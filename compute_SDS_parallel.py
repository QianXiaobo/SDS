#!/usr/bin/env python3
"""
compute_SDS_parallel.py — Singleton Density Score (SDS) computation

Python rewrite of compute_SDS.R with true multi-process parallelism and
chunked streaming for large datasets.

Key optimizations over the R original:
  1. np.searchsorted replaces the sequential linear scan over singletons,
     making each test-SNP's interval computation fully independent — O(log n)
     per lookup instead of O(n) linear scan.
  2. Because each SNP is now independent, the expensive MLE optimization can
     be parallelized across SNPs using ProcessPoolExecutor (true multi-process,
     bypassing the GIL), rather than just parallelizing starting points within
     a single SNP.
  3. Singletons are stored as a list of sorted np arrays (one per individual)
     instead of a padded NaN matrix — less memory, and searchsorted is O(log n).
  4. Chunked streaming: test SNPs are read and processed in configurable chunks,
     keeping memory bounded. This is critical for genome-scale data
     (e.g. 3M SNPs × 12K individuals comfortably fits in 16 GB RAM).
  5. Genotypes stored as int8 (8× memory reduction vs float64).
  6. Vectorized chunk processing (v2): instead of per-SNP per-individual Python
     loops, interval computation is batched — for each individual, all k SNP
     positions in a chunk are processed simultaneously via vectorized
     np.searchsorted. This eliminates O(n_snps × n_ind) Python-call overhead.
  7. Diagnostic summary: comprehensive skip-reason counters and output
     completeness checks are printed to stderr at completion, making it easy
     to verify output integrity.

Memory note (v2):
  The vectorized approach stores (n_ind × chunk_size) intermediate arrays
  (~192 MB per component for 12K ind × 2K SNPs). With the default chunk_size
  of 2000, peak memory is ~500 MB for the chunk arrays + singleton data.
  Reduce --chunk-size if memory is constrained.

Dependencies:
    pip install numpy scipy

Usage:
    python compute_SDS_parallel.py s_file t_file o_file b_file g_file init [options]

Examples:
    # Sequential (single worker)
    python compute_SDS_parallel.py example.singletons example.testsnp \\
        example.observability example.boundaries example.gamma_shapes 1e-6

    # Parallel with 8 workers
    python compute_SDS_parallel.py s.txt t.txt o.txt b.txt g.txt 1e-6 --workers 8

    # Tune chunk size for memory/performance trade-off
    python compute_SDS_parallel.py s.txt t.txt o.txt b.txt g.txt 1e-6 \\
        --chunk-size 3000 --workers 8

    # Debug mode (progress every 1000 SNPs)
    python compute_SDS_parallel.py s.txt t.txt o.txt b.txt g.txt 1e-6 --debug

Input files:
  s_file  : Singleton positions.
            Row i = tab/space-delimited sorted positions of singletons for individual i.
  t_file  : Test SNPs.
            Columns: ID  ancestral_allele  derived_allele  position  genotype...
            genotype entries are 0 (AA), 1 (AD), 2 (DD).
  o_file  : Singleton observability probabilities.
            A single row of tab/space-delimited values, one per individual.
  b_file  : Chromosome boundaries.
            Columns: start  end  (one region per row).
  g_file  : Gamma shape parameters.
            Columns: derived_allele_frequency  shape_parameter  (sorted by frequency).

Output (tab-delimited, one row per test SNP):
  ID  AA  DA  POS  DAF  nG0  nG1  nG2  rSDS  SuggestedInitPoint

Reference:
    Field et al 2016, Detection of human adaptation during the past 2,000 years.
"""

import sys
import argparse
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# ─────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────

PRECISION = 4
E_GRID_NUM_POINTS = 50
E_GRID_SCALE_FACTOR = 20
OPTIM_NUM_ITERATIONS = 5
SKIP_BOUNDARY_FRACTION = 0.05
GENOTYPE_MISSING = -1          # sentinel for missing genotype in int8 arrays


# ─────────────────────────────────────────────────────────────────
# Core numeric functions (must be at module level for pickling)
# ─────────────────────────────────────────────────────────────────

def _minus_log_likelihood(x, dat0, dat1, dat2, A1, A2):
    """
    Negative log-likelihood for the SDS model. Fully vectorized.

    Parameters
    ----------
    x : array_like, shape (2,)
        [logE1, logE2] — log mean tip lengths (ancestral, derived).
    dat0, dat1, dat2 : ndarray
        Singleton intervals for genotype groups 0, 1, 2.
    A1, A2 : float
        Gamma shape parameters.

    Returns
    -------
    float
        Negative log-likelihood (to be minimized).
    """
    logE1, logE2 = float(x[0]), float(x[1])
    logA1, logA2 = np.log(A1), np.log(A2)
    logB1 = logA1 - logE1
    logB2 = logA2 - logE2
    LOG2 = np.log(2.0)

    n0, n1, n2 = len(dat0), len(dat1), len(dat2)
    LL = 0.0

    if n0 > 0:
        log_dat = np.log(dat0)
        ls = np.logaddexp(log_dat, logB1)
        LL0 = (2.0 * A1 * (logB1 - np.mean(ls))
               + np.mean(log_dat) + LOG2 + logA1
               - 2.0 * np.mean(ls)
               + np.mean(np.logaddexp(LOG2 + logA1, 0.0)))
        LL += LL0 * n0

    if n2 > 0:
        log_dat = np.log(dat2)
        ls = np.logaddexp(log_dat, logB2)
        LL2 = (2.0 * A2 * (logB2 - np.mean(ls))
               + np.mean(log_dat) + LOG2 + logA2
               - 2.0 * np.mean(ls)
               + np.mean(np.logaddexp(LOG2 + logA2, 0.0)))
        LL += LL2 * n2

    if n1 > 0:
        log_dat = np.log(dat1)
        ls1_B1 = np.logaddexp(log_dat, logB1)
        ls1_B2 = np.logaddexp(log_dat, logB2)

        term1 = -2.0 * ls1_B1 + logA1 + np.logaddexp(logA1, 0.0)
        term2 = -2.0 * ls1_B2 + logA2 + np.logaddexp(logA2, 0.0)

        ls_combined = np.logaddexp(term1, term2)
        ls_extra = np.logaddexp(LOG2 + logA1 + logA2 - ls1_B1 - ls1_B2, ls_combined)

        LL1 = (A1 * (logB1 - np.mean(ls1_B1))
               + A2 * (logB2 - np.mean(ls1_B2))
               + np.mean(log_dat)
               + np.mean(ls_extra))
        LL += LL1 * n1

    if np.isnan(LL) or np.isinf(LL):
        return 1e15
    return -LL


def _optimize_one_start(args):
    """Run minimize from one starting point; return (params, -loglikelihood)."""
    init_pt, dat0, dat1, dat2, A1, A2 = args
    result = minimize(_minus_log_likelihood, init_pt,
                      args=(dat0, dat1, dat2, A1, A2),
                      method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    return result.x, -result.fun


def _run_mle_for_snp(snp_task):
    """
    Run multi-start Nelder-Mead optimization for a single test-SNP.

    Parameters
    ----------
    snp_task : tuple
        (dat0, dat1, dat2, A1, A2, logE_grid, logE_center)

    Returns
    -------
    tuple
        (best_logE1, best_logE2) or None if optimization fails.
    """
    dat0, dat1, dat2, A1, A2, logE_grid, logE_center = snp_task

    # Build starting points: random from grid + center
    rng = np.random.default_rng()
    init_pts = []
    for _ in range(OPTIM_NUM_ITERATIONS):
        pt = np.array([rng.choice(logE_grid), rng.choice(logE_grid)])
        init_pts.append((pt, dat0, dat1, dat2, A1, A2))
    init_pts.append((np.array([logE_center, logE_center]),
                      dat0, dat1, dat2, A1, A2))

    best_params = None
    best_ll = -np.inf

    for args in init_pts:
        params, ll = _optimize_one_start(args)
        if ll > best_ll:
            best_ll = ll
            best_params = params

    return best_params


# ─────────────────────────────────────────────────────────────────
# Vectorized chunk processing — batched per-individual searchsorted
# ─────────────────────────────────────────────────────────────────

def _process_chunk_vectorized(chunk, singletons_list, sin_obs, boundaries,
                               boundaries_cur, gamma_freq, gamma_shape,
                               logE_grid, logE_center, n_ind):
    """
    Process a chunk of SNPs with fully vectorized interval computation.

    Instead of per-SNP per-individual Python loops (O(k * n_ind) overhead),
    this loops over individuals once and uses batched np.searchsorted on all
    k SNP positions simultaneously. The inner operations are all numpy-level.

    Returns (mle_tasks, task_info, boundaries_cur, skip_stats) where
    skip_stats is a dict of {reason: count}.
    """
    k = len(chunk)

    # ── Extract positions, genotype masks, and metadata ──
    positions = np.empty(k)
    valid_masks = [None] * k
    genotypes_list = [None] * k
    snp_meta = [None] * k

    for j, snp in enumerate(chunk):
        positions[j] = snp['location']
        raw_geno = snp['genotypes']
        if len(raw_geno) != n_ind:
            # Genotype count mismatch — treat all as missing (SNP will be filtered)
            valid_masks[j] = np.zeros(n_ind, dtype=bool)
            genotypes_list[j] = np.array([], dtype=np.int8)
        else:
            mask = raw_geno != GENOTYPE_MISSING
            valid_masks[j] = mask
            genotypes_list[j] = raw_geno[mask]
        snp_meta[j] = (snp['id'], snp['allele1'], snp['allele2'])

    # ── Assign boundaries per SNP ──
    bound_ups = np.full(k, np.nan)
    bound_downs = np.full(k, np.nan)
    snp_in_boundary = np.zeros(k, dtype=bool)

    n_past_all = 0
    n_between = 0

    for j in range(k):
        test_loc = positions[j]
        while boundaries_cur < len(boundaries) and boundaries[boundaries_cur, 1] < test_loc:
            boundaries_cur += 1
        if boundaries_cur >= len(boundaries):
            n_past_all = k - j  # remaining SNPs past last boundary
            break
        if boundaries[boundaries_cur, 0] > test_loc:
            n_between += 1
            continue
        snp_in_boundary[j] = True
        bound_ups[j] = boundaries[boundaries_cur, 0]
        bound_downs[j] = boundaries[boundaries_cur, 1]

    # ── Vectorized interval computation per individual ──
    upstream = np.full((n_ind, k), np.nan)
    downstream = np.full((n_ind, k), np.nan)

    active = np.where(snp_in_boundary)[0]
    if len(active) > 0:
        pos_active = positions[active]
        bu_active = bound_ups[active]
        bd_active = bound_downs[active]

        for i, s in enumerate(singletons_list):
            if len(s) == 0:
                continue

            idx = np.searchsorted(s, pos_active)  # (k_active,)

            # Upstream: singleton just before each test position
            up_ok = idx > 0
            up_idx = np.where(up_ok)[0]
            if len(up_idx) > 0:
                s_before = s[idx[up_idx] - 1]
                within = s_before >= bu_active[up_idx]
                good = up_idx[within]
                if len(good) > 0:
                    j_global = active[good]
                    upstream[i, j_global] = positions[j_global] - s[idx[good] - 1]

            # Downstream: singleton at or after each test position
            down_ok = idx < len(s)
            down_idx = np.where(down_ok)[0]
            if len(down_idx) > 0:
                s_after = s[idx[down_idx]]
                within = s_after <= bd_active[down_idx]
                good = down_idx[within]
                if len(good) > 0:
                    j_global = active[good]
                    downstream[i, j_global] = s[idx[good]] - positions[j_global]

    # ── Per-SNP: NA check, genotype split, build MLE tasks ──
    mle_tasks = []
    task_info = []
    n_na_skip = 0
    n_degenerate = 0

    for j in range(k):
        if not snp_in_boundary[j]:
            continue

        mask = valid_masks[j]
        n_valid = np.count_nonzero(mask)
        if n_valid == 0:
            continue

        up_col = upstream[mask, j]
        down_col = downstream[mask, j]

        na_up = np.mean(np.isnan(up_col))
        na_down = np.mean(np.isnan(down_col))

        if na_up > SKIP_BOUNDARY_FRACTION or na_down > SKIP_BOUNDARY_FRACTION:
            n_na_skip += 1
            continue

        # Fill remaining NAs with max observed in that component
        if na_up > 0:
            up_col[np.isnan(up_col)] = np.nanmax(up_col)
        if na_down > 0:
            down_col[np.isnan(down_col)] = np.nanmax(down_col)

        intervals = (up_col + down_col) * sin_obs[mask]

        genotypes = genotypes_list[j]
        dat0 = intervals[genotypes == 0]
        dat1 = intervals[genotypes == 1]
        dat2 = intervals[genotypes == 2]

        if len(dat0) == 0 and len(dat2) == 0:
            n_degenerate += 1
            continue

        daf = float(np.mean(genotypes)) / 2.0
        A1 = _get_gamma_shape(1.0 - daf, gamma_freq, gamma_shape)
        A2 = _get_gamma_shape(daf, gamma_freq, gamma_shape)

        snp_id, a1, a2 = snp_meta[j]
        task_info.append({
            'id': snp_id,
            'allele1': a1,
            'allele2': a2,
            'location': positions[j],
            'daf': daf,
            'n0': len(dat0),
            'n1': len(dat1),
            'n2': len(dat2),
        })
        mle_tasks.append((dat0, dat1, dat2, A1, A2, logE_grid, logE_center))

    skip_stats = {
        'past_all_boundaries': n_past_all,
        'between_boundaries': n_between,
        'na_fraction': n_na_skip,
        'degenerate': n_degenerate,
    }
    return mle_tasks, task_info, boundaries_cur, skip_stats


# ─────────────────────────────────────────────────────────────────
# Gamma shape interpolation
# ─────────────────────────────────────────────────────────────────

def _get_gamma_shape(freq, freq_arr, shape_arr):
    """Linear interpolation of gamma shape at given allele frequency."""
    if freq <= freq_arr[0]:
        return shape_arr[0]
    if freq >= freq_arr[-1]:
        return shape_arr[-1]
    idx = int(np.searchsorted(freq_arr, freq))
    idx = max(1, min(idx, len(freq_arr) - 1))
    x1, x2 = freq_arr[idx - 1], freq_arr[idx]
    y1, y2 = shape_arr[idx - 1], shape_arr[idx]
    return y1 + (y2 - y1) * (freq - x1) / (x2 - x1)


# ─────────────────────────────────────────────────────────────────
# File I/O utilities
# ─────────────────────────────────────────────────────────────────

def read_singletons(path, max_cols=10000):
    """
    Read singletons file into a list of sorted ndarrays (one per individual).

    Returns list of ndarray, which is more memory-efficient than a padded
    NaN matrix and enables O(log n) lookup via np.searchsorted.

    Each individual's singletons are truncated to the first max_cols entries
    to bound worst-case memory usage.
    """
    singletons = []
    with open(path, 'r') as fh:
        for line in fh:
            tokens = line.strip().split()
            vals = []
            for t in tokens[:max_cols]:
                try:
                    v = float(t)
                    if not np.isnan(v):
                        vals.append(v)
                except ValueError:
                    pass
            singletons.append(np.array(sorted(vals), dtype=np.float64))
    return singletons


def read_observability(path):
    """Read and normalize observability probabilities."""
    with open(path, 'r') as fh:
        line = fh.readline().strip()
    if not line:
        return None
    vals = np.array([float(t) for t in line.split()], dtype=np.float64)
    return vals / np.mean(vals)


def read_boundaries(path):
    """Read boundaries as (n, 2) ndarray, sorted by start position."""
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    return data[np.argsort(data[:, 0])]


def read_gamma_shape(path):
    """Read gamma shape params; return sorted (freq_arr, shape_arr)."""
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def iter_test_snp_chunks(path, chunk_size):
    """
    Generator that yields chunks of test SNPs.

    Each chunk is a list of dicts with keys:
        id, allele1, allele2, location, genotypes (int8 ndarray, -1 = missing)

    This streams through the file without loading all SNPs into memory,
    which is essential for genome-scale datasets (e.g. 3M SNPs × 12K ind).
    """
    chunk = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 5:
                continue

            # Parse genotypes as int8; -1 denotes missing
            n_geno = len(tokens) - 4
            genotypes = np.full(n_geno, GENOTYPE_MISSING, dtype=np.int8)
            for j, t in enumerate(tokens[4:]):
                if t not in ('.', 'NA', 'nan', ''):
                    try:
                        genotypes[j] = int(t)
                    except ValueError:
                        pass  # keep GENOTYPE_MISSING

            chunk.append({
                'id': tokens[0],
                'allele1': tokens[1],
                'allele2': tokens[2],
                'location': float(tokens[3]),
                'genotypes': genotypes,
            })

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if chunk:
        yield chunk


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute Singleton Density Scores (SDS) — parallel Python version',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('s_file',  help='Singleton positions file')
    parser.add_argument('t_file',  help='Test SNPs file')
    parser.add_argument('o_file',  help='Singleton observability file')
    parser.add_argument('b_file',  help='Chromosome boundaries file')
    parser.add_argument('g_file',  help='Gamma shape parameters file')
    parser.add_argument('init', type=float, help='Initial MLE guess (e.g. 1e-6)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for SDS results (default: stdout). '
                             'Use "-" for stdout explicitly.')
    parser.add_argument('--max-singletons', type=int, default=10000,
                        help='Max singletons per individual (default: 10000)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Parallel workers for MLE (default: CPU count - 1)')
    parser.add_argument('--chunk-size', '-c', type=int, default=2000,
                        help='SNPs per processing chunk (default: 2000). '
                             'Lower = less memory; higher = better parallelism.')
    parser.add_argument('--debug', action='store_true',
                        help='Print progress every 1000 SNPs')
    args = parser.parse_args()

    n_workers = args.workers or max(1, cpu_count() - 1)

    # ── Open output file ──
    if args.output and args.output != '-':
        out_fh = open(args.output, 'w', buffering=1)  # line-buffered
    else:
        out_fh = sys.stdout

    # ── Load reference data (small, constant memory) ──
    print(f"# Loading singletons from {args.s_file}...")
    singletons_list = read_singletons(args.s_file, args.max_singletons)
    n_ind = len(singletons_list)
    print(f"#   {n_ind} individuals, "
          f"avg {np.mean([len(s) for s in singletons_list]):.0f} singletons/ind")

    sin_obs = read_observability(args.o_file)
    if sin_obs is None:
        sin_obs = np.ones(n_ind, dtype=np.float64)
    assert len(sin_obs) == n_ind, \
        f"Observability has {len(sin_obs)} values but {n_ind} individuals in singletons"

    boundaries = read_boundaries(args.b_file)
    gamma_freq, gamma_shape = read_gamma_shape(args.g_file)

    print(f"# Loaded: {n_ind} individuals, "
          f"{len(boundaries)} boundary regions, "
          f"{len(gamma_freq)} gamma points")
    print(f"# Chunk size: {args.chunk_size} SNPs, "
          f"{n_workers} parallel workers")

    # ── Precompute log-E grid (same for all SNPs) ──
    e_grid_center = args.init
    logE_grid = np.linspace(
        np.log(e_grid_center) - np.log(E_GRID_SCALE_FACTOR),
        np.log(e_grid_center) + np.log(E_GRID_SCALE_FACTOR),
        E_GRID_NUM_POINTS
    )
    logE_center = np.log(e_grid_center)

    # ── Header ──
    print("ID\tAA\tDA\tPOS\tDAF\tnG0\tnG1\tnG2\trSDS\tSuggestedInitPoint",
          file=out_fh)

    # ── Process SNPs in chunks ──
    boundaries_cur = 0
    total_read = 0
    total_output = 0

    # Track skip reasons for final diagnostic summary
    skip_counts = {
        'past_all_boundaries': 0,
        'between_boundaries': 0,
        'na_fraction': 0,
        'degenerate': 0,
        'mle_failed': 0,
    }

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for chunk_idx, chunk in enumerate(
                    iter_test_snp_chunks(args.t_file, args.chunk_size)):

                total_read += len(chunk)

                if args.debug:
                    print(f"# Chunk {chunk_idx + 1}: {len(chunk)} SNPs "
                          f"(total read: {total_read}), computing intervals...")

                mle_tasks, task_info, boundaries_cur, skip_stats = \
                    _process_chunk_vectorized(
                        chunk, singletons_list, sin_obs, boundaries,
                        boundaries_cur, gamma_freq, gamma_shape,
                        logE_grid, logE_center, n_ind)

                for key in skip_counts:
                    skip_counts[key] += skip_stats.get(key, 0)

                if args.debug:
                    print(f"#   {len(mle_tasks)} valid SNPs, "
                          f"submitting MLE jobs...")
                    if skip_stats['na_fraction'] > 0:
                        print(f"#   Skipped (NA fraction):    {skip_stats['na_fraction']}")
                    if skip_stats['degenerate'] > 0:
                        print(f"#   Skipped (degenerate):     {skip_stats['degenerate']}")
                    if skip_stats['between_boundaries'] > 0:
                        print(f"#   Skipped (no boundary):    {skip_stats['between_boundaries']}")

                if not mle_tasks:
                    if skip_stats['past_all_boundaries'] > 0:
                        if args.debug:
                            print(f"#   Past last boundary — processing complete.")
                        break
                    continue

                # Submit all MLE tasks for this chunk
                future_to_idx = {
                    executor.submit(_run_mle_for_snp, task): i
                    for i, task in enumerate(mle_tasks)
                }

                results = [None] * len(mle_tasks)
                chunk_completed = 0
                n_mle_failures = 0

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    chunk_completed += 1
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        n_mle_failures += 1
                        if args.debug:
                            snp_id = task_info[idx]['id']
                            print(f"#   WARNING: MLE failed for SNP {snp_id}: {e}")
                    if args.debug and chunk_completed % 1000 == 0:
                        print(f"#   MLE {chunk_completed}/{len(mle_tasks)}")

                skip_counts['mle_failed'] += n_mle_failures

                # Output results for this chunk (preserves input order)
                n_written = 0
                for idx, info in enumerate(task_info):
                    best_params = results[idx]
                    if best_params is None:
                        continue

                    logE1, logE2 = best_params
                    rSDS = logE1 - logE2
                    suggested_exp = round(np.mean(best_params) / np.log(10.0))
                    suggested = f"1e{int(suggested_exp)}"
                    pos_str = str(int(info['location']))

                    print(f"{info['id']}\t{info['allele1']}\t{info['allele2']}\t"
                          f"{pos_str}\t"
                          f"{info['daf']:.{PRECISION}f}\t"
                          f"{info['n0']}\t{info['n1']}\t{info['n2']}\t"
                          f"{rSDS:.{PRECISION}f}\t{suggested}",
                          file=out_fh)
                    n_written += 1

                total_output += n_written

                if args.debug:
                    print(f"#   Chunk done: {n_written} output rows. "
                          f"Running total: {total_output} rows.")

        # ── Final diagnostic summary ──
        print(file=sys.stderr)
        print("=" * 56, file=sys.stderr)
        print("  SDS COMPUTATION SUMMARY", file=sys.stderr)
        print("=" * 56, file=sys.stderr)
        print(f"  Total SNPs read:              {total_read:>10d}", file=sys.stderr)
        print(f"  Past last boundary:           {skip_counts['past_all_boundaries']:>10d}", file=sys.stderr)
        print(f"  No boundary region:           {skip_counts['between_boundaries']:>10d}", file=sys.stderr)
        print(f"  High NA fraction (>5%):       {skip_counts['na_fraction']:>10d}", file=sys.stderr)
        print(f"  Degenerate genotypes:         {skip_counts['degenerate']:>10d}", file=sys.stderr)
        print(f"  MLE optimization failures:    {skip_counts['mle_failed']:>10d}", file=sys.stderr)
        print(f"  {'-' * 40}", file=sys.stderr)
        print(f"  Valid output rows written:    {total_output:>10d}", file=sys.stderr)
        print("=" * 56, file=sys.stderr)

        if total_output == 0:
            print(file=sys.stderr)
            print("*** WARNING: Zero output rows! Check your input data. ***",
                  file=sys.stderr)
            print("  - Are test SNP positions within the chromosome boundaries?",
                  file=sys.stderr)
            print("  - Does the singletons file have data for all individuals?",
                  file=sys.stderr)
        elif total_output < total_read * 0.1:
            print(file=sys.stderr)
            print(f"*** NOTE: Only {total_output}/{total_read} "
                  f"({100.0 * total_output / total_read:.1f}%) SNPs produced output. "
                  f"Check if this is expected. ***",
                  file=sys.stderr)
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == '__main__':
    main()
