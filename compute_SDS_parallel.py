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
# Singleton distance computation using searchsorted (O(log n))
# ─────────────────────────────────────────────────────────────────

def _compute_snp_intervals(test_loc, bound_up, bound_down,
                           singletons_list, sin_observability):
    """
    Compute singleton intervals for one test-SNP using binary search.

    Each individual's singletons are stored as a sorted array, so we can
    use np.searchsorted to find the nearest upstream/downstream singleton
    in O(log n) time instead of the R code's O(n) linear scan.

    Parameters
    ----------
    test_loc : float
        Test-SNP position.
    bound_up, bound_down : float
        Boundary limits.
    singletons_list : list of ndarray
        One sorted array per individual.
    sin_observability : ndarray
        Observability correction per individual.

    Returns
    -------
    ndarray or None
        Singleton intervals array, or None if SNP should be skipped.
    """
    n_ind = len(singletons_list)
    upstream = np.full(n_ind, np.nan)
    downstream = np.full(n_ind, np.nan)

    for i in range(n_ind):
        s = singletons_list[i]
        if len(s) == 0:
            continue

        # Find insertion point for test_loc in sorted singletons
        idx = np.searchsorted(s, test_loc)

        # Upstream: nearest singleton before test_loc
        if idx > 0:
            s_loc = s[idx - 1]
            if s_loc >= bound_up:
                upstream[i] = test_loc - s_loc

        # Downstream: nearest singleton at or after test_loc
        if idx < len(s):
            s_loc = s[idx]
            if s_loc <= bound_down:
                downstream[i] = s_loc - test_loc

    # Check boundary missing fraction
    na_up = np.mean(np.isnan(upstream))
    na_down = np.mean(np.isnan(downstream))

    if na_up > SKIP_BOUNDARY_FRACTION or na_down > SKIP_BOUNDARY_FRACTION:
        return None

    # Fill NAs with max observed distance
    if na_up > 0 and na_up <= SKIP_BOUNDARY_FRACTION:
        upstream[np.isnan(upstream)] = np.nanmax(upstream)
    if na_down > 0 and na_down <= SKIP_BOUNDARY_FRACTION:
        downstream[np.isnan(downstream)] = np.nanmax(downstream)

    intervals = (upstream + downstream) * sin_observability
    return intervals


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
    """
    singletons = []
    with open(path, 'r') as fh:
        for line in fh:
            tokens = line.strip().split()
            vals = []
            for t in tokens:
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
# Process a single chunk of SNPs (interval computation + MLE submit)
# ─────────────────────────────────────────────────────────────────

def _process_snp_chunk(chunk, singletons_list, sin_obs, boundaries,
                       boundaries_cur, gamma_freq, gamma_shape,
                       logE_grid, logE_center, n_ind):
    """
    Compute intervals for a chunk of test-SNPs and build MLE task list.

    Returns (mle_tasks, task_info, boundaries_cur) where:
      mle_tasks  — list of tuples ready for _run_mle_for_snp
      task_info  — parallel list of per-SNP metadata dicts for output

    The caller is responsible for submitting mle_tasks to the executor
    and pairing results with task_info.
    """
    mle_tasks = []
    task_info = []

    for snp in chunk:
        test_loc = snp['location']
        raw_genotypes = snp['genotypes']

        # Filter to individuals with valid genotypes
        valid_mask = raw_genotypes != GENOTYPE_MISSING
        if not np.any(valid_mask):
            continue

        genotypes = raw_genotypes[valid_mask]
        obs_valid = sin_obs[valid_mask]

        # Advance boundary index
        while (boundaries_cur < len(boundaries)
               and boundaries[boundaries_cur, 1] < test_loc):
            boundaries_cur += 1

        if boundaries_cur >= len(boundaries):
            break

        if boundaries[boundaries_cur, 0] > test_loc:
            continue

        bound_up = boundaries[boundaries_cur, 0]
        bound_down = boundaries[boundaries_cur, 1]

        # Filter singletons to valid individuals for this SNP
        valid_singletons = [singletons_list[i] for i in range(n_ind)
                            if valid_mask[i]]

        intervals = _compute_snp_intervals(
            test_loc, bound_up, bound_down,
            valid_singletons, obs_valid)

        if intervals is None:
            continue

        # Split by genotype (genotypes are int8 0/1/2 for valid individuals)
        daf = float(np.mean(genotypes)) / 2.0
        dat0 = intervals[genotypes == 0]
        dat1 = intervals[genotypes == 1]
        dat2 = intervals[genotypes == 2]

        # Skip if both homozygous groups are empty (degenerate)
        if len(dat0) == 0 and len(dat2) == 0:
            continue

        # Gamma shape parameters
        A1 = _get_gamma_shape(1.0 - daf, gamma_freq, gamma_shape)
        A2 = _get_gamma_shape(daf, gamma_freq, gamma_shape)

        task_info.append({
            'id': snp['id'],
            'allele1': snp['allele1'],
            'allele2': snp['allele2'],
            'location': test_loc,
            'daf': daf,
            'n0': len(dat0),
            'n1': len(dat1),
            'n2': len(dat2),
        })

        mle_tasks.append((dat0, dat1, dat2, A1, A2, logE_grid, logE_center))

    return mle_tasks, task_info, boundaries_cur


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
        out_fh = open(args.output, 'w')
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
    total_processed = 0
    total_valid = 0

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for chunk_idx, chunk in enumerate(
                    iter_test_snp_chunks(args.t_file, args.chunk_size)):

                if args.debug:
                    print(f"# Chunk {chunk_idx + 1}: {len(chunk)} SNPs, "
                          f"computing intervals...")

                mle_tasks, task_info, boundaries_cur = _process_snp_chunk(
                    chunk, singletons_list, sin_obs, boundaries,
                    boundaries_cur, gamma_freq, gamma_shape,
                    logE_grid, logE_center, n_ind)

                if not mle_tasks:
                    total_processed += len(chunk)
                    continue

                if args.debug:
                    print(f"#   {len(mle_tasks)} valid SNPs, "
                          f"submitting MLE jobs...")

                # Submit all MLE tasks for this chunk
                future_to_idx = {
                    executor.submit(_run_mle_for_snp, task): i
                    for i, task in enumerate(mle_tasks)
                }

                results = [None] * len(mle_tasks)
                chunk_completed = 0

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    chunk_completed += 1
                    if args.debug and chunk_completed % 1000 == 0:
                        print(f"#   MLE {chunk_completed}/{len(mle_tasks)}")

                # Output results for this chunk (preserves input order)
                for idx, info in enumerate(task_info):
                    best_params = results[idx]
                    if best_params is None:
                        continue

                    logE1, logE2 = best_params
                    rSDS = logE1 - logE2
                    suggested_exp = round(np.mean(best_params) / np.log(10.0))
                    suggested = f"1e{int(suggested_exp)}"
                    pos_str = (f"{info['location']:.0f}" if info['location'] < 1e6
                               else f"{info['location']:.4g}")

                    print(f"{info['id']}\t{info['allele1']}\t{info['allele2']}\t"
                          f"{pos_str}\t"
                          f"{info['daf']:.{PRECISION}f}\t"
                          f"{info['n0']}\t{info['n1']}\t{info['n2']}\t"
                          f"{rSDS:.{PRECISION}f}\t{suggested}",
                          file=out_fh)

                total_processed += len(chunk)
                total_valid += len(task_info)

                if args.debug:
                    print(f"#   Chunk done. Total: {total_valid} valid / "
                          f"{total_processed} processed SNPs")

        print(f"# Done. {total_valid} / {total_processed} SNPs processed.")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()


if __name__ == '__main__':
    main()
