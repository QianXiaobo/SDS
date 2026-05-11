#!/usr/bin/env python3
"""
compute_SDS.py — Singleton Density Score (SDS) computation

Python rewrite of compute_SDS.R with NumPy vectorization and ThreadPoolExecutor
parallelism.

Key optimizations over the R original:
  1. np.logaddexp replaces the scalar logsum() function — SIMD vectorized.
  2. NumPy array operations replace R for-loops over individuals.
  3. scipy.optimize.minimize (Nelder-Mead) with ThreadPoolExecutor for parallel
     multi-start optimization.
  4. NumPy broadcasting for genotype-group splitting.

Dependencies:
    pip install numpy scipy

Usage:
    python compute_SDS.py s_file t_file o_file b_file g_file init [options]

Examples:
    # Sequential (single-threaded)
    python compute_SDS.py example.singletons example.testsnp example.observability \\
        example.boundaries example.gamma_shapes 1e-6

    # Parallel with 8 threads
    python compute_SDS.py s.txt t.txt o.txt b.txt g.txt 1e-6 --workers 8

    # Debug mode (progress every 1000 SNPs)
    python compute_SDS.py s.txt t.txt o.txt b.txt g.txt 1e-6 --debug

Input files:
  s_file  : Singleton positions.
            Row i = tab/space-delimited sorted positions of singletons for individual i.
            NA values (or blank) are tolerated.
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

Author: Python rewrite of compute_SDS.R (Field et al 2016,
        Detection of human adaptation during the past 2,000 years).
"""

import sys
import argparse
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count

# ─────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────

DEBUG_MODE = False
PRECISION = 4
E_GRID_NUM_POINTS = 50
E_GRID_SCALE_FACTOR = 20
OPTIM_NUM_ITERATIONS = 5
SKIP_BOUNDARY_FRACTION = 0.05

# ─────────────────────────────────────────────────────────────────
# Core numeric functions
# ─────────────────────────────────────────────────────────────────

def _logsum(log_a, log_b):
    """
    Compute log(exp(log_a) + exp(log_b)) in a numerically stable way.
    Wraps np.logaddexp for clarity; also handles scalar -Inf cases.
    """
    if log_a == -np.inf:
        return log_b
    if log_b == -np.inf:
        return log_a
    return np.logaddexp(log_a, log_b)


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
        ls = np.logaddexp(np.log(dat0), logB1)          # log-sum-exp per element
        LL0 = (2.0 * A1 * (logB1 - np.mean(ls))
               + np.mean(np.log(dat0)) + LOG2 + logA1
               - 2.0 * np.mean(ls)
               + np.mean(np.logaddexp(LOG2 + logA1, 0.0)))
        LL += LL0 * n0

    if n2 > 0:
        ls = np.logaddexp(np.log(dat2), logB2)
        LL2 = (2.0 * A2 * (logB2 - np.mean(ls))
               + np.mean(np.log(dat2)) + LOG2 + logA2
               - 2.0 * np.mean(ls)
               + np.mean(np.logaddexp(LOG2 + logA2, 0.0)))
        LL += LL2 * n2

    if n1 > 0:
        ls1_B1 = np.logaddexp(np.log(dat1), logB1)
        ls1_B2 = np.logaddexp(np.log(dat1), logB2)

        term1 = -2.0 * ls1_B1 + logA1 + np.logaddexp(logA1, 0.0)
        term2 = -2.0 * ls1_B2 + logA2 + np.logaddexp(logA2, 0.0)

        ls_combined = np.logaddexp(term1, term2)
        ls_extra = np.logaddexp(LOG2 + logA1 + logA2 - ls1_B1 - ls1_B2, ls_combined)

        LL1 = (A1 * (logB1 - np.mean(ls1_B1))
               + A2 * (logB2 - np.mean(ls1_B2))
               + np.mean(np.log(dat1))
               + np.mean(ls_extra))
        LL += LL1 * n1

    return -LL


def _optimize_single_init(init_pt, dat0, dat1, dat2, A1, A2):
    """Run minimize from one starting point; return (params, -loglikelihood)."""
    obj = partial(_minus_log_likelihood, dat0=dat0, dat1=dat1, dat2=dat2, A1=A1, A2=A2)
    result = minimize(obj, init_pt, method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    return result.x, -result.fun


# ─────────────────────────────────────────────────────────────────
# Singleton distance computation (vectorized per individual)
# ─────────────────────────────────────────────────────────────────

def _compute_distances(singletons, singletons_current_idx, test_loc,
                       bound_up, bound_down, valid_mask=None):
    """
    Compute upstream and downstream singleton distances.

    Parameters
    ----------
    singletons : ndarray, shape (n_individuals, max_singletons)
    singletons_current_idx : ndarray, shape (n_individuals,)
        Per-individual cursor indices (in-place modified).
    test_loc : float
    bound_up, bound_down : float
    valid_mask : ndarray, shape (n_individuals,), optional
        Boolean mask. If provided, only individuals where mask is True
        have their indices advanced and distances computed. Missing-genotype
        individuals are skipped entirely.

    Returns
    -------
    upstream, downstream : ndarray, shape (n_individuals,)
        Distances for all individuals (NaN for excluded/missing ones).
    """
    n_ind = singletons.shape[0]
    max_col = singletons.shape[1]

    upstream   = np.full(n_ind, np.nan, dtype=np.float64)
    downstream = np.full(n_ind, np.nan, dtype=np.float64)

    # If no mask provided, all individuals are valid
    if valid_mask is None:
        valid_mask = np.ones(n_ind, dtype=bool)

    # Advance cursor and compute distances for VALID individuals only.
    # This ensures missing-genotype individuals do NOT consume their
    # singleton lookahead slots, keeping indices consistent.
    valid_inds = np.where(valid_mask)[0]
    for i in valid_inds:
        ci = singletons_current_idx[i]
        while ci < max_col:
            val = singletons[i, ci]
            if np.isnan(val) or val >= test_loc:
                break
            ci += 1
        singletons_current_idx[i] = ci

    # Upstream: nearest singleton before test_loc
    for i in valid_inds:
        si = singletons_current_idx[i] - 1
        if si >= 0:
            s_loc = singletons[i, si]
            if not np.isnan(s_loc) and s_loc >= bound_up:
                upstream[i] = test_loc - s_loc

    # Downstream: nearest singleton at or after test_loc
    for i in valid_inds:
        ci = singletons_current_idx[i]
        if ci < max_col:
            s_loc = singletons[i, ci]
            if not np.isnan(s_loc) and bound_up <= s_loc <= bound_down:
                downstream[i] = s_loc - test_loc

    return upstream, downstream
    for i in range(n_ind):
        si = singletons_current_idx[i] - 1
        if si >= 0:
            s_loc = singletons[i, si]
            if not np.isnan(s_loc) and s_loc >= bound_up:
                upstream[i] = test_loc - s_loc

    # Compute downstream (nearest singleton at or after test_loc)
    for i in range(n_ind):
        ci = singletons_current_idx[i]
        if ci < max_col:
            s_loc = singletons[i, ci]
            if not np.isnan(s_loc) and bound_up <= s_loc <= bound_down:
                downstream[i] = s_loc - test_loc

    return upstream, downstream


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
    Read singletons file into a (n_individuals, max_cols) ndarray.
    Rows = individuals; columns = singleton positions (NaN-padded).
    """
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            tokens = line.strip().split()
            vals = []
            for t in tokens:
                try:
                    vals.append(float(t))
                except ValueError:
                    vals.append(np.nan)
            # Pad to max_cols
            vals += [np.nan] * (max_cols - len(vals))
            rows.append(vals[:max_cols])
    return np.array(rows, dtype=np.float64)


def read_observability(path):
    """Read and normalize observability probabilities."""
    with open(path, 'r') as fh:
        line = fh.readline().strip()
    vals = np.array([float(t) for t in line.split()], dtype=np.float64)
    return vals / np.mean(vals)


def read_boundaries(path):
    """Read boundaries as (n, 2) ndarray, sorted by start position."""
    data = np.loadtxt(path, dtype=np.float64)
    return data[np.argsort(data[:, 0])]


def read_gamma_shape(path):
    """Read gamma shape params; return sorted (freq_arr, shape_arr)."""
    data = np.loadtxt(path, dtype=np.float64)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute Singleton Density Scores (SDS) — Python/NumPy version',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('s_file',  help='Singleton positions file')
    parser.add_argument('t_file',  help='Test SNPs file')
    parser.add_argument('o_file',  help='Singleton observability file')
    parser.add_argument('b_file',  help='Chromosome boundaries file')
    parser.add_argument('g_file',  help='Gamma shape parameters file')
    parser.add_argument('init', type=float, help='Initial MLE guess (e.g. 1e-6)')
    parser.add_argument('--max_singletons', type=int, default=10000,
                        help='Max singletons per individual (default: 10000)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Parallel workers for MLE (default: CPU count - 1)')
    parser.add_argument('--debug', action='store_true',
                        help='Print progress every 1000 SNPs')
    args = parser.parse_args()

    global DEBUG_MODE
    DEBUG_MODE = args.debug

    n_workers = args.workers or max(1, cpu_count() - 1)

    # ── Load reference data ──
    print(f"# Loading singletons from {args.s_file}...", file=sys.stderr)
    singletons = read_singletons(args.s_file, args.max_singletons)
    n_ind = singletons.shape[0]
    singletons_current_idx = np.ones(n_ind, dtype=int)

    sin_obs = read_observability(args.o_file)
    boundaries = read_boundaries(args.b_file)
    gamma_freq, gamma_shape = read_gamma_shape(args.g_file)

    print(f"# Loaded: {n_ind} individuals, "
          f"{singletons.shape[1]} max singletons, "
          f"{len(boundaries)} boundary regions, "
          f"{len(gamma_freq)} gamma points", file=sys.stderr)
    print(f"# Using {n_workers} workers for parallel MLE optimization.", file=sys.stderr)

    # ── Header ──
    print("ID\tAA\tDA\tPOS\tDAF\tnG0\tnG1\tnG2\trSDS\tSuggestedInitPoint")

    # ── Precompute log-E grid ──
    e_grid_center = args.init
    logE_grid = np.linspace(
        np.log(e_grid_center) - np.log(E_GRID_SCALE_FACTOR),
        np.log(e_grid_center) + np.log(E_GRID_SCALE_FACTOR),
        E_GRID_NUM_POINTS
    )
    logE_center = np.log(e_grid_center)

    # ── MLE worker ──
    def run_mle(dat0, dat1, dat2, A1, A2):
        """Run multi-start Nelder-Mead; return best (logE1, logE2, -loglik)."""
        # Build starting points: OPTIM_NUM_ITERATIONS random + 1 center
        init_pts = [np.array([np.random.choice(logE_grid),
                              np.random.choice(logE_grid)])
                    for _ in range(OPTIM_NUM_ITERATIONS)]
        init_pts.append(np.array([logE_center, logE_center]))

        with ThreadPoolExecutor(max_workers=min(n_workers, len(init_pts))) as pool:
            futures = [pool.submit(_optimize_single_init,
                                    pt, dat0, dat1, dat2, A1, A2)
                       for pt in init_pts]
            results = [f.result() for f in as_completed(futures)]

        best_params, best_ll = max(results, key=lambda r: r[1])
        return best_params, best_ll

    # ── Main SNP loop ──
    boundaries_cur = 0
    processed = 0
    total = 0

    with open(args.t_file, 'r') as snp_fh:
        for line in snp_fh:
            line = line.strip()
            if not line:
                continue
            total += 1

            tokens = line.split()
            if len(tokens) < 5:
                continue

            snp_id   = tokens[0]
            allele1  = tokens[1]   # ancestral
            allele2  = tokens[2]   # derived
            test_loc = float(tokens[3])

            # ── Parse genotypes, treating "." as missing (NaN) ──
            raw_genotypes = np.array([
                np.nan if t in ('.', 'NA', 'nan', '') else float(t)
                for t in tokens[4:]
            ], dtype=np.float64)

            # Skip this SNP if all genotypes are missing
            valid_mask = ~np.isnan(raw_genotypes)
            if not np.any(valid_mask):
                continue

            genotypes = raw_genotypes[valid_mask]   # only valid individuals

            # ── Advance to the correct boundary region ──
            while (boundaries_cur < len(boundaries)
                   and boundaries[boundaries_cur, 1] < test_loc):
                boundaries_cur += 1

            if boundaries_cur >= len(boundaries):
                break

            if boundaries[boundaries_cur, 0] > test_loc:
                continue

            bound_up = boundaries[boundaries_cur, 0]
            bound_down = boundaries[boundaries_cur, 1]

            # ── Singleton distances for VALID individuals only ──
            upstream_all, downstream_all = _compute_distances(
                singletons, singletons_current_idx, test_loc,
                bound_up, bound_down, valid_mask=valid_mask)

            # Keep only valid individuals
            upstream   = upstream_all[valid_mask]
            downstream = downstream_all[valid_mask]

            # Skip if too many missing singleton distances within valid individuals
            if (np.mean(np.isnan(upstream)) > SKIP_BOUNDARY_FRACTION
                    or np.mean(np.isnan(downstream)) > SKIP_BOUNDARY_FRACTION):
                continue

            upstream[np.isnan(upstream)]  = np.nanmax(upstream)
            downstream[np.isnan(downstream)] = np.nanmax(downstream)
            intervals = (upstream + downstream) * sin_obs[valid_mask]

            # ── Genotype groups (no missing values here) ──
            daf = float(np.mean(genotypes)) / 2.0
            dat0 = intervals[genotypes == 0]
            dat1 = intervals[genotypes == 1]
            dat2 = intervals[genotypes == 2]

            # ── Gamma shape parameters ──
            A1 = _get_gamma_shape(1.0 - daf, gamma_freq, gamma_shape)
            A2 = _get_gamma_shape(daf, gamma_freq, gamma_shape)

            # ── Parallel MLE ──
            best_params, _ = run_mle(dat0, dat1, dat2, A1, A2)

            logE1, logE2 = best_params
            rSDS = logE1 - logE2
            suggested = f"1e{np.mean(best_params) / np.log(10.0):.1f}"

            print(f"{snp_id}\t{allele1}\t{allele2}\t{int(test_loc)}\t"
                  f"{daf:.{PRECISION}f}\t{len(dat0)}\t{len(dat1)}\t{len(dat2)}\t"
                  f"{rSDS:.{PRECISION}f}\t{suggested}")

            processed += 1
            if DEBUG_MODE and processed % 1000 == 0:
                print(f"# Processed {processed} SNPs...", file=sys.stderr)

    print(f"# Done. Processed {processed}/{total} SNPs.", file=sys.stderr)


if __name__ == '__main__':
    main()
