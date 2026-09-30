#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_case2_dual_mode_boundary_controlled.py

Theory-aligned Case 2 for the partially high-dimensional S^2ACT framework.

This script uses exactly the same dual-mode boundary-controlled DGP as the
revised Case 1.  Case 2 fixes homogeneous marginal scales (h=0), the common
total factor strength, and the low-dimensional population configuration.  The
only designed difficulty that changes is the distance of the weakest
high-dimensional population correlation spike from the ACT boundary:

    lambda_{1,k}^{pop} = tau_1,n + Delta_pop,
    tau_1,n = 1 + sqrt{p/[q(n-1)]}.

For every Monte Carlo replication, R, C, G, W, and all sketch index streams
are shared across Delta_pop.  Hence comparisons across boundary distances use
common random numbers and isolate high-dimensional spectral separation.

DGP
---
    F_i = A^{1/2} G_i B^{1/2},
    Z_i = R F_i C^T + E_i,
    E_i = sigma W_i,

where the random block-pervasive loadings satisfy R^T R=pI_k and C^T C=qI_k.
The population strength vectors

    t_j = tr(B) a_j,   s_j = tr(A) b_j

have the same fixed total.  The high vector t is recalibrated for each
Delta_pop, whereas the low vector s is calibrated once to

    lambda_{2,k}^{pop} = tau_2,n + low_margin.

Mode-adaptive full estimator
----------------------------
High p-side:
    full correlation -> ACT adjustment using
        rho_{j,n-1} = (p-j) / [q(n-1)]
    -> threshold at tau_1,n.

Low q-side:
    full correlation -> RAW correlation eigenvalues
    -> threshold at tau_2,n = 1 + delta_n.

The default delta_n = n^{-1/4}, valid for the fixed-q polynomial regime used in
the manuscript.

Theory-exact S^2ACT sketch estimator
------------------------------------
BOTH modes are sketched, as in Section 3.3 / Theorem 5.

High p-side:
    sketch pooled columns -> correlation -> ACT adjustment using the ORIGINAL
    q(n-1) aspect ratio -> coordinatewise median -> tau_1,n.

Low q-side:
    sketch pooled rows -> correlation -> RAW eigenvalues -> coordinatewise
    median -> tau_2,n.

Sketch sizes are
    L1 = gamma_1 n q,
    L2 = gamma_2 n p, gamma_2 = 0.20 by default.

Because sampling is with replacement, gamma_1 may exceed one.  Values above
one are approximation diagnostics, not computational-saving configurations.
The sketch fraction gamma changes approximation error only; it does NOT change
ACT's aspect ratio or either decision threshold.

Diagnostics tied to the theory
------------------------------
For each Monte Carlo data set we record

    G1_full = min{ min_{j<=k}(xi_hat_1,j - tau_1,n),
                   tau_1,n - xi_hat_1,k+1 },
    G2_full = min{ min_{j<=k}(xi_hat_2,j - tau_2,n),
                   tau_2,n - xi_hat_2,k+1 },

which are the empirical threshold-separation quantities behind Assumption 8.
We also record the full high-side raw eigengap around the leading k+1
components and the minimum absolute companion Stieltjes-transform value used
in the ACT adjustment, as finite-sample diagnostics related to Assumption 7.

Main outputs
------------
  config.json
  population_settings.csv
  raw_results.csv
  summary_results.csv
  full_sample_summary.csv
  table_case2_full_sample.tex
  table_case2_randomized.tex
  figure_case2_high_accuracy_vs_gamma.{pdf,png}
  figure_case2_pair_accuracy_vs_gamma.{pdf,png}
  figure_case2_full_gap_vs_margin.pdf

Example quick test
------------------
python run_case2_dual_mode_boundary_controlled.py \
  --reps 2 --B 3 --gamma-grid 0.10,0.20 \
  --weak-margins 0.10,0.50 --out-dir ./test_case2

Recommended main run
--------------------
python run_case2_dual_mode_boundary_controlled.py \
  --n 1000 --p 200 --q 10 --k 5 --kmax 8 \
  --factor-total 2 --low-margin 0.15 \
  --weak-margins 0.10,0.20,0.50,1.00,5.00 \
  --gamma-grid 0.030,0.035,0.040,0.045,0.050 \
  --gamma-low-fixed 0.20 --B 15 --reps 100 --n-jobs 5 \
  --out-dir ./results_case2_dual_mode_boundary_controlled
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.linalg import eigvalsh as scipy_eigvalsh
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =============================================================================
# Parsing helpers
# =============================================================================

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(round(float(x))) for x in s.split(",") if x.strip()]


# =============================================================================
# Linear algebra
# =============================================================================

def symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def sorted_eigvals(A: np.ndarray) -> np.ndarray:
    A = symmetrize(np.asarray(A, dtype=float))
    if HAVE_SCIPY:
        vals = scipy_eigvalsh(A, check_finite=False, overwrite_a=False)
    else:
        vals = np.linalg.eigvalsh(A)
    return np.maximum(vals[::-1], 0.0)


def correlation_from_cov(M: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    d = np.maximum(np.diag(M), ridge)
    inv = 1.0 / np.sqrt(d)
    H = inv[:, None] * M * inv[None, :]
    return symmetrize(H)


def prefix_rank(vals: np.ndarray, threshold: float, kmax: int | None = None) -> int:
    """Eq. (15): largest consecutive prefix above the threshold."""
    vals = np.asarray(vals, dtype=float)
    if kmax is None:
        kmax = len(vals)
    kmax = min(int(kmax), len(vals))

    rank = 0
    for x in vals[:kmax]:
        if np.isfinite(x) and x > threshold:
            rank += 1
        else:
            break
    return int(rank)


def max_threshold_rank(vals: np.ndarray, threshold: float, kmax: int) -> int:
    """Full-sample Eq. (13): max{j <= kmax: xi_j > tau}."""
    vals = np.asarray(vals, dtype=float)
    kmax = min(int(kmax), len(vals))
    passed = np.flatnonzero(
        np.isfinite(vals[:kmax]) & (vals[:kmax] > threshold)
    )
    return 0 if len(passed) == 0 else int(passed[-1] + 1)


def eigen_ratio_rank(A: np.ndarray, kmax: int, eps: float = 1e-12) -> int:
    vals = sorted_eigvals(A)
    kmax_eff = min(int(kmax), len(vals) - 1)
    if kmax_eff < 1:
        return 0
    num = np.maximum(vals[:kmax_eff], eps)
    den = np.maximum(vals[1:kmax_eff + 1], eps)
    return int(np.argmax(num / den) + 1)


# =============================================================================
# Thresholds and high-dimensional ACT adjustment
# =============================================================================

def tau_high(n: int, p: int, q: int) -> float:
    return float(1.0 + np.sqrt(p / (q * (n - 1.0))))


def tau_low(n: int, delta_low: float | None = None) -> Tuple[float, float]:
    delta = float(n ** (-0.25) if delta_low is None else delta_low)
    return 1.0 + delta, delta


def adjusted_eigenvalues_high(
    vals: np.ndarray,
    n: int,
    q: int,
    kmax: int,
) -> np.ndarray:
    """
    High-dimensional ACT adjustment under the current matrix-factor theory.

    IMPORTANT: rho_j uses the ORIGINAL pooled effective sample size q(n-1):
        rho_j = (p-j) / [q(n-1)].
    The sketch size L never appears here.
    """
    vals = np.asarray(vals, dtype=float)
    p = len(vals)
    kmax_eff = min(int(kmax), p - 1)
    df = float(q * (n - 1))

    out = np.full(kmax_eff, np.nan, dtype=float)
    for j0 in range(kmax_eff):
        j = j0 + 1
        z = max(float(vals[j0]), 1e-12)
        nxt = max(float(vals[j0 + 1]), 1e-12)

        tail = vals[j0 + 1:]
        denom = tail - z
        denom = np.where(np.abs(denom) < 1e-10, -1e-10, denom)

        stabilizer = (3.0 * z + nxt) / 4.0 - z
        if abs(stabilizer) < 1e-10:
            stabilizer = -1e-10

        mhat = (np.sum(1.0 / denom) + 1.0 / stabilizer) / float(p - j)
        rho_j = (p - j) / df
        mcomp = -(1.0 - rho_j) / z + rho_j * mhat

        out[j0] = np.inf if abs(mcomp) < 1e-12 else -1.0 / mcomp

    return out


def companion_transform_values_high(
    vals: np.ndarray,
    n: int,
    q: int,
    kmax: int,
) -> np.ndarray:
    """Return mbar_{n,j}(lambda_j) values used by the ACT transform."""
    vals = np.asarray(vals, dtype=float)
    p = len(vals)
    kmax_eff = min(int(kmax), p - 1)
    df = float(q * (n - 1))
    out = np.full(kmax_eff, np.nan, dtype=float)

    for j0 in range(kmax_eff):
        j = j0 + 1
        z = max(float(vals[j0]), 1e-12)
        nxt = max(float(vals[j0 + 1]), 1e-12)

        tail = vals[j0 + 1:]
        denom = tail - z
        denom = np.where(np.abs(denom) < 1e-10, -1e-10, denom)

        stabilizer = (3.0 * z + nxt) / 4.0 - z
        if abs(stabilizer) < 1e-10:
            stabilizer = -1e-10

        mhat = (np.sum(1.0 / denom) + 1.0 / stabilizer) / float(p - j)
        rho_j = (p - j) / df
        out[j0] = -(1.0 - rho_j) / z + rho_j * mhat

    return out


# =============================================================================
# Case-1-compatible loadings and dual-mode population calibration
# =============================================================================

def block_pervasive_loading(
    dim: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random dense block-pervasive loading with U^T U = dim I_k."""
    if dim % k != 0:
        raise ValueError(f"dim={dim} must be divisible by k={k}.")

    m = dim // k
    blocks = []
    for _ in range(m):
        Q, _ = np.linalg.qr(rng.normal(size=(k, k)))
        blocks.append(Q.T)

    U = np.sqrt(dim) * np.vstack(blocks) / np.sqrt(m)
    if not np.allclose(U.T @ U, dim * np.eye(k), atol=1e-8):
        raise RuntimeError("Loading normalization failed.")
    return U


def population_correlation_operator(
    U: np.ndarray,
    d_noise: np.ndarray,
    strengths: np.ndarray,
    sigma: float,
) -> np.ndarray:
    strengths = np.asarray(strengths, dtype=float)
    if strengths.ndim != 1 or strengths.shape[0] != U.shape[1]:
        raise ValueError("strengths must have one entry per loading column.")
    if np.any(strengths < 0):
        raise ValueError("Population strengths must be nonnegative.")

    covariance = (
        (U * strengths[None, :]) @ U.T
        + sigma ** 2 * np.diag(np.asarray(d_noise, dtype=float))
    )
    return correlation_from_cov(covariance)


def population_kth_spike(
    U: np.ndarray,
    d_noise: np.ndarray,
    strengths: np.ndarray,
    sigma: float,
    k: int,
) -> float:
    vals = sorted_eigvals(
        population_correlation_operator(U, d_noise, strengths, sigma)
    )
    return float(vals[k - 1])


def boundary_strength_vector(
    total: float,
    weak_strength: float,
    k: int,
) -> np.ndarray:
    """Keep total signal fixed; assign one weak and k-1 equal strong factors."""
    if k < 2:
        raise ValueError("Boundary calibration requires k >= 2.")
    if total <= 0:
        raise ValueError("factor_total must be positive.")
    if not 0.0 <= weak_strength <= total / k:
        raise ValueError("weak_strength must lie in [0, factor_total/k].")

    strong = (total - weak_strength) / float(k - 1)
    return np.concatenate([
        np.full(k - 1, strong, dtype=float),
        np.array([weak_strength], dtype=float),
    ])


def calibrate_boundary_strengths(
    U: np.ndarray,
    d_noise: np.ndarray,
    target: float,
    sigma: float,
    k: int,
    factor_total: float,
    tol: float = 1e-10,
) -> np.ndarray:
    """Calibrate the weakest strength while holding the total fixed."""
    if target <= 1.0:
        raise ValueError("Population kth-spike target must exceed one.")

    lo = 0.0
    hi = factor_total / float(k)
    lo_spike = population_kth_spike(
        U, d_noise, boundary_strength_vector(factor_total, lo, k), sigma, k
    )
    hi_spike = population_kth_spike(
        U, d_noise, boundary_strength_vector(factor_total, hi, k), sigma, k
    )
    if not lo_spike <= target <= hi_spike:
        raise ValueError(
            f"Target {target:.8f} is infeasible for factor_total="
            f"{factor_total:.8f}; attainable kth-spike range is "
            f"[{lo_spike:.8f}, {hi_spike:.8f}]."
        )

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        strengths = boundary_strength_vector(factor_total, mid, k)
        spike = population_kth_spike(U, d_noise, strengths, sigma, k)
        if spike < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, factor_total):
            break

    strengths = boundary_strength_vector(
        factor_total, 0.5 * (lo + hi), k
    )
    achieved = population_kth_spike(U, d_noise, strengths, sigma, k)
    if abs(achieved - target) > 1e-6 * max(1.0, abs(target)):
        raise RuntimeError(
            f"Calibration failed: target={target}, achieved={achieved}."
        )
    return strengths


def generate_from_common_random_numbers(
    R: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
    W: np.ndarray,
    high_strengths: np.ndarray,
    low_strengths: np.ndarray,
    factor_total: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one paired data set from calibrated high/low strengths."""
    sqrt_total = np.sqrt(factor_total)
    a_diag = np.asarray(high_strengths, dtype=float) / sqrt_total
    b_diag = np.asarray(low_strengths, dtype=float) / sqrt_total

    if not np.isclose(a_diag.sum(), sqrt_total, atol=1e-9):
        raise RuntimeError("A calibration failed its trace identity.")
    if not np.isclose(b_diag.sum(), sqrt_total, atol=1e-9):
        raise RuntimeError("B calibration failed its trace identity.")

    F = (
        np.sqrt(a_diag)[None, :, None]
        * G
        * np.sqrt(b_diag)[None, None, :]
    )
    signal = np.einsum("pr,nrs,qs->npq", R, F, C, optimize=True)
    Z = signal + sigma * W
    Z = Z - Z.mean(axis=0, keepdims=True)
    return Z, a_diag, b_diag


def full_operators(
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, p, q = Z.shape
    U_high = Z.transpose(0, 2, 1).reshape(n * q, p)
    U_low = Z.reshape(n * p, q)

    M1 = (U_high.T @ U_high) / float(n * q)
    M2 = (U_low.T @ U_low) / float(n * p)
    H1 = correlation_from_cov(M1)
    H2 = correlation_from_cov(M2)
    return symmetrize(M1), symmetrize(M2), H1, H2, U_high, U_low


# =============================================================================
# Nested sketch banks
# =============================================================================

def build_nested_high_sketch_bank(
    U_high: np.ndarray,
    L_values: Sequence[int],
    B: int,
    n: int,
    q: int,
    kmax: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    High-side sketch bank.

    bank[L] has shape (B, kmax_eff) and stores ACT-adjusted eigenvalues.
    ACT always uses q(n-1), never L.
    """
    pool_size, p = U_high.shape
    L_values_sorted = sorted(set(int(x) for x in L_values))
    if not L_values_sorted:
        raise ValueError("Empty high-side L grid.")
    if min(L_values_sorted) < 1:
        raise ValueError("High-side sketch sizes must be positive.")

    L_max = max(L_values_sorted)
    rng = np.random.default_rng(seed)
    ids_bank = rng.integers(
        0, pool_size, size=(B, L_max), dtype=np.int64
    )

    kmax_eff = min(int(kmax), p - 1)
    banks = {
        L: np.empty((B, kmax_eff), dtype=float)
        for L in L_values_sorted
    }

    for b in range(B):
        gram = np.zeros((p, p), dtype=float)
        start = 0
        ids_stream = ids_bank[b]

        for L in L_values_sorted:
            new_ids = ids_stream[start:L]
            block = U_high[new_ids]
            gram += block.T @ block

            M = gram / float(L)
            H = correlation_from_cov(M)
            vals = sorted_eigvals(H)
            banks[L][b, :] = adjusted_eigenvalues_high(
                vals=vals,
                n=n,
                q=q,
                kmax=kmax_eff,
            )
            start = L

    return banks


def build_nested_low_sketch_bank(
    U_low: np.ndarray,
    L_values: Sequence[int],
    B: int,
    kmax: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    Low-side sketch bank.

    bank[L] stores RAW correlation eigenvalues. No ACT correction is applied.
    """
    pool_size, q = U_low.shape
    L_values_sorted = sorted(set(int(x) for x in L_values))
    if not L_values_sorted:
        raise ValueError("Empty low-side L grid.")
    if min(L_values_sorted) < 1:
        raise ValueError("Low-side sketch sizes must be positive.")

    L_max = max(L_values_sorted)
    rng = np.random.default_rng(seed)
    ids_bank = rng.integers(
        0, pool_size, size=(B, L_max), dtype=np.int64
    )

    kmax_eff = min(int(kmax), q)
    banks = {
        L: np.empty((B, kmax_eff), dtype=float)
        for L in L_values_sorted
    }

    for b in range(B):
        gram = np.zeros((q, q), dtype=float)
        start = 0
        ids_stream = ids_bank[b]

        for L in L_values_sorted:
            new_ids = ids_stream[start:L]
            block = U_low[new_ids]
            gram += block.T @ block

            M = gram / float(L)
            H = correlation_from_cov(M)
            vals = sorted_eigvals(H)
            banks[L][b, :] = vals[:kmax_eff]
            start = L

    return banks


# =============================================================================
# One Monte Carlo replication
# =============================================================================

def run_one_replication(rep: int, cfg: Dict[str, object]):
    n = int(cfg["n"])
    p = int(cfg["p"])
    q = int(cfg["q"])
    k = int(cfg["k"])
    kmax = int(cfg["kmax"])
    sigma = float(cfg["sigma"])

    weak_margins = [float(x) for x in cfg["weak_margins"]]
    gamma_grid = [float(x) for x in cfg["gamma_grid"]]
    gamma_low_fixed = float(cfg["gamma_low_fixed"])
    factor_total = float(cfg["factor_total"])
    low_target = float(cfg["low_target"])
    B = int(cfg["B"])
    seed0 = int(cfg["seed"])
    tau_H = float(cfg["tau_high"])
    tau_L = float(cfg["tau_low"])

    # Common loadings, latent innovations, and noise across all Delta_pop.
    data_seed = seed0 + 10_000_000 * rep + 11
    rng = np.random.default_rng(data_seed)
    R = block_pervasive_loading(p, k, rng)
    C = block_pervasive_loading(q, k, rng)
    dp = np.ones(p, dtype=float)  # Case 2 fixes h=0.
    dq = np.ones(q, dtype=float)

    low_strengths = calibrate_boundary_strengths(
        U=C,
        d_noise=dq,
        target=low_target,
        sigma=sigma,
        k=k,
        factor_total=factor_total,
    )
    pop_low = population_kth_spike(
        C, dq, low_strengths, sigma, k
    )

    G = rng.normal(size=(n, k, k))
    W = rng.normal(size=(n, p, q))

    pool_high = n * q
    pool_low = n * p
    L1_values = [max(1, int(round(g * pool_high))) for g in gamma_grid]
    fixed_L2 = max(1, int(round(gamma_low_fixed * pool_low)))
    L2_values = [fixed_L2 for _ in gamma_grid]

    randomized_rows: List[Dict[str, object]] = []
    full_rows: List[Dict[str, object]] = []

    for margin in weak_margins:
        weak_target = tau_H + margin
        high_strengths = calibrate_boundary_strengths(
            U=R,
            d_noise=dp,
            target=weak_target,
            sigma=sigma,
            k=k,
            factor_total=factor_total,
        )
        pop_high = population_kth_spike(
            R, dp, high_strengths, sigma, k
        )

        Z, a_diag, b_diag = generate_from_common_random_numbers(
            R=R,
            C=C,
            G=G,
            W=W,
            high_strengths=high_strengths,
            low_strengths=low_strengths,
            factor_total=factor_total,
            sigma=sigma,
        )

        M1, M2, H1, H2, U_high, U_low = full_operators(Z)
        vals1 = sorted_eigvals(H1)
        vals2 = sorted_eigvals(H2)
        adj1 = adjusted_eigenvalues_high(vals1, n=n, q=q, kmax=kmax)

        # Full-sample mode-adaptive estimator.
        full_r = max_threshold_rank(adj1, tau_H, kmax)
        full_c = max_threshold_rank(vals2, tau_L, kmax)

        # Full benchmark ER methods (useful context, not the main Case-2 target).
        cov_r = eigen_ratio_rank(M1, kmax)
        cov_c = eigen_ratio_rank(M2, kmax)
        corr_r = eigen_ratio_rank(H1, kmax)
        corr_c = eigen_ratio_rank(H2, kmax)

        # Assumption-8 empirical threshold separations.
        if len(adj1) <= k:
            raise RuntimeError("Need high adjusted eigenvalues through k+1.")
        if len(vals2) <= k:
            raise RuntimeError("Need low raw eigenvalues through k+1.")

        high_signal_margin = float(np.min(adj1[:k] - tau_H))
        high_noise_margin = float(tau_H - adj1[k])
        G1_full = min(high_signal_margin, high_noise_margin)

        low_signal_margin = float(np.min(vals2[:k] - tau_L))
        low_noise_margin = float(tau_L - vals2[k])
        G2_full = min(low_signal_margin, low_noise_margin)

        # Assumption-7 finite-sample diagnostics.
        # min_{j<=k+1}(lambda_j-lambda_{j+1}) needs raw eigenvalues through k+2.
        if len(vals1) >= k + 2:
            raw_high_gap = float(np.min(vals1[:k + 1] - vals1[1:k + 2]))
        else:
            raw_high_gap = np.nan

        mcomp = companion_transform_values_high(
            vals1,
            n=n,
            q=q,
            kmax=min(k + 1, len(vals1) - 1),
        )
        mcomp_abs_min = float(np.nanmin(np.abs(mcomp[:k + 1])))

        full_rows.append({
            "rep": rep,
            "weak_margin": margin,
            "weak_target": weak_target,
            "factor_total": factor_total,
            "pop_high_k": pop_high,
            "pop_low_k": pop_low,
            "high_strength_min": float(np.min(high_strengths)),
            "high_strength_max": float(np.max(high_strengths)),
            "low_strength_min": float(np.min(low_strengths)),
            "low_strength_max": float(np.max(low_strengths)),
            "high_strengths": json.dumps(high_strengths.tolist()),
            "low_strengths": json.dumps(low_strengths.tolist()),
            "a_diag": json.dumps(a_diag.tolist()),
            "b_diag": json.dumps(b_diag.tolist()),
            "tau_high": tau_H,
            "tau_low": tau_L,
            "full_row_rank": full_r,
            "full_col_rank": full_c,
            "full_row_correct": int(full_r == k),
            "full_col_correct": int(full_c == k),
            "full_pair_correct": int(full_r == k and full_c == k),
            "cov_er_row_rank": cov_r,
            "cov_er_col_rank": cov_c,
            "cov_er_pair_correct": int(cov_r == k and cov_c == k),
            "corr_er_row_rank": corr_r,
            "corr_er_col_rank": corr_c,
            "corr_er_pair_correct": int(corr_r == k and corr_c == k),
            "full_high_adj_k": float(adj1[k - 1]),
            "full_high_adj_next": float(adj1[k]),
            "full_low_raw_k": float(vals2[k - 1]),
            "full_low_raw_next": float(vals2[k]),
            "G1_full": float(G1_full),
            "G2_full": float(G2_full),
            "high_signal_margin_full": high_signal_margin,
            "high_noise_margin_full": high_noise_margin,
            "low_signal_margin_full": low_signal_margin,
            "low_noise_margin_full": low_noise_margin,
            "G1_positive": int(G1_full > 0.0),
            "G2_positive": int(G2_full > 0.0),
            "raw_high_eigengap_min_j_le_kplus1": raw_high_gap,
            "mcomp_abs_min_j_le_kplus1": mcomp_abs_min,
        })

        high_banks = build_nested_high_sketch_bank(
            U_high=U_high,
            L_values=L1_values,
            B=B,
            n=n,
            q=q,
            kmax=kmax,
            # Same sampled indices across Delta_pop within a replication.
            seed=seed0 + 10_000_000 * rep + 101,
        )
        low_banks = build_nested_low_sketch_bank(
            U_low=U_low,
            L_values=sorted(set(L2_values)),
            B=B,
            kmax=kmax,
            seed=seed0 + 10_000_000 * rep + 202,
        )

        for gamma, L1, L2 in zip(gamma_grid, L1_values, L2_values):
            high_bank = high_banks[L1]
            low_bank = low_banks[L2]

            A1 = np.nanmedian(high_bank, axis=0)
            A2 = np.nanmedian(low_bank, axis=0)

            s2_r = prefix_rank(A1, tau_H, kmax)
            s2_c = prefix_rank(A2, tau_L, kmax)

            randomized_rows.append({
                "rep": rep,
                "weak_margin": margin,
                "weak_target": weak_target,
                "gamma": gamma,
                "L1_high": L1,
                "L2_low": L2,
                "B": B,
                "tau_high": tau_H,
                "tau_low": tau_L,
                "factor_total": factor_total,
                "pop_high_k": pop_high,
                "pop_low_k": pop_low,
                "row_rank": s2_r,
                "col_rank": s2_c,
                "row_correct": int(s2_r == k),
                "col_correct": int(s2_c == k),
                "pair_correct": int(s2_r == k and s2_c == k),
                "row_over": int(s2_r > k),
                "row_under": int(s2_r < k),
                "col_over": int(s2_c > k),
                "col_under": int(s2_c < k),
                "row_A_k_adj": float(A1[k - 1]),
                "row_A_next_adj": float(A1[k]) if k < len(A1) else np.nan,
                "col_A_k_raw": float(A2[k - 1]),
                "col_A_next_raw": float(A2[k]) if k < len(A2) else np.nan,
                "full_row_rank": full_r,
                "full_col_rank": full_c,
                "full_row_correct": int(full_r == k),
                "full_col_correct": int(full_c == k),
                "full_pair_correct": int(full_r == k and full_c == k),
                "row_agree_full": int(s2_r == full_r),
                "col_agree_full": int(s2_c == full_c),
                "pair_agree_full": int(s2_r == full_r and s2_c == full_c),
                "full_high_adj_k": float(adj1[k - 1]),
                "full_high_adj_next": float(adj1[k]),
                "full_low_raw_k": float(vals2[k - 1]),
                "full_low_raw_next": float(vals2[k]),
                "G1_full": float(G1_full),
                "G2_full": float(G2_full),
                "G1_positive": int(G1_full > 0.0),
                "G2_positive": int(G2_full > 0.0),
                "raw_high_eigengap_min_j_le_kplus1": raw_high_gap,
                "mcomp_abs_min_j_le_kplus1": mcomp_abs_min,
            })

    return randomized_rows, full_rows


# =============================================================================
# Summaries
# =============================================================================

def standard_error(series: pd.Series) -> float:
    count = int(series.notna().sum())
    if count <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(count))


def summarize_randomized(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "weak_margin", "weak_target", "gamma", "L1_high", "L2_low", "B"
    ]
    for keys, group in raw.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        # Explicit assignments keep both SD diagnostics and table-ready SEs.
        row.update({
            "row_accuracy": float(group["row_correct"].mean()),
            "row_accuracy_se": standard_error(group["row_correct"]),
            "col_accuracy": float(group["col_correct"].mean()),
            "col_accuracy_se": standard_error(group["col_correct"]),
            "pair_accuracy": float(group["pair_correct"].mean()),
            "pair_accuracy_se": standard_error(group["pair_correct"]),
            "row_average_rank": float(group["row_rank"].mean()),
            "row_rank_se": standard_error(group["row_rank"]),
            "col_average_rank": float(group["col_rank"].mean()),
            "col_rank_se": standard_error(group["col_rank"]),
            "row_over_rate": float(group["row_over"].mean()),
            "row_under_rate": float(group["row_under"].mean()),
            "col_over_rate": float(group["col_over"].mean()),
            "col_under_rate": float(group["col_under"].mean()),
            "row_full_agreement": float(group["row_agree_full"].mean()),
            "col_full_agreement": float(group["col_agree_full"].mean()),
            "pair_full_agreement": float(group["pair_agree_full"].mean()),
            "full_row_accuracy": float(group["full_row_correct"].mean()),
            "full_col_accuracy": float(group["full_col_correct"].mean()),
            "full_pair_accuracy": float(group["full_pair_correct"].mean()),
            "row_A_k_adj_mean": float(group["row_A_k_adj"].mean()),
            "row_A_k_adj_sd": float(group["row_A_k_adj"].std(ddof=1)),
            "row_A_next_adj_mean": float(group["row_A_next_adj"].mean()),
            "row_A_next_adj_sd": float(group["row_A_next_adj"].std(ddof=1)),
            "col_A_k_raw_mean": float(group["col_A_k_raw"].mean()),
            "col_A_k_raw_sd": float(group["col_A_k_raw"].std(ddof=1)),
            "col_A_next_raw_mean": float(group["col_A_next_raw"].mean()),
            "col_A_next_raw_sd": float(group["col_A_next_raw"].std(ddof=1)),
            "full_high_adj_k_mean": float(group["full_high_adj_k"].mean()),
            "full_high_adj_next_mean": float(group["full_high_adj_next"].mean()),
            "full_low_raw_k_mean": float(group["full_low_raw_k"].mean()),
            "full_low_raw_next_mean": float(group["full_low_raw_next"].mean()),
            "G1_full_mean": float(group["G1_full"].mean()),
            "G1_full_sd": float(group["G1_full"].std(ddof=1)),
            "G2_full_mean": float(group["G2_full"].mean()),
            "G2_full_sd": float(group["G2_full"].std(ddof=1)),
            "G1_positive_rate": float(group["G1_positive"].mean()),
            "G2_positive_rate": float(group["G2_positive"].mean()),
            "raw_high_eigengap_mean": float(
                group["raw_high_eigengap_min_j_le_kplus1"].mean()
            ),
            "mcomp_abs_min_mean": float(
                group["mcomp_abs_min_j_le_kplus1"].mean()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["weak_margin", "gamma"]
    ).reset_index(drop=True)


def summarize_full(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (margin, target), group in full.groupby(
        ["weak_margin", "weak_target"], sort=True
    ):
        rows.append({
            "weak_margin": float(margin),
            "weak_target": float(target),
            "factor_total": float(group["factor_total"].mean()),
            "pop_high_k_mean": float(group["pop_high_k"].mean()),
            "pop_low_k_mean": float(group["pop_low_k"].mean()),
            "full_row_accuracy": float(group["full_row_correct"].mean()),
            "full_row_accuracy_se": standard_error(group["full_row_correct"]),
            "full_col_accuracy": float(group["full_col_correct"].mean()),
            "full_pair_accuracy": float(group["full_pair_correct"].mean()),
            "cov_er_pair_accuracy": float(group["cov_er_pair_correct"].mean()),
            "corr_er_pair_accuracy": float(group["corr_er_pair_correct"].mean()),
            "full_row_average_rank": float(group["full_row_rank"].mean()),
            "full_row_rank_se": standard_error(group["full_row_rank"]),
            "full_col_average_rank": float(group["full_col_rank"].mean()),
            "full_high_adj_k_mean": float(group["full_high_adj_k"].mean()),
            "full_high_adj_k_se": standard_error(group["full_high_adj_k"]),
            "full_high_adj_next_mean": float(group["full_high_adj_next"].mean()),
            "full_high_adj_next_se": standard_error(group["full_high_adj_next"]),
            "full_low_raw_k_mean": float(group["full_low_raw_k"].mean()),
            "full_low_raw_next_mean": float(group["full_low_raw_next"].mean()),
            "G1_full_mean": float(group["G1_full"].mean()),
            "G1_full_sd": float(group["G1_full"].std(ddof=1)),
            "G2_full_mean": float(group["G2_full"].mean()),
            "G2_full_sd": float(group["G2_full"].std(ddof=1)),
            "G1_positive_rate": float(group["G1_positive"].mean()),
            "G2_positive_rate": float(group["G2_positive"].mean()),
            "raw_high_eigengap_mean": float(
                group["raw_high_eigengap_min_j_le_kplus1"].mean()
            ),
            "mcomp_abs_min_mean": float(
                group["mcomp_abs_min_j_le_kplus1"].mean()
            ),
        })
    return pd.DataFrame(rows).sort_values("weak_margin").reset_index(drop=True)


def make_full_sample_table_latex(full_summary: pd.DataFrame) -> str:
    """Table: full-sample adjusted high eigenvalues and rank recovery."""
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{Full-sample adjusted eigenvalues and rank estimation "
            r"across $\Delta_{\mathrm{pop}}$. Standard errors are reported "
            r"in parentheses.}"
        ),
        r"\label{tab:case2_full}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        (
            r"$\Delta_{\mathrm{pop}}$ & "
            r"$\widehat{\lambda}_{1,k}^{\mathrm{adj}}$ & "
            r"$\widehat{\lambda}_{1,k+1}^{\mathrm{adj}}$ & "
            r"$\widehat{k}_1$ & Acc. \\"
        ),
        r"\midrule",
    ]
    for _, row in full_summary.iterrows():
        lines.append(
            f"{row['weak_margin']:.2f} & "
            f"{row['full_high_adj_k_mean']:.2f} "
            f"({row['full_high_adj_k_se']:.3f}) & "
            f"{row['full_high_adj_next_mean']:.2f} "
            f"({row['full_high_adj_next_se']:.3f}) & "
            f"{row['full_row_average_rank']:.2f} "
            f"({row['full_row_rank_se']:.3f}) & "
            f"{row['full_row_accuracy']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def make_randomized_table_latex(summary: pd.DataFrame) -> str:
    """Table: high-mode randomized rank estimates across margin and gamma."""
    gammas = sorted(float(x) for x in summary["gamma"].unique())
    colspec = "c" + "cc" * len(gammas)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (
            r"\caption{High-dimensional rank estimates obtained by the "
            r"randomized median-aggregated estimator across "
            r"$\Delta_{\mathrm{pop}}$ and high-mode sketching ratios "
            r"$\gamma_1$. Standard errors are reported in parentheses.}"
        ),
        r"\label{tab:subsample_RE_case2}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    header = [""] + [
        rf"\multicolumn{{2}}{{c}}{{$\gamma_1={gamma:g}$}}"
        for gamma in gammas
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(" ".join(
        rf"\cmidrule(lr){{{2 + 2*j}-{3 + 2*j}}}"
        for j in range(len(gammas))
    ))
    second = [r"$\Delta_{\mathrm{pop}}$"]
    for _ in gammas:
        second.extend([r"$\widetilde{k}_1^A$", "Acc."])
    lines.append(" & ".join(second) + r" \\")
    lines.append(r"\midrule")

    for margin in sorted(float(x) for x in summary["weak_margin"].unique()):
        cells = [f"{margin:.2f}"]
        for gamma in gammas:
            hit = summary.loc[
                np.isclose(summary["weak_margin"], margin)
                & np.isclose(summary["gamma"], gamma)
            ]
            if hit.empty:
                cells.extend(["--", "--"])
            else:
                row = hit.iloc[0]
                cells.extend([
                    f"{row['row_average_rank']:.2f} "
                    f"({row['row_rank_se']:.2f})",
                    f"{row['row_accuracy']:.2f}",
                ])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


# =============================================================================
# Plots
# =============================================================================

def _plot_accuracy_by_margin(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for margin in sorted(summary["weak_margin"].unique()):
        df = summary[summary["weak_margin"] == margin].sort_values("gamma")
        ax.plot(
            df["gamma"],
            df[metric],
            marker="o",
            linewidth=2.0,
            label=fr"$\Delta_{{\rm pop}}={margin:g}$",
        )
    ax.set_xlabel(r"High-dimensional sketch ratio $\gamma_1=L_1/(nq)$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_gap(full_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    x = full_summary["weak_margin"].to_numpy(float)
    y1 = full_summary["G1_full_mean"].to_numpy(float)
    y2 = full_summary["G2_full_mean"].to_numpy(float)
    ax.plot(x, y1, marker="o", linewidth=2.0, label=r"High mode $G_1^{\rm full}$")
    ax.plot(x, y2, marker="s", linewidth=2.0, label=r"Low mode $G_2^{\rm full}$")
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Population weak-factor margin $\Delta_{\rm pop}$")
    ax.set_ylabel("Mean full-sample threshold separation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_sketch_full_agreement(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for margin in sorted(summary["weak_margin"].unique()):
        df = (
            summary[summary["weak_margin"] == margin]
            .sort_values("gamma")
        )

        ax.plot(
            df["gamma"],
            df["pair_full_agreement"],
            marker="o",
            linewidth=2.0,
            markersize=6.0,
            label=fr"$\Delta_{{\rm pop}}={margin:g}$",
        )

    gamma_ticks = sorted(summary["gamma"].unique())

    ax.set_xticks(gamma_ticks)
    ax.set_xticklabels([f"{g:g}" for g in gamma_ticks])

    ax.set_xlabel(
        r"High-dimensional sketch ratio "
        r"$\gamma_H=L_1/(nq)$"
    )
    ax.set_ylabel(
        "Sketch-to-full rank-pair agreement probability"
    )

    ax.set_ylim(-0.03, 1.05)

    # Optional reference line showing perfect agreement.
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def save_plots(summary: pd.DataFrame, full_summary: pd.DataFrame, out_dir: Path) -> None:
    _plot_accuracy_by_margin(
        summary,
        "row_accuracy",
        "High-mode exact rank recovery probability",
        out_dir / "figure_case2_high_accuracy_vs_gamma",
    )
    _plot_accuracy_by_margin(
        summary,
        "col_accuracy",
        "Low-mode exact rank recovery probability",
        out_dir / "figure_case2_low_accuracy_vs_gamma",
    )
    _plot_accuracy_by_margin(
        summary,
        "pair_accuracy",
        "Exact rank-pair recovery probability",
        out_dir / "figure_case2_pair_accuracy_vs_gamma",
    )
    plot_sketch_full_agreement(
        summary,
        out_dir / "figure_case2_sketch_full_agreement_vs_gamma",
    )
    plot_full_gap(
        full_summary,
        out_dir / "figure_case2_full_gap_vs_margin",
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Theory-exact Case 2: calibrated weak high-side factors under the "
            "partially high-dimensional S2ACT framework."
        )
    )

    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=1.0)

    ap.add_argument(
        "--weak-margins",
        type=parse_float_list,
        default=parse_float_list("0.10,0.25,0.50,1.00,5.00"),
        help=(
            "Delta_pop grid in lambda_{1,k}^{pop}=tau_high+Delta_pop."
        ),
    )
    ap.add_argument(
        "--gamma-grid",
        type=parse_float_list,
        default=parse_float_list("0.25,0.50,0.75,1.00,1.25,1.50"),
        help=(
            "High-mode sketch ratios. Ratios above one are valid under "
            "with-replacement sampling and serve as approximation diagnostics."
        ),
    )
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument(
        "--gamma-low-fixed",
        type=float,
        default=0.20,
        help=(
            "Fixed low-mode sketch fraction L2/(np). Case 2 varies only the "
            "high-mode sketch budget."
        ),
    )
    ap.add_argument(
        "--factor-total",
        type=float,
        default=2.0,
        help=(
            "Fixed common total of high- and low-mode population strengths."
        ),
    )
    ap.add_argument(
        "--low-margin",
        type=float,
        default=0.15,
        help="Fixed low-mode population separation above tau_low.",
    )

    ap.add_argument(
        "--delta-low",
        type=float,
        default=None,
        help=(
            "Low-dimensional threshold buffer delta_n. Default n^{-1/4}; "
            "appropriate for the fixed-q regime used here."
        ),
    )

    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument(
        "--out-dir",
        default="./results_case2_dual_mode_boundary_controlled",
    )
    ap.add_argument("--no-plots", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.n <= 1:
        raise ValueError("n must exceed 1.")
    if args.p % args.k != 0:
        raise ValueError(f"p={args.p} must be divisible by k={args.k}.")
    if args.q % args.k != 0:
        raise ValueError(f"q={args.q} must be divisible by k={args.k}.")
    if args.kmax < args.k + 1:
        raise ValueError("kmax must be at least k+1 for boundary diagnostics.")
    if args.B < 1:
        raise ValueError("B must be positive.")
    if args.factor_total <= 0:
        raise ValueError("factor-total must be positive.")
    if args.low_margin <= 0:
        raise ValueError("low-margin must be positive.")
    if args.gamma_low_fixed <= 0:
        raise ValueError("gamma-low-fixed must be positive.")

    weak_margins = sorted(set(float(x) for x in args.weak_margins))
    gamma_grid = sorted(set(float(x) for x in args.gamma_grid))
    if not weak_margins or min(weak_margins) <= 0:
        raise ValueError("weak-margins must be positive to stay above the ACT boundary.")
    if not gamma_grid or min(gamma_grid) <= 0:
        raise ValueError("gamma-grid must contain positive values.")

    tau_H = tau_high(args.n, args.p, args.q)
    tau_L, delta_used = tau_low(args.n, args.delta_low)
    rho = args.p / float(args.q * (args.n - 1))
    delta_condition_index = np.sqrt(args.n) / float(args.q) * delta_used

    low_target = tau_L + args.low_margin

    pool_high = args.n * args.q
    pool_low = args.n * args.p
    L1_values = [max(1, int(round(g * pool_high))) for g in gamma_grid]
    fixed_L2 = max(1, int(round(args.gamma_low_fixed * pool_low)))
    L2_values = [fixed_L2 for _ in gamma_grid]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({
        "weak_margins": weak_margins,
        "gamma_grid": gamma_grid,
        "tau_high": tau_H,
        "tau_low": tau_L,
        "delta_low_used": delta_used,
        "rho_p_over_q_nminus1": rho,
        "q_over_sqrt_n": args.q / np.sqrt(args.n),
        "delta_condition_index_sqrt_n_over_q_times_delta": delta_condition_index,
        "pool_high_nq": pool_high,
        "pool_low_np": pool_low,
        "L1_values": L1_values,
        "L2_values": L2_values,
        "gamma_low_fixed": args.gamma_low_fixed,
        "h": 0.0,
        "factor_total": args.factor_total,
        "low_margin": args.low_margin,
        "low_target": low_target,
        "paired_across_weak_margins": True,
        "with_replacement_sketching": True,
    })

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print("========== Case 2: dual-mode boundary-controlled experiment ==========")
    print(f"n={args.n}, p={args.p}, q={args.q}, k={args.k}, kmax={args.kmax}")
    print(f"p/[q(n-1)]={rho:.6f}; q/sqrt(n)={args.q/np.sqrt(args.n):.6f}")
    print(f"tau_high={tau_H:.6f}")
    print(f"tau_low=1+delta_n={tau_L:.6f}; delta_n={delta_used:.6f}")
    print(f"sqrt(n)/q * delta_n={delta_condition_index:.6f}")
    print(f"fixed h=0, factor_total={args.factor_total:g}")
    print(f"fixed low target=tau_low+{args.low_margin:g}={low_target:.6f}")
    print(f"weak margins={weak_margins}")
    print(f"gamma grid={gamma_grid}")
    print(f"L1={L1_values}")
    print(f"L2={L2_values}")
    print(f"B={args.B}, reps={args.reps}, n_jobs={args.n_jobs}")
    print("R, C, G, W, and sketch indices are paired across weak margins.")

    cfg = {
        "n": args.n,
        "p": args.p,
        "q": args.q,
        "k": args.k,
        "kmax": args.kmax,
        "sigma": args.sigma,
        "weak_margins": weak_margins,
        "gamma_grid": gamma_grid,
        "gamma_low_fixed": args.gamma_low_fixed,
        "factor_total": args.factor_total,
        "low_target": low_target,
        "B": args.B,
        "seed": args.seed,
        "tau_high": tau_H,
        "tau_low": tau_L,
    }

    all_randomized: List[Dict[str, object]] = []
    all_full: List[Dict[str, object]] = []
    t0 = time.time()

    if args.n_jobs == 1:
        for rep in range(args.reps):
            rt = time.time()
            random_rows, full_rows = run_one_replication(rep, cfg)
            all_randomized.extend(random_rows)
            all_full.extend(full_rows)
            print(
                f"rep {rep + 1:03d}/{args.reps:03d} finished in "
                f"{time.time() - rt:.2f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futures = {
                ex.submit(run_one_replication, rep, cfg): rep
                for rep in range(args.reps)
            }
            done = 0
            for fut in as_completed(futures):
                rep = futures[fut]
                random_rows, full_rows = fut.result()
                all_randomized.extend(random_rows)
                all_full.extend(full_rows)
                done += 1
                print(
                    f"rep {rep + 1:03d}/{args.reps:03d} completed "
                    f"({done}/{args.reps})",
                    flush=True,
                )

    raw = pd.DataFrame(all_randomized)
    full = pd.DataFrame(all_full)

    raw = raw.sort_values(["weak_margin", "gamma", "rep"]).reset_index(drop=True)
    full = full.sort_values(["weak_margin", "rep"]).reset_index(drop=True)

    summary = summarize_randomized(raw)
    full_summary = summarize_full(full)

    population_settings = (
        full.groupby(["weak_margin", "weak_target"], as_index=False)
        .agg(
            factor_total=("factor_total", "mean"),
            population_high_k=("pop_high_k", "mean"),
            population_low_k=("pop_low_k", "mean"),
            high_strength_min=("high_strength_min", "mean"),
            high_strength_max=("high_strength_max", "mean"),
            low_strength_min=("low_strength_min", "mean"),
            low_strength_max=("low_strength_max", "mean"),
        )
        .sort_values("weak_margin")
        .reset_index(drop=True)
    )
    population_settings["population_high_calibration_error"] = (
        population_settings["population_high_k"]
        - population_settings["weak_target"]
    )
    population_settings["population_low_target"] = low_target
    population_settings["population_low_calibration_error"] = (
        population_settings["population_low_k"] - low_target
    )

    raw.to_csv(out_dir / "raw_results.csv", index=False)
    full.to_csv(out_dir / "full_sample_results.csv", index=False)
    summary.to_csv(out_dir / "summary_results.csv", index=False)
    full_summary.to_csv(out_dir / "full_sample_summary.csv", index=False)
    population_settings.to_csv(out_dir / "population_settings.csv", index=False)

    (out_dir / "table_case2_full_sample.tex").write_text(
        make_full_sample_table_latex(full_summary), encoding="utf-8"
    )
    (out_dir / "table_case2_randomized.tex").write_text(
        make_randomized_table_latex(summary), encoding="utf-8"
    )

    if not args.no_plots:
        save_plots(summary, full_summary, out_dir)

    print("\n========== Full-sample summary ==========")
    print(
        full_summary[
            [
                "weak_margin",
                "full_row_accuracy",
                "full_col_accuracy",
                "full_pair_accuracy",
                "full_high_adj_k_mean",
                "full_high_adj_next_mean",
                "G1_full_mean",
                "G1_positive_rate",
                "G2_full_mean",
                "G2_positive_rate",
            ]
        ].to_string(index=False)
    )

    print("\n========== S2ACT summary ==========")
    print(
        summary[
            [
                "weak_margin",
                "gamma",
                "L1_high",
                "L2_low",
                "row_accuracy",
                "col_accuracy",
                "pair_accuracy",
                "pair_full_agreement",
                "G1_full_mean",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {out_dir / 'config.json'}")
    print(f"  {out_dir / 'population_settings.csv'}")
    print(f"  {out_dir / 'raw_results.csv'}")
    print(f"  {out_dir / 'full_sample_results.csv'}")
    print(f"  {out_dir / 'summary_results.csv'}")
    print(f"  {out_dir / 'full_sample_summary.csv'}")
    print(f"  {out_dir / 'table_case2_full_sample.tex'}")
    print(f"  {out_dir / 'table_case2_randomized.tex'}")
    if not args.no_plots:
        for name in [
            "figure_case2_high_accuracy_vs_gamma.pdf",
            "figure_case2_low_accuracy_vs_gamma.pdf",
            "figure_case2_pair_accuracy_vs_gamma.pdf",
            "figure_case2_sketch_full_agreement_vs_gamma.pdf",
            "figure_case2_full_gap_vs_margin.pdf",
        ]:
            print(f"  {out_dir / name}")

    print(f"\nTotal time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
