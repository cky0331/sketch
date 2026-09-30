"""
run_case1_dual_mode_boundary_controlled.py

Theory-aligned Case 1 for the partially high-dimensional S^2ACT framework.

High-dimensional p-side (s=1):
    - sketch L1 pooled p-dimensional column vectors with replacement;
    - form the sketched correlation matrix;
    - apply ACT eigenvalue bias correction;
    - aggregate adjusted eigenvalues coordinatewise by the median;
    - compare with tau_1 = 1 + sqrt{p/[q(n-1)]}.

Low-dimensional q-side (s=2):
    - sketch L2 pooled q-dimensional row vectors with replacement;
    - form the sketched correlation matrix;
    - DO NOT apply ACT correction;
    - aggregate RAW correlation eigenvalues coordinatewise by the median;
    - compare with tau_2 = 1 + delta_n.

For the fixed-q Case-1 setting, delta_n=n^{-1/4} is used by default.
This satisfies the low-dimensional asymptotic requirements delta_n -> 0 and
sqrt(n)/q * delta_n -> infinity.

The full-sample mode-adaptive estimator follows Eq. (13), while the repeated
median sketch estimator follows Eq. (15) in the current manuscript.

Dual-mode boundary-controlled DGP:
    F_i = A^{1/2} G_i B^{1/2},
    Z_i = R F_i C^T + E_i.

For every Monte Carlo replication, diagonal A and B are calibrated numerically
so that

    lambda_{1,k}^{pop} = high_target,
    lambda_{2,k}^{pop} = tau_2 + low_margin,

while sum_j t_j = sum_j s_j = factor_total, where

    t_j = tr(B) a_j,    s_j = tr(A) b_j.

The common total is required by the bilinear factor covariance.  Holding it
fixed also prevents total factor energy from changing with heterogeneity h.

Default batch setting:
    n in {200,500,1000}, p=200, q=10, k=5,
    h in {0,1,2}, omega=0.01,
    high-side population kth correlation spike target=25,
    low-side population margin above tau_2=0.15,
    common total factor strength=2.0,
    gamma in {0.05,0.10,0.15,0.20,0.25,0.30},
    B in {1,5,15,35}.

Both data and sketch banks are regenerated in every Monte Carlo replication.
For each (n,h), the script writes a four-panel eigenvalue boxplot and the raw
and summarized CSV files.  It also writes one publication-ready LaTeX rank
table per n and a combined table across all n values.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import qr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy.linalg import eigvalsh as scipy_eigvalsh
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

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
    A = symmetrize(A)

    if HAVE_SCIPY:
        vals = scipy_eigvalsh(
            A,
            check_finite=False,
            overwrite_a=False,
        )
    else:
        vals = np.linalg.eigvalsh(A)

    return np.maximum(vals[::-1], 0.0)


def correlation_from_cov(
    M: np.ndarray,
    ridge: float = 1e-12,
) -> np.ndarray:
    d = np.maximum(np.diag(M), ridge)
    inv = 1.0 / np.sqrt(d)

    H = (
        inv[:, None]
        * M
        * inv[None, :]
    )

    return symmetrize(H)


def prefix_rank(
    vals: np.ndarray,
    threshold: float,
) -> int:
    """
    Largest consecutive prefix above threshold.
    """
    rank = 0

    for x in vals:
        if np.isfinite(x) and x > threshold:
            rank += 1
        else:
            break

    return int(rank)


def eigen_ratio_rank(
    A: np.ndarray,
    kmax: int,
    eps: float = 1e-12,
) -> int:
    vals = np.maximum(
        sorted_eigvals(A),
        eps,
    )

    kmax_eff = min(
        kmax,
        len(vals) - 1,
    )

    if kmax_eff < 1:
        return 0

    ratios = (
        vals[:kmax_eff]
        /
        vals[1:kmax_eff + 1]
    )

    return int(
        np.argmax(ratios) + 1
    )


# =============================================================================
# Dense pervasive loadings
# =============================================================================

def block_pervasive_loading(
    dim: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if dim % k != 0:
        raise ValueError(
            f"dim={dim} must be divisible by k={k}"
        )

    m = dim // k
    blocks = []

    for _ in range(m):
        Q, _ = qr(
            rng.normal(size=(k, k))
        )
        blocks.append(Q.T)

    Qbig = np.vstack(blocks) / np.sqrt(m)
    U = np.sqrt(dim) * Qbig

    if not np.allclose(
        U.T @ U,
        dim * np.eye(k),
        atol=1e-8,
    ):
        raise RuntimeError(
            "Loading normalization failed."
        )

    return U


# =============================================================================
# Heterogeneity
# =============================================================================

def choose_high_variance_indices(
    dim: int,
    omega: float,
    rng: np.random.Generator,
) -> np.ndarray:
    m = max(
        1,
        int(round(omega * dim)),
    )

    m = min(m, dim)

    return np.sort(
        rng.choice(
            dim,
            size=m,
            replace=False,
        )
    )


def trace_normalized_diag(
    dim: int,
    h: float,
    omega: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Draw the contaminated coordinates even when h=0.  This keeps the RNG
    # stream paired across heterogeneity levels for a fixed (n, replication).
    idx = choose_high_variance_indices(
        dim,
        omega,
        rng,
    )

    if h <= 0:
        return np.ones(
            dim,
            dtype=float,
        )

    d = np.ones(
        dim,
        dtype=float,
    )

    d[idx] = 10.0 ** h

    d *= dim / d.sum()

    return d


# =============================================================================
# High-dimensional ACT side
# =============================================================================

def adjusted_eigenvalues_high(
    vals: np.ndarray,
    n: int,
    q: int,
    kmax: int,
) -> np.ndarray:
    """
    ACT correction ONLY for the high-dimensional p-side.

        rho_j = (p-j) / [q(n-1)]
    """
    vals = np.asarray(
        vals,
        dtype=float,
    )

    p = len(vals)
    kmax_eff = min(
        kmax,
        p - 1,
    )

    df = q * (n - 1)

    out = np.full(
        kmax_eff,
        np.nan,
    )

    for j0 in range(kmax_eff):
        j = j0 + 1

        z = max(
            float(vals[j0]),
            1e-12,
        )

        nxt = max(
            float(vals[j0 + 1]),
            1e-12,
        )

        tail = vals[j0 + 1:]

        denom = tail - z

        denom = np.where(
            np.abs(denom) < 1e-10,
            -1e-10,
            denom,
        )

        stabilizer = (
            (3.0 * z + nxt) / 4.0
            - z
        )

        if abs(stabilizer) < 1e-10:
            stabilizer = -1e-10

        m_hat = (
            np.sum(1.0 / denom)
            + 1.0 / stabilizer
        ) / (p - j)

        rho = (
            p - j
        ) / float(df)

        m_comp = (
            -(1.0 - rho) / z
            + rho * m_hat
        )

        out[j0] = (
            np.inf
            if abs(m_comp) < 1e-12
            else -1.0 / m_comp
        )

    return out


def tau_high(
    n: int,
    p: int,
    q: int,
) -> float:
    return float(
        1.0
        + np.sqrt(
            p / float(q * (n - 1))
        )
    )


def tau_low(
    n: int,
    delta_low: float | None,
) -> Tuple[float, float]:
    """
    If delta_low is not supplied, use delta_n=n^{-1/4}.
    Returns (tau_low, delta_used).
    """
    if delta_low is None:
        delta = float(
            n ** (-0.25)
        )
    else:
        delta = float(delta_low)

    if delta <= 0:
        raise ValueError(
            "delta_low must be positive."
        )

    return 1.0 + delta, delta


# =============================================================================
# Dual-mode population calibration
# =============================================================================

def population_correlation_operator(
    U: np.ndarray,
    d_noise: np.ndarray,
    strengths: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Population correlation for U diag(strengths) U^T + sigma^2 D."""
    strengths = np.asarray(strengths, dtype=float)

    if strengths.ndim != 1 or strengths.shape[0] != U.shape[1]:
        raise ValueError("strengths must have one entry per loading column.")
    if np.any(strengths < 0):
        raise ValueError("Population strengths must be nonnegative.")

    M = (
        (U * strengths[None, :]) @ U.T
        +
        sigma ** 2
        * np.diag(d_noise)
    )

    return correlation_from_cov(M)


def population_kth_spike(
    U: np.ndarray,
    d_noise: np.ndarray,
    strengths: np.ndarray,
    sigma: float,
    k: int,
) -> float:
    H = population_correlation_operator(
        U=U,
        d_noise=d_noise,
        strengths=strengths,
        sigma=sigma,
    )

    vals = sorted_eigvals(H)

    return float(
        vals[k - 1]
    )


def boundary_strength_vector(
    total: float,
    weak_strength: float,
    k: int,
) -> np.ndarray:
    """
    Four/every k-1 stronger factors share the remaining strength, while the
    rank-determining kth factor has strength weak_strength.
    """
    if k < 2:
        raise ValueError("Dual-mode calibration requires k >= 2.")
    if total <= 0:
        raise ValueError("factor_total must be positive.")
    if weak_strength < 0 or weak_strength > total / k:
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
    """
    With sum(strengths)=factor_total, solve for the weakest of k strengths so
    that the kth population correlation eigenvalue equals target.

    The search path is

        strengths(x) = ((T-x)/(k-1), ..., (T-x)/(k-1), x),
        0 <= x <= T/k.

    Therefore the kth factor is the weakest and the total signal is unchanged.
    """
    if target <= 1.0:
        raise ValueError(
            "Population kth-spike target must exceed 1."
        )

    lo = 0.0
    hi = factor_total / float(k)

    lo_vec = boundary_strength_vector(factor_total, lo, k)
    hi_vec = boundary_strength_vector(factor_total, hi, k)
    lo_spike = population_kth_spike(U, d_noise, lo_vec, sigma, k)
    hi_spike = population_kth_spike(U, d_noise, hi_vec, sigma, k)

    if not (lo_spike <= target <= hi_spike):
        raise ValueError(
            f"Target {target:.8f} is infeasible for factor_total="
            f"{factor_total:.8f}. Along the calibration path, the kth "
            f"population spike ranges from {lo_spike:.8f} to "
            f"{hi_spike:.8f}. Increase --factor-total if the target is "
            "above the attainable range."
        )

    for _ in range(120):
        mid = 0.5 * (
            lo + hi
        )

        mid_vec = boundary_strength_vector(factor_total, mid, k)
        spike = population_kth_spike(
            U=U,
            d_noise=d_noise,
            strengths=mid_vec,
            sigma=sigma,
            k=k,
        )

        if spike < target:
            lo = mid
        else:
            hi = mid

        if (
            hi - lo
            <
            tol * max(1.0, factor_total)
        ):
            break

    result = boundary_strength_vector(
        factor_total,
        0.5 * (lo + hi),
        k,
    )

    achieved = population_kth_spike(U, d_noise, result, sigma, k)
    if abs(achieved - target) > 1e-6 * max(1.0, abs(target)):
        raise RuntimeError(
            f"Population calibration did not converge: target={target}, "
            f"achieved={achieved}."
        )

    return result


# =============================================================================
# DGP
# =============================================================================

@dataclass
class GeneratedDataset:
    Z: np.ndarray
    factor_total: float
    high_strengths: np.ndarray
    low_strengths: np.ndarray
    factor_row_variances: np.ndarray
    factor_col_variances: np.ndarray
    population_high_k: float
    population_low_k: float
    dp_ratio: float


def generate_dataset(
    n: int,
    p: int,
    q: int,
    k: int,
    h: float,
    omega: float,
    sigma: float,
    high_target: float,
    low_target: float,
    factor_total: float,
    seed: int,
) -> GeneratedDataset:
    """
    Dual-mode boundary-controlled Case-1 DGP:

        F_i = A^{1/2} G_i B^{1/2},
        Z_i = R F_i C^T + E_i,

        E_i = D_p^{1/2} W_i,

    with D_q = I_q.  The vectors

        t_j = tr(B) a_j,    s_j = tr(A) b_j

    have the same fixed total.  They are calibrated so that the kth
    population correlation spikes equal high_target and low_target.
    """
    rng = np.random.default_rng(
        seed
    )

    R = block_pervasive_loading(
        p,
        k,
        rng,
    )

    C = block_pervasive_loading(
        q,
        k,
        rng,
    )

    dp = trace_normalized_diag(
        dim=p,
        h=h,
        omega=omega,
        rng=rng,
    )

    dq = np.ones(
        q,
        dtype=float,
    )

    high_strengths = calibrate_boundary_strengths(
        U=R,
        d_noise=dp,
        target=high_target,
        sigma=sigma,
        k=k,
        factor_total=factor_total,
    )

    low_strengths = calibrate_boundary_strengths(
        U=C,
        d_noise=dq,
        target=low_target,
        sigma=sigma,
        k=k,
        factor_total=factor_total,
    )

    # Because sum(t)=sum(s)=T, this construction gives
    # tr(A)=tr(B)=sqrt(T), tr(B)*diag(A)=t, tr(A)*diag(B)=s.
    sqrt_total = np.sqrt(factor_total)
    a_diag = high_strengths / sqrt_total
    b_diag = low_strengths / sqrt_total

    if not np.isclose(a_diag.sum(), sqrt_total, atol=1e-9):
        raise RuntimeError("A calibration failed its trace identity.")
    if not np.isclose(b_diag.sum(), sqrt_total, atol=1e-9):
        raise RuntimeError("B calibration failed its trace identity.")

    pop_high = population_kth_spike(
        U=R,
        d_noise=dp,
        strengths=high_strengths,
        sigma=sigma,
        k=k,
    )

    pop_low = population_kth_spike(
        U=C,
        d_noise=dq,
        strengths=low_strengths,
        sigma=sigma,
        k=k,
    )

    G = rng.normal(
        size=(n, k, k)
    )

    W = rng.normal(
        size=(n, p, q)
    )

    F = (
        np.sqrt(a_diag)[None, :, None]
        * G
        * np.sqrt(b_diag)[None, None, :]
    )

    signal = np.einsum(
        "pr,nrs,qs->npq",
        R,
        F,
        C,
        optimize=True,
    )

    noise = (
        sigma
        * np.sqrt(dp)[None, :, None]
        * W
    )

    Z = signal + noise

    Z = Z - Z.mean(
        axis=0,
        keepdims=True,
    )

    return GeneratedDataset(
        Z=Z,
        factor_total=float(factor_total),
        high_strengths=high_strengths,
        low_strengths=low_strengths,
        factor_row_variances=a_diag,
        factor_col_variances=b_diag,
        population_high_k=float(pop_high),
        population_low_k=float(pop_low),
        dp_ratio=float(
            np.max(dp) / np.min(dp)
        ),
    )


# =============================================================================
# Full operators
# =============================================================================

def full_operators(
    Z: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n, p, q = Z.shape

    U1 = (
        Z.transpose(0, 2, 1)
        .reshape(n * q, p)
    )

    U2 = Z.reshape(
        n * p,
        q,
    )

    M1 = (
        U1.T @ U1
    ) / float(n * q)

    M2 = (
        U2.T @ U2
    ) / float(n * p)

    H1 = correlation_from_cov(M1)
    H2 = correlation_from_cov(M2)

    return (
        symmetrize(M1),
        symmetrize(M2),
        H1,
        H2,
    )


# =============================================================================
# Nested high-side sketch bank
# =============================================================================

def build_nested_high_sketch_bank(
    U_high: np.ndarray,
    L_values: List[int],
    B_max: int,
    n: int,
    q: int,
    kmax: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    Return exact ACT-adjusted high-side sketch eigencurves.

    bank[L] has shape (B_max, kmax_eff).
    """
    pool_size, p = U_high.shape

    L_values_sorted = sorted(
        set(
            int(x)
            for x in L_values
        )
    )

    if not L_values_sorted:
        raise ValueError(
            "Empty L grid."
        )

    L_max = max(
        L_values_sorted
    )

    rng = np.random.default_rng(
        seed
    )

    ids_bank = rng.integers(
        0,
        pool_size,
        size=(
            B_max,
            L_max,
        ),
        dtype=np.int64,
    )

    kmax_eff = min(
        kmax,
        p - 1,
    )

    banks = {
        L: np.empty(
            (
                B_max,
                kmax_eff,
            ),
            dtype=float,
        )
        for L
        in L_values_sorted
    }

    for b in range(B_max):
        gram = np.zeros(
            (p, p),
            dtype=float,
        )

        start = 0

        ids_stream = (
            ids_bank[b]
        )

        for L in L_values_sorted:
            new_ids = ids_stream[
                start:L
            ]

            block = U_high[
                new_ids
            ]

            gram += (
                block.T @ block
            )

            M = (
                gram
                /
                float(L)
            )

            H = correlation_from_cov(
                M
            )

            vals = sorted_eigvals(
                H
            )

            banks[L][b, :] = (
                adjusted_eigenvalues_high(
                    vals=vals,
                    n=n,
                    q=q,
                    kmax=kmax,
                )
            )

            start = L

    return banks


# =============================================================================
# One replication
# =============================================================================

def summarize_full_methods(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    return (
        raw.groupby(
            "method",
            as_index=False,
        )
        .agg(
            row_average_rank=(
                "row_rank",
                "mean",
            ),

            row_accuracy=(
                "row_correct",
                "mean",
            ),

            col_average_rank=(
                "col_rank",
                "mean",
            ),

            col_accuracy=(
                "col_correct",
                "mean",
            ),

            pair_accuracy=(
                "pair_correct",
                "mean",
            ),
        )
    )


# =============================================================================
# CLI
# =============================================================================

# THEORY-EXACT OVERRIDES FOR THE CURRENT MANUSCRIPT
# =============================================================================
# The original implementation above treated the low-dimensional q-side as a
# full-data direct estimator.  Section 3.3 and Theorem 5 in the current
# manuscript instead sketch BOTH modes.  The low-dimensional sketch uses RAW
# correlation eigenvalues (no ACT correction), then the same coordinatewise
# median aggregation as the high-dimensional side.  The definitions below
# override the earlier helper functions and main routine accordingly.


def max_threshold_rank(
    vals: np.ndarray,
    threshold: float,
    kmax: int,
) -> int:
    """Full-sample rule in Eq. (13): max{j <= kmax: xi_j > tau}."""
    vals = np.asarray(vals, dtype=float)
    kmax_eff = min(kmax, len(vals))
    passed = np.flatnonzero(
        np.isfinite(vals[:kmax_eff])
        & (vals[:kmax_eff] > threshold)
    )
    return 0 if len(passed) == 0 else int(passed[-1] + 1)


def build_nested_low_sketch_bank(
    U_low: np.ndarray,
    L_values: List[int],
    B_max: int,
    kmax: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    Low-dimensional mode sketch bank.

    Each sketch is formed by uniform sampling with replacement from the np
    pooled q-dimensional row vectors.  The output contains RAW correlation
    eigenvalues, exactly as required for chi_2=0 in Section 3.3 / Theorem 5.
    No ACT adjustment is applied on this side.
    """
    pool_size, q = U_low.shape
    L_values_sorted = sorted(set(int(x) for x in L_values))
    if not L_values_sorted:
        raise ValueError("Empty low-side L grid.")

    L_max = max(L_values_sorted)
    rng = np.random.default_rng(seed)
    ids_bank = rng.integers(
        0,
        pool_size,
        size=(B_max, L_max),
        dtype=np.int64,
    )

    kmax_eff = min(kmax, q)
    banks = {
        L: np.empty((B_max, kmax_eff), dtype=float)
        for L in L_values_sorted
    }

    for b in range(B_max):
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


def run_one_replication(
    rep: int,
    cfg: Dict[str, object],
):
    """
    One Monte Carlo replication of the mode-adaptive S^2ACT estimator.

    High p-side:
        sketch -> correlation -> ACT adjustment -> median -> tau_high.

    Low q-side:
        sketch -> correlation -> RAW eigenvalues -> median -> 1+delta_n.
    """
    n = int(cfg["n"])
    p = int(cfg["p"])
    q = int(cfg["q"])
    k = int(cfg["k"])
    kmax = int(cfg["kmax"])

    h = float(cfg["h"])
    omega = float(cfg["omega"])
    sigma = float(cfg["sigma"])
    high_target = float(cfg["high_target"])
    low_target = float(cfg["low_target"])
    factor_total = float(cfg["factor_total"])

    gamma_grid = [float(x) for x in cfg["gamma_grid"]]
    B_grid = [int(x) for x in cfg["B_grid"]]

    tau_H = float(cfg["tau_high"])
    tau_L = float(cfg["tau_low"])
    seed0 = int(cfg["seed"])

    # Fresh data for every Monte Carlo replication.
    data_seed = seed0 + 10_000_000 * rep + 11
    data = generate_dataset(
        n=n,
        p=p,
        q=q,
        k=k,
        h=h,
        omega=omega,
        sigma=sigma,
        high_target=high_target,
        low_target=low_target,
        factor_total=factor_total,
        seed=data_seed,
    )
    Z = data.Z

    M1, M2, H1, H2 = full_operators(Z)

    # ------------------------------------------------------------------
    # Full-sample mode-adaptive estimator, Eq. (13)
    # ------------------------------------------------------------------
    high_vals = sorted_eigvals(H1)
    high_adj = adjusted_eigenvalues_high(
        vals=high_vals,
        n=n,
        q=q,
        kmax=kmax,
    )

    low_vals_all = sorted_eigvals(H2)
    low_vals = low_vals_all[: min(kmax, q)]

    # Eq. (13) uses max threshold exceedance, not the prefix rule.
    full_high_rank = max_threshold_rank(high_adj, tau_H, kmax)
    full_low_rank = max_threshold_rank(low_vals, tau_L, kmax)

    # Paired benchmark methods on exactly the same generated dataset.
    cov_row = eigen_ratio_rank(M1, kmax)
    cov_col = eigen_ratio_rank(M2, kmax)
    corr_row = eigen_ratio_rank(H1, kmax)
    corr_col = eigen_ratio_rank(H2, kmax)

    full_rows = []

    def append_full_method(method: str, row_rank: int, col_rank: int) -> None:
        full_rows.append({
            "rep": rep,
            "method": method,
            "row_rank": int(row_rank),
            "col_rank": int(col_rank),
            "row_correct": int(row_rank == k),
            "col_correct": int(col_rank == k),
            "pair_correct": int(row_rank == k and col_rank == k),
        })

    append_full_method("Cov-ER", cov_row, cov_col)
    append_full_method("Corr-ER", corr_row, corr_col)
    append_full_method("Full-Mode-Adaptive", full_high_rank, full_low_rank)

    # ------------------------------------------------------------------
    # Pooled vectors for BOTH modes
    # ------------------------------------------------------------------
    # s=1: nq p-dimensional column vectors.
    U_high = Z.transpose(0, 2, 1).reshape(n * q, p)
    # s=2: np q-dimensional row vectors.
    U_low = Z.reshape(n * p, q)

    pool_high = n * q
    pool_low = n * p

    L1_values = [
        max(1, int(round(gamma * pool_high)))
        for gamma in gamma_grid
    ]
    L2_values = [
        max(1, int(round(gamma * pool_low)))
        for gamma in gamma_grid
    ]

    B_max = max(B_grid)

    high_banks = build_nested_high_sketch_bank(
        U_high=U_high,
        L_values=L1_values,
        B_max=B_max,
        n=n,
        q=q,
        kmax=kmax,
        seed=seed0 + 10_000_000 * rep + 101,
    )

    low_banks = build_nested_low_sketch_bank(
        U_low=U_low,
        L_values=L2_values,
        B_max=B_max,
        kmax=kmax,
        seed=seed0 + 10_000_000 * rep + 202,
    )

    # ------------------------------------------------------------------
    # Eq. (15): coordinatewise median on BOTH modes, then prefix rule
    # ------------------------------------------------------------------
    randomized_rows = []
    k_idx = k - 1
    next_idx = k

    for gamma, L1, L2 in zip(gamma_grid, L1_values, L2_values):
        high_bank = high_banks[L1]
        low_bank = low_banks[L2]

        for B in B_grid:
            A1 = np.median(high_bank[:B, :], axis=0)
            A2 = np.median(low_bank[:B, :], axis=0)

            row_rank = prefix_rank(A1, tau_H)
            col_rank = prefix_rank(A2, tau_L)

            randomized_rows.append({
                "rep": rep,
                "n": n,
                "p": p,
                "q": q,
                "k_true": k,
                "h": h,
                "variance_ratio": 10.0 ** h,
                "gamma": gamma,

                "L1_high": L1,
                "L1_fraction": L1 / float(pool_high),
                "L2_low": L2,
                "L2_fraction": L2 / float(pool_low),
                "B": B,

                "tau_high": tau_H,
                "delta_low": float(cfg["delta_low_used"]),
                "tau_low": tau_L,

                "row_rank": int(row_rank),
                "col_rank": int(col_rank),
                "row_correct": int(row_rank == k),
                "col_correct": int(col_rank == k),
                "pair_correct": int(row_rank == k and col_rank == k),
                "row_over": int(row_rank > k),
                "row_under": int(row_rank < k),
                "col_over": int(col_rank > k),
                "col_under": int(col_rank < k),

                # High mode: median ACT-adjusted sketch eigenvalues.
                "row_A_k_adj": (
                    float(A1[k_idx]) if k_idx < len(A1) else np.nan
                ),
                "row_A_next_adj": (
                    float(A1[next_idx]) if next_idx < len(A1) else np.nan
                ),

                # Low mode: median RAW sketch correlation eigenvalues.
                "col_A_k_raw": (
                    float(A2[k_idx]) if k_idx < len(A2) else np.nan
                ),
                "col_A_next_raw": (
                    float(A2[next_idx]) if next_idx < len(A2) else np.nan
                ),

                # Full-sample mode-adaptive references.
                "full_row_adj_k": (
                    float(high_adj[k_idx]) if k_idx < len(high_adj) else np.nan
                ),
                "full_row_adj_next": (
                    float(high_adj[next_idx]) if next_idx < len(high_adj) else np.nan
                ),
                "full_col_raw_k": (
                    float(low_vals[k_idx]) if k_idx < len(low_vals) else np.nan
                ),
                "full_col_raw_next": (
                    float(low_vals[next_idx]) if next_idx < len(low_vals) else np.nan
                ),
                "full_row_rank": int(full_high_rank),
                "full_col_rank": int(full_low_rank),
            })

    diagnostics = {
        "rep": rep,
        "n": n,
        "p": p,
        "q": q,
        "k_true": k,
        "h": h,
        "factor_total": data.factor_total,
        "high_strength_min": float(np.min(data.high_strengths)),
        "high_strength_max": float(np.max(data.high_strengths)),
        "low_strength_min": float(np.min(data.low_strengths)),
        "low_strength_max": float(np.max(data.low_strengths)),
        "high_strengths": json.dumps(data.high_strengths.tolist()),
        "low_strengths": json.dumps(data.low_strengths.tolist()),
        "factor_row_variances_A": json.dumps(
            data.factor_row_variances.tolist()
        ),
        "factor_col_variances_B": json.dumps(
            data.factor_col_variances.tolist()
        ),
        "population_high_target": high_target,
        "population_low_target": low_target,
        "population_high_k": data.population_high_k,
        "population_low_k": data.population_low_k,
        "population_high_error": data.population_high_k - high_target,
        "population_low_error": data.population_low_k - low_target,
        "dp_ratio": data.dp_ratio,

        "tau_high": tau_H,
        "delta_low": float(cfg["delta_low_used"]),
        "tau_low": tau_L,
        "delta_low_condition_index": float(
            cfg["delta_low_condition_index"]
        ),

        "full_row_rank": int(full_high_rank),
        "full_col_rank": int(full_low_rank),
        "full_row_correct": int(full_high_rank == k),
        "full_col_correct": int(full_low_rank == k),
        "full_pair_correct": int(
            full_high_rank == k and full_low_rank == k
        ),

        "full_row_adj_k": (
            float(high_adj[k_idx]) if k_idx < len(high_adj) else np.nan
        ),
        "full_row_adj_next": (
            float(high_adj[next_idx]) if next_idx < len(high_adj) else np.nan
        ),
        "full_col_raw_k": (
            float(low_vals[k_idx]) if k_idx < len(low_vals) else np.nan
        ),
        "full_col_raw_next": (
            float(low_vals[next_idx]) if next_idx < len(low_vals) else np.nan
        ),

        "cov_er_row": int(cov_row),
        "cov_er_col": int(cov_col),
        "corr_er_row": int(corr_row),
        "corr_er_col": int(corr_col),
    }

    return randomized_rows, full_rows, diagnostics


def summarize_randomized(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["gamma", "L1_high", "L2_low", "B"],
            as_index=False,
        )
        .agg(
            row_average_rank=("row_rank", "mean"),
            row_accuracy=("row_correct", "mean"),
            row_over_rate=("row_over", "mean"),
            row_under_rate=("row_under", "mean"),

            col_average_rank=("col_rank", "mean"),
            col_accuracy=("col_correct", "mean"),
            col_over_rate=("col_over", "mean"),
            col_under_rate=("col_under", "mean"),

            pair_accuracy=("pair_correct", "mean"),

            row_A_k_adj_mean=("row_A_k_adj", "mean"),
            row_A_k_adj_sd=("row_A_k_adj", "std"),
            row_A_next_adj_mean=("row_A_next_adj", "mean"),
            row_A_next_adj_sd=("row_A_next_adj", "std"),

            col_A_k_raw_mean=("col_A_k_raw", "mean"),
            col_A_k_raw_sd=("col_A_k_raw", "std"),
            col_A_next_raw_mean=("col_A_next_raw", "mean"),
            col_A_next_raw_sd=("col_A_next_raw", "std"),
        )
        .sort_values(["gamma", "B"])
        .reset_index(drop=True)
    )


# =============================================================================
# Publication outputs
# =============================================================================

PUBLICATION_RENAME = {
    "row_rank": "high_rank",
    "col_rank": "low_rank",
    "row_correct": "high_correct",
    "col_correct": "low_correct",
    "pair_correct": "joint_correct",
    "row_over": "high_over",
    "row_under": "high_under",
    "col_over": "low_over",
    "col_under": "low_under",
    "row_A_k_adj": "high_A_k",
    "row_A_next_adj": "high_A_kplus1",
    "col_A_k_raw": "low_A_k",
    "col_A_next_raw": "low_A_kplus1",
    "full_row_adj_k": "full_high_A_k",
    "full_row_adj_next": "full_high_A_kplus1",
    "full_col_raw_k": "full_low_A_k",
    "full_col_raw_next": "full_low_A_kplus1",
    "full_row_rank": "full_high_rank",
    "full_col_rank": "full_low_rank",
    "full_row_correct": "full_high_correct",
    "full_col_correct": "full_low_correct",
    "full_pair_correct": "full_joint_correct",
    "cov_er_row": "cov_er_high",
    "cov_er_col": "cov_er_low",
    "corr_er_row": "corr_er_high",
    "corr_er_col": "corr_er_low",
    "row_average_rank": "high_average_rank",
    "col_average_rank": "low_average_rank",
    "row_accuracy": "high_accuracy",
    "col_accuracy": "low_accuracy",
    "pair_accuracy": "joint_accuracy",
    "row_over_rate": "high_over_rate",
    "row_under_rate": "high_under_rate",
    "col_over_rate": "low_over_rate",
    "col_under_rate": "low_under_rate",
    "row_A_k_adj_mean": "high_A_k_mean",
    "row_A_k_adj_sd": "high_A_k_sd",
    "row_A_next_adj_mean": "high_A_kplus1_mean",
    "row_A_next_adj_sd": "high_A_kplus1_sd",
    "col_A_k_raw_mean": "low_A_k_mean",
    "col_A_k_raw_sd": "low_A_k_sd",
    "col_A_next_raw_mean": "low_A_kplus1_mean",
    "col_A_next_raw_sd": "low_A_kplus1_sd",
}


def publication_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy row/column labels by high/low mode labels."""
    return df.rename(columns=PUBLICATION_RENAME)


def _condition_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _boxplot_ylim(values: List[np.ndarray], reference: float) -> Tuple[float, float]:
    finite_parts = [
        np.asarray(x, dtype=float)[np.isfinite(x)]
        for x in values
        if len(x)
    ]
    if finite_parts:
        merged = np.concatenate(finite_parts + [np.array([reference])])
    else:
        merged = np.array([reference - 0.05, reference + 0.05])

    lo, hi = np.quantile(merged, [0.01, 0.99])
    span = max(float(hi - lo), 0.04, 0.04 * max(1.0, abs(reference)))
    return float(lo - 0.18 * span), float(hi + 0.18 * span)


def make_eigenvalue_boxplot(
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    n: int,
    h: float,
    gamma_grid: List[float],
    B_grid: List[int],
    output_stem: Path,
) -> List[Path]:
    """Create the four-panel boxplot used for the manuscript diagnostics."""
    panels = [
        (
            "high_A_k",
            "full_high_A_k",
            "tau_high",
            r"(a) High-dimensional mode: last signal $A_1(k)$",
        ),
        (
            "high_A_kplus1",
            "full_high_A_kplus1",
            "tau_high",
            r"(b) High-dimensional mode: first noise $A_1(k+1)$",
        ),
        (
            "low_A_k",
            "full_low_A_k",
            "tau_low",
            r"(c) Low-dimensional mode: last signal $A_2(k)$",
        ),
        (
            "low_A_kplus1",
            "full_low_A_kplus1",
            "tau_low",
            r"(d) Low-dimensional mode: first noise $A_2(k+1)$",
        ),
    ]

    gamma_grid = sorted(float(x) for x in gamma_grid)
    B_grid = sorted(int(x) for x in B_grid)
    centers = np.arange(len(gamma_grid), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(B_grid))
    width = min(0.16, 0.62 / max(1, len(B_grid)))
    colors = plt.cm.Blues(np.linspace(0.75, 0.25, len(B_grid)))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True)

    for ax, (value_col, full_col, tau_col, title) in zip(axes.flat, panels):
        all_values: List[np.ndarray] = []

        for j, (B, offset, color) in enumerate(zip(B_grid, offsets, colors)):
            grouped = []
            for gamma in gamma_grid:
                vals = raw.loc[
                    (raw["B"] == B)
                    & np.isclose(raw["gamma"], gamma),
                    value_col,
                ].dropna().to_numpy()
                grouped.append(vals)
                all_values.append(vals)

            bp = ax.boxplot(
                grouped,
                positions=centers + offset,
                widths=width,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_edgecolor("#4d4d4d")
                patch.set_linewidth(0.8)
            for median in bp["medians"]:
                median.set_color("#e07a1f")
                median.set_linewidth(1.2)
            for key in ("whiskers", "caps"):
                for artist in bp[key]:
                    artist.set_color("#777777")
                    artist.set_linewidth(0.9)

        full_mean = float(diagnostics[full_col].mean())
        threshold = float(diagnostics[tau_col].mean())
        ax.axhline(
            full_mean,
            color="#2f83c5",
            linestyle="--",
            linewidth=1.4,
            zorder=0,
        )
        ax.axhline(
            threshold,
            color="#9b9b9b",
            linestyle=":",
            linewidth=1.5,
            zorder=0,
        )
        ax.set_ylim(*_boxplot_ylim(all_values, full_mean))
        ax.set_title(title, fontsize=11)
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{x:.2f}" for x in gamma_grid])
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)

    axes[0, 0].set_ylabel("Median-aggregated eigenvalue")
    axes[1, 0].set_ylabel("Median-aggregated eigenvalue")
    axes[1, 0].set_xlabel(r"Sketch fraction $\gamma_s=L_s/N_s$")
    axes[1, 1].set_xlabel(r"Sketch fraction $\gamma_s=L_s/N_s$")

    legend_handles = [
        Patch(facecolor=color, edgecolor="#4d4d4d", label=rf"$B={B}$")
        for B, color in zip(B_grid, colors)
    ]
    legend_handles.extend([
        Line2D([0], [0], color="#2f83c5", linestyle="--", label="Full-sample mean"),
    ])

    fig.suptitle(rf"Dual-mode spectral diagnostics: $n={n}$, $h={h:g}$", y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def _se(series: pd.Series) -> float:
    count = int(series.notna().sum())
    if count <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(count))


def publication_table_data(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["n", "h", "gamma", "B"]
    for keys, group in raw.groupby(group_cols, sort=True):
        n, h, gamma, B = keys
        rows.append({
            "n": int(n),
            "h": float(h),
            "gamma": float(gamma),
            "B": int(B),
            "high_average_rank": float(group["high_rank"].mean()),
            "high_rank_se": _se(group["high_rank"]),
            "high_accuracy": float(group["high_correct"].mean()),
            "low_average_rank": float(group["low_rank"].mean()),
            "low_rank_se": _se(group["low_rank"]),
            "low_accuracy": float(group["low_correct"].mean()),
            "joint_accuracy": float(group["joint_correct"].mean()),
        })
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def make_rank_table_latex(
    table_data: pd.DataFrame,
    n_values: List[int],
    h_values: List[float],
    gamma_values: List[float],
    B_values: List[int],
    caption: str,
    label: str,
) -> str:
    """Build a booktabs LaTeX table with mean rank (SE) and accuracy."""
    gamma_values = [float(x) for x in gamma_values]
    ncols = 2 + 2 * len(gamma_values)
    colspec = "cc" + "cc" * len(gamma_values)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]

    header = ["", ""]
    for gamma in gamma_values:
        header.append(rf"\multicolumn{{2}}{{c}}{{${{\gamma_s={gamma:.2f}}}$}}")
    lines.append(" & ".join(header) + r" \\")

    cmids = []
    for j in range(len(gamma_values)):
        start = 3 + 2 * j
        cmids.append(rf"\cmidrule(lr){{{start}-{start+1}}}")
    lines.append(" ".join(cmids))

    second = [r"$h$", r"$B$"]
    for _ in gamma_values:
        second.extend([r"$\widetilde{k}^{A}_{s}$", r"Acc."])
    lines.append(" & ".join(second) + r" \\")
    lines.append(r"\midrule")

    multi_n = len(n_values) > 1
    for n_idx, n in enumerate(n_values):
        if multi_n:
            lines.append(
                rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{$n={n}$}}}} \\"
            )
            lines.append(r"\addlinespace[2pt]")

        for mode, rank_col, se_col, acc_col in [
            ("High-dimensional mode", "high_average_rank", "high_rank_se", "high_accuracy"),
            ("Low-dimensional mode", "low_average_rank", "low_rank_se", "low_accuracy"),
        ]:
            lines.append(
                rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{mode}:}}}} \\"
            )
            lines.append(r"\addlinespace[2pt]")

            for h_idx, h in enumerate(h_values):
                for b_idx, B in enumerate(B_values):
                    h_cell = rf"\multirow{{{len(B_values)}}}{{*}}{{{h:g}}}" if b_idx == 0 else ""
                    cells = [h_cell, str(B)]
                    for gamma in gamma_values:
                        mask = (
                            (table_data["n"] == n)
                            & np.isclose(table_data["h"], h)
                            & np.isclose(table_data["gamma"], gamma)
                            & (table_data["B"] == B)
                        )
                        hit = table_data.loc[mask]
                        if hit.empty:
                            cells.extend(["--", "--"])
                        else:
                            row = hit.iloc[0]
                            cells.extend([
                                f"{row[rank_col]:.2f} ({row[se_col]:.2f})",
                                f"{row[acc_col]:.2f}",
                            ])
                    lines.append(" & ".join(cells) + r" \\")

                if h_idx < len(h_values) - 1:
                    lines.append(r"\addlinespace[2pt]")

            lines.append(r"\midrule")

        if n_idx < len(n_values) - 1:
            lines.append(r"\addlinespace[3pt]")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Dual-mode boundary-controlled Case 1 for the partially "
            "high-dimensional S2ACT framework: the high and low population "
            "kth correlation spikes are calibrated separately subject to "
            "the common bilinear factor-energy constraint."
        )
    )

    ap.add_argument(
        "--n-grid",
        type=parse_int_list,
        default=parse_int_list("200,500,1000"),
        help="Sample sizes run in one batch.",
    )
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=8)

    ap.add_argument(
        "--h-grid",
        type=parse_float_list,
        default=parse_float_list("0,1,2"),
        help="Scale-heterogeneity levels run for every n.",
    )
    ap.add_argument(
        "--omega",
        type=float,
        default=0.01,
        help=(
            "Fraction of high-variance coordinates. Default 0.01 creates "
            "sparse scale contamination on the high-dimensional side."
        ),
    )
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--high-target", type=float, default=25.0)
    ap.add_argument(
        "--low-margin",
        type=float,
        default=0.15,
        help=(
            "Fixed population separation above tau_2. The low-side target "
            "is tau_2 + low_margin."
        ),
    )
    ap.add_argument(
        "--factor-total",
        type=float,
        default=2.0,
        help=(
            "Common sum of high- and low-mode population strengths. Keep "
            "this value fixed across h when comparing scale heterogeneity."
        ),
    )

    ap.add_argument(
        "--delta-low",
        type=float,
        default=None,
        help=(
            "Low-dimensional threshold buffer delta_n. "
            "Default n^{-1/4}, which is theory-valid for the fixed-q "
            "Case-1 setting."
        ),
    )

    ap.add_argument(
        "--gamma-grid",
        type=parse_float_list,
        default=parse_float_list("0.05,0.10,0.15,0.20,0.25,0.30"),
    )
    ap.add_argument(
        "--B-grid",
        type=parse_int_list,
        default=parse_int_list("1,5,15,35"),
    )
    ap.add_argument(
        "--table-gamma-grid",
        type=parse_float_list,
        default=parse_float_list("0.05,0.10,0.15,0.25,0.30"),
        help="Gamma columns included in the publication LaTeX tables.",
    )

    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument(
        "--out-dir",
        default="./results_case1_dual_mode_boundary_controlled",
    )

    return ap


def run_condition(
    args: argparse.Namespace,
    n: int,
    h: float,
    gamma_grid: List[float],
    B_grid: List[int],
    condition_dir: Path,
) -> Dict[str, object]:
    """Run and save one (n,h) condition."""
    tau_H = tau_high(n=n, p=args.p, q=args.q)
    tau_L, delta_used = tau_low(n=n, delta_low=args.delta_low)
    low_target = tau_L + args.low_margin
    delta_condition_index = (np.sqrt(n) / float(args.q)) * delta_used
    pool_high = n * args.q
    pool_low = n * args.p
    L1_values = [max(1, int(round(g * pool_high))) for g in gamma_grid]
    L2_values = [max(1, int(round(g * pool_low))) for g in gamma_grid]

    cfg = {
        "n": n,
        "p": args.p,
        "q": args.q,
        "k": args.k,
        "kmax": args.kmax,
        "h": h,
        "omega": args.omega,
        "sigma": args.sigma,
        "high_target": args.high_target,
        "low_margin": args.low_margin,
        "low_target": low_target,
        "factor_total": args.factor_total,
        "gamma_grid": gamma_grid,
        "B_grid": B_grid,
        "reps": args.reps,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "pool_high": pool_high,
        "pool_low": pool_low,
        "L1_values": L1_values,
        "L2_values": L2_values,
        "tau_high": tau_H,
        "delta_low_used": delta_used,
        "tau_low": tau_L,
        "delta_low_condition_index": delta_condition_index,
        "high_mode": {
            "dimension": args.p,
            "bias_correction": "ACT",
            "aggregation": "median",
            "threshold": "1+sqrt(p/[q(n-1)])",
        },
        "low_mode": {
            "dimension": args.q,
            "bias_correction": False,
            "aggregation": "median of raw correlation eigenvalues",
            "threshold": "1+delta_n",
        },
        "dgp": {
            "factor": "F_i=A^{1/2} G_i B^{1/2}",
            "population_high_k_target": args.high_target,
            "population_low_k_target": low_target,
            "population_low_margin": args.low_margin,
            "common_factor_strength_total": args.factor_total,
        },
    }

    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    print(
        f"\n========== n={n}, h={h:g} | tau_high={tau_H:.6f}, "
        f"tau_low={tau_L:.6f} =========="
    )
    randomized_all: List[Dict[str, object]] = []
    full_all: List[Dict[str, object]] = []
    diagnostics_all: List[Dict[str, object]] = []

    if args.n_jobs <= 1:
        for rep in range(args.reps):
            rand_rows, full_rows, diag = run_one_replication(rep=rep, cfg=cfg)
            randomized_all.extend(rand_rows)
            full_all.extend(full_rows)
            diagnostics_all.append(diag)
            print(
                f"n={n}, h={h:g}: completed {rep+1}/{args.reps}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            future_to_rep = {
                executor.submit(run_one_replication, rep, cfg): rep
                for rep in range(args.reps)
            }
            completed = 0
            for future in as_completed(future_to_rep):
                rep = future_to_rep[future]
                try:
                    rand_rows, full_rows, diag = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Replication {rep} failed for n={n}, h={h}."
                    ) from exc
                randomized_all.extend(rand_rows)
                full_all.extend(full_rows)
                diagnostics_all.append(diag)
                completed += 1
                print(
                    f"n={n}, h={h:g}: completed {completed}/{args.reps}",
                    flush=True,
                )

    randomized_internal = (
        pd.DataFrame(randomized_all)
        .sort_values(["rep", "gamma", "B"])
        .reset_index(drop=True)
    )
    full_internal = (
        pd.DataFrame(full_all)
        .sort_values(["rep", "method"])
        .reset_index(drop=True)
    )
    full_internal.insert(1, "n", n)
    full_internal.insert(2, "h", h)
    diagnostics_internal = (
        pd.DataFrame(diagnostics_all).sort_values("rep").reset_index(drop=True)
    )

    randomized_summary_internal = summarize_randomized(randomized_internal)
    randomized_summary_internal.insert(0, "h", h)
    randomized_summary_internal.insert(0, "n", n)
    full_summary_internal = summarize_full_methods(full_internal)
    full_summary_internal.insert(0, "h", h)
    full_summary_internal.insert(0, "n", n)

    randomized_raw = publication_names(randomized_internal)
    randomized_summary = publication_names(randomized_summary_internal)
    full_raw = publication_names(full_internal)
    full_summary = publication_names(full_summary_internal)
    diagnostics = publication_names(diagnostics_internal)

    files = {
        "randomized_raw": condition_dir / "randomized_raw_results.csv",
        "randomized_summary": condition_dir / "randomized_summary_results.csv",
        "full_raw": condition_dir / "full_paired_raw_results.csv",
        "full_summary": condition_dir / "full_paired_summary_results.csv",
        "diagnostics": condition_dir / "full_mode_adaptive_diagnostics.csv",
    }
    randomized_raw.to_csv(files["randomized_raw"], index=False)
    randomized_summary.to_csv(files["randomized_summary"], index=False)
    full_raw.to_csv(files["full_raw"], index=False)
    full_summary.to_csv(files["full_summary"], index=False)
    diagnostics.to_csv(files["diagnostics"], index=False)

    figure_stem = condition_dir / (
        f"spectral_diagnostics_n{n}_h{_condition_tag(h)}"
    )
    figure_files = make_eigenvalue_boxplot(
        raw=randomized_raw,
        diagnostics=diagnostics,
        n=n,
        h=h,
        gamma_grid=gamma_grid,
        B_grid=B_grid,
        output_stem=figure_stem,
    )

    print("Full-sample methods:")
    print(full_summary.to_string(index=False))
    return {
        "randomized_raw": randomized_raw,
        "randomized_summary": randomized_summary,
        "full_raw": full_raw,
        "full_summary": full_summary,
        "diagnostics": diagnostics,
        "files": list(files.values()) + figure_files + [condition_dir / "config.json"],
    }


def main():
    args = build_parser().parse_args()

    n_grid = sorted(set(int(x) for x in args.n_grid))
    h_grid = sorted(set(float(x) for x in args.h_grid))
    gamma_grid = sorted(set(float(x) for x in args.gamma_grid))
    B_grid = sorted(set(int(x) for x in args.B_grid))
    table_gamma_grid = sorted(set(float(x) for x in args.table_gamma_grid))

    if not n_grid or min(n_grid) <= 1:
        raise ValueError("n-grid must contain integers greater than 1.")
    if args.p % args.k != 0:
        raise ValueError(f"p={args.p} must be divisible by k={args.k}.")
    if args.q % args.k != 0:
        raise ValueError(f"q={args.q} must be divisible by k={args.k}.")
    if args.kmax < args.k + 1:
        raise ValueError("kmax must be at least k+1.")
    if not h_grid or min(h_grid) < 0:
        raise ValueError("h-grid must contain nonnegative values.")
    if not gamma_grid or min(gamma_grid) <= 0 or max(gamma_grid) > 1:
        raise ValueError("gamma-grid must lie in (0,1].")
    if not set(table_gamma_grid).issubset(set(gamma_grid)):
        raise ValueError("table-gamma-grid must be a subset of gamma-grid.")
    if not B_grid or min(B_grid) < 1:
        raise ValueError("B-grid must contain positive integers.")
    if args.low_margin <= 0:
        raise ValueError("low-margin must be positive.")
    if args.factor_total <= 0:
        raise ValueError("factor-total must be positive.")
    if not (0 < args.omega <= 1):
        raise ValueError("omega must lie in (0,1].")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_config = {
        "n_grid": n_grid,
        "h_grid": h_grid,
        "p": args.p,
        "q": args.q,
        "k": args.k,
        "kmax": args.kmax,
        "omega": args.omega,
        "sigma": args.sigma,
        "high_target": args.high_target,
        "low_margin": args.low_margin,
        "factor_total": args.factor_total,
        "gamma_grid": gamma_grid,
        "table_gamma_grid": table_gamma_grid,
        "B_grid": B_grid,
        "reps": args.reps,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
    }
    (out_root / "batch_config.json").write_text(
        json.dumps(batch_config, indent=2), encoding="utf-8"
    )

    print("========== Batch dual-mode boundary-controlled Case 1 ==========")
    print(f"n grid={n_grid}")
    print(f"h grid={h_grid}")
    print(f"gamma grid={gamma_grid}")
    print(f"B grid={B_grid}")
    print(f"reps={args.reps}, n_jobs={args.n_jobs}, omega={args.omega}")

    t0 = time.time()
    randomized_frames = []
    randomized_summary_frames = []
    full_frames = []
    full_summary_frames = []
    diagnostics_frames = []
    saved_files: List[Path] = [out_root / "batch_config.json"]

    for n in n_grid:
        for h in h_grid:
            condition_dir = out_root / f"n_{n}" / f"h_{_condition_tag(h)}"
            result = run_condition(
                args=args,
                n=n,
                h=h,
                gamma_grid=gamma_grid,
                B_grid=B_grid,
                condition_dir=condition_dir,
            )
            randomized_frames.append(result["randomized_raw"])
            randomized_summary_frames.append(result["randomized_summary"])
            full_frames.append(result["full_raw"])
            full_summary_frames.append(result["full_summary"])
            diagnostics_frames.append(result["diagnostics"])
            saved_files.extend(result["files"])

    randomized_all = pd.concat(randomized_frames, ignore_index=True)
    randomized_summary_all = pd.concat(randomized_summary_frames, ignore_index=True)
    full_all = pd.concat(full_frames, ignore_index=True)
    full_summary_all = pd.concat(full_summary_frames, ignore_index=True)
    diagnostics_all = pd.concat(diagnostics_frames, ignore_index=True)

    combined_paths = {
        "randomized_raw": out_root / "all_n_h_randomized_raw.csv",
        "randomized_summary": out_root / "all_n_h_randomized_summary.csv",
        "full_raw": out_root / "all_n_h_full_raw.csv",
        "full_summary": out_root / "all_n_h_full_summary.csv",
        "diagnostics": out_root / "all_n_h_diagnostics.csv",
        "table_data": out_root / "publication_rank_table_data.csv",
    }
    randomized_all.to_csv(combined_paths["randomized_raw"], index=False)
    randomized_summary_all.to_csv(combined_paths["randomized_summary"], index=False)
    full_all.to_csv(combined_paths["full_raw"], index=False)
    full_summary_all.to_csv(combined_paths["full_summary"], index=False)
    diagnostics_all.to_csv(combined_paths["diagnostics"], index=False)

    table_data = publication_table_data(randomized_all)
    table_data.to_csv(combined_paths["table_data"], index=False)
    saved_files.extend(combined_paths.values())

    for n in n_grid:
        tex = make_rank_table_latex(
            table_data=table_data,
            n_values=[n],
            h_values=h_grid,
            gamma_values=table_gamma_grid,
            B_values=B_grid,
            caption=(
                "Rank estimates obtained by the randomized median-aggregated "
                f"estimator for $n={n}$ across heterogeneity levels $h$, "
                "sketching ratios $\\gamma_s$, and repeated sketches $B$. "
                "Standard errors are reported in parentheses."
            ),
            label=f"tab:case1_rank_n{n}",
        )
        table_path = out_root / f"rank_table_n{n}.tex"
        table_path.write_text(tex, encoding="utf-8")
        saved_files.append(table_path)

    combined_tex = make_rank_table_latex(
        table_data=table_data,
        n_values=n_grid,
        h_values=h_grid,
        gamma_values=table_gamma_grid,
        B_values=B_grid,
        caption=(
            "Rank estimates obtained by the randomized median-aggregated "
            "estimator across sample sizes $n$, heterogeneity levels $h$, "
            "sketching ratios $\\gamma_s$, and repeated sketches $B$. "
            "Standard errors are reported in parentheses."
        ),
        label="tab:case1_rank_all_n",
    )
    combined_table_path = out_root / "rank_table_all_n.tex"
    combined_table_path.write_text(combined_tex, encoding="utf-8")
    saved_files.append(combined_table_path)

    print("\n========== Batch completed ==========")
    print(f"Conditions run: {len(n_grid) * len(h_grid)}")
    print(f"Figures created: {2 * len(n_grid) * len(h_grid)} (PNG and PDF)")
    print(f"LaTeX tables created: {len(n_grid) + 1}")
    print(f"Output root: {out_root}")
    print(f"Total elapsed time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()