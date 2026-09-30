#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_case1_randomized_median_partial_hd_theory_exact.py

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

Default setting:
    n=1000, p=200, q=10, k=5, h=2,
    high-side population kth correlation spike target=25,
    gamma in {0.05,0.10,0.15,0.20,0.25,0.30},
    B in {1,5,15,35}.

Both data and sketch banks are regenerated in every Monte Carlo replication.
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
        int(np.floor(omega * dim)),
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
    idx = choose_high_variance_indices(
        dim,
        omega,
        rng,
    )

    # Draw S_p even when h=0 so matched seeds use identical loading, subset,
    # factor, and noise random-number streams across heterogeneity levels.
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
# Population calibration
# =============================================================================

def population_correlation_operator(
    U: np.ndarray,
    d_noise: np.ndarray,
    theta: float,
    sigma: float,
    k: int,
) -> np.ndarray:
    M = (
        theta ** 2
        * k
        * (U @ U.T)
        +
        sigma ** 2
        * np.diag(d_noise)
    )

    return correlation_from_cov(M)


def population_kth_spike(
    U: np.ndarray,
    d_noise: np.ndarray,
    theta: float,
    sigma: float,
    k: int,
) -> float:
    H = population_correlation_operator(
        U=U,
        d_noise=d_noise,
        theta=theta,
        sigma=sigma,
        k=k,
    )

    vals = sorted_eigvals(H)

    return float(
        vals[k - 1]
    )


def calibrate_theta_for_high_target(
    R: np.ndarray,
    dp: np.ndarray,
    target: float,
    sigma: float,
    k: int,
    tol: float = 1e-10,
) -> float:
    if target <= 1.0:
        raise ValueError(
            "high-target must exceed 1."
        )

    H_inf = correlation_from_cov(
        R @ R.T
    )

    max_spike = float(
        sorted_eigvals(H_inf)[k - 1]
    )

    if target >= max_spike - 1e-8:
        raise ValueError(
            f"Requested high-target={target:.6f} is infeasible. "
            f"Maximum attainable kth population correlation spike "
            f"is approximately {max_spike:.6f}."
        )

    lo = 0.0
    hi = 0.01

    while (
        population_kth_spike(
            R,
            dp,
            hi,
            sigma,
            k,
        )
        < target
    ):
        hi *= 2.0

        if hi > 1e3:
            raise RuntimeError(
                "Could not bracket theta."
            )

    for _ in range(100):
        mid = 0.5 * (
            lo + hi
        )

        spike = population_kth_spike(
            R,
            dp,
            mid,
            sigma,
            k,
        )

        if spike < target:
            lo = mid
        else:
            hi = mid

        if (
            hi - lo
            <
            tol * max(1.0, hi)
        ):
            break

    return float(
        0.5 * (lo + hi)
    )


# =============================================================================
# DGP
# =============================================================================

@dataclass
class GeneratedDataset:
    Z: np.ndarray
    theta: float
    population_high_k: float
    population_low_k: float
    dp_ratio: float
    dp_sum: float
    dp_identity_error: float
    center_error: float


def generate_dataset(
    n: int,
    p: int,
    q: int,
    k: int,
    h: float,
    omega: float,
    sigma: float,
    high_target: float,
    seed: int,
) -> GeneratedDataset:
    """
    Case-1 one-sided heterogeneity:

        E_i = D_p^{1/2} W_i,

    with D_q = I_q.
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

    theta = calibrate_theta_for_high_target(
        R=R,
        dp=dp,
        target=high_target,
        sigma=sigma,
        k=k,
    )

    pop_high = population_kth_spike(
        U=R,
        d_noise=dp,
        theta=theta,
        sigma=sigma,
        k=k,
    )

    pop_low = population_kth_spike(
        U=C,
        d_noise=dq,
        theta=theta,
        sigma=sigma,
        k=k,
    )

    G = rng.normal(
        size=(n, k, k)
    )

    W = rng.normal(
        size=(n, p, q)
    )

    signal = theta * np.einsum(
        "pr,nrs,qs->npq",
        R,
        G,
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

    dp_ratio = float(np.max(dp) / np.min(dp))
    dp_sum = float(np.sum(dp))
    dp_identity_error = float(np.max(np.abs(dp - 1.0)))
    center_error = float(np.max(np.abs(np.mean(Z, axis=0))))

    # These are construction checks, not data-dependent performance claims.
    if not np.isclose(dp_sum, float(p), rtol=1e-12, atol=1e-10):
        raise RuntimeError("Trace normalization of D_p failed.")
    if not np.isclose(dp_ratio, 10.0 ** h, rtol=1e-12, atol=1e-10):
        raise RuntimeError("The requested scale-heterogeneity ratio was not attained.")
    if h == 0.0 and dp_identity_error > 1e-12:
        raise RuntimeError("The h=0 design must have D_p=I_p.")
    if abs(float(pop_high) - float(high_target)) > 1e-7:
        raise RuntimeError("Population row-side spike calibration failed.")
    if center_error > 1e-10:
        raise RuntimeError("Sample centering check failed.")

    return GeneratedDataset(
        Z=Z,
        theta=float(theta),
        population_high_k=float(pop_high),
        population_low_k=float(pop_low),
        dp_ratio=dp_ratio,
        dp_sum=dp_sum,
        dp_identity_error=dp_identity_error,
        center_error=center_error,
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
        "theta": data.theta,
        "population_high_k": data.population_high_k,
        "population_low_k": data.population_low_k,
        "dp_ratio": data.dp_ratio,
        "dp_sum": data.dp_sum,
        "dp_identity_error": data.dp_identity_error,
        "center_error": data.center_error,

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Theory-exact Case 1 for the partially high-dimensional "
            "S2ACT framework: high-side ACT sketches and low-side raw "
            "correlation-eigenvalue sketches, with median aggregation "
            "on both modes."
        )
    )

    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=8)

    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--high-target", type=float, default=25.0)

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

    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument(
        "--out-dir",
        default="./results_case1_randomized_median_partial_hd_theory_exact",
    )

    return ap


def main():
    args = build_parser().parse_args()

    if args.p % args.k != 0:
        raise ValueError(f"p={args.p} must be divisible by k={args.k}.")
    if args.q % args.k != 0:
        raise ValueError(f"q={args.q} must be divisible by k={args.k}.")
    if args.kmax < args.k + 1:
        raise ValueError("kmax must be at least k+1.")

    gamma_grid = sorted(set(float(x) for x in args.gamma_grid))
    B_grid = sorted(set(int(x) for x in args.B_grid))

    if not gamma_grid or min(gamma_grid) <= 0:
        raise ValueError("gamma-grid must contain positive values.")
    if not B_grid or min(B_grid) < 1:
        raise ValueError("B-grid must contain positive integers.")

    tau_H = tau_high(
        n=args.n,
        p=args.p,
        q=args.q,
    )

    tau_L, delta_used = tau_low(
        n=args.n,
        delta_low=args.delta_low,
    )

    delta_condition_index = (
        np.sqrt(args.n) / float(args.q)
    ) * delta_used

    pool_high = args.n * args.q
    pool_low = args.n * args.p

    L1_values = [
        max(1, int(round(gamma * pool_high)))
        for gamma in gamma_grid
    ]
    L2_values = [
        max(1, int(round(gamma * pool_low)))
        for gamma in gamma_grid
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n": args.n,
        "p": args.p,
        "q": args.q,
        "k": args.k,
        "kmax": args.kmax,
        "h": args.h,
        "omega": args.omega,
        "sigma": args.sigma,
        "high_target": args.high_target,
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
            "sketch": True,
            "bias_correction": "ACT",
            "aggregation": "median",
            "threshold": "1+sqrt(p/[q(n-1)])",
        },
        "low_mode": {
            "sketch": True,
            "bias_correction": False,
            "eigenvalues": "raw correlation eigenvalues",
            "aggregation": "median",
            "threshold": "1+delta_n",
        },
        "full_rank_rule": "Eq. (13): max threshold exceedance",
        "sketch_rank_rule": "Eq. (15): median + consecutive prefix",
        "sampling": "uniform with replacement",
    }

    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )

    print("========== Theory-exact Case 1: partially-HD S2ACT ==========")
    print(
        f"n={args.n}, p={args.p}, q={args.q}, "
        f"k={args.k}, kmax={args.kmax}"
    )
    print(
        f"p/[q(n-1)]={args.p/(args.q*(args.n-1)):.6f}, "
        f"q/sqrt(n)={args.q/np.sqrt(args.n):.6f}"
    )

    print("\nHigh-dimensional p-side:")
    print("  sketching: ON")
    print("  ACT correction: ON")
    print(f"  tau_1={tau_H:.6f}")
    print(f"  pool nq={pool_high}")
    print(f"  L1={L1_values}")

    print("\nLow-dimensional q-side:")
    print("  sketching: ON")
    print("  ACT correction: OFF")
    print("  per-sketch quantity: RAW correlation eigenvalues")
    print(f"  delta_n={delta_used:.6f}")
    print(f"  tau_2=1+delta_n={tau_L:.6f}")
    print(
        f"  sqrt(n)/q * delta_n={delta_condition_index:.6f}"
    )
    print(f"  pool np={pool_low}")
    print(f"  L2={L2_values}")

    print(f"\ngamma grid={gamma_grid}")
    print(f"B grid={B_grid}")
    print(f"reps={args.reps}, n_jobs={args.n_jobs}")

    randomized_all = []
    full_all = []
    diagnostics_all = []

    t0 = time.time()

    if args.n_jobs <= 1:
        for rep in range(args.reps):
            rep_t0 = time.time()
            rand_rows, full_rows, diag = run_one_replication(
                rep=rep,
                cfg=cfg,
            )
            randomized_all.extend(rand_rows)
            full_all.extend(full_rows)
            diagnostics_all.append(diag)

            print(
                f"completed rep {rep+1}/{args.reps} "
                f"({time.time()-rep_t0:.2f}s)",
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
                        f"Replication {rep} failed."
                    ) from exc

                randomized_all.extend(rand_rows)
                full_all.extend(full_rows)
                diagnostics_all.append(diag)
                completed += 1

                print(
                    f"completed {completed}/{args.reps} (rep={rep})",
                    flush=True,
                )

    randomized_raw = (
        pd.DataFrame(randomized_all)
        .sort_values(["rep", "gamma", "B"])
        .reset_index(drop=True)
    )
    full_raw = (
        pd.DataFrame(full_all)
        .sort_values(["rep", "method"])
        .reset_index(drop=True)
    )
    diagnostics = (
        pd.DataFrame(diagnostics_all)
        .sort_values("rep")
        .reset_index(drop=True)
    )

    randomized_summary = summarize_randomized(randomized_raw)
    full_summary = summarize_full_methods(full_raw)

    randomized_raw_path = out_dir / "randomized_raw_results.csv"
    randomized_summary_path = out_dir / "randomized_summary_results.csv"
    full_raw_path = out_dir / "full_paired_raw_results.csv"
    full_summary_path = out_dir / "full_paired_summary_results.csv"
    diagnostics_path = out_dir / "full_mode_adaptive_diagnostics.csv"

    randomized_raw.to_csv(randomized_raw_path, index=False)
    randomized_summary.to_csv(randomized_summary_path, index=False)
    full_raw.to_csv(full_raw_path, index=False)
    full_summary.to_csv(full_summary_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print("\n========== Paired full-sample methods ==========")
    print(full_summary.to_string(index=False))

    print("\n========== Randomized S2ACT summary ==========")
    show_cols = [
        "gamma",
        "L1_high",
        "L2_low",
        "B",
        "row_average_rank",
        "row_accuracy",
        "col_average_rank",
        "col_accuracy",
        "pair_accuracy",
        "row_A_k_adj_mean",
        "row_A_next_adj_mean",
        "col_A_k_raw_mean",
        "col_A_next_raw_mean",
    ]
    print(randomized_summary[show_cols].to_string(index=False))

    print("\n========== Full mode-adaptive diagnostics ==========")
    print(
        "high side: mean adjusted lambda_k="
        f"{diagnostics['full_row_adj_k'].mean():.6f}, "
        "mean adjusted lambda_(k+1)="
        f"{diagnostics['full_row_adj_next'].mean():.6f}, "
        f"tau_1={tau_H:.6f}"
    )
    print(
        "low side: mean RAW lambda_k="
        f"{diagnostics['full_col_raw_k'].mean():.6f}, "
        "mean RAW lambda_(k+1)="
        f"{diagnostics['full_col_raw_next'].mean():.6f}, "
        f"tau_2={tau_L:.6f}"
    )
    print(
        f"full row accuracy={diagnostics['full_row_correct'].mean():.4f}, "
        f"full col accuracy={diagnostics['full_col_correct'].mean():.4f}, "
        f"full pair accuracy={diagnostics['full_pair_correct'].mean():.4f}"
    )

    print("\nSaved:")
    for path in [
        randomized_raw_path,
        randomized_summary_path,
        full_raw_path,
        full_summary_path,
        diagnostics_path,
        out_dir / "config.json",
    ]:
        print(f"  {path}")

    print(f"\nTotal elapsed time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
