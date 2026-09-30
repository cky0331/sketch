#!/usr/bin/env python3
"""Numerical core for the theory-aligned CEDAR S2ACT analysis."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

try:
    from scipy.linalg import eigvalsh as scipy_eigvalsh
except ImportError:  # NumPy remains a fully functional fallback.
    scipy_eigvalsh = None


EPS = 1.0e-12


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("The list cannot be empty.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("The list cannot be empty.")
    return values


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return (matrix + matrix.T) / 2.0


def sorted_eigvals(matrix: np.ndarray) -> np.ndarray:
    """Return descending eigenvalues after checking numerical PSD error."""
    work = symmetrize(matrix)
    if scipy_eigvalsh is None:
        vals = np.linalg.eigvalsh(work)[::-1]
    else:
        # Divide-and-conquer is normally faster when the complete spectrum is
        # required by the ACT transform.  check_finite=False is safe because
        # callers validate the observations before forming the Gram matrix.
        vals = scipy_eigvalsh(
            work,
            check_finite=False,
            overwrite_a=True,
            driver="evd",
        )[::-1]
    scale = max(1.0, float(np.max(np.abs(vals))))
    if float(np.min(vals)) < -1.0e-8 * scale:
        raise np.linalg.LinAlgError(
            f"Matrix is not positive semidefinite: min eigenvalue={np.min(vals):.3e}"
        )
    vals[vals < 0.0] = 0.0
    return vals


def covariance_from_observations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("x must be a nonempty two-dimensional array.")
    return symmetrize((x.T @ x) / float(x.shape[0]))


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    cov = symmetrize(cov)
    diagonal = np.diag(cov)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= EPS):
        raise ValueError("Correlation normalization found a nonpositive diagonal entry.")
    inv_sd = 1.0 / np.sqrt(diagonal)
    corr = cov * inv_sd[:, None] * inv_sd[None, :]
    np.fill_diagonal(corr, 1.0)
    return symmetrize(corr)


def spectrum_from_observations(
    x: np.ndarray,
    correlation: bool,
) -> np.ndarray:
    """
    Spectrum of X'X/m, using the smaller primal or companion matrix.

    The returned vector always has ambient length d.  When m < d, zeros are
    appended after diagonalizing the m by m companion Gram matrix.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    m, d = x.shape
    if m < 1 or d < 1 or np.any(~np.isfinite(x)):
        raise ValueError("x must be finite and nonempty.")

    work = x
    if correlation:
        diagonal = np.mean(x * x, axis=0)
        if np.any(diagonal <= EPS) or np.any(~np.isfinite(diagonal)):
            raise ValueError("Correlation normalization found a nonpositive variance.")
        work = x / np.sqrt(diagonal)[None, :]

    if m < d:
        nonzero = sorted_eigvals((work @ work.T) / float(m))
        vals = np.concatenate([nonzero, np.zeros(d - m, dtype=float)])
    else:
        vals = sorted_eigvals((work.T @ work) / float(m))
    return vals


def spectrum_from_resampled_indices(
    x: np.ndarray,
    indices: np.ndarray,
    correlation: bool,
) -> np.ndarray:
    """Spectrum of a with-replacement sample, compressing duplicate rows.

    If an original row occurs ``count`` times among ``L`` draws, replacing
    those copies by one row multiplied by ``sqrt(U * count / L)`` gives the
    same covariance (and hence the same correlation matrix), where ``U`` is
    the number of distinct sampled rows.  The smaller companion matrix then
    has dimension ``U`` instead of ``L``.  This is an exact computational
    reduction, not a change to the sketching distribution.
    """
    x = np.asarray(x, dtype=float)
    indices = np.asarray(indices)
    if x.ndim != 2 or indices.ndim != 1 or indices.size < 1:
        raise ValueError("x must be two-dimensional and indices must be nonempty.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must have an integer dtype.")
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= x.shape[0]:
        raise IndexError("A sampled row index is out of range.")

    unique_indices, counts = np.unique(indices, return_counts=True)
    unique_count = unique_indices.size
    weights = np.sqrt(
        unique_count * counts.astype(np.float64) / float(indices.size)
    )
    compressed = x[unique_indices] * weights[:, None]
    return spectrum_from_observations(compressed, correlation=correlation)


def low_mode_matrices(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return low-mode covariance, correlation and pooled observations."""
    n, p, q = z.shape
    u_low = z.reshape(n * p, q)
    cov = covariance_from_observations(u_low)
    corr = correlation_from_cov(cov)
    return cov, corr, u_low


def high_mode_observations(z: np.ndarray) -> np.ndarray:
    n, p, q = z.shape
    return z.transpose(0, 2, 1).reshape(n * q, p)


def tau_high(n: int, p: int, q: int) -> float:
    if n <= 1 or p <= 0 or q <= 0:
        raise ValueError("tau_high requires n>1, p>0 and q>0.")
    return float(1.0 + np.sqrt(p / (q * (n - 1.0))))


def tau_low(n: int, delta_low: float | None = None) -> Tuple[float, float]:
    if n <= 1:
        raise ValueError("tau_low requires n>1.")
    delta = float(n ** (-0.25) if delta_low is None else delta_low)
    if delta <= 0.0:
        raise ValueError("delta_low must be positive.")
    return 1.0 + delta, delta


def adjusted_eigenvalues_high(
    vals: np.ndarray,
    n: int,
    q: int,
    kmax: int,
) -> np.ndarray:
    """
    ACT adjustment for the high-dimensional mode.

    The aspect ratio is always based on the original q(n-1), including for a
    sketch.  The sketch size L must not replace q(n-1) here.
    """
    vals = np.asarray(vals, dtype=float)
    p = len(vals)
    kmax_eff = min(int(kmax), p - 1)
    df = float(q * (n - 1))
    out = np.full(kmax_eff, np.nan, dtype=float)

    for j0 in range(kmax_eff):
        j = j0 + 1
        z = max(float(vals[j0]), EPS)
        nxt = max(float(vals[j0 + 1]), EPS)
        tail = vals[j0 + 1 :]
        denom = tail - z
        denom = np.where(np.abs(denom) < 1.0e-10, -1.0e-10, denom)
        stabilizer = (3.0 * z + nxt) / 4.0 - z
        if abs(stabilizer) < 1.0e-10:
            stabilizer = -1.0e-10
        mhat = (np.sum(1.0 / denom) + 1.0 / stabilizer) / float(p - j)
        rho_j = (p - j) / df
        mcomp = -(1.0 - rho_j) / z + rho_j * mhat
        out[j0] = np.inf if abs(mcomp) < EPS else -1.0 / mcomp
    return out


def companion_transform_values_high(
    vals: np.ndarray,
    n: int,
    q: int,
    kmax: int,
) -> np.ndarray:
    """Return the companion-transform values used by the ACT correction."""
    vals = np.asarray(vals, dtype=float)
    p = len(vals)
    kmax_eff = min(int(kmax), p - 1)
    df = float(q * (n - 1))
    out = np.full(kmax_eff, np.nan, dtype=float)

    for j0 in range(kmax_eff):
        j = j0 + 1
        z = max(float(vals[j0]), EPS)
        nxt = max(float(vals[j0 + 1]), EPS)
        tail = vals[j0 + 1 :]
        denom = tail - z
        denom = np.where(np.abs(denom) < 1.0e-10, -1.0e-10, denom)
        stabilizer = (3.0 * z + nxt) / 4.0 - z
        if abs(stabilizer) < 1.0e-10:
            stabilizer = -1.0e-10
        mhat = (np.sum(1.0 / denom) + 1.0 / stabilizer) / float(p - j)
        rho_j = (p - j) / df
        out[j0] = -(1.0 - rho_j) / z + rho_j * mhat
    return out


def prefix_rank(vals: np.ndarray, threshold: float, kmax: int) -> int:
    """Sketch rule: stop at the first coordinate that fails the threshold."""
    vals = np.asarray(vals, dtype=float)
    rank = 0
    for value in vals[: min(int(kmax), len(vals))]:
        if np.isfinite(value) and value > threshold:
            rank += 1
        else:
            break
    return int(rank)


def max_threshold_rank(vals: np.ndarray, threshold: float, kmax: int) -> int:
    """Full-sample rule: max{j <= kmax: value_j > threshold}."""
    vals = np.asarray(vals, dtype=float)
    passed = np.flatnonzero(
        np.isfinite(vals[: min(int(kmax), len(vals))])
        & (vals[: min(int(kmax), len(vals))] > threshold)
    )
    return 0 if len(passed) == 0 else int(passed[-1] + 1)


def eigen_ratio_rank_from_values(
    vals: np.ndarray,
    kmax: int,
    eps: float = EPS,
) -> int:
    vals = np.asarray(vals, dtype=float)
    kmax_eff = min(int(kmax), len(vals) - 1)
    if kmax_eff < 1:
        return 0
    ratios = np.maximum(vals[:kmax_eff], eps) / np.maximum(
        vals[1 : kmax_eff + 1], eps
    )
    return int(np.argmax(ratios) + 1)


def build_nested_high_sketch_bank(
    u_high: np.ndarray,
    l_values: Sequence[int],
    bank_size: int,
    n: int,
    q: int,
    kmax: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """Create nested high-mode banks of ACT-adjusted sketch eigencurves."""
    pool_size, p = u_high.shape
    levels = sorted(set(int(x) for x in l_values))
    if not levels or min(levels) < 2:
        raise ValueError("High-mode sketch sizes must be at least 2.")
    if bank_size < 1:
        raise ValueError("bank_size must be positive.")

    rng = np.random.default_rng(seed)
    sampled_ids = rng.integers(0, pool_size, size=(bank_size, max(levels)))
    kmax_eff = min(int(kmax), p - 1)
    banks = {level: np.empty((bank_size, kmax_eff)) for level in levels}

    for b in range(bank_size):
        for level in levels:
            x = u_high[sampled_ids[b, :level], :]
            raw = spectrum_from_observations(x, correlation=True)
            banks[level][b, :] = adjusted_eigenvalues_high(raw, n, q, kmax_eff)
    return banks


def build_low_sketch_bank(
    u_low: np.ndarray,
    sketch_size: int,
    bank_size: int,
    kmax: int,
    seed: int,
) -> np.ndarray:
    """Create a low-mode bank of raw-correlation eigencurves (no ACT)."""
    pool_size, q = u_low.shape
    if sketch_size < 2 or bank_size < 1:
        raise ValueError("Low-mode sketch size and bank size must be positive.")
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, pool_size, size=(bank_size, sketch_size))
    out = np.empty((bank_size, min(int(kmax), q)))
    for b in range(bank_size):
        cov = covariance_from_observations(u_low[ids[b]])
        vals = sorted_eigvals(correlation_from_cov(cov))
        out[b, :] = vals[: out.shape[1]]
    return out


def top_correlation_eigenvectors(x: np.ndarray, kmax: int) -> np.ndarray:
    """Leading eigenvectors of the feature correlation without forming d by d."""
    x = np.asarray(x, dtype=float)
    m, d = x.shape
    k = min(int(kmax), m, d)
    if k < 1:
        raise ValueError("kmax must be positive.")
    diagonal = np.mean(x * x, axis=0)
    if np.any(diagonal <= EPS):
        raise ValueError("A sampled sketch has a nonpositive coordinate variance.")
    y = x / np.sqrt(diagonal)[None, :]

    if m <= d:
        gram = symmetrize((y @ y.T) / float(m))
        vals, left = np.linalg.eigh(gram)
        order = np.argsort(vals)[::-1][:k]
        vals = vals[order]
        left = left[:, order]
        keep = vals > EPS
        if int(np.sum(keep)) < k:
            raise np.linalg.LinAlgError("The sketch has fewer positive eigenvalues than kmax.")
        vectors = y.T @ left[:, keep]
        vectors /= np.sqrt(float(m) * vals[keep])[None, :]
        vectors, _ = np.linalg.qr(vectors, mode="reduced")
        return vectors[:, :k]

    corr = correlation_from_cov(covariance_from_observations(x))
    vals, vectors = np.linalg.eigh(corr)
    return vectors[:, np.argsort(vals)[::-1][:k]]


def pairwise_subspace_instability(bases: Iterable[np.ndarray]) -> np.ndarray:
    """Mean 1-||U_b' U_b'||_F^2/k over all pairs, for k=1,...,kmax."""
    bases = list(bases)
    if len(bases) < 2:
        raise ValueError("At least two sketch bases are required.")
    kmax = min(x.shape[1] for x in bases)
    totals = np.zeros(kmax, dtype=float)
    pairs = 0
    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            cross = bases[left].T @ bases[right]
            cumulative = np.cumsum(np.cumsum(cross * cross, axis=0), axis=1)
            for k in range(1, kmax + 1):
                totals[k - 1] += 1.0 - cumulative[k - 1, k - 1] / float(k)
            pairs += 1
    return totals / float(pairs)
