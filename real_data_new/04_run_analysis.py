#!/usr/bin/env python3
"""Run the centered CEDAR full-data and two-sided S2ACT analysis."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    PLOTTING_IMPORT_ERROR = None
except ImportError as exc:  # Analysis can still be smoke-tested without figures.
    plt = None
    ScalarFormatter = None
    PLOTTING_IMPORT_ERROR = exc

from s2act_core import (
    adjusted_eigenvalues_high,
    build_low_sketch_bank,
    build_nested_high_sketch_bank,
    companion_transform_values_high,
    eigen_ratio_rank_from_values,
    high_mode_observations,
    low_mode_matrices,
    max_threshold_rank,
    parse_float_list,
    parse_int_list,
    prefix_rank,
    sorted_eigvals,
    spectrum_from_observations,
    tau_high,
    tau_low,
)


def wilson_interval(successes: int, total: int, z_value: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z_value * z_value / total
    center = (proportion + z_value * z_value / (2.0 * total)) / denominator
    radius = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value * z_value / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def peak_rss_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            value /= 1024.0
        return value / 1024.0
    except (ImportError, AttributeError):
        return float("nan")


def threshold_diagnostics(values: np.ndarray, threshold: float, rank: int) -> dict[str, float]:
    if rank > 0:
        signal_margin = float(np.min(values[:rank] - threshold))
    else:
        signal_margin = float("nan")
    noise_margin = float(threshold - values[rank]) if rank < len(values) else float("nan")
    finite = [x for x in (signal_margin, noise_margin) if np.isfinite(x)]
    gap = float(min(finite)) if finite else float("nan")
    return {
        "signal_margin": signal_margin,
        "noise_margin": noise_margin,
        "G_full": gap,
    }


def save_spectrum_figure(
    high_values: np.ndarray,
    low_values: np.ndarray,
    threshold_high: float,
    threshold_low: float,
    output_stem: Path,
    high_limit: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    high_count = min(high_limit, len(high_values))
    axes[0].plot(np.arange(1, high_count + 1), high_values[:high_count], "o-", ms=3)
    axes[0].axhline(threshold_high, color="crimson", ls="--", label="high-mode threshold")
    axes[0].set_title(f"High mode: first {high_count} ACT values")
    axes[0].set_xlabel("index j")
    axes[0].set_ylabel("ACT-adjusted eigenvalue")
    axes[0].legend(frameon=False)

    low_count = len(low_values)
    axes[1].plot(np.arange(1, low_count + 1), low_values, "o-", ms=4)
    axes[1].axhline(threshold_low, color="crimson", ls="--", label="low-mode threshold")
    axes[1].set_title("Low mode: raw correlation eigenvalues")
    axes[1].set_xlabel("index j")
    axes[1].set_ylabel("raw eigenvalue")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scale_figure(variances: np.ndarray, output_stem: Path) -> None:
    ordered = np.sort(variances)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(np.arange(1, len(ordered) + 1), ordered, lw=1.1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ordered probe index")
    axes[0].set_ylabel("Centered probe variance")
    axes[0].set_title("Sorted probe variances")
    axes[1].hist(np.log10(variances), bins=40, color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel(r"$\log_{10}$(centered probe variance)")
    axes[1].set_ylabel("Number of probes")
    axes[1].set_title("Distribution of probe variances")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_agreement_figure(summary: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    targets = [
        ("row_agreement", "High-mode rank"),
        ("column_agreement", "Low-mode rank"),
        ("pair_agreement", "Joint rank pair"),
    ]
    for ax, (column, title) in zip(axes, targets):
        for gamma, part in summary.groupby("gamma_high", sort=True):
            part = part.sort_values("B")
            y = part[column].to_numpy()
            lower = part[f"{column}_ci_low"].to_numpy()
            upper = part[f"{column}_ci_high"].to_numpy()
            ax.errorbar(
                part["B"],
                y,
                yerr=np.maximum(0.0, np.vstack([y - lower, upper - y])),
                marker="o",
                capsize=2,
                label=f"gamma={gamma:g}",
            )
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary["B"].unique()))
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("median aggregation B")
        ax.set_title(title)
    axes[0].set_ylabel("agreement with Full-ACT")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--p", required=True, type=int)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--kmax", type=int, default=200)
    parser.add_argument("--gamma-grid", default="0.10,0.25,0.5,0.75,1.00")
    parser.add_argument("--gamma-low", type=float, default=0.10)
    parser.add_argument("--B-grid", default="1,5,20,50")
    parser.add_argument("--bank-size", type=int, default=50)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--delta-low", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    gamma_grid = parse_float_list(args.gamma_grid)
    b_grid = parse_int_list(args.B_grid)
    if any(g <= 0.0 or g > 1.0 for g in gamma_grid):
        raise ValueError("Every high-mode gamma must be in (0,1].")
    if not (0.0 < args.gamma_low <= 1.0):
        raise ValueError("--gamma-low must be in (0,1].")
    if any(b <= 0 for b in b_grid) or max(b_grid) > args.bank_size:
        raise ValueError("B values must be positive and no larger than --bank-size.")
    if args.bootstrap_reps < 1 or args.kmax < 2:
        raise ValueError("bootstrap-reps must be positive and kmax must be at least 2.")
    if not args.skip_plots and PLOTTING_IMPORT_ERROR is not None:
        raise ImportError(
            "matplotlib is required for figures; install it or use --skip-plots"
        ) from PLOTTING_IMPORT_ERROR

    processed = args.processed_dir.expanduser().resolve()
    data_file = processed / f"CEDAR_Z_centered_p{args.p}.npy"
    results = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else processed / f"CEDAR_results_p{args.p}"
    )
    results.mkdir(parents=True, exist_ok=True)
    if not data_file.exists():
        raise FileNotFoundError(f"Run 03_make_tensors.py first; missing {data_file}")

    start_total = time.perf_counter()
    z = np.asarray(np.load(data_file), dtype=np.float64)
    if z.ndim != 3 or z.shape[1] != args.p or np.any(~np.isfinite(z)):
        raise ValueError(f"Unexpected tensor: shape={z.shape}, requested p={args.p}")
    pre_recenter_error = float(np.max(np.abs(np.mean(z, axis=0))))
    z = z - np.mean(z, axis=0, keepdims=True)
    center_error = float(np.max(np.abs(np.mean(z, axis=0))))
    n, p, q = z.shape
    kmax_high = min(args.kmax, p - 1)
    kmax_low = min(args.kmax, q)
    threshold_high = tau_high(n, p, q)
    threshold_low, delta_used = tau_low(n, args.delta_low)
    print(f"Centered tensor: n={n}, p={p}, q={q}, p/[q(n-1)]={p/(q*(n-1)):.6f}")
    print(f"Thresholds: high={threshold_high:.6f}; low={threshold_low:.6f}")
    print(f"Centering check: before={pre_recenter_error:.3e}; after={center_error:.3e}")

    u_high = high_mode_observations(z)
    low_cov, low_corr, u_low = low_mode_matrices(z)

    full_start = time.perf_counter()
    high_cov_eigenvalues = spectrum_from_observations(u_high, correlation=False)
    high_corr_eigenvalues = spectrum_from_observations(u_high, correlation=True)
    low_cov_eigenvalues = sorted_eigvals(low_cov)
    low_corr_eigenvalues = sorted_eigvals(low_corr)
    high_adjusted = adjusted_eigenvalues_high(
        high_corr_eigenvalues, n=n, q=q, kmax=kmax_high
    )
    full_seconds = time.perf_counter() - full_start

    full_row_rank = max_threshold_rank(high_adjusted, threshold_high, kmax_high)
    full_column_rank = max_threshold_rank(low_corr_eigenvalues, threshold_low, kmax_low)
    cov_row_rank = eigen_ratio_rank_from_values(high_cov_eigenvalues, kmax_high)
    cov_column_rank = eigen_ratio_rank_from_values(low_cov_eigenvalues, kmax_low)
    corr_row_rank = eigen_ratio_rank_from_values(high_corr_eigenvalues, kmax_high)
    corr_column_rank = eigen_ratio_rank_from_values(low_corr_eigenvalues, kmax_low)
    print(f"Full-ACT rank: ({full_row_rank}, {full_column_rank})")
    print(f"Cov-ER rank:   ({cov_row_rank}, {cov_column_rank})")
    print(f"Corr-ER rank:  ({corr_row_rank}, {corr_column_rank})")

    high_diag = threshold_diagnostics(high_adjusted, threshold_high, full_row_rank)
    low_diag = threshold_diagnostics(low_corr_eigenvalues, threshold_low, full_column_rank)
    gap_end = min(full_row_rank + 1, len(high_corr_eigenvalues) - 1)
    if gap_end >= 1:
        raw_high_gap = float(
            np.min(high_corr_eigenvalues[:gap_end] - high_corr_eigenvalues[1 : gap_end + 1])
        )
    else:
        raw_high_gap = float("nan")
    mcomp_count = min(full_row_rank + 1, len(high_corr_eigenvalues) - 1)
    mcomp = companion_transform_values_high(
        high_corr_eigenvalues, n=n, q=q, kmax=max(1, mcomp_count)
    )
    mcomp_abs_min = float(np.nanmin(np.abs(mcomp[:mcomp_count]))) if mcomp_count else float("nan")
    variances = np.mean(z * z, axis=(0, 2))
    if np.any(variances <= 0.0):
        raise ValueError("Centered high-mode diagonal contains nonpositive values.")

    diagnostics = pd.DataFrame(
        [
            {
                "p": p,
                "n": n,
                "q": q,
                "aspect_p_over_q_nminus1": p / (q * (n - 1)),
                "tau_high": threshold_high,
                "tau_low": threshold_low,
                "delta_low": delta_used,
                "full_row_rank": full_row_rank,
                "full_column_rank": full_column_rank,
                "high_signal_margin_full": high_diag["signal_margin"],
                "high_noise_margin_full": high_diag["noise_margin"],
                "G1_full": high_diag["G_full"],
                "low_signal_margin_full": low_diag["signal_margin"],
                "low_noise_margin_full": low_diag["noise_margin"],
                "G2_full": low_diag["G_full"],
                "raw_high_eigengap_min_j_le_kplus1": raw_high_gap,
                "mcomp_abs_min_j_le_kplus1": mcomp_abs_min,
                "center_error_before_internal_recenter": pre_recenter_error,
                "center_error_after_internal_recenter": center_error,
                "centered_variance_min": float(np.min(variances)),
                "centered_variance_q05": float(np.quantile(variances, 0.05)),
                "centered_variance_median": float(np.median(variances)),
                "centered_variance_q95": float(np.quantile(variances, 0.95)),
                "centered_variance_max": float(np.max(variances)),
                "centered_variance_ratio_max_min": float(np.max(variances) / np.min(variances)),
            }
        ]
    )
    diagnostics.to_csv(results / "full_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "probe_index": np.arange(1, p + 1),
            "centered_variance": variances,
        }
    ).to_csv(results / "scale_heterogeneity.csv", index=False)

    row_spectrum_count = min(args.kmax + 2, p)
    pd.DataFrame(
        {
            "index": np.arange(1, row_spectrum_count + 1),
            "covariance_eigenvalue": high_cov_eigenvalues[:row_spectrum_count],
            "correlation_eigenvalue": high_corr_eigenvalues[:row_spectrum_count],
            "act_adjusted_eigenvalue": np.pad(
                high_adjusted,
                (0, max(0, row_spectrum_count - len(high_adjusted))),
                constant_values=np.nan,
            )[:row_spectrum_count],
            "tau_high": threshold_high,
        }
    ).to_csv(results / "row_spectrum.csv", index=False)
    pd.DataFrame(
        {
            "index": np.arange(1, q + 1),
            "covariance_eigenvalue": low_cov_eigenvalues,
            "raw_correlation_eigenvalue": low_corr_eigenvalues,
            "tau_low": threshold_low,
        }
    ).to_csv(results / "column_spectrum.csv", index=False)

    l_high_values = [max(2, int(round(gamma * n * q))) for gamma in gamma_grid]
    l_low = max(2, int(round(args.gamma_low * n * p)))
    print(f"High sketch sizes: {dict(zip(gamma_grid, l_high_values))}")
    print(f"Low sketch size: gamma_low={args.gamma_low:g}, L2={l_low}")
    high_bank_start = time.perf_counter()
    high_banks = build_nested_high_sketch_bank(
        u_high=u_high,
        l_values=l_high_values,
        bank_size=args.bank_size,
        n=n,
        q=q,
        kmax=kmax_high,
        seed=args.seed + 101,
    )
    high_bank_seconds = time.perf_counter() - high_bank_start
    low_bank_start = time.perf_counter()
    low_bank = build_low_sketch_bank(
        u_low=u_low,
        sketch_size=l_low,
        bank_size=args.bank_size,
        kmax=kmax_low,
        seed=args.seed + 202,
    )
    low_bank_seconds = time.perf_counter() - low_bank_start
    bank_seconds = high_bank_seconds + low_bank_seconds

    npz_content: dict[str, np.ndarray] = {"low": low_bank}
    for gamma, level in zip(gamma_grid, l_high_values):
        key = "high_gamma_" + str(gamma).replace(".", "p")
        npz_content[key] = high_banks[level]
    np.savez_compressed(results / f"sketch_bank_p{p}.npz", **npz_content)

    representative_rows: list[dict[str, float | int | str]] = []
    rank_rows: list[dict[str, float | int | str]] = [
        {
            "method": "Cov-ER",
            "gamma_high": np.nan,
            "gamma_low": np.nan,
            "L1_high": np.nan,
            "L2_low": np.nan,
            "B": np.nan,
            "row_rank": cov_row_rank,
            "column_rank": cov_column_rank,
            "rank_rule": "eigenvalue ratio",
        },
        {
            "method": "Corr-ER",
            "gamma_high": np.nan,
            "gamma_low": np.nan,
            "L1_high": np.nan,
            "L2_low": np.nan,
            "B": np.nan,
            "row_rank": corr_row_rank,
            "column_rank": corr_column_rank,
            "rank_rule": "eigenvalue ratio",
        },
        {
            "method": "Full-ACT",
            "gamma_high": np.nan,
            "gamma_low": np.nan,
            "L1_high": np.nan,
            "L2_low": np.nan,
            "B": np.nan,
            "row_rank": full_row_rank,
            "column_rank": full_column_rank,
            "rank_rule": "max-threshold",
        },
    ]
    for gamma, level in zip(gamma_grid, l_high_values):
        high_bank = high_banks[level]
        for b_value in b_grid:
            median_high = np.nanmedian(high_bank[:b_value], axis=0)
            median_low = np.nanmedian(low_bank[:b_value], axis=0)
            row_rank = prefix_rank(median_high, threshold_high, kmax_high)
            column_rank = prefix_rank(median_low, threshold_low, kmax_low)
            base = {
                "method": "S2ACT",
                "gamma_high": gamma,
                "gamma_low": args.gamma_low,
                "L1_high": level,
                "L2_low": l_low,
                "B": b_value,
                "row_rank": row_rank,
                "column_rank": column_rank,
                "rank_rule": "prefix-threshold after coordinatewise median",
            }
            rank_rows.append(base)
            representative_rows.append(
                {
                    **base,
                    "row_agrees_full": int(row_rank == full_row_rank),
                    "column_agrees_full": int(column_rank == full_column_rank),
                    "pair_agrees_full": int(
                        row_rank == full_row_rank and column_rank == full_column_rank
                    ),
                    "high_value_at_full_rank": (
                        float(median_high[full_row_rank - 1]) if full_row_rank > 0 else np.nan
                    ),
                    "high_value_after_full_rank": (
                        float(median_high[full_row_rank])
                        if full_row_rank < len(median_high)
                        else np.nan
                    ),
                    "low_value_at_full_rank": (
                        float(median_low[full_column_rank - 1])
                        if full_column_rank > 0
                        else np.nan
                    ),
                    "low_value_after_full_rank": (
                        float(median_low[full_column_rank])
                        if full_column_rank < len(median_low)
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(rank_rows).to_csv(results / "rank_summary.csv", index=False)
    pd.DataFrame(representative_rows).to_csv(
        results / "representative_sketch_results.csv", index=False
    )

    bootstrap_start = time.perf_counter()
    rng = np.random.default_rng(args.seed + 303)
    bootstrap_rows: list[dict[str, float | int]] = []
    for gamma, level in zip(gamma_grid, l_high_values):
        high_bank = high_banks[level]
        for b_value in b_grid:
            for replication in range(args.bootstrap_reps):
                high_ids = rng.integers(0, args.bank_size, size=b_value)
                low_ids = rng.integers(0, args.bank_size, size=b_value)
                median_high = np.nanmedian(high_bank[high_ids], axis=0)
                median_low = np.nanmedian(low_bank[low_ids], axis=0)
                row_rank = prefix_rank(median_high, threshold_high, kmax_high)
                column_rank = prefix_rank(median_low, threshold_low, kmax_low)
                bootstrap_rows.append(
                    {
                        "gamma_high": gamma,
                        "gamma_low": args.gamma_low,
                        "L1_high": level,
                        "L2_low": l_low,
                        "B": b_value,
                        "bootstrap_rep": replication,
                        "row_rank": row_rank,
                        "column_rank": column_rank,
                        "row_agreement": int(row_rank == full_row_rank),
                        "column_agreement": int(column_rank == full_column_rank),
                        "pair_agreement": int(
                            row_rank == full_row_rank and column_rank == full_column_rank
                        ),
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(results / "sketch_bootstrap_raw.csv", index=False)

    summary_rows: list[dict[str, float | int]] = []
    group_columns = ["gamma_high", "gamma_low", "L1_high", "L2_low", "B"]
    for keys, part in bootstrap.groupby(group_columns, sort=True):
        row: dict[str, float | int] = dict(zip(group_columns, keys))
        row["bootstrap_reps"] = len(part)
        for target in ("row_agreement", "column_agreement", "pair_agreement"):
            successes = int(part[target].sum())
            lower, upper = wilson_interval(successes, len(part))
            row[target] = float(part[target].mean())
            row[f"{target}_ci_low"] = lower
            row[f"{target}_ci_high"] = upper
        row["mean_abs_row_rank_error"] = float(
            np.mean(np.abs(part["row_rank"] - full_row_rank))
        )
        row["mean_abs_column_rank_error"] = float(
            np.mean(np.abs(part["column_rank"] - full_column_rank))
        )
        summary_rows.append(row)
    agreement = pd.DataFrame(summary_rows)
    agreement.to_csv(results / "sketch_agreement_summary.csv", index=False)
    bootstrap_seconds = time.perf_counter() - bootstrap_start

    accounting = pd.DataFrame(
        [
            {
                "operation": "dense full high-mode correlation",
                "matrix_dimension": p,
                "dense_matrix_memory_mb": 8.0 * p * p / 1024.0**2,
                "implementation": "not formed",
                "dominant_complexity": "O(p^2 n q + p^3)",
            },
            {
                "operation": "companion full high-mode spectrum",
                "matrix_dimension": n * q,
                "dense_matrix_memory_mb": 8.0 * (n * q) ** 2 / 1024.0**2,
                "implementation": "used",
                "dominant_complexity": "O((nq)^2 p + (nq)^3)",
            },
            {
                "operation": "largest high-mode sketch companion",
                "matrix_dimension": max(l_high_values),
                "dense_matrix_memory_mb": 8.0 * max(l_high_values) ** 2 / 1024.0**2,
                "implementation": "used for each sketch",
                "dominant_complexity": "O(L1^2 p + L1^3)",
            },
            {
                "operation": "low-mode sketch correlation",
                "matrix_dimension": q,
                "dense_matrix_memory_mb": 8.0 * q * q / 1024.0**2,
                "implementation": "used for each sketch",
                "dominant_complexity": "O(L2 q^2 + q^3)",
            },
        ]
    )
    accounting.to_csv(results / "computational_accounting.csv", index=False)
    runtime = pd.DataFrame(
        [
            {"stage": "full spectra and ranks", "seconds": full_seconds},
            {"stage": "high-mode sketch bank", "seconds": high_bank_seconds},
            {
                "stage": "high-mode seconds per sketch-gamma",
                "seconds": high_bank_seconds / (args.bank_size * len(l_high_values)),
            },
            {"stage": "low-mode sketch bank", "seconds": low_bank_seconds},
            {
                "stage": "low-mode seconds per sketch",
                "seconds": low_bank_seconds / args.bank_size,
            },
            {"stage": "all sketch banks", "seconds": bank_seconds},
            {"stage": "empirical-bank bootstrap", "seconds": bootstrap_seconds},
            {"stage": "total before figure rendering", "seconds": time.perf_counter() - start_total},
        ]
    )
    runtime["peak_rss_mb_at_write"] = peak_rss_mb()
    runtime.to_csv(results / "runtime_summary.csv", index=False)

    if not args.skip_plots:
        save_spectrum_figure(
            high_adjusted,
            low_corr_eigenvalues,
            threshold_high,
            threshold_low,
            results / f"CEDAR_p{p}_ACT_spectrum",
            high_limit=min(50, args.kmax),
        )
        save_scale_figure(variances, results / f"CEDAR_p{p}_scale_heterogeneity")
        save_agreement_figure(agreement, results / f"CEDAR_p{p}_sketch_agreement")

    config = {
        "data_file": str(data_file),
        "results_dir": str(results),
        "shape": [n, p, q],
        "centered_again_inside_analysis": True,
        "kmax": args.kmax,
        "gamma_high_grid": gamma_grid,
        "gamma_low_fixed": args.gamma_low,
        "L1_high": l_high_values,
        "L2_low": l_low,
        "B_grid": b_grid,
        "independent_bank_size": args.bank_size,
        "empirical_bank_bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "plots_skipped": args.skip_plots,
        "high_mode_bias_correction": "ACT with original q(n-1), never L1",
        "low_mode_bias_correction": "none; raw correlation eigenvalues",
        "full_rank_rule": "max threshold",
        "sketch_rank_rule": "prefix threshold after coordinatewise median",
        "sketch_sides": "both high and low modes",
        "bootstrap_interpretation": (
            "conditional empirical-bank bootstrap; it quantifies sketch sensitivity "
            "for this fixed real dataset and is not a population confidence interval"
        ),
    }
    (results / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Completed. Outputs: {results}")
    print(f"Elapsed seconds: {time.perf_counter() - start_total:.1f}")


if __name__ == "__main__":
    main()
