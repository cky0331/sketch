#!/usr/bin/env python3
"""Create the dense Full-ACT versus S2ACT scalability figure from saved runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def method_rows(
    summary: pd.DataFrame, gamma_high: float, aggregation_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the dense reference and the requested S2ACT configuration."""
    dense = summary[
        summary["method"].astype(str).eq("Full-ACT-dense-direct")
    ].copy()
    gamma_values = pd.to_numeric(summary["gamma_high"], errors="coerce")
    b_values = pd.to_numeric(summary["B"], errors="coerce")
    s2act = summary[
        summary["method"].astype(str).str.startswith("S2ACT")
        & np.isclose(gamma_values, gamma_high, equal_nan=False)
        & np.isclose(b_values, aggregation_size, equal_nan=False)
    ].copy()
    if dense.empty:
        raise ValueError("No Full-ACT-dense-direct rows were found.")
    if s2act.empty:
        raise ValueError(
            "No S2ACT rows match "
            f"gamma_high={gamma_high:g}, B={aggregation_size}."
        )
    return dense, s2act


def combine_rows(dense: pd.DataFrame, s2act: pd.DataFrame) -> pd.DataFrame:
    """Align methods by problem size and calculate dense-reference speedup."""
    size_keys = ["n", "p", "q"]
    for name, frame in [("dense Full-ACT", dense), ("S2ACT", s2act)]:
        duplicates = frame.duplicated(size_keys, keep=False)
        if duplicates.any():
            repeated = frame.loc[duplicates, size_keys].drop_duplicates()
            raise ValueError(
                f"Multiple {name} rows were found for the same size:\n"
                + repeated.to_string(index=False)
            )

    dense_columns = size_keys + [
        "time_median_seconds",
        "time_q25_seconds",
        "time_q75_seconds",
        "peak_memory_median_mb",
    ]
    s2act_columns = size_keys + [
        "aspect_ratio",
        "gamma_high",
        "gamma_low",
        "L1_high",
        "L2_low",
        "B",
        "repetitions",
        "time_median_seconds",
        "time_q25_seconds",
        "time_q75_seconds",
        "peak_memory_median_mb",
        "row_accuracy",
        "column_accuracy",
        "pair_accuracy",
        "row_mae",
        "column_mae",
    ]
    missing = [
        column
        for column in dense_columns
        if column not in dense.columns
    ] + [
        column
        for column in s2act_columns
        if column not in s2act.columns
    ]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(set(missing))))

    combined = dense[dense_columns].merge(
        s2act[s2act_columns],
        on=size_keys,
        how="inner",
        suffixes=("_dense", "_s2act"),
        validate="one_to_one",
    )
    if combined.empty:
        raise ValueError("Dense Full-ACT and S2ACT have no common problem sizes.")
    if len(combined) != len(dense) or len(combined) != len(s2act):
        raise ValueError(
            "Dense Full-ACT and S2ACT problem-size grids do not match exactly."
        )
    combined["speedup_vs_dense_full"] = (
        combined["time_median_seconds_dense"]
        / combined["time_median_seconds_s2act"]
    )
    return combined.sort_values(["p", "n"]).reset_index(drop=True)


def asymmetric_iqr(
    median: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return nonnegative asymmetric errors for matplotlib."""
    return np.maximum(0.0, np.vstack([median - lower, upper - median]))


def save_figure(combined: pd.DataFrame, output_stem: Path, b_value: int) -> None:
    """Draw a two-method scalability figure with runtime IQR bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    x = combined["p"].to_numpy(float)
    dense_time = combined["time_median_seconds_dense"].to_numpy(float)
    sketch_time = combined["time_median_seconds_s2act"].to_numpy(float)
    dense_error = asymmetric_iqr(
        dense_time,
        combined["time_q25_seconds_dense"].to_numpy(float),
        combined["time_q75_seconds_dense"].to_numpy(float),
    )
    sketch_error = asymmetric_iqr(
        sketch_time,
        combined["time_q25_seconds_s2act"].to_numpy(float),
        combined["time_q75_seconds_s2act"].to_numpy(float),
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        x,
        dense_time,
        yerr=dense_error,
        color="#4d4d4d",
        marker="s",
        markersize=5.5,
        linewidth=1.7,
        linestyle="--",
        capsize=3,
        label="Full-ACT",
        zorder=2,
    )
    ax.errorbar(
        x,
        sketch_time,
        yerr=sketch_error,
        color="#c43c39",
        marker="o",
        markersize=6.0,
        linewidth=1.9,
        linestyle="-",
        capsize=3,
        label=rf"$S^2$ACT ($\gamma_1=1$, $B={b_value}$)",
        zorder=3,
    )
    for p_value, runtime, speedup in zip(
        x,
        sketch_time,
        combined["speedup_vs_dense_full"].to_numpy(float),
    ):
        ax.annotate(
            f"{speedup:.2f}x faster",
            (p_value, runtime),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.2,
            color="#9f2f2c",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(value)}" for value in x])
    # Keep the three observed dimensions as the only x-axis labels.  On a
    # log axis, matplotlib otherwise adds labels such as 3x10^3 and 6x10^3
    # next to 2988 and 5988, which makes the labels overlap.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_xlabel(r"High-mode dimension $p$")
    ax.set_ylabel("Median estimator runtime (seconds, log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.10, y=0.25)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma-high", type=float, default=1.0)
    parser.add_argument("--B", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    summary_file = args.summary_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_file)
    dense, s2act = method_rows(summary, args.gamma_high, args.B)
    combined = combine_rows(dense, s2act)

    label = f"B{args.B}"
    csv_file = output_dir / f"S2ACT_scalability_dense_only_{label}.csv"
    figure_stem = output_dir / f"S2ACT_scalability_runtime_{label}"
    combined.to_csv(csv_file, index=False)
    if not args.no_plot:
        save_figure(combined, figure_stem, args.B)

    display_columns = [
        "n",
        "p",
        "time_median_seconds_dense",
        "time_median_seconds_s2act",
        "speedup_vs_dense_full",
        "pair_accuracy",
    ]
    print(combined[display_columns].to_string(index=False), flush=True)
    print(f"Saved: {csv_file}", flush=True)
    if not args.no_plot:
        print(f"Saved: {figure_stem.with_suffix('.pdf')}", flush=True)
        print(f"Saved: {figure_stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
