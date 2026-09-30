#!/usr/bin/env python3
"""Combine independent-outer accuracy with dense Full-ACT benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pareto_mask(time_values: np.ndarray, error_values: np.ndarray) -> np.ndarray:
    """Return points not dominated when both time and error are minimized."""
    keep = np.ones(len(time_values), dtype=bool)
    for i in range(len(time_values)):
        dominated = (
            (time_values <= time_values[i])
            & (error_values <= error_values[i])
            & (
                (time_values < time_values[i])
                | (error_values < error_values[i])
            )
        )
        dominated[i] = False
        keep[i] = not bool(np.any(dominated))
    return keep


def representative_points(pareto: pd.DataFrame) -> pd.DataFrame:
    """Select fastest, balanced and most accurate frontier settings."""
    if pareto.empty:
        return pareto.assign(display_role=pd.Series(dtype=str))

    fastest_index = pareto["time_median_seconds"].idxmin()
    accurate_index = pareto.sort_values(
        ["row_mae", "time_median_seconds"]
    ).index[0]

    log_time = np.log(pareto["time_median_seconds"].to_numpy(float))
    error = pareto["row_mae"].to_numpy(float)
    time_range = float(np.ptp(log_time))
    error_range = float(np.ptp(error))
    normalized_time = (
        np.zeros(len(pareto))
        if time_range == 0.0
        else (log_time - np.min(log_time)) / time_range
    )
    normalized_error = (
        np.zeros(len(pareto))
        if error_range == 0.0
        else (error - np.min(error)) / error_range
    )
    balanced_index = pareto.index[
        int(np.argmin(np.hypot(normalized_time, normalized_error)))
    ]

    # A degenerate frontier can select the same setting for multiple roles.
    # Keep a compact combined label rather than drawing duplicate annotations.
    role_lists: dict[int, list[str]] = {}
    for index, role in [
        (fastest_index, "Fastest"),
        (balanced_index, "Balanced"),
        (accurate_index, "Most accurate"),
    ]:
        role_lists.setdefault(int(index), []).append(role)
    selected = pareto.loc[list(role_lists)].copy()
    selected["display_role"] = [
        " / ".join(role_lists[int(index)]) for index in selected.index
    ]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--p", type=int, default=3000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--show-all-settings",
        action="store_true",
        help=(
            "Plot all tested S2ACT settings. By default, the main-text "
            "figure shows only nondominated S2ACT tuning settings."
        ),
    )
    args = parser.parse_args()

    results = args.results_dir.expanduser().resolve()
    outer_file = results / "outer_independent_summary.csv"
    benchmark_file = results / "runtime_memory_benchmark_summary.csv"
    outer = pd.read_csv(outer_file)
    benchmark = pd.read_csv(benchmark_file)
    outer = outer[outer["p"] == args.p].copy()
    sketch_time = benchmark[
        (benchmark["p"] == args.p)
        & benchmark["method"].astype(str).str.startswith("S2ACT")
    ].copy()
    if outer.empty or sketch_time.empty:
        raise ValueError("No matching outer or benchmark rows were found.")

    for frame in (outer, sketch_time):
        frame["gamma_high"] = frame["gamma_high"].astype(float).round(10)
        frame["B"] = frame["B"].astype(int)
    frontier = outer.merge(
        sketch_time[
            [
                "gamma_high",
                "B",
                "time_median_seconds",
                "time_q25_seconds",
                "time_q75_seconds",
                "peak_memory_median_mb",
                "speedup_vs_dense_full",
            ]
        ],
        on=["gamma_high", "B"],
        how="inner",
        validate="one_to_one",
    )
    if len(frontier) != len(outer):
        missing = outer.merge(
            frontier[["gamma_high", "B"]],
            on=["gamma_high", "B"],
            how="left",
            indicator=True,
        )
        missing = missing[missing["_merge"] == "left_only"]
        raise ValueError(
            "Some outer settings have no benchmark match: "
            + missing[["gamma_high", "B"]].to_string(index=False)
        )

    frontier["pareto_time_mae"] = pareto_mask(
        frontier["time_median_seconds"].to_numpy(float),
        frontier["row_mae"].to_numpy(float),
    ).astype(int)
    frontier["relative_mae_percent"] = (
        100.0 * frontier["row_mae"] / frontier["full_row_rank"]
    )
    frontier = frontier.sort_values(["time_median_seconds", "row_mae"])
    output_csv = results / f"CEDAR_p{args.p}_accuracy_runtime_frontier.csv"

    full_rows = benchmark[
        (benchmark["p"] == args.p)
        & (benchmark["method"] == "Full-ACT-dense-direct")
    ]
    full_times = dict(
        zip(full_rows["method"], full_rows["time_median_seconds"])
    )

    pareto = frontier[frontier["pareto_time_mae"] == 1].sort_values(
        "time_median_seconds"
    )
    representatives = representative_points(pareto)
    frontier["display_role"] = ""
    for index, row in representatives.iterrows():
        frontier.loc[index, "display_role"] = row["display_role"]
    frontier.to_csv(output_csv, index=False)
    representatives.to_csv(
        results / f"CEDAR_p{args.p}_representative_settings.csv",
        index=False,
    )

    figure_suffix = "_all_settings" if args.show_all_settings else ""
    pdf_file = results / (
        f"CEDAR_p{args.p}_accuracy_runtime_frontier{figure_suffix}.pdf"
    )
    png_file = results / (
        f"CEDAR_p{args.p}_accuracy_runtime_frontier{figure_suffix}.png"
    )
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        plot_data = frontier if args.show_all_settings else pareto
        ax.plot(
            pareto["time_median_seconds"],
            pareto["row_mae"],
            color="#c43c39",
            linewidth=1.8,
            zorder=1.5,
            label=r"$S^2$ACT tuning frontier",
        )
        scatter = ax.scatter(
            plot_data["time_median_seconds"],
            plot_data["row_mae"],
            c=plot_data["gamma_high"],
            s=38.0 + 2.0 * plot_data["B"],
            cmap="viridis",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.85,
            zorder=3,
        )
        role_offsets = {
            "Fastest": (8, 9, "left"),
            "Balanced": (8, 9, "left"),
            "Most accurate": (-8, 10, "right"),
        }
        for _, row in representatives.iterrows():
            role = str(row["display_role"])
            offset_x, offset_y, alignment = role_offsets.get(
                role, (8, 9, "left")
            )
            ax.annotate(
                role
                + "\n"
                + fr"$\gamma_1={row['gamma_high']:g},\ B={int(row['B'])}$",
                (row["time_median_seconds"], row["row_mae"]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=8.0,
                ha=alignment,
                va="bottom",
            )
        if "Full-ACT-dense-direct" in full_times:
            ax.axvline(
                full_times["Full-ACT-dense-direct"],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="Full-ACT runtime",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Median runtime (seconds, log scale)")
        ax.set_ylabel("MAE relative to the Full-ACT row rank (rank units)")
        ax.grid(True, which="both", alpha=0.25)
        method_legend = ax.legend(
            frameon=False, fontsize=8.5, loc="upper right"
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label(r"High-mode sketch ratio $\gamma_1$")
        b_values = sorted(plot_data["B"].astype(int).unique())
        size_handles = [
            ax.scatter(
                [],
                [],
                s=38.0 + 2.0 * b_value,
                facecolor="#b8b8b8",
                edgecolor="black",
                linewidth=0.45,
                label=fr"$B={b_value}$",
            )
            for b_value in b_values
        ]
        size_legend = ax.legend(
            handles=size_handles,
            title="Aggregation size",
            frameon=False,
            fontsize=8,
            title_fontsize=8.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(4, len(size_handles)),
            handletextpad=0.3,
            columnspacing=1.0,
        )
        ax.add_artist(method_legend)
        full_rank = int(round(float(frontier["full_row_rank"].iloc[0])))
        ax.text(
            0.015,
            0.02,
            rf"Frontier among $S^2$ACT settings; Full-ACT row rank = {full_rank}",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#555555",
            ha="left",
            va="bottom",
        )
        ax.margins(x=0.08, y=0.10)
        fig.subplots_adjust(left=0.13, right=0.88, top=0.97, bottom=0.25)
        fig.savefig(pdf_file, bbox_inches="tight")
        fig.savefig(png_file, dpi=240, bbox_inches="tight")
        plt.close(fig)

    columns = [
        "gamma_high",
        "B",
        "row_agreement",
        "row_mae",
        "relative_mae_percent",
        "row_rank_median",
        "time_median_seconds",
        "speedup_vs_dense_full",
        "peak_memory_median_mb",
        "pareto_time_mae",
    ]
    print(frontier[columns].to_string(index=False), flush=True)
    print(f"Saved: {output_csv}", flush=True)
    if not args.no_plot:
        print(f"Saved: {pdf_file}", flush=True)


if __name__ == "__main__":
    main()
