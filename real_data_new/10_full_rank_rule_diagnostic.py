#!/usr/bin/env python3
"""Diagnose max-threshold versus prefix-threshold Full-ACT rank rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from s2act_core import (
    adjusted_eigenvalues_high,
    high_mode_observations,
    low_mode_matrices,
    max_threshold_rank,
    prefix_rank,
    spectrum_from_observations,
    tau_high,
    tau_low,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--p", type=int, default=3000)
    parser.add_argument("--kmax", type=int, default=200)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    processed = args.processed_dir.expanduser().resolve()
    results = args.results_dir.expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)
    z = np.asarray(
        np.load(processed / f"CEDAR_Z_centered_p{args.p}.npy"),
        dtype=np.float64,
    )
    z -= np.mean(z, axis=0, keepdims=True)
    n, p, q = z.shape
    high_threshold = tau_high(n, p, q)
    low_threshold, _ = tau_low(n)
    kmax_high = min(args.kmax, p - 1)
    kmax_low = min(args.kmax, q)

    high_raw = spectrum_from_observations(
        high_mode_observations(z), correlation=True
    )
    high_adjusted = adjusted_eigenvalues_high(
        high_raw, n=n, q=q, kmax=kmax_high
    )
    _, low_corr, _ = low_mode_matrices(z)
    low_values = np.linalg.eigvalsh(low_corr)[::-1]

    high_max = max_threshold_rank(high_adjusted, high_threshold, kmax_high)
    high_prefix = prefix_rank(high_adjusted, high_threshold, kmax_high)
    low_max = max_threshold_rank(low_values, low_threshold, kmax_low)
    low_prefix = prefix_rank(low_values, low_threshold, kmax_low)
    passed = np.flatnonzero(
        np.isfinite(high_adjusted) & (high_adjusted > high_threshold)
    ) + 1
    first_failed = high_prefix + 1 if high_prefix < kmax_high else np.nan

    spectrum = pd.DataFrame(
        {
            "index": np.arange(1, len(high_adjusted) + 1),
            "raw_correlation_eigenvalue": high_raw[: len(high_adjusted)],
            "act_adjusted_eigenvalue": high_adjusted,
            "threshold": high_threshold,
            "passes_threshold": (
                np.isfinite(high_adjusted) & (high_adjusted > high_threshold)
            ).astype(int),
        }
    )
    spectrum.to_csv(
        results / f"CEDAR_p{p}_full_rank_rule_spectrum.csv", index=False
    )
    diagnostic = pd.DataFrame(
        [
            {
                "p": p,
                "n": n,
                "q": q,
                "tau_high": high_threshold,
                "high_max_threshold_rank": high_max,
                "high_prefix_threshold_rank": high_prefix,
                "high_first_failed_index": first_failed,
                "high_last_passing_index": int(passed[-1]) if len(passed) else 0,
                "high_number_passing": int(len(passed)),
                "high_nonmonotone_crossing": int(high_max != high_prefix),
                "tau_low": low_threshold,
                "low_max_threshold_rank": low_max,
                "low_prefix_threshold_rank": low_prefix,
            }
        ]
    )
    output_csv = results / f"CEDAR_p{p}_full_rank_rule_diagnostic.csv"
    diagnostic.to_csv(output_csv, index=False)

    plot_end = min(kmax_high, max(high_max + 15, high_prefix + 25, 30))
    plot_start = max(1, min(high_prefix, high_max) - 15)
    part = spectrum[
        (spectrum["index"] >= plot_start) & (spectrum["index"] <= plot_end)
    ]
    pdf_file = results / f"CEDAR_p{p}_full_rank_rule_diagnostic.pdf"
    png_file = results / f"CEDAR_p{p}_full_rank_rule_diagnostic.png"
    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
        axes[0].plot(
            spectrum["index"], spectrum["act_adjusted_eigenvalue"], linewidth=1.1
        )
        axes[0].axhline(high_threshold, color="black", linestyle="--", linewidth=1.1)
        axes[0].set_xlim(1, kmax_high)
        positive = spectrum["act_adjusted_eigenvalue"] > 0
        if bool(positive.all()):
            axes[0].set_yscale("log")
        axes[0].set_xlabel("Component index")
        axes[0].set_ylabel("ACT-adjusted eigenvalue")
        axes[0].set_title("Leading adjusted spectrum")
        axes[0].grid(True, alpha=0.22)

        colors = np.where(part["passes_threshold"] == 1, "#2a788e", "#d1495b")
        axes[1].plot(
            part["index"],
            part["act_adjusted_eigenvalue"],
            color="#555555",
            linewidth=1.0,
        )
        axes[1].scatter(
            part["index"], part["act_adjusted_eigenvalue"], c=colors, s=24, zorder=3
        )
        axes[1].axhline(high_threshold, color="black", linestyle="--", linewidth=1.1)
        axes[1].axvline(
            high_prefix + 0.5,
            color="#d1495b",
            linestyle=":",
            linewidth=1.2,
            label=f"prefix rank={high_prefix}",
        )
        axes[1].axvline(
            high_max,
            color="#2a788e",
            linestyle="-.",
            linewidth=1.2,
            label=f"max rank={high_max}",
        )
        axes[1].set_xlabel("Component index")
        axes[1].set_ylabel("ACT-adjusted eigenvalue")
        axes[1].set_title("Threshold crossings near the rank boundary")
        axes[1].grid(True, alpha=0.22)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(pdf_file, bbox_inches="tight")
        fig.savefig(png_file, dpi=240, bbox_inches="tight")
        plt.close(fig)

    print(diagnostic.to_string(index=False), flush=True)
    print("High-mode indices above threshold:", passed.tolist(), flush=True)
    print(f"Saved: {output_csv}", flush=True)
    if not args.no_plot:
        print(f"Saved: {pdf_file}", flush=True)


if __name__ == "__main__":
    main()
