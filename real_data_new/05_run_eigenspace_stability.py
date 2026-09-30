#!/usr/bin/env python3
"""Compute the pairwise eigenspace-instability diagnostic I_s(k)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTTING_IMPORT_ERROR = None
except ImportError as exc:
    plt = None
    PLOTTING_IMPORT_ERROR = exc


from s2act_core import (
    high_mode_observations,
    parse_float_list,
    top_correlation_eigenvectors,
)


def sampled_bases(
    observations: np.ndarray,
    sketch_size: int,
    repeats: int,
    kmax: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate leading correlation eigenspaces from repeated sketches."""
    rng = np.random.default_rng(seed)

    bases: list[np.ndarray] = []

    for repeat in range(repeats):
        ids = rng.integers(
            0,
            observations.shape[0],
            size=sketch_size,
        )

        bases.append(
            top_correlation_eigenvectors(
                observations[ids],
                kmax,
            )
        )

        print(
            f"  eigenspace sketch {repeat + 1}/{repeats}",
            flush=True,
        )

    return bases


def pairwise_subspace_instability_summary(
    bases: list[np.ndarray],
) -> dict[str, np.ndarray | int]:
    """
    Compute pairwise eigenspace-instability curves and summarize them.

    For two rank-k eigenspaces with orthonormal bases U and V,

        I(k) = 1 - ||U^T V||_F^2 / k.

    The returned mean corresponds to the averaged pairwise instability.
    Quantiles describe the empirical dispersion across sketch pairs and
    are used only as descriptive variability bands.
    """
    if len(bases) < 2:
        raise ValueError(
            "At least two sketch bases are required."
        )

    kmax = min(
        basis.shape[1]
        for basis in bases
    )

    pair_curves: list[np.ndarray] = []

    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):

            cross = (
                bases[left].T
                @ bases[right]
            )

            # cumulative[k-1, k-1] equals the squared Frobenius norm
            # of the leading k x k block of U_left^T U_right.
            cumulative = np.cumsum(
                np.cumsum(
                    cross * cross,
                    axis=0,
                ),
                axis=1,
            )

            curve = np.empty(
                kmax,
                dtype=float,
            )

            for k in range(1, kmax + 1):

                value = (
                    1.0
                    - cumulative[k - 1, k - 1]
                    / float(k)
                )

                # Theoretically I_s(k) is in [0, 1].
                # Clip tiny floating-point excursions.
                curve[k - 1] = np.clip(
                    value,
                    0.0,
                    1.0,
                )

            pair_curves.append(curve)

    pair_curves = np.asarray(
        pair_curves,
        dtype=float,
    )

    return {
        "mean": np.mean(
            pair_curves,
            axis=0,
        ),
        "q25": np.quantile(
            pair_curves,
            0.25,
            axis=0,
        ),
        "q75": np.quantile(
            pair_curves,
            0.75,
            axis=0,
        ),
        "q05": np.quantile(
            pair_curves,
            0.05,
            axis=0,
        ),
        "q95": np.quantile(
            pair_curves,
            0.95,
            axis=0,
        ),
        "num_pairs": pair_curves.shape[0],
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--p",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--gamma-grid",
        default="0.10,0.25,0.50,0.75,1.00",
        help="High-mode sketch ratios.",
    )

    parser.add_argument(
        "--gamma-low-grid",
        default="0.01,0.05,0.10",
        help="Low-mode sketch ratios.",
    )

    parser.add_argument(
        "--B",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--kmax",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260908,
    )

    parser.add_argument(
        "--full-row-rank",
        type=int,
        default=73,
        help=(
            "Full-sample high-mode rank used only "
            "for the vertical reference line."
        ),
    )

    parser.add_argument(
        "--skip-plot",
        action="store_true",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Parse sketch-ratio grids
    # ------------------------------------------------------------

    gamma_grid = parse_float_list(
        args.gamma_grid
    )

    gamma_low_grid = parse_float_list(
        args.gamma_low_grid
    )

    if any(
        gamma <= 0.0 or gamma > 1.0
        for gamma in gamma_grid
    ):
        raise ValueError(
            "Every high-mode gamma must be in (0,1]."
        )

    if any(
        gamma <= 0.0 or gamma > 1.0
        for gamma in gamma_low_grid
    ):
        raise ValueError(
            "Every low-mode gamma must be in (0,1]."
        )

    if args.B < 2:
        raise ValueError(
            "B must be at least 2."
        )

    if args.kmax < 1:
        raise ValueError(
            "kmax must be positive."
        )

    if (
        not args.skip_plot
        and PLOTTING_IMPORT_ERROR is not None
    ):
        raise ImportError(
            "matplotlib is required for the figure; "
            "install it or use --skip-plot"
        ) from PLOTTING_IMPORT_ERROR

    # ------------------------------------------------------------
    # Input / output paths
    # ------------------------------------------------------------

    processed = (
        args.processed_dir
        .expanduser()
        .resolve()
    )

    data_file = (
        processed
        / f"CEDAR_Z_centered_p{args.p}.npy"
    )

    results = (
        args.results_dir
        .expanduser()
        .resolve()
        if args.results_dir is not None
        else processed
        / f"CEDAR_results_p{args.p}"
    )

    results.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Load centered tensor
    # ------------------------------------------------------------

    z = np.asarray(
        np.load(data_file),
        dtype=np.float64,
    )

    if (
        z.ndim != 3
        or z.shape[1] != args.p
        or np.any(~np.isfinite(z))
    ):
        raise ValueError(
            f"Unexpected tensor: "
            f"shape={z.shape}, "
            f"requested p={args.p}"
        )

    # Recenter for numerical safety.
    z = z - np.mean(
        z,
        axis=0,
        keepdims=True,
    )

    n, p, q = z.shape

    print(
        f"Loaded tensor: n={n}, p={p}, q={q}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Construct pooled observations
    # ------------------------------------------------------------

    u_high = high_mode_observations(z)

    # Low-mode pooled row observations.
    u_low = z.reshape(
        n * p,
        q,
    )

    k_high = min(
        args.kmax,
        p,
        n * q,
    )

    k_low = min(
        args.kmax,
        q,
    )

    print(
        f"k_high={k_high}, k_low={k_low}",
        flush=True,
    )

    rows: list[
        dict[str, float | int | str]
    ] = []

    # ============================================================
    # High-dimensional mode
    # ============================================================

    for position, gamma in enumerate(
        gamma_grid
    ):

        sketch_size = max(
            2,
            int(
                round(
                    gamma * n * q
                )
            ),
        )

        print(
            f"High mode gamma={gamma:g}, "
            f"L1={sketch_size}",
            flush=True,
        )

        bases = sampled_bases(
            observations=u_high,
            sketch_size=sketch_size,
            repeats=args.B,
            kmax=k_high,
            seed=(
                args.seed
                + 10_000 * (position + 1)
                + 101
            ),
        )

        summary = (
            pairwise_subspace_instability_summary(
                bases
            )
        )

        for k in range(
            1,
            len(summary["mean"]) + 1,
        ):

            mean_value = float(
                summary["mean"][k - 1]
            )

            q25_value = float(
                summary["q25"][k - 1]
            )

            q75_value = float(
                summary["q75"][k - 1]
            )

            q05_value = float(
                summary["q05"][k - 1]
            )

            q95_value = float(
                summary["q95"][k - 1]
            )

            rows.append(
                {
                    "mode": "high",
                    "gamma": gamma,
                    "sketch_size": sketch_size,
                    "B": args.B,
                    "number_of_pairs": int(
                        summary["num_pairs"]
                    ),
                    "k": k,
                    "I_mean": mean_value,
                    "I_q25": q25_value,
                    "I_q75": q75_value,
                    "I_q05": q05_value,
                    "I_q95": q95_value,
                    "eta": args.eta,
                    "passes_eta": int(
                        mean_value <= args.eta
                    ),
                }
            )

    # ============================================================
    # Low-dimensional cell-type mode
    # ============================================================

    for position, gamma_low in enumerate(
        gamma_low_grid
    ):

        low_size = max(
            2,
            int(
                round(
                    gamma_low * n * p
                )
            ),
        )

        print(
            f"Low mode gamma={gamma_low:g}, "
            f"L2={low_size}",
            flush=True,
        )

        low_bases = sampled_bases(
            observations=u_low,
            sketch_size=low_size,
            repeats=args.B,
            kmax=k_low,
            seed=(
                args.seed
                + 20_000 * (position + 1)
                + 202
            ),
        )

        low_summary = (
            pairwise_subspace_instability_summary(
                low_bases
            )
        )

        for k in range(
            1,
            len(low_summary["mean"]) + 1,
        ):

            mean_value = float(
                low_summary["mean"][k - 1]
            )

            q25_value = float(
                low_summary["q25"][k - 1]
            )

            q75_value = float(
                low_summary["q75"][k - 1]
            )

            q05_value = float(
                low_summary["q05"][k - 1]
            )

            q95_value = float(
                low_summary["q95"][k - 1]
            )

            rows.append(
                {
                    "mode": "low",
                    "gamma": gamma_low,
                    "sketch_size": low_size,
                    "B": args.B,
                    "number_of_pairs": int(
                        low_summary["num_pairs"]
                    ),
                    "k": k,
                    "I_mean": mean_value,
                    "I_q25": q25_value,
                    "I_q75": q75_value,
                    "I_q05": q05_value,
                    "I_q95": q95_value,
                    "eta": args.eta,
                    "passes_eta": int(
                        mean_value <= args.eta
                    ),
                }
            )

    # ============================================================
    # Save numerical results
    # ============================================================

    frame = pd.DataFrame(rows)

    output_csv = (
        results
        / "eigenspace_stability.csv"
    )

    frame.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"Saved numerical results: "
        f"{output_csv}",
        flush=True,
    )

    # ============================================================
    # Plot
    # ============================================================

    if not args.skip_plot:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10.8, 4.3),
            sharey=False,
        )

        # --------------------------------------------------------
        # High mode
        # --------------------------------------------------------

        high = frame[
            frame["mode"] == "high"
        ]

        for gamma, part in high.groupby(
            "gamma",
            sort=True,
        ):

            part = part.sort_values("k")

            axes[0].plot(
                part["k"],
                part["I_mean"],
                "o-",
                linewidth=1.4,
                markersize=3.0,
                label=(
                    fr"$\gamma_1={gamma:g}$"
                ),
            )

        # Stability reference threshold.
        axes[0].axhline(
            args.eta,
            color="crimson",
            ls="--",
            linewidth=1.1,
            label=fr"$\eta={args.eta:g}$",
        )

        # Full-sample high-mode rank.
        if (
            args.full_row_rank >= 1
            and args.full_row_rank <= k_high
        ):
            axes[0].axvline(
                args.full_row_rank,
                color="gray",
                ls=":",
                linewidth=1.1,
            )

        axes[0].set_title(
            "High-mode eigenspace instability"
        )

        axes[0].set_xlabel(
            r"Candidate rank $k$"
        )

        axes[0].set_ylabel(
            r"$I_1(k)$"
        )

        axes[0].legend(
            frameon=False,
            fontsize=8,
            loc="lower right",
        )

        # --------------------------------------------------------
        # Low mode
        # --------------------------------------------------------

        low = frame[
            frame["mode"] == "low"
        ]

        for gamma, part in low.groupby(
            "gamma",
            sort=True,
        ):

            part = part.sort_values("k")

            x = part[
                "k"
            ].to_numpy()

            y = part[
                "I_mean"
            ].to_numpy()

            y_low = part[
                "I_q25"
            ].to_numpy()

            y_high = part[
                "I_q75"
            ].to_numpy()

            # Plot the mean curve first so that the shadow
            # can use exactly the same color.
            line, = axes[1].plot(
                x,
                y,
                "o-",
                linewidth=1.5,
                markersize=4.0,
                label=(
                    fr"$\gamma_2={gamma:g}$"
                ),
            )

            # Interquartile variability band across
            # repeated sketch pairs.
            axes[1].fill_between(
                x,
                y_low,
                y_high,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
            )

        axes[1].set_title(
            "Low-mode eigenspace instability"
        )

        axes[1].set_xlabel(
            r"Candidate rank $k$"
        )

        axes[1].set_ylabel(
            r"$I_2(k)$"
        )

        axes[1].legend(
            frameon=False,
            fontsize=8,
            loc="upper right",
        )

        # --------------------------------------------------------
        # Zoom the low-mode y-axis.
        #
        # Do not share the scale with the high mode because
        # low-mode instability is much smaller.
        # --------------------------------------------------------

        low_upper = float(
            low["I_q75"].max()
        )

        low_ylim_upper = max(
            0.01,
            1.15 * low_upper,
        )

        axes[1].set_ylim(
            0.0,
            low_ylim_upper,
        )

        axes[1].ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(-2, 2),
        )

        fig.tight_layout()

        stem = (
            results
            / f"CEDAR_p{p}_eigenspace_stability"
        )

        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
        )

        fig.savefig(
            stem.with_suffix(".png"),
            dpi=220,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved figure: "
            f"{stem.with_suffix('.pdf')}",
            flush=True,
        )

        print(
            f"Saved figure: "
            f"{stem.with_suffix('.png')}",
            flush=True,
        )


if __name__ == "__main__":
    main()