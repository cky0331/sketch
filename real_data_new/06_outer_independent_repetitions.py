#!/usr/bin/env python3
"""Independent outer repetitions for conditional S2ACT randomization accuracy."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from s2act_core import (
    adjusted_eigenvalues_high,
    high_mode_observations,
    low_mode_matrices,
    max_threshold_rank,
    parse_float_list,
    parse_int_list,
    prefix_rank,
    sorted_eigvals,
    spectrum_from_observations,
    spectrum_from_resampled_indices,
    tau_high,
    tau_low,
)


_U_HIGH: np.ndarray | None = None
_U_LOW: np.ndarray | None = None
_CFG: dict[str, object] | None = None
_THREAD_CONTROLLER = None


def wilson_interval(successes: int, total: int, z_value: float = 1.959964) -> tuple[float, float]:
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
    return max(0.0, center - radius), min(1.0, center + radius)


def initialize_worker(
    u_high: np.ndarray,
    u_low: np.ndarray,
    config: dict[str, object],
) -> None:
    global _U_HIGH, _U_LOW, _CFG, _THREAD_CONTROLLER
    _U_HIGH = u_high
    _U_LOW = u_low
    _CFG = config
    try:
        from threadpoolctl import threadpool_limits

        _THREAD_CONTROLLER = threadpool_limits(
            limits=int(config["worker_blas_threads"])
        )
    except ImportError:
        _THREAD_CONTROLLER = None


def run_one_outer_replication(replication: int) -> list[dict[str, float | int]]:
    if _U_HIGH is None or _U_LOW is None or _CFG is None:
        raise RuntimeError("Worker data were not initialized.")
    cfg = _CFG
    n = int(cfg["n"])
    p = int(cfg["p"])
    q = int(cfg["q"])
    kmax_high = int(cfg["kmax_high"])
    kmax_low = int(cfg["kmax_low"])
    threshold_high = float(cfg["tau_high"])
    threshold_low = float(cfg["tau_low"])
    gamma_grid = [float(x) for x in cfg["gamma_grid"]]
    l_high_values = [int(x) for x in cfg["l_high_values"]]
    l_low = int(cfg["l_low"])
    b_grid = [int(x) for x in cfg["b_grid"]]
    b_max = max(b_grid)
    full_row_rank = int(cfg["full_row_rank"])
    full_column_rank = int(cfg["full_column_rank"])
    seed = int(cfg["seed"])

    high_rng = np.random.default_rng(np.random.SeedSequence([seed, replication, 101]))
    low_rng = np.random.default_rng(np.random.SeedSequence([seed, replication, 202]))
    max_l_high = max(l_high_values)
    high_banks = {
        level: np.empty((b_max, kmax_high), dtype=np.float64)
        for level in sorted(set(l_high_values))
    }
    low_bank = np.empty((b_max, kmax_low), dtype=np.float64)

    # Sketches are independent within and across outer repetitions.  Prefixes
    # of a common sample stream couple gamma values only to reduce comparison
    # noise; this does not alter the marginal distribution at any gamma.
    for b in range(b_max):
        ids = high_rng.integers(0, _U_HIGH.shape[0], size=max_l_high)
        for level in sorted(high_banks):
            raw = spectrum_from_resampled_indices(
                _U_HIGH, ids[:level], correlation=True
            )
            high_banks[level][b, :] = adjusted_eigenvalues_high(
                raw,
                n=n,
                q=q,
                kmax=kmax_high,
            )

        low_ids = low_rng.integers(0, _U_LOW.shape[0], size=l_low)
        low_raw = spectrum_from_observations(_U_LOW[low_ids], correlation=True)
        low_bank[b, :] = low_raw[:kmax_low]

    rows: list[dict[str, float | int]] = []
    for gamma, level in zip(gamma_grid, l_high_values):
        high_bank = high_banks[level]
        for b_value in b_grid:
            high_median = np.nanmedian(high_bank[:b_value], axis=0)
            low_median = np.nanmedian(low_bank[:b_value], axis=0)
            row_rank = prefix_rank(high_median, threshold_high, kmax_high)
            column_rank = prefix_rank(low_median, threshold_low, kmax_low)
            rows.append(
                {
                    "p": p,
                    "outer_rep": replication,
                    "gamma_high": gamma,
                    "gamma_low": float(cfg["gamma_low"]),
                    "L1_high": level,
                    "L2_low": l_low,
                    "B": b_value,
                    "full_row_rank": full_row_rank,
                    "full_column_rank": full_column_rank,
                    "row_rank": row_rank,
                    "column_rank": column_rank,
                    "row_agreement": int(row_rank == full_row_rank),
                    "column_agreement": int(column_rank == full_column_rank),
                    "pair_agreement": int(
                        row_rank == full_row_rank
                        and column_rank == full_column_rank
                    ),
                    "row_error": row_rank - full_row_rank,
                    "column_error": column_rank - full_column_rank,
                    "row_over": int(row_rank > full_row_rank),
                    "row_under": int(row_rank < full_row_rank),
                    "column_over": int(column_rank > full_column_rank),
                    "column_under": int(column_rank < full_column_rank),
                }
            )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "p",
        "gamma_high",
        "gamma_low",
        "L1_high",
        "L2_low",
        "B",
        "full_row_rank",
        "full_column_rank",
    ]
    rows: list[dict[str, float | int]] = []
    for keys, part in raw.groupby(group_columns, sort=True):
        row: dict[str, float | int] = dict(zip(group_columns, keys))
        repetitions = int(part["outer_rep"].nunique())
        row["outer_repetitions"] = repetitions
        for target in ("row_agreement", "column_agreement", "pair_agreement"):
            successes = int(part[target].sum())
            lower, upper = wilson_interval(successes, repetitions)
            probability = successes / repetitions
            row[target] = probability
            row[f"{target}_ci_low"] = lower
            row[f"{target}_ci_high"] = upper
            row[f"{target}_mcse"] = math.sqrt(
                probability * (1.0 - probability) / repetitions
            )
        row["row_rank_mean"] = float(part["row_rank"].mean())
        row["row_rank_median"] = float(part["row_rank"].median())
        row["row_rank_q05"] = float(part["row_rank"].quantile(0.05))
        row["row_rank_q95"] = float(part["row_rank"].quantile(0.95))
        row["row_rank_min"] = int(part["row_rank"].min())
        row["row_rank_max"] = int(part["row_rank"].max())
        row["row_bias"] = float(part["row_error"].mean())
        row["row_mae"] = float(np.abs(part["row_error"]).mean())
        row["row_over_rate"] = float(part["row_over"].mean())
        row["row_under_rate"] = float(part["row_under"].mean())
        row["column_rank_mean"] = float(part["column_rank"].mean())
        row["column_rank_median"] = float(part["column_rank"].median())
        row["column_mae"] = float(np.abs(part["column_error"]).mean())
        row["column_over_rate"] = float(part["column_over"].mean())
        row["column_under_rate"] = float(part["column_under"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--p", required=True, type=int)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--kmax", type=int, default=200)
    parser.add_argument("--gamma-grid", default="0.10,0.25,0.50,0.75,1.00")
    parser.add_argument("--gamma-low", type=float, default=0.10)
    parser.add_argument("--B-grid", default="1,5,20,50")
    parser.add_argument("--outer-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--worker-blas-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gamma_grid = parse_float_list(args.gamma_grid)
    b_grid = parse_int_list(args.B_grid)
    if any(g <= 0.0 or g > 1.0 for g in gamma_grid):
        raise ValueError("Every gamma must be in (0,1].")
    if not (0.0 < args.gamma_low <= 1.0):
        raise ValueError("gamma-low must be in (0,1].")
    if args.outer_reps < 2 or args.workers < 1 or args.worker_blas_threads < 1:
        raise ValueError("outer-reps>=2, workers>=1 and worker-blas-threads>=1 are required.")
    if args.workers > 1:
        try:
            import threadpoolctl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Parallel sketching requires threadpoolctl to prevent BLAS "
                "oversubscription. Install it with: python3 -m pip install threadpoolctl"
            ) from exc

    processed = args.processed_dir.expanduser().resolve()
    results = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else processed / f"CEDAR_results_p{args.p}"
    )
    results.mkdir(parents=True, exist_ok=True)
    data_file = processed / f"CEDAR_Z_centered_p{args.p}.npy"
    z = np.asarray(np.load(data_file), dtype=np.float64)
    if z.ndim != 3 or z.shape[1] != args.p or np.any(~np.isfinite(z)):
        raise ValueError(f"Unexpected tensor: {z.shape}")
    z = z - np.mean(z, axis=0, keepdims=True)
    n, p, q = z.shape
    kmax_high = min(args.kmax, p - 1)
    kmax_low = min(args.kmax, q)
    threshold_high = tau_high(n, p, q)
    threshold_low, delta_low = tau_low(n)
    u_high = high_mode_observations(z)
    low_cov, low_corr, u_low = low_mode_matrices(z)

    print("Computing the Full-ACT reference rank...", flush=True)
    full_start = time.perf_counter()
    full_raw_high = spectrum_from_observations(u_high, correlation=True)
    full_adjusted_high = adjusted_eigenvalues_high(
        full_raw_high, n=n, q=q, kmax=kmax_high
    )
    full_low_values = sorted_eigvals(low_corr)
    full_row_rank = max_threshold_rank(full_adjusted_high, threshold_high, kmax_high)
    full_column_rank = max_threshold_rank(full_low_values, threshold_low, kmax_low)
    full_seconds = time.perf_counter() - full_start
    print(
        f"Full-ACT=({full_row_rank},{full_column_rank}); "
        f"reference time={full_seconds:.3f}s",
        flush=True,
    )

    l_high_values = [max(2, int(round(g * n * q))) for g in gamma_grid]
    l_low = max(2, int(round(args.gamma_low * n * p)))
    signature = {
        "p": p,
        "shape": [n, p, q],
        "kmax": args.kmax,
        "gamma_grid": gamma_grid,
        "gamma_low": args.gamma_low,
        "B_grid": b_grid,
        "outer_reps": args.outer_reps,
        "seed": args.seed,
        "full_row_rank": full_row_rank,
        "full_column_rank": full_column_rank,
        "tau_high": threshold_high,
        "tau_low": threshold_low,
        "delta_low": delta_low,
        "L1_high": l_high_values,
        "L2_low": l_low,
        "workers": args.workers,
        "worker_blas_threads": args.worker_blas_threads,
        "outer_independence": (
            "each outer repetition creates B_max new independent high and low sketches; "
            "B settings use nested prefixes and gamma settings use nested sample streams"
        ),
    }
    raw_file = results / "outer_independent_raw.csv"
    summary_file = results / "outer_independent_summary.csv"
    config_file = results / "outer_independent_config.json"

    if args.overwrite:
        for target in (raw_file, summary_file, config_file):
            if target.exists():
                target.unlink()
    elif config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        comparison_keys = [
            "p",
            "shape",
            "kmax",
            "gamma_grid",
            "gamma_low",
            "B_grid",
            "outer_reps",
            "seed",
        ]
        if any(old.get(key) != signature.get(key) for key in comparison_keys):
            raise RuntimeError(
                "Existing outer-run configuration differs. Use another results directory "
                "or pass --overwrite explicitly."
            )
    config_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    expected_rows_per_rep = len(gamma_grid) * len(b_grid)
    completed: set[int] = set()
    if raw_file.exists():
        existing = pd.read_csv(raw_file)
        counts = existing.groupby("outer_rep").size()
        completed = set(int(x) for x in counts[counts == expected_rows_per_rep].index)
        # Discard a partially written repetition before resuming it.  Each
        # repetition is deterministic from (seed, rep), so rerunning it is safe.
        existing = existing[existing["outer_rep"].isin(completed)]
        existing.to_csv(raw_file, index=False)
        print(f"Resuming: {len(completed)} completed outer repetitions found.", flush=True)
    pending = [rep for rep in range(args.outer_reps) if rep not in completed]
    if not pending:
        print("All requested outer repetitions are already complete.", flush=True)

    worker_config: dict[str, object] = {
        **signature,
        "n": n,
        "q": q,
        "kmax_high": kmax_high,
        "kmax_low": kmax_low,
        "l_high_values": l_high_values,
        "l_low": l_low,
        "b_grid": b_grid,
    }
    start = time.perf_counter()

    def append_result(rows: list[dict[str, float | int]]) -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(
            raw_file,
            mode="a",
            header=not raw_file.exists() or raw_file.stat().st_size == 0,
            index=False,
        )

    if pending and args.workers == 1:
        initialize_worker(u_high, u_low, worker_config)
        for count, rep in enumerate(pending, start=1):
            append_result(run_one_outer_replication(rep))
            print(
                f"completed outer rep {rep + 1}/{args.outer_reps} "
                f"({count}/{len(pending)} pending)",
                flush=True,
            )
    elif pending:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        context = mp.get_context(method)
        with context.Pool(
            processes=args.workers,
            initializer=initialize_worker,
            initargs=(u_high, u_low, worker_config),
        ) as pool:
            for count, rows in enumerate(
                pool.imap_unordered(run_one_outer_replication, pending, chunksize=1),
                start=1,
            ):
                append_result(rows)
                rep = int(rows[0]["outer_rep"])
                print(
                    f"completed outer rep {rep + 1}/{args.outer_reps} "
                    f"({count}/{len(pending)} pending)",
                    flush=True,
                )

    raw = pd.read_csv(raw_file).sort_values(
        ["outer_rep", "gamma_high", "B"]
    )
    raw.to_csv(raw_file, index=False)
    summary = summarize(raw)
    summary.to_csv(summary_file, index=False)
    elapsed = time.perf_counter() - start
    signature["new_work_elapsed_seconds"] = elapsed
    signature["completed_outer_repetitions"] = int(raw["outer_rep"].nunique())
    config_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"Saved raw results: {raw_file}")
    print(f"Saved summary:     {summary_file}")
    print(f"New-work elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
