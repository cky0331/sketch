#!/usr/bin/env python3
"""Fair runtime/memory benchmark of direct dense Full-ACT and S2ACT."""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from s2act_core import (
    adjusted_eigenvalues_high,
    correlation_from_cov,
    covariance_from_observations,
    high_mode_observations,
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

try:
    import psutil
except ImportError as exc:
    raise ImportError("Install psutil before running the memory benchmark.") from exc

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError as exc:
    raise ImportError(
        "Install threadpoolctl before running the fair CPU-thread benchmark."
    ) from exc


_U_HIGH: np.ndarray | None = None
_U_LOW: np.ndarray | None = None
_WORKER_CFG: dict[str, int | float] | None = None
_THREAD_CONTROLLER = None


def initialize_sketch_worker(
    u_high: np.ndarray,
    u_low: np.ndarray,
    config: dict[str, int | float],
) -> None:
    global _U_HIGH, _U_LOW, _WORKER_CFG, _THREAD_CONTROLLER
    _U_HIGH = u_high
    _U_LOW = u_low
    _WORKER_CFG = config
    _THREAD_CONTROLLER = threadpool_limits(
        limits=int(config["worker_blas_threads"])
    )


def run_one_sketch(task: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if _U_HIGH is None or _U_LOW is None or _WORKER_CFG is None:
        raise RuntimeError("Sketch worker was not initialized.")
    seed, l_high, l_low, blas_threads = task
    cfg = _WORKER_CFG
    # When B is smaller than the CPU budget, give each active sketch more
    # BLAS threads.  When B is large, use one thread per worker.  This uses the
    # available cores without oversubscribing them or handicapping B=1.
    with threadpool_limits(limits=blas_threads):
        high_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
        low_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
        high_ids = high_rng.integers(0, _U_HIGH.shape[0], size=l_high)
        low_ids = low_rng.integers(0, _U_LOW.shape[0], size=l_low)
        high_raw = spectrum_from_resampled_indices(
            _U_HIGH, high_ids, correlation=True
        )
        high_adjusted = adjusted_eigenvalues_high(
            high_raw,
            n=int(cfg["n"]),
            q=int(cfg["q"]),
            kmax=int(cfg["kmax_high"]),
        )
        low_values = spectrum_from_observations(_U_LOW[low_ids], correlation=True)
    return high_adjusted, low_values[: int(cfg["kmax_low"])]


def process_tree_memory_mb() -> tuple[float, str]:
    """Return summed PSS when available, otherwise summed RSS."""
    root = psutil.Process(os.getpid())
    processes = [root] + root.children(recursive=True)
    total = 0.0
    metric = "process_tree_pss_mb"
    for process in processes:
        try:
            info = process.memory_full_info()
            value = getattr(info, "pss", None)
            if value is None:
                metric = "process_tree_rss_mb"
                value = process.memory_info().rss
            total += float(value)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / 1024.0**2, metric


class MemoryMonitor:
    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.baseline_mb = float("nan")
        self.peak_mb = float("nan")
        self.metric = "unavailable"
        self.thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while not self.stop_event.is_set():
            value, metric = process_tree_memory_mb()
            self.metric = metric
            self.peak_mb = max(self.peak_mb, value) if np.isfinite(self.peak_mb) else value
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "MemoryMonitor":
        self.baseline_mb, self.metric = process_tree_memory_mb()
        self.peak_mb = self.baseline_mb
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        value, metric = process_tree_memory_mb()
        self.metric = metric
        self.peak_mb = max(self.peak_mb, value)


def timed_call(function):
    gc.collect()
    with MemoryMonitor() as memory:
        start = time.perf_counter()
        result = function()
        seconds = time.perf_counter() - start
    return result, seconds, memory


def summarize_benchmark(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "method",
        "p",
        "gamma_high",
        "gamma_low",
        "L1_high",
        "L2_low",
        "B",
        "cpu_budget",
        "worker_blas_threads",
    ]
    rows: list[dict[str, float | int | str]] = []
    for keys, part in raw.groupby(group_columns, dropna=False, sort=True):
        row: dict[str, float | int | str] = dict(zip(group_columns, keys))
        row["repetitions"] = len(part)
        row["time_median_seconds"] = float(part["time_seconds"].median())
        row["time_q25_seconds"] = float(part["time_seconds"].quantile(0.25))
        row["time_q75_seconds"] = float(part["time_seconds"].quantile(0.75))
        row["peak_memory_median_mb"] = float(part["peak_memory_mb"].median())
        row["peak_memory_max_mb"] = float(part["peak_memory_mb"].max())
        row["incremental_memory_median_mb"] = float(
            part["incremental_memory_mb"].median()
        )
        row["row_rank_median"] = float(part["row_rank"].median())
        row["column_rank_median"] = float(part["column_rank"].median())
        row["row_mae_vs_full"] = float(part["row_abs_error"].mean())
        row["column_mae_vs_full"] = float(part["column_abs_error"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    dense_full_time = float(
        summary.loc[
            summary["method"] == "Full-ACT-dense-direct", "time_median_seconds"
        ].iloc[0]
    )
    summary["speedup_vs_dense_full"] = (
        dense_full_time / summary["time_median_seconds"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--p", required=True, type=int)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--kmax", type=int, default=200)
    parser.add_argument("--gamma-grid", default="0.10,0.25,0.50,0.75,1.00")
    parser.add_argument("--gamma-low", type=float, default=0.10)
    parser.add_argument("--B-grid", default="1,5,20,50")
    parser.add_argument("--benchmark-reps", type=int, default=10)
    parser.add_argument("--cpu-budget", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    gamma_grid = parse_float_list(args.gamma_grid)
    b_grid = parse_int_list(args.B_grid)
    if args.benchmark_reps < 3 or args.cpu_budget < 1:
        raise ValueError("Use at least 3 benchmark repetitions and a positive CPU budget.")
    if any(g <= 0.0 or g > 1.0 for g in gamma_grid):
        raise ValueError("Every gamma must be in (0,1].")

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
    u_high = high_mode_observations(z)
    u_low = z.reshape(n * p, q)
    kmax_high = min(args.kmax, p - 1)
    kmax_low = min(args.kmax, q)
    threshold_high = tau_high(n, p, q)
    threshold_low, _ = tau_low(n)
    l_high_values = [max(2, int(round(gamma * n * q))) for gamma in gamma_grid]
    l_low = max(2, int(round(args.gamma_low * n * p)))

    def full_estimator() -> tuple[int, int]:
        high_raw = spectrum_from_observations(u_high, correlation=True)
        high_adjusted = adjusted_eigenvalues_high(
            high_raw, n=n, q=q, kmax=kmax_high
        )
        low_values = spectrum_from_observations(u_low, correlation=True)
        return (
            max_threshold_rank(high_adjusted, threshold_high, kmax_high),
            max_threshold_rank(low_values, threshold_low, kmax_low),
        )

    def dense_full_estimator() -> tuple[int, int]:
        """Straightforward Full-ACT using explicit mode correlation matrices."""
        high_cov = covariance_from_observations(u_high)
        high_corr = correlation_from_cov(high_cov)
        high_raw = sorted_eigvals(high_corr)
        high_adjusted = adjusted_eigenvalues_high(
            high_raw, n=n, q=q, kmax=kmax_high
        )
        low_cov = covariance_from_observations(u_low)
        low_corr = correlation_from_cov(low_cov)
        low_values = sorted_eigvals(low_corr)
        return (
            max_threshold_rank(high_adjusted, threshold_high, kmax_high),
            max_threshold_rank(low_values, threshold_low, kmax_low),
        )

    print(
        f"Fair CPU budget: {args.cpu_budget} cores. "
        "Full-ACT receives that many BLAS threads; S2ACT divides the same "
        "budget across concurrently active sketches.",
        flush=True,
    )
    with threadpool_limits(limits=args.cpu_budget):
        full_reference = full_estimator()  # warm-up, not recorded
    print(f"Full-ACT reference rank: {full_reference}", flush=True)

    raw_rows: list[dict[str, float | int | str]] = []

    worker_config: dict[str, int | float] = {
        "n": n,
        "q": q,
        "kmax_high": kmax_high,
        "kmax_low": kmax_low,
        "worker_blas_threads": 1,
    }
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    context = mp.get_context(method)
    pool_start = time.perf_counter()
    pool = context.Pool(
        processes=args.cpu_budget,
        initializer=initialize_sketch_worker,
        initargs=(u_high, u_low, worker_config),
    )
    pool_startup_seconds = time.perf_counter() - pool_start
    try:
        # Warm up workers with the smallest sketch.  Pool creation and warm-up
        # are reported separately and excluded from repeated estimator timing.
        warm_tasks = [
            (args.seed + worker, min(l_high_values), l_low, 1)
            for worker in range(args.cpu_budget)
        ]
        pool.map(run_one_sketch, warm_tasks, chunksize=1)

        for gamma_index, (gamma, l_high) in enumerate(
            zip(gamma_grid, l_high_values)
        ):
            for b_value in b_grid:
                for repetition in range(args.benchmark_reps):
                    if b_value <= args.cpu_budget:
                        base_threads = args.cpu_budget // b_value
                        extra_threads = args.cpu_budget % b_value
                        thread_allocations = [
                            base_threads + int(b < extra_threads)
                            for b in range(b_value)
                        ]
                    else:
                        thread_allocations = [1] * b_value
                    tasks = [
                        (
                            int(
                                np.random.SeedSequence(
                                    [
                                        args.seed,
                                        p,
                                        gamma_index,
                                        b_value,
                                        repetition,
                                        b,
                                    ]
                                ).generate_state(1)[0]
                            ),
                            l_high,
                            l_low,
                            thread_allocations[b],
                        )
                        for b in range(b_value)
                    ]

                    def sketch_estimator() -> tuple[int, int]:
                        spectra = pool.map(run_one_sketch, tasks, chunksize=1)
                        high_median = np.nanmedian(
                            np.stack([item[0] for item in spectra]), axis=0
                        )
                        low_median = np.nanmedian(
                            np.stack([item[1] for item in spectra]), axis=0
                        )
                        return (
                            prefix_rank(high_median, threshold_high, kmax_high),
                            prefix_rank(low_median, threshold_low, kmax_low),
                        )

                    ranks, seconds, memory = timed_call(sketch_estimator)
                    raw_rows.append(
                        {
                            "method": "S2ACT-optimized-parallel",
                            "p": p,
                            "gamma_high": gamma,
                            "gamma_low": args.gamma_low,
                            "L1_high": l_high,
                            "L2_low": l_low,
                            "B": b_value,
                            "rep": repetition,
                            "cpu_budget": args.cpu_budget,
                            "worker_blas_threads": max(thread_allocations),
                            "max_blas_threads_per_active_sketch": max(thread_allocations),
                            "time_seconds": seconds,
                            "memory_metric": memory.metric,
                            "baseline_memory_mb": memory.baseline_mb,
                            "peak_memory_mb": memory.peak_mb,
                            "incremental_memory_mb": max(
                                0.0, memory.peak_mb - memory.baseline_mb
                            ),
                            "row_rank": ranks[0],
                            "column_rank": ranks[1],
                            "full_row_rank": full_reference[0],
                            "full_column_rank": full_reference[1],
                            "row_abs_error": abs(ranks[0] - full_reference[0]),
                            "column_abs_error": abs(ranks[1] - full_reference[1]),
                        }
                    )
                    print(
                        f"completed S2ACT gamma={gamma:g}, B={b_value}, "
                        f"benchmark {repetition + 1}/{args.benchmark_reps}",
                        flush=True,
                    )
    finally:
        pool.close()
        pool.join()

    # Time the dense baseline after closing the S2ACT pool so that dense
    # p-by-p work arrays cannot inflate the S2ACT process-tree memory baseline.
    with threadpool_limits(limits=args.cpu_budget):
        dense_reference = dense_full_estimator()  # warm-up, not recorded
    if dense_reference != full_reference:
        raise RuntimeError(
            f"Dense Full-ACT and the reference calculation disagree: "
            f"{dense_reference} vs {full_reference}"
        )
    for repetition in range(args.benchmark_reps):
        with threadpool_limits(limits=args.cpu_budget):
            ranks, seconds, memory = timed_call(dense_full_estimator)
        raw_rows.append(
            {
                "method": "Full-ACT-dense-direct",
                "p": p,
                "gamma_high": np.nan,
                "gamma_low": np.nan,
                "L1_high": np.nan,
                "L2_low": np.nan,
                "B": np.nan,
                "rep": repetition,
                "cpu_budget": args.cpu_budget,
                "worker_blas_threads": np.nan,
                "time_seconds": seconds,
                "memory_metric": memory.metric,
                "baseline_memory_mb": memory.baseline_mb,
                "peak_memory_mb": memory.peak_mb,
                "incremental_memory_mb": max(
                    0.0, memory.peak_mb - memory.baseline_mb
                ),
                "row_rank": ranks[0],
                "column_rank": ranks[1],
                "full_row_rank": full_reference[0],
                "full_column_rank": full_reference[1],
                "row_abs_error": abs(ranks[0] - full_reference[0]),
                "column_abs_error": abs(ranks[1] - full_reference[1]),
            }
        )
        print(
            f"completed dense Full-ACT benchmark "
            f"{repetition + 1}/{args.benchmark_reps}",
            flush=True,
        )

    raw = pd.DataFrame(raw_rows)
    summary = summarize_benchmark(raw)
    raw_file = results / "runtime_memory_benchmark_raw.csv"
    summary_file = results / "runtime_memory_benchmark_summary.csv"
    raw.to_csv(raw_file, index=False)
    summary.to_csv(summary_file, index=False)

    accounting_rows = [
        {
            "method": "direct dense Full-ACT",
            "largest_spectral_matrix_dimension": p,
            "spectral_matrix_storage_mb": 8.0 * p * p / 1024.0**2,
            "dominant_spectral_cost": "O(nq p^2 + p^3)",
        },
    ]
    for gamma, l_high in zip(gamma_grid, l_high_values):
        accounting_rows.append(
            {
                "method": f"S2ACT one high sketch gamma={gamma:g}",
                "largest_spectral_matrix_dimension": l_high,
                "spectral_matrix_storage_mb": 8.0 * l_high * l_high / 1024.0**2,
                "dominant_spectral_cost": "O(L1^2 p + L1^3)",
            }
        )
    pd.DataFrame(accounting_rows).to_csv(
        results / "runtime_memory_complexity_accounting.csv", index=False
    )

    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    config = {
        "data_file": str(data_file),
        "shape": [n, p, q],
        "benchmark_repetitions": args.benchmark_reps,
        "cpu_budget": args.cpu_budget,
        "full_blas_threads": args.cpu_budget,
        "s2act_workers": args.cpu_budget,
        "s2act_thread_allocation": (
            "the CPU budget is divided across active sketches when B<=budget; "
            "for B>budget, budget one-thread sketches run concurrently"
        ),
        "pool_startup_seconds_excluded": pool_startup_seconds,
        "data_loading_centering_and_plotting_excluded_from_timing": True,
        "full_method": (
            "direct construction and eigendecomposition of the p by p "
            "correlation matrix; same CPU budget and vectorized numerical kernels"
        ),
        "memory_measurement": (
            "20-ms process-tree sampling; summed PSS on Linux when available, "
            "otherwise summed RSS; input arrays are included in baseline memory"
        ),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "threadpool_info": threadpool_info(),
        "seed": args.seed,
    }
    (results / "runtime_memory_benchmark_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Saved raw benchmark: {raw_file}")
    print(f"Saved summary:       {summary_file}")


if __name__ == "__main__":
    main()
