#!/usr/bin/env python3
"""Accuracy-aware scalability benchmark for Full-ACT and S2ACT.

The synthetic tensors follow a two-sided bilinear matrix factor model with a
fixed number of strong factors.  Along the dimension grid, q is fixed and
p/[q(n-1)] is held approximately constant.  Every timed method receives the
same total CPU budget.  Direct dense Full-ACT is the sole full-data runtime
comparator and is not artificially delayed.
"""

from __future__ import annotations

import argparse
import gc
import json
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
    raise ImportError("Install psutil before running this benchmark.") from exc

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError as exc:
    raise ImportError(
        "Install threadpoolctl before running the fair CPU benchmark."
    ) from exc


_U_HIGH: np.ndarray | None = None
_U_LOW: np.ndarray | None = None
_WORKER_CONFIG: dict[str, int] | None = None
_THREAD_CONTROLLER = None


def initialize_worker(
    u_high: np.ndarray,
    u_low: np.ndarray,
    config: dict[str, int],
) -> None:
    global _U_HIGH, _U_LOW, _WORKER_CONFIG, _THREAD_CONTROLLER
    _U_HIGH = u_high
    _U_LOW = u_low
    _WORKER_CONFIG = config
    _THREAD_CONTROLLER = threadpool_limits(limits=1, user_api="blas")
    _THREAD_CONTROLLER.__enter__()


def run_one_sketch(
    task: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if _U_HIGH is None or _U_LOW is None or _WORKER_CONFIG is None:
        raise RuntimeError("Sketch worker has not been initialized.")
    seed, l_high, l_low, blas_threads = task
    cfg = _WORKER_CONFIG
    rng_high = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    rng_low = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    high_ids = rng_high.integers(0, _U_HIGH.shape[0], size=l_high)
    low_ids = rng_low.integers(0, _U_LOW.shape[0], size=l_low)
    with threadpool_limits(limits=blas_threads):
        high_raw = spectrum_from_resampled_indices(
            _U_HIGH, high_ids, correlation=True
        )
        high_adjusted = adjusted_eigenvalues_high(
            high_raw,
            n=int(cfg["n"]),
            q=int(cfg["q"]),
            kmax=int(cfg["kmax_high"]),
        )
        low_raw = spectrum_from_observations(_U_LOW[low_ids], correlation=True)
    return high_adjusted, low_raw[: int(cfg["kmax_low"])]


def process_tree_memory_mb() -> tuple[float, str]:
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

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            value, metric = process_tree_memory_mb()
            self.metric = metric
            self.peak_mb = max(self.peak_mb, value)
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "MemoryMonitor":
        self.baseline_mb, self.metric = process_tree_memory_mb()
        self.peak_mb = self.baseline_mb
        self.thread = threading.Thread(target=self._sample, daemon=True)
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


def block_loading(dim: int, rank: int) -> np.ndarray:
    if dim % rank != 0:
        raise ValueError(f"Dimension {dim} must be divisible by rank {rank}.")
    loading = np.zeros((dim, rank), dtype=np.float64)
    block = dim // rank
    for component in range(rank):
        loading[component * block : (component + 1) * block, component] = np.sqrt(
            rank
        )
    if not np.allclose(loading.T @ loading, dim * np.eye(rank)):
        raise RuntimeError("Loading normalization failed.")
    return loading


def generate_tensor(
    n: int,
    p: int,
    q: int,
    rank: int,
    factor_scale: float,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
    """Generate Z_i = R F_i C' + E_i with pervasive block loadings."""
    rng = np.random.default_rng(seed)
    row_loading = block_loading(p, rank)
    column_loading = block_loading(q, rank)
    factors = factor_scale * rng.normal(size=(n, rank, rank))
    z = noise_scale * rng.normal(size=(n, p, q))
    z += np.einsum(
        "pr,nrs,qs->npq",
        row_loading,
        factors,
        column_loading,
        optimize=True,
    )
    z -= np.mean(z, axis=0, keepdims=True)
    return np.asarray(z, dtype=np.float64)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "method",
        "n",
        "p",
        "q",
        "aspect_ratio",
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
            (part["peak_memory_mb"] - part["baseline_memory_mb"])
            .clip(lower=0.0)
            .median()
        )
        row["row_accuracy"] = float(part["row_correct"].mean())
        row["column_accuracy"] = float(part["column_correct"].mean())
        row["pair_accuracy"] = float(part["pair_correct"].mean())
        row["row_rank_median"] = float(part["row_rank"].median())
        row["column_rank_median"] = float(part["column_rank"].median())
        row["row_mae"] = float(
            np.abs(part["row_rank"].to_numpy(float) - part["true_rank"]).mean()
        )
        row["column_mae"] = float(
            np.abs(part["column_rank"].to_numpy(float) - part["true_rank"]).mean()
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["speedup_vs_dense_full"] = np.nan
    for p_value in summary["p"].unique():
        mask_p = summary["p"] == p_value
        dense_time = float(
            summary.loc[
                mask_p & (summary["method"] == "Full-ACT-dense-direct"),
                "time_median_seconds",
            ].iloc[0]
        )
        summary.loc[mask_p, "speedup_vs_dense_full"] = (
            dense_time / summary.loc[mask_p, "time_median_seconds"]
        )
    return summary.sort_values(["p", "method", "gamma_high", "B"])


def save_plots(summary: pd.DataFrame, results_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for keys, part in summary.groupby(
        ["method", "gamma_high", "B"], dropna=False, sort=True
    ):
        method, gamma, b_value = keys
        if method.startswith("S2ACT"):
            label = fr"$S^2$ACT $\gamma={gamma:g},B={int(b_value)}$"
        elif method == "Full-ACT-dense-direct":
            label = "Full-ACT dense"
        else:
            raise ValueError(f"Unexpected benchmark method: {method}")
        part = part.sort_values("p")
        ax.plot(
            part["p"],
            part["time_median_seconds"],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("High-mode dimension $p$")
    ax.set_ylabel("Median estimator time (seconds)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "scalability_runtime.pdf", bbox_inches="tight")
    fig.savefig(results_dir / "scalability_runtime.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for keys, part in summary.groupby(
        ["method", "gamma_high", "B"], dropna=False, sort=True
    ):
        method, gamma, b_value = keys
        if method.startswith("S2ACT"):
            label = fr"$S^2$ACT $\gamma={gamma:g},B={int(b_value)}$"
        elif method == "Full-ACT-dense-direct":
            label = "Full-ACT dense"
        else:
            raise ValueError(f"Unexpected benchmark method: {method}")
        part = part.sort_values("p")
        ax.plot(
            part["p"],
            part["peak_memory_median_mb"],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("High-mode dimension $p$")
    ax.set_ylabel("Median peak process-tree memory (MB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "scalability_memory.pdf", bbox_inches="tight")
    fig.savefig(results_dir / "scalability_memory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sketch = summary[summary["method"].str.startswith("S2ACT")]
    for (gamma, b_value), part in sketch.groupby(
        ["gamma_high", "B"], sort=True
    ):
        part = part.sort_values("p")
        ax.plot(
            part["p"],
            part["pair_accuracy"],
            marker="o",
            linewidth=1.8,
            label=fr"$\gamma={gamma:g},B={int(b_value)}$",
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("High-mode dimension $p$")
    ax.set_ylabel("Exact two-mode rank-recovery rate")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "scalability_rank_recovery.pdf", bbox_inches="tight")
    fig.savefig(
        results_dir / "scalability_rank_recovery.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def loglog_slopes(summary: pd.DataFrame) -> pd.DataFrame:
    """Estimate descriptive log-log slopes when at least three scales exist."""
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["method", "gamma_high", "B"]
    for keys, part in summary.groupby(group_columns, dropna=False, sort=True):
        part = part.sort_values("p")
        if len(part) < 3 or part["p"].nunique() < 3:
            continue
        log_p = np.log(part["p"].to_numpy(float))
        log_time = np.log(part["time_median_seconds"].to_numpy(float))
        log_memory = np.log(part["peak_memory_median_mb"].to_numpy(float))
        rows.append(
            {
                "method": keys[0],
                "gamma_high": keys[1],
                "B": keys[2],
                "number_of_scales": len(part),
                "runtime_loglog_slope": float(np.polyfit(log_p, log_time, 1)[0]),
                "peak_memory_loglog_slope": float(
                    np.polyfit(log_p, log_memory, 1)[0]
                ),
                "pair_accuracy_min": float(part["pair_accuracy"].min()),
            }
        )
    return pd.DataFrame(rows)


def add_row(
    rows: list[dict[str, float | int | str]],
    *,
    method: str,
    n: int,
    p: int,
    q: int,
    gamma_high: float,
    gamma_low: float,
    l_high: float,
    l_low: float,
    b_value: float,
    repetition: int,
    cpu_budget: int,
    worker_threads: float,
    ranks: tuple[int, int],
    seconds: float,
    memory: MemoryMonitor,
    true_rank: int,
) -> None:
    rows.append(
        {
            "method": method,
            "n": n,
            "p": p,
            "q": q,
            "aspect_ratio": p / float(q * (n - 1)),
            "gamma_high": gamma_high,
            "gamma_low": gamma_low,
            "L1_high": l_high,
            "L2_low": l_low,
            "B": b_value,
            "rep": repetition,
            "cpu_budget": cpu_budget,
            "worker_blas_threads": worker_threads,
            "time_seconds": seconds,
            "memory_metric": memory.metric,
            "baseline_memory_mb": memory.baseline_mb,
            "peak_memory_mb": memory.peak_mb,
            "row_rank": ranks[0],
            "column_rank": ranks[1],
            "true_rank": true_rank,
            "row_correct": int(ranks[0] == true_rank),
            "column_correct": int(ranks[1] == true_rank),
            "pair_correct": int(ranks == (true_rank, true_rank)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--n-grid", default="125,250,500")
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--factor-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--gamma-grid", default="0.75,1.00")
    parser.add_argument("--gamma-low", type=float, default=0.10)
    parser.add_argument("--B-grid", default="1,5")
    parser.add_argument("--benchmark-reps", type=int, default=5)
    parser.add_argument("--cpu-budget", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    n_grid = parse_int_list(args.n_grid)
    gamma_grid = parse_float_list(args.gamma_grid)
    b_grid = parse_int_list(args.B_grid)
    if min(n_grid) <= 2 or args.q % args.rank != 0:
        raise ValueError("Use n>2 and choose q divisible by rank.")
    if args.kmax < args.rank + 1:
        raise ValueError("kmax must be at least rank+1.")
    if any(g <= 0.0 or g > 1.0 for g in gamma_grid):
        raise ValueError("gamma-grid values must lie in (0,1].")
    if min(b_grid) < 1 or args.benchmark_reps < 3:
        raise ValueError("Use positive B values and at least 3 repetitions.")

    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, float | int | str]] = []
    generated_dimensions: list[dict[str, float | int]] = []

    for scale_index, n in enumerate(n_grid):
        p = int(round(args.rho * args.q * (n - 1) / args.rank)) * args.rank
        p = max(p, 2 * args.rank)
        print(f"Generating scale {scale_index + 1}/{len(n_grid)}: n={n}, p={p}, q={args.q}", flush=True)
        z = generate_tensor(
            n=n,
            p=p,
            q=args.q,
            rank=args.rank,
            factor_scale=args.factor_scale,
            noise_scale=args.noise_scale,
            seed=args.seed + 1000 * scale_index,
        )
        u_high = high_mode_observations(z)
        u_low = z.reshape(n * p, args.q)
        del z
        gc.collect()

        high_threshold = tau_high(n, p, args.q)
        low_threshold, _ = tau_low(n)
        kmax_high = min(args.kmax, p - 1)
        kmax_low = min(args.kmax, args.q)

        def dense_full() -> tuple[int, int]:
            high_corr = correlation_from_cov(covariance_from_observations(u_high))
            high_raw = sorted_eigvals(high_corr)
            high_adjusted = adjusted_eigenvalues_high(
                high_raw, n=n, q=args.q, kmax=kmax_high
            )
            low_corr = correlation_from_cov(covariance_from_observations(u_low))
            low_raw = sorted_eigvals(low_corr)
            return (
                max_threshold_rank(high_adjusted, high_threshold, kmax_high),
                max_threshold_rank(low_raw, low_threshold, kmax_low),
            )

        generated_dimensions.append(
            {
                "n": n,
                "p": p,
                "q": args.q,
                "true_rank": args.rank,
                "aspect_ratio": p / float(args.q * (n - 1)),
                "q_over_sqrt_n": args.q / np.sqrt(n),
                "tau_high": high_threshold,
                "tau_low": low_threshold,
            }
        )

        worker_config = {
            "n": n,
            "q": args.q,
            "kmax_high": kmax_high,
            "kmax_low": kmax_low,
        }
        context = mp.get_context(
            "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        )
        pool_start = time.perf_counter()
        pool = context.Pool(
            processes=args.cpu_budget,
            initializer=initialize_worker,
            initargs=(u_high, u_low, worker_config),
        )
        pool_startup_seconds = time.perf_counter() - pool_start
        generated_dimensions[-1]["pool_startup_seconds_excluded"] = (
            pool_startup_seconds
        )
        l_low = max(2, int(round(args.gamma_low * n * p)))
        try:
            warmup_start = time.perf_counter()
            warm_tasks = [
                (args.seed + worker, max(2, int(round(min(gamma_grid) * n * args.q))), l_low, 1)
                for worker in range(args.cpu_budget)
            ]
            pool.map(run_one_sketch, warm_tasks, chunksize=1)
            generated_dimensions[-1]["warmup_seconds_excluded"] = (
                time.perf_counter() - warmup_start
            )
            for gamma_index, gamma in enumerate(gamma_grid):
                l_high = max(2, int(round(gamma * n * args.q)))
                for b_value in b_grid:
                    for repetition in range(args.benchmark_reps):
                        if b_value <= args.cpu_budget:
                            base = args.cpu_budget // b_value
                            extra = args.cpu_budget % b_value
                            allocations = [
                                base + int(b < extra) for b in range(b_value)
                            ]
                        else:
                            allocations = [1] * b_value
                        tasks = [
                            (
                                int(
                                    np.random.SeedSequence(
                                        [
                                            args.seed,
                                            scale_index,
                                            gamma_index,
                                            b_value,
                                            repetition,
                                            b,
                                        ]
                                    ).generate_state(1)[0]
                                ),
                                l_high,
                                l_low,
                                allocations[b],
                            )
                            for b in range(b_value)
                        ]

                        def sketch_estimator() -> tuple[int, int]:
                            spectra = pool.map(run_one_sketch, tasks, chunksize=1)
                            high_median = np.median(
                                np.stack([item[0] for item in spectra]), axis=0
                            )
                            low_median = np.median(
                                np.stack([item[1] for item in spectra]), axis=0
                            )
                            return (
                                prefix_rank(high_median, high_threshold, kmax_high),
                                prefix_rank(low_median, low_threshold, kmax_low),
                            )

                        ranks, seconds, memory = timed_call(sketch_estimator)
                        add_row(
                            raw_rows,
                            method="S2ACT-optimized-parallel",
                            n=n,
                            p=p,
                            q=args.q,
                            gamma_high=gamma,
                            gamma_low=args.gamma_low,
                            l_high=l_high,
                            l_low=l_low,
                            b_value=b_value,
                            repetition=repetition,
                            cpu_budget=args.cpu_budget,
                            worker_threads=max(allocations),
                            ranks=ranks,
                            seconds=seconds,
                            memory=memory,
                            true_rank=args.rank,
                        )
                    print(
                        f"completed S2ACT n={n}, p={p}, gamma={gamma:g}, B={b_value}",
                        flush=True,
                    )
        finally:
            pool.close()
            pool.join()

        # Dense timing is last so retained p-by-p work buffers cannot inflate
        # the S2ACT process-tree memory baseline.
        with threadpool_limits(limits=args.cpu_budget):
            dense_reference = dense_full()
        if dense_reference != (args.rank, args.rank):
            raise RuntimeError(
                f"Dense Full-ACT did not recover the designed rank at "
                f"n={n}, p={p}: got {dense_reference}. Increase "
                "factor-scale or inspect the DGP."
            )
        for repetition in range(args.benchmark_reps):
            with threadpool_limits(limits=args.cpu_budget):
                ranks, seconds, memory = timed_call(dense_full)
            add_row(
                raw_rows,
                method="Full-ACT-dense-direct",
                n=n,
                p=p,
                q=args.q,
                gamma_high=np.nan,
                gamma_low=np.nan,
                l_high=np.nan,
                l_low=np.nan,
                b_value=np.nan,
                repetition=repetition,
                cpu_budget=args.cpu_budget,
                worker_threads=np.nan,
                ranks=ranks,
                seconds=seconds,
                memory=memory,
                true_rank=args.rank,
            )

        raw = pd.DataFrame(raw_rows)
        raw.to_csv(results_dir / "scalability_benchmark_raw.csv", index=False)
        pd.DataFrame(generated_dimensions).to_csv(
            results_dir / "scalability_dimensions.csv", index=False
        )
        print(
            f"completed scale n={n}, p={p}; pool startup excluded={pool_startup_seconds:.3f}s",
            flush=True,
        )
        del u_high, u_low
        gc.collect()

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    slopes = loglog_slopes(summary)
    raw.to_csv(results_dir / "scalability_benchmark_raw.csv", index=False)
    summary.to_csv(results_dir / "scalability_benchmark_summary.csv", index=False)
    slopes.to_csv(results_dir / "scalability_loglog_slopes.csv", index=False)
    if not args.no_plots:
        save_plots(summary, results_dir)
    config = {
        **vars(args),
        "results_dir": str(results_dir),
        "dimension_rule": "p is the nearest multiple of rank to rho*q*(n-1)",
        "dgp": "Z_i = R F_i C' + E_i with pervasive block loadings",
        "full_methods": ["direct dense"],
        "s2act_duplicate_compression": "exact multiplicity weighting",
        "timing_excludes": [
            "data generation",
            "centering",
            "pool startup",
            "warm-up",
            "plotting",
        ],
        "memory_metric": "process-tree PSS on Linux, otherwise summed RSS",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threadpool_info": threadpool_info(),
        "seed": args.seed,
    }
    (results_dir / "scalability_benchmark_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved: {results_dir / 'scalability_benchmark_summary.csv'}", flush=True)
    if not args.no_plots:
        print(f"Saved: {results_dir / 'scalability_runtime.pdf'}", flush=True)
        print(f"Saved: {results_dir / 'scalability_memory.pdf'}", flush=True)
        print(
            f"Saved: {results_dir / 'scalability_rank_recovery.pdf'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
