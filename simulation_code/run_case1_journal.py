#!/usr/bin/env python3
"""Run the complete Case-1 experiment and create journal-ready outputs.

The script produces three pieces of evidence from one theory-aligned design:

1. Table 2: full-sample Cov-ER, Corr-ER, and mode-adaptive rank recovery
   over n in {200, 500, 1000} and h in {0, 1, 2}.
2. Figure 1: a 2-by-2 boundary-spectrum plot for both matrix modes at h=2.
3. Table 3: S2ACT rank recovery over h, sketch fraction gamma, and B.

All Monte Carlo data sets and all sketch banks are independently regenerated.
The high-dimensional row mode uses ACT-adjusted correlation eigenvalues; the
fixed-q column mode uses raw correlation eigenvalues.  Both sketched modes use
coordinatewise median aggregation followed by the consecutive-prefix rule.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

# Prevent nested BLAS parallelism when Monte Carlo replications are parallel.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

import simulation_code.case1_core as core


FULL_METHODS = ("Cov-ER", "Corr-ER", "MA-RE")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def stable_condition_seed(base_seed: int, n: int, h: float) -> int:
    """Create a reproducible seed shared by matching full/sketch conditions."""
    state = np.random.SeedSequence(
        [int(base_seed), int(n), int(round(1000.0 * h)), 20260918]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def mcse_mean(values: pd.Series) -> float:
    array = values.to_numpy(float)
    if len(array) <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / np.sqrt(len(array)))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def run_full_replication(task: tuple[int, float, int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the three paired full-sample estimators on one generated data set."""
    n, h, rep, params = task
    p = int(params["p"])
    q = int(params["q"])
    k = int(params["k"])
    kmax = int(params["kmax"])
    condition_seed = stable_condition_seed(int(params["seed"]), n, h)
    data_seed = condition_seed + 10_000_000 * rep + 11

    generated = core.generate_dataset(
        n=n,
        p=p,
        q=q,
        k=k,
        h=h,
        omega=float(params["omega"]),
        sigma=float(params["sigma"]),
        high_target=float(params["high_target"]),
        seed=data_seed,
    )
    center_error = generated.center_error
    m1, m2, h1, h2 = core.full_operators(generated.Z)

    tau_high = core.tau_high(n=n, p=p, q=q)
    tau_low, delta_low = core.tau_low(n=n, delta_low=None)
    high_values = core.sorted_eigvals(h1)
    high_adjusted = core.adjusted_eigenvalues_high(
        vals=high_values,
        n=n,
        q=q,
        kmax=kmax,
    )
    low_values = core.sorted_eigvals(h2)[: min(kmax, q)]

    ranks = {
        "Cov-ER": (
            core.eigen_ratio_rank(m1, kmax),
            core.eigen_ratio_rank(m2, kmax),
        ),
        "Corr-ER": (
            core.eigen_ratio_rank(h1, kmax),
            core.eigen_ratio_rank(h2, kmax),
        ),
        "MA-RE": (
            core.max_threshold_rank(high_adjusted, tau_high, kmax),
            core.max_threshold_rank(low_values, tau_low, kmax),
        ),
    }

    rows: list[dict[str, Any]] = []
    for method, (row_rank, column_rank) in ranks.items():
        rows.append(
            {
                "n": n,
                "p": p,
                "q": q,
                "k_true": k,
                "h": h,
                "variance_ratio": 10.0**h,
                "rep": rep,
                "method": method,
                "row_rank": int(row_rank),
                "row_correct": int(row_rank == k),
                "column_rank": int(column_rank),
                "column_correct": int(column_rank == k),
                "pair_correct": int(row_rank == k and column_rank == k),
                "tau_high": tau_high,
                "delta_low": delta_low,
                "tau_low": tau_low,
                "theta": generated.theta,
                "population_high_k": generated.population_high_k,
                "population_low_k": generated.population_low_k,
                "dp_ratio": generated.dp_ratio,
                "dp_sum": generated.dp_sum,
                "dp_identity_error": generated.dp_identity_error,
                "center_error": center_error,
            }
        )
    return rows


def make_sketch_config(args: argparse.Namespace, h: float) -> dict[str, Any]:
    tau_high = core.tau_high(n=args.sketch_n, p=args.p, q=args.q)
    tau_low, delta_low = core.tau_low(n=args.sketch_n, delta_low=None)
    return {
        "n": args.sketch_n,
        "p": args.p,
        "q": args.q,
        "k": args.k,
        "kmax": args.kmax,
        "h": h,
        "omega": args.omega,
        "sigma": args.sigma,
        "high_target": args.high_target,
        "gamma_grid": args.gamma_grid,
        "B_grid": args.B_grid,
        "reps": args.reps,
        "seed": stable_condition_seed(args.seed, args.sketch_n, h),
        "n_jobs": 1,
        "pool_high": args.sketch_n * args.q,
        "pool_low": args.sketch_n * args.p,
        "L1_values": [
            max(1, int(round(gamma * args.sketch_n * args.q)))
            for gamma in args.gamma_grid
        ],
        "L2_values": [
            max(1, int(round(gamma * args.sketch_n * args.p)))
            for gamma in args.gamma_grid
        ],
        "tau_high": tau_high,
        "delta_low_used": delta_low,
        "tau_low": tau_low,
        "delta_low_condition_index": (
            np.sqrt(args.sketch_n) / float(args.q)
        )
        * delta_low,
    }


def run_sketch_replication(
    task: tuple[float, int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    h, rep, config = task
    randomized, full_rows, diagnostics = core.run_one_replication(rep, config)
    for row in full_rows:
        row["n"] = int(config["n"])
        row["p"] = int(config["p"])
        row["q"] = int(config["q"])
        row["h"] = h
        if row["method"] == "Full-Mode-Adaptive":
            row["method"] = "MA-RE"
    return randomized, full_rows, diagnostics


def completed_full_keys(raw: pd.DataFrame) -> set[tuple[int, float, int]]:
    if raw.empty:
        return set()
    counts = raw.groupby(["n", "h", "rep"])["method"].nunique()
    return {
        (int(n), float(h), int(rep))
        for (n, h, rep), count in counts.items()
        if int(count) == len(FULL_METHODS)
    }


def completed_sketch_keys(
    raw: pd.DataFrame, expected_rows: int
) -> set[tuple[float, int]]:
    if raw.empty:
        return set()
    counts = raw.groupby(["h", "rep"]).size()
    return {
        (float(h), int(rep))
        for (h, rep), count in counts.items()
        if int(count) == expected_rows
    }


def dispatch(
    function: Callable[[Any], Any],
    tasks: list[Any],
    n_jobs: int,
) -> Iterable[tuple[Any, Any]]:
    """Yield (task, result), serially or through a process pool."""
    if n_jobs <= 1:
        for task in tasks:
            yield task, function(task)
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_task = {executor.submit(function, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            yield task, future.result()


def run_full_grid(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    raw_path = out_dir / "case1_full_sample_raw.csv"
    existing = (
        pd.read_csv(raw_path)
        if args.resume and raw_path.exists()
        else pd.DataFrame()
    )
    completed = completed_full_keys(existing)
    params = {
        "p": args.p,
        "q": args.q,
        "k": args.k,
        "kmax": args.kmax,
        "omega": args.omega,
        "sigma": args.sigma,
        "high_target": args.high_target,
        "seed": args.seed,
    }
    tasks = [
        (n, h, rep, params)
        for n in args.n_grid
        for h in args.h_grid
        for rep in range(args.reps)
        if (n, h, rep) not in completed
    ]
    rows = existing.to_dict("records") if not existing.empty else []
    print(
        f"Full-sample grid: {len(tasks)} remaining replications "
        f"({len(completed)} already complete).",
        flush=True,
    )
    for done, (task, result) in enumerate(
        dispatch(run_full_replication, tasks, args.n_jobs), start=1
    ):
        rows.extend(result)
        if done % args.checkpoint_every == 0 or done == len(tasks):
            frame = pd.DataFrame(rows).drop_duplicates(
                ["n", "h", "rep", "method"], keep="last"
            )
            frame = frame.sort_values(["n", "h", "rep", "method"])
            atomic_csv(frame, raw_path)
            print(
                f"  full checkpoint {done}/{len(tasks)}: "
                f"n={task[0]}, h={task[1]:g}, rep={task[2]}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("No full-sample results are available.")
    return pd.DataFrame(rows).drop_duplicates(
        ["n", "h", "rep", "method"], keep="last"
    )


def run_sketch_grid(
    args: argparse.Namespace, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path = out_dir / "case1_s2act_raw.csv"
    full_path = out_dir / "case1_s2act_paired_full_raw.csv"
    diagnostic_path = out_dir / "case1_s2act_diagnostics.csv"
    raw_existing = (
        pd.read_csv(raw_path)
        if args.resume and raw_path.exists()
        else pd.DataFrame()
    )
    full_existing = (
        pd.read_csv(full_path)
        if args.resume and full_path.exists()
        else pd.DataFrame()
    )
    diagnostic_existing = (
        pd.read_csv(diagnostic_path)
        if args.resume and diagnostic_path.exists()
        else pd.DataFrame()
    )
    expected_rows = len(args.gamma_grid) * len(args.B_grid)
    completed = completed_sketch_keys(raw_existing, expected_rows)
    configs = {h: make_sketch_config(args, h) for h in args.h_grid}
    tasks = [
        (h, rep, configs[h])
        for h in args.h_grid
        for rep in range(args.reps)
        if (h, rep) not in completed
    ]
    random_rows = raw_existing.to_dict("records") if not raw_existing.empty else []
    full_rows = full_existing.to_dict("records") if not full_existing.empty else []
    diagnostics = (
        diagnostic_existing.to_dict("records")
        if not diagnostic_existing.empty
        else []
    )
    print(
        f"S2ACT grid: {len(tasks)} remaining replications "
        f"({len(completed)} already complete).",
        flush=True,
    )
    for done, (task, result) in enumerate(
        dispatch(run_sketch_replication, tasks, args.n_jobs), start=1
    ):
        randomized, paired_full, diagnostic = result
        random_rows.extend(randomized)
        full_rows.extend(paired_full)
        diagnostics.append(diagnostic)
        if done % args.checkpoint_every == 0 or done == len(tasks):
            random_frame = pd.DataFrame(random_rows).drop_duplicates(
                ["h", "rep", "gamma", "B"], keep="last"
            )
            full_frame = pd.DataFrame(full_rows).drop_duplicates(
                ["h", "rep", "method"], keep="last"
            )
            diagnostic_frame = pd.DataFrame(diagnostics).drop_duplicates(
                ["h", "rep"], keep="last"
            )
            atomic_csv(
                random_frame.sort_values(["h", "rep", "gamma", "B"]),
                raw_path,
            )
            atomic_csv(
                full_frame.sort_values(["h", "rep", "method"]), full_path
            )
            atomic_csv(
                diagnostic_frame.sort_values(["h", "rep"]), diagnostic_path
            )
            print(
                f"  sketch checkpoint {done}/{len(tasks)}: "
                f"h={task[0]:g}, rep={task[1]}",
                flush=True,
            )
    if not random_rows:
        raise RuntimeError("No S2ACT results are available.")
    return (
        pd.DataFrame(random_rows).drop_duplicates(
            ["h", "rep", "gamma", "B"], keep="last"
        ),
        pd.DataFrame(full_rows).drop_duplicates(
            ["h", "rep", "method"], keep="last"
        ),
        pd.DataFrame(diagnostics).drop_duplicates(["h", "rep"], keep="last"),
    )


def summarize_table2(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["row_over"] = (raw["row_rank"] > raw["k_true"]).astype(int)
    raw["row_under"] = (raw["row_rank"] < raw["k_true"]).astype(int)
    raw["column_over"] = (raw["column_rank"] > raw["k_true"]).astype(int)
    raw["column_under"] = (raw["column_rank"] < raw["k_true"]).astype(int)
    pieces = []
    for mode, rank_column, correct_column, over_column, under_column in (
        ("row", "row_rank", "row_correct", "row_over", "row_under"),
        (
            "column",
            "column_rank",
            "column_correct",
            "column_over",
            "column_under",
        ),
    ):
        summary = (
            raw.groupby(["n", "h", "method"], as_index=False)
            .agg(
                replications=("rep", "nunique"),
                average_rank=(rank_column, "mean"),
                rank_sd=(rank_column, "std"),
                accuracy=(correct_column, "mean"),
                over_rate=(over_column, "mean"),
                under_rate=(under_column, "mean"),
            )
        )
        rank_mcse = (
            raw.groupby(["n", "h", "method"])[rank_column]
            .apply(mcse_mean)
            .rename("rank_mcse")
            .reset_index()
        )
        summary = summary.merge(
            rank_mcse, on=["n", "h", "method"], validate="one_to_one"
        )
        summary["accuracy_mcse"] = np.sqrt(
            summary["accuracy"]
            * (1.0 - summary["accuracy"])
            / summary["replications"]
        )
        intervals = [
            wilson_interval(int(round(row.accuracy * row.replications)), int(row.replications))
            for row in summary.itertuples()
        ]
        summary["accuracy_ci_low"] = [interval[0] for interval in intervals]
        summary["accuracy_ci_high"] = [interval[1] for interval in intervals]
        summary.insert(0, "mode", mode)
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["mode", "n", "h", "method"]
    )


def format_table2(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["rank_cell"] = frame.apply(
        lambda row: f"{row['average_rank']:.2f} ({row['rank_mcse']:.2f})",
        axis=1,
    )
    frame["accuracy_cell"] = frame["accuracy"].map(lambda value: f"{value:.2f}")
    rows = []
    for (mode, n, h), part in frame.groupby(["mode", "n", "h"], sort=True):
        record: dict[str, Any] = {"mode": mode, "n": int(n), "h": float(h)}
        indexed = part.set_index("method")
        for method in FULL_METHODS:
            slug = method.lower().replace("-", "_")
            record[f"{slug}_rank"] = indexed.loc[method, "rank_cell"]
            record[f"{slug}_accuracy"] = indexed.loc[method, "accuracy_cell"]
        rows.append(record)
    return pd.DataFrame(rows)


def table2_latex(formatted: pd.DataFrame, replications: int) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Rank recovery of the full-sample estimators across "
            r"sample sizes and scale-heterogeneity levels. Values in "
            r"parentheses are Monte Carlo standard errors of the average "
            rf"estimated rank; Acc. is based on {replications} replications.}}"
        ),
        r"\label{tab:case1_full}",
        r"\begin{tabular}{rr|cc|cc|cc}",
        r"\toprule",
        r"& & \multicolumn{2}{c|}{Cov-ER} & \multicolumn{2}{c|}{Corr-ER} & \multicolumn{2}{c}{MA-RE} \\",
        r"$n$ & $h$ & $\overline{\widehat{k}}$ & Acc. & $\overline{\widehat{k}}$ & Acc. & $\overline{\widehat{k}}$ & Acc. \\",
        r"\midrule",
    ]
    for mode, title in (("row", "Row-side rank recovery"), ("column", "Column-side rank recovery")):
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\")
        part = formatted[formatted["mode"] == mode].sort_values(["n", "h"])
        last_n: int | None = None
        for _, row in part.iterrows():
            n_value = int(row["n"])
            n_text = str(n_value) if n_value != last_n else ""
            last_n = n_value
            lines.append(
                f"{n_text} & {row['h']:g} & {row['cov_er_rank']} & "
                f"{row['cov_er_accuracy']} & {row['corr_er_rank']} & "
                f"{row['corr_er_accuracy']} & {row['ma_re_rank']} & "
                f"{row['ma_re_accuracy']} \\\\"
            )
        if mode == "row":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def summarize_table3(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["h", "gamma", "L1_high", "L2_low", "B"], as_index=False)
    summary = grouped.agg(
        replications=("rep", "nunique"),
        row_average_rank=("row_rank", "mean"),
        row_rank_sd=("row_rank", "std"),
        row_accuracy=("row_correct", "mean"),
        row_over_rate=("row_over", "mean"),
        row_under_rate=("row_under", "mean"),
        column_average_rank=("col_rank", "mean"),
        column_rank_sd=("col_rank", "std"),
        column_accuracy=("col_correct", "mean"),
        column_over_rate=("col_over", "mean"),
        column_under_rate=("col_under", "mean"),
        pair_accuracy=("pair_correct", "mean"),
        row_signal_mean=("row_A_k_adj", "mean"),
        row_noise_mean=("row_A_next_adj", "mean"),
        column_signal_mean=("col_A_k_raw", "mean"),
        column_noise_mean=("col_A_next_raw", "mean"),
    )
    summary["row_rank_mcse"] = summary["row_rank_sd"].fillna(0.0) / np.sqrt(
        summary["replications"]
    )
    summary["column_rank_mcse"] = summary["column_rank_sd"].fillna(0.0) / np.sqrt(
        summary["replications"]
    )
    for column in ("row_accuracy", "column_accuracy", "pair_accuracy"):
        summary[f"{column}_mcse"] = np.sqrt(
            summary[column] * (1.0 - summary[column]) / summary["replications"]
        )
        intervals = [
            wilson_interval(int(round(row_value * count)), int(count))
            for row_value, count in zip(summary[column], summary["replications"])
        ]
        summary[f"{column}_ci_low"] = [interval[0] for interval in intervals]
        summary[f"{column}_ci_high"] = [interval[1] for interval in intervals]
    return summary.sort_values(["h", "B", "gamma"]).reset_index(drop=True)


def format_table3(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["row_rank_cell"] = frame["row_average_rank"].map(lambda value: f"{value:.2f}")
    frame["row_accuracy_cell"] = frame["row_accuracy"].map(lambda value: f"{value:.2f}")
    frame["column_rank_cell"] = frame["column_average_rank"].map(lambda value: f"{value:.2f}")
    frame["column_accuracy_cell"] = frame["column_accuracy"].map(lambda value: f"{value:.2f}")
    rows = []
    gamma_values = sorted(frame["gamma"].unique())
    for (mode, h, b_value), part in pd.concat(
        [
            frame.assign(mode="row"),
            frame.assign(mode="column"),
        ],
        ignore_index=True,
    ).groupby(["mode", "h", "B"], sort=True):
        indexed = part.set_index("gamma")
        record: dict[str, Any] = {"mode": mode, "h": h, "B": int(b_value)}
        for gamma in gamma_values:
            prefix = f"gamma_{gamma:.2f}"
            record[f"{prefix}_rank"] = indexed.loc[
                gamma, f"{mode}_rank_cell"
            ]
            record[f"{prefix}_accuracy"] = indexed.loc[
                gamma, f"{mode}_accuracy_cell"
            ]
        rows.append(record)
    return pd.DataFrame(rows)


def table3_latex(
    formatted: pd.DataFrame,
    gamma_values: list[float],
    replications: int,
) -> str:
    column_spec = "rr|" + "cc" * len(gamma_values)
    header_groups = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{${{\gamma}}={gamma:.2f}$}}"
        for gamma in gamma_values
    )
    subheaders = " & ".join(
        [r"$\overline{\widetilde{k}_s^A}$ & Acc."] * len(gamma_values)
    )
    total_columns = 2 + 2 * len(gamma_values)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Rank recovery of the randomized median-aggregated "
            r"estimator across heterogeneity levels, sketch fractions, and "
            rf"numbers of repeated sketches, based on {replications} replications.}}"
        ),
        r"\label{tab:case1_s2act}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        rf"$h$ & $B$ & {header_groups} \\",
        rf"& & {subheaders} \\",
        r"\midrule",
    ]
    for mode, title in (("row", "Panel A: Row-side rank recovery"), ("column", "Panel B: Column-side rank recovery")):
        lines.append(
            rf"\multicolumn{{{total_columns}}}{{l}}{{\textit{{{title}}}}} \\")
        part = formatted[formatted["mode"] == mode].sort_values(["h", "B"])
        last_h: float | None = None
        for _, row in part.iterrows():
            h_value = float(row["h"])
            h_text = f"{h_value:g}" if h_value != last_h else ""
            last_h = h_value
            cells = []
            for gamma in gamma_values:
                prefix = f"gamma_{gamma:.2f}"
                cells.extend(
                    [row[f"{prefix}_rank"], row[f"{prefix}_accuracy"]]
                )
            lines.append(
                f"{h_text} & {int(row['B'])} & " + " & ".join(cells) + r" \\"
            )
        if mode == "row":
            lines.append(r"\midrule")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    )
    return "\n".join(lines)


def minimum_gamma_table(summary: pd.DataFrame) -> pd.DataFrame:
    """First observed gamma reaching each target accuracy, without smoothing."""
    rows: list[dict[str, Any]] = []
    for mode, accuracy_column in (
        ("row", "row_accuracy"),
        ("column", "column_accuracy"),
    ):
        for (h, b_value), part in summary.groupby(["h", "B"], sort=True):
            part = part.sort_values("gamma")
            for target in (0.90, 0.95):
                eligible = part[part[accuracy_column] >= target]
                rows.append(
                    {
                        "mode": mode,
                        "h": h,
                        "B": int(b_value),
                        "target_accuracy": target,
                        "minimum_gamma": (
                            float(eligible["gamma"].iloc[0])
                            if not eligible.empty
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def representative_comparison(
    sketch_raw: pd.DataFrame,
    paired_full: pd.DataFrame,
    figure_h: float,
) -> pd.DataFrame:
    """Create the h=2 comparison used in the overall-rank-recovery paragraph."""
    full = paired_full[
        np.isclose(paired_full["h"].astype(float), figure_h)
    ].copy()
    full_summary = (
        full.groupby("method", as_index=False)
        .agg(
            replications=("rep", "nunique"),
            row_average_rank=("row_rank", "mean"),
            row_accuracy=("row_correct", "mean"),
            column_average_rank=("col_rank", "mean"),
            column_accuracy=("col_correct", "mean"),
            pair_accuracy=("pair_correct", "mean"),
        )
    )
    full_summary.insert(1, "gamma", np.nan)
    full_summary.insert(2, "B", np.nan)

    sketch = sketch_raw[
        np.isclose(sketch_raw["h"].astype(float), figure_h)
        & (
            (
                np.isclose(sketch_raw["gamma"].astype(float), 0.20)
                & (sketch_raw["B"].astype(int) == 15)
            )
            | (
                np.isclose(sketch_raw["gamma"].astype(float), 0.30)
                & (sketch_raw["B"].astype(int) == 1)
            )
        )
    ].copy()
    sketch_summary = (
        sketch.groupby(["gamma", "B"], as_index=False)
        .agg(
            replications=("rep", "nunique"),
            row_average_rank=("row_rank", "mean"),
            row_accuracy=("row_correct", "mean"),
            column_average_rank=("col_rank", "mean"),
            column_accuracy=("col_correct", "mean"),
            pair_accuracy=("pair_correct", "mean"),
        )
    )
    sketch_summary.insert(0, "method", "S2ACT")
    return pd.concat([full_summary, sketch_summary], ignore_index=True)


def representative_latex(frame: pd.DataFrame, figure_h: float) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Overall rank recovery at $h={figure_h:g}$.}}",
        r"\label{tab:case1_representative}",
        r"\begin{tabular}{lcc|cc|cc|c}",
        r"\toprule",
        r"Method & $\gamma$ & $B$ & Row avg. & Row acc. & Column avg. & Column acc. & Pair acc. \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        gamma = "--" if pd.isna(row["gamma"]) else f"{row['gamma']:.2f}"
        b_value = "--" if pd.isna(row["B"]) else str(int(row["B"]))
        lines.append(
            f"{row['method']} & {gamma} & {b_value} & "
            f"{row['row_average_rank']:.2f} & {row['row_accuracy']:.2f} & "
            f"{row['column_average_rank']:.2f} & {row['column_accuracy']:.2f} & "
            f"{row['pair_accuracy']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def make_boundary_figure(
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_stem: Path,
    figure_h: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    selected = raw[np.isclose(raw["h"].astype(float), figure_h)].copy()
    diag = diagnostics[
        np.isclose(diagnostics["h"].astype(float), figure_h)
    ].copy()
    if selected.empty or diag.empty:
        raise ValueError(f"No raw results are available for figure h={figure_h:g}.")

    gamma_values = sorted(selected["gamma"].unique())
    b_values = sorted(selected["B"].astype(int).unique())
    colors = ["#2b6f9e", "#5f91b4", "#93b6ca", "#c7d9e4"]
    if len(b_values) > len(colors):
        colors = [plt.cm.Blues(0.45 + 0.45 * i / max(1, len(b_values) - 1)) for i in range(len(b_values))]
    positions = np.arange(len(gamma_values), dtype=float)
    width = min(0.16, 0.72 / max(1, len(b_values)))
    offsets = (np.arange(len(b_values)) - (len(b_values) - 1) / 2.0) * width

    panels = [
        ("row_A_k_adj", "full_row_adj_k", None, r"(a) Row mode: last signal $A_1(k)$"),
        ("row_A_next_adj", "full_row_adj_next", "tau_high", r"(b) Row mode: first noise $A_1(k+1)$"),
        ("col_A_k_raw", "full_col_raw_k", "tau_low", r"(c) Column mode: last signal $A_2(k)$"),
        ("col_A_next_raw", "full_col_raw_next", "tau_low", r"(d) Column mode: first noise $A_2(k+1)$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
    for axis, (value_column, full_column, threshold_column, title) in zip(
        axes.flat, panels
    ):
        for b_index, (b_value, color) in enumerate(zip(b_values, colors)):
            values = [
                selected[
                    np.isclose(selected["gamma"].astype(float), gamma)
                    & (selected["B"].astype(int) == b_value)
                ][value_column].dropna().to_numpy(float)
                for gamma in gamma_values
            ]
            boxes = axis.boxplot(
                values,
                positions=positions + offsets[b_index],
                widths=width * 0.86,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                medianprops={"color": "#d97706", "linewidth": 1.2},
                boxprops={"facecolor": color, "edgecolor": "#2f2f2f", "linewidth": 0.7},
                whiskerprops={"color": "#4d4d4d", "linewidth": 0.7},
                capprops={"color": "#4d4d4d", "linewidth": 0.7},
            )
            del boxes
        full_mean = float(diag[full_column].mean())
        axis.axhline(
            full_mean,
            color="#377eb8",
            linestyle="--",
            linewidth=1.1,
            label="Full-sample mean",
        )
        if threshold_column is not None:
            threshold = float(diag[threshold_column].iloc[0])
            axis.axhline(
                threshold,
                color="#8c8c8c",
                linestyle=":",
                linewidth=1.2,
                label="Decision threshold",
            )
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.22)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{gamma:.2f}" for gamma in gamma_values])
        axis.margins(x=0.04)

    axes[0, 0].set_ylabel("Median-aggregated eigenvalue")
    axes[1, 0].set_ylabel("Median-aggregated eigenvalue")
    axes[1, 0].set_xlabel(r"Sketch fraction $\gamma=L_s/N_s$")
    axes[1, 1].set_xlabel(r"Sketch fraction $\gamma=L_s/N_s$")
    legend_handles = [
        Patch(facecolor=color, edgecolor="#2f2f2f", label=rf"$B={b_value}$")
        for b_value, color in zip(b_values, colors)
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color="#377eb8", linestyle="--", label="Full-sample mean"),
            plt.Line2D([0], [0], color="#8c8c8c", linestyle=":", label="Decision threshold"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_outputs(args: argparse.Namespace, out_dir: Path) -> None:
    full_raw_path = out_dir / "case1_full_sample_raw.csv"
    sketch_raw_path = out_dir / "case1_s2act_raw.csv"
    diagnostic_path = out_dir / "case1_s2act_diagnostics.csv"
    paired_full_path = out_dir / "case1_s2act_paired_full_raw.csv"

    if full_raw_path.exists():
        full_raw = pd.read_csv(full_raw_path)
        table2 = summarize_table2(full_raw)
        formatted2 = format_table2(table2)
        atomic_csv(table2, out_dir / "table2_full_sample_summary_long.csv")
        atomic_csv(formatted2, out_dir / "table2_full_sample_formatted.csv")
        (out_dir / "table2_full_sample.tex").write_text(
            table2_latex(formatted2, args.reps), encoding="utf-8"
        )
        diagnostic_source = full_raw.drop_duplicates(["n", "h", "rep"])
        simulation_diagnostics = (
            diagnostic_source.groupby(["n", "h"], as_index=False)
            .agg(
                replications=("rep", "nunique"),
                maximum_centering_error=("center_error", "max"),
                mean_theta=("theta", "mean"),
                mean_population_row_k=("population_high_k", "mean"),
                maximum_abs_row_target_error=(
                    "population_high_k",
                    lambda values: float(
                        np.max(np.abs(values.to_numpy(float) - args.high_target))
                    ),
                ),
                mean_population_column_k=("population_low_k", "mean"),
                minimum_population_column_k=("population_low_k", "min"),
                maximum_population_column_k=("population_low_k", "max"),
                minimum_dp_ratio=("dp_ratio", "min"),
                maximum_dp_ratio=("dp_ratio", "max"),
                maximum_trace_normalization_error=(
                    "dp_sum",
                    lambda values: float(
                        np.max(np.abs(values.to_numpy(float) - args.p))
                    ),
                ),
                maximum_dp_identity_deviation=("dp_identity_error", "max"),
                tau_high=("tau_high", "first"),
                tau_low=("tau_low", "first"),
            )
        )
        atomic_csv(
            simulation_diagnostics,
            out_dir / "simulation_diagnostics.csv",
        )

    if sketch_raw_path.exists() and diagnostic_path.exists():
        sketch_raw = pd.read_csv(sketch_raw_path)
        diagnostics = pd.read_csv(diagnostic_path)
        table3 = summarize_table3(sketch_raw)
        display_table3 = table3[
            table3["gamma"].apply(
                lambda value: any(
                    np.isclose(value, target) for target in args.table_gamma_grid
                )
            )
        ].copy()
        formatted3 = format_table3(display_table3)
        atomic_csv(table3, out_dir / "table3_s2act_summary_long.csv")
        atomic_csv(formatted3, out_dir / "table3_s2act_formatted.csv")
        (out_dir / "table3_s2act.tex").write_text(
            table3_latex(
                formatted3,
                sorted(display_table3["gamma"].unique()),
                args.reps,
            ),
            encoding="utf-8",
        )
        atomic_csv(
            minimum_gamma_table(table3),
            out_dir / "minimum_gamma_for_target_accuracy.csv",
        )
        figure_data = sketch_raw[
            np.isclose(sketch_raw["h"].astype(float), args.figure_h)
        ].copy()
        atomic_csv(
            figure_data.sort_values(["rep", "gamma", "B"]),
            out_dir / "figure1_boundary_spectra_data.csv",
        )
        if paired_full_path.exists():
            paired_full = pd.read_csv(paired_full_path)
            representative = representative_comparison(
                sketch_raw, paired_full, args.figure_h
            )
            atomic_csv(
                representative,
                out_dir / "representative_h2_comparison.csv",
            )
            (out_dir / "representative_h2_comparison.tex").write_text(
                representative_latex(representative, args.figure_h),
                encoding="utf-8",
            )
        if not args.no_figure:
            make_boundary_figure(
                sketch_raw,
                diagnostics,
                out_dir / "figure1_case1_boundary_spectrum_2x2",
                args.figure_h,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "full", "sketch", "postprocess"),
        default="all",
    )
    parser.add_argument("--n-grid", type=parse_int_list, default=parse_int_list("200,500,1000"))
    parser.add_argument("--sketch-n", type=int, default=1000)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--h-grid", type=parse_float_list, default=parse_float_list("0,1,2"))
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--high-target", type=float, default=25.0)
    parser.add_argument(
        "--gamma-grid",
        type=parse_float_list,
        default=parse_float_list("0.05,0.10,0.15,0.20,0.25,0.30"),
    )
    parser.add_argument(
        "--table-gamma-grid",
        type=parse_float_list,
        default=parse_float_list("0.05,0.10,0.15,0.25,0.30"),
        help="Gamma values displayed in Table 3; all gamma-grid values remain in raw output.",
    )
    parser.add_argument("--B-grid", type=parse_int_list, default=parse_int_list("1,5,15,35"))
    parser.add_argument("--figure-h", type=float, default=2.0)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("./results_case1_journal"))
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Create CSV/LaTeX outputs without importing matplotlib.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from complete replications already present in the output directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.n_grid = sorted(set(args.n_grid))
    args.h_grid = sorted(set(args.h_grid))
    args.gamma_grid = sorted(set(args.gamma_grid))
    args.table_gamma_grid = sorted(set(args.table_gamma_grid))
    args.B_grid = sorted(set(args.B_grid))
    if args.p % args.k or args.q % args.k:
        raise ValueError("Both p and q must be divisible by k.")
    if args.kmax < args.k + 1:
        raise ValueError("kmax must be at least k+1.")
    if min(args.n_grid) <= 1 or args.sketch_n <= 1:
        raise ValueError("Every sample size must exceed one.")
    if min(args.h_grid) < 0:
        raise ValueError("Heterogeneity levels must be nonnegative.")
    if min(args.gamma_grid) <= 0 or max(args.gamma_grid) > 1:
        raise ValueError("Sketch fractions must lie in (0,1].")
    if not args.table_gamma_grid or any(
        not any(np.isclose(value, gamma) for gamma in args.gamma_grid)
        for value in args.table_gamma_grid
    ):
        raise ValueError("Every table-gamma-grid value must occur in gamma-grid.")
    if min(args.B_grid) < 1 or args.reps < 1 or args.n_jobs < 1:
        raise ValueError("B, reps, and n-jobs must be positive.")
    if args.figure_h not in args.h_grid:
        raise ValueError("figure-h must be included in h-grid.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update(
        {
            "row_mode": "ACT-adjusted correlation eigenvalues",
            "column_mode": "raw correlation eigenvalues",
            "full_rank_rule": "maximum threshold exceedance",
            "sketch_rank_rule": "coordinatewise median plus consecutive prefix",
            "sampling": "uniform with replacement, independent across replications",
        }
    )
    config_path = out_dir / "case1_journal_config.json"
    comparison_keys = {
        "n_grid",
        "sketch_n",
        "p",
        "q",
        "k",
        "kmax",
        "h_grid",
        "omega",
        "sigma",
        "high_target",
        "gamma_grid",
        "B_grid",
        "reps",
        "seed",
    }
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [
            key
            for key in sorted(comparison_keys)
            if previous.get(key) != config.get(key)
        ]
        if changed:
            raise ValueError(
                "The output directory contains checkpoints from a different "
                f"simulation configuration ({', '.join(changed)} changed). "
                "Use a new --out-dir, or use --no-resume only after moving the "
                "old result files out of this directory."
            )
    config_path.write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Case 1 journal experiment", flush=True)
    print(
        f"stage={args.stage}; reps={args.reps}; n_jobs={args.n_jobs}; "
        f"output={out_dir}",
        flush=True,
    )
    start = time.time()
    if args.stage in ("all", "full"):
        run_full_grid(args, out_dir)
    if args.stage in ("all", "sketch"):
        run_sketch_grid(args, out_dir)
    create_outputs(args, out_dir)
    print(f"Completed in {time.time() - start:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
