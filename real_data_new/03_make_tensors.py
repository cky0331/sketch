#!/usr/bin/env python3
"""Build explicitly aligned raw and donor-centered CEDAR tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CELL_ORDER = (
    "B cell|CD14-positive monocyte|CD15 positive leukocyte|platelet|"
    "T cell|thymocyte"
)


def basename_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("\\", "/", regex=False).str.rsplit("/").str[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--p-list", default="3000,5000")
    parser.add_argument("--cell-order", default=DEFAULT_CELL_ORDER)
    args = parser.parse_args()

    processed = args.processed_dir.expanduser().resolve()
    p_list = [int(x) for x in args.p_list.split(",") if x.strip()]
    cell_order = [x.strip() for x in args.cell_order.split("|") if x.strip()]
    if len(cell_order) < 2 or len(set(cell_order)) != len(cell_order):
        raise ValueError("--cell-order must contain distinct pipe-separated labels.")

    for p_target in p_list:
        expression_file = processed / f"CEDAR_expression_p{p_target}.csv.gz"
        metadata_file = processed / f"CEDAR_metadata_p{p_target}.csv"
        if not expression_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"Missing expression/metadata pair for p={p_target}: "
                f"{expression_file}, {metadata_file}"
            )

        metadata = pd.read_csv(metadata_file)
        required = {"donor", "cell_type", "idat"}
        if not required.issubset(metadata.columns):
            raise ValueError(f"{metadata_file} lacks columns {sorted(required)}")
        metadata = metadata.copy()
        metadata["idat_key"] = basename_series(metadata["idat"])
        if metadata["idat_key"].duplicated().any():
            duplicate = metadata.loc[metadata["idat_key"].duplicated(), "idat_key"].iloc[0]
            raise ValueError(f"Duplicate IDAT basename in metadata: {duplicate}")

        expression = pd.read_csv(expression_file, index_col=0)
        expression.columns = basename_series(pd.Series(expression.columns)).to_numpy()
        if expression.columns.duplicated().any():
            raise ValueError("Expression matrix has duplicate IDAT basenames.")
        missing = sorted(set(metadata["idat_key"]) - set(expression.columns))
        extra = sorted(set(expression.columns) - set(metadata["idat_key"]))
        if missing or extra:
            raise ValueError(
                "Expression/metadata IDAT mismatch. "
                f"missing_in_expression={missing[:3]}, extra_in_expression={extra[:3]}"
            )
        expression = expression.loc[:, metadata["idat_key"].to_list()]
        values = expression.to_numpy(dtype=np.float64, copy=False)
        if values.shape[0] != p_target or np.any(~np.isfinite(values)):
            raise ValueError(f"Invalid p={p_target} expression matrix: shape={values.shape}")

        unknown = sorted(set(metadata["cell_type"]) - set(cell_order))
        missing_cells = sorted(set(cell_order) - set(metadata["cell_type"]))
        if unknown or missing_cells:
            raise ValueError(
                f"Cell labels do not match --cell-order; unknown={unknown}, missing={missing_cells}"
            )
        counts = metadata.groupby(["donor", "cell_type"], sort=False).size()
        if not (counts >= 1).all():
            raise ValueError("Every retained donor/cell combination must occur at least once.")
        donors = sorted(metadata["donor"].astype(str).unique())
        expected = len(donors) * len(cell_order)
        if len(counts) != expected:
            raise ValueError(
                f"Incomplete donor-by-cell grid: observed_pairs={len(counts)}, expected={expected}"
            )

        column_index = {name: i for i, name in enumerate(expression.columns)}
        lookup: dict[tuple[str, str], list[int]] = {}
        for row in metadata.itertuples(index=False):
            key = (str(row.donor), str(row.cell_type))
            lookup.setdefault(key, []).append(column_index[row.idat_key])
        z_raw = np.empty((len(donors), p_target, len(cell_order)), dtype=np.float64)
        for i, donor in enumerate(donors):
            for j, cell_type in enumerate(cell_order):
                replicate_columns = lookup[(donor, cell_type)]
                z_raw[i, :, j] = np.mean(values[:, replicate_columns], axis=1)

        z_centered = z_raw - np.mean(z_raw, axis=0, keepdims=True)
        raw_path = processed / f"CEDAR_Z_raw_p{p_target}.npy"
        centered_path = processed / f"CEDAR_Z_centered_p{p_target}.npy"
        np.save(raw_path, z_raw)
        np.save(centered_path, z_centered)

        diagonal = np.mean(z_centered * z_centered, axis=(0, 2))
        manifest = {
            "p": p_target,
            "n_donors": len(donors),
            "q_cell_types": len(cell_order),
            "n_selected_arrays": len(metadata),
            "n_donor_cell_pairs": expected,
            "n_extra_technical_replicate_arrays": len(metadata) - expected,
            "max_technical_replicates_per_pair": int(counts.max()),
            "technical_replicate_rule": (
                "all retained IDATs were normalized jointly, then expression was "
                "averaged probe-by-probe within each donor-cell pair"
            ),
            "shape": list(z_centered.shape),
            "cell_order_raw_sdrf_labels": cell_order,
            "donors_sorted": donors,
            "expression_file": str(expression_file),
            "metadata_file": str(metadata_file),
            "raw_tensor": str(raw_path),
            "centered_tensor": str(centered_path),
            "max_abs_raw_feature_cell_mean": float(np.max(np.abs(np.mean(z_raw, axis=0)))),
            "max_abs_centered_feature_cell_mean": float(
                np.max(np.abs(np.mean(z_centered, axis=0)))
            ),
            "centered_variance_min": float(np.min(diagonal)),
            "centered_variance_median": float(np.median(diagonal)),
            "centered_variance_max": float(np.max(diagonal)),
            "centered_variance_ratio_max_min": float(np.max(diagonal) / np.min(diagonal)),
        }
        manifest_path = processed / f"CEDAR_tensor_manifest_p{p_target}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"p={p_target}: shape={z_centered.shape}")
        print(f"  raw:      {raw_path}")
        print(f"  centered: {centered_path}")
        print(f"  manifest: {manifest_path}")
        print(
            "  max abs centered mean:",
            f"{manifest['max_abs_centered_feature_cell_mean']:.3e}",
        )


if __name__ == "__main__":
    main()
