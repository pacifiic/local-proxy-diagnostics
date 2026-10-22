#!/usr/bin/env python3
"""Audit and summarize exp27's A+B+C comparison."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP21_AUDIT = (
    REPO_ROOT
    / "experiments"
    / "exp21_direct_calibration"
    / "audit_direct_calibration.py"
)

_spec = importlib.util.spec_from_file_location("exp21_audit", EXP21_AUDIT)
audit21 = importlib.util.module_from_spec(_spec)
sys.modules["exp21_audit"] = audit21
_spec.loader.exec_module(audit21)  # type: ignore[union-attr]

KEYS = ["dataset", "path_id", "repeat", "d"]
REQUIRED = ["A+B+C", "A+B", "A-only", "isotonic_calibration", "spline_ridge"]
COMPARE = ["A+B", "A-only", "isotonic_calibration", "spline_ridge"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        *KEYS,
        "baseline",
        "actual_gain",
        "predicted_gain",
        "abs_error",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    if df.duplicated([*KEYS, "baseline"]).any():
        raise ValueError("duplicate dataset/path/repeat/d/baseline rows")
    if not np.isfinite(
        df[["actual_gain", "predicted_gain", "abs_error"]].to_numpy()
    ).all():
        raise ValueError("non-finite prediction values")

    supports = {}
    for baseline in REQUIRED:
        supports[baseline] = set(
            map(tuple, df.loc[df["baseline"] == baseline, KEYS].to_numpy())
        )
    reference = supports["A+B+C"]
    for baseline, support in supports.items():
        if support != reference:
            raise ValueError(
                f"{baseline} support differs from A+B+C: "
                f"{len(support)} vs {len(reference)}"
            )

    abc = df[df["baseline"] == "A+B+C"][KEYS + ["predicted_gain"]].rename(
        columns={"predicted_gain": "pred_abc"}
    )
    ab = df[df["baseline"] == "A+B"][KEYS + ["predicted_gain"]].rename(
        columns={"predicted_gain": "pred_ab"}
    )
    paired = abc.merge(ab, on=KEYS, validate="one_to_one")
    paired["c_derived"] = (
        (paired["pred_abc"] - paired["pred_ab"])
        / np.power(paired["d"], 1.5)
    )

    # One C is fitted per outer split, so all radii within a split must imply
    # exactly the same coefficient up to floating-point rounding.
    spread = (
        paired.groupby(["dataset", "path_id", "repeat"])["c_derived"]
        .agg(lambda x: float(np.max(x) - np.min(x)))
    )
    if float(spread.max()) > 1e-9:
        raise ValueError(
            f"C changes across radii within a split (max spread={spread.max():.3g})"
        )
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    path = args.outdir / "direct_calibration_predictions.csv"
    df = pd.read_csv(path)
    paired = validate(df)
    endpoint, curve = audit21.curve_mae_table(df)
    endpoint.to_csv(args.outdir / "endpoint_mae_by_d.csv", index=False)
    curve.to_csv(args.outdir / "curve_mae.csv", index=False)

    rows = []
    for dataset in sorted(df["dataset"].unique()):
        for baseline in COMPARE:
            result = audit21.path_clustered_bootstrap_diff(
                df,
                dataset,
                "A+B+C",
                baseline,
                d=None,
                n_boot=args.n_boot,
                seed=0,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "metric": "CurveMAE",
                    "mae_A+B+C": result["mae_a"],
                    "mae_baseline": result["mae_b"],
                    "diff_ABC_minus_baseline": result["diff_mean"],
                    "ci_lo": result["ci_lo"],
                    "ci_hi": result["ci_hi"],
                    "n_paths": result["n_paths"],
                    "n_paired_cells": result["n_paired_cells"],
                }
            )
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(
        args.outdir / "paired_diff_curvemae_bootstrap.csv", index=False
    )

    c_by_split = (
        paired.groupby(["dataset", "path_id", "repeat"], as_index=False)
        ["c_derived"]
        .first()
    )
    c_summary = (
        c_by_split.groupby("dataset")["c_derived"]
        .agg(
            n="size",
            mean="mean",
            std="std",
            median="median",
            q05=lambda x: x.quantile(0.05),
            q95=lambda x: x.quantile(0.95),
            max_abs=lambda x: np.max(np.abs(x)),
        )
        .reset_index()
    )
    c_by_split.to_csv(args.outdir / "c_coefficient_by_split.csv", index=False)
    c_summary.to_csv(args.outdir / "c_coefficient_summary.csv", index=False)

    print("CurveMAE:")
    print(
        curve[curve["baseline"].isin(REQUIRED)]
        .pivot(index="dataset", columns="baseline", values="curve_mae")
        .to_string()
    )
    print("\nA+B+C minus comparator; negative favors A+B+C:")
    print(comparisons.to_string(index=False))
    print("\nC stability:")
    print(c_summary.to_string(index=False))
    print("\nAUDIT PASSED")


if __name__ == "__main__":
    main()
