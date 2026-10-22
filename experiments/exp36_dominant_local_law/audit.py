#!/usr/bin/env python3
"""Audit and summarize exp36 dominant-local-law predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_D = (0.01, 0.025, 0.05, 0.075, 0.10)
REFERENCE = "A+B_moment"
COMPARATORS = (
    "calibration_curve",
    "empirical_two_term",
    "empirical_cubic",
    "direct_spline",
    "A+B_residual_spline",
)
KEY_PAIRWISE_COMPARISONS = (
    ("empirical_two_term", "A+B_moment"),
    ("empirical_cubic", "empirical_two_term"),
    ("direct_spline", "empirical_two_term"),
    ("A+B_residual_spline", "empirical_two_term"),
)


def path_bootstrap_ci(
    paired: pd.DataFrame,
    n_bootstrap: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap paths, retaining all repeats within each sampled path."""
    by_path = (
        paired.groupby("path_id", as_index=False)["difference"]
        .mean()
        .sort_values("path_id")
    )
    values = by_path["difference"].to_numpy(dtype=float)
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    draw = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = draw.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return estimate, float(low), float(high)


def complete_unit_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    expected = set(EXPECTED_D)
    rows = []
    for key, group in predictions.groupby(
        ["dataset", "method", "path_id", "repeat"], sort=False
    ):
        dataset, method, path_id, repeat = key
        radii = set(np.round(group["d"].to_numpy(dtype=float), 12))
        if radii != expected:
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "path_id": path_id,
                "repeat": int(repeat),
                "curve_mae": float(group["abs_error"].mean()),
                "curve_rmse": float(
                    np.sqrt(np.mean(np.square(group["residual"])))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    prediction_path = args.outdir / "predictions.csv"
    predictions = pd.read_csv(prediction_path)

    required = {
        "dataset",
        "path_id",
        "repeat",
        "d",
        "method",
        "information_setting",
        "actual_gain",
        "predicted_gain",
        "residual",
        "abs_error",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    if predictions.duplicated(
        ["dataset", "path_id", "repeat", "d", "method"]
    ).any():
        raise ValueError("duplicate prediction keys")
    numeric = predictions[
        ["actual_gain", "predicted_gain", "residual", "abs_error"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite prediction values")
    if not np.allclose(
        predictions["residual"],
        predictions["actual_gain"] - predictions["predicted_gain"],
        atol=1e-12,
    ):
        raise ValueError("residual column is inconsistent")
    if not np.allclose(
        predictions["abs_error"],
        np.abs(predictions["residual"]),
        atol=1e-12,
    ):
        raise ValueError("absolute-error column is inconsistent")

    actual_counts = (
        predictions.groupby(["dataset", "path_id", "repeat", "d"])[
            "actual_gain"
        ]
        .nunique()
    )
    if int(actual_counts.max()) != 1:
        raise ValueError("actual gain changes across methods")

    prospective = predictions[
        predictions["information_setting"] == "calibration_only_pre_pool"
    ]
    if (prospective["method"] == "eval_oracle_A+B").any():
        raise ValueError("eval oracle was mislabeled as prospective")

    units = complete_unit_errors(predictions)
    units.to_csv(args.outdir / "curve_error_by_unit.csv", index=False)
    summary = (
        units.groupby(["dataset", "method"], as_index=False)
        .agg(
            curve_mae=("curve_mae", "mean"),
            curve_rmse=("curve_rmse", "mean"),
            n_complete_units=("curve_mae", "size"),
            n_paths=("path_id", "nunique"),
        )
        .sort_values(["dataset", "curve_mae", "method"])
    )
    summary.to_csv(args.outdir / "curve_error_summary.csv", index=False)

    support = (
        predictions.groupby(["dataset", "path_id", "repeat"], as_index=False)[
            "cal_max_reachable_d"
        ]
        .first()
    )
    support["common_reachable_support"] = (
        support["cal_max_reachable_d"] > max(EXPECTED_D) + 1e-10
    )
    common_units = units.merge(
        support[
            ["dataset", "path_id", "repeat", "common_reachable_support"]
        ],
        on=["dataset", "path_id", "repeat"],
        how="left",
        validate="many_to_one",
    )
    common_units = common_units[common_units["common_reachable_support"]]
    common_support_summary = (
        common_units.groupby(["dataset", "method"], as_index=False)
        .agg(
            curve_mae=("curve_mae", "mean"),
            n_complete_units=("curve_mae", "size"),
            n_paths=("path_id", "nunique"),
        )
        .sort_values(["dataset", "curve_mae", "method"])
    )
    common_support_summary.to_csv(
        args.outdir / "common_reachable_support_summary.csv", index=False
    )
    excluded_support = (
        support.groupby("dataset")["common_reachable_support"]
        .agg(n_units="size", n_common_support="sum")
        .reset_index()
    )
    excluded_support["n_excluded"] = (
        excluded_support["n_units"] - excluded_support["n_common_support"]
    )
    excluded_support.to_csv(
        args.outdir / "common_reachable_support_accounting.csv", index=False
    )

    paired_rows = []
    for dataset in sorted(units["dataset"].unique()):
        reference = units[
            (units["dataset"] == dataset) & (units["method"] == REFERENCE)
        ][["path_id", "repeat", "curve_mae"]].rename(
            columns={"curve_mae": "reference_mae"}
        )
        for method in COMPARATORS:
            comparator = units[
                (units["dataset"] == dataset) & (units["method"] == method)
            ][["path_id", "repeat", "curve_mae"]].rename(
                columns={"curve_mae": "comparator_mae"}
            )
            paired = reference.merge(
                comparator, on=["path_id", "repeat"], how="inner"
            )
            paired["difference"] = (
                paired["comparator_mae"] - paired["reference_mae"]
            )
            estimate, low, high = path_bootstrap_ci(paired)
            paired_rows.append(
                {
                    "dataset": dataset,
                    "comparator": method,
                    "difference_comparator_minus_A+B": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_paired_units": len(paired),
                    "n_paths": paired["path_id"].nunique(),
                    "fraction_A+B_better": float(
                        (paired["difference"] > 0).mean()
                    ),
                }
            )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(
        args.outdir / "paired_vs_A+B_bootstrap.csv", index=False
    )

    key_pair_rows = []
    for dataset in sorted(units["dataset"].unique()):
        dataset_units = units[units["dataset"] == dataset]
        for first_method, second_method in KEY_PAIRWISE_COMPARISONS:
            first = dataset_units[
                dataset_units["method"] == first_method
            ][["path_id", "repeat", "curve_mae"]].rename(
                columns={"curve_mae": "first_mae"}
            )
            second = dataset_units[
                dataset_units["method"] == second_method
            ][["path_id", "repeat", "curve_mae"]].rename(
                columns={"curve_mae": "second_mae"}
            )
            paired = first.merge(
                second, on=["path_id", "repeat"], how="inner"
            )
            paired["difference"] = (
                paired["first_mae"] - paired["second_mae"]
            )
            estimate, low, high = path_bootstrap_ci(paired)
            key_pair_rows.append(
                {
                    "dataset": dataset,
                    "first_method": first_method,
                    "second_method": second_method,
                    "difference_first_minus_second": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_paired_units": len(paired),
                    "n_paths": paired["path_id"].nunique(),
                }
            )
    pd.DataFrame(key_pair_rows).to_csv(
        args.outdir / "key_pairwise_bootstrap.csv", index=False
    )

    oracle = predictions[predictions["method"] == "eval_oracle_A+B"].copy()
    oracle_units = []
    for key, group in oracle.groupby(
        ["dataset", "path_id", "repeat"], sort=False
    ):
        dataset, path_id, repeat = key
        if set(np.round(group["d"], 12)) != set(EXPECTED_D):
            continue
        exact = group["actual_gain"].to_numpy(dtype=float)
        residual = group["residual"].to_numpy(dtype=float)
        exact_energy = float(np.sum(np.square(exact)))
        residual_energy = float(np.sum(np.square(residual)))
        oracle_units.append(
            {
                "dataset": dataset,
                "path_id": path_id,
                "repeat": int(repeat),
                "mean_abs_truncation_residual": float(
                    np.mean(np.abs(residual))
                ),
                "max_abs_truncation_residual": float(
                    np.max(np.abs(residual))
                ),
                "relative_residual_rms": (
                    float(np.sqrt(residual_energy / exact_energy))
                    if exact_energy > 0
                    else np.nan
                ),
                "gain_energy_explained": (
                    1.0 - residual_energy / exact_energy
                    if exact_energy > 0
                    else np.nan
                ),
            }
        )
    oracle_units = pd.DataFrame(oracle_units)
    oracle_units.to_csv(
        args.outdir / "oracle_truncation_by_unit.csv", index=False
    )
    approximation = (
        oracle_units.groupby("dataset", as_index=False)
        .agg(
            median_mean_abs_residual=(
                "mean_abs_truncation_residual",
                "median",
            ),
            median_max_abs_residual=(
                "max_abs_truncation_residual",
                "median",
            ),
            median_relative_residual_rms=(
                "relative_residual_rms",
                "median",
            ),
            median_gain_energy_explained=(
                "gain_energy_explained",
                "median",
            ),
            n_units=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
        .sort_values("dataset")
    )
    approximation.to_csv(
        args.outdir / "oracle_truncation_summary.csv", index=False
    )

    table_methods = [
        "A+B_moment",
        "empirical_two_term",
        "empirical_cubic",
        "direct_spline",
        "A+B_residual_spline",
    ]
    table = summary[summary["method"].isin(table_methods)].pivot(
        index="dataset", columns="method", values="curve_mae"
    )
    table = table.reindex(columns=table_methods).reset_index()
    table.to_csv(args.outdir / "model_comparison.csv", index=False)
    (args.outdir / "model_comparison.md").write_text(
        table.to_markdown(index=False, floatfmt=".5f") + "\n",
        encoding="utf-8",
    )

    audit = {
        "status": "passed",
        "prediction_rows": int(len(predictions)),
        "datasets": sorted(predictions["dataset"].unique().tolist()),
        "methods": sorted(predictions["method"].unique().tolist()),
        "complete_curve_units": int(len(units)),
        "common_reachable_support": excluded_support.to_dict(
            orient="records"
        ),
        "prospective_information_check": "passed",
        "actual_gain_invariance_check": "passed",
        "formula_consistency_check": "passed",
        "notes": [
            (
                "eval_oracle_A+B uses evaluation gold only retrospectively to "
                "measure local-law truncation error."
            ),
            (
                "All other methods use calibration items only; evaluation "
                "proxy and gold enter only the realized target."
            ),
        ],
    }
    (args.outdir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
