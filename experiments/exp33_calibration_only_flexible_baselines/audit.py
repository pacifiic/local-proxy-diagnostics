#!/usr/bin/env python3
"""Audit and summarize strict calibration-only flexible baselines."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BUILD_SPEC = importlib.util.spec_from_file_location(
    "exp33_builder", HERE / "build.py"
)
builder = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules["exp33_builder"] = builder
BUILD_SPEC.loader.exec_module(builder)

AUDIT21_PATH = (
    ROOT / "experiments" / "exp21_direct_calibration"
    / "audit_direct_calibration.py"
)
AUDIT21_SPEC = importlib.util.spec_from_file_location(
    "exp21_audit", AUDIT21_PATH
)
audit21 = importlib.util.module_from_spec(AUDIT21_SPEC)
sys.modules["exp21_audit"] = audit21
AUDIT21_SPEC.loader.exec_module(audit21)

STRICT_TO_TARGET = {
    "isotonic_calibration_only": "isotonic_calibration",
    "binned_calibration_only": "binned_shrinkage",
    "spline_calibration_only": "spline_ridge",
    "mlp_calibration_only": "mlp_calibration",
}


def validate_saved_rows(frame: pd.DataFrame) -> dict:
    key = ["dataset", "path_id", "repeat", "d"]
    if frame.duplicated(key + ["baseline"]).any():
        raise AssertionError("duplicate prediction cells")
    error_drift = float(
        np.max(
            np.abs(
                frame["abs_error"].to_numpy()
                - np.abs(
                    frame["predicted_gain"].to_numpy()
                    - frame["actual_gain"].to_numpy()
                )
            )
        )
    )
    actual_spread = (
        frame.groupby(key)["actual_gain"].agg(lambda values: values.max() - values.min())
    )
    if error_drift > 1e-12 or float(actual_spread.max()) > 1e-12:
        raise AssertionError("saved prediction identities failed")
    return {
        "n_rows": len(frame),
        "n_paths": int(frame["path_id"].nunique()),
        "max_abs_error_identity_drift": error_drift,
        "max_actual_gain_spread_across_baselines": float(
            actual_spread.max()
        ),
    }


def compare_shared_predictions(
    strict: pd.DataFrame, reference: pd.DataFrame
) -> dict:
    reference = reference[
        reference["dataset"].isin(strict["dataset"].unique())
    ].copy()
    shared = [
        baseline
        for baseline in builder.SHARED_BASELINES
        if baseline in set(reference["baseline"])
    ]
    keys = ["dataset", "path_id", "repeat", "d", "baseline"]
    left = strict[strict["baseline"].isin(shared)][
        keys + ["actual_gain", "predicted_gain"]
    ]
    right = reference[reference["baseline"].isin(shared)][
        keys + ["actual_gain", "predicted_gain"]
    ]
    paired = left.merge(
        right,
        on=keys,
        suffixes=("_strict", "_reference"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise AssertionError(f"shared baseline cell mismatch: {counts}")
    actual_drift = float(
        np.max(
            np.abs(
                paired["actual_gain_strict"]
                - paired["actual_gain_reference"]
            )
        )
    )
    prediction_drift = float(
        np.max(
            np.abs(
                paired["predicted_gain_strict"]
                - paired["predicted_gain_reference"]
            )
        )
    )
    if actual_drift > 1e-12 or prediction_drift > 1e-12:
        raise AssertionError("strict experiment does not reuse exp21 splits")
    return {
        "shared_baselines": shared,
        "n_paired_cells": len(paired),
        "max_actual_gain_drift": actual_drift,
        "max_prediction_drift": prediction_drift,
    }


def explicit_information_audit() -> dict:
    """Perturb evaluation proxies and verify frozen predictions do not move."""
    cfg = builder.exp21.CONFIGS[0]
    paths = list(
        builder.exp21.calib.build_paths(builder.exp21._make_args(cfg))
    )
    path = paths[0]
    rng = np.random.default_rng(builder.exp21.RANDOM_SEED)
    cal_idx, eval_idx = builder.exp21.calib.split_indices(
        len(path.item_ids), builder.exp21.CALIBRATION_FRAC, rng
    )
    r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
    j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
    r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
    j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
    models = builder.exp21.fit_split_models(
        r_cal, j_cal, r_cal, builder.exp21.FIT_MAX_D
    )
    pred_1, _ = builder.calibration_only_predictions(
        models, r_cal, j_cal, 0.10
    )

    # Prediction has no evaluation argument. Change evaluation proxy geometry
    # only in the realized outcome calculation.
    permuted = r_eval[::-1].copy()
    pred_2, _ = builder.calibration_only_predictions(
        models, r_cal, j_cal, 0.10
    )
    prediction_drift = max(
        abs(pred_1[name] - pred_2[name]) for name in pred_1
    )
    weights_1 = builder.exp21.calib.exact_weights_at_d(r_eval, 0.10)
    weights_2 = builder.exp21.calib.exact_weights_at_d(permuted, 0.10)
    actual_1 = float(np.sum(weights_1 * j_eval) - np.mean(j_eval))
    actual_2 = float(np.sum(weights_2 * j_eval) - np.mean(j_eval))
    actual_change = abs(actual_1 - actual_2)
    if prediction_drift > 0 or actual_change <= 1e-8:
        raise AssertionError("evaluation-proxy invariance audit failed")
    return {
        "prediction_drift_after_eval_proxy_permutation": prediction_drift,
        "realized_gain_change_after_eval_proxy_permutation": actual_change,
        "pass": True,
    }


def summarize_information_settings(
    strict_curve: pd.DataFrame, target_curve: pd.DataFrame
) -> pd.DataFrame:
    target_curve = target_curve[
        target_curve["dataset"].isin(strict_curve["dataset"].unique())
    ].copy()
    rows = []
    for strict_name, target_name in STRICT_TO_TARGET.items():
        left = strict_curve[
            strict_curve["baseline"].eq(strict_name)
        ][["dataset", "curve_mae", "n_complete_units"]]
        right = target_curve[
            target_curve["baseline"].eq(target_name)
        ][["dataset", "curve_mae", "n_complete_units"]]
        merged = left.merge(
            right,
            on="dataset",
            suffixes=("_calibration_only", "_target_aware"),
            validate="one_to_one",
        )
        for record in merged.to_dict("records"):
            rows.append(
                {
                    "dataset": record["dataset"],
                    "model_family": strict_name.replace(
                        "_calibration_only", ""
                    ),
                    "calibration_only_curve_mae": record[
                        "curve_mae_calibration_only"
                    ],
                    "target_aware_curve_mae": record[
                        "curve_mae_target_aware"
                    ],
                    "calibration_only_complete_units": record[
                        "n_complete_units_calibration_only"
                    ],
                    "target_aware_complete_units": record[
                        "n_complete_units_target_aware"
                    ],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--target-aware-reference", type=Path, required=True
    )
    args = parser.parse_args()

    strict = pd.read_csv(args.outdir / "calibration_only_predictions.csv")
    target = pd.read_csv(
        args.target_aware_reference / "direct_calibration_predictions.csv"
    )
    saved_audit = validate_saved_rows(strict)
    shared_audit = compare_shared_predictions(strict, target)
    information_audit = explicit_information_audit()

    endpoint, curve = audit21.curve_mae_table(strict)
    target_endpoint, target_curve = audit21.curve_mae_table(target)
    endpoint.to_csv(args.outdir / "endpoint_mae_by_d.csv", index=False)
    curve.to_csv(args.outdir / "curve_mae.csv", index=False)
    comparison = summarize_information_settings(curve, target_curve)
    comparison.to_csv(
        args.outdir / "information_setting_comparison.csv", index=False
    )

    paired_rows = []
    for dataset in strict["dataset"].unique():
        for baseline in builder.STRICT_BASELINES + [
            "empirical_curve",
            "empirical_fit",
            "A-only",
            "zero",
        ]:
            result = audit21.path_clustered_bootstrap_diff(
                strict, dataset, "A+B", baseline, d=None
            )
            paired_rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "metric": "CurveMAE",
                    "mae_A+B": result["mae_a"],
                    "mae_baseline": result["mae_b"],
                    "n_paths": result["n_paths"],
                    "n_paired_cells": result["n_paired_cells"],
                    "diff_AplusB_minus_baseline": result["diff_mean"],
                    "ci_lo": result["ci_lo"],
                    "ci_hi": result["ci_hi"],
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        args.outdir / "paired_diff_curvemae_bootstrap.csv", index=False
    )

    audit = {
        "status": "PASS",
        "saved_rows": saved_audit,
        "shared_exp21_identity": shared_audit,
        "evaluation_proxy_invariance": information_audit,
    }
    (args.outdir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(curve.pivot_table(
        index="dataset", columns="baseline", values="curve_mae"
    ).to_string())
    print("\nCalibration-only versus target-aware:")
    print(comparison.to_string(index=False))
    print("\nPASS")


if __name__ == "__main__":
    main()
