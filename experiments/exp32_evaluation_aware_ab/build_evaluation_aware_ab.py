#!/usr/bin/env python3
"""Evaluate conditional-mean, evaluation-aware A+B projections.

For each calibration-only estimator h(r), this experiment evaluates h on the
unlabeled evaluation proxy scores and computes the moment coefficients of
(h(r_eval), r_eval). It then predicts

    Delta(d) = A_h sqrt(d) + B_h d.

The corresponding direct calibrator integrates the same h(r_eval) under the
exact evaluation proxy tilt. Their difference therefore isolates local-law
truncation after matching both the learned conditional mean and the available
evaluation-side proxy geometry.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP21_PATH = (
    REPO_ROOT
    / "experiments"
    / "exp21_direct_calibration"
    / "build_direct_calibration.py"
)

_spec = importlib.util.spec_from_file_location("exp21_eval_aware", EXP21_PATH)
exp21 = importlib.util.module_from_spec(_spec)
sys.modules["exp21_eval_aware"] = exp21
_spec.loader.exec_module(exp21)  # type: ignore[union-attr]

SOURCE_TO_PROJECTION = {
    "isotonic_calibration": "A+B (isotonic h)",
    "binned_shrinkage": "A+B (binned h)",
    "spline_ridge": "A+B (spline h)",
    "mlp_calibration": "A+B (MLP h)",
}

_original_fit_split_models = exp21.fit_split_models
_original_predictions_at_d = exp21.predictions_at_d


def fit_split_models(
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    r_eval: np.ndarray,
    fit_max_d: float,
) -> dict:
    models = _original_fit_split_models(r_cal, j_cal, r_eval, fit_max_d)
    projection_coefficients = {}
    for source, h_eval in models["h_eval"].items():
        if source not in SOURCE_TO_PROJECTION:
            continue
        projection_coefficients[source] = exp21.calib.theory_coefficients(
            np.asarray(h_eval, dtype=float),
            np.asarray(r_eval, dtype=float),
        )
    models["evaluation_aware_coefficients"] = projection_coefficients
    return models


def predictions_at_d(
    split_models: dict,
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    eval_weights: np.ndarray,
    d: float,
) -> tuple[dict[str, float], dict[str, str]]:
    predictions, errors = _original_predictions_at_d(
        split_models, r_cal, j_cal, eval_weights, d
    )
    sqrt_d = math.sqrt(max(0.0, float(d)))
    coefficients = split_models["evaluation_aware_coefficients"]
    for source, output_name in SOURCE_TO_PROJECTION.items():
        if source not in coefficients:
            errors[output_name] = (
                f"conditional-mean fit unavailable for source {source}"
            )
            continue
        coeff = coefficients[source]
        predictions[output_name] = (
            coeff.a_exact * sqrt_d + coeff.b_theory * float(d)
        )
    return predictions, errors


def output_dir_from_argv() -> Path:
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--outdir":
            return Path(sys.argv[index + 1])
        if arg.startswith("--outdir="):
            return Path(arg.split("=", 1)[1])
    return exp21.OUT_DIR


def augment_run_spec(outdir: Path) -> None:
    spec_path = outdir / "direct_calibration_run_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["evaluation_aware_conditional_mean_ab"] = {
        "fit_labels": "calibration gold labels only",
        "evaluation_inputs": "evaluation proxy scores only; no evaluation gold",
        "coefficient_distribution": "(h(r_eval), r_eval)",
        "formula": "A_h*sqrt(d)+B_h*d",
        "direct_comparator": (
            "exact integration of the same h(r_eval) under evaluation tilt"
        ),
        "source_to_projection": SOURCE_TO_PROJECTION,
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    exp21.fit_split_models = fit_split_models
    exp21.predictions_at_d = predictions_at_d
    projections = list(SOURCE_TO_PROJECTION.values())
    exp21.ALL_BASELINES = [*exp21.ALL_BASELINES, *projections]
    exp21.NEW_BASELINES = [*exp21.NEW_BASELINES, *projections]
    exp21.main()
    augment_run_spec(output_dir_from_argv())


if __name__ == "__main__":
    main()
