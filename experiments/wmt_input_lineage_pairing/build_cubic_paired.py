#!/usr/bin/env python3
"""Run the calibration-only cubic moment extension with consolidated exp21."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP21_PATH = (
    ROOT / "experiments" / "exp21_direct_calibration" / "build_direct_calibration.py"
)
spec = importlib.util.spec_from_file_location("paired_exp21_builder", EXP21_PATH)
exp21 = importlib.util.module_from_spec(spec)
sys.modules["paired_exp21_builder"] = exp21
spec.loader.exec_module(exp21)

BASELINE = "A+B+C"
original_fit = exp21.fit_split_models
original_predict = exp21.predictions_at_d


def cubic_moment_coefficient(gold: np.ndarray, proxy: np.ndarray) -> float:
    j = np.asarray(gold, dtype=float)
    r = np.asarray(proxy, dtype=float)
    sigma_r = float(np.std(r, ddof=0))
    if sigma_r <= 1e-12:
        raise ValueError("proxy variance is zero")
    z = (r - np.mean(r)) / sigma_r
    jc = j - np.mean(j)
    m1 = float(np.mean(jc * z))
    m2 = float(np.mean(jc * z**2))
    gamma1 = float(np.mean(z**3))
    gamma2 = float(np.mean(z**4) - 3.0)
    tau3 = float(np.mean(jc * z**3) - 3.0 * m1)
    return math.sqrt(2.0) * (
        tau3 / 3.0
        - (2.0 / 3.0) * m2 * gamma1
        + (m1 / 36.0) * (20.0 * gamma1**2 - 9.0 * gamma2)
    )


def fit_split_models(r_cal, j_cal, r_eval, fit_max_d):
    models = original_fit(r_cal, j_cal, r_eval, fit_max_d)
    models["c_moment"] = cubic_moment_coefficient(j_cal, r_cal)
    return models


def predictions_at_d(split_models, r_cal, j_cal, eval_weights, d):
    predictions, errors = original_predict(
        split_models, r_cal, j_cal, eval_weights, d
    )
    predictions[BASELINE] = (
        predictions["A+B"] + split_models["c_moment"] * float(d) ** 1.5
    )
    return predictions, errors


def output_dir() -> Path:
    for index, arg in enumerate(sys.argv[1:]):
        if arg == "--outdir":
            return Path(sys.argv[index + 2])
        if arg.startswith("--outdir="):
            return Path(arg.split("=", 1)[1])
    return exp21.OUT_DIR


def main() -> None:
    exp21.fit_split_models = fit_split_models
    exp21.predictions_at_d = predictions_at_d
    exp21.ALL_BASELINES = [*exp21.ALL_BASELINES, BASELINE]
    exp21.NEW_BASELINES = [*exp21.NEW_BASELINES, BASELINE]
    exp21.main()
    spec_path = output_dir() / "direct_calibration_run_spec.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["cubic_moment_term"] = {
        "baseline": BASELINE,
        "fit_data": "calibration split only",
        "moment_convention": "population empirical moments; ddof=0",
    }
    spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
