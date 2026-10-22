#!/usr/bin/env python3
"""Evaluate the tuning-free third Puiseux term C d^(3/2).

This experiment reuses exp21's data loading, outer splits, exact evaluation
frontier, and direct-calibration baselines. The only added predictor is

    A sqrt(d) + B d + C d^(3/2),

where C is computed from calibration-split population moments. Evaluation
labels are never used to estimate A, B, or C.
"""
from __future__ import annotations

import importlib.util
import hashlib
import importlib.metadata
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

_spec = importlib.util.spec_from_file_location("exp21_builder", EXP21_PATH)
exp21 = importlib.util.module_from_spec(_spec)
sys.modules["exp21_builder"] = exp21
_spec.loader.exec_module(exp21)  # type: ignore[union-attr]

BASELINE = "A+B+C"
_original_fit_split_models = exp21.fit_split_models
_original_predictions_at_d = exp21.predictions_at_d


def cubic_moment_coefficient(gold: np.ndarray, proxy: np.ndarray) -> float:
    """Return C in Delta(d)=A sqrt(d)+B d+C d^(3/2)+O(d^2).

    Moments use the empirical distribution (ddof=0), matching the finite-item
    tilt and the existing A/B implementation.
    """
    j = np.asarray(gold, dtype=float)
    r = np.asarray(proxy, dtype=float)
    if j.ndim != 1 or r.ndim != 1 or len(j) != len(r):
        raise ValueError("gold and proxy must be one-dimensional and aligned")
    if len(j) < 4:
        raise ValueError("at least four aligned items are required")
    if not np.all(np.isfinite(j)) or not np.all(np.isfinite(r)):
        raise ValueError("gold and proxy must be finite")

    sigma_r = float(np.std(r, ddof=0))
    if sigma_r <= 1e-12:
        raise ValueError("proxy variance is zero")

    z = (r - np.mean(r)) / sigma_r
    j_centered = j - np.mean(j)
    m1 = float(np.mean(j_centered * z))
    m2 = float(np.mean(j_centered * z**2))
    gamma1 = float(np.mean(z**3))
    gamma2 = float(np.mean(z**4) - 3.0)
    tau3 = float(np.mean(j_centered * z**3) - 3.0 * m1)

    c = math.sqrt(2.0) * (
        tau3 / 3.0
        - (2.0 / 3.0) * m2 * gamma1
        + (m1 / 36.0) * (20.0 * gamma1**2 - 9.0 * gamma2)
    )
    if not math.isfinite(c):
        raise ValueError("non-finite cubic moment coefficient")
    return c


def fit_split_models(
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    r_eval: np.ndarray,
    fit_max_d: float,
) -> dict:
    models = _original_fit_split_models(r_cal, j_cal, r_eval, fit_max_d)
    models["c_moment"] = cubic_moment_coefficient(j_cal, r_cal)
    return models


def predictions_at_d(
    split_models: dict,
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    eval_weights: np.ndarray,
    d: float,
) -> tuple[dict[str, float], dict[str, str]]:
    preds, errors = _original_predictions_at_d(
        split_models, r_cal, j_cal, eval_weights, d
    )
    preds[BASELINE] = (
        preds["A+B"] + split_models["c_moment"] * float(d) ** 1.5
    )
    return preds, errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_dir_from_argv() -> Path:
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--outdir":
            return Path(sys.argv[index + 1])
        if arg.startswith("--outdir="):
            return Path(arg.split("=", 1)[1])
    return Path(exp21.OUT_DIR)


def augment_run_spec(outdir: Path) -> None:
    spec_path = outdir / "direct_calibration_run_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    input_paths = {
        "summeval_long": exp21.DATA_DIR / "summeval_long.csv",
        "gpt4o_proxy": exp21.DATA_DIR / "model_proxy_4o_aggregated.csv",
        "wmt_mqm_long": exp21.DATA_DIR / "wmt_mqm_long.csv",
        "openmeva_long": exp21.DATA_DIR / "openmeva_long.csv",
    }
    code_paths = {
        "exp27_builder": Path(__file__).resolve(),
        "exp21_builder": EXP21_PATH.resolve(),
        "canonical_engine": exp21.CANON_PATH.resolve(),
    }
    spec["cubic_moment_term"] = {
        "baseline": BASELINE,
        "formula": (
            "sqrt(2)*(tau3/3-(2/3)*m2*gamma1"
            "+m1*(20*gamma1^2-9*gamma2)/36)"
        ),
        "moment_convention": "population empirical moments; ddof=0",
        "fit_data": "calibration split only",
    }
    spec["input_sha256"] = {
        name: sha256_file(path) for name, path in input_paths.items()
    }
    spec["code_sha256"] = {
        name: sha256_file(path) for name, path in code_paths.items()
    }
    spec["package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn")
    }
    temporary = spec_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    temporary.replace(spec_path)


def main() -> None:
    # exp21 workers resolve these names from their defining module at runtime.
    # They use Linux fork workers, so the patched globals and fixed split plan
    # are inherited without refitting or changing random-number consumption.
    exp21.fit_split_models = fit_split_models
    exp21.predictions_at_d = predictions_at_d
    exp21.ALL_BASELINES = [*exp21.ALL_BASELINES, BASELINE]
    exp21.NEW_BASELINES = [*exp21.NEW_BASELINES, BASELINE]
    exp21.main()
    augment_run_spec(output_dir_from_argv())


if __name__ == "__main__":
    main()
