#!/usr/bin/env python3
"""Test whether flexible radius regressions improve on the local coefficient law.

All deployable predictors are fit on the 30% calibration item split.  The 70%
evaluation split is used only to calculate the exact finite-tilt gain target.
The eval-oracle A+B row is explicitly retrospective: it isolates truncation
error and is not a prospective predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP21_PATH = (
    ROOT / "experiments" / "exp21_direct_calibration"
    / "build_direct_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("exp21_builder_exp36", EXP21_PATH)
exp21 = importlib.util.module_from_spec(SPEC)
sys.modules["exp21_builder_exp36"] = exp21
SPEC.loader.exec_module(exp21)

D_TARGETS = tuple(float(d) for d in exp21.D_TARGETS)
FIT_D_GRID = np.linspace(0.0, exp21.FIT_MAX_D, 41)
SPLINE_N_KNOTS = 6
SPLINE_DEGREE = 3
SPLINE_ALPHAS = np.logspace(-8, 2, 21)

PROSPECTIVE_METHODS = (
    "A+B_moment",
    "A-only_moment",
    "calibration_curve",
    "empirical_two_term",
    "empirical_cubic",
    "direct_spline",
    "A+B_residual_spline",
)
DIAGNOSTIC_METHODS = ("eval_oracle_A+B",)
ALL_METHODS = PROSPECTIVE_METHODS + DIAGNOSTIC_METHODS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_origin_polynomial(d: np.ndarray, gain: np.ndarray, order: int) -> np.ndarray:
    """Fit sum_{k=1}^order beta_k (sqrt(d))^k without an intercept."""
    x = np.sqrt(np.asarray(d, dtype=float))
    y = np.asarray(gain, dtype=float)
    design = np.column_stack([x ** k for k in range(1, order + 1)])
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(coefficient, dtype=float)


def predict_origin_polynomial(coefficient: np.ndarray, d: float) -> float:
    x = math.sqrt(max(0.0, float(d)))
    return float(
        sum(float(beta) * x ** k for k, beta in enumerate(coefficient, start=1))
    )


def fit_centered_spline(d: np.ndarray, values: np.ndarray):
    """Fit a fixed-complexity cubic spline and force its value at d=0 to zero."""
    x = np.sqrt(np.asarray(d, dtype=float)).reshape(-1, 1)
    y = np.asarray(values, dtype=float)
    model = make_pipeline(
        SplineTransformer(
            n_knots=SPLINE_N_KNOTS,
            degree=SPLINE_DEGREE,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
        ),
        RidgeCV(alphas=SPLINE_ALPHAS, fit_intercept=True),
    )
    model.fit(x, y)
    origin = float(model.predict(np.zeros((1, 1), dtype=float))[0])

    def predict(d_new: float) -> float:
        x_new = np.array([[math.sqrt(max(0.0, float(d_new)))]])
        return float(model.predict(x_new)[0] - origin)

    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridgecv"]
    n_basis = int(transformer.transform(x[:1]).shape[1])
    metadata = {
        "n_knots": SPLINE_N_KNOTS,
        "degree": SPLINE_DEGREE,
        "n_basis": n_basis,
        "ridge_alpha": float(ridge.alpha_),
    }
    return predict, metadata


def fit_calibration_curve_models(
    proxy: np.ndarray,
    gold: np.ndarray,
) -> dict:
    """Fit radius-only models on the exact calibration finite-tilt curve."""
    r = np.asarray(proxy, dtype=float)
    j = np.asarray(gold, dtype=float)
    coefficient = exp21.calib.theory_coefficients(j, r)
    max_d = exp21.max_reachable_d(r)
    fit_max_d = min(float(exp21.FIT_MAX_D), 0.95 * max_d)
    if fit_max_d <= 1e-8:
        raise ValueError("calibration proxy has zero reachable KL")

    d_fit = FIT_D_GRID[FIT_D_GRID <= fit_max_d + 1e-12]
    if len(d_fit) < 8:
        d_fit = np.linspace(0.0, fit_max_d, 21)
    # Solve the tilt separately at every requested d. Interpolating a regular
    # t-grid onto d introduced errors large enough to obscure the very small
    # spline-versus-two-term differences this experiment is designed to audit.
    gain_fit = np.array(
        [
            exp21.calib.exact_delta_at_d(r, j, float(d))
            if d > 0
            else 0.0
            for d in d_fit
        ],
        dtype=float,
    )

    two_term = fit_origin_polynomial(d_fit, gain_fit, order=2)
    cubic = fit_origin_polynomial(d_fit, gain_fit, order=3)
    direct_spline, direct_meta = fit_centered_spline(d_fit, gain_fit)

    moment_fit = (
        coefficient.a_exact * np.sqrt(d_fit)
        + coefficient.b_theory * d_fit
    )
    residual_fit = gain_fit - moment_fit
    residual_spline, residual_meta = fit_centered_spline(
        d_fit, residual_fit
    )
    return {
        "coefficient": coefficient,
        "max_d": max_d,
        "fit_max_d": fit_max_d,
        "two_term": two_term,
        "cubic": cubic,
        "direct_spline": direct_spline,
        "direct_meta": direct_meta,
        "residual_spline": residual_spline,
        "residual_meta": residual_meta,
    }


def predictions_at_d(models: dict, d: float) -> dict[str, float]:
    coefficient = models["coefficient"]
    sqrt_d = math.sqrt(max(0.0, d))
    base = coefficient.a_exact * sqrt_d + coefficient.b_theory * d
    predictions = {
        "A+B_moment": base,
        "A-only_moment": coefficient.a_exact * sqrt_d,
        "empirical_two_term": predict_origin_polynomial(
            models["two_term"], d
        ),
        "empirical_cubic": predict_origin_polynomial(models["cubic"], d),
        "direct_spline": models["direct_spline"](d),
        "A+B_residual_spline": base + models["residual_spline"](d),
    }
    if d < models["max_d"] - 1e-10:
        predictions["calibration_curve"] = exp21.calib.exact_delta_at_d(
            models["proxy"], models["gold"], d
        )
    return predictions


def run_path_task(task):
    cfg, path, split_plan = task
    rows: list[dict] = []
    skipped: list[dict] = []
    try:
        from threadpoolctl import threadpool_limits

        thread_limit = threadpool_limits(limits=1)
    except ImportError:
        from contextlib import nullcontext

        thread_limit = nullcontext()

    with thread_limit:
        for repeat, (cal_idx, eval_idx) in enumerate(split_plan):
            r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
            r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
            j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
            try:
                models = fit_calibration_curve_models(r_cal, j_cal)
                models["proxy"] = r_cal
                models["gold"] = j_cal
                eval_coefficient = exp21.calib.theory_coefficients(
                    j_eval, r_eval
                )
            except Exception as exc:  # noqa: BLE001
                for d in D_TARGETS:
                    skipped.append(
                        {
                            "dataset": cfg["dataset_name"],
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": d,
                            "method": "ALL",
                            "reason": (
                                f"split_fit_failed: {type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                continue

            for d in D_TARGETS:
                try:
                    eval_weights = exp21.calib.exact_weights_at_d(r_eval, d)
                    actual = float(
                        np.sum(eval_weights * j_eval) - np.mean(j_eval)
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped.append(
                        {
                            "dataset": cfg["dataset_name"],
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": d,
                            "method": "ALL",
                            "reason": (
                                "evaluation_unreachable: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                    continue

                predictions = predictions_at_d(models, d)
                predictions["eval_oracle_A+B"] = (
                    eval_coefficient.a_exact * math.sqrt(d)
                    + eval_coefficient.b_theory * d
                )
                for method in ALL_METHODS:
                    if method not in predictions:
                        skipped.append(
                            {
                                "dataset": cfg["dataset_name"],
                                "path_id": path.path_id,
                                "repeat": repeat,
                                "d": d,
                                "method": method,
                                "reason": (
                                    f"calibration max KL={models['max_d']:.6g} "
                                    f"cannot reach d={d:g}"
                                ),
                            }
                        )
                        continue
                    predicted = float(predictions[method])
                    information_setting = (
                        "retrospective_eval_gold_diagnostic"
                        if method in DIAGNOSTIC_METHODS
                        else "calibration_only_pre_pool"
                    )
                    rows.append(
                        {
                            "dataset": cfg["dataset_name"],
                            "proxy_mode": (
                                "external_model_proxy"
                                if cfg.get("proxy_csv")
                                else "heldout_human_proxy"
                            ),
                            "group_key": path.group_key,
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": d,
                            "method": method,
                            "information_setting": information_setting,
                            "actual_gain": actual,
                            "predicted_gain": predicted,
                            "residual": actual - predicted,
                            "abs_error": abs(actual - predicted),
                            "n_cal": int(len(cal_idx)),
                            "n_eval": int(len(eval_idx)),
                            "cal_max_reachable_d": models["max_d"],
                            "curve_fit_max_d": models["fit_max_d"],
                            "direct_spline_n_basis": models[
                                "direct_meta"
                            ]["n_basis"],
                            "direct_spline_alpha": models[
                                "direct_meta"
                            ]["ridge_alpha"],
                            "residual_spline_n_basis": models[
                                "residual_meta"
                            ]["n_basis"],
                            "residual_spline_alpha": models[
                                "residual_meta"
                            ]["ridge_alpha"],
                            "A_cal": models["coefficient"].a_exact,
                            "B_cal": models["coefficient"].b_theory,
                            "A_eval_diagnostic": eval_coefficient.a_exact,
                            "B_eval_diagnostic": eval_coefficient.b_theory,
                        }
                    )
    return rows, skipped


def run_config(cfg: dict, n_jobs: int):
    paths = list(exp21.calib.build_paths(exp21._make_args(cfg)))
    rng = np.random.default_rng(exp21.RANDOM_SEED)
    tasks = []
    for path in paths:
        split_plan = [
            exp21.calib.split_indices(
                len(path.item_ids), exp21.CALIBRATION_FRAC, rng
            )
            for _ in range(exp21.N_REPEATS)
        ]
        tasks.append((cfg, path, split_plan))

    if n_jobs == 1:
        results = [run_path_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context("fork"),
        ) as executor:
            results = list(executor.map(run_path_task, tasks, chunksize=1))
    rows = [row for path_rows, _ in results for row in path_rows]
    skipped = [row for _, path_skips in results for row in path_skips]
    return pd.DataFrame(rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[cfg["config_key"] for cfg in exp21.CONFIGS],
        default=[cfg["config_key"] for cfg in exp21.CONFIGS],
    )
    parser.add_argument(
        "--jobs", type=int, default=min(32, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    args.outdir.mkdir(parents=True, exist_ok=False)

    selected_keys = set(args.datasets)
    selected = [
        cfg for cfg in exp21.CONFIGS
        if cfg["config_key"] in selected_keys
    ]
    frames: list[pd.DataFrame] = []
    skipped: list[dict] = []
    for cfg in selected:
        print(f"=== {cfg['dataset_name']} ===", flush=True)
        frame, cfg_skips = run_config(cfg, args.jobs)
        print(
            f"rows={len(frame)} paths="
            f"{frame['path_id'].nunique() if len(frame) else 0} "
            f"skipped={len(cfg_skips)}",
            flush=True,
        )
        frames.append(frame)
        skipped.extend(cfg_skips)

    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.outdir / "skipped.csv", index=False)

    input_records = []
    for cfg in selected:
        record = {
            "dataset": cfg["dataset_name"],
            "human_long_csv": exp21._portable_path(cfg["human_long_csv"]),
            "human_long_sha256": exp21._sha256(cfg["human_long_csv"]),
            "proxy_csv": None,
            "proxy_sha256": None,
        }
        if cfg.get("proxy_csv"):
            record["proxy_csv"] = exp21._portable_path(cfg["proxy_csv"])
            record["proxy_sha256"] = exp21._sha256(cfg["proxy_csv"])
        input_records.append(record)

    run_spec = {
        "experiment": "exp36_dominant_local_law",
        "purpose": (
            "Test whether flexible regressions in radius d materially improve "
            "held-out gain prediction beyond the moment-derived local law."
        ),
        "information_settings": {
            "prospective": (
                "30% calibration proxy and gold plus requested d only; no "
                "evaluation proxy or gold enters prediction"
            ),
            "eval_oracle_A+B": (
                "evaluation gold and proxy; retrospective truncation diagnostic "
                "only, never a predictor"
            ),
        },
        "datasets": input_records,
        "calibration_fraction": exp21.CALIBRATION_FRAC,
        "n_repeats": exp21.N_REPEATS,
        "random_seed": exp21.RANDOM_SEED,
        "d_targets": D_TARGETS,
        "curve_fit_grid": FIT_D_GRID.tolist(),
        "curve_fit_max_d": exp21.FIT_MAX_D,
        "curve_fit_evaluation": (
            "independent exact KL root solve at every fit-grid radius; "
            "no interpolation from a t-grid"
        ),
        "methods": list(ALL_METHODS),
        "model_complexity": {
            "A+B_moment": "2 moment-derived coefficients; no curve fit",
            "A-only_moment": "1 moment-derived coefficient; no curve fit",
            "calibration_curve": (
                "exact calibration gain transferred at each requested radius"
            ),
            "empirical_two_term": "2 fitted coefficients in sqrt(d), d",
            "empirical_cubic": (
                "3 fitted coefficients in sqrt(d), d, d^(3/2)"
            ),
            "direct_spline": (
                f"degree-{SPLINE_DEGREE} spline with "
                f"{SPLINE_N_KNOTS} knots and RidgeCV smoothing"
            ),
            "A+B_residual_spline": (
                "moment A+B plus the same spline fitted to calibration residuals"
            ),
        },
        "spline_alphas": SPLINE_ALPHAS.tolist(),
        "n_jobs": args.jobs,
        "parallelization_unit": "path",
        "source_sha256": {
            "build.py": sha256(HERE / "build.py"),
            "audit.py": sha256(HERE / "audit.py"),
            "data_chain_audit.py": sha256(HERE / "data_chain_audit.py"),
            "run.sh": sha256(HERE / "run.sh"),
            "exp21_builder": sha256(EXP21_PATH),
            "canonical_engine": sha256(exp21.CANON_PATH),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
