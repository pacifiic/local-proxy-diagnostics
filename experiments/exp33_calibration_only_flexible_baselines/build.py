#!/usr/bin/env python3
"""Strict calibration-only flexible baselines on the exp21 protocol."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP21_PATH = (
    ROOT / "experiments" / "exp21_direct_calibration"
    / "build_direct_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("exp21_builder", EXP21_PATH)
exp21 = importlib.util.module_from_spec(SPEC)
sys.modules["exp21_builder"] = exp21
SPEC.loader.exec_module(exp21)

STRICT_NAME = {
    "isotonic_calibration": "isotonic_calibration_only",
    "binned_shrinkage": "binned_calibration_only",
    "spline_ridge": "spline_calibration_only",
    "mlp_calibration": "mlp_calibration_only",
}
SHARED_BASELINES = [
    "A+B",
    "A-only",
    "A_nu",
    "pearson_corr",
    "spearman_corr",
    "empirical_curve",
    "empirical_fit",
    "zero",
]
STRICT_BASELINES = list(STRICT_NAME.values())
ALL_BASELINES = SHARED_BASELINES + STRICT_BASELINES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_only_predictions(
    models: dict,
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    d: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Predict using calibration values only; evaluation arrays are unavailable."""
    coeff = models["coeff"]
    sqrt_d = math.sqrt(max(0.0, d))
    predictions = {
        "A+B": coeff.a_exact * sqrt_d + coeff.b_theory * d,
        "A-only": coeff.a_exact * sqrt_d,
        "A_nu": coeff.a_nu * sqrt_d,
        "pearson_corr": (
            math.sqrt(2.0)
            * models["sigma_j"]
            * models["rho_pearson"]
            * sqrt_d
        ),
        "spearman_corr": (
            math.sqrt(2.0)
            * models["sigma_j"]
            * models["rho_spearman"]
            * sqrt_d
        ),
        "zero": 0.0,
    }
    errors: dict[str, str] = {}

    fit_coeff = models["fit_coeff"]
    if fit_coeff is not None:
        predictions["empirical_fit"] = (
            fit_coeff.a_fit * sqrt_d + fit_coeff.b_fit * d
        )
    else:
        errors["empirical_fit"] = models["fit_errors"].get(
            "empirical_fit", "empirical fit unavailable"
        )

    if d < models["cal_max_d"] - 1e-10:
        cal_weights = exp21.calib.exact_weights_at_d(r_cal, d)
        predictions["empirical_curve"] = float(
            np.sum(cal_weights * j_cal) - np.mean(j_cal)
        )
        for old_name, values in models["h_eval"].items():
            predictions[STRICT_NAME[old_name]] = float(
                np.sum(cal_weights * values) - np.mean(values)
            )
    else:
        reason = (
            f"calibration frontier max KL={models['cal_max_d']:.6g} "
            f"cannot reach d={d:g}"
        )
        errors["empirical_curve"] = reason
        for strict_name in STRICT_BASELINES:
            errors[strict_name] = reason

    for old_name, reason in models["fit_errors"].items():
        if old_name in STRICT_NAME:
            errors[STRICT_NAME[old_name]] = reason
    return predictions, errors


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
                # Passing r_cal in the third argument makes every fitted h value
                # a calibration-pool value. No evaluation proxy enters prediction.
                models = exp21.fit_split_models(
                    r_cal, j_cal, r_cal, exp21.FIT_MAX_D
                )
            except Exception as exc:  # noqa: BLE001
                for d in exp21.D_TARGETS:
                    skipped.append(
                        {
                            "dataset": cfg["dataset_name"],
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": d,
                            "baseline": "ALL",
                            "reason": (
                                f"calibration_fit_failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                continue

            for d in exp21.D_TARGETS:
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
                            "baseline": "ALL",
                            "reason": (
                                f"evaluation_unreachable: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                    continue

                predictions, errors = calibration_only_predictions(
                    models, r_cal, j_cal, float(d)
                )
                for baseline, reason in errors.items():
                    skipped.append(
                        {
                            "dataset": cfg["dataset_name"],
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": d,
                            "baseline": baseline,
                            "reason": reason,
                        }
                    )
                for baseline in ALL_BASELINES:
                    if baseline not in predictions:
                        continue
                    predicted = float(predictions[baseline])
                    if not np.isfinite(predicted):
                        raise RuntimeError(
                            f"non-finite prediction for {baseline}"
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
                            "d": float(d),
                            "baseline": baseline,
                            "information_setting": "calibration_only",
                            "actual_gain": actual,
                            "predicted_gain": predicted,
                            "abs_error": abs(predicted - actual),
                            "n_cal": int(len(cal_idx)),
                            "n_eval": int(len(eval_idx)),
                            "cal_max_reachable_d": models["cal_max_d"],
                            "empirical_fit_max_d": models[
                                "empirical_fit_max_d"
                            ],
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
            max_workers=n_jobs, mp_context=mp.get_context("fork")
        ) as executor:
            results = list(executor.map(run_path_task, tasks, chunksize=1))
    rows = [row for result, _ in results for row in result]
    skips = [skip for _, result in results for skip in result]
    return pd.DataFrame(rows), skips


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

    selected = [
        cfg for cfg in exp21.CONFIGS
        if cfg["config_key"] in set(args.datasets)
    ]
    frames = []
    skipped = []
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

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        args.outdir / "calibration_only_predictions.csv", index=False
    )
    pd.DataFrame(skipped).to_csv(
        args.outdir / "calibration_only_skipped.csv", index=False
    )

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

    spec = {
        "experiment": "exp33_calibration_only_flexible_baselines",
        "information_setting": {
            "prediction_inputs": (
                "calibration proxy and gold values plus requested radius d"
            ),
            "evaluation_proxy": "used only to compute realized held-out gain",
            "evaluation_gold": "used only to compute realized held-out gain",
            "target_pool_available_at_prediction_time": False,
        },
        "datasets": input_records,
        "calibration_fraction": exp21.CALIBRATION_FRAC,
        "n_repeats": exp21.N_REPEATS,
        "d_targets": exp21.D_TARGETS,
        "fit_max_d": exp21.FIT_MAX_D,
        "random_seed": exp21.RANDOM_SEED,
        "n_jobs": args.jobs,
        "baselines": ALL_BASELINES,
        "source_sha256": {
            "build.py": sha256(HERE / "build.py"),
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
        json.dumps(spec, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
