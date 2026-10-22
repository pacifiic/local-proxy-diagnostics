#!/usr/bin/env python3
"""Cross-dataset transfer of A/B and direct h(r) calibrators.

Each source path/repeat is standardized from its source calibration split.
The source prior is either the mean A/B pair or the mean fitted h(z) function.
On the target, both proxy and gold are standardized using target-calibration
statistics (the paper's z_cal convention). Predictions blend source and target
estimators with the same target weight lambda. Results are standardized gains,
not raw-score gains.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP21_PATH = (
    ROOT
    / "experiments"
    / "exp21_direct_calibration"
    / "build_direct_calibration.py"
)
_spec = importlib.util.spec_from_file_location("exp21_ood_engine", EXP21_PATH)
exp21 = importlib.util.module_from_spec(_spec)
sys.modules["exp21_ood_engine"] = exp21
_spec.loader.exec_module(exp21)  # type: ignore[union-attr]

SOURCE_CALIBRATION_FRAC = 0.30
TARGET_CALIBRATION_FRACS = [0.05, 0.10, 0.20, 0.30]
D_TARGETS = [0.01, 0.025, 0.05, 0.075, 0.10]
LAMBDAS = [0.0, 0.25, 0.50, 0.75, 1.0]
N_REPEATS = 50
# Wide enough that target z-scores from small calibration splits do not hit an
# arbitrary interpolation boundary. The fitted isotonic/spline models still use
# their declared constant extrapolation beyond observed source support.
Z_GRID = np.linspace(-30.0, 30.0, 6001)
METHODS = ["A+B", "isotonic_calibration", "spline_ridge"]
ROUTES = [
    ("summeval", "wmt"),
    ("wmt", "summeval"),
    ("summeval", "openmeva"),
]


def safe_std(values: np.ndarray) -> float:
    value = float(np.std(np.asarray(values, dtype=float)))
    return value if value > 1e-12 else 1.0


def standardize_cal_eval(
    cal: np.ndarray, eva: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(cal))
    std = safe_std(cal)
    return (cal - mean) / std, (eva - mean) / std


def source_path_task(task):
    path, plans = task
    rows = []
    iso_sum = np.zeros_like(Z_GRID)
    spline_sum = np.zeros_like(Z_GRID)
    n_curves = 0
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:
        from contextlib import nullcontext

        limiter = nullcontext()
    with limiter:
        for repeat, (cal_idx, _) in enumerate(plans):
            r = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j = np.asarray(path.gold_scores[cal_idx], dtype=float)
            r = (r - float(np.mean(r))) / safe_std(r)
            j = (j - float(np.mean(j))) / safe_std(j)
            coeff = exp21.calib.theory_coefficients(j, r)
            iso = exp21.calib.fit_isotonic(r, j)
            spline, spline_meta = exp21.fit_spline_ridge(r, j)
            iso_values = np.asarray(iso.predict(Z_GRID), dtype=float)
            spline_values = np.asarray(spline(Z_GRID), dtype=float)
            # Gain is invariant to an additive constant. Center each source
            # function before averaging so path-specific intercepts cannot
            # dominate the ensemble numerically.
            zero_idx = len(Z_GRID) // 2
            iso_sum += iso_values - iso_values[zero_idx]
            spline_sum += spline_values - spline_values[zero_idx]
            n_curves += 1
            rows.append(
                {
                    "path_id": path.path_id,
                    "repeat": repeat,
                    "a_exact": coeff.a_exact,
                    "b_theory": coeff.b_theory,
                    "spline_n_knots": spline_meta["n_knots"],
                    "spline_ridge_alpha": spline_meta["ridge_alpha"],
                    "n_cal": int(len(cal_idx)),
                }
            )
    return rows, iso_sum, spline_sum, n_curves


def build_source_profile(cfg: dict, n_jobs: int) -> tuple[dict, pd.DataFrame]:
    paths = list(exp21.calib.build_paths(exp21._make_args(cfg)))
    rng = np.random.default_rng(exp21.RANDOM_SEED)
    tasks = []
    for path in paths:
        plans = [
            exp21.calib.split_indices(
                len(path.item_ids), SOURCE_CALIBRATION_FRAC, rng
            )
            for _ in range(N_REPEATS)
        ]
        tasks.append((path, plans))
    if n_jobs == 1:
        results = [source_path_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=mp.get_context("fork")
        ) as executor:
            results = list(executor.map(source_path_task, tasks, chunksize=1))
    rows = [row for path_rows, _, _, _ in results for row in path_rows]
    n_fits = sum(n_curves for _, _, _, n_curves in results)
    iso = sum((curve_sum for _, curve_sum, _, _ in results), np.zeros_like(Z_GRID))
    spline = sum(
        (curve_sum for _, _, curve_sum, _ in results),
        np.zeros_like(Z_GRID),
    )
    coeffs = pd.DataFrame(rows)
    profile = {
        "source_name": cfg["config_key"],
        "a_exact": float(coeffs["a_exact"].mean()),
        "b_theory": float(coeffs["b_theory"].mean()),
        "iso_grid": iso / n_fits,
        "spline_grid": spline / n_fits,
        "n_paths": len(paths),
        "n_fits": n_fits,
    }
    return profile, coeffs


def gain_from_values(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values) - np.mean(values))


def target_path_task(task):
    route_name, path, frac_plans, source = task
    rows = []
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
    except ImportError:
        from contextlib import nullcontext

        limiter = nullcontext()
    with limiter:
        for frac in TARGET_CALIBRATION_FRACS:
            for repeat, (cal_idx, eval_idx) in enumerate(frac_plans[frac]):
                r_cal_raw = np.asarray(
                    path.proxy_scores[cal_idx], dtype=float
                )
                r_eval_raw = np.asarray(
                    path.proxy_scores[eval_idx], dtype=float
                )
                j_cal_raw = np.asarray(
                    path.gold_scores[cal_idx], dtype=float
                )
                j_eval_raw = np.asarray(
                    path.gold_scores[eval_idx], dtype=float
                )
                r_cal, r_eval = standardize_cal_eval(
                    r_cal_raw, r_eval_raw
                )
                j_cal, j_eval = standardize_cal_eval(
                    j_cal_raw, j_eval_raw
                )
                coeff = exp21.calib.theory_coefficients(j_cal, r_cal)
                iso = exp21.calib.fit_isotonic(r_cal, j_cal)
                spline, _ = exp21.fit_spline_ridge(r_cal, j_cal)
                h_target_iso = np.asarray(iso.predict(r_eval), dtype=float)
                h_target_spline = np.asarray(spline(r_eval), dtype=float)
                h_source_iso = np.interp(
                    r_eval,
                    Z_GRID,
                    source["iso_grid"],
                    left=source["iso_grid"][0],
                    right=source["iso_grid"][-1],
                )
                h_source_spline = np.interp(
                    r_eval,
                    Z_GRID,
                    source["spline_grid"],
                    left=source["spline_grid"][0],
                    right=source["spline_grid"][-1],
                )

                for d in D_TARGETS:
                    try:
                        weights = exp21.calib.exact_weights_at_d(r_eval, d)
                    except (RuntimeError, ValueError):
                        continue
                    actual = gain_from_values(j_eval, weights)
                    source_iso_gain = gain_from_values(
                        h_source_iso, weights
                    )
                    target_iso_gain = gain_from_values(
                        h_target_iso, weights
                    )
                    source_spline_gain = gain_from_values(
                        h_source_spline, weights
                    )
                    target_spline_gain = gain_from_values(
                        h_target_spline, weights
                    )
                    for target_weight in LAMBDAS:
                        source_weight = 1.0 - target_weight
                        predictions = {
                            "A+B": (
                                (
                                    source_weight * source["a_exact"]
                                    + target_weight * coeff.a_exact
                                )
                                * np.sqrt(d)
                                + (
                                    source_weight * source["b_theory"]
                                    + target_weight * coeff.b_theory
                                )
                                * d
                            ),
                            "isotonic_calibration": (
                                source_weight * source_iso_gain
                                + target_weight * target_iso_gain
                            ),
                            "spline_ridge": (
                                source_weight * source_spline_gain
                                + target_weight * target_spline_gain
                            ),
                        }
                        for method, prediction in predictions.items():
                            rows.append(
                                {
                                    "route": route_name,
                                    "target_dataset": path.meta.get(
                                        "dataset_name", ""
                                    ),
                                    "path_id": path.path_id,
                                    "repeat": repeat,
                                    "calibration_frac": frac,
                                    "d": d,
                                    "target_weight": target_weight,
                                    "method": method,
                                    "actual_gain_z": actual,
                                    "predicted_gain_z": prediction,
                                    "abs_error_z": abs(
                                        prediction - actual
                                    ),
                                    "n_cal": int(len(cal_idx)),
                                    "n_eval": int(len(eval_idx)),
                                    "eval_proxy_outside_source_grid": float(
                                        np.mean(
                                            (r_eval < Z_GRID[0])
                                            | (r_eval > Z_GRID[-1])
                                        )
                                    ),
                                }
                            )
    return rows


def target_plans(paths) -> dict:
    plans = {path.path_id: {} for path in paths}
    for frac in TARGET_CALIBRATION_FRACS:
        rng = np.random.default_rng(exp21.RANDOM_SEED)
        for path in paths:
            plans[path.path_id][frac] = [
                exp21.calib.split_indices(len(path.item_ids), frac, rng)
                for _ in range(N_REPEATS)
            ]
    return plans


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--jobs", type=int, default=min(32, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output: {args.outdir}"
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    configs = {
        cfg["config_key"]: cfg
        for cfg in exp21.CONFIGS
        if cfg["config_key"] != "gpt4o"
    }
    source_names = sorted({source for source, _ in ROUTES})
    profiles = {}
    for source_name in source_names:
        print(f"=== source profile: {source_name} ===", flush=True)
        profile, coeffs = build_source_profile(
            configs[source_name], args.jobs
        )
        profiles[source_name] = profile
        coeffs.to_csv(
            args.outdir / f"source_coefficients_{source_name}.csv",
            index=False,
        )
        np.savez(
            args.outdir / f"source_profile_{source_name}.npz",
            z_grid=Z_GRID,
            iso_grid=profile["iso_grid"],
            spline_grid=profile["spline_grid"],
            a_exact=profile["a_exact"],
            b_theory=profile["b_theory"],
        )

    split_frames = []
    route_accounting = []
    cells_path = args.outdir / "ood_prediction_cells.csv.gz"
    keys = [
        "route",
        "path_id",
        "repeat",
        "calibration_frac",
        "target_weight",
        "method",
    ]
    with cells_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                write_header = True
                for source_name, target_name in ROUTES:
                    route_name = f"{source_name}_to_{target_name}"
                    print(f"=== route: {route_name} ===", flush=True)
                    target_cfg = configs[target_name]
                    paths = list(
                        exp21.calib.build_paths(
                            exp21._make_args(target_cfg)
                        )
                    )
                    plans = target_plans(paths)
                    tasks = [
                        (
                            route_name,
                            path,
                            plans[path.path_id],
                            profiles[source_name],
                        )
                        for path in paths
                    ]
                    if args.jobs == 1:
                        results = map(target_path_task, tasks)
                        executor = None
                    else:
                        executor = ProcessPoolExecutor(
                            max_workers=args.jobs,
                            mp_context=mp.get_context("fork"),
                        )
                        results = executor.map(
                            target_path_task, tasks, chunksize=1
                        )
                    route_cells = 0
                    try:
                        for path_rows in results:
                            frame = pd.DataFrame(path_rows)
                            route_cells += len(frame)
                            frame.to_csv(
                                text, index=False, header=write_header
                            )
                            text.flush()
                            write_header = False
                            split_frames.append(
                                frame.groupby(keys, dropna=False)
                                .agg(
                                    curve_mae_z=("abs_error_z", "mean"),
                                    n_radii=("d", "size"),
                                    n_cal=("n_cal", "first"),
                                    n_eval=("n_eval", "first"),
                                    source_grid_exceedance=(
                                        "eval_proxy_outside_source_grid",
                                        "max",
                                    ),
                                )
                                .reset_index()
                            )
                    finally:
                        if executor is not None:
                            executor.shutdown()
                    route_accounting.append(
                        {
                            "route": route_name,
                            "n_target_paths": len(paths),
                            "n_prediction_cells": route_cells,
                        }
                    )

    splits = pd.concat(split_frames, ignore_index=True)
    splits.to_csv(args.outdir / "ood_split_metrics.csv", index=False)
    pd.DataFrame(route_accounting).to_csv(
        args.outdir / "ood_accounting.csv", index=False
    )

    spec = {
        "experiment": "exp24_direct_calibration_followup_ood",
        "routes": [f"{s}_to_{t}" for s, t in ROUTES],
        "source_calibration_frac": SOURCE_CALIBRATION_FRAC,
        "target_calibration_fracs": TARGET_CALIBRATION_FRACS,
        "d_targets": D_TARGETS,
        "target_weights": LAMBDAS,
        "n_repeats": N_REPEATS,
        "random_seed": exp21.RANDOM_SEED,
        "standardization": (
            "z_cal: target proxy and gold use target-calibration mean/std; "
            "each source estimator uses its source-calibration mean/std"
        ),
        "source_function_prior": (
            "equal-weight mean of centered path/repeat h(z) functions"
        ),
        "source_coefficient_prior": (
            "equal-weight mean of path/repeat A and B estimates"
        ),
        "blend": (
            "target_weight * target estimator + "
            "(1-target_weight) * source estimator"
        ),
        "methods": METHODS,
        "units": "standardized target-gold gain",
        "jobs": args.jobs,
    }
    (args.outdir / "ood_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(
        f"-> {cells_path} "
        f"({sum(row['n_prediction_cells'] for row in route_accounting)} rows)"
    )
    print(
        f"-> {args.outdir / 'ood_split_metrics.csv'} "
        f"({len(splits)} rows)"
    )


if __name__ == "__main__":
    main()
