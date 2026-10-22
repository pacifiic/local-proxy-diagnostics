#!/usr/bin/env python3
"""Add calibration-only binned baselines to the fixed-pool DPO audit.

The fit uses only calibration prompts. Bin count and shrinkage are selected by
five-fold GroupKFold, keeping all candidates from a prompt in the same fold.
Existing experiment outputs are read for comparison and are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BUILDER_PATH = HERE / "build_prompt_conditional_calibration.py"
_spec = importlib.util.spec_from_file_location("prompt_calibration", BUILDER_PATH)
prompt_calibration = importlib.util.module_from_spec(_spec)
sys.modules["prompt_calibration"] = prompt_calibration
_spec.loader.exec_module(prompt_calibration)  # type: ignore[union-attr]

MODES = ("raw", "centered", "zscore", "rank")
BIN_CANDIDATES = (3, 5, 8, 12)
SHRINK_K_CANDIDATES = (0.0, 5.0, 20.0)
N_FOLDS = 5
COMPARISON_MODELS = (
    "A+B",
    "A-only",
    "A_nu",
    "Zero",
    "cal_raw_isotonic",
    "cal_rank_isotonic",
    "cal_raw_spline",
    "cal_rank_spline",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_bins(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
    shrink_k: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.unique(
        np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    )
    if len(edges) < 3:
        midpoint = float(np.median(x))
        edges = np.array(
            [float(np.min(x)) - 1e-12, midpoint, float(np.max(x)) + 1e-12]
        )
    inner_edges = edges[1:-1]
    bin_ids = np.digitize(x, inner_edges)
    n_actual_bins = len(edges) - 1
    global_mean = float(np.mean(y))
    values = np.full(n_actual_bins, global_mean, dtype=float)

    for bin_id in range(n_actual_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        local_mean = float(np.mean(y[mask]))
        values[bin_id] = (
            count * local_mean + shrink_k * global_mean
        ) / (count + shrink_k)
    return inner_edges, values, global_mean


def predict_bins(
    x: np.ndarray,
    inner_edges: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    bin_ids = np.digitize(np.asarray(x, dtype=float), inner_edges)
    bin_ids = np.clip(bin_ids, 0, len(values) - 1)
    return values[bin_ids]


def transform_proxy(proxy: np.ndarray, mode: str) -> np.ndarray:
    """Transform evaluation proxy scores without accessing evaluation gold."""
    values = np.asarray(proxy, dtype=float)
    if mode == "raw":
        return values
    if mode == "centered":
        return values - np.mean(values)
    if mode == "zscore":
        return (values - np.mean(values)) / max(
            float(np.std(values)),
            1e-12,
        )
    if mode == "rank":
        return (rankdata(values, method="average") - 0.5) / len(values)
    raise ValueError(f"unknown mode: {mode}")


def fit_grouped_binned(
    x: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_ids = np.asarray(group_ids)
    if len(np.unique(group_ids)) < N_FOLDS:
        raise ValueError("not enough calibration prompts for grouped CV")

    folds = GroupKFold(n_splits=N_FOLDS)
    candidates = []
    for n_bins in BIN_CANDIDATES:
        for shrink_k in SHRINK_K_CANDIDATES:
            fold_mses = []
            for train_idx, validation_idx in folds.split(
                x, y, groups=group_ids
            ):
                edges, values, _ = fit_bins(
                    x[train_idx],
                    y[train_idx],
                    n_bins,
                    shrink_k,
                )
                prediction = predict_bins(x[validation_idx], edges, values)
                fold_mses.append(
                    float(np.mean((prediction - y[validation_idx]) ** 2))
                )
            candidates.append(
                (float(np.mean(fold_mses)), n_bins, shrink_k)
            )

    cv_mse, n_bins, shrink_k = min(
        candidates,
        key=lambda row: (row[0], row[1], row[2]),
    )
    edges, values, _ = fit_bins(x, y, n_bins, shrink_k)

    def predict(new_x: np.ndarray) -> np.ndarray:
        return predict_bins(new_x, edges, values)

    return predict, {
        "n_bins_requested": int(n_bins),
        "n_bins_actual": int(len(values)),
        "shrink_k": float(shrink_k),
        "cv_mse": float(cv_mse),
        "cv_scheme": "5-fold GroupKFold by prompt",
    }


def fit_models(groups):
    prompt_ids = sorted(
        prompt_id
        for split, prompt_id in groups
        if split == "calibration"
    )
    models = {}
    metadata = []
    ranges = {}
    for mode in MODES:
        x, y, group_ids = prompt_calibration.pooled_xy(
            groups,
            prompt_ids,
            "calibration",
            mode,
        )
        predict, meta = fit_grouped_binned(x, y, group_ids)
        name = f"cal_{mode}_binned"
        models[name] = (mode, predict)
        ranges[mode] = (float(np.min(x)), float(np.max(x)))
        metadata.append(
            {
                "model": name,
                "fit_split": "calibration",
                "n_prompts": len(prompt_ids),
                "n_candidates": len(x),
                **meta,
            }
        )
    return models, pd.DataFrame(metadata), ranges


def evaluate(groups, models, ranges, multiseed_root: Path, seeds: list[int]):
    eval_cache = {}
    for split, prompt_id in groups:
        if split != "eval":
            continue
        proxy, gold = prompt_calibration.arrays(groups[(split, prompt_id)])
        transformed = {
            mode: transform_proxy(proxy, mode)
            for mode in MODES
        }
        eval_cache[prompt_id] = {
            "proxy": proxy,
            "gold": gold,
            "transformed": transformed,
            "predictions": {
                name: predict(transformed[mode])
                for name, (mode, predict) in models.items()
            },
        }

    rows = []
    for seed in seeds:
        source_path = (
            multiseed_root / f"seed_{seed}" / "exp09_final_per_prompt.csv"
        )
        source = pd.read_csv(source_path)
        local_tags = (
            source.groupby("tag")
            .agg(d_med=("d", "median"), ess_med=("ess", "median"))
            .query("d_med <= 1.0 and ess_med >= 8.0")
            .index
        )
        source = source[source["tag"].isin(local_tags)].copy()
        if len(local_tags) != 6 or len(source) != 1200:
            raise ValueError(f"seed {seed}: expected 6 tags x 200 prompts")

        for row in source.itertuples():
            cache = eval_cache[row.prompt_id]
            proxy = cache["proxy"]
            gold = cache["gold"]
            temperature = prompt_calibration.dpo_spline.find_t(
                proxy,
                max(float(row.d), 0.0),
            )
            weights = prompt_calibration.dpo_spline.tilt_weights(
                proxy,
                temperature,
            )
            frontier = float(np.sum(weights * gold) - np.mean(gold))
            if not math.isclose(
                frontier,
                float(row.frontier),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"frontier mismatch: seed={seed}, tag={row.tag}, "
                    f"prompt={row.prompt_id}"
                )

            for name, fitted_gold in cache["predictions"].items():
                mode = name.removeprefix("cal_").removesuffix("_binned")
                low, high = ranges[mode]
                transformed_proxy = cache["transformed"][mode]
                prediction = float(
                    np.sum(weights * fitted_gold) - np.mean(fitted_gold)
                )
                outside_mass = float(
                    np.sum(
                        weights[
                            (transformed_proxy < low)
                            | (transformed_proxy > high)
                        ]
                    )
                )
                rows.append(
                    {
                        "seed": seed,
                        "tag": row.tag,
                        "prompt_id": row.prompt_id,
                        "d": float(row.d),
                        "ess": float(row.ess),
                        "frontier": frontier,
                        "dpo_realized": float(row.dpo_reweight_gold),
                        "model": name,
                        "prediction": prediction,
                        "fit_scope": "calibration_only",
                        "outside_calibration_mass": outside_mass,
                    }
                )

    result = pd.DataFrame(rows)
    expected_rows = len(seeds) * 6 * 200 * len(MODES)
    if len(result) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(result)}")
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("non-finite values in binned predictions")
    result["abs_error_frontier"] = np.abs(
        result["prediction"] - result["frontier"]
    )
    result["abs_error_dpo"] = np.abs(
        result["prediction"] - result["dpo_realized"]
    )
    return result


def aggregate(result: pd.DataFrame):
    checkpoint = (
        result.groupby(["seed", "tag", "model", "fit_scope"], as_index=False)
        .agg(
            d_med=("d", "median"),
            ess_med=("ess", "median"),
            prediction=("prediction", "mean"),
            frontier=("frontier", "mean"),
            dpo_realized=("dpo_realized", "mean"),
            outside_calibration_mass=("outside_calibration_mass", "mean"),
            n_prompts=("prompt_id", "nunique"),
        )
    )
    checkpoint["abs_error_frontier"] = np.abs(
        checkpoint["prediction"] - checkpoint["frontier"]
    )
    checkpoint["abs_error_dpo"] = np.abs(
        checkpoint["prediction"] - checkpoint["dpo_realized"]
    )

    scoped_seed_frames = []
    for scope, max_d in (
        ("strict_local_d_le_0p10", 0.10),
        ("historical_six_checkpoints", np.inf),
    ):
        scoped = checkpoint[checkpoint["d_med"] <= max_d]
        seed_level = (
            scoped.groupby(["seed", "model", "fit_scope"], as_index=False)
            .agg(
                mae_frontier=("abs_error_frontier", "mean"),
                mae_dpo=("abs_error_dpo", "mean"),
                mean_outside_calibration_mass=(
                    "outside_calibration_mass",
                    "mean",
                ),
                n_checkpoints=("tag", "nunique"),
            )
        )
        seed_level.insert(0, "scope", scope)
        scoped_seed_frames.append(seed_level)
    scoped_seed = pd.concat(scoped_seed_frames, ignore_index=True)
    summary = (
        scoped_seed.groupby(
            ["scope", "model", "fit_scope"],
            as_index=False,
        )
        .agg(
            mean_mae_frontier=("mae_frontier", "mean"),
            sd_mae_frontier=("mae_frontier", "std"),
            mean_mae_dpo=("mae_dpo", "mean"),
            sd_mae_dpo=("mae_dpo", "std"),
            mean_outside_calibration_mass=(
                "mean_outside_calibration_mass",
                "mean",
            ),
            min_checkpoints_per_seed=("n_checkpoints", "min"),
            max_checkpoints_per_seed=("n_checkpoints", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["scope", "mean_mae_dpo"])
    )
    return checkpoint, scoped_seed, summary


def build_comparison(
    binned_summary: pd.DataFrame,
    existing_exp28_dir: Path,
) -> pd.DataFrame:
    existing = pd.read_csv(existing_exp28_dir / "window_summary_mean_sd.csv")
    existing = existing[existing["model"].isin(COMPARISON_MODELS)].copy()
    existing["mean_outside_calibration_mass"] = np.nan
    columns = [
        "scope",
        "model",
        "fit_scope",
        "mean_mae_frontier",
        "sd_mae_frontier",
        "mean_mae_dpo",
        "sd_mae_dpo",
        "mean_outside_calibration_mass",
        "min_checkpoints_per_seed",
        "max_checkpoints_per_seed",
        "n_seeds",
    ]
    combined = pd.concat(
        [existing[columns], binned_summary[columns]],
        ignore_index=True,
    )
    return combined.sort_values(["scope", "mean_mae_dpo"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--pool-scored", type=Path, required=True)
    parser.add_argument("--existing-exp28-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups = prompt_calibration.load_pool(args.pool_scored)
    models, metadata, ranges = fit_models(groups)
    result = evaluate(
        groups,
        models,
        ranges,
        args.multiseed_root,
        args.seeds,
    )
    checkpoint, seed_level, summary = aggregate(result)
    comparison = build_comparison(summary, args.existing_exp28_dir)

    metadata.to_csv(args.outdir / "model_fit_metadata.csv", index=False)
    result.to_csv(args.outdir / "per_prompt_predictions.csv", index=False)
    checkpoint.to_csv(
        args.outdir / "checkpoint_predictions.csv",
        index=False,
    )
    seed_level.to_csv(args.outdir / "seed_level_mae.csv", index=False)
    summary.to_csv(args.outdir / "summary_mean_sd.csv", index=False)
    comparison.to_csv(args.outdir / "comparison_summary.csv", index=False)

    run_spec = {
        "seeds": args.seeds,
        "modes": MODES,
        "family": "equal-frequency binned calibration with shrinkage",
        "bin_candidates": BIN_CANDIDATES,
        "shrink_k_candidates": SHRINK_K_CANDIDATES,
        "cv": "5-fold GroupKFold by calibration prompt",
        "fit_scope": "calibration_only",
        "strict_local_window": "checkpoint median d <= 0.10",
        "historical_scope": "six checkpoints with d <= 1 and ESS >= 8",
        "input_sha256": {
            "pool_scored": sha256_file(args.pool_scored),
            "existing_exp28_window_summary": sha256_file(
                args.existing_exp28_dir / "window_summary_mean_sd.csv"
            ),
            **{
                f"seed_{seed}_per_prompt": sha256_file(
                    args.multiseed_root
                    / f"seed_{seed}"
                    / "exp09_final_per_prompt.csv"
                )
                for seed in args.seeds
            },
        },
        "code_sha256": {
            "binned_followup": sha256_file(Path(__file__).resolve()),
            "exp28_builder": sha256_file(BUILDER_PATH),
        },
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2),
        encoding="utf-8",
    )
    (args.outdir / "WORKFLOW_COMPLETE").write_text(
        "PASS\n",
        encoding="utf-8",
    )
    print(metadata.to_string(index=False))
    print("\nComparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
