#!/usr/bin/env python3
"""Diagnose direct calibration under prompt-conditional DPO geometry.

The experiment separates two errors:

  prediction - realized DPO
    = (prediction - exact prompt-tilt frontier)
    + (exact prompt-tilt frontier - realized DPO).

Calibration-only isotonic and spline models are fit using raw, prompt-centered,
prompt-z-scored, or within-prompt rank proxy coordinates. Evaluation prompt
gold labels are used only for targets. A separate evaluation-prompt cross-fit
is reported as a descriptive distribution-shift diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SPLINE_PATH = (
    REPO_ROOT
    / "experiments"
    / "exp09_dpo_validation"
    / "19_spline_baseline.py"
)
_spec = importlib.util.spec_from_file_location("dpo_spline", SPLINE_PATH)
dpo_spline = importlib.util.module_from_spec(_spec)
sys.modules["dpo_spline"] = dpo_spline
_spec.loader.exec_module(dpo_spline)  # type: ignore[union-attr]

MODES = ("raw", "centered", "zscore", "rank")
FAMILIES = ("isotonic", "spline")
KNOT_CANDIDATES = (4, 6, 8)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(
    proxy: np.ndarray,
    gold: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(proxy, dtype=float)
    j = np.asarray(gold, dtype=float)
    if mode == "raw":
        return r, j
    j_centered = j - np.mean(j)
    if mode == "centered":
        return r - np.mean(r), j_centered
    if mode == "zscore":
        return (r - np.mean(r)) / max(float(np.std(r)), 1e-12), j_centered
    if mode == "rank":
        return (rankdata(r, method="average") - 0.5) / len(r), j_centered
    raise ValueError(f"unknown mode: {mode}")


def load_pool(path: Path):
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] not in {"calibration", "eval"}:
                continue
            key = (row["prompt_id"], row["split"], row["sample_idx"])
            if key in seen:
                raise ValueError(f"duplicate candidate key: {key}")
            seen.add(key)
            if not str(row["response"]).strip():
                raise ValueError(f"empty response: {key}")
            values = (float(row["proxy_score"]), float(row["gold_score"]))
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite score: {key}")
            groups[(row["split"], row["prompt_id"])].append(row)

    for split, expected_prompts in (("calibration", 100), ("eval", 200)):
        split_groups = {
            prompt_id: rows
            for (candidate_split, prompt_id), rows in groups.items()
            if candidate_split == split
        }
        if len(split_groups) != expected_prompts:
            raise ValueError(
                f"{split}: expected {expected_prompts} prompts, got "
                f"{len(split_groups)}"
            )
        if {len(rows) for rows in split_groups.values()} != {32}:
            raise ValueError(f"{split}: expected exactly 32 candidates/prompt")
    return groups


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: int(row["sample_idx"]))
    return (
        np.array([float(row["proxy_score"]) for row in ordered]),
        np.array([float(row["gold_score"]) for row in ordered]),
    )


def pooled_xy(groups, prompt_ids: list[str], split: str, mode: str):
    transformed = []
    group_ids = []
    for prompt_id in prompt_ids:
        pair = transform(*arrays(groups[(split, prompt_id)]), mode)
        transformed.append(pair)
        group_ids.extend([prompt_id] * len(pair[0]))
    return (
        np.concatenate([pair[0] for pair in transformed]),
        np.concatenate([pair[1] for pair in transformed]),
        np.asarray(group_ids),
    )


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    family: str,
):
    if family == "isotonic":
        model = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        ).fit(x, y)
        return (
            lambda new_x: model.predict(np.asarray(new_x, dtype=float)),
            {
                "n_thresholds": int(len(model.X_thresholds_)),
                "n_knots": np.nan,
                "ridge_alpha": np.nan,
                "cv_mse": np.nan,
            },
        )
    if family == "spline":
        x_2d = np.asarray(x, dtype=float).reshape(-1, 1)
        folds = GroupKFold(n_splits=5)
        best = None
        for n_knots in KNOT_CANDIDATES:
            for alpha in RIDGE_ALPHAS:
                candidate = make_pipeline(
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=3,
                        extrapolation="constant",
                    ),
                    Ridge(alpha=float(alpha)),
                )
                mse = float(
                    -np.mean(
                        cross_val_score(
                            candidate,
                            x_2d,
                            y,
                            groups=group_ids,
                            cv=folds,
                            scoring="neg_mean_squared_error",
                        )
                    )
                )
                if best is None or mse < best[0]:
                    best = (mse, n_knots, float(alpha))
        assert best is not None
        model = make_pipeline(
            SplineTransformer(
                n_knots=best[1],
                degree=3,
                extrapolation="constant",
            ),
            Ridge(alpha=best[2]),
        ).fit(x_2d, y)
        return (
            lambda new_x: model.predict(
                np.asarray(new_x, dtype=float).reshape(-1, 1)
            ),
            {
                "n_thresholds": np.nan,
                "n_knots": int(best[1]),
                "ridge_alpha": float(best[2]),
                "cv_mse": float(best[0]),
                "cv_scheme": "5-fold GroupKFold by prompt",
            },
        )
    raise ValueError(f"unknown family: {family}")


def fit_calibration_models(groups):
    prompt_ids = sorted(
        prompt_id
        for split, prompt_id in groups
        if split == "calibration"
    )
    models = {}
    metadata = []
    ranges = {}
    for mode in MODES:
        x, y, group_ids = pooled_xy(
            groups, prompt_ids, "calibration", mode
        )
        ranges[mode] = (float(np.min(x)), float(np.max(x)))
        for family in FAMILIES:
            predict, meta = fit_model(x, y, group_ids, family)
            name = f"cal_{mode}_{family}"
            models[name] = (mode, predict)
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


def fit_eval_crossfit_models(groups, n_folds: int = 5):
    """Fit on four evaluation-prompt folds and predict the held-out fold.

    These rows diagnose calibration-to-evaluation shift and are not deployable
    baselines because they use labels from other evaluation prompts.
    """
    prompt_ids = sorted(
        prompt_id for split, prompt_id in groups if split == "eval"
    )
    fold_of = {prompt_id: index % n_folds for index, prompt_id in enumerate(prompt_ids)}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    metadata = []
    for fold in range(n_folds):
        train_ids = [p for p in prompt_ids if fold_of[p] != fold]
        test_ids = [p for p in prompt_ids if fold_of[p] == fold]
        for mode in MODES:
            x, y, group_ids = pooled_xy(groups, train_ids, "eval", mode)
            for family in FAMILIES:
                predict, meta = fit_model(x, y, group_ids, family)
                name = f"eval_xfit_{mode}_{family}"
                for prompt_id in test_ids:
                    r, j = arrays(groups[("eval", prompt_id)])
                    transformed_proxy, _ = transform(r, j, mode)
                    predictions[(name, prompt_id)] = predict(transformed_proxy)
                metadata.append(
                    {
                        "model": name,
                        "fit_split": f"eval_crossfit_fold_{fold}",
                        "n_prompts": len(train_ids),
                        "n_candidates": len(x),
                        **meta,
                    }
                )
    return predictions, pd.DataFrame(metadata)


def prompt_geometry(groups) -> pd.DataFrame:
    rows = []
    for split in ("calibration", "eval"):
        prompt_rows = []
        all_r, all_j, centered_r, centered_j = [], [], [], []
        within_corr = []
        for candidate_split, prompt_id in groups:
            if candidate_split != split:
                continue
            r, j = arrays(groups[(split, prompt_id)])
            all_r.extend(r)
            all_j.extend(j)
            centered_r.extend(r - r.mean())
            centered_j.extend(j - j.mean())
            corr = np.corrcoef(r, j)[0, 1] if np.std(r) and np.std(j) else np.nan
            if np.isfinite(corr):
                within_corr.append(float(corr))
            prompt_rows.append(
                {
                    "prompt_id": prompt_id,
                    "proxy_mean": r.mean(),
                    "gold_mean": j.mean(),
                }
            )
        prompt_df = pd.DataFrame(prompt_rows)
        all_r = np.asarray(all_r)
        all_j = np.asarray(all_j)
        centered_r = np.asarray(centered_r)
        centered_j = np.asarray(centered_j)
        rows.append(
            {
                "split": split,
                "global_pearson": float(np.corrcoef(all_r, all_j)[0, 1]),
                "between_prompt_mean_pearson": float(
                    np.corrcoef(
                        prompt_df["proxy_mean"], prompt_df["gold_mean"]
                    )[0, 1]
                ),
                "within_centered_pearson": float(
                    np.corrcoef(centered_r, centered_j)[0, 1]
                ),
                "mean_within_prompt_pearson": float(np.mean(within_corr)),
                "median_within_prompt_pearson": float(np.median(within_corr)),
                "proxy_between_variance_fraction": float(
                    prompt_df["proxy_mean"].var(ddof=1)
                    / np.var(all_r, ddof=1)
                ),
                "gold_between_variance_fraction": float(
                    prompt_df["gold_mean"].var(ddof=1)
                    / np.var(all_j, ddof=1)
                ),
                "n_prompts": len(prompt_df),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    groups,
    calibration_models,
    eval_crossfit_predictions,
    calibration_ranges,
    multiseed_root: Path,
    seeds: list[int],
):
    eval_cache = {}
    for split, prompt_id in groups:
        if split != "eval":
            continue
        r, j = arrays(groups[(split, prompt_id)])
        transformed = {mode: transform(r, j, mode)[0] for mode in MODES}
        predictions = {
            name: predict(transformed[mode])
            for name, (mode, predict) in calibration_models.items()
        }
        for name in (
            f"eval_xfit_{mode}_{family}"
            for mode in MODES
            for family in FAMILIES
        ):
            predictions[name] = eval_crossfit_predictions[(name, prompt_id)]
        eval_cache[prompt_id] = {
            "proxy": r,
            "gold": j,
            "transformed": transformed,
            "predictions": predictions,
        }

    rows = []
    for seed in seeds:
        per_path = multiseed_root / f"seed_{seed}" / "exp09_final_per_prompt.csv"
        per = pd.read_csv(per_path)
        local_tags = (
            per.groupby("tag")
            .agg(d_med=("d", "median"), ess_med=("ess", "median"))
            .query("d_med <= 1.0 and ess_med >= 8.0")
            .index
        )
        per = per[per["tag"].isin(local_tags)].copy()
        if len(local_tags) != 6 or len(per) != 1200:
            raise ValueError(f"seed {seed}: expected 6 x 200 local rows")

        for row in per.itertuples():
            cache = eval_cache[row.prompt_id]
            r = cache["proxy"]
            j = cache["gold"]
            t = dpo_spline.find_t(r, max(float(row.d), 0.0))
            weights = dpo_spline.tilt_weights(r, t)
            frontier = float(np.sum(weights * j) - np.mean(j))
            if not math.isclose(
                frontier,
                float(row.frontier),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"frontier re-derivation mismatch for seed={seed}, "
                    f"tag={row.tag}, prompt={row.prompt_id}"
                )

            base = {
                "seed": seed,
                "tag": row.tag,
                "prompt_id": row.prompt_id,
                "d": float(row.d),
                "ess": float(row.ess),
                "frontier": frontier,
                "dpo_realized": float(row.dpo_reweight_gold),
            }
            existing = {
                "A+B": float(row.pred_AB),
                "A-only": float(row.pred_Aonly),
                "A_nu": float(row.pred_Anu),
                "Spearman": float(row.pred_spearman),
                "Zero": 0.0,
            }
            for name, prediction in existing.items():
                rows.append(
                    {
                        **base,
                        "model": name,
                        "prediction": prediction,
                        "fit_scope": "coefficient_or_fixed_baseline",
                        "outside_calibration_mass": np.nan,
                    }
                )
            for name, h in cache["predictions"].items():
                gain = float(np.sum(weights * h) - np.mean(h))
                if name.startswith("cal_"):
                    mode = name.removeprefix("cal_").rsplit("_", 1)[0]
                    low, high = calibration_ranges[mode]
                    transformed_proxy = cache["transformed"][mode]
                    outside_mass = float(
                        np.sum(
                            weights[
                                (transformed_proxy < low)
                                | (transformed_proxy > high)
                            ]
                        )
                    )
                    fit_scope = "calibration_only"
                else:
                    outside_mass = np.nan
                    fit_scope = "eval_prompt_crossfit_diagnostic"
                rows.append(
                    {
                        **base,
                        "model": name,
                        "prediction": gain,
                        "fit_scope": fit_scope,
                        "outside_calibration_mass": outside_mass,
                    }
                )
    result = pd.DataFrame(rows)
    result["signed_calibration_error"] = result["prediction"] - result["frontier"]
    result["mechanism_gap"] = result["frontier"] - result["dpo_realized"]
    result["signed_total_error"] = result["prediction"] - result["dpo_realized"]
    residual = (
        result["signed_total_error"]
        - result["signed_calibration_error"]
        - result["mechanism_gap"]
    )
    if float(np.max(np.abs(residual))) > 1e-12:
        raise ValueError("error decomposition identity failed")
    result["abs_error_frontier"] = np.abs(result["signed_calibration_error"])
    result["abs_error_dpo"] = np.abs(result["signed_total_error"])
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
    seed_level = (
        checkpoint.groupby(["seed", "model", "fit_scope"], as_index=False)
        .agg(
            mae_frontier=("abs_error_frontier", "mean"),
            mae_dpo=("abs_error_dpo", "mean"),
            mean_outside_calibration_mass=("outside_calibration_mass", "mean"),
            n_checkpoints=("tag", "nunique"),
        )
    )
    summary = (
        seed_level.groupby(["model", "fit_scope"], as_index=False)
        .agg(
            mean_mae_frontier=("mae_frontier", "mean"),
            sd_mae_frontier=("mae_frontier", "std"),
            mean_mae_dpo=("mae_dpo", "mean"),
            sd_mae_dpo=("mae_dpo", "std"),
            mean_outside_calibration_mass=(
                "mean_outside_calibration_mass",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_mae_dpo")
    )

    scoped_frames = []
    for scope, max_d in (
        ("strict_local_d_le_0p10", 0.10),
        ("historical_six_checkpoints", np.inf),
    ):
        scoped = checkpoint[checkpoint["d_med"] <= max_d].copy()
        per_seed = (
            scoped.groupby(
                ["seed", "model", "fit_scope"], as_index=False
            )
            .agg(
                mae_frontier=("abs_error_frontier", "mean"),
                mae_dpo=("abs_error_dpo", "mean"),
                n_checkpoints=("tag", "nunique"),
            )
        )
        scoped_summary = (
            per_seed.groupby(["model", "fit_scope"], as_index=False)
            .agg(
                mean_mae_frontier=("mae_frontier", "mean"),
                sd_mae_frontier=("mae_frontier", "std"),
                mean_mae_dpo=("mae_dpo", "mean"),
                sd_mae_dpo=("mae_dpo", "std"),
                min_checkpoints_per_seed=("n_checkpoints", "min"),
                max_checkpoints_per_seed=("n_checkpoints", "max"),
                n_seeds=("seed", "nunique"),
            )
        )
        scoped_summary.insert(0, "scope", scope)
        scoped_frames.append(scoped_summary)
    window_summary = pd.concat(scoped_frames, ignore_index=True)

    decomposition = checkpoint.copy()
    decomposition["signed_calibration_error"] = (
        decomposition["prediction"] - decomposition["frontier"]
    )
    decomposition["mechanism_gap"] = (
        decomposition["frontier"] - decomposition["dpo_realized"]
    )
    decomposition["signed_total_error"] = (
        decomposition["prediction"] - decomposition["dpo_realized"]
    )
    decomposition["errors_oppose"] = (
        np.sign(decomposition["signed_calibration_error"])
        != np.sign(decomposition["mechanism_gap"])
    )
    decomposition_summary = (
        decomposition.groupby(["model", "fit_scope"], as_index=False)
        .agg(
            mean_signed_calibration_error=(
                "signed_calibration_error",
                "mean",
            ),
            mean_mechanism_gap=("mechanism_gap", "mean"),
            mean_signed_total_error=("signed_total_error", "mean"),
            opposing_error_rate=("errors_oppose", "mean"),
        )
    )
    mechanism = (
        checkpoint[checkpoint["model"] == "A+B"][
            ["seed", "tag", "d_med", "ess_med", "frontier", "dpo_realized"]
        ]
        .copy()
        .sort_values(["seed", "d_med"])
    )
    mechanism["mechanism_gap"] = (
        mechanism["frontier"] - mechanism["dpo_realized"]
    )
    mechanism["realization_ratio"] = (
        mechanism["dpo_realized"] / mechanism["frontier"]
    )
    return (
        checkpoint,
        seed_level,
        summary,
        window_summary,
        decomposition_summary,
        mechanism,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--pool-scored", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups = load_pool(args.pool_scored)
    geometry = prompt_geometry(groups)
    calibration_models, cal_metadata, ranges = fit_calibration_models(groups)
    eval_crossfit, xfit_metadata = fit_eval_crossfit_models(groups)
    result = evaluate(
        groups,
        calibration_models,
        eval_crossfit,
        ranges,
        args.multiseed_root,
        args.seeds,
    )
    (
        checkpoint,
        seed_level,
        summary,
        window_summary,
        decomposition_summary,
        mechanism,
    ) = aggregate(result)

    geometry.to_csv(args.outdir / "prompt_geometry.csv", index=False)
    pd.concat([cal_metadata, xfit_metadata], ignore_index=True).to_csv(
        args.outdir / "model_fit_metadata.csv", index=False
    )
    result.to_csv(args.outdir / "per_prompt_predictions.csv", index=False)
    checkpoint.to_csv(args.outdir / "checkpoint_predictions.csv", index=False)
    seed_level.to_csv(args.outdir / "seed_level_mae.csv", index=False)
    summary.to_csv(args.outdir / "summary_mean_sd.csv", index=False)
    window_summary.to_csv(
        args.outdir / "window_summary_mean_sd.csv", index=False
    )
    decomposition_summary.to_csv(
        args.outdir / "error_decomposition_summary.csv", index=False
    )
    mechanism.to_csv(
        args.outdir / "mechanism_by_checkpoint.csv", index=False
    )
    (args.outdir / "run_spec.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "modes": MODES,
                "families": FAMILIES,
                "evaluation_crossfit_folds": 5,
                "spline_cv": "5-fold GroupKFold by prompt",
                "strict_local_window": "checkpoint median d <= 0.10",
                "historical_scope": "six checkpoints with d <= 1 and ESS >= 8",
                "main_fit_scope": "calibration_only",
                "input_sha256": {
                    "pool_scored": sha256_file(args.pool_scored),
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
                    "builder": sha256_file(Path(__file__).resolve()),
                    "spline_helper": sha256_file(SPLINE_PATH.resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Prompt geometry:")
    print(geometry.to_string(index=False))
    print("\nCalibration-only and fixed baselines:")
    print(
        summary[
            summary["fit_scope"].isin(
                ["calibration_only", "coefficient_or_fixed_baseline"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
