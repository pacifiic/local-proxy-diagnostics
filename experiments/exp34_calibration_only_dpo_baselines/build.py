#!/usr/bin/env python3
"""Fair calibration-only flexible baselines for the fixed-pool DPO audit.

The flexible models are fit on calibration candidates and integrated over the
calibration prompt pools.  Their gain curves are frozen before evaluation
candidate proxy scores exist.  At evaluation time they receive only the KL
radius measured for each DPO prompt, matching the information used by A+B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

MODES = ("raw", "rank")
FAMILIES = ("isotonic", "binned", "spline", "mlp")
BIN_CANDIDATES = (3, 5, 8, 12)
SHRINK_CANDIDATES = (0.0, 5.0, 20.0)
KNOT_CANDIDATES = (4, 6, 8)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
MLP_HIDDEN_LAYERS = (16, 16)
MLP_ALPHA = 1e-3
MLP_MAX_ITER = 1000
MLP_SEED = 0
GRID_SIZE = 3001
FIXED_MODELS = ("A+B", "A-only", "A_nu", "Zero")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_proxy(proxy: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(proxy, dtype=float)
    if mode == "raw":
        return values
    if mode == "rank":
        return (rankdata(values, method="average") - 0.5) / len(values)
    raise ValueError(f"unknown mode: {mode}")


def transform_gold(gold: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(gold, dtype=float)
    if mode == "raw":
        return values
    if mode == "rank":
        return values - np.mean(values)
    raise ValueError(f"unknown mode: {mode}")


def load_calibration(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    groups: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "calibration":
                continue
            key = (str(row["prompt_id"]), int(row["sample_idx"]))
            if key in seen:
                raise ValueError(f"duplicate calibration candidate: {key}")
            seen.add(key)
            pair = (float(row["proxy_score"]), float(row["gold_score"]))
            if not np.all(np.isfinite(pair)):
                raise ValueError(f"non-finite calibration score: {key}")
            groups[key[0]].append(row)

    prompt_ids = sorted(groups)
    if len(prompt_ids) != 100:
        raise ValueError(f"expected 100 calibration prompts, got {len(prompt_ids)}")
    proxy_rows, gold_rows = [], []
    for prompt_id in prompt_ids:
        rows = sorted(groups[prompt_id], key=lambda row: int(row["sample_idx"]))
        if len(rows) != 32:
            raise ValueError(f"{prompt_id}: expected 32 candidates, got {len(rows)}")
        if [int(row["sample_idx"]) for row in rows] != list(range(32)):
            raise ValueError(f"{prompt_id}: sample_idx is not exactly 0..31")
        proxy_rows.append([float(row["proxy_score"]) for row in rows])
        gold_rows.append([float(row["gold_score"]) for row in rows])
    return prompt_ids, np.asarray(proxy_rows), np.asarray(gold_rows)


def pooled_xy(
    prompt_ids: list[str],
    proxy: np.ndarray,
    gold: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = [transform_proxy(row, mode) for row in proxy]
    y_rows = [transform_gold(row, mode) for row in gold]
    return (
        np.concatenate(x_rows),
        np.concatenate(y_rows),
        np.repeat(np.asarray(prompt_ids), proxy.shape[1]),
    )


def fit_bins(
    x: np.ndarray, y: np.ndarray, n_bins: int, shrink: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        edges = np.array(
            [float(np.min(x)) - 1e-12, float(np.median(x)),
             float(np.max(x)) + 1e-12]
        )
    inner = edges[1:-1]
    ids = np.digitize(x, inner)
    global_mean = float(np.mean(y))
    values = np.full(len(edges) - 1, global_mean)
    for bin_id in range(len(values)):
        mask = ids == bin_id
        count = int(mask.sum())
        if count:
            values[bin_id] = (
                count * float(np.mean(y[mask])) + shrink * global_mean
            ) / (count + shrink)
    return inner, values


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    family: str,
) -> tuple[callable, dict]:
    folds = GroupKFold(n_splits=5)
    if family == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        return model.predict, {
            "complexity": f"{len(model.X_thresholds_)} thresholds",
            "cv_scheme": "none",
        }

    if family == "binned":
        candidates = []
        for n_bins in BIN_CANDIDATES:
            for shrink in SHRINK_CANDIDATES:
                losses = []
                for train, validation in folds.split(x, y, groups):
                    edges, values = fit_bins(
                        x[train], y[train], n_bins, shrink
                    )
                    pred = values[
                        np.clip(
                            np.digitize(x[validation], edges),
                            0,
                            len(values) - 1,
                        )
                    ]
                    losses.append(float(np.mean((pred - y[validation]) ** 2)))
                candidates.append((float(np.mean(losses)), n_bins, shrink))
        cv_mse, n_bins, shrink = min(candidates)
        edges, values = fit_bins(x, y, n_bins, shrink)

        def predict(new_x: np.ndarray) -> np.ndarray:
            ids = np.clip(np.digitize(new_x, edges), 0, len(values) - 1)
            return values[ids]

        return predict, {
            "complexity": f"{len(values)} bins; shrink={shrink:g}",
            "cv_mse": cv_mse,
            "cv_scheme": "5-fold GroupKFold by prompt",
        }

    if family == "spline":
        x2 = x.reshape(-1, 1)
        candidates = []
        for knots in KNOT_CANDIDATES:
            for alpha in RIDGE_ALPHAS:
                candidate = make_pipeline(
                    SplineTransformer(
                        n_knots=knots,
                        degree=3,
                        extrapolation="constant",
                    ),
                    Ridge(alpha=float(alpha)),
                )
                mse = -float(
                    np.mean(
                        cross_val_score(
                            candidate,
                            x2,
                            y,
                            groups=groups,
                            cv=folds,
                            scoring="neg_mean_squared_error",
                        )
                    )
                )
                candidates.append((mse, knots, float(alpha)))
        cv_mse, knots, alpha = min(candidates)
        model = make_pipeline(
            SplineTransformer(
                n_knots=knots, degree=3, extrapolation="constant"
            ),
            Ridge(alpha=alpha),
        ).fit(x2, y)
        n_basis = int(model.named_steps["splinetransformer"].n_features_out_)
        return (
            lambda new_x: model.predict(np.asarray(new_x).reshape(-1, 1)),
            {
                "complexity": f"{n_basis} spline coefficients; alpha={alpha:g}",
                "cv_mse": cv_mse,
                "cv_scheme": "5-fold GroupKFold by prompt",
            },
        )

    if family == "mlp":
        scaler = StandardScaler().fit(x.reshape(-1, 1))
        y_mean, y_std = float(np.mean(y)), max(float(np.std(y)), 1e-12)
        model = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=MLP_ALPHA,
            learning_rate_init=1e-3,
            max_iter=MLP_MAX_ITER,
            random_state=MLP_SEED,
            early_stopping=False,
            tol=1e-4,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(scaler.transform(x.reshape(-1, 1)), (y - y_mean) / y_std)

        def predict(new_x: np.ndarray) -> np.ndarray:
            scaled = scaler.transform(np.asarray(new_x).reshape(-1, 1))
            return y_mean + y_std * model.predict(scaled)

        n_parameters = int(
            sum(value.size for value in model.coefs_)
            + sum(value.size for value in model.intercepts_)
        )
        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        return predict, {
            "complexity": f"1-16-16-1 ReLU MLP; {n_parameters} parameters",
            "alpha": MLP_ALPHA,
            "max_iter": MLP_MAX_ITER,
            "random_seed": MLP_SEED,
            "n_iter": int(model.n_iter_),
            "converged": bool(converged),
            "cv_scheme": "fixed pre-specified architecture",
        }
    raise ValueError(f"unknown family: {family}")


def fit_calibrators(
    prompt_ids: list[str], proxy: np.ndarray, gold: np.ndarray
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    fitted_values: dict[str, np.ndarray] = {"cal_empirical_curve": gold.copy()}
    metadata = [
        {
            "model": "cal_empirical_curve",
            "mode": "raw",
            "family": "empirical",
            "n_prompts": len(prompt_ids),
            "n_candidates": int(proxy.size),
            "complexity": "stored calibration gold values",
            "cv_scheme": "none",
        }
    ]
    for mode in MODES:
        x, y, groups = pooled_xy(prompt_ids, proxy, gold, mode)
        prompt_x = np.vstack([transform_proxy(row, mode) for row in proxy])
        for family in FAMILIES:
            predict, meta = fit_model(x, y, groups, family)
            name = f"cal_{mode}_{family}"
            fitted_values[name] = np.asarray(
                predict(prompt_x.reshape(-1)), dtype=float
            ).reshape(prompt_x.shape)
            metadata.append(
                {
                    "model": name,
                    "mode": mode,
                    "family": family,
                    "n_prompts": len(prompt_ids),
                    "n_candidates": int(proxy.size),
                    **meta,
                }
            )
    return fitted_values, pd.DataFrame(metadata)


def max_reachable_kl(proxy: np.ndarray) -> np.ndarray:
    top_count = np.sum(proxy == np.max(proxy, axis=1, keepdims=True), axis=1)
    return np.log(proxy.shape[1] / top_count)


def weights_at_d(proxy: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact weights, saturated at each prompt's maximum KL."""
    n_prompts, n_candidates = proxy.shape
    if d <= 0:
        return np.full_like(proxy, 1.0 / n_candidates), np.zeros(n_prompts)

    maximum = max_reachable_kl(proxy)
    saturated = d >= maximum - 1e-12
    active = ~saturated
    weights = np.zeros_like(proxy, dtype=float)
    if np.any(saturated):
        top = proxy[saturated] == np.max(
            proxy[saturated], axis=1, keepdims=True
        )
        weights[saturated] = top / np.sum(top, axis=1, keepdims=True)

    if np.any(active):
        values = proxy[active]
        lo = np.zeros(len(values))
        hi = np.ones(len(values))

        def kl_at_t(t: np.ndarray) -> np.ndarray:
            logits = t[:, None] * (
                values - np.max(values, axis=1, keepdims=True)
            )
            current = np.exp(logits)
            current /= np.sum(current, axis=1, keepdims=True)
            return np.log(n_candidates) + np.sum(
                current * np.log(np.clip(current, 1e-300, None)), axis=1
            )

        for _ in range(20):
            need = kl_at_t(hi) < d
            if not np.any(need):
                break
            hi[need] *= 2.0
        else:
            raise ValueError(f"could not bracket d={d:g}")
        for _ in range(70):
            mid = (lo + hi) / 2.0
            below = kl_at_t(mid) < d
            lo[below] = mid[below]
            hi[~below] = mid[~below]
        logits = hi[:, None] * (
            values - np.max(values, axis=1, keepdims=True)
        )
        current = np.exp(logits)
        current /= np.sum(current, axis=1, keepdims=True)
        weights[active] = current
    return weights, saturated.astype(float)


def build_frozen_curves(
    proxy: np.ndarray,
    fitted_values: dict[str, np.ndarray],
    max_d: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sqrt_grid = np.linspace(0.0, math.sqrt(max_d), GRID_SIZE)
    d_grid = sqrt_grid**2
    centered = {
        name: values - np.mean(values, axis=1, keepdims=True)
        for name, values in fitted_values.items()
    }
    rows, support = [], []
    for d in d_grid:
        weights, saturated = weights_at_d(proxy, float(d))
        support.append(
            {
                "d": float(d),
                "n_prompts": proxy.shape[0],
                "n_saturated_prompts": int(np.sum(saturated)),
                "saturated_fraction": float(np.mean(saturated)),
            }
        )
        for name, values in centered.items():
            prompt_gain = np.sum(weights * values, axis=1)
            rows.append(
                {
                    "d": float(d),
                    "model": name,
                    "prediction": float(np.mean(prompt_gain)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(support)


def interpolate_curves(curves: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (
            frame["d"].to_numpy(),
            frame["prediction"].to_numpy(),
        )
        for name, frame in curves.groupby("model", sort=False)
    }


def load_dpo_rows(multiseed_root: Path, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = multiseed_root / f"seed_{seed}" / "exp09_final_per_prompt.csv"
        frame = pd.read_csv(path)
        required = {
            "tag", "prompt_id", "d", "ess", "pred_AB", "pred_Aonly",
            "pred_Anu", "frontier", "dpo_reweight_gold",
        }
        if missing := required - set(frame):
            raise ValueError(f"seed {seed}: missing columns {sorted(missing)}")
        if frame.duplicated(["tag", "prompt_id"]).any():
            raise ValueError(f"seed {seed}: duplicate tag/prompt rows")
        local_tags = (
            frame.groupby("tag")
            .agg(d_med=("d", "median"), ess_med=("ess", "median"))
            .query("d_med <= 1.0 and ess_med >= 8.0")
            .index
        )
        frame = frame[frame["tag"].isin(local_tags)].copy()
        if len(local_tags) != 6 or len(frame) != 1200:
            raise ValueError(f"seed {seed}: expected 6 tags x 200 prompts")
        if frame["prompt_id"].nunique() != 200:
            raise ValueError(f"seed {seed}: expected 200 evaluation prompts")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_predictions(
    source: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    lookup = interpolate_curves(curves)
    rows = []
    fixed_columns = {
        "A+B": "pred_AB",
        "A-only": "pred_Aonly",
        "A_nu": "pred_Anu",
    }
    for row in source.itertuples():
        base = {
            "seed": int(row.seed),
            "tag": row.tag,
            "prompt_id": row.prompt_id,
            "d": float(row.d),
            "ess": float(row.ess),
            "frontier": float(row.frontier),
            "dpo_realized": float(row.dpo_reweight_gold),
        }
        for model, column in fixed_columns.items():
            rows.append(
                {
                    **base,
                    "model": model,
                    "prediction": float(getattr(row, column)),
                    "information_setting": "calibration_only",
                }
            )
        rows.append(
            {
                **base,
                "model": "Zero",
                "prediction": 0.0,
                "information_setting": "fixed_baseline",
            }
        )
        for model, (grid, values) in lookup.items():
            rows.append(
                {
                    **base,
                    "model": model,
                    "prediction": float(np.interp(row.d, grid, values)),
                    "information_setting": "calibration_only_frozen_curve",
                }
            )
    result = pd.DataFrame(rows)
    result["abs_error_frontier"] = np.abs(
        result["prediction"] - result["frontier"]
    )
    result["abs_error_dpo"] = np.abs(
        result["prediction"] - result["dpo_realized"]
    )
    return result


def aggregate(
    result: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = (
        result.groupby(
            ["seed", "tag", "model", "information_setting"], as_index=False
        )
        .agg(
            d_med=("d", "median"),
            ess_med=("ess", "median"),
            prediction=("prediction", "mean"),
            frontier=("frontier", "mean"),
            dpo_realized=("dpo_realized", "mean"),
            n_prompts=("prompt_id", "nunique"),
        )
    )
    checkpoint["abs_error_frontier"] = np.abs(
        checkpoint["prediction"] - checkpoint["frontier"]
    )
    checkpoint["abs_error_dpo"] = np.abs(
        checkpoint["prediction"] - checkpoint["dpo_realized"]
    )
    seed_rows = []
    for scope, max_d in (
        ("strict_local_d_le_0p10", 0.10),
        ("historical_six_checkpoints", np.inf),
    ):
        scoped = checkpoint[checkpoint["d_med"] <= max_d]
        summary = (
            scoped.groupby(
                ["seed", "model", "information_setting"], as_index=False
            )
            .agg(
                mae_frontier=("abs_error_frontier", "mean"),
                mae_dpo=("abs_error_dpo", "mean"),
                n_checkpoints=("tag", "nunique"),
            )
        )
        summary.insert(0, "scope", scope)
        seed_rows.append(summary)
    seed_level = pd.concat(seed_rows, ignore_index=True)
    summary = (
        seed_level.groupby(
            ["scope", "model", "information_setting"], as_index=False
        )
        .agg(
            mean_mae_frontier=("mae_frontier", "mean"),
            sd_mae_frontier=("mae_frontier", "std"),
            mean_mae_dpo=("mae_dpo", "mean"),
            sd_mae_dpo=("mae_dpo", "std"),
            min_checkpoints=("n_checkpoints", "min"),
            max_checkpoints=("n_checkpoints", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["scope", "mean_mae_dpo"])
    )

    decomposition_rows = []
    for scope, max_d in (
        ("strict_local_d_le_0p10", 0.10),
        ("historical_six_checkpoints", np.inf),
    ):
        scoped = checkpoint[checkpoint["d_med"] <= max_d].copy()
        scoped["frontier_signed_error"] = (
            scoped["prediction"] - scoped["frontier"]
        )
        scoped["optimizer_realization_gap"] = (
            scoped["frontier"] - scoped["dpo_realized"]
        )
        scoped["dpo_signed_error"] = (
            scoped["prediction"] - scoped["dpo_realized"]
        )
        per_seed = (
            scoped.groupby(
                ["seed", "model", "information_setting"], as_index=False
            )
            .agg(
                frontier_signed_error=("frontier_signed_error", "mean"),
                optimizer_realization_gap=(
                    "optimizer_realization_gap", "mean"
                ),
                dpo_signed_error=("dpo_signed_error", "mean"),
            )
        )
        across_seed = (
            per_seed.groupby(
                ["model", "information_setting"], as_index=False
            )
            .agg(
                mean_frontier_signed_error=(
                    "frontier_signed_error", "mean"
                ),
                sd_frontier_signed_error=("frontier_signed_error", "std"),
                mean_optimizer_realization_gap=(
                    "optimizer_realization_gap", "mean"
                ),
                sd_optimizer_realization_gap=(
                    "optimizer_realization_gap", "std"
                ),
                mean_dpo_signed_error=("dpo_signed_error", "mean"),
                sd_dpo_signed_error=("dpo_signed_error", "std"),
                n_seeds=("model", "size"),
            )
        )
        across_seed.insert(0, "scope", scope)
        decomposition_rows.append(across_seed)
    decomposition = pd.concat(decomposition_rows, ignore_index=True)
    return checkpoint, seed_level, summary, decomposition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-scored", type=Path, required=True)
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    prompt_ids, proxy, gold = load_calibration(args.pool_scored)
    source = load_dpo_rows(args.multiseed_root, args.seeds)
    max_d = max(float(source["d"].max()), 1e-6)
    fitted_values, metadata = fit_calibrators(prompt_ids, proxy, gold)
    curves, support = build_frozen_curves(proxy, fitted_values, max_d)
    result = make_predictions(source, curves)
    checkpoint, seed_level, summary, decomposition = aggregate(result)

    metadata.to_csv(args.outdir / "model_fit_metadata.csv", index=False)
    curves.to_csv(args.outdir / "frozen_calibration_curves.csv", index=False)
    support.to_csv(args.outdir / "curve_support.csv", index=False)
    result.to_csv(args.outdir / "per_prompt_predictions.csv", index=False)
    checkpoint.to_csv(args.outdir / "checkpoint_predictions.csv", index=False)
    seed_level.to_csv(args.outdir / "seed_level_mae.csv", index=False)
    summary.to_csv(args.outdir / "summary_mean_sd.csv", index=False)
    decomposition.to_csv(
        args.outdir / "error_decomposition_mean_sd.csv", index=False
    )

    run_spec = {
        "experiment": "exp34_calibration_only_dpo_baselines",
        "information_setting": {
            "operational_example": (
                "decide from a rated calibration pool before future "
                "optimization candidates have been collected"
            ),
            "fit_inputs": "100 calibration prompts x 32 proxy/gold candidates",
            "prediction_inputs": (
                "frozen calibration gain curve plus each DPO prompt's "
                "measured KL radius d"
            ),
            "evaluation_proxy_scores": "not loaded by the predictor",
            "evaluation_gold_scores": "used only in saved DPO outcomes",
            "target_pool_available_at_prediction_time": False,
        },
        "seeds": args.seeds,
        "n_calibration_prompts": len(prompt_ids),
        "candidates_per_prompt": int(proxy.shape[1]),
        "modes": MODES,
        "families": FAMILIES,
        "grid_size": GRID_SIZE,
        "max_grid_d": max_d,
        "checkpoint_scope": "six checkpoints with median d <= 1 and ESS >= 8",
        "strict_local_scope": "checkpoint median d <= 0.10",
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
        "code_sha256": {"build.py": sha256_file(Path(__file__).resolve())},
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
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
