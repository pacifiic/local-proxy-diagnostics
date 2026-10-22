#!/usr/bin/env python3
"""Shared prompt-conditional gain-curve baselines for exp25.

All models are fit on calibration prompts only. Evaluation proxy scores are
used to construct the requested KL tilt, but evaluation gold scores are used
only after predictions have been frozen.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


BASELINES = [
    "A+B",
    "A-only",
    "A_nu",
    "empirical_curve",
    "empirical_fit",
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
    "mlp_calibration",
    "zero",
]

DIRECT_BASELINES = [
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
    "mlp_calibration",
]


@dataclass(frozen=True)
class Coefficients:
    a: float
    b: float
    a_nu: float
    n_prompts_used: int
    n_prompts_skipped: int


@dataclass
class FittedModels:
    coefficients: Coefficients
    empirical_a: float
    empirical_b: float
    direct_predictors: dict[str, Callable[[np.ndarray], np.ndarray]]
    metadata: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_empty_outdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_pool(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported candidate-pool format: {path}")
    return frame


def validate_pool(
    frame: pd.DataFrame,
    *,
    name: str,
    prompt_col: str = "prompt_id",
    proxy_col: str = "proxy_score",
    gold_col: str = "gold_score",
    candidate_col: str | None = None,
) -> dict[str, object]:
    required = {prompt_col, proxy_col, gold_col}
    if candidate_col:
        required.add(candidate_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name}: empty candidate pool")
    if frame[prompt_col].isna().any():
        raise ValueError(f"{name}: missing prompt IDs")
    scores = frame[[proxy_col, gold_col]].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"{name}: non-finite proxy or gold scores")
    if candidate_col and frame.duplicated([prompt_col, candidate_col]).any():
        raise ValueError(f"{name}: duplicate prompt/candidate keys")
    counts = frame.groupby(prompt_col, sort=True).size()
    if int(counts.min()) < 2:
        raise ValueError(f"{name}: at least one prompt has fewer than two candidates")
    nonconstant_proxy = frame.groupby(prompt_col)[proxy_col].var(ddof=0).gt(0)
    nonconstant_gold = frame.groupby(prompt_col)[gold_col].var(ddof=0).gt(0)
    return {
        "name": name,
        "n_rows": int(len(frame)),
        "n_prompts": int(frame[prompt_col].nunique()),
        "min_candidates_per_prompt": int(counts.min()),
        "max_candidates_per_prompt": int(counts.max()),
        "n_nonconstant_proxy_prompts": int(nonconstant_proxy.sum()),
        "n_nonconstant_gold_prompts": int(nonconstant_gold.sum()),
        "proxy_gold_pearson": float(
            frame[[proxy_col, gold_col]].corr().iloc[0, 1]
        ),
    }


def _central_moments(
    gold: np.ndarray, proxy: np.ndarray
) -> tuple[float, float, float, float, float]:
    jc = gold - float(np.mean(gold))
    rc = proxy - float(np.mean(proxy))
    return (
        float(np.mean(jc**2)),
        float(np.mean(rc**2)),
        float(np.mean(jc * rc)),
        float(np.mean(rc**3)),
        float(np.mean(jc * rc**2)),
    )


def _coefficient_from_moments(
    var_j: float,
    var_r: float,
    cov_jr: float,
    kappa_r: float,
    kappa_jrr: float,
) -> tuple[float, float, float]:
    if var_r <= 0:
        raise ValueError("proxy variance is zero")
    sigma_r = math.sqrt(var_r)
    sigma_j = math.sqrt(max(var_j, 0.0))
    a = math.sqrt(2.0) * cov_jr / sigma_r
    b = (
        kappa_jrr / var_r
        - 2.0 * cov_jr * kappa_r / (3.0 * var_r**2)
    )
    a_nu = math.sqrt(2.0) * sigma_j * sigma_j / sigma_r
    return a, b, a_nu


def estimate_coefficients(
    frame: pd.DataFrame,
    *,
    geometry: str,
    prompt_col: str,
    proxy_col: str,
    gold_col: str,
) -> Coefficients:
    moments: list[tuple[float, float, float, float, float]] = []
    prompt_coefficients: list[tuple[float, float, float]] = []
    skipped = 0
    for _, group in frame.groupby(prompt_col, sort=True):
        proxy = group[proxy_col].to_numpy(dtype=float)
        gold = group[gold_col].to_numpy(dtype=float)
        vals = _central_moments(gold, proxy)
        moments.append(vals)
        if vals[1] <= 0 or vals[0] <= 0:
            skipped += 1
            continue
        prompt_coefficients.append(_coefficient_from_moments(*vals))

    if geometry == "common_temperature":
        averaged = np.asarray(moments, dtype=float).mean(axis=0)
        a, b, a_nu = _coefficient_from_moments(*averaged.tolist())
        used = len(moments)
        skipped = 0
    elif geometry == "fixed_prompt_kl":
        if not prompt_coefficients:
            raise ValueError("no prompts have nonzero proxy and gold variance")
        a, b, a_nu = np.asarray(prompt_coefficients, dtype=float).mean(axis=0)
        used = len(prompt_coefficients)
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    return Coefficients(
        a=float(a),
        b=float(b),
        a_nu=float(a_nu),
        n_prompts_used=int(used),
        n_prompts_skipped=int(skipped),
    )


def tilt_weights(proxy: np.ndarray, t: float) -> np.ndarray:
    shifted = t * (proxy - float(np.max(proxy)))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def prompt_kl(proxy: np.ndarray, t: float) -> float:
    weights = tilt_weights(proxy, t)
    positive = weights > 0
    return float(
        np.sum(
            weights[positive]
            * (np.log(weights[positive]) + math.log(float(len(proxy))))
        )
    )


def _group_arrays(
    frame: pd.DataFrame,
    prompt_col: str,
    proxy_col: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in frame.groupby(prompt_col, sort=True):
        positions = group["_row_position"].to_numpy(dtype=int)
        proxy = group[proxy_col].to_numpy(dtype=float)
        groups.append((positions, proxy))
    return groups


def _find_t_common(groups: list[tuple[np.ndarray, np.ndarray]], d: float) -> float:
    if d <= 0:
        return 0.0

    def mean_kl(t: float) -> float:
        return float(np.mean([prompt_kl(proxy, t) for _, proxy in groups]))

    lo, hi = 0.0, 1.0
    while mean_kl(hi) < d:
        lo, hi = hi, 2.0 * hi
        if hi > 1e6:
            raise RuntimeError(f"could not reach common-temperature KL d={d}")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if mean_kl(mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def _find_t_prompt(proxy: np.ndarray, d: float) -> float:
    if d <= 0 or float(np.var(proxy)) == 0:
        return 0.0
    max_d = math.log(float(len(proxy)) / float(np.sum(proxy == np.max(proxy))))
    if d >= max_d - 1e-12:
        # Match the audited exp09/exp11 convention: prompts whose discrete
        # proxy maximum is tied saturate at the largest available raw-score
        # tilt. Reachability is reported separately by exp25 rather than
        # silently treating the achieved KL as the requested KL.
        return 1e4
    lo, hi = 0.0, 1.0
    while prompt_kl(proxy, hi) < d:
        lo, hi = hi, 2.0 * hi
        if hi > 1e4:
            return 1e4
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if prompt_kl(proxy, mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def prompt_reachability(
    frame: pd.DataFrame,
    d_values: Iterable[float],
    *,
    prompt_col: str = "prompt_id",
    proxy_col: str = "proxy_score",
) -> pd.DataFrame:
    maxima: list[tuple[str, float]] = []
    for prompt_id, group in frame.groupby(prompt_col, sort=True):
        proxy = group[proxy_col].to_numpy(dtype=float)
        n_max = int(np.sum(proxy == np.max(proxy)))
        maxima.append(
            (str(prompt_id), math.log(float(len(proxy)) / float(n_max)))
        )
    rows: list[dict[str, object]] = []
    for d_raw in d_values:
        d = float(d_raw)
        reachable = [max_d + 1e-12 >= d for _, max_d in maxima]
        rows.append(
            {
                "d": d,
                "n_prompts": len(maxima),
                "n_reachable": int(np.sum(reachable)),
                "reachable_fraction": float(np.mean(reachable)),
                "min_max_reachable_d": float(min(v for _, v in maxima)),
                "max_max_reachable_d": float(max(v for _, v in maxima)),
            }
        )
    return pd.DataFrame(rows)


def weights_for_radius(
    frame: pd.DataFrame,
    d: float,
    *,
    geometry: str,
    prompt_col: str,
    proxy_col: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = _group_arrays(frame, prompt_col, proxy_col)
    if geometry == "common_temperature":
        common_t = _find_t_common(groups, d)
        return [
            (positions, tilt_weights(proxy, common_t))
            for positions, proxy in groups
        ]
    if geometry == "fixed_prompt_kl":
        return [
            (positions, tilt_weights(proxy, _find_t_prompt(proxy, d)))
            for positions, proxy in groups
        ]
    raise ValueError(f"unknown geometry: {geometry}")


def gains_from_values(
    values: np.ndarray,
    grouped_weights: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray]:
    per_prompt = np.asarray(
        [
            float(np.sum(weights * values[positions]) - np.mean(values[positions]))
            for positions, weights in grouped_weights
        ],
        dtype=float,
    )
    return float(np.mean(per_prompt)), per_prompt


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    out["_row_position"] = np.arange(len(out), dtype=int)
    return out


def _group_cv(prompt_ids: np.ndarray, max_folds: int = 5) -> GroupKFold:
    n_groups = int(pd.Series(prompt_ids).nunique())
    if n_groups < 2:
        raise ValueError("at least two calibration prompts are required for CV")
    return GroupKFold(n_splits=min(max_folds, n_groups))


def _fit_binned(
    proxy: np.ndarray,
    gold: np.ndarray,
    prompt_ids: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    bin_candidates = [b for b in (3, 5, 8, 12) if b * 3 <= len(proxy)]
    if not bin_candidates:
        bin_candidates = [2]
    shrink_candidates = (0.0, 5.0, 20.0)
    splitter = _group_cv(prompt_ids)

    def fit_one(
        x: np.ndarray, y: np.ndarray, n_bins: int, shrink: float
    ) -> tuple[np.ndarray, np.ndarray]:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.asarray(
                [float(np.min(x)) - 1e-9, float(np.median(x)), float(np.max(x)) + 1e-9]
            )
        inner = edges[1:-1]
        ids = np.digitize(x, inner)
        global_mean = float(np.mean(y))
        means = np.full(len(edges) - 1, global_mean, dtype=float)
        for index in range(len(means)):
            mask = ids == index
            count = int(mask.sum())
            if count:
                means[index] = (
                    count * float(np.mean(y[mask])) + shrink * global_mean
                ) / (count + shrink)
        return inner, means

    best: tuple[float, int, float] | None = None
    for n_bins in bin_candidates:
        for shrink in shrink_candidates:
            errors: list[float] = []
            for train, validation in splitter.split(proxy, gold, prompt_ids):
                inner, means = fit_one(
                    proxy[train], gold[train], n_bins, shrink
                )
                ids = np.clip(
                    np.digitize(proxy[validation], inner), 0, len(means) - 1
                )
                errors.append(
                    float(np.mean((means[ids] - gold[validation]) ** 2))
                )
            candidate = (float(np.mean(errors)), n_bins, shrink)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    _, n_bins, shrink = best
    inner, means = fit_one(proxy, gold, n_bins, shrink)

    def predict(new_proxy: np.ndarray) -> np.ndarray:
        ids = np.clip(np.digitize(new_proxy, inner), 0, len(means) - 1)
        return means[ids]

    return predict, {"n_bins": int(n_bins), "shrink_k": float(shrink)}


def _fit_spline(
    proxy: np.ndarray,
    gold: np.ndarray,
    prompt_ids: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    splitter = _group_cv(prompt_ids)
    alphas = np.logspace(-3, 3, 9)
    best: tuple[float, int, float] | None = None
    x = proxy.reshape(-1, 1)
    for n_knots in (4, 6, 8):
        for alpha in alphas:
            errors: list[float] = []
            for train, validation in splitter.split(x, gold, prompt_ids):
                model = make_pipeline(
                    SplineTransformer(
                        n_knots=n_knots,
                        degree=3,
                        extrapolation="constant",
                    ),
                    Ridge(alpha=float(alpha)),
                )
                model.fit(x[train], gold[train])
                prediction = model.predict(x[validation])
                errors.append(
                    float(np.mean((prediction - gold[validation]) ** 2))
                )
            candidate = (float(np.mean(errors)), n_knots, float(alpha))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    _, n_knots, alpha = best
    model = make_pipeline(
        SplineTransformer(
            n_knots=n_knots, degree=3, extrapolation="constant"
        ),
        Ridge(alpha=alpha),
    )
    model.fit(x, gold)

    def predict(new_proxy: np.ndarray) -> np.ndarray:
        return model.predict(np.asarray(new_proxy, dtype=float).reshape(-1, 1))

    return predict, {"n_knots": int(n_knots), "ridge_alpha": float(alpha)}


def _fit_mlp(
    proxy: np.ndarray,
    gold: np.ndarray,
    seed: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    x_scaler = StandardScaler()
    x = x_scaler.fit_transform(proxy.reshape(-1, 1))
    y_mean = float(np.mean(gold))
    y_std = float(np.std(gold))
    if y_std <= 0:
        return (
            lambda new_proxy: np.full(len(new_proxy), y_mean, dtype=float),
            {"hidden_layers": [16, 16], "alpha": 0.001, "constant": True},
        )
    model = MLPRegressor(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=seed,
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, (gold - y_mean) / y_std)

    def predict(new_proxy: np.ndarray) -> np.ndarray:
        scaled = x_scaler.transform(
            np.asarray(new_proxy, dtype=float).reshape(-1, 1)
        )
        return y_mean + y_std * model.predict(scaled)

    return predict, {
        "hidden_layers": [16, 16],
        "alpha": 0.001,
        "max_iter": 500,
        "seed": int(seed),
        "constant": False,
    }


def fit_models(
    calibration: pd.DataFrame,
    *,
    geometry: str,
    d_values: Iterable[float],
    fit_max_d: float,
    seed: int,
    prompt_col: str = "prompt_id",
    proxy_col: str = "proxy_score",
    gold_col: str = "gold_score",
) -> FittedModels:
    calibration = prepare_frame(calibration)
    coefficients = estimate_coefficients(
        calibration,
        geometry=geometry,
        prompt_col=prompt_col,
        proxy_col=proxy_col,
        gold_col=gold_col,
    )
    proxy = calibration[proxy_col].to_numpy(dtype=float)
    gold = calibration[gold_col].to_numpy(dtype=float)
    prompt_ids = calibration[prompt_col].astype(str).to_numpy()

    direct: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    metadata: dict[str, object] = {}
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    isotonic.fit(proxy, gold)
    direct["isotonic_calibration"] = lambda values: isotonic.predict(values)
    direct["binned_shrinkage"], metadata["binned_shrinkage"] = _fit_binned(
        proxy, gold, prompt_ids
    )
    direct["spline_ridge"], metadata["spline_ridge"] = _fit_spline(
        proxy, gold, prompt_ids
    )
    direct["mlp_calibration"], metadata["mlp_calibration"] = _fit_mlp(
        proxy, gold, seed
    )

    fit_radii = np.linspace(max(0.001, fit_max_d / 25.0), fit_max_d, 25)
    fit_gains: list[float] = []
    for d in fit_radii:
        weights = weights_for_radius(
            calibration,
            float(d),
            geometry=geometry,
            prompt_col=prompt_col,
            proxy_col=proxy_col,
        )
        gain, _ = gains_from_values(gold, weights)
        fit_gains.append(gain)
    design = np.column_stack([np.sqrt(fit_radii), fit_radii])
    empirical_a, empirical_b = np.linalg.lstsq(
        design, np.asarray(fit_gains), rcond=None
    )[0]
    metadata["fit_radii"] = fit_radii.tolist()
    metadata["coefficients"] = asdict(coefficients)
    metadata["requested_d_values"] = [float(d) for d in d_values]
    return FittedModels(
        coefficients=coefficients,
        empirical_a=float(empirical_a),
        empirical_b=float(empirical_b),
        direct_predictors=direct,
        metadata=metadata,
    )


def evaluate_models(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    models: FittedModels,
    *,
    geometry: str,
    d_values: Iterable[float],
    prompt_col: str = "prompt_id",
    proxy_col: str = "proxy_score",
    gold_col: str = "gold_score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = prepare_frame(calibration)
    evaluation = prepare_frame(evaluation)
    cal_gold = calibration[gold_col].to_numpy(dtype=float)
    eval_gold = evaluation[gold_col].to_numpy(dtype=float)
    eval_proxy = evaluation[proxy_col].to_numpy(dtype=float)
    direct_values = {
        name: predictor(eval_proxy)
        for name, predictor in models.direct_predictors.items()
    }

    aggregate_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    prompt_ids = [
        str(prompt_id)
        for prompt_id, _ in evaluation.groupby(prompt_col, sort=True)
    ]
    for d_raw in d_values:
        d = float(d_raw)
        cal_weights = weights_for_radius(
            calibration,
            d,
            geometry=geometry,
            prompt_col=prompt_col,
            proxy_col=proxy_col,
        )
        eval_weights = weights_for_radius(
            evaluation,
            d,
            geometry=geometry,
            prompt_col=prompt_col,
            proxy_col=proxy_col,
        )
        empirical_curve, _ = gains_from_values(cal_gold, cal_weights)
        actual, actual_per_prompt = gains_from_values(eval_gold, eval_weights)
        sqrt_d = math.sqrt(d)
        scalar_predictions = {
            "A+B": models.coefficients.a * sqrt_d
            + models.coefficients.b * d,
            "A-only": models.coefficients.a * sqrt_d,
            "A_nu": models.coefficients.a_nu * sqrt_d,
            "empirical_curve": empirical_curve,
            "empirical_fit": models.empirical_a * sqrt_d
            + models.empirical_b * d,
            "zero": 0.0,
        }
        direct_predictions: dict[str, tuple[float, np.ndarray]] = {
            name: gains_from_values(values, eval_weights)
            for name, values in direct_values.items()
        }
        for baseline in BASELINES:
            if baseline in direct_predictions:
                predicted, predicted_per_prompt = direct_predictions[baseline]
            else:
                predicted = float(scalar_predictions[baseline])
                predicted_per_prompt = np.full(
                    len(actual_per_prompt), predicted, dtype=float
                )
            aggregate_rows.append(
                {
                    "d": d,
                    "baseline": baseline,
                    "predicted_gain": float(predicted),
                    "actual_gain": float(actual),
                    "error": float(predicted - actual),
                    "abs_error": float(abs(predicted - actual)),
                }
            )
            for prompt_id, prediction_i, actual_i in zip(
                prompt_ids, predicted_per_prompt, actual_per_prompt
            ):
                prompt_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "d": d,
                        "baseline": baseline,
                        "predicted_gain": float(prediction_i),
                        "actual_gain": float(actual_i),
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(prompt_rows)
