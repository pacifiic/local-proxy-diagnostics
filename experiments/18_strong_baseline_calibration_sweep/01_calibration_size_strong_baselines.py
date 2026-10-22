#!/usr/bin/env python3
"""Calibration-size sweep with strong calibration baselines.

This experiment extends the prospective split protocol by asking how strong
calibration baselines behave as the calibration set gets small.  It compares
the paper's moment predictor (A+B) against direct empirical curve transfer,
empirical curve fitting, isotonic and MLP score calibration, correlation
baselines, reliability-only A_nu, and zero gain.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import (  # noqa: E402
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
    mkdir,
    parse_cols,
    spearman_corr,
    split_indices,
    theory_coefficients,
    tilt_weights,
)


BASELINES = [
    "A+B",
    "A-only",
    "A_nu",
    "empirical_curve",
    "empirical_fit",
    "pearson_corr",
    "spearman_corr",
    "isotonic_calibration",
    "mlp_calibration",
    "zero",
]


def ensure_empty_outdir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Use --overwrite or choose a fresh --outdir."
        )
    return mkdir(path)


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def parse_int_list(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one hidden-layer width.")
    if any(v <= 0 for v in vals):
        raise ValueError(f"MLP hidden-layer widths must be positive: {vals}")
    return vals


def safe_corr(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    val = spearman_corr(x, y) if rank else float(np.corrcoef(x, y)[0, 1])
    return float(val) if np.isfinite(val) else 0.0


def curve_deltas(curve: Any, d_values: Sequence[float]) -> dict[float, float]:
    q0 = float(curve.q[0])
    return {float(d): float(np.interp(float(d), curve.d, curve.q) - q0) for d in d_values}


def exact_weights_at_d_from_curve(proxy: np.ndarray, curve: Any, d: float) -> np.ndarray:
    r = np.asarray(proxy, dtype=float)
    if d <= 0:
        return np.ones(len(r), dtype=float) / float(len(r))
    t = float(np.interp(float(d), curve.d, curve.t))
    return tilt_weights(r, t)


def isotonic_predictions(
    cal_proxy: np.ndarray,
    cal_gold: np.ndarray,
    eval_proxy: np.ndarray,
    eval_curve: Any,
    d_values: Sequence[float],
) -> dict[float, float]:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(cal_proxy, dtype=float), np.asarray(cal_gold, dtype=float))
    h = model.predict(np.asarray(eval_proxy, dtype=float))
    base = float(np.mean(h))
    out: dict[float, float] = {}
    for d in d_values:
        w = exact_weights_at_d_from_curve(eval_proxy, eval_curve, float(d))
        out[float(d)] = float(np.sum(w * h) - base)
    return out


def mlp_predictions(
    cal_proxy: np.ndarray,
    cal_gold: np.ndarray,
    eval_proxy: np.ndarray,
    eval_curve: Any,
    d_values: Sequence[float],
    *,
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    learning_rate_init: float,
    random_seed: int,
) -> dict[float, float]:
    """Black-box score-calibration baseline.

    The MLP is fit only on calibration items as a one-dimensional calibrator
    h(r) ~= E[J | r].  Predicted gains on the evaluation split are then computed
    by integrating h(r) under the same exact tilt weights used by the other
    calibration baselines.
    """
    x_cal = np.asarray(cal_proxy, dtype=float).reshape(-1, 1)
    y_cal = np.asarray(cal_gold, dtype=float)
    x_eval = np.asarray(eval_proxy, dtype=float).reshape(-1, 1)
    if len(x_cal) < 4 or float(np.nanstd(x_cal)) == 0.0 or float(np.nanstd(y_cal)) == 0.0:
        return {float(d): 0.0 for d in d_values}

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_cal)
    x_test = x_scaler.transform(x_eval)
    y_mean = float(np.mean(y_cal))
    y_std = float(np.std(y_cal))
    if y_std == 0.0 or not np.isfinite(y_std):
        return {float(d): 0.0 for d in d_values}
    y_train = (y_cal - y_mean) / y_std

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=False,
        n_iter_no_change=max(20, max_iter // 10),
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)

    h = y_mean + y_std * model.predict(x_test)
    base = float(np.mean(h))
    out: dict[float, float] = {}
    for d in d_values:
        w = exact_weights_at_d_from_curve(eval_proxy, eval_curve, float(d))
        out[float(d)] = float(np.sum(w * h) - base)
    return out


def baseline_predictions(
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    r_eval: np.ndarray,
    eval_curve: Any,
    d_values: Sequence[float],
    fit_max_d: float,
    n_grid: int,
    active_baselines: Sequence[str],
    *,
    mlp_hidden_layers: tuple[int, ...],
    mlp_alpha: float,
    mlp_max_iter: int,
    mlp_learning_rate_init: float,
    mlp_seed: int,
) -> dict[str, dict[float, float]]:
    coeff = theory_coefficients(j_cal, r_cal)
    curve_cal = exact_tilt_curve(r_cal, j_cal, d_max=fit_max_d, n_grid=n_grid)
    fit_cal = fit_local_law(curve_cal.d, curve_cal.q, fit_max_d=fit_max_d)
    cal_deltas = {float(d): float(np.interp(float(d), curve_cal.d, curve_cal.q) - float(curve_cal.q[0])) for d in d_values}
    iso = isotonic_predictions(r_cal, j_cal, r_eval, eval_curve, d_values)
    mlp = (
        mlp_predictions(
            r_cal,
            j_cal,
            r_eval,
            eval_curve,
            d_values,
            hidden_layers=mlp_hidden_layers,
            alpha=mlp_alpha,
            max_iter=mlp_max_iter,
            learning_rate_init=mlp_learning_rate_init,
            random_seed=mlp_seed,
        )
        if "mlp_calibration" in active_baselines
        else {}
    )

    sigma_j = float(np.std(j_cal))
    rho_pearson = safe_corr(r_cal, j_cal, rank=False)
    rho_spearman = safe_corr(r_cal, j_cal, rank=True)
    a_pearson = math.sqrt(2.0) * sigma_j * rho_pearson
    a_spearman = math.sqrt(2.0) * sigma_j * rho_spearman

    preds: dict[str, dict[float, float]] = {name: {} for name in active_baselines}
    for d in d_values:
        d = float(d)
        sqrt_d = math.sqrt(max(0.0, d))
        if "A+B" in preds:
            preds["A+B"][d] = coeff.a_exact * sqrt_d + coeff.b_theory * d
        if "A-only" in preds:
            preds["A-only"][d] = coeff.a_exact * sqrt_d
        if "A_nu" in preds:
            preds["A_nu"][d] = coeff.a_nu * sqrt_d
        if "empirical_curve" in preds:
            preds["empirical_curve"][d] = cal_deltas[d]
        if "empirical_fit" in preds:
            preds["empirical_fit"][d] = fit_cal.a_fit * sqrt_d + fit_cal.b_fit * d
        if "pearson_corr" in preds:
            preds["pearson_corr"][d] = a_pearson * sqrt_d
        if "spearman_corr" in preds:
            preds["spearman_corr"][d] = a_spearman * sqrt_d
        if "isotonic_calibration" in preds:
            preds["isotonic_calibration"][d] = iso[d]
        if "mlp_calibration" in preds:
            preds["mlp_calibration"][d] = mlp[d]
        if "zero" in preds:
            preds["zero"][d] = 0.0
    return preds


def aggregate_proxy_if_needed(
    proxy_df: pd.DataFrame,
    item_col: str,
    score_col: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    cols = [*group_cols, item_col, score_col]
    missing = [c for c in cols if c not in proxy_df.columns]
    if missing:
        raise ValueError(f"proxy csv is missing required columns: {missing}")
    return (
        proxy_df[cols]
        .groupby([*group_cols, item_col], dropna=False)[score_col]
        .mean()
        .reset_index()
    )


def build_paths(args: argparse.Namespace):
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    if args.proxy_csv:
        proxy = pd.read_csv(args.proxy_csv)
        proxy = aggregate_proxy_if_needed(proxy, args.proxy_item_col, args.proxy_score_col, group_cols)
        proxy_score_col = args.proxy_score_col
        if proxy_score_col != "proxy_score":
            proxy = proxy.rename(columns={proxy_score_col: "proxy_score"})
            proxy_score_col = "proxy_score"
        proxy_item_col = args.proxy_item_col
        if proxy_item_col != "item_id":
            proxy = proxy.rename(columns={proxy_item_col: "item_id"})
            proxy_item_col = "item_id"
        return list(
            generate_model_paths(
                human_long=human,
                proxy_df=proxy,
                item_cols=item_cols,
                score_col=args.score_col,
                group_cols=group_cols,
                proxy_item_col=proxy_item_col,
                proxy_score_col=proxy_score_col,
                min_items=args.min_items,
            )
        )
    return list(
        generate_human_paths(
            df=human,
            item_cols=item_cols,
            rater_col=args.rater_col,
            score_col=args.score_col,
            group_cols=group_cols,
            min_items=args.min_items,
            min_raters=args.min_raters,
        )
    )


def summarize_endpoint(pred: pd.DataFrame, endpoint_d: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint = pred[np.isclose(pred["d"], endpoint_d)]
    for key, g in endpoint.groupby(["dataset", "proxy_mode", "calibration_frac", "baseline"], dropna=False):
        dataset, proxy_mode, frac, baseline = key
        path_means = g.groupby("path_id", dropna=False)["abs_error"].mean()
        rows.append(
            {
                "dataset": dataset,
                "proxy_mode": proxy_mode,
                "calibration_frac": float(frac),
                "d": endpoint_d,
                "baseline": baseline,
                "n_rows": int(len(g)),
                "n_paths": int(g["path_id"].nunique()),
                "mae": float(g["abs_error"].mean()),
                "median_abs_error": float(g["abs_error"].median()),
                "path_se": float(path_means.std(ddof=1) / math.sqrt(len(path_means))) if len(path_means) > 1 else float("nan"),
                "mean_actual_gain": float(g["actual_gain"].mean()),
                "sign_accuracy": float((np.sign(g["predicted_gain"]) == np.sign(g["actual_gain"])).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "proxy_mode", "calibration_frac", "mae", "baseline"])


def summarize_curve(pred: pd.DataFrame) -> pd.DataFrame:
    per_curve = (
        pred.groupby(["dataset", "proxy_mode", "calibration_frac", "baseline", "path_id", "repeat"], dropna=False)
        .agg(
            grid_iae=("abs_error", "mean"),
            max_error=("abs_error", "max"),
            mean_actual_gain=("actual_gain", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for key, g in per_curve.groupby(["dataset", "proxy_mode", "calibration_frac", "baseline"], dropna=False):
        dataset, proxy_mode, frac, baseline = key
        path_means = g.groupby("path_id", dropna=False)["grid_iae"].mean()
        rows.append(
            {
                "dataset": dataset,
                "proxy_mode": proxy_mode,
                "calibration_frac": float(frac),
                "baseline": baseline,
                "n_curves": int(len(g)),
                "n_paths": int(g["path_id"].nunique()),
                "grid_iae": float(g["grid_iae"].mean()),
                "grid_iae_path_se": float(path_means.std(ddof=1) / math.sqrt(len(path_means))) if len(path_means) > 1 else float("nan"),
                "median_max_error": float(g["max_error"].median()),
                "p90_max_error": float(g["max_error"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "proxy_mode", "calibration_frac", "grid_iae", "baseline"])


def run(args: argparse.Namespace) -> None:
    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    fractions = parse_float_list(args.calibration_fracs)
    grid = parse_float_list(args.grid_d_values)
    mlp_hidden_layers = parse_int_list(args.mlp_hidden_layers)
    active_baselines = [b for b in BASELINES if not (args.skip_mlp and b == "mlp_calibration")]
    endpoint_d = float(args.endpoint_d)
    if endpoint_d not in grid:
        grid = sorted(set([*grid, endpoint_d]))
    paths = build_paths(args)
    if not paths:
        raise RuntimeError("No admissible paths were generated.")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frac in fractions:
        rng = np.random.default_rng(args.random_seed)
        for path in paths:
            n = len(path.item_ids)
            for repeat in range(args.n_repeats):
                cal_idx, eval_idx = split_indices(n, frac, rng)
                r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
                j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
                r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
                j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
                try:
                    eval_curve = exact_tilt_curve(r_eval, j_eval, d_max=max(grid), n_grid=args.n_grid)
                    actual = curve_deltas(eval_curve, grid)
                    preds = baseline_predictions(
                        r_cal,
                        j_cal,
                        r_eval,
                        eval_curve,
                        grid,
                        args.fit_max_d,
                        args.n_grid,
                        active_baselines,
                        mlp_hidden_layers=mlp_hidden_layers,
                        mlp_alpha=args.mlp_alpha,
                        mlp_max_iter=args.mlp_max_iter,
                        mlp_learning_rate_init=args.mlp_learning_rate_init,
                        mlp_seed=args.mlp_seed,
                    )
                except Exception as exc:
                    skipped.append(
                        {
                            "calibration_frac": frac,
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "reason": str(exc),
                        }
                    )
                    continue
                for d in grid:
                    d = float(d)
                    for baseline in active_baselines:
                        pred = preds[baseline][d]
                        act = actual[d]
                        rows.append(
                            {
                                "dataset": args.dataset_name,
                                "proxy_mode": "external_model_proxy" if args.proxy_csv else "heldout_human_proxy",
                                "group_key": path.group_key,
                                "path_id": path.path_id,
                                "repeat": repeat,
                                "calibration_frac": float(frac),
                                "d": d,
                                "baseline": baseline,
                                "actual_gain": act,
                                "predicted_gain": pred,
                                "error": pred - act,
                                "abs_error": abs(pred - act),
                                "n_cal": int(len(cal_idx)),
                                "n_eval": int(len(eval_idx)),
                            }
                        )

    pred = pd.DataFrame(rows)
    if pred.empty:
        pd.DataFrame(skipped).to_csv(outdir / "calibration_size_strong_baselines_skipped.csv", index=False)
        raise RuntimeError("No prediction rows were produced. See skipped CSV.")
    endpoint = summarize_endpoint(pred, endpoint_d)
    curve = summarize_curve(pred)
    pred.to_csv(outdir / "calibration_size_strong_baselines_predictions.csv", index=False)
    endpoint.to_csv(outdir / "calibration_size_strong_baselines_endpoint_summary.csv", index=False)
    curve.to_csv(outdir / "calibration_size_strong_baselines_curve_summary.csv", index=False)
    (outdir / "calibration_size_strong_baselines_endpoint_summary.md").write_text(endpoint.to_markdown(index=False), encoding="utf-8")
    (outdir / "calibration_size_strong_baselines_curve_summary.md").write_text(curve.to_markdown(index=False), encoding="utf-8")
    pd.DataFrame(skipped).to_csv(outdir / "calibration_size_strong_baselines_skipped.csv", index=False)
    spec = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_name": args.dataset_name,
        "proxy_mode": "external_model_proxy" if args.proxy_csv else "heldout_human_proxy",
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "item_cols": parse_cols(args.item_cols),
        "group_cols": parse_cols(args.group_cols),
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "proxy_item_col": args.proxy_item_col,
        "proxy_score_col": args.proxy_score_col,
        "calibration_fracs": fractions,
        "n_repeats": args.n_repeats,
        "fit_max_d": args.fit_max_d,
        "grid_d_values": grid,
        "endpoint_d": endpoint_d,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "random_seed": args.random_seed,
        "n_grid": args.n_grid,
        "mlp_hidden_layers": list(mlp_hidden_layers),
        "mlp_alpha": args.mlp_alpha,
        "mlp_max_iter": args.mlp_max_iter,
        "mlp_learning_rate_init": args.mlp_learning_rate_init,
        "mlp_seed": args.mlp_seed,
        "skip_mlp": args.skip_mlp,
        "n_paths": len(paths),
        "n_prediction_rows": len(pred),
        "n_skipped": len(skipped),
        "baselines": active_baselines,
    }
    (outdir / "calibration_size_strong_baselines_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(endpoint[endpoint["d"].eq(endpoint_d)].to_string(index=False))
    print(f"Saved calibration-size strong-baseline outputs in: {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration-size sweep with strong calibration baselines.")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", required=True)
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score_mean")
    ap.add_argument("--calibration-fracs", default="0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--grid-d-values", default="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")
    ap.add_argument("--endpoint-d", type=float, default=0.10)
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--mlp-hidden-layers", default="16,16")
    ap.add_argument("--mlp-alpha", type=float, default=1e-3)
    ap.add_argument("--mlp-max-iter", type=int, default=300)
    ap.add_argument("--mlp-learning-rate-init", type=float, default=1e-3)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
