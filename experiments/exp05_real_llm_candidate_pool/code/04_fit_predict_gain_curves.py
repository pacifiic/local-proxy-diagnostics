#!/usr/bin/env python3
"""Fit and evaluate prompt-conditional LLM candidate-pool gain curves."""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
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

from common_experiment_utils import fit_local_law, mkdir, spearman_corr, tilt_weights  # noqa: E402


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


@dataclass
class ConditionalCurve:
    t: np.ndarray
    d: np.ndarray
    q: np.ndarray
    ess_mean: np.ndarray


@dataclass
class ConditionalCoefficients:
    a_exact: float
    b_theory: float
    a_nu: float
    cov_jr: float
    var_r: float
    var_j: float
    kappa_r: float
    kappa_jrr: float


def ensure_empty_outdir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Use --overwrite or choose a fresh --outdir."
        )
    return mkdir(path)


def parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def parse_int_list(text: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not out or any(v <= 0 for v in out):
        raise ValueError(f"Invalid hidden-layer list: {text}")
    return out


def grouped_arrays(df: pd.DataFrame, prompt_col: str, proxy_col: str, gold_col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, g in df.groupby(prompt_col, sort=True, dropna=False):
        if len(g) < 2:
            continue
        r = g[proxy_col].to_numpy(dtype=float)
        j = g[gold_col].to_numpy(dtype=float)
        mask = np.isfinite(r) & np.isfinite(j)
        if int(mask.sum()) >= 2:
            groups.append((r[mask], j[mask]))
    if not groups:
        raise ValueError("No prompt groups with at least two finite candidates.")
    return groups


def central_moments(j: np.ndarray, r: np.ndarray) -> tuple[float, float, float, float, float]:
    jm = float(np.mean(j))
    rm = float(np.mean(r))
    jc = j - jm
    rc = r - rm
    var_j = float(np.mean(jc**2))
    var_r = float(np.mean(rc**2))
    cov = float(np.mean(jc * rc))
    kappa_r = float(np.mean(rc**3))
    kappa_jrr = float(np.mean(jc * rc**2))
    return var_j, var_r, cov, kappa_r, kappa_jrr


def conditional_coefficients(df: pd.DataFrame, prompt_col: str, proxy_col: str, gold_col: str) -> ConditionalCoefficients:
    parts = grouped_arrays(df, prompt_col, proxy_col, gold_col)
    vals = np.asarray([central_moments(j, r) for r, j in parts], dtype=float)
    var_j, var_r, cov, kappa_r, kappa_jrr = [float(np.mean(vals[:, i])) for i in range(vals.shape[1])]
    if var_r <= 0.0:
        raise ValueError("Mean within-prompt proxy variance is zero.")
    sigma_r = math.sqrt(var_r)
    sigma_j = math.sqrt(max(var_j, 0.0))
    a_exact = math.sqrt(2.0) * cov / sigma_r
    a_nu = math.sqrt(2.0) * sigma_j * (sigma_j / sigma_r) if sigma_r > 0 else float("nan")
    b_theory = kappa_jrr / var_r - 2.0 * cov * kappa_r / (3.0 * var_r**2)
    return ConditionalCoefficients(
        a_exact=a_exact,
        b_theory=b_theory,
        a_nu=a_nu,
        cov_jr=cov,
        var_r=var_r,
        var_j=var_j,
        kappa_r=kappa_r,
        kappa_jrr=kappa_jrr,
    )


def prompt_kl_and_q(r: np.ndarray, j: np.ndarray, t: float) -> tuple[float, float, float]:
    w = tilt_weights(r, t)
    q = float(np.sum(w * j))
    base_log_prob = math.log(float(len(r)))
    kl = float(np.sum(w[w > 0] * (np.log(w[w > 0]) + base_log_prob)))
    ess = 1.0 / float(np.sum(w**2))
    return kl, q, ess


def conditional_stats(groups: Sequence[tuple[np.ndarray, np.ndarray]], t: float) -> tuple[float, float, float]:
    vals = [prompt_kl_and_q(r, j, t) for r, j in groups]
    arr = np.asarray(vals, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


def conditional_tilt_curve(
    df: pd.DataFrame,
    prompt_col: str,
    proxy_col: str,
    gold_col: str,
    d_max: float,
    n_grid: int,
) -> ConditionalCurve:
    groups = grouped_arrays(df, prompt_col, proxy_col, gold_col)
    if d_max <= 0:
        d, q, ess = conditional_stats(groups, 0.0)
        return ConditionalCurve(t=np.asarray([0.0]), d=np.asarray([d]), q=np.asarray([q]), ess_mean=np.asarray([ess]))
    lo, hi = 0.0, 1.0
    while conditional_stats(groups, hi)[0] < d_max:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket t for requested prompt-conditional d_max.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if conditional_stats(groups, mid)[0] < d_max:
            lo = mid
        else:
            hi = mid
    t_grid = np.linspace(0.0, hi, n_grid)
    d_vals: list[float] = []
    q_vals: list[float] = []
    ess_vals: list[float] = []
    for t in t_grid:
        d, q, ess = conditional_stats(groups, float(t))
        d_vals.append(d)
        q_vals.append(q)
        ess_vals.append(ess)
    return ConditionalCurve(t=t_grid, d=np.asarray(d_vals), q=np.asarray(q_vals), ess_mean=np.asarray(ess_vals))


def curve_deltas(curve: ConditionalCurve, d_values: Sequence[float]) -> dict[float, float]:
    q0 = float(curve.q[0])
    return {float(d): float(np.interp(float(d), curve.d, curve.q) - q0) for d in d_values}


def t_at_d(curve: ConditionalCurve, d: float) -> float:
    if d <= 0:
        return 0.0
    return float(np.interp(float(d), curve.d, curve.t))


def calibrator_predictions(
    cal_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_curve: ConditionalCurve,
    d_values: Sequence[float],
    prompt_col: str,
    proxy_col: str,
    gold_col: str,
    kind: str,
    *,
    hidden_layers: tuple[int, ...],
    mlp_alpha: float,
    mlp_max_iter: int,
    mlp_seed: int,
) -> dict[float, float]:
    x_cal = cal_df[proxy_col].to_numpy(dtype=float).reshape(-1, 1)
    y_cal = cal_df[gold_col].to_numpy(dtype=float)
    x_eval = eval_df[proxy_col].to_numpy(dtype=float).reshape(-1, 1)
    if kind == "isotonic":
        model: Any = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(x_cal.ravel(), y_cal)
        h = model.predict(x_eval.ravel())
    elif kind == "mlp":
        if len(x_cal) < 8 or float(np.nanstd(x_cal)) == 0.0 or float(np.nanstd(y_cal)) == 0.0:
            return {float(d): 0.0 for d in d_values}
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_cal)
        x_test = x_scaler.transform(x_eval)
        y_mean = float(np.mean(y_cal))
        y_std = float(np.std(y_cal))
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=mlp_alpha,
            max_iter=mlp_max_iter,
            random_state=mlp_seed,
            learning_rate_init=1e-3,
            early_stopping=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, (y_cal - y_mean) / y_std)
        h = y_mean + y_std * model.predict(x_test)
    else:
        raise ValueError(f"Unknown calibrator kind: {kind}")

    tmp = eval_df[[prompt_col, proxy_col]].copy()
    tmp["_h"] = h
    base_vals = tmp.groupby(prompt_col, dropna=False)["_h"].mean().to_numpy(dtype=float)
    base = float(np.mean(base_vals))
    out: dict[float, float] = {}
    for d in d_values:
        t = t_at_d(eval_curve, float(d))
        qs: list[float] = []
        for _, g in tmp.groupby(prompt_col, sort=True, dropna=False):
            w = tilt_weights(g[proxy_col].to_numpy(dtype=float), t)
            qs.append(float(np.sum(w * g["_h"].to_numpy(dtype=float))))
        out[float(d)] = float(np.mean(qs) - base)
    return out


def safe_corr(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    val = spearman_corr(x, y) if rank else float(np.corrcoef(x, y)[0, 1])
    return float(val) if np.isfinite(val) else 0.0


def split_prompt_ids(prompt_ids: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[set[str], set[str]]:
    ids = np.asarray(sorted(pd.unique(prompt_ids)), dtype=str)
    order = np.arange(len(ids))
    rng.shuffle(order)
    n_cal = int(round(frac * len(ids)))
    n_cal = max(1, min(n_cal, len(ids) - 1))
    cal = set(ids[order[:n_cal]].tolist())
    eva = set(ids[order[n_cal:]].tolist())
    return cal, eva


def predictions_for_split(
    df: pd.DataFrame,
    cal_ids: set[str],
    eval_ids: set[str],
    args: argparse.Namespace,
    d_values: Sequence[float],
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cal = df[df[args.prompt_col].astype(str).isin(cal_ids)].copy()
    eva = df[df[args.prompt_col].astype(str).isin(eval_ids)].copy()
    coeff = conditional_coefficients(cal, args.prompt_col, args.proxy_col, args.gold_col)
    eval_coeff = conditional_coefficients(eva, args.prompt_col, args.proxy_col, args.gold_col)
    cal_curve_max_d = max(float(args.fit_max_d), max(float(d) for d in d_values))
    cal_curve = conditional_tilt_curve(cal, args.prompt_col, args.proxy_col, args.gold_col, cal_curve_max_d, args.n_grid)
    eval_curve = conditional_tilt_curve(eva, args.prompt_col, args.proxy_col, args.gold_col, max(d_values), args.n_grid)
    actual = curve_deltas(eval_curve, d_values)
    cal_deltas = curve_deltas(cal_curve, d_values)
    fit = fit_local_law(cal_curve.d, cal_curve.q, fit_max_d=args.fit_max_d)
    iso = calibrator_predictions(
        cal,
        eva,
        eval_curve,
        d_values,
        args.prompt_col,
        args.proxy_col,
        args.gold_col,
        "isotonic",
        hidden_layers=parse_int_list(args.mlp_hidden_layers),
        mlp_alpha=args.mlp_alpha,
        mlp_max_iter=args.mlp_max_iter,
        mlp_seed=args.mlp_seed + repeat,
    )
    mlp = calibrator_predictions(
        cal,
        eva,
        eval_curve,
        d_values,
        args.prompt_col,
        args.proxy_col,
        args.gold_col,
        "mlp",
        hidden_layers=parse_int_list(args.mlp_hidden_layers),
        mlp_alpha=args.mlp_alpha,
        mlp_max_iter=args.mlp_max_iter,
        mlp_seed=args.mlp_seed + repeat,
    )
    sigma_j = math.sqrt(max(coeff.var_j, 0.0))
    flat_r = cal[args.proxy_col].to_numpy(dtype=float)
    flat_j = cal[args.gold_col].to_numpy(dtype=float)
    a_pearson = math.sqrt(2.0) * sigma_j * safe_corr(flat_r, flat_j, rank=False)
    a_spearman = math.sqrt(2.0) * sigma_j * safe_corr(flat_r, flat_j, rank=True)

    rows: list[dict[str, Any]] = []
    for d in d_values:
        d = float(d)
        sqrt_d = math.sqrt(max(0.0, d))
        preds = {
            "A+B": coeff.a_exact * sqrt_d + coeff.b_theory * d,
            "A-only": coeff.a_exact * sqrt_d,
            "A_nu": coeff.a_nu * sqrt_d,
            "empirical_curve": cal_deltas[d],
            "empirical_fit": fit.a_fit * sqrt_d + fit.b_fit * d,
            "pearson_corr": a_pearson * sqrt_d,
            "spearman_corr": a_spearman * sqrt_d,
            "isotonic_calibration": iso[d],
            "mlp_calibration": mlp[d],
            "zero": 0.0,
        }
        for baseline in BASELINES:
            pred = float(preds[baseline])
            act = float(actual[d])
            rows.append(
                {
                    "repeat": repeat,
                    "d": d,
                    "baseline": baseline,
                    "predicted_gain": pred,
                    "actual_gain": act,
                    "error": pred - act,
                    "abs_error": abs(pred - act),
                    "n_cal_prompts": len(cal_ids),
                    "n_eval_prompts": len(eval_ids),
                    "n_cal_candidates": int(len(cal)),
                    "n_eval_candidates": int(len(eva)),
                }
            )
    coef_row = {
        "repeat": repeat,
        "n_cal_prompts": len(cal_ids),
        "n_eval_prompts": len(eval_ids),
        "a_cal": coeff.a_exact,
        "b_cal": coeff.b_theory,
        "a_nu_cal": coeff.a_nu,
        "a_eval": eval_coeff.a_exact,
        "b_eval": eval_coeff.b_theory,
        "a_nu_eval": eval_coeff.a_nu,
        "cal_fit_a": fit.a_fit,
        "cal_fit_b": fit.b_fit,
        "cal_fit_r2": fit.r2,
    }
    return rows, coef_row


def summarize_endpoint(pred: pd.DataFrame, endpoint_d: float) -> pd.DataFrame:
    endpoint = pred[np.isclose(pred["d"], endpoint_d)].copy()
    rows: list[dict[str, Any]] = []
    for baseline, g in endpoint.groupby("baseline", dropna=False):
        rows.append(
            {
                "baseline": baseline,
                "d": endpoint_d,
                "n_rows": int(len(g)),
                "mae": float(g["abs_error"].mean()),
                "median_abs_error": float(g["abs_error"].median()),
                "repeat_se": float(g.groupby("repeat")["abs_error"].mean().std(ddof=1) / math.sqrt(g["repeat"].nunique()))
                if g["repeat"].nunique() > 1
                else float("nan"),
                "mean_actual_gain": float(g["actual_gain"].mean()),
                "sign_accuracy": float((np.sign(g["predicted_gain"]) == np.sign(g["actual_gain"])).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "baseline"])


def summarize_curve(pred: pd.DataFrame) -> pd.DataFrame:
    per = (
        pred.groupby(["baseline", "repeat"], dropna=False)
        .agg(grid_iae=("abs_error", "mean"), max_error=("abs_error", "max"))
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for baseline, g in per.groupby("baseline", dropna=False):
        rows.append(
            {
                "baseline": baseline,
                "n_curves": int(len(g)),
                "grid_iae": float(g["grid_iae"].mean()),
                "grid_iae_repeat_se": float(g["grid_iae"].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else float("nan"),
                "median_max_error": float(g["max_error"].median()),
                "p90_max_error": float(g["max_error"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values(["grid_iae", "baseline"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate prompt-conditional LLM candidate-pool gain-curve prediction.")
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--prompt-col", default="prompt_id")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--proxy-col", default="proxy_score")
    ap.add_argument("--gold-col", default="gold_score")
    ap.add_argument("--use-existing-split", action="store_true")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--grid-d-values", default="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10")
    ap.add_argument("--endpoint-d", type=float, default=0.10)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--mlp-hidden-layers", default="16,16")
    ap.add_argument("--mlp-alpha", type=float, default=1e-3)
    ap.add_argument("--mlp-max-iter", type=int, default=300)
    ap.add_argument("--mlp-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    if not (0.0 < args.calibration_frac < 1.0):
        raise ValueError("--calibration-frac must be in (0,1).")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be positive.")
    if args.n_grid < 2:
        raise ValueError("--n-grid must be at least 2.")
    if args.fit_max_d <= 0.0:
        raise ValueError("--fit-max-d must be positive.")
    if args.endpoint_d <= 0.0:
        raise ValueError("--endpoint-d must be positive.")
    df = pd.read_csv(args.candidates_csv)
    required = {args.prompt_col, args.proxy_col, args.gold_col}
    if args.use_existing_split:
        required.add(args.split_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidates CSV missing columns: {sorted(missing)}")
    before_drop = len(df)
    df = df.dropna(subset=[args.prompt_col, args.proxy_col, args.gold_col]).copy()
    if df.empty:
        raise ValueError("No rows remain after dropping missing prompt/proxy/gold values.")
    d_values = parse_float_list(args.grid_d_values)
    if not any(np.isclose(args.endpoint_d, d) for d in d_values):
        d_values = sorted(set([*d_values, args.endpoint_d]))
    rows: list[dict[str, Any]] = []
    coefs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.random_seed)
    prompt_ids = df[args.prompt_col].astype(str).to_numpy()
    n_repeats = 1 if args.use_existing_split else args.n_repeats
    for repeat in range(n_repeats):
        if args.use_existing_split:
            cal_ids = set(df.loc[df[args.split_col].astype(str).eq("calibration"), args.prompt_col].astype(str))
            eval_ids = set(df.loc[df[args.split_col].astype(str).eq("eval"), args.prompt_col].astype(str))
            if not cal_ids or not eval_ids:
                raise ValueError("Existing split must contain both 'calibration' and 'eval' prompts.")
        else:
            cal_ids, eval_ids = split_prompt_ids(prompt_ids, args.calibration_frac, rng)
        try:
            pred_rows, coef_row = predictions_for_split(df, cal_ids, eval_ids, args, d_values, repeat)
        except Exception as exc:
            skipped.append({"repeat": repeat, "reason": str(exc)})
            continue
        rows.extend(pred_rows)
        coefs.append(coef_row)
    pred = pd.DataFrame(rows)
    pd.DataFrame(skipped).to_csv(outdir / "llm_gain_curve_skipped.csv", index=False)
    if pred.empty:
        raise RuntimeError("No prediction rows produced; see llm_gain_curve_skipped.csv")
    endpoint = summarize_endpoint(pred, args.endpoint_d)
    curve = summarize_curve(pred)
    coef_df = pd.DataFrame(coefs)
    pred.to_csv(outdir / "llm_gain_curve_predictions.csv", index=False)
    endpoint.to_csv(outdir / "llm_gain_curve_endpoint_summary.csv", index=False)
    curve.to_csv(outdir / "llm_gain_curve_curve_summary.csv", index=False)
    coef_df.to_csv(outdir / "llm_gain_curve_coefficients.csv", index=False)
    (outdir / "llm_gain_curve_endpoint_summary.md").write_text(endpoint.to_markdown(index=False), encoding="utf-8")
    (outdir / "llm_gain_curve_curve_summary.md").write_text(curve.to_markdown(index=False), encoding="utf-8")
    spec = {
        "script": "experiments/exp05_real_llm_candidate_pool/code/04_fit_predict_gain_curves.py",
        "candidates_csv": args.candidates_csv,
        "prompt_col": args.prompt_col,
        "split_col": args.split_col,
        "proxy_col": args.proxy_col,
        "gold_col": args.gold_col,
        "use_existing_split": args.use_existing_split,
        "calibration_frac": args.calibration_frac,
        "n_repeats": n_repeats,
        "random_seed": args.random_seed,
        "grid_d_values": d_values,
        "endpoint_d": args.endpoint_d,
        "fit_max_d": args.fit_max_d,
        "n_grid": args.n_grid,
        "mlp_hidden_layers": parse_int_list(args.mlp_hidden_layers),
        "mlp_alpha": args.mlp_alpha,
        "mlp_max_iter": args.mlp_max_iter,
        "mlp_seed": args.mlp_seed,
        "n_input_rows": int(len(df)),
        "n_dropped_missing_prompt_proxy_gold": int(before_drop - len(df)),
        "n_prompt_ids": int(pd.Series(prompt_ids).nunique()),
        "n_prediction_rows": int(len(pred)),
        "n_skipped": int(len(skipped)),
        "baselines": BASELINES,
    }
    (outdir / "llm_gain_curve_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(endpoint.to_string(index=False))
    print(f"Wrote LLM gain-curve outputs to: {outdir}")


if __name__ == "__main__":
    main()
