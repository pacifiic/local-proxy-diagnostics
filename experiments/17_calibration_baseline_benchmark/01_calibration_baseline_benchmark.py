#!/usr/bin/env python3
"""Calibration-baseline benchmark for prospective gain prediction.

This experiment compares the theorem coefficient predictor (A+B) against
natural empirical calibration baselines:

- empirical calibration curve: use the calibration split's exact gain curve
- empirical two-parameter fit: fit alpha sqrt(d) + beta d to the calibration curve
- Pearson/Spearman correlation A-only baselines
- isotonic score calibration: learn h(r) ~= E[J | r] on calibration items, then
  integrate h under the evaluation proxy tilt

The split protocol mirrors exp27: estimate all deployable predictors on a
calibration split and evaluate against the exact finite-tilt gain on held-out
evaluation items.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import (  # noqa: E402
    exact_tilt_curve,
    find_t_for_dmax,
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
    if any(v < 0 for v in vals):
        raise ValueError("d targets must be nonnegative.")
    return vals


def exact_delta_at_d(proxy: np.ndarray, gold: np.ndarray, d: float) -> float:
    """Exact finite-tilt gain at KL radius d."""
    r = np.asarray(proxy, dtype=float)
    j = np.asarray(gold, dtype=float)
    if d <= 0:
        return 0.0
    t = find_t_for_dmax(r, float(d))
    w = tilt_weights(r, t)
    return float(np.sum(w * j) - np.mean(j))


def exact_weights_at_d(proxy: np.ndarray, d: float) -> np.ndarray:
    r = np.asarray(proxy, dtype=float)
    if d <= 0:
        return np.ones(len(r), dtype=float) / float(len(r))
    t = find_t_for_dmax(r, float(d))
    return tilt_weights(r, t)


def fit_isotonic(cal_proxy: np.ndarray, cal_gold: np.ndarray) -> IsotonicRegression:
    # Increasing=True matches the paper's fixed proxy convention: higher proxy
    # scores should indicate better candidates. Out-of-range evaluation scores
    # are clipped to the nearest calibrated endpoint.
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(cal_proxy, dtype=float), np.asarray(cal_gold, dtype=float))
    return model


def predict_isotonic_gain(model: IsotonicRegression, eval_proxy: np.ndarray, d: float) -> float:
    h = model.predict(np.asarray(eval_proxy, dtype=float))
    w = exact_weights_at_d(eval_proxy, d)
    return float(np.sum(w * h) - np.mean(h))


def safe_corr(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    if rank:
        val = spearman_corr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    else:
        val = float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])
    if not np.isfinite(val):
        return 0.0
    return val


def baseline_predictions(
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    r_eval: np.ndarray,
    d: float,
    fit_max_d: float,
) -> dict[str, float]:
    coeff = theory_coefficients(j_cal, r_cal)

    curve_cal = exact_tilt_curve(r_cal, j_cal, d_max=fit_max_d)
    fit_cal = fit_local_law(curve_cal.d, curve_cal.q, fit_max_d=fit_max_d)
    iso = fit_isotonic(r_cal, j_cal)

    sigma_j = float(np.std(j_cal))
    rho_pearson = safe_corr(r_cal, j_cal, rank=False)
    rho_spearman = safe_corr(r_cal, j_cal, rank=True)
    a_pearson = math.sqrt(2.0) * sigma_j * rho_pearson
    a_spearman = math.sqrt(2.0) * sigma_j * rho_spearman

    sqrt_d = math.sqrt(max(0.0, d))
    return {
        "A+B": coeff.a_exact * sqrt_d + coeff.b_theory * d,
        "A-only": coeff.a_exact * sqrt_d,
        "A_nu": coeff.a_nu * sqrt_d,
        "empirical_curve": exact_delta_at_d(r_cal, j_cal, d),
        "empirical_fit": fit_cal.a_fit * sqrt_d + fit_cal.b_fit * d,
        "pearson_corr": a_pearson * sqrt_d,
        "spearman_corr": a_spearman * sqrt_d,
        "isotonic_calibration": predict_isotonic_gain(iso, r_eval, d),
        "zero": 0.0,
    }


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


def summarize(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in pred_df.groupby(["dataset", "proxy_mode", "d", "baseline"], dropna=False):
        dataset, proxy_mode, d, baseline = key
        finite = g[np.isfinite(g["actual_gain"]) & np.isfinite(g["predicted_gain"])]
        rows.append(
            {
                "dataset": dataset,
                "proxy_mode": proxy_mode,
                "d": float(d),
                "baseline": baseline,
                "n_rows": int(len(finite)),
                "n_paths": int(finite["path_id"].nunique()) if not finite.empty else 0,
                "mean_actual_gain": float(finite["actual_gain"].mean()) if not finite.empty else float("nan"),
                "mean_predicted_gain": float(finite["predicted_gain"].mean()) if not finite.empty else float("nan"),
                "mae": float(finite["abs_error"].mean()) if not finite.empty else float("nan"),
                "median_abs_error": float(finite["abs_error"].median()) if not finite.empty else float("nan"),
                "sign_accuracy": float((np.sign(finite["predicted_gain"]) == np.sign(finite["actual_gain"])).mean())
                if not finite.empty
                else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["dataset", "proxy_mode", "d", "mae", "baseline"])


def run(args: argparse.Namespace) -> None:
    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    d_targets = parse_float_list(args.d_targets)
    paths = build_paths(args)
    if not paths:
        raise RuntimeError("No admissible paths were generated.")

    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in paths:
        n = len(path.item_ids)
        for repeat in range(args.n_repeats):
            cal_idx, eval_idx = split_indices(n, args.calibration_frac, rng)
            r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
            r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
            j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
            for d in d_targets:
                try:
                    actual = exact_delta_at_d(r_eval, j_eval, d)
                    preds = baseline_predictions(r_cal, j_cal, r_eval, d, args.fit_max_d)
                except Exception as exc:
                    skipped.append(
                        {
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": d,
                            "reason": str(exc),
                        }
                    )
                    continue
                for baseline in BASELINES:
                    pred = preds[baseline]
                    rows.append(
                        {
                            "dataset": args.dataset_name,
                            "proxy_mode": "external_model_proxy" if args.proxy_csv else "heldout_human_proxy",
                            "group_key": path.group_key,
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "d": float(d),
                            "baseline": baseline,
                            "actual_gain": actual,
                            "predicted_gain": pred,
                            "error": pred - actual,
                            "abs_error": abs(pred - actual),
                            "n_cal": int(len(cal_idx)),
                            "n_eval": int(len(eval_idx)),
                        }
                    )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        pd.DataFrame(skipped).to_csv(outdir / "calibration_baseline_skipped.csv", index=False)
        raise RuntimeError("No prediction rows were produced. See skipped CSV.")

    summary_df = summarize(pred_df)
    pred_df.to_csv(outdir / "calibration_baseline_predictions.csv", index=False)
    summary_df.to_csv(outdir / "calibration_baseline_summary.csv", index=False)
    (outdir / "calibration_baseline_summary.md").write_text(summary_df.to_markdown(index=False), encoding="utf-8")

    wide = summary_df.pivot_table(
        index=["dataset", "proxy_mode", "d"],
        columns="baseline",
        values="mae",
        aggfunc="mean",
    ).reset_index()
    wide.to_csv(outdir / "calibration_baseline_mae_wide.csv", index=False)
    (outdir / "calibration_baseline_mae_wide.md").write_text(wide.to_markdown(index=False), encoding="utf-8")

    pd.DataFrame(skipped).to_csv(outdir / "calibration_baseline_skipped.csv", index=False)

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
        "calibration_frac": args.calibration_frac,
        "n_repeats": args.n_repeats,
        "fit_max_d": args.fit_max_d,
        "d_targets": d_targets,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "random_seed": args.random_seed,
        "n_paths": len(paths),
        "n_prediction_rows": len(pred_df),
        "n_skipped": len(skipped),
        "baselines": BASELINES,
    }
    (outdir / "calibration_baseline_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Saved calibration-baseline benchmark outputs in: {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark calibration baselines against A+B gain prediction.")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", required=True)
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score_mean")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--d-targets", default="0.05,0.10")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
