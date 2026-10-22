"""Finite-radius residuals of the local law.

This script checks where the local approximation itself begins to break down:

    Q(d)-Q(0) ~= A sqrt(d) + B d.

It computes exact exponential-tilt curves on each admissible path and reports
max/mean/RMSE residuals over D in {0.05, 0.10, 0.20}.  If a finite empirical
path cannot realize the largest requested radius, the script keeps the smaller
reachable windows instead of dropping the path.  It also includes A-only,
reliability-only A_nu, zero-gain, least-squares A+B, and optional third-order
A+B+C fits as descriptive controls.

Example:
    python experiments/12_accept_boost/02_curve_residual_breakdown.py \
      --human-long-csv data/summeval_long.csv \
      --item-cols article_id,model_id --group-cols dimension,rater_type \
      --outdir all_results_reproduced/12_accept_boost/residual_human
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure lib/ is on sys.path before importing shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accept_boost_utils import load_paths_from_args, mkdir, parse_float_list, save_markdown
from common_experiment_utils import exact_tilt_curve, theory_coefficients


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) < 2:
        return float("nan")
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 if sst == 0.0 else 1.0 - sse / sst


def _fit_predict(d: np.ndarray, y: np.ndarray, order: str) -> tuple[np.ndarray, np.ndarray]:
    if order == "fit_ab":
        X = np.column_stack([np.sqrt(d), d])
    elif order == "fit_abc":
        X = np.column_stack([np.sqrt(d), d, d ** 1.5])
    else:
        raise ValueError(order)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, X @ coef


def _reachable_curve(r: np.ndarray, j: np.ndarray, d_ranges: list[float], n_grid: int):
    """Return an exact tilt curve for the largest reachable requested window."""
    errors: list[str] = []
    for D in sorted(set(float(d) for d in d_ranges), reverse=True):
        try:
            return exact_tilt_curve(r, j, d_max=D, n_grid=n_grid), D
        except Exception as exc:
            errors.append(f"D={D:g}: {exc}")
    raise RuntimeError("Could not compute any requested residual window; " + "; ".join(errors))


def analyze_path(path, d_ranges: list[float], n_grid: int) -> list[dict]:
    r = np.asarray(path.proxy_scores, dtype=float)
    j = np.asarray(path.gold_scores, dtype=float)
    coeff = theory_coefficients(j, r)
    curve, reached_D = _reachable_curve(r, j, d_ranges=d_ranges, n_grid=n_grid)
    q_delta = curve.q - float(curve.q[0])
    rows: list[dict] = []
    for D in d_ranges:
        if float(D) > float(curve.d[-1]) + 1e-12:
            continue
        mask = (curve.d > 0) & (curve.d <= D) & np.isfinite(q_delta)
        if int(mask.sum()) < 5:
            continue
        d = curve.d[mask]
        y = q_delta[mask]
        pred_theory = coeff.a_exact * np.sqrt(d) + coeff.b_theory * d
        pred_aonly = coeff.a_exact * np.sqrt(d)
        pred_anu = coeff.a_nu * np.sqrt(d)
        _, pred_fit_ab = _fit_predict(d, y, "fit_ab")
        _, pred_fit_abc = _fit_predict(d, y, "fit_abc")
        pred_zero = np.zeros_like(y)
        for name, pred in {
            "theory_ab": pred_theory,
            "aonly": pred_aonly,
            "fit_ab": pred_fit_ab,
            "anu": pred_anu,
            "fit_abc": pred_fit_abc,
            "zero": pred_zero,
        }.items():
            resid = y - pred
            rows.append({
                "group_key": path.group_key,
                "path_id": path.path_id,
                "proxy_source": path.meta.get("proxy_source"),
                "D": float(D),
                "reached_D": float(reached_D),
                "curve_max_d": float(curve.d[-1]),
                "predictor": name,
                "n_points": int(len(y)),
                "A_theory": coeff.a_exact,
                "B_theory": coeff.b_theory,
                "max_abs_resid": float(np.max(np.abs(resid))),
                "mean_abs_resid": float(np.mean(np.abs(resid))),
                "rmse_resid": float(np.sqrt(np.mean(resid ** 2))),
                "r2": _r2(y, pred),
                "curve_range_abs": float(np.max(np.abs(y))),
            })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group_key, D, predictor), g in df.groupby(["group_key", "D", "predictor"], dropna=False):
        rows.append({
            "group_key": group_key,
            "D": D,
            "predictor": predictor,
            "n_paths": g["path_id"].nunique(),
            "median_max_abs_resid": g["max_abs_resid"].median(),
            "q90_max_abs_resid": g["max_abs_resid"].quantile(0.90),
            "median_mean_abs_resid": g["mean_abs_resid"].median(),
            "median_rmse_resid": g["rmse_resid"].median(),
            "median_r2": g["r2"].median(),
            "median_curve_range_abs": g["curve_range_abs"].median(),
        })
    group_summary = pd.DataFrame(rows)
    rows = []
    for (D, predictor), g in df.groupby(["D", "predictor"], dropna=False):
        rows.append({
            "group_key": "__OVERALL__",
            "D": D,
            "predictor": predictor,
            "n_paths": g["path_id"].nunique(),
            "median_max_abs_resid": g["max_abs_resid"].median(),
            "q90_max_abs_resid": g["max_abs_resid"].quantile(0.90),
            "median_mean_abs_resid": g["mean_abs_resid"].median(),
            "median_rmse_resid": g["rmse_resid"].median(),
            "median_r2": g["r2"].median(),
            "median_curve_range_abs": g["curve_range_abs"].median(),
        })
    overall = pd.DataFrame(rows)
    return pd.concat([group_summary, overall], ignore_index=True)


def plot_residual(summary: pd.DataFrame, outdir: Path) -> None:
    overall = summary[summary["group_key"].eq("__OVERALL__")].copy()
    keep = ["theory_ab", "aonly", "fit_ab", "fit_abc"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for pred in keep:
        sub = overall[overall["predictor"].eq(pred)].sort_values("D")
        if sub.empty:
            continue
        ax.plot(sub["D"], sub["median_max_abs_resid"], marker="o", linewidth=1.8, label=pred)
    ax.set_xlabel("Maximum radius D for residual window")
    ax.set_ylabel("Median max |residual| over d <= D")
    ax.set_title("Finite-radius residual of the local law")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_curve_residual_breakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_curve_residual_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Finite-radius residual/breakdown metrics for exact tilt curves.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--d-ranges", default="0.05,0.10,0.20")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--n-grid", type=int, default=320)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    d_ranges = parse_float_list(args.d_ranges)
    paths = load_paths_from_args(args)
    print(f"Loaded {len(paths)} paths; residual windows={d_ranges}")

    rows = []
    for path in paths:
        try:
            rows.extend(analyze_path(path, d_ranges=d_ranges, n_grid=args.n_grid))
        except Exception as exc:
            print(f"[skip] {path.path_id}: {exc}")
    if not rows:
        raise RuntimeError("No residual rows produced.")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "curve_residual_paths.csv", index=False)

    summ = summarize(df)
    summ.to_csv(outdir / "curve_residual_summary.csv", index=False)
    save_markdown(summ, outdir / "curve_residual_summary.md")
    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "item_cols": args.item_cols,
        "group_cols": args.group_cols,
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "proxy_item_col": args.proxy_item_col,
        "proxy_score_col": args.proxy_score_col,
        "d_ranges": d_ranges,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "n_grid": args.n_grid,
        "n_paths": len(paths),
        "n_rows": len(df),
        "outputs": {
            "paths": str(outdir / "curve_residual_paths.csv"),
            "summary": str(outdir / "curve_residual_summary.csv"),
        },
    }
    (outdir / "curve_residual_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    plot_residual(summ, outdir)

    print(summ[summ["group_key"].eq("__OVERALL__")].to_string(index=False))
    print(f"Saved residual outputs in: {outdir}")


if __name__ == "__main__":
    main()
