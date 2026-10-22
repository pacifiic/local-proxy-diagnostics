"""  [exp28]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common_experiment_utils import (
    PathData,
    delta_at_d,
    exact_tilt_curve,
    fourth_cumulant_jrrr,
    fourth_cumulant_rrrr,
    generate_human_paths,
    generate_model_paths,
    kl_to_uniform,
    mkdir,
    parse_cols,
    tilt_weights,
)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x))


def audit_one_path(path: PathData, d_max: float, n_grid: int, tail_quantiles: List[float]) -> Dict[str, Any]:
    r = path.proxy_scores.astype(float)
    j = path.gold_scores.astype(float)
    curve = exact_tilt_curve(r, j, d_max=d_max, n_grid=n_grid)

    max_abs_k4_rrrr = 0.0
    max_abs_k4_jrrr = 0.0
    min_var = float("inf")
    min_ess = float("inf")
    t_at_min_var = 0.0
    for t in curve.t:
        w = tilt_weights(r, float(t))
        var_r = float(np.sum(w * (r - weighted_mean(r, w)) ** 2))
        k4_rrrr = fourth_cumulant_rrrr(r, w)
        k4_jrrr = fourth_cumulant_jrrr(j, r, w)
        ess = 1.0 / float(np.sum(w**2))
        if var_r < min_var:
            min_var = var_r
            t_at_min_var = float(t)
        min_ess = min(min_ess, ess)
        max_abs_k4_rrrr = max(max_abs_k4_rrrr, abs(k4_rrrr))
        max_abs_k4_jrrr = max(max_abs_k4_jrrr, abs(k4_jrrr))

    baseline = float(np.mean(j))
    max_proxy = float(np.max(r))
    at_max_mask = r == np.max(r)
    tail_at_max = float(np.mean(j[at_max_mask]))
    row: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "proxy_source": path.meta.get("proxy_source"),
        "n_items": len(r),
        "baseline_q0": baseline,
        "max_proxy_value": max_proxy,
        "tail_mean_at_max_proxy": tail_at_max,
        "tail_mean_at_max_proxy_minus_baseline": tail_at_max - baseline,
        "min_var_pi_t_d_le_max": min_var,
        "t_at_min_var": t_at_min_var,
        "min_ess_d_le_max": min_ess,
        "max_abs_kappa4_rrrr_d_le_max": max_abs_k4_rrrr,
        "max_abs_kappa4_jrrr_d_le_max": max_abs_k4_jrrr,
        "d_grid_max": float(curve.d[-1]),
        "delta_at_dmax": delta_at_d(curve, d_max),
    }

    # Quantile-based tail diagnostics; useful when proxy is continuous.
    for q in tail_quantiles:
        q = float(q)
        cutoff = float(np.quantile(r, q))
        mask = r >= cutoff
        suffix = str(q).replace(".", "p")
        row[f"tail_q_{suffix}_cutoff"] = cutoff
        row[f"tail_q_{suffix}_n"] = int(mask.sum())
        row[f"tail_q_{suffix}_mean_j"] = float(np.mean(j[mask]))
        row[f"tail_q_{suffix}_minus_baseline"] = float(np.mean(j[mask]) - baseline)

    return row


def summarize(df: pd.DataFrame, tail_quantiles: List[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_key, gdf in df.groupby("group_key", dropna=False):
        row: Dict[str, Any] = {
            "group_key": group_key,
            "n_paths": len(gdf),
            "min_var_pi_t_median": gdf["min_var_pi_t_d_le_max"].median(),
            "min_ess_median": gdf["min_ess_d_le_max"].median(),
            "tail_at_max_minus_baseline_median": gdf["tail_mean_at_max_proxy_minus_baseline"].median(),
            "frac_tail_at_max_below_baseline": (gdf["tail_mean_at_max_proxy_minus_baseline"] < 0).mean(),
            "max_abs_kappa4_rrrr_median": gdf["max_abs_kappa4_rrrr_d_le_max"].median(),
            "max_abs_kappa4_jrrr_median": gdf["max_abs_kappa4_jrrr_d_le_max"].median(),
        }
        for q in tail_quantiles:
            suffix = str(q).replace(".", "p")
            col = f"tail_q_{suffix}_minus_baseline"
            row[f"median_{col}"] = gdf[col].median()
            row[f"frac_{col}_lt_0"] = (gdf[col] < 0).mean()
        rows.append(row)

    overall = {
        "group_key": "__OVERALL__",
        "n_paths": len(df),
        "min_var_pi_t_median": df["min_var_pi_t_d_le_max"].median(),
        "min_ess_median": df["min_ess_d_le_max"].median(),
        "tail_at_max_minus_baseline_median": df["tail_mean_at_max_proxy_minus_baseline"].median(),
        "frac_tail_at_max_below_baseline": (df["tail_mean_at_max_proxy_minus_baseline"] < 0).mean(),
        "max_abs_kappa4_rrrr_median": df["max_abs_kappa4_rrrr_d_le_max"].median(),
        "max_abs_kappa4_jrrr_median": df["max_abs_kappa4_jrrr_d_le_max"].median(),
    }
    for q in tail_quantiles:
        suffix = str(q).replace(".", "p")
        col = f"tail_q_{suffix}_minus_baseline"
        overall[f"median_{col}"] = df[col].median()
        overall[f"frac_{col}_lt_0"] = (df[col] < 0).mean()
    rows.append(overall)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Finite-range stability and tail-diagnostic audit for local law appendix claims.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="", help="Optional model-proxy CSV. If omitted, use held-out human paths.")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--tail-quantiles", default="0.90,0.95,0.99")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    tail_quantiles = [float(x) for x in parse_cols(args.tail_quantiles)]

    if args.proxy_csv:
        proxy = pd.read_csv(args.proxy_csv)
        paths = list(
            generate_model_paths(
                human_long=human,
                proxy_df=proxy,
                item_cols=item_cols,
                score_col=args.score_col,
                group_cols=group_cols,
                proxy_item_col=args.proxy_item_col,
                proxy_score_col=args.proxy_score_col,
                min_items=args.min_items,
            )
        )
    else:
        paths = list(
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

    rows = [audit_one_path(path, args.fit_max_d, args.n_grid, tail_quantiles) for path in paths]
    paths_df = pd.DataFrame(rows)
    paths_csv = outdir / "exp28_stability_paths.csv"
    paths_df.to_csv(paths_csv, index=False)

    summary_df = summarize(paths_df, tail_quantiles)
    summary_csv = outdir / "exp28_stability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "fit_max_d": args.fit_max_d,
        "n_grid": args.n_grid,
        "tail_quantiles": tail_quantiles,
        "n_paths": len(paths),
    }
    (outdir / "exp28_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved path-level results to {paths_csv}")
    print(f"Saved summary results to {summary_csv}")


if __name__ == "__main__":
    main()
