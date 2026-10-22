#!/usr/bin/env python3
"""  [exp23]Exp27: empirical monotone-condition audit for Theorem 1.

This script evaluates whether m(u)=E[eps | r=u] appears empirically non-decreasing.

For each held-out path in a repeated-rating dataset:
    J = leave-one-rater-out gold
    r = held-out rater proxy
    eps = r - J

We then estimate monotonicity using:
- quantile-binned trend (Spearman over bin means),
- adjacent-violation fraction of binned means,
- isotonic increasing fit R^2.

Recommended use:
- SummEval / QGEval / OpenMEVA / WMT MQM long CSVs.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supplemental_audit_utils import (
    canonicalize_long,
    compute_ab,
    iter_heldout_paths,
    monotone_metrics,
    read_table,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--item-col", type=str, default=None)
    ap.add_argument("--rater-col", type=str, default=None)
    ap.add_argument("--score-col", type=str, default="score")
    ap.add_argument("--group-cols", type=str, default="")
    ap.add_argument("--min-items", type=int, default=100)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--gold-estimator", type=str, default="loo_mean", choices=["loo_mean", "median_others", "same_mean"])
    ap.add_argument("--score-transform", type=str, default="raw", choices=["raw", "rater_z"])
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--random-state", type=int, default=0)
    return ap.parse_args()


def summarize_group(df: pd.DataFrame) -> dict[str, float | int | str]:
    row = {
        "group_id": df["group_id"].iloc[0],
        "n_paths": int(df.shape[0]),
        "frac_bin_spearman_pos": float((df["bin_spearman"] > 0).mean()),
        "frac_bin_spearman_ci_pos": float((df["bin_spearman_ci_lo"] > 0).mean()),
        "frac_zero_violations": float((df["bin_zero_violations"] == 1).mean()),
        "median_bin_violation_frac": float(df["bin_violation_frac"].median()),
        "median_isotonic_r2": float(df["isotonic_r2"].median()),
        "median_bin_spearman": float(df["bin_spearman"].median()),
        "frac_B_neg": float((df["B_theory"] < 0).mean()),
    }
    if df["B_theory"].std() > 0 and df["bin_spearman"].std() > 0:
        row["corr_B_vs_bin_spearman"] = float(np.corrcoef(df["B_theory"], df["bin_spearman"])[0, 1])
    else:
        row["corr_B_vs_bin_spearman"] = np.nan
    return row


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_table(args.data)
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()] or None
    df, group_cols_used = canonicalize_long(
        raw,
        item_col=args.item_col,
        rater_col=args.rater_col,
        score_col=args.score_col,
        group_cols=group_cols,
    )

    path_rows = []
    bin_rows = []

    for path in iter_heldout_paths(
        df,
        min_items=args.min_items,
        min_raters=args.min_raters,
        score_transform=args.score_transform,
        gold_estimator=args.gold_estimator,
    ):
        J = path["J"]
        r = path["r"]
        ab = compute_ab(J, r)
        metrics, bins = monotone_metrics(
            J, r,
            n_bins=args.n_bins,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
        row = {
            "group_id": path["group_id"],
            "heldout_rater": path["heldout_rater"],
            "n_items": path["n_items"],
            "n_raters_total": path["n_raters_total"],
            "icc_a1": path["icc_a1"],
            "A_theory": ab["A"],
            "B_theory": ab["B"],
            "rho": ab["rho"],
            "sigma_J_sq": ab["sigma_J_sq"],
            "sigma_r_sq": ab["sigma_r_sq"],
        }
        row.update(metrics)
        path_rows.append(row)

        if not bins.empty:
            bb = bins.copy()
            bb["group_id"] = path["group_id"]
            bb["heldout_rater"] = path["heldout_rater"]
            bin_rows.append(bb)

    paths_df = pd.DataFrame(path_rows)
    bins_df = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
    by_group_df = pd.DataFrame([summarize_group(g) for _, g in paths_df.groupby("group_id")]) if not paths_df.empty else pd.DataFrame()
    summary_df = pd.DataFrame([{
        "n_groups": int(paths_df["group_id"].nunique()) if not paths_df.empty else 0,
        "n_paths_total": int(paths_df.shape[0]),
        "frac_bin_spearman_pos_pooled": float((paths_df["bin_spearman"] > 0).mean()) if not paths_df.empty else np.nan,
        "frac_bin_spearman_ci_pos_pooled": float((paths_df["bin_spearman_ci_lo"] > 0).mean()) if not paths_df.empty else np.nan,
        "frac_zero_violations_pooled": float((paths_df["bin_zero_violations"] == 1).mean()) if not paths_df.empty else np.nan,
        "median_violation_frac_pooled": float(paths_df["bin_violation_frac"].median()) if not paths_df.empty else np.nan,
        "median_isotonic_r2_pooled": float(paths_df["isotonic_r2"].median()) if not paths_df.empty else np.nan,
    }])

    paths_df.to_csv(outdir / "exp23_monotone_paths.csv", index=False)
    bins_df.to_csv(outdir / "exp23_monotone_bins.csv", index=False)
    by_group_df.to_csv(outdir / "exp23_monotone_by_group.csv", index=False)
    summary_df.to_csv(outdir / "exp23_monotone_summary.csv", index=False)

    print("Saved monotone audit to", outdir)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
