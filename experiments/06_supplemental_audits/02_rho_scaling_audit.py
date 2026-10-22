#!/usr/bin/env python3
"""  [exp24]Exp28: rho / nu^2 scaling normalization audit.

This script evaluates whether the normalized nu^2 scaling approximation is empirically
plausible in a repeated-rating dataset.

Per held-out path it computes:
- rho = Cov(J, eps)
- A_exact = sqrt(2) * (sigma_J^2 + rho) / sigma_r
- A_nu   = sqrt(2) * sigma_J * nu_hat, where nu_hat^2 = max(ICC(A,1), 0)
- B_theory, B_fit, c_loc = -B_theory when B_theory < 0
- headroom_exact = A_exact^2 / (4 c_loc) if A>0 and B<0
- headroom_nu    = A_nu^2   / (4 c_loc) if A_nu>0 and B<0

It summarizes three normalization checks:
1) |rho| / sigma_J^2 is small;
2) sigma_J is comparable across paths within a group;
3) c_loc is comparable across paths within a group.
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
    fit_local_ab,
    iter_heldout_paths,
    read_table,
    tilt_curve,
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
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--rho-rel-thresh", type=float, default=0.10)
    ap.add_argument("--cv-thresh", type=float, default=0.20)
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--t-n", type=int, default=400)
    return ap.parse_args()


def safe_cv(x: pd.Series) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    return sd / abs(mu) if abs(mu) > 1e-12 else np.nan


def summarize_group(df: pd.DataFrame, rho_rel_thresh: float, cv_thresh: float) -> dict[str, float | int | str]:
    sigmaJ = np.sqrt(df["sigma_J_sq"].clip(lower=0))
    c_loc = df["c_loc"].replace([np.inf, -np.inf], np.nan)
    return {
        "group_id": df["group_id"].iloc[0],
        "n_paths": int(df.shape[0]),
        "mean_icc_a1": float(df["icc_a1"].median()),
        "corr_A_exact_fit": float(np.corrcoef(df["A_exact"], df["A_fit"])[0, 1]) if df["A_exact"].std() > 0 and df["A_fit"].std() > 0 and len(df) >= 2 else np.nan,
        "corr_A_nu_fit": float(np.corrcoef(df["A_nu"], df["A_fit"])[0, 1]) if df["A_nu"].std() > 0 and df["A_fit"].std() > 0 and len(df) >= 2 else np.nan,
        "corr_B_fit": float(np.corrcoef(df["B_theory"], df["B_fit"])[0, 1]) if df["B_theory"].std() > 0 and df["B_fit"].std() > 0 and len(df) >= 2 else np.nan,
        "median_abs_rho_rel": float(df["abs_rho_rel"].median()),
        "frac_abs_rho_rel_le_thresh": float((df["abs_rho_rel"] <= rho_rel_thresh).mean()),
        "cv_sigma_J": float(safe_cv(sigmaJ)),
        "cv_c_loc": float(safe_cv(c_loc.dropna())),
        "median_headroom_rel_error_nu": float(df["headroom_rel_error_nu"].replace([np.inf, -np.inf], np.nan).median()),
        "rho_small_flag": float((df["abs_rho_rel"] <= rho_rel_thresh).mean() >= 0.8),
        "sigmaJ_comparable_flag": float(safe_cv(sigmaJ) <= cv_thresh) if np.isfinite(safe_cv(sigmaJ)) else np.nan,
        "c_loc_comparable_flag": float(safe_cv(c_loc.dropna()) <= cv_thresh) if np.isfinite(safe_cv(c_loc.dropna())) else np.nan,
    }


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

    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.t_n)])
    rows = []

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
        d, q = tilt_curve(J, r, t_grid)
        A_fit, B_fit, r2 = fit_local_ab(d, q, max_d=args.fit_max_d)

        icc_a1 = path["icc_a1"]
        nu_hat_sq = max(float(icc_a1), 0.0) if np.isfinite(icc_a1) else np.nan
        A_nu = float(np.sqrt(2.0) * ab["sigma_J"] * np.sqrt(nu_hat_sq)) if np.isfinite(nu_hat_sq) else np.nan

        A_exact = ab["A"]
        B_theory = ab["B"]
        c_loc = float(-B_theory) if B_theory < 0 else np.nan
        abs_rho_rel = abs(ab["rho"]) / max(ab["sigma_J_sq"], 1e-18)

        headroom_exact = A_exact**2 / (4 * c_loc) if np.isfinite(c_loc) and c_loc > 0 and A_exact > 0 else np.nan
        headroom_nu = A_nu**2 / (4 * c_loc) if np.isfinite(c_loc) and c_loc > 0 and np.isfinite(A_nu) and A_nu > 0 else np.nan
        headroom_rel_error_nu = abs(headroom_nu - headroom_exact) / abs(headroom_exact) if np.isfinite(headroom_exact) and abs(headroom_exact) > 1e-12 and np.isfinite(headroom_nu) else np.nan

        rows.append({
            "group_id": path["group_id"],
            "heldout_rater": path["heldout_rater"],
            "n_items": path["n_items"],
            "n_raters_total": path["n_raters_total"],
            "icc_a1": icc_a1,
            "nu_hat_sq": nu_hat_sq,
            "A_exact": A_exact,
            "A_nu": A_nu,
            "A_fit": A_fit,
            "B_theory": B_theory,
            "B_fit": B_fit,
            "r2": r2,
            "rho": ab["rho"],
            "abs_rho_rel": abs_rho_rel,
            "sigma_J_sq": ab["sigma_J_sq"],
            "sigma_r_sq": ab["sigma_r_sq"],
            "c_loc": c_loc,
            "headroom_exact": headroom_exact,
            "headroom_nu": headroom_nu,
            "headroom_rel_error_nu": headroom_rel_error_nu,
        })

    paths_df = pd.DataFrame(rows)
    by_group_df = pd.DataFrame(
        [summarize_group(g, args.rho_rel_thresh, args.cv_thresh) for _, g in paths_df.groupby("group_id")]
    ) if not paths_df.empty else pd.DataFrame()

    summary_df = pd.DataFrame([{
        "n_groups": int(paths_df["group_id"].nunique()) if not paths_df.empty else 0,
        "n_paths_total": int(paths_df.shape[0]),
        "median_abs_rho_rel_pooled": float(paths_df["abs_rho_rel"].median()) if not paths_df.empty else np.nan,
        "frac_abs_rho_rel_le_thresh_pooled": float((paths_df["abs_rho_rel"] <= args.rho_rel_thresh).mean()) if not paths_df.empty else np.nan,
        "median_headroom_rel_error_nu_pooled": float(paths_df["headroom_rel_error_nu"].replace([np.inf, -np.inf], np.nan).median()) if not paths_df.empty else np.nan,
        "corr_A_exact_fit_pooled": float(np.corrcoef(paths_df["A_exact"], paths_df["A_fit"])[0, 1]) if len(paths_df) >= 2 and paths_df["A_exact"].std() > 0 and paths_df["A_fit"].std() > 0 else np.nan,
        "corr_A_nu_fit_pooled": float(np.corrcoef(paths_df["A_nu"], paths_df["A_fit"])[0, 1]) if len(paths_df) >= 2 and paths_df["A_nu"].std() > 0 and paths_df["A_fit"].std() > 0 else np.nan,
        "corr_B_fit_pooled": float(np.corrcoef(paths_df["B_theory"], paths_df["B_fit"])[0, 1]) if len(paths_df) >= 2 and paths_df["B_theory"].std() > 0 and paths_df["B_fit"].std() > 0 else np.nan,
    }])

    paths_df.to_csv(outdir / "exp24_scaling_paths.csv", index=False)
    by_group_df.to_csv(outdir / "exp24_scaling_by_group.csv", index=False)
    summary_df.to_csv(outdir / "exp24_scaling_summary.csv", index=False)

    print("Saved scaling audit to", outdir)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
