#!/usr/bin/env python3
"""  [exp19]WMT MQM direct-tilt experiments.

Main-text candidate experiment for the paper.

Protocol
--------
1. Aggregate raw MQM error annotations to segment-level higher-is-better scores:
      score = - weighted_error_sum
2. Build held-out-rater paths within each dataset split.
3. Use exact exponential tilt over items:
      pi_t(i) ∝ exp(t r_i)
4. Fit local law:
      Q(d)-Q(0) = A sqrt(d) + B d

Outputs
-------
- wmt_mqm_long.csv
- wmt_mqm_paths.csv
- wmt_mqm_traj.csv
- wmt_mqm_summary.csv
- wmt_mqm_by_dataset.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wmt_mtbench_utils import (
    build_heldout_paths_from_long,
    compute_ab,
    filter_ref_like_systems,
    fit_local_ab,
    icc1_from_long,
    load_wmt_long,
    tilt_curve,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Extracted root containing wmt-mqm-human-evaluation/")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--gold-estimator", type=str, default="loo_mean", choices=["loo_mean", "median_others"])
    ap.add_argument("--score-transform", type=str, default="raw", choices=["raw", "rater_z"])
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-other-raters", type=int, default=2)
    ap.add_argument("--exclude-ref-like", action="store_true")
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--t-n", type=int, default=400)
    return ap.parse_args()


def summarize_dataset(df: pd.DataFrame) -> dict[str, float | int | str]:
    row = {"dataset_id": df["dataset_id"].iloc[0], "n_paths": int(df.shape[0])}
    for a, b, name in [
        ("A_theory", "A_fit", "corr_A"),
        ("B_theory", "B_fit", "corr_B"),
    ]:
        x = df[a].to_numpy(dtype=float)
        y = df[b].to_numpy(dtype=float)
        row[name] = float(np.corrcoef(x, y)[0, 1]) if len(df) >= 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    row["sign_agreement"] = int(((np.sign(df["B_theory"]) == np.sign(df["B_fit"])) | ((df["B_theory"].abs() < 1e-12) & (df["B_fit"].abs() < 1e-12))).sum())
    row["frac_B_theory_neg"] = float((df["B_theory"] < 0).mean())
    row["frac_B_fit_neg"] = float((df["B_fit"] < 0).mean())
    row["median_r2"] = float(df["r2"].median())
    row["median_n_items"] = int(df["n_items"].median())
    row["mean_icc1"] = float(df["icc1"].dropna().iloc[0]) if df["icc1"].notna().any() else np.nan
    row["icc1_items_used"] = int(df["icc1_items_used"].dropna().iloc[0]) if df["icc1_items_used"].notna().any() else 0
    row["icc1_k"] = int(df["icc1_k"].dropna().iloc[0]) if df["icc1_k"].notna().any() else 0
    return row


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = load_wmt_long(args.root, mode="mqm")
    if args.exclude_ref_like:
        long_df = filter_ref_like_systems(long_df)
    long_df.to_csv(outdir / "wmt_mqm_long.csv", index=False)

    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.t_n)])
    path_rows = []
    traj_rows = []
    dataset_rows = []

    for dataset_id in sorted(long_df["dataset_id"].unique().tolist()):
        sub = long_df[long_df["dataset_id"] == dataset_id].copy()
        icc1, icc_n, icc_k = icc1_from_long(sub, item_col="item_id", score_col="score")
        paths = build_heldout_paths_from_long(
            sub,
            dataset_id,
            gold_estimator=args.gold_estimator,
            score_transform=args.score_transform,
            min_items=args.min_items,
            min_other_raters=args.min_other_raters,
        )
        for p in paths:
            J = p["J"]
            r = p["r"]
            ab = compute_ab(J, r)
            d, q = tilt_curve(J, r, t_grid)
            A_fit, B_fit, r2 = fit_local_ab(d, q, max_d=args.fit_max_d)
            path_rows.append({
                "dataset_id": dataset_id,
                "heldout_rater": p["heldout_rater"],
                "n_items": p["n_items"],
                "A_theory": ab["A"],
                "B_theory": ab["B"],
                "A_fit": A_fit,
                "B_fit": B_fit,
                "r2": r2,
                "rho": ab["rho"],
                "sigma_J_sq": ab["sigma_J_sq"],
                "sigma_r_sq": ab["sigma_r_sq"],
                "icc1": icc1,
                "icc1_items_used": icc_n,
                "icc1_k": icc_k,
                "fit_max_d": args.fit_max_d,
                "score_transform": args.score_transform,
                "gold_estimator": args.gold_estimator,
            })
            for t, dv, qv in zip(t_grid, d, q, strict=False):
                traj_rows.append({
                    "dataset_id": dataset_id,
                    "heldout_rater": p["heldout_rater"],
                    "t": float(t),
                    "d": float(dv),
                    "q": float(qv),
                })
        if paths:
            ds = pd.DataFrame([r for r in path_rows if r["dataset_id"] == dataset_id])
            dataset_rows.append(summarize_dataset(ds))
        else:
            dataset_rows.append({
                "dataset_id": dataset_id,
                "n_paths": 0,
                "corr_A": np.nan,
                "corr_B": np.nan,
                "sign_agreement": 0,
                "frac_B_theory_neg": np.nan,
                "frac_B_fit_neg": np.nan,
                "median_r2": np.nan,
                "median_n_items": 0,
                "mean_icc1": icc1,
                "icc1_items_used": icc_n,
                "icc1_k": icc_k,
            })

    paths_df = pd.DataFrame(path_rows)
    traj_df = pd.DataFrame(traj_rows)
    ds_df = pd.DataFrame(dataset_rows)
    summary_df = pd.DataFrame([
        {
            "n_datasets": int(ds_df.shape[0]),
            "datasets_with_paths": int((ds_df["n_paths"] > 0).sum()),
            "n_paths_total": int(paths_df.shape[0]),
            "corr_A_pooled": float(np.corrcoef(paths_df["A_theory"], paths_df["A_fit"])[0, 1]) if len(paths_df) >= 2 and paths_df["A_theory"].std() > 0 and paths_df["A_fit"].std() > 0 else np.nan,
            "corr_B_pooled": float(np.corrcoef(paths_df["B_theory"], paths_df["B_fit"])[0, 1]) if len(paths_df) >= 2 and paths_df["B_theory"].std() > 0 and paths_df["B_fit"].std() > 0 else np.nan,
            "sign_agreement_total": int(((np.sign(paths_df["B_theory"]) == np.sign(paths_df["B_fit"])) | ((paths_df["B_theory"].abs() < 1e-12) & (paths_df["B_fit"].abs() < 1e-12))).sum()) if not paths_df.empty else 0,
            "frac_B_theory_neg_pooled": float((paths_df["B_theory"] < 0).mean()) if not paths_df.empty else np.nan,
            "frac_B_fit_neg_pooled": float((paths_df["B_fit"] < 0).mean()) if not paths_df.empty else np.nan,
            "median_r2_pooled": float(paths_df["r2"].median()) if not paths_df.empty else np.nan,
        }
    ])

    paths_df.to_csv(outdir / "wmt_mqm_paths.csv", index=False)
    traj_df.to_csv(outdir / "wmt_mqm_traj.csv", index=False)
    ds_df.to_csv(outdir / "wmt_mqm_by_dataset.csv", index=False)
    summary_df.to_csv(outdir / "wmt_mqm_summary.csv", index=False)
    print("Saved:", outdir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
