#!/usr/bin/env python3
"""  [exp25]Exp29: measurement intervention before/after.

Core supplemental-audit experiment:
Does improving measurement (averaging more proxy raters) shift the local
coefficients in the predicted direction?

Protocol
--------
Within each analysis group:
1. Randomly choose a fixed disjoint gold subset G of raters.
2. Order the remaining proxy raters.
3. Construct nested proxy measurements using the first k raters for k in K.
4. For each k, compute J = mean_G, r = mean_proxy(k), then fit the exact local law.
5. Compare before/after changes in:
   - nu_tilde
   - A_exact
   - B_theory / c_loc
   - local headroom A^2/(4 c_loc)

The same random partition is reused across k so that deltas are paired.
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
    bootstrap_ci,
    build_nested_intervention_paths,
    canonicalize_long,
    compute_ab,
    fit_local_ab,
    read_table,
    spearman_brown,
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
    ap.add_argument("--score-transform", type=str, default="raw", choices=["raw", "rater_z"])
    ap.add_argument("--k-list", type=str, default="1,2,3")
    ap.add_argument("--gold-size", type=int, default=None)
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--min-items", type=int, default=100)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--t-n", type=int, default=400)
    return ap.parse_args()


def parse_k_list(s: str) -> list[int]:
    return sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))


def summarize_k(df: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "group_id": df["group_id"].iloc[0],
        "k_proxy": int(df["k_proxy"].iloc[0]),
        "n_paths": int(df.shape[0]),
        "median_n_items": int(df["n_items"].median()),
        "mean_icc_a1_full": float(df["icc_a1_full"].median()),
        "mean_spearman_brown_icc_k": float(df["icc_k_spearman_brown"].median()),
        "mean_nu_tilde": float(df["nu_tilde"].mean()),
        "mean_corr_rJ": float(df["corr_rJ"].mean()),
        "mean_A_exact": float(df["A_exact"].mean()),
        "mean_B_theory": float(df["B_theory"].mean()),
        "frac_B_neg": float((df["B_theory"] < 0).mean()),
        "mean_c_loc": float(df["c_loc"].dropna().mean()) if df["c_loc"].notna().any() else np.nan,
        "mean_headroom": float(df["headroom_exact"].dropna().mean()) if df["headroom_exact"].notna().any() else np.nan,
        "median_r2": float(df["r2"].median()),
    }


def paired_deltas(df: pd.DataFrame, k_lo: int, k_hi: int) -> pd.DataFrame:
    sub = df[df["k_proxy"].isin([k_lo, k_hi])].copy()
    key_cols = ["group_id", "rep_id"]
    lo = sub[sub["k_proxy"] == k_lo].copy()
    hi = sub[sub["k_proxy"] == k_hi].copy()
    merged = lo.merge(
        hi,
        on=key_cols,
        suffixes=(f"_{k_lo}", f"_{k_hi}"),
        how="inner",
    )
    if merged.empty:
        return merged
    for metric in ["nu_tilde", "corr_rJ", "A_exact", "B_theory", "c_loc", "headroom_exact"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_{k_hi}"] - merged[f"{metric}_{k_lo}"]
    return merged


def summarize_pair(df: pd.DataFrame, k_lo: int, k_hi: int) -> dict[str, float | int | str]:
    row = {
        "group_id": df["group_id"].iloc[0],
        "k_before": int(k_lo),
        "k_after": int(k_hi),
        "n_pairs": int(df.shape[0]),
    }
    for metric in ["nu_tilde", "corr_rJ", "A_exact", "B_theory", "c_loc", "headroom_exact"]:
        col = f"delta_{metric}"
        vals = df[col].to_numpy(dtype=float)
        row[f"mean_{col}"] = float(np.nanmean(vals)) if len(vals) else np.nan
        row[f"median_{col}"] = float(np.nanmedian(vals)) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals[np.isfinite(vals)])
        row[f"{col}_ci_lo"] = lo
        row[f"{col}_ci_hi"] = hi
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

    k_list = parse_k_list(args.k_list)
    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.t_n)])

    rows = []
    for _, g in df.groupby("group_id"):
        paths = build_nested_intervention_paths(
            g,
            k_list=k_list,
            gold_size=args.gold_size,
            n_reps=args.n_reps,
            random_state=args.random_state,
            score_transform=args.score_transform,
            min_items=args.min_items,
        )
        for p in paths:
            ab = compute_ab(p.J, p.r)
            d, q = tilt_curve(p.J, p.r, t_grid)
            A_fit, B_fit, r2 = fit_local_ab(d, q, max_d=args.fit_max_d)
            c_loc = float(-ab["B"]) if ab["B"] < 0 else np.nan
            headroom_exact = ab["A"]**2 / (4 * c_loc) if np.isfinite(c_loc) and c_loc > 0 and ab["A"] > 0 else np.nan
            corr_rJ = float(np.corrcoef(p.J, p.r)[0, 1]) if len(p.J) > 1 and np.std(p.J) > 0 and np.std(p.r) > 0 else np.nan

            rows.append({
                "group_id": p.group_id,
                "rep_id": p.rep_id,
                "k_proxy": p.k_proxy,
                "proxy_raters": "|".join(p.proxy_raters),
                "gold_raters": "|".join(p.gold_raters),
                "n_items": p.n_items,
                "icc_a1_full": p.icc_a1_full,
                "icc_k_spearman_brown": spearman_brown(p.icc_a1_full, p.k_proxy),
                "nu_tilde": ab["nu_tilde"],
                "corr_rJ": corr_rJ,
                "A_exact": ab["A"],
                "B_theory": ab["B"],
                "A_fit": A_fit,
                "B_fit": B_fit,
                "r2": r2,
                "rho": ab["rho"],
                "sigma_J_sq": ab["sigma_J_sq"],
                "sigma_r_sq": ab["sigma_r_sq"],
                "c_loc": c_loc,
                "headroom_exact": headroom_exact,
            })

    paths_df = pd.DataFrame(rows)
    k_summary_df = pd.DataFrame(
        [summarize_k(g) for _, g in paths_df.groupby(["group_id", "k_proxy"])]
    ) if not paths_df.empty else pd.DataFrame()

    pair_rows = []
    pair_delta_frames = []
    if not paths_df.empty:
        for group_id, g in paths_df.groupby("group_id"):
            k_avail = sorted(g["k_proxy"].unique().tolist())
            for a, b in zip(k_avail[:-1], k_avail[1:], strict=False):
                dd = paired_deltas(g, a, b)
                if dd.empty:
                    continue
                pair_delta_frames.append(dd)
                pair_rows.append(summarize_pair(dd, a, b))
    pair_df = pd.DataFrame(pair_rows)
    pair_long_df = pd.concat(pair_delta_frames, ignore_index=True) if pair_delta_frames else pd.DataFrame()

    summary_df = pd.DataFrame([{
        "n_groups": int(paths_df["group_id"].nunique()) if not paths_df.empty else 0,
        "n_paths_total": int(paths_df.shape[0]),
        "available_k_values": ",".join(map(str, sorted(paths_df["k_proxy"].unique().tolist()))) if not paths_df.empty else "",
    }])

    paths_df.to_csv(outdir / "exp25_intervention_paths.csv", index=False)
    k_summary_df.to_csv(outdir / "exp25_intervention_by_k.csv", index=False)
    pair_long_df.to_csv(outdir / "exp25_intervention_pair_deltas_long.csv", index=False)
    pair_df.to_csv(outdir / "exp25_intervention_pair_summary.csv", index=False)
    summary_df.to_csv(outdir / "exp25_intervention_summary.csv", index=False)

    print("Saved intervention experiment to", outdir)
    if not pair_df.empty:
        print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
