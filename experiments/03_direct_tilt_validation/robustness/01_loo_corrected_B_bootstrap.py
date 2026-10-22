#!/usr/bin/env python3
"""  [exp11]Experiment 15: LOO corrected-B bootstrap on human multi-rater data.

Why run this:
- upgrades the old plug-in B experiment to held-out LOO gold
- puts bootstrap CIs on A, B, rho, and sign(B<0)
- optionally checks exact-tilt local fit alignment in the same run

Expected input format: long CSV with columns
    article_id, model_id, dimension, rater_type, rater_idx, score
or equivalent plus item_id.

Example:
python exp15_loo_corrected_B_bootstrap.py \
  --data data/summeval_long.csv \
  --outdir results/exp15 \
  --n-bootstrap 1000 \
  --fit-max-d 0.10
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))

import argparse
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from pathlib import Path

import numpy as np
import pandas as pd

from verif_experiment_utils import (
    DIMENSIONS_DEFAULT,
    compute_ab,
    fit_local_ab,
    gold_from_matrix,
    load_long_csv,
    make_matrix,
    parse_str_list,
    tilt_curve,
    zscore_within_rater,
)


def stable_offset(*parts, modulo: int = 10_000) -> int:
    text = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def bootstrap_path_metrics(
    J: np.ndarray,
    r: np.ndarray,
    *,
    n_boot: int,
    fit_max_d: float,
    t_grid: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(J)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Jb = J[idx]
        rb = r[idx]
        th = compute_ab(Jb, rb)
        d, q = tilt_curve(Jb, rb, t_grid)
        A_fit, B_fit, r2 = fit_local_ab(d, q, fit_max_d)
        rows.append({
            "boot_idx": b,
            "A_theory": th["A"],
            "B_theory": th["B"],
            "rho": th["rho"],
            "nu_tilde": th["nu_tilde"],
            "A_fit": A_fit,
            "B_fit": B_fit,
            "r2": r2,
        })
    return pd.DataFrame(rows)


def ci(series: pd.Series, probs=(0.025, 0.5, 0.975)) -> tuple[float, float, float]:
    vals = series.dropna().to_numpy()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return tuple(float(x) for x in np.quantile(vals, probs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pools", default="expert,crowd")
    ap.add_argument("--dimensions", default=",".join(DIMENSIONS_DEFAULT))
    ap.add_argument("--gold-estimator", default="loo_mean",
                    choices=["loo_mean", "same_mean", "median_others", "trimmed_mean_others"])
    ap.add_argument("--score-transform", default="raw", choices=["raw", "rater_z"])
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--n-t", type=int, default=400)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_long_csv(args.data)
    if args.score_transform == "rater_z":
        df = zscore_within_rater(df, "score")

    pools = parse_str_list(args.pools)
    dimensions = parse_str_list(args.dimensions)
    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.n_t)])

    path_rows = []
    boot_rows = []
    summary_rows = []

    for pool in pools:
        for dim in dimensions:
            X, item_ids, rater_ids = make_matrix(df, dimension=dim, pool=pool, score_col="score")
            n_items, n_raters = X.shape
            label = f"{pool[:3]}-{dim[:5]}"
            print(f"[{label}] items={n_items}, raters={n_raters}")

            for j in range(n_raters):
                J, r = gold_from_matrix(X, j, estimator=args.gold_estimator)
                th = compute_ab(J, r)
                d, q = tilt_curve(J, r, t_grid)
                A_fit, B_fit, r2 = fit_local_ab(d, q, args.fit_max_d)
                path_rows.append({
                    "pool": pool,
                    "dimension": dim,
                    "label": label,
                    "rater_idx": j,
                    "rater_key": rater_ids[j],
                    "n_items": n_items,
                    "n_raters": n_raters,
                    "gold_estimator": args.gold_estimator,
                    "score_transform": args.score_transform,
                    "fit_max_d": args.fit_max_d,
                    "A_theory": th["A"],
                    "B_theory": th["B"],
                    "rho": th["rho"],
                    "nu_tilde": th["nu_tilde"],
                    "A_fit": A_fit,
                    "B_fit": B_fit,
                    "r2": r2,
                    "sign_match": int(np.sign(th["B"]) == np.sign(B_fit)),
                    "B_negative": int(th["B"] < 0),
                })

                boot = bootstrap_path_metrics(
                    J,
                    r,
                    n_boot=args.n_bootstrap,
                    fit_max_d=args.fit_max_d,
                    t_grid=t_grid,
                    seed=args.seed + 1000 * stable_offset(pool, dim, j),
                )
                lo, med, hi = ci(boot["B_theory"])
                lo_fit, med_fit, hi_fit = ci(boot["B_fit"])
                summary_rows.append({
                    "pool": pool,
                    "dimension": dim,
                    "label": label,
                    "rater_idx": j,
                    "rater_key": rater_ids[j],
                    "B_theory": th["B"],
                    "B_theory_ci_lo": lo,
                    "B_theory_ci_med": med,
                    "B_theory_ci_hi": hi,
                    "B_fit": B_fit,
                    "B_fit_ci_lo": lo_fit,
                    "B_fit_ci_med": med_fit,
                    "B_fit_ci_hi": hi_fit,
                    "Pr_B_theory_lt_0": float((boot["B_theory"] < 0).mean()),
                    "Pr_B_fit_lt_0": float((boot["B_fit"] < 0).mean()),
                    "mean_r2_boot": float(boot["r2"].mean()),
                })
                boot["pool"] = pool
                boot["dimension"] = dim
                boot["label"] = label
                boot["rater_idx"] = j
                boot_rows.append(boot)

    path_df = pd.DataFrame(path_rows)
    boot_df = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    summ_df = pd.DataFrame(summary_rows)

    dataset_summary = (
        path_df.groupby(["pool", "dimension", "label"], as_index=False)
        .agg(
            n_raters=("rater_idx", "nunique"),
            mean_A_theory=("A_theory", "mean"),
            mean_B_theory=("B_theory", "mean"),
            mean_A_fit=("A_fit", "mean"),
            mean_B_fit=("B_fit", "mean"),
            mean_rho=("rho", "mean"),
            mean_nu_tilde=("nu_tilde", "mean"),
            sign_match_rate=("sign_match", "mean"),
            B_negative_rate=("B_negative", "mean"),
            mean_r2=("r2", "mean"),
        )
    )

    path_file = outdir / "table_exp15_loo_corrected_B_paths.csv"
    boot_file = outdir / "table_exp15_loo_corrected_B_bootstrap.csv"
    summ_file = outdir / "table_exp15_loo_corrected_B_summary.csv"
    dataset_file = outdir / "table_exp15_loo_corrected_B_dataset_summary.csv"

    path_df.to_csv(path_file, index=False)
    boot_df.to_csv(boot_file, index=False)
    summ_df.to_csv(summ_file, index=False)
    dataset_summary.to_csv(dataset_file, index=False)

    print(f"→ saved {path_file}")
    print(f"→ saved {boot_file}")
    print(f"→ saved {summ_file}")
    print(f"→ saved {dataset_file}")


if __name__ == "__main__":
    main()
