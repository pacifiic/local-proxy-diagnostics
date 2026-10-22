#!/usr/bin/env python3
"""  [exp14]Experiment 18: Direct-tilt estimator / preprocessing sensitivity.

Goal:
Check whether the strong exact-tilt validation is robust to:
- gold estimator choice: loo_mean / same_mean / median_others / trimmed_mean_others
- score preprocessing: raw / rater_z
- local fit range: multiple max_d values
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))

import argparse
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
    parse_float_list,
    parse_str_list,
    tilt_curve,
    zscore_within_rater,
)


def run_one_setting(
    df: pd.DataFrame,
    *,
    pool: str,
    dimension: str,
    gold_estimator: str,
    score_transform: str,
    fit_max_d_vals: list[float],
    t_grid: np.ndarray,
) -> list[dict]:
    X, _, rater_ids = make_matrix(df, dimension=dimension, pool=pool, score_col="score")
    rows = []
    label = f"{pool[:3]}-{dimension[:5]}"
    for j in range(X.shape[1]):
        J, r = gold_from_matrix(X, j, estimator=gold_estimator)
        th = compute_ab(J, r)
        d, q = tilt_curve(J, r, t_grid)
        for max_d in fit_max_d_vals:
            A_fit, B_fit, r2 = fit_local_ab(d, q, max_d)
            rows.append({
                "score_transform": score_transform,
                "pool": pool,
                "dimension": dimension,
                "label": label,
                "rater_idx": j,
                "rater_key": rater_ids[j],
                "gold_estimator": gold_estimator,
                "fit_max_d": max_d,
                "A_theory": th["A"],
                "B_theory": th["B"],
                "rho": th["rho"],
                "nu_tilde": th["nu_tilde"],
                "A_fit": A_fit,
                "B_fit": B_fit,
                "r2": r2,
                "sign_match": int(np.sign(th["B"]) == np.sign(B_fit)),
                "B_theory_negative": int(th["B"] < 0),
                "B_fit_negative": int(B_fit < 0),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pools", default="expert,crowd")
    ap.add_argument("--dimensions", default=",".join(DIMENSIONS_DEFAULT))
    ap.add_argument("--gold-estimators", default="loo_mean,same_mean,median_others,trimmed_mean_others")
    ap.add_argument("--score-transforms", default="raw,rater_z")
    ap.add_argument("--fit-max-d-vals", default="0.05,0.10,0.20,0.50,1.00")
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--n-t", type=int, default=400)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_df = load_long_csv(args.data)
    pools = parse_str_list(args.pools)
    dimensions = parse_str_list(args.dimensions)
    gold_estimators = parse_str_list(args.gold_estimators)
    score_transforms = parse_str_list(args.score_transforms)
    fit_max_d_vals = parse_float_list(args.fit_max_d_vals)
    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.n_t)])

    rows = []
    for transform in score_transforms:
        df = base_df.copy()
        if transform == "rater_z":
            df = zscore_within_rater(df, "score")
        for pool in pools:
            for dim in dimensions:
                for estimator in gold_estimators:
                    print(f"[{transform}] [{pool}] [{dim}] [{estimator}]")
                    try:
                        rows.extend(
                            run_one_setting(
                                df,
                                pool=pool,
                                dimension=dim,
                                gold_estimator=estimator,
                                score_transform=transform,
                                fit_max_d_vals=fit_max_d_vals,
                                t_grid=t_grid,
                            )
                        )
                    except Exception as exc:
                        print(f"  skipped: {exc}")

    out = pd.DataFrame(rows)

    summary = (
        out.groupby(["score_transform", "gold_estimator", "fit_max_d"], as_index=False)
        .agg(
            n_paths=("label", "count"),
            r_A=("A_fit", lambda s: np.corrcoef(out.loc[s.index, "A_theory"], s)[0, 1] if len(s) > 1 else np.nan),
            r_B=("B_fit", lambda s: np.corrcoef(out.loc[s.index, "B_theory"], s)[0, 1] if len(s) > 1 else np.nan),
            sign_match_rate=("sign_match", "mean"),
            B_theory_negative_rate=("B_theory_negative", "mean"),
            B_fit_negative_rate=("B_fit_negative", "mean"),
            mean_r2=("r2", "mean"),
            mean_rho=("rho", "mean"),
        )
    )

    by_dataset = (
        out.groupby(["score_transform", "gold_estimator", "fit_max_d", "label"], as_index=False)
        .agg(
            mean_A_theory=("A_theory", "mean"),
            mean_B_theory=("B_theory", "mean"),
            mean_A_fit=("A_fit", "mean"),
            mean_B_fit=("B_fit", "mean"),
            sign_match_rate=("sign_match", "mean"),
            mean_r2=("r2", "mean"),
        )
    )

    out_file = outdir / "table_exp18_tilt_estimator_sensitivity_paths.csv"
    summary_file = outdir / "table_exp18_tilt_estimator_sensitivity_summary.csv"
    by_dataset_file = outdir / "table_exp18_tilt_estimator_sensitivity_by_dataset.csv"
    out.to_csv(out_file, index=False)
    summary.to_csv(summary_file, index=False)
    by_dataset.to_csv(by_dataset_file, index=False)
    print(f"→ saved {out_file}")
    print(f"→ saved {summary_file}")
    print(f"→ saved {by_dataset_file}")


if __name__ == "__main__":
    main()
