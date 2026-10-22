#!/usr/bin/env python3
"""  [exp13]Experiment 17: Matched-rater-count ICC robustness.

Goal:
Compare expert vs crowd verifiability after matching the number of crowd raters to
the expert pool size, with optional rater-wise z-scoring.

This addresses the fairness objection that expert (3 raters) vs crowd (5 raters)
may not be a like-for-like comparison on SummEval.
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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from verif_experiment_utils import (
    DIMENSIONS_DEFAULT,
    icc_a1,
    icc_c1,
    load_long_csv,
    make_matrix,
    parse_str_list,
    zscore_within_rater,
)


def stable_offset(*parts, modulo: int = 10_000) -> int:
    text = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def bootstrap_icc(X: np.ndarray, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        rows.append({"boot_idx": b, "ICC_A1": icc_a1(Xb), "ICC_C1": icc_c1(Xb)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dimensions", default=",".join(DIMENSIONS_DEFAULT))
    ap.add_argument("--score-transform", default="raw", choices=["raw", "rater_z"])
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_long_csv(args.data)
    if args.score_transform == "rater_z":
        df = zscore_within_rater(df, "score")

    dimensions = parse_str_list(args.dimensions)
    point_rows = []
    boot_rows = []

    for dim in dimensions:
        X_exp, _, exp_raters = make_matrix(df, dimension=dim, pool="expert")
        X_cro, _, cro_raters = make_matrix(df, dimension=dim, pool="crowd")

        k_exp = X_exp.shape[1]
        k_cro = X_cro.shape[1]
        if k_cro < k_exp:
            raise ValueError(f"Crowd raters ({k_cro}) fewer than expert raters ({k_exp}) for {dim}")

        # Full pools
        point_rows.append({
            "dimension": dim,
            "condition": "expert_full",
            "n_items": X_exp.shape[0],
            "n_raters": k_exp,
            "ICC_A1": icc_a1(X_exp),
            "ICC_C1": icc_c1(X_exp),
            "nu_proxy": float(np.sqrt(max(icc_a1(X_exp), 0.0))),
        })
        point_rows.append({
            "dimension": dim,
            "condition": "crowd_full",
            "n_items": X_cro.shape[0],
            "n_raters": k_cro,
            "ICC_A1": icc_a1(X_cro),
            "ICC_C1": icc_c1(X_cro),
            "nu_proxy": float(np.sqrt(max(icc_a1(X_cro), 0.0))),
        })

        exp_boot = bootstrap_icc(X_exp, n_boot=args.n_bootstrap, seed=args.seed + stable_offset(dim, "exp"))
        exp_boot["dimension"] = dim
        exp_boot["condition"] = "expert_full"
        boot_rows.append(exp_boot)

        cro_boot = bootstrap_icc(X_cro, n_boot=args.n_bootstrap, seed=args.seed + stable_offset(dim, "cro"))
        cro_boot["dimension"] = dim
        cro_boot["condition"] = "crowd_full"
        boot_rows.append(cro_boot)

        # Matched-rater crowd subsets: use all combinations if small enough, else random sample.
        all_subsets = list(combinations(range(k_cro), k_exp))
        for subset_idx, subset in enumerate(all_subsets):
            X_sub = X_cro[:, subset]
            label = f"crowd_match_{subset_idx:02d}"
            point_rows.append({
                "dimension": dim,
                "condition": label,
                "n_items": X_sub.shape[0],
                "n_raters": X_sub.shape[1],
                "ICC_A1": icc_a1(X_sub),
                "ICC_C1": icc_c1(X_sub),
                "nu_proxy": float(np.sqrt(max(icc_a1(X_sub), 0.0))),
                "subset": ";".join(map(str, subset)),
            })
            sub_boot = bootstrap_icc(
                X_sub,
                n_boot=args.n_bootstrap,
                seed=args.seed + subset_idx + stable_offset(dim, "sub"),
            )
            sub_boot["dimension"] = dim
            sub_boot["condition"] = label
            sub_boot["subset"] = ";".join(map(str, subset))
            boot_rows.append(sub_boot)

    point_df = pd.DataFrame(point_rows)
    boot_df = pd.concat(boot_rows, ignore_index=True)

    # Aggregate crowd-matched subsets into one summary per dimension.
    matched = point_df[point_df["condition"].str.startswith("crowd_match_")].copy()
    matched_summary = (
        matched.groupby("dimension", as_index=False)
        .agg(
            crowd_match_ICC_A1_mean=("ICC_A1", "mean"),
            crowd_match_ICC_A1_min=("ICC_A1", "min"),
            crowd_match_ICC_A1_max=("ICC_A1", "max"),
            crowd_match_nu_mean=("nu_proxy", "mean"),
            n_subsets=("condition", "count"),
        )
    )

    expert_full = point_df[point_df["condition"] == "expert_full"][ ["dimension", "ICC_A1", "nu_proxy"] ].rename(columns={"ICC_A1":"expert_ICC_A1","nu_proxy":"expert_nu"})
    crowd_full = point_df[point_df["condition"] == "crowd_full"][ ["dimension", "ICC_A1", "nu_proxy"] ].rename(columns={"ICC_A1":"crowd_full_ICC_A1","nu_proxy":"crowd_full_nu"})
    final_summary = expert_full.merge(crowd_full, on="dimension").merge(matched_summary, on="dimension")
    final_summary["expert_minus_crowd_full"] = final_summary["expert_ICC_A1"] - final_summary["crowd_full_ICC_A1"]
    final_summary["expert_minus_crowd_matched_mean"] = final_summary["expert_ICC_A1"] - final_summary["crowd_match_ICC_A1_mean"]

    point_file = outdir / "table_exp17_matched_rater_icc_points.csv"
    boot_file = outdir / "table_exp17_matched_rater_icc_bootstrap.csv"
    summary_file = outdir / "table_exp17_matched_rater_icc_summary.csv"
    point_df.to_csv(point_file, index=False)
    boot_df.to_csv(boot_file, index=False)
    final_summary.to_csv(summary_file, index=False)
    print(f"→ saved {point_file}")
    print(f"→ saved {boot_file}")
    print(f"→ saved {summary_file}")


if __name__ == "__main__":
    main()
