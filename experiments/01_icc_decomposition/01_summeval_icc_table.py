#!/usr/bin/env python3
"""  [exp01]
Experiment 1 — SummEval Decomposition Table

Compute ICC(A,1), ICC(C,1), G_pers for each dimension × pool.
Outputs: Table 1, Figure 1, Figure 2, bootstrap stats.
"""

import os
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
sys.path.insert(0, str(ROOT / "lib"))

from src.fast_icc import (
    df_to_matrix, icc_a1, icc_c1,
    bootstrap_icc_fast, bootstrap_gap_fast,
)
from src.plotting import fig1_icc_by_dimension, fig2_personalization_gap

# ---------------------------------------------------------------------------
DATA_PATH = DATA_ROOT / "summeval_long.csv"
TABLE_OUT  = ROOT / "results" / "tables" / "table1_summeval_decomposition.csv"
BOOT_OUT   = ROOT / "results" / "intermediate" / "bootstrap_summeval_icc.json"
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
POOLS      = ["all", "crowd", "expert"]
N_BOOT     = 2000
SEED       = 42
# ---------------------------------------------------------------------------


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["target"] = df["article_id"] + "___" + df["model_id"]
    df["rater"]  = df["rater_type"] + "_" + df["rater_idx"].astype(str)
    return df


def run():
    print("=" * 60)
    print("Experiment 1: SummEval Decomposition Table")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df):,} rows, {df['target'].nunique()} targets, "
          f"{df['rater'].nunique()} raters\n")

    rows = []
    boot_results = {}

    for dim in DIMENSIONS:
        df_dim = df[df["dimension"] == dim]

        for pool in POOLS:
            print(f"  [{dim}][{pool}] ", end="", flush=True)

            if pool == "all":
                df_pool = df_dim
            elif pool == "crowd":
                df_pool = df_dim[df_dim["rater_type"] == "turker"]
            else:  # expert
                df_pool = df_dim[df_dim["rater_type"] == "expert"]

            X = df_to_matrix(df_pool)

            # Point estimates
            a1_val = icc_a1(X)
            c1_val = icc_c1(X)
            g_val  = (c1_val - a1_val) if not (np.isnan(a1_val) or np.isnan(c1_val)) else np.nan

            # Bootstrap CIs
            boot_a1 = bootstrap_icc_fast(X, icc_a1, N_BOOT, SEED)
            boot_c1 = bootstrap_icc_fast(X, icc_c1, N_BOOT, SEED)
            boot_g  = bootstrap_gap_fast(X, N_BOOT, SEED)

            rows.append({
                "dimension":        dim,
                "pool":             pool,
                "N_examples":       df_pool["target"].nunique(),
                "N_raters":         df_pool["rater"].nunique(),
                "ICC_A1":           a1_val,
                "ICC_A1_CI_lower":  boot_a1["ci95_lower"],
                "ICC_A1_CI_upper":  boot_a1["ci95_upper"],
                "ICC_C1":           c1_val,
                "ICC_C1_CI_lower":  boot_c1["ci95_lower"],
                "ICC_C1_CI_upper":  boot_c1["ci95_upper"],
                "G_pers":           g_val,
                "G_pers_CI_lower":  boot_g["ci95_lower"],
                "G_pers_CI_upper":  boot_g["ci95_upper"],
                "mean_score":       df_pool["score"].mean(),
                "std_score":        df_pool["score"].std(),
            })
            boot_results[f"{dim}_{pool}"] = {"a1": boot_a1, "c1": boot_c1, "gap": boot_g}

            print(f"ICC(A,1)={a1_val:.3f}  ICC(C,1)={c1_val:.3f}  G_pers={g_val:.3f}")

    # ── Save ──
    df_table = pd.DataFrame(rows)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(TABLE_OUT, index=False)
    print(f"\nSaved: {TABLE_OUT}")

    BOOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BOOT_OUT, "w") as f:
        json.dump(boot_results, f, indent=2)
    print(f"Saved: {BOOT_OUT}")

    # ── Figures ──
    print("\nGenerating figures...")
    fig1_df = df_table[["dimension","pool","ICC_A1","ICC_A1_CI_lower","ICC_A1_CI_upper"]].copy()
    fig1_df.rename(columns={"ICC_A1_CI_lower":"CI_lower","ICC_A1_CI_upper":"CI_upper"}, inplace=True)
    fig1_icc_by_dimension(fig1_df)

    fig2_personalization_gap(df_table[["dimension","pool","G_pers"]].copy())

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY — Table 1")
    print("=" * 60)
    print(df_table[["dimension","pool","ICC_A1","ICC_C1","G_pers","mean_score"]]
          .to_string(index=False, float_format="%.3f"))

    return df_table


if __name__ == "__main__":
    run()
