#!/usr/bin/env python3
"""  [exp03]
Experiment 2 — SummEval Annotator-Pool Perturbation

Compare crowd-only vs expert-only ICC to quantify expert lift.
Outputs: Table 2, Figure 3, Figure 4.
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
    bootstrap_icc_fast, bootstrap_lift_fast,
)
from src.plotting import fig3_expert_crowd_lift, fig4_expert_crowd_scatter

# ---------------------------------------------------------------------------
DATA_PATH = DATA_ROOT / "summeval_long.csv"
TABLE_OUT  = ROOT / "results" / "tables" / "table2_summeval_expert_crowd_lift.csv"
BOOT_OUT   = ROOT / "results" / "intermediate" / "bootstrap_summeval_lift.json"
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
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
    print("Experiment 2: Annotator-Pool Perturbation")
    print("=" * 60)

    df = load_data()
    rows = []
    boot_results = {}

    for dim in DIMENSIONS:
        print(f"  [{dim}] ", end="", flush=True)
        df_dim    = df[df["dimension"] == dim]
        df_crowd  = df_dim[df_dim["rater_type"] == "turker"]
        df_expert = df_dim[df_dim["rater_type"] == "expert"]

        X_crowd  = df_to_matrix(df_crowd)
        X_expert = df_to_matrix(df_expert)

        a1_crowd  = icc_a1(X_crowd)
        a1_expert = icc_a1(X_expert)
        c1_crowd  = icc_c1(X_crowd)
        c1_expert = icc_c1(X_expert)
        lift_a1   = a1_expert - a1_crowd
        lift_c1   = c1_expert - c1_crowd

        # Bootstrap lift (paired on shared targets → same target ordering)
        boot_lift = bootstrap_lift_fast(X_crowd, X_expert, icc_a1, N_BOOT, SEED)

        rows.append({
            "dimension":      dim,
            "ICC_A1_crowd":   a1_crowd,
            "ICC_A1_expert":  a1_expert,
            "Lift_expert":    lift_a1,
            "Lift_CI_lower":  boot_lift["ci95_lower"],
            "Lift_CI_upper":  boot_lift["ci95_upper"],
            "ICC_C1_crowd":   c1_crowd,
            "ICC_C1_expert":  c1_expert,
            "Lift_C1":        lift_c1,
            "mean_crowd":     df_crowd["score"].mean(),
            "mean_expert":    df_expert["score"].mean(),
            "delta_mean":     df_expert["score"].mean() - df_crowd["score"].mean(),
        })
        boot_results[dim] = {
            "lift_a1": {k: float(v) for k, v in boot_lift.items()}
        }

        print(f"Crowd={a1_crowd:.3f}  Expert={a1_expert:.3f}  "
              f"Lift={lift_a1:.3f} [{boot_lift['ci95_lower']:.3f}, {boot_lift['ci95_upper']:.3f}]")

    df_table = pd.DataFrame(rows)
    df_table.to_csv(TABLE_OUT, index=False)
    print(f"\nSaved: {TABLE_OUT}")

    with open(BOOT_OUT, "w") as f:
        json.dump(boot_results, f, indent=2)
    print(f"Saved: {BOOT_OUT}")

    print("\nGenerating figures...")
    fig3_expert_crowd_lift(df_table)
    fig4_expert_crowd_scatter(df_table)

    print("\n" + "=" * 60)
    print("SUMMARY — Table 2")
    print("=" * 60)
    print(df_table[["dimension","ICC_A1_crowd","ICC_A1_expert","Lift_expert","delta_mean"]]
          .to_string(index=False, float_format="%.3f"))

    return df_table


if __name__ == "__main__":
    run()
