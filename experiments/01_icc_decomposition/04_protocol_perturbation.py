#!/usr/bin/env python3
"""  [exp04]
Experiment 3 — SummEval Protocol Perturbation

Compare split-half stability under different aggregation protocols:
  P0: mean, P1: median, P2: trimmed_mean
Outputs: Table 3, Figure 5.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
sys.path.insert(0, str(ROOT / "lib"))

from src.reliability import compute_split_half_stability
from src.plotting import fig5_protocol_stability

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = DATA_ROOT / "summeval_long.csv"
TABLE_OUT = ROOT / "results" / "tables" / "table3_summeval_protocol_perturbation.csv"
BOOT_OUT = ROOT / "results" / "intermediate" / "bootstrap_protocol.json"
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
PROTOCOLS = ["mean", "median", "trimmed_mean"]
POOLS = ["all", "crowd", "expert"]
N_SPLITS = 500
SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["target"] = df["article_id"] + "___" + df["model_id"]
    df["rater"] = df["rater_type"] + "_" + df["rater_idx"].astype(str)
    return df


def run():
    print("=" * 60)
    print("Experiment 3: Protocol Perturbation (Split-half Stability)")
    print("=" * 60)

    df = load_data()
    rows = []

    for dim in DIMENSIONS:
        df_dim = df[df["dimension"] == dim]

        for pool in POOLS:
            if pool == "all":
                df_pool = df_dim
            elif pool == "crowd":
                df_pool = df_dim[df_dim["rater_type"] == "turker"]
            else:
                df_pool = df_dim[df_dim["rater_type"] == "expert"]

            n_raters = df_pool["rater"].nunique()

            for protocol in PROTOCOLS:
                print(f"  [{dim}][{pool}][{protocol}] ", end="", flush=True)

                if n_raters < 4:
                    print("SKIP (too few raters)")
                    rows.append({
                        "dimension": dim,
                        "pool": pool,
                        "protocol": protocol,
                        "mean_corr": np.nan,
                        "std_corr": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "n_raters": n_raters,
                    })
                    continue

                result = compute_split_half_stability(
                    df_pool, "target", "rater", "score",
                    agg_func=protocol, n_splits=N_SPLITS, seed=SEED,
                )

                rows.append({
                    "dimension": dim,
                    "pool": pool,
                    "protocol": protocol,
                    "mean_corr": result["mean_corr"],
                    "std_corr": result["std_corr"],
                    "ci_lower": result["ci95"][0],
                    "ci_upper": result["ci95"][1],
                    "n_raters": n_raters,
                })
                print(f"r={result['mean_corr']:.3f} [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]")

    df_table = pd.DataFrame(rows)
    df_table.to_csv(TABLE_OUT, index=False)
    print(f"\nSaved: {TABLE_OUT}")

    # Save as JSON too
    with open(BOOT_OUT, "w") as f:
        json.dump(rows, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"Saved: {BOOT_OUT}")

    # Figure 5: split-half stability by protocol (all pool only for clarity)
    print("\nGenerating figures...")
    fig5_df = df_table[df_table["pool"] == "all"]
    fig5_protocol_stability(fig5_df)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY — Table 3 (all pool)")
    print("=" * 60)
    print(fig5_df[["dimension", "protocol", "mean_corr"]].to_string(index=False, float_format="%.3f"))

    return df_table


if __name__ == "__main__":
    run()
