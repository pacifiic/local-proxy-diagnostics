#!/usr/bin/env python3
"""  [exp05]
Experiment 4 — OpenMEVA Cross-benchmark Replication

Replicate reliability decomposition on OpenMEVA MAGS (ROC + WP).
Outputs: Table 4, Figure 6, Figure 7.
"""

import argparse
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
from src.plotting import fig6_dataset_replication, fig7_roc_vs_wp

# ---------------------------------------------------------------------------
DATA_PATH   = DATA_ROOT / "openmeva_long.csv"
TABLE1_PATH = ROOT / "results" / "tables" / "table1_summeval_decomposition.csv"
TABLE_OUT   = ROOT / "results" / "tables" / "table4_openmeva_replication.csv"
BOOT_OUT    = ROOT / "results" / "intermediate" / "bootstrap_openmeva.json"
N_BOOT      = 2000
SEED        = 42
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenMEVA MAGS ICC replication")
    ap.add_argument("--outdir", default="", help="Optional output directory for CSV/JSON results.")
    ap.add_argument("--skip-figures", action="store_true", help="Skip figure generation.")
    ap.add_argument("--table1-path", default=str(TABLE1_PATH), help="SummEval Table 1 CSV for figure generation.")
    return ap.parse_args()


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Include the OpenMEVA source split in the target id.  ROC and WP reuse story
    # ids, so omitting dataset would collapse the combined row from 2,000
    # story-model targets to 1,000 artificial target ids.
    df["target"] = df["dataset"].astype(str) + "___" + df["story_id"].astype(str) + "___" + df["model_id"]
    df["rater"]  = "rater_" + df["rater_idx"].astype(str)
    return df


def run(
    table_out: Path = TABLE_OUT,
    boot_out: Path = BOOT_OUT,
    table1_path: Path = TABLE1_PATH,
    make_figures: bool = True,
):
    print("=" * 60)
    print("Experiment 4: OpenMEVA Cross-benchmark Replication")
    print("=" * 60)

    df = load_data()
    rows = []
    boot_results = {}

    for source in ["roc", "wp", "combined"]:
        df_src = df if source == "combined" else df[df["dataset"] == source]
        print(f"\n  [{source}] {df_src['target'].nunique()} targets, "
              f"{df_src['rater'].nunique()} raters", end="", flush=True)

        X = df_to_matrix(df_src)
        a1_val = icc_a1(X)
        c1_val = icc_c1(X)
        g_val  = (c1_val - a1_val) if not (np.isnan(a1_val) or np.isnan(c1_val)) else np.nan

        if source != "combined":
            boot_a1 = bootstrap_icc_fast(X, icc_a1, N_BOOT, SEED)
            boot_c1 = bootstrap_icc_fast(X, icc_c1, N_BOOT, SEED)
            boot_g  = bootstrap_gap_fast(X, N_BOOT, SEED)
            boot_results[source] = {
                "a1": boot_a1, "c1": boot_c1, "gap": boot_g
            }
        else:
            boot_a1 = boot_c1 = boot_g = {"ci95_lower": np.nan, "ci95_upper": np.nan}

        rows.append({
            "source":           source,
            "dimension":        "holistic",
            "N_examples":       df_src["target"].nunique(),
            "N_raters":         df_src["rater"].nunique(),
            "ICC_A1":           a1_val,
            "ICC_A1_CI_lower":  boot_a1["ci95_lower"],
            "ICC_A1_CI_upper":  boot_a1["ci95_upper"],
            "ICC_C1":           c1_val,
            "ICC_C1_CI_lower":  boot_c1["ci95_lower"],
            "ICC_C1_CI_upper":  boot_c1["ci95_upper"],
            "G_pers":           g_val,
            "G_pers_CI_lower":  boot_g["ci95_lower"],
            "G_pers_CI_upper":  boot_g["ci95_upper"],
            "mean_score":       df_src["score"].mean(),
            "std_score":        df_src["score"].std(),
        })
        print(f"  ICC(A,1)={a1_val:.3f}  ICC(C,1)={c1_val:.3f}  G_pers={g_val:.3f}")

    table_out.parent.mkdir(parents=True, exist_ok=True)
    boot_out.parent.mkdir(parents=True, exist_ok=True)

    df_table = pd.DataFrame(rows)
    df_table.to_csv(table_out, index=False)
    print(f"\nSaved: {table_out}")

    with open(boot_out, "w") as f:
        json.dump(boot_results, f, indent=2)
    print(f"Saved: {boot_out}")

    # ── Figures ──
    if make_figures:
        print("\nGenerating figures...")

        # Fig 6: SummEval vs OpenMEVA
        t1 = pd.read_csv(table1_path)
        t1_all = t1[t1["pool"] == "all"][["dimension", "ICC_A1"]].copy()
        t1_all["dataset"] = "SummEval"
        t1_all.rename(columns={"dimension": "dimension_label"}, inplace=True)

        ome = df_table[df_table["source"].isin(["roc","wp"])][["source","ICC_A1"]].copy()
        ome["dataset"] = "OpenMEVA"
        ome.rename(columns={"source": "dimension_label"}, inplace=True)

        fig6_dataset_replication(pd.concat([t1_all, ome], ignore_index=True))

        # Fig 7: ROC vs WP
        fig7_roc_vs_wp(df_table[df_table["source"].isin(["roc","wp"])][["source","ICC_A1"]])

    print("\n" + "=" * 60)
    print("SUMMARY — Table 4")
    print("=" * 60)
    print(df_table[["source","ICC_A1","ICC_C1","G_pers","mean_score"]]
          .to_string(index=False, float_format="%.3f"))

    return df_table


if __name__ == "__main__":
    args = parse_args()
    if args.outdir:
        outdir = Path(args.outdir)
        run(
            table_out=outdir / "table4_openmeva_replication.csv",
            boot_out=outdir / "bootstrap_openmeva.json",
            table1_path=Path(args.table1_path),
            make_figures=not args.skip_figures,
        )
    else:
        run(table1_path=Path(args.table1_path), make_figures=not args.skip_figures)
