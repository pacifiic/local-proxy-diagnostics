#!/usr/bin/env python3
"""  [exp01b]
Experiment 1b — BoN Simulation: ICC → Optimization Gain

This is THE key experiment connecting theory to empirics.
Uses proxy (crowd) / oracle (expert) annotator split.

Outputs: Table A, Figure A (BoN curves), Figure B (ΔJ_max vs ICC scatter).
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

from src.bon_simulation import (
    compute_proxy_oracle_scores,
    simulate_bon_curve,
    compute_delta_J_max,
    compute_delta_at_Kmax,
    compute_proxy_oracle_corr,
    compute_oracle_score_range,
    bootstrap_bon_delta,
)
from src.plotting import figA_bon_curves, figB_delta_vs_icc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = DATA_ROOT / "summeval_long.csv"
TABLE1_PATH = ROOT / "results" / "tables" / "table1_summeval_decomposition.csv"
TABLE_OUT = ROOT / "results" / "tables" / "tableA_bon_gain_summary.csv"
BOOT_OUT = ROOT / "results" / "intermediate" / "bootstrap_bon_delta.json"
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]

# Primary split: proxy = 3 crowd raters, oracle = 3 expert raters
PROXY_RATERS = [("turker", 0), ("turker", 1), ("turker", 2)]
ORACLE_RATERS = [("expert", 0), ("expert", 1), ("expert", 2)]

N_TRIALS = 500          # quick: 100 / final: 500
N_BOOT = 2000           # quick: 200 / final: 2000
N_TRIALS_PER_BOOT = 100 # quick: 30  / final: 100
SEED = 42

# Secondary split: proxy = 2 experts, oracle = 1 held-out expert
# Tests Theorem 2 theoretical setting (high-quality proxy)
PROXY_RATERS_SEC = [("expert", 0), ("expert", 1)]
ORACLE_RATERS_SEC = [("expert", 2)]
TABLE_OUT_SEC = ROOT / "results" / "tables" / "tableA_bon_secondary.csv"
BOOT_OUT_SEC  = ROOT / "results" / "intermediate" / "bootstrap_bon_delta_secondary.json"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def load_icc_from_table1():
    """Load ICC(A,1) from Experiment 1 results.

    Use expert-only pool because:
    - Expert ICC measures true quality signal consistency per dimension
    - Shows strong dimension variation (relevance=0.47 ~ consistency=0.90),
      making the Figure B scatter meaningful
    - All-pool ICC (~0.22 flat) and crowd-only (~0.48 flat) show almost no
      dimension separation, making ΔJ_max ∝ ν² unverifiable
    - Theoretically: expert ICC ≈ ceiling on how much proxy selection can gain
    """
    t1 = pd.read_csv(TABLE1_PATH)
    t1_expert = t1[t1["pool"] == "expert"].set_index("dimension")
    return t1_expert["ICC_A1"].to_dict()


def run():
    print("=" * 60)
    print("Experiment 1b: BoN Simulation")
    print("=" * 60)

    df = load_data()
    icc_values = load_icc_from_table1()
    print(f"ICC(A,1) from Exp 1: {icc_values}\n")

    bon_curves = {}
    delta_results = []
    boot_all = {}

    for dim in DIMENSIONS:
        print(f"  [{dim}] Computing proxy/oracle scores ...", flush=True)

        proxy_scores, oracle_scores = compute_proxy_oracle_scores(
            df, PROXY_RATERS, ORACLE_RATERS, dim,
        )

        print(f"    Articles: {len(proxy_scores)}, Systems per article: "
              f"{len(next(iter(proxy_scores.values())))}")

        # Diagnostic: proxy-oracle correlation & oracle score range
        po_corr = compute_proxy_oracle_corr(proxy_scores, oracle_scores)
        o_range = compute_oracle_score_range(oracle_scores)
        print(f"    proxy-oracle corr = {po_corr:.4f},  oracle range = {o_range:.4f}")

        # BoN curve
        print(f"    Simulating BoN curve (n_trials={N_TRIALS}) ...", flush=True)
        curve = simulate_bon_curve(proxy_scores, oracle_scores,
                                    n_trials=N_TRIALS, seed=SEED)
        bon_curves[dim] = curve

        delta = compute_delta_J_max(curve)
        delta_raw = compute_delta_at_Kmax(curve)
        print(f"    ΔJ_max = {delta:.4f},  Δ@K_max = {delta_raw:.4f}")

        # Bootstrap
        print(f"    Bootstrap (n_boot={N_BOOT}) ...", flush=True)
        boot = bootstrap_bon_delta(
            df, dim, PROXY_RATERS, ORACLE_RATERS,
            n_boot=N_BOOT, n_trials_per_boot=N_TRIALS_PER_BOOT, seed=SEED,
        )
        print(f"    ΔJ_max 95% CI: [{boot['ci95_lower']:.4f}, {boot['ci95_upper']:.4f}]")
        print(f"    Δ@K_max 95% CI: [{boot['raw_ci95_lower']:.4f}, {boot['raw_ci95_upper']:.4f}]")

        icc_a1 = icc_values.get(dim, np.nan)
        delta_results.append({
            "dimension":              dim,
            "ICC_A1":                 icc_a1,
            "delta_J_max":            delta,
            "delta_ci_lo":            boot["ci95_lower"],
            "delta_ci_hi":            boot["ci95_upper"],
            "delta_at_Kmax":          delta_raw,
            "delta_at_Kmax_ci_lo":    boot["raw_ci95_lower"],
            "delta_at_Kmax_ci_hi":    boot["raw_ci95_upper"],
            "proxy_oracle_corr":      po_corr,
            "oracle_score_range":     o_range,
            "ratio_delta_over_icc_sq": delta / (icc_a1 ** 2) if icc_a1 > 0 else np.nan,
        })
        boot_all[dim] = {
            "mean":           float(boot["mean"]),
            "std":            float(boot["std"]),
            "ci95_lower":     float(boot["ci95_lower"]),
            "ci95_upper":     float(boot["ci95_upper"]),
            "raw_mean":       float(boot["raw_mean"]),
            "raw_std":        float(boot["raw_std"]),
            "raw_ci95_lower": float(boot["raw_ci95_lower"]),
            "raw_ci95_upper": float(boot["raw_ci95_upper"]),
        }

    # Table A
    df_table = pd.DataFrame(delta_results)
    df_table.to_csv(TABLE_OUT, index=False)
    print(f"\nSaved: {TABLE_OUT}")

    # Bootstrap intermediate
    with open(BOOT_OUT, "w") as f:
        json.dump(boot_all, f, indent=2)
    print(f"Saved: {BOOT_OUT}")

    # Figures
    print("\nGenerating figures...")
    figA_bon_curves(bon_curves)

    # For Fig B, also fetch ICC CI from bootstrap
    boot_icc_path = ROOT / "results" / "intermediate" / "bootstrap_summeval_icc.json"
    boot_icc = {}
    if boot_icc_path.exists():
        with open(boot_icc_path) as f:
            boot_icc = json.load(f)
        for r in delta_results:
            dim = r["dimension"]
            key = f"{dim}_expert"
            if key in boot_icc:
                r["ICC_A1_ci_lo"] = boot_icc[key]["a1"]["ci95_lower"]
                r["ICC_A1_ci_hi"] = boot_icc[key]["a1"]["ci95_upper"]
            else:
                r["ICC_A1_ci_lo"] = r["ICC_A1"]
                r["ICC_A1_ci_hi"] = r["ICC_A1"]
    else:
        for r in delta_results:
            r["ICC_A1_ci_lo"] = r["ICC_A1"]
            r["ICC_A1_ci_hi"] = r["ICC_A1"]

    figB_delta_vs_icc(pd.DataFrame(delta_results))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY — Table A: BoN Gain")
    print("=" * 60)
    print(df_table.to_string(index=False, float_format="%.4f"))

    # Check rank consistency
    icc_rank = df_table.sort_values("ICC_A1", ascending=False)["dimension"].tolist()
    delta_rank = df_table.sort_values("delta_J_max", ascending=False)["dimension"].tolist()
    print(f"\nICC(A,1) rank:  {icc_rank}")
    print(f"ΔJ_max rank:    {delta_rank}")
    match = sum(a == b for a, b in zip(icc_rank, delta_rank))
    print(f"Rank match: {match}/4  → {'SUCCESS' if match >= 3 else 'PARTIAL' if match >= 2 else 'CHECK NEEDED'}")

    # -----------------------------------------------------------------------
    # SECONDARY SPLIT: proxy = expert[0,1], oracle = expert[2]
    # Purpose: test Theorem 2 in a high-quality proxy setting and show that
    #          proxy quality moderates BoN gain relative to primary (crowd) split
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECONDARY SPLIT: proxy=expert[0,1], oracle=expert[2]")
    print("=" * 60)

    bon_curves_sec = {}
    delta_results_sec = []
    boot_all_sec = {}

    for dim in DIMENSIONS:
        print(f"  [{dim}] Computing proxy/oracle scores ...", flush=True)

        proxy_scores_sec, oracle_scores_sec = compute_proxy_oracle_scores(
            df, PROXY_RATERS_SEC, ORACLE_RATERS_SEC, dim,
        )

        po_corr_sec = compute_proxy_oracle_corr(proxy_scores_sec, oracle_scores_sec)
        o_range_sec = compute_oracle_score_range(oracle_scores_sec)
        print(f"    proxy-oracle corr = {po_corr_sec:.4f},  oracle range = {o_range_sec:.4f}")

        print(f"    Simulating BoN curve (n_trials={N_TRIALS}) ...", flush=True)
        curve_sec = simulate_bon_curve(
            proxy_scores_sec, oracle_scores_sec,
            n_trials=N_TRIALS, seed=SEED,
        )
        bon_curves_sec[dim] = curve_sec
        delta_sec = compute_delta_J_max(curve_sec)
        delta_raw_sec = compute_delta_at_Kmax(curve_sec)
        print(f"    ΔJ_max = {delta_sec:.4f},  Δ@K_max = {delta_raw_sec:.4f}")

        print(f"    Bootstrap (n_boot={N_BOOT}) ...", flush=True)
        boot_sec = bootstrap_bon_delta(
            df, dim, PROXY_RATERS_SEC, ORACLE_RATERS_SEC,
            n_boot=N_BOOT, n_trials_per_boot=N_TRIALS_PER_BOOT, seed=SEED,
        )
        print(f"    ΔJ_max 95% CI: [{boot_sec['ci95_lower']:.4f}, {boot_sec['ci95_upper']:.4f}]")
        print(f"    Δ@K_max 95% CI: [{boot_sec['raw_ci95_lower']:.4f}, {boot_sec['raw_ci95_upper']:.4f}]")

        icc_a1 = icc_values.get(dim, np.nan)
        delta_results_sec.append({
            "dimension":              dim,
            "ICC_A1":                 icc_a1,
            "delta_J_max":            delta_sec,
            "delta_ci_lo":            boot_sec["ci95_lower"],
            "delta_ci_hi":            boot_sec["ci95_upper"],
            "delta_at_Kmax":          delta_raw_sec,
            "delta_at_Kmax_ci_lo":    boot_sec["raw_ci95_lower"],
            "delta_at_Kmax_ci_hi":    boot_sec["raw_ci95_upper"],
            "proxy_oracle_corr":      po_corr_sec,
            "oracle_score_range":     o_range_sec,
            "ratio_delta_over_icc_sq": delta_sec / (icc_a1 ** 2) if icc_a1 > 0 else np.nan,
        })
        boot_all_sec[dim] = {
            "mean":           float(boot_sec["mean"]),
            "std":            float(boot_sec["std"]),
            "ci95_lower":     float(boot_sec["ci95_lower"]),
            "ci95_upper":     float(boot_sec["ci95_upper"]),
            "raw_mean":       float(boot_sec["raw_mean"]),
            "raw_std":        float(boot_sec["raw_std"]),
            "raw_ci95_lower": float(boot_sec["raw_ci95_lower"]),
            "raw_ci95_upper": float(boot_sec["raw_ci95_upper"]),
        }

    df_table_sec = pd.DataFrame(delta_results_sec)
    df_table_sec.to_csv(TABLE_OUT_SEC, index=False)
    print(f"\nSaved: {TABLE_OUT_SEC}")

    with open(BOOT_OUT_SEC, "w") as f:
        json.dump(boot_all_sec, f, indent=2)
    print(f"Saved: {BOOT_OUT_SEC}")

    # Figures for secondary split
    print("\nGenerating figures (secondary split)...")
    figA_bon_curves(bon_curves_sec, filename="figA_bon_curves_expert_split.png")

    # Attach ICC CI to secondary results for Fig B
    if boot_icc_path.exists():
        for r in delta_results_sec:
            dim = r["dimension"]
            key = f"{dim}_expert"
            if key in boot_icc:
                r["ICC_A1_ci_lo"] = boot_icc[key]["a1"]["ci95_lower"]
                r["ICC_A1_ci_hi"] = boot_icc[key]["a1"]["ci95_upper"]
            else:
                r["ICC_A1_ci_lo"] = r["ICC_A1"]
                r["ICC_A1_ci_hi"] = r["ICC_A1"]
    else:
        for r in delta_results_sec:
            r["ICC_A1_ci_lo"] = r["ICC_A1"]
            r["ICC_A1_ci_hi"] = r["ICC_A1"]

    figB_delta_vs_icc(pd.DataFrame(delta_results_sec),
                      filename="figB_delta_J_max_vs_icc_expert_split.png")

    # Normalized Fig B for expert split — controls for ceiling effect
    # ΔJ_max_norm = ΔJ_max / oracle_score_range
    df_sec_norm = pd.DataFrame(delta_results_sec).copy()
    df_sec_norm["delta_J_max_norm"] = df_sec_norm["delta_J_max"] / df_sec_norm["oracle_score_range"]
    df_sec_norm["delta_ci_lo_norm"] = df_sec_norm["delta_ci_lo"] / df_sec_norm["oracle_score_range"]
    df_sec_norm["delta_ci_hi_norm"] = df_sec_norm["delta_ci_hi"] / df_sec_norm["oracle_score_range"]
    figB_delta_vs_icc(df_sec_norm,
                      filename="figB_delta_normalized_expert_split.png",
                      y_col="delta_J_max_norm")

    print("\n" + "=" * 60)
    print("SECONDARY SUMMARY — Table A (expert split)")
    print("=" * 60)
    print(df_table_sec.to_string(index=False, float_format="%.4f"))

    icc_rank_sec = df_table_sec.sort_values("ICC_A1", ascending=False)["dimension"].tolist()
    delta_rank_sec = df_table_sec.sort_values("delta_J_max", ascending=False)["dimension"].tolist()
    print(f"\nICC(A,1) rank:  {icc_rank_sec}")
    print(f"ΔJ_max rank:    {delta_rank_sec}")
    match_sec = sum(a == b for a, b in zip(icc_rank_sec, delta_rank_sec))
    print(f"Rank match: {match_sec}/4  → {'SUCCESS' if match_sec >= 3 else 'PARTIAL' if match_sec >= 2 else 'CHECK NEEDED'}")

    # -----------------------------------------------------------------------
    # COMPARISON: crowd proxy vs expert proxy
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: Crowd proxy vs Expert proxy")
    print("=" * 60)
    df_compare = df_table[["dimension", "ICC_A1", "delta_J_max", "delta_at_Kmax",
                            "proxy_oracle_corr", "oracle_score_range"]].rename(
        columns={
            "delta_J_max":       "dJ_max_crowd",
            "delta_at_Kmax":     "d@Kmax_crowd",
            "proxy_oracle_corr": "po_corr_crowd",
        }
    ).merge(
        df_table_sec[["dimension", "delta_J_max", "delta_at_Kmax",
                       "proxy_oracle_corr"]].rename(
            columns={
                "delta_J_max":       "dJ_max_expert",
                "delta_at_Kmax":     "d@Kmax_expert",
                "proxy_oracle_corr": "po_corr_expert",
            }
        ),
        on="dimension",
    )
    df_compare["proxy_gain"] = df_compare["dJ_max_expert"] - df_compare["dJ_max_crowd"]
    print(df_compare.to_string(index=False, float_format="%.4f"))

    return df_table


if __name__ == "__main__":
    run()
