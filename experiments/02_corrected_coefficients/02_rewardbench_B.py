#!/usr/bin/env python3
"""RewardBench corrected-B coefficient analysis.

Loads the prepared RewardBench pivot, z-normalizes each model column, computes
the corrected local-law coefficients, and summarizes the same quantities by
subset.
"""
import sys
from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
LEGACY_FIG_OUT = ROOT / "results" / "figures" / "fig07_rewardbench_B.png"
LEGACY_TABLE_OUT = ROOT / "results" / "tables" / "table07_rewardbench_B.csv"

N_BOOT = 2000
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="RewardBench corrected-B coefficient analysis")
    p.add_argument("--data-root", default=str(DATA_ROOT),
                   help="Directory containing rewardbench/rewardbench_*.csv")
    p.add_argument("--outdir", default=None,
                   help=("Output directory for table/figure. If omitted, preserves the legacy "
                         "results/tables and results/figures locations."))
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    return p.parse_args()


def compute_all_cumulants(X):
    """Compute corrected B and all component cumulants from (items × raters) matrix."""
    J = np.nanmean(X, axis=1)
    eps = X - J[:, np.newaxis]
    n_items, n_raters = X.shape

    mask = ~np.isnan(X)
    J_expanded = np.repeat(J[:, np.newaxis], n_raters, axis=1)
    J_vals = J_expanded[mask]
    eps_vals = eps[mask]
    r_vals = X[mask]

    sigma_r_sq = np.var(r_vals)
    sigma_J_sq = np.var(J)
    sigma_r = np.sqrt(sigma_r_sq)
    sigma_J = np.sqrt(sigma_J_sq)

    rho = np.cov(J_vals, eps_vals)[0, 1]
    nu = sigma_J / sigma_r if sigma_r > 0 else 0

    r_centered = r_vals - np.mean(r_vals)
    J_centered = J_vals - np.mean(J_vals)
    eps_centered = eps_vals - np.mean(eps_vals)

    kappa_r = np.mean(r_centered**3)
    kappa_Jrr = np.mean(J_centered * r_centered**2)
    kappa3_old = np.mean(eps_centered * r_centered**2)

    term1 = kappa_Jrr / sigma_r_sq if sigma_r_sq > 0 else 0
    term2 = 2 * (sigma_J_sq + rho) * kappa_r / (3 * sigma_r_sq**2) if sigma_r_sq > 0 else 0
    B = term1 - term2

    A = np.sqrt(2) * (sigma_J_sq + rho) / sigma_r if sigma_r > 0 else 0

    return {
        "sigma_r_sq": sigma_r_sq, "sigma_J_sq": sigma_J_sq,
        "rho": rho, "nu": nu,
        "kappa_r": kappa_r, "kappa_Jrr": kappa_Jrr,
        "kappa3_old": kappa3_old,
        "A": A, "B": B, "B_term1": term1, "B_term2": term2,
        "alpha_old": kappa3_old / sigma_r_sq if sigma_r_sq > 0 else 0,
        "n_items": n_items, "n_raters": n_raters,
    }


def bootstrap_B(X, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    boot_B = np.empty(n_boot)
    boot_k3old = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        res = compute_all_cumulants(X[idx])
        boot_B[b] = res["B"]
        boot_k3old[b] = res["kappa3_old"]
    return {
        "B_ci": (np.percentile(boot_B, 2.5), np.percentile(boot_B, 97.5)),
        "P_B_neg": np.mean(boot_B < 0),
        "boot_B": boot_B,
        "boot_k3old": boot_k3old,
    }


def load_rewardbench(data_root):
    pivot_path = Path(data_root) / "rewardbench" / "rewardbench_pivot.csv"
    pivot = pd.read_csv(pivot_path, index_col=0)
    X_raw = pivot.values.astype(float)
    # Z-normalize per model. Newer RewardBench snapshots include a few
    # partial model files, so use nan-aware moments and let downstream masks
    # ignore genuinely missing item-model scores.
    stds = np.nanstd(X_raw, axis=0)
    X_z = (X_raw - np.nanmean(X_raw, axis=0, keepdims=True)) / (stds[np.newaxis, :] + 1e-12)
    return X_z, pivot.index.values


def load_rewardbench_subsets(data_root):
    scores_path = Path(data_root) / "rewardbench" / "rewardbench_scores.csv"
    df = pd.read_csv(scores_path)
    return df


def run():
    args = parse_args()
    data_root = Path(args.data_root)
    if args.outdir:
        outdir = Path(args.outdir)
        table_out = outdir / "table07_rewardbench_B.csv"
        fig_out = outdir / "fig07_rewardbench_B.png"
        spec_out = outdir / "run_spec_rewardbench_B.json"
    else:
        table_out = LEGACY_TABLE_OUT
        fig_out = LEGACY_FIG_OUT
        spec_out = ROOT / "results" / "tables" / "run_spec_rewardbench_B.json"

    print("=" * 70)
    print("RewardBench corrected-B coefficient analysis")
    print("=" * 70)

    X_z, item_ids = load_rewardbench(data_root)
    print(f"Loaded: {X_z.shape[0]} items × {X_z.shape[1]} models (Z-normalized)")

    results = []

    # ── Overall ──
    print(f"\n{'─'*50}")
    print("  RewardBench (Overall, Z-normalized)")
    est = compute_all_cumulants(X_z)
    boot = bootstrap_B(X_z, n_boot=args.n_boot)

    print(f"  σ_r² = {est['sigma_r_sq']:.6f}")
    print(f"  σ_J² = {est['sigma_J_sq']:.6f}")
    print(f"  ν    = {est['nu']:.4f}")
    print(f"  ρ    = {est['rho']:+.6f}")
    print(f"  κ_r  = {est['kappa_r']:+.6f}")
    print(f"  κ_{{Jrr}} = {est['kappa_Jrr']:+.6f}")
    print(f"  κ₃_old = {est['kappa3_old']:+.6f}")
    print(f"  B_term1 = {est['B_term1']:+.6f}")
    print(f"  B_term2 = {est['B_term2']:+.6f}")
    print(f"  B       = {est['B']:+.6f}  [{boot['B_ci'][0]:+.6f}, {boot['B_ci'][1]:+.6f}]")
    print(f"  -α_old  = {-est['alpha_old']:+.6f}")
    print(f"  P(B<0)  = {boot['P_B_neg']:.3f}")

    results.append({
        "dataset": "RewardBench-Overall",
        **{k: est[k] for k in est},
        "B_ci_lo": boot["B_ci"][0], "B_ci_hi": boot["B_ci"][1],
        "P_B_neg": boot["P_B_neg"],
    })

    # ── Per subset ──
    try:
        df_scores = load_rewardbench_subsets(data_root)
        subsets = df_scores["subset"].unique()

        # Group subsets into categories
        subset_groups = {}
        for s in subsets:
            if "alpaca" in s.lower() or "mt-bench" in s.lower():
                cat = "chat"
            elif "refusal" in s.lower() or "xstest" in s.lower() or "donotanswer" in s.lower():
                cat = "safety"
            elif "math" in s.lower() or "hep" in s.lower() or "human" in s.lower():
                cat = "reasoning"
            else:
                cat = "other"
            if cat not in subset_groups:
                subset_groups[cat] = []
            subset_groups[cat].append(s)

        for cat, cat_subsets in sorted(subset_groups.items()):
            cat_items = df_scores[df_scores["subset"].isin(cat_subsets)]["item_id"].unique()
            # Find matching rows in pivot
            mask = np.isin(np.arange(X_z.shape[0]), cat_items)
            if mask.sum() < 50:
                continue

            X_sub = X_z[mask]
            print(f"\n  RewardBench-{cat} ({X_sub.shape[0]} items)")
            est_s = compute_all_cumulants(X_sub)
            boot_s = bootstrap_B(X_sub, n_boot=min(1000, args.n_boot))

            print(f"    B = {est_s['B']:+.6f}  [{boot_s['B_ci'][0]:+.6f}, {boot_s['B_ci'][1]:+.6f}]")
            print(f"    P(B<0) = {boot_s['P_B_neg']:.3f}")

            results.append({
                "dataset": f"RB-{cat}",
                **{k: est_s[k] for k in est_s},
                "B_ci_lo": boot_s["B_ci"][0], "B_ci_hi": boot_s["B_ci"][1],
                "P_B_neg": boot_s["P_B_neg"],
            })
    except Exception as e:
        print(f"  Subset analysis skipped: {e}")

    df = pd.DataFrame(results)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_out, index=False)
    print(f"\n→ Saved: {table_out}")
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.write_text(json.dumps({
        "data_root": str(data_root),
        "n_boot": args.n_boot,
        "seed": SEED,
        "n_items": int(X_z.shape[0]),
        "n_models": int(X_z.shape[1]),
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"→ Saved: {spec_out}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("RewardBench corrected B coefficient", fontweight="normal")

    # Panel A: B decomposition
    ax = axes[0]
    ax.bar(["κ_{Jrr}/σ²_r", "−2(σ²+ρ)κ_r/(3σ⁴_r)", "B (net)"],
           [est["B_term1"], -est["B_term2"], est["B"]],
           color=["steelblue", "salmon", "black"], alpha=0.8)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("Coefficient value")
    ax.set_title("A. B decomposition (overall)", fontweight="normal")

    # Panel B: Bootstrap
    ax = axes[1]
    ax.hist(boot["boot_B"], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linewidth=2, linestyle="--", label="B = 0")
    ax.axvline(est["B"], color="black", linewidth=2, label=f"B = {est['B']:+.4f}")
    ax.set_xlabel("B (bootstrap)")
    ax.set_ylabel("Count")
    ax.set_title("B. Bootstrap distribution", fontweight="normal")
    ax.legend()
    ax.text(0.95, 0.95, f"P(B<0) = {boot['P_B_neg']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel C: Old κ₃ vs corrected B bootstrap
    ax = axes[2]
    ax.scatter(boot["boot_k3old"], boot["boot_B"], s=2, alpha=0.1, color="steelblue")
    ax.axhline(0, color="red", linewidth=1, alpha=0.5)
    ax.axvline(0, color="red", linewidth=1, alpha=0.5)
    ax.set_xlabel("Old κ₃(ε,r,r)")
    ax.set_ylabel("Corrected B")
    ax.set_title("C. Old κ₃ vs corrected B", fontweight="normal")

    plt.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")
    print(f"→ Saved: {fig_out}")
    pdf_out = fig_out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"→ Saved: {pdf_out}")
    plt.close()

    return df


if __name__ == "__main__":
    run()
