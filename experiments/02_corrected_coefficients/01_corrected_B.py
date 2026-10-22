#!/usr/bin/env python3
"""  [exp06]
Experiment 11 — Corrected B Coefficient Computation
=====================================================

The corrected Theorem 2 gives:

  Q(d) = Q(0) + √2·(σ_J² + ρ)/σ_r · √d + B·d + O(d^{3/2})

where:
  B = κ_{Jrr}/σ_r² − 2·(σ_J² + ρ)·κ_r / (3·σ_r⁴)

  κ_{Jrr} = κ₃(J, r, r)   [mixed third cumulant of J with r,r]
  κ_r     = κ₃(r, r, r)    [third cumulant = skewness of r]
  ρ       = Cov(J, ε)      [correlation between true quality and noise]
  σ_r²    = Var(r)          [total variance of scores]
  σ_J²    = Var(J)          [true quality variance]

Previously we only computed κ₃(ε, r, r). Now we need the FULL B.

Key question: Does B have the same sign as the old κ₃(ε,r,r)?

Data: SummEval (expert + crowd)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
sys.path.insert(0, str(ROOT / "lib"))

FIG_OUT = ROOT / "results" / "figures" / "fig_corrected_B.png"
TABLE_OUT = ROOT / "results" / "tables" / "table_corrected_B.csv"

N_BOOT = 2000
SEED = 42


# =============================================================
# Core: Compute all quantities from (items × raters) matrix
# =============================================================

def compute_all_cumulants(X):
    """
    From an (n_items, n_raters) score matrix X, compute:

    Notation:
      r_ij = score of item i by rater j  (= X[i,j])
      J_i  = item mean = mean_j(r_ij)    (proxy for true quality)
      ε_ij = r_ij - J_i                   (residual = noise)

    Computes:
      σ_r²     = Var(r)           total variance of all scores
      σ_J²     = Var(J)           variance of item means
      ρ        = Cov(J, ε)        covariance of true quality and noise
      κ_r      = κ₃(r,r,r)       skewness of score distribution (unnormalized)
      κ_{Jrr}  = κ₃(J,r,r)       mixed third cumulant
      κ₃_old   = κ₃(ε,r,r)       the OLD quantity we computed before
      B        = corrected d-coefficient
      A        = corrected √d coefficient
    """
    n_items, n_raters = X.shape

    # Item means (proxy for J)
    J = np.nanmean(X, axis=1)  # (n_items,)

    # Residuals (proxy for ε)
    eps = X - J[:, np.newaxis]  # (n_items, n_raters)

    # Grand mean
    grand_mean = np.nanmean(X)

    # ── Basic variances ──
    r_flat = X[~np.isnan(X)].ravel()
    sigma_r_sq = np.var(r_flat)
    sigma_r = np.sqrt(sigma_r_sq)

    sigma_J_sq = np.var(J)
    sigma_J = np.sqrt(sigma_J_sq)

    # ── ρ = Cov(J, ε) ──
    # For each (i,j) pair, J_i and ε_ij
    # Since J_i = mean(r_ij), and ε_ij = r_ij - J_i,
    # Cov(J, ε) across all (i,j) pairs
    J_expanded = np.repeat(J[:, np.newaxis], n_raters, axis=1)
    mask = ~np.isnan(X)

    J_vals = J_expanded[mask]
    eps_vals = eps[mask]
    r_vals = X[mask]

    rho = np.cov(J_vals, eps_vals)[0, 1]

    # ── κ_r = κ₃(r, r, r) = E[(r - μ_r)³] ──
    r_centered = r_vals - np.mean(r_vals)
    kappa_r = np.mean(r_centered**3)

    # ── κ_{Jrr} = κ₃(J, r, r) ──
    # = E[(J - μ_J)(r - μ_r)²] - 2·Cov(J,r)·E[(r - μ_r)] - Var(r)·E[(J - μ_J)]
    # Since E[(r-μ_r)] = 0 and E[(J-μ_J)] = 0 (centered),
    # κ₃(J,r,r) = E[(J - μ_J)(r - μ_r)²]  (for centered variables, this IS the third cumulant)
    #
    # Actually, for the third cumulant:
    # κ₃(X,Y,Z) = E[(X-μ_X)(Y-μ_Y)(Z-μ_Z)]  when all means are removed
    # This is true for third order (unlike higher orders where cumulants ≠ central moments)

    J_centered = J_vals - np.mean(J_vals)
    kappa_Jrr = np.mean(J_centered * r_centered**2)

    # ── κ₃_old = κ₃(ε, r, r) = E[(ε - μ_ε)(r - μ_r)²] ──
    eps_centered = eps_vals - np.mean(eps_vals)
    kappa3_old = np.mean(eps_centered * r_centered**2)

    # ── Also compute κ₃(ε,r,r) the old way for verification ──
    # Old way: E[ε · (r - μ_r)²] where ε already has mean ≈ 0
    kappa3_old_v2 = np.mean(eps_vals * r_centered**2)

    # ── Corrected coefficients ──
    # A = √2 · (σ_J² + ρ) / σ_r
    A = np.sqrt(2) * (sigma_J_sq + rho) / sigma_r

    # B = κ_{Jrr}/σ_r² − 2·(σ_J² + ρ)·κ_r / (3·σ_r⁴)
    term1 = kappa_Jrr / sigma_r_sq
    term2 = 2 * (sigma_J_sq + rho) * kappa_r / (3 * sigma_r_sq**2)
    B = term1 - term2

    # Old α = κ₃(ε,r,r) / σ_r²
    alpha_old = kappa3_old / sigma_r_sq

    # ── ν = σ_J / σ_r (signal-to-total ratio) ──
    nu = sigma_J / sigma_r

    # ── A under ρ=0: √2·σ_J·ν ──
    A_rho0 = np.sqrt(2) * sigma_J * nu

    return {
        "sigma_r_sq": sigma_r_sq,
        "sigma_J_sq": sigma_J_sq,
        "rho": rho,
        "nu": nu,
        "kappa_r": kappa_r,
        "kappa_Jrr": kappa_Jrr,
        "kappa3_old": kappa3_old,
        "kappa3_old_v2": kappa3_old_v2,
        "A": A,
        "A_rho0": A_rho0,
        "B": B,
        "B_term1": term1,
        "B_term2": term2,
        "alpha_old": alpha_old,
        "n_items": n_items,
        "n_raters": n_raters,
    }


def bootstrap_B(X, n_boot=N_BOOT, seed=SEED):
    """Bootstrap B and other quantities by resampling items."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]

    boot_B = np.empty(n_boot)
    boot_kappa3_old = np.empty(n_boot)
    boot_kappa_Jrr = np.empty(n_boot)
    boot_kappa_r = np.empty(n_boot)
    boot_rho = np.empty(n_boot)
    boot_term1 = np.empty(n_boot)
    boot_term2 = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        res = compute_all_cumulants(X[idx])
        boot_B[b] = res["B"]
        boot_kappa3_old[b] = res["kappa3_old"]
        boot_kappa_Jrr[b] = res["kappa_Jrr"]
        boot_kappa_r[b] = res["kappa_r"]
        boot_rho[b] = res["rho"]
        boot_term1[b] = res["B_term1"]
        boot_term2[b] = res["B_term2"]

    def ci(arr):
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    return {
        "B_ci": ci(boot_B),
        "B_p_negative": np.mean(boot_B < 0),
        "kappa3_old_ci": ci(boot_kappa3_old),
        "kappa_Jrr_ci": ci(boot_kappa_Jrr),
        "kappa_r_ci": ci(boot_kappa_r),
        "rho_ci": ci(boot_rho),
        "term1_ci": ci(boot_term1),
        "term2_ci": ci(boot_term2),
        "boot_B": boot_B,
        "boot_kappa3_old": boot_kappa3_old,
    }


# =============================================================
# Data loading
# =============================================================

def load_summeval_matrix(rater_type, dimension):
    """Load SummEval as (items × raters) matrix."""
    df = pd.read_csv(DATA_ROOT / "summeval_long.csv")
    sub = df[(df["rater_type"] == rater_type) & (df["dimension"] == dimension)]

    # Create item ID from article_id + model_id
    sub = sub.copy()
    sub["item_id"] = sub["article_id"] + "_" + sub["model_id"]

    pivot = sub.pivot_table(index="item_id", columns="rater_idx",
                            values="score", aggfunc="first")
    X = pivot.values.astype(float)

    # Remove items with < 2 ratings
    valid = np.sum(~np.isnan(X), axis=1) >= 2
    X = X[valid]

    return X



# =============================================================
# Main
# =============================================================

def run():
    print("=" * 70)
    print("Experiment 11: Corrected B Coefficient")
    print("=" * 70)
    print()
    print("B = κ_{Jrr}/σ_r² − 2·(σ_J²+ρ)·κ_r / (3·σ_r⁴)")
    print("Old: -α = -κ₃(ε,r,r)/σ_r²")
    print()

    results = []
    all_boots = {}

    # ── SummEval ──
    for rater_type in ["expert", "crowd"]:
        for dimension in ["coherence", "consistency", "fluency", "relevance"]:
            label = f"SummEval-{rater_type[:3]}-{dimension[:5]}"
            print(f"\n{'─'*60}")
            print(f"  {label}")

            # Map rater_type name
            rt = "expert" if rater_type == "expert" else "turker"
            X = load_summeval_matrix(rt if rt == "expert" else "turker", dimension)
            print(f"  Matrix: {X.shape[0]} items × {X.shape[1]} raters")

            est = compute_all_cumulants(X)
            boot = bootstrap_B(X)

            print(f"  σ_r² = {est['sigma_r_sq']:.4f}")
            print(f"  σ_J² = {est['sigma_J_sq']:.4f}")
            print(f"  ν    = {est['nu']:.4f}")
            print(f"  ρ    = {est['rho']:+.6f}  {boot['rho_ci']}")
            print(f"  κ_r  = {est['kappa_r']:+.6f}  {boot['kappa_r_ci']}")
            print(f"  κ_{{Jrr}} = {est['kappa_Jrr']:+.6f}  {boot['kappa_Jrr_ci']}")
            print(f"  κ₃_old = {est['kappa3_old']:+.6f}  {boot['kappa3_old_ci']}")
            print(f"")
            print(f"  B_term1 (κ_{{Jrr}}/σ_r²)        = {est['B_term1']:+.6f}")
            print(f"  B_term2 (2(σ²+ρ)κ_r/3σ_r⁴)    = {est['B_term2']:+.6f}")
            print(f"  ────────────────────────────────────")
            print(f"  B (corrected)                   = {est['B']:+.6f}  {boot['B_ci']}")
            print(f"  -α (old)                        = {-est['alpha_old']:+.6f}")
            print(f"  P(B < 0) = {boot['B_p_negative']:.3f}")
            print(f"  Sign match: {'YES' if (est['B'] < 0) == (est['kappa3_old'] < 0) else 'NO'}")

            results.append({
                "dataset": label,
                "n_items": est["n_items"],
                "n_raters": est["n_raters"],
                "sigma_r_sq": est["sigma_r_sq"],
                "sigma_J_sq": est["sigma_J_sq"],
                "nu": est["nu"],
                "rho": est["rho"],
                "kappa_r": est["kappa_r"],
                "kappa_Jrr": est["kappa_Jrr"],
                "kappa3_old": est["kappa3_old"],
                "B_term1": est["B_term1"],
                "B_term2": est["B_term2"],
                "B": est["B"],
                "neg_alpha_old": -est["alpha_old"],
                "B_ci_lo": boot["B_ci"][0],
                "B_ci_hi": boot["B_ci"][1],
                "P_B_neg": boot["B_p_negative"],
                "A": est["A"],
                "A_rho0": est["A_rho0"],
                "sign_match": (est["B"] < 0) == (est["kappa3_old"] < 0),
            })
            all_boots[label] = boot

    df = pd.DataFrame(results)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_OUT, index=False)
    print(f"\n→ Saved: {TABLE_OUT}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: B vs old -α comparison")
    print(f"{'='*70}")

    n_match = df["sign_match"].sum()
    n_total = len(df)
    n_B_neg = (df["B"] < 0).sum()
    n_old_neg = (df["kappa3_old"] < 0).sum()

    print(f"\n  Sign match (B and old κ₃ same sign): {n_match}/{n_total}")
    print(f"  B < 0: {n_B_neg}/{n_total}")
    print(f"  old κ₃(ε,r,r) < 0: {n_old_neg}/{n_total}")
    print(f"  P(B < 0) range: [{df['P_B_neg'].min():.3f}, {df['P_B_neg'].max():.3f}]")

    print(f"\n  Key comparison:")
    print(f"  {'Dataset':<25} {'B':>10} {'-α_old':>10} {'Match':>6} {'|ρ/σ_J²|':>10}")
    print(f"  {'─'*65}")
    for _, row in df.iterrows():
        rho_ratio = abs(row["rho"]) / row["sigma_J_sq"] if row["sigma_J_sq"] > 0 else 0
        print(f"  {row['dataset']:<25} {row['B']:>+10.5f} {row['neg_alpha_old']:>+10.5f} "
              f"{'YES' if row['sign_match'] else 'NO':>6} {rho_ratio:>10.4f}")

    # ── Plot ──
    plot_comparison(df, all_boots)

    return df


def plot_comparison(df, all_boots):
    """4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Corrected B Coefficient vs Old κ₃(ε,r,r)",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel A: B vs -α scatter ──
    ax = axes[0, 0]
    colors = ["tab:blue" if "exp" in d else ("tab:orange" if "cro" in d else "tab:red")
              for d in df["dataset"]]

    ax.scatter(df["neg_alpha_old"], df["B"], c=colors, s=80,
               edgecolor="black", linewidth=0.5, zorder=5)

    # Identity line
    lims = [min(df["neg_alpha_old"].min(), df["B"].min()) - 0.02,
            max(df["neg_alpha_old"].max(), df["B"].max()) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.3, label="B = -α")
    ax.axhline(0, color="red", linewidth=1, alpha=0.3)
    ax.axvline(0, color="red", linewidth=1, alpha=0.3)

    for _, row in df.iterrows():
        ax.annotate(row["dataset"].replace("SummEval-", "SE-"),
                   (row["neg_alpha_old"], row["B"]),
                   fontsize=6, textcoords="offset points", xytext=(3, 3))

    r, p = stats.pearsonr(df["neg_alpha_old"], df["B"])
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("-α (old coefficient)")
    ax.set_ylabel("B (corrected coefficient)")
    ax.set_title("A. Old vs Corrected d-Coefficient", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel B: B decomposition (term1 vs term2) ──
    ax = axes[0, 1]
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["B_term1"], width, label="κ_{Jrr}/σ_r²", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, -df["B_term2"], width, label="-2(σ²+ρ)κ_r/(3σ_r⁴)", color="salmon", alpha=0.8)
    ax.scatter(x, df["B"], color="black", s=50, zorder=5, label="B (net)")

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("SummEval-", "SE-") for d in df["dataset"]],
                       rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Coefficient value")
    ax.set_title("B. B Decomposition: Two Terms", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel C: P(B < 0) across datasets ──
    ax = axes[1, 0]
    ax.barh(range(len(df)), df["P_B_neg"], color="steelblue", alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([d.replace("SummEval-", "SE-") for d in df["dataset"]], fontsize=8)
    ax.axvline(0.5, color="red", linewidth=2, linestyle="--", label="P = 0.5")
    ax.set_xlabel("P(B < 0)")
    ax.set_title("C. Bootstrap P(B < 0)", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)

    # ── Panel D: Bootstrap distributions for key datasets ──
    ax = axes[1, 1]
    key_datasets = ["SummEval-exp-coher", "SummEval-cro-coher", "SummEval-exp-consi"]
    colors_d = ["tab:blue", "tab:orange", "tab:red"]

    for label, color in zip(key_datasets, colors_d):
        if label in all_boots:
            boot = all_boots[label]
            ax.hist(boot["boot_B"], bins=40, alpha=0.5, color=color,
                    label=label.replace("SummEval-", "SE-"), density=True)

    ax.axvline(0, color="red", linewidth=2, linestyle="--")
    ax.set_xlabel("B (bootstrap)")
    ax.set_ylabel("Density")
    ax.set_title("D. Bootstrap Distribution of B", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"→ Saved: {FIG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    df = run()
