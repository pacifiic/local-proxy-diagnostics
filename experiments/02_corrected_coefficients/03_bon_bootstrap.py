#!/usr/bin/env python3
"""BoN bootstrap diagnostics from reproduced score matrices.

Instead of using synthetic heteroscedastic distributions, this diagnostic
bootstraps directly from the prepared score matrices.

For each dataset:
  - "Gold" = mean score across available raters/models for that item (J_i)
  - "Proxy" = score from one randomly selected rater/model
  - BoN: sample N items, pick the one with highest proxy, record its gold

The script then fits Q(d) = A sqrt(d) + B d at small d and compares the fitted
coefficients with the local-law predictions.
"""
import sys
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import matplotlib

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
FIG_OUT = ROOT / "results" / "figures" / "fig_bon_bridge_bootstrap.png"
RESULTS_PDF_OUT = ROOT / "results" / "figures" / "fig_bon_bridge_bootstrap.pdf"
PAPER_PDF_OUT = ROOT / "artifacts" / "figures" / "fig_bon_bridge_bootstrap.pdf"
TABLE_OUT = ROOT / "results" / "tables" / "table08v2_bon_bootstrap.csv"

N_SIMS = 10_000
# Focus on small N to test local expansion, plus larger N for full picture
N_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 128, 256, 512, 1024]
SEED = 42
CANONICAL_PDF_DATE = b"D:20000101000000Z"
PDF_DATE_FIELD_RE = re.compile(rb"(/(CreationDate|ModDate)\s*\()([^)]*)(\))")


def normalize_pdf_metadata(path):
    data = path.read_bytes()

    def replace_date(match):
        value = match.group(3)
        if len(value) != len(CANONICAL_PDF_DATE):
            return match.group(0)
        return match.group(1) + CANONICAL_PDF_DATE + match.group(4)

    path.write_bytes(PDF_DATE_FIELD_RE.sub(replace_date, data))


def panel_label(ax, label):
    """Use the same bold panel-label placement as the paper figures."""
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def panel_title(ax, title):
    ax.set_title(title, fontsize=12, fontweight="normal", pad=8)


def load_summeval_pivot(dimension, rater_type):
    """Load SummEval data as (n_items × n_raters) pivot matrix."""
    path = DATA_ROOT / "summeval_long.csv"
    df = pd.read_csv(path)
    sub = df[(df["dimension"] == dimension) & (df["rater_type"] == rater_type)]
    # Create unique item id
    sub = sub.copy()
    sub["item"] = sub["article_id"] + "_" + sub["model_id"]
    pivot = sub.pivot_table(index="item", columns="rater_idx", values="score", aggfunc="first")
    return pivot.dropna().values  # shape: (n_items, n_raters)




def load_rewardbench_pivot():
    """Load the prepared RewardBench pivot and z-normalize each model column."""
    path = DATA_ROOT / "rewardbench" / "rewardbench_pivot.csv"
    df = pd.read_csv(path, index_col=0)
    X = df.values
    # Z-normalize per model using nan-aware moments for partial snapshots.
    col_mean = np.nanmean(X, axis=0, keepdims=True)
    col_std = np.nanstd(X, axis=0, keepdims=True)
    col_std[col_std < 1e-8] = 1.0
    X_z = (X - col_mean) / col_std
    return X_z  # shape: (n_items, n_models)


def compute_cumulants_from_matrix(X):
    """Compute local-law moments from a rater/model matrix."""
    n_items, n_raters = X.shape

    # Handle NaN: compute per-item means and residuals
    item_means = np.nanmean(X, axis=1)  # J_i estimates
    grand_mean = np.nanmean(item_means)

    J = item_means - grand_mean  # centered true quality

    # Collect all (J_i, r_ij) pairs where r_ij is not NaN
    J_all, r_all = [], []
    for i in range(n_items):
        for j in range(n_raters):
            if not np.isnan(X[i, j]):
                J_all.append(J[i])
                r_all.append(X[i, j] - grand_mean)
    J_all = np.array(J_all)
    r_all = np.array(r_all)
    eps_all = r_all - J_all

    sigma_r_sq = np.var(r_all)
    sigma_J_sq = np.var(J)
    sigma_r = np.sqrt(sigma_r_sq)
    nu = np.sqrt(sigma_J_sq / sigma_r_sq) if sigma_r_sq > 0 else 0
    rho = np.mean(J_all * eps_all)

    # Cumulants
    kappa_r = np.mean(r_all**3)  # third central moment of r
    kappa_Jrr = np.mean(J_all * r_all**2) - sigma_r_sq * np.mean(J_all)
    # Since J_all is centered, E[J]=0, so kappa_Jrr = E[J * r²]

    # Corrected A and B
    A = np.sqrt(2) * (sigma_J_sq + rho) / sigma_r
    B_term1 = kappa_Jrr / sigma_r_sq
    B_term2 = 2 * (sigma_J_sq + rho) * kappa_r / (3 * sigma_r_sq**2)
    B = B_term1 - B_term2

    return {
        "sigma_r_sq": sigma_r_sq, "sigma_J_sq": sigma_J_sq,
        "nu": nu, "rho": rho,
        "kappa_r": kappa_r, "kappa_Jrr": kappa_Jrr,
        "A": A, "B": B, "B_term1": B_term1, "B_term2": B_term2,
        "n_items": n_items, "n_raters": n_raters,
    }


def bon_bootstrap(X, N_values, n_sims=N_SIMS, seed=SEED):
    """Vectorized BoN simulation via bootstrap from real data matrix X.

    For each trial:
      1. Sample N items (with replacement)
      2. For each item, sample one rater's score as proxy
      3. Select item with highest proxy
      4. Record gold (mean of all raters for that item)
    """
    rng = np.random.RandomState(seed)
    n_items, n_raters = X.shape

    # Precompute gold (item means)
    gold_values = np.nanmean(X, axis=1)  # shape: (n_items,)

    # Handle NaN: fill NaN with a per-item random valid score for vectorization
    # Build a "clean" matrix where NaN is replaced by a random valid entry
    X_clean = X.copy()
    nan_mask = np.isnan(X_clean)
    if nan_mask.any():
        # For items with NaN, count valid raters
        n_valid = (~nan_mask).sum(axis=1)  # per item
        for i in range(n_items):
            if nan_mask[i].any():
                valid_cols = np.where(~nan_mask[i])[0]
                if len(valid_cols) > 0:
                    # Fill NaN columns with random valid values
                    for j in np.where(nan_mask[i])[0]:
                        X_clean[i, j] = X[i, valid_cols[rng.randint(len(valid_cols))]]

    d_values, gold_means, proxy_means = [], [], []

    for N in N_values:
        # Vectorized: sample items and raters
        items = rng.randint(0, n_items, size=(n_sims, N))  # (n_sims, N)
        raters = rng.randint(0, n_raters, size=(n_sims, N))  # (n_sims, N)

        # Get proxy scores: X_clean[items[i,k], raters[i,k]]
        proxies = X_clean[items, raters]  # (n_sims, N)

        # Best-of-N
        best_k = np.argmax(proxies, axis=1)  # (n_sims,)
        best_items = items[np.arange(n_sims), best_k]

        gold_batch = gold_values[best_items]
        proxy_batch = proxies[np.arange(n_sims), best_k]

        d_values.append(np.log(N) if N > 1 else 0.0)
        gold_means.append(np.mean(gold_batch))
        proxy_means.append(np.mean(proxy_batch))

    return np.array(d_values), np.array(gold_means), np.array(proxy_means)


def fit_local_surrogate(d, gold, max_d):
    """Fit Q(d)-Q(0) = A√d + Bd for d ≤ max_d."""
    mask = (d > 0) & (d <= max_d)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    d_f = d[mask]
    y = gold[mask] - gold[0]
    X_mat = np.column_stack([np.sqrt(d_f), d_f])
    coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    A_fit, B_fit = coeffs

    pred = X_mat @ coeffs
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return A_fit, B_fit, r2


def run():
    print("=" * 70)
    print("BoN bootstrap from reproduced score matrices")
    print("=" * 70)

    # Load all datasets
    datasets = []

    # SummEval expert (4 dimensions)
    for dim in ["coherence", "consistency", "fluency", "relevance"]:
        try:
            X = load_summeval_pivot(dim, "expert")
            if X.shape[0] > 10:
                datasets.append({"name": f"SE-exp-{dim[:5]}", "X": X, "domain": "NLP/Likert"})
        except Exception as e:
            print(f"  Warning: SE-exp-{dim}: {e}")

    # SummEval crowd (4 dimensions)
    for dim in ["coherence", "consistency", "fluency", "relevance"]:
        try:
            X = load_summeval_pivot(dim, "crowd")
            if X.shape[0] > 10:
                datasets.append({"name": f"SE-cro-{dim[:5]}", "X": X, "domain": "NLP/Likert"})
        except Exception as e:
            print(f"  Warning: SE-cro-{dim}: {e}")

    # RewardBench
    try:
        X = load_rewardbench_pivot()
        datasets.append({"name": "RwdBench", "X": X, "domain": "Neural RM"})
    except Exception as e:
        print(f"  Warning: RewardBench: {e}")

    results = []
    all_traj = {}

    for ds in datasets:
        name = ds["name"]
        X = ds["X"]
        print(f"\n  {name}: {X.shape[0]} items × {X.shape[1]} raters")

        # Compute theoretical cumulants
        cums = compute_cumulants_from_matrix(X)
        print(f"    ν={cums['nu']:.4f}, A_th={cums['A']:.4f}, B_th={cums['B']:+.6f}")
        print(f"    κ_r={cums['kappa_r']:+.4f}, κ_Jrr={cums['kappa_Jrr']:+.4f}, "
              f"ρ={cums['rho']:+.6f}")

        # Bootstrap BoN
        d, gold, proxy = bon_bootstrap(X, N_VALUES, n_sims=N_SIMS, seed=SEED)
        all_traj[name] = {"d": d, "gold": gold, "proxy": proxy, "cums": cums, "domain": ds["domain"]}

        # Fit local surrogate at different ranges
        for max_d in [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]:
            A_f, B_f, r2 = fit_local_surrogate(d, gold, max_d)
            if np.isnan(A_f):
                continue
            results.append({
                "dataset": name,
                "domain": ds["domain"],
                "nu": cums["nu"],
                "A_theory": cums["A"],
                "B_theory": cums["B"],
                "max_d": max_d,
                "A_fit": A_f,
                "B_fit": B_f,
                "r2": r2,
            })

        # Print key fits
        A_s, B_s, r2_s = fit_local_surrogate(d, gold, 1.5)
        A_f, B_f, r2_f = fit_local_surrogate(d, gold, 7.0)
        print(f"    Fit d≤1.5: A={A_s:.4f}, B={B_s:+.6f}, R²={r2_s:.4f}")
        print(f"    Fit d≤7.0: A={A_f:.4f}, B={B_f:+.6f}, R²={r2_f:.4f}")
        print(f"    Gold(N=1024)={gold[-1]:.4f}, Gap={proxy[-1]-gold[-1]:.4f}")

    df = pd.DataFrame(results)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_OUT, index=False)
    print(f"\n→ Saved: {TABLE_OUT}")

    # ── Key comparison: Theory vs Bootstrap at small d ──
    print(f"\n{'='*60}")
    print("  Theory vs Bootstrap Comparison (d ≤ 1.5):")
    small_d = df[df["max_d"] == 1.5].dropna()
    if len(small_d) > 2:
        r_A = np.corrcoef(small_d["A_theory"], small_d["A_fit"])[0, 1]
        r_B = np.corrcoef(small_d["B_theory"], small_d["B_fit"])[0, 1]
        print(f"  Corr(A_theory, A_fit): {r_A:.4f}")
        print(f"  Corr(B_theory, B_fit): {r_B:.4f}")

        # Sign agreement for B
        sign_match = np.sum(np.sign(small_d["B_theory"]) == np.sign(small_d["B_fit"]))
        print(f"  B sign agreement: {sign_match}/{len(small_d)}")

    print("\n  Theory vs Bootstrap (d ≤ 3.0):")
    med_d = df[df["max_d"] == 3.0].dropna()
    if len(med_d) > 2:
        r_A = np.corrcoef(med_d["A_theory"], med_d["A_fit"])[0, 1]
        r_B = np.corrcoef(med_d["B_theory"], med_d["B_fit"])[0, 1]
        print(f"  Corr(A_theory, A_fit): {r_A:.4f}")
        print(f"  Corr(B_theory, B_fit): {r_B:.4f}")
        sign_match = np.sum(np.sign(med_d["B_theory"]) == np.sign(med_d["B_fit"]))
        print(f"  B sign agreement: {sign_match}/{len(med_d)}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Best-of-N bootstrap: theory vs real-data simulation",
                 fontsize=13, fontweight="normal")

    domain_colors = {"NLP/Likert": "#1f77b4", "Grant": "#d62728", "Neural RM": "#2ca02c"}

    # Panel A: Gold trajectories (gain over N=1)
    ax = axes[0, 0]
    for name, traj in all_traj.items():
        d, gold = traj["d"], traj["gold"]
        color = domain_colors.get(traj["domain"], "gray")
        ax.plot(d, gold - gold[0], color=color, alpha=0.5,
                marker="o", markersize=2, linewidth=1, label=name)
    ax.set_xlabel("d = log(N)")
    ax.set_ylabel("Gold gain over N=1")
    panel_title(ax, "Bootstrap BoN trajectories")
    panel_label(ax, "A")
    ax.legend(fontsize=5, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel B: A_theory vs A_fit (d ≤ 1.5)
    ax = axes[0, 1]
    sub = df[df["max_d"] == 1.5].dropna()
    colors = [domain_colors.get(d, "gray") for d in sub["domain"]]
    ax.scatter(sub["A_theory"], sub["A_fit"], c=colors, s=60, edgecolors="black", linewidth=0.5)
    lims_a = [0, max(sub["A_theory"].max(), sub["A_fit"].max()) * 1.1]
    ax.plot(lims_a, lims_a, "k--", alpha=0.5, label="y=x")
    r_A = np.corrcoef(sub["A_theory"], sub["A_fit"])[0, 1] if len(sub) > 2 else 0
    ax.set_xlabel("Theory A")
    ax.set_ylabel("A (Bootstrap fit, d≤1.5)")
    panel_title(ax, f"A coefficient (r={r_A:.3f})")
    panel_label(ax, "B")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: B_theory vs B_fit (d ≤ 1.5)
    ax = axes[0, 2]
    ax.scatter(sub["B_theory"], sub["B_fit"], c=colors, s=60, edgecolors="black", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    r_B = np.corrcoef(sub["B_theory"], sub["B_fit"])[0, 1] if len(sub) > 2 else 0
    ax.set_xlabel("Theory B")
    ax.set_ylabel("B (Bootstrap fit, d≤1.5)")
    panel_title(ax, f"B coefficient d≤1.5 (r={r_B:.3f})")
    panel_label(ax, "C")
    # Quadrant shading
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.fill_between([xlim[0], 0], ylim[0], ylim[1], alpha=0.05, color="blue")
    ax.fill_between([0, xlim[1]], ylim[0], ylim[1], alpha=0.05, color="red")
    ax.grid(True, alpha=0.3)

    # Panel D: B_fit across fit ranges
    ax = axes[1, 0]
    for name, traj in all_traj.items():
        sub_ds = df[df["dataset"] == name]
        color = domain_colors.get(traj["domain"], "gray")
        ax.plot(sub_ds["max_d"], sub_ds["B_fit"], color=color, alpha=0.5,
                marker="o", markersize=3, linewidth=1)
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Fit range (max d)")
    ax.set_ylabel("B_fit")
    panel_title(ax, "B_fit vs fit range")
    panel_label(ax, "D")
    ax.grid(True, alpha=0.3)

    # Panel E: Proxy-Gold gap vs ν
    ax = axes[1, 1]
    gaps, nus = [], []
    for name, traj in all_traj.items():
        gap = traj["proxy"][-1] - traj["gold"][-1]
        gaps.append(gap)
        nus.append(traj["cums"]["nu"])
    colors_e = [domain_colors.get(all_traj[n]["domain"], "gray") for n in all_traj]
    ax.scatter(nus, gaps, c=colors_e, s=60, edgecolors="black", linewidth=0.5)
    r_gap = np.corrcoef(nus, gaps)[0, 1] if len(nus) > 2 else 0
    ax.set_xlabel("ν (ICC)")
    ax.set_ylabel("Proxy-gold gap at N=1024")
    panel_title(ax, f"Proxy gap vs ν (r={r_gap:.3f})")
    panel_label(ax, "E")
    ax.grid(True, alpha=0.3)

    # Panel F: Gold at N=1024 vs ν
    ax = axes[1, 2]
    golds_1024 = [all_traj[n]["gold"][-1] for n in all_traj]
    ax.scatter(nus, golds_1024, c=colors_e, s=60, edgecolors="black", linewidth=0.5)
    r_gold = np.corrcoef(nus, golds_1024)[0, 1] if len(nus) > 2 else 0
    ax.set_xlabel("ν (ICC)")
    ax.set_ylabel("Gold at N=1024")
    panel_title(ax, f"ν predicts gold (r={r_gold:.3f})")
    panel_label(ax, "F")
    ax.grid(True, alpha=0.3)
    # Labels
    for name, nu, g in zip(all_traj.keys(), nus, golds_1024):
        ax.annotate(name, (nu, g), fontsize=5, textcoords="offset points", xytext=(3, 3))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d) for d, c in domain_colors.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PAPER_PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_PDF_OUT, bbox_inches="tight")
    fig.savefig(PAPER_PDF_OUT, bbox_inches="tight")
    normalize_pdf_metadata(RESULTS_PDF_OUT)
    normalize_pdf_metadata(PAPER_PDF_OUT)
    print(f"→ Saved: {FIG_OUT}")
    print(f"→ Saved: {RESULTS_PDF_OUT}")
    print(f"→ Saved: {PAPER_PDF_OUT}")
    plt.close()

    return df


if __name__ == "__main__":
    run()
