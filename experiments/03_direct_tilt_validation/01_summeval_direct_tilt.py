#!/usr/bin/env python3
"""  [exp10]
Direct tilt validation on SummEval
==================================
Implements the theorem-aligned flagship experiment:
  π_t(i) ∝ exp(t r_i)
for a frozen held-out proxy r_i and held-out gold J_i (LOO mean).

Uses existing summeval_long.csv (not JSONL).
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
DATA = DATA_ROOT / "summeval_long.csv"
FIG_OUT = ROOT / "results" / "figures" / "fig_direct_tilt_summeval.png"
TABLE_OUT = ROOT / "results" / "tables" / "table_direct_tilt_summeval.csv"
SUMMARY_OUT = ROOT / "results" / "tables" / "table_direct_tilt_summeval_summary.csv"


def compute_ab(J, r):
    """Compute theoretical A, B from corrected Theorem 2."""
    J = np.asarray(J, float)
    r = np.asarray(r, float)
    Jc = J - J.mean()
    rc = r - r.mean()
    sigma_J_sq = np.mean(Jc**2)
    sigma_r_sq = np.mean(rc**2)
    sigma_r = np.sqrt(sigma_r_sq)
    eps = r - J
    rho = np.mean(Jc * (eps - eps.mean()))
    kappa_r = np.mean(rc**3)
    kappa_Jrr = np.mean(Jc * rc**2)
    A = np.sqrt(2) * (sigma_J_sq + rho) / sigma_r
    B = kappa_Jrr / sigma_r_sq - 2 * (sigma_J_sq + rho) * kappa_r / (3 * sigma_r_sq**2)
    nu_tilde = np.sqrt(sigma_J_sq / sigma_r_sq)
    return A, B, rho, nu_tilde, sigma_J_sq, sigma_r_sq, kappa_r, kappa_Jrr


def tilt_curve(J, r, t_grid):
    """Compute exact exponential tilt Q(d) trajectory."""
    J = np.asarray(J, float)
    r = np.asarray(r, float)
    n = len(r)
    d_vals, q_vals = [], []
    for t in t_grid:
        logits = t * r
        m = np.max(logits)
        w = np.exp(logits - m)
        pi = w / w.sum()
        # KL(π_t || π_ref) where π_ref = uniform
        log_pi = np.log(pi + 1e-300)
        log_uniform = -np.log(n)
        kl = float(np.dot(pi, log_pi - log_uniform))
        d_vals.append(kl)
        q_vals.append(float(np.dot(pi, J)))
    return np.array(d_vals), np.array(q_vals)


def fit_local(d, q, max_d):
    """Fit Q(d)-Q(0) = A√d + Bd over d ≤ max_d."""
    mask = (d > 1e-10) & (d <= max_d)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.sqrt(d[mask]), d[mask]])
    y = q[mask] - q[0]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return coef[0], coef[1], r2


def main():
    print("=" * 70)
    print("Direct Tilt Experiment: SummEval")
    print("=" * 70)

    df = pd.read_csv(DATA)
    df["item_id"] = df["article_id"] + "___" + df["model_id"]

    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, 4.0, 400)])

    rows = []
    all_curves = {}

    for pool in ["expert", "crowd"]:
        rater_col = "expert" if pool == "expert" else "crowd"
        sub_pool = df[df["rater_type"] == rater_col]
        if len(sub_pool) == 0:
            # Try alternate name
            sub_pool = df[df["rater_type"] == pool]
        if len(sub_pool) == 0:
            print(f"  Warning: no data for pool={pool}, trying turker")
            sub_pool = df[df["rater_type"] == "turker"]

        for dim in ["coherence", "consistency", "fluency", "relevance"]:
            sub = sub_pool[sub_pool["dimension"] == dim]
            if len(sub) == 0:
                continue

            X = sub.pivot_table(
                index="item_id", columns="rater_idx",
                values="score", aggfunc="first"
            ).dropna().values.astype(float)

            n_items, n_raters = X.shape
            label = f"{pool[:3]}-{dim[:5]}"
            print(f"\n  {label}: {n_items} items × {n_raters} raters")

            # Aggregate across LOO splits
            curves_d, curves_q = [], []
            for j in range(n_raters):
                r_j = X[:, j]
                J_j = (X.sum(axis=1) - r_j) / (n_raters - 1)  # LOO gold

                A_th, B_th, rho, nu_t, sJ2, sr2, kr, kJrr = compute_ab(J_j, r_j)

                d, q = tilt_curve(J_j, r_j, t_grid)
                curves_d.append(d)
                curves_q.append(q)

                for max_d in [0.05, 0.10, 0.20, 0.50, 1.00, 2.00]:
                    A_fit, B_fit, r2 = fit_local(d, q, max_d)
                    rows.append({
                        "pool": pool,
                        "dimension": dim,
                        "label": label,
                        "rater_idx": j,
                        "fit_max_d": max_d,
                        "A_theory": A_th,
                        "B_theory": B_th,
                        "A_fit": A_fit,
                        "B_fit": B_fit,
                        "r2": r2,
                        "rho": rho,
                        "nu_tilde": nu_t,
                    })

            # Store mean curve for plotting
            mean_d = np.mean(curves_d, axis=0)
            mean_q = np.mean(curves_q, axis=0)
            all_curves[label] = {"d": mean_d, "q": mean_q}

            # Print summary for this dimension
            A_th_mean = np.mean([r["A_theory"] for r in rows if r["label"] == label and r["fit_max_d"] == 0.20])
            B_th_mean = np.mean([r["B_theory"] for r in rows if r["label"] == label and r["fit_max_d"] == 0.20])
            sub_fit = [r for r in rows if r["label"] == label and r["fit_max_d"] == 0.20]
            A_fit_mean = np.mean([r["A_fit"] for r in sub_fit])
            B_fit_mean = np.mean([r["B_fit"] for r in sub_fit])
            rho_mean = np.mean([r["rho"] for r in sub_fit])
            print(f"    ρ={rho_mean:+.4f}, ν̃={sub_fit[0]['nu_tilde']:.4f}")
            print(f"    A: theory={A_th_mean:.4f}, fit(d≤0.2)={A_fit_mean:.4f}")
            print(f"    B: theory={B_th_mean:+.4f}, fit(d≤0.2)={B_fit_mean:+.4f}")

    out = pd.DataFrame(rows)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_OUT, index=False)
    print(f"\n→ Saved: {TABLE_OUT}")

    # ── Summary: correlations across datasets at each fit range ──
    print(f"\n{'='*60}")
    print("  Theory vs Fit Correlations (averaged over LOO splits):")

    summary_rows = []
    for max_d in [0.05, 0.10, 0.20, 0.50, 1.00, 2.00]:
        sub = out[out["fit_max_d"] == max_d].dropna()
        # Average over rater splits per dataset
        agg = sub.groupby("label").agg({
            "A_theory": "mean", "B_theory": "mean",
            "A_fit": "mean", "B_fit": "mean", "r2": "mean",
        }).dropna()

        if len(agg) > 2:
            r_A = np.corrcoef(agg["A_theory"], agg["A_fit"])[0, 1]
            r_B = np.corrcoef(agg["B_theory"], agg["B_fit"])[0, 1]
            sign_B = np.sum(np.sign(agg["B_theory"]) == np.sign(agg["B_fit"]))
            mean_r2 = agg["r2"].mean()
            print(f"  d≤{max_d:.2f}: r(A)={r_A:.3f}, r(B)={r_B:.3f}, "
                  f"B sign={sign_B}/{len(agg)}, R²={mean_r2:.4f}")
            summary_rows.append({
                "max_d": max_d, "r_A": r_A, "r_B": r_B,
                "B_sign_match": f"{sign_B}/{len(agg)}",
                "mean_r2": mean_r2, "n_datasets": len(agg),
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
        print(f"→ Saved: {SUMMARY_OUT}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Direct Tilt Experiment: Theorem 2 Validation on SummEval (LOO Gold)",
                 fontsize=13, fontweight="bold")

    colors = {
        "exp-coher": "#1f77b4", "exp-consi": "#ff7f0e",
        "exp-fluen": "#2ca02c", "exp-relev": "#d62728",
        "cro-coher": "#9467bd", "cro-consi": "#8c564b",
        "cro-fluen": "#e377c2", "cro-relev": "#7f7f7f",
    }

    # Panel A: Q(d) - Q(0) trajectories
    ax = axes[0, 0]
    for label, curve in all_curves.items():
        d, q = curve["d"], curve["q"]
        gain = q - q[0]
        color = colors.get(label, "gray")
        ax.plot(d, gain, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("d = KL(π_t || π_ref)")
    ax.set_ylabel("Q(d) - Q(0)")
    ax.set_title("A. Quality Trajectories (Tilt Path)", fontweight="bold")
    ax.set_xlim(0, 3)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Zoomed local region
    ax = axes[0, 1]
    for label, curve in all_curves.items():
        d, q = curve["d"], curve["q"]
        gain = q - q[0]
        color = colors.get(label, "gray")
        ax.plot(d, gain, color=color, linewidth=1.5, label=label)

        # Overlay theory prediction
        agg = out[(out["label"] == label) & (out["fit_max_d"] == 0.20)]
        if len(agg) > 0:
            A_th = agg["A_theory"].mean()
            B_th = agg["B_theory"].mean()
            d_smooth = np.linspace(0, 0.5, 100)
            pred = A_th * np.sqrt(d_smooth) + B_th * d_smooth
            ax.plot(d_smooth, pred, color=color, linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("d (zoomed)")
    ax.set_ylabel("Q(d) - Q(0)")
    ax.set_title("B. Local Region + Theory (dashed)", fontweight="bold")
    ax.set_xlim(0, 0.5)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel C: A_theory vs A_fit (d ≤ 0.20)
    ax = axes[0, 2]
    agg = out[out["fit_max_d"] == 0.20].groupby("label").agg({
        "A_theory": "mean", "B_theory": "mean",
        "A_fit": "mean", "B_fit": "mean",
    }).dropna()
    c_list = [colors.get(l, "gray") for l in agg.index]
    ax.scatter(agg["A_theory"], agg["A_fit"], c=c_list, s=80, edgecolors="black", linewidth=0.5)
    lims = [0, max(agg["A_theory"].max(), agg["A_fit"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.5)
    r_A = np.corrcoef(agg["A_theory"], agg["A_fit"])[0, 1] if len(agg) > 2 else 0
    ax.set_xlabel("A (Theorem 2)")
    ax.set_ylabel("A (Tilt fit, d≤0.2)")
    ax.set_title(f"C. A Coefficient (r={r_A:.3f})", fontweight="bold")
    for label in agg.index:
        ax.annotate(label, (agg.loc[label, "A_theory"], agg.loc[label, "A_fit"]),
                    fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.grid(True, alpha=0.3)

    # Panel D: B_theory vs B_fit (d ≤ 0.20)
    ax = axes[1, 0]
    ax.scatter(agg["B_theory"], agg["B_fit"], c=c_list, s=80, edgecolors="black", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    r_B = np.corrcoef(agg["B_theory"], agg["B_fit"])[0, 1] if len(agg) > 2 else 0
    ax.set_xlabel("B (Theorem 2)")
    ax.set_ylabel("B (Tilt fit, d≤0.2)")
    ax.set_title(f"D. B Coefficient (r={r_B:.3f})", fontweight="bold")
    for label in agg.index:
        ax.annotate(label, (agg.loc[label, "B_theory"], agg.loc[label, "B_fit"]),
                    fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.grid(True, alpha=0.3)

    # Panel E: r(B) across fit ranges
    ax = axes[1, 1]
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        ax.plot(sdf["max_d"], sdf["r_B"], "o-", color="steelblue", linewidth=2, label="r(B)")
        ax.plot(sdf["max_d"], sdf["r_A"], "s--", color="darkred", linewidth=2, label="r(A)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Fit range (max d)")
        ax.set_ylabel("Correlation")
        ax.set_title("E. Theory-Fit Correlation vs Range", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel F: Example trajectory + fit + theory for one dataset
    ax = axes[1, 2]
    example = "exp-consi"
    if example in all_curves:
        d, q = all_curves[example]["d"], all_curves[example]["q"]
        gain = q - q[0]
        ax.plot(d, gain, "k-", linewidth=2, label="Exact tilt")

        agg_ex = out[(out["label"] == example) & (out["fit_max_d"] == 0.20)]
        if len(agg_ex) > 0:
            A_th = agg_ex["A_theory"].mean()
            B_th = agg_ex["B_theory"].mean()
            A_ft = agg_ex["A_fit"].mean()
            B_ft = agg_ex["B_fit"].mean()
            d_sm = np.linspace(0, 1.0, 200)
            ax.plot(d_sm, A_th * np.sqrt(d_sm) + B_th * d_sm,
                    "r--", linewidth=1.5, label=f"Theory: A={A_th:.3f}, B={B_th:+.3f}")
            ax.plot(d_sm, A_ft * np.sqrt(d_sm) + B_ft * d_sm,
                    "b:", linewidth=1.5, label=f"Fit: A={A_ft:.3f}, B={B_ft:+.3f}")

        ax.set_xlabel("d")
        ax.set_ylabel("Q(d) - Q(0)")
        ax.set_title(f"F. {example}: Exact vs Theory vs Fit", fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"→ Saved: {FIG_OUT}")
    plt.close()


if __name__ == "__main__":
    main()
