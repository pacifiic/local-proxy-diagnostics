"""
Plotting utilities for all experiments.

All figures are saved as 300-dpi PNGs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(os.environ.get("RESULTS_ROOT", _ROOT / "results")) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color palette
COLORS = {
    "all": "#2196F3",
    "crowd": "#FF9800",
    "turker": "#FF9800",
    "expert": "#4CAF50",
    "coherence": "#E91E63",
    "consistency": "#9C27B0",
    "fluency": "#00BCD4",
    "relevance": "#FF5722",
    "roc": "#3F51B5",
    "wp": "#FF9800",
}

DIM_ORDER = ["coherence", "consistency", "fluency", "relevance"]


def _save(fig, name: str, dpi: int = 300):
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 1: ICC(A,1) by dimension (grouped bar)
# ---------------------------------------------------------------------------

def fig1_icc_by_dimension(df: pd.DataFrame, filename="fig1_summeval_icc_by_dimension.png"):
    """df columns: dimension, pool, ICC_A1, CI_lower, CI_upper"""
    pools = ["all", "crowd", "expert"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DIM_ORDER))
    width = 0.25

    for i, pool in enumerate(pools):
        sub = df[df["pool"] == pool].set_index("dimension").loc[DIM_ORDER]
        bars = ax.bar(
            x + i * width, sub["ICC_A1"], width,
            yerr=[sub["ICC_A1"] - sub["CI_lower"], sub["CI_upper"] - sub["ICC_A1"]],
            label=pool, color=COLORS.get(pool, "#888"), capsize=3, alpha=0.85,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(DIM_ORDER, fontsize=11)
    ax.set_ylabel("ICC(A,1)", fontsize=12)
    ax.set_title("SummEval: ICC(A,1) by Dimension and Rater Pool", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 2: Personalization gap by dimension
# ---------------------------------------------------------------------------

def fig2_personalization_gap(df: pd.DataFrame, filename="fig2_summeval_personalization_gap.png"):
    """df columns: dimension, pool, G_pers"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pools = df["pool"].unique()
    x = np.arange(len(DIM_ORDER))
    width = 0.8 / len(pools)

    for i, pool in enumerate(pools):
        sub = df[df["pool"] == pool].set_index("dimension").reindex(DIM_ORDER)
        ax.bar(x + i * width, sub["G_pers"], width,
               label=pool, color=COLORS.get(pool, "#888"), alpha=0.85)

    ax.set_xticks(x + width * (len(pools) - 1) / 2)
    ax.set_xticklabels(DIM_ORDER, fontsize=11)
    ax.set_ylabel("G_pers = ICC(C,1) - ICC(A,1)", fontsize=11)
    ax.set_title("Personalization Gap by Dimension", fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig A: BoN curves by dimension
# ---------------------------------------------------------------------------

def figA_bon_curves(
    bon_curves: Dict[str, Dict[int, float]],
    filename="figA_bon_curves_by_dimension.png",
):
    """bon_curves: {dimension: {K: mean_oracle_score}}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for dim in DIM_ORDER:
        if dim not in bon_curves:
            continue
        curve = bon_curves[dim]
        ks = sorted(curve.keys())
        vals = [curve[k] for k in ks]
        ax.plot(ks, vals, marker="o", label=dim, color=COLORS.get(dim, "#888"), lw=2)

    ax.set_xlabel("K (number of candidate systems)", fontsize=12)
    ax.set_ylabel("E[oracle score | top-1 by proxy]", fontsize=12)
    ax.set_title("BoN Curves by Dimension (SummEval)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig B: ΔJ_max vs ICC(A,1) scatter — THE key figure
# ---------------------------------------------------------------------------

def figB_delta_vs_icc(
    df: pd.DataFrame,
    filename="figB_delta_J_max_vs_icc_scatter.png",
    y_col: str = "delta_J_max",
):
    """df columns: dimension, ICC_A1, <y_col>,
                   ICC_A1_ci_lo, ICC_A1_ci_hi, delta_ci_lo, delta_ci_hi

    y_col can be "delta_J_max" (raw) or "delta_J_max_norm" (normalized by oracle range).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    y_ci_lo_col = "delta_ci_lo_norm" if y_col == "delta_J_max_norm" else "delta_ci_lo"
    y_ci_hi_col = "delta_ci_hi_norm" if y_col == "delta_J_max_norm" else "delta_ci_hi"

    for _, row in df.iterrows():
        dim = row["dimension"]
        y_val = row[y_col]
        xerr_lo = max(0.0, row["ICC_A1"] - row.get("ICC_A1_ci_lo", row["ICC_A1"]))
        xerr_hi = max(0.0, row.get("ICC_A1_ci_hi", row["ICC_A1"]) - row["ICC_A1"])
        yerr_lo = max(0.0, y_val - row.get(y_ci_lo_col, y_val))
        yerr_hi = max(0.0, row.get(y_ci_hi_col, y_val) - y_val)
        ax.errorbar(
            row["ICC_A1"], y_val,
            xerr=[[xerr_lo], [xerr_hi]],
            yerr=[[yerr_lo], [yerr_hi]],
            fmt="o", ms=10, color=COLORS.get(dim, "#888"), capsize=4, lw=1.5,
        )
        ax.annotate(dim, (row["ICC_A1"], y_val),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    # Fit line for reference
    x = df["ICC_A1"].values
    y = df[y_col].values
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min() * 0.9, x.max() * 1.1, 50)
        ax.plot(xline, np.polyval(z, xline), ls="--", color="gray", alpha=0.5, lw=1)

    ax.set_xlabel("ICC(A,1)  ≈  ν²", fontsize=12)
    if y_col == "delta_J_max_norm":
        ax.set_ylabel("ΔJ_max / oracle range (normalized)", fontsize=12)
    else:
        ax.set_ylabel("ΔJ_max (BoN gain)", fontsize=12)
    ax.set_title("ΔJ_max vs ICC(A,1): Theorem 2 Prediction", fontsize=13)
    ax.grid(alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 3: Expert vs crowd lift bar chart
# ---------------------------------------------------------------------------

def fig3_expert_crowd_lift(df: pd.DataFrame, filename="fig3_expert_vs_crowd_lift.png"):
    """df columns: dimension, Lift_expert, Lift_CI_lower, Lift_CI_upper"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(DIM_ORDER))
    sub = df.set_index("dimension").reindex(DIM_ORDER)

    yerr_lo = sub["Lift_expert"] - sub.get("Lift_CI_lower", sub["Lift_expert"])
    yerr_hi = sub.get("Lift_CI_upper", sub["Lift_expert"]) - sub["Lift_expert"]

    ax.bar(x, sub["Lift_expert"], color=[COLORS.get(d, "#888") for d in DIM_ORDER],
           yerr=[yerr_lo, yerr_hi], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_ORDER, fontsize=11)
    ax.set_ylabel("Expert Lift = ICC(A,1)_expert - ICC(A,1)_crowd", fontsize=10)
    ax.set_title("Expert-vs-Crowd Lift by Dimension", fontsize=13)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 4: Expert vs crowd scatter
# ---------------------------------------------------------------------------

def fig4_expert_crowd_scatter(df: pd.DataFrame, filename="fig4_expert_vs_crowd_scatter.png"):
    """df columns: dimension, ICC_A1_crowd, ICC_A1_expert"""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for _, row in df.iterrows():
        dim = row["dimension"]
        ax.scatter(row["ICC_A1_crowd"], row["ICC_A1_expert"],
                   s=120, color=COLORS.get(dim, "#888"), zorder=5)
        ax.annotate(dim, (row["ICC_A1_crowd"], row["ICC_A1_expert"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=10)

    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05]
    ax.plot(lims, lims, ls="--", color="gray", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("ICC(A,1) — Crowd", fontsize=12)
    ax.set_ylabel("ICC(A,1) — Expert", fontsize=12)
    ax.set_title("Crowd vs Expert Reliability", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 5: Protocol split-half stability
# ---------------------------------------------------------------------------

def fig5_protocol_stability(df: pd.DataFrame, filename="fig5_protocol_split_half_stability.png"):
    """df columns: dimension, protocol, mean_corr, ci_lower, ci_upper"""
    protocols = sorted(df["protocol"].unique())
    dims = DIM_ORDER

    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4.5), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        sub = df[df["dimension"] == dim].set_index("protocol").reindex(protocols)
        x = np.arange(len(protocols))
        yerr_lo = sub["mean_corr"] - sub["ci_lower"]
        yerr_hi = sub["ci_upper"] - sub["mean_corr"]
        ax.bar(x, sub["mean_corr"], yerr=[yerr_lo, yerr_hi],
               capsize=3, color=COLORS.get(dim, "#888"), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, rotation=30, fontsize=9)
        ax.set_title(dim, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Split-half correlation", fontsize=11)
    fig.suptitle("Protocol Perturbation: Split-half Stability", fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 6: Dataset replication comparison
# ---------------------------------------------------------------------------

def fig6_dataset_replication(df: pd.DataFrame, filename="fig6_dataset_replication.png"):
    """df columns: dataset, dimension_label, ICC_A1"""
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = df["dataset"].unique()
    labels = df["dimension_label"].unique()
    x = np.arange(len(labels))
    width = 0.35

    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].set_index("dimension_label").reindex(labels)
        ax.bar(x + i * width, sub["ICC_A1"], width,
               label=ds, color=COLORS.get(ds.lower(), "#888"), alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, fontsize=10, rotation=15)
    ax.set_ylabel("ICC(A,1)", fontsize=12)
    ax.set_title("Reliability Profile: SummEval vs OpenMEVA", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 7: ROC vs WP
# ---------------------------------------------------------------------------

def fig7_roc_vs_wp(df: pd.DataFrame, filename="fig7_roc_vs_wp_comparison.png"):
    """df columns: source, ICC_A1"""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sources = ["roc", "wp"]
    vals = [df[df["source"] == s]["ICC_A1"].values[0] for s in sources]
    ax.bar(sources, vals, color=[COLORS.get(s, "#888") for s in sources], alpha=0.85)
    ax.set_ylabel("ICC(A,1)", fontsize=12)
    ax.set_title("ROC vs WP: Overall Reliability", fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 8: Intervention scatter
# ---------------------------------------------------------------------------

def fig8_intervention_scatter(df: pd.DataFrame, filename="fig8_intervention_scatter.png"):
    """df columns: dimension, G_pers, Lift_expert, ICC_A1, diagnosis"""
    fig, ax = plt.subplots(figsize=(7, 6))

    diag_colors = {
        "personalization-limited": "#E91E63",
        "measurement-limited": "#2196F3",
        "deeply-sparse": "#FF9800",
        "mixed": "#9E9E9E",
    }

    for _, row in df.iterrows():
        c = diag_colors.get(row["diagnosis"], "#888")
        size = max(row["ICC_A1"] * 300, 50)
        ax.scatter(row["G_pers"], row["Lift_expert"], s=size, c=c, alpha=0.8, zorder=5,
                   edgecolors="white", linewidth=1)
        ax.annotate(row["dimension"], (row["G_pers"], row["Lift_expert"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    # Legend for diagnosis
    for diag, color in diag_colors.items():
        ax.scatter([], [], c=color, s=80, label=diag)
    ax.legend(fontsize=9, title="Diagnosis", loc="best")

    ax.set_xlabel("G_pers", fontsize=12)
    ax.set_ylabel("Expert Lift", fontsize=12)
    ax.set_title("Intervention Map", fontsize=13)
    ax.grid(alpha=0.3)
    return _save(fig, filename)
