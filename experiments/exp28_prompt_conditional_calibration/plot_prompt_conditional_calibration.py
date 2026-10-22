#!/usr/bin/env python3
"""Plot the prompt-geometry and DPO mechanism diagnosis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = [
    "A+B",
    "A-only",
    "A_nu",
    "Spearman",
    "cal_raw_isotonic",
    "cal_centered_isotonic",
    "cal_zscore_isotonic",
    "cal_rank_isotonic",
    "cal_rank_spline",
]
LABELS = {
    "A+B": "A+B",
    "A-only": "A-only",
    "A_nu": r"$A_\nu$",
    "Spearman": "Spearman",
    "cal_raw_isotonic": "Iso: raw",
    "cal_centered_isotonic": "Iso: centered",
    "cal_zscore_isotonic": "Iso: z-score",
    "cal_rank_isotonic": "Iso: rank",
    "cal_rank_spline": "Spline: rank",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    windows = pd.read_csv(args.outdir / "window_summary_mean_sd.csv")
    geometry = pd.read_csv(args.outdir / "prompt_geometry.csv")
    mechanism = pd.read_csv(args.outdir / "mechanism_by_checkpoint.csv")
    strict = windows[
        windows["scope"] == "strict_local_d_le_0p10"
    ].set_index("model")
    historical = windows[
        windows["scope"] == "historical_six_checkpoints"
    ].set_index("model")
    selected = [name for name in ORDER if name in strict.index]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = [
        "#2878B5",
        "#F28E2B",
        "#9B59B6",
        "#B0B0B0",
        "#2A9D8F",
        "#59A14F",
        "#76B7B2",
        "#E76F51",
        "#8C564B",
    ]

    ax = axes[0, 0]
    x = np.arange(len(geometry))
    width = 0.34
    ax.bar(
        x - width / 2,
        geometry["proxy_between_variance_fraction"],
        width,
        label="Proxy",
        color="#2878B5",
    )
    ax.bar(
        x + width / 2,
        geometry["gold_between_variance_fraction"],
        width,
        label="Gold",
        color="#F28E2B",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Calibration", "Evaluation"])
    ax.set_ylabel("Between-prompt variance fraction")
    ax.set_ylim(0, 1)
    ax.set_title("A  Prompt offsets dominate raw scores")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for ax, table, mean_col, sd_col, title in [
        (
            axes[0, 1],
            strict,
            "mean_mae_frontier",
            "sd_mae_frontier",
            "B  Exact frontier, strict d <= 0.10",
        ),
        (
            axes[1, 0],
            historical,
            "mean_mae_dpo",
            "sd_mae_dpo",
            "C  Realized DPO, six checkpoints",
        ),
    ]:
        values = table.loc[selected, mean_col].to_numpy()
        errors = table.loc[selected, sd_col].to_numpy()
        x = np.arange(len(selected))
        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            color=colors,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[name] for name in selected],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("MAE (mean +/- SD over 3 seeds)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    for seed, seed_df in mechanism.groupby("seed"):
        seed_df = seed_df.sort_values("d_med")
        ax.plot(
            seed_df["d_med"],
            seed_df["realization_ratio"],
            marker="o",
            alpha=0.7,
            label=f"seed {seed}",
        )
    ax.axvline(0.10, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Measured prompt-level KL radius")
    ax.set_ylabel("Realized DPO gain / exact frontier gain")
    ax.set_ylim(0, 1)
    ax.set_title("D  DPO realizes only part of the tilt frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.outdir / "fig_prompt_conditional_calibration.pdf")
    fig.savefig(args.outdir / "fig_prompt_conditional_calibration.png", dpi=180)


if __name__ == "__main__":
    main()
