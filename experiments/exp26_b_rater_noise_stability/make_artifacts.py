#!/usr/bin/env python3
"""Create compact tables and a deterministic figure for exp26."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "b_abs_error_vs_eval",
    "b_sign_matches_eval",
    "b_sign_matches_same_k_clean",
    "b_contribution_share_d0p1",
    "curve_mae_ab",
    "curve_mae_aonly",
    "ab_minus_aonly_curve_mae",
    "decision_disagree_d0p1",
    "d0p1_reachable",
)


def cluster_ci(values: pd.DataFrame, metric: str, seed: int) -> tuple[float, float]:
    per_path = values.groupby("path_id", sort=True)[metric].mean()
    array = per_path.to_numpy(dtype=float)
    if len(array) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(5000, len(array)))
    draws = array[indices].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "rater_type",
        "calibration_frac",
        "gold_rater_count",
        "max_gold_raters",
        "gold_rater_fraction",
        "noise_family",
        "noise_scale_sd",
    ]
    output: list[dict[str, object]] = []
    for key, group in rows.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, key))
        record["n_paths"] = int(group["path_id"].nunique())
        record["n_path_repeats"] = int(
            group[["path_id", "repeat"]].drop_duplicates().shape[0]
        )
        per_path = group.groupby("path_id", sort=True)[list(METRICS)].mean()
        for metric in METRICS:
            record[metric] = float(per_path[metric].mean())
        seed = stable_seed_from_values(
            record["rater_type"],
            record["calibration_frac"],
            record["gold_rater_count"],
            record["noise_family"],
            record["noise_scale_sd"],
        )
        lo, hi = cluster_ci(group, "ab_minus_aonly_curve_mae", seed)
        record["ab_minus_aonly_ci_lo"] = lo
        record["ab_minus_aonly_ci_hi"] = hi
        output.append(record)
    return pd.DataFrame(output).sort_values(keys).reset_index(drop=True)


def stable_seed_from_values(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_figure(summary: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    colors = {"expert": "#2369a8", "turker": "#d34a35"}
    markers = {"expert": "o", "turker": "s"}

    clean = summary[
        summary["noise_family"].eq("clean")
        & summary["calibration_frac"].eq(0.30)
    ]
    for rater_type, group in clean.groupby("rater_type", sort=True):
        axes[0].plot(
            group["gold_rater_count"],
            group["b_abs_error_vs_eval"],
            marker=markers[rater_type],
            color=colors[rater_type],
            label=rater_type.capitalize(),
        )
    axes[0].set_title("Gold-rater budget")
    axes[0].set_xlabel("Gold raters per item")
    axes[0].set_ylabel(r"Mean $|\hat B-B_{\rm eval}|$")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    noise = summary[
        summary["calibration_frac"].eq(0.30)
        & summary["gold_rater_count"].eq(summary["max_gold_raters"])
        & ~summary["noise_family"].eq("clean")
    ]
    for (rater_type, family), group in noise.groupby(
        ["rater_type", "noise_family"], sort=True
    ):
        linestyle = "-" if family == "gaussian" else "--"
        axes[1].plot(
            group["noise_scale_sd"],
            100 * group["b_sign_matches_same_k_clean"],
            marker=markers[rater_type],
            linestyle=linestyle,
            color=colors[rater_type],
            label=f"{rater_type.capitalize()}, {family.replace('_', '-')}",
        )
    axes[1].set_title("Noise stability")
    axes[1].set_xlabel("Noise scale (rating SD)")
    axes[1].set_ylabel(r"$B$-sign agreement with clean (%)")
    axes[1].set_ylim(0, 102)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    for (rater_type, family), group in noise.groupby(
        ["rater_type", "noise_family"], sort=True
    ):
        linestyle = "-" if family == "gaussian" else "--"
        axes[2].plot(
            group["noise_scale_sd"],
            group["ab_minus_aonly_curve_mae"],
            marker=markers[rater_type],
            linestyle=linestyle,
            color=colors[rater_type],
            label=f"{rater_type.capitalize()}, {family.replace('_', '-')}",
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Second-order predictive effect")
    axes[2].set_xlabel("Noise scale (rating SD)")
    axes[2].set_ylabel("CurveMAE(A+B) - CurveMAE(A-only)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        outdir / "fig_b_rater_noise_stability.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "fig_b_rater_noise_stability.pdf",
        bbox_inches="tight",
        metadata={
            "Title": "B rater-count and noise stability",
            "Author": "Local Proxy Diagnostics Authors",
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rows_path = args.outdir / "b_rater_noise_rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    rows = pd.read_csv(rows_path)
    summary = summarize(rows)
    summary.to_csv(args.outdir / "condition_summary.csv", index=False)

    selected = summary[
        summary["calibration_frac"].eq(0.30)
        & (
            summary["noise_family"].eq("clean")
            | summary["noise_scale_sd"].eq(0.50)
        )
    ].copy()
    selected.to_csv(args.outdir / "selected_conditions.csv", index=False)
    make_figure(summary, args.outdir)

    artifact_spec = {
        "summary_unit": (
            "mean over repeats within path, then equal-weight mean over paths"
        ),
        "paired_difference_ci": (
            "5000-draw path-cluster bootstrap over path-level means"
        ),
        "figure_filters": {
            "panel_a": "clean, 30% calibration, all available gold-rater counts",
            "panels_b_c": (
                "30% calibration, maximum gold raters, nonzero noise"
            ),
        },
    }
    (args.outdir / "artifact_spec.json").write_text(
        json.dumps(artifact_spec, indent=2), encoding="utf-8"
    )
    print(f"wrote summaries and figure to {args.outdir}")


if __name__ == "__main__":
    main()
