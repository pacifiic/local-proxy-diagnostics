#!/usr/bin/env python3
"""Summarize and visualize exp29 same-prompt holdout results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "A+B": "A+B",
    "A+B-shrunk": "A+B (shrunk B)",
    "A-only": "A-only",
    "A_nu": r"$A_\nu$",
    "empirical_curve": "Empirical curve",
    "empirical_fit": "Empirical fit",
    "isotonic_calibration": "Isotonic",
    "binned_shrinkage": "Binned",
    "spline_ridge": "Spline",
    "mlp_calibration": "MLP",
    "zero": "Zero",
}

ORDER = [
    "A+B",
    "A+B-shrunk",
    "A-only",
    "A_nu",
    "empirical_curve",
    "empirical_fit",
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
    "mlp_calibration",
    "zero",
]

COLORS = {
    "A+B": "#1f77b4",
    "A+B-shrunk": "#17becf",
    "A-only": "#ff7f0e",
    "A_nu": "#d62728",
    "empirical_curve": "#9467bd",
    "empirical_fit": "#2ca02c",
    "isotonic_calibration": "#8c564b",
    "binned_shrinkage": "#e377c2",
    "spline_ridge": "#7f7f7f",
    "mlp_calibration": "#bcbd22",
    "zero": "#111111",
}


def model_family(run: str) -> str:
    if "ministral" in run.lower():
        return "Ministral-3 8B"
    if "llama" in run.lower():
        return "Llama-3.1 8B"
    return run


def seed_from_run(run: str) -> int:
    marker = "_seed"
    if marker not in run:
        return -1
    return int(run.split(marker, 1)[1].split("_", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.input_dir / "aggregate_predictions.csv")
    metadata = pd.read_csv(args.input_dir / "fit_metadata.csv")
    predictions["model"] = predictions["run"].map(model_family)
    predictions["generation_seed"] = predictions["run"].map(seed_from_run)
    metadata["model"] = metadata["run"].map(model_family)
    metadata["generation_seed"] = metadata["run"].map(seed_from_run)

    local = predictions[predictions["d"].le(0.10 + 1e-12)].copy()
    per_split = (
        local.groupby(
            ["model", "generation_seed", "run", "repeat", "baseline"],
            sort=True,
        )
        .agg(
            curve_mae=("abs_error", "mean"),
            prompt_mae=("prompt_mae", "mean"),
            n_valid_prompts=("n_valid_prompts", "min"),
        )
        .reset_index()
    )
    endpoint = (
        local[np.isclose(local["d"], 0.10)]
        .set_index(["model", "generation_seed", "run", "repeat", "baseline"])[
            "abs_error"
        ]
        .rename("endpoint_mae")
        .reset_index()
    )
    per_split = per_split.merge(
        endpoint,
        on=["model", "generation_seed", "run", "repeat", "baseline"],
        how="left",
        validate="one_to_one",
    )
    per_split.to_csv(args.outdir / "per_split_metrics.csv", index=False)

    summary = (
        per_split.groupby(["model", "baseline"], sort=True)
        .agg(
            curve_mae_mean=("curve_mae", "mean"),
            curve_mae_std=("curve_mae", "std"),
            endpoint_mae_mean=("endpoint_mae", "mean"),
            endpoint_mae_std=("endpoint_mae", "std"),
            prompt_mae_mean=("prompt_mae", "mean"),
            n_run_repeats=("curve_mae", "size"),
            n_valid_prompts_min=("n_valid_prompts", "min"),
        )
        .reset_index()
    )
    summary["baseline_order"] = summary["baseline"].map(
        {name: index for index, name in enumerate(ORDER)}
    )
    summary = summary.sort_values(["model", "baseline_order"]).drop(
        columns="baseline_order"
    )
    summary.to_csv(args.outdir / "table_same_prompt_summary.csv", index=False)

    pivot = per_split.pivot_table(
        index=["model", "generation_seed", "run", "repeat"],
        columns="baseline",
        values="curve_mae",
    ).reset_index()
    pair_rows: list[dict[str, object]] = []
    for model, group in pivot.groupby("model", sort=True):
        for baseline in ORDER:
            if baseline == "A+B" or baseline not in group:
                continue
            difference = group["A+B"] - group[baseline]
            pair_rows.append(
                {
                    "model": model,
                    "comparison": f"A+B minus {baseline}",
                    "baseline": baseline,
                    "mean_difference": float(difference.mean()),
                    "std_difference": float(difference.std(ddof=1)),
                    "a_plus_b_win_fraction": float((difference < 0).mean()),
                    "n_run_repeats": int(len(difference)),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(args.outdir / "table_paired_differences.csv", index=False)

    skip_summary = (
        metadata.groupby("model", sort=True)
        .agg(
            attempted_prompts=("n_attempted_prompts", "sum"),
            valid_prompts=("n_valid_prompts", "sum"),
            skipped_prompts=("n_skipped_prompts", "sum"),
            mean_b_shrink_weight=("mean_b_shrink_weight", "mean"),
            min_b_shrink_weight=("mean_b_shrink_weight", "min"),
            max_b_shrink_weight=("mean_b_shrink_weight", "max"),
            leakage_prediction_difference_max=(
                "leakage_audit_prediction_difference",
                lambda x: float(np.max(np.abs(x))),
            ),
            leakage_actual_difference_min_abs=(
                "leakage_audit_actual_difference",
                lambda x: float(np.min(np.abs(x))),
            ),
        )
        .reset_index()
    )
    skip_summary.to_csv(args.outdir / "table_audit_summary.csv", index=False)

    mpl.rcParams.update(
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
    selected = [
        "A+B",
        "A+B-shrunk",
        "A-only",
        "A_nu",
        "empirical_curve",
        "isotonic_calibration",
        "binned_shrinkage",
        "spline_ridge",
        "mlp_calibration",
    ]
    models = summary["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(12.2, 4.1),
        sharey=False,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        subset = summary[
            summary["model"].eq(model) & summary["baseline"].isin(selected)
        ].set_index("baseline")
        names = [name for name in selected if name in subset.index]
        x = np.arange(len(names))
        heights = [subset.loc[name, "curve_mae_mean"] for name in names]
        errors = [
            subset.loc[name, "curve_mae_std"]
            / math_sqrt_safe(subset.loc[name, "n_run_repeats"])
            for name in names
        ]
        ax.bar(
            x,
            heights,
            yerr=errors,
            capsize=2,
            color=[COLORS[name] for name in names],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[name] for name in names], rotation=45, ha="right")
        ax.set_ylabel(r"Curve MAE on $d\leq0.10$")
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Same-prompt candidate holdout (10 calibration / 22 evaluation candidates)"
    )
    fig.savefig(args.outdir / "fig_same_prompt_candidate_holdout.pdf")
    fig.savefig(args.outdir / "fig_same_prompt_candidate_holdout.png", dpi=220)
    plt.close(fig)

    run_spec = json.loads((args.input_dir / "run_spec.json").read_text())
    analysis_spec = {
        "experiment": "exp29_same_prompt_candidate_holdout_analysis",
        "input_dir": str(args.input_dir),
        "local_window": "d <= 0.10",
        "curve_mae_definition": (
            "mean absolute error of the aggregate predicted gain over "
            "d in {0.01, 0.025, 0.05, 0.075, 0.10}"
        ),
        "error_bars": "standard error over generation-seed x candidate-split repeats",
        "input_leakage_audit_pass": run_spec["leakage_audit_pass"],
        "outputs": [
            "per_split_metrics.csv",
            "table_same_prompt_summary.csv",
            "table_paired_differences.csv",
            "table_audit_summary.csv",
            "fig_same_prompt_candidate_holdout.pdf",
            "fig_same_prompt_candidate_holdout.png",
        ],
    }
    (args.outdir / "analysis_run_spec.json").write_text(
        json.dumps(analysis_spec, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def math_sqrt_safe(value: float) -> float:
    value = float(value)
    return math.sqrt(value) if value > 0 else 1.0


if __name__ == "__main__":
    main()
