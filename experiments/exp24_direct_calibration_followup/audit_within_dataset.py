#!/usr/bin/env python3
"""Summarize and audit exp24 within-dataset follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARE = [
    "A-only",
    "empirical_curve",
    "empirical_fit",
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
    "zero",
]
PLOT_BASELINES = [
    "A+B",
    "A-only",
    "empirical_fit",
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
]
DATASET_LABELS = {
    "SummEval-human_heldout": "SummEval human proxy",
    "SummEval-GPT4o": "GPT-4o proxy",
    "WMT-MQM-human_heldout": "WMT MQM human proxy",
    "OpenMEVA-MAGS-human_heldout": "OpenMEVA-MAGS human proxy",
}
BASELINE_LABELS = {
    "A+B": "A+B",
    "A-only": "A-only",
    "empirical_fit": "Empirical fit",
    "isotonic_calibration": "Isotonic",
    "binned_shrinkage": "Binned",
    "spline_ridge": "Spline",
}
N_BOOT = 2000
SEED = 0


def clustered_diff(
    frame: pd.DataFrame,
    metric: str,
    baseline: str,
    n_boot: int = N_BOOT,
) -> dict:
    a = frame[frame["baseline"] == "A+B"][
        ["path_id", "repeat", metric]
    ].rename(columns={metric: "err_a"})
    b = frame[frame["baseline"] == baseline][
        ["path_id", "repeat", metric]
    ].rename(columns={metric: "err_b"})
    paired = a.merge(
        b, on=["path_id", "repeat"], how="inner", validate="one_to_one"
    ).dropna(subset=["err_a", "err_b"])
    by_path = paired.groupby("path_id")[["err_a", "err_b"]].mean()
    delta = (by_path["err_a"] - by_path["err_b"]).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.mean(
            delta[rng.integers(0, len(delta), size=len(delta))]
        )
    return {
        "n_paths": int(len(delta)),
        "n_paired_splits": int(len(paired)),
        "mae_A+B": float(by_path["err_a"].mean()),
        "mae_baseline": float(by_path["err_b"].mean()),
        "diff_AplusB_minus_baseline": float(delta.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    splits = pd.read_csv(args.outdir / "within_split_metrics.csv")
    spec = json.loads(
        (args.outdir / "within_run_spec.json").read_text(encoding="utf-8")
    )

    assert not splits.duplicated(
        ["dataset", "path_id", "repeat", "baseline", "calibration_frac"]
    ).any()
    deployable = splits[splits["baseline"] != "empirical_curve"]
    assert np.isfinite(
        deployable.loc[
            deployable["n_local_radii"] == 5, "local_curve_mae"
        ].to_numpy(float)
    ).all()
    assert (
        deployable["local_curve_mae"].isna()
        == (deployable["n_local_radii"] != 5)
    ).all()
    assert (
        deployable["extended_curve_mae"].isna()
        == (deployable["n_extended_radii"] != 7)
    ).all()

    learning = (
        splits.groupby(["dataset", "calibration_frac", "baseline"])
        .agg(
            local_curve_mae=("local_curve_mae", "mean"),
            extended_curve_mae=("extended_curve_mae", "mean"),
            n_paths=("path_id", "nunique"),
            n_splits=("repeat", "size"),
            n_complete_local=("local_curve_mae", "count"),
            n_complete_extended=("extended_curve_mae", "count"),
        )
        .reset_index()
    )
    learning.to_csv(args.outdir / "calibration_size_summary.csv", index=False)
    compact_learning = (
        learning[
            learning["calibration_frac"].isin([0.05, 0.10, 0.30])
            & learning["baseline"].isin(
                [
                    "A+B",
                    "empirical_fit",
                    "isotonic_calibration",
                    "spline_ridge",
                ]
            )
        ]
        .pivot(
            index=["dataset", "calibration_frac"],
            columns="baseline",
            values="local_curve_mae",
        )
        .reset_index()
    )
    compact_learning.to_csv(
        args.outdir / "calibration_comparison_table.csv", index=False
    )
    (args.outdir / "calibration_comparison_table.tex").write_text(
        compact_learning.to_latex(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        encoding="utf-8",
    )

    radius_rows = []
    for d in spec["d_targets"]:
        suffix = str(d).replace(".", "p")
        col = f"abs_error_d{suffix}"
        sub = (
            splits.groupby(["dataset", "calibration_frac", "baseline"])[col]
            .agg(mae="mean", n_reachable="count", n_attempted="size")
            .reset_index()
        )
        sub["reachable_rate"] = sub["n_reachable"] / sub["n_attempted"]
        sub["d"] = d
        radius_rows.append(sub)
    radius = pd.concat(radius_rows, ignore_index=True)
    radius.to_csv(args.outdir / "radius_summary.csv", index=False)
    compact_radius = (
        radius[
            np.isclose(radius["calibration_frac"], 0.30)
            & radius["d"].isin([0.10, 0.20])
            & radius["baseline"].isin(
                [
                    "A+B",
                    "empirical_fit",
                    "isotonic_calibration",
                    "spline_ridge",
                ]
            )
        ]
        .pivot(
            index=["dataset", "d"],
            columns="baseline",
            values="mae",
        )
        .reset_index()
    )
    compact_radius.to_csv(
        args.outdir / "radius_comparison_table.csv", index=False
    )
    (args.outdir / "radius_comparison_table.tex").write_text(
        compact_radius.to_latex(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        encoding="utf-8",
    )

    diff_rows = []
    for (dataset, frac), frame in splits.groupby(
        ["dataset", "calibration_frac"], sort=False
    ):
        for baseline in COMPARE:
            result = clustered_diff(
                frame, "local_curve_mae", baseline
            )
            result.update(
                {
                    "dataset": dataset,
                    "calibration_frac": frac,
                    "baseline": baseline,
                    "metric": "local_curve_mae",
                }
            )
            diff_rows.append(result)
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(
        args.outdir / "paired_local_curvemae_bootstrap.csv", index=False
    )

    plot_learning(learning, args.outdir / "fig_calibration_efficiency.png")
    plot_radius(
        radius[np.isclose(radius["calibration_frac"], 0.30)],
        args.outdir / "fig_radius_scope.png",
    )
    print(
        learning[
            learning["baseline"].isin(
                ["A+B", "empirical_fit", "isotonic_calibration", "spline_ridge"]
            )
        ].to_string(index=False)
    )
    print("AUDIT PASS")


def plot_learning(frame: pd.DataFrame, out: Path) -> None:
    datasets = frame["dataset"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, dataset in zip(axes.flat, datasets):
        sub = frame[frame["dataset"] == dataset]
        for baseline in PLOT_BASELINES:
            line = sub[sub["baseline"] == baseline].sort_values(
                "calibration_frac"
            )
            ax.plot(
                100 * line["calibration_frac"],
                line["local_curve_mae"],
                marker="o",
                label=BASELINE_LABELS[baseline],
            )
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Calibration items (%)")
        ax.set_ylabel("CurveMAE, d <= 0.10")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=180)


def plot_radius(frame: pd.DataFrame, out: Path) -> None:
    datasets = frame["dataset"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, dataset in zip(axes.flat, datasets):
        sub = frame[frame["dataset"] == dataset]
        for baseline in PLOT_BASELINES:
            line = sub[sub["baseline"] == baseline].sort_values("d")
            ax.plot(
                line["d"],
                line["mae"],
                marker="o",
                label=BASELINE_LABELS[baseline],
            )
        ax.axvline(0.10, color="black", linestyle=":", linewidth=1)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("KL radius d")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
