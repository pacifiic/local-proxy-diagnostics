"""Aggregate direct-calibration predictions and paired uncertainty.

Report endpoint MAE and CurveMAE, with paired differences against A+B.
Bootstrap paths, not individual repeated split rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

COMPARE_BASELINES = ["binned_shrinkage", "spline_ridge", "isotonic_calibration", "empirical_fit", "A-only", "zero"]
N_BOOT = 2000
SEED = 0


def curve_mae_table(df: pd.DataFrame) -> pd.DataFrame:
    """Endpoint MAE by (dataset, d, baseline) + CurveMAE (mean over d grid)."""
    endpoint = (
        df.groupby(["dataset", "d", "baseline"])["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "mae"})
    )
    # CurveMAE: per (dataset, path, repeat) mean abs error across the d grid, then
    # averaged -- matches the paper's existing CurveMAE definition (mean over the
    # reported d grid of the per-point absolute error), not a mean-of-per-d-means.
    expected_n_d = df["d"].nunique()
    per_unit = (
        df.groupby(["dataset", "path_id", "repeat", "baseline"])["abs_error"]
        .agg(curve_mae="mean", n_d="size")
        .reset_index()
    )
    curve = (
        per_unit[per_unit["n_d"] == expected_n_d]
        .groupby(["dataset", "baseline"])
        .agg(curve_mae=("curve_mae", "mean"),
             n_complete_units=("curve_mae", "size"))
        .reset_index()
    )
    attempted = (
        per_unit.groupby(["dataset", "baseline"]).size()
        .rename("n_attempted_units").reset_index()
    )
    curve = curve.merge(attempted, on=["dataset", "baseline"], how="right")
    curve["complete_unit_rate"] = (
        curve["n_complete_units"] / curve["n_attempted_units"]
    )
    return endpoint, curve


def path_clustered_bootstrap_diff(
    df: pd.DataFrame, dataset: str, baseline_a: str, baseline_b: str, d: float | None, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Bootstrap CI for mean(abs_error[baseline_a]) - mean(abs_error[baseline_b]),
    resampling PATHS with replacement (not rows), so within-path repeat correlation
    doesn't understate the CI width."""
    sub = df[df["dataset"] == dataset]
    if d is not None:
        sub = sub[np.isclose(sub["d"], d)]
    keys = ["path_id", "repeat", "d"]
    a_rows = sub[sub["baseline"] == baseline_a][keys + ["abs_error"]].rename(
        columns={"abs_error": "err_a"}
    )
    b_rows = sub[sub["baseline"] == baseline_b][keys + ["abs_error"]].rename(
        columns={"abs_error": "err_b"}
    )
    paired = a_rows.merge(b_rows, on=keys, how="inner", validate="one_to_one")
    by_path = paired.groupby("path_id")[["err_a", "err_b"]].mean()
    a = by_path["err_a"].to_numpy()
    b = by_path["err_b"].to_numpy()
    if len(by_path) < 3:
        return {"n_paths": len(by_path), "n_paired_cells": len(paired),
                "mae_a": float("nan"), "mae_b": float("nan"),
                "diff_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    n = len(by_path)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])
    return {
        "n_paths": n,
        "n_paired_cells": len(paired),
        "mae_a": float(np.mean(a)),
        "mae_b": float(np.mean(b)),
        "diff_mean": float(np.mean(a) - np.mean(b)),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    df = pd.read_csv(out_dir / "direct_calibration_predictions.csv")
    assert df["predicted_gain"].notna().all(), "NaN predictions present -- fix build step before auditing"

    endpoint, curve = curve_mae_table(df)
    endpoint.to_csv(out_dir / "endpoint_mae_by_d.csv", index=False)
    curve.to_csv(out_dir / "curve_mae.csv", index=False)
    print("Endpoint MAE (wide, by dataset/d):")
    wide = endpoint.pivot_table(index=["dataset", "d"], columns="baseline", values="mae").reset_index()
    print(wide.to_string(index=False))
    print("\nCurveMAE (mean abs error across full d grid):")
    print(curve.pivot_table(index="dataset", columns="baseline", values="curve_mae").to_string())

    datasets = df["dataset"].unique().tolist()
    d_targets = sorted(df["d"].unique().tolist())
    rows = []
    for dataset in datasets:
        for baseline in COMPARE_BASELINES:
            res = path_clustered_bootstrap_diff(
                df, dataset, "A+B", baseline, d=None
            )
            rows.append({
                "dataset": dataset, "baseline": baseline, "metric": "CurveMAE",
                "mae_A+B": res.pop("mae_a"),
                "mae_baseline": res.pop("mae_b"),
                "n_paths": res.pop("n_paths"),
                "n_paired_cells": res.pop("n_paired_cells"),
                "diff_AplusB_minus_baseline": res.pop("diff_mean"),
                "ci_lo": res.pop("ci_lo"),
                "ci_hi": res.pop("ci_hi"),
            })

    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(out_dir / "paired_diff_curvemae_bootstrap.csv", index=False)
    print("\nPaired CurveMAE difference (A+B minus baseline), path-clustered bootstrap 95% CI:")
    print(diff_df.to_string(index=False))

    plot(endpoint, curve, out_dir / "fig_direct_calibration.png")


def plot(endpoint: pd.DataFrame, curve: pd.DataFrame, out_path: Path) -> None:
    datasets = endpoint["dataset"].unique().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]
    baselines_to_plot = ["zero", "A-only", "A+B", "empirical_fit", "isotonic_calibration", "binned_shrinkage", "spline_ridge"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(baselines_to_plot)))
    for ax, dataset in zip(axes, datasets):
        sub = endpoint[endpoint["dataset"] == dataset]
        for color, baseline in zip(colors, baselines_to_plot):
            bsub = sub[sub["baseline"] == baseline].sort_values("d")
            if bsub.empty:
                continue
            ax.plot(bsub["d"], bsub["mae"], "o-", label=baseline, color=color)
        ax.set_xlabel("KL radius d")
        ax.set_ylabel("MAE vs realized held-out gain")
        ax.set_title(dataset)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
