"""Compare smooth selection-rule gains with local predictions at matched KL.

These are descriptive in-window comparisons, not raw-score-tilt theorem tests.
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
D_VALID = 0.10
ERROR_COLUMNS = {
    "A+B": "abs_err_theory",
    "A-only": "abs_err_aonly",
    "A_nu": "abs_err_anu",
    "empirical_fit": "abs_err_empfit",
    "zero": "abs_err_zero",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    df = pd.read_csv(out_dir / "mixture_ranktilt_rows.csv")
    inwin = df[(df["d"] > 0) & (df["d"] <= D_VALID)].copy()

    # d-bucket for aggregation (fine bins across the checked window).
    bins = [0, 0.01, 0.025, 0.05, 0.075, 0.10]
    inwin["d_bucket"] = pd.cut(inwin["d"], bins=bins, include_lowest=True)

    by_mech_bucket = (
        inwin.groupby(["mechanism", "d_bucket"], observed=True)
        .agg(n=("abs_err_theory", "size"), median_d=("d", "median"),
             mae_theory=("abs_err_theory", "mean"))
        .reset_index()
    )
    # correlation needs paired columns; compute separately
    corrs = []
    for (mech, bucket), g in inwin.groupby(["mechanism", "d_bucket"], observed=True):
        if len(g) >= 3 and g["delta_realized"].std() > 0 and g["theory_pred"].std() > 0:
            corrs.append({"mechanism": mech, "d_bucket": bucket,
                           "pearson_r": float(np.corrcoef(g["delta_realized"], g["theory_pred"])[0, 1])})
        else:
            corrs.append({"mechanism": mech, "d_bucket": bucket, "pearson_r": float("nan")})
    corr_df = pd.DataFrame(corrs)
    by_mech_bucket = by_mech_bucket.merge(corr_df, on=["mechanism", "d_bucket"], how="left")
    by_mech_bucket.to_csv(out_dir / "inwindow_summary_by_bucket.csv", index=False)
    print("In-window (d<=0.10) A+B accuracy, by mechanism and d-bucket:")
    print(by_mech_bucket.to_string(index=False))

    by_dataset = (
        inwin.groupby(["dataset", "mechanism"])
        .agg(n=("abs_err_theory", "size"), mae_theory=("abs_err_theory", "mean"),
             mean_realized=("delta_realized", "mean"))
        .reset_index()
    )
    by_dataset.to_csv(out_dir / "inwindow_summary_by_dataset.csv", index=False)
    print("\nIn-window MAE by dataset x mechanism:")
    print(by_dataset.to_string(index=False))

    # Zero-gain baseline for reference (how much of the realized signal is nontrivial).
    inwin_nonzero = inwin[inwin["d"] > 0.005]
    zero_mae = float(inwin_nonzero["delta_realized"].abs().mean())
    theory_mae = float(inwin_nonzero["abs_err_theory"].mean())
    print(f"\nOverall in-window (d>0.005): zero-gain MAE={zero_mae:.4f}, A+B MAE={theory_mae:.4f}, "
          f"improvement factor={zero_mae / theory_mae:.2f}x")

    # Baseline-complete, path-macro summary. Each path receives equal weight even
    # though mechanisms contribute different numbers of radius points.
    macro_rows = []
    for baseline, err_col in ERROR_COLUMNS.items():
        per_path = (
            inwin_nonzero[np.isfinite(inwin_nonzero[err_col])]
            .groupby(["dataset", "mechanism", "path_id"])[err_col]
            .mean().reset_index(name="path_mae")
        )
        for (dataset, mechanism), g in per_path.groupby(["dataset", "mechanism"]):
            macro_rows.append({
                "dataset": dataset,
                "mechanism": mechanism,
                "baseline": baseline,
                "n_paths": g["path_id"].nunique(),
                "path_macro_mae": float(g["path_mae"].mean()),
            })
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(out_dir / "inwindow_baseline_path_macro.csv", index=False)
    print("\nIn-window baseline comparison (path-macro MAE, d>0.005):")
    print(macro.to_string(index=False))

    # Path-level correlation avoids treating many monotone radius points from one
    # path as independent evidence.
    corr_rows = []
    for (dataset, mechanism, path_id), g in inwin_nonzero.groupby(
        ["dataset", "mechanism", "path_id"]
    ):
        if len(g) >= 3 and g["theory_pred"].std() > 0 and g["delta_realized"].std() > 0:
            corr_rows.append({
                "dataset": dataset, "mechanism": mechanism, "path_id": path_id,
                "pearson_within_path": float(np.corrcoef(
                    g["theory_pred"], g["delta_realized"]
                )[0, 1]),
            })
    corr_path = pd.DataFrame(corr_rows)
    corr_path.to_csv(out_dir / "inwindow_correlation_by_path.csv", index=False)

    plot(inwin, out_dir / "fig_mixture_ranktilt.png")


def plot(inwin: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mech in zip(axes, ["alpha_mixture", "rank_tilt"]):
        sub = inwin[inwin["mechanism"] == mech]
        ax.scatter(sub["theory_pred"], sub["delta_realized"], s=4, alpha=0.2, color="steelblue")
        lims = [min(sub["theory_pred"].min(), sub["delta_realized"].min()),
                max(sub["theory_pred"].max(), sub["delta_realized"].max())]
        ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
        ax.set_xlabel("A+B predicted gain")
        ax.set_ylabel("Realized gain")
        ax.set_title(f"{mech} (in-window, d<=0.10)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
