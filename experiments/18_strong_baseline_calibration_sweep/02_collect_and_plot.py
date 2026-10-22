#!/usr/bin/env python3
"""Collect and plot Experiment 18 calibration-size sweep outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


DISPLAY_BASELINES = ["A+B", "empirical_fit", "isotonic_calibration", "mlp_calibration", "A-only"]
LABELS = {
    "A+B": "A+B",
    "empirical_fit": "empirical fit",
    "isotonic_calibration": "isotonic",
    "mlp_calibration": "MLP calibrator",
    "A-only": "A-only",
}
COLORS = {
    "A+B": "#1f77b4",
    "empirical_fit": "#2ca02c",
    "isotonic_calibration": "#9467bd",
    "mlp_calibration": "#d62728",
    "A-only": "#ff7f0e",
}
MARKERS = {
    "A+B": "o",
    "empirical_fit": "s",
    "isotonic_calibration": "^",
    "mlp_calibration": "P",
    "A-only": "D",
}


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoints = []
    curves = []
    for p in sorted(root.glob("*/calibration_size_strong_baselines_endpoint_summary.csv")):
        endpoints.append(pd.read_csv(p))
    for p in sorted(root.glob("*/calibration_size_strong_baselines_curve_summary.csv")):
        curves.append(pd.read_csv(p))
    if not endpoints:
        raise FileNotFoundError(f"No endpoint summaries found under {root}")
    endpoint = pd.concat(endpoints, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return endpoint, curve


def plot_endpoint(endpoint: pd.DataFrame, outpath: Path) -> None:
    plot_df = endpoint[endpoint["baseline"].isin(DISPLAY_BASELINES)].copy()
    datasets = list(dict.fromkeys(plot_df["dataset"].tolist()))
    n = len(datasets)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 4.1 * nrows), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        sub = plot_df[plot_df["dataset"] == dataset]
        for baseline in DISPLAY_BASELINES:
            g = sub[sub["baseline"] == baseline].sort_values("calibration_frac")
            if g.empty:
                continue
            x = 100.0 * g["calibration_frac"].to_numpy()
            y = g["mae"].to_numpy()
            yerr = g["path_se"].fillna(0.0).to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                label=LABELS[baseline],
                color=COLORS[baseline],
                marker=MARKERS[baseline],
                linewidth=2.0,
                markersize=5,
                capsize=2,
            )
        ax.set_title(dataset.replace("_", " "))
        ax.set_xlabel("Calibration split (%)")
        ax.set_ylabel("MAE at d = 0.10")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(datasets) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(DISPLAY_BASELINES), frameon=False)
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    endpoint, curve = collect(root)
    endpoint.to_csv(outdir / "all_calibration_size_strong_baselines_endpoint_summary.csv", index=False)
    endpoint.to_markdown(outdir / "all_calibration_size_strong_baselines_endpoint_summary.md", index=False)
    if not curve.empty:
        curve.to_csv(outdir / "all_calibration_size_strong_baselines_curve_summary.csv", index=False)
        curve.to_markdown(outdir / "all_calibration_size_strong_baselines_curve_summary.md", index=False)
    compact = endpoint[endpoint["d"].round(10).eq(0.10)].pivot_table(
        index=["dataset", "proxy_mode", "calibration_frac"],
        columns="baseline",
        values="mae",
        aggfunc="mean",
    ).reset_index()
    compact.to_csv(outdir / "compact_endpoint_mae_wide.csv", index=False)
    compact.to_markdown(outdir / "compact_endpoint_mae_wide.md", index=False)
    plot_endpoint(endpoint, outdir / "fig_calibration_size_strong_baselines.pdf")
    print(f"Wrote collected tables and figure to: {outdir}")


if __name__ == "__main__":
    main()
