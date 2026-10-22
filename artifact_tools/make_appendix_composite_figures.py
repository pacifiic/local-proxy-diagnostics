#!/usr/bin/env python3
"""Build appendix composite figures from accept-boost summary CSVs."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANONICAL_PDF_DATE = b"D:20000101000000Z"
PDF_DATE_FIELD_RE = re.compile(rb"(/(CreationDate|ModDate)\s*\()([^)]*)(\))")

PREDICTORS = [
    ("mae_theory", "A+B coefficient", "#1f77b4", "-", "o"),
    ("mae_aonly", "A-only", "#ff7f0e", "-", "s"),
    ("mae_empfit", "empirical fit", "#2ca02c", "-", "^"),
    ("mae_anu", r"reliability-only $A_\nu$", "#d62728", "-", "D"),
    ("mae_zero", "zero gain", "#9467bd", "-", "X"),
]

RESIDUAL_PREDICTORS = [
    ("theory_ab", "A+B", "#1f77b4", "o"),
    ("aonly", "A-only", "#ff7f0e", "s"),
    ("fit_ab", "same-pool fit", "#2ca02c", "^"),
    ("anu", r"$A_\nu$", "#d62728", "D"),
    ("zero", "zero gain", "#7f7f7f", "X"),
]

SELECTIVE_PREDICTORS = {
    "theory": ("A+B", "#1f77b4", "o"),
    "aonly": ("A-only", "#ff7f0e", "s"),
    "empfit": ("calibration fit", "#2ca02c", "^"),
    "anu": (r"$A_\nu$", "#d62728", "D"),
}


def _normalize_pdf_metadata(path: Path) -> None:
    data = path.read_bytes()

    def replace_date(match: re.Match[bytes]) -> bytes:
        value = match.group(3)
        if len(value) != len(CANONICAL_PDF_DATE):
            return match.group(0)
        return match.group(1) + CANONICAL_PDF_DATE + match.group(4)

    path.write_bytes(PDF_DATE_FIELD_RE.sub(replace_date, data))


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def _title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, fontsize=9.6, fontweight="normal", pad=8)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _make_figure_grid() -> tuple[plt.Figure, np.ndarray, plt.Axes]:
    fig = plt.figure(figsize=(8.8, 6.3), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.08],
        hspace=0.32,
        left=0.08,
        right=0.985,
        top=0.92,
        bottom=0.055,
    )
    plots = outer[0].subgridspec(
        2,
        2,
        wspace=0.22,
        hspace=0.45,
    )
    axes = np.array(
        [
            [fig.add_subplot(plots[0, 0]), fig.add_subplot(plots[0, 1])],
            [fig.add_subplot(plots[1, 0]), fig.add_subplot(plots[1, 1])],
        ]
    )
    legend_ax = fig.add_subplot(outer[1])
    legend_ax.axis("off")
    return fig, axes, legend_ax


def _make_three_panel_grid() -> tuple[plt.Figure, np.ndarray, plt.Axes]:
    fig = plt.figure(figsize=(8.8, 3.95), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.16],
        hspace=0.42,
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.065,
    )
    plots = outer[0].subgridspec(1, 3, wspace=0.34)
    axes = np.array([fig.add_subplot(plots[0, i]) for i in range(3)])
    legend_ax = fig.add_subplot(outer[1])
    legend_ax.axis("off")
    return fig, axes, legend_ax


def _draw_shared_legend(
    legend_ax: plt.Axes,
    handles: list[object],
    labels: list[str],
    *,
    ncol: int,
    fontsize: float = 8.3,
    handlelength: float = 1.6,
    columnspacing: float = 1.0,
) -> None:
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=handlelength,
        columnspacing=columnspacing,
        handletextpad=0.5,
    )


def _plot_d_sweep(ax: plt.Axes, df: pd.DataFrame, title: str, panel: str) -> None:
    df = df.sort_values("d")
    for col, label, color, linestyle, marker in PREDICTORS:
        if col not in df:
            continue
        is_empfit = col == "mae_empfit"
        ax.plot(
            df["d"],
            df[col],
            marker=marker,
            markersize=3.5 if is_empfit else 3.7,
            linewidth=0.95 if is_empfit else 1.6,
            linestyle=linestyle,
            color=color,
            alpha=0.72 if is_empfit else 1.0,
            label=label,
        )
        lo, hi = f"{col}_ci_lo", f"{col}_ci_hi"
        if lo in df and hi in df:
            ax.fill_between(df["d"], df[lo], df[hi], color=color, alpha=0.045 if is_empfit else 0.10, linewidth=0)
    ax.axvspan(0, 0.10, color="#E3F2FD", alpha=0.45, zorder=-5)
    ax.axvline(0.10, color="#1565C0", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("KL radius $d$")
    ax.set_ylabel("MAE")
    _title(ax, title)
    _panel_label(ax, panel)
    ax.grid(True, alpha=0.25)


def _plot_calibration(ax: plt.Axes, df: pd.DataFrame, title: str, panel: str) -> None:
    df = df[np.isclose(df["d"], 0.10)].sort_values("calibration_frac")
    x = 100.0 * df["calibration_frac"]
    for col, label, color, linestyle, marker in PREDICTORS:
        if col not in df:
            continue
        is_empfit = col == "mae_empfit"
        ax.plot(
            x,
            df[col],
            marker=marker,
            markersize=3.5 if is_empfit else 3.7,
            linewidth=0.95 if is_empfit else 1.6,
            linestyle=linestyle,
            color=color,
            alpha=0.72 if is_empfit else 1.0,
            label=label,
        )
        lo, hi = f"{col}_ci_lo", f"{col}_ci_hi"
        if lo in df and hi in df:
            ax.fill_between(x, df[lo], df[hi], color=color, alpha=0.045 if is_empfit else 0.10, linewidth=0)
    ax.set_xlabel("Calibration split fraction (%)")
    ax.set_ylabel("MAE at $d=0.10$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    _title(ax, title)
    _panel_label(ax, panel)
    ax.grid(True, alpha=0.25)


def build_radius_calibration(results_root: Path, outdir: Path) -> Path:
    base = results_root / "12_accept_boost"
    human_d = _load_csv(base / "d_sweep_human" / "d_sweep_summary_overall.csv")
    gpt_d = _load_csv(base / "d_sweep_gpt4o" / "d_sweep_summary_overall.csv")
    human_cal = _load_csv(base / "cal_size_human" / "calibration_size_summary_overall.csv")
    gpt_cal = _load_csv(base / "cal_size_gpt4o" / "calibration_size_summary_overall.csv")

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5})
    fig, axes, legend_ax = _make_figure_grid()
    _plot_d_sweep(axes[0, 0], human_d, "SummEval human proxy: radius sweep", "A")
    _plot_d_sweep(axes[0, 1], gpt_d, "GPT-4o: radius sweep", "B")
    _plot_calibration(axes[1, 0], human_cal, "SummEval human proxy: calibration size", "C")
    _plot_calibration(axes[1, 1], gpt_cal, "GPT-4o: calibration size", "D")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _draw_shared_legend(legend_ax, handles, labels, ncol=5)
    out = outdir / "fig_radius_calibration_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _normalize_pdf_metadata(out)
    return out


def _plot_residual(ax: plt.Axes, df: pd.DataFrame, title: str, panel: str) -> None:
    overall = df[df["group_key"].eq("__OVERALL__")].copy()
    for pred, label, color, marker in RESIDUAL_PREDICTORS:
        sub = overall[overall["predictor"].eq(pred)].sort_values("D")
        if sub.empty:
            continue
        linewidth = 0.95 if pred == "fit_ab" else 1.6
        alpha = 0.75 if pred == "fit_ab" else 1.0
        ax.plot(
            sub["D"],
            sub["median_max_abs_resid"],
            marker=marker,
            markersize=3.5 if pred == "fit_ab" else 3.7,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            label=label,
        )
    ax.set_xlabel("Residual window $D$")
    ax.set_ylabel(r"Median max $|\widehat{\Delta}-\Delta|$")
    _title(ax, title)
    _panel_label(ax, panel)
    ax.grid(True, alpha=0.25)


def _plot_selective(ax: plt.Axes, df: pd.DataFrame, title: str, panel: str) -> None:
    for pred, sub in df.groupby("predictor"):
        label, color, marker = SELECTIVE_PREDICTORS.get(pred, (pred, "#666666", "o"))
        sub = sub.sort_values("coverage")
        ax.plot(100.0 * sub["coverage"], 100.0 * sub["sign_acc_covered"], marker=marker, markersize=3.7, linewidth=1.6, color=color, label=label)
    ax.axhline(50, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Coverage after abstention (%)")
    ax.set_ylabel("Covered sign accuracy (%)")
    _title(ax, title)
    _panel_label(ax, panel)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)


def _plot_policy(ax: plt.Axes, df: pd.DataFrame, title: str, panel: str) -> None:
    for pred, sub in df.groupby("predictor"):
        label, color, marker = SELECTIVE_PREDICTORS.get(pred, (pred, "#666666", "o"))
        sub = sub.sort_values("threshold")
        ax.plot(sub["threshold"], sub["policy_realized_gain"], marker=marker, markersize=3.7, linewidth=1.6, color=color, label=label)
    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.set_xlabel("Predicted-margin threshold $\\tau$")
    ax.set_ylabel("Mean realized policy gain")
    _title(ax, title)
    _panel_label(ax, panel)
    ax.grid(True, alpha=0.25)


def build_residual_selective(results_root: Path, outdir: Path) -> Path:
    base = results_root / "12_accept_boost"
    human_resid = _load_csv(base / "residual_human" / "curve_residual_summary.csv")
    gpt_resid = _load_csv(base / "residual_gpt4o" / "curve_residual_summary.csv")
    selective = _load_csv(base / "selective_gpt4o" / "selective_prediction_gate_summary.csv")

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5})
    fig, axes, legend_ax = _make_three_panel_grid()
    _plot_residual(axes[0], human_resid, "SummEval human proxy\nSame-pool approximation", "A")
    _plot_residual(axes[1], gpt_resid, "GPT-4o proxy\nSame-pool approximation", "B")
    _plot_policy(axes[2], selective, "Optimize/abstain policy", "C")
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[2].get_legend_handles_labels()
    seen: set[str] = set()
    all_handles: list[object] = []
    all_labels: list[str] = []
    for h, lab in [*zip(handles, labels), *zip(handles2, labels2)]:
        if lab not in seen:
            seen.add(lab)
            all_handles.append(h)
            all_labels.append(lab)
    _draw_shared_legend(
        legend_ax,
        all_handles,
        all_labels,
        ncol=3,
        fontsize=7.7,
        handlelength=1.25,
        columnspacing=0.65,
    )
    out = outdir / "fig_residual_selective_diagnostics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    _normalize_pdf_metadata(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build appendix composite figures B.8 and B.9.")
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out1 = build_radius_calibration(args.results_root, args.outdir)
    out2 = build_residual_selective(args.results_root, args.outdir)
    print(f"Saved {out1}")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
