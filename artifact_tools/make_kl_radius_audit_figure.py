#!/usr/bin/env python3
"""Build a KL-radius sensitivity/reachability audit figure.

This script is intentionally additive: it reads existing d-sweep summaries and
writes a publication figure, without changing the experiment code that
generated those summaries.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
CANONICAL_PDF_DATE = b"D:20000101000000Z"
PDF_DATE_FIELD_RE = re.compile(
    rb"(/(?:CreationDate|ModDate)\s*\(\s*)(D:[^)]*)(\s*\))"
)


def _normalize_pdf_metadata(path: Path) -> None:
    data = path.read_bytes()

    def replace_date(match: re.Match[bytes]) -> bytes:
        value = match.group(2)
        if len(value) != len(CANONICAL_PDF_DATE):
            return match.group(0)
        return match.group(1) + CANONICAL_PDF_DATE + match.group(3)

    normalized = PDF_DATE_FIELD_RE.sub(replace_date, data)
    if normalized != data:
        path.write_bytes(normalized)


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "group_key" in df.columns:
        df = df[df["group_key"].astype(str).eq("__OVERALL__")].copy()
    if df.empty:
        raise ValueError(f"No overall rows found in {path}")
    return df.sort_values("d").reset_index(drop=True)


def _series(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype=float)


def _plot_mae(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    x = _series(df, "d")
    predictors = [
        ("A+B", "mae_theory", "#2563eb", "o"),
        ("A-only", "mae_aonly", "#d97706", "s"),
        ("empirical fit", "mae_empfit", "#16a34a", "^"),
        (r"$A_\nu$", "mae_anu", "#dc2626", "D"),
        ("zero", "mae_zero", "#6b7280", "X"),
    ]
    for label, col, color, marker in predictors:
        if col not in df.columns:
            continue
        y = _series(df, col)
        is_empfit = col == "mae_empfit"
        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=0.95 if is_empfit else 1.7,
            markersize=3.2 if is_empfit else 3.4,
            color=color,
            alpha=0.72 if is_empfit else 1.0,
            label=label,
        )
        lo_col = f"{col}_ci_lo"
        hi_col = f"{col}_ci_hi"
        if lo_col in df.columns and hi_col in df.columns:
            ax.fill_between(
                x,
                _series(df, lo_col),
                _series(df, hi_col),
                color=color,
                alpha=0.04 if is_empfit else 0.08,
                linewidth=0,
            )
    ax.axvspan(0, 0.10, color="#dbeafe", alpha=0.28, linewidth=0)
    ax.axvline(0.10, color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("KL radius $d$")
    ax.set_ylabel("MAE of gain prediction")
    ax.set_xlim(0, max(0.205, float(np.nanmax(x)) * 1.03))
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)


def _plot_sign(ax: plt.Axes, frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        x = _series(df, "d")
        y = 100.0 * _series(df, "sign_acc_theory")
        color = "#2563eb" if name.startswith("Human") else "#dc2626"
        marker = "o" if name.startswith("Human") else "s"
        ax.plot(x, y, marker=marker, linewidth=1.7, markersize=3.4, color=color, label=name)
    ax.axhline(50, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.axvspan(0, 0.10, color="#dbeafe", alpha=0.28, linewidth=0)
    ax.axvline(0.10, color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.set_title("Predicted gain sign")
    ax.set_xlabel("KL radius $d$")
    ax.set_ylabel("A+B sign accuracy (%)")
    max_x = max(float(np.nanmax(_series(df, "d"))) for df in frames.values())
    ax.set_xlim(0, max(0.205, max_x * 1.03))
    ax.set_ylim(-3, 103)
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="lower left")


def _plot_reachability(ax: plt.Axes, frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        x = _series(df, "d")
        path_pct = 100.0 * _series(df, "valid_paths") / _series(df, "n_paths")
        row_pct = 100.0 * _series(df, "valid_rows") / _series(df, "n_rows")
        color = "#2563eb" if name.startswith("Human") else "#dc2626"
        marker = "o" if name.startswith("Human") else "s"
        counts = sorted(set(df["n_rows"].astype(int)))
        base = f"n={counts[0]:,}" if len(counts) == 1 else "radius-specific n"
        ax.plot(x, row_pct, marker=marker, linewidth=1.7, markersize=3.4, color=color, label=f"{name} ({base})")
        changed = np.where(path_pct < 99.999)[0]
        annotate_idx = set()
        if len(changed):
            annotate_idx.add(int(changed[0]))
        if len(x):
            annotate_idx.add(len(x) - 1)
        for i in sorted(annotate_idx):
            if i < 0 or i >= len(x):
                continue
            if x[i] <= 0.10 and path_pct[i] >= 99.999:
                continue
            dy = -14 if name.startswith("Human") else 8
            ax.annotate(
                f"{int(df.loc[i, 'valid_paths'])}/{int(df.loc[i, 'n_paths'])} paths",
                (x[i], row_pct[i]),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=6.8,
                color=color,
            )
    ax.axvspan(0, 0.10, color="#dbeafe", alpha=0.28, linewidth=0)
    ax.axvline(0.10, color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.set_title("Split coverage; labels count paths")
    ax.set_xlabel("KL radius $d$")
    ax.set_ylabel("Reachable retained splits (%)")
    max_x = max(float(np.nanmax(_series(df, "d"))) for df in frames.values())
    ax.set_xlim(0, max(0.205, max_x * 1.03))
    min_pct = min(float((100.0 * _series(df, "valid_rows") / _series(df, "n_rows")).min()) for df in frames.values())
    ax.set_ylim(max(0, min(79, min_pct - 5)), 103)
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, loc="lower left")


def build_figure(human: pd.DataFrame, gpt4o: pd.DataFrame, out_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.65), constrained_layout=True)
    for ax, letter in zip(axes, ["A", "B", "C"]):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")

    _plot_mae(axes[0], human, "Human proxy")
    _plot_mae(axes[1], gpt4o, "GPT-4o proxy")
    _plot_reachability(axes[2], {"Human proxy": human, "GPT-4o proxy": gpt4o})

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    _normalize_pdf_metadata(pdf_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a KL-radius MAE/sign/reachability audit figure.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--human-summary", default="12_accept_boost/d_sweep_human/d_sweep_summary_overall.csv")
    parser.add_argument("--gpt4o-summary", default="12_accept_boost/d_sweep_gpt4o/d_sweep_summary_overall.csv")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--stem", default="fig_kl_radius_audit")
    args = parser.parse_args()

    root = Path.cwd()
    results = root / args.results_root
    human = _load_summary(results / args.human_summary)
    gpt4o = _load_summary(results / args.gpt4o_summary)
    out_prefix = root / args.outdir / args.stem
    build_figure(human, gpt4o, out_prefix)
    print(f"Saved {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
