#!/usr/bin/env python3
"""Build tables and figures for tail/outlier and BoN scope audits."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
CANONICAL_PDF_DATE = b"D:20000101000000Z"
PDF_DATE_FIELD_RE = re.compile(rb"(/(?:CreationDate|ModDate)\s*\(\s*)(D:[^)]*)(\s*\))")


DATASET_ORDER = ["SummEval", "WMT-MQM-clusters", "OpenMEVA-MAGS"]
CONDITION_ORDER = ["winsor_1_99", "winsor_5_95", "rank_normal", "outlier_gold", "outlier_proxy"]
CONDITION_LABELS = {
    "original": "Original",
    "winsor_1_99": "Winsor 1/99",
    "winsor_5_95": "Winsor 5/95",
    "rank_normal": "Rank-normal",
    "outlier_gold": "Gold outlier",
    "outlier_proxy": "Proxy outlier",
}


def normalize_pdf_metadata(path: Path) -> None:
    data = path.read_bytes()

    def replace_date(match: re.Match[bytes]) -> bytes:
        value = match.group(2)
        if len(value) != len(CANONICAL_PDF_DATE):
            return match.group(0)
        return match.group(1) + CANONICAL_PDF_DATE + match.group(3)

    normalized = PDF_DATE_FIELD_RE.sub(replace_date, data)
    if normalized != data:
        path.write_bytes(normalized)


def fnum(x: Any, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def pct(x: Any, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{100.0 * float(x):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) + "\n"


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=False).replace("%", r"\%")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            body.rstrip(),
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def write_table(df: pd.DataFrame, outdir: Path, stem: str, caption: str, label: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(markdown_table(df), encoding="utf-8")
    (outdir / f"{stem}.tex").write_text(latex_table(df, caption, label), encoding="utf-8")


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    normalize_pdf_metadata(pdf_path)
    plt.close(fig)


def load_tail(root: Path) -> pd.DataFrame:
    files = [
        root / "summeval" / "tail_outlier_robustness_summary.csv",
        root / "openmeva_mags" / "tail_outlier_robustness_summary.csv",
        root / "wmt_mqm_clusters_outlier20" / "tail_outlier_robustness_summary.csv",
    ]
    frames = []
    for path in files:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No tail summary files found under {root}")
    return pd.concat(frames, ignore_index=True)


def make_tail_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[df["condition"].isin(CONDITION_ORDER)].copy()
    keep["dataset_order"] = keep["dataset"].map({v: i for i, v in enumerate(DATASET_ORDER)})
    keep["condition_order"] = keep["condition"].map({v: i for i, v in enumerate(CONDITION_ORDER)})
    keep = keep.sort_values(["dataset_order", "condition_order"])
    rows = []
    for _, row in keep.iterrows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Condition": CONDITION_LABELS.get(row["condition"], row["condition"]),
                "Paths": str(int(row["n_paths"])),
                "corr(A)": fnum(row["corr_a_exact_vs_original"], 3),
                "corr(B)": fnum(row["corr_b_theory_vs_original"], 3),
                "B sign (%)": pct(row["b_sign_agreement_vs_original"], 1),
                "Gain sign (%)": pct(row["delta_sign_agreement_vs_original"], 1),
                "median abs gain shift": fnum(row["median_abs_delta_change"], 3),
                "MAE": fnum(row["mae_theory"], 3),
            }
        )
    return pd.DataFrame(rows)


def plot_tail(df: pd.DataFrame, outdir: Path) -> None:
    keep = df[df["condition"].isin(CONDITION_ORDER)].copy()
    dataset_colors = {
        "SummEval": "#2563eb",
        "WMT-MQM-clusters": "#dc2626",
        "OpenMEVA-MAGS": "#16a34a",
    }
    fig = plt.figure(figsize=(8.6, 3.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.18, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    x = np.arange(len(CONDITION_ORDER))
    width = 0.24
    for i, dataset in enumerate(DATASET_ORDER):
        g = keep[keep["dataset"].eq(dataset)].set_index("condition").reindex(CONDITION_ORDER)
        offset = (i - 1) * width
        axes[0].bar(
            x + offset,
            g["corr_b_theory_vs_original"].to_numpy(float),
            width=width,
            label=dataset,
            color=dataset_colors[dataset],
            alpha=0.86,
        )
        axes[1].bar(
            x + offset,
            g["median_abs_delta_change"].to_numpy(float),
            width=width,
            label=dataset,
            color=dataset_colors[dataset],
            alpha=0.86,
        )
    for ax, letter, title, ylabel in [
        (axes[0], "A", "B-correlation stability", "corr(B, original)"),
        (axes[1], "B", "Gain perturbation size", "median abs gain shift"),
    ]:
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.7)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0].axhline(0, color="#9ca3af", linewidth=0.8)
    axes[0].set_ylim(min(-0.15, float(keep["corr_b_theory_vs_original"].min()) - 0.05), 1.05)
    handles, labels = axes[1].get_legend_handles_labels()
    legend_ax.legend(handles, labels, frameon=False, loc="center", ncol=3)
    save_figure(fig, outdir, "fig_tail_outlier_robustness")


def load_bon(root: Path) -> pd.DataFrame:
    files = [
        root / "summeval" / "prompt_conditional_bon_summary.csv",
        root / "openmeva_mags" / "prompt_conditional_bon_summary.csv",
    ]
    frames = []
    for path in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        dataset = "SummEval" if path.parent.name == "summeval" else "OpenMEVA-MAGS"
        df["dataset"] = dataset
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No prompt-conditional BoN summaries found under {root}")
    return pd.concat(frames, ignore_index=True)


def make_bon_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ["SummEval", "OpenMEVA-MAGS"]:
        g = df[df["dataset"].eq(dataset)].sort_values("n_best")
        for _, row in g.iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "N": str(int(row["n_best"])),
                    "mean KL": fnum(row["mean_prompt_kl"], 3),
                    "median KL": fnum(row["median_prompt_kl"], 3),
                    "prompts KL<=.10 (%)": pct(row["mean_prompt_share_kl_le_0p10"], 1),
                    "pooled tilt reachable, d<=0.25 (%)": pct(row["share_paths_global_tilt_reachable"], 1),
                    "mean BoN gain": fnum(row["mean_conditional_bon_gain"], 3),
                }
            )
    return pd.DataFrame(rows)


def plot_bon(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    colors = {"SummEval": "#2563eb", "OpenMEVA-MAGS": "#16a34a"}
    for dataset in ["SummEval", "OpenMEVA-MAGS"]:
        g = df[df["dataset"].eq(dataset)].sort_values("n_best")
        axes[0].plot(g["n_best"], g["mean_prompt_kl"], marker="o", linewidth=1.8, color=colors[dataset], label=dataset)
        axes[1].plot(
            g["n_best"],
            100 * g["share_paths_global_tilt_reachable"],
            marker="s",
            linewidth=1.8,
            color=colors[dataset],
            label=dataset,
        )
    axes[0].axhline(0.10, color="#dc2626", linestyle="--", linewidth=1.0, label="local window")
    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_xticks([2, 4, 8, 16, 32, 64])
    axes[1].set_xticks([2, 4, 8, 16, 32, 64])
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_title("Prompt-conditional BoN radius")
    axes[1].set_title("Pooled-tilt reachability")
    axes[0].set_xlabel("Best-of-N")
    axes[1].set_xlabel("Best-of-N")
    axes[0].set_ylabel("Mean prompt KL")
    axes[1].set_ylabel(r"Reachable within $d\leq0.25$ (%)")
    axes[1].set_ylim(-3, 103)
    for ax, letter in zip(axes, ["A", "B"]):
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        ax.text(-0.10, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left")
    save_figure(fig, outdir, "fig_prompt_conditional_bon_scope")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tail/outlier and BoN scope audit artifacts.")
    parser.add_argument(
        "--tail-root",
        required=True,
        help="Output directory from experiments/14_tail_outlier_robustness.",
    )
    parser.add_argument(
        "--bon-root",
        required=True,
        help="Output directory from experiments/15_prompt_conditional_bon_scope.",
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--tables-out", default=repo_root / "artifacts/tables")
    parser.add_argument("--figures-out", default=repo_root / "artifacts/figures")
    args = parser.parse_args()

    tables_out = Path(args.tables_out)
    figures_out = Path(args.figures_out)

    tail_df = load_tail(Path(args.tail_root))
    tail_table = make_tail_table(tail_df)
    write_table(
        tail_table,
        tables_out,
        "table_tail_outlier_robustness",
        "Tail and outlier robustness audit. Winsorization and gold outliers leave the coefficient geometry largely stable, while proxy outliers can strongly perturb curvature.",
        "tab:tail-outlier-robustness",
    )
    plot_tail(tail_df, figures_out)

    bon_df = load_bon(Path(args.bon_root))
    bon_table = make_bon_table(bon_df)
    write_table(
        bon_table,
        tables_out,
        "table_prompt_conditional_bon_scope",
        "Prompt-conditional Best-of-N scope audit. Even small Best-of-N interventions often leave the shared local KL window, especially for N greater than two.",
        "tab:prompt-conditional-bon-scope",
    )
    plot_bon(bon_df, figures_out)

    print(f"Wrote tables to {tables_out}")
    print(f"Wrote figures to {figures_out}")


if __name__ == "__main__":
    main()
