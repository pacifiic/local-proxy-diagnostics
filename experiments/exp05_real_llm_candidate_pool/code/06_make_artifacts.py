#!/usr/bin/env python3
"""Build compact tables and figures for the real LLM candidate-pool experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


DISPLAY = ["A+B", "empirical_fit", "isotonic_calibration", "mlp_calibration", "A-only", "zero"]
LABELS = {
    "A+B": "A+B",
    "empirical_fit": "empirical fit",
    "isotonic_calibration": "isotonic",
    "mlp_calibration": "MLP",
    "A-only": "A-only",
    "zero": "zero",
}
COLORS = {
    "A+B": "#1f77b4",
    "empirical_fit": "#2ca02c",
    "isotonic_calibration": "#9467bd",
    "mlp_calibration": "#d62728",
    "A-only": "#ff7f0e",
    "zero": "#7f7f7f",
}


def plot_endpoint(endpoint: pd.DataFrame, outpath: Path, variant_label: str = "") -> None:
    sub = endpoint[endpoint["baseline"].isin(DISPLAY)].copy()
    sub["label"] = sub["baseline"].map(LABELS)
    sub = sub.set_index("baseline").loc[[b for b in DISPLAY if b in set(sub["baseline"])]].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(
        sub["label"],
        sub["mae"],
        color=[COLORS[b] for b in sub["baseline"]],
        yerr=sub["repeat_se"].fillna(0.0),
        capsize=3,
    )
    ax.set_ylabel("MAE at target KL radius")
    ax.set_title(f"Held-out candidate-pool gain prediction {variant_label}".strip())
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, sub["mae"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=180)
    plt.close(fig)


def plot_bon(bon: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes[0].plot(bon["n_best"], bon["mean_prompt_kl"], marker="o", color="#1f77b4")
    axes[0].axhline(0.10, color="#d62728", linestyle="--", linewidth=1.5, label="local d=0.10")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Best-of-N")
    axes[0].set_ylabel("Mean prompt KL")
    axes[0].set_title("BoN moves outside local window")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(bon["n_best"], bon["mean_actual_bon_gain"], marker="o", label="actual BoN gain", color="#2ca02c")
    axes[1].plot(bon["n_best"], bon["mean_local_ab_at_mean_kl"], marker="s", label="A+B at mean KL", color="#1f77b4")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Best-of-N")
    axes[1].set_ylabel("Gold gain")
    axes[1].set_title("Discrete scope check")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compact tables/figures for Experiment 19.")
    ap.add_argument("--gain-root", required=True)
    ap.add_argument("--bon-root", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--variant-label", default="", help="Appended to figure titles, e.g. '(SYNTHETIC pipeline check)' or '(real LLM candidates)'.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    endpoint = pd.read_csv(Path(args.gain_root) / "llm_gain_curve_endpoint_summary.csv")
    curve = pd.read_csv(Path(args.gain_root) / "llm_gain_curve_curve_summary.csv")
    endpoint.to_csv(outdir / "table_llm_candidate_pool_endpoint.csv", index=False)
    endpoint.to_markdown(outdir / "table_llm_candidate_pool_endpoint.md", index=False)
    curve.to_csv(outdir / "table_llm_candidate_pool_curve.csv", index=False)
    curve.to_markdown(outdir / "table_llm_candidate_pool_curve.md", index=False)
    plot_endpoint(endpoint, outdir / "fig_llm_candidate_pool_endpoint.pdf", variant_label=args.variant_label)
    if args.bon_root:
        bon_path = Path(args.bon_root) / "llm_bon_scope_summary.csv"
        if bon_path.exists():
            bon = pd.read_csv(bon_path)
            bon.to_csv(outdir / "table_llm_bon_scope.csv", index=False)
            bon.to_markdown(outdir / "table_llm_bon_scope.md", index=False)
            plot_bon(bon, outdir / "fig_llm_bon_scope.pdf")
    print(f"Wrote analysis artifacts to: {outdir}")


if __name__ == "__main__":
    main()
