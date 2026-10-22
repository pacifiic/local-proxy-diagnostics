"""Prediction accuracy as the KL radius d varies.

This is the main local-validity stress test.  It reruns the prospective
calibration/evaluation split protocol over a grid of KL radii and compares the
coefficient predictor against A-only, empirical-fit, reliability-only A_nu,
zero-gain, and oracle-evaluation baselines.

Example: human-heldout SummEval
    python experiments/12_accept_boost/01_d_sweep_prediction.py \
      --human-long-csv data/summeval_long.csv \
      --item-cols article_id,model_id --group-cols dimension,rater_type \
      --d-grid 0.005,0.01,0.02,0.05,0.10,0.15,0.20 \
      --outdir all_results_reproduced/12_accept_boost/d_sweep_human

Example: GPT-4o proxy
    python experiments/12_accept_boost/01_d_sweep_prediction.py \
      --human-long-csv data/summeval_long.csv \
      --item-cols article_id,model_id --group-cols dimension \
      --proxy-csv data/model_proxy_4o_aggregated.csv \
      --proxy-item-col item_id --proxy-score-col proxy_score_mean \
      --outdir all_results_reproduced/12_accept_boost/d_sweep_gpt4o
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accept_boost_utils import (
    PREDICTOR_LABELS,
    PREDICTORS,
    load_paths_from_args,
    mkdir,
    parse_float_list,
    run_prediction_splits,
    save_markdown,
    summarize_prediction_long,
)


def plot_mae(summary: pd.DataFrame, outdir: Path, predictors: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for p in predictors:
        col = f"mae_{p}"
        if col not in summary:
            continue
        sub = summary.sort_values("d")
        y = sub[col].to_numpy(dtype=float)
        lo = sub.get(f"{col}_ci_lo", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        hi = sub.get(f"{col}_ci_hi", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        ax.plot(sub["d"], y, marker="o", linewidth=1.8, label=PREDICTOR_LABELS.get(p, p))
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(sub["d"], lo, hi, alpha=0.12)
    ax.set_xlabel("KL radius d")
    ax.set_ylabel("MAE of predicted human gain")
    ax.set_title("Local prediction error increases with optimization radius")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_d_sweep_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_d_sweep_mae.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sign(summary: pd.DataFrame, outdir: Path, predictors: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for p in predictors:
        col = f"sign_acc_{p}"
        if col not in summary:
            continue
        sub = summary.sort_values("d")
        y = 100.0 * sub[col].to_numpy(dtype=float)
        lo = 100.0 * sub.get(f"{col}_ci_lo", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        hi = 100.0 * sub.get(f"{col}_ci_hi", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        ax.plot(sub["d"], y, marker="o", linewidth=1.8, label=PREDICTOR_LABELS.get(p, p))
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(sub["d"], lo, hi, alpha=0.12)
    ax.set_xlabel("KL radius d")
    ax.set_ylabel("Sign accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Sign prediction across local and non-local radii")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_d_sweep_sign_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_d_sweep_sign_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="KL-radius d-sweep for prospective local-gain prediction.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--d-grid", default="0.005,0.01,0.02,0.05,0.10,0.15,0.20")
    ap.add_argument("--empfit-max-d", type=float, default=0.10,
                    help="Fit window for empirical A+B baseline on calibration curve.")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--n-grid", type=int, default=240)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--predictors", default="theory,aonly,empfit,anu,zero,oracle")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    d_grid = parse_float_list(args.d_grid)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    paths = load_paths_from_args(args)
    print(f"Loaded {len(paths)} paths; d_grid={d_grid}; repeats/path={args.n_repeats}")

    long_df = run_prediction_splits(
        paths=paths,
        d_grid=d_grid,
        calibration_frac=args.calibration_frac,
        n_repeats=args.n_repeats,
        seed=args.random_seed,
        empfit_max_d=args.empfit_max_d,
        n_grid=args.n_grid,
    )
    long_csv = outdir / "d_sweep_paths_long.csv"
    long_df.to_csv(long_csv, index=False)

    by_group = summarize_prediction_long(
        long_df,
        group_cols=["group_key", "d"],
        predictors=predictors,
        n_boot=args.n_boot,
        seed=args.random_seed,
    )
    by_group.to_csv(outdir / "d_sweep_summary_by_group.csv", index=False)
    save_markdown(by_group, outdir / "d_sweep_summary_by_group.md")

    overall = summarize_prediction_long(
        long_df.assign(group_key="__OVERALL__"),
        group_cols=["group_key", "d"],
        predictors=predictors,
        n_boot=args.n_boot,
        seed=args.random_seed,
    )
    overall.to_csv(outdir / "d_sweep_summary_overall.csv", index=False)
    save_markdown(overall, outdir / "d_sweep_summary_overall.md")

    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "item_cols": args.item_cols,
        "group_cols": args.group_cols,
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "proxy_item_col": args.proxy_item_col,
        "proxy_score_col": args.proxy_score_col,
        "calibration_frac": args.calibration_frac,
        "n_repeats": args.n_repeats,
        "d_grid": d_grid,
        "empfit_max_d": args.empfit_max_d,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "random_seed": args.random_seed,
        "n_grid": args.n_grid,
        "n_boot": args.n_boot,
        "predictors": predictors,
        "n_paths": len(paths),
        "n_rows": len(long_df),
        "outputs": {
            "paths": str(long_csv),
            "summary_by_group": str(outdir / "d_sweep_summary_by_group.csv"),
            "summary_overall": str(outdir / "d_sweep_summary_overall.csv"),
        },
    }
    (outdir / "d_sweep_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    plot_mae(overall, outdir, predictors=[p for p in predictors if p != "oracle"])
    plot_sign(overall, outdir, predictors=[p for p in predictors if p != "oracle"])

    print(f"Saved long rows: {long_csv}")
    print(f"Saved summaries and figures in: {outdir}")
    print(overall[["d", "n_rows", "n_paths", "mae_theory", "mae_aonly", "mae_zero", "mae_anu"]].to_string(index=False))


if __name__ == "__main__":
    main()
