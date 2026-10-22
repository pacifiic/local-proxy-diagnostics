"""Calibration-size learning curve for local gain prediction.

This script asks how much repeated-rating calibration is needed before the
coefficient-based diagnostic becomes stable.  It reruns the prospective split
experiment for multiple calibration fractions and compares A+B against A-only,
zero, empirical-fit, A_nu, and oracle baselines.

Example:
    python experiments/12_accept_boost/03_calibration_size_curve.py \
      --human-long-csv data/summeval_long.csv \
      --item-cols article_id,model_id --group-cols dimension,rater_type \
      --calibration-fracs 0.10,0.20,0.30,0.50,0.70 \
      --d-targets 0.05,0.10 \
      --outdir all_results_reproduced/12_accept_boost/cal_size_human
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
    load_paths_from_args,
    mkdir,
    parse_float_list,
    run_prediction_splits,
    save_markdown,
    summarize_prediction_long,
)


def plot_calibration_curve(summary: pd.DataFrame, outdir: Path, d_value: float, predictors: list[str]) -> None:
    sub0 = summary[summary["d"].eq(float(d_value))].copy()
    if sub0.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for p in predictors:
        col = f"mae_{p}"
        if col not in sub0:
            continue
        sub = sub0.sort_values("calibration_frac")
        x = 100.0 * sub["calibration_frac"].to_numpy(dtype=float)
        y = sub[col].to_numpy(dtype=float)
        lo = sub.get(f"{col}_ci_lo", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        hi = sub.get(f"{col}_ci_hi", pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=PREDICTOR_LABELS.get(p, p))
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, alpha=0.12)
    ax.set_xlabel("Calibration split fraction (%)")
    ax.set_ylabel(f"MAE at d={d_value:g}")
    ax.set_title("Calibration-size learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="center right", frameon=True)
    fig.tight_layout()
    stem = f"fig_calibration_size_mae_d{str(d_value).replace('.', 'p')}"
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration-size learning curve for prospective prediction.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--calibration-fracs", default="0.10,0.20,0.30,0.50,0.70")
    ap.add_argument("--d-targets", default="0.05,0.10")
    ap.add_argument("--empfit-max-d", type=float, default=0.10)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--n-grid", type=int, default=240)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--predictors", default="theory,aonly,empfit,anu,zero,oracle")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    cal_fracs = parse_float_list(args.calibration_fracs)
    d_targets = parse_float_list(args.d_targets)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    paths = load_paths_from_args(args)
    print(f"Loaded {len(paths)} paths; calibration_fracs={cal_fracs}; d_targets={d_targets}")

    all_rows = []
    for i, frac in enumerate(cal_fracs):
        # Use the same base permutation schedule for each calibration fraction.
        # This makes the learning curve a paired calibration-size comparison and
        # keeps the 30% row aligned with the main prospective/d-sweep setting.
        print(f"Running calibration fraction {frac:.3f}")
        df = run_prediction_splits(
            paths=paths,
            d_grid=d_targets,
            calibration_frac=float(frac),
            n_repeats=args.n_repeats,
            seed=args.random_seed,
            empfit_max_d=args.empfit_max_d,
            n_grid=args.n_grid,
        )
        all_rows.append(df)
    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_csv(outdir / "calibration_size_paths_long.csv", index=False)

    by_group = summarize_prediction_long(
        long_df,
        group_cols=["group_key", "calibration_frac", "d"],
        predictors=predictors,
        n_boot=args.n_boot,
        seed=args.random_seed,
    )
    by_group.to_csv(outdir / "calibration_size_summary_by_group.csv", index=False)
    save_markdown(by_group, outdir / "calibration_size_summary_by_group.md")

    overall = summarize_prediction_long(
        long_df.assign(group_key="__OVERALL__"),
        group_cols=["group_key", "calibration_frac", "d"],
        predictors=predictors,
        n_boot=args.n_boot,
        seed=args.random_seed,
    )
    overall.to_csv(outdir / "calibration_size_summary_overall.csv", index=False)
    save_markdown(overall, outdir / "calibration_size_summary_overall.md")

    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "item_cols": args.item_cols,
        "group_cols": args.group_cols,
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "proxy_item_col": args.proxy_item_col,
        "proxy_score_col": args.proxy_score_col,
        "calibration_fracs": cal_fracs,
        "d_targets": d_targets,
        "empfit_max_d": args.empfit_max_d,
        "n_repeats": args.n_repeats,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "random_seed": args.random_seed,
        "split_seed_policy": "paired across calibration fractions; same base seed as the 30% prospective/d-sweep setting",
        "n_grid": args.n_grid,
        "n_boot": args.n_boot,
        "predictors": predictors,
        "n_paths": len(paths),
        "n_rows": len(long_df),
        "outputs": {
            "paths": str(outdir / "calibration_size_paths_long.csv"),
            "summary_by_group": str(outdir / "calibration_size_summary_by_group.csv"),
            "summary_overall": str(outdir / "calibration_size_summary_overall.csv"),
        },
    }
    (outdir / "calibration_size_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    for d in d_targets:
        plot_calibration_curve(overall, outdir, d, predictors=[p for p in predictors if p != "oracle"])

    print(overall[["calibration_frac", "d", "n_rows", "n_paths", "mae_theory", "mae_aonly", "mae_zero", "mae_anu"]].to_string(index=False))
    print(f"Saved calibration-size outputs in: {outdir}")


if __name__ == "__main__":
    main()
