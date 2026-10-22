#!/usr/bin/env python3
"""Run the paper's WMT direct-validation logic from a prepared long CSV."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT
sys.path.insert(0, str(PAPER / "lib"))

from wmt_mtbench_utils import (  # noqa: E402
    build_heldout_paths_from_long,
    compute_ab,
    fit_local_ab,
    icc1_from_long,
    tilt_curve,
)

DIRECT_PATH = (
    PAPER
    / "experiments"
    / "05_wmt_mtbench"
    / "01_wmt_mqm_direct_tilt.py"
)
RESIDUAL_PATH = (
    PAPER
    / "experiments"
    / "03_direct_tilt_validation"
    / "02_residual_metrics.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = load_module("paper_wmt_direct", DIRECT_PATH)
residual = load_module("paper_wmt_residual", RESIDUAL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fit-max-d", type=float, default=0.10)
    parser.add_argument("--min-items", type=int, default=200)
    parser.add_argument("--min-other-raters", type=int, default=2)
    parser.add_argument("--t-max", type=float, default=4.0)
    parser.add_argument("--t-n", type=int, default=400)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    long_df = pd.read_csv(args.long_csv)
    required = {"dataset_id", "item_id", "rater", "score"}
    missing = required.difference(long_df.columns)
    if missing:
        raise ValueError(f"prepared WMT input missing columns: {sorted(missing)}")
    long_df.to_csv(args.outdir / "wmt_mqm_long.csv", index=False)

    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.t_n)])
    path_rows = []
    traj_rows = []
    dataset_rows = []
    for dataset_id in sorted(long_df["dataset_id"].unique()):
        sub = long_df[long_df["dataset_id"] == dataset_id].copy()
        icc1, icc_n, icc_k = icc1_from_long(
            sub, item_col="item_id", score_col="score"
        )
        paths = build_heldout_paths_from_long(
            sub,
            dataset_id,
            gold_estimator="loo_mean",
            score_transform="raw",
            min_items=args.min_items,
            min_other_raters=args.min_other_raters,
        )
        for path in paths:
            gold = path["J"]
            proxy = path["r"]
            coeff = compute_ab(gold, proxy)
            d_values, gains = tilt_curve(gold, proxy, t_grid)
            a_fit, b_fit, r2 = fit_local_ab(
                d_values, gains, max_d=args.fit_max_d
            )
            path_rows.append(
                {
                    "dataset_id": dataset_id,
                    "heldout_rater": path["heldout_rater"],
                    "n_items": path["n_items"],
                    "A_theory": coeff["A"],
                    "B_theory": coeff["B"],
                    "A_fit": a_fit,
                    "B_fit": b_fit,
                    "r2": r2,
                    "rho": coeff["rho"],
                    "sigma_J_sq": coeff["sigma_J_sq"],
                    "sigma_r_sq": coeff["sigma_r_sq"],
                    "icc1": icc1,
                    "icc1_items_used": icc_n,
                    "icc1_k": icc_k,
                    "fit_max_d": args.fit_max_d,
                    "score_transform": "raw",
                    "gold_estimator": "loo_mean",
                }
            )
            for temperature, radius, gain in zip(
                t_grid, d_values, gains, strict=False
            ):
                traj_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "heldout_rater": path["heldout_rater"],
                        "t": float(temperature),
                        "d": float(radius),
                        "q": float(gain),
                    }
                )
        if paths:
            local = pd.DataFrame(
                row for row in path_rows if row["dataset_id"] == dataset_id
            )
            dataset_rows.append(direct.summarize_dataset(local))

    paths_df = pd.DataFrame(path_rows)
    traj_df = pd.DataFrame(traj_rows)
    datasets_df = pd.DataFrame(dataset_rows)
    summary_df = pd.DataFrame(
        [
            {
                "n_datasets": int(datasets_df.shape[0]),
                "datasets_with_paths": int((datasets_df["n_paths"] > 0).sum()),
                "n_paths_total": int(paths_df.shape[0]),
                "corr_A_pooled": float(
                    np.corrcoef(paths_df["A_theory"], paths_df["A_fit"])[0, 1]
                ),
                "corr_B_pooled": float(
                    np.corrcoef(paths_df["B_theory"], paths_df["B_fit"])[0, 1]
                ),
                "sign_agreement_total": int(
                    (np.sign(paths_df["B_theory"]) == np.sign(paths_df["B_fit"])).sum()
                ),
                "frac_B_theory_neg_pooled": float(
                    (paths_df["B_theory"] < 0).mean()
                ),
                "frac_B_fit_neg_pooled": float(
                    (paths_df["B_fit"] < 0).mean()
                ),
                "median_r2_pooled": float(paths_df["r2"].median()),
            }
        ]
    )
    paths_df.to_csv(args.outdir / "wmt_mqm_paths.csv", index=False)
    traj_df.to_csv(args.outdir / "wmt_mqm_traj.csv", index=False)
    datasets_df.to_csv(args.outdir / "wmt_mqm_by_dataset.csv", index=False)
    summary_df.to_csv(args.outdir / "wmt_mqm_summary.csv", index=False)

    residual_rows = residual.analyze_traj(
        traj_df,
        "WMT MQM",
        ["dataset_id", "heldout_rater"],
        fit_max_d=args.fit_max_d,
    )
    pd.DataFrame(residual_rows).to_csv(
        args.outdir / "direct_validation_residual_metrics.csv", index=False
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
