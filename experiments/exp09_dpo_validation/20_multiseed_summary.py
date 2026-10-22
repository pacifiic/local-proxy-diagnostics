#!/usr/bin/env python3
"""Aggregate seed-level DPO realized-gain MAE as mean and sample SD."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASELINES = {
    "A+B": "pred_AB",
    "A-only": "pred_Aonly",
    "A_nu": "pred_Anu",
    "Pearson": "pred_pearson",
    "Spearman": "pred_spearman",
    "Isotonic": "pred_isotonic",
    "Zero": None,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    spline = pd.read_csv(args.root / "spline" / "spline_summary.csv")
    spline_by_source = dict(
        zip(spline["source"], spline["spline_mae_vs_dpo_realized"])
    )
    rows = []
    for seed in args.seeds:
        source = f"seed_{seed}"
        path = args.root / source / "exp09_final_summary.csv"
        data = pd.read_csv(path)
        local = data[
            (data["d_med"] <= 1.0) & (data["ess_med"] >= 8.0)
        ].copy()
        if len(local) != 6:
            raise ValueError(f"{source}: expected six local checkpoints")
        target = local["dpo_reweight"].to_numpy()
        for baseline, column in BASELINES.items():
            prediction = (
                np.zeros(len(local))
                if column is None
                else local[column].to_numpy()
            )
            rows.append(
                {
                    "seed": seed,
                    "baseline": baseline,
                    "mae_vs_dpo_realized": float(
                        np.mean(np.abs(prediction - target))
                    ),
                    "d_min": float(local["d_med"].min()),
                    "d_max": float(local["d_med"].max()),
                }
            )
        rows.append(
            {
                "seed": seed,
                "baseline": "Spline",
                "mae_vs_dpo_realized": float(spline_by_source[source]),
                "d_min": float(local["d_med"].min()),
                "d_max": float(local["d_med"].max()),
            }
        )

    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(args.root / "seed_level_mae.csv", index=False)
    summary = (
        seed_level.groupby("baseline", as_index=False)
        .agg(
            mean_mae=("mae_vs_dpo_realized", "mean"),
            sd_mae=("mae_vs_dpo_realized", "std"),
            min_mae=("mae_vs_dpo_realized", "min"),
            max_mae=("mae_vs_dpo_realized", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_mae")
    )
    summary.to_csv(args.root / "multiseed_mean_sd.csv", index=False)
    print("Seed-level MAE:")
    print(
        seed_level.pivot(
            index="baseline", columns="seed", values="mae_vs_dpo_realized"
        ).to_string()
    )
    print("\nMean +/- sample SD:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
