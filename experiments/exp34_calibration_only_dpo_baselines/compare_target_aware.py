#!/usr/bin/env python3
"""Compare exp34 with the earlier target-pool-aware DPO calibrators."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = (
    "cal_raw_isotonic",
    "cal_rank_isotonic",
    "cal_raw_spline",
    "cal_rank_spline",
    "cal_raw_binned",
    "cal_rank_binned",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--target-aware-primary", type=Path, required=True
    )
    parser.add_argument(
        "--target-aware-binned", type=Path, required=True
    )
    args = parser.parse_args()

    fair = pd.read_csv(args.outdir / "checkpoint_predictions.csv")
    target = pd.concat(
        [
            pd.read_csv(args.target_aware_primary),
            pd.read_csv(args.target_aware_binned),
        ],
        ignore_index=True,
    )
    rows = []
    for scope, max_d in (
        ("strict_local_d_le_0p10", 0.10),
        ("historical_six_checkpoints", np.inf),
    ):
        for model in MODELS:
            left = fair[(fair["model"] == model) & (fair["d_med"] <= max_d)]
            right = target[
                (target["model"] == model) & (target["d_med"] <= max_d)
            ]
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "calibration_only_frontier_mae": left[
                        "abs_error_frontier"
                    ].mean(),
                    "target_pool_aware_frontier_mae": right[
                        "abs_error_frontier"
                    ].mean(),
                    "calibration_only_dpo_mae": left["abs_error_dpo"].mean(),
                    "target_pool_aware_dpo_mae": right[
                        "abs_error_dpo"
                    ].mean(),
                    "n_calibration_only_checkpoints": len(left),
                    "n_target_pool_aware_checkpoints": len(right),
                }
            )
    result = pd.DataFrame(rows)
    if result.isna().any().any():
        raise ValueError("missing model/scope in information-setting comparison")
    result.to_csv(
        args.outdir / "information_setting_comparison.csv", index=False
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
