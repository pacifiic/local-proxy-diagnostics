#!/usr/bin/env python3
"""Fail-fast integrity and arithmetic audit for exp26."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_KEY = [
    "path_id",
    "repeat",
    "calibration_frac",
    "gold_rater_count",
    "noise_family",
    "noise_scale_sd",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.outdir / "b_rater_noise_rows.csv")
    spec = json.loads((args.outdir / "run_spec.json").read_text())

    require(not rows.empty, "raw result is empty")
    require(not rows.duplicated(PRIMARY_KEY).any(), "duplicate primary keys")
    nullable_endpoint = {
        "actual_gain_d0p1",
        "abs_error_ab_d0p1",
        "abs_error_aonly_d0p1",
    }
    numeric = rows.select_dtypes(include=[np.number])
    always_finite = numeric.drop(columns=list(nullable_endpoint))
    require(
        np.isfinite(always_finite.to_numpy()).all(),
        "non-finite non-endpoint numeric result",
    )
    endpoint = rows["d0p1_reachable"].astype(bool)
    require(
        np.isfinite(rows.loc[endpoint, list(nullable_endpoint)].to_numpy()).all(),
        "reachable d=0.10 endpoint has non-finite values",
    )
    require(
        rows.loc[~endpoint, list(nullable_endpoint)].isna().all().all(),
        "unreachable d=0.10 endpoint was not marked missing",
    )
    require(rows["path_id"].nunique() == 32, "expected 32 paths")
    require(set(rows["calibration_frac"]) == {0.1, 0.2, 0.3, 0.5}, "bad fractions")
    require(
        set(rows["noise_family"]) == {"clean", "gaussian", "student_t3"},
        "bad noise families",
    )
    require(
        rows.loc[rows["noise_family"].eq("clean"), "noise_scale_sd"].eq(0).all(),
        "clean rows must have zero noise scale",
    )
    require(
        rows.loc[~rows["noise_family"].eq("clean"), "noise_scale_sd"]
        .isin([0.25, 0.5, 1.0])
        .all(),
        "unexpected nonzero noise scale",
    )

    expected_per_repeat_fraction = 0
    for rater_type, n_paths, max_gold in (
        ("expert", 12, 2),
        ("turker", 20, 4),
    ):
        subset = rows[rows["rater_type"].eq(rater_type)]
        require(subset["path_id"].nunique() == n_paths, f"bad {rater_type} paths")
        require(
            subset["max_gold_raters"].eq(max_gold).all(),
            f"bad {rater_type} gold-rater maximum",
        )
        expected_per_repeat_fraction += n_paths * max_gold * 7
    expected_rows = (
        expected_per_repeat_fraction
        * int(spec["n_repeats"])
        * len(spec["calibration_fractions"])
    )
    require(len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}")
    require(int(spec["n_rows"]) == len(rows), "run-spec row count mismatch")
    reachability = pd.read_csv(args.outdir / "reachability_accounting.csv")
    require(
        not reachability.duplicated(
            ["path_id", "repeat", "calibration_frac"]
        ).any(),
        "duplicate reachability-accounting key",
    )
    require(
        len(reachability)
        == 32 * int(spec["n_repeats"]) * len(spec["calibration_fractions"]),
        "reachability-accounting row count mismatch",
    )
    require(
        int(spec["n_d0p1_unreachable_path_repeat_fractions"])
        == int((~reachability["d0p1_reachable"].astype(bool)).sum()),
        "run-spec reachability count mismatch",
    )

    clean = rows[rows["noise_family"].eq("clean")]
    require(
        np.allclose(clean["a_shift_from_same_k_clean"], 0.0, atol=1e-12),
        "clean A shift is not zero",
    )
    require(
        np.allclose(clean["b_shift_from_same_k_clean"], 0.0, atol=1e-12),
        "clean B shift is not zero",
    )
    require(
        np.allclose(
            rows["ab_minus_aonly_curve_mae"],
            rows["curve_mae_ab"] - rows["curve_mae_aonly"],
            atol=1e-12,
        ),
        "paired CurveMAE arithmetic mismatch",
    )
    require(
        np.allclose(
            rows["pred_ab_d0p1"] - rows["pred_aonly_d0p1"],
            rows["b_cal"] * 0.1,
            atol=1e-12,
        ),
        "Bd endpoint identity failed",
    )
    require(
        (
            rows["decision_disagree_d0p1"]
            == (rows["proceed_ab_d0p1"] != rows["proceed_aonly_d0p1"])
        ).all(),
        "decision-disagreement identity failed",
    )

    for calibration_frac, group in rows.groupby("calibration_frac"):
        expected_n_cal = int(round(1600 * calibration_frac))
        require(
            group["n_cal_items"].eq(expected_n_cal).all(),
            f"bad n_cal for {calibration_frac}",
        )
        require(
            group["n_eval_items"].eq(1600 - expected_n_cal).all(),
            f"bad n_eval for {calibration_frac}",
        )
    require(
        rows.groupby(["path_id", "repeat", "calibration_frac"])["split_sha256"]
        .nunique()
        .eq(1)
        .all(),
        "conditions within a paired comparison use different item splits",
    )

    summary = pd.read_csv(args.outdir / "condition_summary.csv")
    require(not summary.empty, "condition summary is empty")
    require(
        (args.outdir / "fig_b_rater_noise_stability.pdf").exists(),
        "missing PDF figure",
    )
    require(
        (args.outdir / "fig_b_rater_noise_stability.png").exists(),
        "missing PNG figure",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "paths": int(rows["path_id"].nunique()),
                "repeats": int(spec["n_repeats"]),
                "conditions": int(
                    rows[
                        [
                            "calibration_frac",
                            "gold_rater_count",
                            "noise_family",
                            "noise_scale_sd",
                            "rater_type",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
                "d0p1_unreachable_path_repeat_fractions": int(
                    spec["n_d0p1_unreachable_path_repeat_fractions"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
