#!/usr/bin/env python3
"""Independent integrity audit for exp29 outputs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "exp25_real_llm_calibration_followup"
sys.path.insert(0, str(BASELINE_DIR))

from baseline_core import (  # noqa: E402
    _central_moments,
    _coefficient_from_moments,
    _find_t_prompt,
    sha256_file,
    tilt_weights,
)


FIT_RADII = np.linspace(0.004, 0.10, 25)
ANALYTIC_BASELINES = {
    "A+B",
    "A+B-shrunk",
    "A-only",
    "A_nu",
    "empirical_curve",
    "empirical_fit",
    "zero",
}


def gain_at_radius(proxy: np.ndarray, values: np.ndarray, d: float) -> float:
    weights = tilt_weights(proxy, _find_t_prompt(proxy, d))
    return float(np.sum(weights * values) - np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {args.report}")

    spec = json.loads((args.input_dir / "run_spec.json").read_text())
    manifest = pd.read_csv(args.input_dir / "candidate_split_manifest.csv.gz")
    aggregate = pd.read_csv(args.input_dir / "aggregate_predictions.csv")
    prompt = pd.read_csv(args.input_dir / "prompt_predictions.csv.gz")
    coefficients = pd.read_csv(args.input_dir / "prompt_coefficients.csv.gz")
    metadata = pd.read_csv(args.input_dir / "fit_metadata.csv")

    split_counts = (
        manifest.groupby(["repeat", "prompt_id", "role"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    split_count_pass = bool(
        split_counts["calibration"]
        .eq(spec["n_calibration_candidates_per_prompt"])
        .all()
        and split_counts["evaluation"]
        .eq(spec["n_evaluation_candidates_per_prompt"])
        .all()
    )
    overlap_max = 0
    for _, group in manifest.groupby("repeat", sort=True):
        calibration_keys = set(
            zip(
                group.loc[group["role"].eq("calibration"), "prompt_id"].astype(str),
                group.loc[group["role"].eq("calibration"), "sample_idx"].astype(int),
            )
        )
        evaluation_keys = set(
            zip(
                group.loc[group["role"].eq("evaluation"), "prompt_id"].astype(str),
                group.loc[group["role"].eq("evaluation"), "sample_idx"].astype(int),
            )
        )
        overlap_max = max(overlap_max, len(calibration_keys & evaluation_keys))

    expected_baselines = sorted(spec["baselines"])
    baseline_pass = sorted(aggregate["baseline"].unique()) == expected_baselines
    finite_pass = bool(
        np.isfinite(
            aggregate[
                [
                    "predicted_gain",
                    "actual_gain",
                    "error",
                    "abs_error",
                    "prompt_mae",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            prompt[
                ["predicted_gain", "actual_gain", "error", "abs_error"]
            ].to_numpy(dtype=float)
        ).all()
    )

    actual_spread = (
        prompt.groupby(["run", "repeat", "prompt_id", "d"], sort=True)[
            "actual_gain"
        ]
        .agg(["min", "max"])
        .reset_index()
    )
    actual_frozen_pass = bool(
        np.max(np.abs(actual_spread["max"] - actual_spread["min"])) <= 1e-14
    )
    recalculated = (
        prompt.groupby(["run", "repeat", "d", "baseline"], sort=True)
        .agg(
            predicted_gain=("predicted_gain", "mean"),
            actual_gain=("actual_gain", "mean"),
            prompt_mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    aggregate_check = aggregate.merge(
        recalculated,
        on=["run", "repeat", "d", "baseline"],
        suffixes=("_saved", "_recalculated"),
        validate="one_to_one",
    )
    aggregate_max_difference = float(
        max(
            np.max(
                np.abs(
                    aggregate_check["predicted_gain_saved"]
                    - aggregate_check["predicted_gain_recalculated"]
                )
            ),
            np.max(
                np.abs(
                    aggregate_check["actual_gain_saved"]
                    - aggregate_check["actual_gain_recalculated"]
                )
            ),
            np.max(
                np.abs(
                    aggregate_check["prompt_mae_saved"]
                    - aggregate_check["prompt_mae_recalculated"]
                )
            ),
        )
    )

    raw_actual = (
        prompt.loc[
            prompt["baseline"].eq("A+B"),
            ["run", "repeat", "prompt_id", "d", "actual_gain"],
        ]
        .drop_duplicates()
        .sort_values(["run", "repeat", "prompt_id", "d"])
    )
    raw_max_difference = 0.0
    coefficient_max_difference = 0.0
    analytic_prediction_max_difference = 0.0
    n_empirical_fit_prompts_recomputed = 0
    error_identity_max_difference = float(
        max(
            np.max(np.abs(prompt["error"] - (prompt["predicted_gain"] - prompt["actual_gain"]))),
            np.max(np.abs(prompt["abs_error"] - prompt["error"].abs())),
        )
    )
    input_hash_pass = True
    input_key_pass = True
    coefficient_groups = {
        (str(run), int(repeat)): group
        for (run, repeat), group in coefficients.groupby(
            ["run", "repeat"], sort=True
        )
    }
    metadata_lookup = {
        (str(row.run), int(row.repeat)): row
        for row in metadata.itertuples(index=False)
    }
    analytic_groups = {
        (str(run), int(repeat)): group
        for (run, repeat), group in prompt[
            prompt["baseline"].isin(ANALYTIC_BASELINES)
        ].groupby(["run", "repeat"], sort=True)
    }
    for pool_record in spec["pools"]:
        run_name = pool_record["run"]
        path = Path(pool_record["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        input_hash_pass = input_hash_pass and (
            sha256_file(path) == pool_record["sha256"]
        )
        raw = pd.read_json(path, lines=True)[
            ["prompt_id", "sample_idx", "proxy_score", "gold_score"]
        ]
        raw["prompt_id"] = raw["prompt_id"].astype(str)
        run_actual = raw_actual[raw_actual["run"].eq(run_name)]
        for repeat, repeat_actual in run_actual.groupby("repeat", sort=True):
            manifest_repeat = manifest[manifest["repeat"].eq(int(repeat))]
            roles = manifest_repeat[
                manifest_repeat["role"].eq("evaluation")
            ][["prompt_id", "sample_idx"]]
            evaluation = raw.merge(
                roles,
                on=["prompt_id", "sample_idx"],
                how="inner",
                validate="one_to_one",
            )
            expected_keys = len(roles)
            input_key_pass = input_key_pass and len(evaluation) == expected_keys
            by_prompt = {
                str(prompt_id): group
                for prompt_id, group in evaluation.groupby("prompt_id", sort=True)
            }
            for row in repeat_actual.itertuples(index=False):
                group = by_prompt[str(row.prompt_id)]
                proxy = group["proxy_score"].to_numpy(dtype=float)
                gold = group["gold_score"].to_numpy(dtype=float)
                weights = tilt_weights(
                    proxy, _find_t_prompt(proxy, float(row.d))
                )
                recomputed = float(
                    np.sum(weights * gold) - np.mean(gold)
                )
                raw_max_difference = max(
                    raw_max_difference, abs(recomputed - float(row.actual_gain))
                )

            valid_coefficients = coefficient_groups[(run_name, int(repeat))]
            valid_coefficients = valid_coefficients[
                valid_coefficients["status"].eq("valid")
            ].copy()
            coefficient_by_prompt = {
                str(row.prompt_id): row
                for row in valid_coefficients.itertuples(index=False)
            }
            repeat_metadata = metadata_lookup[(run_name, int(repeat))]
            tau2 = float(repeat_metadata.b_prior_tau2)
            calibration_roles = manifest_repeat[
                manifest_repeat["role"].eq("calibration")
            ][["prompt_id", "sample_idx"]]
            calibration = raw.merge(
                calibration_roles,
                on=["prompt_id", "sample_idx"],
                how="inner",
                validate="one_to_one",
            )
            calibration_by_prompt = {
                str(prompt_id): group
                for prompt_id, group in calibration.groupby("prompt_id", sort=True)
            }
            saved_analytic = analytic_groups[(run_name, int(repeat))]
            saved_lookup = {
                (str(row.prompt_id), float(row.d), str(row.baseline)): float(
                    row.predicted_gain
                )
                for row in saved_analytic.itertuples(index=False)
            }
            empirical_fit_audit_prompt = min(coefficient_by_prompt)
            for prompt_id, coefficient in coefficient_by_prompt.items():
                cal_group = calibration_by_prompt[prompt_id]
                cal_proxy = cal_group["proxy_score"].to_numpy(dtype=float)
                cal_gold = cal_group["gold_score"].to_numpy(dtype=float)
                a, b, a_nu = _coefficient_from_moments(
                    *_central_moments(cal_gold, cal_proxy)
                )
                coefficient_max_difference = max(
                    coefficient_max_difference,
                    abs(a - float(coefficient.a)),
                    abs(b - float(coefficient.b)),
                    abs(a_nu - float(coefficient.a_nu)),
                )
                empirical_a = empirical_b = math.nan
                if prompt_id == empirical_fit_audit_prompt:
                    fit_gains = np.asarray(
                        [
                            gain_at_radius(cal_proxy, cal_gold, float(d))
                            for d in FIT_RADII
                        ],
                        dtype=float,
                    )
                    design = np.column_stack([np.sqrt(FIT_RADII), FIT_RADII])
                    empirical_a, empirical_b = np.linalg.lstsq(
                        design, fit_gains, rcond=None
                    )[0]
                    n_empirical_fit_prompts_recomputed += 1
                b_se = float(coefficient.b_bootstrap_se)
                shrink_weight = (
                    tau2 / (tau2 + b_se**2)
                    if np.isfinite(b_se) and tau2 > 0
                    else 0.0
                )
                for d in spec["d_values"]:
                    d = float(d)
                    expected = {
                        "A+B": a * np.sqrt(d) + b * d,
                        "A+B-shrunk": a * np.sqrt(d) + shrink_weight * b * d,
                        "A-only": a * np.sqrt(d),
                        "A_nu": a_nu * np.sqrt(d),
                        "empirical_curve": gain_at_radius(
                            cal_proxy, cal_gold, d
                        ),
                        "zero": 0.0,
                    }
                    if prompt_id == empirical_fit_audit_prompt:
                        expected["empirical_fit"] = (
                            empirical_a * np.sqrt(d) + empirical_b * d
                        )
                    for baseline, prediction in expected.items():
                        key = (prompt_id, d, baseline)
                        if key not in saved_lookup:
                            raise ValueError(
                                f"missing saved analytic prediction: "
                                f"{run_name}/{repeat}/{key}"
                            )
                        analytic_prediction_max_difference = max(
                            analytic_prediction_max_difference,
                            abs(prediction - saved_lookup[key]),
                        )

    checks = {
        "split_counts_exact": bool(split_count_pass),
        "calibration_evaluation_overlap_zero": bool(overlap_max == 0),
        "baseline_set_exact": bool(baseline_pass),
        "all_predictions_finite": bool(finite_pass),
        "actual_gain_identical_across_baselines": bool(actual_frozen_pass),
        "aggregate_recomputation_pass": bool(
            aggregate_max_difference <= 1e-12
        ),
        "raw_actual_gain_recomputation_pass": bool(
            raw_max_difference <= 1e-12
        ),
        "coefficient_recomputation_pass": bool(
            coefficient_max_difference <= 1e-12
        ),
        "analytic_prediction_recomputation_pass": (
            bool(analytic_prediction_max_difference <= 1e-12)
        ),
        "error_column_identity_pass": bool(
            error_identity_max_difference <= 1e-12
        ),
        "input_sha256_pass": bool(input_hash_pass),
        "input_candidate_key_pass": bool(input_key_pass),
        "recorded_leakage_audit_pass": bool(spec["leakage_audit_pass"]),
    }
    report = {
        "experiment": "exp29_same_prompt_candidate_holdout_integrity_audit",
        "pass": bool(all(checks.values())),
        "checks": checks,
        "max_candidate_overlap": int(overlap_max),
        "aggregate_max_abs_difference": aggregate_max_difference,
        "raw_actual_gain_max_abs_difference": raw_max_difference,
        "coefficient_max_abs_difference": coefficient_max_difference,
        "analytic_prediction_max_abs_difference": (
            analytic_prediction_max_difference
        ),
        "error_identity_max_abs_difference": error_identity_max_difference,
        "n_aggregate_rows": int(len(aggregate)),
        "n_prompt_prediction_rows": int(len(prompt)),
        "n_raw_actual_cells_recomputed": int(len(raw_actual)),
        "n_empirical_fit_prompts_recomputed": int(
            n_empirical_fit_prompts_recomputed
        ),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["pass"]:
        raise RuntimeError(f"exp29 integrity audit failed: {checks}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
