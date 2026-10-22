#!/usr/bin/env python3
"""Independent data-layer and label-leakage checks for exp25."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_core import evaluate_models, fit_models, load_pool, validate_pool


def fixed_split_leakage_audit(frame: pd.DataFrame) -> dict[str, float | bool]:
    calibration = frame[frame["split"].eq("calibration")].copy()
    evaluation = frame[frame["split"].eq("eval")].copy()
    models = fit_models(
        calibration,
        geometry="common_temperature",
        d_values=[0.01, 0.05, 0.10],
        fit_max_d=0.05,
        seed=0,
    )
    original, _ = evaluate_models(
        calibration,
        evaluation,
        models,
        geometry="common_temperature",
        d_values=[0.01, 0.05, 0.10],
    )
    perturbed = evaluation.copy()
    perturbed["gold_score"] = (
        perturbed.groupby("prompt_id", sort=False)["gold_score"]
        .transform(lambda values: values.iloc[::-1].to_numpy())
        .to_numpy()
    )
    corrupted, _ = evaluate_models(
        calibration,
        perturbed,
        models,
        geometry="common_temperature",
        d_values=[0.01, 0.05, 0.10],
    )
    merged = original.merge(
        corrupted,
        on=["d", "baseline"],
        suffixes=("_original", "_corrupted"),
        validate="one_to_one",
    )
    prediction_drift = float(
        np.max(
            np.abs(
                merged["predicted_gain_original"]
                - merged["predicted_gain_corrupted"]
            )
        )
    )
    actual_change = float(
        np.max(
            np.abs(
                merged["actual_gain_original"]
                - merged["actual_gain_corrupted"]
            )
        )
    )
    return {
        "prediction_drift_after_eval_gold_permutation": prediction_drift,
        "actual_gain_change_after_eval_gold_permutation": actual_change,
        "prediction_invariance_pass": prediction_drift < 1e-12,
        "perturbation_nontrivial_pass": actual_change > 1e-4,
    }


def old_identity_checks(root: Path, frame: pd.DataFrame, zephyr: pd.DataFrame) -> dict:
    calibration = frame[frame["split"].eq("calibration")]
    evaluation = frame[frame["split"].eq("eval")]
    models = fit_models(
        calibration,
        geometry="common_temperature",
        d_values=[0.01, 0.05, 0.10],
        fit_max_d=0.05,
        seed=0,
    )
    current, _ = evaluate_models(
        calibration,
        evaluation,
        models,
        geometry="common_temperature",
        d_values=[0.01, 0.05, 0.10],
    )
    previous = pd.read_csv(
        root
        / "outputs"
        / "exp05_real_llm_20260706_032457"
        / "gain_curves"
        / "llm_gain_curve_predictions.csv"
    )
    merged = previous.merge(
        current, on=["d", "baseline"], suffixes=("_old", "_new")
    )
    exact_baselines = ["A+B", "A-only", "A_nu", "zero"]
    exact = merged[merged["baseline"].isin(exact_baselines)]
    exp05_prediction_drift = float(
        np.max(
            np.abs(
                exact["predicted_gain_old"] - exact["predicted_gain_new"]
            )
        )
    )

    z_cal = zephyr[zephyr["split"].eq("calibration")]
    z_eval = zephyr[zephyr["split"].eq("eval")]
    z_models = fit_models(
        z_cal,
        geometry="fixed_prompt_kl",
        d_values=[0.10, 0.20, 0.30, 0.50],
        fit_max_d=0.10,
        seed=0,
    )
    z_current, _ = evaluate_models(
        z_cal,
        z_eval,
        z_models,
        geometry="fixed_prompt_kl",
        d_values=[0.10, 0.20, 0.30, 0.50],
    )
    z_previous = pd.read_csv(
        root
        / "outputs"
        / "exp11_cross_model_20260710_071053"
        / "cross_model_per_d.csv"
    )
    z_ab = z_current[z_current["baseline"].eq("A+B")]
    z_merged = z_previous.merge(z_ab, on="d")
    exp11_prediction_drift = float(
        np.max(
            np.abs(z_merged["within_AB"] - z_merged["predicted_gain"])
        )
    )
    exp11_frontier_drift = float(
        np.max(
            np.abs(z_merged["frontier"] - z_merged["actual_gain"])
        )
    )
    return {
        "exp05_exact_baseline_prediction_drift": exp05_prediction_drift,
        "exp05_identity_pass": exp05_prediction_drift < 1e-12,
        "exp11_A+B_prediction_drift": exp11_prediction_drift,
        "exp11_frontier_drift": exp11_frontier_drift,
        "exp11_identity_pass": max(
            exp11_prediction_drift, exp11_frontier_drift
        )
        < 1e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--within-outdir", type=Path, required=True)
    parser.add_argument("--transfer-outdir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    exp05_path = (
        args.repo_root
        / "outputs"
        / "exp05_real_llm_20260706_032457"
        / "candidates_scored.csv"
    )
    qwen_path = (
        args.repo_root
        / "outputs"
        / "exp09_dpo_v4_20260710_051319"
        / "pool_scored.jsonl"
    )
    zephyr_path = (
        args.repo_root
        / "outputs"
        / "exp11_cross_model_20260710_071053"
        / "pool_scored.jsonl"
    )
    exp05 = load_pool(exp05_path)
    qwen = load_pool(qwen_path)
    zephyr = load_pool(zephyr_path)
    audits = {
        "exp05_pool": validate_pool(
            exp05, name="exp05", candidate_col="candidate_id"
        ),
        "qwen_pool": validate_pool(
            qwen, name="qwen", candidate_col="sample_idx"
        ),
        "zephyr_pool": validate_pool(
            zephyr, name="zephyr", candidate_col="sample_idx"
        ),
        "label_leakage": fixed_split_leakage_audit(exp05),
        "legacy_identity": old_identity_checks(
            args.repo_root, exp05, zephyr
        ),
    }
    for split in ("calibration", "eval"):
        q_ids = set(qwen.loc[qwen["split"].eq(split), "prompt_id"])
        z_ids = set(zephyr.loc[zephyr["split"].eq(split), "prompt_id"])
        if q_ids != z_ids:
            raise AssertionError(f"{split} Qwen/Zephyr prompt IDs differ")
    if set(qwen.loc[qwen["split"].eq("calibration"), "prompt_id"]) & set(
        qwen.loc[qwen["split"].eq("eval"), "prompt_id"]
    ):
        raise AssertionError("Qwen calibration/eval prompts overlap")
    if set(zephyr.loc[zephyr["split"].eq("calibration"), "prompt_id"]) & set(
        zephyr.loc[zephyr["split"].eq("eval"), "prompt_id"]
    ):
        raise AssertionError("Zephyr calibration/eval prompts overlap")
    audits["split_integrity"] = {
        "qwen_zephyr_prompt_ids_match": True,
        "calibration_eval_prompt_overlap": 0,
    }

    within_spec = json.loads(
        (args.within_outdir / "within_run_spec.json").read_text(
            encoding="utf-8"
        )
    )
    transfer_spec = json.loads(
        (args.transfer_outdir / "transfer_run_spec.json").read_text(
            encoding="utf-8"
        )
    )
    expected_within = (
        within_spec["n_repeats"]
        * len(within_spec["calibration_fractions"])
        * len(within_spec["d_values"])
        * len(within_spec["baselines"])
    )
    within = pd.read_csv(args.within_outdir / "within_predictions.csv")
    transfer = pd.read_csv(args.transfer_outdir / "transfer_predictions.csv")
    audits["output_integrity"] = {
        "within_rows": int(len(within)),
        "within_expected_rows": expected_within,
        "within_count_pass": len(within) == expected_within,
        "transfer_rows": int(len(transfer)),
        "transfer_expected_rows": (
            len(transfer_spec["routes"])
            * len(transfer_spec["estimation_scopes"])
            * len(transfer_spec["d_values"])
            * len(transfer_spec["baselines"])
        ),
        "transfer_count_pass": len(transfer)
        == (
            len(transfer_spec["routes"])
            * len(transfer_spec["estimation_scopes"])
            * len(transfer_spec["d_values"])
            * len(transfer_spec["baselines"])
        ),
    }
    pass_flags: list[bool] = []
    for section in audits.values():
        if isinstance(section, dict):
            pass_flags.extend(
                bool(value)
                for key, value in section.items()
                if key.endswith("_pass")
            )
    audits["all_explicit_checks_pass"] = all(pass_flags)
    if not audits["all_explicit_checks_pass"]:
        raise AssertionError(json.dumps(audits, indent=2))
    args.out_json.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
