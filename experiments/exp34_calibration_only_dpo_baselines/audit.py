#!/usr/bin/env python3
"""Integrity, information-boundary, and interpolation audits for exp34."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp34_build", HERE / "build.py")
build = importlib.util.module_from_spec(SPEC)
sys.modules["exp34_build"] = build
SPEC.loader.exec_module(build)


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def exact_mean_gain(
    proxy: np.ndarray, fitted: np.ndarray, d: float
) -> float:
    weights, _ = build.weights_at_d(proxy, d)
    centered = fitted - np.mean(fitted, axis=1, keepdims=True)
    return float(np.mean(np.sum(weights * centered, axis=1)))


def load_full_pool(path: Path) -> tuple[set[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    calibration_ids: set[str] = set()
    evaluation: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt_id = str(row["prompt_id"])
            if row["split"] == "calibration":
                calibration_ids.add(prompt_id)
            elif row["split"] == "eval":
                evaluation.setdefault(prompt_id, []).append(row)
    if len(calibration_ids) != 100 or len(evaluation) != 200:
        raise AssertionError("canonical pool split count mismatch")
    if calibration_ids & set(evaluation):
        raise AssertionError("calibration/evaluation prompt overlap")
    arrays = {}
    for prompt_id, rows in evaluation.items():
        rows = sorted(rows, key=lambda row: int(row["sample_idx"]))
        if len(rows) != 32:
            raise AssertionError(f"{prompt_id}: evaluation pool size mismatch")
        arrays[prompt_id] = (
            np.asarray([float(row["proxy_score"]) for row in rows]),
            np.asarray([float(row["gold_score"]) for row in rows]),
        )
    return calibration_ids, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-scored", type=Path, required=True)
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    per = pd.read_csv(args.outdir / "per_prompt_predictions.csv")
    checkpoint = pd.read_csv(args.outdir / "checkpoint_predictions.csv")
    curves = pd.read_csv(args.outdir / "frozen_calibration_curves.csv")
    metadata = pd.read_csv(args.outdir / "model_fit_metadata.csv")
    prompt_ids, proxy, gold = build.load_calibration(args.pool_scored)
    calibration_ids, evaluation = load_full_pool(args.pool_scored)

    key = ["seed", "tag", "prompt_id", "model"]
    if per.duplicated(key).any():
        raise AssertionError("duplicate prediction rows")
    if not (checkpoint["n_prompts"] == 200).all():
        raise AssertionError("checkpoint missing evaluation prompts")
    identity_frontier = float(
        np.max(
            np.abs(
                per["abs_error_frontier"]
                - np.abs(per["prediction"] - per["frontier"])
            )
        )
    )
    identity_dpo = float(
        np.max(
            np.abs(
                per["abs_error_dpo"]
                - np.abs(per["prediction"] - per["dpo_realized"])
            )
        )
    )

    # Fixed predictions must be exactly the canonical exp09 columns.
    source = build.load_dpo_rows(args.multiseed_root, [0, 1, 2])
    eval_id_match = {}
    for seed, frame in source.groupby("seed"):
        source_ids = set(frame["prompt_id"])
        eval_id_match[str(seed)] = {
            "missing_from_dpo": sorted(set(evaluation) - source_ids),
            "extra_in_dpo": sorted(source_ids - set(evaluation)),
        }

    # Re-derive every saved ideal frontier from canonical candidate scores.
    frontier_drift = []
    for row in source.itertuples():
        r_eval, j_eval = evaluation[row.prompt_id]
        weights, _ = build.weights_at_d(r_eval[None, :], float(row.d))
        gain = float(np.sum(weights[0] * j_eval) - np.mean(j_eval))
        frontier_drift.append(abs(gain - float(row.frontier)))
    max_frontier_drift = float(max(frontier_drift))

    canonical_pool_hash = build.sha256_file(args.pool_scored)
    seed_pool_hashes = {}
    for seed in (0, 1, 2):
        seed_pool = args.multiseed_root / f"seed_{seed}" / "pool_scored.jsonl"
        seed_pool_hashes[str(seed)] = (
            build.sha256_file(seed_pool) if seed_pool.exists() else None
        )
    fixed_map = {
        "A+B": "pred_AB",
        "A-only": "pred_Aonly",
        "A_nu": "pred_Anu",
    }
    fixed_drift = {}
    source_key = ["seed", "tag", "prompt_id"]
    for model, column in fixed_map.items():
        paired = per[per["model"].eq(model)][
            source_key + ["prediction"]
        ].merge(
            source[source_key + [column]],
            on=source_key,
            validate="one_to_one",
        )
        fixed_drift[model] = float(
            np.max(np.abs(paired["prediction"] - paired[column]))
        )

    # Refit once for an independent exact-vs-grid interpolation audit.
    fitted_values, _ = build.fit_calibrators(prompt_ids, proxy, gold)
    lookup = build.interpolate_curves(curves)
    d_values = np.unique(per["d"].to_numpy())
    sample_index = np.linspace(
        0, len(d_values) - 1, min(40, len(d_values)), dtype=int
    )
    interpolation_drift = {}
    for model, fitted in fitted_values.items():
        grid, values = lookup[model]
        drift = []
        for d in d_values[sample_index]:
            exact = exact_mean_gain(proxy, fitted, float(d))
            approximate = float(np.interp(d, grid, values))
            drift.append(abs(exact - approximate))
        interpolation_drift[model] = float(max(drift))

    # The loader ignores every evaluation row. Mutating those rows cannot
    # alter the calibration arrays consumed by any flexible predictor.
    original_hash = array_hash(proxy, gold)
    mutated_rows = []
    with args.pool_scored.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "eval":
                row["proxy_score"] = -999999.0
                row["gold_score"] = 999999.0
            mutated_rows.append(row)
    temporary = args.outdir / "_eval_mutation_audit.jsonl"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in mutated_rows:
            handle.write(json.dumps(row) + "\n")
    try:
        _, proxy_mutated, gold_mutated = build.load_calibration(temporary)
    finally:
        temporary.unlink()
    mutated_hash = array_hash(proxy_mutated, gold_mutated)

    failures = []
    if identity_frontier > 1e-12 or identity_dpo > 1e-12:
        failures.append("saved error identity")
    if max(fixed_drift.values()) > 1e-12:
        failures.append("canonical fixed-baseline identity")
    if max(interpolation_drift.values()) > 2e-5:
        failures.append("curve interpolation tolerance")
    if original_hash != mutated_hash:
        failures.append("evaluation-row mutation invariance")
    if set(metadata["model"]) != set(fitted_values):
        failures.append("fit metadata coverage")
    if set(prompt_ids) != calibration_ids:
        failures.append("calibration prompt identity")
    if any(
        record["missing_from_dpo"] or record["extra_in_dpo"]
        for record in eval_id_match.values()
    ):
        failures.append("DPO/evaluation prompt identity")
    if max_frontier_drift > 1e-9:
        failures.append("canonical frontier re-derivation")
    if any(
        value is not None and value != canonical_pool_hash
        for value in seed_pool_hashes.values()
    ):
        failures.append("seed candidate-pool hash")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "n_prediction_rows": len(per),
        "n_checkpoint_rows": len(checkpoint),
        "n_calibration_prompts": len(prompt_ids),
        "max_error_identity_drift": {
            "frontier": identity_frontier,
            "dpo": identity_dpo,
        },
        "canonical_fixed_prediction_max_drift": fixed_drift,
        "exact_vs_interpolated_max_drift": interpolation_drift,
        "evaluation_row_mutation": {
            "calibration_array_hash_before": original_hash,
            "calibration_array_hash_after": mutated_hash,
            "pass": original_hash == mutated_hash,
        },
        "data_chain": {
            "calibration_prompt_count": len(calibration_ids),
            "evaluation_prompt_count": len(evaluation),
            "calibration_evaluation_overlap": len(
                calibration_ids & set(evaluation)
            ),
            "dpo_evaluation_prompt_identity": eval_id_match,
            "max_frontier_rederivation_drift": max_frontier_drift,
            "canonical_pool_sha256": canonical_pool_hash,
            "seed_pool_sha256": seed_pool_hashes,
        },
        "failures": failures,
    }
    (args.outdir / "audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
