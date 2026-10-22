#!/usr/bin/env python3
"""Qwen/Zephyr source-only direct-calibration transfer comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from baseline_core import (
    BASELINES,
    ensure_empty_outdir,
    evaluate_models,
    fit_models,
    load_pool,
    prompt_reachability,
    sha256_file,
    validate_pool,
)


D_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
GEOMETRY = "fixed_prompt_kl"
FIT_MAX_D = 0.10


def prompt_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return (
        frame[frame["split"].eq(split)][["prompt_id", "prompt"]]
        .drop_duplicates()
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-pool", type=Path, required=True)
    parser.add_argument("--zephyr-pool", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    ensure_empty_outdir(args.outdir)

    qwen = load_pool(args.qwen_pool)
    zephyr = load_pool(args.zephyr_pool)
    audit = {
        "qwen": validate_pool(
            qwen,
            name="UltraFeedback Qwen2.5-3B candidate pool",
            candidate_col="sample_idx",
        ),
        "zephyr": validate_pool(
            zephyr,
            name="UltraFeedback Zephyr-7B candidate pool",
            candidate_col="sample_idx",
        ),
    }
    for split in ("calibration", "eval"):
        q_table = prompt_table(qwen, split)
        z_table = prompt_table(zephyr, split)
        if not q_table.equals(z_table):
            raise ValueError(
                f"Qwen and Zephyr {split} prompt IDs/text are not identical"
            )
    if set(qwen["split"]) != {"calibration", "eval"}:
        raise ValueError("Qwen pool has unexpected split labels")
    if set(zephyr["split"]) != {"calibration", "eval"}:
        raise ValueError("Zephyr pool has unexpected split labels")

    pools = {"qwen": qwen, "zephyr": zephyr}
    aggregate_frames: list[pd.DataFrame] = []
    prompt_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    reachability_frames: list[pd.DataFrame] = []
    for pool_name, pool in pools.items():
        for split in ("calibration", "eval"):
            reachability = prompt_reachability(
                pool[pool["split"].eq(split)], D_VALUES
            )
            reachability.insert(0, "pool", pool_name)
            reachability.insert(1, "split", split)
            reachability_frames.append(reachability)
    for source_name, target_name in (("qwen", "zephyr"), ("zephyr", "qwen")):
        source = pools[source_name]
        target = pools[target_name]
        target_eval = target[target["split"].eq("eval")].copy()
        route = f"{source_name}_to_{target_name}"
        for estimation_scope, fit_pool in (
            ("source_only", source[source["split"].eq("calibration")].copy()),
            ("target_calibrated", target[target["split"].eq("calibration")].copy()),
        ):
            seed = args.random_seed + (0 if source_name == "qwen" else 1000)
            if estimation_scope == "target_calibrated":
                seed += 100
            models = fit_models(
                fit_pool,
                geometry=GEOMETRY,
                d_values=D_VALUES,
                fit_max_d=FIT_MAX_D,
                seed=seed,
            )
            aggregate, prompt = evaluate_models(
                fit_pool,
                target_eval,
                models,
                geometry=GEOMETRY,
                d_values=D_VALUES,
            )
            for frame in (aggregate, prompt):
                frame.insert(0, "route", route)
                frame.insert(1, "estimation_scope", estimation_scope)
                frame["n_fit_prompts"] = int(fit_pool["prompt_id"].nunique())
                frame["n_eval_prompts"] = int(
                    target_eval["prompt_id"].nunique()
                )
            aggregate_frames.append(aggregate)
            prompt_frames.append(prompt)
            metadata_rows.append(
                {
                    "route": route,
                    "estimation_scope": estimation_scope,
                    "a": models.coefficients.a,
                    "b": models.coefficients.b,
                    "a_nu": models.coefficients.a_nu,
                    "empirical_a": models.empirical_a,
                    "empirical_b": models.empirical_b,
                    "model_metadata": json.dumps(
                        models.metadata, sort_keys=True
                    ),
                }
            )
            print(f"completed {route} / {estimation_scope}", flush=True)

    aggregate = pd.concat(aggregate_frames, ignore_index=True).sort_values(
        ["route", "estimation_scope", "d", "baseline"]
    )
    prompt = pd.concat(prompt_frames, ignore_index=True).sort_values(
        ["route", "estimation_scope", "prompt_id", "d", "baseline"]
    )
    aggregate.to_csv(args.outdir / "transfer_predictions.csv", index=False)
    prompt.to_csv(
        args.outdir / "transfer_prompt_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(metadata_rows).sort_values(
        ["route", "estimation_scope"]
    ).to_csv(args.outdir / "transfer_fit_metadata.csv", index=False)
    pd.concat(reachability_frames, ignore_index=True).to_csv(
        args.outdir / "transfer_reachability.csv", index=False
    )
    spec = {
        "experiment": "exp25_real_llm_cross_model_transfer",
        "qwen_pool": str(args.qwen_pool),
        "qwen_pool_sha256": sha256_file(args.qwen_pool),
        "zephyr_pool": str(args.zephyr_pool),
        "zephyr_pool_sha256": sha256_file(args.zephyr_pool),
        "input_audit": audit,
        "prompt_identity_check": "exact ID and prompt-text equality",
        "geometry": GEOMETRY,
        "d_values": D_VALUES,
        "local_window": "d <= 0.10",
        "fit_max_d": FIT_MAX_D,
        "random_seed": args.random_seed,
        "split_unit": "prompt_id",
        "direct_model_cv_unit": "prompt_id",
        "baselines": BASELINES,
        "routes": ["qwen_to_zephyr", "zephyr_to_qwen"],
        "estimation_scopes": ["source_only", "target_calibrated"],
        "unreachable_prompt_policy": (
            "saturate tied/constant proxy prompts at t=1e4 (or t=0 for "
            "zero variance), matching exp09/exp11; report reachability by radius"
        ),
        "n_prediction_rows": int(len(aggregate)),
        "n_prompt_prediction_rows": int(len(prompt)),
    }
    (args.outdir / "transfer_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(f"wrote transfer outputs to {args.outdir}")


if __name__ == "__main__":
    main()
