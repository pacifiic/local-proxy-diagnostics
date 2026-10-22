#!/usr/bin/env python3
"""Calibration-budget and radius sweep on real LLM-generated candidates."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_core import (
    BASELINES,
    ensure_empty_outdir,
    evaluate_models,
    fit_models,
    load_pool,
    sha256_file,
    validate_pool,
)


CALIBRATION_FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]
D_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
GEOMETRY = "common_temperature"
FIT_MAX_D = 0.10

_FRAME: pd.DataFrame | None = None


def _set_frame(frame: pd.DataFrame) -> None:
    global _FRAME
    _FRAME = frame


def _task(
    repeat: int,
    calibration_frac: float,
    calibration_ids: list[str],
    evaluation_ids: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if _FRAME is None:
        raise RuntimeError("worker candidate pool was not initialized")
    prompt_text = _FRAME["prompt_id"].astype(str)
    calibration = _FRAME[prompt_text.isin(calibration_ids)].copy()
    evaluation = _FRAME[prompt_text.isin(evaluation_ids)].copy()
    models = fit_models(
        calibration,
        geometry=GEOMETRY,
        d_values=D_VALUES,
        fit_max_d=FIT_MAX_D,
        seed=seed,
    )
    aggregate, prompt = evaluate_models(
        calibration,
        evaluation,
        models,
        geometry=GEOMETRY,
        d_values=D_VALUES,
    )
    for frame in (aggregate, prompt):
        frame.insert(0, "repeat", repeat)
        frame.insert(1, "calibration_frac", calibration_frac)
        frame["n_cal_prompts"] = len(calibration_ids)
        frame["n_eval_prompts"] = len(evaluation_ids)
    metadata = {
        "repeat": repeat,
        "calibration_frac": calibration_frac,
        "n_cal_prompts": len(calibration_ids),
        "n_eval_prompts": len(evaluation_ids),
        "a": models.coefficients.a,
        "b": models.coefficients.b,
        "a_nu": models.coefficients.a_nu,
        "empirical_a": models.empirical_a,
        "empirical_b": models.empirical_b,
        "model_metadata": json.dumps(models.metadata, sort_keys=True),
    }
    return aggregate, prompt, metadata


def build_nested_splits(
    prompt_ids: list[str],
    n_repeats: int,
    seed: int,
) -> tuple[list[tuple[int, float, list[str], list[str], int]], pd.DataFrame]:
    tasks: list[tuple[int, float, list[str], list[str], int]] = []
    manifest: list[dict[str, object]] = []
    ids = np.asarray(sorted(prompt_ids), dtype=str)
    for repeat in range(n_repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, repeat]))
        ordered = ids.copy()
        rng.shuffle(ordered)
        for frac in CALIBRATION_FRACS:
            n_cal = max(2, min(int(round(frac * len(ids))), len(ids) - 2))
            calibration = ordered[:n_cal].tolist()
            evaluation = ordered[n_cal:].tolist()
            task_seed = int(
                np.random.SeedSequence([seed, repeat, int(frac * 1000)])
                .generate_state(1)[0]
            )
            tasks.append((repeat, frac, calibration, evaluation, task_seed))
            cal_set = set(calibration)
            for prompt_id in ids:
                manifest.append(
                    {
                        "repeat": repeat,
                        "calibration_frac": frac,
                        "prompt_id": prompt_id,
                        "role": (
                            "calibration" if prompt_id in cal_set else "evaluation"
                        ),
                    }
                )
    return tasks, pd.DataFrame(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.n_repeats < 1 or args.jobs < 1:
        parser.error("--n-repeats and --jobs must be positive")
    ensure_empty_outdir(args.outdir)

    frame = load_pool(args.candidates)
    audit = validate_pool(
        frame,
        name="SummEval Qwen2.5-7B candidate pool",
        candidate_col="candidate_id",
    )
    tasks, manifest = build_nested_splits(
        frame["prompt_id"].astype(str).unique().tolist(),
        args.n_repeats,
        args.random_seed,
    )
    manifest.to_csv(args.outdir / "within_split_manifest.csv", index=False)

    aggregate_frames: list[pd.DataFrame] = []
    prompt_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    _set_frame(frame)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        mp_context=context,
        initializer=_set_frame,
        initargs=(frame,),
    ) as executor:
        futures = [executor.submit(_task, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            aggregate, prompt, metadata = future.result()
            aggregate_frames.append(aggregate)
            prompt_frames.append(prompt)
            metadata_rows.append(metadata)
            if index % 25 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)} split fits", flush=True)

    aggregate = pd.concat(aggregate_frames, ignore_index=True).sort_values(
        ["calibration_frac", "repeat", "d", "baseline"]
    )
    prompt = pd.concat(prompt_frames, ignore_index=True).sort_values(
        ["calibration_frac", "repeat", "prompt_id", "d", "baseline"]
    )
    metadata = pd.DataFrame(metadata_rows).sort_values(
        ["calibration_frac", "repeat"]
    )
    aggregate.to_csv(args.outdir / "within_predictions.csv", index=False)
    prompt.to_csv(
        args.outdir / "within_prompt_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metadata.to_csv(args.outdir / "within_fit_metadata.csv", index=False)

    spec = {
        "experiment": "exp25_real_llm_calibration_followup_within",
        "input": str(args.candidates),
        "input_sha256": sha256_file(args.candidates),
        "input_audit": audit,
        "geometry": GEOMETRY,
        "calibration_fractions": CALIBRATION_FRACS,
        "d_values": D_VALUES,
        "local_window": "d <= 0.10",
        "fit_max_d": FIT_MAX_D,
        "n_repeats": args.n_repeats,
        "random_seed": args.random_seed,
        "nested_splits": True,
        "split_unit": "prompt_id",
        "baselines": BASELINES,
        "direct_model_cv_unit": "prompt_id",
        "n_prediction_rows": int(len(aggregate)),
        "n_prompt_prediction_rows": int(len(prompt)),
    }
    (args.outdir / "within_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(f"wrote within-dataset outputs to {args.outdir}")


if __name__ == "__main__":
    main()
