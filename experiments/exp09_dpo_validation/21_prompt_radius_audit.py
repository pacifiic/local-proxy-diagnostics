#!/usr/bin/env python3
"""Summarize prompt-level KL radii for the canonical three-seed DPO audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-predictions",
        type=Path,
        default=ROOT
        / "results/actual_dpo/three_seed_canonical/checkpoint_predictions.csv",
    )
    parser.add_argument(
        "--per-prompt-root",
        type=Path,
        default=ROOT
        / "results/actual_dpo/training_runs",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT
        / "results/actual_dpo/three_seed_canonical/prompt_radius_audit",
    )
    parser.add_argument("--mean-kl-max", type=float, default=0.10)
    return parser.parse_args()


def summarize(values: pd.Series) -> dict[str, float | int]:
    return {
        "n_prompt_checkpoint_cells": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "fraction_gt_0p10": float((values > 0.10).mean()),
        "mean_d_pow_1p5": float(np.mean(np.power(values, 1.5))),
    }


def main() -> None:
    args = parse_args()
    checkpoints = pd.read_csv(args.checkpoint_predictions)
    candidates = checkpoints.loc[
        (checkpoints["model"] == "A+B")
        & (checkpoints["fit_scope"] == "coefficient_or_fixed_baseline"),
        ["seed", "tag", "d_med"],
    ].drop_duplicates()

    frames: list[pd.DataFrame] = []
    selected_records: list[dict[str, float | int | str]] = []
    for row in candidates.itertuples(index=False):
        path = (
            args.per_prompt_root
            / f"seed_{int(row.seed)}"
            / str(row.tag)
            / "per_prompt.csv"
        )
        frame = pd.read_csv(path)
        if len(frame) != 200 or frame["prompt_id"].nunique() != 200:
            raise RuntimeError(f"unexpected prompt accounting in {path}")
        if not np.isfinite(frame["d_reweight"]).all():
            raise RuntimeError(f"non-finite prompt KL in {path}")
        checkpoint_mean = float(frame["d_reweight"].mean())
        if checkpoint_mean > args.mean_kl_max:
            continue
        selected_records.append(
            {
                "seed": int(row.seed),
                "tag": str(row.tag),
                "checkpoint_median_prompt_kl": float(row.d_med),
                "checkpoint_mean_prompt_kl": checkpoint_mean,
            }
        )
        frame = frame[["prompt_id", "d_reweight"]].copy()
        frame.insert(0, "tag", str(row.tag))
        frame.insert(0, "seed", int(row.seed))
        frames.append(frame)

    selected = pd.DataFrame(selected_records)
    if len(selected) != 10:
        raise RuntimeError(f"expected 10 selected seed-checkpoints, found {len(selected)}")
    median_selected = {
        (int(row.seed), str(row.tag))
        for row in candidates.loc[candidates["d_med"] <= args.mean_kl_max].itertuples(
            index=False
        )
    }
    mean_selected = set(zip(selected["seed"], selected["tag"]))
    if mean_selected != median_selected:
        raise RuntimeError("mean-KL and median-KL checkpoint gates select different sets")
    cells = pd.concat(frames, ignore_index=True)
    if len(cells) != 2_000:
        raise RuntimeError(f"expected 2,000 prompt-checkpoint cells, found {len(cells)}")

    summaries = [{"scope": "pooled", "seed": -1, **summarize(cells["d_reweight"])}]
    for seed, frame in cells.groupby("seed", sort=True):
        summaries.append(
            {"scope": "seed", "seed": int(seed), **summarize(frame["d_reweight"])}
        )
    summary = pd.DataFrame(summaries)

    args.outdir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.outdir / "prompt_radius_cells.csv", index=False)
    summary.to_csv(args.outdir / "prompt_radius_summary.csv", index=False)
    run_spec = {
        "checkpoint_predictions": display_path(args.checkpoint_predictions),
        "per_prompt_root": display_path(args.per_prompt_root),
        "selection": {
            "model": "A+B",
            "fit_scope": "coefficient_or_fixed_baseline",
            "checkpoint_mean_prompt_kl_max": args.mean_kl_max,
        },
        "selected_seed_checkpoints": selected.sort_values(["seed", "tag"]).to_dict(
            orient="records"
        ),
        "n_prompt_checkpoint_cells": len(cells),
        "interpretation": (
            "The checkpoint gate uses mean prompt KL. It retains the same ten "
            "checkpoints as the earlier median-KL gate. The reported distribution "
            "audits prompt-level radii inside those retained checkpoints."
        ),
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
