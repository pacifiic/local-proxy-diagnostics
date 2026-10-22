#!/usr/bin/env python3
"""Independent output-contract audit for exp28."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    per = pd.read_csv(args.outdir / "per_prompt_predictions.csv")
    checkpoint = pd.read_csv(args.outdir / "checkpoint_predictions.csv")
    seed = pd.read_csv(args.outdir / "seed_level_mae.csv")
    summary = pd.read_csv(args.outdir / "summary_mean_sd.csv")
    windows = pd.read_csv(args.outdir / "window_summary_mean_sd.csv")
    decomposition = pd.read_csv(
        args.outdir / "error_decomposition_summary.csv"
    )
    mechanism = pd.read_csv(args.outdir / "mechanism_by_checkpoint.csv")
    fit_metadata = pd.read_csv(args.outdir / "model_fit_metadata.csv")
    spec = json.loads((args.outdir / "run_spec.json").read_text())

    expected_models = {
        "A+B",
        "A-only",
        "A_nu",
        "Spearman",
        "Zero",
        *{
            f"cal_{mode}_{family}"
            for mode in spec["modes"]
            for family in spec["families"]
        },
        *{
            f"eval_xfit_{mode}_{family}"
            for mode in spec["modes"]
            for family in spec["families"]
        },
    }
    if set(per["model"]) != expected_models:
        raise ValueError("model set mismatch")
    if per.duplicated(["seed", "tag", "prompt_id", "model"]).any():
        raise ValueError("duplicate per-prompt prediction key")
    if set(per["seed"]) != set(spec["seeds"]):
        raise ValueError("seed support mismatch")
    expected_rows = len(spec["seeds"]) * 6 * 200 * len(expected_models)
    if len(per) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(per)}")
    required_numeric = [
        "d",
        "ess",
        "frontier",
        "dpo_realized",
        "prediction",
        "signed_calibration_error",
        "mechanism_gap",
        "signed_total_error",
        "abs_error_frontier",
        "abs_error_dpo",
    ]
    if not np.isfinite(per[required_numeric].to_numpy()).all():
        raise ValueError("non-finite core prediction output")
    identity = (
        per["signed_calibration_error"]
        + per["mechanism_gap"]
        - per["signed_total_error"]
    )
    if float(np.max(np.abs(identity))) > 1e-10:
        raise ValueError("decomposition identity mismatch")
    if not (checkpoint["n_prompts"] == 200).all():
        raise ValueError("checkpoint aggregation does not contain 200 prompts")
    if not (seed["n_checkpoints"] == 6).all():
        raise ValueError("seed aggregation does not contain six checkpoints")
    if not (summary["n_seeds"] == len(spec["seeds"])).all():
        raise ValueError("summary does not contain all seeds")
    if set(windows["scope"]) != {
        "strict_local_d_le_0p10",
        "historical_six_checkpoints",
    }:
        raise ValueError("window summary scope mismatch")
    historical = windows[
        windows["scope"] == "historical_six_checkpoints"
    ]
    if not (
        (historical["min_checkpoints_per_seed"] == 6)
        & (historical["max_checkpoints_per_seed"] == 6)
    ).all():
        raise ValueError("historical scope must contain six checkpoints")
    strict = windows[windows["scope"] == "strict_local_d_le_0p10"]
    if (
        (strict["min_checkpoints_per_seed"] < 1)
        | (strict["max_checkpoints_per_seed"] > 6)
    ).any():
        raise ValueError("strict local checkpoint support is invalid")
    if set(decomposition["model"]) != expected_models:
        raise ValueError("decomposition model set mismatch")
    if len(mechanism) != len(spec["seeds"]) * 6:
        raise ValueError("mechanism table must contain six checkpoints/seed")
    if not (
        (mechanism["mechanism_gap"] > 0)
        & mechanism["realization_ratio"].between(0, 1)
    ).all():
        raise ValueError("unexpected DPO mechanism-gap direction")
    spline_metadata = fit_metadata[
        fit_metadata["model"].str.endswith("_spline")
    ]
    if set(spline_metadata["cv_scheme"].dropna()) != {
        "5-fold GroupKFold by prompt"
    }:
        raise ValueError("spline CV is not prompt-grouped")

    # The historical raw isotonic implementation must be reproduced exactly.
    raw = per[per["model"] == "cal_raw_isotonic"]
    existing = []
    for seed_value in spec["seeds"]:
        source = pd.read_csv(
            args.outdir.parent
            / f"seed_{seed_value}"
            / "exp09_final_per_prompt.csv"
        )[["tag", "prompt_id", "pred_isotonic"]]
        source["seed"] = seed_value
        existing.append(source)
    existing = pd.concat(existing, ignore_index=True)
    paired = raw.merge(
        existing,
        on=["seed", "tag", "prompt_id"],
        validate="one_to_one",
    )
    max_diff = float(
        np.max(np.abs(paired["prediction"] - paired["pred_isotonic"]))
    )
    if max_diff > 1e-10:
        raise ValueError(f"raw isotonic reproduction mismatch: {max_diff}")

    print(f"rows={len(per)}, models={len(expected_models)}, max raw-isotonic diff={max_diff:.3g}")
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
