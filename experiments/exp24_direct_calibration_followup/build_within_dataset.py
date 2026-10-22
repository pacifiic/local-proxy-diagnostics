#!/usr/bin/env python3
"""Calibration-budget and radius follow-up for exp21.

This script reuses the audited exp21 implementation. It changes only the
calibration fraction and radius grid, writes every prediction cell to a
deterministic gzip CSV, and also emits a compact split-level table for paired
inference. Existing exp21 outputs are never modified.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP21_PATH = (
    ROOT
    / "experiments"
    / "exp21_direct_calibration"
    / "build_direct_calibration.py"
)

_spec = importlib.util.spec_from_file_location("exp21_followup_engine", EXP21_PATH)
exp21 = importlib.util.module_from_spec(_spec)
sys.modules["exp21_followup_engine"] = exp21
_spec.loader.exec_module(exp21)  # type: ignore[union-attr]

CALIBRATION_FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]
D_TARGETS = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
N_REPEATS = 50
KEEP_BASELINES = [
    "A+B",
    "A-only",
    "A_nu",
    "empirical_curve",
    "empirical_fit",
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
    "zero",
]


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def split_metrics(cells: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "path_id",
        "repeat",
        "baseline",
        "calibration_frac",
        "n_cal",
        "n_eval",
    ]
    local = (
        cells[cells["d"] <= 0.10 + 1e-12]
        .groupby(keys, dropna=False)["abs_error"]
        .agg(local_curve_mae="mean", n_local_radii="size")
        .reset_index()
    )
    extended = (
        cells.groupby(keys, dropna=False)["abs_error"]
        .agg(extended_curve_mae="mean", n_extended_radii="size")
        .reset_index()
    )
    local.loc[local["n_local_radii"] != 5, "local_curve_mae"] = np.nan
    extended.loc[
        extended["n_extended_radii"] != len(D_TARGETS),
        "extended_curve_mae",
    ] = np.nan
    out = local.merge(extended, on=keys, how="outer", validate="one_to_one")
    for d in D_TARGETS:
        suffix = str(d).replace(".", "p")
        drows = cells[np.isclose(cells["d"], d)][
            keys + ["abs_error", "actual_gain", "predicted_gain"]
        ].rename(
            columns={
                "abs_error": f"abs_error_d{suffix}",
                "actual_gain": f"actual_gain_d{suffix}",
                "predicted_gain": f"predicted_gain_d{suffix}",
            }
        )
        out = out.merge(drows, on=keys, how="outer", validate="one_to_one")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--jobs", type=int, default=min(32, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[cfg["config_key"] for cfg in exp21.CONFIGS],
        default=[cfg["config_key"] for cfg in exp21.CONFIGS],
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    ensure_empty(args.outdir)

    exp21.D_TARGETS = list(D_TARGETS)
    exp21.N_REPEATS = N_REPEATS

    cells_path = args.outdir / "within_prediction_cells.csv.gz"
    split_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict] = []
    accounting_rows: list[dict] = []

    with cells_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                write_header = True
                for frac in CALIBRATION_FRACS:
                    exp21.CALIBRATION_FRAC = frac
                    selected = [
                        cfg for cfg in exp21.CONFIGS
                        if cfg["config_key"] in set(args.datasets)
                    ]
                    for cfg in selected:
                        print(
                            f"=== {cfg['dataset_name']} | calibration={frac:.0%} ===",
                            flush=True,
                        )
                        frame, skipped = exp21.run_config(cfg, n_jobs=args.jobs)
                        frame["calibration_frac"] = frac
                        frame = frame[
                            frame["baseline"].isin(KEEP_BASELINES)
                        ].copy()
                        frame.to_csv(text, index=False, header=write_header)
                        text.flush()
                        write_header = False
                        split_frames.append(split_metrics(frame))
                        for row in skipped:
                            row["calibration_frac"] = frac
                        skipped_rows.extend(skipped)
                        accounting_rows.append(
                            {
                                "dataset": cfg["dataset_name"],
                                "calibration_frac": frac,
                                "n_paths": int(frame["path_id"].nunique()),
                                "n_prediction_cells": int(len(frame)),
                                "n_skipped_records": int(len(skipped)),
                            }
                        )

    splits = pd.concat(split_frames, ignore_index=True)
    splits.to_csv(args.outdir / "within_split_metrics.csv", index=False)
    pd.DataFrame(accounting_rows).to_csv(
        args.outdir / "within_accounting.csv", index=False
    )
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            args.outdir / "within_skipped.csv", index=False
        )

    spec = {
        "experiment": "exp24_direct_calibration_followup_within_dataset",
        "datasets": [cfg["dataset_name"] for cfg in exp21.CONFIGS],
        "calibration_fracs": CALIBRATION_FRACS,
        "d_targets": D_TARGETS,
        "local_window": "d <= 0.10",
        "extended_window": "d <= 0.20 (scope audit, not a local-law claim)",
        "n_repeats": N_REPEATS,
        "random_seed": exp21.RANDOM_SEED,
        "jobs": args.jobs,
        "split_policy": (
            "fresh RNG(seed=0) per calibration fraction; identical path/repeat "
            "permutations make calibration sets nested across fractions"
        ),
        "fit_policy": (
            "all calibration models fit once per path/repeat/fraction and reused "
            "at every radius; hyperparameter CV uses calibration labels only"
        ),
        "baselines": KEEP_BASELINES,
        "engine": str(EXP21_PATH.relative_to(ROOT)),
    }
    (args.outdir / "within_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(f"-> {cells_path}")
    print(f"-> {args.outdir / 'within_split_metrics.csv'} ({len(splits)} rows)")


if __name__ == "__main__":
    main()
