#!/usr/bin/env python3
"""Verify an independent fixed_141 WMT Exp32 reproduction."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CANONICAL = (
    REPO_ROOT / "results" / "evaluation_aware_ab" / "wmt_fixed_141"
)

EXACT_FILES = (
    "direct_calibration_run_spec.json",
    "evaluation_aware_ab_summary.csv",
    "evaluation_aware_ab_paired_bootstrap.csv",
    "EVALUATION_AWARE_AB_RESULTS.md",
)


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def prediction_path(directory: Path) -> Path:
    raw = directory / "direct_calibration_predictions.csv"
    compressed = directory / "direct_calibration_predictions.csv.gz"
    if raw.exists():
        return raw
    if compressed.exists():
        return compressed
    raise FileNotFoundError(
        f"Missing direct_calibration_predictions.csv[.gz] in {directory}"
    )


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def payload_sha256(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return sha256_stream(handle)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-dir", type=Path, default=DEFAULT_CANONICAL
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    canonical_predictions = prediction_path(args.canonical_dir)
    repeat_predictions = prediction_path(args.repeat_dir)
    canonical_df = read_predictions(canonical_predictions)
    repeat_df = read_predictions(repeat_predictions)

    exact_files = {}
    for name in EXACT_FILES:
        canonical_hash = sha256_file(args.canonical_dir / name)
        repeat_hash = sha256_file(args.repeat_dir / name)
        exact_files[name] = {
            "canonical_sha256": canonical_hash,
            "repeat_sha256": repeat_hash,
            "byte_identical": canonical_hash == repeat_hash,
        }

    manifest_path = (
        REPO_ROOT / "results" / "evaluation_aware_ab" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source_hash = manifest["canonical_lineages"]["wmt"][
        "source_predictions_sha256"
    ]
    repeat_payload_hash = payload_sha256(repeat_predictions)

    dimensions = {
        "rows": int(len(repeat_df)),
        "paths": int(repeat_df["path_id"].nunique()),
        "repeats": int(repeat_df["repeat"].nunique()),
        "radii": int(repeat_df["d"].nunique()),
        "baselines": int(repeat_df["baseline"].nunique()),
    }
    expected_dimensions = {
        "rows": 564000,
        "paths": 141,
        "repeats": 50,
        "radii": 5,
        "baselines": 16,
    }

    checks = {
        "prediction_values_and_order_identical": bool(
            canonical_df.equals(repeat_df)
        ),
        "repeat_payload_matches_original_source_hash": (
            repeat_payload_hash == expected_source_hash
        ),
        "all_small_artifacts_byte_identical": all(
            item["byte_identical"] for item in exact_files.values()
        ),
        "dimensions_match": dimensions == expected_dimensions,
        "no_skipped_file": not (
            args.repeat_dir / "direct_calibration_skipped.csv"
        ).exists(),
    }
    report = {
        "experiment": "exp32_evaluation_aware_ab",
        "dataset_lineage": "WMT fixed_141",
        "protocol": {
            "calibration_fraction": 0.30,
            "repeats": 50,
            "random_seed": 0,
            "radii": [0.01, 0.025, 0.05, 0.075, 0.10],
        },
        "expected_source_predictions_sha256": expected_source_hash,
        "repeat_predictions_payload_sha256": repeat_payload_hash,
        "dimensions": dimensions,
        "expected_dimensions": expected_dimensions,
        "exact_files": exact_files,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }

    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
