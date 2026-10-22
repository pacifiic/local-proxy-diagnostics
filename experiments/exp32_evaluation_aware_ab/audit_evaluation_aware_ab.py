#!/usr/bin/env python3
"""Audit evaluation-aware conditional-mean A+B projections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {
    "isotonic": ("isotonic_calibration", "A+B (isotonic h)"),
    "binned": ("binned_shrinkage", "A+B (binned h)"),
    "spline": ("spline_ridge", "A+B (spline h)"),
    "MLP": ("mlp_calibration", "A+B (MLP h)"),
}
KEYS = ["dataset", "path_id", "repeat", "d"]
ENDPOINT_D = 0.10
N_BOOT = 2000
SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_curve_errors(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    subset = frame[frame["baseline"] == baseline]
    expected = frame["d"].nunique()
    result = (
        subset.groupby(["dataset", "path_id", "repeat"])
        .agg(curve_mae=("abs_error", "mean"), n_d=("d", "size"))
        .reset_index()
    )
    return result[result["n_d"] == expected].drop(columns="n_d")


def support_matched_curve_errors(
    frame: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    """Average available paired radius cells within path, then across paths."""
    subset = frame[frame["baseline"] == baseline]
    return (
        subset.groupby(["dataset", "path_id"])
        .agg(curve_mae=("abs_error", "mean"))
        .reset_index()
    )


def endpoint_errors(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    subset = frame[
        (frame["baseline"] == baseline)
        & np.isclose(frame["d"], ENDPOINT_D, rtol=0.0, atol=1e-12)
    ]
    return subset[
        ["dataset", "path_id", "repeat", "abs_error"]
    ].rename(columns={"abs_error": "endpoint_mae"})


def support_matched_endpoint_errors(
    frame: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    subset = frame[
        (frame["baseline"] == baseline)
        & np.isclose(frame["d"], ENDPOINT_D, rtol=0.0, atol=1e-12)
    ]
    return (
        subset.groupby(["dataset", "path_id"])
        .agg(endpoint_mae=("abs_error", "mean"))
        .reset_index()
    )


def apply_support_exclusions(
    frame: pd.DataFrame, exclusions_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["dataset", "path_id", "repeat", "d"]
    exclusions = pd.read_csv(exclusions_path)
    missing = set(keys) - set(exclusions.columns)
    if missing:
        raise ValueError(
            f"support exclusions missing columns: {sorted(missing)}"
        )
    exclusions = exclusions[keys].drop_duplicates()
    marked = frame.merge(
        exclusions.assign(_exclude=True),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    excluded = marked["_exclude"].eq(True)
    matched_keys = marked.loc[excluded, keys].drop_duplicates()
    if len(matched_keys) != len(exclusions):
        raise ValueError(
            "some requested support exclusions do not match prediction cells"
        )
    filtered = marked.loc[~excluded].drop(columns="_exclude")
    report = {
        "mode": "common_support",
        "exclusions_csv": str(exclusions_path),
        "n_excluded_radius_cells": int(len(exclusions)),
        "excluded_cells_by_dataset": {
            str(key): int(value)
            for key, value in exclusions.groupby("dataset").size().items()
        },
        "prediction_rows_before": int(len(frame)),
        "prediction_rows_after": int(len(filtered)),
    }
    return filtered, report


def clustered_interval(
    paired: pd.DataFrame,
    left: str,
    right: str,
) -> tuple[float, float, float]:
    by_path = paired.groupby("path_id")[[left, right]].mean()
    differences = (by_path[left] - by_path[right]).to_numpy()
    rng = np.random.default_rng(SEED)
    samples = np.empty(N_BOOT)
    for index in range(N_BOOT):
        selected = rng.integers(0, len(differences), size=len(differences))
        samples[index] = float(np.mean(differences[selected]))
    return (
        float(np.mean(differences)),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Prediction CSV or CSV.GZ. Defaults to direct_calibration_predictions"
            " inside --outdir."
        ),
    )
    parser.add_argument(
        "--support-exclusions",
        type=Path,
        help=(
            "CSV of dataset/path_id/repeat/d cells unavailable to any displayed"
            " comparator. When provided, all methods are evaluated on this"
            " common support."
        ),
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        args.predictions
        if args.predictions is not None
        else args.outdir / "direct_calibration_predictions.csv"
    )
    if not predictions_path.exists() and predictions_path.suffix == ".csv":
        gzip_path = predictions_path.with_suffix(".csv.gz")
        if gzip_path.exists():
            predictions_path = gzip_path
    frame = pd.read_csv(predictions_path)
    support_report: dict[str, object] = {
        "mode": "all_available_cells",
        "predictions_csv": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_rows": int(len(frame)),
    }
    support_matched = args.support_exclusions is not None
    if args.support_exclusions is not None:
        frame, support_report = apply_support_exclusions(
            frame, args.support_exclusions
        )
        support_report["predictions_csv"] = str(predictions_path)
        support_report["predictions_sha256"] = sha256_file(predictions_path)
        support_report["support_exclusions_sha256"] = sha256_file(
            args.support_exclusions
        )

    required = {"A+B", *(name for pair in PAIRS.values() for name in pair)}
    missing = required - set(frame["baseline"])
    if missing:
        raise ValueError(f"missing baselines: {sorted(missing)}")

    summary_rows = []
    paired_rows = []
    for dataset in frame["dataset"].drop_duplicates():
        dataset_frame = frame[frame["dataset"] == dataset]
        curve_errors = (
            support_matched_curve_errors
            if support_matched
            else complete_curve_errors
        )
        endpoint_error_rows = (
            support_matched_endpoint_errors
            if support_matched
            else endpoint_errors
        )
        raw = curve_errors(dataset_frame, "A+B")
        raw = raw.rename(columns={"curve_mae": "raw_ab"})
        for label, (direct_name, projected_name) in PAIRS.items():
            direct = curve_errors(dataset_frame, direct_name).rename(
                columns={"curve_mae": "direct"}
            )
            projected = curve_errors(dataset_frame, projected_name).rename(
                columns={"curve_mae": "projected"}
            )
            paired = raw.merge(
                direct,
                on=["dataset", "path_id"]
                + ([] if support_matched else ["repeat"]),
                validate="one_to_one",
            ).merge(
                projected,
                on=["dataset", "path_id"]
                + ([] if support_matched else ["repeat"]),
                validate="one_to_one",
            )
            raw_endpoint = endpoint_error_rows(
                dataset_frame, "A+B"
            ).rename(columns={"endpoint_mae": "raw_ab_endpoint"})
            direct_endpoint = endpoint_error_rows(
                dataset_frame, direct_name
            ).rename(columns={"endpoint_mae": "direct_endpoint"})
            projected_endpoint = endpoint_error_rows(
                dataset_frame, projected_name
            ).rename(columns={"endpoint_mae": "projected_endpoint"})
            endpoint_keys = ["dataset", "path_id"] + (
                [] if support_matched else ["repeat"]
            )
            endpoint = raw_endpoint.merge(
                direct_endpoint,
                on=endpoint_keys,
                validate="one_to_one",
            ).merge(
                projected_endpoint,
                on=endpoint_keys,
                validate="one_to_one",
            )

            direct_predictions = dataset_frame[
                dataset_frame["baseline"] == direct_name
            ][KEYS + ["predicted_gain"]].rename(
                columns={"predicted_gain": "direct_prediction"}
            )
            projected_predictions = dataset_frame[
                dataset_frame["baseline"] == projected_name
            ][KEYS + ["predicted_gain"]].rename(
                columns={"predicted_gain": "projected_prediction"}
            )
            truncation = direct_predictions.merge(
                projected_predictions, on=KEYS, validate="one_to_one"
            )
            truncation["abs_gap"] = np.abs(
                truncation["projected_prediction"]
                - truncation["direct_prediction"]
            )
            if support_matched:
                mean_truncation_gap = float(
                    truncation.groupby("path_id")["abs_gap"].mean().mean()
                )
            else:
                mean_truncation_gap = float(truncation["abs_gap"].mean())

            summary_rows.append(
                {
                    "dataset": dataset,
                    "conditional_mean_model": label,
                    "n_paths": paired["path_id"].nunique(),
                    "n_path_repeats": int(
                        dataset_frame.loc[
                            dataset_frame["baseline"] == "A+B",
                            ["path_id", "repeat"],
                        ].drop_duplicates().shape[0]
                    ),
                    "raw_A+B_curve_mae": paired["raw_ab"].mean(),
                    "evaluation_aware_A+B_curve_mae": paired[
                        "projected"
                    ].mean(),
                    "direct_calibrator_curve_mae": paired["direct"].mean(),
                    "raw_A+B_endpoint_mae_d0p10": endpoint[
                        "raw_ab_endpoint"
                    ].mean(),
                    "evaluation_aware_A+B_endpoint_mae_d0p10": endpoint[
                        "projected_endpoint"
                    ].mean(),
                    "direct_calibrator_endpoint_mae_d0p10": endpoint[
                        "direct_endpoint"
                    ].mean(),
                    "mean_abs_projection_vs_direct": mean_truncation_gap,
                }
            )
            for comparator in ["raw_ab", "direct"]:
                mean, low, high = clustered_interval(
                    paired, "projected", comparator
                )
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "conditional_mean_model": label,
                        "comparison": (
                            "evaluation-aware A+B minus "
                            + (
                                "raw A+B"
                                if comparator == "raw_ab"
                                else "direct calibrator"
                            )
                        ),
                        "difference": mean,
                        "ci_lo": low,
                        "ci_hi": high,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(paired_rows)
    summary.to_csv(args.outdir / "evaluation_aware_ab_summary.csv", index=False)
    comparisons.to_csv(
        args.outdir / "evaluation_aware_ab_paired_bootstrap.csv", index=False
    )
    (args.outdir / "evaluation_aware_ab_support.json").write_text(
        json.dumps(support_report, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Evaluation-aware conditional-mean A+B",
        "",
        "> **Information setting.** This analysis assumes the target candidate pool",
        "> has already been collected and proxy-scored. It is not a fair pre-pool",
        "> comparison for a decision made from 300 rated items before the future 1,000",
        "> candidates exist; use exp33 for that setting.",
        "",
        (
            "> **Evaluation support.** All displayed methods use the shared"
            " radius cells defined by `--support-exclusions`."
            if support_matched
            else "> **Evaluation support.** All available prediction cells are used."
        ),
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Path-clustered paired differences",
        "",
        comparisons.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Negative differences favor evaluation-aware A+B.",
        "",
    ]
    (args.outdir / "EVALUATION_AWARE_AB_RESULTS.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print()
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
