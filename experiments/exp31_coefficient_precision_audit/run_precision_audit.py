#!/usr/bin/env python3
"""Generate compact coefficient-precision audits.

The script reconciles endpoint and curve metrics, measures
calibration-size coefficient errors, summarizes B contribution by radius, and
extracts a compact strict-local DPO table. Existing outputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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

_spec = importlib.util.spec_from_file_location("exp21_precision_engine", EXP21_PATH)
exp21 = importlib.util.module_from_spec(_spec)
sys.modules["exp21_precision_engine"] = exp21
_spec.loader.exec_module(exp21)  # type: ignore[union-attr]

CALIBRATION_FRACS = (0.05, 0.10, 0.30)
D_VALUES = (0.05, 0.10)
N_REPEATS = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_outdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def coefficient_error_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for frac in CALIBRATION_FRACS:
        for cfg in exp21.CONFIGS:
            args = exp21._make_args(cfg)
            paths = list(exp21.calib.build_paths(args))
            rng = np.random.default_rng(exp21.RANDOM_SEED)
            for path in paths:
                for repeat in range(N_REPEATS):
                    cal_idx, eval_idx = exp21.calib.split_indices(
                        len(path.item_ids), frac, rng
                    )
                    r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
                    j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
                    r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
                    j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
                    try:
                        cal = exp21.calib.theory_coefficients(j_cal, r_cal)
                        evaluation = exp21.calib.theory_coefficients(
                            j_eval, r_eval
                        )
                    except (ValueError, ZeroDivisionError):
                        continue
                    for d in D_VALUES:
                        try:
                            weights = exp21.calib.exact_weights_at_d(r_eval, d)
                        except (ValueError, RuntimeError):
                            continue
                        actual = float(
                            np.sum(weights * j_eval) - np.mean(j_eval)
                        )
                        predicted_ab = (
                            cal.a_exact * math.sqrt(d)
                            + cal.b_theory * d
                        )
                        local_eval = (
                            evaluation.a_exact * math.sqrt(d)
                            + evaluation.b_theory * d
                        )
                        ga = abs(cal.a_exact) * math.sqrt(d)
                        gb = abs(cal.b_theory) * d
                        rows.append(
                            {
                                "dataset": cfg["dataset_name"],
                                "path_id": path.path_id,
                                "repeat": repeat,
                                "calibration_frac": frac,
                                "n_cal": len(cal_idx),
                                "n_eval": len(eval_idx),
                                "d": d,
                                "a_cal": cal.a_exact,
                                "b_cal": cal.b_theory,
                                "a_eval": evaluation.a_exact,
                                "b_eval": evaluation.b_theory,
                                "abs_a_error": abs(
                                    cal.a_exact - evaluation.a_exact
                                ),
                                "abs_b_error": abs(
                                    cal.b_theory - evaluation.b_theory
                                ),
                                "b_sign_agreement": float(
                                    np.sign(cal.b_theory)
                                    == np.sign(evaluation.b_theory)
                                ),
                                "scaled_a_error": math.sqrt(d)
                                * abs(cal.a_exact - evaluation.a_exact),
                                "scaled_b_error": d
                                * abs(cal.b_theory - evaluation.b_theory),
                                "eval_remainder_abs": abs(
                                    actual - local_eval
                                ),
                                "actual_gain": actual,
                                "predicted_ab": predicted_ab,
                                "ab_abs_error": abs(predicted_ab - actual),
                                "curvature_share": (
                                    gb / (ga + gb) if ga + gb > 0 else 0.0
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def summarize_coefficient_errors(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["dataset", "calibration_frac", "d"], as_index=False)
        .agg(
            n_path_splits=("abs_b_error", "size"),
            median_abs_a_error=("abs_a_error", "median"),
            median_abs_b_error=("abs_b_error", "median"),
            b_sign_agreement=("b_sign_agreement", "mean"),
            median_scaled_a_error=("scaled_a_error", "median"),
            median_scaled_b_error=("scaled_b_error", "median"),
            median_eval_remainder_abs=("eval_remainder_abs", "median"),
            median_curvature_share=("curvature_share", "median"),
        )
        .sort_values(["dataset", "calibration_frac", "d"])
    )


def reconcile_metrics(
    paper_human: Path,
    paper_gpt: Path,
    direct_predictions: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    for setting, path in [
        ("SummEval-human_heldout", paper_human),
        ("SummEval-GPT4o", paper_gpt),
    ]:
        data = pd.read_csv(path)
        local = data[
            data["d"].le(0.10 + 1e-12) & data["actual_delta"].notna()
        ]
        per_curve = (
            local.groupby(["path_id", "repeat"])["abs_err_theory"]
            .agg(["mean", "size"])
            .reset_index()
        )
        rows.append(
            {
                "setting": setting,
                "protocol": "paper Grid IAE",
                "radius_grid": ",".join(
                    f"{x:g}" for x in sorted(local["d"].unique())
                ),
                "n_complete_curves": len(per_curve),
                "a_plus_b_error": per_curve["mean"].mean(),
                "reachability_policy": (
                    "d-sweep retained 1,589/1,600 SummEval path-splits"
                    if setting.startswith("SummEval-human")
                    else "all 200 path-splits"
                ),
            }
        )

    direct = pd.read_csv(direct_predictions)
    direct = direct[direct["baseline"].eq("A+B")]
    for setting, data in direct.groupby("dataset", sort=False):
        per_curve = (
            data.groupby(["path_id", "repeat"])["abs_error"]
            .agg(["mean", "size"])
            .reset_index()
        )
        rows.append(
            {
                "setting": setting,
                "protocol": "local-grid MAE",
                "radius_grid": ",".join(
                    f"{x:g}" for x in sorted(data["d"].unique())
                ),
                "n_complete_curves": int((per_curve["size"] == 5).sum()),
                "a_plus_b_error": per_curve.loc[
                    per_curve["size"] == 5, "mean"
                ].mean(),
                "reachability_policy": (
                    "retain each reachable radius; all A+B path-splits complete"
                ),
            }
        )
    return pd.DataFrame(rows)


def compact_dpo_table(dpo_summary_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dpo_summary_path)
    keep = {
        "A+B": "A+B",
        "A-only": "A-only",
        "cal_zscore_binned": "prompt-z-score binned",
        "Zero": "zero",
    }
    data = data[
        data["scope"].eq("strict_local_d_le_0p10")
        & data["model"].isin(keep)
    ].copy()
    data["predictor"] = data["model"].map(keep)
    return data[
        [
            "predictor",
            "n_seeds",
            "n_checkpoints_total",
            "min_checkpoints_per_seed",
            "max_checkpoints_per_seed",
            "frontier_mae",
            "frontier_mae_sd",
            "realized_mae",
            "realized_mae_sd",
            "frontier_signed_bias",
            "optimizer_gap_mean",
        ]
    ].sort_values("realized_mae")


def validate_against_exp24(
    rows: pd.DataFrame, split_metrics_path: Path
) -> dict:
    old = pd.read_csv(split_metrics_path)
    old = old[
        old["baseline"].eq("A+B")
        & old["calibration_frac"].isin(CALIBRATION_FRACS)
    ]
    frames = []
    for d, suffix in [(0.05, "0p05"), (0.10, "0p1")]:
        frame = old[
            [
                "dataset",
                "path_id",
                "repeat",
                "calibration_frac",
                f"actual_gain_d{suffix}",
                f"predicted_gain_d{suffix}",
                f"abs_error_d{suffix}",
            ]
        ].copy()
        frame["d"] = d
        frame = frame.rename(
            columns={
                f"actual_gain_d{suffix}": "actual_old",
                f"predicted_gain_d{suffix}": "predicted_old",
                f"abs_error_d{suffix}": "abs_error_old",
            }
        )
        frames.append(frame)
    old_long = pd.concat(frames, ignore_index=True).dropna(
        subset=["actual_old", "predicted_old", "abs_error_old"]
    )
    keys = ["dataset", "path_id", "repeat", "calibration_frac", "d"]
    merged = rows.merge(
        old_long, on=keys, how="outer", indicator=True, validate="one_to_one"
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("exp24 cross-check has unmatched split/radius keys")
    comparisons = {
        "actual_gain": "actual_old",
        "predicted_ab": "predicted_old",
        "ab_abs_error": "abs_error_old",
    }
    max_differences = {
        new: float((merged[new] - merged[old_name]).abs().max())
        for new, old_name in comparisons.items()
    }
    if any(value > 1e-12 for value in max_differences.values()):
        raise ValueError(
            f"exp24 cross-check numerical mismatch: {max_differences}"
        )
    return {
        "status": "PASS",
        "n_cells": len(merged),
        "key": keys,
        "max_absolute_differences": max_differences,
    }


def write_results(
    path: Path,
    metric: pd.DataFrame,
    coeff: pd.DataFrame,
    dpo: pd.DataFrame,
) -> None:
    b_at_point_one = coeff[np.isclose(coeff["d"], 0.10)]
    compact_b = b_at_point_one[
        [
            "dataset",
            "calibration_frac",
            "median_abs_b_error",
            "b_sign_agreement",
            "median_scaled_a_error",
            "median_scaled_b_error",
            "median_eval_remainder_abs",
        ]
    ]
    contribution = coeff[
        np.isclose(coeff["calibration_frac"], 0.30)
    ][
        [
            "dataset",
            "d",
            "median_curvature_share",
        ]
    ]
    parts = [
        "# Coefficient Precision Audit",
        "",
        "## Metric reconciliation",
        "",
        metric.to_markdown(index=False),
        "",
        "The two metrics are both means of pointwise absolute error, but they "
        "use different radius grids and reachability accounting. They should "
        "not be presented as the same numerical experiment.",
        "",
        "## Calibration-size coefficient error at d=0.10",
        "",
        compact_b.to_markdown(index=False),
        "",
        "The scaled A/B columns are the first two coefficient-shift terms in "
        "the paper's prediction-error decomposition. The remainder is computed "
        "on the evaluation split using its own A and B.",
        "",
        "## Curvature contribution over radius",
        "",
        contribution.to_markdown(index=False),
        "",
        "## Strict-local DPO comparison",
        "",
        dpo.to_markdown(index=False),
        "",
        "The SD columns are sample standard deviations across three DPO seeds. "
        "The strict window contains 10 seed-checkpoint observations: three or "
        "four checkpoints per seed because checkpoint median KL differs by seed.",
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-human-d-sweep", type=Path, required=True)
    parser.add_argument("--paper-gpt-d-sweep", type=Path, required=True)
    parser.add_argument("--direct-predictions", type=Path, required=True)
    parser.add_argument("--dpo-summary", type=Path, required=True)
    parser.add_argument("--exp24-split-metrics", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    fresh_outdir(args.outdir)

    coefficient_rows = coefficient_error_rows()
    coefficient_summary = summarize_coefficient_errors(coefficient_rows)
    metric = reconcile_metrics(
        args.paper_human_d_sweep,
        args.paper_gpt_d_sweep,
        args.direct_predictions,
    )
    dpo = compact_dpo_table(args.dpo_summary)
    exp24_check = validate_against_exp24(
        coefficient_rows, args.exp24_split_metrics
    )

    coefficient_rows.to_csv(
        args.outdir / "coefficient_error_rows.csv", index=False
    )
    coefficient_summary.to_csv(
        args.outdir / "coefficient_error_summary.csv", index=False
    )
    metric.to_csv(args.outdir / "metric_reconciliation.csv", index=False)
    dpo.to_csv(args.outdir / "strict_local_dpo.csv", index=False)
    (args.outdir / "exp24_crosscheck.json").write_text(
        json.dumps(exp24_check, indent=2) + "\n", encoding="utf-8"
    )
    write_results(
        args.outdir / "RESULTS.md", metric, coefficient_summary, dpo
    )

    inputs = [
        args.paper_human_d_sweep,
        args.paper_gpt_d_sweep,
        args.direct_predictions,
        args.dpo_summary,
        args.exp24_split_metrics,
    ]
    spec = {
        "calibration_fracs": CALIBRATION_FRACS,
        "d_values": D_VALUES,
        "n_repeats": N_REPEATS,
        "random_seed": exp21.RANDOM_SEED,
        "split_policy": (
            "fresh RNG(seed=0) per dataset/fraction; matches exp24 nested "
            "calibration-size protocol"
        ),
        "input_files": [
            {"path": str(input_path), "sha256": sha256_file(input_path)}
            for input_path in inputs
        ],
        "code_sha256": sha256_file(Path(__file__)),
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    (args.outdir / "WORKFLOW_COMPLETE").write_text("PASS\n", encoding="ascii")


if __name__ == "__main__":
    main()
