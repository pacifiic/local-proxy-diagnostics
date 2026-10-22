#!/usr/bin/env python3
"""  [exp31]Exp35: cross-dataset / OOD prediction of local proxy-optimization geometry.

This experiment is analysis-only: it does not fine-tune a model. It reuses repeated-rating
path construction and exact exponential-tilt utilities from proxy_experiment_package.

It reports strict source-only transfer, target small-calibration prediction, source-prior shrinkage,
A_nu reliability-only baselines, and a zero-gain baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import (  # noqa: E402
    PathData,
    TheoryCoefficients,
    delta_at_d,
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    local_peak_from_coefficients,
    mkdir,
    parse_cols,
    spearman_corr,
    split_indices,
    theory_coefficients,
)


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(np.asarray(x, dtype=float)))
    return s if s > 1e-12 else 1.0


def standardize_pair(j: np.ndarray, r: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Transform one (gold, proxy) path.

    Modes:
      none   : raw scores.
      center : center both scores, keep scale.
      z_both : z-score gold and proxy separately. Recommended for cross-dataset transfer.
      z_gold : z-score gold only and center proxy.
    """
    mode = mode.lower()
    j = np.asarray(j, dtype=float).copy()
    r = np.asarray(r, dtype=float).copy()
    if mode == "none":
        return j, r
    if mode == "center":
        return j - float(np.mean(j)), r - float(np.mean(r))
    if mode == "z_both":
        return (j - float(np.mean(j))) / _safe_std(j), (r - float(np.mean(r))) / _safe_std(r)
    if mode == "z_gold":
        return (j - float(np.mean(j))) / _safe_std(j), r - float(np.mean(r))
    raise ValueError(f"Unknown standardize mode: {mode}")


def transform_path(path: PathData, mode: str, dataset_name: str) -> PathData:
    j, r = standardize_pair(path.gold_scores, path.proxy_scores, mode)
    meta = dict(path.meta)
    meta.update({"dataset_name": dataset_name, "standardize_pairs": mode})
    return PathData(
        group_key=f"{dataset_name}::{path.group_key}",
        path_id=f"{dataset_name}::{path.path_id}",
        item_ids=path.item_ids,
        proxy_scores=r,
        gold_scores=j,
        meta=meta,
    )


def build_paths(
    csv_path: str,
    dataset_name: str,
    item_cols: List[str],
    group_cols: List[str],
    rater_col: str,
    score_col: str,
    min_items: int,
    min_raters: int,
    standardize_pairs: str,
) -> List[PathData]:
    df = pd.read_csv(csv_path)
    paths = list(
        generate_human_paths(
            df=df,
            item_cols=item_cols,
            rater_col=rater_col,
            score_col=score_col,
            group_cols=group_cols,
            min_items=min_items,
            min_raters=min_raters,
        )
    )
    return [transform_path(p, standardize_pairs, dataset_name) for p in paths]


def coeff_to_row(coeff: TheoryCoefficients, prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}a_exact": coeff.a_exact,
        f"{prefix}b_theory": coeff.b_theory,
        f"{prefix}a_nu": coeff.a_nu,
        f"{prefix}sigma_j2": coeff.sigma_j2,
        f"{prefix}sigma_r2": coeff.sigma_r2,
        f"{prefix}rho": coeff.rho,
        f"{prefix}kappa_r": coeff.kappa_r,
        f"{prefix}kappa_jrr": coeff.kappa_jrr,
    }


def collect_source_prior(
    source_paths: List[PathData],
    rng: np.random.Generator,
    calibration_frac: float,
    n_repeats: int,
    estimator: str,
    standardize_cal_only: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for path in source_paths:
        n = len(path.item_ids)
        if n < 30:
            continue
        for rep in range(n_repeats):
            cal_idx, _ = split_indices(n, calibration_frac, rng)
            j_cal = path.gold_scores[cal_idx]
            r_cal = path.proxy_scores[cal_idx]
            if standardize_cal_only:
                j_mean, j_std = float(np.mean(j_cal)), _safe_std(j_cal)
                r_mean, r_std = float(np.mean(r_cal)), _safe_std(r_cal)
                j_cal = (j_cal - j_mean) / j_std
                r_cal = (r_cal - r_mean) / r_std
            try:
                coeff = theory_coefficients(j_cal, r_cal)
            except ValueError:
                continue
            row: Dict[str, Any] = {
                "dataset": path.meta.get("dataset_name"),
                "group_key": path.group_key,
                "path_id": path.path_id,
                "rep": rep,
            }
            row.update(coeff_to_row(coeff))
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid source calibration coefficients were produced.")
    if estimator == "mean":
        agg = df[["a_exact", "b_theory", "a_nu"]].mean(numeric_only=True)
    elif estimator == "median":
        agg = df[["a_exact", "b_theory", "a_nu"]].median(numeric_only=True)
    else:
        raise ValueError("--source-prior-estimator must be mean or median")
    return {
        "source_a_exact": float(agg["a_exact"]),
        "source_b_theory": float(agg["b_theory"]),
        "source_a_nu": float(agg["a_nu"]),
    }, df


def evaluate_target_split(
    path: PathData,
    cal_idx: np.ndarray,
    eva_idx: np.ndarray,
    source_prior: Dict[str, float],
    shrinkage_lambda: float,
    fit_max_d: float,
    d_targets: List[float],
    standardize_cal_only: bool = False,
) -> Dict[str, Any]:
    j_cal = path.gold_scores[cal_idx]
    r_cal = path.proxy_scores[cal_idx]
    j_eva = path.gold_scores[eva_idx]
    r_eva = path.proxy_scores[eva_idx]

    # z_cal mode: standardize both splits using calibration-split statistics only
    if standardize_cal_only:
        j_mean, j_std = float(np.mean(j_cal)), _safe_std(j_cal)
        r_mean, r_std = float(np.mean(r_cal)), _safe_std(r_cal)
        j_cal = (j_cal - j_mean) / j_std
        r_cal = (r_cal - r_mean) / r_std
        j_eva = (j_eva - j_mean) / j_std
        r_eva = (r_eva - r_mean) / r_std

    coeff_cal = theory_coefficients(j_cal, r_cal)
    coeff_eval = theory_coefficients(j_eva, r_eva)
    curve_eval = exact_tilt_curve(r_eva, j_eva, d_max=fit_max_d)
    fit_eval = fit_local_law(curve_eval.d, curve_eval.q, fit_max_d=fit_max_d)

    src_a = source_prior["source_a_exact"]
    src_b = source_prior["source_b_theory"]
    prior_a = shrinkage_lambda * coeff_cal.a_exact + (1.0 - shrinkage_lambda) * src_a
    prior_b = shrinkage_lambda * coeff_cal.b_theory + (1.0 - shrinkage_lambda) * src_b

    row: Dict[str, Any] = {
        "target_dataset": path.meta.get("dataset_name"),
        "target_group_key": path.group_key,
        "target_path_id": path.path_id,
        "n_cal": int(len(cal_idx)),
        "n_eval": int(len(eva_idx)),
        "shrinkage_lambda": shrinkage_lambda,
        "eval_a_fit": fit_eval.a_fit,
        "eval_b_fit": fit_eval.b_fit,
        "eval_r2": fit_eval.r2,
        "cal_spearman_r_j": spearman_corr(r_cal, j_cal),
        "eval_spearman_r_j": spearman_corr(r_eva, j_eva),
        "source_a_exact": src_a,
        "source_b_theory": src_b,
        "source_a_nu": source_prior["source_a_nu"],
        "source_prior_a": prior_a,
        "source_prior_b": prior_b,
    }
    row.update(coeff_to_row(coeff_cal, prefix="cal_"))
    row.update(coeff_to_row(coeff_eval, prefix="eval_"))

    for name, a, b in [
        ("source", src_a, src_b),
        ("target_cal", coeff_cal.a_exact, coeff_cal.b_theory),
        ("source_prior", prior_a, prior_b),
        ("eval_fit", fit_eval.a_fit, fit_eval.b_fit),
    ]:
        d_peak, headroom = local_peak_from_coefficients(float(a), float(b))
        row[f"{name}_peak_d"] = d_peak
        row[f"{name}_headroom"] = headroom

    for d_star in d_targets:
        suffix = str(d_star).replace(".", "p")
        actual = delta_at_d(curve_eval, d_star)
        predictions = {
            "source_mean": src_a * math.sqrt(d_star) + src_b * d_star,
            "target_cal": coeff_cal.a_exact * math.sqrt(d_star) + coeff_cal.b_theory * d_star,
            "source_prior": prior_a * math.sqrt(d_star) + prior_b * d_star,
            "target_anu": coeff_cal.a_nu * math.sqrt(d_star),
            "source_anu": source_prior["source_a_nu"] * math.sqrt(d_star),
            "zero": 0.0,
        }
        row[f"actual_eval_delta_{suffix}"] = actual
        for pred_name, pred in predictions.items():
            row[f"pred_{pred_name}_delta_{suffix}"] = pred
            if not np.isnan(actual):
                row[f"abs_err_{pred_name}_{suffix}"] = abs(pred - actual)
                row[f"sign_correct_{pred_name}_{suffix}"] = int(np.sign(pred) == np.sign(actual))
    return row


def summarize(rows_df: pd.DataFrame, d_targets: List[float]) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    grouped = list(rows_df.groupby(["target_dataset", "target_group_key"], dropna=False))
    grouped.append((("__OVERALL__", "__OVERALL__"), rows_df))

    for (target_dataset, target_group), gdf in grouped:
        row: Dict[str, Any] = {
            "target_dataset": target_dataset,
            "target_group_key": target_group,
            "n_splits": int(len(gdf)),
            "eval_r2_median": float(gdf["eval_r2"].median()),
            "cal_spearman_mean": float(gdf["cal_spearman_r_j"].mean()),
            "eval_spearman_mean": float(gdf["eval_spearman_r_j"].mean()),
            "cal_a_exact_mean": float(gdf["cal_a_exact"].mean()),
            "cal_b_theory_mean": float(gdf["cal_b_theory"].mean()),
            "eval_a_theory_mean": float(gdf["eval_a_exact"].mean()),
            "eval_b_theory_mean": float(gdf["eval_b_theory"].mean()),
        }
        for d_star in d_targets:
            suffix = str(d_star).replace(".", "p")
            act_col = f"actual_eval_delta_{suffix}"
            row[f"actual_delta_mean_{suffix}"] = float(gdf[act_col].mean())
            row[f"actual_delta_median_{suffix}"] = float(gdf[act_col].median())
            for pred_name in ["source_mean", "target_cal", "source_prior", "target_anu", "source_anu", "zero"]:
                err_col = f"abs_err_{pred_name}_{suffix}"
                sign_col = f"sign_correct_{pred_name}_{suffix}"
                if err_col in gdf:
                    row[f"mae_{pred_name}_{suffix}"] = float(gdf[err_col].mean())
                if sign_col in gdf:
                    row[f"sign_acc_{pred_name}_{suffix}"] = float(gdf[sign_col].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp35: OOD / cross-dataset prediction experiment.")
    ap.add_argument("--source-human-long-csv", required=True)
    ap.add_argument("--source-name", default="source")
    ap.add_argument("--source-item-cols", default="item_id")
    ap.add_argument("--source-group-cols", default="")
    ap.add_argument("--source-rater-col", default="rater_idx")
    ap.add_argument("--source-score-col", default="score")
    ap.add_argument("--target-human-long-csv", required=True)
    ap.add_argument("--target-name", default="target")
    ap.add_argument("--target-item-cols", default="item_id")
    ap.add_argument("--target-group-cols", default="")
    ap.add_argument("--target-rater-col", default="rater_idx")
    ap.add_argument("--target-score-col", default="score")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--source-prior-repeats", type=int, default=50)
    ap.add_argument("--source-prior-estimator", choices=["mean", "median"], default="mean")
    ap.add_argument("--shrinkage-lambda", type=float, default=0.75, help="0=source only, 1=target calibration only for source_prior predictor.")
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--d-targets", default="0.05,0.10")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--standardize-pairs", choices=["none", "center", "z_both", "z_gold", "z_cal"], default="none")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if not (0.0 <= args.shrinkage_lambda <= 1.0):
        raise ValueError("--shrinkage-lambda must be in [0,1]")

    outdir = mkdir(args.outdir)
    d_targets = [float(x) for x in parse_cols(args.d_targets)]
    rng = np.random.default_rng(args.random_seed)

    # z_cal mode: build paths without standardization, apply cal-only z-scoring at split time
    is_z_cal = args.standardize_pairs == "z_cal"
    build_std = "none" if is_z_cal else args.standardize_pairs

    source_paths = build_paths(
        csv_path=args.source_human_long_csv,
        dataset_name=args.source_name,
        item_cols=parse_cols(args.source_item_cols),
        group_cols=parse_cols(args.source_group_cols),
        rater_col=args.source_rater_col,
        score_col=args.source_score_col,
        min_items=args.min_items,
        min_raters=args.min_raters,
        standardize_pairs=build_std,
    )
    target_paths = build_paths(
        csv_path=args.target_human_long_csv,
        dataset_name=args.target_name,
        item_cols=parse_cols(args.target_item_cols),
        group_cols=parse_cols(args.target_group_cols),
        rater_col=args.target_rater_col,
        score_col=args.target_score_col,
        min_items=args.min_items,
        min_raters=args.min_raters,
        standardize_pairs=build_std,
    )

    source_prior, source_coeffs_df = collect_source_prior(
        source_paths=source_paths,
        rng=rng,
        calibration_frac=args.calibration_frac,
        n_repeats=args.source_prior_repeats,
        estimator=args.source_prior_estimator,
        standardize_cal_only=is_z_cal,
    )
    source_coeffs_csv = outdir / "exp31_source_coefficients.csv"
    source_coeffs_df.to_csv(source_coeffs_csv, index=False)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for path in target_paths:
        n = len(path.item_ids)
        if n < 30:
            continue
        for rep in range(args.n_repeats):
            cal_idx, eva_idx = split_indices(n, args.calibration_frac, rng)
            try:
                row = evaluate_target_split(
                    path=path,
                    cal_idx=cal_idx,
                    eva_idx=eva_idx,
                    source_prior=source_prior,
                    shrinkage_lambda=args.shrinkage_lambda,
                    fit_max_d=args.fit_max_d,
                    d_targets=d_targets,
                    standardize_cal_only=is_z_cal,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                skipped += 1
                continue
            row["split_rep"] = rep
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid target splits were produced.")

    paths_df = pd.DataFrame(rows)
    paths_csv = outdir / "exp31_ood_transfer_paths.csv"
    paths_df.to_csv(paths_csv, index=False)

    summary_df = summarize(paths_df, d_targets)
    summary_csv = outdir / "exp31_ood_transfer_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    spec = {
        "source_name": args.source_name,
        "target_name": args.target_name,
        "source_human_long_csv": args.source_human_long_csv,
        "target_human_long_csv": args.target_human_long_csv,
        "n_source_paths": len(source_paths),
        "n_target_paths": len(target_paths),
        "n_target_rows": len(paths_df),
        "n_skipped_target_splits": skipped,
        "source_prior": source_prior,
        "standardize_pairs": args.standardize_pairs,
        "calibration_frac": args.calibration_frac,
        "n_repeats": args.n_repeats,
        "source_prior_repeats": args.source_prior_repeats,
        "source_prior_estimator": args.source_prior_estimator,
        "shrinkage_lambda": args.shrinkage_lambda,
        "fit_max_d": args.fit_max_d,
        "d_targets": d_targets,
    }
    (outdir / "exp31_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print(f"Source paths: {len(source_paths)} | Target paths: {len(target_paths)}")
    print(f"Source prior: {source_prior}")
    print(f"Saved source coefficients to {source_coeffs_csv}")
    print(f"Saved target path-level results to {paths_csv}")
    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
