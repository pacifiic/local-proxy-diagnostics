#!/usr/bin/env python3
"""  [exp32]Exp36: proxy perturbation baselines for the local-geometry diagnostic.

This experiment tests whether the exact coefficient diagnostic is doing nontrivial work.
It destroys or degrades proxy--gold alignment while preserving much of the score scale, then compares
exact-coefficient predictions against the reliability-only / variance-only A_nu surrogate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import (  # noqa: E402
    PathData,
    delta_at_d,
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
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


def perturb_proxy(
    r: np.ndarray,
    j: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    j = np.asarray(j, dtype=float)
    mode = mode.lower()
    r_sd = _safe_std(r)
    if mode == "original":
        return r.copy()
    if mode == "shuffle":
        return rng.permutation(r)
    if mode == "random_normal":
        return rng.normal(loc=float(np.mean(r)), scale=r_sd, size=len(r))
    if mode == "noisy":
        return r + float(noise_sigma) * r_sd * rng.normal(size=len(r))
    if mode == "sign_flip":
        return -r
    if mode == "oracle":
        # Positive control: same marginal scale as the original proxy, but ranked exactly by gold.
        jz = (j - float(np.mean(j))) / _safe_std(j)
        return float(np.mean(r)) + r_sd * jz
    raise ValueError(f"Unknown perturbation mode: {mode}")


def topk_overlap(r: np.ndarray, j: np.ndarray, frac: float = 0.10) -> float:
    n = len(r)
    k = max(1, int(round(frac * n)))
    top_r = set(np.argsort(r)[-k:].tolist())
    top_j = set(np.argsort(j)[-k:].tolist())
    return float(len(top_r.intersection(top_j)) / k)


def evaluate_one(
    path: PathData,
    r_pert: np.ndarray,
    mode: str,
    noise_sigma: float,
    baseline_rep: int,
    split_rep: int,
    cal_idx: np.ndarray,
    eva_idx: np.ndarray,
    fit_max_d: float,
    d_targets: List[float],
) -> Dict[str, Any]:
    j = path.gold_scores
    r = np.asarray(r_pert, dtype=float)
    j_cal, r_cal = j[cal_idx], r[cal_idx]
    j_eva, r_eva = j[eva_idx], r[eva_idx]

    coeff_cal = theory_coefficients(j_cal, r_cal)
    coeff_eval = theory_coefficients(j_eva, r_eva)
    curve_eval = exact_tilt_curve(r_eva, j_eva, d_max=fit_max_d)
    fit_eval = fit_local_law(curve_eval.d, curve_eval.q, fit_max_d=fit_max_d)

    d_peak_cal, headroom_cal = local_peak_from_coefficients(coeff_cal.a_exact, coeff_cal.b_theory)
    d_peak_eval, headroom_eval = local_peak_from_coefficients(fit_eval.a_fit, fit_eval.b_fit)

    row: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "proxy_source": path.meta.get("proxy_source"),
        "mode": mode,
        "noise_sigma": noise_sigma,
        "baseline_rep": baseline_rep,
        "split_rep": split_rep,
        "n_cal": len(cal_idx),
        "n_eval": len(eva_idx),
        "cal_a_exact": coeff_cal.a_exact,
        "cal_b_theory": coeff_cal.b_theory,
        "cal_a_nu": coeff_cal.a_nu,
        "cal_sigma_j2": coeff_cal.sigma_j2,
        "cal_sigma_r2": coeff_cal.sigma_r2,
        "cal_rho": coeff_cal.rho,
        "eval_a_exact": coeff_eval.a_exact,
        "eval_b_theory": coeff_eval.b_theory,
        "eval_a_fit": fit_eval.a_fit,
        "eval_b_fit": fit_eval.b_fit,
        "eval_r2": fit_eval.r2,
        "cal_spearman_r_j": spearman_corr(r_cal, j_cal),
        "eval_spearman_r_j": spearman_corr(r_eva, j_eva),
        "cal_pearson_r_j": float(np.corrcoef(r_cal, j_cal)[0, 1]) if np.std(r_cal) > 0 and np.std(j_cal) > 0 else np.nan,
        "eval_pearson_r_j": float(np.corrcoef(r_eva, j_eva)[0, 1]) if np.std(r_eva) > 0 and np.std(j_eva) > 0 else np.nan,
        "eval_top10_overlap": topk_overlap(r_eva, j_eva, frac=0.10),
        "cal_peak_d_theory": d_peak_cal,
        "cal_headroom_theory": headroom_cal,
        "eval_peak_d_fit": d_peak_eval,
        "eval_headroom_fit": headroom_eval,
    }

    # Calibration empirical-gain baseline: use the calibration exact-tilt gain as a naive predictor.
    curve_cal = None
    try:
        curve_cal = exact_tilt_curve(r_cal, j_cal, d_max=fit_max_d)
    except (RuntimeError, ValueError):
        curve_cal = None

    for d_star in d_targets:
        suffix = str(d_star).replace(".", "p")
        actual = delta_at_d(curve_eval, d_star)
        predictions = {
            "theory": coeff_cal.a_exact * math.sqrt(d_star) + coeff_cal.b_theory * d_star,
            "anu": coeff_cal.a_nu * math.sqrt(d_star),
            "zero": 0.0,
            "cal_mean_gain": delta_at_d(curve_cal, d_star) if curve_cal is not None else np.nan,
        }
        row[f"actual_eval_delta_{suffix}"] = actual
        for name, pred in predictions.items():
            row[f"pred_{name}_delta_{suffix}"] = pred
            if not np.isnan(actual) and not np.isnan(pred):
                row[f"abs_err_{name}_{suffix}"] = abs(pred - actual)
                row[f"sign_correct_{name}_{suffix}"] = int(np.sign(pred) == np.sign(actual))
    return row


def summarize(df: pd.DataFrame, d_targets: List[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = list(df.groupby(["mode", "noise_sigma"], dropna=False))
    grouped.append((("__OVERALL__", np.nan), df))
    for (mode, sigma), gdf in grouped:
        row: Dict[str, Any] = {
            "mode": mode,
            "noise_sigma": sigma,
            "n_rows": int(len(gdf)),
            "n_paths": int(gdf["path_id"].nunique()) if "path_id" in gdf else np.nan,
            "cal_a_exact_mean": float(gdf["cal_a_exact"].mean()),
            "cal_a_nu_mean": float(gdf["cal_a_nu"].mean()),
            "cal_b_theory_mean": float(gdf["cal_b_theory"].mean()),
            "eval_a_fit_mean": float(gdf["eval_a_fit"].mean()),
            "eval_b_fit_mean": float(gdf["eval_b_fit"].mean()),
            "eval_spearman_mean": float(gdf["eval_spearman_r_j"].mean()),
            "eval_pearson_mean": float(gdf["eval_pearson_r_j"].mean()),
            "eval_top10_overlap_mean": float(gdf["eval_top10_overlap"].mean()),
            "eval_r2_median": float(gdf["eval_r2"].median()),
        }
        for d_star in d_targets:
            suffix = str(d_star).replace(".", "p")
            act_col = f"actual_eval_delta_{suffix}"
            row[f"actual_delta_mean_{suffix}"] = float(gdf[act_col].mean())
            row[f"actual_delta_median_{suffix}"] = float(gdf[act_col].median())
            for name in ["theory", "anu", "zero", "cal_mean_gain"]:
                pred_col = f"pred_{name}_delta_{suffix}"
                err_col = f"abs_err_{name}_{suffix}"
                sign_col = f"sign_correct_{name}_{suffix}"
                if pred_col in gdf:
                    row[f"pred_{name}_mean_{suffix}"] = float(gdf[pred_col].mean())
                if err_col in gdf:
                    row[f"mae_{name}_{suffix}"] = float(gdf[err_col].mean())
                if sign_col in gdf:
                    row[f"sign_acc_{name}_{suffix}"] = float(gdf[sign_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def generate_paths_from_args(args: argparse.Namespace) -> List[PathData]:
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    if args.proxy_csv:
        proxy = pd.read_csv(args.proxy_csv)
        return list(
            generate_model_paths(
                human_long=human,
                proxy_df=proxy,
                item_cols=item_cols,
                score_col=args.score_col,
                group_cols=group_cols,
                proxy_item_col=args.proxy_item_col,
                proxy_score_col=args.proxy_score_col,
                min_items=args.min_items,
            )
        )
    return list(
        generate_human_paths(
            df=human,
            item_cols=item_cols,
            rater_col=args.rater_col,
            score_col=args.score_col,
            group_cols=group_cols,
            min_items=args.min_items,
            min_raters=args.min_raters,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp36: random/noisy/shuffled proxy baseline experiment.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="", help="Optional external item-level proxy CSV; if omitted, uses held-out human proxies.")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--modes", default="original,shuffle,random_normal,noisy,sign_flip,oracle")
    ap.add_argument("--noise-sigmas", default="0.5,1.0,2.0", help="Only used for mode=noisy.")
    ap.add_argument("--baseline-repeats", type=int, default=10, help="Number of perturbation draws per path and mode.")
    ap.add_argument("--n-repeats", type=int, default=20, help="Number of calibration/evaluation splits per perturbation draw.")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--d-targets", default="0.05,0.10")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    rng = np.random.default_rng(args.random_seed)
    modes = parse_cols(args.modes)
    noise_sigmas = [float(x) for x in parse_cols(args.noise_sigmas)]
    d_targets = [float(x) for x in parse_cols(args.d_targets)]

    paths = generate_paths_from_args(args)
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for path in paths:
        n = len(path.item_ids)
        if n < 30:
            continue
        for mode in modes:
            sigmas = noise_sigmas if mode == "noisy" else [0.0]
            reps = args.baseline_repeats if mode in {"shuffle", "random_normal", "noisy"} else 1
            for sigma in sigmas:
                for baseline_rep in range(reps):
                    r_pert = perturb_proxy(path.proxy_scores, path.gold_scores, mode, rng, sigma)
                    for split_rep in range(args.n_repeats):
                        cal_idx, eva_idx = split_indices(n, args.calibration_frac, rng)
                        try:
                            row = evaluate_one(
                                path=path,
                                r_pert=r_pert,
                                mode=mode,
                                noise_sigma=float(sigma),
                                baseline_rep=baseline_rep,
                                split_rep=split_rep,
                                cal_idx=cal_idx,
                                eva_idx=eva_idx,
                                fit_max_d=args.fit_max_d,
                                d_targets=d_targets,
                            )
                        except (RuntimeError, ValueError, FloatingPointError):
                            skipped += 1
                            continue
                        rows.append(row)

    if not rows:
        raise RuntimeError("No valid perturbation rows were produced.")

    paths_df = pd.DataFrame(rows)
    paths_csv = outdir / "exp32_proxy_baseline_paths.csv"
    paths_df.to_csv(paths_csv, index=False)

    summary_df = summarize(paths_df, d_targets)
    summary_csv = outdir / "exp32_proxy_baseline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "n_paths": len(paths),
        "n_rows": len(paths_df),
        "n_skipped": skipped,
        "modes": modes,
        "noise_sigmas": noise_sigmas,
        "baseline_repeats": args.baseline_repeats,
        "n_repeats": args.n_repeats,
        "calibration_frac": args.calibration_frac,
        "fit_max_d": args.fit_max_d,
        "d_targets": d_targets,
    }
    (outdir / "exp32_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print(f"Paths: {len(paths)} | Rows: {len(paths_df)} | Skipped: {skipped}")
    print(f"Saved path-level results to {paths_csv}")
    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
