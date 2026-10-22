#!/usr/bin/env python3
"""Audit coefficient robustness to clipping and injected outliers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

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
    mkdir,
    parse_cols,
    spearman_corr,
    theory_coefficients,
)


def stable_seed(*parts: object, modulo: int = 2**32) -> int:
    text = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


def safe_std(x: np.ndarray) -> float:
    s = float(np.std(np.asarray(x, dtype=float)))
    return s if s > 1e-12 else 1.0


def winsorize(x: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    return np.clip(x, lo, hi)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    probs = (ranks - 0.5) / float(n)
    eps = 1.0 / (2.0 * n)
    probs = np.clip(probs, eps, 1.0 - eps)
    nd = NormalDist()
    return np.array([nd.inv_cdf(float(p)) for p in probs], dtype=float)


def inject_outliers(
    x: np.ndarray,
    rng: np.random.Generator,
    frac: float,
    magnitude: float,
    direction: str,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    sign = -1.0 if direction == "negative" else 1.0
    x[idx] += sign * float(magnitude) * safe_std(x)
    return x


def transform_scores(
    j: np.ndarray,
    r: np.ndarray,
    condition: str,
    rng: np.random.Generator,
    outlier_frac: float,
    outlier_magnitude: float,
    outlier_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    j = np.asarray(j, dtype=float)
    r = np.asarray(r, dtype=float)
    if condition == "original":
        return j.copy(), r.copy()
    if condition == "winsor_1_99":
        return winsorize(j, 0.01, 0.99), winsorize(r, 0.01, 0.99)
    if condition == "winsor_5_95":
        return winsorize(j, 0.05, 0.95), winsorize(r, 0.05, 0.95)
    if condition == "rank_normal":
        return rank_normalize(j), rank_normalize(r)
    if condition == "outlier_proxy":
        return j.copy(), inject_outliers(r, rng, outlier_frac, outlier_magnitude, outlier_direction)
    if condition == "outlier_gold":
        return inject_outliers(j, rng, outlier_frac, outlier_magnitude, outlier_direction), r.copy()
    raise ValueError(f"Unknown condition: {condition}")


def path_metrics(j: np.ndarray, r: np.ndarray, d_target: float, fit_max_d: float) -> dict[str, float]:
    coeff = theory_coefficients(j, r)
    curve = exact_tilt_curve(r, j, d_max=fit_max_d)
    fit = fit_local_law(curve.d, curve.q, fit_max_d=fit_max_d)
    actual = delta_at_d(curve, d_target)
    pred = coeff.a_exact * math.sqrt(d_target) + coeff.b_theory * d_target
    return {
        "a_exact": coeff.a_exact,
        "b_theory": coeff.b_theory,
        "a_nu": coeff.a_nu,
        "rho": coeff.rho,
        "sigma_j2": coeff.sigma_j2,
        "sigma_r2": coeff.sigma_r2,
        "kappa_r": coeff.kappa_r,
        "kappa_jrr": coeff.kappa_jrr,
        "a_fit": fit.a_fit,
        "b_fit": fit.b_fit,
        "r2": fit.r2,
        "actual_delta": actual,
        "pred_delta": pred,
        "abs_err_theory": abs(pred - actual),
        "spearman": spearman_corr(r, j),
        "min_var_r": float(np.min(curve.var_r)),
        "min_ess": float(np.min(curve.ess)),
    }


def corr_or_nan(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def sign_agree(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if not mask.any():
        return float("nan")
    return float((np.sign(x[mask]) == np.sign(y[mask])).mean())


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for (dataset, condition), g in rows.groupby(["dataset", "condition"], dropna=False):
        out.append(
            {
                "dataset": dataset,
                "condition": condition,
                "n_rows": int(len(g)),
                "n_paths": int(g["path_id"].nunique()),
                "corr_a_exact_vs_original": corr_or_nan(g["a_exact"], g["base_a_exact"]),
                "corr_b_theory_vs_original": corr_or_nan(g["b_theory"], g["base_b_theory"]),
                "b_sign_agreement_vs_original": sign_agree(g["b_theory"], g["base_b_theory"]),
                "delta_sign_agreement_vs_original": sign_agree(g["actual_delta"], g["base_actual_delta"]),
                "median_abs_a_change": float((g["a_exact"] - g["base_a_exact"]).abs().median()),
                "median_abs_b_change": float((g["b_theory"] - g["base_b_theory"]).abs().median()),
                "median_abs_delta_change": float((g["actual_delta"] - g["base_actual_delta"]).abs().median()),
                "mae_theory": float(g["abs_err_theory"].mean()),
                "median_min_ess": float(g["min_ess"].median()),
                "median_min_var_r": float(g["min_var_r"].median()),
            }
        )
    return pd.DataFrame(out)


def ensure_empty_outdir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Use --overwrite or choose a fresh --outdir."
        )
    return mkdir(path)


def parse_conditions(text: str) -> list[str]:
    valid = {
        "original",
        "winsor_1_99",
        "winsor_5_95",
        "rank_normal",
        "outlier_proxy",
        "outlier_gold",
    }
    out = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"Unknown condition(s): {bad}. Valid: {sorted(valid)}")
    if "original" not in out:
        out.insert(0, "original")
    return out


def load_paths(args: argparse.Namespace) -> list[PathData]:
    df = pd.read_csv(args.human_long_csv)
    paths = list(
        generate_human_paths(
            df=df,
            item_cols=parse_cols(args.item_cols),
            rater_col=args.rater_col,
            score_col=args.score_col,
            group_cols=parse_cols(args.group_cols),
            min_items=args.min_items,
            min_raters=args.min_raters,
        )
    )
    if not paths:
        raise RuntimeError("No admissible paths generated.")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Tail/outlier robustness audit for local coefficients.")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", required=True)
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--conditions", default="original,winsor_1_99,winsor_5_95,rank_normal,outlier_proxy,outlier_gold")
    ap.add_argument("--outlier-repeats", type=int, default=20)
    ap.add_argument("--outlier-frac", type=float, default=0.01)
    ap.add_argument("--outlier-magnitude", type=float, default=8.0)
    ap.add_argument("--outlier-direction", choices=["positive", "negative"], default="positive")
    ap.add_argument("--d-target", type=float, default=0.10)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.outlier_frac < 1.0):
        raise ValueError("--outlier-frac must be in (0,1)")
    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    conditions = parse_conditions(args.conditions)
    paths = load_paths(args)

    spec = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_name": args.dataset_name,
        "human_long_csv": args.human_long_csv,
        "item_cols": args.item_cols,
        "group_cols": args.group_cols,
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "conditions": conditions,
        "outlier_repeats": args.outlier_repeats,
        "outlier_frac": args.outlier_frac,
        "outlier_magnitude": args.outlier_magnitude,
        "outlier_direction": args.outlier_direction,
        "d_target": args.d_target,
        "fit_max_d": args.fit_max_d,
        "random_seed": args.random_seed,
        "seed_policy": "SHA-256 over dataset/path/condition/repeat/base seed",
        "n_paths": len(paths),
    }
    (outdir / "tail_outlier_robustness_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    base_by_path: dict[str, dict[str, float]] = {}
    for path in paths:
        try:
            base_by_path[path.path_id] = path_metrics(path.gold_scores, path.proxy_scores, args.d_target, args.fit_max_d)
        except Exception as exc:
            print(f"Skipping base path {path.path_id}: {exc}")

    rows: list[dict[str, Any]] = []
    stochastic = {"outlier_proxy", "outlier_gold"}
    for path in paths:
        if path.path_id not in base_by_path:
            continue
        base = base_by_path[path.path_id]
        for condition in conditions:
            reps = args.outlier_repeats if condition in stochastic else 1
            for rep in range(reps):
                seed = stable_seed(args.dataset_name, path.path_id, condition, rep, args.random_seed)
                rng = np.random.default_rng(seed)
                try:
                    j_new, r_new = transform_scores(
                        path.gold_scores,
                        path.proxy_scores,
                        condition=condition,
                        rng=rng,
                        outlier_frac=args.outlier_frac,
                        outlier_magnitude=args.outlier_magnitude,
                        outlier_direction=args.outlier_direction,
                    )
                    metrics = path_metrics(j_new, r_new, args.d_target, args.fit_max_d)
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": args.dataset_name,
                            "group_key": path.group_key,
                            "path_id": path.path_id,
                            "condition": condition,
                            "rep": rep,
                            "seed": seed,
                            "status": "skipped",
                            "error": str(exc),
                        }
                    )
                    continue
                row: dict[str, Any] = {
                    "dataset": args.dataset_name,
                    "group_key": path.group_key,
                    "path_id": path.path_id,
                    "condition": condition,
                    "rep": rep,
                    "seed": seed,
                    "status": "ok",
                    "n_items": len(path.item_ids),
                }
                row.update(metrics)
                for key, value in base.items():
                    row[f"base_{key}"] = value
                rows.append(row)

    if not rows:
        raise RuntimeError("No robustness rows were produced.")
    paths_df = pd.DataFrame(rows)
    paths_df.to_csv(outdir / "tail_outlier_robustness_paths.csv", index=False)
    ok_df = paths_df[paths_df["status"].eq("ok")].copy()
    summary_df = summarize(ok_df)
    summary_df.to_csv(outdir / "tail_outlier_robustness_summary.csv", index=False)
    (outdir / "tail_outlier_robustness_summary.md").write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Saved tail/outlier robustness outputs in: {outdir}")


if __name__ == "__main__":
    main()

