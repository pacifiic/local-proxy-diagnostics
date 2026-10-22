"""Shared helpers for the final accept-boost experiment suite.

These utilities intentionally depend only on the repository's existing
``common_experiment_utils`` module plus numpy/pandas/matplotlib.  The goal is
for the new experiments to be easy to run next to the existing exp27 outputs
without introducing a new dependency stack.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

# Allow importing when this file is imported from experiments/12_accept_boost.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

from common_experiment_utils import (  # noqa: E402
    PathData,
    delta_at_d,
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
    mkdir,
    parse_cols,
    split_indices,
    theory_coefficients,
)

PREDICTORS: List[str] = ["theory", "aonly", "empfit", "anu", "zero", "oracle"]
PREDICTOR_LABELS: Dict[str, str] = {
    "theory": "A+B coefficient",
    "aonly": "A-only",
    "empfit": "empirical fit",
    "anu": "reliability-only A_nu",
    "zero": "zero gain",
    "oracle": "oracle eval A+B",
}


def d_suffix(d: float) -> str:
    """Stable compact suffix used in CSV column names, e.g. 0.10 -> '0p1'.

    Uses the same format as existing experiments (str(d).replace('.', 'p'))
    to ensure column name consistency across the codebase.
    """
    return str(d).replace(".", "p").replace("-", "m")


def parse_float_list(text: str | None, default: Sequence[float] | None = None) -> List[float]:
    if text is None or not str(text).strip():
        return list(default or [])
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def load_paths_from_args(args: Any) -> List[PathData]:
    """Load either held-out-human paths or model-proxy paths from argparse args."""
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    if getattr(args, "proxy_csv", ""):
        proxy = pd.read_csv(args.proxy_csv)
        paths = list(
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
    else:
        paths = list(
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
    if not paths:
        raise RuntimeError("No admissible paths were generated. Check item/group/rater columns and min-items.")
    return paths


def predict_split_rows(
    path: PathData,
    cal_idx: np.ndarray,
    eval_idx: np.ndarray,
    d_grid: Sequence[float],
    eval_max_d: float | None = None,
    empfit_max_d: float = 0.10,
    n_grid: int = 240,
    extra: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Evaluate all predictors for one calibration/evaluation split.

    The population/theory coefficients are estimated on the calibration split.
    The actual target is the exact tilt gain on the evaluation split.  The
    empirical-fit baseline fits A and B to the calibration exact curve over
    ``empfit_max_d``.  The oracle baseline uses evaluation coefficients and is
    included only as a diagnostic upper bound, not a deployable predictor.

    If the exact tilt curve cannot reach a requested d value (e.g. because the
    calibration or evaluation split is too small or B is too negative), that d
    point gets NaN for actual_delta rather than skipping the entire split.
    """
    d_grid = [float(d) for d in d_grid]
    if not d_grid:
        raise ValueError("d_grid is empty")

    r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
    j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
    r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
    j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)

    coeff_cal = theory_coefficients(j_cal, r_cal)
    coeff_eval = theory_coefficients(j_eval, r_eval)

    # Compute calibration curve up to empfit_max_d (always needed for emp-fit).
    # Use a conservative d_max to avoid bracketing failures on cal side.
    try:
        curve_cal = exact_tilt_curve(r_cal, j_cal, d_max=float(empfit_max_d), n_grid=n_grid)
        fit_cal = fit_local_law(curve_cal.d, curve_cal.q, fit_max_d=float(empfit_max_d))
    except Exception:
        # If even empfit_max_d fails, we cannot produce any row for this split.
        raise

    # Compute evaluation curve up to the largest requested d.
    # If the full range fails, try progressively smaller d_max values
    # so that reachable d points still get actual delta values.
    eval_max_d_requested = float(eval_max_d if eval_max_d is not None else max(d_grid))
    curve_eval = None
    eval_actual_max_d = eval_max_d_requested
    for try_d in sorted(set([eval_max_d_requested] + sorted(d_grid, reverse=True)), reverse=True):
        try:
            curve_eval = exact_tilt_curve(r_eval, j_eval, d_max=float(try_d), n_grid=n_grid)
            eval_actual_max_d = float(try_d)
            break
        except Exception:
            continue
    # curve_eval may cover less than the full d_grid; delta_at_d will return
    # NaN for d values beyond the curve's range.

    base: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "proxy_source": path.meta.get("proxy_source"),
        "n_cal": int(len(cal_idx)),
        "n_eval": int(len(eval_idx)),
        "cal_a_exact": coeff_cal.a_exact,
        "cal_b_theory": coeff_cal.b_theory,
        "cal_a_nu": coeff_cal.a_nu,
        "cal_a_fit": fit_cal.a_fit,
        "cal_b_fit": fit_cal.b_fit,
        "eval_a_exact": coeff_eval.a_exact,
        "eval_b_theory": coeff_eval.b_theory,
        "eval_actual_max_d": eval_actual_max_d,
    }
    if extra:
        base.update(extra)

    rows: List[Dict[str, Any]] = []
    for d_star in d_grid:
        d_star = float(d_star)
        # delta_at_d returns NaN if d_star is beyond the curve's range
        actual = delta_at_d(curve_eval, d_star) if curve_eval is not None else np.nan
        pred_theory = coeff_cal.a_exact * math.sqrt(d_star) + coeff_cal.b_theory * d_star
        pred_aonly = coeff_cal.a_exact * math.sqrt(d_star)
        pred_empfit = fit_cal.a_fit * math.sqrt(d_star) + fit_cal.b_fit * d_star
        pred_anu = coeff_cal.a_nu * math.sqrt(d_star)
        pred_zero = 0.0
        pred_oracle = coeff_eval.a_exact * math.sqrt(d_star) + coeff_eval.b_theory * d_star
        preds = {
            "theory": pred_theory,
            "aonly": pred_aonly,
            "empfit": pred_empfit,
            "anu": pred_anu,
            "zero": pred_zero,
            "oracle": pred_oracle,
        }
        row = dict(base)
        row.update({"d": d_star, "d_suffix": d_suffix(d_star), "actual_delta": actual})
        for name, pred in preds.items():
            err = pred - actual if np.isfinite(actual) else np.nan
            row[f"pred_{name}"] = pred
            row[f"err_{name}"] = err
            row[f"abs_err_{name}"] = abs(err) if np.isfinite(err) else np.nan
            # When actual == 0, sign prediction is undefined; mark as NaN rather
            # than penalizing all non-zero predictors.
            if not np.isfinite(actual) or actual == 0.0:
                row[f"sign_correct_{name}"] = np.nan
            else:
                row[f"sign_correct_{name}"] = float(np.sign(pred) == np.sign(actual))
        rows.append(row)
    return rows


def run_prediction_splits(
    paths: Sequence[PathData],
    d_grid: Sequence[float],
    calibration_frac: float,
    n_repeats: int,
    seed: int,
    empfit_max_d: float,
    n_grid: int = 240,
) -> pd.DataFrame:
    """Generate long-form prediction rows for all paths and repeated splits."""
    rng = np.random.default_rng(seed)
    eval_max_d = max(float(d) for d in d_grid)
    all_rows: List[Dict[str, Any]] = []
    for path in paths:
        n = len(path.item_ids)
        for rep in range(int(n_repeats)):
            cal_idx, eval_idx = split_indices(n, calibration_frac, rng)
            try:
                rows = predict_split_rows(
                    path=path,
                    cal_idx=cal_idx,
                    eval_idx=eval_idx,
                    d_grid=d_grid,
                    eval_max_d=eval_max_d,
                    empfit_max_d=empfit_max_d,
                    n_grid=n_grid,
                    extra={"repeat": rep, "calibration_frac": calibration_frac},
                )
            except Exception as exc:  # keep large sweeps robust to degenerate paths
                print(f"[skip] {path.path_id} repeat={rep}: {exc}")
                continue
            all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("No split rows were produced.")
    return pd.DataFrame(all_rows)


def _cluster_bootstrap_for_group(
    g: pd.DataFrame,
    predictors: Sequence[str],
    cluster_col: str,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Path-level clustered bootstrap CIs for MAE, sign accuracy, and paired diffs."""
    if n_boot <= 0 or cluster_col not in g.columns:
        return {}
    clusters = sorted(g[cluster_col].dropna().unique())
    if len(clusters) <= 1:
        return {}
    rng = np.random.default_rng(seed)
    columns = {}
    for p in predictors:
        if f"abs_err_{p}" in g:
            columns[f"abs_err_{p}"] = f"mae_{p}"
        if f"sign_correct_{p}" in g:
            columns[f"sign_correct_{p}"] = f"sign_acc_{p}"
    if not columns:
        return {}
    grouped = g.groupby(cluster_col)[list(columns)]
    sums = grouped.sum().reindex(clusters).to_numpy(dtype=float)
    counts = grouped.count().reindex(clusters).to_numpy(dtype=float)
    cols = list(columns.values())
    col_index = {c: i for i, c in enumerate(cols)}
    boot = np.empty((n_boot, len(cols)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        # Resample clusters without changing the pooled valid-split estimand.
        denominator = counts[idx].sum(axis=0)
        boot[b] = np.divide(
            sums[idx].sum(axis=0), denominator,
            out=np.full(len(cols), np.nan), where=denominator > 0,
        )
    out: Dict[str, float] = {}
    for i, col in enumerate(cols):
        finite = boot[np.isfinite(boot[:, i]), i]
        if finite.size:
            out[f"{col}_ci_lo"] = float(np.percentile(finite, 2.5))
            out[f"{col}_ci_hi"] = float(np.percentile(finite, 97.5))
        else:
            out[f"{col}_ci_lo"] = np.nan
            out[f"{col}_ci_hi"] = np.nan
    # Paired differences relative to theory; positive means baseline has worse MAE.
    if "mae_theory" in col_index:
        theory_vals = boot[:, col_index["mae_theory"]]
        for p in predictors:
            c = f"mae_{p}"
            if p == "theory" or c not in col_index:
                continue
            diff = boot[:, col_index[c]] - theory_vals
            finite = diff[np.isfinite(diff)]
            if finite.size:
                out[f"mae_diff_{p}_minus_theory_ci_lo"] = float(np.percentile(finite, 2.5))
                out[f"mae_diff_{p}_minus_theory_ci_hi"] = float(np.percentile(finite, 97.5))
            else:
                out[f"mae_diff_{p}_minus_theory_ci_lo"] = np.nan
                out[f"mae_diff_{p}_minus_theory_ci_hi"] = np.nan
    return out


def summarize_prediction_long(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    predictors: Sequence[str] = PREDICTORS,
    cluster_col: str = "path_id",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Summarize long-form prediction rows with optional path-level CIs."""
    rows: List[Dict[str, Any]] = []
    if not group_cols:
        grouped = [((), df)]
    else:
        grouped = df.groupby(list(group_cols), dropna=False)
    for key, g in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: Dict[str, Any] = {col: val for col, val in zip(group_cols, key_tuple)}
        row["n_rows"] = int(len(g))
        row["n_paths"] = int(g[cluster_col].nunique()) if cluster_col in g else np.nan
        valid_actual = g[np.isfinite(g["actual_delta"])].copy()
        row["valid_rows"] = int(len(valid_actual))
        row["valid_paths"] = int(valid_actual[cluster_col].nunique()) if cluster_col in valid_actual else np.nan
        row["nan_actual_rows"] = int(len(g) - len(valid_actual))
        row["actual_delta_mean"] = float(valid_actual["actual_delta"].mean()) if len(valid_actual) else np.nan
        row["actual_abs_delta_mean"] = float(valid_actual["actual_delta"].abs().mean()) if len(valid_actual) else np.nan
        row["actual_positive_rate"] = float((valid_actual["actual_delta"] > 0).mean()) if len(valid_actual) else np.nan
        for p in predictors:
            if f"abs_err_{p}" not in g:
                continue
            row[f"mae_{p}"] = float(g[f"abs_err_{p}"].mean())
            if f"err_{p}" in g:
                finite_err = g.loc[np.isfinite(g[f"err_{p}"]), f"err_{p}"].to_numpy(dtype=float)
                row[f"rmse_{p}"] = float(np.sqrt(np.mean(finite_err ** 2))) if finite_err.size else np.nan
                row[f"bias_{p}"] = float(np.mean(finite_err)) if finite_err.size else np.nan
            else:
                row[f"rmse_{p}"] = np.nan
                row[f"bias_{p}"] = np.nan
            row[f"sign_acc_{p}"] = float(g[f"sign_correct_{p}"].mean()) if f"sign_correct_{p}" in g else np.nan
            if p != "theory" and f"abs_err_theory" in g:
                row[f"mae_diff_{p}_minus_theory"] = float(g[f"abs_err_{p}"].mean() - g["abs_err_theory"].mean())
        if n_boot > 0:
            row.update(_cluster_bootstrap_for_group(g, predictors, cluster_col, n_boot, seed))
        rows.append(row)
    return pd.DataFrame(rows)


def add_overall_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Add __OVERALL__ group labels for summary calls."""
    out = df.copy()
    for col in group_cols:
        if col != "d" and col != "calibration_frac":
            out[col] = "__OVERALL__"
    return out


def save_markdown(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    try:
        p.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        # tabulate is optional; fall back to a plain text representation.
        p.write_text(df.to_string(index=False), encoding="utf-8")
