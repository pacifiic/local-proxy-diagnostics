"""  [exp27]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from common_experiment_utils import (
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_one_split(
    path: PathData,
    cal_idx: np.ndarray,
    eva_idx: np.ndarray,
    fit_max_d: float,
    d_targets: List[float],
) -> Dict[str, Any]:
    r_cal = path.proxy_scores[cal_idx]
    j_cal = path.gold_scores[cal_idx]
    r_eva = path.proxy_scores[eva_idx]
    j_eva = path.gold_scores[eva_idx]

    coeff_cal = theory_coefficients(j_cal, r_cal)
    curve_cal = exact_tilt_curve(r_cal, j_cal, d_max=fit_max_d)
    fit_cal = fit_local_law(curve_cal.d, curve_cal.q, fit_max_d=fit_max_d)

    curve_eva = exact_tilt_curve(r_eva, j_eva, d_max=fit_max_d)
    fit_eva = fit_local_law(curve_eva.d, curve_eva.q, fit_max_d=fit_max_d)
    coeff_eva = theory_coefficients(j_eva, r_eva)

    row: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "proxy_source": path.meta.get("proxy_source"),
        "n_cal": len(cal_idx),
        "n_eval": len(eva_idx),
        "cal_a_exact": coeff_cal.a_exact,
        "cal_b_theory": coeff_cal.b_theory,
        "cal_a_nu": coeff_cal.a_nu,
        "cal_a_fit": fit_cal.a_fit,
        "cal_b_fit": fit_cal.b_fit,
        "eval_a_theory": coeff_eva.a_exact,
        "eval_b_theory": coeff_eva.b_theory,
        "eval_a_fit": fit_eva.a_fit,
        "eval_b_fit": fit_eva.b_fit,
        "eval_r2": fit_eva.r2,
        "cal_spearman_r_j": spearman_corr(r_cal, j_cal),
        "eval_spearman_r_j": spearman_corr(r_eva, j_eva),
        "theory_sign_correct": int(np.sign(coeff_cal.b_theory) == np.sign(fit_eva.b_fit)),
        "empfit_sign_correct": int(np.sign(fit_cal.b_fit) == np.sign(fit_eva.b_fit)),
    }

    cal_peak_d_theory, cal_headroom_theory = local_peak_from_coefficients(coeff_cal.a_exact, coeff_cal.b_theory)
    cal_peak_d_emp, cal_headroom_emp = local_peak_from_coefficients(fit_cal.a_fit, fit_cal.b_fit)
    eva_peak_d_fit, eva_headroom_fit = local_peak_from_coefficients(fit_eva.a_fit, fit_eva.b_fit)
    row.update(
        {
            "cal_peak_d_theory": cal_peak_d_theory,
            "cal_headroom_theory": cal_headroom_theory,
            "cal_peak_d_empfit": cal_peak_d_emp,
            "cal_headroom_empfit": cal_headroom_emp,
            "eval_peak_d_fit": eva_peak_d_fit,
            "eval_headroom_fit": eva_headroom_fit,
        }
    )

    for d_star in d_targets:
        suffix = str(d_star).replace(".", "p")
        theory_pred = coeff_cal.a_exact * np.sqrt(d_star) + coeff_cal.b_theory * d_star
        anu_pred = coeff_cal.a_nu * np.sqrt(d_star)
        empfit_pred = fit_cal.a_fit * np.sqrt(d_star) + fit_cal.b_fit * d_star
        actual = delta_at_d(curve_eva, d_star)
        # Additional baselines
        a_only_pred = coeff_cal.a_exact * np.sqrt(d_star)  # A*sqrt(d) only, no B
        zero_pred = 0.0
        oracle_pred = coeff_eva.a_exact * np.sqrt(d_star) + coeff_eva.b_theory * d_star

        row[f"pred_theory_delta_{suffix}"] = theory_pred
        row[f"pred_anu_delta_{suffix}"] = anu_pred
        row[f"pred_empfit_delta_{suffix}"] = empfit_pred
        row[f"pred_aonly_delta_{suffix}"] = a_only_pred
        row[f"pred_zero_delta_{suffix}"] = zero_pred
        row[f"pred_oracle_delta_{suffix}"] = oracle_pred
        row[f"actual_eval_delta_{suffix}"] = actual
        if not np.isnan(actual):
            row[f"abs_err_theory_{suffix}"] = abs(theory_pred - actual)
            row[f"abs_err_anu_{suffix}"] = abs(anu_pred - actual)
            row[f"abs_err_empfit_{suffix}"] = abs(empfit_pred - actual)
            row[f"abs_err_aonly_{suffix}"] = abs(a_only_pred - actual)
            row[f"abs_err_zero_{suffix}"] = abs(zero_pred - actual)
            row[f"abs_err_oracle_{suffix}"] = abs(oracle_pred - actual)
            row[f"reversal_by_{suffix}"] = int(actual < 0)
            row[f"theory_reversal_pred_{suffix}"] = int(theory_pred < 0)
            row[f"anu_reversal_pred_{suffix}"] = int(anu_pred < 0)
            row[f"empfit_reversal_pred_{suffix}"] = int(empfit_pred < 0)

    return row


def _bootstrap_mae_ci(
    df: pd.DataFrame,
    d_targets: List[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Path-level bootstrap 95% CIs for MAE and paired MAE differences (A_nu - theory).

    Resamples paths (independent units) rather than individual split rows to
    respect within-path correlation across split repetitions.
    Positive mae_diff_anu_minus_theory means A_nu has higher (worse) MAE.
    """
    rng = np.random.default_rng(seed)

    err_cols = [c for c in df.columns if c.startswith("abs_err_")]
    sign_col = "theory_sign_correct"
    agg_dict: Dict[str, str] = {c: "mean" for c in err_cols}
    if sign_col in df.columns:
        agg_dict[sign_col] = "mean"

    path_df = df.groupby("path_id").agg(agg_dict)
    mat = path_df.values
    col_names = list(path_df.columns)
    col_index = {c: i for i, c in enumerate(col_names)}
    n_paths = len(mat)

    boot_means = np.empty((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_paths, size=n_paths)
        boot_means[b] = mat[idx].mean(axis=0)

    result: Dict[str, float] = {}
    for i, col in enumerate(col_names):
        result[f"{col}_ci_lo"] = float(np.percentile(boot_means[:, i], 2.5))
        result[f"{col}_ci_hi"] = float(np.percentile(boot_means[:, i], 97.5))

    for d_star in d_targets:
        suffix = str(d_star).replace(".", "p")
        col_t = f"abs_err_theory_{suffix}"
        col_a = f"abs_err_anu_{suffix}"
        if col_t in col_index and col_a in col_index:
            diffs = boot_means[:, col_index[col_a]] - boot_means[:, col_index[col_t]]
            result[f"mae_diff_anu_minus_theory_{suffix}_ci_lo"] = float(np.percentile(diffs, 2.5))
            result[f"mae_diff_anu_minus_theory_{suffix}_ci_hi"] = float(np.percentile(diffs, 97.5))

    return result


def summarize_results(df: pd.DataFrame, d_targets: List[float], n_boot: int = 1000) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_key, gdf in df.groupby("group_key", dropna=False):
        row: Dict[str, Any] = {
            "group_key": group_key,
            "n_splits": len(gdf),
            "theory_sign_acc": gdf["theory_sign_correct"].mean(),
            "empfit_sign_acc": gdf["empfit_sign_correct"].mean(),
            "eval_r2_median": gdf["eval_r2"].median(),
        }
        for d_star in d_targets:
            suffix = str(d_star).replace(".", "p")
            for name in ["theory", "anu", "empfit", "aonly", "zero", "oracle"]:
                col = f"abs_err_{name}_{suffix}"
                if col in gdf:
                    row[f"mae_{name}_{suffix}"] = gdf[col].mean()
            act = f"actual_eval_delta_{suffix}"
            if act in gdf:
                row[f"actual_delta_mean_{suffix}"] = gdf[act].mean()
        if n_boot > 0 and "path_id" in gdf.columns:
            row.update(_bootstrap_mae_ci(gdf, d_targets, n_boot=n_boot))
        rows.append(row)

    overall: Dict[str, Any] = {
        "group_key": "__OVERALL__",
        "n_splits": len(df),
        "theory_sign_acc": df["theory_sign_correct"].mean(),
        "empfit_sign_acc": df["empfit_sign_correct"].mean(),
        "eval_r2_median": df["eval_r2"].median(),
    }
    for d_star in d_targets:
        suffix = str(d_star).replace(".", "p")
        for name in ["theory", "anu", "empfit", "aonly", "zero", "oracle"]:
            col = f"abs_err_{name}_{suffix}"
            if col in df:
                overall[f"mae_{name}_{suffix}"] = df[col].mean()
        act = f"actual_eval_delta_{suffix}"
        if act in df:
            overall[f"actual_delta_mean_{suffix}"] = df[act].mean()
    if n_boot > 0 and "path_id" in df.columns:
        overall.update(_bootstrap_mae_ci(df, d_targets, n_boot=n_boot))
    rows.append(overall)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prospective prediction experiment for local optimization geometry.")
    ap.add_argument("--human-long-csv", required=True, help="Human long-format ratings CSV.")
    ap.add_argument("--item-cols", default="item_id", help="Columns that define a unique item.")
    ap.add_argument("--group-cols", default="", help="Grouping columns.")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="", help="Optional item-level model proxy CSV. If omitted, uses held-out human proxy paths.")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--d-targets", default="0.05,0.10")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap resamples for MAE CIs (0 = skip).")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    d_targets = [float(x) for x in parse_cols(args.d_targets)]

    if args.proxy_csv:
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

    rng = np.random.default_rng(args.random_seed)
    rows: List[Dict[str, Any]] = []
    n_skipped_splits = 0
    for path in paths:
        n = len(path.item_ids)
        if n < 30:
            continue
        for rep in range(args.n_repeats):
            cal_idx, eva_idx = split_indices(n, args.calibration_frac, rng)
            try:
                row = evaluate_one_split(path, cal_idx, eva_idx, args.fit_max_d, d_targets)
            except (RuntimeError, ValueError):
                n_skipped_splits += 1
                continue
            row["split_rep"] = rep
            rows.append(row)

    if n_skipped_splits > 0:
        total_attempted = sum(args.n_repeats for p in paths if len(p.item_ids) >= 30)
        print(f"Skipped {n_skipped_splits}/{total_attempted} splits (tilt curve or fit infeasible)")

    paths_df = pd.DataFrame(rows)
    paths_csv = outdir / "exp27_prospective_paths.csv"
    paths_df.to_csv(paths_csv, index=False)

    summary_df = summarize_results(paths_df, d_targets, n_boot=args.n_boot)
    summary_csv = outdir / "exp27_prospective_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    spec = {
        "human_long_csv": args.human_long_csv,
        "human_long_sha256": sha256_file(args.human_long_csv),
        "proxy_csv": args.proxy_csv or None,
        "proxy_csv_sha256": sha256_file(args.proxy_csv) if args.proxy_csv else None,
        "proxy_item_col": args.proxy_item_col,
        "proxy_score_col": args.proxy_score_col,
        "group_cols": group_cols,
        "item_cols": item_cols,
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "calibration_frac": args.calibration_frac,
        "n_repeats": args.n_repeats,
        "fit_max_d": args.fit_max_d,
        "d_targets": d_targets,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "random_seed": args.random_seed,
        "n_boot": args.n_boot,
        "n_paths": len(paths),
        "n_attempted_splits": sum(args.n_repeats for p in paths if len(p.item_ids) >= 30),
        "n_rows": len(paths_df),
        "n_skipped_splits": n_skipped_splits,
    }
    (outdir / "exp27_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved path-level results to {paths_csv}")
    print(f"Saved summary results to {summary_csv}")


if __name__ == "__main__":
    main()
