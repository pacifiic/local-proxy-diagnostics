"""  [exp30]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from common_experiment_utils import (
    PathData,
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
    mkdir,
    parse_cols,
    pop_cov,
    pop_var,
    spearman_corr,
    theory_coefficients,
)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xv = x - x.mean()
    yv = y - y.mean()
    denom = math.sqrt(float(np.sum(xv**2) * np.sum(yv**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)



def aggregate_proxy_if_needed(proxy_df: pd.DataFrame, item_col: str, score_col: str, group_cols: Sequence[str]) -> pd.DataFrame:
    cols = [*group_cols, item_col, score_col]
    missing = [c for c in cols if c not in proxy_df.columns]
    if missing:
        raise ValueError(f"proxy csv is missing required columns: {missing}")
    return (
        proxy_df[[*group_cols, item_col, score_col]]
        .groupby([*group_cols, item_col], dropna=False)[score_col]
        .mean()
        .reset_index()
    )



def build_model_paths(args: argparse.Namespace) -> List[PathData]:
    human = pd.read_csv(args.human_long_csv)
    proxy = pd.read_csv(args.proxy_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    proxy = aggregate_proxy_if_needed(proxy, args.proxy_item_col, args.proxy_score_col, group_cols)
    if args.proxy_item_col != "item_id":
        proxy = proxy.rename(columns={args.proxy_item_col: "item_id"})
    score_col = args.proxy_score_col
    if score_col != "proxy_score":
        proxy = proxy.rename(columns={score_col: "proxy_score"})
        score_col = "proxy_score"
    return list(
        generate_model_paths(
            human_long=human,
            proxy_df=proxy,
            item_cols=item_cols,
            score_col=args.score_col,
            group_cols=group_cols,
            proxy_item_col="item_id",
            proxy_score_col=score_col,
            min_items=args.min_items,
        )
    )



def build_human_compare_paths(args: argparse.Namespace) -> List[PathData]:
    if not args.human_compare:
        return []
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.human_compare_group_cols or args.group_cols)
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



def top_idx(x: np.ndarray, frac: float) -> np.ndarray:
    n = len(x)
    k = max(1, int(math.ceil(frac * n)))
    order = np.argsort(x, kind="mergesort")
    return np.sort(order[-k:])



def iqr(x: np.ndarray) -> float:
    q75, q25 = np.quantile(x, [0.75, 0.25])
    return float(q75 - q25)



def tail_metrics(r: np.ndarray, j: np.ndarray, frac: float) -> Dict[str, float]:
    idx_r = top_idx(r, frac)
    idx_j = top_idx(j, frac)
    baseline = float(np.mean(j))
    k = len(idx_r)
    overlap = len(set(idx_r.tolist()) & set(idx_j.tolist())) / float(k)
    proxy_tail_gold = float(np.mean(j[idx_r]))
    gold_tail_gold = float(np.mean(j[idx_j]))
    return {
        f"tail_q_{str(frac).replace('.', 'p')}_gold_delta": proxy_tail_gold - baseline,
        f"tail_q_{str(frac).replace('.', 'p')}_oracle_gap": gold_tail_gold - proxy_tail_gold,
        f"tail_q_{str(frac).replace('.', 'p')}_overlap": overlap,
        f"tail_q_{str(frac).replace('.', 'p')}_proxy_mean": float(np.mean(r[idx_r])) - float(np.mean(r)),
    }



def path_stats(path: PathData, fit_max_d: float, top_fracs: Sequence[float]) -> Dict[str, Any]:
    r = np.asarray(path.proxy_scores, dtype=float)
    j = np.asarray(path.gold_scores, dtype=float)
    coeff = theory_coefficients(j, r)
    curve = exact_tilt_curve(r, j, d_max=fit_max_d, n_grid=200)
    fit = fit_local_law(curve.d, curve.q, fit_max_d=fit_max_d)

    row: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "proxy_source": path.meta.get("proxy_source"),
        "n_items": len(r),
        "proxy_std": float(np.std(r, ddof=0)),
        "proxy_iqr": iqr(r),
        "proxy_p90_p10": float(np.quantile(r, 0.90) - np.quantile(r, 0.10)),
        "proxy_unique_fraction": float(pd.Series(r).nunique(dropna=False) / len(r)),
        "proxy_tie_rate": 1.0 - float(pd.Series(r).nunique(dropna=False) / len(r)),
        "gold_std": float(np.std(j, ddof=0)),
        "spearman_r_j": spearman_corr(r, j),
        "pearson_r_j": pearson_corr(r, j),
        "sigma_j2": coeff.sigma_j2,
        "sigma_r2": coeff.sigma_r2,
        "rho": coeff.rho,
        "rho_over_sigma_j2": float(coeff.rho / coeff.sigma_j2) if coeff.sigma_j2 > 0 else float("nan"),
        "a_exact": coeff.a_exact,
        "a_nu": coeff.a_nu,
        "b_theory": coeff.b_theory,
        "kappa_r": coeff.kappa_r,
        "kappa_jrr": coeff.kappa_jrr,
        "a_fit": fit.a_fit,
        "b_fit": fit.b_fit,
        "fit_r2": fit.r2,
        "delta_at_0p1": float(np.interp(min(fit_max_d, curve.d[-1]), curve.d, curve.q) - curve.q[0]),
        "min_var_pi_t": float(np.min(curve.var_r)),
        "min_ess": float(np.min(curve.ess)),
    }
    for frac in top_fracs:
        row.update(tail_metrics(r, j, frac))
    return row



def summarize_rows(df: pd.DataFrame, group_key_name: str = "group_key") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        c
        for c in df.columns
        if c not in {group_key_name, "path_id", "proxy_source"}
    ]
    for gk, gdf in df.groupby(group_key_name, dropna=False):
        row: Dict[str, Any] = {group_key_name: gk, "n_paths": len(gdf)}
        for c in metric_cols:
            row[f"median_{c}"] = gdf[c].median()
            row[f"mean_{c}"] = gdf[c].mean()
        rows.append(row)
    overall: Dict[str, Any] = {group_key_name: "__OVERALL__", "n_paths": len(df)}
    for c in metric_cols:
        overall[f"median_{c}"] = df[c].median()
        overall[f"mean_{c}"] = df[c].mean()
    rows.append(overall)
    return pd.DataFrame(rows)



def compare_model_to_human(model_summary: pd.DataFrame, human_summary: pd.DataFrame) -> pd.DataFrame:
    ms = model_summary.copy()
    hs = human_summary.copy()
    ms = ms[ms["group_key"] != "__OVERALL__"]
    hs = hs[hs["group_key"] != "__OVERALL__"]
    joined = ms.merge(hs, on="group_key", how="inner", suffixes=("_model", "_human"))
    out: List[Dict[str, Any]] = []
    metrics = [
        ("median_proxy_std", "proxy_std_ratio_model_over_human"),
        ("median_proxy_p90_p10", "proxy_p90_p10_ratio_model_over_human"),
        ("median_spearman_r_j", "delta_spearman_model_minus_human"),
        ("median_a_exact", "delta_a_exact_model_minus_human"),
        ("median_b_theory", "delta_b_theory_model_minus_human"),
        ("median_tail_q_0p1_gold_delta", "delta_tail_top10_model_minus_human"),
        ("median_tail_q_0p05_oracle_gap", "delta_oracle_gap_top5_model_minus_human"),
        ("median_tail_q_0p1_overlap", "delta_top10_overlap_model_minus_human"),
    ]
    for _, row in joined.iterrows():
        rec = {"group_key": row["group_key"]}
        for col, name in metrics:
            mcol = f"{col}_model"
            hcol = f"{col}_human"
            mv = row.get(mcol, np.nan)
            hv = row.get(hcol, np.nan)
            if "ratio" in name:
                rec[name] = float(mv / hv) if pd.notna(mv) and pd.notna(hv) and hv != 0 else np.nan
            else:
                rec[name] = float(mv - hv) if pd.notna(mv) and pd.notna(hv) else np.nan
        out.append(rec)
    return pd.DataFrame(out)



def summarize_exp31_margins(
    paths_df: pd.DataFrame,
    label: str,
    include_clarified_sign_columns: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    d_suffix = "0p1"
    pred_col = f"pred_theory_delta_{d_suffix}"
    act_col = f"actual_eval_delta_{d_suffix}"

    def _fixed_gain_sign_acc(df: pd.DataFrame, mask: pd.Series | None = None) -> float:
        if pred_col not in df.columns or act_col not in df.columns:
            return np.nan
        sub = df if mask is None else df.loc[mask]
        if sub.empty:
            return np.nan
        return float((np.sign(sub[pred_col]) == np.sign(sub[act_col])).mean())

    for gk, gdf in paths_df.groupby("group_key", dropna=False):
        row = {
            "source": label,
            "group_key": gk,
            "n_splits": len(gdf),
            "theory_sign_acc": gdf["theory_sign_correct"].mean(),
            "empfit_sign_acc": gdf["empfit_sign_correct"].mean(),
            "median_abs_actual_delta_0p1": gdf[act_col].abs().median(),
            "mean_abs_actual_delta_0p1": gdf[act_col].abs().mean(),
            "corr_pred_vs_actual_0p1": gdf[[pred_col, act_col]].corr().iloc[0, 1],
            "frac_reversal_0p1": gdf["reversal_by_0p1"].mean(),
        }
        if include_clarified_sign_columns:
            row.update(
                {
                    "b_sign_acc_theory": gdf["theory_sign_correct"].mean(),
                    "b_sign_acc_empfit": gdf["empfit_sign_correct"].mean(),
                    "fixed_gain_sign_acc_theory_0p1": _fixed_gain_sign_acc(gdf),
                }
            )
        for eps in (0.005, 0.01, 0.02):
            mask = gdf[act_col].abs() > eps
            tag = str(eps).replace(".", "p")
            row[f"frac_abs_actual_gt_{tag}"] = mask.mean()
            row[f"theory_sign_acc_abs_actual_gt_{tag}"] = gdf.loc[mask, "theory_sign_correct"].mean() if mask.any() else np.nan
            if include_clarified_sign_columns:
                row[f"fixed_gain_sign_acc_theory_0p1_abs_actual_gt_{tag}"] = _fixed_gain_sign_acc(gdf, mask)
        rows.append(row)
    overall = {
        "source": label,
        "group_key": "__OVERALL__",
        "n_splits": len(paths_df),
        "theory_sign_acc": paths_df["theory_sign_correct"].mean(),
        "empfit_sign_acc": paths_df["empfit_sign_correct"].mean(),
        "median_abs_actual_delta_0p1": paths_df[act_col].abs().median(),
        "mean_abs_actual_delta_0p1": paths_df[act_col].abs().mean(),
        "corr_pred_vs_actual_0p1": paths_df[[pred_col, act_col]].corr().iloc[0, 1],
        "frac_reversal_0p1": paths_df["reversal_by_0p1"].mean(),
    }
    if include_clarified_sign_columns:
        overall.update(
            {
                "b_sign_acc_theory": paths_df["theory_sign_correct"].mean(),
                "b_sign_acc_empfit": paths_df["empfit_sign_correct"].mean(),
                "fixed_gain_sign_acc_theory_0p1": _fixed_gain_sign_acc(paths_df),
            }
        )
    for eps in (0.005, 0.01, 0.02):
        mask = paths_df[act_col].abs() > eps
        tag = str(eps).replace(".", "p")
        overall[f"frac_abs_actual_gt_{tag}"] = mask.mean()
        overall[f"theory_sign_acc_abs_actual_gt_{tag}"] = paths_df.loc[mask, "theory_sign_correct"].mean() if mask.any() else np.nan
        if include_clarified_sign_columns:
            overall[f"fixed_gain_sign_acc_theory_0p1_abs_actual_gt_{tag}"] = _fixed_gain_sign_acc(paths_df, mask)
    rows.append(overall)
    return pd.DataFrame(rows)



def build_markdown_report(
    model_summary: pd.DataFrame,
    compare_df: pd.DataFrame,
    margin_df: Optional[pd.DataFrame],
    out_path: Path,
) -> None:
    overall = model_summary.loc[model_summary["group_key"] == "__OVERALL__"]
    lines = ["# GPT proxy diagnostic report", ""]
    if not overall.empty:
        o = overall.iloc[0]
        lines += [
            "## Overall model-proxy profile",
            f"- Median proxy std: **{o.get('median_proxy_std', float('nan')):.4f}**",
            f"- Median Spearman(proxy, gold): **{o.get('median_spearman_r_j', float('nan')):.4f}**",
            f"- Median top-10% gold delta: **{o.get('median_tail_q_0p1_gold_delta', float('nan')):.4f}**",
            f"- Median top-10% overlap with gold top set: **{o.get('median_tail_q_0p1_overlap', float('nan')):.4f}**",
            "",
        ]
    if compare_df is not None and not compare_df.empty:
        lines += ["## Model vs human comparison (group-level)", ""]
        for _, row in compare_df.iterrows():
            lines.append(
                f"- **{row['group_key']}**: std ratio={row.get('proxy_std_ratio_model_over_human', float('nan')):.3f}, "
                f"Δspearman={row.get('delta_spearman_model_minus_human', float('nan')):.3f}, "
                f"Δtop10 overlap={row.get('delta_top10_overlap_model_minus_human', float('nan')):.3f}"
            )
        lines.append("")
    if margin_df is not None and not margin_df.empty:
        lines += ["## Predictive-margin analysis", ""]
        for _, row in margin_df.iterrows():
            if row["group_key"] != "__OVERALL__":
                continue
            lines.append(
                f"- **{row['source']}** overall: sign acc={row.get('theory_sign_acc', float('nan')):.3f}, "
                f"median |actual Δ(0.1)|={row.get('median_abs_actual_delta_0p1', float('nan')):.4f}, "
                f"sign acc when |actual Δ|>0.01 = {row.get('theory_sign_acc_abs_actual_gt_0p01', float('nan')):.3f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze GPT/model proxy ranking, tail, and margin diagnostics.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--proxy-csv", required=True, help="Raw or aggregated model proxy CSV.")
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--human-compare", action="store_true", help="Also build human held-out paths for comparison using the same human long CSV.")
    ap.add_argument("--human-compare-group-cols", default="", help="Optional group cols for human compare paths; defaults to --group-cols.")
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--top-fracs", default="0.01,0.05,0.10")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--exp31-model-paths", default="", help="Optional exp31 model-path CSV for margin analysis.")
    ap.add_argument("--exp31-human-paths", default="", help="Optional exp31 human-path CSV for margin analysis.")
    ap.add_argument(
        "--clarified-sign-columns",
        action="store_true",
        help="Add explicit B-sign and fixed-gain sign columns while keeping legacy sign columns.",
    )
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    top_fracs = [float(x) for x in parse_cols(args.top_fracs)]

    model_paths = build_model_paths(args)
    model_rows = [path_stats(p, args.fit_max_d, top_fracs) for p in model_paths]
    model_df = pd.DataFrame(model_rows)
    model_summary = summarize_rows(model_df)
    model_df.to_csv(outdir / "exp30_model_proxy_stats_paths.csv", index=False)
    model_summary.to_csv(outdir / "exp30_model_proxy_stats_summary.csv", index=False)

    compare_df = pd.DataFrame()
    human_df = pd.DataFrame()
    human_summary = pd.DataFrame()
    if args.human_compare:
        human_paths = build_human_compare_paths(args)
        human_rows = [path_stats(p, args.fit_max_d, top_fracs) for p in human_paths]
        human_df = pd.DataFrame(human_rows)
        human_summary = summarize_rows(human_df)
        human_df.to_csv(outdir / "exp30_human_proxy_stats_paths.csv", index=False)
        human_summary.to_csv(outdir / "exp30_human_proxy_stats_summary.csv", index=False)
        compare_df = compare_model_to_human(model_summary, human_summary)
        compare_df.to_csv(outdir / "exp30_model_vs_human_compare.csv", index=False)

    margin_frames: List[pd.DataFrame] = []
    if args.exp31_model_paths:
        exp31_model = pd.read_csv(args.exp31_model_paths)
        margin_frames.append(
            summarize_exp31_margins(
                exp31_model,
                label="model_proxy",
                include_clarified_sign_columns=args.clarified_sign_columns,
            )
        )
    if args.exp31_human_paths:
        exp31_human = pd.read_csv(args.exp31_human_paths)
        margin_frames.append(
            summarize_exp31_margins(
                exp31_human,
                label="human_baseline",
                include_clarified_sign_columns=args.clarified_sign_columns,
            )
        )
    margin_df = pd.concat(margin_frames, ignore_index=True) if margin_frames else pd.DataFrame()
    if not margin_df.empty:
        margin_df.to_csv(outdir / "exp30_margin_summary.csv", index=False)

    build_markdown_report(model_summary, compare_df, margin_df if not margin_df.empty else None, outdir / "exp30_report.md")

    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv,
        "group_cols": parse_cols(args.group_cols),
        "fit_max_d": args.fit_max_d,
        "top_fracs": top_fracs,
        "n_model_paths": len(model_paths),
        "human_compare": bool(args.human_compare),
        "exp31_model_paths": args.exp31_model_paths or None,
        "exp31_human_paths": args.exp31_human_paths or None,
    }
    (outdir / "exp30_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Saved model-proxy path stats to {outdir / 'exp30_model_proxy_stats_paths.csv'}")
    if args.human_compare:
        print(f"Saved model-vs-human comparison to {outdir / 'exp30_model_vs_human_compare.csv'}")
    if not margin_df.empty:
        print(f"Saved margin analysis to {outdir / 'exp30_margin_summary.csv'}")
    print(f"Saved markdown report to {outdir / 'exp30_report.md'}")


if __name__ == "__main__":
    main()
