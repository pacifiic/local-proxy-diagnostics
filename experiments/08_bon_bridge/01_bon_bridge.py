from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from common_experiment_utils import (
    PathData,
    delta_at_d,
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
    mkdir,
    parse_cols,
    theory_coefficients,
)


@dataclass
class BoNPoint:
    N: int
    d_bon: float
    q_bon: float
    delta_bon: float
    selected_proxy_mean: float
    p_max: float
    n_support: int


def aggregate_proxy_if_needed(
    proxy_df: pd.DataFrame,
    item_col: str,
    score_col: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    cols = [*group_cols, item_col, score_col]
    missing = [c for c in cols if c not in proxy_df.columns]
    if missing:
        raise ValueError(f"proxy csv is missing required columns: {missing}")
    out = (
        proxy_df[[*group_cols, item_col, score_col]]
        .groupby([*group_cols, item_col], dropna=False)[score_col]
        .mean()
        .reset_index()
    )
    return out


# ---------- BoN distribution helpers ----------

def bon_distribution_uniform_with_ties(r: np.ndarray, N: int) -> np.ndarray:
    """  [exp29]Exact selected-item distribution for best-of-N with replacement.

    The base distribution is uniform over items. Tie-breaking is uniform among
    all items that achieve the maximum proxy score in a draw.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    r = np.asarray(r, dtype=float)
    m = r.size
    if m == 0:
        raise ValueError("empty score vector")

    order = np.argsort(r, kind="mergesort")
    rs = r[order]
    uniq, idx_start, counts = np.unique(rs, return_index=True, return_counts=True)
    cum = np.cumsum(counts).astype(float)
    cum_prev = np.concatenate([[0.0], cum[:-1]])
    p_group = (cum / m) ** N - (cum_prev / m) ** N

    ps = np.zeros(m, dtype=float)
    for start, cnt, pg in zip(idx_start, counts, p_group):
        ps[start : start + cnt] = pg / float(cnt)

    p = np.zeros(m, dtype=float)
    p[order] = ps
    p /= p.sum()
    return p



def kl_to_uniform_from_distribution(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    m = p.size
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] * m)))



def bon_points_for_path(r: np.ndarray, j: np.ndarray, N_list: Sequence[int]) -> List[BoNPoint]:
    r = np.asarray(r, dtype=float)
    j = np.asarray(j, dtype=float)
    q0 = float(np.mean(j))
    out: List[BoNPoint] = []
    for N in N_list:
        p = bon_distribution_uniform_with_ties(r, int(N))
        q = float(np.sum(p * j))
        d = kl_to_uniform_from_distribution(p)
        mu_r = float(np.sum(p * r))
        out.append(
            BoNPoint(
                N=int(N),
                d_bon=d,
                q_bon=q,
                delta_bon=q - q0,
                selected_proxy_mean=mu_r,
                p_max=float(np.max(p)),
                n_support=int(np.sum(p > 0)),
            )
        )
    return out


# ---------- path building ----------

def build_paths(args: argparse.Namespace) -> List[PathData]:
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)

    if args.proxy_csv:
        proxy = pd.read_csv(args.proxy_csv)
        proxy = aggregate_proxy_if_needed(proxy, args.proxy_item_col, args.proxy_score_col, group_cols)
        score_col = args.proxy_score_col
        if score_col != "proxy_score":
            proxy = proxy.rename(columns={score_col: "proxy_score"})
            score_col = "proxy_score"
        item_col = args.proxy_item_col
        if item_col != "item_id":
            proxy = proxy.rename(columns={item_col: "item_id"})
            item_col = "item_id"
        return list(
            generate_model_paths(
                human_long=human,
                proxy_df=proxy,
                item_cols=item_cols,
                score_col=args.score_col,
                group_cols=group_cols,
                proxy_item_col=item_col,
                proxy_score_col=score_col,
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


# ---------- summary helpers ----------

def safe_corr(x: pd.Series, y: pd.Series) -> float:
    vals = pd.concat([x, y], axis=1).dropna()
    if len(vals) < 2:
        return float("nan")
    if vals.iloc[:, 0].nunique() < 2 or vals.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(vals.iloc[:, 0].corr(vals.iloc[:, 1]))



def classify_window_status(n_nontrivial_points: int) -> str:
    if n_nontrivial_points <= 0:
        return "nontrivial_empty"
    if n_nontrivial_points == 1:
        return "nontrivial_sparse"
    return "nontrivial_ok"



def first_matching_N(df: pd.DataFrame, predicate_col: str) -> float:
    g = df.loc[df[predicate_col] == 1, "N"]
    if g.empty:
        return float("nan")
    return float(g.min())



def make_markdown_report(
    outdir: Path,
    path_summary_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    shared_fit_max_d: float,
    bridge_max_d: float,
) -> None:
    overall = path_summary_df.loc[path_summary_df["group_key"] == "__OVERALL__"].head(1)
    if overall.empty:
        text = "# BoN bridge report\n\nNo paths were processed.\n"
        (outdir / "exp29_bridge_report.md").write_text(text, encoding="utf-8")
        return
    o = overall.iloc[0]
    n2 = ladder_df.loc[ladder_df["N"] == 2]
    n2_med = float(n2["median_d_bon"].iloc[0]) if not n2.empty else float("nan")
    text = f"""# BoN bridge report

## Executive summary

This run uses two KL windows:
- shared local-law window: `d <= {shared_fit_max_d:.3f}`
- BoN bridge window: `d <= {bridge_max_d:.3f}`

Across all processed paths:
- median share of **all** BoN points in the shared window: `{o['median_share_all_in_shared_window']:.3f}`
- median share of **nontrivial** BoN points in the shared window: `{o['median_share_nontrivial_in_shared_window']:.3f}`
- median share of **nontrivial** BoN points in the bridge window: `{o['median_share_nontrivial_in_bridge_window']:.3f}`
- fraction of paths with **no nontrivial** BoN points inside the shared window: `{o['frac_protocol_mismatch_shared']:.3f}`
- median KL of the first nontrivial BoN step (`N=2`): `{n2_med:.3f}`

## Interpretation

If the first nontrivial BoN step already lands above the shared local-law window, then a lack of shared-window coverage is best interpreted as a **protocol mismatch between discrete BoN selection and the chosen local KL window**, rather than as evidence against the local law itself.

The bridge-window columns are designed to make this distinction explicit.
"""
    (outdir / "exp29_bridge_report.md").write_text(text, encoding="utf-8")


# ---------- main path computation ----------

def path_to_rows(
    path: PathData,
    N_list: Sequence[int],
    shared_fit_max_d: float,
    bridge_max_d: float,
    var_eps: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    proxy = np.asarray(path.proxy_scores, dtype=float)
    gold = np.asarray(path.gold_scores, dtype=float)
    n_items = len(path.item_ids)
    n_unique_proxy = int(pd.Series(proxy).nunique(dropna=False))
    proxy_var = float(np.var(proxy))
    base_q = float(np.mean(gold))
    bon_points = bon_points_for_path(proxy, gold, N_list)

    # Always keep the path, even if theory/fit becomes undefined.
    status_bits: List[str] = []
    coeff = None
    fit_shared = None
    fit_bridge = None
    curve = None

    if proxy_var <= var_eps or n_unique_proxy < 2:
        status_bits.append("degenerate_proxy_variance")
    else:
        coeff = theory_coefficients(gold, proxy)
        d_curve_max = max(bridge_max_d, max(x.d_bon for x in bon_points) + 1e-9)
        curve = exact_tilt_curve(proxy, gold, d_max=d_curve_max, n_grid=400)
        try:
            fit_shared = fit_local_law(curve.d, curve.q, fit_max_d=shared_fit_max_d)
        except Exception:
            status_bits.append("shared_fit_failed")
        try:
            fit_bridge = fit_local_law(curve.d, curve.q, fit_max_d=bridge_max_d)
        except Exception:
            status_bits.append("bridge_fit_failed")

    rows: List[Dict[str, Any]] = []
    for bp in bon_points:
        d = bp.d_bon
        in_shared = int(d <= shared_fit_max_d + 1e-12)
        in_bridge = int(d <= bridge_max_d + 1e-12)
        nontrivial = int(bp.N > 1)
        theory_shared = float("nan")
        empfit_shared = float("nan")
        empfit_bridge = float("nan")
        tilt_delta = float("nan")

        if coeff is not None:
            theory_shared = coeff.a_exact * math.sqrt(max(d, 0.0)) + coeff.b_theory * d
        if fit_shared is not None:
            empfit_shared = fit_shared.a_fit * math.sqrt(max(d, 0.0)) + fit_shared.b_fit * d
        if fit_bridge is not None:
            empfit_bridge = fit_bridge.a_fit * math.sqrt(max(d, 0.0)) + fit_bridge.b_fit * d
        if curve is not None and d <= float(curve.d[-1]):
            tilt_delta = delta_at_d(curve, d)

        rows.append(
            {
                "group_key": path.group_key,
                "path_id": path.path_id,
                "proxy_source": path.meta.get("proxy_source"),
                "n_items": n_items,
                "n_unique_proxy": n_unique_proxy,
                "proxy_var": proxy_var,
                "path_status": "ok" if not status_bits else ";".join(status_bits),
                "N": int(bp.N),
                "logN": math.log(bp.N),
                "is_nontrivial": nontrivial,
                "d_bon": bp.d_bon,
                "q0": base_q,
                "q_bon": bp.q_bon,
                "delta_bon": bp.delta_bon,
                "selected_proxy_mean": bp.selected_proxy_mean,
                "p_max": bp.p_max,
                "n_support": bp.n_support,
                "in_shared_window": in_shared,
                "in_bridge_window": in_bridge,
                "in_nontrivial_shared_window": int(nontrivial and in_shared),
                "in_nontrivial_bridge_window": int(nontrivial and in_bridge),
                "a_exact": getattr(coeff, "a_exact", float("nan")),
                "b_theory": getattr(coeff, "b_theory", float("nan")),
                "a_fit_shared": getattr(fit_shared, "a_fit", float("nan")),
                "b_fit_shared": getattr(fit_shared, "b_fit", float("nan")),
                "a_fit_bridge": getattr(fit_bridge, "a_fit", float("nan")),
                "b_fit_bridge": getattr(fit_bridge, "b_fit", float("nan")),
                "tilt_delta_at_dbon": tilt_delta,
                "theory_local_delta_at_dbon": theory_shared,
                "empfit_shared_delta_at_dbon": empfit_shared,
                "empfit_bridge_delta_at_dbon": empfit_bridge,
                "abs_err_theory_vs_bon": abs(theory_shared - bp.delta_bon) if math.isfinite(theory_shared) else float("nan"),
                "abs_err_empfit_shared_vs_bon": abs(empfit_shared - bp.delta_bon) if math.isfinite(empfit_shared) else float("nan"),
                "abs_err_empfit_bridge_vs_bon": abs(empfit_bridge - bp.delta_bon) if math.isfinite(empfit_bridge) else float("nan"),
                "abs_err_tilt_vs_bon": abs(tilt_delta - bp.delta_bon) if math.isfinite(tilt_delta) else float("nan"),
            }
        )

    rows_df = pd.DataFrame(rows)
    n_nontrivial_total = int(rows_df["is_nontrivial"].sum())
    n_nontrivial_shared = int(rows_df["in_nontrivial_shared_window"].sum())
    n_nontrivial_bridge = int(rows_df["in_nontrivial_bridge_window"].sum())
    protocol_mismatch_shared = int(n_nontrivial_total > 0 and n_nontrivial_shared == 0)
    protocol_mismatch_bridge = int(n_nontrivial_total > 0 and n_nontrivial_bridge == 0)

    summary: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "proxy_source": path.meta.get("proxy_source"),
        "n_items": n_items,
        "n_unique_proxy": n_unique_proxy,
        "proxy_var": proxy_var,
        "path_status": "ok" if not status_bits else ";".join(status_bits),
        "shared_fit_max_d": shared_fit_max_d,
        "bridge_max_d": bridge_max_d,
        "max_d_bon": rows_df["d_bon"].max(),
        "d_bon_at_N2": float(rows_df.loc[rows_df["N"] == 2, "d_bon"].iloc[0]) if (rows_df["N"] == 2).any() else float("nan"),
        "share_all_in_shared_window": rows_df["in_shared_window"].mean(),
        "share_all_in_bridge_window": rows_df["in_bridge_window"].mean(),
        "share_nontrivial_in_shared_window": rows_df["in_nontrivial_shared_window"].mean() if n_nontrivial_total > 0 else float("nan"),
        "share_nontrivial_in_bridge_window": rows_df["in_nontrivial_bridge_window"].mean() if n_nontrivial_total > 0 else float("nan"),
        "n_nontrivial_total": n_nontrivial_total,
        "n_nontrivial_shared": n_nontrivial_shared,
        "n_nontrivial_bridge": n_nontrivial_bridge,
        "shared_window_status": classify_window_status(n_nontrivial_shared),
        "bridge_window_status": classify_window_status(n_nontrivial_bridge),
        "protocol_mismatch_shared": protocol_mismatch_shared,
        "protocol_mismatch_bridge": protocol_mismatch_bridge,
        "first_N_in_shared_window": first_matching_N(rows_df, "in_shared_window"),
        "first_nontrivial_N_in_shared_window": first_matching_N(rows_df, "in_nontrivial_shared_window"),
        "first_N_in_bridge_window": first_matching_N(rows_df, "in_bridge_window"),
        "first_nontrivial_N_in_bridge_window": first_matching_N(rows_df, "in_nontrivial_bridge_window"),
        "first_N_above_shared_window": float(rows_df.loc[rows_df["d_bon"] > shared_fit_max_d, "N"].min()) if (rows_df["d_bon"] > shared_fit_max_d).any() else float("nan"),
        "first_N_above_bridge_window": float(rows_df.loc[rows_df["d_bon"] > bridge_max_d, "N"].min()) if (rows_df["d_bon"] > bridge_max_d).any() else float("nan"),
        "corr_bon_vs_theory": safe_corr(rows_df["delta_bon"], rows_df["theory_local_delta_at_dbon"]),
        "corr_bon_vs_empfit_shared": safe_corr(rows_df["delta_bon"], rows_df["empfit_shared_delta_at_dbon"]),
        "corr_bon_vs_empfit_bridge": safe_corr(rows_df["delta_bon"], rows_df["empfit_bridge_delta_at_dbon"]),
        "corr_bon_vs_tilt": safe_corr(rows_df["delta_bon"], rows_df["tilt_delta_at_dbon"]),
        "mae_bon_vs_theory_all": rows_df["abs_err_theory_vs_bon"].mean(),
        "mae_bon_vs_empfit_shared_all": rows_df["abs_err_empfit_shared_vs_bon"].mean(),
        "mae_bon_vs_empfit_bridge_all": rows_df["abs_err_empfit_bridge_vs_bon"].mean(),
        "mae_bon_vs_tilt_all": rows_df["abs_err_tilt_vs_bon"].mean(),
        "mae_bon_vs_theory_shared": rows_df.loc[rows_df["in_shared_window"] == 1, "abs_err_theory_vs_bon"].mean(),
        "mae_bon_vs_empfit_shared_shared": rows_df.loc[rows_df["in_shared_window"] == 1, "abs_err_empfit_shared_vs_bon"].mean(),
        "mae_bon_vs_empfit_bridge_shared": rows_df.loc[rows_df["in_shared_window"] == 1, "abs_err_empfit_bridge_vs_bon"].mean(),
        "mae_bon_vs_tilt_shared": rows_df.loc[rows_df["in_shared_window"] == 1, "abs_err_tilt_vs_bon"].mean(),
        "mae_bon_vs_theory_bridge": rows_df.loc[rows_df["in_bridge_window"] == 1, "abs_err_theory_vs_bon"].mean(),
        "mae_bon_vs_empfit_shared_bridge": rows_df.loc[rows_df["in_bridge_window"] == 1, "abs_err_empfit_shared_vs_bon"].mean(),
        "mae_bon_vs_empfit_bridge_bridge": rows_df.loc[rows_df["in_bridge_window"] == 1, "abs_err_empfit_bridge_vs_bon"].mean(),
        "mae_bon_vs_tilt_bridge": rows_df.loc[rows_df["in_bridge_window"] == 1, "abs_err_tilt_vs_bon"].mean(),
    }
    return rows_df, summary



def summarize_by_group(path_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        "share_all_in_shared_window",
        "share_all_in_bridge_window",
        "share_nontrivial_in_shared_window",
        "share_nontrivial_in_bridge_window",
        "max_d_bon",
        "d_bon_at_N2",
        "corr_bon_vs_theory",
        "corr_bon_vs_empfit_shared",
        "corr_bon_vs_empfit_bridge",
        "corr_bon_vs_tilt",
        "mae_bon_vs_theory_all",
        "mae_bon_vs_empfit_shared_all",
        "mae_bon_vs_empfit_bridge_all",
        "mae_bon_vs_tilt_all",
        "mae_bon_vs_theory_shared",
        "mae_bon_vs_empfit_shared_shared",
        "mae_bon_vs_empfit_bridge_shared",
        "mae_bon_vs_tilt_shared",
        "mae_bon_vs_theory_bridge",
        "mae_bon_vs_empfit_shared_bridge",
        "mae_bon_vs_empfit_bridge_bridge",
        "mae_bon_vs_tilt_bridge",
        "proxy_var",
    ]

    def add_group(group_key: str, gdf: pd.DataFrame) -> None:
        row: Dict[str, Any] = {"group_key": group_key, "n_paths": len(gdf)}
        row["frac_protocol_mismatch_shared"] = float(np.mean(gdf["protocol_mismatch_shared"]))
        row["frac_protocol_mismatch_bridge"] = float(np.mean(gdf["protocol_mismatch_bridge"]))
        row["frac_status_not_ok"] = float(np.mean(gdf["path_status"] != "ok"))
        for col in metric_cols:
            row[f"median_{col}"] = float(gdf[col].median()) if col in gdf.columns else float("nan")
            row[f"mean_{col}"] = float(gdf[col].mean()) if col in gdf.columns else float("nan")
        rows.append(row)

    for group_key, gdf in path_summary_df.groupby("group_key", dropna=False):
        add_group(str(group_key), gdf)
    add_group("__OVERALL__", path_summary_df)
    return pd.DataFrame(rows)



def summarize_ladder(points_df: pd.DataFrame, group_by: str = "N") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if points_df.empty:
        return pd.DataFrame()
    for N, gdf in points_df.groupby(group_by, dropna=False):
        rows.append(
            {
                "N": int(N),
                "n_rows": len(gdf),
                "median_d_bon": float(gdf["d_bon"].median()),
                "mean_d_bon": float(gdf["d_bon"].mean()),
                "min_d_bon": float(gdf["d_bon"].min()),
                "max_d_bon": float(gdf["d_bon"].max()),
                "share_in_shared_window": float(gdf["in_shared_window"].mean()),
                "share_in_bridge_window": float(gdf["in_bridge_window"].mean()),
                "share_nontrivial_in_shared_window": float(gdf["in_nontrivial_shared_window"].mean()),
                "share_nontrivial_in_bridge_window": float(gdf["in_nontrivial_bridge_window"].mean()),
                "median_delta_bon": float(gdf["delta_bon"].median()),
                "mean_delta_bon": float(gdf["delta_bon"].mean()),
                "median_abs_err_tilt_vs_bon": float(gdf["abs_err_tilt_vs_bon"].median()),
                "median_abs_err_theory_vs_bon": float(gdf["abs_err_theory_vs_bon"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)



def summarize_status(path_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if path_summary_df.empty:
        return pd.DataFrame()
    for key, gdf in path_summary_df.groupby("path_status", dropna=False):
        rows.append({"path_status": key, "n_paths": len(gdf), "frac_paths": len(gdf) / len(path_summary_df)})
    return pd.DataFrame(rows).sort_values(["n_paths", "path_status"], ascending=[False, True]).reset_index(drop=True)



def main() -> None:
    ap = argparse.ArgumentParser(description="Improved BoN-vs-tilt bridge experiment with explicit window diagnostics.")
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", default="item_id")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="", help="Optional external/model proxy CSV. If omitted, uses held-out human paths.")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score")
    ap.add_argument("--N-list", default="1,2,4,8,16,32,64")
    ap.add_argument("--shared-fit-max-d", type=float, default=0.10)
    ap.add_argument("--bridge-max-d", type=float, default=0.25)
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--proxy-var-eps", type=float, default=1e-12)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    N_list = [int(x) for x in parse_cols(args.N_list)]
    if not N_list:
        raise ValueError("N-list must not be empty")
    if args.bridge_max_d < args.shared_fit_max_d:
        raise ValueError("bridge-max-d must be >= shared-fit-max-d")

    paths = build_paths(args)
    point_frames: List[pd.DataFrame] = []
    path_summaries: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in paths:
        try:
            rows_df, summary = path_to_rows(
                path,
                N_list=N_list,
                shared_fit_max_d=args.shared_fit_max_d,
                bridge_max_d=args.bridge_max_d,
                var_eps=args.proxy_var_eps,
            )
        except (RuntimeError, ValueError) as exc:
            skipped.append(f"{path.group_key}/{path.path_id}: {type(exc).__name__}")
            continue
        point_frames.append(rows_df)
        path_summaries.append(summary)

    if skipped:
        print(f"Skipped {len(skipped)}/{len(paths)} paths (tilt curve or fit infeasible):")
        for s in skipped:
            print(f"  {s}")

    points_df = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    path_summary_df = pd.DataFrame(path_summaries)
    group_summary_df = summarize_by_group(path_summary_df) if not path_summary_df.empty else pd.DataFrame()
    ladder_df = summarize_ladder(points_df)
    status_df = summarize_status(path_summary_df)

    points_path = outdir / "exp29_bon_bridge_points.csv"
    path_summary_path = outdir / "exp29_bon_bridge_path_summary.csv"
    group_summary_path = outdir / "exp29_bon_bridge_summary.csv"
    ladder_path = outdir / "exp29_bon_bridge_ladder_summary.csv"
    status_path = outdir / "exp29_bon_bridge_status_summary.csv"
    points_df.to_csv(points_path, index=False)
    path_summary_df.to_csv(path_summary_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    ladder_df.to_csv(ladder_path, index=False)
    status_df.to_csv(status_path, index=False)

    spec = {
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "N_list": N_list,
        "shared_fit_max_d": args.shared_fit_max_d,
        "bridge_max_d": args.bridge_max_d,
        "n_paths": len(paths),
        "proxy_var_eps": args.proxy_var_eps,
    }
    (outdir / "exp29_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    make_markdown_report(outdir, group_summary_df, ladder_df, args.shared_fit_max_d, args.bridge_max_d)

    print(f"Saved point-level results to {points_path}")
    print(f"Saved path-level summary to {path_summary_path}")
    print(f"Saved group summary to {group_summary_path}")
    print(f"Saved N-to-KL ladder summary to {ladder_path}")
    print(f"Saved status summary to {status_path}")


if __name__ == "__main__":
    main()
