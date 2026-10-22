#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from verif_experiment_utils import compute_ab, fit_local_ab, tilt_curve

DIMS = [
    "fluency",
    "clarity",
    "conciseness",
    "relevance",
    "consistency",
    "answerability",
    "answer_consistency",
]

NUMERIC_METRICS = [
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "BERTScore",
    "MoverScore",
    "BLEURT",
    "Q-BLEU4",
    "QSTS",
    "BARTScore-ref",
    "BARTScore-src",
    "QRelScore",
    "RQUGE",
]
DICT_METRICS = ["GPTScore-ref", "GPTScore-src", "UniEval"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QGEval metric-proxy exact tilt experiment")
    p.add_argument("--scores-xlsx", required=True)
    p.add_argument("--metric-xlsx", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--score-transform", default="z", choices=["raw", "z"])
    p.add_argument("--fit-max-d", type=float, default=0.10)
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--n-grid", type=int, default=401)
    return p.parse_args()


def parse_metric_cell(val: Any, dim: str) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if not s:
        return np.nan
    if s.startswith("{") and s.endswith("}"):
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                if dim in d:
                    return float(d[dim])
                if "overall" in d:
                    return float(d["overall"])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def load_merged(scores_xlsx: str, metric_xlsx: str) -> pd.DataFrame:
    h = pd.read_excel(scores_xlsx)
    m = pd.read_excel(metric_xlsx)
    h["item_id"] = h["id"].astype(str) + "___" + h["source"].astype(str)
    m["item_id"] = m["id"].astype(str) + "___" + m["source"].astype(str)
    h["dataset"] = h["source"].astype(str).str.split("_").str[0]
    m["dataset"] = m["source"].astype(str).str.split("_").str[0]
    keep_h = ["item_id", "dataset", "source"] + DIMS
    keep_m = ["item_id", "dataset", "source"] + [c for c in NUMERIC_METRICS + DICT_METRICS if c in m.columns]
    merged = h[keep_h].merge(m[keep_m], on=["item_id", "dataset", "source"], how="inner", suffixes=("", "_metric"))
    return merged


def get_metric_vector(df: pd.DataFrame, metric: str, dim: str) -> np.ndarray:
    if metric in NUMERIC_METRICS:
        return pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    if metric in DICT_METRICS:
        return df[metric].map(lambda x: parse_metric_cell(x, dim)).to_numpy(dtype=float)
    raise ValueError(f"Unknown metric: {metric}")


def maybe_zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, dim), g in df.groupby(["dataset_eval", "dimension"]):
        out.append({
            "dataset_eval": dataset,
            "dimension": dim,
            "n_metrics": len(g),
            "frac_B_theory_neg": float((g["B_theory"] < 0).mean()),
            "frac_B_fit_neg": float((g["B_fit"] < 0).mean()),
            "corr_theory_fit_B": np.corrcoef(g["B_theory"], g["B_fit"])[0, 1] if len(g) > 1 else np.nan,
            "median_r2": float(g["local_r2"].median()),
            "best_corr_metric": g.sort_values("corr_J_r", ascending=False).iloc[0]["metric"],
            "largest_B_metric": g.sort_values("B_fit", ascending=False).iloc[0]["metric"],
            "most_negative_B_metric": g.sort_values("B_fit", ascending=True).iloc[0]["metric"],
        })
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_merged(args.scores_xlsx, args.metric_xlsx)
    datasets = sorted(df["dataset"].unique().tolist()) + ["pooled"]
    metrics = [m for m in NUMERIC_METRICS + DICT_METRICS if m in df.columns]
    t_grid = np.linspace(0.0, args.t_max, args.n_grid)

    rows = []
    trajs = []
    for dataset in datasets:
        sub = df if dataset == "pooled" else df[df["dataset"] == dataset]
        for dim in DIMS:
            J_full = pd.to_numeric(sub[dim], errors="coerce").to_numpy(dtype=float)
            for metric in metrics:
                r_full = get_metric_vector(sub, metric, dim)
                mask = np.isfinite(J_full) & np.isfinite(r_full)
                J = J_full[mask]
                r = r_full[mask]
                if len(J) < 20:
                    continue
                if args.score_transform == "z":
                    r = maybe_zscore(r)
                ab = compute_ab(J, r)
                d_vals, q_vals = tilt_curve(J, r, t_grid)
                A_fit, B_fit, r2 = fit_local_ab(d_vals, q_vals, max_d=args.fit_max_d)
                rows.append({
                    "dataset_eval": dataset,
                    "dimension": dim,
                    "metric": metric,
                    "n_items": len(J),
                    "fit_max_d": args.fit_max_d,
                    "A_theory": ab["A"],
                    "B_theory": ab["B"],
                    "A_fit": A_fit,
                    "B_fit": B_fit,
                    "local_r2": r2,
                    "corr_J_r": float(np.corrcoef(J, r)[0, 1]) if len(J) > 1 else np.nan,
                    "rho": ab["rho"],
                    "nu_tilde": ab["nu_tilde"],
                })
                for t, d, q in zip(t_grid, d_vals, q_vals):
                    trajs.append({
                        "dataset_eval": dataset,
                        "dimension": dim,
                        "metric": metric,
                        "t": t,
                        "d": d,
                        "q": q,
                        "q_delta": q - q_vals[0],
                    })

    df_rows = pd.DataFrame(rows).sort_values(["dataset_eval", "dimension", "metric"])
    df_traj = pd.DataFrame(trajs)
    df_summary = summarize(df_rows)

    df_rows.to_csv(outdir / "table_qgeval_metric_proxy_tilt.csv", index=False)
    df_traj.to_csv(outdir / "table_qgeval_metric_proxy_tilt_traj.csv", index=False)
    df_summary.to_csv(outdir / "table_qgeval_metric_proxy_tilt_summary.csv", index=False)

    report = [
        "# QGEval metric-proxy tilt\n",
        f"- score_transform: `{args.score_transform}`\n",
        f"- fit_max_d: `{args.fit_max_d}`\n",
        "\n## Summary\n",
        df_summary.to_markdown(index=False),
        "\n\n## Notes\n",
        "- This experiment uses the human average score as gold and automatic metrics as proxies.\n",
        "- The key finding is that a metric with high correlation does not necessarily have a good local `B`.\n",
        "- Metrics that consistently yield a negative `B_fit` are the closest reward-model analogues.\n",
    ]
    (outdir / "report_qgeval_metric_proxy_tilt.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
