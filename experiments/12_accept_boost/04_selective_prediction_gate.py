"""Selective prediction and optimize/abstain policy gate.

This script turns GPT-4o proxy predictions into an operational selective
diagnostic. It can read either:
  1. the existing wide exp27_prospective_paths.csv; or
  2. the long d_sweep_paths_long.csv produced by 01_d_sweep_prediction.py.

For each threshold tau, the script reports:
  - coverage under |predicted gain| > tau;
  - sign accuracy and MAE among covered cases;
  - optimize/avoid/abstain rates;
  - realized policy gain when optimizing only if predicted gain > tau;
  - bad-optimization rate among optimized cases.

Example using existing exp27 output:
    python experiments/12_accept_boost/04_selective_prediction_gate.py \
      --paths-csv all_results_reproduced/07_prospective_prediction/exp27_model_proxy_4o/exp27_prospective_paths.csv \
      --d-target 0.10 --outdir all_results_reproduced/12_accept_boost/selective_gpt4o

Example using d-sweep output:
    python experiments/12_accept_boost/04_selective_prediction_gate.py \
      --paths-csv all_results_reproduced/12_accept_boost/d_sweep_gpt4o/d_sweep_paths_long.csv \
      --input-format long --d-target 0.10 \
      --outdir all_results_reproduced/12_accept_boost/selective_gpt4o
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accept_boost_utils import PREDICTOR_LABELS, d_suffix, mkdir, parse_float_list, save_markdown


def to_long_predictions(df: pd.DataFrame, d_target: float, predictors: List[str], input_format: str) -> pd.DataFrame:
    """Normalize wide exp27 or long d-sweep output to columns actual_delta, pred_<p>."""
    if input_format == "auto":
        input_format = "long" if {"d", "actual_delta"}.issubset(df.columns) else "wide"
    if input_format == "long":
        if "d" not in df or "actual_delta" not in df:
            raise ValueError("Long input must contain columns d and actual_delta.")
        sub = df[np.isclose(df["d"].astype(float), float(d_target))].copy()
        if sub.empty:
            raise ValueError(f"No rows found for d={d_target}. Available: {sorted(df['d'].unique())[:20]}")
        return sub
    if input_format != "wide":
        raise ValueError(f"Unknown input format: {input_format}")

    suf = d_suffix(d_target)
    actual_col = f"actual_eval_delta_{suf}"
    if actual_col not in df.columns:
        # Existing exp27 often used Python's str(0.10) -> 0p1; d_suffix already gives that,
        # but keep a useful fallback message.
        candidates = [c for c in df.columns if c.startswith("actual_eval_delta_")]
        raise ValueError(f"Missing {actual_col}; available actual columns: {candidates}")
    out = df.copy()
    out["actual_delta"] = out[actual_col]
    for p in predictors:
        src = f"pred_{p}_delta_{suf}"
        if src in out.columns:
            out[f"pred_{p}"] = out[src]
    if "d" not in out:
        out["d"] = float(d_target)
    return out


def _metrics(actual: np.ndarray, pred: np.ndarray, tau: float, margin_mode: str) -> Dict[str, float]:
    margin = np.abs(pred) if margin_mode == "predicted" else np.abs(actual)
    covered = np.ones_like(actual, dtype=bool) if tau <= 0 else margin > tau
    optimize = pred > tau
    avoid = pred < -tau
    abstain = ~(optimize | avoid)

    n = len(actual)
    n_cov = int(covered.sum())
    out: Dict[str, float] = {
        "threshold": float(tau),
        "n": float(n),
        "n_covered": float(n_cov),
        "coverage": float(n_cov / n) if n else np.nan,
        "optimize_rate": float(optimize.mean()) if n else np.nan,
        "avoid_rate": float(avoid.mean()) if n else np.nan,
        "abstain_rate": float(abstain.mean()) if n else np.nan,
        "mean_actual_gain_all": float(actual.mean()) if n else np.nan,
        "mean_abs_actual_all": float(np.abs(actual).mean()) if n else np.nan,
        "policy_realized_gain": float(np.where(optimize, actual, 0.0).mean()) if n else np.nan,
        "always_optimize_gain": float(actual.mean()) if n else np.nan,
        "never_optimize_gain": 0.0,
        "oracle_positive_gain": float(np.maximum(actual, 0.0).mean()) if n else np.nan,
    }
    if n_cov > 0:
        out["sign_acc_covered"] = float((np.sign(pred[covered]) == np.sign(actual[covered])).mean())
        out["mae_covered"] = float(np.abs(pred[covered] - actual[covered]).mean())
        out["mean_abs_pred_covered"] = float(np.abs(pred[covered]).mean())
        out["mean_abs_actual_covered"] = float(np.abs(actual[covered]).mean())
    else:
        out.update({
            "sign_acc_covered": np.nan,
            "mae_covered": np.nan,
            "mean_abs_pred_covered": np.nan,
            "mean_abs_actual_covered": np.nan,
        })
    if optimize.sum() > 0:
        out["optimize_precision_positive"] = float((actual[optimize] > 0).mean())
        out["bad_optimization_rate"] = float((actual[optimize] < 0).mean())
        out["mean_gain_when_optimized"] = float(actual[optimize].mean())
    else:
        out["optimize_precision_positive"] = np.nan
        out["bad_optimization_rate"] = np.nan
        out["mean_gain_when_optimized"] = np.nan
    if avoid.sum() > 0:
        out["avoid_precision_negative"] = float((actual[avoid] < 0).mean())
        out["mean_gain_when_avoided"] = float(actual[avoid].mean())
    else:
        out["avoid_precision_negative"] = np.nan
        out["mean_gain_when_avoided"] = np.nan
    out["regret_vs_oracle_positive_gain"] = out["oracle_positive_gain"] - out["policy_realized_gain"]
    out["gain_vs_always_optimize"] = out["policy_realized_gain"] - out["always_optimize_gain"]
    return out


def _cluster_bootstrap(g: pd.DataFrame, pred_col: str, thresholds: List[float], margin_mode: str, n_boot: int, seed: int) -> pd.DataFrame:
    if n_boot <= 0 or "path_id" not in g.columns or g["path_id"].nunique() <= 1:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    clusters = sorted(g["path_id"].dropna().unique())
    by_cluster = {c: g[g["path_id"].eq(c)].copy() for c in clusters}
    metric_names = [
        "coverage", "sign_acc_covered", "mae_covered", "optimize_rate", "bad_optimization_rate",
        "policy_realized_gain", "gain_vs_always_optimize", "regret_vs_oracle_positive_gain",
    ]
    boot_rows = []
    for tau in thresholds:
        vals = {m: [] for m in metric_names}
        for _ in range(n_boot):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            sdf = pd.concat([by_cluster[c] for c in sampled], ignore_index=True)
            m = _metrics(sdf["actual_delta"].to_numpy(float), sdf[pred_col].to_numpy(float), tau, margin_mode)
            for name in metric_names:
                vals[name].append(m.get(name, np.nan))
        row: Dict[str, float] = {"threshold": float(tau)}
        for name, arr in vals.items():
            arr = np.asarray(arr, dtype=float)
            row[f"{name}_ci_lo"] = float(np.nanpercentile(arr, 2.5))
            row[f"{name}_ci_hi"] = float(np.nanpercentile(arr, 97.5))
        boot_rows.append(row)
    return pd.DataFrame(boot_rows)


def run_selective(df: pd.DataFrame, predictors: List[str], thresholds: List[float], margin_mode: str, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for p in predictors:
        pred_col = f"pred_{p}"
        if pred_col not in df.columns:
            print(f"[skip] predictor {p}: missing {pred_col}")
            continue
        sub = df[[c for c in ["path_id", "group_key", "d", "actual_delta", pred_col] if c in df.columns]].dropna().copy()
        if sub.empty:
            continue
        boot = _cluster_bootstrap(sub, pred_col, thresholds, margin_mode, n_boot=n_boot, seed=seed)
        for tau in thresholds:
            m = _metrics(sub["actual_delta"].to_numpy(float), sub[pred_col].to_numpy(float), tau, margin_mode)
            m["predictor"] = p
            m["predictor_label"] = PREDICTOR_LABELS.get(p, p)
            if not boot.empty:
                brow = boot[boot["threshold"].eq(float(tau))]
                if not brow.empty:
                    m.update(brow.iloc[0].to_dict())
            rows.append(m)
    return pd.DataFrame(rows)


def plot_coverage_sign(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for p, sub in summary.groupby("predictor"):
        sub = sub.sort_values("coverage")
        ax.plot(100.0 * sub["coverage"], 100.0 * sub["sign_acc_covered"], marker="o", linewidth=1.8,
                label=PREDICTOR_LABELS.get(p, p))
    ax.set_xlabel("Coverage after abstention (%)")
    ax.set_ylabel("Sign accuracy on covered cases (%)")
    ax.set_title("Selective prediction: abstain on small margins")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_selective_coverage_sign.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_selective_coverage_sign.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_policy_gain(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for p, sub in summary.groupby("predictor"):
        sub = sub.sort_values("threshold")
        ax.plot(sub["threshold"], sub["policy_realized_gain"], marker="o", linewidth=1.8,
                label=PREDICTOR_LABELS.get(p, p))
    ax.axhline(0, linewidth=0.8, color="black")
    ax.set_xlabel("Predicted-margin/action threshold tau")
    ax.set_ylabel("Mean realized gain under optimize-if-pred>tau policy")
    ax.set_title("Optimize / abstain policy gate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_policy_gate_realized_gain.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig_policy_gate_realized_gain.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Selective prediction and optimize/abstain gate from prospective prediction rows.")
    ap.add_argument("--paths-csv", required=True)
    ap.add_argument("--input-format", choices=["auto", "wide", "long"], default="auto")
    ap.add_argument("--d-target", type=float, default=0.10)
    ap.add_argument("--predictors", default="theory,aonly,empfit,anu")
    ap.add_argument("--thresholds", default="0,0.0025,0.005,0.01,0.02,0.05,0.10")
    ap.add_argument("--margin-mode", choices=["predicted", "actual"], default="predicted",
                    help="Use predicted margins for deployable abstention; actual margins only for diagnostics.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = mkdir(args.outdir)
    raw = pd.read_csv(args.paths_csv)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    thresholds = parse_float_list(args.thresholds)
    df = to_long_predictions(raw, d_target=args.d_target, predictors=predictors, input_format=args.input_format)
    summary = run_selective(df, predictors, thresholds, args.margin_mode, args.n_boot, args.random_seed)
    summary.to_csv(outdir / "selective_prediction_gate_summary.csv", index=False)
    save_markdown(summary, outdir / "selective_prediction_gate_summary.md")
    spec = {
        "paths_csv": args.paths_csv,
        "input_format": args.input_format,
        "d_target": args.d_target,
        "predictors": predictors,
        "thresholds": thresholds,
        "margin_mode": args.margin_mode,
        "n_boot": args.n_boot,
        "random_seed": args.random_seed,
        "n_raw_rows": len(raw),
        "n_rows_at_target": len(df),
        "n_summary_rows": len(summary),
        "outputs": {
            "summary": str(outdir / "selective_prediction_gate_summary.csv"),
        },
    }
    (outdir / "selective_prediction_gate_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    plot_coverage_sign(summary, outdir)
    plot_policy_gain(summary, outdir)

    cols = ["predictor", "threshold", "coverage", "sign_acc_covered", "mae_covered", "policy_realized_gain", "bad_optimization_rate"]
    print(summary[cols].to_string(index=False))
    print(f"Saved selective prediction outputs in: {outdir}")


if __name__ == "__main__":
    main()
