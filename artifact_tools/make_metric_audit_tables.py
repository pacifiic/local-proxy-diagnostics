#!/usr/bin/env python3
"""Build metric-audit tables without touching experiment outputs.

This script is intentionally read-only with respect to existing results.  It
collects the sign-metric audit, fixed-gain margin sweep, curve-level metrics,
and GPT-4o reproducibility-record fields used by the submission artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROSPECTIVE_RUNS = {
    "SummEval human": "exp27_human_baseline",
    "WMT MQM clusters": "exp27_wmt_mqm",
    "OpenMEVA-MAGS": "exp27_openmeva",
    "GPT-4o proxy": "exp27_model_proxy_4o",
}

D_SWEEP_RUNS = {
    "SummEval human": "d_sweep_human",
    "GPT-4o proxy": "d_sweep_gpt4o",
}

PREDICTORS = {
    "theory": "A+B",
    "aonly": "A-only",
    "empfit": "empirical fit",
    "anu": "$A_\\nu$",
    "zero": "zero",
    "oracle": "oracle",
}


def fmt(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def pct(x: Any, digits: int = 1) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{100.0 * val:.{digits}f}%"


def latex_cell(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    if "$" in x:
        return x.replace(" -> ", r" $\rightarrow$ ")
    return (
        x.replace(" -> ", r" $\rightarrow$ ")
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def write_table(df: pd.DataFrame, outdir: Path, stem: str, *, review_markup: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    latex_df = df.copy()
    latex_df.columns = [latex_cell(str(col)) for col in latex_df.columns]
    for col in latex_df.columns:
        if latex_df[col].dtype == object:
            latex_df[col] = latex_df[col].map(latex_cell)
    if review_markup and stem == "table_curve_level_metrics":
        latex_df = latex_df.rename(columns={
            "Grid MAE": r"\textcolor{red}{Grid MAE} \textcolor{blue}{(Grid IAE)}",
        })
    (outdir / f"{stem}.tex").write_text(
        latex_df.to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def overall_row(csv_path: Path, key_col: str = "group_key") -> pd.Series:
    df = pd.read_csv(csv_path)
    return df.loc[df[key_col].eq("__OVERALL__")].iloc[0]


def build_sign_metric_audit(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    prospective_root = results_root / "07_prospective_prediction"
    accept_root = results_root / "12_accept_boost"

    for setting, run in PROSPECTIVE_RUNS.items():
        row = overall_row(prospective_root / run / "exp27_prospective_summary.csv")
        rows.append(
            {
                "Setting": setting,
                "Metric": "B-sign agreement",
                "Rows": str(int(row["n_splits"])),
                "Accuracy": pct(row["theory_sign_acc"]),
                "Source": f"07_prospective_prediction/{run}",
                "Interpretation": "sign(calibration B) equals sign(evaluation fitted B)",
            }
        )

    for setting, run in D_SWEEP_RUNS.items():
        summary = pd.read_csv(accept_root / run / "d_sweep_summary_overall.csv")
        row = summary.loc[summary["d"].round(8).eq(0.10)].iloc[0]
        rows.append(
            {
                "Setting": setting,
                "Metric": "fixed-gain sign at $d=0.10$",
                "Rows": str(int(row["valid_rows"])),
                "Accuracy": pct(row["sign_acc_theory"]),
                "Source": f"12_accept_boost/{run}",
                "Interpretation": "sign(predicted Delta) equals sign(actual Delta)",
            }
        )
    return pd.DataFrame(rows)


def build_fixed_gain_margin_sweep(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    thresholds: list[float | None] = [None, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    accept_root = results_root / "12_accept_boost"
    for setting, run in D_SWEEP_RUNS.items():
        df = pd.read_csv(accept_root / run / "d_sweep_paths_long.csv")
        df = df[df["d"].round(8).eq(0.10) & df["actual_delta"].notna()].copy()
        total = len(df)
        for eps in thresholds:
            if eps is None:
                sub = df
                condition = "all"
            else:
                sub = df[df["actual_delta"].abs() > eps]
                condition = f"$|\\Delta|>{eps:g}$"
            rows.append(
                {
                    "Setting": setting,
                    "Condition": condition,
                    "Coverage": pct(len(sub) / total if total else np.nan),
                    "$n$": str(len(sub)),
                    "Fixed-gain sign acc.": pct(sub["sign_correct_theory"].mean() if len(sub) else np.nan),
                    "MAE": fmt(sub["abs_err_theory"].mean() if len(sub) else np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_curve_level_metrics(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    accept_root = results_root / "12_accept_boost"
    id_cols = ["group_key", "path_id", "repeat"]
    for setting, run in D_SWEEP_RUNS.items():
        df = pd.read_csv(accept_root / run / "d_sweep_paths_long.csv")
        df = df[df["d"].le(0.10) & df["actual_delta"].notna()].copy()
        for pred, label in PREDICTORS.items():
            err_col = f"abs_err_{pred}"
            if err_col not in df.columns:
                continue
            per_curve = df.groupby(id_cols, dropna=False)[err_col].agg(["mean", "max"]).reset_index()
            rows.append(
                {
                    "Setting": setting,
                    "Predictor": label,
                    "Curves": str(len(per_curve)),
                    "Grid MAE": fmt(per_curve["mean"].mean()),
                    "Median max error": fmt(per_curve["max"].median()),
                    "90% max error": fmt(per_curve["max"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def _sha12(path: Path) -> str:
    if not path.exists():
        return "--"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_gpt4o_repro_record(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    proxy_csv = repo_root / "data/model_proxy_4o_aggregated.csv"
    system_prompt = repo_root / "experiments/09_gpt_proxy_analysis/examples/summary_system_prompt.txt"
    user_template = repo_root / "experiments/09_gpt_proxy_analysis/examples/summary_user_template.txt"
    generator = repo_root / "experiments/09_gpt_proxy_analysis/01_generate_proxy_scores.py"

    if proxy_csv.exists():
        proxy = pd.read_csv(proxy_csv)
        n_rows = len(proxy)
        n_judges = sorted(proxy["n_judges"].dropna().unique().tolist()) if "n_judges" in proxy else []
        dimensions = sorted(proxy["dimension"].dropna().unique().tolist()) if "dimension" in proxy else []
    else:
        n_rows = 0
        n_judges = []
        dimensions = []

    rows.extend(
        [
            {
                "Field": "Aggregated proxy artifact",
                "Value": str(proxy_csv.relative_to(repo_root)) if proxy_csv.exists() else "missing",
                "Status": f"{n_rows} rows; dimensions={','.join(map(str, dimensions))}; n_judges={n_judges}",
            },
            {
                "Field": "Exact model snapshot",
                "Value": "not stored in aggregated artifact",
                "Status": "filename indicates GPT-4o; raw per-call log is needed for an exact model id",
            },
            {
                "Field": "System prompt",
                "Value": str(system_prompt.relative_to(repo_root)) if system_prompt.exists() else "missing",
                "Status": f"sha256[:12]={_sha12(system_prompt)}",
            },
            {
                "Field": "User prompt template",
                "Value": str(user_template.relative_to(repo_root)) if user_template.exists() else "missing",
                "Status": f"sha256[:12]={_sha12(user_template)}",
            },
            {
                "Field": "Score parser",
                "Value": "Pydantic RatingOutput(score, confidence, rationale)",
                "Status": str(generator.relative_to(repo_root)) if generator.exists() else "generator missing",
            },
            {
                "Field": "Score range",
                "Value": "1.0--5.0 in generator defaults",
                "Status": "score is clamped to range after parsing",
            },
            {
                "Field": "Decoding parameters",
                "Value": "not explicitly set in generator",
                "Status": "temperature/top_p/max_tokens use API defaults unless original command overrode code",
            },
            {
                "Field": "Invalid-output handling",
                "Value": "structured parse with retries",
                "Status": "default max_retries=5; invalid count not stored in aggregate",
            },
            {
                "Field": "Normalization split",
                "Value": "not stored in proxy artifact",
                "Status": "downstream experiments standardize inside each analysis protocol",
            },
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="all_results_reproduced")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="paper_tables_metric_audits")
    ap.add_argument("--review-markup", action="store_true",
                    help="Mark the corrected LaTeX header red and its old wording blue.")
    args = ap.parse_args()
    results_root = Path(args.results_root)
    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.out)

    tables = {
        "table_sign_metric_audit": build_sign_metric_audit(results_root),
        "table_fixed_gain_margin_sweep": build_fixed_gain_margin_sweep(results_root),
        "table_curve_level_metrics": build_curve_level_metrics(results_root),
        "table_gpt4o_repro_record": build_gpt4o_repro_record(repo_root),
    }
    for stem, df in tables.items():
        write_table(df, outdir, stem, review_markup=args.review_markup)
        print(f"wrote {outdir / (stem + '.csv')} ({len(df)} rows)")


if __name__ == "__main__":
    main()
