"""  [exp27 supplement]
Margin-conditioned sign accuracy sweep for prospective prediction.

Reads exp27_prospective_paths.csv and computes sign accuracy and MAE
conditioned on |actual Delta(d)| > threshold for multiple thresholds.

Usage:
    python 02_margin_threshold_sweep.py \
        --paths-csv RESULTS/07_prospective_prediction/exp27_human_baseline/exp27_prospective_paths.csv \
        --outdir RESULTS/07_prospective_prediction/exp27_human_baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def margin_sweep(
    df: pd.DataFrame,
    d_suffix: str = "0p1",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

    actual = df[f"actual_eval_delta_{d_suffix}"].values
    pred   = df[f"pred_theory_delta_{d_suffix}"].values

    rows = []
    for thresh in thresholds:
        mask = np.abs(actual) > thresh if thresh > 0 else np.ones(len(actual), dtype=bool)
        n = mask.sum()
        if n == 0:
            continue
        rows.append({
            "threshold": thresh,
            "condition": "all" if thresh == 0 else f"|Delta|>{thresh}",
            "n": n,
            "coverage_pct": n / len(actual) * 100,
            "sign_acc_pct": (np.sign(pred[mask]) == np.sign(actual[mask])).mean() * 100,
            "mae": np.abs(pred[mask] - actual[mask]).mean(),
            "mean_abs_actual": np.abs(actual[mask]).mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-csv", required=True, help="exp27_prospective_paths.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--d-suffix", default="0p1", help="Delta suffix (default: 0p1 = d=0.10)")
    args = ap.parse_args()

    df = pd.read_csv(args.paths_csv)
    result = margin_sweep(df, d_suffix=args.d_suffix)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "margin_threshold_sweep.csv"
    result.to_csv(out_csv, index=False)
    print(result.to_markdown(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
