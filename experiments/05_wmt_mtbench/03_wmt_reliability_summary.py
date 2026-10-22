#!/usr/bin/env python3
"""  [exp21]WMT reliability / coverage summary.

Summarizes how many repeated ratings are actually available per WMT split and computes
ICC(1,1) where appropriate. This is useful both for selecting main-text datasets and for
reporting data coverage before running the direct-tilt scripts.

Outputs
-------
- wmt_mqm_reliability.csv
- wmt_psqm_reliability.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wmt_mtbench_utils import icc1_from_long, load_wmt_long


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    return ap.parse_args()


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_id, sub in long_df.groupby("dataset_id"):
        counts = sub.groupby("item_id")["rater"].nunique()
        icc1, icc_n, icc_k = icc1_from_long(sub, item_col="item_id", score_col="score")
        rows.append({
            "dataset_id": dataset_id,
            "n_rows": int(sub.shape[0]),
            "n_items": int(counts.shape[0]),
            "n_raters_total": int(sub["rater"].nunique()),
            "mean_raters_per_item": float(counts.mean()),
            "min_raters_per_item": int(counts.min()),
            "max_raters_per_item": int(counts.max()),
            "frac_items_ge2": float((counts >= 2).mean()),
            "frac_items_ge3": float((counts >= 3).mean()),
            "icc1": icc1,
            "icc1_items_used": icc_n,
            "icc1_k_modal": icc_k,
            "score_mean": float(sub["score"].mean()),
            "score_std": float(sub["score"].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mqm = load_wmt_long(args.root, mode="mqm")
    psqm = load_wmt_long(args.root, mode="psqm")
    mqm_out = summarize(mqm)
    psqm_out = summarize(psqm)
    mqm_out.to_csv(outdir / "wmt_mqm_reliability.csv", index=False)
    psqm_out.to_csv(outdir / "wmt_psqm_reliability.csv", index=False)
    print("MQM\n", mqm_out.to_string(index=False))
    print("\npSQM\n", psqm_out.to_string(index=False))


if __name__ == "__main__":
    main()
