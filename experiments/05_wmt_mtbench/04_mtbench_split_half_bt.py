#!/usr/bin/env python3
"""  [exp22]MT-Bench split-half Bradley-Terry direct-tilt experiment.

Data preparation
----------------
The MT-Bench human judgments parquet file is not included in this repository.
Download it using ``experiments/data_preparation/download_mtbench.py``::

    python experiments/data_preparation/download_mtbench.py

Then pass the resulting parquet file via ``--parquet data/mt_bench_human_judgments/human.parquet``.

Why this experiment exists
--------------------------
MT-Bench human judgments are pairwise rather than scalar repeated ratings, so the exact
SummEval/WMT direct-tilt protocol does not apply directly. This script therefore builds
*question-turn-specific* Bradley-Terry utilities from random split-halves of judges:

1. Split judges into two random halves within each question-turn group.
2. Fit BT utilities on each half separately.
3. Treat one split-half utility vector as gold J and the other as proxy r.
4. Pool per-question-turn model utilities into item-level arrays and run the same
   exact exponential-tilt / local-fit procedure as in the scalar-rating experiments.

Interpretation
--------------
This is best viewed as an appendix / robustness experiment about human pairwise BT geometry,
not as a drop-in replacement for scalar repeated-rating direct tilt.

Requirements
------------
Requires a parquet engine locally (pyarrow or fastparquet). If pandas.read_parquet fails,
install one and rerun.

Outputs
-------
- mtbench_bt_split_items.csv
- mtbench_bt_split_summary.csv
- mtbench_bt_overall.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wmt_mtbench_utils import split_half_bt_items, read_mtbench_parquet


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True, help="Path to human parquet file")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--min-judges-per-group", type=int, default=4)
    ap.add_argument("--min-rows-per-split-group", type=int, default=4)
    ap.add_argument("--no-standardize-within-group", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_mtbench_parquet(args.parquet)
    items_df, summary_df = split_half_bt_items(
        df,
        n_splits=args.n_splits,
        random_state=args.random_state,
        standardize_within_group=not args.no_standardize_within_group,
        min_judges_per_group=args.min_judges_per_group,
        min_rows_per_split_group=args.min_rows_per_split_group,
    )
    items_df.to_csv(outdir / "mtbench_bt_split_items.csv", index=False)
    summary_df.to_csv(outdir / "mtbench_bt_split_summary.csv", index=False)

    overall = pd.DataFrame([
        {
            "n_splits_requested": args.n_splits,
            "n_splits_successful": int(summary_df.shape[0]),
            "median_n_items": float(summary_df["n_items"].median()) if not summary_df.empty else np.nan,
            "median_n_groups_used": float(summary_df["n_groups_used"].median()) if not summary_df.empty else np.nan,
            "corr_A_pooled": float(np.corrcoef(summary_df["A_theory"], summary_df["A_fit"])[0, 1]) if len(summary_df) >= 2 and summary_df["A_theory"].std() > 0 and summary_df["A_fit"].std() > 0 else np.nan,
            "corr_B_pooled": float(np.corrcoef(summary_df["B_theory"], summary_df["B_fit"])[0, 1]) if len(summary_df) >= 2 and summary_df["B_theory"].std() > 0 and summary_df["B_fit"].std() > 0 else np.nan,
            "sign_agreement": int(((np.sign(summary_df["B_theory"]) == np.sign(summary_df["B_fit"])) | ((summary_df["B_theory"].abs() < 1e-12) & (summary_df["B_fit"].abs() < 1e-12))).sum()) if not summary_df.empty else 0,
            "frac_B_theory_neg": float((summary_df["B_theory"] < 0).mean()) if not summary_df.empty else np.nan,
            "frac_B_fit_neg": float((summary_df["B_fit"] < 0).mean()) if not summary_df.empty else np.nan,
            "median_r2": float(summary_df["r2"].median()) if not summary_df.empty else np.nan,
            "median_corr_J_r": float(summary_df["corr_J_r"].median()) if not summary_df.empty else np.nan,
        }
    ])
    overall.to_csv(outdir / "mtbench_bt_overall.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
