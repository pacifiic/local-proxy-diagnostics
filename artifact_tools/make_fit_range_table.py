#!/usr/bin/env python3
"""Reaggregate the eight-group SummEval raw-score LOO fit-range table."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/fit_range_2/03_direct_tilt_validation/robustness/table_exp18_tilt_estimator_sensitivity_by_dataset.csv"


def build(source, outdir):
    frame = pd.read_csv(source)
    frame = frame[(frame.score_transform == "raw") & (frame.gold_estimator == "loo_mean")]
    columns = ["mean_A_theory", "mean_A_fit", "mean_B_theory", "mean_B_fit", "mean_r2"]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("Nonfinite fit-range input")
    if frame.duplicated(["fit_max_d", "label"]).any():
        raise ValueError("Duplicate group/cutoff")
    expected = [.05, .10, .20, .50, 1., 2.]
    if sorted(frame.fit_max_d.unique()) != expected:
        raise ValueError("Incomplete fit cutoffs")
    if not frame.groupby("fit_max_d").size().eq(8).all():
        raise ValueError("Expected eight pool-dimension groups at each cutoff")
    rows = []
    for cutoff, group in frame.groupby("fit_max_d", sort=True):
        rows.append(dict(
            fit_cutoff=cutoff, groups=len(group),
            corr_A=group.mean_A_theory.corr(group.mean_A_fit),
            corr_B=group.mean_B_theory.corr(group.mean_B_fit),
            sign_agree=int((np.sign(group.mean_B_theory) == np.sign(group.mean_B_fit)).sum()),
            mean_r2=group.mean_r2.mean(),
        ))
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/"table_fit_range_sensitivity.csv", index=False)
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Fit cutoff $D$ & corr($A$) & corr($B$) & sign agree & mean $R^2$ \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['fit_cutoff']:.2f} & {r['corr_A']:.4f} & {r['corr_B']:.4f} & "
                     f"{r['sign_agree']}/{r['groups']} & {r['mean_r2']:.6f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir/"table_fit_range_sensitivity.tex").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--outdir", type=Path, default=ROOT/"tables")
    args = parser.parse_args()
    build(args.source, args.outdir)
