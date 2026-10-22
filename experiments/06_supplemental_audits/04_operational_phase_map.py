#!/usr/bin/env python3
"""  [exp26]Exp30: operational phase-map thresholds.

This is not a new dataset experiment. It operationalizes the Phase A-D map
using explicit thresholds and optional confidence intervals.

Expected input columns
----------------------
Required:
- unit_id
- nu_pop

Optional:
- nu_pop_ci_lo
- nu_meas
- nu_pers

Interpretation
--------------
measurement_gap = max(nu_meas - nu_pop, 0)
personalization_gap = max(nu_pers - nu_pop, 0)

Phases
------
A: high verifiability already
B: moderate verifiability, small measurement and personalization gaps
C: measurement-limited
D: personalization-limited
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--unit-col", type=str, default="unit_id")
    ap.add_argument("--nu-pop-col", type=str, default="nu_pop")
    ap.add_argument("--nu-pop-ci-lo-col", type=str, default="nu_pop_ci_lo")
    ap.add_argument("--nu-meas-col", type=str, default="nu_meas")
    ap.add_argument("--nu-pers-col", type=str, default="nu_pers")
    ap.add_argument("--tau-high", type=float, default=0.90)
    ap.add_argument("--tau-mid", type=float, default=0.70)
    ap.add_argument("--tau-gap", type=float, default=0.05)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    up = args.nu_pop_col
    ul = args.nu_pop_ci_lo_col
    um = args.nu_meas_col
    ur = args.nu_pers_col

    if args.unit_col not in df.columns or up not in df.columns:
        raise ValueError("Input must contain unit_id and nu_pop columns (or custom names).")

    df = df.copy()
    if um not in df.columns:
        df[um] = np.nan
    if ur not in df.columns:
        df[ur] = np.nan
    if ul not in df.columns:
        df[ul] = np.nan

    df["measurement_gap"] = (df[um] - df[up]).clip(lower=0)
    df["personalization_gap"] = (df[ur] - df[up]).clip(lower=0)

    phases = []
    for row in df.itertuples(index=False):
        nu = getattr(row, up)
        nu_lo = getattr(row, ul)
        mg = getattr(row, "measurement_gap")
        pg = getattr(row, "personalization_gap")

        if np.isfinite(nu_lo) and nu_lo >= args.tau_high:
            phase = "A"
        elif (np.isfinite(mg) and mg >= args.tau_gap) and (not np.isfinite(pg) or pg < args.tau_gap):
            phase = "C"
        elif (np.isfinite(pg) and pg >= args.tau_gap) and (not np.isfinite(mg) or mg < args.tau_gap):
            phase = "D"
        elif np.isfinite(nu) and nu >= args.tau_mid:
            phase = "B"
        else:
            phase = "B"
        phases.append(phase)

    df["phase"] = phases
    df.to_csv(out, index=False)
    print("Saved phase map to", out)
    print(df[[args.unit_col, up, "measurement_gap", "personalization_gap", "phase"]].to_string(index=False))


if __name__ == "__main__":
    main()
