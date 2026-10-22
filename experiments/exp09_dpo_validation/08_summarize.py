#!/usr/bin/env python3
"""[exp09-08] Pool beta results, compare realized gains against the pre-locked prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--coefficients-json", required=True)
    args = ap.parse_args()

    coef = json.load(open(args.coefficients_json))
    A, B, A_nu = coef["A"], coef["B"], coef["A_nu"]

    rows = []
    for sd in sorted(Path(args.run_root).glob("beta_*")):
        if not sd.is_dir():
            continue
        sj = sd / "summary.json"
        pp = sd / "per_prompt.csv"
        if not sj.exists() or not pp.exists():
            print(f"skip incomplete {sd.name}")
            continue
        s = json.load(open(sj))
        df = pd.read_csv(pp)
        d = df.d_reweight.clip(lower=0)
        pred_ab = (A * np.sqrt(d) + B * d)
        pred_a = (A * np.sqrt(d))
        pred_anu = (A_nu * np.sqrt(d))
        rows.append({
            "beta": s["beta"],
            "median_d_reweight": s["median_d_reweight"],
            "median_ess": s["median_ess"],
            "mean_seq_kl": s["mean_seq_kl_nats"],
            "realized_reweight_gold": s["mean_gain_reweight_gold"],
            "realized_fresh_gold": s["mean_gain_fresh_gold"],
            "realized_fresh_proxy": s["mean_gain_fresh_proxy"],
            "pred_AB": float(pred_ab.mean()),
            "pred_Aonly": float(pred_a.mean()),
            "pred_Anu": float(pred_anu.mean()),
            "abs_err_AB_vs_reweight": float(abs(pred_ab.mean() - s["mean_gain_reweight_gold"])),
            "abs_err_AB_vs_fresh": float(abs(pred_ab.mean() - s["mean_gain_fresh_gold"])),
            "abs_err_Aonly_vs_fresh": float(abs(pred_a.mean() - s["mean_gain_fresh_gold"])),
            "abs_err_zero_vs_fresh": float(abs(s["mean_gain_fresh_gold"])),
            "bridge_term_fresh_minus_reweight": s["mean_gain_fresh_gold"] - s["mean_gain_reweight_gold"],
        })
    out = pd.DataFrame(rows).sort_values("beta", ascending=False)
    out.to_csv(Path(args.run_root) / "summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nA={A:.5f} B={B:.5f} A_nu={A_nu:.5f} (calibration-locked)")


if __name__ == "__main__":
    main()
