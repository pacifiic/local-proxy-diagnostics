#!/usr/bin/env python3
"""exp08 — Proxy selection via predicted human-gain (downstream-validation answer, R5/③).

For each external proxy we already have a prospective run (predict A+B gain on a calibration
split, measure realized gain on disjoint eval items). This script pools those CANONICAL,
bug-fixed runs and asks the decision-relevant question: if a practitioner ranked the proxies
by *predicted* gain alone (no eval labels), would they pick the proxy that actually delivers
the most human gain? Reports proxy-level predicted vs realized gain, pairwise ranking
agreement, and Spearman rho over proxies.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# (label, paths_csv) — all from 2026-07-06 canonical corrected runs (STATUS.md §2)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    R = Path(args.repo_root)
    O = R / "outputs"
    SOURCES = [
        ("GPT-4o judge",   O/"exp06_final_20260706_022627"/"proxy_gpt4o"/"exp27_prospective_paths.csv"),
        ("Open judge (Qwen-32B)", O/"exp06_final_20260706_022627"/"proxy_openjudge"/"exp27_prospective_paths.csv"),
        ("Open RM (Skywork-8B)",  O/"exp06_final_20260706_022627"/"proxy_rm_skywork"/"exp27_prospective_paths.csv"),
        ("ROUGE-1", O/"exp02_fixed_rouge_20260706_020516"/"metric_rouge1"/"exp27_prospective_paths.csv"),
        ("ROUGE-2", O/"exp02_fixed_rouge_20260706_020516"/"metric_rouge2"/"exp27_prospective_paths.csv"),
        ("ROUGE-L", O/"exp02_fixed_rouge_20260706_020516"/"metric_rougeL"/"exp27_prospective_paths.csv"),
    ]
    rows = []
    for label, p in SOURCES:
        if not p.exists():
            print(f"SKIP (missing): {label} {p}"); continue
        df = pd.read_csv(p)
        rows.append({
            "proxy": label,
            "predicted_gain": float(df["pred_theory_delta_0p1"].mean()),
            "realized_gain": float(df["actual_eval_delta_0p1"].mean()),
            "n_paths": int(len(df)),
        })
    t = pd.DataFrame(rows)
    t["abs_pred_error"] = (t.predicted_gain - t.realized_gain).abs()
    t = t.sort_values("realized_gain", ascending=False).reset_index(drop=True)

    # ranking agreement
    pred_rank = t.predicted_gain.rank(ascending=False)
    real_rank = t.realized_gain.rank(ascending=False)
    rho, _ = spearmanr(t.predicted_gain, t.realized_gain)
    n = len(t)
    concordant = sum(
        (t.predicted_gain[i] - t.predicted_gain[j]) * (t.realized_gain[i] - t.realized_gain[j]) > 0
        for i in range(n) for j in range(i+1, n)
    )
    total_pairs = n*(n-1)//2

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    t.to_csv(Path(args.outdir)/"proxy_selection_table.csv", index=False)
    summary = {
        "n_proxies": n,
        "spearman_rho_predicted_vs_realized": float(rho),
        "pairwise_concordant": int(concordant),
        "pairwise_total": int(total_pairs),
        "top1_by_predicted_equals_top1_by_realized": bool(pred_rank.idxmin()==real_rank.idxmin()),
        "max_abs_pred_error": float(t.abs_pred_error.max()),
        "mean_abs_pred_error": float(t.abs_pred_error.mean()),
    }
    json.dump(summary, open(Path(args.outdir)/"proxy_selection_summary.json","w"), indent=2)
    print(t.to_string(index=False))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
