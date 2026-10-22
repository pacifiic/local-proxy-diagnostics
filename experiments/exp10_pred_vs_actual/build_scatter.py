#!/usr/bin/env python3
"""[exp10] THE scatter: predicted vs actual optimization gain, pooled across every
canonical prospective run (per-path points), + Pearson/Spearman/R^2.

Sources (all 2026-07-06+ canonical, bug-fixed):
  exp06 proxies (GPT-4o, open judge, Skywork RM), exp02 ROUGE-1/2/L  -> exp27 paths files
  exp09 DPO checkpoints (local window)                                -> per-prompt aggregates
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    R = Path(args.repo_root); O = R/"outputs"
    SRC = [
        ("GPT-4o judge",  O/"exp06_final_20260706_022627/proxy_gpt4o/exp27_prospective_paths.csv"),
        ("Open judge",    O/"exp06_final_20260706_022627/proxy_openjudge/exp27_prospective_paths.csv"),
        ("Open RM",       O/"exp06_final_20260706_022627/proxy_rm_skywork/exp27_prospective_paths.csv"),
        ("ROUGE-1",       O/"exp02_fixed_rouge_20260706_020516/metric_rouge1/exp27_prospective_paths.csv"),
        ("ROUGE-2",       O/"exp02_fixed_rouge_20260706_020516/metric_rouge2/exp27_prospective_paths.csv"),
        ("ROUGE-L",       O/"exp02_fixed_rouge_20260706_020516/metric_rougeL/exp27_prospective_paths.csv"),
    ]
    rows = []
    for label, p in SRC:
        df = pd.read_csv(p)
        for t in df.itertuples():
            rows.append({"source": label, "kind": "reweighting (exp27)",
                         "pred": t.pred_theory_delta_0p1, "actual": t.actual_eval_delta_0p1})
    # exp09 DPO local-window checkpoints (per-checkpoint aggregate, reweight realized)
    fs = sorted(glob.glob(str(O/"exp09_dpo_v4_*/exp09_final_summary.csv")))
    if fs:
        s = pd.read_csv(fs[-1])
        loc = s[(s.d_med <= 1.0) & (s.ess_med >= 8)]
        for t in loc.itertuples():
            rows.append({"source": f"DPO {t.tag}", "kind": "actual DPO training (exp09)",
                         "pred": t.pred_AB, "actual": t.dpo_reweight})
    d = pd.DataFrame(rows).dropna()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out/"scatter_points.csv", index=False)

    stats = {}
    for name, sub in [("all", d)] + [(k, g) for k, g in d.groupby("kind")]:
        pr = pearsonr(sub.pred, sub.actual); sp = spearmanr(sub.pred, sub.actual)
        ss_res = float(((sub.actual - sub.pred) ** 2).sum())
        ss_tot = float(((sub.actual - sub.actual.mean()) ** 2).sum())
        stats[name] = {"n": len(sub), "pearson": float(pr.statistic),
                       "spearman": float(sp.statistic), "r2_identity": 1 - ss_res / ss_tot}
    json.dump(stats, open(out/"scatter_stats.json", "w"), indent=2)
    for k, v in stats.items():
        print(f"{k:32s} n={v['n']:4d}  Pearson={v['pearson']:+.3f}  Spearman={v['spearman']:+.3f}  R2(y=x)={v['r2_identity']:+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    colors = {"GPT-4o judge":"#4269d0","Open judge":"#efb118","Open RM":"#ff725c",
              "ROUGE-1":"#6cc5b0","ROUGE-2":"#3ca951","ROUGE-L":"#97bbf5"}
    for src, g in d[d.kind.str.startswith("reweighting")].groupby("source"):
        ax.scatter(g.actual, g.pred, s=10, alpha=0.45, label=src, color=colors.get(src, "#9c6b4e"), linewidths=0)
    dp = d[d.kind.str.startswith("actual DPO")]
    if len(dp):
        ax.scatter(dp.actual, dp.pred, s=70, marker="*", color="#ff8ab7",
                   edgecolors="#333", linewidths=0.6, label="DPO checkpoints (exp09)", zorder=5)
    lim = [min(d.actual.min(), d.pred.min()) - 0.01, max(d.actual.max(), d.pred.max()) + 0.01]
    ax.plot(lim, lim, ls="--", c="#888", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual held-out gain @ d=0.1")
    ax.set_ylabel("Predicted gain  $\\hat{A}\\sqrt{d}+\\hat{B}d$")
    a = stats["all"]
    ax.set_title(f"Predicted vs actual optimization gain\nn={a['n']}, Pearson {a['pearson']:.3f}, Spearman {a['spearman']:.3f}, $R^2$(y=x) {a['r2_identity']:.3f}", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out/"fig_pred_vs_actual_gain.pdf")
    fig.savefig(out/"fig_pred_vs_actual_gain.png", dpi=180)
    print("figure ->", out/"fig_pred_vs_actual_gain.png")

if __name__ == "__main__":
    main()
