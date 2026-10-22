#!/usr/bin/env python3
"""[exp12] Unified strength-sweep figure (LLM-review Tier-1 assembly item).

One law across optimization strength AND mechanism, with the validity window
explicit — pure assembly of three canonical artifacts (no new computation):
  A: exp05 real-LLM exponential tilt, d in {0.01,0.05,0.1}, 20 prompt-resplits
  B: exp09 real DPO training, d 0.01..3.4 (local window + off-support collapse)
  C: exp11 second policy family (zephyr), within vs cross calibration
Every plotted number is printed alongside its source row for eyeball-diffing.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    R = Path(args.repo_root); O = R / "outputs"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # --- A: exp05 tilt sweep (20 resplits) ---
    p05 = pd.read_csv(O / "exp05_real_llm_20260706_032457/gain_curves_20resplit/llm_gain_curve_predictions.csv")
    a_ab = p05[p05.baseline == "A+B"].groupby("d").agg(
        pred_m=("predicted_gain", "mean"), pred_s=("predicted_gain", "std"),
        act_m=("actual_gain", "mean"), act_s=("actual_gain", "std")).reset_index()
    print("[A] exp05 tilt sweep (A+B vs actual, 20 resplits):")
    print(a_ab.round(4).to_string(index=False))

    # --- B: exp09 DPO sweep ---
    s09 = pd.read_csv(O / "exp09_dpo_v4_20260710_051319/exp09_final_summary.csv").sort_values("d_med")
    print("\n[B] exp09 DPO sweep:")
    print(s09[["tag", "d_med", "ess_med", "pred_AB", "frontier", "dpo_reweight", "dpo_fresh"]]
          .round(4).to_string(index=False))

    # --- C: exp11 zephyr ---
    c11 = pd.read_csv(O / "exp11_cross_model_20260710_071053/cross_model_per_d.csv")
    print("\n[C] exp11 zephyr per-d:")
    print(c11[["d", "frontier", "within_AB", "cross_AB"]].round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # A
    ax[0].errorbar(a_ab.d, a_ab.act_m, yerr=a_ab.act_s, marker="o", c="k",
                   label="actual tilt gain (20 resplits)", capsize=3)
    ax[0].errorbar(a_ab.d, a_ab.pred_m, yerr=a_ab.pred_s, marker="s", c="C0",
                   label=r"predicted $\hat A\sqrt{d}+\hat B d$", capsize=3)
    ax[0].set_xlabel("KL budget d (nats)"); ax[0].set_ylabel("gold gain")
    ax[0].set_title("A. Exponential tilt on real-LLM pool\n(Qwen-7B candidates, exp05)", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    # B
    loc = s09[(s09.d_med <= 1.0) & (s09.ess_med >= 8)]
    off = s09[~((s09.d_med <= 1.0) & (s09.ess_med >= 8))]
    ax[1].axvspan(0.007, 1.0, color="C0", alpha=0.07, label="local window (d≤1, ESS≥8)")
    ax[1].plot(s09.d_med, s09.pred_AB, "-s", c="C0", label=r"predicted $\hat A\sqrt{d}+\hat B d$")
    ax[1].plot(s09.d_med, s09.frontier, "--o", c="k", label="exact tilt frontier @ d̂")
    ax[1].plot(loc.d_med, loc.dpo_reweight, "^", c="C2", ms=8, label="DPO realized (reweight)")
    ax[1].plot(off.d_med, off.dpo_reweight, "x", c="C3", ms=9, mew=2,
               label="off-support: ESS collapse")
    ax[1].set_xscale("log"); ax[1].set_xlabel("measured policy KL d̂ (nats, log)")
    ax[1].set_title("B. Actual DPO training strength sweep\n(Qwen-3B + LoRA, steps 64→1200, exp09)", fontsize=10)
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.25)

    # C
    ax[2].plot(c11.d, c11.frontier, "k-o", label="Zephyr tilt frontier")
    ax[2].plot(c11.d, c11.within_AB, "-s", c="C0", label="within-Zephyr A+B (MAE 0.0066)")
    ax[2].plot(c11.d, c11.cross_AB, "-^", c="C1", label="Qwen-calibrated A+B (MAE 0.0488)")
    ax[2].set_xlabel("KL budget d (nats)")
    ax[2].set_title("C. Second policy family\n(zephyr-7b-beta, exp11)", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

    for a in ax:
        a.axhline(0, c="#999", lw=0.7)
    fig.suptitle("One local law across optimization strength and mechanism — validity window explicit",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_strength_sweep_unified.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_strength_sweep_unified.png", dpi=180, bbox_inches="tight")
    print(f"\nfigure -> {out/'fig_strength_sweep_unified.png'}")


if __name__ == "__main__":
    main()
