#!/usr/bin/env python3
"""[exp11-14] Cross-model generalization of the local law (Tier-2 review item).

Two claims, evaluated on a SECOND policy family (zephyr-7b-beta; Qwen was 3B):
  P1 (replication, unconditional): (A,B) fit on Zephyr's own calibration split
     predicts Zephyr's eval-pool exact tilt frontier — the law replicates.
  P2 (transfer, bonus): Qwen-locked (A,B) from exp09 predicting the Zephyr
     frontier. Theory does NOT require this (coefficients are policy-conditional
     moments); either outcome is informative and reported honestly.

Reuses the AUDITED primitives from exp09-12 (frontier/tilt/fit) via import so the
estimator is byte-identical to the one validated by 13_poolsize_audit.
Prompt-level bootstrap CIs over the eval set.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "ps12", HERE.parent / "exp09_dpo_validation" / "12_poolsize_robustness.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)


def per_prompt_frontier(ev, d_grid):
    """frontier_i(d) for every eval prompt — matrix (n_prompts, n_d)."""
    pids = sorted(ev)
    M = np.zeros((len(pids), len(d_grid)))
    for i, pid in enumerate(pids):
        r_arr, j_arr = ev[pid]
        base = j_arr.mean()
        for k, d in enumerate(d_grid):
            t = ps.find_t(r_arr, d)
            M[i, k] = float(np.sum(ps.tilt_w(r_arr, t) * j_arr) - base)
    return pids, M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zephyr-run", required=True, help="dir with zephyr pool_scored.jsonl + coefficients.json")
    ap.add_argument("--qwen-coefficients", required=True, help="exp09 V4 coefficients.json (locked)")
    ap.add_argument("--d-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    zr = Path(args.zephyr_run)
    cal, ev = ps.load_pool(zr / "pool_scored.jsonl")
    cz = json.load(open(zr / "coefficients.json"))
    cq = json.load(open(args.qwen_coefficients))
    d_grid = np.array(args.d_grid, float)
    print(f"zephyr pool: cal {len(cal)} prompts, eval {len(ev)} prompts")
    print(f"within (zephyr) A={cz['A']:.4f} B={cz['B']:.4f} A_nu={cz['A_nu']:.4f} (n_used {cz['n_prompts_used']})")
    print(f"cross  (qwen)   A={cq['A']:.4f} B={cq['B']:.4f} A_nu={cq['A_nu']:.4f}")

    # self-check (data layer): refit on zephyr cal reproduces coefficients.json
    fit = ps.fit_AB(cal, 8)
    drift = max(abs(fit[0]-cz["A"]), abs(fit[1]-cz["B"]), abs(fit[2]-cz["A_nu"]))
    if drift > 1e-9:
        raise SystemExit(f"SELF-CHECK FAILED: zephyr refit != coefficients.json (drift {drift:.2e})")
    print(f"[self-check] zephyr identity refit reproduces lock (drift {drift:.2e}) — PASS\n")

    pids, F = per_prompt_frontier(ev, d_grid)   # (P, D)
    frontier = F.mean(axis=0)

    preds = {
        "within_AB":   cz["A"]*np.sqrt(d_grid) + cz["B"]*d_grid,
        "within_Aonly": cz["A"]*np.sqrt(d_grid),
        "within_Anu":  cz["A_nu"]*np.sqrt(d_grid),
        "cross_AB":    cq["A"]*np.sqrt(d_grid) + cq["B"]*d_grid,
        "cross_Aonly": cq["A"]*np.sqrt(d_grid),
        "zero":        np.zeros(len(d_grid)),
    }

    # prompt-level bootstrap of the frontier -> CI on each predictor's MAE
    rng = np.random.default_rng(0)
    P = F.shape[0]
    boot_mae = {k: [] for k in preds}
    for _ in range(args.n_boot):
        idx = rng.integers(0, P, P)
        fb = F[idx].mean(axis=0)
        for k, pr in preds.items():
            boot_mae[k].append(float(np.mean(np.abs(pr - fb))))

    rows = []
    for k, pr in preds.items():
        mae = float(np.mean(np.abs(pr - frontier)))
        lo, hi = np.percentile(boot_mae[k], [2.5, 97.5])
        rows.append({"predictor": k, "mae_vs_frontier": mae,
                     "ci_lo": float(lo), "ci_hi": float(hi)})
        print(f"{k:13s}: MAE {mae:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    per_d = pd.DataFrame({"d": d_grid, "frontier": frontier, **preds})
    per_d.to_csv(f"{args.out_prefix}_per_d.csv", index=False)
    pd.DataFrame(F, index=pids, columns=[f"d{d}" for d in d_grid]).to_csv(
        f"{args.out_prefix}_per_prompt_frontier.csv")

    # headline ratios
    mz = summ.set_index("predictor").mae_vs_frontier
    print(f"\nP1 replication: within_AB {mz['within_AB']:.4f} vs zero {mz['zero']:.4f} "
          f"({mz['zero']/max(mz['within_AB'],1e-9):.1f}x)")
    print(f"P2 transfer:    cross_AB  {mz['cross_AB']:.4f} vs zero {mz['zero']:.4f} "
          f"({mz['zero']/max(mz['cross_AB'],1e-9):.1f}x)  | vs within {mz['cross_AB']/max(mz['within_AB'],1e-9):.2f}x")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.plot(d_grid, frontier, "k-o", lw=2, label="Zephyr exact tilt frontier")
        ax.plot(d_grid, preds["within_AB"], "-s", c="C0", label=f"within: A√d+Bd (A={cz['A']:.3f})")
        ax.plot(d_grid, preds["cross_AB"], "-^", c="C1", label=f"cross (Qwen-locked, A={cq['A']:.3f})")
        ax.plot(d_grid, preds["within_Anu"], "--", c="C2", label="within: reliability-only A_ν√d")
        ax.axhline(0, c="#999", lw=0.8)
        ax.set_xlabel("KL budget d (nats)"); ax.set_ylabel("gold gain")
        ax.set_title("Cross-model: law refit on Zephyr-7B;\nQwen-3B-calibrated coefficients as transfer test", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(f"{args.out_prefix}_figure.png", dpi=150)
        print(f"figure -> {args.out_prefix}_figure.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
