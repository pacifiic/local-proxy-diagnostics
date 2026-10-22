#!/usr/bin/env python3
"""[exp09-12] Pool-size (samples-per-prompt m) robustness of the local law.

Tests sensitivity to the fixed candidate-pool size: subsample m' in a grid of the 32 samples/prompt,
refit (A,B) on the CALIBRATION subsample (exactly as 05), recompute the exact proxy-
tilt gold frontier over the EVAL subsample on a fixed d-grid (exactly as 09/11), and
track prediction-vs-frontier MAE plus coefficient dispersion as functions of m'.
Pure CPU re-analysis of pool_scored.jsonl — decoupled from DPO so it isolates the
Monte-Carlo pool-size effect alone.

DATA-LAYER SELF-CHECK (hard-fail): at m'=32 the subsample is a permutation of all 32
samples, so the refit (A,B,A_nu) MUST equal the locked coefficients.json to <1e-9
(moments are permutation-invariant). Any drift => a re-implementation bug in the fit
or the join, caught loudly before any figure is produced.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common_experiment_utils import theory_coefficients  # noqa: E402


# --- frontier machinery: byte-identical to 09_frontier_check / 11_combined_analysis ---
def tilt_w(r, t):
    a = t * (r - r.max()); w = np.exp(a); return w / w.sum()


def kl(r, t):
    w = tilt_w(r, t); m = len(r)
    return float(np.log(m) + np.sum(w * np.log(np.clip(w, 1e-300, None))))


def find_t(r, d, cap=1e4):
    if d <= 0 or np.var(r) == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while kl(r, hi) < d:
        lo, hi = hi, hi * 2
        if hi > cap:
            return cap
    for _ in range(80):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if kl(r, mid) < d else (lo, mid)
    return hi


def fit_AB(by_prompt_sub, min_samples):
    """Replicates 05_fit_calibration: per-prompt theory_coefficients, uniform mean.
    Same skip rule (len<min_samples or zero proxy/gold variance)."""
    a_l, b_l, anu_l = [], [], []
    for pid, (r_arr, j_arr) in by_prompt_sub.items():
        if len(r_arr) < min_samples or np.var(r_arr) == 0 or np.var(j_arr) == 0:
            continue
        c = theory_coefficients(j_arr, r_arr)
        a_l.append(float(c.a_exact)); b_l.append(float(c.b_theory)); anu_l.append(float(c.a_nu))
    if not a_l:
        return None
    return float(np.mean(a_l)), float(np.mean(b_l)), float(np.mean(anu_l)), len(a_l)


def frontier_grid(eval_sub, d_grid):
    """Mean over ALL eval prompts of exact proxy-tilt gold gain at each d.
    Matches 09/11 exactly: a zero-proxy-variance prompt yields find_t=0 -> uniform
    tilt -> 0 gain, and is INCLUDED (not skipped) so the denominator is n_eval."""
    out = np.zeros(len(d_grid))
    n = 0
    for pid, (r_arr, j_arr) in eval_sub.items():
        n += 1
        base = j_arr.mean()
        for k, d in enumerate(d_grid):
            t = find_t(r_arr, d)  # 0 when var==0 -> contributes 0
            out[k] += float(np.sum(tilt_w(r_arr, t) * j_arr) - base)
    return out / max(n, 1), n


def load_pool(path):
    cal, ev = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    for l in open(path):
        r = json.loads(l)
        tgt = cal if r["split"] == "calibration" else ev if r["split"] == "eval" else None
        if tgt is None:
            continue
        tgt[r["prompt_id"]][0].append(r["proxy_score"])
        tgt[r["prompt_id"]][1].append(r["gold_score"])
    to_np = lambda d: {p: (np.array(rr, float), np.array(jj, float)) for p, (rr, jj) in d.items()}
    return to_np(cal), to_np(ev)


def subsample(by_prompt, m, rng):
    sub = {}
    for pid, (r_arr, j_arr) in by_prompt.items():
        n = len(r_arr)
        idx = np.arange(n) if m >= n else rng.choice(n, m, replace=False)
        sub[pid] = (r_arr[idx], j_arr[idx])
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="a completed exp09 run (uses its pool_scored.jsonl + coefficients.json)")
    ap.add_argument("--m-grid", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    ap.add_argument("--d-grid", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    root = Path(args.run_dir)
    cal, ev = load_pool(root / "pool_scored.jsonl")
    locked = json.load(open(root / "coefficients.json"))
    d_grid = np.array(args.d_grid, float)
    full_m = max(len(v[0]) for v in cal.values())
    print(f"cal prompts {len(cal)}, eval prompts {len(ev)}, samples/prompt {full_m}")
    print(f"locked A={locked['A']:.6f} B={locked['B']:.6f} A_nu={locked['A_nu']:.6f}")

    # ---- data-layer self-check: m'=full identity refit must reproduce the lock ----
    fit_full = fit_AB(cal, args.min_samples)
    assert fit_full is not None, "identity refit produced no usable prompts"
    dA, dB, dNu = abs(fit_full[0] - locked["A"]), abs(fit_full[1] - locked["B"]), abs(fit_full[2] - locked["A_nu"])
    print(f"[self-check] identity refit A={fit_full[0]:.6f} B={fit_full[1]:.6f} A_nu={fit_full[2]:.6f} "
          f"| |dA|={dA:.2e} |dB|={dB:.2e} |dNu|={dNu:.2e}")
    if max(dA, dB, dNu) > 1e-9:
        raise SystemExit(f"SELF-CHECK FAILED: refit != locked coefficients (drift {max(dA,dB,dNu):.2e}) — fit re-impl bug")
    print("[self-check] PASS — refit reproduces locked coefficients exactly\n")

    per_rows, agg_rows = [], []
    for mi, m in enumerate(args.m_grid):
        A_s, B_s, Anu_s, nfit_s = [], [], [], []
        front_s = []  # (reps, len(d_grid))
        for rep in range(args.reps):
            seed = mi * 100000 + rep  # deterministic, no Date/random
            rng = np.random.default_rng(seed)
            cal_sub = subsample(cal, m, rng)
            ev_sub = subsample(ev, m, rng)
            fit = fit_AB(cal_sub, args.min_samples)
            if fit is None:
                continue
            A, B, Anu, nfit = fit
            fr, nev = frontier_grid(ev_sub, d_grid)
            A_s.append(A); B_s.append(B); Anu_s.append(Anu); nfit_s.append(nfit)
            front_s.append(fr)
            for k, d in enumerate(d_grid):
                pred = A * np.sqrt(d) + B * d
                per_rows.append({"m": m, "rep": rep, "d": float(d), "A": A, "B": B,
                                 "pred_AB": pred, "frontier": float(fr[k]),
                                 "abs_err": abs(pred - float(fr[k]))})
        A_s = np.array(A_s); B_s = np.array(B_s); front_s = np.array(front_s)  # (reps,D)
        # prediction per rep on the d-grid, then MAE over d, then over reps
        pred_grid = A_s[:, None] * np.sqrt(d_grid)[None, :] + B_s[:, None] * d_grid[None, :]
        mae_pred = np.abs(pred_grid - front_s).mean(axis=1)          # per-rep MAE vs frontier
        mae_zero = np.abs(front_s).mean(axis=1)                       # zero-baseline MAE vs frontier
        agg_rows.append({
            "m": m, "n_rep": len(A_s),
            "A_mean": A_s.mean(), "A_std": A_s.std(ddof=1),
            "B_mean": B_s.mean(), "B_std": B_s.std(ddof=1),
            "Anu_mean": np.array(Anu_s).mean(),
            "nfit_med": int(np.median(nfit_s)),
            "frontier_mean_over_d": front_s.mean(),
            "mae_pred_mean": mae_pred.mean(), "mae_pred_std": mae_pred.std(ddof=1),
            "mae_zero_mean": mae_zero.mean(),
            "skill_vs_zero": mae_zero.mean() / max(mae_pred.mean(), 1e-9),
        })
        r = agg_rows[-1]
        print(f"m={m:2d} | A {r['A_mean']:.4f}±{r['A_std']:.4f}  B {r['B_mean']:.4f}±{r['B_std']:.4f} "
              f"| MAE(pred vs frontier) {r['mae_pred_mean']:.4f}±{r['mae_pred_std']:.4f} "
              f"| MAE(zero) {r['mae_zero_mean']:.4f} | skill×{r['skill_vs_zero']:.1f}")

    import pandas as pd
    pd.DataFrame(per_rows).to_csv(f"{args.out_prefix}_per_rep.csv", index=False)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nwrote {args.out_prefix}_summary.csv  and  _per_rep.csv")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].errorbar(agg.m, agg.A_mean, yerr=agg.A_std, marker="o", label="A (refit)", capsize=3)
        ax[0].errorbar(agg.m, agg.B_mean, yerr=agg.B_std, marker="s", label="B (refit)", capsize=3)
        ax[0].axhline(locked["A"], ls="--", c="C0", alpha=.6, label="A locked (m=32)")
        ax[0].axhline(locked["B"], ls="--", c="C1", alpha=.6, label="B locked (m=32)")
        ax[0].set_xlabel("samples per prompt m'"); ax[0].set_ylabel("coefficient")
        ax[0].set_title("Coefficient stability vs pool size"); ax[0].legend(fontsize=8)
        ax[1].plot(agg.m, agg.mae_pred_mean, marker="o", label="MAE: A√d+Bd vs frontier")
        ax[1].fill_between(agg.m, agg.mae_pred_mean - agg.mae_pred_std,
                           agg.mae_pred_mean + agg.mae_pred_std, alpha=.2)
        ax[1].plot(agg.m, agg.mae_zero_mean, marker="x", ls="--", label="MAE: zero baseline")
        ax[1].set_xlabel("samples per prompt m'"); ax[1].set_ylabel("MAE over d-grid")
        ax[1].set_title("Prediction accuracy vs pool size"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_figure.png", dpi=150)
        print(f"wrote {args.out_prefix}_figure.png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
