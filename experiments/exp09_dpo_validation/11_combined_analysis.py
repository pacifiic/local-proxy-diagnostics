#!/usr/bin/env python3
"""[exp09-11] Final analysis: pool local-window DPO points across runs, per-prompt
prediction, 3-rung frontier decomposition, baseline comparison.

Per eval prompt at each (run, step): d_i = d_reweight, and
  frontier_i = exact proxy-tilt gold gain at KL d_i over the 32-sample eval pool
  pred_AB_i  = A√d_i + B·d_i   (locked A,B from disjoint calibration)
Rung 1 (theory):        pred_AB   vs frontier
Rung 2 (DPO efficiency): frontier vs gain_reweight_gold (DPO's implied reweighting)
Rung 3 (off-support):    gain_reweight_gold vs gain_fresh_gold  (V4/v5 fp32 only)
Baselines: A-only, A_nu (reliability), zero — all per-prompt then averaged.
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--coefficients-json", required=True)
    ap.add_argument("--d-max", type=float, default=1.0, help="local-window ceiling on d_reweight")
    ap.add_argument("--ess-min", type=float, default=8.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    coef = json.load(open(args.coefficients_json))
    A, B, A_nu = coef["A"], coef["B"], coef["A_nu"]

    # eval + calibration pools per-prompt (proxy,gold) — identical across runs (shared prep)
    pool, cal_pool = defaultdict(list), defaultdict(list)
    for l in open(Path(args.run_dirs[0]) / "pool_scored.jsonl"):
        r = json.loads(l)
        if r["split"] == "eval":
            pool[r["prompt_id"]].append((r["proxy_score"], r["gold_score"]))
        elif r["split"] == "calibration":
            cal_pool[r["prompt_id"]].append((r["proxy_score"], r["gold_score"]))

    # exp05-suite baselines, ALL fit on calibration only (no eval leakage), same
    # per-prompt-conditional-then-average convention as A,B:
    #   pearson/spearman: a_corr = sqrt(2)*sigma_j*corr(r,j) per cal prompt, averaged
    #   isotonic: monotone proxy->gold map h fit on pooled cal rows (scale-fair
    #             "trust the proxy"); prediction later = tilt-average of h(r_eval)
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression
    a_p_l, a_s_l = [], []
    for pid, vals in cal_pool.items():
        r_arr = np.array([v[0] for v in vals]); j_arr = np.array([v[1] for v in vals])
        if len(vals) < 8 or np.std(r_arr) == 0 or np.std(j_arr) == 0:
            continue
        sig_j = float(np.std(j_arr))  # population, matches theory_coefficients pop_var
        a_p_l.append(np.sqrt(2.0) * sig_j * float(np.corrcoef(r_arr, j_arr)[0, 1]))
        a_s_l.append(np.sqrt(2.0) * sig_j * float(spearmanr(r_arr, j_arr).statistic))
    A_pear, A_spear = float(np.mean(a_p_l)), float(np.mean(a_s_l))
    cal_r = np.array([v[0] for vals in cal_pool.values() for v in vals])
    cal_j = np.array([v[1] for vals in cal_pool.values() for v in vals])
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(cal_r, cal_j)

    rows = []
    for rd in args.run_dirs:
        for pp in sorted(glob.glob(f"{rd}/*/per_prompt.csv")):
            tag = Path(pp).parent.name
            df = pd.read_csv(pp)
            for t in df.itertuples():
                d = max(float(t.d_reweight), 0.0)
                vals = pool.get(t.prompt_id)
                if not vals:
                    continue
                r_arr = np.array([v[0] for v in vals]); j_arr = np.array([v[1] for v in vals])
                tt = find_t(r_arr, d)
                w = tilt_w(r_arr, tt)
                frontier = float(np.sum(w * j_arr) - j_arr.mean())
                proxy_frontier = float(np.sum(w * r_arr) - r_arr.mean())  # naive: tilt proxy, MEASURE proxy
                h = iso.predict(r_arr)  # calibrated proxy->gold, then tilt-average (exp05 convention)
                rows.append({
                    "run": Path(rd).name, "tag": tag, "prompt_id": t.prompt_id,
                    "d": d, "ess": float(t.ess),
                    "pred_AB": A * np.sqrt(d) + B * d,
                    "pred_Aonly": A * np.sqrt(d),
                    "pred_Anu": A_nu * np.sqrt(d),
                    "pred_pearson": A_pear * np.sqrt(d),
                    "pred_spearman": A_spear * np.sqrt(d),
                    "pred_isotonic": float(np.sum(w * h) - h.mean()),
                    "frontier": frontier,
                    "proxy_frontier": proxy_frontier,
                    "dpo_reweight_gold": float(t.gain_reweight_gold),
                    "dpo_reweight_proxy": float(getattr(t, "gain_reweight_proxy", np.nan)),
                    "dpo_fresh_gold": float(getattr(t, "gain_fresh_gold", np.nan)),
                })
    per = pd.DataFrame(rows)
    # guard against silent under-pooling: every run dir must contribute rows
    # (an incomplete run has no per_prompt.csv and would vanish from the glob)
    contributed = set(per.run.unique()) if len(per) else set()
    missing = [Path(rd).name for rd in args.run_dirs if Path(rd).name not in contributed]
    if missing:
        raise SystemExit(f"run dirs contributed no per_prompt rows (incomplete?): {missing}")
    per.to_csv(f"{args.out_prefix}_per_prompt.csv", index=False)

    # per (run,tag) aggregate, local-window filter on the group's median d + ess
    g = per.groupby(["run", "tag"])
    agg = g.agg(d_med=("d", "median"), ess_med=("ess", "median"),
                pred_AB=("pred_AB", "mean"), pred_Aonly=("pred_Aonly", "mean"),
                pred_Anu=("pred_Anu", "mean"),
                pred_pearson=("pred_pearson", "mean"), pred_spearman=("pred_spearman", "mean"),
                pred_isotonic=("pred_isotonic", "mean"), frontier=("frontier", "mean"),
                proxy_frontier=("proxy_frontier", "mean"),
                dpo_reweight=("dpo_reweight_gold", "mean"),
                dpo_reweight_proxy=("dpo_reweight_proxy", "mean"),
                dpo_fresh=("dpo_fresh_gold", "mean")).reset_index()
    agg = agg.sort_values("d_med")
    loc = agg[(agg.d_med <= args.d_max) & (agg.ess_med >= args.ess_min)].copy()
    # rungs
    for t in (agg, loc):
        t["rung1_theory_err"] = (t.pred_AB - t.frontier).abs()
        t["rung2_dpo_gap"] = t.frontier - t.dpo_reweight
        t["rung3_offsupport"] = t.dpo_fresh - t.dpo_reweight
    agg.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print("=== ALL points (sorted by d) ===")
    print(agg[["run", "tag", "d_med", "ess_med", "pred_AB", "frontier",
               "dpo_reweight", "dpo_fresh"]].round(4).to_string(index=False))
    print(f"\n=== LOCAL WINDOW (d<= {args.d_max}, ESS>= {args.ess_min}): {len(loc)} points ===")
    if len(loc):
        print(loc[["tag", "d_med", "ess_med", "pred_AB", "frontier", "dpo_reweight",
                   "rung1_theory_err", "rung2_dpo_gap"]].round(4).to_string(index=False))
        # baseline comparison vs DPO realized reweight gold
        for name, col in [("A+B", "pred_AB"), ("A-only", "pred_Aonly"),
                          ("A_nu", "pred_Anu"), ("pearson", "pred_pearson"),
                          ("spearman", "pred_spearman"), ("isotonic", "pred_isotonic"),
                          ("zero", None), ("frontier", "frontier")]:
            pred = np.zeros(len(loc)) if col is None else loc[col].values
            mae = float(np.mean(np.abs(pred - loc.dpo_reweight.values)))
            mae_f = float(np.mean(np.abs((np.zeros(len(loc)) if col is None else loc[col].values) - loc.frontier.values)))
            print(f"  {name:9s}: MAE vs DPO-realized {mae:.4f} | MAE vs frontier {mae_f:.4f}")
        # "trust the proxy" illustration. NOTE: raw proxy and gold live on different
        # scales (RM logit vs regression head), so raw proxy gain is NOT a fair MAE
        # competitor — it is an OVER-OPTIMISM factor. The scale-fair naive (OLS map
        # proxy->gold, beta=cov/var_r) reduces to A*sqrt(d)=A-only at leading order,
        # already tabled above; nothing fair beats the law.
        overopt = float(loc.proxy_frontier.mean() / max(loc.frontier.mean(), 1e-9))
        print(f"  [proxy over-optimism] proxy-frontier {loc.proxy_frontier.mean():.3f} vs gold-frontier "
              f"{loc.frontier.mean():.4f}  ->  raw proxy inflates gain {overopt:.1f}x (scale-fair naive == A-only)")
    print(f"\nA={A:.4f} B={B:.4f} A_nu={A_nu:.4f}")


if __name__ == "__main__":
    main()
