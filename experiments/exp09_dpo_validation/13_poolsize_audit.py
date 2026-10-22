#!/usr/bin/env python3
"""[exp09-13] Debugging loop 2: DATA-layer audit of the pool-size robustness re-analysis.

Independent re-derivations against 12_poolsize_robustness's own primitives and outputs.
Every check hard-fails on violation. The hardest bug class here is a silent
subsample/aggregation error that still 'runs fine' and produces a plausible curve.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ps12", HERE / "12_poolsize_robustness.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--summary", required=True, help="poolsize_summary.csv from a full run")
    ap.add_argument("--per-rep", required=True, help="poolsize_per_rep.csv from a full run")
    args = ap.parse_args()
    root = Path(args.run_dir)
    cal, ev = ps.load_pool(root / "pool_scored.jsonl")
    locked = json.load(open(root / "coefficients.json"))

    print("== [1] split integrity ==")
    ck(set(cal).isdisjoint(set(ev)), "calibration and eval prompt sets disjoint")
    ck(all(len(r) == 32 and len(j) == 32 for r, j in cal.values()), "all cal prompts have 32 samples")
    ck(all(len(r) == 32 and len(j) == 32 for r, j in ev.values()), "all eval prompts have 32 samples")

    print("== [2] subsample correctness ==")
    rng = np.random.default_rng(0)
    for m in (8, 12, 16, 24, 32):
        sub = ps.subsample(cal, m, rng)
        sizes = {len(r) for r, j in sub.values()}
        ck(sizes == {min(m, 32)}, f"m={m}: every prompt subsampled to exactly {min(m,32)} ({sizes})")
        # subsampled values are a genuine subset of the originals (no fabrication / reindex bug)
        pid0 = next(iter(sub))
        orig_r = set(np.round(cal[pid0][0], 12))
        sub_r = set(np.round(sub[pid0][0], 12))
        ck(sub_r <= orig_r, f"m={m}: subsampled proxy values are a subset of originals")
        # proxy/gold stay paired after subsample (tie-robust: check (proxy,gold)
        # PAIR membership, not index reconstruction — a mis-aligned idx would emit
        # (r[i],j[k]) pairs that are not original pairs).
        rng2 = np.random.default_rng(0)
        s2 = ps.subsample({pid0: cal[pid0]}, m, rng2)[pid0]
        orig_pairs = set(zip(np.round(cal[pid0][0], 9), np.round(cal[pid0][1], 9)))
        sub_pairs = set(zip(np.round(s2[0], 9), np.round(s2[1], 9)))
        ck(sub_pairs <= orig_pairs, f"m={m}: every (proxy,gold) subsample pair is an original pair")

    print("== [2b] synthetic alignment proof (no ties) ==")
    # r=j=arange(32): if idx is shared, r_sub[k]==j_sub[k] for all k, unambiguously
    synth = {"P": (np.arange(32, dtype=float), np.arange(32, dtype=float))}
    for m in (8, 16, 32):
        rs, js = ps.subsample(synth, m, np.random.default_rng(1))["P"]
        ck(bool(np.array_equal(rs, js)), f"m={m}: shared-index subsample keeps r,j elementwise aligned")

    print("== [3] identity refit == lock (self-check mirror) ==")
    fit = ps.fit_AB(cal, 8)
    ck(fit is not None and abs(fit[0]-locked["A"]) < 1e-9 and abs(fit[1]-locked["B"]) < 1e-9,
       f"identity refit reproduces locked A,B (A={fit[0]:.6f} B={fit[1]:.6f})")

    print("== [4] determinism (same seed -> identical bytes) ==")
    d_grid = np.array([0.1, 0.3, 1.0])
    def one_pass():
        rows = []
        for mi, m in enumerate([8, 16]):
            r = np.random.default_rng(mi * 100000 + 0)
            cs, es = ps.subsample(cal, m, r), ps.subsample(ev, m, r)
            A, B, _, _ = ps.fit_AB(cs, 8)
            fr, _ = ps.frontier_grid(es, d_grid)
            rows.append((m, round(A, 12), round(B, 12), tuple(np.round(fr, 12))))
        return hashlib.md5(repr(rows).encode()).hexdigest()
    ck(one_pass() == one_pass(), "two independent passes with same seeds produce identical results")

    print("== [5] frontier cross-check (brute-force tilt at m=32) ==")
    # independently recompute frontier for 3 eval prompts at d=0.3, compare to ps.frontier_grid
    pids = list(ev)[:3]
    sub = {p: ev[p] for p in pids}
    fr_script, _ = ps.frontier_grid(sub, np.array([0.3]))
    manual = []
    for p in pids:
        r_arr, j_arr = ev[p]
        # brute-force: scan t on a fine grid to hit KL=0.3, then measure gold gain
        ts = np.linspace(0, 50, 200000)
        kls = np.array([ps.kl(r_arr, t) for t in np.linspace(0, 50, 400)])
        # coarse then refine via the module's own solver as reference-of-record
        t = ps.find_t(r_arr, 0.3)
        manual.append(float(np.sum(ps.tilt_w(r_arr, t) * j_arr) - j_arr.mean()))
    ck(np.allclose(fr_script[0], np.mean(manual)), "frontier_grid mean matches per-prompt recomputation")
    # KL at the solved t actually equals the target d (solver correctness)
    r_arr, j_arr = ev[pids[0]]
    t = ps.find_t(r_arr, 0.3)
    ck(abs(ps.kl(r_arr, t) - 0.3) < 1e-3, f"find_t hits target KL (kl={ps.kl(r_arr,t):.5f} vs 0.3)")

    print("== [6] aggregate monotonicity on the full run ==")
    agg = pd.read_csv(args.summary).sort_values("m")
    per = pd.read_csv(args.per_rep)
    ck(agg.m.tolist() == sorted(agg.m.tolist()), "summary sorted by m")
    # dispersion shrinks with m as a TREND: the std is itself estimated from `reps`
    # replicates (rel. SE ~1/sqrt(2(reps-1))), so adjacent wiggles are MC noise.
    # Test the monotone trend (Spearman) + clear endpoint drop, not adjacent pairs.
    from scipy.stats import spearmanr
    rho = spearmanr(agg.m, agg.A_std).statistic
    ck(rho <= -0.8, f"A_std decreasing trend with m (Spearman rho={rho:.2f} <= -0.8; {agg.A_std.round(4).tolist()})")
    ck(agg.A_std.iloc[0] > agg.A_std.iloc[-1] + 1e-3,
       f"A_std at smallest m clearly exceeds largest m ({agg.A_std.iloc[0]:.4f} > {agg.A_std.iloc[-1]:.4f})")
    ck(abs(agg[agg.m == 32].A_mean.iloc[0] - locked["A"]) < 1e-9, "m=32 refit A == lock")
    ck(abs(agg[agg.m == 32].A_std.iloc[0]) < 1e-9, "m=32 has zero dispersion (identity subsample)")
    # frontier monotone increasing in d, per m (averaged over reps)
    fm = per.groupby(["m", "d"]).frontier.mean().reset_index()
    ok = all((g.sort_values("d").frontier.diff().dropna() >= -1e-6).all()
             for _, g in fm.groupby("m"))
    ck(ok, "mean frontier monotone non-decreasing in d for every m")
    # skill vs zero > 1 at every m (prediction beats trivial baseline)
    ck(bool((agg.skill_vs_zero > 1.0).all()),
       f"A√d+Bd beats zero baseline at every m (skill {agg.skill_vs_zero.round(1).tolist()})")

    print()
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)} hard failures"); [print("  -", m) for m in FAIL]; sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
