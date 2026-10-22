#!/usr/bin/env python3
"""[exp13-audit] Debugging loop 2 for build_paired_stats: independent re-derivations.

Loop-2 history (what this audit already caught):
  * exp03 'gpt' track == exp06 exp27 gpt4o (per-path MAE identical to 4e-7)
    -> double-counting; the builder now keeps only the exp27 version.
  * first negative control (permute the OTHER predictor's errors across paths) was
    WRONG: it destroys the marginal mean difference too, so the sign test correctly
    rejects — the check itself had a logic bug while "running fine". The valid null
    for a paired sign test is RANDOM SIGN FLIPS of the paired differences.
  * published merged_*.csv aggregation is the PATH-BALANCED mean — matches our unit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="output dir of build_paired_stats.py")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    out = Path(args.outdir); O = Path(args.repo_root) / "outputs"

    print("== [1] no duplicated sources in summary ==")
    res = pd.read_csv(out / "paired_stats_summary.csv")
    ck("GPT-4o judge (exp03 track)" not in set(res.source),
       "exp03-gpt duplicate excluded from summary")
    per_gpt27 = pd.read_csv(out / "per_path_gpt4o_exp27.csv", index_col=0)
    sp = pd.read_csv(O / "exp03_boost50_20260706_004007/splits_gpt_boost50.csv")
    sp = sp[np.isclose(sp.cal_frac, 0.30)]
    dup = sp.groupby("path_id").abs_err_theory.mean().sort_index()
    ck(bool(np.allclose(dup.values, per_gpt27["theory"].sort_index().values, atol=1e-5)),
       f"duplication re-confirmed (exp03-gpt == exp27-gpt, max diff "
       f"{np.abs(dup.values - per_gpt27['theory'].sort_index().values).max():.1e}) — exclusion justified")

    print("== [2] valid negative control: random sign flips of paired diffs ==")
    rng = np.random.default_rng(1)
    h = pd.read_csv(out / "per_path_summeval_exp03.csv", index_col=0)
    d = (h["theory"] - h["aonly"]).values
    rej = 0
    for _ in range(400):
        flip = d * rng.choice([-1, 1], len(d))
        w = int((flip < 0).sum()); l = int((flip > 0).sum())
        if binomtest(w, w + l, 0.5).pvalue < 0.05:
            rej += 1
    rate = rej / 400
    ck(0.01 <= rate <= 0.10, f"sign-flip null rejection rate {rate:.3f} in [0.01, 0.10] (~0.05)")

    print("== [3] bootstrap CI coherence ==")
    # mean_diff must lie inside its own CI; significant flag consistent with CI & n
    inside = bool(((res.mean_diff >= res.ci_lo) & (res.mean_diff <= res.ci_hi)).all())
    ck(inside, "mean_diff within [ci_lo, ci_hi] for every row")
    flag_ok = bool(((res.significant) == (((res.ci_hi < 0) | (res.ci_lo > 0)) & (res.n_paths >= 8))).all())
    ck(flag_ok, "significant flag == (CI excludes 0) AND (n_paths >= 8)")

    print("== [4] per-path tables: units, NaN, rep counts ==")
    for f in sorted(out.glob("per_path_*.csv")):
        t = pd.read_csv(f, index_col=0)
        ck(int(t.isna().sum().sum()) == 0, f"{f.name}: no NaN")
        ck(t.index.is_unique, f"{f.name}: path_id unique")

    print("== [5] determinism: builder rerun byte-identical ==")
    import hashlib, subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, str(Path(__file__).parent / "build_paired_stats.py"),
                        "--outdir", td], check=True, capture_output=True)
        m1 = hashlib.md5(open(out / "paired_stats_summary.csv", "rb").read()).hexdigest()
        m2 = hashlib.md5(open(Path(td) / "paired_stats_summary.csv", "rb").read()).hexdigest()
        ck(m1 == m2, f"rerun summary md5 identical ({m1[:10]}…)")

    print()
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)}"); [print("  -", m) for m in FAIL]; sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
