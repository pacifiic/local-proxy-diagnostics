#!/usr/bin/env python3
"""[exp18-audit] Debugging loop 2: independent re-derivations for the cal-budget sweep.

Focus: the data seams — (a) one full cell recomputed end-to-end from the raw CSV must
match the stored row bit-for-bit; (b) at the largest budget the coefficients must
approach the full-cal-pool values (convergence sanity); (c) stored 'actual' must match
exp14's conditional actual for the same paths (cross-experiment consistency).
"""
import argparse
import importlib.util
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("xb", HERE / "build_calbudget.py")
xb = importlib.util.module_from_spec(_s); _s.loader.exec_module(xb)
bc = xb.bc

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    cells = pd.read_csv(out / "calbudget_cells.csv")

    print("== [1] structural ==")
    ck(int(cells.isna().sum().sum()) == 0, "no NaN")
    ck(bool((cells[["err_AB", "err_Aonly", "err_empfit"]] >= 0).all().all()),
       "all errors nonnegative")
    ck(bool((cells.groupby(["dataset", "path"]).actual.nunique() == 1).all()),
       "one fixed eval frontier per path")

    print("== [2] one cell recomputed end-to-end from raw CSV ==")
    row = cells[(cells.dataset == "SummEval") & (cells.n_cal == 20)].iloc[0]
    ds_want, path_want = row.dataset, row.path
    for ds_label, grp, mat, raters in bc.dataset_matrices():
        if ds_label != ds_want:
            continue
        for pr in raters:
            if f"{grp}__r{pr}" == path_want:
                pools = bc.pools_from_matrix(mat, pr, raters)
                seed = zlib.crc32(f"{ds_label}|{path_want}".encode()) % 10**6
                rows, actual, reason = xb.run_path(pools, seed, xb.REPS[ds_label])
                ck(reason is None, "recomputed path reaches the evaluation radius")
                sub = [r for r in rows if r["n_cal"] == row.n_cal and r["rep"] == row.rep]
                ck(len(sub) == 1, "cell uniquely recomputed")
                for k in ("actual", "err_AB", "err_Aonly", "err_empfit"):
                    ck(abs(sub[0][k] - row[k]) < 1e-12, f"{k} matches stored ({sub[0][k]:.6f})")
                # [3] convergence: n_cal=30 mean coefficients vs full cal-pool
                prompts = sorted(pools)
                rng0 = np.random.default_rng(seed)
                order = rng0.permutation(len(prompts))
                half = len(prompts) // 2
                cal_pool = {prompts[i]: pools[prompts[i]] for i in order[half:]}
                A_full, B_full, _, _ = bc.fit_conditional(cal_pool)
                A_s = []
                for rep in range(10):
                    rng = np.random.default_rng(seed * 1000 + 30 * 37 + rep)
                    ids = sorted(cal_pool)
                    pick = list(rng.choice(len(ids), min(30, len(ids)), replace=False))
                    fit = bc.fit_conditional({ids[i]: cal_pool[ids[i]] for i in pick})
                    if fit:
                        A_s.append(fit[0])
                print("== [3] convergence at n_cal=30 ==")
                ck(abs(np.mean(A_s) - A_full) < 0.05,
                   f"n_cal=30 mean A {np.mean(A_s):.4f} ≈ full-pool A {A_full:.4f}")
                break
        break

    print("== [4] cross-experiment: actual vs exp14 conditional actual (ballpark) ==")
    e14 = pd.read_csv(Path(open("outputs/.exp14_current").read().strip()) / "conditional_per_path.csv")
    m14 = e14[e14.dataset == "SummEval"].actual_mean.mean()
    m18 = cells[cells.dataset == "SummEval"].groupby("path").actual.first().mean()
    ck(abs(m14 - m18) < 0.05,
       f"SummEval mean actualΔ: exp18 {m18:.4f} ≈ exp14 {m14:.4f} (approximate comparison: different split protocols)")

    print()
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)}"); [print("  -", m) for m in FAIL]; sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
