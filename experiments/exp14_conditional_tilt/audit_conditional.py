#!/usr/bin/env python3
"""[exp14-audit] Debugging loop 2: independent re-derivations on REAL data.

The new (unaudited) code in build_conditional is the long-CSV -> per-prompt pool
pivot and the path loop. Every check below re-derives from the raw CSVs by a
DIFFERENT pandas route and compares, plus determinism and pooled-context checks.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
_s = importlib.util.spec_from_file_location("bc", HERE / "build_conditional.py")
bc = importlib.util.module_from_spec(_s); _s.loader.exec_module(bc)
ps = bc.ps

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--pooled-splits",
        type=Path,
        default=None,
        help="Optional pooled-tilt split CSV used only for the contextual comparison in check 6.",
    )
    args = ap.parse_args()
    out = Path(args.outdir)
    pp = pd.read_csv(out / "conditional_per_path.csv")

    print("== [1] path counts per dataset (units) ==")
    cnt = pp.groupby("dataset").path.nunique().to_dict()
    ck(cnt.get("SummEval") == 32, f"SummEval paths {cnt.get('SummEval')} == 32")
    ck(cnt.get("OpenMEVA") == 10, f"OpenMEVA paths {cnt.get('OpenMEVA')} == 10")
    ck(cnt.get("WMT-MQM", 0) >= 100, f"WMT paths {cnt.get('WMT-MQM')} >= 100")
    ck(bool((pp.n_reps == bc.N_REPEATS).all()), f"all paths have n_reps == {bc.N_REPEATS}")
    ck(int(pp.isna().sum().sum()) == 0, "no NaN in per-path table")

    print("== [2] independent pivot re-derivation (real SummEval prompt) ==")
    d = pd.read_csv(DATA_DIR / "summeval_long.csv")
    g = d[(d.dimension == "coherence") & (d.rater_type == "expert")]
    mat = g.pivot_table(index="article_id", columns=["model_id", "rater_idx"],
                        values="score", aggfunc="first")
    raters = sorted(g.rater_idx.unique())
    pools = bc.pools_from_matrix(mat, raters[0], raters)
    art = sorted(pools)[0]
    # different route: plain filtering, no pivot
    sub = g[g.article_id == art].sort_values("model_id")
    r_direct = sub[sub.rater_idx == raters[0]].set_index("model_id").score
    j_direct = (sub[sub.rater_idx != raters[0]].groupby("model_id").score.mean())
    r_pool, j_pool = pools[art]
    ck(bool(np.allclose(np.sort(r_pool), np.sort(r_direct.values))),
       "proxy scores match direct filter route")
    ck(bool(np.allclose(np.sort(j_pool), np.sort(j_direct.values))),
       "gold means match direct filter route")
    # alignment (not just multisets): same system order for r and j
    aligned = np.allclose(r_pool, r_direct.reindex(j_direct.index).values) and \
              np.allclose(j_pool, j_direct.values)
    ck(bool(aligned), "r and j are SYSTEM-ALIGNED (same index order), not just equal as sets")

    print("== [3] WMT parse spot-check ==")
    w = pd.read_csv(DATA_DIR / "wmt_mqm_long.csv")
    parts = w.item_id.astype(str).str.split("___", n=1, expand=True)
    ck(bool((w.item_id.astype(str).str.count("___") == 1).all()),
       "every WMT item_id splits exactly once on '___'")
    recon = parts[0] + "___" + parts[1]
    ck(bool((recon == w.item_id.astype(str)).all()), "parse is lossless (reconstruction identity)")

    print("== [4] common-temperature policy and coefficient formula ==")
    toy = {k: pools[k] for k in sorted(pools)[:10]}
    t_common = bc.find_common_t(toy, bc.D_EVAL)
    prompt_kls = np.array([ps.kl(r, t_common) for r, _ in toy.values()])
    ck(abs(prompt_kls.mean() - bc.D_EVAL) < 1e-8,
       "one shared t reaches the target mean prompt KL")
    ck(float(prompt_kls.std()) > 1e-5,
       "per-prompt KLs vary under common t (not equal-KL per-prompt tilts)")
    fit = bc.fit_conditional(toy)
    vr, c1, kr, kjrr = [], [], [], []
    for r, j in toy.values():
        rc, jc = r - r.mean(), j - j.mean()
        vr.append(np.mean(rc**2)); c1.append(np.mean(jc * rc))
        kr.append(np.mean(rc**3)); kjrr.append(np.mean(jc * rc**2))
    v, c, k3, kj = map(float, map(np.mean, (vr, c1, kr, kjrr)))
    a_manual = np.sqrt(2.0) * c / np.sqrt(v)
    b_manual = kj / v - 2.0 * c * k3 / (3.0 * v**2)
    ck(abs(fit[0] - a_manual) < 1e-12 and abs(fit[1] - b_manual) < 1e-12,
       "stored A_c/B_c equal independently computed context-averaged cumulants")

    print("== [5] determinism: one full path recomputed == stored row ==")
    import zlib
    row = pp[(pp.dataset == "SummEval")].iloc[0]
    grp = row.group
    dim, rt = grp.split("|")
    g2 = d[(d.dimension == dim) & (d.rater_type == rt)]
    mat2 = g2.pivot_table(index="article_id", columns=["model_id", "rater_idx"],
                          values="score", aggfunc="first")
    raters2 = sorted(g2.rater_idx.unique())
    pr = int(row.path.rsplit("__r", 1)[1])
    pools2 = bc.pools_from_matrix(mat2, pr, raters2)
    pseed = zlib.crc32(f"SummEval|{grp}|{pr}".encode()) % 10**6
    accs = []
    for rep in range(bc.N_REPEATS):
        res = bc.run_path(pools2, np.random.default_rng(pseed * 100 + rep))
        if res:
            accs.append(res)
    df = pd.DataFrame(accs)
    for col in ("theory", "aonly", "empfit", "zero"):
        ck(abs(df[col].mean() - row[col]) < 1e-12,
           f"recomputed path MAE[{col}] == stored ({df[col].mean():.6f})")

    print("== [6] pooled-vs-conditional context (not a pass/fail) ==")
    if args.pooled_splits is None:
        print("  SKIP optional pooled comparison (pass --pooled-splits to enable it)")
    else:
        sp = pd.read_csv(args.pooled_splits)
        required = {"cal_frac", "path_id", "abs_err_theory", "actual_delta"}
        missing = required.difference(sp.columns)
        ck(not missing, f"pooled split CSV has required columns (missing={sorted(missing)})")
        if not missing:
            sp = sp[np.isclose(sp.cal_frac, 0.3)]
            pooled_theory = sp.groupby("path_id").abs_err_theory.mean().mean()
            cond_theory = pp[pp.dataset == "SummEval"].theory.mean()
            pooled_act = sp.groupby("path_id").actual_delta.mean().mean()
            cond_act = pp[pp.dataset == "SummEval"].actual_mean.mean()
            print(f"  SummEval pooled: actualΔ {pooled_act:+.4f}, theory MAE {pooled_theory:.4f}")
            print(f"  SummEval cond  : actualΔ {cond_act:+.4f}, theory MAE {cond_theory:.4f}")

    print()
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)}"); [print("  -", m) for m in FAIL]; sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
