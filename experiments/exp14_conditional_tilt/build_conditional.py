#!/usr/bin/env python3
"""[exp14/R2'] Prompt-conditional tilt re-analysis on the HUMAN-gold datasets —
the prompt-conditional scope question: does the law hold when the prompt marginal is
FIXED (tilt within each prompt's own response set), not pooled across items?

Conditional units (verified in R2.0'):
  SummEval : prompt=article (100), items=16 systems, groups=(dimension, rater_type),
             paths=held-out rater (3 expert / 5 turker) -> 32 paths
  OpenMEVA : prompt=story (200), items=5 systems, groups=dataset(roc/wp), 10 paths
  WMT-MQM  : prompt=doc-segment (item_id='SYSTEM___doc__seg', parse verified: '___'
             exactly once in 100% rows), items=10-15 systems, groups=dataset_id,
             paths=held-out rater

Protocol:
  prompt-level 30/70 cal/eval split, n_repeats=20, seeds fixed; conditional (A,B)
  via context-averaged cumulants over calibration prompts; conditional actual uses
  one common temperature t across evaluation prompts, chosen so that the mean
  prompt-level KL is d=0.1 (zero-variance prompts contribute zero and are included);
  predictors A+B / A-only / A_nu / empfit(2-coef LS on cal conditional curve) / zero;
  resplits averaged WITHIN path; same paired stats as exp13.
  SUCCESS CRITERION (pre-registered): conditional A+B beats zero and A_nu.
  A-only ordering is secondary (R1' framing). Unfavorable cells reported as-is.

DATA-LAYER SELF-CHECKS (hard-fail): synthetic-long pivot round-trip; held-out rater
excluded from gold (leak check); grid completeness; d target hit by solver.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
_s = importlib.util.spec_from_file_location(
    "ps12", HERE.parent / "exp09_dpo_validation" / "12_poolsize_robustness.py")
ps = importlib.util.module_from_spec(_s); _s.loader.exec_module(ps)
_s2 = importlib.util.spec_from_file_location(
    "e13", HERE.parents[1] / "engines" / "13_paired_stats" / "build_paired_stats.py")
e13 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(e13)

D_EVAL = 0.1
D_CAL_GRID = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
N_REPEATS = 20
CAL_FRAC = 0.30
MIN_PROMPTS_PER_GROUP = 30

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def pools_from_matrix(mat: pd.DataFrame, proxy_rater, raters):
    """mat: index=prompt, columns=MultiIndex (system, rater) scores.
    Returns {prompt: (r_arr over systems, j_arr over systems)} for one held-out path."""
    others = [r for r in raters if r != proxy_rater]
    pools = {}
    for prompt, row in mat.iterrows():
        wide = row.unstack()  # index=system, columns=rater
        r_arr = wide[proxy_rater].to_numpy(float)
        j_arr = wide[others].mean(axis=1).to_numpy(float)
        m = np.isfinite(r_arr) & np.isfinite(j_arr)
        if m.sum() >= 3:
            pools[str(prompt)] = (r_arr[m], j_arr[m])
    return pools


def fit_conditional(pools_cal):
    """Common-temperature coefficients from context-averaged cumulants."""
    moments = []
    for r, j in pools_cal.values():
        if len(r) < 3:
            continue
        rc = r - np.mean(r)
        jc = j - np.mean(j)
        moments.append({
            "v_r": float(np.mean(rc**2)),
            "v_j": float(np.mean(jc**2)),
            "c1": float(np.mean(jc * rc)),
            "k_r": float(np.mean(rc**3)),
            "k_jrr": float(np.mean(jc * rc**2)),
        })
    if not moments:
        return None
    m = pd.DataFrame(moments).mean()
    v_r = float(m.v_r)
    if not np.isfinite(v_r) or v_r <= 0:
        return None
    c1 = float(m.c1)
    k_r = float(m.k_r)
    k_jrr = float(m.k_jrr)
    a = np.sqrt(2.0) * c1 / np.sqrt(v_r)
    b = k_jrr / v_r - 2.0 * c1 * k_r / (3.0 * v_r**2)
    # Reliability-only analogue under the same common-temperature geometry:
    # replace each within-prompt Cov(J,r) by Var(J), then average.
    a_nu = np.sqrt(2.0) * float(m.v_j) / np.sqrt(v_r)
    return float(a), float(b), float(a_nu), len(moments)


def pack_pools(pools):
    """Stack equal-sized prompt pools so KL/gain evaluation is vectorized."""
    grouped = {}
    for r, j in pools.values():
        grouped.setdefault(len(r), []).append((r, j))
    return {
        n: (
            np.stack([r for r, _ in values]).astype(float),
            np.stack([j for _, j in values]).astype(float),
        )
        for n, values in grouped.items()
    }


def mean_prompt_kl_packed(packed, t):
    total, count = 0.0, 0
    for r, _ in packed.values():
        z = t * r
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        log_mean_exp = z_max[:, 0] + np.log(np.mean(exp_z, axis=1))
        kl = t * np.sum(w * r, axis=1) - log_mean_exp
        total += float(np.sum(kl))
        count += len(kl)
    return total / count


def mean_prompt_kl(pools, t):
    return mean_prompt_kl_packed(pack_pools(pools), t)


def max_common_kl_packed(packed):
    """Limit of the mean prompt KL as t -> infinity, accounting for max ties."""
    limits = []
    for n, (r, _) in packed.items():
        row_max = np.max(r, axis=1, keepdims=True)
        n_max = np.sum(np.isclose(r, row_max, rtol=0.0, atol=1e-12), axis=1)
        limits.extend(np.log(n / n_max))
    return float(np.mean(limits))


def max_common_kl(pools):
    return max_common_kl_packed(pack_pools(pools))


def find_common_t_packed(packed, d, tol=1e-10):
    """Find one shared t whose mean prompt-level KL equals d."""
    if d < 0:
        raise ValueError("d must be non-negative")
    if d == 0:
        return 0.0
    max_d = max_common_kl_packed(packed)
    if d >= max_d - tol:
        raise RuntimeError(
            f"Common-temperature frontier max KL={max_d:.6g} cannot reach d={d:g}"
        )
    lo, hi = 0.0, 1.0
    d_hi = mean_prompt_kl_packed(packed, hi)
    for _ in range(60):
        if d_hi >= d:
            break
        lo, hi = hi, hi * 2.0
        next_d = mean_prompt_kl_packed(packed, hi)
        if next_d <= d_hi + 1e-14:
            raise RuntimeError(f"Common-temperature frontier cannot reach d={d:g}")
        d_hi = next_d
    else:
        raise RuntimeError(f"Could not bracket common t for d={d:g}")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        d_mid = mean_prompt_kl_packed(packed, mid)
        if abs(d_mid - d) <= tol:
            return mid
        if d_mid < d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def find_common_t(pools, d, tol=1e-10):
    return find_common_t_packed(pack_pools(pools), d, tol)


def frontier_delta(pools, d):
    packed = pack_pools(pools)
    t = find_common_t_packed(packed, d)
    total, count = 0.0, 0
    for r, j in packed.values():
        z = t * r
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        w = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        gains = np.sum(w * j, axis=1) - np.mean(j, axis=1)
        total += float(np.sum(gains))
        count += len(gains)
    return total / count


def empfit_coeffs(pools_cal):
    fit_max = min(float(D_CAL_GRID[-1]), 0.95 * max_common_kl(pools_cal))
    if fit_max <= 0:
        raise RuntimeError("Calibration frontier has zero reachable KL radius")
    fit_grid = D_CAL_GRID[D_CAL_GRID <= fit_max]
    if len(fit_grid) < 3:
        fit_grid = np.linspace(fit_max / 5.0, fit_max, 5)
    Q = np.array([frontier_delta(pools_cal, d) for d in fit_grid])
    X = np.column_stack([np.sqrt(fit_grid), fit_grid])
    a, b = np.linalg.lstsq(X, Q, rcond=None)[0]
    return float(a), float(b), float(fit_grid[-1])


def run_path(pools, rng):
    prompts = sorted(pools)
    n_cal = max(3, int(round(CAL_FRAC * len(prompts))))
    order = rng.permutation(len(prompts))
    cal = {prompts[i]: pools[prompts[i]] for i in order[:n_cal]}
    ev = {prompts[i]: pools[prompts[i]] for i in order[n_cal:]}
    fit = fit_conditional(cal)
    if fit is None:
        return None
    A, B, Anu, _ = fit
    try:
        ae, be, empfit_max_d = empfit_coeffs(cal)
        actual = frontier_delta(ev, D_EVAL)
    except RuntimeError:
        return None
    sq, dd = np.sqrt(D_EVAL), D_EVAL
    preds = {"theory": A * sq + B * dd, "aonly": A * sq, "anu": Anu * sq,
             "empfit": ae * sq + be * dd, "zero": 0.0}
    return {k: abs(v - actual) for k, v in preds.items()} | {
        "actual": actual,
        "empfit_fit_max_d": empfit_max_d,
        "cal_max_d": max_common_kl(cal),
        "eval_max_d": max_common_kl(ev),
    }


def dataset_matrices():
    """yield (dataset_label, group_label, matrix(prompt x (system,rater)), raters)."""
    d = pd.read_csv(DATA_DIR / "summeval_long.csv")
    for (dim, rt), g in d.groupby(["dimension", "rater_type"]):
        mat = g.pivot_table(index="article_id", columns=["model_id", "rater_idx"],
                            values="score", aggfunc="first")
        yield "SummEval", f"{dim}|{rt}", mat, sorted(g.rater_idx.unique())
    o = pd.read_csv(DATA_DIR / "openmeva_long.csv")
    for ds, g in o.groupby("dataset"):
        mat = g.pivot_table(index="story_id", columns=["model_id", "rater_idx"],
                            values="score", aggfunc="first")
        yield "OpenMEVA", str(ds), mat, sorted(g.rater_idx.unique())
    w = pd.read_csv(DATA_DIR / "wmt_mqm_long.csv")
    parts = w.item_id.astype(str).str.split("___", n=1, expand=True)
    w["system"], w["seg_key"] = parts[0], parts[1]
    for ds, g in w.groupby("dataset_id"):
        mat = g.pivot_table(index="seg_key", columns=["system", "rater"],
                            values="score", aggfunc="first")
        if mat.shape[0] < MIN_PROMPTS_PER_GROUP:
            continue
        yield "WMT-MQM", str(ds), mat, sorted(g.rater.unique())


def self_checks():
    print("== data-layer self-checks ==")
    # [1] synthetic pivot round-trip: known long -> pools must reproduce known values
    long = pd.DataFrame([{"prompt": p, "system": s, "rater": r,
                          "score": (i * 7 + j * 3 + k) * 0.1}
                         for i, p in enumerate(("p1", "p2"))
                         for j, s in enumerate(("s1", "s2", "s3"))
                         for k, r in enumerate((0, 1, 2))])
    mat = long.pivot_table(index="prompt", columns=["system", "rater"],
                           values="score", aggfunc="first")
    pools = pools_from_matrix(mat, proxy_rater=1, raters=[0, 1, 2])
    r_exp = np.array([(0 * 7 + j * 3 + 1) * 0.1 for j in range(3)])   # p1, rater1
    j_exp = np.array([np.mean([(0 * 7 + j * 3 + 0) * 0.1, (0 * 7 + j * 3 + 2) * 0.1])
                      for j in range(3)])                               # p1, raters 0&2
    ck(bool(np.allclose(pools["p1"][0], r_exp) and np.allclose(pools["p1"][1], j_exp)),
       "synthetic pivot round-trip: proxy=rater1 scores, gold=mean(rater0,2), system-aligned")
    # [2] leak check: gold must change when proxy rater changes
    pools2 = pools_from_matrix(mat, proxy_rater=0, raters=[0, 1, 2])
    ck(not np.allclose(pools["p1"][1], pools2["p1"][1]),
       "held-out rater excluded from gold (gold differs across paths)")
    # [3] one common t hits the average prompt-level KL target
    toy = {
        "p1": (np.array([0.1, 0.5, 0.2, 0.9, 0.4]), np.arange(5.0)),
        "p2": (np.array([0.0, 0.2, 0.8, 1.2]), np.arange(4.0)),
    }
    t = find_common_t(toy, 0.1)
    d_hit = mean_prompt_kl(toy, t)
    ck(abs(d_hit - 0.1) < 1e-8,
       f"common t hits mean prompt KL d=0.1 (t={t:.4f}, mean KL={d_hit:.8f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    self_checks()

    per_path_rows = []
    for ds_label, grp, mat, raters in dataset_matrices():
        for pr in raters:
            pools = pools_from_matrix(mat, pr, raters)
            if len(pools) < MIN_PROMPTS_PER_GROUP:
                continue
            accs = []
            # deterministic path seed: python hash() is per-process salted -> crc32
            import zlib
            pseed = zlib.crc32(f"{ds_label}|{grp}|{pr}".encode()) % 10**6
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(pseed * 100 + rep)
                res = run_path(pools, rng)
                if res:
                    accs.append(res)
            if not accs:
                continue
            df = pd.DataFrame(accs)
            per_path_rows.append({"dataset": ds_label, "group": grp, "path": f"{grp}__r{pr}",
                                  "n_prompts": len(pools), "n_reps": len(accs),
                                  "actual_mean": df.actual.mean(),
                                  "empfit_fit_max_d_min": df.empfit_fit_max_d.min(),
                                  "cal_max_d_min": df.cal_max_d.min(),
                                  "eval_max_d_min": df.eval_max_d.min(),
                                  **{k: df[k].mean() for k in
                                     ("theory", "aonly", "anu", "empfit", "zero")}})
    pp = pd.DataFrame(per_path_rows)
    pp.to_csv(out / "conditional_per_path.csv", index=False)

    print("\n== conditional per-dataset paired stats (unit=path, d=0.1) ==")
    rng = np.random.default_rng(0)
    stats = []
    for ds, g in pp.groupby("dataset"):
        tab = g.set_index("path")[["theory", "aonly", "anu", "empfit", "zero"]]
        print(f"\n[{ds}] paths={len(tab)}  actualΔ(0.1) mean={g.actual_mean.mean():+.4f}  "
              f"MAE: theory {tab.theory.mean():.4f} | aonly {tab.aonly.mean():.4f} | "
              f"anu {tab.anu.mean():.4f} | empfit {tab.empfit.mean():.4f} | zero {tab.zero.mean():.4f}")
        for other in ("aonly", "anu", "empfit", "zero"):
            st = e13.paired_stats(tab, "theory", other, rng)
            stats.append({"dataset": ds, **st})
    res = pd.DataFrame(stats)
    res.to_csv(out / "conditional_paired_stats.csv", index=False)
    show = res[["dataset", "contrast", "n_paths", "mean_diff", "ci_lo", "ci_hi",
                "wins", "ties", "losses", "sign_p", "significant"]].copy()
    for c in ("mean_diff", "ci_lo", "ci_hi"):
        show[c] = show[c].round(4)
    show["sign_p"] = show["sign_p"].round(3)
    print("\n" + show.to_string(index=False))

    json.dump({"policy": "prompt_conditional_common_temperature",
               "d_definition": "mean_prompt_level_kl",
               "d_eval": D_EVAL, "cal_grid": D_CAL_GRID.tolist(),
               "n_repeats": N_REPEATS, "random_seed": "crc32(path)*100+repeat",
               "cal_frac": CAL_FRAC, "min_prompts": MIN_PROMPTS_PER_GROUP},
              open(out / "run_spec.json", "w"), indent=2)
    print(f"\nwrote {out}/conditional_per_path.csv, conditional_paired_stats.csv")
    if FAIL:
        raise SystemExit(f"SELF-CHECKS FAILED: {FAIL}")


if __name__ == "__main__":
    main()
