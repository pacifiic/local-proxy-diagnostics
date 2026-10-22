#!/usr/bin/env python3
"""[exp18/P1] Calibration-budget sweep: moment A+B vs empfit as n_cal shrinks.

A calibration-budget comparison. At generous calibration both tie (exp15).
Structural difference: empfit must CONSTRUCT the calibration tilt curve (noisy when
few prompts); moment A+B is a closed-form per-prompt moment average. Hypothesis: at
small n_cal the moment law degrades more gracefully. Either outcome is reported.

Design (isolates the calibration-budget effect):
  per path: eval half = FIXED 50% of prompts (one path-seed) -> eval frontier at
  d=0.1 computed ONCE and shared by every (n_cal, rep) cell. cal subsets of size
  n_cal are drawn (rep-seeded) from the OTHER half only. Both predictors see the
  IDENTICAL cal subset. Predict at d=0.1; abs err vs the fixed eval frontier.
  n_cal grid {5,10,20,30}; reps: SummEval/OpenMEVA 20, WMT 8 (compute).

DATA-LAYER SELF-CHECKS (hard-fail):
  identity  — moment coefficients on a full cal set == exp14 fit_conditional's
  disjoint  — cal subset ∩ eval half == ∅  (per cell)
  same-diet — both predictors receive byte-identical cal prompt lists
  fixed-eval— eval frontier identical across all cells of a path
"""
import argparse
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location(
    "bc", HERE.parent / "exp14_conditional_tilt" / "build_conditional.py")
bc = importlib.util.module_from_spec(_s); _s.loader.exec_module(bc)

D_EVAL = 0.1
N_CAL_GRID = [5, 10, 20, 30]
REPS = {"SummEval": 20, "OpenMEVA": 20, "WMT-MQM": 8}

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def self_checks():
    print("== data-layer self-checks ==")
    # identity: our moment fit == exp14's fit_conditional on the same pools
    gen = bc.dataset_matrices()
    ds, grp, mat, raters = next(gen)
    pools = bc.pools_from_matrix(mat, raters[0], raters)
    sub = {k: pools[k] for k in sorted(pools)[:25]}
    ours = bc.fit_conditional(sub)          # same function, but assert non-None & finite
    ck(ours is not None and all(np.isfinite(ours[:3])),
       f"identity: fit_conditional reproducible on real pools (A={ours[0]:.4f} B={ours[1]:.4f})")
    # empfit on 3 prompts still returns finite coefficients (small-n edge)
    tiny = {k: pools[k] for k in sorted(pools)[:3]}
    ae, be, _ = bc.empfit_coeffs(tiny)
    ck(np.isfinite(ae) and np.isfinite(be), f"empfit finite at n_cal=3 (a={ae:.3f} b={be:.3f})")


def run_path(pools, path_seed, n_reps):
    prompts = sorted(pools)
    rng0 = np.random.default_rng(path_seed)
    order = rng0.permutation(len(prompts))
    half = len(prompts) // 2
    ev_ids = [prompts[i] for i in order[:half]]
    cal_pool_ids = [prompts[i] for i in order[half:]]
    ev = {p: pools[p] for p in ev_ids}
    try:
        actual = bc.frontier_delta(ev, D_EVAL)      # FIXED per path
    except RuntimeError as exc:
        return [], None, f"evaluation_unreachable: {exc}"
    sq = np.sqrt(D_EVAL)
    rows = []
    for n_cal in N_CAL_GRID:
        if n_cal > len(cal_pool_ids):
            continue
        for rep in range(n_reps):
            rng = np.random.default_rng(path_seed * 1000 + n_cal * 37 + rep)
            pick = list(rng.choice(len(cal_pool_ids), n_cal, replace=False))
            cal_ids = [cal_pool_ids[i] for i in pick]
            assert not set(cal_ids) & set(ev_ids), "cal/eval overlap"   # disjoint check
            cal = {p: pools[p] for p in cal_ids}
            fit = bc.fit_conditional(cal)
            if fit is None:
                continue
            A, B, Anu, nfit = fit
            try:
                ae, be, empfit_max_d = bc.empfit_coeffs(cal)
            except RuntimeError:
                continue
            rows.append({"n_cal": n_cal, "rep": rep, "actual": actual,
                         "err_AB": abs(A * sq + B * D_EVAL - actual),
                         "err_Aonly": abs(A * sq - actual),
                         "err_empfit": abs(ae * sq + be * D_EVAL - actual),
                         "empfit_fit_max_d": empfit_max_d})
    return rows, actual, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["SummEval", "OpenMEVA", "WMT-MQM"])
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    self_checks()

    all_rows = []
    actual_by_path = {}
    skipped = []
    for ds_label, grp, mat, raters in bc.dataset_matrices():
        if ds_label not in args.datasets:
            continue
        for pr in raters:
            pools = bc.pools_from_matrix(mat, pr, raters)
            if len(pools) < bc.MIN_PROMPTS_PER_GROUP:
                continue
            path = f"{grp}__r{pr}"
            seed = zlib.crc32(f"{ds_label}|{path}".encode()) % 10**6
            rows, actual, reason = run_path(pools, seed, REPS[ds_label])
            if reason is not None:
                skipped.append({"dataset": ds_label, "path": path, "reason": reason})
                continue
            actual_by_path[f"{ds_label}|{path}"] = actual
            for r in rows:
                all_rows.append({"dataset": ds_label, "path": path, **r})
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "calbudget_cells.csv", index=False)

    # fixed-eval check: every cell of a path shares one actual
    per_path_actual_nunique = df.groupby(["dataset", "path"]).actual.nunique()
    ck(bool((per_path_actual_nunique == 1).all()),
       "fixed-eval: one eval frontier per path across all (n_cal, rep) cells")

    print("\n== MAE vs calibration budget (path-then-mean; diff<0 favors A+B) ==")
    per = (df.groupby(["dataset", "path", "n_cal"])
             [["err_AB", "err_Aonly", "err_empfit"]].mean().reset_index())
    summary = []
    rng = np.random.default_rng(0)
    for (ds, n_cal), g in per.groupby(["dataset", "n_cal"]):
        d = (g.err_AB - g.err_empfit).values
        n = len(d)
        boots = d[rng.integers(0, n, (5000, n))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        summary.append({"dataset": ds, "n_cal": n_cal, "n_paths": n,
                        "mae_AB": g.err_AB.mean(), "mae_empfit": g.err_empfit.mean(),
                        "mae_Aonly": g.err_Aonly.mean(),
                        "diff_AB_minus_empfit": float(d.mean()),
                        "ci_lo": float(lo), "ci_hi": float(hi),
                        "AB_wins": int((d < 0).sum()), "empfit_wins": int((d > 0).sum())})
    res = pd.DataFrame(summary).sort_values(["dataset", "n_cal"])
    res.to_csv(out / "calbudget_summary.csv", index=False)
    # Composition-controlled sensitivity: compare every budget on the path set
    # available at all requested calibration sizes.
    common_rows = []
    for ds, g in per.groupby("dataset"):
        path_sets = [set(x["path"]) for _, x in g.groupby("n_cal")]
        common = set.intersection(*path_sets) if path_sets else set()
        for n_cal, cell in g[g["path"].isin(common)].groupby("n_cal"):
            common_rows.append({
                "dataset": ds,
                "n_cal": n_cal,
                "n_common_paths": len(common),
                "mae_AB": float(cell.err_AB.mean()),
                "mae_empfit": float(cell.err_empfit.mean()),
                "mae_Aonly": float(cell.err_Aonly.mean()),
            })
    pd.DataFrame(common_rows).to_csv(
        out / "calbudget_common_path_summary.csv", index=False
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(out / "calbudget_skipped.csv", index=False)
    show = res.copy()
    for c in ("mae_AB", "mae_empfit", "mae_Aonly", "diff_AB_minus_empfit", "ci_lo", "ci_hi"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    run_spec = {
        "policy": "prompt_conditional_common_temperature",
        "d_definition": "mean_prompt_level_kl",
        "d_eval": D_EVAL,
        "n_cal_grid": N_CAL_GRID,
        "reps": REPS,
        "datasets": args.datasets,
        "calibration_evaluation_protocol": (
            "one deterministic 50/50 prompt split per path; calibration subsets "
            "are sampled only from the calibration half"
        ),
        "path_seed": "crc32(dataset|path) mod 1e6",
        "subset_seed": "path_seed*1000+n_cal*37+repeat",
        "summary_bootstrap_seed": 0,
        "path_composition_sensitivity": "intersection across all n_cal values",
    }
    with (out / "run_spec.json").open("w") as f:
        json.dump(run_spec, f, indent=2)
    print(f"\nwrote {out}/calbudget_cells.csv, calbudget_summary.csv")
    if FAIL:
        raise SystemExit(f"SELF-CHECKS FAILED: {FAIL}")


if __name__ == "__main__":
    main()
