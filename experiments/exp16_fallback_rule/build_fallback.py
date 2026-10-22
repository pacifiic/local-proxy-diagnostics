#!/usr/bin/env python3
"""[exp16/R4'] Pre-registered data-driven fallback rule: when to trust B (use A+B)
vs drop to A-only. B depends on the 3rd moment and can be unstable;
they ask for a data-driven rule to fall back to A-only when B is unreliable.

PRE-REGISTERED RULE (fixed before evaluation; calibration-only, no eval labels):
  Bootstrap the calibration prompts B_BOOT times; recompute B̂ each resample.
  If the 90% percentile CI of B̂ CONTAINS 0  ->  B is not reliably signed  ->  use A-only.
  Else                                       ->  use A+B.
  Exactly one rule, one threshold (90%), applied untuned to every setting.

EVALUATION (honest, all settings incl. where the rule loses):
  For each path, compute MAE of {A+B, A-only, rule-selected} vs the actual gain.
  Report: (i) how often the rule picks the ex-post better of {A+B, A-only};
          (ii) rule MAE vs always-A+B and always-A-only;
          (iii) settings where the rule underperforms (reported, not hidden).

Settings pooled: exp14 conditional human paths (SummEval/WMT/OpenMEVA). Reuses
the audited common-temperature primitives. Self-check: rule is a pure function
of calibration (decision identical regardless of eval labels).
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_s2 = importlib.util.spec_from_file_location(
    "bc", HERE.parent / "exp14_conditional_tilt" / "build_conditional.py")
bc = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(bc)

B_BOOT = 1000
CI = 90.0
D_EVAL = 0.1


def prompt_moments(pools):
    rows = []
    for r, j in pools.values():
        if len(r) < 3:
            continue
        rc = r - np.mean(r)
        jc = j - np.mean(j)
        rows.append(
            [
                np.mean(rc**2),
                np.mean(jc**2),
                np.mean(jc * rc),
                np.mean(rc**3),
                np.mean(jc * rc**2),
            ]
        )
    return np.asarray(rows, dtype=float)


def coefficients_from_moments(moment_means):
    v_r, v_j, c1, k_r, k_jrr = np.moveaxis(moment_means, -1, 0)
    valid = np.isfinite(v_r) & (v_r > 1e-12)
    if np.ndim(v_r) == 0 and not bool(valid):
        raise RuntimeError("non-positive proxy variance in moment aggregate")
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(valid, np.sqrt(2.0) * c1 / np.sqrt(v_r), np.nan)
        b = np.where(
            valid,
            k_jrr / v_r - 2.0 * c1 * k_r / (3.0 * v_r**2),
            np.nan,
        )
        a_nu = np.where(
            valid, np.sqrt(2.0) * v_j / np.sqrt(v_r), np.nan
        )
    return a, b, a_nu


def bootstrap_b(moments, rng):
    """Collect exactly B_BOOT valid B draws, skipping degenerate resamples."""
    n = len(moments)
    valid_draws = []
    total_draws = 0
    while sum(len(values) for values in valid_draws) < B_BOOT:
        if total_draws >= 20 * B_BOOT:
            raise RuntimeError(
                "could not obtain enough positive-variance bootstrap draws"
            )
        batch_size = min(B_BOOT, 20 * B_BOOT - total_draws)
        sampled = rng.integers(0, n, (batch_size, n))
        boot_means = moments[sampled].mean(axis=1)
        _, boot_b, _ = coefficients_from_moments(boot_means)
        finite = np.asarray(boot_b, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            valid_draws.append(finite)
        total_draws += batch_size
    return np.concatenate(valid_draws)[:B_BOOT], total_draws


def coeffs_and_decision(pools_cal, rng):
    """Return (A, B, A_nu, use_AB) — the pre-registered rule, calibration-only."""
    fit = bc.fit_conditional(pools_cal)
    if fit is None:
        return None
    A, B, Anu, _ = fit
    moments = prompt_moments(pools_cal)
    if not len(moments):
        return None
    point = coefficients_from_moments(moments.mean(axis=0))
    if not np.allclose(point, (A, B, Anu), rtol=1e-12, atol=1e-12):
        raise RuntimeError("vectorized moments disagree with exp14 coefficients")
    boot_B, n_boot_draws = bootstrap_b(moments, rng)
    lo, hi = np.percentile(boot_B, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    use_AB = not (lo <= 0 <= hi)   # CI excludes 0 -> B reliably signed -> trust A+B
    return A, B, Anu, use_AB, (float(lo), float(hi)), n_boot_draws


def eval_path(pools, rng):
    prompts = sorted(pools)
    n_cal = max(3, int(round(0.30 * len(prompts))))
    order = rng.permutation(len(prompts))
    cal = {prompts[i]: pools[prompts[i]] for i in order[:n_cal]}
    ev = {prompts[i]: pools[prompts[i]] for i in order[n_cal:]}
    dec = coeffs_and_decision(cal, rng)
    if dec is None:
        return None
    A, B, Anu, use_AB, ci, n_boot_draws = dec
    actual = bc.frontier_delta(ev, D_EVAL)
    sq = np.sqrt(D_EVAL)
    e_ab = abs(A * sq + B * D_EVAL - actual)
    e_ao = abs(A * sq - actual)
    e_rule = e_ab if use_AB else e_ao
    return {"use_AB": use_AB, "ci_lo": ci[0], "ci_hi": ci[1],
            "n_boot_draws": n_boot_draws,
            "mae_AB": e_ab, "mae_Aonly": e_ao, "mae_rule": e_rule,
            "ideal": min(e_ab, e_ao), "rule_is_ideal": abs(e_rule - min(e_ab, e_ao)) < 1e-12}


def iter_settings(datasets):
    for ds_label, grp, mat, raters in bc.dataset_matrices():
        if ds_label not in datasets:
            continue
        for pr in raters:
            pools = bc.pools_from_matrix(mat, pr, raters)
            if len(pools) >= bc.MIN_PROMPTS_PER_GROUP:
                yield ds_label, f"{grp}__r{pr}", pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=["SummEval", "OpenMEVA", "WMT-MQM"],
        default=["SummEval", "OpenMEVA", "WMT-MQM"],
    )
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)

    import zlib
    rows = []
    for ds_label, path, pools in iter_settings(set(args.datasets)):
        accs = []
        seed = zlib.crc32(f"{ds_label}|{path}".encode()) % 10**6
        for rep in range(bc.N_REPEATS):
            r = eval_path(pools, np.random.default_rng(seed * 100 + rep))
            if r:
                accs.append(r)
        if accs:
            df = pd.DataFrame(accs)
            rows.append({"dataset": ds_label, "path": path,
                         "use_AB_rate": df.use_AB.mean(),
                         "max_boot_draws": int(df.n_boot_draws.max()),
                         "mae_AB": df.mae_AB.mean(), "mae_Aonly": df.mae_Aonly.mean(),
                         "mae_rule": df.mae_rule.mean(), "mae_ideal": df.ideal.mean(),
                         "rule_ideal_rate": df.rule_is_ideal.mean()})

    # The rule accepts calibration pools only. Verify deterministic replay of
    # every returned value under a fixed bootstrap seed.
    _, _, pools0 = next(iter_settings(set(args.datasets)))
    calibration0 = {k: pools0[k] for k in list(pools0)[:20]}
    d1 = coeffs_and_decision(calibration0, np.random.default_rng(5))
    d2 = coeffs_and_decision(calibration0, np.random.default_rng(5))
    assert np.allclose(d1[:3], d2[:3]) and d1[3:] == d2[3:], (
        "calibration-only decision not reproducible under fixed seed"
    )
    print("[self-check] calibration-only rule replays exactly — PASS\n")

    res = pd.DataFrame(rows)
    res.to_csv(out / "fallback_per_path.csv", index=False)

    print("=== per-dataset: fallback rule vs always-A+B vs always-A-only (mean path MAE) ===")
    agg = res.groupby("dataset").agg(
        n_paths=("path", "count"), use_AB_rate=("use_AB_rate", "mean"),
        mae_AB=("mae_AB", "mean"), mae_Aonly=("mae_Aonly", "mean"),
        mae_rule=("mae_rule", "mean"), mae_ideal=("mae_ideal", "mean"),
        rule_ideal_rate=("rule_ideal_rate", "mean")).round(4)
    print(agg.to_string())
    overall = res[["mae_AB", "mae_Aonly", "mae_rule", "mae_ideal"]].mean()
    print(f"\nPOOLED mean path MAE: A+B {overall.mae_AB:.4f} | A-only {overall.mae_Aonly:.4f} | "
          f"rule {overall.mae_rule:.4f} | ideal {overall.mae_ideal:.4f}")
    print(f"rule picks ex-post-better predictor {res.rule_ideal_rate.mean()*100:.0f}% of the time")
    loses = res[(res.mae_rule > res.mae_AB + 1e-9) & (res.mae_rule > res.mae_Aonly + 1e-9)]
    print(f"settings where rule is WORSE than BOTH fixed choices: {len(loses)}/{len(res)} "
          f"(reported, not hidden)")
    if len(loses):
        print(loses[["dataset", "path", "use_AB_rate", "mae_rule", "mae_AB", "mae_Aonly"]]
              .round(4).head(10).to_string(index=False))

    source_files = {
        "build_fallback.py": HERE / "build_fallback.py",
        "build_conditional.py": (
            HERE.parent / "exp14_conditional_tilt" / "build_conditional.py"
        ),
    }
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_files.items()
    }
    input_files = {
        "summeval_long.csv": ROOT / "data" / "summeval_long.csv",
        "wmt_mqm_long.csv": bc.WMT_MQM_LONG_CSV,
        "openmeva_long.csv": ROOT / "data" / "openmeva_long.csv",
    }
    input_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in input_files.items()
    }
    payload = {
        "B_BOOT": B_BOOT,
        "CI": CI,
        "d_eval": D_EVAL,
        "rule": "90% bootstrap CI of B̂ contains 0 -> A-only, else A+B",
        "coefficient_geometry": "prompt_conditional_common_temperature",
        "datasets": args.datasets,
        "pooled": overall.to_dict(),
        "rule_ideal_rate": float(res.rule_ideal_rate.mean()),
        "random_seed": "crc32(dataset|path)*100+repeat",
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with (out / "fallback_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}/fallback_per_path.csv, fallback_summary.json")


if __name__ == "__main__":
    main()
