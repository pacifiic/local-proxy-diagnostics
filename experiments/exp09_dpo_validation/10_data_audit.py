#!/usr/bin/env python3
"""[exp09-10] Debugging loop 2: DATA-layer audit of real run artifacts.

The hardest bug class lives in the data files between codes that all "run fine".
Every check below is an independent re-derivation on the actual artifacts, with
positive controls. Hard-fails on contract violations; prints WARN for soft ones.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FAIL = []
WARN = []


def check(cond, msg, hard=True):
    if cond:
        print(f"  OK   {msg}")
    elif hard:
        FAIL.append(msg)
        print(f"  FAIL {msg}")
    else:
        WARN.append(msg)
        print(f"  WARN {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument(
        "--prep-root",
        default=None,
        help="Root containing prompts_*.jsonl, pool.jsonl, and pool_p.jsonl. "
             "Defaults to --run-root for self-contained runs.",
    )
    args = ap.parse_args()
    root = Path(args.run_root)
    prep = Path(args.prep_root) if args.prep_root else root

    def artifact(name: str, *, prep_only: bool = False) -> Path:
        candidates = [prep / name] if prep_only else [root / name, prep / name]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Required artifact {name!r} not found under "
            f"run_root={root} or prep_root={prep}"
        )

    spec = json.load(open(artifact("prompts_run_spec.json")))
    ns = json.load(open(artifact("pool.jsonl.run_spec.json")))

    print("== [1] prompt splits ==")
    splits = {}
    for name in ["train", "calibration", "eval"]:
        rows = [json.loads(l) for l in open(artifact(f"prompts_{name}.jsonl", prep_only=True))]
        splits[name] = set(r["prompt_id"] for r in rows)
        check(len(rows) == len(splits[name]), f"{name}: prompt_id unique ({len(rows)})")
    check(not (splits["train"] & splits["eval"]) and not (splits["train"] & splits["calibration"])
          and not (splits["calibration"] & splits["eval"]), "splits pairwise disjoint")

    print("== [2] pool ==")
    pool = [json.loads(l) for l in open(artifact("pool.jsonl", prep_only=True))]
    cnt = Counter(r["split"] for r in pool)
    exp = {"train": spec["n_train"] * ns["n_train"],
           "calibration": spec["n_cal"] * ns["n_cal"],
           "eval": spec["n_eval"] * ns["n_eval"]}
    for k, v in exp.items():
        check(cnt[k] == v, f"pool[{k}] rows {cnt[k]} == expected {v}")
    keys = [(r["prompt_id"], r["split"], r["sample_idx"]) for r in pool]
    check(len(keys) == len(set(keys)), "pool key (prompt_id,split,sample_idx) unique")
    nonempty = np.mean([bool(r["response"].strip()) for r in pool])
    check(nonempty > 0.95, f"non-empty responses {nonempty:.3f} > 0.95")
    check(all(r["prompt_id"] in splits[r["split"]] for r in pool),
          "every pool row's prompt_id belongs to its declared split")

    print("== [3] scores ==")
    pool_by_key = {
        (r["prompt_id"], r["split"], r["sample_idx"]): r
        for r in pool
    }
    proxy_rows = list(map(json.loads, open(artifact("pool_p.jsonl", prep_only=True))))
    pp = {(r["prompt_id"], r["split"], r["sample_idx"]): r["proxy_score"]
          for r in proxy_rows}
    check(len(pp) == len(pool), "proxy scores cover all pool rows")
    check(set(pp) == set(pool_by_key), "proxy-score keys exactly equal pool keys")
    check(all(
        r["prompt"] == pool_by_key[(r["prompt_id"], r["split"], r["sample_idx"])]["prompt"]
        and r["response"] == pool_by_key[(r["prompt_id"], r["split"], r["sample_idx"])]["response"]
        for r in proxy_rows
    ), "proxy scoring preserves prompt/response text exactly")
    sc = [json.loads(l) for l in open(artifact("pool_scored.jsonl"))]
    check(len(sc) == exp["calibration"] + exp["eval"], "scored file covers cal+eval exactly")
    scored_keys = {(r["prompt_id"], r["split"], r["sample_idx"]) for r in sc}
    expected_scored_keys = {
        key for key in pool_by_key if key[1] in {"calibration", "eval"}
    }
    check(len(scored_keys) == len(sc), "scored candidate keys are unique")
    check(scored_keys == expected_scored_keys, "scored keys exactly equal calibration+eval pool keys")
    check(all(
        r["prompt"] == pool_by_key[(r["prompt_id"], r["split"], r["sample_idx"])]["prompt"]
        and r["response"] == pool_by_key[(r["prompt_id"], r["split"], r["sample_idx"])]["response"]
        for r in sc
    ), "gold/proxy scoring preserves calibration/eval text exactly")
    pv = np.array([r["proxy_score"] for r in sc])
    gv = np.array([r["gold_score"] for r in sc])
    check(np.isfinite(pv).all() and np.isfinite(gv).all(), "no NaN/inf in proxy/gold")
    corr = float(np.corrcoef(pv, gv)[0, 1])
    check(corr > 0.05, f"positive control: proxy-gold corr {corr:.3f} > 0.05")
    check(all(abs(pp[(r['prompt_id'], r['split'], r['sample_idx'])] - r["proxy_score"]) < 1e-9
              for r in sc), "proxy score identical across pool_p and pool_scored (join integrity)")

    print("== [4] preference pairs (re-derived) ==")
    by_tp = defaultdict(list)
    for r in map(json.loads, open(artifact("pool_p.jsonl", prep_only=True))):
        if r["split"] == "train":
            by_tp[r["prompt_id"]].append(r)
    pairs = [json.loads(l) for l in open(artifact("pref_pairs.jsonl"))]
    check(all(p["prompt_id"] in splits["train"] for p in pairs), "pairs use train prompts only")
    ok_dir = ok_ident = 0
    for p in pairs:
        rows = by_tp[p["prompt_id"]]
        max_score = max(r["proxy_score"] for r in rows)
        min_score = min(r["proxy_score"] for r in rows)
        best_responses = {
            r["response"] for r in rows
            if np.isclose(r["proxy_score"], max_score, rtol=0.0, atol=1e-12)
        }
        worst_responses = {
            r["response"] for r in rows
            if np.isclose(r["proxy_score"], min_score, rtol=0.0, atol=1e-12)
        }
        if max_score > min_score:
            ok_dir += 1
        if p["chosen"] in best_responses and p["rejected"] in worst_responses:
            ok_ident += 1
    check(ok_dir == len(pairs), f"chosen>rejected proxy direction {ok_dir}/{len(pairs)}")
    check(ok_ident == len(pairs), f"pair text == argmax/argmin re-derivation {ok_ident}/{len(pairs)}")

    print("== [5] coefficients ==")
    coef = json.load(open(artifact("coefficients.json")))
    check(np.isfinite(coef["A"]) and np.isfinite(coef["B"]), "A,B finite")
    check(coef["A"] > 0, f"positive control: A={coef['A']:.4f} > 0 (aligned RMs)", hard=False)
    check(coef["n_prompts_used"] >= 0.8 * spec["n_cal"],
          f"calibration prompts used {coef['n_prompts_used']} >= 80%")

    print("== [6] ref logp ==")
    lp = [json.loads(l) for l in open(artifact("pool_eval_reflogp.jsonl"))]
    check(len(lp) == exp["eval"], f"ref logp rows {len(lp)} == eval pool {exp['eval']}")
    lp_keys = {(r["prompt_id"], r["split"], r["sample_idx"]) for r in lp}
    expected_eval_keys = {key for key in pool_by_key if key[1] == "eval"}
    check(len(lp_keys) == len(lp), "ref-logp candidate keys are unique")
    check(lp_keys == expected_eval_keys, "ref-logp keys exactly equal eval pool keys")
    check(all(
        r["prompt"] == pool_by_key[(r["prompt_id"], r["split"], r["sample_idx"])]["prompt"]
        and r["response"] == pool_by_key[(r["prompt_id"], r["split"], r["sample_idx"])]["response"]
        for r in lp
    ), "ref-logp computation preserves eval text exactly")
    check(all(r["logp_ref"] < 0 and r["logp_ref_ntok"] > 0 for r in lp),
          "logp_ref < 0 and ntok > 0 for all rows")

    print("== [7] per-beta results ==")
    m = ns["n_eval"]
    logm = math.log(m)
    prev_d = None
    result_dirs = sorted(
        path.parent for path in root.glob("*/per_prompt.csv")
    )
    beta_dirs = [
        path for path in result_dirs if path.name.startswith("beta_")
    ]
    if beta_dirs:
        result_dirs = sorted(
            beta_dirs, key=lambda p: -float(p.name.split("_")[1])
        )
    for sd in result_dirs:
        f = sd / "per_prompt.csv"
        if not f.exists():
            check(False, f"{sd.name}: per_prompt.csv missing")
            continue
        df = pd.read_csv(f)
        check(len(df) == spec["n_eval"], f"{sd.name}: rows {len(df)} == n_eval {spec['n_eval']}")
        check(bool((df.d_reweight.between(-1e-9, logm + 1e-6)).all()),
              f"{sd.name}: d_reweight within [0, log {m}={logm:.2f}]")
        check(bool((df.ess.between(1 - 1e-9, m + 1e-6)).all()), f"{sd.name}: ESS in (1,{m}]")
        core_numeric = [
            "d_reweight", "ess", "gain_reweight_gold",
            "gain_reweight_proxy", "ref_gold_mean", "ref_proxy_mean",
        ]
        check(
            bool(np.isfinite(df[core_numeric].to_numpy()).all()),
            f"{sd.name}: core reweight metrics all finite",
        )
        if df["gain_fresh_proxy"].isna().all():
            check(
                True,
                f"{sd.name}: fresh evaluation explicitly skipped",
            )
        else:
            fresh_numeric = [
                "fresh_gold_mean", "fresh_proxy_mean",
                "fresh_ref_gold_mean", "fresh_ref_proxy_mean",
                "gain_fresh_gold", "gain_fresh_proxy",
            ]
            check(
                bool(np.isfinite(df[fresh_numeric].to_numpy()).all()),
                f"{sd.name}: fresh metrics all finite",
            )
            fp = df.gain_fresh_proxy.mean()
            check(fp > 0, f"{sd.name}: positive control — DPO raised proxy on fresh ({fp:+.3f})", hard=False)
        if beta_dirs and prev_d is not None:
            check(df.d_reweight.median() >= prev_d - 0.05,
                  f"{sd.name}: KL roughly increases as beta decreases", hard=False)
        prev_d = df.d_reweight.median()
    final_path = root / "exp09_final_per_prompt.csv"
    if final_path.exists():
        final = pd.read_csv(final_path)
        check(not final.duplicated(["tag", "prompt_id"]).any(),
              "final table key (tag,prompt_id) unique")
        check(bool((final.groupby("tag").size() == spec["n_eval"]).all()),
              f"every final checkpoint has {spec['n_eval']} eval prompts")
        check(bool(final["d"].between(-1e-9, logm + 1e-6).all()),
              f"final d within [0, log {m}={logm:.2f}]")
        check(bool(final["ess"].between(1 - 1e-9, m + 1e-6).all()),
              f"final ESS in [1,{m}]")
        required_final = [
            "d", "ess", "pred_AB", "pred_Aonly", "pred_Anu",
            "pred_pearson", "pred_spearman", "pred_isotonic",
            "frontier", "proxy_frontier", "dpo_reweight_gold",
            "dpo_reweight_proxy",
        ]
        check(bool(np.isfinite(final[required_final].to_numpy()).all()),
              "final checkpoint core table has no NaN/inf")
    else:
        check(bool(result_dirs),
              "run contains checkpoint per_prompt.csv or exp09_final_per_prompt.csv")

    print()
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)} hard failures"); [print("  -", m) for m in FAIL]
        sys.exit(1)
    print(f"AUDIT PASSED ({len(WARN)} warnings)"); [print("  WARN:", m) for m in WARN]


if __name__ == "__main__":
    main()
