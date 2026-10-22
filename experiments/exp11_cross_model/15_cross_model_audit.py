#!/usr/bin/env python3
"""[exp11-15] Debugging loop 2: DATA-layer audit of the zephyr cross-model run.

Exp11-specific trap: if generation silently reused the Qwen pool (cache, wrong
model path, wrong out-jsonl), 'transfer' would look trivially perfect. So the
first checks compare the zephyr pool AGAINST the qwen pool row-by-row.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FAIL = []


def ck(cond, msg, hard=True):
    print(("  OK   " if cond else ("  FAIL " if hard else "  WARN ")) + msg)
    if not cond and hard:
        FAIL.append(msg)


def load(p):
    return [json.loads(l) for l in open(p)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zephyr-run", required=True)
    ap.add_argument("--qwen-run", required=True, help="exp09 V4 run dir (reference pool)")
    args = ap.parse_args()
    zr, qr = Path(args.zephyr_run), Path(args.qwen_run)

    print("== [1] zephyr pool is genuinely a different policy's data ==")
    zp = load(zr / "pool.jsonl")
    qp = [r for r in map(json.loads, open(qr / "pool_scored.jsonl"))]
    zkey = {(r["prompt_id"], r["split"], r["sample_idx"]): r["response"] for r in zp}
    qkey = {(r["prompt_id"], r["split"], r["sample_idx"]): r["response"] for r in qp}
    shared = set(zkey) & set(qkey)
    ck(len(shared) > 5000, f"keyspace overlap {len(shared)} rows (same prompts/splits by design)")
    same = sum(1 for k in shared if zkey[k].strip() == qkey[k].strip())
    ck(same / max(len(shared), 1) < 0.01,
       f"identical responses across models: {same}/{len(shared)} (<1% tolerated, coincidence)")
    spec = json.load(open(zr / "pool.jsonl.run_spec.json"))
    ck("zephyr" in spec["model"].lower(), f"run_spec.model == {spec['model']}")

    print("== [2] structure ==")
    cnt = Counter(r["split"] for r in zp)
    ck(cnt.get("train", 0) == 0, f"no train rows (got {cnt.get('train', 0)})")
    ck(cnt["calibration"] == 100 * 32, f"cal rows {cnt['calibration']} == 3200")
    ck(cnt["eval"] == 200 * 32, f"eval rows {cnt['eval']} == 6400")
    keys = [(r["prompt_id"], r["split"], r["sample_idx"]) for r in zp]
    ck(len(keys) == len(set(keys)), "key (prompt_id,split,sample_idx) unique")
    nonempty = float(np.mean([bool(r["response"].strip()) for r in zp]))
    ck(nonempty > 0.95, f"non-empty responses {nonempty:.3f}")
    # chat-template artifacts leaking into text (wrong template application)
    art = sum(1 for r in zp if "<|assistant|>" in r["response"] or "<|user|>" in r["response"]
              or "<|im_start|>" in r["response"])
    ck(art / len(zp) < 0.01, f"template-artifact leakage {art}/{len(zp)} rows (<1%)")

    print("== [3] scoring joins ==")
    sc = load(zr / "pool_scored.jsonl")
    ck(len(sc) == cnt["calibration"] + cnt["eval"], f"pool_scored rows {len(sc)} == cal+eval")
    pv = np.array([r["proxy_score"] for r in sc]); gv = np.array([r["gold_score"] for r in sc])
    ck(bool(np.isfinite(pv).all() and np.isfinite(gv).all()), "no NaN/inf in scores")
    pp = {(r["prompt_id"], r["split"], r["sample_idx"]): r["proxy_score"]
          for r in map(json.loads, open(zr / "pool_p.jsonl"))}
    ck(all(abs(pp[(r["prompt_id"], r["split"], r["sample_idx"])] - r["proxy_score"]) < 1e-9
           for r in sc), "proxy join integrity (pool_p == pool_scored)")
    gg = {(r["prompt_id"], r["split"], r["sample_idx"]): r["gold_score"]
          for r in map(json.loads, open(zr / "pool_g.jsonl"))}
    ck(all(abs(gg[(r["prompt_id"], r["split"], r["sample_idx"])] - r["gold_score"]) < 1e-9
           for r in sc), "gold join integrity (pool_g == pool_scored)")
    # scored responses match generated pool (no cross-file text drift)
    ck(all(zkey[(r["prompt_id"], r["split"], r["sample_idx"])] == r["response"] for r in sc),
       "pool_scored responses byte-match pool.jsonl")

    print("== [4] positive controls ==")
    corr = float(np.corrcoef(pv, gv)[0, 1])
    ck(corr > 0.05, f"proxy-gold corr {corr:.3f} > 0.05")
    # zephyr scores are NOT identical to qwen scores on the same key (different text)
    qs = {(r["prompt_id"], r["split"], r["sample_idx"]): r["proxy_score"] for r in qp}
    both = [k for k in qs if k in pp]
    same_score = sum(1 for k in both if abs(qs[k] - pp[k]) < 1e-9)
    ck(same_score / max(len(both), 1) < 0.01,
       f"identical proxy scores across models {same_score}/{len(both)} (<1%)")
    coef = json.load(open(zr / "coefficients.json"))
    ck(bool(np.isfinite(coef["A"]) and np.isfinite(coef["B"])), "zephyr A,B finite")
    ck(coef["n_prompts_used"] >= 80, f"cal prompts used {coef['n_prompts_used']} >= 80")

    print()
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)}"); [print("  -", m) for m in FAIL]; sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
