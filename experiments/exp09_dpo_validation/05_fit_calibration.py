#!/usr/bin/env python3
"""[exp09-05] Estimate (A, B) from the CALIBRATION pool only (prospective lock).

Prompt-conditional moments: theory_coefficients(J=gold, r=proxy) per calibration
prompt, averaged uniformly over prompts (same convention as the conditional-tilt
design and exp05). Eval labels are never touched here.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common_experiment_utils import theory_coefficients  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-jsonl", required=True)
    ap.add_argument("--proxy-key", default="proxy_score")
    ap.add_argument("--gold-key", default="gold_score")
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    by_prompt = defaultdict(list)
    for l in open(args.scored_jsonl):
        r = json.loads(l)
        if r["split"] == "calibration":
            by_prompt[r["prompt_id"]].append(r)

    rows, skipped = [], 0
    for pid, rs in by_prompt.items():
        r_arr = np.array([x[args.proxy_key] for x in rs], float)
        j_arr = np.array([x[args.gold_key] for x in rs], float)
        if len(rs) < args.min_samples or np.var(r_arr) == 0 or np.var(j_arr) == 0:
            skipped += 1
            continue
        c = theory_coefficients(j_arr, r_arr)
        rows.append({"prompt_id": pid, "n": len(rs),
                     "a": float(c.a_exact), "b": float(c.b_theory), "a_nu": float(c.a_nu)})

    if not rows:
        raise SystemExit("no usable calibration prompts (all skipped) — refusing to emit NaN coefficients")
    A = float(np.mean([r["a"] for r in rows]))
    B = float(np.mean([r["b"] for r in rows]))
    A_nu = float(np.mean([r["a_nu"] for r in rows]))
    out = {
        "A": A, "B": B, "A_nu": A_nu,
        "n_prompts_used": len(rows), "n_prompts_skipped": skipped,
        "proxy_key": args.proxy_key, "gold_key": args.gold_key,
        "per_prompt": rows,
    }
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"A={A:.5f} B={B:.5f} A_nu={A_nu:.5f} (prompts used {len(rows)}, skipped {skipped})")


if __name__ == "__main__":
    main()
