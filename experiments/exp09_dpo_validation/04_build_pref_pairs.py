#!/usr/bin/env python3
"""[exp09-04] Build DPO preference pairs from proxy scores on TRAIN prompts only.

chosen = argmax proxy_score, rejected = argmin proxy_score per prompt.
This IS the definition of optimizing the proxy — direction check is enforced.
"""
import argparse
import json
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-jsonl", required=True)
    ap.add_argument("--score-key", default="proxy_score")
    ap.add_argument("--min-gap", type=float, default=0.0)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    by_prompt = defaultdict(list)
    for l in open(args.scored_jsonl):
        r = json.loads(l)
        if r["split"] == "train":
            by_prompt[r["prompt_id"]].append(r)

    n_written = n_skipped = 0
    with open(args.out_jsonl, "w") as f:
        for pid, rows in by_prompt.items():
            rows.sort(key=lambda r: r[args.score_key])
            lo, hi = rows[0], rows[-1]
            gap = hi[args.score_key] - lo[args.score_key]
            if gap <= args.min_gap or not hi["response"] or not lo["response"]:
                n_skipped += 1
                continue
            assert hi[args.score_key] >= lo[args.score_key]  # chosen must be higher-proxy
            f.write(json.dumps({
                "prompt_id": pid,
                "prompt": hi["prompt"],
                "chosen": hi["response"],
                "rejected": lo["response"],
                "proxy_gap": gap,
            }, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"pairs: {n_written} written, {n_skipped} skipped (zero-gap/empty)")
    json.dump(vars(args) | {"n_pairs": n_written, "n_skipped": n_skipped},
              open(args.out_jsonl + ".run_spec.json", "w"), indent=2)


if __name__ == "__main__":
    main()
