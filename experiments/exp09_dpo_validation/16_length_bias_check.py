#!/usr/bin/env python3
"""[exp09-16] Length-bias disclosure: is the proxy->gold alignment our law captures
just a shared response-length artifact?

Regression reward models are known to carry length bias, so we MEASURE it rather
than deny it. Reports corr(response length, gold) and corr(length, proxy) on the
scored pool. The defensive point is empirical: if proxy and gold have OPPOSITE-
signed length correlations, the alignment (A) is not driven by a shared length
confound. Pure post-processing of pool_scored.jsonl.
"""
import argparse
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-jsonl", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'file/split':40s} {'n':>6s} {'r(len,GOLD)':>12s} {'ρ(len,GOLD)':>12s} {'r(len,PROXY)':>13s}")
    for path in args.scored_jsonl:
        for split in ("calibration", "eval"):
            W, G, P = [], [], []
            for l in open(path):
                r = json.loads(l)
                if r.get("split") != split:
                    continue
                W.append(len(r["response"].split()))
                G.append(r["gold_score"]); P.append(r["proxy_score"])
            if len(W) < 3:
                continue
            W, G, P = np.array(W), np.array(G), np.array(P)
            rg = pearsonr(W, G).statistic
            sg = spearmanr(W, G).statistic
            rp = pearsonr(W, P).statistic
            tag = f"{path.split('/')[-2]}/{split}"
            print(f"{tag:40s} {len(W):6d} {rg:+12.3f} {sg:+12.3f} {rp:+13.3f}")
            if rg * rp < 0:
                print(f"    -> opposite-signed length bias: gold {rg:+.2f} vs proxy {rp:+.2f} "
                      f"=> alignment is NOT a shared length artifact")


if __name__ == "__main__":
    main()
