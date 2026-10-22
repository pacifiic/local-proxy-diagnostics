#!/usr/bin/env python3
"""[exp09-09] Design-review addendum: exact-tilt frontier at matched d̂.

For each eval prompt and each beta, solve the exponential tilt in r over the SAME
32-sample ref pool at the SAME measured d_reweight, and record its gold gain.
Three-rung decomposition per prompt:
  pred (A√d+Bd)  vs  frontier (exact tilt @ d̂)   -> theory error
  frontier       vs  gain_reweight (DPO implied)  -> mechanism gap (Prop. 1 direction)
  gain_reweight  vs  gain_fresh                   -> off-support / W1 term (Thm. C)
Pure post-processing of saved artifacts; no GPU.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def tilt_weights(r: np.ndarray, t: float) -> np.ndarray:
    a = t * (r - r.max())
    w = np.exp(a)
    return w / w.sum()


def kl_of_tilt(r: np.ndarray, t: float) -> float:
    w = tilt_weights(r, t)
    m = len(r)
    return float(np.log(m) + np.sum(w * np.log(np.clip(w, 1e-300, None))))


def find_t(r: np.ndarray, d: float, t_cap: float = 1e4) -> float:
    if d <= 0 or np.var(r) == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while kl_of_tilt(r, hi) < d:
        lo, hi = hi, hi * 2
        if hi > t_cap:
            return t_cap
    for _ in range(80):
        mid = (lo + hi) / 2
        if kl_of_tilt(r, mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--coefficients-json", required=True)
    args = ap.parse_args()
    root = Path(args.run_root)
    coef = json.load(open(args.coefficients_json))
    A, B = coef["A"], coef["B"]

    pool = defaultdict(list)
    for l in open(root / "pool_scored.jsonl"):
        r = json.loads(l)
        if r["split"] == "eval":
            pool[r["prompt_id"]].append((r["proxy_score"], r["gold_score"]))

    all_rows = []
    result_dirs = sorted(
        path.parent for path in root.glob("*/per_prompt.csv")
    )
    for sd in result_dirs:
        pp = sd / "per_prompt.csv"
        df = pd.read_csv(pp)
        beta = json.load(open(sd / "summary.json"))["beta"]
        for row in df.itertuples():
            vals = pool.get(row.prompt_id)
            if not vals:
                continue
            r_arr = np.array([v[0] for v in vals])
            j_arr = np.array([v[1] for v in vals])
            d = max(float(row.d_reweight), 0.0)
            t = find_t(r_arr, d)
            w = tilt_weights(r_arr, t)
            frontier = float(np.sum(w * j_arr) - j_arr.mean())
            all_rows.append({
                "tag": sd.name, "beta": beta,
                "prompt_id": row.prompt_id, "d": d,
                "pred_AB": A * np.sqrt(d) + B * d,
                "frontier_tilt_gold": frontier,
                "gain_reweight_gold": row.gain_reweight_gold,
                "gain_fresh_gold": getattr(row, "gain_fresh_gold", np.nan),
            })
    out = pd.DataFrame(all_rows)
    if out.empty:
        raise ValueError(
            f"no direct-child per_prompt.csv artifacts found under {root}"
        )
    out.to_csv(root / "frontier_decomposition_per_prompt.csv", index=False)
    g = out.groupby(["tag", "beta"]).agg(
        mean_d=("d", "mean"),
        pred_AB=("pred_AB", "mean"),
        frontier=("frontier_tilt_gold", "mean"),
        dpo_reweight=("gain_reweight_gold", "mean"),
        fresh=("gain_fresh_gold", "mean"),
    ).reset_index().sort_values("mean_d")
    g["theory_err(pred-frontier)"] = g.pred_AB - g.frontier
    g["mechanism_gap(frontier-dpo)"] = g.frontier - g.dpo_reweight
    g["offsupport(fresh-dpo)"] = g.fresh - g.dpo_reweight
    g.to_csv(root / "frontier_decomposition_summary.csv", index=False)
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
