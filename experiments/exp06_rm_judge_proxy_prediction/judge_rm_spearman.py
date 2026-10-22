#!/usr/bin/env python3
"""Judge-RM alignment for the fresh-BoN pool (response-bank checklist item 4).

Reports Spearman/Pearson between the RM proxy score and the judge gold score,
both pooled and within-prompt (mean/median across prompts), on the exp01
fresh_bon sample pool.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-csv", required=True)
    ap.add_argument("--gold-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    proxy = pd.read_csv(args.proxy_csv)
    gold = pd.read_csv(args.gold_csv)
    keys = ["prompt_idx", "sample_idx"]
    for name, frame, score in (
        ("proxy", proxy, "proxy_score"),
        ("gold", gold, "gold_score"),
    ):
        missing = set(keys + [score]).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            examples = frame.loc[frame.duplicated(keys, keep=False), keys].head()
            raise ValueError(f"{name} CSV has duplicate candidate keys:\n{examples}")
        if not np.isfinite(frame[score].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} CSV has non-finite {score} values")

    joined = proxy.merge(gold, on=keys, how="outer", validate="one_to_one", indicator=True)
    unmatched = joined.loc[joined["_merge"] != "both", keys + ["_merge"]]
    if not unmatched.empty:
        raise ValueError(
            "proxy/gold candidate key sets differ; refusing a lossy inner join:\n"
            + unmatched.head(10).to_string(index=False)
        )
    df = joined.drop(columns="_merge")
    if len(df) != len(proxy) or len(df) != len(gold):
        raise AssertionError("one-to-one merge cardinality changed unexpectedly")

    pooled_s = spearmanr(df.proxy_score, df.gold_score)
    pooled_p = pearsonr(df.proxy_score, df.gold_score)

    within = []
    for pid, g in df.groupby("prompt_idx"):
        if g.proxy_score.nunique() > 1 and g.gold_score.nunique() > 1:
            within.append(spearmanr(g.proxy_score, g.gold_score).statistic)
    within = np.array(within, dtype=float)

    out = pd.DataFrame(
        [
            {"metric": "pooled_spearman", "value": pooled_s.statistic, "n": len(df)},
            {"metric": "pooled_spearman_p", "value": pooled_s.pvalue, "n": len(df)},
            {"metric": "pooled_pearson", "value": pooled_p.statistic, "n": len(df)},
            {"metric": "within_prompt_spearman_mean", "value": within.mean(), "n": len(within)},
            {"metric": "within_prompt_spearman_median", "value": np.median(within), "n": len(within)},
            {"metric": "within_prompt_spearman_frac_positive", "value": float((within > 0).mean()), "n": len(within)},
        ]
    )
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
