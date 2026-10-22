#!/usr/bin/env python3
"""Generate ROUGE-1/2/L proxy CSVs (dimension,item_id,proxy_score) from summeval_items.csv.

Replicates the original ROUGE proxy computation (rouge_score, use_stemmer=True,
F-measure vs the reference summary; identical value replicated across dimensions),
verified to match the stored CSVs when run on the same summary text.
"""
import argparse

import pandas as pd
from rouge_score import rouge_scorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--out-prefix", required=True, help="e.g. data/proxy_ -> data/proxy_rouge1.csv ...")
    args = ap.parse_args()

    items = pd.read_csv(args.items_csv)
    uniq = items.drop_duplicates("item_id")
    dims = sorted(items.dimension.unique())
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {m: {} for m in ["rouge1", "rouge2", "rougeL"]}
    for r in uniq.itertuples():
        res = sc.score(str(r.reference), str(r.summary))
        for m in scores:
            scores[m][r.item_id] = res[m].fmeasure

    for m in scores:
        rows = [
            {"dimension": d, "item_id": iid, "proxy_score": v}
            for d in dims
            for iid, v in scores[m].items()
        ]
        out = f"{args.out_prefix}{m}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
