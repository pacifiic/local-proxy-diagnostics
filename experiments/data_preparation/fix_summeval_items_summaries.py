#!/usr/bin/env python3
"""Repair the summary column of summeval_items.csv.

Bug (found 2026-07-06): summaries in summeval_items.csv were assigned to the
wrong model_id within each article via a fixed permutation (e.g. the text shown
for M0 actually belongs to M11) — a sorted-order join bug in the original
generation code. Human scores (summeval_long.csv) and the reference/article
columns are correct; only the summary text column is permuted. Every consumer
that scored `summary` (exp01 RM proxy, exp01 open judge, ROUGE proxy CSVs)
therefore scored the wrong text for each item_id.

This script rewrites the summary column from the ground-truth
model_annotations.aligned.jsonl (`decoded` field) keyed by (id, model_id),
verifies 100% ownership, and refuses to write otherwise.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--aligned-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--backup-csv", default="", help="Optional path to save the buggy original.")
    args = ap.parse_args()

    items = pd.read_csv(args.items_csv)
    truth = {}
    with open(args.aligned_jsonl) as f:
        for line in f:
            d = json.loads(line)
            truth[(d["id"], d["model_id"])] = (d.get("decoded") or "").strip()

    missing = [
        (a, m)
        for a, m in items[["article_id", "model_id"]].drop_duplicates().itertuples(index=False)
        if (a, m) not in truth
    ]
    if missing:
        raise SystemExit(f"{len(missing)} (article_id, model_id) pairs missing from aligned jsonl: {missing[:5]}")

    before_correct = sum(
        str(r.summary).strip() == truth[(r.article_id, r.model_id)]
        for r in items.drop_duplicates("item_id").itertuples()
    )
    items["summary"] = [truth[(a, m)] for a, m in zip(items.article_id, items.model_id)]
    after_correct = sum(
        str(r.summary).strip() == truth[(r.article_id, r.model_id)]
        for r in items.drop_duplicates("item_id").itertuples()
    )
    n_items = items.item_id.nunique()
    print(f"summary ownership correct: before {before_correct}/{n_items} -> after {after_correct}/{n_items}")
    if after_correct != n_items:
        raise SystemExit("Repair failed verification; not writing output.")

    if args.backup_csv:
        Path(args.backup_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(args.items_csv).to_csv(args.backup_csv, index=False)
        print(f"buggy original saved to {args.backup_csv}")
    items.to_csv(args.out_csv, index=False)
    print(f"fixed items written to {args.out_csv}")


if __name__ == "__main__":
    main()
