"""Freeze validated input text, human targets and common item-level split IDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import (COMPONENT, DIMENSIONS, HERE, digest, environment, load_models,
                    new_directory, read_csv, sha256, shared_article, validate_inputs,
                    verify_aligned_source, write_json)

sys.path.insert(0, str(COMPONENT / "lib"))
from common_experiment_utils import split_indices


def make_splits(ids, repeats, calibration_frac, seed):
    if len(ids) < 30 or len(set(ids)) != len(ids) or repeats < 1:
        raise ValueError("Need unique IDs, >=30 items and positive repeats")
    rng = np.random.default_rng(seed)
    rows = []
    for dimension in DIMENSIONS:
        for repeat in range(repeats):
            cal, eva = split_indices(len(ids), calibration_frac, rng)
            rows.append(dict(dimension=dimension, repeat=repeat,
                             calibration=[ids[i] for i in cal], evaluation=[ids[i] for i in eva]))
    return rows


def prepare_records(items, tokenizers, models):
    contexts = {}
    for article_id, group in items.groupby("article_id", sort=True):
        article = group.article.iloc[0]
        # Recheck all summaries after shortening: token counts need not be
        # strictly monotone under a character-prefix change (BPE boundaries).
        while True:
            previous = article
            for summary in group.summary:
                article, _ = shared_article(article, summary, tokenizers, models)
            if article == previous:
                break
        contexts[article_id] = article
    prepared = []
    for row in items.itertuples():
        article, lengths = shared_article(contexts[row.article_id], row.summary, tokenizers, models)
        if article != contexts[row.article_id]:
            raise ValueError("Common article context was not stable")
        record = dict(item_id=row.item_id, article_id=row.article_id, model_id=row.model_id,
                      article=article, summary=row.summary,
                      original_article_sha256=digest(row.article),
                      text_sha256=digest([article, row.summary]),
                      article_chars_original=len(row.article), article_chars_used=len(article))
        record.update({f"tokens_{k}": n for k, n in lengths.items()})
        prepared.append(record)
    return prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items-csv", type=Path, required=True)
    p.add_argument("--aligned-jsonl", type=Path, required=True)
    p.add_argument("--human-csv", type=Path, default=COMPONENT / "data/summeval_long.csv")
    p.add_argument("--models", type=Path, default=HERE / "models.json")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--calibration-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    items, human = validate_inputs(read_csv(args.items_csv), read_csv(args.human_csv))
    source_validation = verify_aligned_source(items, human, args.aligned_jsonl)
    models = load_models(args.models)
    splits = make_splits(items.item_id.tolist(), args.repeats, args.calibration_frac, args.seed)
    from transformers import AutoTokenizer
    tokenizers = {k: AutoTokenizer.from_pretrained(v["model_id"], revision=v["revision"],
                  cache_dir=args.cache_dir, local_files_only=args.local_files_only)
                  for k, v in models.items()}
    prepared = prepare_records(items, tokenizers, models)
    import pandas as pd
    out = new_directory(args.outdir)
    pd.DataFrame(prepared).to_csv(out / "items.csv", index=False)
    human.to_csv(out / "human.csv", index=False)
    write_json(out / "models.json", models)
    write_json(out / "source_validation.json", source_validation)
    with (out / "splits.jsonl").open("x") as f:
        for row in splits:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(out / "manifest.json", dict(
        schema_version=2, n_items=len(items), dimensions=DIMENSIONS,
        repeats=args.repeats, seed=args.seed, calibration_frac=args.calibration_frac,
        gold="Mean of three expert and five crowd scores per item/dimension",
        split_unit="item, not article (matches original external-proxy protocol)",
        context="Common article prefix across all three tokenizers and all summaries of that article; full summaries retained",
        input_hashes={"items": sha256(args.items_csv), "human": sha256(args.human_csv)},
        files={n: sha256(out / n) for n in ["items.csv", "human.csv", "models.json", "splits.jsonl", "source_validation.json"]},
        code_sha256={n: sha256(HERE / n) for n in ["prepare.py", "common.py"]}, env=environment()))
    print(out)


if __name__ == "__main__":
    main()
