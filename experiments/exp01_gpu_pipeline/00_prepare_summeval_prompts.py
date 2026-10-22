"""Prepare one-prompt-per-line SummEval summarization prompts for GPU external-proxy runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--max-article-chars", type=int, default=12000)
    args = ap.parse_args()

    items_csv = Path(args.items_csv)
    df = pd.read_csv(items_csv)
    missing = [c for c in ["article_id", "article"] if c not in df.columns]
    if missing:
        raise ValueError(f"{items_csv} missing required columns: {missing}")

    articles = df[["article_id", "article"]].drop_duplicates("article_id").sort_values("article_id")
    if args.limit_prompts is not None:
        articles = articles.head(args.limit_prompts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompts = []
    for _, row in articles.iterrows():
        article = " ".join(str(row.article).split())[: args.max_article_chars]
        prompts.append(
            "Summarize the following news article in a concise, factual paragraph.\n\n"
            f"Article:\n{article}"
        )
    out.write_text("\n".join(" ".join(p.split()) for p in prompts) + "\n", encoding="utf-8")

    spec = {
        **vars(args),
        "items_csv_sha256": sha256_file(items_csv),
        "n_prompts": len(prompts),
        "article_ids": articles["article_id"].tolist(),
    }
    out.with_suffix(".run_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(prompts)} prompts to {out}")


if __name__ == "__main__":
    main()
