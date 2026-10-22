"""Input contracts shared by preparation, model scoring and coefficient analysis."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMPONENT = HERE.parents[1]
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
PROMPT = "Summarize the following news article.\n\n{article}"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def environment():
    out = {}
    for name in ["numpy", "pandas", "scipy", "torch", "transformers", "huggingface-hub", "tokenizers"]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False,
                       dtype={k: str for k in ["article_id", "model_id", "item_id", "dimension", "rater_type"]})


def require_columns(df, columns):
    if missing := set(columns) - set(df):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[columns].isna().any().any() or any(df[c].astype(str).str.strip().eq("").any() for c in columns):
        raise ValueError("Missing/empty required values")


def item_ids(df):
    require_columns(df, ["article_id", "model_id"])
    expected = df.article_id.astype(str) + "||" + df.model_id.astype(str)
    if "item_id" in df and not expected.equals(df.item_id.astype(str)):
        raise ValueError("item_id does not match article_id||model_id")
    return df.assign(item_id=expected)


def validate_inputs(items, human, expected_items=1600):
    items, human = item_ids(items), item_ids(human)
    require_columns(items, ["item_id", "article", "summary"])
    if items.groupby("item_id")[["article", "summary"]].nunique().gt(1).any().any():
        raise ValueError("Conflicting article/summary for the same item_id")
    if items.groupby("article_id").article.nunique().gt(1).any():
        raise ValueError("Conflicting article text for the same article_id")
    items = items.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    if len(items) != expected_items:
        raise ValueError(f"Expected {expected_items} items; got {len(items)}")
    require_columns(human, ["item_id", "dimension", "rater_type", "rater_idx", "score"])
    keys = ["item_id", "dimension", "rater_type", "rater_idx"]
    if human.duplicated(keys).any():
        raise ValueError("Duplicate human-rating key")
    human["score"] = pd.to_numeric(human.score, errors="raise")
    if not np.isfinite(human.score).all() or not human.score.between(1, 5).all():
        raise ValueError("Nonfinite/out-of-range human scores")
    pools = set(human.rater_type)
    if set(human.dimension) != set(DIMENSIONS) or pools not in ({"expert", "crowd"}, {"expert", "turker"}):
        raise ValueError("Unexpected dimensions or rater pools")
    crowd_label = "turker" if "turker" in pools else "crowd"
    for dim in DIMENSIONS:
        h = human[human.dimension == dim]
        if set(h.item_id) != set(items.item_id):
            raise ValueError(f"Item coverage mismatch in {dim}")
        counts = h.groupby(["item_id", "rater_type"]).size().unstack(fill_value=0)
        if not (counts.expert.eq(3).all() and counts[crowd_label].eq(5).all()):
            raise ValueError("Expected three expert and five crowd ratings per item/dimension")
    return items, human


def load_models(path):
    models = json.loads(Path(path).read_text())
    if len(models) != 3:
        raise ValueError("This comparison requires exactly three registered models")
    for alias, m in models.items():
        if not alias.replace("_", "").isalnum():
            raise ValueError("Unsafe model alias")
        if len(m["revision"]) != 40 or any(c not in "0123456789abcdef" for c in m["revision"]):
            raise ValueError("Pin model/tokenizer to a full immutable commit")
        if m["input_format"] not in {"chat", "pair"} or m["output_field"] not in {"logits", "score"}:
            raise ValueError("Unsupported model adapter")
    return models


def verify_aligned_source(items, human, aligned_jsonl):
    truth = {}
    with Path(aligned_jsonl).open() as f:
        for line in f:
            record = json.loads(line)
            key = (str(record["id"]), str(record["model_id"]))
            if key in truth:
                raise ValueError(f"Duplicate item in aligned source: {key}")
            truth[key] = record
    keys = set(zip(items.article_id, items.model_id))
    if keys != set(truth):
        raise ValueError("Aligned source and item table have different item coverage")
    wrong = [row.item_id for row in items.itertuples()
             if row.summary.strip() != str(truth[(row.article_id, row.model_id)]["decoded"]).strip()]
    if wrong:
        raise ValueError(f"Summary ownership mismatch for {len(wrong)} items: {wrong[:3]}")
    for row in human.itertuples():
        field = "expert_annotations" if row.rater_type == "expert" else "turker_annotations"
        index = int(row.rater_idx)
        if index != float(row.rater_idx) or index < 0:
            raise ValueError("Rater index must be a nonnegative integer")
        expected = truth[(row.article_id, row.model_id)][field][index][row.dimension]
        if float(row.score) != float(expected):
            raise ValueError(f"Human rating differs from aligned source: {row.item_id}/{row.dimension}")
    return dict(status="PASS", aligned_jsonl_sha256=sha256(aligned_jsonl),
                n_verified_summaries=len(items), n_verified_ratings=len(human),
                article_text="Aligned JSONL has no source article text; article context comes from supplied item CSV")


def encode_pair(tokenizer, config, article, summary):
    prompt = PROMPT.format(article=article)
    if config["input_format"] == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": summary}],
            tokenize=False, add_generation_prompt=False)
        # The chat template already emits special tokens, including BOS.
        return tokenizer(text, add_special_tokens=False, truncation=False)
    return tokenizer(prompt, summary, add_special_tokens=True, truncation=False)


def shared_article(article, summary, tokenizers, models):
    def lengths(text):
        return {k: len(encode_pair(tokenizers[k], v, text, summary)["input_ids"])
                for k, v in models.items()}

    def fits(sizes):
        return all(sizes[k] <= models[k]["max_length"] for k in models)

    original = lengths(article)
    if fits(original):
        return article, original
    if not fits(lengths("")):
        raise ValueError("Full summary plus instructions exceeds a context limit; never truncate it silently")
    # All models see the same character-prefix of the article, with the full summary.
    lo, hi = 0, len(article)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(lengths(article[:mid])):
            lo = mid
        else:
            hi = mid
    result = article[:lo]
    sizes = lengths(result)
    if not result.strip() or not fits(sizes):
        raise ValueError("No usable common article context")
    return result, sizes


def verify_prepared(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    if manifest.get("schema_version") != 2 or "source_validation.json" not in manifest["files"]:
        raise ValueError("Prepared data require aligned-source summary/rating verification (schema 2)")
    audit = json.loads((root / "source_validation.json").read_text())
    if audit["status"] != "PASS" or audit["n_verified_summaries"] != manifest["n_items"]:
        raise ValueError("Incomplete source validation")
    return manifest


def validate_scores(scores, items):
    require_columns(scores, ["item_id", "text_sha256", "proxy_score"])
    if scores.item_id.duplicated().any() or set(scores.item_id) != set(items.item_id):
        raise ValueError("Missing/duplicate/extra score keys; refusing an inner-join comparison")
    aligned = items[["item_id", "text_sha256"]].merge(scores, on="item_id", validate="one_to_one", suffixes=("_expected", ""))
    if not aligned.text_sha256.equals(aligned.text_sha256_expected):
        raise ValueError("Score/text hash mismatch")
    aligned["proxy_score"] = pd.to_numeric(aligned.proxy_score, errors="raise")
    if not np.isfinite(aligned.proxy_score).all():
        raise ValueError("Nonfinite proxy scores")
    return aligned.set_index("item_id").loc[items.item_id, "proxy_score"].to_numpy(float)
