"""[gpu-01] Score SummEval summaries with open reward models -> proxy CSVs.

External-proxy experiment: adds learned-RM external proxies to the prospective
pipeline. Output format matches the repo convention:
    dimension,item_id,proxy_score
so it plugs directly into experiments/07_prospective_prediction.

Usage (from repo root, after parse_summeval + building data/summeval_items.csv):
  python gpu/01_rm_proxy_summeval.py \
    --items-csv data/summeval_items.csv \
    --model Skywork/Skywork-Reward-V2-Llama-3.1-8B --alias skywork8b \
    --outdir data
  # repeat with 1-2 more RewardBench-ranked open RMs for a multi-RM table.

Then:
  python experiments/07_prospective_prediction/01_prospective_prediction.py \
    --human-long-csv data/summeval_long.csv --item-cols article_id,model_id \
    --group-cols dimension --proxy-csv data/proxy_rm_<alias>.csv \
    --proxy-item-col item_id --proxy-score-col proxy_score \
    --n-repeats 50 --outdir <out>/rm_<alias>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from repro_utils import env_snapshot, resolve_hf_revision

PROMPT_TMPL = "Summarize the following news article.\n\n{article}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}; got {list(df.columns)}")


def render_pair(tokenizer, article: str, summary: str) -> str:
    chat = [
        {"role": "user", "content": PROMPT_TMPL.format(article=article)},
        {"role": "assistant", "content": summary},
    ]
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False)
    except Exception:
        return f"User: {chat[0]['content']}\nAssistant: {summary}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True, help="summeval_items.csv (article, summary, item_id, dimension)")
    ap.add_argument("--model", required=True, help="HF id of a sequence-classification reward model")
    ap.add_argument("--alias", required=True, help="short name used in the output filename")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit-items", type=int, default=None, help="debug limit on unique item_id count")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--model-revision", default=None, help="Optional HF revision/commit for --model")
    args = ap.parse_args()

    items_csv = Path(args.items_csv)
    df = pd.read_csv(items_csv)
    require_columns(df, ["item_id", "dimension", "article", "summary"], items_csv)
    uniq = df.drop_duplicates(subset=["item_id"])[["item_id", "article", "summary"]].reset_index(drop=True)
    if args.limit_items is not None:
        uniq = uniq.head(args.limit_items).copy()
        df = df[df["item_id"].isin(set(uniq["item_id"]))].copy()
    dims = df[["item_id", "dimension"]].drop_duplicates()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"proxy_rm_{args.alias}.csv"
    ckpt_path = outdir / f"proxy_rm_{args.alias}.partial.csv"
    spec_path = outdir / f"proxy_rm_{args.alias}.run_spec.json"
    model_sha = resolve_hf_revision(args.model, args.model_revision)
    spec = {
        **vars(args),
        "items_csv": str(items_csv),
        "items_csv_sha256": sha256_file(items_csv),
        "model_resolved_sha": model_sha,
        "env": env_snapshot(),
        "n_unique_items": int(len(uniq)),
    }
    done: dict[str, float] = {}
    if ckpt_path.exists():
        if spec_path.exists():
            old = json.loads(spec_path.read_text())
            keys = ["model", "model_revision", "alias", "max_length", "dtype", "items_csv_sha256"]
            if old.get("model_resolved_sha") is not None:
                keys.append("model_resolved_sha")
            comparable = {k: old.get(k) for k in keys}
            current = {k: spec.get(k) for k in keys}
            if comparable != current:
                raise RuntimeError(f"Refusing to resume from incompatible checkpoint: {comparable} != {current}")
        prev = pd.read_csv(ckpt_path)
        done = dict(zip(prev.item_id, prev.score))
        print(f"resume: {len(done)} items already scored")

    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False))

    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()

    todo = uniq[~uniq.item_id.isin(done)].reset_index(drop=True)
    rows = [{"item_id": k, "score": v} for k, v in done.items()]
    with torch.no_grad():
        for start in range(0, len(todo), args.batch_size):
            batch = todo.iloc[start:start + args.batch_size]
            texts = []
            for _, r in batch.iterrows():
                texts.append(render_pair(tok, str(r.article), str(r.summary)))
            # apply_chat_template already emits BOS; add_special_tokens=False avoids
            # the double-BOS input that Llama-3.1 reward models were not trained on.
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      add_special_tokens=False,
                      max_length=args.max_length).to(model.device)
            logits = model(**enc).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            for (_, r), s in zip(batch.iterrows(), logits):
                rows.append({"item_id": r.item_id, "score": float(s)})
            if (start // args.batch_size) % 20 == 0:
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)
                print(f"{len(rows)}/{len(uniq)} scored")
    pd.DataFrame(rows).to_csv(ckpt_path, index=False)

    scores = pd.DataFrame(rows)
    if scores["item_id"].duplicated().any():
        raise RuntimeError("duplicate item_id rows in score checkpoint")
    out = dims.merge(scores, on="item_id").rename(columns={"score": "proxy_score"})
    out[["dimension", "item_id", "proxy_score"]].to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(out)} rows, {scores.item_id.nunique()} items)")


if __name__ == "__main__":
    main()
