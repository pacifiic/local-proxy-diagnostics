#!/usr/bin/env python3
"""Score generated candidates with a Hugging Face sequence-classification model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_hf_token(token_file: str) -> bool:
    if not token_file:
        return False
    token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Empty Hugging Face token file: {token_file}")
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return True


def format_pair(prompt: str, response: str, mode: str, tokenizer: object | None = None) -> str:
    if mode == "plain":
        return f"Prompt:\n{prompt}\n\nResponse:\n{response}"
    if mode == "instruction":
        return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    if mode == "chat":
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("--input-format chat requires a tokenizer with apply_chat_template.")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    raise ValueError(f"Unknown input format: {mode}")


def score_from_logits(logits: np.ndarray, mode: str) -> np.ndarray:
    if logits.ndim == 1:
        return logits.astype(float)
    if mode == "scalar":
        return logits[:, 0].astype(float)
    if mode == "last":
        return logits[:, -1].astype(float)
    if mode == "logit_diff":
        if logits.shape[1] < 2:
            raise ValueError("logit_diff requires at least two logits.")
        return (logits[:, -1] - logits[:, 0]).astype(float)
    if mode == "auto":
        if logits.shape[1] == 1:
            return logits[:, 0].astype(float)
        return (logits[:, -1] - logits[:, 0]).astype(float)
    raise ValueError(f"Unknown --logit-mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score LLM candidates with a sequence-classification reward model.")
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--score-col", required=True)
    ap.add_argument("--input-format", choices=["plain", "instruction", "chat", "pair"], default="plain")
    ap.add_argument("--logit-mode", choices=["auto", "scalar", "last", "logit_diff"], default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--hf-token-file", default="", help="Optional path to a Hugging Face token file. The token is not written to outputs.")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-spec-json", default="")
    args = ap.parse_args()

    token_used = load_hf_token(args.hf_token_file)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.max_length < 1:
        raise ValueError("--max-length must be positive.")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Scoring requires torch and transformers. Install them in the GPU environment."
        ) from exc

    df = pd.read_csv(args.candidates_csv)
    required = {"prompt_id", "candidate_id", "prompt", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidates CSV missing columns: {sorted(missing)}")

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(args.device)

    if args.input_format == "pair":
        # canonical two-segment encoding for encoder RMs (e.g. OA deberta):
        # tokenizer(question, answer) -> [CLS] q [SEP] a [SEP]
        texts = [(str(row.prompt), str(row.response)) for row in df.itertuples(index=False)]
    else:
        texts = [
            format_pair(str(row.prompt), str(row.response), args.input_format, tokenizer)
            for row in df.itertuples(index=False)
        ]
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            if args.input_format == "pair":
                tok = tokenizer(
                    [q for q, _ in batch],
                    [a for _, a in batch],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
            else:
                tok = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    # chat mode: apply_chat_template already emitted special tokens;
                    # avoid the double-BOS footgun for decoder RMs.
                    add_special_tokens=(args.input_format != "chat"),
                    return_tensors="pt",
                )
            tok = {k: v.to(args.device) for k, v in tok.items()}
            logits = model(**tok).logits.detach().float().cpu().numpy()
            scores.extend(score_from_logits(logits, args.logit_mode).tolist())

    out = df.copy()
    out[args.score_col] = np.asarray(scores, dtype=float)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    spec = {
        "script": "experiments/exp05_real_llm_candidate_pool/code/03_score_sequence_classifier.py",
        "candidates_csv": args.candidates_csv,
        "model": args.model,
        "score_col": args.score_col,
        "input_format": args.input_format,
        "logit_mode": args.logit_mode,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "device": args.device,
        "trust_remote_code": args.trust_remote_code,
        "hf_token_file_used": token_used,
        "n_rows": int(len(out)),
    }
    spec_path = Path(args.out_spec_json) if args.out_spec_json else out_path.with_suffix(f".{args.score_col}.run_spec.json")
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote scores to {out_path} column {args.score_col}")


if __name__ == "__main__":
    main()
