#!/usr/bin/env python3
"""[exp09-03] Score (prompt, response) pairs with a reward model.

Two model kinds, both with the double-BOS footgun (B3) avoided:
- skywork: AutoModelForSequenceClassification; chat template rendered to TEXT then
  tokenized with add_special_tokens=False (template already emits BOS).
- armorm: RLHFlow/ArmoRM custom architecture (trust_remote_code); tokenized DIRECTLY
  via apply_chat_template(tokenize=True) per model card, score read from output.score.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import site
import sys
from pathlib import Path

import torch


KNOWN_REVISIONS = {
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": (
        "d4117fbfd81b72f41b96341238baa1e3e90a4ce1"
    ),
    "OpenAssistant/reward-model-deberta-v3-large-v2": (
        "c355404efa9ad2ad069f3a197cae0523c14244fc"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--revision",
        help="Pinned 40-character model commit.",
    )
    ap.add_argument("--kind", choices=["skywork", "armorm", "deberta"], required=True)
    ap.add_argument("--score-key", required=True, help="key added to each row, e.g. proxy_score / gold_score")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--require-user-site-disabled",
        action="store_true",
        help="Fail unless PYTHONNOUSERSITE disables user-site packages.",
    )
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()
    revision = args.revision or KNOWN_REVISIONS.get(args.model)
    if revision is not None and len(revision) != 40:
        ap.error("--revision must be a 40-character commit")
    if min(args.batch_size, args.max_length) < 1:
        ap.error("--batch-size and --max-length must be positive")
    if args.require_user_site_disabled and site.ENABLE_USER_SITE:
        ap.error(
            "--require-user-site-disabled needs PYTHONNOUSERSITE=1 "
            "set before Python starts"
        )

    input_path = Path(args.in_jsonl)
    output_path = Path(args.out_jsonl)
    spec_path = Path(f"{args.out_jsonl}.run_spec.json")
    existing = [path for path in (output_path, spec_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite scorer artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with input_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise ValueError("scorer input is empty")
    identity_fields = ("prompt_id", "split", "sample_idx")
    identities: list[tuple[object, ...]] = []
    for index, row in enumerate(rows):
        missing = [name for name in ("prompt", "response") if name not in row]
        if missing:
            raise ValueError(f"row {index}: missing fields {missing}")
        if not str(row["prompt"]).strip() or not str(row["response"]).strip():
            raise ValueError(f"row {index}: empty prompt or response")
        if args.score_key in row:
            raise ValueError(
                f"row {index}: refusing to overwrite existing {args.score_key}"
            )
        if all(name in row for name in identity_fields):
            identities.append(tuple(row[name] for name in identity_fields))
    if identities and len(identities) != len(rows):
        raise ValueError("candidate identity fields are present on only some rows")
    if identities and len(identities) != len(set(identities)):
        raise ValueError("duplicate candidate identity in scorer input")

    load_kwargs = {
        "trust_remote_code": args.kind == "armorm",
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        **load_kwargs,
    )
    model.eval()

    scores = []
    with torch.no_grad():
        if args.kind == "skywork":
            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                texts = [tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]},
                     {"role": "assistant", "content": r["response"]}], tokenize=False)
                    for r in batch]
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length, add_special_tokens=False).to(model.device)
                s = model(**enc).logits.squeeze(-1).float().cpu().tolist()
                scores.extend([s] if isinstance(s, float) else s)
        elif args.kind == "deberta":  # encoder RM: canonical two-segment pair encoding
            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                enc = tok([r["prompt"] for r in batch], [r["response"] for r in batch],
                          return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length).to(model.device)
                s = model(**enc).logits.squeeze(-1).float().cpu().tolist()
                scores.extend([s] if isinstance(s, float) else s)
        else:  # armorm: per model card, one example at a time via tokenized chat template
            for r in rows:
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]},
                     {"role": "assistant", "content": r["response"]}],
                    return_tensors="pt", tokenize=True, return_dict=True,
                    truncation=True, max_length=args.max_length).to(model.device)
                out = model(**enc)
                scores.append(float(out.score.float().item()))

    if len(scores) != len(rows):
        raise RuntimeError(
            f"scorer returned {len(scores)} scores for {len(rows)} rows"
        )
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("scorer produced a non-finite score")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"refusing to overwrite {temporary_path}")
    with temporary_path.open("w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            scored = dict(r)
            scored[args.score_key] = float(s)
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    resolved_revision = (
        revision
        or getattr(model.config, "_commit_hash", None)
        or getattr(tok, "init_kwargs", {}).get("_commit_hash")
    )
    spec = {
        "model": args.model,
        "revision": resolved_revision,
        "kind": args.kind,
        "n": len(rows),
        "score_key": args.score_key,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "torch_dtype": "bfloat16",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "accelerate": package_version("accelerate"),
        },
        "environment": {
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "user_site_enabled": bool(site.ENABLE_USER_SITE),
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"scored {len(rows)} rows [{args.score_key}] -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
