"""Score one registered reward model. No training or human-label access."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HERE, encode_pair, environment, load_models, new_directory,
                    read_csv, sha256, validate_scores, verify_prepared, write_json)


def scalar_score(output, field):
    value = getattr(output, field).detach().float().cpu().numpy().reshape(-1)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError("Expected one finite scalar reward; do not flatten multi-objective rewards")
    return float(value[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--alias", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-remote-code", action="store_true")
    args = p.parse_args()
    manifest = verify_prepared(args.prepared)
    models = load_models(args.prepared / "models.json")
    if args.alias not in models:
        raise ValueError("Unknown model alias")
    cfg = models[args.alias]
    if cfg["trust_remote_code"] and not args.allow_remote_code:
        raise ValueError("ArmoRM requires explicit --allow-remote-code for its pinned custom module")
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    if args.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Explicitly set CUDA_VISIBLE_DEVICES to the GPU allocated to this run")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    # The pinned ArmoRM module imports a docstring constant removed in recent
    # Transformers. Restoring only this documentation symbol leaves inference unchanged.
    if args.alias == "armorm":
        from transformers.models.llama import modeling_llama
        if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
            modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
    kwargs = dict(revision=cfg["revision"], cache_dir=args.cache_dir,
                  local_files_only=args.local_files_only, trust_remote_code=cfg["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], **kwargs, torch_dtype=getattr(torch, cfg["dtype"]),
        device_map={"": args.device}, attn_implementation="eager")
    model.eval()
    items = read_csv(args.prepared / "items.csv")
    out = new_directory(args.outdir)
    write_json(out / "run_spec.json", dict(
        alias=args.alias, model=cfg, prepared_manifest_sha256=sha256(args.prepared / "manifest.json"),
        files=manifest["files"], batch_size=1, seed=0, device=args.device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), attention="eager",
        env=environment(), code_sha256={n: sha256(HERE / n) for n in ["score.py", "common.py"]}))
    rows = []
    # Batch size one avoids model-specific padding/pooling differences.
    # Flush an append-only journal so partial work survives an interrupted run.
    with (out / "scores.partial.jsonl").open("x") as journal, torch.inference_mode():
        for row in items.itertuples():
            enc = encode_pair(tokenizer, cfg, row.article, row.summary)
            n_tokens = len(enc["input_ids"])
            if n_tokens > cfg["max_length"] or n_tokens != getattr(row, f"tokens_{args.alias}"):
                raise ValueError(f"Tokenization drift at {row.item_id}")
            tensors = {k: torch.tensor([v], device=args.device) for k, v in enc.items()}
            value = scalar_score(model(**tensors), cfg["output_field"])
            record = dict(item_id=row.item_id, text_sha256=row.text_sha256,
                          proxy_score=value, n_tokens=n_tokens)
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            rows.append(record)
            if len(rows) % 100 == 0:
                print(f"{args.alias}: {len(rows)}/{len(items)}", flush=True)
    scores = pd.DataFrame(rows)
    validate_scores(scores, items)
    scores.to_csv(out / "scores.csv", index=False)
    write_json(out / "complete.json", dict(n_items=len(scores),
               scores_sha256=sha256(out / "scores.csv"), spec_sha256=sha256(out / "run_spec.json")))
    print(out)


if __name__ == "__main__":
    main()
