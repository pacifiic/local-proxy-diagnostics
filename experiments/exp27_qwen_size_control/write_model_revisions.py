#!/usr/bin/env python3
"""Resolve locally cached generator and scorer revisions for one run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def revision(model_id: str) -> str:
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub",
        )
    )
    repo = cache_root / f"models--{model_id.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(f"missing local main ref for {model_id}: {ref}")
    value = ref.read_text(encoding="utf-8").strip()
    if len(value) != 40:
        raise ValueError(f"unexpected revision for {model_id}: {value}")
    if not (repo / "snapshots" / value).is_dir():
        raise FileNotFoundError(f"missing local snapshot for {model_id}@{value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    generator = json.loads(
        (args.run / "pool.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    proxy = json.loads(
        (args.run / "pool_proxy.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    gold = json.loads(
        (args.run / "pool_gold.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    output = {
        "generator": {
            "model": generator["model"],
            "revision": generator.get("model_revision")
            or revision(generator["model"]),
        },
        "proxy_scorer": {
            "model": proxy["model"],
            "revision": proxy.get("revision") or revision(proxy["model"]),
        },
        "gold_scorer": {
            "model": gold["model"],
            "revision": gold.get("revision") or revision(gold["model"]),
        },
    }
    path = args.run / "model_revisions.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
