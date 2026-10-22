#!/usr/bin/env python3
"""Strictly merge independently scored candidate pools and audit every join."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


KEY = ("prompt_id", "split", "sample_idx")
IDENTITY = ("prompt_id", "split", "sample_idx", "prompt", "response")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, score_key: str | None = None) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    keys = [tuple(row[name] for name in KEY) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate candidate key")
    for row in rows:
        missing = set(IDENTITY) - set(row)
        if missing:
            raise ValueError(f"{path}: missing identity fields {sorted(missing)}")
        if not str(row["prompt"]).strip() or not str(row["response"]).strip():
            raise ValueError(f"{path}: empty prompt or response")
        if score_key is not None:
            if score_key not in row or not math.isfinite(float(row[score_key])):
                raise ValueError(f"{path}: missing or non-finite {score_key}")
    return rows, {tuple(row[name] for name in KEY): row for row in rows}


def require_same_identity(base: dict, scored: dict, label: str) -> None:
    if set(base) != set(scored):
        missing = sorted(set(base) - set(scored))[:3]
        extra = sorted(set(scored) - set(base))[:3]
        raise ValueError(
            f"{label}: key-set mismatch; missing={missing}, extra={extra}"
        )
    for key in sorted(base):
        left = tuple(base[key][name] for name in IDENTITY)
        right = tuple(scored[key][name] for name in IDENTITY)
        if left != right:
            raise ValueError(f"{label}: identity drift at key {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.out, args.audit_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    pool_rows, pool = load(args.pool)
    _, proxy = load(args.proxy, "proxy_score")
    _, gold = load(args.gold, "gold_score")
    require_same_identity(pool, proxy, "proxy join")
    require_same_identity(pool, gold, "gold join")

    split_counts: dict[str, int] = {}
    prompt_candidates: dict[tuple[str, str], int] = {}
    with args.out.open("w", encoding="utf-8") as handle:
        for row in pool_rows:
            key = tuple(row[name] for name in KEY)
            merged = dict(row)
            merged["proxy_score"] = float(proxy[key]["proxy_score"])
            merged["gold_score"] = float(gold[key]["gold_score"])
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            split = str(row["split"])
            split_counts[split] = split_counts.get(split, 0) + 1
            prompt_key = (split, str(row["prompt_id"]))
            prompt_candidates[prompt_key] = prompt_candidates.get(prompt_key, 0) + 1

    if split_counts != {"calibration": 3200, "eval": 6400}:
        raise ValueError(f"unexpected split counts: {split_counts}")
    if set(prompt_candidates.values()) != {32}:
        raise ValueError("every prompt must have exactly 32 candidates")
    prompt_counts = {
        split: len({prompt for row_split, prompt in prompt_candidates if row_split == split})
        for split in split_counts
    }
    if prompt_counts != {"calibration": 100, "eval": 200}:
        raise ValueError(f"unexpected prompt counts: {prompt_counts}")

    report = {
        "status": "PASS",
        "join_key": list(KEY),
        "identity_fields_checked": list(IDENTITY),
        "n_rows": len(pool_rows),
        "split_rows": split_counts,
        "split_prompts": prompt_counts,
        "candidates_per_prompt": 32,
        "inputs": {
            "pool": {"path": str(args.pool), "sha256": sha256_file(args.pool)},
            "proxy": {"path": str(args.proxy), "sha256": sha256_file(args.proxy)},
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
        },
        "output": {"path": str(args.out), "sha256": sha256_file(args.out)},
    }
    args.audit_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
