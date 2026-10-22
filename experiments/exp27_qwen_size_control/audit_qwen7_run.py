#!/usr/bin/env python3
"""Independent data-layer audit for the Qwen2.5-7B candidate-pool run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["prompt_id", "split", "sample_idx"]
EXPECTED_GENERATOR = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_PROXY = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
EXPECTED_GOLD = "OpenAssistant/reward-model-deberta-v3-large-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def identity_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[KEY + ["prompt", "response"]].sort_values(KEY).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qwen-3b", type=Path, required=True)
    parser.add_argument("--zephyr-7b", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    pool = load(args.run / "pool.jsonl")
    proxy = load(args.run / "pool_proxy.jsonl")
    gold = load(args.run / "pool_gold.jsonl")
    scored = load(args.run / "pool_scored.jsonl")
    qwen3 = load(args.qwen_3b)
    zephyr = load(args.zephyr_7b)
    for name, frame in {
        "pool": pool,
        "proxy": proxy,
        "gold": gold,
        "scored": scored,
    }.items():
        if len(frame) != 9600:
            raise ValueError(f"{name}: expected 9600 rows, found {len(frame)}")
        if frame.duplicated(KEY).any():
            raise ValueError(f"{name}: duplicate candidate key")
    reference = identity_table(pool)
    if not reference.equals(identity_table(proxy)):
        raise ValueError("proxy-scored rows drift from generated pool")
    if not reference.equals(identity_table(gold)):
        raise ValueError("gold-scored rows drift from generated pool")
    if not reference.equals(identity_table(scored)):
        raise ValueError("merged rows drift from generated pool")
    if not np.isfinite(scored[["proxy_score", "gold_score"]].to_numpy(float)).all():
        raise ValueError("non-finite merged scores")

    generator = json.loads(
        (args.run / "pool.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    proxy_spec = json.loads(
        (args.run / "pool_proxy.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    gold_spec = json.loads(
        (args.run / "pool_gold.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    if generator["model"] != EXPECTED_GENERATOR:
        raise ValueError(f"wrong generator: {generator['model']}")
    expected_sampling = {
        "n_train": 0,
        "n_cal": 32,
        "n_eval": 32,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 320,
        "seed": 0,
    }
    for key, expected in expected_sampling.items():
        if not math.isclose(float(generator[key]), float(expected)):
            raise ValueError(f"wrong generation setting {key}={generator[key]}")
    if proxy_spec["model"] != EXPECTED_PROXY or proxy_spec["score_key"] != "proxy_score":
        raise ValueError("wrong proxy scorer specification")
    if gold_spec["model"] != EXPECTED_GOLD or gold_spec["score_key"] != "gold_score":
        raise ValueError("wrong gold scorer specification")
    merge = json.loads((args.run / "merge_audit.json").read_text(encoding="utf-8"))
    if merge["status"] != "PASS":
        raise ValueError("merge audit is not PASS")
    if merge["output"]["sha256"] != sha256_file(args.run / "pool_scored.jsonl"):
        raise ValueError("merged pool hash drift")

    def identical_response_stats(other: pd.DataFrame) -> tuple[float, float]:
        left = pool.set_index(KEY)["response"].astype(str)
        right = other.set_index(KEY)["response"].astype(str)
        if set(left.index) != set(right.index):
            raise ValueError("comparison pool key sets differ")
        right = right.loc[left.index]
        same = left.str.strip() == right.str.strip()
        long_same = same & left.str.len().ge(100)
        return float(same.mean()), float(long_same.mean())

    qwen3_same, qwen3_long_same = identical_response_stats(qwen3)
    zephyr_same, zephyr_long_same = identical_response_stats(zephyr)
    # Short label-like responses can legitimately coincide across related
    # checkpoints. A reused candidate pool would also duplicate long responses.
    if (
        qwen3_same >= 0.05
        or zephyr_same >= 0.05
        or qwen3_long_same >= 0.001
        or zephyr_long_same >= 0.001
    ):
        raise ValueError(
            "Qwen7 responses suspiciously duplicate a reference pool: "
            f"Qwen3={qwen3_same:.4f} (long={qwen3_long_same:.4f}), "
            f"Zephyr={zephyr_same:.4f} (long={zephyr_long_same:.4f})"
        )
    report = {
        "status": "PASS",
        "generator": generator["model"],
        "sampling": expected_sampling,
        "proxy_scorer": proxy_spec["model"],
        "gold_scorer": gold_spec["model"],
        "n_rows": len(scored),
        "n_prompts": int(scored["prompt_id"].nunique()),
        "candidates_per_prompt": [
            int(value)
            for value in sorted(
                scored.groupby(["split", "prompt_id"]).size().unique().tolist()
            )
        ],
        "proxy_gold_pearson": float(
            scored[["proxy_score", "gold_score"]].corr().iloc[0, 1]
        ),
        "identical_response_fraction_vs_qwen3": qwen3_same,
        "identical_long_response_fraction_vs_qwen3": qwen3_long_same,
        "identical_response_fraction_vs_zephyr7": zephyr_same,
        "identical_long_response_fraction_vs_zephyr7": zephyr_long_same,
        "pool_scored_sha256": sha256_file(args.run / "pool_scored.jsonl"),
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
