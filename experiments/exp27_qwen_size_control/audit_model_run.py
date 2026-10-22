#!/usr/bin/env python3
"""Audit one deterministic generator/scorer run without model-specific assumptions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["prompt_id", "split", "sample_idx"]
IDENTITY = KEY + ["prompt", "response"]
EXPECTED_PROXY = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
EXPECTED_GOLD = "OpenAssistant/reward-model-deberta-v3-large-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-generator", required=True)
    parser.add_argument("--expected-cal-prompts", type=int, default=100)
    parser.add_argument("--expected-eval-prompts", type=int, default=200)
    parser.add_argument("--expected-candidates", type=int, default=32)
    parser.add_argument("--expected-base-seed", type=int, default=0)
    parser.add_argument(
        "--expected-pretokenize-prompts", action="store_true"
    )
    parser.add_argument("--expected-fix-mistral-regex", action="store_true")
    parser.add_argument("--expected-skip-vllm-tokenizer", action="store_true")
    parser.add_argument("--expected-mistral-common-chat", action="store_true")
    parser.add_argument("--expected-text-only", action="store_true")
    parser.add_argument(
        "--expected-vllm-tokenizer-mode", default="auto"
    )
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.out_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_json}")
    frames = {
        name: pd.read_json(args.run / filename, lines=True)
        for name, filename in {
            "pool": "pool.jsonl",
            "proxy": "pool_proxy.jsonl",
            "gold": "pool_gold.jsonl",
            "scored": "pool_scored.jsonl",
        }.items()
    }
    expected_rows = args.expected_candidates * (
        args.expected_cal_prompts + args.expected_eval_prompts
    )
    reference = frames["pool"][IDENTITY].sort_values(KEY).reset_index(drop=True)
    for name, frame in frames.items():
        if len(frame) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, got {len(frame)}")
        if frame.duplicated(KEY).any():
            raise ValueError(f"{name}: duplicate key")
        identity = frame[IDENTITY].sort_values(KEY).reset_index(drop=True)
        if not reference.equals(identity):
            raise ValueError(f"{name}: candidate identity drift")
    scored = frames["scored"]
    if not np.isfinite(scored[["proxy_score", "gold_score"]].to_numpy(float)).all():
        raise ValueError("non-finite scores")
    counts = scored.groupby(["split", "prompt_id"]).size()
    if set(counts.tolist()) != {args.expected_candidates}:
        raise ValueError("candidate count per prompt mismatch")
    prompts = scored.groupby("split")["prompt_id"].nunique().to_dict()
    if prompts != {
        "calibration": args.expected_cal_prompts,
        "eval": args.expected_eval_prompts,
    }:
        raise ValueError(f"prompt count mismatch: {prompts}")
    prompt_identity = scored[["prompt_id", "prompt"]].drop_duplicates()
    expected_prompt_ids = prompt_identity["prompt"].map(
        lambda prompt: hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:16]
    )
    if not prompt_identity["prompt_id"].astype(str).reset_index(drop=True).equals(
        expected_prompt_ids.reset_index(drop=True)
    ):
        raise ValueError("prompt_id does not match SHA-256(prompt)[:16]")

    proxy_values = frames["proxy"][KEY + ["proxy_score"]]
    gold_values = frames["gold"][KEY + ["gold_score"]]
    score_check = (
        scored[KEY + ["proxy_score", "gold_score"]]
        .merge(proxy_values, on=KEY, suffixes=("_merged", "_source"), validate="one_to_one")
        .merge(gold_values, on=KEY, suffixes=("", "_gold_source"), validate="one_to_one")
    )
    if not np.array_equal(
        score_check["proxy_score_merged"].to_numpy(dtype=float),
        score_check["proxy_score_source"].to_numpy(dtype=float),
    ):
        raise ValueError("merged proxy scores differ from proxy scorer output")
    if not np.array_equal(
        score_check["gold_score"].to_numpy(dtype=float),
        score_check["gold_score_gold_source"].to_numpy(dtype=float),
    ):
        raise ValueError("merged gold scores differ from gold scorer output")

    generator = json.loads(
        (args.run / "pool.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    if generator["model"] != args.expected_generator:
        raise ValueError(f"wrong generator: {generator['model']}")
    for key, expected in {
        "n_cal": args.expected_candidates,
        "n_eval": args.expected_candidates,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 320,
        "base_seed": args.expected_base_seed,
    }.items():
        if not math.isclose(float(generator[key]), float(expected)):
            raise ValueError(f"wrong generator setting {key}={generator[key]}")
    if "one vLLM request with n=1 per candidate" not in generator["seed_policy"]:
        raise ValueError("candidate-level seed policy not recorded")
    if bool(generator.get("pretokenize_prompts", False)) != (
        args.expected_pretokenize_prompts
    ):
        raise ValueError("prompt tokenization mode mismatch")
    if bool(generator.get("fix_mistral_regex", False)) != (
        args.expected_fix_mistral_regex
    ):
        raise ValueError("Mistral regex-fix setting mismatch")
    if bool(generator.get("skip_vllm_tokenizer", False)) != (
        args.expected_skip_vllm_tokenizer
    ):
        raise ValueError("vLLM tokenizer initialization setting mismatch")
    if generator.get("vllm_tokenizer_mode", "auto") != (
        args.expected_vllm_tokenizer_mode
    ):
        raise ValueError("vLLM tokenizer mode mismatch")
    if bool(generator.get("mistral_common_chat", False)) != (
        args.expected_mistral_common_chat
    ):
        raise ValueError("Mistral chat-encoding setting mismatch")
    if bool(generator.get("text_only", False)) != args.expected_text_only:
        raise ValueError("text-only model setting mismatch")
    revisions = json.loads(
        (args.run / "model_revisions.json").read_text(encoding="utf-8")
    )
    if revisions["generator"]["model"] != args.expected_generator:
        raise ValueError("generator revision manifest model mismatch")
    for role, record in revisions.items():
        if len(record["revision"]) != 40:
            raise ValueError(f"{role}: invalid revision {record['revision']}")
    proxy_spec = json.loads(
        (args.run / "pool_proxy.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    gold_spec = json.loads(
        (args.run / "pool_gold.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    if proxy_spec["model"] != EXPECTED_PROXY or proxy_spec["score_key"] != "proxy_score":
        raise ValueError("wrong proxy scorer")
    if gold_spec["model"] != EXPECTED_GOLD or gold_spec["score_key"] != "gold_score":
        raise ValueError("wrong gold scorer")
    for role, scorer_spec, source_name in (
        ("proxy_scorer", proxy_spec, "pool"),
        ("gold_scorer", gold_spec, "pool"),
    ):
        if scorer_spec.get("revision") is not None and (
            scorer_spec["revision"] != revisions[role]["revision"]
        ):
            raise ValueError(f"{role}: scorer spec/revision manifest mismatch")
        if scorer_spec.get("input_sha256") is not None and (
            scorer_spec["input_sha256"]
            != sha256_file(args.run / f"{source_name}.jsonl")
        ):
            raise ValueError(f"{role}: scorer input hash mismatch")
        if scorer_spec.get("output_sha256") is not None:
            output_name = "pool_proxy.jsonl" if role == "proxy_scorer" else "pool_gold.jsonl"
            if scorer_spec["output_sha256"] != sha256_file(args.run / output_name):
                raise ValueError(f"{role}: scorer output hash mismatch")
        environment = scorer_spec.get("environment", {})
        if environment and environment.get("user_site_enabled") is not False:
            raise ValueError(f"{role}: user-site packages were not disabled")
    merge = json.loads((args.run / "merge_audit.json").read_text(encoding="utf-8"))
    if merge["output"]["sha256"] != sha256_file(args.run / "pool_scored.jsonl"):
        raise ValueError("merge hash drift")
    report = {
        "status": "PASS",
        "generator": generator["model"],
        "model_revision": revisions["generator"]["revision"],
        "proxy_revision": revisions["proxy_scorer"]["revision"],
        "gold_revision": revisions["gold_scorer"]["revision"],
        "n_rows": len(scored),
        "n_prompts": int(scored["prompt_id"].nunique()),
        "candidates_per_prompt": args.expected_candidates,
        "proxy_scorer": proxy_spec["model"],
        "gold_scorer": gold_spec["model"],
        "proxy_gold_pearson": float(
            scored[["proxy_score", "gold_score"]].corr().iloc[0, 1]
        ),
        "pool_sha256": sha256_file(args.run / "pool.jsonl"),
        "pool_scored_sha256": sha256_file(args.run / "pool_scored.jsonl"),
        "score_propagation_exact": True,
        "prompt_hash_identity_pass": True,
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
