#!/usr/bin/env python3
"""Audit prompt -> generation -> scoring -> merge lineage for real-LLM pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


KEY = ("prompt_id", "split", "sample_idx")
IDENTITY = KEY + ("prompt", "response")
EXPECTED_PROXY = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
EXPECTED_GOLD = "OpenAssistant/reward-model-deberta-v3-large-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(
    base_seed: int, split: str, prompt_id: str, sample_idx: int
) -> int:
    payload = f"{base_seed}|{split}|{prompt_id}|{sample_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def keyed(rows: list[dict], path: Path) -> dict[tuple[object, ...], dict]:
    result: dict[tuple[object, ...], dict] = {}
    for index, row in enumerate(rows):
        missing = [field for field in IDENTITY if field not in row]
        if missing:
            raise ValueError(f"{path}: row {index} missing {missing}")
        key = tuple(row[field] for field in KEY)
        if key in result:
            raise ValueError(f"{path}: duplicate candidate key {key}")
        if not str(row["prompt"]).strip() or not str(row["response"]).strip():
            raise ValueError(f"{path}: empty prompt or response at {key}")
        result[key] = row
    return result


def require_same_identity(
    reference: dict[tuple[object, ...], dict],
    other: dict[tuple[object, ...], dict],
    *,
    label: str,
) -> None:
    if set(reference) != set(other):
        raise ValueError(f"{label}: candidate key sets differ")
    for key in reference:
        left = tuple(reference[key][field] for field in IDENTITY)
        right = tuple(other[key][field] for field in IDENTITY)
        if left != right:
            raise ValueError(f"{label}: candidate identity drift at {key}")


def audit_prompt_files(run: Path) -> tuple[dict[str, dict[str, str]], dict]:
    split_records: dict[str, dict[str, str]] = {}
    hashes: dict[str, str] = {}
    id_sets: dict[str, set[str]] = {}
    expected_counts = {"calibration": 100, "eval": 200}
    for split, expected_count in expected_counts.items():
        path = run / f"prompts_{split}.jsonl"
        rows = load_jsonl(path)
        if len(rows) != expected_count:
            raise ValueError(
                f"{run.name}: {split} has {len(rows)} prompts, "
                f"expected {expected_count}"
            )
        records: dict[str, str] = {}
        for row in rows:
            prompt = str(row["prompt"])
            prompt_id = str(row["prompt_id"])
            expected_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            if prompt_id != expected_id:
                raise ValueError(
                    f"{run.name}: prompt hash mismatch {prompt_id} != {expected_id}"
                )
            if prompt_id in records:
                raise ValueError(f"{run.name}: duplicate {split} prompt {prompt_id}")
            records[prompt_id] = prompt
        split_records[split] = records
        id_sets[split] = set(records)
        hashes[split] = sha256_file(path)
    if id_sets["calibration"] & id_sets["eval"]:
        raise ValueError(f"{run.name}: calibration/eval prompt overlap")
    return split_records, hashes


def audit_run(run: Path) -> dict[str, object]:
    prompt_records, prompt_hashes = audit_prompt_files(run)
    paths = {
        "pool": run / "pool.jsonl",
        "proxy": run / "pool_proxy.jsonl",
        "gold": run / "pool_gold.jsonl",
        "scored": run / "pool_scored.jsonl",
    }
    rows = {name: load_jsonl(path) for name, path in paths.items()}
    maps = {name: keyed(rows[name], paths[name]) for name in paths}
    reference = maps["pool"]
    for name in ("proxy", "gold", "scored"):
        require_same_identity(reference, maps[name], label=f"{run.name}/{name}")

    if len(reference) != 9600:
        raise ValueError(f"{run.name}: expected 9600 candidates, got {len(reference)}")
    expected_keys: set[tuple[object, ...]] = set()
    for split, prompts in prompt_records.items():
        for prompt_id, prompt in prompts.items():
            for sample_idx in range(32):
                expected_keys.add((prompt_id, split, sample_idx))
                row = reference.get((prompt_id, split, sample_idx))
                if row is None:
                    raise ValueError(
                        f"{run.name}: missing {split}/{prompt_id}/{sample_idx}"
                    )
                if row["prompt"] != prompt:
                    raise ValueError(
                        f"{run.name}: prompt text drift at "
                        f"{split}/{prompt_id}/{sample_idx}"
                    )
    if set(reference) != expected_keys:
        raise ValueError(f"{run.name}: unexpected candidate keys")

    generation_spec = json.loads(
        (run / "pool.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    base_seed = int(generation_spec["base_seed"])
    derived_seeds: list[int] = []
    for key, row in reference.items():
        prompt_id, split, sample_idx = key
        expected_seed = stable_seed(
            base_seed, str(split), str(prompt_id), int(sample_idx)
        )
        derived_seeds.append(expected_seed)
        if "candidate_seed" in row and int(row["candidate_seed"]) != expected_seed:
            raise ValueError(f"{run.name}: candidate seed mismatch at {key}")
    seed_collision_count = len(derived_seeds) - len(set(derived_seeds))

    for key in reference:
        proxy_score = float(maps["proxy"][key].get("proxy_score", math.nan))
        gold_score = float(maps["gold"][key].get("gold_score", math.nan))
        merged_proxy = float(maps["scored"][key].get("proxy_score", math.nan))
        merged_gold = float(maps["scored"][key].get("gold_score", math.nan))
        if not all(
            math.isfinite(value)
            for value in (proxy_score, gold_score, merged_proxy, merged_gold)
        ):
            raise ValueError(f"{run.name}: non-finite score at {key}")
        if proxy_score != merged_proxy:
            raise ValueError(f"{run.name}: proxy score merge drift at {key}")
        if gold_score != merged_gold:
            raise ValueError(f"{run.name}: gold score merge drift at {key}")

    proxy_spec = json.loads(
        (run / "pool_proxy.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    gold_spec = json.loads(
        (run / "pool_gold.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    if (
        proxy_spec.get("model") != EXPECTED_PROXY
        or proxy_spec.get("kind") != "skywork"
        or proxy_spec.get("score_key") != "proxy_score"
    ):
        raise ValueError(f"{run.name}: wrong proxy scorer specification")
    if (
        gold_spec.get("model") != EXPECTED_GOLD
        or gold_spec.get("kind") != "deberta"
        or gold_spec.get("score_key") != "gold_score"
    ):
        raise ValueError(f"{run.name}: wrong gold scorer specification")
    for spec, output_name in (
        (proxy_spec, "proxy"),
        (gold_spec, "gold"),
    ):
        if spec.get("input_sha256") is not None and (
            spec["input_sha256"] != sha256_file(paths["pool"])
        ):
            raise ValueError(f"{run.name}: {output_name} scorer input hash drift")
        if spec.get("output_sha256") is not None and (
            spec["output_sha256"] != sha256_file(paths[output_name])
        ):
            raise ValueError(f"{run.name}: {output_name} scorer output hash drift")

    merge = json.loads((run / "merge_audit.json").read_text(encoding="utf-8"))
    expected_merge_hashes = {
        "pool": sha256_file(paths["pool"]),
        "proxy": sha256_file(paths["proxy"]),
        "gold": sha256_file(paths["gold"]),
    }
    for role, expected_hash in expected_merge_hashes.items():
        if merge["inputs"][role]["sha256"] != expected_hash:
            raise ValueError(f"{run.name}: merge input hash drift for {role}")
    if merge["output"]["sha256"] != sha256_file(paths["scored"]):
        raise ValueError(f"{run.name}: merge output hash drift")

    revisions = json.loads(
        (run / "model_revisions.json").read_text(encoding="utf-8")
    )
    for role in ("generator", "proxy_scorer", "gold_scorer"):
        revision = str(revisions[role]["revision"])
        if len(revision) != 40:
            raise ValueError(f"{run.name}: invalid {role} revision")
    if revisions["generator"]["model"] != generation_spec["model"]:
        raise ValueError(f"{run.name}: generator model/revision manifest drift")
    if revisions["proxy_scorer"]["model"] != EXPECTED_PROXY:
        raise ValueError(f"{run.name}: proxy model/revision manifest drift")
    if revisions["gold_scorer"]["model"] != EXPECTED_GOLD:
        raise ValueError(f"{run.name}: gold model/revision manifest drift")

    constant_response_prompts = []
    for (split, prompt_id), group in _group_by_prompt(rows["pool"]).items():
        if len({row["response"] for row in group}) == 1:
            constant_response_prompts.append(f"{split}/{prompt_id}")

    return {
        "run": run.name,
        "path": str(run),
        "generator": generation_spec["model"],
        "generator_revision": revisions["generator"]["revision"],
        "base_seed": base_seed,
        "n_rows": len(reference),
        "prompt_file_sha256": prompt_hashes,
        "pool_sha256": sha256_file(paths["pool"]),
        "scored_sha256": sha256_file(paths["scored"]),
        "candidate_seed_collisions": seed_collision_count,
        "constant_response_prompt_count": len(constant_response_prompts),
        "constant_response_prompts": constant_response_prompts,
        "proxy_score_min": min(
            float(row["proxy_score"]) for row in rows["proxy"]
        ),
        "proxy_score_max": max(
            float(row["proxy_score"]) for row in rows["proxy"]
        ),
        "gold_score_min": min(float(row["gold_score"]) for row in rows["gold"]),
        "gold_score_max": max(float(row["gold_score"]) for row in rows["gold"]),
        "score_propagation_exact": True,
        "prompt_identity_exact": True,
        "candidate_grid_complete": True,
    }


def _group_by_prompt(
    rows: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["prompt_id"]))
        groups.setdefault(key, []).append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if len({run.resolve() for run in args.run}) != len(args.run):
        parser.error("duplicate --run path")

    records = [audit_run(run) for run in args.run]
    calibration_hashes = {
        record["prompt_file_sha256"]["calibration"] for record in records
    }
    eval_hashes = {record["prompt_file_sha256"]["eval"] for record in records}
    cross_run_prompt_identity = (
        len(calibration_hashes) == 1 and len(eval_hashes) == 1
    )
    if not cross_run_prompt_identity:
        raise ValueError("prompt files differ across runs")
    seed_collisions = sum(
        int(record["candidate_seed_collisions"]) for record in records
    )
    checks = {
        "all_run_lineage_checks_pass": True,
        "prompt_files_identical_across_runs": cross_run_prompt_identity,
        "candidate_seed_collisions_zero": seed_collisions == 0,
    }
    report = {
        "experiment": "real_llm_source_pipeline_lineage_audit",
        "pass": bool(all(checks.values())),
        "checks": checks,
        "n_runs": len(records),
        "runs": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["pass"]:
        raise RuntimeError(f"source pipeline audit failed: {checks}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
