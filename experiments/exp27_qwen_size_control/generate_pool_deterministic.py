#!/usr/bin/env python3
"""Generate one vLLM request per candidate with a stable candidate-level seed."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, split: str, prompt_id: str, sample_idx: int) -> int:
    payload = f"{base_seed}|{split}|{prompt_id}|{sample_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def read_prompts(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if limit > 0:
        rows = rows[:limit]
    ids = [str(row["prompt_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate prompt_id")
    if any(not str(row["prompt"]).strip() for row in rows):
        raise ValueError(f"{path}: empty prompt")
    for row in rows:
        expected = hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest()[:16]
        if str(row["prompt_id"]) != expected:
            raise ValueError(
                f"{path}: prompt_id does not match prompt text: "
                f"{row['prompt_id']} != {expected}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-id",
        help="Public model ID to record when --model is a local directory",
    )
    parser.add_argument(
        "--model-revision",
        help="Pinned 40-character revision for a local model directory",
    )
    parser.add_argument("--n-cal", type=int, default=32)
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--pretokenize-prompts",
        action="store_true",
        help="Pass chat-template token IDs directly to vLLM",
    )
    parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        help="Enable the Transformers tokenizer fix for affected Mistral models",
    )
    parser.add_argument(
        "--skip-vllm-tokenizer",
        action="store_true",
        help="Use external prompt token IDs and decode generated token IDs",
    )
    parser.add_argument(
        "--vllm-tokenizer-mode",
        choices=("auto", "hf", "slow", "mistral"),
        default="auto",
    )
    parser.add_argument(
        "--mistral-common-chat",
        action="store_true",
        help=(
            "Encode chat prompts with the official mistral-common tokenizer "
            "from the local tekken.json"
        ),
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Disable multimodal inputs when loading a multimodal text generator",
    )
    parser.add_argument("--limit-prompts-per-split", type=int, default=0)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()
    if args.out_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_jsonl}")
    if min(args.n_cal, args.n_eval) < 1:
        parser.error("--n-cal and --n-eval must be positive")
    if args.skip_vllm_tokenizer and not args.pretokenize_prompts:
        parser.error("--skip-vllm-tokenizer requires --pretokenize-prompts")
    if args.mistral_common_chat and not args.pretokenize_prompts:
        parser.error("--mistral-common-chat requires --pretokenize-prompts")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_path = Path(args.model)
    if model_path.is_dir():
        if not args.model_id or not args.model_revision:
            parser.error(
                "local --model requires --model-id and --model-revision"
            )
        if len(args.model_revision) != 40:
            parser.error("--model-revision must be a 40-character commit")
        load_target = str(model_path.resolve())
        recorded_model = args.model_id
        model_revision = args.model_revision
    else:
        snapshot_path = snapshot_download(args.model, local_files_only=True)
        load_target = args.model
        recorded_model = args.model
        model_revision = Path(snapshot_path).name
    mistral_tokenizer = None
    if args.mistral_common_chat:
        if not model_path.is_dir():
            parser.error("--mistral-common-chat requires a local model directory")
        tekken_path = model_path / "tekken.json"
        if not tekken_path.is_file():
            parser.error(f"missing Mistral tokenizer file: {tekken_path}")
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        mistral_tokenizer = MistralTokenizer.from_file(str(tekken_path))
        tokenizer = None
    else:
        tokenizer_kwargs = (
            {"fix_mistral_regex": True} if args.fix_mistral_regex else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(load_target, **tokenizer_kwargs)
    jobs: list[dict[str, object]] = []
    rendered: list[str | TokensPrompt] = []
    sampling: list[SamplingParams] = []
    for split, n_candidates in (
        ("calibration", args.n_cal),
        ("eval", args.n_eval),
    ):
        rows = read_prompts(
            args.prompts_dir / f"prompts_{split}.jsonl",
            args.limit_prompts_per_split,
        )
        for row in rows:
            prompt_id = str(row["prompt_id"])
            prompt = str(row["prompt"])
            messages = [{"role": "user", "content": prompt}]
            if args.pretokenize_prompts:
                if mistral_tokenizer is not None:
                    encoded = mistral_tokenizer.encode_chat_completion(
                        ChatCompletionRequest(
                            messages=[UserMessage(content=prompt)]
                        )
                    )
                    token_ids = encoded.tokens
                else:
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    token_ids = (
                        encoded["input_ids"]
                        if isinstance(encoded, Mapping)
                        else encoded
                    )
                if not token_ids or not all(
                    isinstance(token_id, int) for token_id in token_ids
                ):
                    raise TypeError("chat template did not return integer token IDs")
                chat: str | TokensPrompt = TokensPrompt(
                    prompt_token_ids=list(token_ids)
                )
            else:
                if tokenizer is None:
                    raise RuntimeError(
                        "mistral-common chat encoding requires pretokenization"
                    )
                chat = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            for sample_idx in range(n_candidates):
                candidate_seed = stable_seed(
                    args.base_seed, split, prompt_id, sample_idx
                )
                jobs.append(
                    {
                        "prompt_id": prompt_id,
                        "split": split,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "candidate_seed": candidate_seed,
                    }
                )
                rendered.append(chat)
                sampling.append(
                    SamplingParams(
                        n=1,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        seed=candidate_seed,
                    )
                )

    llm_kwargs = {}
    if args.text_only:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(
        model=load_target,
        dtype="bfloat16",
        seed=args.base_seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        skip_tokenizer_init=args.skip_vllm_tokenizer,
        tokenizer_mode=args.vllm_tokenizer_mode,
        **llm_kwargs,
    )
    generations = llm.generate(rendered, sampling)
    if len(generations) != len(jobs):
        raise RuntimeError("vLLM returned the wrong number of requests")
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for job, generation in zip(jobs, generations):
            if len(generation.outputs) != 1:
                raise RuntimeError("candidate-level request returned n != 1")
            if args.skip_vllm_tokenizer:
                token_ids = list(generation.outputs[0].token_ids)
                if mistral_tokenizer is not None:
                    response = mistral_tokenizer.decode(token_ids).strip()
                else:
                    response = tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    ).strip()
            else:
                response = generation.outputs[0].text.strip()
            if not response:
                raise ValueError(
                    "empty response for "
                    f"{job['split']}/{job['prompt_id']}/{job['sample_idx']}"
                )
            row = {
                "prompt_id": job["prompt_id"],
                "split": job["split"],
                "sample_idx": job["sample_idx"],
                "prompt": job["prompt"],
                "response": response,
                "candidate_seed": job["candidate_seed"],
                "generated_tokens": len(generation.outputs[0].token_ids),
                "finish_reason": generation.outputs[0].finish_reason,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    spec = {
        "generator": "generate_pool_deterministic.py",
        "model": recorded_model,
        "model_revision": model_revision,
        "model_loaded_from_local_directory": model_path.is_dir(),
        "n_cal": args.n_cal,
        "n_eval": args.n_eval,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "base_seed": args.base_seed,
        "seed_policy": (
            "SHA-256(base_seed, split, prompt_id, sample_idx), first 32 bits; "
            "one vLLM request with n=1 per candidate"
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "pretokenize_prompts": args.pretokenize_prompts,
        "fix_mistral_regex": args.fix_mistral_regex,
        "skip_vllm_tokenizer": args.skip_vllm_tokenizer,
        "vllm_tokenizer_mode": args.vllm_tokenizer_mode,
        "mistral_common_chat": args.mistral_common_chat,
        "text_only": args.text_only,
        "limit_prompts_per_split": args.limit_prompts_per_split,
        "n_samples": len(jobs),
        "prompt_inputs": {
            split: {
                "path": str(
                    args.prompts_dir / f"prompts_{split}.jsonl"
                ),
                "sha256": sha256_file(
                    args.prompts_dir / f"prompts_{split}.jsonl"
                ),
            }
            for split in ("calibration", "eval")
        },
    }
    Path(f"{args.out_jsonl}.run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(jobs)} deterministic candidates -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
