#!/usr/bin/env python3
"""Generate candidate responses for prompt-level reranking experiments with vLLM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def format_prompt(prompt: str, template: str, tokenizer: object | None = None) -> str:
    if template == "plain":
        return prompt
    if template == "instruction":
        return f"### Instruction:\n{prompt}\n\n### Response:\n"
    if template == "chat":
        # render the model's own chat template so instruct models generate
        # on-distribution; the stored prompt column stays plain.
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    raise ValueError(f"Unknown --prompt-template: {template}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LLM candidates with vLLM.")
    ap.add_argument("--prompts-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--hf-token-file", default="", help="Optional path to a Hugging Face token file. The token is not written to outputs.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--prompt-template", choices=["plain", "instruction", "chat"], default="plain")
    ap.add_argument("--n-candidates", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-spec-json", default="")
    args = ap.parse_args()

    token_used = load_hf_token(args.hf_token_file)
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0,1].")
    if args.max_model_len < 0:
        raise ValueError("--max-model-len must be nonnegative.")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive.")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is required for candidate generation. Install it in a GPU environment, "
            "for example with `pip install vllm`."
        ) from exc

    prompts = pd.read_csv(args.prompts_csv)
    required = {"prompt_id", "prompt", "split"}
    missing = required - set(prompts.columns)
    if missing:
        raise ValueError(f"Prompts CSV missing columns: {sorted(missing)}")
    if args.n_candidates < 1:
        raise ValueError("--n-candidates must be positive.")

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.max_model_len > 0:
        llm_kwargs["max_model_len"] = args.max_model_len
    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        # Older vLLM releases did not expose every constructor keyword.
        llm_kwargs.pop("seed", None)
        llm = LLM(**llm_kwargs)
    sampling_kwargs = {
        "n": args.n_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    try:
        sampling = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("seed", None)
        sampling = SamplingParams(**sampling_kwargs)
    chat_tokenizer = None
    if args.prompt_template == "chat":
        from transformers import AutoTokenizer
        chat_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    formatted = [format_prompt(str(p), args.prompt_template, chat_tokenizer) for p in prompts["prompt"].tolist()]
    generations = llm.generate(formatted, sampling)

    rows: list[dict[str, object]] = []
    for prompt_row, request_output in zip(prompts.to_dict(orient="records"), generations):
        for j, output in enumerate(request_output.outputs):
            rows.append(
                {
                    "prompt_id": prompt_row["prompt_id"],
                    "split": prompt_row["split"],
                    "candidate_id": f"{prompt_row['prompt_id']}__cand_{j:03d}",
                    "candidate_index": j,
                    "prompt": prompt_row["prompt"],
                    "response": output.text,
                    "finish_reason": getattr(output, "finish_reason", None),
                    "generation_model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                }
            )
    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    spec = {
        "script": "experiments/exp05_real_llm_candidate_pool/code/02_generate_candidates_vllm.py",
        "prompts_csv": args.prompts_csv,
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "prompt_template": args.prompt_template,
        "n_candidates": args.n_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "n_prompts": int(len(prompts)),
        "n_rows": int(len(out)),
    }
    spec_path = Path(args.out_spec_json) if args.out_spec_json else out_path.with_suffix(".run_spec.json")
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} candidates to {out_path}")


if __name__ == "__main__":
    main()
