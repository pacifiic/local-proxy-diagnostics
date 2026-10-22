#!/usr/bin/env python3
"""[exp09-02] Sample the reference policy (vLLM) for train / calibration / eval prompt sets.

The vLLM request seed and model revision are fixed; the chat template is rendered
with the model's own pinned tokenizer. Output: samples.jsonl rows
{prompt_id, split, sample_idx, prompt, response}.
"""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def read_jsonl(p):
    with open(p, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()
    if len(args.revision) != 40:
        ap.error("--revision must be a 40-character commit")
    output_path = Path(args.out_jsonl)
    spec_path = Path(f"{args.out_jsonl}.run_spec.json")
    existing = [path for path in (output_path, spec_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite pool artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    jobs = []  # (prompt_id, split, rendered)
    n_per = {"train": args.n_train, "calibration": args.n_cal, "eval": args.n_eval}
    prompt_text = {}
    for split, n in n_per.items():
        if n <= 0:
            continue  # e.g. cross-model runs need no train split (SamplingParams rejects n=0)
        for r in read_jsonl(Path(args.prompts_dir) / f"prompts_{split}.jsonl"):
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            jobs.append((r["prompt_id"], split, rendered, n))
            prompt_text[r["prompt_id"]] = r["prompt"]

    llm = LLM(model=args.model, revision=args.revision,
              tokenizer_revision=args.revision, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = output_path.open("w", encoding="utf-8")
    total = 0
    # group jobs by n to batch with a single SamplingParams each
    for n in sorted(set(j[3] for j in jobs)):
        sub = [j for j in jobs if j[3] == n]
        sp = SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, seed=args.seed)
        gens = llm.generate([j[2] for j in sub], sp)
        if len(gens) != len(sub):
            raise RuntimeError(
                f"vLLM returned {len(gens)} prompt outputs for {len(sub)} inputs"
            )
        for (pid, split, _, _), g in zip(sub, gens):
            if len(g.outputs) != n:
                raise RuntimeError(
                    f"{split}/{pid}: vLLM returned {len(g.outputs)} "
                    f"candidates, expected {n}"
                )
            for k, o in enumerate(g.outputs):
                out.write(json.dumps({
                    "prompt_id": pid, "split": split, "sample_idx": k,
                    "prompt": prompt_text[pid], "response": o.text.strip(),
                }, ensure_ascii=False) + "\n")
                total += 1
    out.close()
    expected_total = sum(
        len(read_jsonl(Path(args.prompts_dir) / f"prompts_{split}.jsonl")) * n
        for split, n in n_per.items() if n > 0
    )
    if total != expected_total:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"generated {total} candidates, expected {expected_total}"
        )
    spec_path.write_text(
        json.dumps(
            vars(args)
            | {
                "n_samples": total,
                "output_sha256": sha256_file(output_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {total} samples -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
