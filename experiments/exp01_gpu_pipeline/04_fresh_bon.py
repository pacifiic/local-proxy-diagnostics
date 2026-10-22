"""[gpu-04] Fresh-generation BoN: does the diagnostic predict gains on NEWLY GENERATED outputs?

Fresh-generation scope experiment. Removes the "fixed item pool" premise entirely:
calibration and evaluation items are fresh samples from an SFT policy, BoN selection
is a real selection mechanism over fresh candidates, and gold is an open judge
(synthetic-gold compromise as in Gao et al. 2023 — documented caveat).

Protocol per prompt:
  1. generate n_cal + n_pool completions from the generator (disjoint sets);
  2. proxy r = reward model score; gold J = judge score (1-10), both on all samples;
  3. estimate (A_hat, B_hat) from the CALIBRATION samples only (plug-in moments);
  4. for N in N_list: repeat K times {draw N pool samples, BoN-select by r, record J};
     realized gain = mean selected J - mean pool J;
  5. predicted gain at d_N = log N - (N-1)/N:  A_hat*sqrt(d_N) + B_hat*d_N;
     baselines: A-only, zero.

Usage:
  python gpu/04_fresh_bon.py \
    --gen-model HuggingFaceH4/mistral-7b-sft-beta \
    --rm-model Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --judge-model meta-llama/Llama-3.3-70B-Instruct \
    --prompts-file prompts.txt --n-prompts 100 \
    --n-cal 32 --n-pool 64 --N-list 2,4,8,16 --outdir fresh_bon_out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import numpy as np
import pandas as pd
import torch
from repro_utils import env_snapshot, resolve_hf_revision


def theory_coeffs(j: np.ndarray, r: np.ndarray):
    sj2, sr2 = float(np.var(j)), float(np.var(r))
    if sr2 <= 0:
        return None
    sr = math.sqrt(sr2)
    cov = float(np.mean((j - j.mean()) * (r - r.mean())))
    kr = float(np.mean((r - r.mean()) ** 3))
    kjrr = float(np.mean((j - j.mean()) * (r - r.mean()) ** 2))
    A = math.sqrt(2.0) * cov / sr
    B = kjrr / sr2 - 2.0 * cov * kr / (3.0 * sr2 ** 2)
    return A, B


JUDGE_SYS = "You are a strict, consistent evaluator of assistant responses. JSON only."
JUDGE_TMPL = (
    "Rate the overall quality (helpfulness, correctness, clarity) of the response "
    "on an integer scale 1 (very poor) to 10 (excellent).\n\n"
    "Instruction:\n{prompt}\n\nResponse:\n{resp}\n\n"
    'Answer with JSON only: {{"score": <1-10>}}'
)


def parse_score(text: str, lo=1.0, hi=10.0):
    m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not m:
        m = re.search(r"\b(10|[1-9])\b", text)
    return min(hi, max(lo, float(m.group(1)))) if m else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sampling_params(SamplingParams, **kwargs):
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def chat_prompt(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"User: {prompt}\nAssistant:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", required=True)
    ap.add_argument("--rm-model", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--gen-revision", default=None)
    ap.add_argument("--rm-revision", default=None)
    ap.add_argument("--judge-revision", default=None)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-pool", type=int, default=64)
    ap.add_argument("--N-list", default="2,4,8,16")
    ap.add_argument("--K", type=int, default=200, help="BoN resamples per N")
    ap.add_argument("--n-judge-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--judge-tensor-parallel", type=int, default=1)
    ap.add_argument("--gen-tensor-parallel", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    prompts_file = Path(args.prompts_file)
    prompts = [l.strip() for l in prompts_file.open(encoding="utf-8") if l.strip()][: args.n_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    n_total = args.n_cal + args.n_pool
    Ns = [int(x) for x in args.N_list.split(",")]
    if max(Ns) > args.n_pool:
        raise ValueError(f"max N={max(Ns)} exceeds n_pool={args.n_pool}")
    model_revisions = {
        "gen_model_resolved_sha": resolve_hf_revision(args.gen_model, args.gen_revision),
        "rm_model_resolved_sha": resolve_hf_revision(args.rm_model, args.rm_revision),
        "judge_model_resolved_sha": resolve_hf_revision(args.judge_model, args.judge_revision),
    }

    # 1) generate fresh samples
    gen = LLM(
        model=args.gen_model,
        revision=args.gen_revision,
        seed=args.seed,
        max_model_len=4096,
        tensor_parallel_size=args.gen_tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    gtok = AutoTokenizer.from_pretrained(
        args.gen_model,
        revision=args.gen_revision,
        trust_remote_code=args.trust_remote_code,
    )
    chats = [chat_prompt(gtok, p) for p in prompts]
    sp = make_sampling_params(
        SamplingParams,
        n=n_total,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    gouts = gen.generate(chats, sp)
    samples = [[c.text for c in o.outputs] for o in gouts]
    sample_rows = []
    for i, (p, ss) in enumerate(zip(prompts, samples)):
        for k, text in enumerate(ss):
            sample_rows.append({"prompt_idx": i, "sample_idx": k, "prompt": p, "response": text})
    pd.DataFrame(sample_rows).to_csv(outdir / "fresh_bon_generated_samples.csv", index=False)
    del gen; torch.cuda.empty_cache()

    # 2a) proxy scores
    rtok = AutoTokenizer.from_pretrained(
        args.rm_model,
        revision=args.rm_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model,
        revision=args.rm_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    if getattr(rm.config, "pad_token_id", None) is None and rtok.pad_token_id is not None:
        rm.config.pad_token_id = rtok.pad_token_id

    @torch.no_grad()
    def rm_scores(p, texts, bs=8):
        out = []
        for i in range(0, len(texts), bs):
            enc_texts = [rtok.apply_chat_template(
                [{"role": "user", "content": p}, {"role": "assistant", "content": t}],
                tokenize=False) for t in texts[i:i + bs]]
            enc = rtok(enc_texts, return_tensors="pt", padding=True, truncation=True,
                       add_special_tokens=False,
                       max_length=4096).to(rm.device)
            s = rm(**enc).logits.squeeze(-1).float().cpu().tolist()
            out.extend([s] if isinstance(s, float) else s)
        return np.array(out)

    R = [rm_scores(p, ss) for p, ss in zip(prompts, samples)]
    proxy_rows = []
    for i, scores in enumerate(R):
        for k, s in enumerate(scores):
            proxy_rows.append({"prompt_idx": i, "sample_idx": k, "proxy_score": float(s)})
    pd.DataFrame(proxy_rows).to_csv(outdir / "fresh_bon_proxy_scores.csv", index=False)
    del rm; torch.cuda.empty_cache()

    # 2b) judge (gold) scores
    judge = LLM(
        model=args.judge_model,
        revision=args.judge_revision,
        seed=args.seed,
        max_model_len=4096,
        tensor_parallel_size=args.judge_tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    jsp = make_sampling_params(
        SamplingParams,
        n=args.n_judge_samples,
        temperature=0.0 if args.n_judge_samples == 1 else 0.3,
        max_tokens=24,
        seed=args.seed,
    )
    jmsgs, jkeys = [], []
    for i, (p, ss) in enumerate(zip(prompts, samples)):
        for k, t in enumerate(ss):
            jmsgs.append([{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": JUDGE_TMPL.format(prompt=p, resp=t[:8000])}])
            jkeys.append((i, k))
    jouts = judge.chat(jmsgs, jsp)
    J = [np.full(n_total, np.nan) for _ in prompts]
    judge_rows = []
    for (i, k), o in zip(jkeys, jouts):
        vals = [parse_score(c.text) for c in o.outputs]
        for sidx, (choice, score) in enumerate(zip(o.outputs, vals)):
            judge_rows.append({
                "prompt_idx": i,
                "sample_idx": k,
                "judge_sample": sidx,
                "raw_text": choice.text,
                "score": score,
            })
        vals = [v for v in vals if v is not None]
        if vals:
            J[i][k] = float(np.mean(vals))
    pd.DataFrame(judge_rows).to_csv(outdir / "fresh_bon_judge_raw_outputs.csv", index=False)
    pd.DataFrame([
        {"prompt_idx": i, "sample_idx": k, "gold_score": float(v) if not np.isnan(v) else np.nan}
        for i, arr in enumerate(J)
        for k, v in enumerate(arr)
    ]).to_csv(outdir / "fresh_bon_gold_scores.csv", index=False)
    del judge; torch.cuda.empty_cache()

    # 3-5) per-prompt calibration -> BoN prediction
    rows, skipped = [], []
    for i, p in enumerate(prompts):
        j, r = J[i], R[i]
        ok = ~np.isnan(j)
        if ok.sum() < n_total * 0.9:
            skipped.append({"prompt_idx": i, "reason": "too_many_unparsed_judge_scores", "n_valid": int(ok.sum())})
            continue
        j, r = j[ok], r[ok]
        if len(j) < args.n_cal + max(Ns):
            skipped.append({"prompt_idx": i, "reason": "not_enough_valid_samples_after_parsing", "n_valid": int(len(j))})
            continue
        perm = rng.permutation(len(j))
        cal, pool = perm[: args.n_cal], perm[args.n_cal:]
        if len(pool) < max(Ns):
            skipped.append({"prompt_idx": i, "reason": "pool_smaller_than_max_N", "pool": int(len(pool))})
            continue
        co = theory_coeffs(j[cal], r[cal])
        if co is None:
            skipped.append({"prompt_idx": i, "reason": "zero_proxy_variance"})
            continue
        A, B = co
        base = float(j[pool].mean())
        for N in Ns:
            sel = []
            for _ in range(args.K):
                pick = rng.choice(pool, size=N, replace=False)
                sel.append(j[pick[np.argmax(r[pick])]])
            realized = float(np.mean(sel) - base)
            dN = math.log(N) - (N - 1) / N
            pred = A * math.sqrt(dN) + B * dN
            rows.append({"prompt_idx": i, "N": N, "d_N": dN,
                         "realized": realized, "pred_AB": pred,
                         "pred_Aonly": A * math.sqrt(dN), "pred_zero": 0.0,
                         "A_cal": A, "B_cal": B})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "fresh_bon_per_prompt.csv", index=False)
    pd.DataFrame(skipped).to_csv(outdir / "fresh_bon_skipped.csv", index=False)
    if df.empty:
        raise RuntimeError("No valid prompt rows; see fresh_bon_skipped.csv")
    summary_rows = []
    for N, g in df.groupby("N", sort=True):
        summary_rows.append({
            "N": N,
            "n_prompts": len(g),
            "mean_realized": g.realized.mean(),
            "mae_AB": (g.pred_AB - g.realized).abs().mean(),
            "mae_Aonly": (g.pred_Aonly - g.realized).abs().mean(),
            "mae_zero": g.realized.abs().mean(),
        })
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(outdir / "fresh_bon_summary.csv", index=False)
    spec = vars(args) | {
        **model_revisions,
        "prompts_file_sha256": sha256_file(prompts_file),
        "env": env_snapshot(),
        "n_prompts_loaded": len(prompts),
        "n_skipped_prompts": len(skipped),
    }
    (outdir / "run_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
