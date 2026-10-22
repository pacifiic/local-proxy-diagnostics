"""[gpu-03] W1 audit: does real policy optimization shift proxy-score mass like a KL-matched tilt?

Audit of the learned-reward-model pipeline and Theorem C term (i).
Label-free: no human ratings anywhere.

Protocol per prompt x:
  1. sample n_ref completions from the reference (SFT) policy and n_opt from the
     optimized (DPO/RLHF) policy;
  2. score all completions with a proxy reward model r;
  3. estimate the per-prompt policy KL  d = E_opt[log p_opt - log p_ref]
     (teacher-forced logprobs of the opt samples under both models);
  4. tilt the reference score sample to radius d (exponential tilt in r);
  5. report W1(opt scores, tilted ref scores)  vs  the no-tilt baseline
     W1(opt scores, raw ref scores).

If W1_tilt << W1_raw, real optimization moves score mass like the KL-matched tilt
(term (i) small). Either outcome supports the paper: small = bridge validated,
large = the label-free warning signal works.

Usage:
  python gpu/03_w1_audit.py \
    --ref-model HuggingFaceH4/mistral-7b-sft-beta \
    --opt-model HuggingFaceH4/zephyr-7b-beta \
    --rm-model Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --prompts-file prompts.txt --n-prompts 200 --n-ref 64 --n-opt 64 \
    --outdir w1_audit_out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import numpy as np
import pandas as pd
import torch
from repro_utils import env_snapshot, resolve_hf_revision


# ---- tilt helpers (mirrors lib/common_experiment_utils.py) ----
def log_mean_exp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.mean(np.exp(x - m))))


def tilt_weights(r: np.ndarray, t: float) -> np.ndarray:
    a = t * r - np.max(t * r)
    w = np.exp(a)
    return w / w.sum()


def kl_of_tilt(r: np.ndarray, t: float) -> float:
    w = tilt_weights(r, t)
    return float(t * np.sum(w * r) - log_mean_exp(t * r))


def find_t(r: np.ndarray, d: float) -> float:
    if d <= 0 or np.var(r) == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while kl_of_tilt(r, hi) < d:
        lo, hi = hi, hi * 2
        if hi > 1e6:
            return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if kl_of_tilt(r, mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def w1_weighted(x: np.ndarray, wx: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(x, y, u_weights=wx))


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


# ---- logprob scoring ----
@torch.no_grad()
def seq_logprobs(model, tok, prompts, completions, batch_size, device):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(prompts), batch_size):
        ps, cs = prompts[i:i + batch_size], completions[i:i + batch_size]
        full = [p + c for p, c in zip(ps, cs)]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
        plen = [len(tok(p).input_ids) for p in ps]
        logits = model(**enc).logits.float()
        logsm = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = enc.input_ids[:, 1:]
        lp_tok = logsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = enc.attention_mask[:, 1:].clone()
        for j, pl in enumerate(plen):
            mask[j, : max(pl - 1, 0)] = 0
        out.extend((lp_tok * mask).sum(-1).cpu().tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--opt-model", required=True)
    ap.add_argument("--rm-model", required=True)
    ap.add_argument("--ref-revision", default=None)
    ap.add_argument("--opt-revision", default=None)
    ap.add_argument("--rm-revision", default=None)
    ap.add_argument("--prompts-file", required=True, help="one prompt per line (e.g., UltraFeedback test prompts)")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--n-ref", type=int, default=64)
    ap.add_argument("--n-opt", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tilt-dcap-frac", type=float, default=0.5,
                    help="Cap the tilt KL target at this fraction of log(n_ref) "
                         "(feasible radius of an n-atom tilt); d_hat beyond the cap "
                         "is recorded as saturated.")
    ap.add_argument("--lp-batch", type=int, default=8)
    ap.add_argument("--gen-tensor-parallel", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    prompts_file = Path(args.prompts_file)
    prompts = [l.strip() for l in prompts_file.open(encoding="utf-8") if l.strip()][: args.n_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    model_revisions = {
        "ref_model_resolved_sha": resolve_hf_revision(args.ref_model, args.ref_revision),
        "opt_model_resolved_sha": resolve_hf_revision(args.opt_model, args.opt_revision),
        "rm_model_resolved_sha": resolve_hf_revision(args.rm_model, args.rm_revision),
    }

    # 1) generation (vLLM, one model at a time to fit memory)
    def generate(model_id, revision, n):
        llm = LLM(
            model=model_id,
            revision=revision,
            seed=args.seed,
            max_model_len=4096,
            tensor_parallel_size=args.gen_tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )
        tok = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
        )
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        sp = make_sampling_params(
            SamplingParams,
            n=n,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        outs = llm.generate(chats, sp)
        del llm
        torch.cuda.empty_cache()
        return chats, [[c.text for c in o.outputs] for o in outs]

    ref_chats, ref_samps = generate(args.ref_model, args.ref_revision, args.n_ref)
    opt_chats, opt_samps = generate(args.opt_model, args.opt_revision, args.n_opt)

    # 2) proxy scores
    rm_tok = AutoTokenizer.from_pretrained(
        args.rm_model,
        revision=args.rm_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model,
        revision=args.rm_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    if getattr(rm.config, "pad_token_id", None) is None and rm_tok.pad_token_id is not None:
        rm.config.pad_token_id = rm_tok.pad_token_id

    @torch.no_grad()
    def rm_score(prompt, texts):
        scores = []
        for i in range(0, len(texts), args.lp_batch):
            chunk = texts[i:i + args.lp_batch]
            enc_texts = [rm_tok.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": t}],
                tokenize=False) for t in chunk]
            enc = rm_tok(enc_texts, return_tensors="pt", padding=True, truncation=True,
                         add_special_tokens=False,
                         max_length=4096).to(rm.device)
            s = rm(**enc).logits.squeeze(-1).float().cpu().tolist()
            scores.extend([s] if isinstance(s, float) else s)
        return np.array(scores)

    # 3) KL: logprobs of opt samples under both models (HF teacher forcing)
    def load_lm(mid, revision):
        m = AutoModelForCausalLM.from_pretrained(
            mid,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        ).eval()
        t = AutoTokenizer.from_pretrained(
            mid,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
        )
        t.pad_token = t.pad_token or t.eos_token
        return m, t

    rows = []
    lm_opt, tok_opt = load_lm(args.opt_model, args.opt_revision)
    flat_prompts_opt = [opt_chats[i] for i in range(len(prompts)) for _ in opt_samps[i]]
    flat_comps = [c for cs in opt_samps for c in cs]
    lp_opt = seq_logprobs(lm_opt, tok_opt, flat_prompts_opt, flat_comps, args.lp_batch, lm_opt.device)
    del lm_opt; torch.cuda.empty_cache()
    lm_ref, tok_ref = load_lm(args.ref_model, args.ref_revision)
    flat_prompts_ref = [ref_chats[i] for i in range(len(prompts)) for _ in opt_samps[i]]
    lp_ref = seq_logprobs(lm_ref, tok_ref, flat_prompts_ref, flat_comps, args.lp_batch, lm_ref.device)
    del lm_ref; torch.cuda.empty_cache()

    idx = 0
    for i, p in enumerate(prompts):
        k = len(opt_samps[i])
        d_hat = float(np.mean([lo - lr for lo, lr in zip(lp_opt[idx:idx + k], lp_ref[idx:idx + k])]))
        idx += k
        d_hat = max(d_hat, 0.0)
        r_ref = rm_score(p, ref_samps[i])
        r_opt = rm_score(p, opt_samps[i])
        # An exponential tilt over n atoms cannot exceed KL = log(n); solving for
        # the raw sequence-level d_hat (tens of nats) saturates the tilt onto a
        # single atom (ESS=1), which makes W1(tilt) a point-mass artifact.  Clip
        # the target to a feasible radius and record the saturation explicitly.
        d_feasible = args.tilt_dcap_frac * math.log(len(r_ref))
        d_used = min(d_hat, d_feasible)
        saturated = d_hat > d_feasible
        t = find_t(r_ref, d_used)
        w = tilt_weights(r_ref, t)
        w1_raw = w1_weighted(r_ref, np.ones_like(r_ref) / len(r_ref), r_opt)
        w1_tilt = w1_weighted(r_ref, w, r_opt)
        rows.append({
            "prompt_idx": i, "d_hat": d_hat, "d_used": d_used,
            "d_feasible": d_feasible, "saturated": bool(saturated), "t": t,
            "w1_raw": w1_raw,
            "w1_tilt": w1_tilt,
            "w1_reduction": float(1.0 - w1_tilt / w1_raw) if w1_raw > 0 else np.nan,
            "mean_r_ref": float(r_ref.mean()), "mean_r_opt": float(r_opt.mean()),
            "mean_r_tilt": float((w * r_ref).sum()),
            "sd_r_ref": float(r_ref.std()),
            "ess": float(1.0 / np.sum(w ** 2)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "w1_audit_per_prompt.csv", index=False)
    valid_ratio = df.loc[df.w1_raw > 0, "w1_tilt"] / df.loc[df.w1_raw > 0, "w1_raw"]
    summary = {
        "n_prompts": len(df),
        "median_d_hat": float(df.d_hat.median()),
        "median_d_used": float(df.d_used.median()),
        "n_saturated": int(df.saturated.sum()),
        "median_ess": float(df.ess.median()),
        "median_w1_raw": float(df.w1_raw.median()),
        "median_w1_tilt": float(df.w1_tilt.median()),
        "median_reduction": float(1 - valid_ratio.median()) if len(valid_ratio) else None,
        "frac_tilt_better": float((df.w1_tilt < df.w1_raw).mean()),
        "config": vars(args) | {
            **model_revisions,
            "prompts_file_sha256": sha256_file(prompts_file),
            "env": env_snapshot(),
        },
    }
    (outdir / "w1_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
