#!/usr/bin/env python3
"""[exp09-07] One-beta worker: LoRA DPO train -> merged policy -> measure.

Runs entirely on the single visible GPU. Steps:
  1) TRL DPOTrainer (LoRA) on proxy preference pairs, given --beta.
  2) merge_and_unload -> in-memory merged policy.
  3) Teacher-forcing logp of the EVAL reference pool under the merged policy
     (ref logp comes precomputed from 06_logp.py) -> implied reweighting
     w ~ exp(dlogp), d_reweight = KL(w||uniform) per prompt (bounded by log m,
     the same units as the theory's d — avoids the B2 saturation trap).
  4) Fresh generation from the merged policy on EVAL prompts (transformers
     generate, same temperature as the ref pool), logp under merged + ref
     for a sequence-KL estimate.
  5) Gold (DeBERTa RM) + proxy (Skywork) scoring of fresh samples.
  6) Per-prompt CSV + summary JSON. No model weights written to disk.
"""
import argparse
import gc
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free():
    # callers must del/None their own references first (del here only unbinds the local)
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--pairs-jsonl", required=True)
    ap.add_argument("--pool-scored-jsonl", required=True, help="eval pool rows with proxy/gold scores")
    ap.add_argument("--pool-ref-logp-jsonl", required=True, help="eval pool rows with logp_ref")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    ap.add_argument("--proxy-model", default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    ap.add_argument("--gold-model", default="OpenAssistant/reward-model-deberta-v3-large-v2")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 overrides epochs (KL sweep via early stop)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--n-fresh", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--skip-fresh-evaluation",
        action="store_true",
        help=(
            "Stop after held-out pool reweight evaluation. This preserves the "
            "DPO-realized metric used by the baseline table while skipping "
            "fresh generation and reward rescoring."
        ),
    )
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if len(args.base_revision) != 40:
        ap.error("--base-revision must be a 40-character commit")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    # ---------- 1) DPO training ----------
    pairs = [json.loads(l) for l in open(args.pairs_jsonl)]
    ds = Dataset.from_list([
        {"prompt": [{"role": "user", "content": p["prompt"]}],
         "chosen": [{"role": "assistant", "content": p["chosen"]}],
         "rejected": [{"role": "assistant", "content": p["rejected"]}]}
        for p in pairs
    ])
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    _epochs_kw = {} if args.max_steps > 0 else {"num_train_epochs": args.epochs}
    cfg = DPOConfig(
        output_dir=str(outdir / "trainer_tmp"), beta=args.beta,
        max_steps=args.max_steps, learning_rate=args.lr,
        **_epochs_kw,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        max_length=1024,
        bf16=True, logging_steps=20, save_strategy="no", report_to=[],
        seed=args.seed,
    )
    trainer = DPOTrainer(model=policy, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_cfg)
    train_out = trainer.train()
    train_loss = float(train_out.training_loss)
    merged = trainer.model.merge_and_unload()
    merged = merged.float()  # fp32 for low-noise logp
    merged.eval()
    del trainer, policy
    free()

    # ---------- 2) reweighting measurement on eval ref pool ----------
    logp_mod = _load("logp06", "06_logp.py")
    pool = [json.loads(l) for l in open(args.pool_ref_logp_jsonl)]
    pool_keys = [(r["prompt_id"], r["sample_idx"]) for r in pool]
    if len(pool_keys) != len(set(pool_keys)):
        raise ValueError("duplicate evaluation key in reference-logp pool")
    scored_rows = [
        r for r in map(json.loads, open(args.pool_scored_jsonl))
        if r["split"] == "eval"
    ]
    scored_keys = [(r["prompt_id"], r["sample_idx"]) for r in scored_rows]
    if len(scored_keys) != len(set(scored_keys)):
        raise ValueError("duplicate evaluation key in scored pool")
    scored = {
        (r["prompt_id"], r["sample_idx"]): r for r in scored_rows
    }
    if set(pool_keys) != set(scored):
        raise ValueError("reference-logp and scored evaluation key sets differ")
    lp_dpo = logp_mod.seq_logprobs(merged, tok, [r["prompt"] for r in pool],
                                   [r["response"] for r in pool], 8, merged.device)
    if len(lp_dpo) != len(pool):
        raise RuntimeError(
            f"policy logp returned {len(lp_dpo)} rows for {len(pool)} candidates"
        )
    by_prompt = defaultdict(list)
    for r, (lp, _) in zip(pool, lp_dpo):
        key = (r["prompt_id"], r["sample_idx"])
        s = scored[key]
        by_prompt[r["prompt_id"]].append(
            (lp - r["logp_ref"], s["gold_score"], s["proxy_score"]))

    rows = []
    for pid, items in by_prompt.items():
        dl = np.array([x[0] for x in items])
        gold = np.array([x[1] for x in items])
        prox = np.array([x[2] for x in items])
        m = len(items)
        w = np.exp(dl - dl.max())
        w = w / w.sum()
        d_rw = float(np.log(m) + np.sum(w * np.log(np.clip(w, 1e-300, None))))
        rows.append({
            "prompt_id": pid, "m": m,
            "d_reweight": max(d_rw, 0.0),
            "ess": float(1.0 / np.sum(w ** 2)),
            "gain_reweight_gold": float(np.sum(w * gold) - gold.mean()),
            "gain_reweight_proxy": float(np.sum(w * prox) - prox.mean()),
            "ref_gold_mean": float(gold.mean()),
            "ref_proxy_mean": float(prox.mean()),
        })

    if args.skip_fresh_evaluation:
        import pandas as pd

        df = pd.DataFrame(rows)
        df["fresh_gold_mean"] = np.nan
        df["fresh_proxy_mean"] = np.nan
        df["fresh_ref_gold_mean"] = np.nan
        df["fresh_ref_proxy_mean"] = np.nan
        df["gain_fresh_gold"] = np.nan
        df["gain_fresh_proxy"] = np.nan
        df.to_csv(outdir / "per_prompt.csv", index=False)
        summary = {
            "beta": args.beta,
            "max_steps": args.max_steps,
            "train_loss": train_loss,
            "n_pairs": len(pairs),
            "n_eval_prompts": len(rows),
            "median_d_reweight": float(df.d_reweight.median()),
            "mean_d_reweight": float(df.d_reweight.mean()),
            "median_ess": float(df.ess.median()),
            "mean_gain_reweight_gold": float(df.gain_reweight_gold.mean()),
            "mean_gain_fresh_gold": None,
            "mean_gain_fresh_proxy": None,
            "mean_seq_kl_nats": None,
            "mean_seq_kl_per_token": None,
            "fresh_evaluation_skipped": True,
            "config": vars(args),
        }
        json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k != "config"},
                indent=2,
            )
        )
        return

    # ---------- 3) fresh generation from merged policy ----------
    eval_prompts = sorted({r["prompt_id"]: r["prompt"] for r in pool}.items())
    fresh = []
    gen_bs = 16
    for i in range(0, len(eval_prompts), gen_bs):
        chunk = eval_prompts[i:i + gen_bs]
        prefixes = [tok.apply_chat_template([{"role": "user", "content": p}],
                                            tokenize=False, add_generation_prompt=True)
                    for _, p in chunk]
        enc = tok(prefixes, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to(merged.device)
        with torch.no_grad():
            gen = merged.generate(**enc, do_sample=True, temperature=args.temperature,
                                  top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                  num_return_sequences=args.n_fresh,
                                  pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
        for j, (pid, p) in enumerate(chunk):
            for k in range(args.n_fresh):
                fresh.append({"prompt_id": pid, "split": "fresh", "sample_idx": k,
                              "prompt": p, "response": texts[j * args.n_fresh + k].strip()})
    with open(outdir / "fresh_samples.jsonl", "w") as f:
        for r in fresh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lp_fresh_dpo = logp_mod.seq_logprobs(merged, tok, [r["prompt"] for r in fresh],
                                         [r["response"] for r in fresh], 8, merged.device)
    del merged
    free()

    ref = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.float32,
        device_map="cuda",
    )
    ref.eval()
    lp_fresh_ref = logp_mod.seq_logprobs(ref, tok, [r["prompt"] for r in fresh],
                                         [r["response"] for r in fresh], 8, ref.device)
    if len(lp_fresh_dpo) != len(fresh) or len(lp_fresh_ref) != len(fresh):
        raise RuntimeError(
            "fresh-sample logp cardinality mismatch: "
            f"samples={len(fresh)}, policy={len(lp_fresh_dpo)}, "
            f"reference={len(lp_fresh_ref)}"
        )
    # matched HF-generate ref baseline (fixes vLLM-vs-HF stack offset, review F8)
    # dtype must ALSO match the merged policy's generation dtype (fp32) — a bf16
    # baseline produced a constant +0.145 gold offset at d=0 in v3.
    ref_bf16 = ref  # fp32, same as merged
    fresh_ref = []
    for i in range(0, len(eval_prompts), gen_bs):
        chunk = eval_prompts[i:i + gen_bs]
        prefixes = [tok.apply_chat_template([{"role": "user", "content": p}],
                                            tokenize=False, add_generation_prompt=True)
                    for _, p in chunk]
        enc = tok(prefixes, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to(ref_bf16.device)
        with torch.no_grad():
            gen = ref_bf16.generate(**enc, do_sample=True, temperature=args.temperature,
                                    top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                    num_return_sequences=args.n_fresh, pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
        for j, (pid, p) in enumerate(chunk):
            for k in range(args.n_fresh):
                fresh_ref.append({"prompt_id": pid, "split": "fresh_ref", "sample_idx": k,
                                  "prompt": p, "response": texts[j * args.n_fresh + k].strip()})
    with open(outdir / "fresh_ref_samples.jsonl", "w") as f:
        for r in fresh_ref:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del ref, ref_bf16
    free()
    seq_kl = [a[0] - b[0] for a, b in zip(lp_fresh_dpo, lp_fresh_ref)]
    ntoks = [max(a[1], 1) for a in lp_fresh_dpo]

    # ---------- 4) score fresh samples (gold + proxy) ----------
    # The scorer deliberately refuses in-place writes. Keep each stage as a
    # separate artifact so a failed/resumed run cannot silently mix old and new
    # scores.
    import subprocess

    def score_staged(stem: str) -> Path:
        raw = outdir / f"{stem}.jsonl"
        gold = outdir / f"{stem}_gold.jsonl"
        scored = outdir / f"{stem}_scored.jsonl"
        subprocess.run(
            [
                sys.executable, str(HERE / "03_score_rm.py"),
                "--in-jsonl", str(raw),
                "--model", args.gold_model,
                "--kind", "deberta",
                "--score-key", "gold_score",
                "--out-jsonl", str(gold),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, str(HERE / "03_score_rm.py"),
                "--in-jsonl", str(gold),
                "--model", args.proxy_model,
                "--kind", "skywork",
                "--score-key", "proxy_score",
                "--out-jsonl", str(scored),
            ],
            check=True,
        )
        return scored

    fresh_scored = score_staged("fresh_samples")
    fresh_ref_scored = score_staged("fresh_ref_samples")

    def group(path):
        g = defaultdict(list)
        seen = set()
        for r in map(json.loads, open(path)):
            key = (r["prompt_id"], r["split"], r["sample_idx"])
            if key in seen:
                raise ValueError(f"duplicate scored fresh-sample key: {key}")
            seen.add(key)
            g[r["prompt_id"]].append((r["gold_score"], r["proxy_score"]))
        return g
    fg = group(fresh_scored)
    fgr = group(fresh_ref_scored)
    expected_prompt_ids = {prompt_id for prompt_id, _ in eval_prompts}
    if set(fg) != expected_prompt_ids or set(fgr) != expected_prompt_ids:
        raise ValueError("fresh scored outputs do not cover the exact eval prompts")
    if any(len(values) != args.n_fresh for values in fg.values()):
        raise ValueError("fresh policy scores have incomplete candidate groups")
    if any(len(values) != args.n_fresh for values in fgr.values()):
        raise ValueError("fresh reference scores have incomplete candidate groups")
    for row in rows:
        vals = fg.get(row["prompt_id"], []); base = fgr.get(row["prompt_id"], [])
        if vals and base:
            row["fresh_gold_mean"] = float(np.mean([v[0] for v in vals]))
            row["fresh_proxy_mean"] = float(np.mean([v[1] for v in vals]))
            row["fresh_ref_gold_mean"] = float(np.mean([v[0] for v in base]))
            row["fresh_ref_proxy_mean"] = float(np.mean([v[1] for v in base]))
            # matched HF-generate baseline (not the vLLM pool)
            row["gain_fresh_gold"] = row["fresh_gold_mean"] - row["fresh_ref_gold_mean"]
            row["gain_fresh_proxy"] = row["fresh_proxy_mean"] - row["fresh_ref_proxy_mean"]

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_prompt.csv", index=False)
    summary = {
        "beta": args.beta, "max_steps": args.max_steps, "train_loss": train_loss, "n_pairs": len(pairs),
        "n_eval_prompts": len(rows),
        "median_d_reweight": float(df.d_reweight.median()),
        "mean_d_reweight": float(df.d_reweight.mean()),
        "median_ess": float(df.ess.median()),
        "mean_gain_reweight_gold": float(df.gain_reweight_gold.mean()),
        "mean_gain_fresh_gold": float(df.gain_fresh_gold.mean()),
        "mean_gain_fresh_proxy": float(df.gain_fresh_proxy.mean()),
        "mean_seq_kl_nats": float(np.mean(seq_kl)),
        "mean_seq_kl_per_token": float(np.mean([k / n for k, n in zip(seq_kl, ntoks)])),
        "config": vars(args),
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
