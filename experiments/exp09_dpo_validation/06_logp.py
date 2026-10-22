#!/usr/bin/env python3
"""[exp09-06] Teacher-forcing sequence log-probs of (prompt, response) rows under a model.

Chat prefix rendered with add_generation_prompt=True; response tokens only are summed.
Same construction for ref and DPO models so shared-prefix effects cancel in the diff.
"""
import argparse
import json

import numpy as np
import torch

DEFAULT_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def seq_logprobs(model, tok, prompts, responses, batch_size, device, max_length=2048):
    out = []
    for i in range(0, len(prompts), batch_size):
        ps, rs = prompts[i:i + batch_size], responses[i:i + batch_size]
        prefixes = [tok.apply_chat_template([{"role": "user", "content": p}],
                                            tokenize=False, add_generation_prompt=True)
                    for p in ps]
        fulls = [pre + r for pre, r in zip(prefixes, rs)]
        pre_ids = [tok(pre, add_special_tokens=False)["input_ids"] for pre in prefixes]
        enc = tok(fulls, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for b in range(len(ps)):
            ids = enc["input_ids"][b]
            attn = enc["attention_mask"][b]
            plen = min(len(pre_ids[b]), int(attn.sum()) - 1)
            total = 0.0
            n_tok = 0
            for t in range(plen, int(attn.sum())):
                if t == 0:
                    continue
                total += float(logprobs[b, t - 1, ids[t]])
                n_tok += 1
            out.append((total, n_tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True, help="rows with prompt/response")
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--logp-key", required=True)
    ap.add_argument("--filter-split", default="", help="only rows with this split value (empty=all)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=["bfloat16","float32"], default="float32")
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()
    if len(args.revision) != 40:
        ap.error("--revision must be a 40-character commit")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(args.in_jsonl)]
    if args.filter_split:
        rows = [r for r in rows if r.get("split") == args.filter_split]
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dt,
        device_map="cuda",
    )
    model.eval()
    lp = seq_logprobs(model, tok, [r["prompt"] for r in rows],
                      [r["response"] for r in rows], args.batch_size, model.device)
    assert len(lp) == len(rows)
    with open(args.out_jsonl, "w") as f:
        for r, (total, ntok) in zip(rows, lp):
            r[args.logp_key] = total
            r[args.logp_key + "_ntok"] = ntok
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"logp[{args.logp_key}] for {len(rows)} rows -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
