"""[gpu-02] Open-judge SummEval scoring with anchored rubric prompts.

External-proxy experiment: same structure as exp09 (GPT-4o), but the judge is a
local open model served through vLLM — fixed weights, fixed seed, fully reproducible.
Output matches data/model_proxy_4o_aggregated.csv:
    dimension,item_id,proxy_score_mean,proxy_score_sd,n_judges

This version avoids score collapse by using dimension-specific anchors, a brief
rationale before the final JSON score, decimal scores on the 1.0--5.0 scale, and
a wider generation budget. It also reports basic score-collapse diagnostics.

Usage:
  python 02_open_judge_summeval.py \
    --items-csv data/summeval_items.csv \
    --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel 2 \
    --n-judges 5 \
    --sanity-proxy-csv data/model_proxy_4o_aggregated.csv \
    --out data/model_proxy_openjudge72_aggregated.csv

Then run the prospective pipeline exactly as for GPT-4o:
  python experiments/07_prospective_prediction/01_prospective_prediction.py \
    --human-long-csv data/summeval_long.csv --item-cols article_id,model_id \
    --group-cols dimension --proxy-csv data/model_proxy_openjudge_aggregated.csv \
    --proxy-item-col item_id --proxy-score-col proxy_score_mean \
    --n-repeats 50 --outdir <out>/openjudge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import numpy as np
import pandas as pd
from repro_utils import env_snapshot, resolve_hf_revision

RUBRICS = {
    "coherence": (
        "the collective quality of all sentences: how well the summary is structured and organized",
        "1 = disjointed fragments, no logical flow; 3 = mostly connected but with abrupt jumps "
        "or ordering issues; 5 = flows naturally sentence-to-sentence into a coherent whole",
    ),
    "consistency": (
        "factual alignment with the source article: a consistent summary contains only statements entailed by the source",
        "1 = contains clear fabrications or contradictions of the article; 3 = mostly faithful with "
        "one questionable or exaggerated claim; 5 = every statement is directly supported by the article",
    ),
    "fluency": (
        "the linguistic quality of individual sentences: grammar, word choice, readability",
        "1 = ungrammatical or garbled sentences; 3 = understandable but with awkward phrasing or "
        "minor grammatical errors; 5 = polished, natural, error-free prose",
    ),
    "relevance": (
        "selection of important content: the summary should capture the key points of the article",
        "1 = misses the main points or dwells on trivia; 3 = covers some key points but omits or "
        "buries others; 5 = captures exactly the most important content",
    ),
}

SYS = (
    "You are a meticulous evaluator of news summaries. You first reason briefly, then give a score. "
    "You use the FULL 1.0-5.0 range and avoid defaulting to middle scores: reserve 3.0 for genuinely "
    "middling cases and commit to low/high scores when deserved. Decimal scores (e.g., 2.5, 4.5) are encouraged."
)

USER_TMPL = (
    "Evaluate the summary below on **{dim}** — {definition}.\n"
    "Anchors: {rubric}.\n\n"
    "Article:\n{article}\n\n"
    "Summary:\n{summary}\n\n"
    "First write a 2-3 sentence analysis of the summary's {dim}, citing specifics. "
    "Then, on the final line, output exactly:\n"
    'JSON: {{"score": <float between 1.0 and 5.0>}}'
)


def parse_score(text: str) -> float | None:
    matches = re.findall(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if matches:
        return min(5.0, max(1.0, float(matches[-1])))
    matches = re.findall(
        r"(?:final\s+)?(?:score|rating)\s*(?:is|=|:)?\s*([1-5](?:\.[0-9]+)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return min(5.0, max(1.0, float(matches[-1])))
    matches = re.findall(
        r"^\s*(?:coherence|consistency|fluency|relevance)\s*:\s*([1-5](?:\.[0-9]+)?)\b",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if matches:
        return min(5.0, max(1.0, float(matches[-1])))
    matches = re.findall(r"\b([1-5](?:\.[0-9])?)\s*(?:/\s*5)?\s*$", text.strip())
    return min(5.0, max(1.0, float(matches[-1]))) if matches else None


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


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}; got {list(df.columns)}")


def sanity_report(out: pd.DataFrame, sanity_csv: str | None) -> dict:
    rep: dict[str, float | int] = {}
    print("\n=== SANITY CHECK ===")
    if out.empty:
        rep["empty_output"] = 1
        print("empty output << do not use")
        return rep
    for dim, g in out.groupby("dimension"):
        n = int(len(g))
        top2 = g.proxy_score_mean.round(1).value_counts(normalize=True).head(2).sum()
        std = float(g.proxy_score_mean.std())
        rep[f"{dim}_n"] = n
        rep[f"{dim}_std"] = std
        rep[f"{dim}_top2_mass"] = float(top2)
        if n < 20:
            print(f"{dim:12s} n={n:<4d} std={std:.3f}  top-2-value mass={top2:.2f}  (sample too small for collapse check)")
        else:
            warning = top2 > 0.7 or std < 0.4
            print(
                f"{dim:12s} n={n:<4d} std={std:.3f}  top-2-value mass={top2:.2f}"
                + ("  << COLLAPSE WARNING" if warning else "")
            )
    if sanity_csv:
        ref = pd.read_csv(sanity_csv)
        if "proxy_score_mean" not in ref.columns and "proxy_score" in ref.columns:
            ref = ref.rename(columns={"proxy_score": "proxy_score_mean"})
        require_columns(ref, ["dimension", "item_id", "proxy_score_mean"], Path(sanity_csv))
        m = out.merge(ref, on=["dimension", "item_id"], suffixes=("", "_ref"))
        rep["sanity_overlap_rows"] = int(len(m))
        if m.empty:
            print("reference proxy overlap is empty << check item_id/dimension mapping")
        for dim, g in m.groupby("dimension"):
            corr = g.proxy_score_mean.corr(g.proxy_score_mean_ref)
            c = float(corr) if pd.notna(corr) else float("nan")
            rep[f"{dim}_corr_ref"] = c
            print(
                f"{dim:12s} corr vs reference proxy = {c:.3f}"
                + ("  << TOO LOW, do not use" if pd.notna(corr) and c < 0.3 else "  OK")
            )
    print("Rule: proceed only if there is no collapse warning and optional reference correlations are adequate.\n")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-judges", type=int, default=5, help="samples per (item, dimension)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--model-revision", default=None, help="Optional HF revision/commit for --model")
    ap.add_argument(
        "--sanity-proxy-csv",
        default=None,
        help="Optional reference proxy CSV, e.g. GPT-4o aggregate, for correlation sanity check",
    )
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items_csv = Path(args.items_csv)
    df = pd.read_csv(items_csv)
    require_columns(df, ["item_id", "article", "summary"], items_csv)
    items = df.drop_duplicates(subset=["item_id"])[["item_id", "article", "summary"]]
    if args.limit_items is not None:
        items = items.head(args.limit_items).copy()
    model_sha = resolve_hf_revision(args.model, args.model_revision)
    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    sp = make_sampling_params(
        SamplingParams,
        n=args.n_judges,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    prompts, keys = [], []
    for dim, (definition, rubric) in RUBRICS.items():
        for _, r in items.iterrows():
            msg = [
                {"role": "system", "content": SYS},
                {"role": "user", "content": USER_TMPL.format(
                    dim=dim, definition=definition, rubric=rubric,
                    article=str(r.article)[:12000], summary=str(r.summary))},
            ]
            prompts.append(msg)
            keys.append((dim, r.item_id))

    outs = llm.chat(prompts, sp)
    rows, raw_rows, n_invalid = [], [], 0
    for (dim, item_id), o in zip(keys, outs):
        parsed = [parse_score(c.text) for c in o.outputs]
        for j, (choice, score) in enumerate(zip(o.outputs, parsed)):
            raw_rows.append({
                "dimension": dim,
                "item_id": item_id,
                "judge_sample": j,
                "text": choice.text,
                "score": score,
            })
        scores = parsed
        scores = [s for s in scores if s is not None]
        n_invalid += args.n_judges - len(scores)
        if not scores:
            continue
        rows.append({
            "dimension": dim, "item_id": item_id,
            "proxy_score_mean": float(np.mean(scores)),
            "proxy_score_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n_judges": len(scores),
        })
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    raw_out = Path(args.raw_out) if args.raw_out else Path(args.out).with_suffix(".raw_outputs.jsonl")
    with raw_out.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sanity = sanity_report(out, args.sanity_proxy_csv)
    spec = {
        **vars(args),
        "prompt_version": "v2_rationale_float_rubric",
        "items_csv_sha256": sha256_file(items_csv),
        "model_resolved_sha": model_sha,
        "env": env_snapshot(),
        "n_items": int(len(items)),
        "n_rows": len(out),
        "n_invalid": n_invalid,
        "raw_out": str(raw_out),
        "sanity": sanity,
    }
    Path(args.out).with_suffix(".run_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"saved {args.out}: {len(out)} rows, invalid generations: {n_invalid}")


if __name__ == "__main__":
    main()
