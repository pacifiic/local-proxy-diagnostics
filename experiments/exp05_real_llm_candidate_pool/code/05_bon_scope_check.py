#!/usr/bin/env python3
"""Prompt-conditional Best-of-N scope check for generated LLM candidates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import mkdir  # noqa: E402


def ensure_empty_outdir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Use --overwrite or choose a fresh --outdir."
        )
    return mkdir(path)


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"Invalid integer list: {text}")
    return vals


def central_moments(j: np.ndarray, r: np.ndarray) -> tuple[float, float, float, float]:
    jm = float(np.mean(j))
    rm = float(np.mean(r))
    jc = j - jm
    rc = r - rm
    var_r = float(np.mean(rc**2))
    cov = float(np.mean(jc * rc))
    kappa_r = float(np.mean(rc**3))
    kappa_jrr = float(np.mean(jc * rc**2))
    return var_r, cov, kappa_r, kappa_jrr


def conditional_ab(df: pd.DataFrame, prompt_col: str, proxy_col: str, gold_col: str) -> tuple[float, float]:
    vals: list[tuple[float, float, float, float]] = []
    for _, g in df.groupby(prompt_col, sort=True, dropna=False):
        if len(g) < 2:
            continue
        r = g[proxy_col].to_numpy(dtype=float)
        j = g[gold_col].to_numpy(dtype=float)
        mask = np.isfinite(r) & np.isfinite(j)
        if int(mask.sum()) >= 2:
            vals.append(central_moments(j[mask], r[mask]))
    if not vals:
        raise ValueError("No prompt groups with at least two finite candidates.")
    arr = np.asarray(vals, dtype=float)
    var_r, cov, kappa_r, kappa_jrr = [float(np.mean(arr[:, i])) for i in range(arr.shape[1])]
    if var_r <= 0:
        raise ValueError("Mean within-prompt proxy variance is zero.")
    a = math.sqrt(2.0) * cov / math.sqrt(var_r)
    b = kappa_jrr / var_r - 2.0 * cov * kappa_r / (3.0 * var_r**2)
    return a, b


def split_prompt_ids(prompt_ids: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[set[str], set[str]]:
    ids = np.asarray(sorted(pd.unique(prompt_ids)), dtype=str)
    order = np.arange(len(ids))
    rng.shuffle(order)
    n_cal = int(round(frac * len(ids)))
    n_cal = max(1, min(n_cal, len(ids) - 1))
    return set(ids[order[:n_cal]].tolist()), set(ids[order[n_cal:]].tolist())


def bon_weights_with_replacement(proxy_scores: np.ndarray, n_best: int) -> np.ndarray:
    r = np.asarray(proxy_scores, dtype=float)
    m = len(r)
    if m == 0:
        raise ValueError("Need at least one candidate.")
    weights = np.zeros(m, dtype=float)
    values = np.sort(np.unique(r))
    n_lower = 0
    for value in values:
        idx = np.flatnonzero(r == value)
        n_equal = len(idx)
        p_block = ((n_lower + n_equal) / m) ** n_best - (n_lower / m) ** n_best
        weights[idx] = p_block / n_equal
        n_lower += n_equal
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid BoN weights.")
    return weights / total


def kl_to_uniform(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    m = len(w)
    mask = w > 0
    return float(np.sum(w[mask] * np.log(w[mask] * m)))


def evaluate_bon(df: pd.DataFrame, prompt_col: str, proxy_col: str, gold_col: str, n_best: int) -> tuple[float, float, float]:
    gains: list[float] = []
    kls: list[float] = []
    prompt_sizes: list[int] = []
    for _, g in df.groupby(prompt_col, sort=True, dropna=False):
        if len(g) < 2:
            continue
        r = g[proxy_col].to_numpy(dtype=float)
        j = g[gold_col].to_numpy(dtype=float)
        w = bon_weights_with_replacement(r, n_best)
        gains.append(float(np.sum(w * j) - np.mean(j)))
        kls.append(kl_to_uniform(w))
        prompt_sizes.append(int(len(g)))
    if not gains:
        raise ValueError("No evaluable prompt groups.")
    return float(np.mean(gains)), float(np.mean(kls)), float(np.median(prompt_sizes))


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt-conditional BoN scope check for LLM candidate pools.")
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--prompt-col", default="prompt_id")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--proxy-col", default="proxy_score")
    ap.add_argument("--gold-col", default="gold_score")
    ap.add_argument("--use-existing-split", action="store_true")
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--n-list", default="2,4,8,16")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    if not (0.0 < args.calibration_frac < 1.0):
        raise ValueError("--calibration-frac must be in (0,1).")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be positive.")
    df = pd.read_csv(args.candidates_csv)
    required = {args.prompt_col, args.proxy_col, args.gold_col}
    if args.use_existing_split:
        required.add(args.split_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidates CSV missing columns: {sorted(missing)}")
    before_drop = len(df)
    df = df.dropna(subset=[args.prompt_col, args.proxy_col, args.gold_col]).copy()
    if df.empty:
        raise ValueError("No rows remain after dropping missing prompt/proxy/gold values.")
    n_list = parse_int_list(args.n_list)
    rng = np.random.default_rng(args.random_seed)
    prompt_ids = df[args.prompt_col].astype(str).to_numpy()
    n_repeats = 1 if args.use_existing_split else args.n_repeats
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for repeat in range(n_repeats):
        if args.use_existing_split:
            cal_ids = set(df.loc[df[args.split_col].astype(str).eq("calibration"), args.prompt_col].astype(str))
            eval_ids = set(df.loc[df[args.split_col].astype(str).eq("eval"), args.prompt_col].astype(str))
            if not cal_ids or not eval_ids:
                raise ValueError("Existing split must contain both 'calibration' and 'eval' prompts.")
        else:
            cal_ids, eval_ids = split_prompt_ids(prompt_ids, args.calibration_frac, rng)
        cal = df[df[args.prompt_col].astype(str).isin(cal_ids)].copy()
        eva = df[df[args.prompt_col].astype(str).isin(eval_ids)].copy()
        try:
            a, b = conditional_ab(cal, args.prompt_col, args.proxy_col, args.gold_col)
            for n_best in n_list:
                actual_gain, mean_kl, median_prompt_size = evaluate_bon(eva, args.prompt_col, args.proxy_col, args.gold_col, n_best)
                pred = a * math.sqrt(max(mean_kl, 0.0)) + b * mean_kl
                rows.append(
                    {
                        "repeat": repeat,
                        "n_best": n_best,
                        "n_cal_prompts": len(cal_ids),
                        "n_eval_prompts": len(eval_ids),
                        "median_prompt_size": median_prompt_size,
                        "mean_prompt_kl": mean_kl,
                        "actual_bon_gain": actual_gain,
                        "local_ab_at_mean_kl": pred,
                        "abs_error_local_ab": abs(pred - actual_gain),
                        "a_cal": a,
                        "b_cal": b,
                    }
                )
        except Exception as exc:
            skipped.append({"repeat": repeat, "reason": str(exc)})
    row_df = pd.DataFrame(rows)
    pd.DataFrame(skipped).to_csv(outdir / "llm_bon_scope_skipped.csv", index=False)
    if row_df.empty:
        raise RuntimeError("No BoN rows produced; see llm_bon_scope_skipped.csv")
    summary = (
        row_df.groupby("n_best", dropna=False)
        .agg(
            n_rows=("repeat", "count"),
            mean_prompt_kl=("mean_prompt_kl", "mean"),
            median_prompt_kl=("mean_prompt_kl", "median"),
            mean_actual_bon_gain=("actual_bon_gain", "mean"),
            mean_local_ab_at_mean_kl=("local_ab_at_mean_kl", "mean"),
            mae_local_ab=("abs_error_local_ab", "mean"),
            median_prompt_size=("median_prompt_size", "median"),
        )
        .reset_index()
    )
    row_df.to_csv(outdir / "llm_bon_scope_rows.csv", index=False)
    summary.to_csv(outdir / "llm_bon_scope_summary.csv", index=False)
    (outdir / "llm_bon_scope_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    spec = {
        "script": "experiments/exp05_real_llm_candidate_pool/code/05_bon_scope_check.py",
        "candidates_csv": args.candidates_csv,
        "prompt_col": args.prompt_col,
        "proxy_col": args.proxy_col,
        "gold_col": args.gold_col,
        "use_existing_split": args.use_existing_split,
        "calibration_frac": args.calibration_frac,
        "n_repeats": n_repeats,
        "random_seed": args.random_seed,
        "n_list": n_list,
        "n_input_rows": int(len(df)),
        "n_dropped_missing_prompt_proxy_gold": int(before_drop - len(df)),
        "n_prompt_ids": int(pd.Series(prompt_ids).nunique()),
        "n_rows": int(len(row_df)),
        "n_skipped": int(len(skipped)),
    }
    (outdir / "llm_bon_scope_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote LLM BoN scope outputs to: {outdir}")


if __name__ == "__main__":
    main()
