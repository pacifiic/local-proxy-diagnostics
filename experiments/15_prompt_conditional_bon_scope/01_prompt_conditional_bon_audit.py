#!/usr/bin/env python3
"""Prompt-conditional Best-of-N scope audit.

The local theory in the paper targets a smooth raw-score exponential tilt on a
fixed finite item set.  Best-of-N instead selects within each prompt group using
a discrete rank-score policy.  This script quantifies that mismatch without
claiming BoN is another validation target for the local tilt law.
"""
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

from common_experiment_utils import (  # noqa: E402
    PathData,
    find_t_for_dmax,
    tilt_weights,
    generate_human_paths,
    mkdir,
    parse_cols,
    theory_coefficients,
)


def ensure_empty_outdir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Use --overwrite or choose a fresh --outdir."
        )
    return mkdir(path)


def parse_n_list(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if any(n < 1 for n in out):
        raise ValueError("--n-list values must be positive integers.")
    return out


def prompt_keys_from_item_ids(item_ids: np.ndarray, sep: str, prompt_col_index: int) -> np.ndarray:
    keys: list[str] = []
    for item_id in item_ids:
        parts = str(item_id).split(sep)
        if prompt_col_index >= len(parts):
            raise ValueError(
                f"Item id {item_id!r} has only {len(parts)} component(s); "
                f"cannot take prompt-col-index {prompt_col_index}."
            )
        keys.append(parts[prompt_col_index])
    return np.asarray(keys, dtype=str)


def bon_weights_with_replacement(proxy_scores: np.ndarray, n_best: int) -> np.ndarray:
    """Exact BoN selection probabilities under uniform with-replacement draws.

    For each tied proxy-score block, the probability that the block is the top
    observed block is allocated uniformly across tied items by symmetry.
    """
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


def evaluate_path(path: PathData, prompt_keys: np.ndarray, n_list: list[int], tilt_max_d: float) -> list[dict[str, Any]]:
    coeff = theory_coefficients(path.gold_scores, path.proxy_scores)
    if not np.isfinite(tilt_max_d) or tilt_max_d <= 0:
        raise ValueError("tilt_max_d must be finite and positive.")
    r = np.asarray(path.proxy_scores, dtype=float)
    pooled_j = np.asarray(path.gold_scores, dtype=float)
    if r.shape != pooled_j.shape or not np.all(np.isfinite(r)) or not np.all(np.isfinite(pooled_j)):
        raise ValueError("Proxy and gold must be aligned finite vectors.")
    n_top = int(np.count_nonzero(r == np.max(r)))
    pooled_kl_supremum = math.log(len(r) / n_top)

    rows: list[dict[str, Any]] = []
    prompts = sorted(pd.unique(prompt_keys))
    for n_best in n_list:
        prompt_gains: list[float] = []
        prompt_kls: list[float] = []
        prompt_sizes: list[int] = []
        for prompt in prompts:
            idx = np.flatnonzero(prompt_keys == prompt)
            if len(idx) < 2:
                continue
            w = bon_weights_with_replacement(path.proxy_scores[idx], n_best)
            j = path.gold_scores[idx]
            prompt_gains.append(float(np.sum(w * j) - np.mean(j)))
            prompt_kls.append(kl_to_uniform(w))
            prompt_sizes.append(int(len(idx)))
        if not prompt_gains:
            continue
        d_mean = float(np.mean(prompt_kls))
        conditional_gain = float(np.mean(prompt_gains))
        local_pred = coeff.a_exact * math.sqrt(d_mean) + coeff.b_theory * d_mean if d_mean >= 0 else float("nan")
        tilt_pred = float("nan")
        reachable = d_mean <= tilt_max_d and d_mean < pooled_kl_supremum
        if d_mean == 0.0:
            reachable = True
            tilt_pred = 0.0
        elif reachable:
            # Solve at the requested radius, not at an unreachable outer cutoff.
            scale = float(np.std(r))
            standardized_r = (r - np.mean(r)) / scale
            t = find_t_for_dmax(standardized_r, d_mean)
            w = tilt_weights(standardized_r, t)
            if not np.isclose(kl_to_uniform(w), d_mean, rtol=1e-7, atol=1e-10):
                raise RuntimeError("Pooled tilt did not attain the requested KL.")
            tilt_pred = float(np.dot(w, pooled_j) - np.mean(pooled_j))
        rows.append(
            {
                "group_key": path.group_key,
                "path_id": path.path_id,
                "n_best": n_best,
                "n_prompts": len(prompt_gains),
                "median_prompt_size": float(np.median(prompt_sizes)),
                "conditional_bon_gain": conditional_gain,
                "prompt_mean_kl": d_mean,
                "prompt_median_kl": float(np.median(prompt_kls)),
                "prompt_kl_le_0p10": float(np.mean(np.asarray(prompt_kls) <= 0.10)),
                "prompt_kl_le_0p05": float(np.mean(np.asarray(prompt_kls) <= 0.05)),
                "global_tilt_reachable": bool(reachable),
                "pooled_kl_supremum": pooled_kl_supremum,
                "global_tilt_at_mean_kl": tilt_pred,
                "local_law_at_mean_kl": local_pred,
                "abs_err_global_tilt": abs(tilt_pred - conditional_gain) if np.isfinite(tilt_pred) else float("nan"),
                "abs_err_local_law": abs(local_pred - conditional_gain) if np.isfinite(local_pred) else float("nan"),
                "a_exact": coeff.a_exact,
                "b_theory": coeff.b_theory,
            }
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_best, g in df.groupby("n_best", dropna=False):
        rows.append(
            {
                "n_best": int(n_best),
                "n_paths": int(g["path_id"].nunique()),
                "n_rows": int(len(g)),
                "mean_conditional_bon_gain": float(g["conditional_bon_gain"].mean()),
                "median_conditional_bon_gain": float(g["conditional_bon_gain"].median()),
                "mean_prompt_kl": float(g["prompt_mean_kl"].mean()),
                "median_prompt_kl": float(g["prompt_median_kl"].median()),
                "mean_prompt_share_kl_le_0p10": float(g["prompt_kl_le_0p10"].mean()),
                "mean_prompt_share_kl_le_0p05": float(g["prompt_kl_le_0p05"].mean()),
                "share_paths_global_tilt_reachable": float(g["global_tilt_reachable"].mean()),
                "mae_global_tilt_at_mean_kl": float(g["abs_err_global_tilt"].mean()),
                "mae_local_law_at_mean_kl": float(g["abs_err_local_law"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt-conditional BoN scope audit.")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", required=True)
    ap.add_argument("--prompt-col-index", type=int, default=0)
    ap.add_argument("--item-id-sep", default="||")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--n-list", default="2,4,8,16,32,64")
    ap.add_argument("--tilt-max-d", type=float, default=0.25)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    df = pd.read_csv(args.human_long_csv)
    paths = list(
        generate_human_paths(
            df=df,
            item_cols=parse_cols(args.item_cols),
            rater_col=args.rater_col,
            score_col=args.score_col,
            group_cols=parse_cols(args.group_cols),
            min_items=args.min_items,
            min_raters=args.min_raters,
        )
    )
    if not paths:
        raise RuntimeError("No admissible repeated-rating paths generated.")
    n_list = parse_n_list(args.n_list)

    spec = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_name": args.dataset_name,
        "human_long_csv": args.human_long_csv,
        "item_cols": args.item_cols,
        "prompt_col_index": args.prompt_col_index,
        "item_id_sep": args.item_id_sep,
        "group_cols": args.group_cols,
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "n_list": n_list,
        "tilt_max_d": args.tilt_max_d,
        "n_paths": len(paths),
        "policy": "exact prompt-conditional with-replacement BoN probabilities",
    }
    (outdir / "prompt_conditional_bon_run_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for path in paths:
        prompt_keys = prompt_keys_from_item_ids(path.item_ids, args.item_id_sep, args.prompt_col_index)
        for row in evaluate_path(path, prompt_keys, n_list, args.tilt_max_d):
            row["dataset"] = args.dataset_name
            rows.append(row)
    if not rows:
        raise RuntimeError("No prompt-conditional BoN rows produced.")

    paths_df = pd.DataFrame(rows)
    summary_df = summarize(paths_df)
    paths_df.to_csv(outdir / "prompt_conditional_bon_paths.csv", index=False)
    summary_df.to_csv(outdir / "prompt_conditional_bon_summary.csv", index=False)
    (outdir / "prompt_conditional_bon_summary.md").write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Saved prompt-conditional BoN scope outputs in: {outdir}")


if __name__ == "__main__":
    main()
