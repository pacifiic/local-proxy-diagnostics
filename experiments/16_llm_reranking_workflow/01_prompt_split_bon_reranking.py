#!/usr/bin/env python3
"""Prompt-split candidate-pool BoN/reranking experiment.

This experiment is a lightweight proxy for an LLM candidate-selection workflow.
It treats each prompt's existing candidate outputs as the candidate pool, uses a
fixed proxy to select the best item among N uniformly sampled candidates, and
measures the resulting held-out gold gain on evaluation prompts.

Unlike the scope-only BoN audit, this script estimates A and B on calibration
prompts and evaluates prediction error on disjoint evaluation prompts. It does
not train an LLM or call an API; it is intended as a fast, reproducible
pre-RLHF/reranking diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import (  # noqa: E402
    PathData,
    ensure_item_id,
    generate_human_paths,
    generate_model_paths,
    mkdir,
    parse_cols,
    theory_coefficients,
)


@dataclass
class PredictorCoefficients:
    a_exact: float
    b_theory: float
    a_nu: float
    n_cal_items: int
    n_cal_prompts: int


def ensure_empty_outdir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}\n"
            "Use --overwrite or choose a fresh --outdir."
        )
    return mkdir(path)


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer.")
    if any(v < 1 for v in vals):
        raise ValueError("All list values must be positive.")
    return vals


def parse_nonnegative_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer.")
    if any(v < 0 for v in vals):
        raise ValueError("All list values must be nonnegative.")
    return vals


def stable_seed(*parts: object, base_seed: int = 0) -> int:
    payload = "::".join(str(p) for p in (base_seed, *parts))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def prompt_keys_from_item_ids(item_ids: np.ndarray, sep: str, prompt_col_indices: Sequence[int]) -> np.ndarray:
    keys: list[str] = []
    for item_id in item_ids:
        parts = str(item_id).split(sep)
        if any(i >= len(parts) or i < 0 for i in prompt_col_indices):
            raise ValueError(
                f"Item id {item_id!r} has {len(parts)} component(s); "
                f"cannot take prompt-col-indices {list(prompt_col_indices)}."
            )
        keys.append(sep.join(parts[i] for i in prompt_col_indices))
    return np.asarray(keys, dtype=str)


def split_prompt_keys(
    prompt_keys: np.ndarray,
    calibration_frac: float,
    rng: np.random.Generator,
    min_cal_prompts: int,
    min_eval_prompts: int,
) -> tuple[set[str], set[str]]:
    prompts = np.asarray(sorted(pd.unique(prompt_keys)), dtype=str)
    if len(prompts) < min_cal_prompts + min_eval_prompts:
        raise ValueError(
            f"Need at least {min_cal_prompts + min_eval_prompts} prompts; got {len(prompts)}."
        )
    shuffled = prompts.copy()
    rng.shuffle(shuffled)
    n_cal = int(round(calibration_frac * len(shuffled)))
    n_cal = max(min_cal_prompts, n_cal)
    n_cal = min(n_cal, len(shuffled) - min_eval_prompts)
    return set(shuffled[:n_cal]), set(shuffled[n_cal:])


def bon_weights_with_replacement(proxy_scores: np.ndarray, n_best: int) -> np.ndarray:
    """Exact selected-item probabilities for Best-of-N with replacement.

    The base generator is uniform over the candidate pool. Ties are handled by
    assigning the tied top-score block uniformly across the tied items.
    """
    r = np.asarray(proxy_scores, dtype=float)
    m = len(r)
    if m == 0:
        raise ValueError("Need at least one candidate.")
    values = np.sort(np.unique(r))
    weights = np.zeros(m, dtype=float)
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
    val = float(np.sum(w[mask] * np.log(w[mask] * m)))
    if abs(val) < 1e-12:
        return 0.0
    return max(0.0, val)


def estimate_coefficients(path: PathData, mask: np.ndarray, n_cal_prompts: int) -> PredictorCoefficients:
    j = path.gold_scores[mask]
    r = path.proxy_scores[mask]
    coeff = theory_coefficients(j, r)
    return PredictorCoefficients(
        a_exact=coeff.a_exact,
        b_theory=coeff.b_theory,
        a_nu=coeff.a_nu,
        n_cal_items=int(mask.sum()),
        n_cal_prompts=int(n_cal_prompts),
    )


def eval_prompt_bon(
    path: PathData,
    prompt_keys: np.ndarray,
    eval_prompts: set[str],
    n_best: int,
    min_candidates_per_prompt: int,
) -> dict[str, float]:
    gains: list[float] = []
    kls: list[float] = []
    sizes: list[int] = []
    for prompt in sorted(eval_prompts):
        idx = np.flatnonzero(prompt_keys == prompt)
        if len(idx) < min_candidates_per_prompt:
            continue
        w = bon_weights_with_replacement(path.proxy_scores[idx], n_best)
        j = path.gold_scores[idx]
        gains.append(float(np.sum(w * j) - np.mean(j)))
        kls.append(kl_to_uniform(w))
        sizes.append(int(len(idx)))
    if not gains:
        raise ValueError("No evaluation prompts survived min_candidates_per_prompt.")
    return {
        "actual_gain": float(np.mean(gains)),
        "mean_prompt_kl": float(np.mean(kls)),
        "median_prompt_kl": float(np.median(kls)),
        "mean_candidates_per_prompt": float(np.mean(sizes)),
        "median_candidates_per_prompt": float(np.median(sizes)),
        "n_eval_prompts_used": int(len(gains)),
        "share_eval_prompts_d_le_0p10": float(np.mean(np.asarray(kls) <= 0.10)),
        "share_eval_prompts_d_le_0p05": float(np.mean(np.asarray(kls) <= 0.05)),
    }


def predict_gain(coeff: PredictorCoefficients, d_values: np.ndarray, predictor: str) -> float:
    d = np.maximum(0.0, np.asarray(d_values, dtype=float))
    if predictor == "A+B":
        vals = coeff.a_exact * np.sqrt(d) + coeff.b_theory * d
    elif predictor == "A-only":
        vals = coeff.a_exact * np.sqrt(d)
    elif predictor == "A_nu":
        vals = coeff.a_nu * np.sqrt(d)
    elif predictor == "zero":
        vals = np.zeros_like(d)
    else:
        raise ValueError(f"Unknown predictor: {predictor}")
    return float(np.mean(vals))


def eval_prompt_kl_values(
    path: PathData,
    prompt_keys: np.ndarray,
    eval_prompts: set[str],
    n_best: int,
    min_candidates_per_prompt: int,
) -> np.ndarray:
    vals: list[float] = []
    for prompt in sorted(eval_prompts):
        idx = np.flatnonzero(prompt_keys == prompt)
        if len(idx) < min_candidates_per_prompt:
            continue
        vals.append(kl_to_uniform(bon_weights_with_replacement(path.proxy_scores[idx], n_best)))
    return np.asarray(vals, dtype=float)


def aggregate_proxy_if_needed(
    proxy_df: pd.DataFrame,
    item_col: str,
    score_col: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    cols = [*group_cols, item_col, score_col]
    missing = [c for c in cols if c not in proxy_df.columns]
    if missing:
        raise ValueError(f"proxy csv is missing required columns: {missing}")
    return (
        proxy_df[cols]
        .groupby([*group_cols, item_col], dropna=False)[score_col]
        .mean()
        .reset_index()
    )


def build_paths(args: argparse.Namespace) -> list[PathData]:
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    if args.proxy_csv:
        proxy = pd.read_csv(args.proxy_csv)
        proxy = aggregate_proxy_if_needed(proxy, args.proxy_item_col, args.proxy_score_col, group_cols)
        proxy_score_col = args.proxy_score_col
        if proxy_score_col != "proxy_score":
            proxy = proxy.rename(columns={proxy_score_col: "proxy_score"})
            proxy_score_col = "proxy_score"
        proxy_item_col = args.proxy_item_col
        if proxy_item_col != "item_id":
            proxy = proxy.rename(columns={proxy_item_col: "item_id"})
            proxy_item_col = "item_id"
        return list(
            generate_model_paths(
                human_long=human,
                proxy_df=proxy,
                item_cols=item_cols,
                score_col=args.score_col,
                group_cols=group_cols,
                proxy_item_col=proxy_item_col,
                proxy_score_col=proxy_score_col,
                min_items=args.min_items,
            )
        )
    return list(
        generate_human_paths(
            df=human,
            item_cols=item_cols,
            rater_col=args.rater_col,
            score_col=args.score_col,
            group_cols=group_cols,
            min_items=args.min_items,
            min_raters=args.min_raters,
        )
    )


def rows_to_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "proxy_mode", "n_best", "predictor"]
    for key, g in rows_df.groupby(group_cols, dropna=False):
        dataset, proxy_mode, n_best, predictor = key
        rows.append(
            {
                "dataset": dataset,
                "proxy_mode": proxy_mode,
                "n_best": int(n_best),
                "predictor": predictor,
                "n_rows": int(len(g)),
                "n_paths": int(g["path_id"].nunique()),
                "mean_actual_gain": float(g["actual_gain"].mean()),
                "mean_prompt_kl": float(g["mean_prompt_kl"].mean()),
                "mae": float(g["abs_error"].mean()),
                "median_abs_error": float(g["abs_error"].median()),
                "sign_accuracy": float((np.sign(g["predicted_gain"]) == np.sign(g["actual_gain"])).mean()),
                "mean_share_eval_prompts_d_le_0p10": float(g["share_eval_prompts_d_le_0p10"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "proxy_mode", "n_best", "predictor"])


def write_table_views(summary: pd.DataFrame, outdir: Path) -> None:
    summary.to_csv(outdir / "prompt_split_bon_reranking_summary.csv", index=False)
    (outdir / "prompt_split_bon_reranking_summary.md").write_text(
        summary.to_markdown(index=False), encoding="utf-8"
    )
    compact = summary.pivot_table(
        index=["dataset", "proxy_mode", "n_best"],
        columns="predictor",
        values="mae",
        aggfunc="mean",
    ).reset_index()
    compact.to_csv(outdir / "prompt_split_bon_reranking_mae_wide.csv", index=False)
    (outdir / "prompt_split_bon_reranking_mae_wide.md").write_text(
        compact.to_markdown(index=False), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    outdir = ensure_empty_outdir(Path(args.outdir), args.overwrite)
    item_cols = parse_cols(args.item_cols)
    prompt_col_indices = parse_nonnegative_int_list(args.prompt_col_indices)
    n_list = parse_int_list(args.n_list)
    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    paths = build_paths(args)
    if not paths:
        raise RuntimeError("No admissible paths were generated.")

    spec = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_name": args.dataset_name,
        "proxy_mode": "external_model_proxy" if args.proxy_csv else "heldout_human_proxy",
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv,
        "item_cols": item_cols,
        "prompt_col_indices": prompt_col_indices,
        "item_id_sep": args.item_id_sep,
        "group_cols": parse_cols(args.group_cols),
        "n_list": n_list,
        "predictors": predictors,
        "calibration_frac": args.calibration_frac,
        "n_repeats": args.n_repeats,
        "random_seed": args.random_seed,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "min_cal_prompts": args.min_cal_prompts,
        "min_eval_prompts": args.min_eval_prompts,
        "min_candidates_per_prompt": args.min_candidates_per_prompt,
        "n_paths": len(paths),
        "policy": "prompt-level exact with-replacement Best-of-N over existing candidate pools",
    }
    (outdir / "prompt_split_bon_reranking_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )

    all_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in paths:
        try:
            prompt_keys = prompt_keys_from_item_ids(path.item_ids, args.item_id_sep, prompt_col_indices)
        except Exception as exc:
            skipped.append({"path_id": path.path_id, "stage": "prompt_key", "reason": str(exc)})
            continue
        candidate_counts = pd.Series(prompt_keys).value_counts()
        prompt_keep = set(candidate_counts[candidate_counts >= args.min_candidates_per_prompt].index.astype(str))
        keep_mask = np.asarray([p in prompt_keep for p in prompt_keys], dtype=bool)
        if len(prompt_keep) < args.min_cal_prompts + args.min_eval_prompts:
            skipped.append(
                {
                    "path_id": path.path_id,
                    "stage": "prompt_filter",
                    "reason": f"only {len(prompt_keep)} prompts have >= {args.min_candidates_per_prompt} candidates",
                }
            )
            continue

        for repeat in range(args.n_repeats):
            rng = np.random.default_rng(stable_seed(path.path_id, repeat, base_seed=args.random_seed))
            try:
                cal_prompts, eval_prompts = split_prompt_keys(
                    prompt_keys[keep_mask],
                    args.calibration_frac,
                    rng,
                    args.min_cal_prompts,
                    args.min_eval_prompts,
                )
                cal_mask = np.asarray([p in cal_prompts for p in prompt_keys], dtype=bool)
                coeff = estimate_coefficients(path, cal_mask, len(cal_prompts))
            except Exception as exc:
                skipped.append({"path_id": path.path_id, "repeat": repeat, "stage": "calibration", "reason": str(exc)})
                continue

            for n_best in n_list:
                try:
                    eval_stats = eval_prompt_bon(
                        path,
                        prompt_keys,
                        eval_prompts,
                        n_best,
                        args.min_candidates_per_prompt,
                    )
                    eval_d = eval_prompt_kl_values(
                        path,
                        prompt_keys,
                        eval_prompts,
                        n_best,
                        args.min_candidates_per_prompt,
                    )
                except Exception as exc:
                    skipped.append(
                        {
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "n_best": n_best,
                            "stage": "evaluation",
                            "reason": str(exc),
                        }
                    )
                    continue
                for predictor in predictors:
                    pred = predict_gain(coeff, eval_d, predictor)
                    all_rows.append(
                        {
                            "dataset": args.dataset_name,
                            "proxy_mode": spec["proxy_mode"],
                            "group_key": path.group_key,
                            "path_id": path.path_id,
                            "repeat": repeat,
                            "n_best": n_best,
                            "predictor": predictor,
                            "predicted_gain": pred,
                            "actual_gain": eval_stats["actual_gain"],
                            "abs_error": abs(pred - eval_stats["actual_gain"]),
                            "mean_prompt_kl": eval_stats["mean_prompt_kl"],
                            "median_prompt_kl": eval_stats["median_prompt_kl"],
                            "n_cal_prompts": coeff.n_cal_prompts,
                            "n_cal_items": coeff.n_cal_items,
                            "n_eval_prompts_used": eval_stats["n_eval_prompts_used"],
                            "mean_candidates_per_prompt": eval_stats["mean_candidates_per_prompt"],
                            "median_candidates_per_prompt": eval_stats["median_candidates_per_prompt"],
                            "share_eval_prompts_d_le_0p10": eval_stats["share_eval_prompts_d_le_0p10"],
                            "share_eval_prompts_d_le_0p05": eval_stats["share_eval_prompts_d_le_0p05"],
                            **{f"cal_{k}": v for k, v in asdict(coeff).items()},
                        }
                    )

    rows_df = pd.DataFrame(all_rows)
    if rows_df.empty:
        pd.DataFrame(skipped).to_csv(outdir / "prompt_split_bon_reranking_skipped.csv", index=False)
        raise RuntimeError("No prediction rows were produced. See skipped CSV.")
    rows_df.to_csv(outdir / "prompt_split_bon_reranking_predictions.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(outdir / "prompt_split_bon_reranking_skipped.csv", index=False)
    else:
        pd.DataFrame(columns=["path_id", "repeat", "n_best", "stage", "reason"]).to_csv(
            outdir / "prompt_split_bon_reranking_skipped.csv", index=False
        )
    summary = rows_to_summary(rows_df)
    write_table_views(summary, outdir)
    print(summary.to_string(index=False))
    print(f"Saved prompt-split BoN reranking outputs in: {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt-split candidate-pool BoN/reranking experiment.")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--human-long-csv", required=True)
    ap.add_argument("--item-cols", required=True)
    ap.add_argument("--prompt-col-indices", default="0")
    ap.add_argument("--item-id-sep", default="||")
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--rater-col", default="rater_idx")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--proxy-csv", default="")
    ap.add_argument("--proxy-item-col", default="item_id")
    ap.add_argument("--proxy-score-col", default="proxy_score_mean")
    ap.add_argument("--min-items", type=int, default=200)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--min-candidates-per-prompt", type=int, default=2)
    ap.add_argument("--min-cal-prompts", type=int, default=10)
    ap.add_argument("--min-eval-prompts", type=int, default=10)
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--n-list", default="1,2,4,8,16,32")
    ap.add_argument("--predictors", default="A+B,A-only,A_nu,zero")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
