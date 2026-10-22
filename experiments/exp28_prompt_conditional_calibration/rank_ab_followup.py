#!/usr/bin/env python3
"""Compare raw-score and within-prompt-rank A+B predictions for DPO.

Raw A+B describes the local frontier induced by a raw-score exponential tilt.
Rank A+B instead describes a within-prompt rank-score tilt. The two frontiers
are therefore evaluated separately before either prediction is compared with
realized DPO gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "lib"))

from common_experiment_utils import theory_coefficients  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tilt_weights(score: np.ndarray, t: float) -> np.ndarray:
    logits = t * (score - np.max(score))
    weights = np.exp(logits)
    return weights / np.sum(weights)


def kl_at_t(score: np.ndarray, t: float) -> float:
    weights = tilt_weights(score, t)
    return float(
        np.log(len(score))
        + np.sum(weights * np.log(np.clip(weights, 1e-300, None)))
    )


def find_t(score: np.ndarray, d: float) -> float:
    if d <= 0.0:
        return 0.0
    if np.var(score) == 0.0:
        raise ValueError("cannot tilt a constant score")
    max_kl = math.log(len(score))
    if d >= max_kl:
        raise ValueError(f"d={d} is outside the finite-pool KL range")
    lo, hi = 0.0, 1.0
    while kl_at_t(score, hi) < d:
        lo, hi = hi, 2.0 * hi
        if hi > 1e6:
            raise ValueError(f"could not bracket d={d}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_at_t(score, mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def normalized_rank(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    return (rankdata(score, method="average") - 0.5) / len(score)


def load_pool(path: Path):
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] not in {"calibration", "eval"}:
                continue
            key = (row["split"], row["prompt_id"], int(row["sample_idx"]))
            if key in seen:
                raise ValueError(f"duplicate candidate key: {key}")
            seen.add(key)
            values = (float(row["proxy_score"]), float(row["gold_score"]))
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite score at {key}")
            groups[(row["split"], row["prompt_id"])].append(row)

    for split, expected_prompts in (("calibration", 100), ("eval", 200)):
        subset = {
            prompt_id: rows
            for (candidate_split, prompt_id), rows in groups.items()
            if candidate_split == split
        }
        if len(subset) != expected_prompts:
            raise ValueError(
                f"{split}: expected {expected_prompts} prompts, got {len(subset)}"
            )
        if {len(rows) for rows in subset.values()} != {32}:
            raise ValueError(f"{split}: expected exactly 32 candidates per prompt")
    return groups


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: int(row["sample_idx"]))
    return (
        np.asarray([row["proxy_score"] for row in ordered], dtype=float),
        np.asarray([row["gold_score"] for row in ordered], dtype=float),
    )


def fit_coefficients(groups, mode: str):
    rows = []
    skipped = []
    for (split, prompt_id), candidates in sorted(groups.items()):
        if split != "calibration":
            continue
        proxy, gold = arrays(candidates)
        if np.var(proxy) == 0.0 or np.var(gold) == 0.0:
            skipped.append(
                {
                    "prompt_id": prompt_id,
                    "mode": mode,
                    "reason": (
                        "zero proxy variance"
                        if np.var(proxy) == 0.0
                        else "zero gold variance"
                    ),
                }
            )
            continue
        direction = proxy if mode == "raw" else normalized_rank(proxy)
        coefficients = theory_coefficients(gold, direction)
        rows.append(
            {
                "prompt_id": prompt_id,
                "mode": mode,
                "A": coefficients.a_exact,
                "B": coefficients.b_theory,
                "n_candidates": len(direction),
            }
        )
    frame = pd.DataFrame(rows)
    skipped_frame = pd.DataFrame(skipped)
    if len(frame) + len(skipped_frame) != 100:
        raise ValueError(f"{mode}: calibration accounting does not sum to 100")
    return (
        frame,
        skipped_frame,
        float(frame["A"].mean()),
        float(frame["B"].mean()),
    )


def evaluate(
    groups,
    multiseed_root: Path,
    seeds: list[int],
    raw_a: float,
    raw_b: float,
    rank_a: float,
    rank_b: float,
):
    eval_pool = {
        prompt_id: arrays(rows)
        for (split, prompt_id), rows in groups.items()
        if split == "eval"
    }
    records = []
    for seed in seeds:
        per_path = (
            multiseed_root / f"seed_{seed}" / "exp09_final_per_prompt.csv"
        )
        per = pd.read_csv(per_path)
        required = {
            "tag",
            "prompt_id",
            "d",
            "ess",
            "frontier",
            "dpo_reweight_gold",
            "pred_AB",
        }
        missing = required - set(per.columns)
        if missing:
            raise ValueError(f"seed {seed}: missing columns {sorted(missing)}")
        if per.duplicated(["tag", "prompt_id"]).any():
            raise ValueError(f"seed {seed}: duplicate checkpoint/prompt rows")

        local_tags = (
            per.groupby("tag")
            .agg(d_med=("d", "median"), ess_med=("ess", "median"))
            .query("d_med <= 1.0 and ess_med >= 8.0")
            .index
        )
        per = per[per["tag"].isin(local_tags)].copy()
        if len(local_tags) != 6 or len(per) != 1200:
            raise ValueError(f"seed {seed}: expected 6 x 200 local rows")

        for row in per.itertuples():
            proxy, gold = eval_pool[row.prompt_id]
            rank_score = normalized_rank(proxy)
            d = max(float(row.d), 0.0)

            raw_weights = tilt_weights(proxy, find_t(proxy, d))
            rank_weights = tilt_weights(rank_score, find_t(rank_score, d))
            raw_frontier = float(np.sum(raw_weights * gold) - np.mean(gold))
            rank_frontier = float(np.sum(rank_weights * gold) - np.mean(gold))
            if not math.isclose(
                raw_frontier,
                float(row.frontier),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"seed {seed}, {row.tag}, {row.prompt_id}: "
                    "raw frontier mismatch"
                )

            raw_prediction = raw_a * math.sqrt(d) + raw_b * d
            if not math.isclose(
                raw_prediction,
                float(row.pred_AB),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"seed {seed}, {row.tag}, {row.prompt_id}: "
                    "raw A+B prediction mismatch"
                )
            records.append(
                {
                    "seed": seed,
                    "tag": row.tag,
                    "prompt_id": row.prompt_id,
                    "d": d,
                    "ess": float(row.ess),
                    "raw_frontier": raw_frontier,
                    "rank_frontier": rank_frontier,
                    "dpo_realized": float(row.dpo_reweight_gold),
                    "raw_ab": raw_prediction,
                    "rank_ab": rank_a * math.sqrt(d) + rank_b * d,
                }
            )
    return pd.DataFrame(records)


def summarize(per_prompt: pd.DataFrame):
    checkpoint = (
        per_prompt.groupby(["seed", "tag"], as_index=False)
        .agg(
            d_med=("d", "median"),
            ess_med=("ess", "median"),
            raw_frontier=("raw_frontier", "mean"),
            rank_frontier=("rank_frontier", "mean"),
            dpo_realized=("dpo_realized", "mean"),
            raw_ab=("raw_ab", "mean"),
            rank_ab=("rank_ab", "mean"),
            n_prompts=("prompt_id", "nunique"),
        )
    )
    if not (checkpoint["n_prompts"] == 200).all():
        raise ValueError("a checkpoint is missing evaluation prompts")

    metric_rows = []
    definitions = (
        ("Raw A+B", "raw_ab", "raw_frontier"),
        ("Rank A+B", "rank_ab", "rank_frontier"),
        ("Rank A+B vs raw frontier", "rank_ab", "raw_frontier"),
    )
    for seed, seed_frame in checkpoint.groupby("seed"):
        for model, prediction, frontier in definitions:
            metric_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "frontier_target": frontier,
                    "mae_matching_or_named_frontier": float(
                        np.mean(
                            np.abs(
                                seed_frame[prediction] - seed_frame[frontier]
                            )
                        )
                    ),
                    "mae_realized_dpo": float(
                        np.mean(
                            np.abs(
                                seed_frame[prediction]
                                - seed_frame["dpo_realized"]
                            )
                        )
                    ),
                }
            )
    seed_metrics = pd.DataFrame(metric_rows)
    summary = (
        seed_metrics.groupby(["model", "frontier_target"], as_index=False)
        .agg(
            mean_mae_frontier=(
                "mae_matching_or_named_frontier",
                "mean",
            ),
            sd_mae_frontier=(
                "mae_matching_or_named_frontier",
                "std",
            ),
            mean_mae_dpo=("mae_realized_dpo", "mean"),
            sd_mae_dpo=("mae_realized_dpo", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_mae_dpo")
    )

    window_rows = []
    for scope, max_d in (
        ("strict_local_d_le_0p10", 0.10),
        ("historical_six_checkpoints", np.inf),
    ):
        scoped = checkpoint[checkpoint["d_med"] <= max_d]
        for seed, seed_frame in scoped.groupby("seed"):
            for model, prediction, frontier in definitions:
                window_rows.append(
                    {
                        "scope": scope,
                        "seed": seed,
                        "model": model,
                        "frontier_target": frontier,
                        "n_checkpoints": len(seed_frame),
                        "mae_frontier": float(
                            np.mean(
                                np.abs(
                                    seed_frame[prediction]
                                    - seed_frame[frontier]
                                )
                            )
                        ),
                        "mae_dpo": float(
                            np.mean(
                                np.abs(
                                    seed_frame[prediction]
                                    - seed_frame["dpo_realized"]
                                )
                            )
                        ),
                    }
                )
    window_seed = pd.DataFrame(window_rows)
    window_summary = (
        window_seed.groupby(
            ["scope", "model", "frontier_target"], as_index=False
        )
        .agg(
            mean_mae_frontier=("mae_frontier", "mean"),
            sd_mae_frontier=("mae_frontier", "std"),
            mean_mae_dpo=("mae_dpo", "mean"),
            sd_mae_dpo=("mae_dpo", "std"),
            min_checkpoints_per_seed=("n_checkpoints", "min"),
            max_checkpoints_per_seed=("n_checkpoints", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["scope", "mean_mae_dpo"])
    )
    return (
        checkpoint,
        seed_metrics,
        summary,
        window_seed,
        window_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-scored", type=Path, required=True)
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups = load_pool(args.pool_scored)
    raw_per_prompt, raw_skipped, raw_a, raw_b = fit_coefficients(groups, "raw")
    rank_per_prompt, rank_skipped, rank_a, rank_b = fit_coefficients(
        groups, "rank"
    )
    if set(raw_per_prompt["prompt_id"]) != set(rank_per_prompt["prompt_id"]):
        raise ValueError("raw and rank coefficients use different prompts")

    for seed in args.seeds:
        coefficient_path = (
            args.multiseed_root / f"seed_{seed}" / "coefficients.json"
        )
        locked = json.loads(coefficient_path.read_text(encoding="utf-8"))
        if not (
            math.isclose(raw_a, float(locked["A"]), abs_tol=1e-12)
            and math.isclose(raw_b, float(locked["B"]), abs_tol=1e-12)
        ):
            raise ValueError(f"seed {seed}: raw coefficients do not match lock")

    per_prompt = evaluate(
        groups,
        args.multiseed_root,
        args.seeds,
        raw_a,
        raw_b,
        rank_a,
        rank_b,
    )
    (
        checkpoint,
        seed_metrics,
        summary,
        window_seed,
        window_summary,
    ) = summarize(per_prompt)

    pd.concat([raw_per_prompt, rank_per_prompt], ignore_index=True).to_csv(
        args.outdir / "calibration_coefficients_by_prompt.csv",
        index=False,
    )
    pd.concat([raw_skipped, rank_skipped], ignore_index=True).to_csv(
        args.outdir / "calibration_skipped_prompts.csv",
        index=False,
    )
    per_prompt.to_csv(args.outdir / "per_prompt_predictions.csv", index=False)
    checkpoint.to_csv(args.outdir / "checkpoint_predictions.csv", index=False)
    seed_metrics.to_csv(args.outdir / "seed_metrics.csv", index=False)
    summary.to_csv(args.outdir / "summary_mean_sd.csv", index=False)
    window_seed.to_csv(args.outdir / "window_seed_metrics.csv", index=False)
    window_summary.to_csv(
        args.outdir / "window_summary_mean_sd.csv",
        index=False,
    )
    (args.outdir / "run_spec.json").write_text(
        json.dumps(
            {
                "description": (
                    "Raw A+B is evaluated against the raw-score tilt frontier; "
                    "rank A+B is evaluated against a within-prompt rank-score "
                    "tilt frontier. Both are also compared with realized DPO."
                ),
                "rank_coordinate": "(average_rank - 0.5) / candidates_per_prompt",
                "seeds": args.seeds,
                "historical_scope": "six checkpoints with median d <= 1 and ESS >= 8",
                "coefficients": {
                    "raw": {"A": raw_a, "B": raw_b},
                    "rank": {"A": rank_a, "B": rank_b},
                },
                "calibration_accounting": {
                    "used_prompts": len(raw_per_prompt),
                    "skipped_prompts": len(raw_skipped),
                },
                "input_sha256": {
                    "pool_scored": sha256_file(args.pool_scored),
                    **{
                        f"seed_{seed}_per_prompt": sha256_file(
                            args.multiseed_root
                            / f"seed_{seed}"
                            / "exp09_final_per_prompt.csv"
                        )
                        for seed in args.seeds
                    },
                },
                "code_sha256": sha256_file(Path(__file__).resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"raw coefficients:  A={raw_a:.8f}, B={raw_b:.8f}")
    print(f"rank coefficients: A={rank_a:.8f}, B={rank_b:.8f}")
    print(summary.to_string(index=False))
    print("\nWindow summaries:")
    print(window_summary.to_string(index=False))
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
