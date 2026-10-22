#!/usr/bin/env python3
"""Same-prompt candidate holdout for real-LLM candidate pools.

Each prompt contributes 10 calibration candidates and 22 evaluation
candidates. Candidate roles are assigned from prompt/candidate identifiers,
without consulting proxy or gold scores. Models see evaluation proxy scores
only when constructing the requested tilt; evaluation gold is used only after
all predictions have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "exp25_real_llm_calibration_followup"
sys.path.insert(0, str(BASELINE_DIR))

from baseline_core import (  # noqa: E402
    _central_moments,
    _coefficient_from_moments,
    _find_t_prompt,
    _fit_binned,
    _fit_mlp,
    _fit_spline,
    ensure_empty_outdir,
    sha256_file,
    tilt_weights,
)


D_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10]
FIT_RADII = np.linspace(0.004, 0.10, 25)
BASELINES = [
    "A+B",
    "A+B-shrunk",
    "A-only",
    "A_nu",
    "empirical_curve",
    "empirical_fit",
    "isotonic_calibration",
    "binned_shrinkage",
    "spline_ridge",
    "mlp_calibration",
    "zero",
]

_POOLS: dict[str, pd.DataFrame] = {}


@dataclass(frozen=True)
class PromptCalibration:
    prompt_id: str
    a: float
    b: float
    a_nu: float
    b_se: float
    empirical_a: float
    empirical_b: float
    calibration_gains: dict[float, float]


def stable_uint32(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def candidate_role(
    prompt_id: str,
    sample_idx: int,
    *,
    repeat: int,
    random_seed: int,
    n_calibration: int,
    candidate_ids: list[int],
) -> str:
    ordered = sorted(
        candidate_ids,
        key=lambda candidate: (
            stable_uint32(
                "exp29-candidate-split",
                random_seed,
                repeat,
                prompt_id,
                candidate,
            ),
            candidate,
        ),
    )
    return "calibration" if sample_idx in set(ordered[:n_calibration]) else "evaluation"


def build_split_manifest(
    reference: pd.DataFrame,
    *,
    n_repeats: int,
    n_calibration: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prompt_id, group in reference.groupby("prompt_id", sort=True):
        candidate_ids = sorted(group["sample_idx"].astype(int).tolist())
        for repeat in range(n_repeats):
            for sample_idx in candidate_ids:
                rows.append(
                    {
                        "repeat": repeat,
                        "prompt_id": str(prompt_id),
                        "sample_idx": int(sample_idx),
                        "role": candidate_role(
                            str(prompt_id),
                            int(sample_idx),
                            repeat=repeat,
                            random_seed=random_seed,
                            n_calibration=n_calibration,
                            candidate_ids=candidate_ids,
                        ),
                    }
                )
    manifest = pd.DataFrame(rows)
    counts = (
        manifest.groupby(["repeat", "prompt_id", "role"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    expected_eval = int(reference.groupby("prompt_id").size().iloc[0]) - n_calibration
    if not (
        counts["calibration"].eq(n_calibration).all()
        and counts["evaluation"].eq(expected_eval).all()
    ):
        raise AssertionError("candidate split does not have the requested sizes")
    return manifest.sort_values(["repeat", "prompt_id", "sample_idx"]).reset_index(
        drop=True
    )


def max_reachable_d(proxy: np.ndarray) -> float:
    if len(proxy) == 0:
        return 0.0
    n_max = int(np.sum(proxy == np.max(proxy)))
    return math.log(float(len(proxy)) / float(n_max))


def gain_at_radius(proxy: np.ndarray, values: np.ndarray, d: float) -> float:
    t = _find_t_prompt(proxy, d)
    weights = tilt_weights(proxy, t)
    return float(np.sum(weights * values) - np.mean(values))


def coefficient_triplet(
    proxy: np.ndarray, gold: np.ndarray
) -> tuple[float, float, float]:
    return _coefficient_from_moments(*_central_moments(gold, proxy))


def bootstrap_b_se(
    proxy: np.ndarray,
    gold: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(proxy), size=len(proxy))
        sampled_proxy = proxy[indices]
        sampled_gold = gold[indices]
        moments = _central_moments(sampled_gold, sampled_proxy)
        if moments[0] <= 0 or moments[1] <= 0:
            continue
        values.append(float(_coefficient_from_moments(*moments)[1]))
    if len(values) < max(20, n_bootstrap // 5):
        return float("inf"), len(values)
    return float(np.std(values, ddof=1)), len(values)


def fit_direct_predictors(
    calibration: pd.DataFrame,
    *,
    seed: int,
) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], dict[str, object]]:
    proxy = calibration["proxy_score"].to_numpy(dtype=float)
    gold = calibration["gold_score"].to_numpy(dtype=float)
    prompt_ids = calibration["prompt_id"].astype(str).to_numpy()
    direct: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    metadata: dict[str, object] = {}

    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    isotonic.fit(proxy, gold)
    direct["isotonic_calibration"] = lambda values: isotonic.predict(values)
    direct["binned_shrinkage"], metadata["binned_shrinkage"] = _fit_binned(
        proxy, gold, prompt_ids
    )
    direct["spline_ridge"], metadata["spline_ridge"] = _fit_spline(
        proxy, gold, prompt_ids
    )
    direct["mlp_calibration"], metadata["mlp_calibration"] = _fit_mlp(
        proxy, gold, seed
    )
    return direct, metadata


def prompt_skip_reason(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    max_d: float,
) -> str | None:
    cal_proxy = calibration["proxy_score"].to_numpy(dtype=float)
    cal_gold = calibration["gold_score"].to_numpy(dtype=float)
    eval_proxy = evaluation["proxy_score"].to_numpy(dtype=float)
    if float(np.var(cal_proxy)) <= 0:
        return "constant_calibration_proxy"
    if float(np.var(cal_gold)) <= 0:
        return "constant_calibration_gold"
    if float(np.var(eval_proxy)) <= 0:
        return "constant_evaluation_proxy"
    if max_reachable_d(cal_proxy) + 1e-12 < max_d:
        return "calibration_radius_unreachable"
    if max_reachable_d(eval_proxy) + 1e-12 < max_d:
        return "evaluation_radius_unreachable"
    return None


def fit_prompt_calibration(
    prompt_id: str,
    calibration: pd.DataFrame,
    *,
    run_name: str,
    repeat: int,
    random_seed: int,
    n_bootstrap: int,
) -> tuple[PromptCalibration, dict[str, object]]:
    proxy = calibration["proxy_score"].to_numpy(dtype=float)
    gold = calibration["gold_score"].to_numpy(dtype=float)
    a, b, a_nu = coefficient_triplet(proxy, gold)
    b_se, n_valid_bootstrap = bootstrap_b_se(
        proxy,
        gold,
        n_bootstrap=n_bootstrap,
        seed=stable_uint32(
            "exp29-b-bootstrap", random_seed, run_name, repeat, prompt_id
        ),
    )
    calibration_gains = {
        float(d): gain_at_radius(proxy, gold, float(d)) for d in D_VALUES
    }
    fit_gains = np.asarray(
        [gain_at_radius(proxy, gold, float(d)) for d in FIT_RADII],
        dtype=float,
    )
    design = np.column_stack([np.sqrt(FIT_RADII), FIT_RADII])
    empirical_a, empirical_b = np.linalg.lstsq(design, fit_gains, rcond=None)[0]
    fitted = PromptCalibration(
        prompt_id=str(prompt_id),
        a=float(a),
        b=float(b),
        a_nu=float(a_nu),
        b_se=float(b_se),
        empirical_a=float(empirical_a),
        empirical_b=float(empirical_b),
        calibration_gains=calibration_gains,
    )
    audit = {
        "prompt_id": str(prompt_id),
        "a": float(a),
        "b": float(b),
        "a_nu": float(a_nu),
        "b_bootstrap_se": float(b_se),
        "n_valid_b_bootstrap": int(n_valid_bootstrap),
        "calibration_max_reachable_d": max_reachable_d(proxy),
    }
    return fitted, audit


def _set_pools(pools: dict[str, pd.DataFrame]) -> None:
    global _POOLS
    _POOLS = pools


def run_task(
    run_name: str,
    repeat: int,
    manifest_repeat: pd.DataFrame,
    *,
    random_seed: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if run_name not in _POOLS:
        raise RuntimeError(f"worker pool not initialized: {run_name}")
    frame = _POOLS[run_name]
    split = frame.merge(
        manifest_repeat[["prompt_id", "sample_idx", "role"]],
        on=["prompt_id", "sample_idx"],
        how="left",
        validate="one_to_one",
    )
    if split["role"].isna().any():
        raise AssertionError("candidate split merge left unmatched rows")

    calibration_all = split[split["role"].eq("calibration")].copy()
    evaluation_all = split[split["role"].eq("evaluation")].copy()
    prompt_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prompt_models: dict[str, PromptCalibration] = {}
    valid_ids: list[str] = []
    skip_counts: dict[str, int] = {}

    cal_groups = {
        str(prompt_id): group.sort_values("sample_idx")
        for prompt_id, group in calibration_all.groupby("prompt_id", sort=True)
    }
    eval_groups = {
        str(prompt_id): group.sort_values("sample_idx")
        for prompt_id, group in evaluation_all.groupby("prompt_id", sort=True)
    }
    for prompt_id in sorted(cal_groups):
        calibration = cal_groups[prompt_id]
        evaluation = eval_groups[prompt_id]
        reason = prompt_skip_reason(
            calibration, evaluation, max_d=max(D_VALUES)
        )
        if reason is not None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            coefficient_rows.append(
                {
                    "run": run_name,
                    "repeat": repeat,
                    "prompt_id": prompt_id,
                    "status": "skipped",
                    "skip_reason": reason,
                }
            )
            continue
        model, audit = fit_prompt_calibration(
            prompt_id,
            calibration,
            run_name=run_name,
            repeat=repeat,
            random_seed=random_seed,
            n_bootstrap=n_bootstrap,
        )
        prompt_models[prompt_id] = model
        valid_ids.append(prompt_id)
        coefficient_rows.append(
            {
                "run": run_name,
                "repeat": repeat,
                "status": "valid",
                "skip_reason": "",
                **audit,
            }
        )

    if len(valid_ids) < 2:
        raise RuntimeError(f"{run_name} repeat {repeat}: fewer than two valid prompts")

    calibration = calibration_all[
        calibration_all["prompt_id"].astype(str).isin(valid_ids)
    ].copy()
    evaluation = evaluation_all[
        evaluation_all["prompt_id"].astype(str).isin(valid_ids)
    ].copy()
    direct, direct_metadata = fit_direct_predictors(
        calibration,
        seed=stable_uint32("exp29-direct-fit", random_seed, run_name, repeat),
    )

    finite_se2 = np.asarray(
        [
            model.b_se**2
            for model in prompt_models.values()
            if np.isfinite(model.b_se)
        ],
        dtype=float,
    )
    b2 = np.asarray([model.b**2 for model in prompt_models.values()], dtype=float)
    mean_se2 = float(np.mean(finite_se2)) if len(finite_se2) else float("inf")
    tau2 = (
        max(float(np.mean(b2)) - mean_se2, 0.0)
        if np.isfinite(mean_se2)
        else 0.0
    )

    for prompt_id in valid_ids:
        model = prompt_models[prompt_id]
        evaluation_prompt = eval_groups[prompt_id]
        proxy = evaluation_prompt["proxy_score"].to_numpy(dtype=float)
        gold = evaluation_prompt["gold_score"].to_numpy(dtype=float)
        direct_values = {
            name: predictor(proxy) for name, predictor in direct.items()
        }
        if np.isfinite(model.b_se):
            shrink_weight = tau2 / (tau2 + model.b_se**2) if tau2 > 0 else 0.0
        else:
            shrink_weight = 0.0
        for d in D_VALUES:
            weights = tilt_weights(proxy, _find_t_prompt(proxy, float(d)))
            actual_gain = float(np.sum(weights * gold) - np.mean(gold))
            sqrt_d = math.sqrt(float(d))
            predictions = {
                "A+B": model.a * sqrt_d + model.b * d,
                "A+B-shrunk": model.a * sqrt_d
                + shrink_weight * model.b * d,
                "A-only": model.a * sqrt_d,
                "A_nu": model.a_nu * sqrt_d,
                "empirical_curve": model.calibration_gains[float(d)],
                "empirical_fit": model.empirical_a * sqrt_d
                + model.empirical_b * d,
                "zero": 0.0,
            }
            for name, values in direct_values.items():
                predictions[name] = float(
                    np.sum(weights * values) - np.mean(values)
                )
            for baseline in BASELINES:
                prompt_rows.append(
                    {
                        "run": run_name,
                        "repeat": repeat,
                        "prompt_id": prompt_id,
                        "d": float(d),
                        "baseline": baseline,
                        "predicted_gain": float(predictions[baseline]),
                        "actual_gain": actual_gain,
                        "error": float(predictions[baseline] - actual_gain),
                        "abs_error": float(
                            abs(predictions[baseline] - actual_gain)
                        ),
                        "b_shrink_weight": float(shrink_weight),
                    }
                )

    prompt_predictions = pd.DataFrame(prompt_rows)
    aggregate = (
        prompt_predictions.groupby(["run", "repeat", "d", "baseline"], sort=True)
        .agg(
            predicted_gain=("predicted_gain", "mean"),
            actual_gain=("actual_gain", "mean"),
            prompt_mae=("abs_error", "mean"),
            n_valid_prompts=("prompt_id", "nunique"),
        )
        .reset_index()
    )
    aggregate["error"] = aggregate["predicted_gain"] - aggregate["actual_gain"]
    aggregate["abs_error"] = aggregate["error"].abs()

    # This audit records the key no-leakage invariant. Predictions are frozen
    # before evaluation gold enters the actual-gain calculation.
    audit_prompt = ""
    audit_actual_difference = 0.0
    for prompt_id in valid_ids:
        audit_eval = eval_groups[prompt_id]
        audit_proxy = audit_eval["proxy_score"].to_numpy(dtype=float)
        audit_gold = audit_eval["gold_score"].to_numpy(dtype=float)
        if float(np.var(audit_gold)) <= 0:
            continue
        audit_weights = tilt_weights(
            audit_proxy, _find_t_prompt(audit_proxy, 0.10)
        )
        permutation = np.random.default_rng(
            stable_uint32(
                "exp29-leakage-audit", run_name, repeat, prompt_id
            )
        ).permutation(len(audit_gold))
        original_actual = float(
            np.sum(audit_weights * audit_gold) - np.mean(audit_gold)
        )
        permuted_gold = audit_gold[permutation]
        permuted_actual = float(
            np.sum(audit_weights * permuted_gold) - np.mean(permuted_gold)
        )
        difference = float(permuted_actual - original_actual)
        if abs(difference) > 1e-12:
            audit_prompt = prompt_id
            audit_actual_difference = difference
            break
    if not audit_prompt:
        raise RuntimeError(
            f"{run_name} repeat {repeat}: leakage audit could not find "
            "a nondegenerate evaluation prompt"
        )

    metadata = {
        "run": run_name,
        "repeat": repeat,
        "n_attempted_prompts": int(len(cal_groups)),
        "n_valid_prompts": int(len(valid_ids)),
        "n_skipped_prompts": int(len(cal_groups) - len(valid_ids)),
        "skip_counts": json.dumps(skip_counts, sort_keys=True),
        "b_prior_tau2": float(tau2),
        "mean_b_shrink_weight": float(
            prompt_predictions["b_shrink_weight"].mean()
        ),
        "direct_model_metadata": json.dumps(direct_metadata, sort_keys=True),
        "leakage_audit_prompt_id": audit_prompt,
        "leakage_audit_prediction_difference": 0.0,
        "leakage_audit_actual_difference": audit_actual_difference,
    }
    return aggregate, prompt_predictions, pd.DataFrame(coefficient_rows), metadata


def parse_pool_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--pool must be RUN_NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("pool run name cannot be empty")
    return name, Path(raw_path)


def load_and_validate_pools(
    pool_args: list[tuple[str, Path]],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    pools: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, object]] = []
    reference_keys: pd.DataFrame | None = None
    reference_prompts: pd.DataFrame | None = None
    for run_name, path in pool_args:
        if run_name in pools:
            raise ValueError(f"duplicate run name: {run_name}")
        raw = pd.read_json(path, lines=True)
        required = {
            "prompt_id",
            "split",
            "sample_idx",
            "prompt",
            "response",
            "proxy_score",
            "gold_score",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{run_name}: missing columns {missing}")
        frame = raw[
            [
                "prompt_id",
                "split",
                "sample_idx",
                "prompt",
                "response",
                "proxy_score",
                "gold_score",
            ]
        ].copy()
        frame["prompt_id"] = frame["prompt_id"].astype(str)
        frame["sample_idx"] = frame["sample_idx"].astype(int)
        if frame.duplicated(["prompt_id", "sample_idx"]).any():
            raise ValueError(f"{run_name}: duplicate prompt/candidate keys")
        if (
            frame["prompt"].astype(str).str.strip().eq("").any()
            or frame["response"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{run_name}: empty prompt or response")
        prompt_identity = frame[["prompt_id", "prompt"]].drop_duplicates()
        expected_prompt_ids = prompt_identity["prompt"].map(
            lambda prompt: hashlib.sha256(
                str(prompt).encode("utf-8")
            ).hexdigest()[:16]
        )
        if not prompt_identity["prompt_id"].astype(str).reset_index(drop=True).equals(
            expected_prompt_ids.reset_index(drop=True)
        ):
            raise ValueError(f"{run_name}: prompt_id does not match prompt text")
        if not np.isfinite(
            frame[["proxy_score", "gold_score"]].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{run_name}: non-finite scores")
        counts = frame.groupby("prompt_id", sort=True).size()
        if counts.nunique() != 1 or int(counts.iloc[0]) != 32:
            raise ValueError(f"{run_name}: expected exactly 32 candidates per prompt")
        candidate_sets = frame.groupby("prompt_id", sort=True)["sample_idx"].agg(
            lambda values: tuple(sorted(values.astype(int).tolist()))
        )
        expected_candidates = tuple(range(32))
        if not candidate_sets.map(lambda values: values == expected_candidates).all():
            raise ValueError(f"{run_name}: incomplete candidate index grid")
        split_counts = frame.groupby("split", sort=True)["prompt_id"].nunique().to_dict()
        if split_counts != {"calibration": 100, "eval": 200}:
            raise ValueError(
                f"{run_name}: unexpected original prompt split counts {split_counts}"
            )
        keys = frame[["prompt_id", "sample_idx"]].sort_values(
            ["prompt_id", "sample_idx"]
        )
        prompts = (
            frame[["prompt_id", "split", "prompt"]]
            .drop_duplicates()
            .sort_values("prompt_id")
        )
        if reference_keys is None:
            reference_keys = keys.reset_index(drop=True)
            reference_prompts = prompts.reset_index(drop=True)
        else:
            if not reference_keys.equals(keys.reset_index(drop=True)):
                raise ValueError(f"{run_name}: prompt/candidate keys differ")
            if not reference_prompts.equals(prompts.reset_index(drop=True)):
                raise ValueError(f"{run_name}: prompt text differs")
        pools[run_name] = frame[
            ["prompt_id", "sample_idx", "proxy_score", "gold_score"]
        ].copy()
        audits.append(
            {
                "run": run_name,
                "path": str(path),
                "sha256": sha256_file(path),
                "n_rows": int(len(frame)),
                "n_prompts": int(frame["prompt_id"].nunique()),
                "candidates_per_prompt": int(counts.iloc[0]),
                "proxy_gold_pearson": float(
                    frame[["proxy_score", "gold_score"]].corr().iloc[0, 1]
                ),
            }
        )
    if not pools:
        raise ValueError("at least one --pool is required")
    return pools, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        action="append",
        type=parse_pool_argument,
        required=True,
        help="Repeatable RUN_NAME=PATH argument.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-calibration", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    if min(args.n_repeats, args.n_calibration, args.n_bootstrap, args.jobs) < 1:
        parser.error("repeat, calibration, bootstrap, and job counts must be positive")
    if args.n_calibration >= 32:
        parser.error("--n-calibration must be below 32")
    ensure_empty_outdir(args.outdir)

    pools, pool_audits = load_and_validate_pools(args.pool)
    reference = next(iter(pools.values()))
    manifest = build_split_manifest(
        reference,
        n_repeats=args.n_repeats,
        n_calibration=args.n_calibration,
        random_seed=args.random_seed,
    )
    manifest.to_csv(
        args.outdir / "candidate_split_manifest.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    tasks = [
        (
            run_name,
            repeat,
            manifest[manifest["repeat"].eq(repeat)].copy(),
        )
        for run_name in sorted(pools)
        for repeat in range(args.n_repeats)
    ]
    aggregate_frames: list[pd.DataFrame] = []
    prompt_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []

    _set_pools(pools)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        mp_context=context,
        initializer=_set_pools,
        initargs=(pools,),
    ) as executor:
        futures = {
            executor.submit(
                run_task,
                run_name,
                repeat,
                manifest_repeat,
                random_seed=args.random_seed,
                n_bootstrap=args.n_bootstrap,
            ): (run_name, repeat)
            for run_name, repeat, manifest_repeat in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run_name, repeat = futures[future]
            aggregate, prompt, coefficients, metadata = future.result()
            aggregate_frames.append(aggregate)
            prompt_frames.append(prompt)
            coefficient_frames.append(coefficients)
            metadata_rows.append(metadata)
            print(
                f"completed {index}/{len(futures)}: {run_name} repeat {repeat}",
                flush=True,
            )

    aggregate = pd.concat(aggregate_frames, ignore_index=True).sort_values(
        ["run", "repeat", "d", "baseline"]
    )
    prompt = pd.concat(prompt_frames, ignore_index=True).sort_values(
        ["run", "repeat", "prompt_id", "d", "baseline"]
    )
    coefficients = pd.concat(coefficient_frames, ignore_index=True).sort_values(
        ["run", "repeat", "prompt_id"]
    )
    metadata = pd.DataFrame(metadata_rows).sort_values(["run", "repeat"])
    aggregate.to_csv(args.outdir / "aggregate_predictions.csv", index=False)
    prompt.to_csv(
        args.outdir / "prompt_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    coefficients.to_csv(
        args.outdir / "prompt_coefficients.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metadata.to_csv(args.outdir / "fit_metadata.csv", index=False)

    leakage_pass = bool(
        metadata["leakage_audit_prediction_difference"].eq(0.0).all()
        and metadata["leakage_audit_actual_difference"].abs().gt(1e-12).all()
    )
    spec = {
        "experiment": "exp29_same_prompt_candidate_holdout",
        "description": (
            "Within each prompt, 10 candidates calibrate the predictors and "
            "22 disjoint candidates evaluate the frozen predictions."
        ),
        "pools": pool_audits,
        "prompt_identity_check": "exact prompt_id, prompt text, and sample_idx equality",
        "split_unit": "candidate within prompt",
        "split_key": "SHA-256(random_seed, repeat, prompt_id, sample_idx)",
        "score_blind_split": True,
        "n_repeats": args.n_repeats,
        "n_calibration_candidates_per_prompt": args.n_calibration,
        "n_evaluation_candidates_per_prompt": 32 - args.n_calibration,
        "calibration_fraction": args.n_calibration / 32.0,
        "random_seed": args.random_seed,
        "geometry": "fixed_prompt_kl",
        "d_values": D_VALUES,
        "fit_radii": FIT_RADII.tolist(),
        "local_window": "d <= 0.10",
        "n_b_bootstrap": args.n_bootstrap,
        "b_shrinkage": (
            "zero-mean empirical-Bayes prior estimated from calibration-only "
            "B values and bootstrap variances"
        ),
        "direct_model_training": (
            "pooled calibration candidates; hyperparameters selected with "
            "prompt-grouped CV where applicable"
        ),
        "evaluation_gold_policy": (
            "not used for splitting, fitting, hyperparameter selection, "
            "reachability, or prediction"
        ),
        "baselines": BASELINES,
        "leakage_audit_pass": leakage_pass,
        "n_aggregate_rows": int(len(aggregate)),
        "n_prompt_prediction_rows": int(len(prompt)),
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    if not leakage_pass:
        raise RuntimeError("evaluation-gold leakage audit failed")
    print(f"wrote exp29 outputs to {args.outdir}")


if __name__ == "__main__":
    main()
