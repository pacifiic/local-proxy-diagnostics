#!/usr/bin/env python3
"""Audit B under reduced gold-rater counts and noisy calibration ratings.

The held-out human proxy is kept fixed. Only the calibration gold is degraded.
Evaluation gold always uses the clean mean of every non-proxy rater, which
isolates coefficient-estimation error from changes to the evaluation target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT / "lib"))
from common_experiment_utils import theory_coefficients  # noqa: E402


CALIBRATION_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
D_VALUES = np.asarray((0.01, 0.025, 0.05, 0.075, 0.10), dtype=float)
NOISE_CONDITIONS = (
    ("clean", 0.0),
    ("gaussian", 0.25),
    ("gaussian", 0.50),
    ("gaussian", 1.00),
    ("student_t3", 0.25),
    ("student_t3", 0.50),
    ("student_t3", 1.00),
)
GAMMA_FRACTION = 0.10


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_empty_outdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _weights_and_kl(proxy: np.ndarray, temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = temperatures[:, None] * (proxy[None, :] - float(np.max(proxy)))
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    positive = weights > 0
    kl = np.sum(
        np.where(
            positive,
            weights * (np.log(np.maximum(weights, np.finfo(float).tiny)) + math.log(len(proxy))),
            0.0,
        ),
        axis=1,
    )
    return weights, kl


def exact_gains(
    proxy: np.ndarray, gold: np.ndarray, d_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact gains at every reachable finite-pool KL radius.

    A tied proxy maximum can make a requested radius unreachable. Those radii
    are returned as NaN while lower reachable radii remain available.
    """
    proxy = np.asarray(proxy, dtype=float)
    gold = np.asarray(gold, dtype=float)
    if len(proxy) != len(gold) or len(proxy) < 2:
        raise ValueError("proxy and gold must have the same nontrivial length")
    if not np.isfinite(proxy).all() or not np.isfinite(gold).all():
        raise ValueError("non-finite score in exact_gains")
    if float(np.var(proxy)) == 0.0:
        raise ValueError("proxy variance is zero")

    n_max = int(np.sum(proxy == np.max(proxy)))
    max_reachable = math.log(len(proxy) / n_max)
    reachable = d_values < max_reachable
    if not reachable.any():
        return np.full(len(d_values), np.nan), reachable, max_reachable
    reachable_d = d_values[reachable]

    hi = 1.0
    while True:
        _, kl_hi = _weights_and_kl(proxy, np.asarray([hi]))
        if float(kl_hi[0]) >= float(np.max(reachable_d)):
            break
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("failed to bracket the requested KL radius")

    lo_values = np.zeros(len(reachable_d), dtype=float)
    hi_values = np.full(len(reachable_d), hi, dtype=float)
    for _ in range(64):
        mid = 0.5 * (lo_values + hi_values)
        _, kl_mid = _weights_and_kl(proxy, mid)
        below = kl_mid < reachable_d
        lo_values = np.where(below, mid, lo_values)
        hi_values = np.where(below, hi_values, mid)

    weights, _ = _weights_and_kl(proxy, hi_values)
    gains = np.full(len(d_values), np.nan)
    gains[reachable] = weights @ gold - float(np.mean(gold))
    return gains, reachable, max_reachable


def load_tasks(data_path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(data_path)
    required = {
        "article_id",
        "model_id",
        "rater_type",
        "rater_idx",
        "dimension",
        "score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SummEval input is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("SummEval input contains missing required values")

    frame = frame.copy()
    frame["item_id"] = (
        frame["article_id"].astype(str) + "||" + frame["model_id"].astype(str)
    )
    tasks: list[dict[str, object]] = []
    for (dimension, rater_type), group in frame.groupby(
        ["dimension", "rater_type"], sort=True
    ):
        matrix = group.pivot(
            index="item_id", columns="rater_idx", values="score"
        ).sort_index()
        if matrix.isna().any().any():
            raise ValueError(f"incomplete rating matrix for {dimension}/{rater_type}")
        expected_raters = 3 if rater_type == "expert" else 5
        if matrix.shape != (1600, expected_raters):
            raise ValueError(
                f"unexpected matrix shape for {dimension}/{rater_type}: "
                f"{matrix.shape}"
            )
        values = matrix.to_numpy(dtype=float)
        raters = [str(value) for value in matrix.columns]
        for proxy_position, proxy_rater in enumerate(raters):
            tasks.append(
                {
                    "dimension": str(dimension),
                    "rater_type": str(rater_type),
                    "path_id": (
                        f"{dimension}__{rater_type}__heldout_{proxy_rater}"
                    ),
                    "item_ids": matrix.index.to_numpy(dtype=str),
                    "ratings": values,
                    "proxy_position": proxy_position,
                    "proxy_rater": proxy_rater,
                    "score_min": float(group["score"].min()),
                    "score_max": float(group["score"].max()),
                    "rating_sd": float(group["score"].std(ddof=0)),
                }
            )
    if len(tasks) != 32:
        raise ValueError(f"expected 32 human-proxy paths, found {len(tasks)}")
    return tasks


def _condition_gold(
    gold_matrix: np.ndarray,
    selected: np.ndarray,
    family: str,
    scale: float,
    rating_sd: float,
    score_min: float,
    score_max: float,
    gaussian_noise: np.ndarray,
    student_noise: np.ndarray,
) -> np.ndarray:
    observed = gold_matrix[:, selected]
    if family == "clean":
        perturbed = observed
    elif family == "gaussian":
        perturbed = observed + scale * rating_sd * gaussian_noise[:, selected]
    elif family == "student_t3":
        perturbed = observed + scale * rating_sd * student_noise[:, selected]
    else:
        raise ValueError(f"unknown noise family: {family}")
    return np.clip(perturbed, score_min, score_max).mean(axis=1)


def run_path(
    task: dict[str, object],
    n_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    ratings = np.asarray(task["ratings"], dtype=float)
    item_ids = np.asarray(task["item_ids"], dtype=str)
    proxy_position = int(task["proxy_position"])
    other_positions = np.asarray(
        [idx for idx in range(ratings.shape[1]) if idx != proxy_position],
        dtype=int,
    )
    proxy = ratings[:, proxy_position]
    gold_matrix = ratings[:, other_positions]
    full_clean_gold = gold_matrix.mean(axis=1)
    max_gold_raters = gold_matrix.shape[1]
    path_id = str(task["path_id"])
    rows: list[dict[str, object]] = []

    for repeat in range(n_repeats):
        split_rng = np.random.default_rng(
            stable_seed(random_seed, path_id, repeat, "split")
        )
        ordered_items = split_rng.permutation(len(item_ids))
        rater_rng = np.random.default_rng(
            stable_seed(random_seed, path_id, repeat, "rater_order")
        )
        selected_order = rater_rng.permutation(max_gold_raters)
        gaussian_rng = np.random.default_rng(
            stable_seed(random_seed, path_id, repeat, "gaussian_noise")
        )
        student_rng = np.random.default_rng(
            stable_seed(random_seed, path_id, repeat, "student_t3_noise")
        )
        gaussian_noise = gaussian_rng.normal(size=gold_matrix.shape)
        student_noise = student_rng.standard_t(df=3, size=gold_matrix.shape)
        student_noise /= math.sqrt(3.0)

        for calibration_frac in CALIBRATION_FRACTIONS:
            n_cal = int(round(calibration_frac * len(item_ids)))
            cal_idx = np.sort(ordered_items[:n_cal])
            eval_idx = np.sort(ordered_items[n_cal:])
            split_digest = hashlib.sha256(
                "\n".join(item_ids[cal_idx]).encode("utf-8")
            ).hexdigest()

            eval_proxy = proxy[eval_idx]
            eval_gold = full_clean_gold[eval_idx]
            eval_coeff = theory_coefficients(eval_gold, eval_proxy)
            actual_curve, reachable_d, max_reachable_kl = exact_gains(
                eval_proxy, eval_gold, D_VALUES
            )
            if not reachable_d.any():
                raise RuntimeError(
                    f"no requested radius is reachable for {path_id}, repeat "
                    f"{repeat}, calibration fraction {calibration_frac}"
                )
            clean_full_cal_gold = full_clean_gold[cal_idx]
            gamma = GAMMA_FRACTION * float(
                np.std(clean_full_cal_gold, ddof=0)
            )

            clean_coefficients: dict[int, object] = {}
            for gold_rater_count in range(1, max_gold_raters + 1):
                selected = selected_order[:gold_rater_count]
                clean_gold = gold_matrix[:, selected].mean(axis=1)
                clean_coefficients[gold_rater_count] = theory_coefficients(
                    clean_gold[cal_idx], proxy[cal_idx]
                )
            full_clean_coeff = clean_coefficients[max_gold_raters]

            for gold_rater_count in range(1, max_gold_raters + 1):
                selected = selected_order[:gold_rater_count]
                same_k_clean = clean_coefficients[gold_rater_count]
                for noise_family, noise_scale in NOISE_CONDITIONS:
                    observed_gold = _condition_gold(
                        gold_matrix=gold_matrix,
                        selected=selected,
                        family=noise_family,
                        scale=noise_scale,
                        rating_sd=float(task["rating_sd"]),
                        score_min=float(task["score_min"]),
                        score_max=float(task["score_max"]),
                        gaussian_noise=gaussian_noise,
                        student_noise=student_noise,
                    )
                    coeff = theory_coefficients(
                        observed_gold[cal_idx], proxy[cal_idx]
                    )
                    prediction_ab = (
                        coeff.a_exact * np.sqrt(D_VALUES)
                        + coeff.b_theory * D_VALUES
                    )
                    prediction_aonly = coeff.a_exact * np.sqrt(D_VALUES)
                    curve_mae_ab = float(
                        np.mean(
                            np.abs(
                                prediction_ab[reachable_d]
                                - actual_curve[reachable_d]
                            )
                        )
                    )
                    curve_mae_aonly = float(
                        np.mean(
                            np.abs(
                                prediction_aonly[reachable_d]
                                - actual_curve[reachable_d]
                            )
                        )
                    )
                    d_ref = float(D_VALUES[-1])
                    d0p1_reachable = bool(reachable_d[-1])
                    actual_d0p1 = (
                        float(actual_curve[-1])
                        if d0p1_reachable
                        else float("nan")
                    )
                    g_a = coeff.a_exact * math.sqrt(d_ref)
                    g_b = coeff.b_theory * d_ref
                    rows.append(
                        {
                            "dataset": "SummEval",
                            "dimension": task["dimension"],
                            "rater_type": task["rater_type"],
                            "path_id": path_id,
                            "proxy_rater": task["proxy_rater"],
                            "repeat": repeat,
                            "calibration_frac": calibration_frac,
                            "split_sha256": split_digest,
                            "n_cal_items": len(cal_idx),
                            "n_eval_items": len(eval_idx),
                            "gold_rater_count": gold_rater_count,
                            "max_gold_raters": max_gold_raters,
                            "gold_rater_fraction": (
                                gold_rater_count / max_gold_raters
                            ),
                            "noise_family": noise_family,
                            "noise_scale_sd": noise_scale,
                            "noise_clipping": True,
                            "max_reachable_kl": max_reachable_kl,
                            "n_reachable_d": int(reachable_d.sum()),
                            "d0p1_reachable": d0p1_reachable,
                            "a_cal": coeff.a_exact,
                            "b_cal": coeff.b_theory,
                            "a_eval_clean": eval_coeff.a_exact,
                            "b_eval_clean": eval_coeff.b_theory,
                            "a_shift_from_same_k_clean": (
                                coeff.a_exact - same_k_clean.a_exact
                            ),
                            "b_shift_from_same_k_clean": (
                                coeff.b_theory - same_k_clean.b_theory
                            ),
                            "a_shift_from_full_rater_clean": (
                                coeff.a_exact - full_clean_coeff.a_exact
                            ),
                            "b_shift_from_full_rater_clean": (
                                coeff.b_theory - full_clean_coeff.b_theory
                            ),
                            "b_abs_error_vs_eval": abs(
                                coeff.b_theory - eval_coeff.b_theory
                            ),
                            "b_sign_matches_eval": bool(
                                np.sign(coeff.b_theory)
                                == np.sign(eval_coeff.b_theory)
                            ),
                            "b_sign_matches_same_k_clean": bool(
                                np.sign(coeff.b_theory)
                                == np.sign(same_k_clean.b_theory)
                            ),
                            "g_a_d0p1": g_a,
                            "g_b_d0p1": g_b,
                            "b_contribution_share_d0p1": (
                                abs(g_b)
                                / max(abs(g_a) + abs(g_b), 1e-12)
                            ),
                            "actual_gain_d0p1": actual_d0p1,
                            "pred_ab_d0p1": float(prediction_ab[-1]),
                            "pred_aonly_d0p1": float(prediction_aonly[-1]),
                            "abs_error_ab_d0p1": (
                                abs(float(prediction_ab[-1] - actual_d0p1))
                                if d0p1_reachable
                                else float("nan")
                            ),
                            "abs_error_aonly_d0p1": (
                                abs(
                                    float(prediction_aonly[-1] - actual_d0p1)
                                )
                                if d0p1_reachable
                                else float("nan")
                            ),
                            "curve_mae_ab": curve_mae_ab,
                            "curve_mae_aonly": curve_mae_aonly,
                            "ab_minus_aonly_curve_mae": (
                                curve_mae_ab - curve_mae_aonly
                            ),
                            "gamma": gamma,
                            "proceed_ab_d0p1": bool(
                                prediction_ab[-1] > gamma
                            ),
                            "proceed_aonly_d0p1": bool(
                                prediction_aonly[-1] > gamma
                            ),
                            "decision_disagree_d0p1": bool(
                                (prediction_ab[-1] > gamma)
                                != (prediction_aonly[-1] > gamma)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "summeval_long.csv",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--jobs", type=int, default=min(16, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    if args.n_repeats < 1 or args.jobs < 1:
        parser.error("--n-repeats and --jobs must be positive")
    ensure_empty_outdir(args.outdir)
    tasks = load_tasks(args.data)

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_path, task, args.n_repeats, args.random_seed
            ): str(task["path_id"])
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            path_id = futures[future]
            frame = future.result()
            frames.append(frame)
            print(
                f"completed {index}/{len(futures)} paths: "
                f"{path_id} ({len(frame)} rows)",
                flush=True,
            )

    rows = pd.concat(frames, ignore_index=True).sort_values(
        [
            "rater_type",
            "dimension",
            "path_id",
            "repeat",
            "calibration_frac",
            "gold_rater_count",
            "noise_family",
            "noise_scale_sd",
        ]
    )
    rows.to_csv(args.outdir / "b_rater_noise_rows.csv", index=False)
    reachability = (
        rows[
            [
                "rater_type",
                "dimension",
                "path_id",
                "repeat",
                "calibration_frac",
                "n_eval_items",
                "max_reachable_kl",
                "n_reachable_d",
                "d0p1_reachable",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["rater_type", "dimension", "path_id", "repeat", "calibration_frac"]
        )
    )
    reachability.to_csv(
        args.outdir / "reachability_accounting.csv", index=False
    )
    spec = {
        "experiment": "exp26_b_rater_noise_stability",
        "input": str(args.data),
        "input_sha256": sha256_file(args.data),
        "input_rows": 51200,
        "human_proxy_rule": "one fixed held-out rater",
        "calibration_gold_rule": (
            "mean of a nested subset of non-proxy raters, with optional "
            "synthetic noise"
        ),
        "evaluation_gold_rule": (
            "clean mean of every non-proxy rater, independent of calibration "
            "rater count and noise"
        ),
        "calibration_fractions": list(CALIBRATION_FRACTIONS),
        "d_values": D_VALUES.tolist(),
        "noise_conditions": [
            {"family": family, "scale_in_group_rating_sd": scale}
            for family, scale in NOISE_CONDITIONS
        ],
        "noise_policy": (
            "independent per-rating Gaussian or variance-normalized t(df=3) "
            "noise, scaled by group rating SD and clipped to the observed "
            "rating range"
        ),
        "nested_item_splits": True,
        "nested_gold_rater_subsets": True,
        "n_repeats": args.n_repeats,
        "random_seed": args.random_seed,
        "seed_policy": "SHA-256-derived streams per path/repeat/component",
        "split_unit": "item",
        "n_paths": 32,
        "n_rows": int(len(rows)),
        "n_path_repeat_fractions": int(len(reachability)),
        "n_d0p1_unreachable_path_repeat_fractions": int(
            (~reachability["d0p1_reachable"]).sum()
        ),
        "reachability_policy": (
            "retain every path/repeat; compute curve metrics over requested "
            "radii reachable by that evaluation proxy and mark d=0.10 "
            "endpoint metrics missing when d=0.10 is unreachable"
        ),
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
