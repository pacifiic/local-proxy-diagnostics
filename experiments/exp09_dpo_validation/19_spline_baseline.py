#!/usr/bin/env python3
"""Add a calibration-only spline baseline to completed DPO analyses.

The protocol matches exp09's isotonic baseline: fit h(r) ~= E[J|r] on pooled
calibration candidates only, then integrate h(r_eval) under each prompt's
proxy-tilt weights at that prompt/checkpoint's measured KL radius.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

KNOT_CANDIDATES = (4, 6, 8)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tilt_weights(proxy: np.ndarray, t: float) -> np.ndarray:
    logits = t * (proxy - np.max(proxy))
    weights = np.exp(logits)
    return weights / np.sum(weights)


def kl_at_t(proxy: np.ndarray, t: float) -> float:
    weights = tilt_weights(proxy, t)
    return float(
        np.log(len(proxy))
        + np.sum(weights * np.log(np.clip(weights, 1e-300, None)))
    )


def find_t(proxy: np.ndarray, d: float) -> float:
    if d <= 0 or np.var(proxy) == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while kl_at_t(proxy, hi) < d:
        lo, hi = hi, 2.0 * hi
        if hi > 1e4:
            raise ValueError(f"could not reach d={d}")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if kl_at_t(proxy, mid) < d:
            lo = mid
        else:
            hi = mid
    return hi


def fit_spline(proxy: np.ndarray, gold: np.ndarray):
    x = np.asarray(proxy, dtype=float).reshape(-1, 1)
    y = np.asarray(gold, dtype=float)
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    best = None
    for n_knots in KNOT_CANDIDATES:
        model = make_pipeline(
            SplineTransformer(
                n_knots=n_knots,
                degree=3,
                extrapolation="constant",
            ),
            RidgeCV(alphas=RIDGE_ALPHAS),
        )
        mse = float(
            -np.mean(
                cross_val_score(
                    model,
                    x,
                    y,
                    cv=folds,
                    scoring="neg_mean_squared_error",
                )
            )
        )
        if best is None or mse < best[0]:
            best = (mse, n_knots)
    assert best is not None
    model = make_pipeline(
        SplineTransformer(
            n_knots=best[1],
            degree=3,
            extrapolation="constant",
        ),
        RidgeCV(alphas=RIDGE_ALPHAS),
    )
    model.fit(x, y)
    return model, {
        "n_knots": int(best[1]),
        "ridge_alpha": float(model.named_steps["ridgecv"].alpha_),
        "cv_mse": float(best[0]),
    }


def load_pool(path: Path):
    calibration = []
    evaluation: dict[str, list[tuple[float, float]]] = {}
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] not in {"calibration", "eval"}:
                continue
            key = (row["prompt_id"], row["split"], row["sample_idx"])
            if key in seen:
                raise ValueError(f"duplicate candidate key: {key}")
            seen.add(key)
            pair = (float(row["proxy_score"]), float(row["gold_score"]))
            if not np.all(np.isfinite(pair)):
                raise ValueError(f"non-finite score at {key}")
            if row["split"] == "calibration":
                calibration.append(pair)
            else:
                evaluation.setdefault(row["prompt_id"], []).append(pair)
    if not calibration or not evaluation:
        raise ValueError("pool lacks calibration or evaluation rows")
    return calibration, evaluation


def evaluate_source(label: str, per_prompt_path: Path, pool_path: Path):
    per = pd.read_csv(per_prompt_path)
    required = {
        "run",
        "tag",
        "prompt_id",
        "d",
        "ess",
        "dpo_reweight_gold",
    }
    missing = required - set(per.columns)
    if missing:
        raise ValueError(f"{label}: missing per-prompt columns {sorted(missing)}")
    if per.duplicated(["run", "tag", "prompt_id"]).any():
        raise ValueError(f"{label}: duplicate checkpoint/prompt rows")
    checkpoint_scope = (
        per.groupby(["run", "tag"], as_index=False)
        .agg(d_med=("d", "median"), ess_med=("ess", "median"))
    )
    checkpoint_scope = checkpoint_scope[
        (checkpoint_scope["d_med"] <= 1.0)
        & (checkpoint_scope["ess_med"] >= 8.0)
    ][["run", "tag"]]
    if len(checkpoint_scope) != 6:
        raise ValueError(
            f"{label}: expected six local checkpoints, got "
            f"{len(checkpoint_scope)}"
        )
    per = per.merge(
        checkpoint_scope,
        on=["run", "tag"],
        how="inner",
        validate="many_to_one",
    )

    calibration, evaluation = load_pool(pool_path)
    cal_r = np.array([x[0] for x in calibration])
    cal_j = np.array([x[1] for x in calibration])
    model, metadata = fit_spline(cal_r, cal_j)

    predictions = []
    for row in per.itertuples():
        if row.prompt_id not in evaluation:
            raise ValueError(f"{label}: missing evaluation prompt {row.prompt_id}")
        values = evaluation[row.prompt_id]
        r = np.array([x[0] for x in values])
        h = model.predict(r.reshape(-1, 1))
        weights = tilt_weights(r, find_t(r, max(float(row.d), 0.0)))
        predictions.append(float(np.sum(weights * h) - np.mean(h)))
    per = per.copy()
    per.insert(0, "source", label)
    per["pred_spline"] = predictions

    grouped = (
        per.groupby(["source", "run", "tag"], as_index=False)
        .agg(
            d_med=("d", "median"),
            ess_med=("ess", "median"),
            pred_spline=("pred_spline", "mean"),
            dpo_reweight=("dpo_reweight_gold", "mean"),
            n_prompts=("prompt_id", "nunique"),
        )
        .sort_values("d_med")
    )
    local = grouped[
        (grouped["d_med"] <= 1.0) & (grouped["ess_med"] >= 8.0)
    ].copy()
    if len(local) != 6:
        raise ValueError(f"{label}: expected six local checkpoints, got {len(local)}")
    if not (local["n_prompts"] == 200).all():
        raise ValueError(f"{label}: checkpoint missing evaluation prompts")
    mae = float(
        np.mean(np.abs(local["pred_spline"] - local["dpo_reweight"]))
    )
    summary = {
        "source": label,
        "spline_mae_vs_dpo_realized": mae,
        "n_local_checkpoints": int(len(local)),
        "n_calibration_candidates": int(len(calibration)),
        **metadata,
        "per_prompt_sha256": sha256_file(per_prompt_path),
        "pool_scored_sha256": sha256_file(pool_path),
    }
    return per, grouped, summary


def parse_source(text: str):
    parts = text.split("::")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--source must be LABEL::PER_PROMPT_CSV::POOL_SCORED_JSONL"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_frames = []
    checkpoint_frames = []
    summaries = []
    for label, per_path, pool_path in args.source:
        per, checkpoints, summary = evaluate_source(label, per_path, pool_path)
        per_frames.append(per)
        checkpoint_frames.append(checkpoints)
        summaries.append(summary)

    pd.concat(per_frames, ignore_index=True).to_csv(
        args.outdir / "spline_per_prompt.csv", index=False
    )
    pd.concat(checkpoint_frames, ignore_index=True).to_csv(
        args.outdir / "spline_by_checkpoint.csv", index=False
    )
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.outdir / "spline_summary.csv", index=False)
    (args.outdir / "run_spec.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "pooled calibration-only B-spline+ridge; integrated under "
                    "per-prompt evaluation proxy tilt at measured d"
                ),
                "knot_candidates": KNOT_CANDIDATES,
                "ridge_alphas": RIDGE_ALPHAS.tolist(),
                "cv": "5-fold shuffled KFold, seed 0",
                "sources": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
