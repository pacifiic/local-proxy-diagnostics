#!/usr/bin/env python3
"""Compare promptwise target-aware A+B with direct DPO calibrators.

All conditional-mean models are fit on calibration prompts only. For each
evaluation prompt, the frozen model is evaluated on that prompt's proxy scores.
The same pseudo-quality vector is then used in two ways:

1. compute prompt-specific moment coefficients and predict A_h sqrt(d) + B_h d;
2. integrate the pseudo-quality exactly under the raw-proxy exponential tilt.

Evaluation silver labels are used only to score predictions and to re-derive
the already stored idealized frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BINNED_PATH = HERE / "binned_followup.py"

_spec = importlib.util.spec_from_file_location("dpo_binned_followup", BINNED_PATH)
binned = importlib.util.module_from_spec(_spec)
sys.modules["dpo_binned_followup"] = binned
_spec.loader.exec_module(binned)  # type: ignore[union-attr]
prompt_calibration = binned.prompt_calibration

sys.path.insert(0, str(ROOT / "lib"))
from common_experiment_utils import theory_coefficients  # noqa: E402


FAMILIES = ("isotonic", "spline", "binned")
MODES = ("raw", "centered", "zscore", "rank")
PAPER_BASELINES = ("A+B", "A-only", "A_nu", "Zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-scored",
        type=Path,
        default=ROOT / "data/dpo_candidate_pool/pool_scored.jsonl",
    )
    parser.add_argument(
        "--multiseed-root",
        type=Path,
        default=ROOT / "results/actual_dpo/training_runs",
    )
    parser.add_argument(
        "--existing-primary-checkpoints",
        type=Path,
        default=ROOT
        / "results/prompt_conditional_calibration/checkpoint_predictions.csv",
    )
    parser.add_argument(
        "--existing-binned-checkpoints",
        type=Path,
        default=ROOT
        / "results/actual_dpo/calibration_models/binned"
        / "checkpoint_predictions.csv",
    )
    parser.add_argument(
        "--existing-primary-metadata",
        type=Path,
        default=ROOT
        / "results/prompt_conditional_calibration/model_fit_metadata.csv",
    )
    parser.add_argument(
        "--existing-binned-metadata",
        type=Path,
        default=ROOT
        / "results/actual_dpo/calibration_models/binned"
        / "model_fit_metadata.csv",
    )
    parser.add_argument(
        "--canonical-checkpoints",
        type=Path,
        default=ROOT
        / "results/actual_dpo/three_seed_canonical/checkpoint_predictions.csv",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--mean-kl-max", type=float, default=0.10)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def build_models(
    groups,
    primary_metadata_path: Path,
    binned_metadata_path: Path,
):
    """Refit the exact hyperparameters selected by the canonical direct runs."""
    primary_metadata = pd.read_csv(primary_metadata_path)
    primary_metadata = primary_metadata[
        primary_metadata["fit_split"] == "calibration"
    ].copy()
    binned_metadata = pd.read_csv(binned_metadata_path)
    prompt_ids = sorted(
        prompt_id for split, prompt_id in groups if split == "calibration"
    )
    models = {}
    metadata_rows = []
    for mode in MODES:
        x, y, group_ids = prompt_calibration.pooled_xy(
            groups, prompt_ids, "calibration", mode
        )
        isotonic_name = f"cal_{mode}_isotonic"
        isotonic_predict, _ = prompt_calibration.fit_model(
            x, y, group_ids, "isotonic"
        )
        models[isotonic_name] = (mode, isotonic_predict)
        isotonic_meta = primary_metadata[
            primary_metadata["model"] == isotonic_name
        ]
        if len(isotonic_meta) != 1:
            raise RuntimeError(f"missing locked metadata for {isotonic_name}")
        metadata_rows.append(isotonic_meta.iloc[0].to_dict())

        spline_name = f"cal_{mode}_spline"
        spline_meta = primary_metadata[
            primary_metadata["model"] == spline_name
        ]
        if len(spline_meta) != 1:
            raise RuntimeError(f"missing locked metadata for {spline_name}")
        spline_meta = spline_meta.iloc[0]
        spline_model = prompt_calibration.make_pipeline(
            prompt_calibration.SplineTransformer(
                n_knots=int(spline_meta["n_knots"]),
                degree=3,
                extrapolation="constant",
            ),
            prompt_calibration.Ridge(alpha=float(spline_meta["ridge_alpha"])),
        )
        spline_model.fit(np.asarray(x, dtype=float).reshape(-1, 1), y)
        models[spline_name] = (
            mode,
            lambda new_x, fitted=spline_model: fitted.predict(
                np.asarray(new_x, dtype=float).reshape(-1, 1)
            ),
        )
        metadata_rows.append(spline_meta.to_dict())

        binned_name = f"cal_{mode}_binned"
        binned_meta = binned_metadata[
            binned_metadata["model"] == binned_name
        ]
        if len(binned_meta) != 1:
            raise RuntimeError(f"missing locked metadata for {binned_name}")
        binned_meta = binned_meta.iloc[0]
        edges, values, _ = binned.fit_bins(
            x,
            y,
            int(binned_meta["n_bins_requested"]),
            float(binned_meta["shrink_k"]),
        )
        models[binned_name] = (
            mode,
            lambda new_x, e=edges, v=values: binned.predict_bins(new_x, e, v),
        )
        metadata_rows.append(binned_meta.to_dict())

    expected = {
        f"cal_{mode}_{family}" for mode in MODES for family in FAMILIES
    }
    if set(models) != expected:
        raise RuntimeError(
            f"conditional-mean model mismatch: {sorted(set(models) ^ expected)}"
        )
    metadata = pd.DataFrame(metadata_rows)
    metadata["hyperparameter_source"] = "locked canonical direct-run metadata"
    return models, metadata


def build_eval_cache(groups, models):
    cache = {}
    for split, prompt_id in sorted(groups):
        if split != "eval":
            continue
        proxy, gold = prompt_calibration.arrays(groups[(split, prompt_id)])
        degenerate_proxy = float(np.std(proxy)) == 0.0
        transformed = {
            mode: binned.transform_proxy(proxy, mode) for mode in MODES
        }
        model_rows = {}
        for name, (mode, predict) in models.items():
            pseudo_gold = np.asarray(predict(transformed[mode]), dtype=float)
            if pseudo_gold.shape != proxy.shape or not np.isfinite(pseudo_gold).all():
                raise RuntimeError(f"invalid pseudo-quality for {prompt_id}/{name}")
            coeff = (
                None
                if degenerate_proxy
                else theory_coefficients(pseudo_gold, proxy)
            )
            model_rows[name] = {
                "pseudo_gold": pseudo_gold,
                "A_h": np.nan if coeff is None else float(coeff.a_exact),
                "B_h": np.nan if coeff is None else float(coeff.b_theory),
            }
        cache[prompt_id] = {
            "proxy": proxy,
            "gold": gold,
            "models": model_rows,
            "degenerate_proxy": degenerate_proxy,
        }
    if len(cache) != 200:
        raise RuntimeError(f"expected 200 evaluation prompts, found {len(cache)}")
    return cache


def selected_source_rows(
    multiseed_root: Path,
    seeds: list[int],
    mean_kl_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    selected = []
    for seed in seeds:
        path = multiseed_root / f"seed_{seed}" / "exp09_final_per_prompt.csv"
        frame = pd.read_csv(path)
        if len(frame) != 1_200 or frame["prompt_id"].nunique() != 200:
            raise RuntimeError(f"unexpected source accounting in {path}")
        by_tag = (
            frame.groupby("tag", as_index=False)
            .agg(
                checkpoint_mean_prompt_kl=("d", "mean"),
                checkpoint_median_prompt_kl=("d", "median"),
                n_prompts=("prompt_id", "nunique"),
            )
            .sort_values("tag")
        )
        keep = by_tag.loc[
            by_tag["checkpoint_mean_prompt_kl"] <= mean_kl_max
        ].copy()
        keep.insert(0, "seed", seed)
        selected.append(keep)
        frame = frame[frame["tag"].isin(keep["tag"])].copy()
        frame.insert(0, "seed", seed)
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    selected_frame = pd.concat(selected, ignore_index=True)
    if len(selected_frame) != 10 or len(source) != 2_000:
        raise RuntimeError(
            "expected 10 seed-checkpoints and 2,000 prompt-checkpoint rows, "
            f"found {len(selected_frame)} and {len(source)}"
        )
    if source.duplicated(["seed", "tag", "prompt_id"]).any():
        raise RuntimeError("duplicate source prompt-checkpoint key")
    return source, selected_frame


def evaluate(source: pd.DataFrame, cache: dict) -> pd.DataFrame:
    rows = []
    for row in source.itertuples(index=False):
        prompt = cache[str(row.prompt_id)]
        proxy = prompt["proxy"]
        gold = prompt["gold"]
        d = max(float(row.d), 0.0)
        if prompt["degenerate_proxy"] and d != 0.0:
            raise RuntimeError(
                f"positive radius with zero proxy variance: {row.prompt_id}"
            )
        temperature = prompt_calibration.dpo_spline.find_t(proxy, d)
        weights = prompt_calibration.dpo_spline.tilt_weights(proxy, temperature)
        frontier = float(np.sum(weights * gold) - np.mean(gold))
        if not math.isclose(
            frontier, float(row.frontier), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError(
                f"frontier mismatch: seed={row.seed}, tag={row.tag}, "
                f"prompt={row.prompt_id}"
            )
        for model, values in prompt["models"].items():
            pseudo_gold = values["pseudo_gold"]
            projection = (
                0.0
                if prompt["degenerate_proxy"]
                else values["A_h"] * math.sqrt(d) + values["B_h"] * d
            )
            direct = float(np.sum(weights * pseudo_gold) - np.mean(pseudo_gold))
            rows.append(
                {
                    "seed": int(row.seed),
                    "tag": str(row.tag),
                    "prompt_id": str(row.prompt_id),
                    "model": model,
                    "d": d,
                    "A_h": values["A_h"],
                    "B_h": values["B_h"],
                    "projection": projection,
                    "direct": direct,
                    "frontier": frontier,
                    "dpo_implied_gain": float(row.dpo_reweight_gold),
                    "abs_projection_minus_direct": abs(projection - direct),
                }
            )
    result = pd.DataFrame(rows)
    expected = 2_000 * len(FAMILIES) * len(MODES)
    if len(result) != expected:
        raise RuntimeError(f"expected {expected} result rows, found {len(result)}")
    if result.duplicated(["seed", "tag", "prompt_id", "model"]).any():
        raise RuntimeError("duplicate target-aware prediction key")
    return result


def aggregate(result: pd.DataFrame):
    checkpoint = (
        result.groupby(["seed", "tag", "model"], as_index=False)
        .agg(
            mean_prompt_kl=("d", "mean"),
            projection=("projection", "mean"),
            direct=("direct", "mean"),
            frontier=("frontier", "mean"),
            dpo_implied_gain=("dpo_implied_gain", "mean"),
            mean_prompt_abs_projection_direct=(
                "abs_projection_minus_direct",
                "mean",
            ),
            n_prompts=("prompt_id", "nunique"),
        )
        .sort_values(["model", "seed", "tag"])
    )
    checkpoint["projection_frontier_error"] = np.abs(
        checkpoint["projection"] - checkpoint["frontier"]
    )
    checkpoint["direct_frontier_error"] = np.abs(
        checkpoint["direct"] - checkpoint["frontier"]
    )
    checkpoint["projection_dpo_error"] = np.abs(
        checkpoint["projection"] - checkpoint["dpo_implied_gain"]
    )
    checkpoint["direct_dpo_error"] = np.abs(
        checkpoint["direct"] - checkpoint["dpo_implied_gain"]
    )
    checkpoint["aggregate_projection_direct_gap"] = np.abs(
        checkpoint["projection"] - checkpoint["direct"]
    )

    seed_level = (
        checkpoint.groupby(["seed", "model"], as_index=False)
        .agg(
            projection_frontier_mae=("projection_frontier_error", "mean"),
            direct_frontier_mae=("direct_frontier_error", "mean"),
            projection_dpo_mae=("projection_dpo_error", "mean"),
            direct_dpo_mae=("direct_dpo_error", "mean"),
            aggregate_projection_direct_gap=(
                "aggregate_projection_direct_gap",
                "mean",
            ),
            prompt_cell_projection_direct_gap=(
                "mean_prompt_abs_projection_direct",
                "mean",
            ),
            n_checkpoints=("tag", "nunique"),
        )
        .sort_values(["model", "seed"])
    )

    summary = (
        seed_level.groupby("model", as_index=False)
        .agg(
            projection_frontier_mae_mean=("projection_frontier_mae", "mean"),
            projection_frontier_mae_sd=("projection_frontier_mae", "std"),
            direct_frontier_mae_mean=("direct_frontier_mae", "mean"),
            direct_frontier_mae_sd=("direct_frontier_mae", "std"),
            projection_dpo_mae_mean=("projection_dpo_mae", "mean"),
            projection_dpo_mae_sd=("projection_dpo_mae", "std"),
            direct_dpo_mae_mean=("direct_dpo_mae", "mean"),
            direct_dpo_mae_sd=("direct_dpo_mae", "std"),
            aggregate_projection_direct_gap_mean=(
                "aggregate_projection_direct_gap",
                "mean",
            ),
            aggregate_projection_direct_gap_sd=(
                "aggregate_projection_direct_gap",
                "std",
            ),
            prompt_cell_projection_direct_gap_mean=(
                "prompt_cell_projection_direct_gap",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
            min_checkpoints_per_seed=("n_checkpoints", "min"),
            max_checkpoints_per_seed=("n_checkpoints", "max"),
        )
        .sort_values("model")
    )
    return checkpoint, seed_level, summary


def verify_direct_predictions(
    checkpoint: pd.DataFrame,
    primary_path: Path,
    binned_path: Path,
) -> float:
    existing = pd.concat(
        [pd.read_csv(primary_path), pd.read_csv(binned_path)],
        ignore_index=True,
    )
    existing = existing[
        existing["model"].isin(checkpoint["model"].unique())
    ][["seed", "tag", "model", "prediction"]].drop_duplicates()
    paired = checkpoint.merge(
        existing,
        on=["seed", "tag", "model"],
        validate="one_to_one",
    )
    if len(paired) != len(checkpoint):
        raise RuntimeError(
            f"direct audit accounting mismatch: {len(paired)} vs {len(checkpoint)}"
        )
    paired["abs_diff"] = np.abs(paired["direct"] - paired["prediction"])
    max_diff = float(paired["abs_diff"].max())
    if max_diff > 1e-10:
        by_model = (
            paired.groupby("model")["abs_diff"].max().sort_values(ascending=False)
        )
        raise RuntimeError(
            "existing direct prediction mismatch; max by model:\n"
            + by_model.to_string()
        )
    return max_diff


def paper_baselines(
    canonical_path: Path,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    canonical = pd.read_csv(canonical_path)
    canonical = canonical[canonical["model"].isin(PAPER_BASELINES)].copy()
    keep = selected[["seed", "tag"]].drop_duplicates()
    canonical = canonical.merge(keep, on=["seed", "tag"], validate="many_to_one")
    if len(canonical) != len(keep) * len(PAPER_BASELINES):
        raise RuntimeError("paper-baseline checkpoint accounting mismatch")
    per_seed = (
        canonical.groupby(["seed", "model"], as_index=False)
        .agg(
            frontier_mae=("abs_error_frontier", "mean"),
            dpo_mae=("abs_error_dpo", "mean"),
            n_checkpoints=("tag", "nunique"),
        )
    )
    return (
        per_seed.groupby("model", as_index=False)
        .agg(
            frontier_mae_mean=("frontier_mae", "mean"),
            frontier_mae_sd=("frontier_mae", "std"),
            dpo_mae_mean=("dpo_mae", "mean"),
            dpo_mae_sd=("dpo_mae", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("model")
    )


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.5f} +/- {sd:.5f}"


def render_markdown(
    summary: pd.DataFrame,
    baselines: pd.DataFrame,
    selected: pd.DataFrame,
    max_direct_diff: float,
) -> str:
    lines = [
        "# DPO Target-Pool-Aware A+B and Direct Calibration",
        "",
        "All conditional-mean models are fit on the 100 calibration prompts only.",
        "For each of 200 evaluation prompts, the frozen model is evaluated on",
        "that prompt's proxy scores. `Target-aware A+B` computes prompt-specific",
        "moments of `(h(r_eval), r_eval)` and predicts",
        "`A_h sqrt(d_x) + B_h d_x`; `Direct` exactly integrates the same",
        "`h(r_eval)` under the raw-proxy tilt at the same `d_x`. Evaluation",
        "silver scores are used only to compute the idealized frontier and the",
        "DPO-implied fixed-pool gain after predictions are frozen.",
        "",
        "The ten checkpoints are selected by checkpoint mean prompt KL <= 0.10.",
        "This is a mean-KL gate, not a guarantee that every prompt is local.",
        "",
        "## Original Paper Baselines",
        "",
        "| Predictor | Idealized-frontier MAE | DPO-implied-gain MAE |",
        "|---|---:|---:|",
    ]
    for row in baselines.itertuples(index=False):
        lines.append(
            f"| {row.model} | {fmt(row.frontier_mae_mean, row.frontier_mae_sd)} "
            f"| {fmt(row.dpo_mae_mean, row.dpo_mae_sd)} |"
        )

    lines.extend(
        [
            "",
            "## Target-Pool-Aware Projection vs. Direct Integration",
            "",
            "Each cell is mean MAE +/- sample SD across the three DPO training seeds.",
            "",
            "| Conditional-mean model | Target-aware A+B vs frontier | Direct vs frontier | Target-aware A+B vs DPO-implied gain | Direct vs DPO-implied gain | Projection-direct gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        label = row.model.removeprefix("cal_").replace("_", " ")
        lines.append(
            f"| {label} | "
            f"{fmt(row.projection_frontier_mae_mean, row.projection_frontier_mae_sd)} | "
            f"{fmt(row.direct_frontier_mae_mean, row.direct_frontier_mae_sd)} | "
            f"{fmt(row.projection_dpo_mae_mean, row.projection_dpo_mae_sd)} | "
            f"{fmt(row.direct_dpo_mae_mean, row.direct_dpo_mae_sd)} | "
            f"{fmt(row.aggregate_projection_direct_gap_mean, row.aggregate_projection_direct_gap_sd)} |"
        )

    prompt_counts = (
        selected.groupby("seed")["n_prompts"].sum().astype(int).to_dict()
    )
    lines.extend(
        [
            "",
            "The projection-direct gap is computed between the two aggregate",
            "checkpoint predictions before averaging across checkpoints and seeds.",
            "The full prompt-cell discrepancy is retained in the CSV outputs.",
            "",
            "## Interpretation",
            "",
            "- This is a later-stage target-pool-aware analysis. It is not the",
            "  paper's pre-pool estimator and should not replace the matched",
            "  calibration-only comparison.",
            "- Projection and direct rows use exactly the same fitted conditional",
            "  mean and evaluation proxy geometry. Their difference measures the",
            "  local two-term projection discrepancy for that fitted curve.",
            "- MAE against DPO-implied gain also includes the optimizer-realization",
            "  gap, so a lower value can result from error cancellation.",
            "- Common paper coefficients and prompt-specific target-aware",
            "  coefficients answer different operational questions.",
            "",
            "## Integrity Checks",
            "",
            f"- Selected seed-checkpoints: {len(selected)}; prompt counts by seed: "
            f"`{prompt_counts}`.",
            f"- Maximum absolute difference from the previously saved direct",
            f"  checkpoint predictions: `{max_direct_diff:.3g}`.",
            "- Every stored idealized frontier was re-derived from raw candidate",
            "  scores and matched the original source row.",
            "- The script refuses to overwrite a non-empty output directory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups = prompt_calibration.load_pool(args.pool_scored)
    models, metadata = build_models(
        groups,
        args.existing_primary_metadata,
        args.existing_binned_metadata,
    )
    cache = build_eval_cache(groups, models)
    source, selected = selected_source_rows(
        args.multiseed_root, args.seeds, args.mean_kl_max
    )
    per_prompt = evaluate(source, cache)
    checkpoint, seed_level, summary = aggregate(per_prompt)
    max_direct_diff = verify_direct_predictions(
        checkpoint,
        args.existing_primary_checkpoints,
        args.existing_binned_checkpoints,
    )
    baselines = paper_baselines(args.canonical_checkpoints, selected)

    per_prompt.to_csv(args.outdir / "per_prompt_predictions.csv", index=False)
    checkpoint.to_csv(args.outdir / "checkpoint_predictions.csv", index=False)
    seed_level.to_csv(args.outdir / "seed_level_mae.csv", index=False)
    summary.to_csv(args.outdir / "summary_mean_sd.csv", index=False)
    baselines.to_csv(args.outdir / "paper_baseline_summary.csv", index=False)
    metadata.to_csv(args.outdir / "model_fit_metadata.csv", index=False)
    selected.to_csv(args.outdir / "selected_checkpoints.csv", index=False)

    report = render_markdown(summary, baselines, selected, max_direct_diff)
    (args.outdir / "DPO_TARGET_POOL_AWARE_AB_DIRECT_COMPARISON.md").write_text(
        report + "\n", encoding="utf-8"
    )
    run_spec = {
        "experiment": "DPO target-pool-aware conditional-mean A+B follow-up",
        "information_setting": {
            "conditional_mean_fit": "calibration prompts only",
            "evaluation_proxy_geometry": "used",
            "evaluation_silver_labels_for_prediction": "not used",
            "coefficient_scope": "prompt-specific moments of (h(r_eval), r_eval)",
            "direct_comparator": (
                "exact integration of the same h(r_eval) under raw-proxy tilt"
            ),
        },
        "selection": {
            "checkpoint_mean_prompt_kl_max": args.mean_kl_max,
            "n_seed_checkpoints": len(selected),
        },
        "seeds": args.seeds,
        "modes": MODES,
        "families": FAMILIES,
        "inputs": {
            "pool_scored": display_path(args.pool_scored),
            "multiseed_root": display_path(args.multiseed_root),
            "existing_primary_checkpoints": display_path(
                args.existing_primary_checkpoints
            ),
            "existing_binned_checkpoints": display_path(
                args.existing_binned_checkpoints
            ),
            "existing_primary_metadata": display_path(
                args.existing_primary_metadata
            ),
            "existing_binned_metadata": display_path(
                args.existing_binned_metadata
            ),
            "canonical_checkpoints": display_path(args.canonical_checkpoints),
        },
        "input_sha256": {
            "pool_scored": sha256_file(args.pool_scored),
            "existing_primary_checkpoints": sha256_file(
                args.existing_primary_checkpoints
            ),
            "existing_binned_checkpoints": sha256_file(
                args.existing_binned_checkpoints
            ),
            "existing_primary_metadata": sha256_file(
                args.existing_primary_metadata
            ),
            "existing_binned_metadata": sha256_file(
                args.existing_binned_metadata
            ),
            "canonical_checkpoints": sha256_file(args.canonical_checkpoints),
            **{
                f"seed_{seed}_per_prompt": sha256_file(
                    args.multiseed_root
                    / f"seed_{seed}"
                    / "exp09_final_per_prompt.csv"
                )
                for seed in args.seeds
            },
        },
        "code_sha256": {
            "this_script": sha256_file(Path(__file__).resolve()),
            "primary_builder": sha256_file(
                HERE / "build_prompt_conditional_calibration.py"
            ),
            "binned_builder": sha256_file(BINNED_PATH),
        },
        "integrity": {
            "max_abs_direct_prediction_difference": max_direct_diff,
            "n_per_prompt_rows": len(per_prompt),
            "n_checkpoint_rows": len(checkpoint),
        },
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
