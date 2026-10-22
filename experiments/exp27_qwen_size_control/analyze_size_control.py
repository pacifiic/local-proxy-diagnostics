#!/usr/bin/env python3
"""Matched-prompt Qwen size control and same-size Qwen/Zephyr diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP25 = HERE.parent / "exp25_real_llm_calibration_followup"
sys.path.insert(0, str(EXP25))
from baseline_core import (  # noqa: E402
    BASELINES,
    ensure_empty_outdir,
    evaluate_models,
    fit_models,
    load_pool,
    prompt_reachability,
    sha256_file,
    validate_pool,
)


D_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
LOCAL_MAX_D = 0.10
GEOMETRY = "fixed_prompt_kl"
LABELS = {
    "qwen_3b": "Qwen2.5-3B",
    "qwen_7b": "Qwen2.5-7B",
    "qwen_14b": "Qwen2.5-14B",
    "qwen_32b": "Qwen2.5-32B",
    "zephyr_7b": "Zephyr-7B",
}
NOMINAL_PARAMETERS = {
    "qwen_3b": 3.0,
    "qwen_7b": 7.0,
    "qwen_14b": 14.0,
    "qwen_32b": 32.0,
    "zephyr_7b": 7.0,
}


def prompt_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return (
        frame[frame["split"].eq(split)][["prompt_id", "prompt"]]
        .drop_duplicates()
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )


def response_sha(frame: pd.DataFrame) -> str:
    import hashlib

    payload = "\n".join(
        frame.sort_values(["split", "prompt_id", "sample_idx"])["response"]
        .astype(str)
        .tolist()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bootstrap_curve_mae(
    prompt: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    local = prompt[prompt["d"].le(LOCAL_MAX_D + 1e-12)]
    output: list[dict[str, object]] = []
    for (pool, baseline), group in local.groupby(["pool", "baseline"], sort=True):
        prompt_ids = sorted(group["prompt_id"].astype(str).unique())
        d_values = sorted(group["d"].unique())
        lookup = {
            d: group[group["d"].eq(d)]
            .assign(prompt_id=lambda x: x["prompt_id"].astype(str))
            .set_index("prompt_id")[["predicted_gain", "actual_gain"]]
            .loc[prompt_ids]
            .to_numpy(float)
            for d in d_values
        }

        def score(indices: np.ndarray) -> float:
            return float(
                np.mean(
                    [
                        abs(
                            values[indices, 0].mean()
                            - values[indices, 1].mean()
                        )
                        for values in lookup.values()
                    ]
                )
            )

        full = np.arange(len(prompt_ids))
        point = score(full)
        rng = np.random.default_rng(seed)
        boot = np.asarray(
            [
                score(rng.integers(0, len(prompt_ids), len(prompt_ids)))
                for _ in range(n_boot)
            ]
        )
        output.append(
            {
                "pool": pool,
                "baseline": baseline,
                "curve_mae": point,
                "ci_lo": float(np.quantile(boot, 0.025)),
                "ci_hi": float(np.quantile(boot, 0.975)),
                "n_eval_prompts": len(prompt_ids),
                "n_radii": len(d_values),
            }
        )
    return pd.DataFrame(output).sort_values(["pool", "curve_mae"])


def paired_model_comparisons(
    prompt: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    local = prompt[
        prompt["d"].le(LOCAL_MAX_D + 1e-12)
        & prompt["baseline"].eq("A+B")
    ]
    prompt_ids = sorted(local["prompt_id"].astype(str).unique())
    d_values = sorted(local["d"].unique())
    lookup: dict[str, dict[float, np.ndarray]] = {}
    available = set(local["pool"])
    for pool in (name for name in LABELS if name in available):
        group = local[local["pool"].eq(pool)].assign(
            prompt_id=lambda x: x["prompt_id"].astype(str)
        )
        lookup[pool] = {
            d: group[group["d"].eq(d)]
            .set_index("prompt_id")[["predicted_gain", "actual_gain"]]
            .loc[prompt_ids]
            .to_numpy(float)
            for d in d_values
        }

    def score(pool: str, indices: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    abs(
                        values[indices, 0].mean()
                        - values[indices, 1].mean()
                    )
                    for values in lookup[pool].values()
                ]
            )
        )

    candidate_comparisons = [
        ("qwen_7b_minus_qwen_3b", "qwen_7b", "qwen_3b"),
        ("qwen_14b_minus_qwen_7b", "qwen_14b", "qwen_7b"),
        ("qwen_32b_minus_qwen_14b", "qwen_32b", "qwen_14b"),
        ("qwen_7b_minus_zephyr_7b", "qwen_7b", "zephyr_7b"),
        ("qwen_32b_minus_zephyr_7b", "qwen_32b", "zephyr_7b"),
        ("qwen_3b_minus_zephyr_7b", "qwen_3b", "zephyr_7b"),
    ]
    comparisons = [
        comparison
        for comparison in candidate_comparisons
        if comparison[1] in lookup and comparison[2] in lookup
    ]
    full = np.arange(len(prompt_ids))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(prompt_ids), size=(n_boot, len(prompt_ids))
    )
    rows: list[dict[str, object]] = []
    for label, left, right in comparisons:
        point = score(left, full) - score(right, full)
        draws = np.asarray(
            [score(left, indices) - score(right, indices) for indices in sampled]
        )
        rows.append(
            {
                "comparison": label,
                "left_pool": left,
                "right_pool": right,
                "A+B_curve_mae_difference": point,
                "ci_lo": float(np.quantile(draws, 0.025)),
                "ci_hi": float(np.quantile(draws, 0.975)),
                "prob_left_error_greater": float(np.mean(draws > 0)),
                "n_eval_prompts": len(prompt_ids),
                "n_radii": len(d_values),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    summary: pd.DataFrame, diagnostics: pd.DataFrame, outdir: Path
) -> None:
    available = set(diagnostics["pool"])
    order = [name for name in LABELS if name in available]
    diagnostic = diagnostics.set_index("pool").loc[order]
    ab = (
        summary[summary["baseline"].eq("A+B")]
        .set_index("pool")
        .loc[order]
    )
    palette = {
        "qwen_3b": "#377eb8",
        "qwen_7b": "#4daf4a",
        "qwen_14b": "#984ea3",
        "qwen_32b": "#ff7f00",
        "zephyr_7b": "#e41a1c",
    }
    colors = [palette[name] for name in order]
    x = np.arange(len(order))
    tick_labels = [LABELS[name] for name in order]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))

    axes[0].bar(x, diagnostic["proxy_gold_pearson_all"], color=colors)
    axes[0].set_title("Proxy-gold alignment")
    axes[0].set_ylabel("Pearson correlation")

    width = 0.34
    axes[1].bar(
        x - width / 2, diagnostic["A"], width, label="A", color="#377eb8"
    )
    axes[1].bar(
        x + width / 2, diagnostic["B"], width, label="B", color="#ff7f00"
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Calibration coefficients")
    axes[1].set_ylabel("Coefficient value")
    axes[1].legend(frameon=False)

    axes[2].vlines(x, ab["ci_lo"], ab["ci_hi"], color="black", linewidth=1)
    axes[2].scatter(x, ab["curve_mae"], color=colors, zorder=3, s=42)
    axes[2].set_title("Local A+B prediction")
    axes[2].set_ylabel(r"CurveMAE, $d\leq0.10$")

    for axis in axes:
        axis.set_xticks(x, tick_labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Matched UltraFeedback prompts and reward scorers")
    fig.tight_layout()
    metadata = {
        "Title": "Qwen size control",
        "Author": "Local Proxy Diagnostics Authors",
        "Creator": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(outdir / "fig_qwen_size_control.pdf", metadata=metadata)
    fig.savefig(outdir / "fig_qwen_size_control.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-3b", type=Path, required=True)
    parser.add_argument("--qwen-7b", type=Path, required=True)
    parser.add_argument("--qwen-14b", type=Path)
    parser.add_argument("--qwen-32b", type=Path)
    parser.add_argument("--zephyr-7b", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    ensure_empty_outdir(args.outdir)

    paths = {
        "qwen_3b": args.qwen_3b,
        "qwen_7b": args.qwen_7b,
        "zephyr_7b": args.zephyr_7b,
    }
    if args.qwen_14b is not None:
        paths["qwen_14b"] = args.qwen_14b
    if args.qwen_32b is not None:
        paths["qwen_32b"] = args.qwen_32b
    pools = {name: load_pool(path) for name, path in paths.items()}
    audits = {
        name: validate_pool(
            frame,
            name=LABELS[name],
            candidate_col="sample_idx",
        )
        for name, frame in pools.items()
    }
    reference = pools["qwen_3b"]
    for split in ("calibration", "eval"):
        expected = prompt_table(reference, split)
        for name, frame in pools.items():
            if not expected.equals(prompt_table(frame, split)):
                raise ValueError(f"{name}: {split} prompt IDs/text differ")
    for name, frame in pools.items():
        if set(frame["split"]) != {"calibration", "eval"}:
            raise ValueError(f"{name}: unexpected split labels")
        if len(frame) != 9600:
            raise ValueError(f"{name}: expected 9600 rows, got {len(frame)}")

    aggregates: list[pd.DataFrame] = []
    prompts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    reachability: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for name, frame in pools.items():
        calibration = frame[frame["split"].eq("calibration")].copy()
        evaluation = frame[frame["split"].eq("eval")].copy()
        models = fit_models(
            calibration,
            geometry=GEOMETRY,
            d_values=D_VALUES,
            fit_max_d=LOCAL_MAX_D,
            seed=args.random_seed,
        )
        aggregate, prompt = evaluate_models(
            calibration,
            evaluation,
            models,
            geometry=GEOMETRY,
            d_values=D_VALUES,
        )
        aggregate.insert(0, "pool", name)
        prompt.insert(0, "pool", name)
        aggregates.append(aggregate)
        prompts.append(prompt)
        for split in ("calibration", "eval"):
            reach = prompt_reachability(
                frame[frame["split"].eq(split)], D_VALUES
            )
            reach.insert(0, "pool", name)
            reach.insert(1, "split", split)
            reachability.append(reach)
        diagnostics.append(
            {
                "pool": name,
                "generator_label": LABELS[name],
                "n_parameters_billions_nominal": NOMINAL_PARAMETERS[name],
                "proxy_gold_pearson_all": float(
                    frame[["proxy_score", "gold_score"]].corr().iloc[0, 1]
                ),
                "proxy_gold_pearson_calibration": float(
                    calibration[["proxy_score", "gold_score"]].corr().iloc[0, 1]
                ),
                "proxy_gold_pearson_eval": float(
                    evaluation[["proxy_score", "gold_score"]].corr().iloc[0, 1]
                ),
                "proxy_mean": float(frame["proxy_score"].mean()),
                "proxy_sd": float(frame["proxy_score"].std(ddof=0)),
                "gold_mean": float(frame["gold_score"].mean()),
                "gold_sd": float(frame["gold_score"].std(ddof=0)),
                "response_char_mean": float(frame["response"].astype(str).str.len().mean()),
                "response_char_sd": float(frame["response"].astype(str).str.len().std(ddof=0)),
                "response_sha256": response_sha(frame),
                "A": models.coefficients.a,
                "B": models.coefficients.b,
                "A_nu": models.coefficients.a_nu,
                "n_calibration_prompts_used": models.coefficients.n_prompts_used,
                "n_calibration_prompts_skipped": models.coefficients.n_prompts_skipped,
            }
        )
        metadata.append(
            {
                "pool": name,
                "metadata": json.dumps(models.metadata, sort_keys=True),
            }
        )

    aggregate = pd.concat(aggregates, ignore_index=True).sort_values(
        ["pool", "d", "baseline"]
    )
    prompt = pd.concat(prompts, ignore_index=True).sort_values(
        ["pool", "prompt_id", "d", "baseline"]
    )
    key = ["pool", "d", "baseline"]
    if aggregate.duplicated(key).any():
        raise ValueError("duplicate aggregate prediction key")
    prompt_key = key + ["prompt_id"]
    if prompt.duplicated(prompt_key).any():
        raise ValueError("duplicate prompt prediction key")
    numeric = aggregate[["predicted_gain", "actual_gain", "abs_error"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite prediction result")

    aggregate.to_csv(args.outdir / "size_control_predictions.csv", index=False)
    prompt.to_csv(
        args.outdir / "size_control_prompt_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame.to_csv(
        args.outdir / "pool_diagnostics.csv", index=False
    )
    pd.DataFrame(metadata).to_csv(args.outdir / "fit_metadata.csv", index=False)
    pd.concat(reachability, ignore_index=True).to_csv(
        args.outdir / "reachability.csv", index=False
    )
    summary = bootstrap_curve_mae(prompt, args.n_boot, args.random_seed)
    summary.to_csv(args.outdir / "local_curve_summary.csv", index=False)
    comparisons = paired_model_comparisons(
        prompt, args.n_boot, args.random_seed
    )
    comparisons.to_csv(
        args.outdir / "a_plus_b_model_comparisons.csv", index=False
    )
    make_figure(summary, diagnostics_frame, args.outdir)

    spec = {
        "experiment": "exp27_qwen_size_control",
        "purpose": (
            "diagnostic matched-prompt comparison: the Qwen size sweep "
            "measures within-family scaling, while Qwen 7B versus Zephyr 7B "
            "holds nominal size approximately fixed but changes family/training"
        ),
        "causal_scope": (
            "descriptive rather than causal because there is one checkpoint "
            "per model-size/family cell"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "input_audits": audits,
        "prompt_identity_check": "exact prompt_id and prompt-text equality",
        "geometry": GEOMETRY,
        "d_values": D_VALUES,
        "local_window": "d <= 0.10",
        "fit_max_d": LOCAL_MAX_D,
        "random_seed": args.random_seed,
        "n_boot": args.n_boot,
        "baselines": BASELINES,
        "gold_definition": (
            "OpenAssistant/reward-model-deberta-v3-large-v2 score; this is "
            "model-based silver gold, not a new human rating"
        ),
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(f"wrote size-control analysis to {args.outdir}")


if __name__ == "__main__":
    main()
