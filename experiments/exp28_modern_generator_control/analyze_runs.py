#!/usr/bin/env python3
"""Analyze matched modern-generator candidate pools across generation seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
EXP25 = HERE.parent / "exp25_real_llm_calibration_followup"
EXP27 = HERE.parent / "exp27_qwen_size_control"
sys.path.insert(0, str(EXP25))
sys.path.insert(0, str(EXP27))
from analyze_size_control import bootstrap_curve_mae  # noqa: E402
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


D_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
LOCAL_MAX_D = 0.10
GEOMETRY = "fixed_prompt_kl"


def prompt_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return (
        frame[frame["split"].eq(split)][["prompt_id", "prompt"]]
        .drop_duplicates()
        .assign(prompt_id=lambda x: x["prompt_id"].astype(str))
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )


def run_metadata(run: Path) -> dict[str, object]:
    spec = json.loads(
        (run / "pool.jsonl.run_spec.json").read_text(encoding="utf-8")
    )
    audit = json.loads((run / "run_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError(f"{run}: run audit did not pass")
    if audit["generator"] != spec["model"]:
        raise ValueError(f"{run}: generator identity mismatch")
    revisions = json.loads(
        (run / "model_revisions.json").read_text(encoding="utf-8")
    )
    model_revision = spec.get("model_revision") or revisions["generator"][
        "revision"
    ]
    return {
        "run": run.name,
        "model": spec["model"],
        "model_revision": model_revision,
        "base_seed": int(spec["base_seed"]),
        "pool_path": run / "pool_scored.jsonl",
        "pool_sha256": sha256_file(run / "pool_scored.jsonl"),
    }


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_figure(
    diagnostics: pd.DataFrame,
    local: pd.DataFrame,
    seed_summary: pd.DataFrame,
    outdir: Path,
) -> None:
    models = list(dict.fromkeys(diagnostics["model"]))
    short = {model: model.split("/")[-1].replace("-Instruct", "") for model in models}
    x = np.arange(len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 3)))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))

    for index, model in enumerate(models):
        rows = diagnostics[diagnostics["model"].eq(model)]
        jitter = np.linspace(-0.08, 0.08, len(rows)) if len(rows) > 1 else [0]
        axes[0].scatter(
            index + np.asarray(jitter),
            rows["proxy_gold_pearson_all"],
            color=colors[index],
            s=35,
        )
        axes[1].scatter(
            index + np.asarray(jitter),
            rows["B"],
            color=colors[index],
            s=35,
        )
        ab = local[
            local["model"].eq(model) & local["baseline"].eq("A+B")
        ]
        jitter_ab = (
            np.linspace(-0.08, 0.08, len(ab)) if len(ab) > 1 else [0]
        )
        axes[2].scatter(
            index + np.asarray(jitter_ab),
            ab["curve_mae"],
            color=colors[index],
            s=35,
        )

    axes[0].set_title("Proxy-silver alignment")
    axes[0].set_ylabel("Pearson correlation")
    axes[1].set_title("Curvature across seeds")
    axes[1].set_ylabel("B")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Local A+B prediction")
    axes[2].set_ylabel(r"CurveMAE, $d\leq0.10$")
    for axis in axes:
        axis.set_xticks(x, [short[m] for m in models], rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    metadata = {
        "Title": "Modern generator control",
        "Author": "Local Proxy Diagnostics Authors",
        "Creator": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(outdir / "fig_modern_generator_control.pdf", metadata=metadata)
    fig.savefig(outdir / "fig_modern_generator_control.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    ensure_empty_outdir(args.outdir)

    records = [run_metadata(run) for run in args.run]
    keys = [(record["model"], record["base_seed"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate model/base-seed input")

    pools = {
        record["run"]: load_pool(record["pool_path"]) for record in records
    }
    reference = pools[records[0]["run"]]
    for split in ("calibration", "eval"):
        expected = prompt_table(reference, split)
        for name, frame in pools.items():
            if not expected.equals(prompt_table(frame, split)):
                raise ValueError(f"{name}: {split} prompts differ")

    aggregates: list[pd.DataFrame] = []
    prompts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    reaches: list[pd.DataFrame] = []
    for record in records:
        name = str(record["run"])
        frame = pools[name]
        audit = validate_pool(
            frame, name=name, candidate_col="sample_idx"
        )
        if audit["n_rows"] != 9600 or audit["n_prompts"] != 300:
            raise ValueError(f"{name}: unexpected pool shape")
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
        for output in (aggregate, prompt):
            output.insert(0, "pool", name)
            output.insert(1, "model", record["model"])
            output.insert(2, "base_seed", record["base_seed"])
        aggregates.append(aggregate)
        prompts.append(prompt)
        for split in ("calibration", "eval"):
            reach = prompt_reachability(
                frame[frame["split"].eq(split)], D_VALUES
            )
            reach.insert(0, "run", name)
            reach.insert(1, "model", record["model"])
            reach.insert(2, "base_seed", record["base_seed"])
            reach.insert(3, "split", split)
            reaches.append(reach)
        diagnostics.append(
            {
                **record,
                "proxy_gold_pearson_all": float(
                    frame[["proxy_score", "gold_score"]].corr().iloc[0, 1]
                ),
                "proxy_gold_pearson_calibration": float(
                    calibration[["proxy_score", "gold_score"]]
                    .corr()
                    .iloc[0, 1]
                ),
                "proxy_gold_pearson_eval": float(
                    evaluation[["proxy_score", "gold_score"]]
                    .corr()
                    .iloc[0, 1]
                ),
                "response_char_mean": float(
                    frame["response"].astype(str).str.len().mean()
                ),
                "A": models.coefficients.a,
                "B": models.coefficients.b,
                "A_nu": models.coefficients.a_nu,
                "n_calibration_prompts_used": (
                    models.coefficients.n_prompts_used
                ),
                "n_calibration_prompts_skipped": (
                    models.coefficients.n_prompts_skipped
                ),
            }
        )

    aggregate = pd.concat(aggregates, ignore_index=True)
    prompt = pd.concat(prompts, ignore_index=True)
    diagnostics_frame = pd.DataFrame(diagnostics)
    aggregate.to_csv(args.outdir / "predictions.csv", index=False)
    prompt.to_csv(
        args.outdir / "prompt_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    diagnostics_frame.drop(columns=["pool_path"]).to_csv(
        args.outdir / "pool_diagnostics.csv", index=False
    )
    pd.concat(reaches, ignore_index=True).to_csv(
        args.outdir / "reachability.csv", index=False
    )

    bootstrap_input = prompt.drop(columns=["model", "base_seed"])
    local = bootstrap_curve_mae(
        bootstrap_input, args.n_boot, args.random_seed
    )
    metadata = diagnostics_frame[["run", "model", "base_seed"]].rename(
        columns={"run": "pool"}
    )
    local = local.merge(metadata, on="pool", validate="many_to_one")
    local.to_csv(args.outdir / "local_curve_summary.csv", index=False)
    seed_summary = (
        local.groupby(["model", "baseline"], sort=True)["curve_mae"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_curve_mae",
                "std": "sd_across_seeds",
                "min": "min_curve_mae",
                "max": "max_curve_mae",
                "count": "n_seeds",
            }
        )
    )
    seed_summary.to_csv(args.outdir / "across_seed_summary.csv", index=False)
    make_figure(diagnostics_frame, local, seed_summary, args.outdir)

    spec = {
        "experiment": "exp28_modern_generator_control",
        "runs": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in record.items()
            }
            for record in records
        ],
        "prompt_identity_check": "exact prompt_id and prompt-text equality",
        "geometry": GEOMETRY,
        "d_values": D_VALUES,
        "local_window": "d <= 0.10",
        "fit_max_d": LOCAL_MAX_D,
        "random_seed": args.random_seed,
        "n_boot": args.n_boot,
        "baselines": BASELINES,
        "gold_definition": (
            "OpenAssistant/reward-model-deberta-v3-large-v2 model-based "
            "silver gold; no new human ratings"
        ),
        "source_sha256": {
            "analyze_runs.py": source_sha256(HERE / "analyze_runs.py"),
            "baseline_core.py": source_sha256(EXP25 / "baseline_core.py"),
            "analyze_size_control.py": source_sha256(
                EXP27 / "analyze_size_control.py"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(local.to_string(index=False))
    print(seed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
