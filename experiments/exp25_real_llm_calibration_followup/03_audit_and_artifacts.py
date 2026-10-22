#!/usr/bin/env python3
"""Audit exp25 outputs and make compact analysis artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_core import BASELINES


PLOT_BASELINES = [
    "A+B",
    "empirical_fit",
    "isotonic_calibration",
    "spline_ridge",
    "mlp_calibration",
    "A-only",
    "A_nu",
]
LABELS = {
    "A+B": "A+B",
    "A-only": "A-only",
    "A_nu": "A_nu",
    "empirical_curve": "Empirical curve",
    "empirical_fit": "Empirical fit",
    "isotonic_calibration": "Isotonic",
    "binned_shrinkage": "Binned",
    "spline_ridge": "Spline",
    "mlp_calibration": "MLP",
    "zero": "Zero",
}


def within_artifacts(outdir: Path) -> None:
    predictions = pd.read_csv(outdir / "within_predictions.csv")
    manifest = pd.read_csv(outdir / "within_split_manifest.csv")
    spec = json.loads(
        (outdir / "within_run_spec.json").read_text(encoding="utf-8")
    )
    key = ["calibration_frac", "repeat", "d", "baseline"]
    assert not predictions.duplicated(key).any()
    assert set(predictions["baseline"]) == set(BASELINES)
    assert np.isfinite(
        predictions[
            ["predicted_gain", "actual_gain", "abs_error"]
        ].to_numpy(float)
    ).all()
    expected = (
        len(spec["calibration_fractions"])
        * spec["n_repeats"]
        * len(spec["d_values"])
        * len(BASELINES)
    )
    assert len(predictions) == expected
    role_counts = (
        manifest.groupby(["repeat", "calibration_frac", "role"])["prompt_id"]
        .nunique()
        .unstack()
    )
    assert (role_counts.sum(axis=1) == spec["input_audit"]["n_prompts"]).all()

    endpoint = (
        predictions[np.isclose(predictions["d"], 0.10)]
        .groupby(["calibration_frac", "baseline"])
        .agg(
            endpoint_mae=("abs_error", "mean"),
            endpoint_se=(
                "abs_error",
                lambda x: float(x.std(ddof=1) / np.sqrt(len(x))),
            ),
            mean_actual_gain=("actual_gain", "mean"),
        )
        .reset_index()
    )
    per_curve = (
        predictions[predictions["d"].le(0.10 + 1e-12)]
        .groupby(["calibration_frac", "repeat", "baseline"])["abs_error"]
        .mean()
        .rename("curve_mae")
        .reset_index()
    )
    curve = (
        per_curve.groupby(["calibration_frac", "baseline"])
        .agg(
            curve_mae=("curve_mae", "mean"),
            curve_se=(
                "curve_mae",
                lambda x: float(x.std(ddof=1) / np.sqrt(len(x))),
            ),
        )
        .reset_index()
    )
    radius = (
        predictions[np.isclose(predictions["calibration_frac"], 0.30)]
        .groupby(["d", "baseline"])
        .agg(
            mae=("abs_error", "mean"),
            se=(
                "abs_error",
                lambda x: float(x.std(ddof=1) / np.sqrt(len(x))),
            ),
        )
        .reset_index()
    )
    endpoint.to_csv(outdir / "within_endpoint_summary.csv", index=False)
    curve.to_csv(outdir / "within_curve_summary.csv", index=False)
    radius.to_csv(outdir / "within_radius_summary.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    a = per_curve[per_curve["baseline"].eq("A+B")][
        ["calibration_frac", "repeat", "curve_mae"]
    ].rename(columns={"curve_mae": "a_error"})
    for baseline in BASELINES:
        if baseline == "A+B":
            continue
        b = per_curve[per_curve["baseline"].eq(baseline)][
            ["calibration_frac", "repeat", "curve_mae"]
        ].rename(columns={"curve_mae": "b_error"})
        paired = a.merge(
            b,
            on=["calibration_frac", "repeat"],
            validate="one_to_one",
        )
        for frac, group in paired.groupby("calibration_frac"):
            delta = (group["a_error"] - group["b_error"]).to_numpy(float)
            rng = np.random.default_rng(0)
            boot = np.asarray(
                [
                    np.mean(delta[rng.integers(0, len(delta), len(delta))])
                    for _ in range(5000)
                ]
            )
            paired_rows.append(
                {
                    "calibration_frac": frac,
                    "baseline": baseline,
                    "A+B_minus_baseline": float(np.mean(delta)),
                    "ci_lo": float(np.quantile(boot, 0.025)),
                    "ci_hi": float(np.quantile(boot, 0.975)),
                    "n_repeats": len(delta),
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        outdir / "within_paired_bootstrap.csv", index=False
    )
    selected_methods = [
        "A+B",
        "empirical_fit",
        "isotonic_calibration",
        "spline_ridge",
        "mlp_calibration",
        "A-only",
        "A_nu",
    ]
    compact = (
        curve[
            curve["calibration_frac"].isin([0.05, 0.10, 0.30, 0.50])
            & curve["baseline"].isin(selected_methods)
        ]
        .pivot(
            index="calibration_frac",
            columns="baseline",
            values="curve_mae",
        )
        .reindex(columns=selected_methods)
        .reset_index()
        .rename(
            columns={
                "calibration_frac": "Calibration fraction",
                **{name: LABELS[name] for name in selected_methods},
            }
        )
    )
    compact["Calibration fraction"] = (
        100 * compact["Calibration fraction"]
    ).astype(int)
    compact.to_csv(outdir / "within_dataset_table.csv", index=False)
    (outdir / "within_dataset_table.tex").write_text(
        compact.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for baseline in PLOT_BASELINES:
        data = curve[curve["baseline"].eq(baseline)]
        axes[0].errorbar(
            100 * data["calibration_frac"],
            data["curve_mae"],
            yerr=data["curve_se"],
            marker="o",
            capsize=2,
            label=LABELS[baseline],
        )
        data = radius[radius["baseline"].eq(baseline)]
        axes[1].errorbar(
            data["d"],
            data["mae"],
            yerr=data["se"],
            marker="o",
            capsize=2,
            label=LABELS[baseline],
        )
    axes[0].set_xlabel("Calibration prompts (%)")
    axes[0].set_ylabel("CurveMAE, d <= 0.10")
    axes[0].set_title("Real LLM candidates: calibration budget")
    axes[1].axvspan(0, 0.10, color="#dcebf7", alpha=0.55)
    axes[1].axvline(0.10, color="#377eb8", linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("KL radius d")
    axes[1].set_ylabel("Endpoint MAE")
    axes[1].set_title("30% calibration: radius audit")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(
        outdir / "fig_real_llm_within_followup.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(outdir / "fig_real_llm_within_followup.png", dpi=180)
    plt.close(fig)


def bootstrap_transfer(group: pd.DataFrame, n_boot: int = 5000) -> dict:
    d_values = sorted(group["d"].unique())
    prompt_ids = sorted(group["prompt_id"].unique())
    lookup = {
        d: group[group["d"].eq(d)]
        .set_index("prompt_id")[["predicted_gain", "actual_gain"]]
        .loc[prompt_ids]
        .to_numpy(float)
        for d in d_values
    }
    point = float(
        np.mean(
            [
                abs(values[:, 0].mean() - values[:, 1].mean())
                for values in lookup.values()
            ]
        )
    )
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.integers(0, len(prompt_ids), len(prompt_ids))
        boot[index] = np.mean(
            [
                abs(
                    values[sampled, 0].mean()
                    - values[sampled, 1].mean()
                )
                for values in lookup.values()
            ]
        )
    return {
        "curve_mae": point,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "n_eval_prompts": len(prompt_ids),
        "n_radii": len(d_values),
    }


def bootstrap_transfer_difference(
    group: pd.DataFrame,
    baseline: str,
    n_boot: int = 5000,
) -> dict[str, float | int | str]:
    a = group[group["baseline"].eq("A+B")]
    b = group[group["baseline"].eq(baseline)]
    prompt_ids = sorted(a["prompt_id"].unique())
    d_values = sorted(a["d"].unique())
    a_lookup = {
        d: a[a["d"].eq(d)]
        .set_index("prompt_id")[["predicted_gain", "actual_gain"]]
        .loc[prompt_ids]
        .to_numpy(float)
        for d in d_values
    }
    b_lookup = {
        d: b[b["d"].eq(d)]
        .set_index("prompt_id")[["predicted_gain", "actual_gain"]]
        .loc[prompt_ids]
        .to_numpy(float)
        for d in d_values
    }

    def difference(indices: np.ndarray) -> float:
        a_error = np.mean(
            [
                abs(
                    values[indices, 0].mean()
                    - values[indices, 1].mean()
                )
                for values in a_lookup.values()
            ]
        )
        b_error = np.mean(
            [
                abs(
                    values[indices, 0].mean()
                    - values[indices, 1].mean()
                )
                for values in b_lookup.values()
            ]
        )
        return float(a_error - b_error)

    full = np.arange(len(prompt_ids))
    point = difference(full)
    rng = np.random.default_rng(0)
    boot = np.asarray(
        [
            difference(
                rng.integers(0, len(prompt_ids), size=len(prompt_ids))
            )
            for _ in range(n_boot)
        ]
    )
    return {
        "baseline": baseline,
        "A+B_minus_baseline": point,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "prob_A+B_better": float(np.mean(boot < 0)),
        "n_eval_prompts": len(prompt_ids),
        "n_radii": len(d_values),
    }


def transfer_artifacts(outdir: Path) -> None:
    predictions = pd.read_csv(outdir / "transfer_predictions.csv")
    prompt = pd.read_csv(outdir / "transfer_prompt_predictions.csv.gz")
    reachability = pd.read_csv(outdir / "transfer_reachability.csv")
    key = ["route", "estimation_scope", "d", "baseline"]
    assert not predictions.duplicated(key).any()
    assert set(predictions["baseline"]) == set(BASELINES)
    assert np.isfinite(
        predictions[
            ["predicted_gain", "actual_gain", "abs_error"]
        ].to_numpy(float)
    ).all()
    prompt_key = key + ["prompt_id"]
    assert not prompt.duplicated(prompt_key).any()
    assert not reachability.duplicated(["pool", "split", "d"]).any()
    assert reachability["reachable_fraction"].between(0, 1).all()

    local = prompt[prompt["d"].le(0.10 + 1e-12)]
    rows: list[dict[str, object]] = []
    for keys, group in local.groupby(
        ["route", "estimation_scope", "baseline"]
    ):
        route, scope, baseline = keys
        row = bootstrap_transfer(group)
        row.update(
            {
                "route": route,
                "estimation_scope": scope,
                "baseline": baseline,
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["route", "estimation_scope", "curve_mae"]
    )
    summary.to_csv(outdir / "transfer_local_curve_summary.csv", index=False)
    paired_rows: list[dict[str, object]] = []
    for keys, group in local.groupby(["route", "estimation_scope"]):
        route, scope = keys
        for baseline in BASELINES:
            if baseline == "A+B":
                continue
            row = bootstrap_transfer_difference(group, baseline)
            row.update({"route": route, "estimation_scope": scope})
            paired_rows.append(row)
    pd.DataFrame(paired_rows).sort_values(
        ["route", "estimation_scope", "baseline"]
    ).to_csv(outdir / "transfer_paired_bootstrap.csv", index=False)
    selected_methods = [
        "A+B",
        "empirical_fit",
        "isotonic_calibration",
        "spline_ridge",
        "mlp_calibration",
        "A-only",
        "A_nu",
    ]
    compact = (
        summary[
            summary["estimation_scope"].eq("source_only")
            & summary["baseline"].isin(selected_methods)
        ]
        .pivot(index="route", columns="baseline", values="curve_mae")
        .reindex(columns=selected_methods)
        .reset_index()
        .rename(
            columns={
                "route": "Transfer route",
                **{name: LABELS[name] for name in selected_methods},
            }
        )
    )
    compact.to_csv(outdir / "transfer_table.csv", index=False)
    (outdir / "transfer_table.tex").write_text(
        compact.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )

    source = summary[
        summary["estimation_scope"].eq("source_only")
        & summary["baseline"].isin(PLOT_BASELINES)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    for axis, route in zip(
        axes, ["qwen_to_zephyr", "zephyr_to_qwen"]
    ):
        data = source[source["route"].eq(route)].copy()
        order = [b for b in PLOT_BASELINES if b in set(data["baseline"])]
        data = data.set_index("baseline").loc[order].reset_index()
        positions = np.arange(len(data))
        axis.bar(
            positions,
            data["curve_mae"],
            color=plt.get_cmap("tab10").colors[: len(data)],
        )
        axis.errorbar(
            positions,
            data["curve_mae"],
            yerr=[
                data["curve_mae"] - data["ci_lo"],
                data["ci_hi"] - data["curve_mae"],
            ],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )
        axis.set_xticks(positions, [LABELS[b] for b in order], rotation=25)
        axis.set_title(route.replace("_", " "))
        axis.set_xlabel("Source-calibrated predictor")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("CurveMAE, d <= 0.10")
    fig.suptitle("Cross-model transfer on matched UltraFeedback prompts")
    fig.tight_layout()
    fig.savefig(
        outdir / "fig_real_llm_cross_model_transfer.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(outdir / "fig_real_llm_cross_model_transfer.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--within-outdir", type=Path, required=True)
    parser.add_argument("--transfer-outdir", type=Path, required=True)
    args = parser.parse_args()
    within_artifacts(args.within_outdir)
    transfer_artifacts(args.transfer_outdir)
    print("exp25 audits and artifacts completed")


if __name__ == "__main__":
    main()
