#!/usr/bin/env python3
"""Audit and summarize exp24 cross-dataset transfer."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 0
ROUTE_LABELS = {
    "summeval_to_wmt": "SummEval to WMT MQM",
    "wmt_to_summeval": "WMT MQM to SummEval",
    "summeval_to_openmeva": "SummEval to OpenMEVA",
}
METHOD_LABELS = {
    "A+B": "A+B",
    "isotonic_calibration": "Isotonic",
    "spline_ridge": "Spline",
}


def paired_diff(frame: pd.DataFrame, baseline: str) -> dict:
    a = frame[frame["method"] == "A+B"][
        ["path_id", "repeat", "curve_mae_z"]
    ].rename(columns={"curve_mae_z": "err_a"})
    b = frame[frame["method"] == baseline][
        ["path_id", "repeat", "curve_mae_z"]
    ].rename(columns={"curve_mae_z": "err_b"})
    paired = a.merge(
        b, on=["path_id", "repeat"], validate="one_to_one"
    )
    by_path = paired.groupby("path_id")[["err_a", "err_b"]].mean()
    delta = (by_path["err_a"] - by_path["err_b"]).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boot[i] = np.mean(
            delta[rng.integers(0, len(delta), len(delta))]
        )
    return {
        "n_paths": len(delta),
        "n_paired_splits": len(paired),
        "mae_A+B": float(by_path["err_a"].mean()),
        "mae_baseline": float(by_path["err_b"].mean()),
        "diff_AplusB_minus_baseline": float(delta.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    splits = pd.read_csv(args.outdir / "ood_split_metrics.csv")
    key = [
        "route",
        "path_id",
        "repeat",
        "calibration_frac",
        "target_weight",
        "method",
    ]
    assert not splits.duplicated(key).any()
    assert set(splits["n_radii"]) == {5}
    assert np.isfinite(splits["curve_mae_z"]).all()

    summary = (
        splits.groupby(
            ["route", "calibration_frac", "target_weight", "method"]
        )
        .agg(
            curve_mae_z=("curve_mae_z", "mean"),
            n_paths=("path_id", "nunique"),
            n_splits=("repeat", "size"),
            max_source_grid_exceedance=(
                "source_grid_exceedance",
                "max",
            ),
        )
        .reset_index()
    )
    summary.to_csv(args.outdir / "ood_summary.csv", index=False)
    compact_table = (
        summary[
            np.isclose(summary["calibration_frac"], 0.10)
            & summary["target_weight"].isin([0.0, 0.75, 1.0])
        ]
        .pivot(
            index=["route", "calibration_frac", "target_weight"],
            columns="method",
            values="curve_mae_z",
        )
        .reset_index()
    )
    compact_table.to_csv(
        args.outdir / "ood_calibration_table.csv", index=False
    )
    (args.outdir / "ood_calibration_table.tex").write_text(
        compact_table.to_latex(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        encoding="utf-8",
    )

    rows = []
    for (route, frac, weight), frame in splits.groupby(
        ["route", "calibration_frac", "target_weight"], sort=False
    ):
        for baseline in ["isotonic_calibration", "spline_ridge"]:
            result = paired_diff(frame, baseline)
            result.update(
                {
                    "route": route,
                    "calibration_frac": frac,
                    "target_weight": weight,
                    "baseline": baseline,
                }
            )
            rows.append(result)
    paired = pd.DataFrame(rows)
    paired.to_csv(args.outdir / "ood_paired_bootstrap.csv", index=False)
    plot(summary, args.outdir / "fig_ood_transfer.png")

    compact = summary[
        summary["target_weight"].isin([0.0, 0.75, 1.0])
    ]
    print(compact.to_string(index=False))
    print("AUDIT PASS")


def plot(summary: pd.DataFrame, out: Path) -> None:
    routes = summary["route"].unique()
    fig, axes = plt.subplots(1, len(routes), figsize=(15, 4), sharey=False)
    for ax, route in zip(axes, routes):
        sub = summary[
            (summary["route"] == route)
            & np.isclose(summary["target_weight"], 0.75)
        ]
        for method in ["A+B", "isotonic_calibration", "spline_ridge"]:
            line = sub[sub["method"] == method].sort_values(
                "calibration_frac"
            )
            ax.plot(
                100 * line["calibration_frac"],
                line["curve_mae_z"],
                marker="o",
                label=METHOD_LABELS[method],
            )
        ax.set_title(ROUTE_LABELS.get(route, route))
        ax.set_xlabel("Target calibration (%)")
        ax.set_ylabel("Standardized CurveMAE")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Fixed target-heavy transfer blend (target weight = 0.75)")
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
