#!/usr/bin/env python3
"""Audit the claims used to position the local coefficient law.

This script reads completed direct-calibration, B-stability, and DPO outputs.
It does not refit any model or modify its inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "path_id", "repeat", "d"]
DIRECT_METHODS = [
    "A-only",
    "A+B",
    "empirical_fit",
    "binned_shrinkage",
    "spline_ridge",
    "isotonic_calibration",
    "mlp_calibration",
]

DIRECT_COMPLEXITY = [
    {
        "method": "A-only",
        "fitted_representation": "1 closed-form coefficient",
        "candidate_configurations": 1,
        "selection": "none",
        "additional_structure": "calibration moments of proxy and gold",
    },
    {
        "method": "A+B",
        "fitted_representation": "2 closed-form coefficients",
        "candidate_configurations": 1,
        "selection": "none",
        "additional_structure": "calibration moments of proxy and gold",
    },
    {
        "method": "empirical_fit",
        "fitted_representation": "2 fitted coefficients",
        "candidate_configurations": 1,
        "selection": "least squares on the calibration gain curve",
        "additional_structure": "exact calibration curve on the radius grid",
    },
    {
        "method": "binned_shrinkage",
        "fitted_representation": "3-12 fitted bin values",
        "candidate_configurations": 12,
        "selection": "calibration-internal K-fold CV",
        "additional_structure": "quantile bins; 4 bin counts x 3 shrinkages",
    },
    {
        "method": "spline_ridge",
        "fitted_representation": "6/8/10 cubic-spline basis values + intercept",
        "candidate_configurations": 39,
        "selection": "calibration-internal K-fold CV",
        "additional_structure": "3 knot counts x 13 ridge strengths",
    },
    {
        "method": "isotonic_calibration",
        "fitted_representation": "data-adaptive monotone steps",
        "candidate_configurations": 1,
        "selection": "no tuned hyperparameter",
        "additional_structure": "at most one threshold per distinct proxy score",
    },
    {
        "method": "mlp_calibration",
        "fitted_representation": "1-16-16-1 ReLU network; 321 parameters",
        "candidate_configurations": 1,
        "selection": "none; pre-specified architecture and 1000-step Adam budget",
        "additional_structure": "standardized proxy input and gold target",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_outdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_dpo(primary_path: Path, binned_path: Path) -> pd.DataFrame:
    required = {
        "seed",
        "tag",
        "model",
        "d_med",
        "ess_med",
        "prediction",
        "frontier",
        "dpo_realized",
    }
    primary = pd.read_csv(primary_path)
    binned = pd.read_csv(binned_path)
    for name, frame in [("primary", primary), ("binned", binned)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} DPO input is missing columns: {missing}")
        if frame.duplicated(["seed", "tag", "model"]).any():
            raise ValueError(f"{name} DPO input has duplicate checkpoint keys")
    overlap = set(primary["model"]) & set(binned["model"])
    if overlap:
        raise ValueError(f"DPO model sets overlap: {sorted(overlap)}")
    data = pd.concat([primary, binned], ignore_index=True)
    identity = (
        (data["prediction"] - data["frontier"])
        + (data["frontier"] - data["dpo_realized"])
        - (data["prediction"] - data["dpo_realized"])
    )
    if float(identity.abs().max()) > 1e-12:
        raise ValueError("DPO error-decomposition identity failed")
    return data


def summarize_dpo(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "strict_local_d_le_0p10": data["d_med"] <= 0.10 + 1e-12,
        "historical_six_checkpoints": (
            (data["d_med"] <= 1.0) & (data["ess_med"] >= 8.0)
        ),
    }
    for scope, mask in scopes.items():
        local = data.loc[mask].copy()
        local["frontier_error"] = local["prediction"] - local["frontier"]
        local["optimizer_gap"] = local["frontier"] - local["dpo_realized"]
        local["total_error"] = local["prediction"] - local["dpo_realized"]
        local["opposes_gap"] = (
            np.sign(local["frontier_error"])
            * np.sign(local["optimizer_gap"])
            < 0
        )
        for model, frame in local.groupby("model", sort=False):
            seed_level = (
                frame.groupby("seed", as_index=False)
                .agg(
                    n_checkpoints=("tag", "size"),
                    frontier_signed_bias=("frontier_error", "mean"),
                    frontier_mae=("frontier_error", lambda x: x.abs().mean()),
                    optimizer_gap_mean=("optimizer_gap", "mean"),
                    optimizer_gap_mae=(
                        "optimizer_gap",
                        lambda x: x.abs().mean(),
                    ),
                    realized_signed_bias=("total_error", "mean"),
                    realized_mae=("total_error", lambda x: x.abs().mean()),
                    opposition_rate=("opposes_gap", "mean"),
                )
            )
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "n_checkpoints_total": len(frame),
                    "min_checkpoints_per_seed": int(
                        seed_level["n_checkpoints"].min()
                    ),
                    "max_checkpoints_per_seed": int(
                        seed_level["n_checkpoints"].max()
                    ),
                    "n_seeds": len(seed_level),
                    "frontier_signed_bias": seed_level[
                        "frontier_signed_bias"
                    ].mean(),
                    "frontier_mae": seed_level["frontier_mae"].mean(),
                    "frontier_mae_sd": seed_level["frontier_mae"].std(ddof=1),
                    "optimizer_gap_mean": seed_level[
                        "optimizer_gap_mean"
                    ].mean(),
                    "optimizer_gap_mae": seed_level[
                        "optimizer_gap_mae"
                    ].mean(),
                    "realized_signed_bias": seed_level[
                        "realized_signed_bias"
                    ].mean(),
                    "realized_mae": seed_level["realized_mae"].mean(),
                    "realized_mae_sd": seed_level["realized_mae"].std(ddof=1),
                    "opposition_rate": seed_level["opposition_rate"].mean(),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(["scope", "realized_mae", "model"])


def load_direct(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    required = set(KEYS) | {
        "baseline",
        "actual_gain",
        "predicted_gain",
        "abs_error",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"direct-calibration input missing columns: {missing}")
    if data.duplicated(KEYS + ["baseline"]).any():
        raise ValueError("duplicate direct-calibration prediction key")
    recomputed = (data["predicted_gain"] - data["actual_gain"]).abs()
    if float((recomputed - data["abs_error"]).abs().max()) > 1e-10:
        raise ValueError("stored direct-calibration absolute error mismatch")
    return data


def summarize_direct(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[data["baseline"].isin(DIRECT_METHODS)].copy()
    summary = (
        selected.groupby(["dataset", "baseline"], as_index=False)
        .agg(curve_mae=("abs_error", "mean"), n_rows=("abs_error", "size"))
    )
    pivot = summary.pivot(
        index="dataset", columns="baseline", values="curve_mae"
    )
    rows = []
    for dataset, values in pivot.iterrows():
        a_only = float(values["A-only"])
        ab = float(values["A+B"])
        fit = float(values["empirical_fit"])
        rows.append(
            {
                "dataset": dataset,
                "a_only_curve_mae": a_only,
                "ab_curve_mae": ab,
                "empirical_two_term_curve_mae": fit,
                "b_relative_mae_reduction_vs_a_only": (a_only - ab) / a_only,
                "ab_minus_empirical_two_term_mae": ab - fit,
                "binned_curve_mae": float(values["binned_shrinkage"]),
                "spline_curve_mae": float(values["spline_ridge"]),
                "isotonic_curve_mae": float(
                    values["isotonic_calibration"]
                ),
                "mlp_curve_mae": float(values["mlp_calibration"]),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset")


def direct_complexity_table() -> pd.DataFrame:
    return pd.DataFrame(DIRECT_COMPLEXITY)


def dpo_complexity_table(
    fit_metadata_path: Path, binned_metadata_path: Path
) -> pd.DataFrame:
    fit = pd.read_csv(fit_metadata_path)
    binned = pd.read_csv(binned_metadata_path)
    fit = fit[fit["fit_split"] == "calibration"].copy()
    binned = binned[binned["fit_split"] == "calibration"].copy()
    if set(fit["model"]) != {
        "cal_raw_isotonic",
        "cal_raw_spline",
        "cal_centered_isotonic",
        "cal_centered_spline",
        "cal_zscore_isotonic",
        "cal_zscore_spline",
        "cal_rank_isotonic",
        "cal_rank_spline",
    }:
        raise ValueError("unexpected calibration models in DPO fit metadata")
    if set(binned["model"]) != {
        "cal_raw_binned",
        "cal_centered_binned",
        "cal_zscore_binned",
        "cal_rank_binned",
    }:
        raise ValueError("unexpected calibration models in DPO binned metadata")

    rows = []
    for row in fit.itertuples(index=False):
        if row.model.endswith("_isotonic"):
            representation = f"{int(row.n_thresholds)} monotone thresholds"
            candidates = 1
            selection = "no tuned hyperparameter"
        else:
            knots = int(row.n_knots)
            basis = knots + 2
            representation = (
                f"{knots} knots / {basis} cubic-spline basis values + intercept"
            )
            candidates = 39
            selection = (
                f"{row.cv_scheme}; selected ridge alpha={row.ridge_alpha:g}"
            )
        rows.append(
            {
                "model": row.model,
                "fit_prompts": int(row.n_prompts),
                "fit_candidates": int(row.n_candidates),
                "fitted_representation": representation,
                "candidate_configurations": candidates,
                "selection": selection,
                "coordinate_information": row.model.removeprefix("cal_").rsplit(
                    "_", 1
                )[0],
            }
        )
    for row in binned.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "fit_prompts": int(row.n_prompts),
                "fit_candidates": int(row.n_candidates),
                "fitted_representation": (
                    f"{int(row.n_bins_actual)} fitted bin values"
                ),
                "candidate_configurations": 12,
                "selection": (
                    f"{row.cv_scheme}; selected shrinkage={row.shrink_k:g}"
                ),
                "coordinate_information": row.model.removeprefix("cal_").rsplit(
                    "_", 1
                )[0],
            }
        )
    return pd.DataFrame(rows).sort_values("model")


def summarize_rich_disagreement(data: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "A+B",
        "isotonic_calibration",
        "spline_ridge",
        "mlp_calibration",
    ]
    selected = data[data["baseline"].isin(methods)]
    pivot = selected.pivot(
        index=KEYS, columns="baseline", values="predicted_gain"
    ).reset_index()
    rows = []
    for method in [
        "isotonic_calibration",
        "spline_ridge",
        "mlp_calibration",
    ]:
        pivot["difference"] = (pivot[method] - pivot["A+B"]).abs()
        unit = (
            pivot.groupby(["dataset", "path_id", "repeat"], as_index=False)
            ["difference"]
            .max()
        )
        for dataset, frame in unit.groupby("dataset"):
            values = frame["difference"].to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "direct_method": method,
                    "n_path_repeat_units": len(values),
                    "mean_sup_curve_disagreement": float(values.mean()),
                    "median_sup_curve_disagreement": float(
                        np.median(values)
                    ),
                    "p90_sup_curve_disagreement": float(
                        np.quantile(values, 0.90)
                    ),
                    "max_sup_curve_disagreement": float(values.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "direct_method"])


def build_gate_rows(
    direct: pd.DataFrame, stability_path: Path
) -> pd.DataFrame:
    stability = pd.read_csv(stability_path)
    required = set(KEYS) | {"b_ci_includes_zero"}
    missing = required - set(stability.columns)
    if missing:
        raise ValueError(f"B-stability input missing columns: {missing}")
    flags = stability[KEYS + ["b_ci_includes_zero"]].drop_duplicates(KEYS)
    if len(flags) != len(stability):
        raise ValueError("B-stability input has duplicate prediction keys")

    selected = direct[direct["baseline"].isin(["A+B", "A-only"])].copy()
    actual_counts = selected.groupby(KEYS)["actual_gain"].nunique()
    if int(actual_counts.max()) != 1:
        raise ValueError("actual gain differs between A+B and A-only")
    predictions = selected.pivot(
        index=KEYS, columns="baseline", values="predicted_gain"
    ).reset_index()
    actual = selected.groupby(KEYS, as_index=False)["actual_gain"].first()
    rows = predictions.merge(actual, on=KEYS, validate="one_to_one")
    rows = rows.merge(flags, on=KEYS, validate="one_to_one")
    if len(rows) != len(flags):
        raise ValueError(
            f"gate join lost rows: predictions={len(rows)}, flags={len(flags)}"
        )
    rows["gated_prediction"] = np.where(
        rows["b_ci_includes_zero"], rows["A-only"], rows["A+B"]
    )
    rows["a_only_abs_error"] = (
        rows["A-only"] - rows["actual_gain"]
    ).abs()
    rows["ab_abs_error"] = (rows["A+B"] - rows["actual_gain"]).abs()
    rows["gated_abs_error"] = (
        rows["gated_prediction"] - rows["actual_gain"]
    ).abs()
    return rows


def summarize_gate(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("dataset", as_index=False)
        .agg(
            n_rows=("gated_abs_error", "size"),
            n_paths=("path_id", "nunique"),
            gate_to_a_only_rate=("b_ci_includes_zero", "mean"),
            a_only_curve_mae=("a_only_abs_error", "mean"),
            ab_curve_mae=("ab_abs_error", "mean"),
            gated_curve_mae=("gated_abs_error", "mean"),
        )
        .sort_values("dataset")
    )


def bootstrap_gate(
    rows: pd.DataFrame, n_boot: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = []
    comparisons = {
        "ab_minus_gated": ("ab_abs_error", "gated_abs_error"),
        "a_only_minus_gated": ("a_only_abs_error", "gated_abs_error"),
    }
    for dataset, frame in rows.groupby("dataset", sort=True):
        path_ids = np.array(sorted(frame["path_id"].unique()), dtype=object)
        cluster_sizes = frame.groupby("path_id").size()
        if cluster_sizes.nunique() != 1:
            raise ValueError(
                f"{dataset}: unequal path cluster sizes require weighted "
                "cluster bootstrap"
            )
        sampled_indices = rng.integers(
            0, len(path_ids), size=(n_boot, len(path_ids))
        )
        for name, (left, right) in comparisons.items():
            path_differences = (
                frame.assign(difference=frame[left] - frame[right])
                .groupby("path_id")["difference"]
                .mean()
                .reindex(path_ids)
                .to_numpy()
            )
            values = path_differences[sampled_indices].mean(axis=1)
            observed = float((frame[left] - frame[right]).mean())
            output.append(
                {
                    "dataset": dataset,
                    "comparison": name,
                    "observed_mae_difference": observed,
                    "ci_lo": float(np.quantile(values, 0.025)),
                    "ci_hi": float(np.quantile(values, 0.975)),
                    "n_boot": n_boot,
                    "bootstrap_unit": "path_id",
                }
            )
    return pd.DataFrame(output)


def write_report(
    path: Path,
    dpo: pd.DataFrame,
    direct: pd.DataFrame,
    rich: pd.DataFrame,
    gate: pd.DataFrame,
    gate_boot: pd.DataFrame,
    direct_complexity: pd.DataFrame,
    dpo_complexity: pd.DataFrame,
) -> None:
    historical = dpo[dpo["scope"] == "historical_six_checkpoints"]
    strict = dpo[dpo["scope"] == "strict_local_d_le_0p10"]
    focus = [
        "A-only",
        "A+B",
        "A_nu",
        "cal_raw_spline",
        "cal_rank_spline",
        "cal_rank_isotonic",
        "cal_zscore_binned",
        "cal_rank_binned",
        "Zero",
    ]

    def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
        local = frame[columns].copy()
        for column in local.select_dtypes(include=[np.number]).columns:
            local[column] = local[column].map(lambda x: f"{x:.5f}")
        return local.to_markdown(index=False)

    parts = [
        "# Method Positioning Audit",
        "",
        "## DPO signed-bias decomposition",
        "",
        "The identity is `prediction - realized = "
        "(prediction - frontier) + (frontier - realized)`.",
        "A negative frontier bias opposes the positive optimizer gap.",
        "",
        "### Historical six checkpoints",
        "",
        md_table(
            historical[historical["model"].isin(focus)],
            [
                "model",
                "frontier_signed_bias",
                "optimizer_gap_mean",
                "realized_signed_bias",
                "frontier_mae",
                "realized_mae",
                "opposition_rate",
            ],
        ),
        "",
        "### Pre-specified local window (`d <= 0.10`)",
        "",
        md_table(
            strict[strict["model"].isin(focus)],
            [
                "model",
                "frontier_signed_bias",
                "optimizer_gap_mean",
                "realized_signed_bias",
                "frontier_mae",
                "realized_mae",
                "opposition_rate",
            ],
        ),
        "",
        "The cancellation account is supported when frontier bias is negative "
        "and opposition rate is high. It does not imply that the method models "
        "optimizer realization.",
        "",
        "## Closed-form versus freely fitted two-term law",
        "",
        md_table(
            direct,
            [
                "dataset",
                "a_only_curve_mae",
                "ab_curve_mae",
                "empirical_two_term_curve_mae",
                "binned_curve_mae",
                "spline_curve_mae",
                "isotonic_curve_mae",
                "mlp_curve_mae",
                "b_relative_mae_reduction_vs_a_only",
                "ab_minus_empirical_two_term_mae",
            ],
        ),
        "",
        "## Rich-calibrator disagreement with A+B",
        "",
        md_table(
            rich,
            [
                "dataset",
                "direct_method",
                "median_sup_curve_disagreement",
                "p90_sup_curve_disagreement",
                "max_sup_curve_disagreement",
            ],
        ),
        "",
        "This is the descriptive supremum disagreement over the checked radius "
        "grid. A formal adequacy gate would additionally require a joint "
        "bootstrap band for this difference; these values should not be given "
        "a universal threshold.",
        "",
        "## Hard uncertainty gate for B",
        "",
        md_table(
            gate,
            [
                "dataset",
                "gate_to_a_only_rate",
                "a_only_curve_mae",
                "ab_curve_mae",
                "gated_curve_mae",
            ],
        ),
        "",
        md_table(
            gate_boot,
            [
                "dataset",
                "comparison",
                "observed_mae_difference",
                "ci_lo",
                "ci_hi",
            ],
        ),
        "",
        "The hard CI gate is an uncertainty-reporting and decision-safety "
        "device, not an MAE improvement in these data. It falls between A-only "
        "and unconditional A+B and should not be advertised as a predictive "
        "win.",
        "",
        "> **Direct-calibration information setting.** The direct flexible-calibrator",
        "> rows in this report use target-pool-aware exp21 outputs. Use exp33 for the",
        "> pre-pool case with 300 rated items and 1,000 future candidates not yet",
        "> collected.",
        "",
        "## Model-complexity accounting",
        "",
        "Complexity is reported as fitted representation size and model-selection "
        "search size. Isotonic complexity is data-adaptive, so its fitted step "
        "count is reported rather than calling it a fixed-parameter model.",
        "",
        "### Direct calibration",
        "",
        direct_complexity.to_markdown(index=False),
        "",
        "### DPO calibration models selected on 100 calibration prompts",
        "",
        dpo_complexity.to_markdown(index=False),
        "",
        "Prompt-centered, z-score, and rank coordinates use prompt grouping "
        "during calibration. They do not change the raw-score exponential-tilt "
        "weights used to define the evaluated frontier.",
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-checkpoints", type=Path, required=True)
    parser.add_argument("--dpo-binned-checkpoints", type=Path, required=True)
    parser.add_argument("--dpo-fit-metadata", type=Path, required=True)
    parser.add_argument("--dpo-binned-fit-metadata", type=Path, required=True)
    parser.add_argument(
        "--direct-predictions", type=Path, nargs="+", required=True
    )
    parser.add_argument("--b-stability-rows", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    fresh_outdir(args.outdir)
    dpo_raw = load_dpo(args.dpo_checkpoints, args.dpo_binned_checkpoints)
    dpo_summary = summarize_dpo(dpo_raw)
    direct_raw = load_direct(args.direct_predictions)
    direct_summary = summarize_direct(direct_raw)
    rich_summary = summarize_rich_disagreement(direct_raw)
    gate_rows = build_gate_rows(direct_raw, args.b_stability_rows)
    gate_summary = summarize_gate(gate_rows)
    gate_boot = bootstrap_gate(
        gate_rows, args.bootstrap_reps, args.random_seed
    )
    direct_complexity = direct_complexity_table()
    dpo_complexity = dpo_complexity_table(
        args.dpo_fit_metadata, args.dpo_binned_fit_metadata
    )

    dpo_summary.to_csv(args.outdir / "dpo_signed_bias_summary.csv", index=False)
    direct_summary.to_csv(
        args.outdir / "direct_positioning_summary.csv", index=False
    )
    rich_summary.to_csv(
        args.outdir / "rich_curve_disagreement_summary.csv", index=False
    )
    gate_summary.to_csv(
        args.outdir / "b_uncertainty_gate_summary.csv", index=False
    )
    gate_boot.to_csv(
        args.outdir / "b_uncertainty_gate_bootstrap.csv", index=False
    )
    direct_complexity.to_csv(
        args.outdir / "direct_model_complexity.csv", index=False
    )
    dpo_complexity.to_csv(
        args.outdir / "dpo_model_complexity.csv", index=False
    )
    write_report(
        args.outdir / "RESULTS.md",
        dpo_summary,
        direct_summary,
        rich_summary,
        gate_summary,
        gate_boot,
        direct_complexity,
        dpo_complexity,
    )

    inputs = [
        args.dpo_checkpoints,
        args.dpo_binned_checkpoints,
        args.dpo_fit_metadata,
        args.dpo_binned_fit_metadata,
        *args.direct_predictions,
        args.b_stability_rows,
    ]
    spec = {
        "bootstrap_reps": args.bootstrap_reps,
        "random_seed": args.random_seed,
        "input_files": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in inputs
        ],
        "code_sha256": sha256_file(Path(__file__)),
    }
    (args.outdir / "run_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    (args.outdir / "WORKFLOW_COMPLETE").write_text("PASS\n", encoding="ascii")
    print(dpo_summary.to_string(index=False))
    print("\nDirect positioning:")
    print(direct_summary.to_string(index=False))
    print("\nB uncertainty gate:")
    print(gate_summary.to_string(index=False))


if __name__ == "__main__":
    main()
