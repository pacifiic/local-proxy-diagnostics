#!/usr/bin/env python3
"""Validate the two completed WMT input lineages and write a compact report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LINEAGES = ("submitted_139", "fixed_141")
EXPECTED_HASHES = {
    "submitted_139": "5649ee3041340f7ee96c3079a4f1f859b9ddd625b2d82f731407d32fd88553c0",
    "fixed_141": "6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8",
}
EXPECTED_PATHS = {
    "exp02_filter_min400": {"submitted_139": 132, "fixed_141": 141},
    "exp02_filter_min100": {"submitted_139": 139, "fixed_141": 141},
    "exp03_calibration_crossover": {"submitted_139": 139, "fixed_141": 141},
    "exp14_conditional_tilt": {"submitted_139": 132, "fixed_141": 141},
    "exp16_fallback_rule": {"submitted_139": 132, "fixed_141": 141},
    "exp18_calbudget": {"submitted_139": 132, "fixed_141": 141},
    "exp20_bon_window_gate": {"submitted_139": 139, "fixed_141": 141},
    "exp21_direct_calibration": {"submitted_139": 139, "fixed_141": 141},
    "exp22_b_stability": {"submitted_139": 139, "fixed_141": 141},
    "exp23_mixture_ranktilt": {"submitted_139": 139, "fixed_141": 141},
    "exp24_within_dataset": {"submitted_139": 139, "fixed_141": 141},
    "exp32_evaluation_aware_ab": {"submitted_139": 139, "fixed_141": 141},
    "exp33_calibration_only": {"submitted_139": 139, "fixed_141": 141},
    "paper_exp19_direct_validation": {"submitted_139": 139, "fixed_141": 141},
    "paper_exp23_monotone": {"submitted_139": 139, "fixed_141": 141},
    "paper_exp24_rho_scaling": {"submitted_139": 139, "fixed_141": 141},
    "paper_exp27_prospective": {"submitted_139": 139, "fixed_141": 141},
    "paper_exp28_stability_tail": {"submitted_139": 139, "fixed_141": 141},
    "paper_exp32_proxy_perturbation": {"submitted_139": 139, "fixed_141": 141},
    "paper_tail_outlier_robustness": {
        "submitted_139": 139,
        "fixed_141": 141,
    },
    "paper_calibration_size": {"submitted_139": 139, "fixed_141": 141},
    "cubic_moment": {"submitted_139": 139, "fixed_141": 141},
    "paper_ood_zboth_summeval_to_wmt": {
        "submitted_139": 139,
        "fixed_141": 141,
    },
}
PATH_FILES = {
    "exp02_filter_min400": ("exp27_prospective_paths.csv", "path_id"),
    "exp02_filter_min100": ("exp27_prospective_paths.csv", "path_id"),
    "exp03_calibration_crossover": ("splits_wmt.csv", "path_id"),
    "exp14_conditional_tilt": ("conditional_per_path.csv", "path"),
    "exp16_fallback_rule": ("fallback_per_path.csv", "path"),
    "exp18_calbudget": ("calbudget_cells.csv", "path"),
    "exp20_bon_window_gate": ("bon_gate_points.csv", "path_id"),
    "exp21_direct_calibration": (
        "direct_calibration_predictions.csv",
        "path_id",
    ),
    "exp22_b_stability": ("b_stability_rows.csv", "path_id"),
    "exp23_mixture_ranktilt": ("mixture_ranktilt_rows.csv", "path_id"),
    "exp24_within_dataset": ("within_split_metrics.csv", "path_id"),
    "exp32_evaluation_aware_ab": (
        "direct_calibration_predictions.csv",
        "path_id",
    ),
    "exp33_calibration_only": (
        "calibration_only_predictions.csv",
        "path_id",
    ),
    "paper_exp19_direct_validation": (
        "direct_validation_residual_metrics.csv",
        "path",
    ),
    "paper_exp23_monotone": (
        "exp23_monotone_paths.csv",
        ("group_id", "heldout_rater"),
    ),
    "paper_exp24_rho_scaling": (
        "exp24_scaling_paths.csv",
        ("group_id", "heldout_rater"),
    ),
    "paper_exp27_prospective": ("exp27_prospective_paths.csv", "path_id"),
    "paper_exp28_stability_tail": ("exp28_stability_paths.csv", "path_id"),
    "paper_exp32_proxy_perturbation": (
        "exp32_proxy_baseline_paths.csv",
        "path_id",
    ),
    "paper_tail_outlier_robustness": (
        "tail_outlier_robustness_paths.csv",
        "path_id",
    ),
    "paper_calibration_size": ("calibration_size_paths_long.csv", "path_id"),
    "cubic_moment": (
        "direct_calibration_predictions.csv",
        "path_id",
    ),
    "paper_ood_zboth_summeval_to_wmt": (
        "exp31_ood_transfer_paths.csv",
        "target_path_id",
    ),
}

EXPECTED_DATASETS = {"submitted_139": 45, "fixed_141": 47}
EXPECTED_TASK_COUNT = 43


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_status(lineage_root: Path) -> dict[str, int]:
    status = {}
    for record_path in sorted(lineage_root.glob("*/task_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        status[record_path.parent.name] = int(record["returncode"])
    retry = lineage_root / "exp14_conditional_audit" / "retry_record.json"
    if retry.is_file() and not json.loads(retry.read_text())["returncode"]:
        status["exp14_conditional_audit"] = 0
    return status


def task_records(lineage_root: Path) -> dict[str, dict]:
    records = {}
    for record_path in sorted(lineage_root.glob("*/task_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        records[record_path.parent.name] = record
    return records


def path_counts(root: Path) -> tuple[list[dict], list[str]]:
    rows = []
    failures = []
    for experiment, (filename, columns) in PATH_FILES.items():
        row = {"experiment": experiment}
        for lineage in LINEAGES:
            path = root / lineage / experiment / "results" / filename
            usecols = [columns] if isinstance(columns, str) else list(columns)
            frame = pd.read_csv(path, usecols=usecols)
            count = int(len(frame.drop_duplicates()))
            row[lineage] = count
            expected = EXPECTED_PATHS[experiment][lineage]
            if count != expected:
                failures.append(
                    f"{lineage}/{experiment}: paths={count}, expected={expected}"
                )
        rows.append(row)
    return rows, failures


def endpoint_metrics(root: Path) -> list[dict]:
    rows = []
    for lineage in LINEAGES:
        path = (
            root
            / lineage
            / "exp21_direct_calibration"
            / "results"
            / "direct_calibration_predictions.csv"
        )
        frame = pd.read_csv(path, usecols=["d", "baseline", "abs_error"])
        local = frame[np.isclose(frame["d"], 0.10)]
        means = local.groupby("baseline")["abs_error"].mean()
        rows.append(
            {
                "lineage": lineage,
                "A+B": float(means["A+B"]),
                "A-only": float(means["A-only"]),
                "A_nu": float(means["A_nu"]),
                "zero": float(means["zero"]),
                "isotonic": float(means["isotonic_calibration"]),
                "spline": float(means["spline_ridge"]),
            }
        )
    return rows


def addon_dataset_counts(root: Path) -> tuple[list[dict], list[str]]:
    """Check WMT dataset accounting for addon analyses without path IDs."""
    rows = []
    failures = []
    files = {
        "paper_exp21_reliability": "wmt_mqm_reliability.csv",
        "paper_exp25_measurement_intervention": (
            "exp25_intervention_pair_summary.csv"
        ),
    }
    for experiment, filename in files.items():
        row = {"experiment": experiment}
        for lineage in LINEAGES:
            path = root / lineage / experiment / "results" / filename
            count = len(pd.read_csv(path))
            row[lineage] = count
            expected = EXPECTED_DATASETS[lineage]
            if count != expected:
                failures.append(
                    f"{lineage}/{experiment}: datasets={count}, "
                    f"expected={expected}"
                )
        rows.append(row)
    return rows, failures


def ood_metrics(root: Path) -> list[dict]:
    rows = []
    for lineage in LINEAGES:
        path = (
            root
            / lineage
            / "exp24_ood_transfer"
            / "results"
            / "ood_split_metrics.csv"
        )
        frame = pd.read_csv(
            path,
            usecols=[
                "route",
                "calibration_frac",
                "target_weight",
                "method",
                "curve_mae_z",
            ],
        )
        selected = frame[
            np.isclose(frame["calibration_frac"], 0.30)
            & np.isclose(frame["target_weight"], 0.75)
            & frame["method"].eq("A+B")
        ]
        for route, value in selected.groupby("route")["curve_mae_z"].mean().items():
            rows.append(
                {
                    "lineage": lineage,
                    "route": route,
                    "curve_mae_z": float(value),
                }
            )
    return rows


def dpo_independence() -> dict:
    dpo_root = ROOT
    code_roots = [
        dpo_root / "experiments" / "exp09_dpo_validation",
        dpo_root / "experiments" / "exp28_prompt_conditional_calibration",
        ROOT / "experiments" / "exp34_calibration_only_dpo_baselines",
    ]
    pattern = re.compile(r"\b(?:wmt|mqm)\b", re.IGNORECASE)
    hits = []
    for code_root in code_roots:
        for path in code_root.rglob("*"):
            if path.suffix not in {".py", ".sh", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                hits.append(str(path))
    run_spec = (
        dpo_root / "results/prompt_conditional_calibration/run_spec.json"
    )
    spec = json.loads(run_spec.read_text(encoding="utf-8"))
    return {
        "wmt_references_in_dpo_code_or_specs": hits,
        "canonical_run_spec": str(run_spec),
        "input_keys": sorted(spec["input_sha256"]),
        "independent_of_wmt": not hits,
    }


def markdown(report: dict) -> str:
    lines = [
        "# WMT Input-Lineage Pair Validation",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "## Inputs",
        "",
        "| Lineage | SHA-256 |",
        "|---|---|",
    ]
    for lineage in LINEAGES:
        lines.append(f"| `{lineage}` | `{report['input_hashes'][lineage]}` |")
    lines.extend(
        [
            "",
            "## Task Matrix",
            "",
            "| Task | submitted_139 | fixed_141 |",
            "|---|---:|---:|",
        ]
    )
    task_names = sorted(
        set().union(*(report["task_statuses"][lineage] for lineage in LINEAGES))
    )
    for name in task_names:
        states = []
        for lineage in LINEAGES:
            value = report["task_statuses"][lineage].get(name)
            states.append("PASS" if value == 0 else "MISSING/FAIL")
        lines.append(f"| `{name}` | {states[0]} | {states[1]} |")
    lines.extend(
        [
            "",
            "## Path Accounting",
            "",
            "| Experiment | submitted_139 | fixed_141 |",
            "|---|---:|---:|",
        ]
    )
    for row in report["path_counts"]:
        lines.append(
            f"| `{row['experiment']}` | {row['submitted_139']} | {row['fixed_141']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Accounting",
            "",
            "| Experiment | submitted_139 | fixed_141 |",
            "|---|---:|---:|",
        ]
    )
    for row in report["dataset_counts"]:
        lines.append(
            f"| `{row['experiment']}` | {row['submitted_139']} | "
            f"{row['fixed_141']} |"
        )
    lines.extend(
        [
            "",
            "## Endpoint Prediction",
            "",
            "Mean WMT absolute error at `d=0.10`, 30% calibration, 50 repeats.",
            "",
            "| Lineage | A+B | A-only | A_nu | Zero | Isotonic | Spline |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["endpoint_metrics"]:
        lines.append(
            f"| `{row['lineage']}` | {row['A+B']:.6f} | "
            f"{row['A-only']:.6f} | {row['A_nu']:.6f} | "
            f"{row['zero']:.6f} | {row['isotonic']:.6f} | "
            f"{row['spline']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## DPO Boundary",
            "",
            "DPO training and candidate generation were not rerun. The DPO code and",
            "canonical run specification contain no WMT/MQM input reference. The same",
            "read-only DPO summaries are used only by the downstream positioning audit.",
            "",
            f"- Independent of WMT: `{report['dpo']['independent_of_wmt']}`",
            f"- DPO input keys: `{', '.join(report['dpo']['input_keys'])}`",
            "",
            "## Audit Status",
            "",
            f"- Failed checks: `{len(report['failures'])}`",
        ]
    )
    for failure in report["failures"]:
        lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    input_paths = {
        "submitted_139": ROOT / "data" / "wmt_mqm_long_main_protocol.csv",
        "fixed_141": ROOT / "data" / "wmt_mqm_long.csv",
    }
    input_hashes = {key: sha256(path) for key, path in input_paths.items()}
    for lineage, expected in EXPECTED_HASHES.items():
        if input_hashes[lineage] != expected:
            failures.append(f"{lineage}: input hash mismatch")

    counts, count_failures = path_counts(args.root)
    failures.extend(count_failures)
    dataset_counts, dataset_failures = addon_dataset_counts(args.root)
    failures.extend(dataset_failures)
    task_statuses = {
        lineage: task_status(args.root / lineage) for lineage in LINEAGES
    }
    for lineage, statuses in task_statuses.items():
        failed = sorted(name for name, status in statuses.items() if status)
        if failed:
            failures.append(f"{lineage}: failed tasks {failed}")
        if len(statuses) != EXPECTED_TASK_COUNT:
            failures.append(
                f"{lineage}: recorded task count {len(statuses)}, "
                f"expected {EXPECTED_TASK_COUNT}"
            )
    if set(task_statuses[LINEAGES[0]]) != set(task_statuses[LINEAGES[1]]):
        failures.append("The two lineages do not contain identical task sets")
    dpo = dpo_independence()
    if not dpo["independent_of_wmt"]:
        failures.append("DPO code/spec unexpectedly references WMT")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "input_hashes": input_hashes,
        "path_counts": counts,
        "dataset_counts": dataset_counts,
        "endpoint_metrics": endpoint_metrics(args.root),
        "ood_metrics": ood_metrics(args.root),
        "task_statuses": task_statuses,
        "dpo": dpo,
        "failures": failures,
    }
    (args.root / "PAIR_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.root / "PAIR_VALIDATION.md").write_text(
        markdown(report), encoding="utf-8"
    )
    provenance = {
        "inputs": {
            lineage: {
                "path": str(input_paths[lineage]),
                "sha256": input_hashes[lineage],
            }
            for lineage in LINEAGES
        },
        "task_records": {
            lineage: task_records(args.root / lineage) for lineage in LINEAGES
        },
        "dpo": dpo,
    }
    (args.root / "PAIR_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(report["status"])
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
