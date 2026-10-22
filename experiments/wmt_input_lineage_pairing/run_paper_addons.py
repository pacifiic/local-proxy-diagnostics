#!/usr/bin/env python3
"""Add paired results for WMT analyses unique to the paper workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import run_paired as pair


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parent
PAPER = ROOT
PYTHON = os.environ.get("PYTHON", "python")


def output(lineage_root: Path, name: str) -> str:
    return str(lineage_root / name / "results")


def ood_command(
    lineage_root: Path,
    name: str,
    source_csv: Path,
    source_name: str,
    source_items: str,
    source_groups: str,
    source_rater: str,
    target_csv: Path,
    target_name: str,
    target_items: str,
    target_groups: str,
    target_rater: str,
    standardize: str,
    shrinkage: float,
) -> tuple[str, ...]:
    script = (
        PAPER
        / "experiments"
        / "10_ood_transfer"
        / "01_ood_transfer_prediction.py"
    )
    return (
        PYTHON,
        str(script),
        "--source-human-long-csv", str(source_csv),
        "--source-name", source_name,
        "--source-item-cols", source_items,
        "--source-group-cols", source_groups,
        "--source-rater-col", source_rater,
        "--target-human-long-csv", str(target_csv),
        "--target-name", target_name,
        "--target-item-cols", target_items,
        "--target-group-cols", target_groups,
        "--target-rater-col", target_rater,
        "--calibration-frac", "0.30",
        "--n-repeats", "50",
        "--source-prior-repeats", "50",
        "--shrinkage-lambda", str(shrinkage),
        "--fit-max-d", "0.10",
        "--d-targets", "0.05,0.10",
        "--standardize-pairs", standardize,
        "--outdir", output(lineage_root, name),
    )


def tasks(lineage_root: Path, wmt: Path, jobs: int) -> list[pair.Task]:
    scripts = PAPER / "experiments"
    summeval = ROOT / "data" / "summeval_long.csv"
    items = "item_id"
    groups = "dataset_id"
    common_long = (
        "--data", str(wmt),
        "--item-col", items,
        "--rater-col", "rater",
        "--score-col", "score",
        "--group-cols", groups,
    )
    result = [
        pair.Task(
            "paper_exp19_direct_validation",
            (
                PYTHON,
                str(HERE / "run_prepared_direct_validation.py"),
                "--long-csv", str(wmt),
                "--outdir", output(lineage_root, "paper_exp19_direct_validation"),
            ),
        ),
        pair.Task(
            "paper_exp21_reliability",
            (
                PYTHON,
                str(HERE / "run_prepared_reliability.py"),
                "--long-csv", str(wmt),
                "--outdir", output(lineage_root, "paper_exp21_reliability"),
            ),
        ),
        pair.Task(
            "paper_exp23_monotone",
            (
                PYTHON,
                str(scripts / "06_supplemental_audits" / "01_monotone_condition_audit.py"),
                *common_long,
                "--score-transform", "raw",
                "--gold-estimator", "loo_mean",
                "--min-items", "200",
                "--min-raters", "3",
                "--n-bootstrap", "200",
                "--outdir", output(lineage_root, "paper_exp23_monotone"),
            ),
        ),
        pair.Task(
            "paper_exp24_rho_scaling",
            (
                PYTHON,
                str(scripts / "06_supplemental_audits" / "02_rho_scaling_audit.py"),
                *common_long,
                "--score-transform", "raw",
                "--gold-estimator", "loo_mean",
                "--min-items", "200",
                "--min-raters", "3",
                "--fit-max-d", "0.10",
                "--outdir", output(lineage_root, "paper_exp24_rho_scaling"),
            ),
        ),
        pair.Task(
            "paper_exp25_measurement_intervention",
            (
                PYTHON,
                str(scripts / "06_supplemental_audits" / "03_measurement_intervention.py"),
                *common_long,
                "--score-transform", "raw",
                "--k-list", "1,2",
                "--gold-size", "1",
                "--n-reps", "500",
                "--min-items", "200",
                "--fit-max-d", "0.10",
                "--outdir",
                output(lineage_root, "paper_exp25_measurement_intervention"),
            ),
        ),
        pair.Task(
            "paper_exp27_prospective",
            (
                PYTHON,
                str(scripts / "07_prospective_prediction" / "01_prospective_prediction.py"),
                "--human-long-csv", str(wmt),
                "--item-cols", items,
                "--group-cols", groups,
                "--rater-col", "rater",
                "--score-col", "score",
                "--fit-max-d", "0.10",
                "--outdir", output(lineage_root, "paper_exp27_prospective"),
            ),
        ),
        pair.Task(
            "paper_exp28_stability_tail",
            (
                PYTHON,
                str(scripts / "07_prospective_prediction" / "02_stability_tail_audit.py"),
                "--human-long-csv", str(wmt),
                "--item-cols", items,
                "--group-cols", groups,
                "--rater-col", "rater",
                "--score-col", "score",
                "--fit-max-d", "0.10",
                "--outdir", output(lineage_root, "paper_exp28_stability_tail"),
            ),
        ),
        pair.Task(
            "paper_exp32_proxy_perturbation",
            (
                PYTHON,
                str(scripts / "11_proxy_perturbation" / "01_proxy_perturbation_baselines.py"),
                "--human-long-csv", str(wmt),
                "--item-cols", items,
                "--group-cols", groups,
                "--rater-col", "rater",
                "--score-col", "score",
                "--fit-max-d", "0.10",
                "--outdir", output(lineage_root, "paper_exp32_proxy_perturbation"),
            ),
        ),
        pair.Task(
            "paper_tail_outlier_robustness",
            (
                PYTHON,
                str(
                    scripts
                    / "14_tail_outlier_robustness"
                    / "01_tail_outlier_robustness_audit.py"
                ),
                "--dataset-name", "WMT-MQM-clusters",
                "--human-long-csv", str(wmt),
                "--item-cols", items,
                "--group-cols", groups,
                "--rater-col", "rater",
                "--outdir",
                output(lineage_root, "paper_tail_outlier_robustness"),
            ),
        ),
        pair.Task(
            "paper_calibration_size",
            (
                PYTHON,
                str(scripts / "12_accept_boost" / "03_calibration_size_curve.py"),
                "--human-long-csv", str(wmt),
                "--item-cols", items,
                "--group-cols", groups,
                "--rater-col", "rater",
                "--calibration-fracs", "0.10,0.20,0.30,0.50,0.70",
                "--d-targets", "0.05,0.10",
                "--n-repeats", "50",
                "--n-boot", "1000",
                "--outdir", output(lineage_root, "paper_calibration_size"),
            ),
        ),
        pair.Task(
            "cubic_moment",
            (
                PYTHON,
                str(HERE / "build_cubic_paired.py"),
                "--datasets", "wmt",
                "--jobs", str(jobs),
                "--outdir", output(lineage_root, "cubic_moment"),
            ),
        ),
    ]

    result.append(
        pair.Task(
            "paper_ood_zboth_summeval_to_wmt",
            ood_command(
                lineage_root,
                "paper_ood_zboth_summeval_to_wmt",
                summeval, "SummEval", "article_id,model_id",
                "dimension,rater_type", "rater_idx",
                wmt, "WMT_MQM", items, groups, "rater",
                "z_both", 0.75,
            ),
        )
    )
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        stem = str(value).replace(".", "p")
        name = f"paper_ood_zcal_summeval_to_wmt_lambda_{stem}"
        result.append(
            pair.Task(
                name,
                ood_command(
                    lineage_root, name,
                    summeval, "SummEval", "article_id,model_id",
                    "dimension,rater_type", "rater_idx",
                    wmt, "WMT_MQM", items, groups, "rater",
                    "z_cal", value,
                ),
            )
        )
        reverse = f"paper_ood_zcal_wmt_to_summeval_lambda_{stem}"
        result.append(
            pair.Task(
                reverse,
                ood_command(
                    lineage_root, reverse,
                    wmt, "WMT_MQM", items, groups, "rater",
                    summeval, "SummEval", "article_id,model_id",
                    "dimension,rater_type", "rater_idx",
                    "z_cal", value,
                ),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--max-parallel", type=int, default=8)
    args = parser.parse_args()
    for lineage, wmt in pair.WMT_INPUTS.items():
        lineage_root = args.root / lineage
        if not lineage_root.is_dir():
            raise FileNotFoundError(lineage_root)
        env = os.environ.copy()
        env.update(
            {
                "WMT_MQM_LONG_CSV": str(wmt.resolve()),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        records = pair.run_phase(
            tasks(lineage_root, wmt, args.jobs),
            lineage_root,
            env,
            args.max_parallel,
        )
        (lineage_root / "paper_addon_records.json").write_text(
            __import__("json").dumps(records, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
