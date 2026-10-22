#!/usr/bin/env python3
"""Run all WMT-dependent analyses against both preserved WMT inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = os.environ.get("PYTHON", "python")

WMT_INPUTS = {
    "submitted_139": ROOT / "data" / "wmt_mqm_long_main_protocol.csv",
    "fixed_141": ROOT / "data" / "wmt_mqm_long.csv",
}

DPO_ROOT = ROOT
DPO_CHECKPOINTS = DPO_ROOT / "results/prompt_conditional_calibration/checkpoint_predictions.csv"
DPO_FIT_METADATA = DPO_CHECKPOINTS.with_name("model_fit_metadata.csv")
DPO_BINNED_CHECKPOINTS = DPO_ROOT / "results/actual_dpo/calibration_models/binned/checkpoint_predictions.csv"
DPO_BINNED_FIT_METADATA = DPO_BINNED_CHECKPOINTS.with_name(
    "model_fit_metadata.csv"
)


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def command_string(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_task(task: Task, lineage_root: Path, env: dict[str, str]) -> dict:
    task_root = lineage_root / task.name
    ensure_fresh(task_root)
    log_path = task_root / "command.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_string(task.command)}\n\n")
        log.flush()
        process = subprocess.run(
            task.command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record = {
        "task": task.name,
        "command": list(task.command),
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path.resolve().relative_to(ROOT.resolve())),
    }
    (task_root / "task_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(
            f"{task.name} failed with code {process.returncode}; see {log_path}"
        )
    return record


def primary_tasks(lineage_root: Path, jobs: int) -> list[Task]:
    exp = ROOT / "experiments"
    engines = ROOT / "engines"
    out = lambda name: str(lineage_root / name / "results")
    prospect = (
        engines
        / "07_prospective_prediction"
        / "01_prospective_prediction.py"
    )
    wmt = os.environ["WMT_MQM_LONG_CSV"]
    return [
        Task(
            "exp02_filter_min400",
            (
                PYTHON, str(prospect),
                "--human-long-csv", wmt,
                "--item-cols", "item_id",
                "--group-cols", "dataset_id",
                "--rater-col", "rater",
                "--min-items", "400",
                "--n-repeats", "8",
                "--n-boot", "300",
                "--outdir", out("exp02_filter_min400"),
            ),
        ),
        Task(
            "exp02_filter_min100",
            (
                PYTHON, str(prospect),
                "--human-long-csv", wmt,
                "--item-cols", "item_id",
                "--group-cols", "dataset_id",
                "--rater-col", "rater",
                "--min-items", "100",
                "--n-repeats", "8",
                "--n-boot", "300",
                "--outdir", out("exp02_filter_min100"),
            ),
        ),
        Task(
            "exp03_calibration_crossover",
            (
                PYTHON,
                str(exp / "exp03_calibration_crossover" / "01_calibration_crossover.py"),
                "--track", "wmt",
                "--human-long-csv", wmt,
                "--cal-fracs", "0.05,0.10,0.20,0.30,0.50",
                "--n-repeats", "50",
                "--min-items", "100",
                "--n-boot", "500",
                "--extended",
                "--out-name", "merged_wmt.csv",
                "--splits-name", "splits_wmt.csv",
                "--outdir", out("exp03_calibration_crossover"),
            ),
        ),
        Task(
            "exp14_conditional_tilt",
            (
                PYTHON,
                str(exp / "exp14_conditional_tilt" / "build_conditional.py"),
                "--datasets", "WMT-MQM",
                "--outdir", out("exp14_conditional_tilt"),
            ),
        ),
        Task(
            "exp16_fallback_rule",
            (
                PYTHON,
                str(exp / "exp16_fallback_rule" / "build_fallback.py"),
                "--datasets", "WMT-MQM",
                "--outdir", out("exp16_fallback_rule"),
            ),
        ),
        Task(
            "exp18_calbudget",
            (
                PYTHON,
                str(exp / "exp18_calbudget" / "build_calbudget.py"),
                "--datasets", "WMT-MQM",
                "--outdir", out("exp18_calbudget"),
            ),
        ),
        Task(
            "exp20_bon_window_gate",
            (
                PYTHON,
                str(exp / "exp20_bon_window_gate" / "build_bon_gate.py"),
                "--datasets", "WMT-MQM-human_heldout",
                "--outdir", out("exp20_bon_window_gate"),
            ),
        ),
        Task(
            "exp21_direct_calibration",
            (
                PYTHON,
                str(exp / "exp21_direct_calibration" / "build_direct_calibration.py"),
                "--datasets", "wmt",
                "--jobs", str(jobs),
                "--outdir", out("exp21_direct_calibration"),
            ),
        ),
        Task(
            "exp22_b_stability",
            (
                PYTHON,
                str(exp / "exp22_b_stability" / "build_b_stability.py"),
                "--datasets", "WMT-MQM-human_heldout",
                "--outdir", out("exp22_b_stability"),
            ),
        ),
        Task(
            "exp23_mixture_ranktilt",
            (
                PYTHON,
                str(exp / "exp23_mixture_ranktilt" / "build_mixture_ranktilt.py"),
                "--datasets", "WMT-MQM-human_heldout",
                "--outdir", out("exp23_mixture_ranktilt"),
            ),
        ),
        Task(
            "exp24_within_dataset",
            (
                PYTHON,
                str(exp / "exp24_direct_calibration_followup" / "build_within_dataset.py"),
                "--datasets", "wmt",
                "--jobs", str(jobs),
                "--outdir", out("exp24_within_dataset"),
            ),
        ),
        Task(
            "exp24_ood_transfer",
            (
                PYTHON,
                str(exp / "exp24_direct_calibration_followup" / "build_ood_transfer.py"),
                "--jobs", str(jobs),
                "--outdir", out("exp24_ood_transfer"),
            ),
        ),
        Task(
            "exp32_evaluation_aware_ab",
            (
                PYTHON,
                str(exp / "exp32_evaluation_aware_ab" / "build_evaluation_aware_ab.py"),
                "--datasets", "wmt",
                "--jobs", str(jobs),
                "--outdir", out("exp32_evaluation_aware_ab"),
            ),
        ),
        Task(
            "exp33_calibration_only",
            (
                PYTHON,
                str(exp / "exp33_calibration_only_flexible_baselines" / "build.py"),
                "--datasets", "wmt",
                "--jobs", str(jobs),
                "--outdir", out("exp33_calibration_only"),
            ),
        ),
    ]


def derived_tasks(lineage_root: Path) -> list[Task]:
    exp = ROOT / "experiments"
    result = lambda name: lineage_root / name / "results"
    return [
        Task(
            "exp13_paired_stats",
            (
                PYTHON,
                str(exp / "exp13_paired_stats" / "build_paired_stats.py"),
                "--wmt-splits",
                str(result("exp03_calibration_crossover") / "splits_wmt.csv"),
                "--expected-wmt-paths",
                "139" if lineage_root.name == "submitted_139" else "141",
                "--outdir",
                str(lineage_root / "exp13_paired_stats" / "results"),
            ),
        ),
        Task(
            "exp14_conditional_audit",
            (
                PYTHON,
                str(exp / "exp14_conditional_tilt" / "audit_conditional.py"),
                "--outdir", str(result("exp14_conditional_tilt")),
            ),
        ),
        Task(
            "exp24_within_audit",
            (
                PYTHON,
                str(exp / "exp24_direct_calibration_followup" / "audit_within_dataset.py"),
                "--outdir", str(result("exp24_within_dataset")),
            ),
        ),
        Task(
            "exp24_ood_audit",
            (
                PYTHON,
                str(exp / "exp24_direct_calibration_followup" / "audit_ood_transfer.py"),
                "--outdir", str(result("exp24_ood_transfer")),
            ),
        ),
        Task(
            "exp32_evaluation_aware_audit",
            (
                PYTHON,
                str(exp / "exp32_evaluation_aware_ab" / "audit_evaluation_aware_ab.py"),
                "--outdir", str(result("exp32_evaluation_aware_ab")),
            ),
        ),
        Task(
            "exp33_calibration_only_audit",
            (
                PYTHON,
                str(exp / "exp33_calibration_only_flexible_baselines" / "audit.py"),
                "--outdir", str(result("exp33_calibration_only")),
                "--target-aware-reference", str(result("exp21_direct_calibration")),
            ),
        ),
        Task(
            "exp30_positioning_audit",
            (
                PYTHON,
                str(exp / "exp30_positioning_audit" / "run_positioning_audit.py"),
                "--dpo-checkpoints", str(DPO_CHECKPOINTS),
                "--dpo-binned-checkpoints", str(DPO_BINNED_CHECKPOINTS),
                "--dpo-fit-metadata", str(DPO_FIT_METADATA),
                "--dpo-binned-fit-metadata", str(DPO_BINNED_FIT_METADATA),
                "--direct-predictions",
                str(result("exp21_direct_calibration") / "direct_calibration_predictions.csv"),
                "--b-stability-rows",
                str(result("exp22_b_stability") / "b_stability_rows.csv"),
                "--bootstrap-reps", "10000",
                "--random-seed", "0",
                "--outdir", str(lineage_root / "exp30_positioning_audit" / "results"),
            ),
        ),
    ]


def run_phase(
    tasks: list[Task],
    lineage_root: Path,
    env: dict[str, str],
    max_parallel: int,
) -> list[dict]:
    records = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_task, task, lineage_root, env): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            record = future.result()
            print(
                f"PASS {lineage_root.name}/{task.name} "
                f"({record['elapsed_seconds']:.1f}s)",
                flush=True,
            )
            records.append(record)
    return sorted(records, key=lambda row: row["task"])


def validate_required_inputs() -> None:
    required = [*WMT_INPUTS.values(), DPO_CHECKPOINTS, DPO_FIT_METADATA,
                DPO_BINNED_CHECKPOINTS, DPO_BINNED_FIT_METADATA]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument(
        "--lineages",
        nargs="+",
        choices=sorted(WMT_INPUTS),
        default=sorted(WMT_INPUTS),
    )
    parser.add_argument(
        "--resume-root",
        action="store_true",
        help="Allow an existing root, but still require selected lineages to be absent.",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.max_parallel < 1:
        parser.error("--jobs and --max-parallel must be positive")
    if args.resume_root:
        args.out_root.mkdir(parents=True, exist_ok=True)
    else:
        ensure_fresh(args.out_root)
    validate_required_inputs()

    all_records = {}
    manifest_path = args.out_root / "manifest.json"
    if args.resume_root and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_records.update(existing.get("lineages", {}))
    for lineage in args.lineages:
        input_path = WMT_INPUTS[lineage]
        lineage_root = args.out_root / lineage
        lineage_root.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(
            {
                "WMT_MQM_LONG_CSV": str(input_path.resolve()),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        os.environ["WMT_MQM_LONG_CSV"] = str(input_path.resolve())
        lineage_record = {
            "lineage": lineage,
            "wmt_input": str(input_path.relative_to(ROOT)),
            "wmt_input_sha256": sha256(input_path),
            "primary": run_phase(
                primary_tasks(lineage_root, args.jobs),
                lineage_root,
                env,
                args.max_parallel,
            ),
        }
        lineage_record["derived"] = run_phase(
            derived_tasks(lineage_root),
            lineage_root,
            env,
            args.max_parallel,
        )
        all_records[lineage] = lineage_record
        (lineage_root / "lineage_record.json").write_text(
            json.dumps(lineage_record, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "experiment": "wmt_input_lineage_pairing",
        "dpo_training_rerun": False,
        "dpo_role": "read-only shared input to exp30 positioning audit",
        "lineages": all_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COMPLETE -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
