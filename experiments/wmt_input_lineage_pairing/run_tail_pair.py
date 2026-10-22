#!/usr/bin/env python3
"""Run only the paired WMT tail/outlier robustness addon."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import run_paired as pair
import run_paper_addons as addons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    for lineage, wmt in pair.WMT_INPUTS.items():
        lineage_root = args.root / lineage
        selected = [
            task
            for task in addons.tasks(lineage_root, wmt, jobs=1)
            if task.name == "paper_tail_outlier_robustness"
        ]
        if len(selected) != 1:
            raise RuntimeError("Tail/outlier addon task was not found uniquely")
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
        records = pair.run_phase(selected, lineage_root, env, max_parallel=1)
        if records[0]["returncode"]:
            raise SystemExit(records[0]["returncode"])


if __name__ == "__main__":
    main()
