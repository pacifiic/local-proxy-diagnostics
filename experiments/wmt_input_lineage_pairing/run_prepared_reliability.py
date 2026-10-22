#!/usr/bin/env python3
"""Run the paper WMT MQM reliability summary from a prepared long CSV."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT
sys.path.insert(0, str(PAPER / "lib"))
SCRIPT = (
    PAPER
    / "experiments"
    / "05_wmt_mtbench"
    / "03_wmt_reliability_summary.py"
)
spec = importlib.util.spec_from_file_location("paper_wmt_reliability", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules["paper_wmt_reliability"] = module
spec.loader.exec_module(module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.long_csv)
    summary = module.summarize(frame)
    summary.to_csv(args.outdir / "wmt_mqm_reliability.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
