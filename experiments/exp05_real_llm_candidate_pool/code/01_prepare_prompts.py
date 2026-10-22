#!/usr/bin/env python3
"""Prepare prompt-level splits for real LLM candidate-pool experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_input(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.input_csv) == bool(args.input_jsonl):
        raise ValueError("Pass exactly one of --input-csv or --input-jsonl.")
    if args.input_csv:
        return pd.read_csv(args.input_csv)
    rows: list[dict[str, Any]] = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare prompts and prompt-level calibration/evaluation split.")
    ap.add_argument("--input-csv", default="")
    ap.add_argument("--input-jsonl", default="")
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--prompt-id-col", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibration-frac", type=float, default=0.30)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-spec-json", default="")
    args = ap.parse_args()

    if not (0.0 < args.calibration_frac < 1.0):
        raise ValueError("--calibration-frac must be in (0,1).")
    df = read_input(args)
    if args.prompt_col not in df.columns:
        raise ValueError(f"Missing prompt column: {args.prompt_col}")
    out = pd.DataFrame()
    out["prompt"] = df[args.prompt_col].astype(str)
    if args.prompt_id_col:
        if args.prompt_id_col not in df.columns:
            raise ValueError(f"Missing prompt id column: {args.prompt_id_col}")
        out["prompt_id"] = df[args.prompt_id_col].astype(str)
    else:
        out["prompt_id"] = [f"prompt_{i:06d}" for i in range(len(out))]
    out = out.drop_duplicates(subset=["prompt_id"]).reset_index(drop=True)
    if args.limit and args.limit > 0:
        rng = np.random.default_rng(args.random_seed)
        keep = np.sort(rng.choice(len(out), size=min(args.limit, len(out)), replace=False))
        out = out.iloc[keep].reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("Need at least two unique prompts to form calibration/evaluation splits.")

    rng = np.random.default_rng(args.random_seed)
    order = np.arange(len(out))
    rng.shuffle(order)
    n_cal = int(round(args.calibration_frac * len(out)))
    n_cal = max(1, min(n_cal, len(out) - 1))
    split = np.full(len(out), "eval", dtype=object)
    split[order[:n_cal]] = "calibration"
    out["split"] = split

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    spec = {
        "script": "experiments/exp05_real_llm_candidate_pool/code/01_prepare_prompts.py",
        "input_csv": args.input_csv or None,
        "input_jsonl": args.input_jsonl or None,
        "prompt_col": args.prompt_col,
        "prompt_id_col": args.prompt_id_col or None,
        "limit": args.limit,
        "calibration_frac": args.calibration_frac,
        "random_seed": args.random_seed,
        "n_prompts": int(len(out)),
        "n_calibration": int((out["split"] == "calibration").sum()),
        "n_eval": int((out["split"] == "eval").sum()),
    }
    spec_path = Path(args.out_spec_json) if args.out_spec_json else out_path.with_suffix(".run_spec.json")
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} prompts to {out_path}")


if __name__ == "__main__":
    main()
