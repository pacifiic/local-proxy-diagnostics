#!/usr/bin/env python3
"""
Download MT-Bench human pairwise judgments from HuggingFace.

Source: https://huggingface.co/datasets/lmsys/mt_bench_human_judgments

Requires:
    pip install datasets

Output:
    data/mt_bench_human_judgments/human.parquet
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HF_DATASET = "lmsys/mt_bench_human_judgments"
DEFAULT_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
LEGACY_PARQUET_NAME = "human-00000-of-00001-25f4910818759289.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "data" / "mt_bench_human_judgments"),
                    help="Output directory (default: data/mt_bench_human_judgments/)")
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help=f"HuggingFace revision for {HF_DATASET} (default: {DEFAULT_REVISION})")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "human.parquet"
    legacy_path = out_dir / LEGACY_PARQUET_NAME

    from datasets import load_dataset

    print("Downloading MT-Bench human judgments from HuggingFace...")
    ds = load_dataset(HF_DATASET, split="human", revision=args.revision)
    ds.to_parquet(str(out_path))
    print(f"Saved: {out_path}  ({len(ds)} rows)")
    if legacy_path != out_path:
        shutil.copy2(out_path, legacy_path)
        print(f"Saved legacy alias: {legacy_path}")

    manifest_rows = []
    for path in [out_path, legacy_path]:
        manifest_rows.append({
            "dataset": HF_DATASET,
            "revision": args.revision,
            "filename": path.name,
            "path": str(path),
            "n_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest_path = out_dir / "mtbench_manifest.csv"
    import pandas as pd
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path}")

    spec = {
        "dataset": HF_DATASET,
        "revision": args.revision,
        "split": "human",
        "outputs": {
            "canonical_parquet": str(out_path),
            "legacy_parquet_alias": str(legacy_path),
            "manifest_csv": str(manifest_path),
        },
        "n_rows": int(len(ds)),
        "columns": list(ds.column_names),
    }
    spec_path = out_dir / "mtbench_run_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {spec_path}")


if __name__ == "__main__":
    main()
