"""
Parse OpenMEVA MAGS data (mans_roc.json, mans_wp.json) into long-format CSV.

Structure:
  - 200 stories × 5 models × 5 raters = 5,000 rows per dataset
  - score: list of 5 annotator scores (1–5 scale, overall quality)
  - No dimension breakdown (single holistic score)

Output files:
  data/openmeva_long.csv    — one row per (dataset, story_id, model, rater_idx)
  data/openmeva_scores.csv  — mean score per (dataset, story_id, model)
"""

import argparse
import hashlib
import json
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

HF_DATASET = "Jiann/OpenMEVA"
DEFAULT_REVISION = "5ee4fe59caafce625670f91d3983f5c1ddab8c1d"
DATASET_FILES = {
    "roc": "mans_roc.json",
    "wp":  "mans_wp.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_openmeva_files(outdir: Path, revision: str, retries: int = 3) -> dict:
    """Download mans_roc.json and mans_wp.json from HuggingFace if not already present."""
    paths = {}
    for key, filename in DATASET_FILES.items():
        local_path = outdir / filename
        if local_path.exists():
            print(f"  {filename}: already exists, skipping download")
            paths[key] = local_path
            continue

        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{revision}/data/mans_data/{filename}"
        print(f"  Downloading {filename} from HuggingFace...")
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                with open(local_path, "w") as f:
                    f.write(resp.text)
                print(f"  Saved: {local_path}")
                paths[key] = local_path
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Failed to download {filename}: {e}") from e
                time.sleep(2 ** attempt)
    return paths


def parse_openmeva(filepath: Path, dataset_name: str) -> pd.DataFrame:
    """Parse a MAGS JSON file into long-format DataFrame."""
    with open(filepath) as f:
        data = json.load(f)

    rows = []
    for story_id, entry in data.items():
        for model_id, mdata in entry["gen"].items():
            scores = mdata["score"]
            for rater_idx, score in enumerate(scores):
                rows.append({
                    "dataset": dataset_name,
                    "story_id": story_id,
                    "model_id": model_id,
                    "rater_idx": rater_idx,
                    "score": score,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(DATA_DIR),
                    help="Output directory (default: data/)")
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help=f"HuggingFace revision for {HF_DATASET} (default: {DEFAULT_REVISION})")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Downloading OpenMEVA source files...")
    dataset_paths = download_openmeva_files(outdir, args.revision)

    manifest_rows = []
    for name, path in dataset_paths.items():
        manifest_rows.append({
            "dataset": HF_DATASET,
            "revision": args.revision,
            "split": name,
            "filename": path.name,
            "path": str(path),
            "n_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest_path = outdir / "openmeva_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path}")

    all_dfs = []
    for name, path in dataset_paths.items():
        print(f"Reading {path.name} ...")
        df = parse_openmeva(path, name)
        print(f"  {name}: {len(df):,} rows, "
              f"{df['story_id'].nunique()} stories, "
              f"{df['model_id'].nunique()} models")
        all_dfs.append(df)

    df_long = pd.concat(all_dfs, ignore_index=True)
    long_path = outdir / "openmeva_long.csv"
    df_long.to_csv(long_path, index=False)
    print(f"\nSaved: {long_path}  ({len(df_long):,} rows)")

    # Mean score per (dataset, story_id, model)
    df_scores = (
        df_long
        .groupby(["dataset", "story_id", "model_id"])["score"]
        .agg(mean_score="mean", n_raters="count")
        .reset_index()
    )
    scores_path = outdir / "openmeva_scores.csv"
    df_scores.to_csv(scores_path, index=False)
    print(f"Saved: {scores_path}  ({len(df_scores):,} rows)")

    run_spec = {
        "dataset": HF_DATASET,
        "revision": args.revision,
        "source_files": DATASET_FILES,
        "outputs": {
            "long_csv": str(long_path),
            "scores_csv": str(scores_path),
            "manifest_csv": str(manifest_path),
        },
        "n_long_rows": int(len(df_long)),
        "n_score_rows": int(len(df_scores)),
        "n_stories_by_dataset": {k: int(v) for k, v in df_long.groupby("dataset")["story_id"].nunique().items()},
        "n_models_by_dataset": {k: int(v) for k, v in df_long.groupby("dataset")["model_id"].nunique().items()},
    }
    spec_path = outdir / "openmeva_run_spec.json"
    spec_path.write_text(json.dumps(run_spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {spec_path}")

    # Quick summary
    print("\n--- Score summary by dataset & model ---")
    summary = (
        df_long.groupby(["dataset", "model_id"])["score"]
        .agg(["mean", "std"])
        .round(3)
    )
    print(summary.to_string())

    print("\n--- Overall ICC setup ---")
    print(f"  ROC: {df_long[df_long.dataset=='roc']['story_id'].nunique()} stories × "
          f"{df_long[df_long.dataset=='roc']['model_id'].nunique()} models × 5 raters")
    print(f"  WP : {df_long[df_long.dataset=='wp']['story_id'].nunique()} stories × "
          f"{df_long[df_long.dataset=='wp']['model_id'].nunique()} models × 5 raters")


if __name__ == "__main__":
    main()
