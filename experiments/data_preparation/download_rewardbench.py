#!/usr/bin/env python3
"""
Download RewardBench result data
=================================

Downloads per-item reward model scores from allenai/reward-bench-results
and saves them to data/rewardbench/.

Usage:
    python experiments/data_preparation/download_rewardbench.py

Required packages:
    pip install requests pandas tqdm
"""

import json
import hashlib
import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────
import argparse as _argparse
ROOT     = Path(__file__).resolve().parents[2]
DEFAULT_REVISION = "96302cc604e4f257dc526272c803c4eff27e3469"

def _parse_args():
    ap = _argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "data" / "rewardbench"),
                    help="Output directory (default: data/rewardbench/)")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "rewardbench" / "raw_json"),
                    help="JSON cache directory (default: data/rewardbench/raw_json/)")
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help=f"Hugging Face revision or commit SHA to read (default: {DEFAULT_REVISION}).")
    ap.add_argument("--include-result-labels", action="store_true",
                    help=("Also parse results-only files as binary labels. By default the "
                          "paper snapshot keeps only comparable continuous chosen/rejected scores."))
    return ap.parse_args()

_args    = _parse_args()
OUT_DIR  = Path(_args.outdir)
OUT_CSV  = OUT_DIR / "rewardbench_scores.csv"
OUT_JSON = Path(_args.cache_dir)
REVISION = _args.revision
INCLUDE_RESULT_LABELS = _args.include_result_labels

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HuggingFace URL
# ─────────────────────────────────────────────────────────────
HF_API      = "https://huggingface.co/api/datasets/allenai/reward-bench-results"
HF_RESOLVE  = f"https://huggingface.co/datasets/allenai/reward-bench-results/resolve/{REVISION}"


def list_model_files():
    """Return all JSON file paths under eval-set-scores/."""
    print("Fetching file list...")

    # Use the recursive tree API so nested model paths are not silently missed.
    resp = requests.get(
        f"{HF_API}/tree/{REVISION}/eval-set-scores",
        params={"recursive": "true"},
        timeout=30
    )
    resp.raise_for_status()

    entries = resp.json()
    json_files = sorted(
        entry["path"] for entry in entries
        if entry.get("path", "").endswith(".json")
    )

    print(f"Found {len(json_files)} model JSON files")
    return json_files


def download_json(hf_path: str, retries=3):
    """
    Download a single JSON file. Reuses local cache if available.

    hf_path example: eval-set-scores/allenai/tulu-2-dpo-13b.json
    """
    # Local cache path
    local_path = OUT_JSON / hf_path.replace("/", "__")

    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)

    url = f"{HF_RESOLVE}/{hf_path}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Save to cache
            with open(local_path, "w") as f:
                json.dump(data, f)

            return data

        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # exponential backoff


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_float(value):
    """Convert RewardBench scalar scores, unwrapping singleton lists."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar or singleton score list, got length {len(value)}")
        value = value[0]
    return float(value)


def _result_rows(results, model_name: str, subsets=None):
    """Parse results-only RewardBench files as binary labels when requested."""
    if subsets is None:
        subsets = [None] * len(results)
    rows = []
    for i, (score, subset) in enumerate(zip(results, subsets)):
        if isinstance(score, dict):
            item_id = score.get("id") or score.get("item_id") or i
            if "score_chosen" in score and "score_rejected" in score:
                sc = _as_float(score["score_chosen"])
                sr = _as_float(score["score_rejected"])
            elif "chosen" in score and "rejected" in score:
                sc = _as_float(score["chosen"])
                sr = _as_float(score["rejected"])
            elif "score" in score:
                sc = _as_float(score["score"])
                sr = 0.0
            else:
                sc = _as_float(score.get("result", 0.0))
                sr = 0.0
            subset = score.get("subset", subset)
        else:
            item_id = i
            sc = _as_float(score)
            sr = 0.0
        rows.append({
            "item_id": item_id,
            "model": model_name,
            "score_chosen": sc,
            "score_rejected": sr,
            "margin": sc - sr,
            "subset": subset,
        })
    return rows


def parse_model_json(data, model_name: str, include_result_labels: bool = False):
    """
    Convert JSON data to long-format rows.

    Primary structure (allenai/reward-bench-results eval-set-scores):
      {"scores_chosen": [...], "scores_rejected": [...], "subset": [...], ...}

    Fallback structures:
      A. [0, 1, 0, 1, ...]                       — flat int list
      B. {"scores": [0.3, -1.2, ...]}             — float scores
      C. {"chosen": [...], "rejected": [...]}      — paired scores
      D. [{"score_chosen": 1.2, "score_rejected": -0.3, ...}]
      E. {"results": [...]}                        — nested
    """
    rows = []

    # ── Primary: {"scores_chosen": [...], "scores_rejected": [...], "subset": [...]} ──
    if isinstance(data, dict) and "scores_chosen" in data and "scores_rejected" in data:
        chosen = data["scores_chosen"]
        rejected = data["scores_rejected"]
        subsets = data.get("subset", [None] * len(chosen))
        for i, (c, r, s) in enumerate(zip(chosen, rejected, subsets)):
            sc = _as_float(c)
            sr = _as_float(r)
            rows.append({
                "item_id": i,
                "model": model_name,
                "score_chosen": sc,
                "score_rejected": sr,
                "margin": sc - sr,
                "subset": s,
            })
        return rows

    # ── Fallback A: flat list (int or float) ──
    if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], dict):
        return _result_rows(data, model_name) if include_result_labels else []

    # ── Fallback B: {"scores": [...]} ──
    if isinstance(data, dict) and "scores" in data:
        scores = data["scores"]
        if not isinstance(scores, list):
            scores = list(scores.values()) if isinstance(scores, dict) else [scores]
        for i, score in enumerate(scores):
            sc = _as_float(score)
            rows.append({"item_id": i, "model": model_name,
                         "score_chosen": sc, "score_rejected": 0.0,
                         "margin": sc, "subset": None})
        return rows

    # ── Fallback C: {"chosen": [...], "rejected": [...]} ──
    if isinstance(data, dict) and "chosen" in data and "rejected" in data:
        chosen = data["chosen"]
        rejected = data["rejected"]
        subsets = data.get("subset", [None] * len(chosen))
        for i, (c, r, s) in enumerate(zip(chosen, rejected, subsets)):
            sc = _as_float(c)
            sr = _as_float(r)
            rows.append({"item_id": i, "model": model_name,
                         "score_chosen": sc, "score_rejected": sr,
                         "margin": sc - sr, "subset": s})
        return rows

    # ── Fallback D: [{"score_chosen": ..., "score_rejected": ...}, ...] ──
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if not include_result_labels and not (
            "score_chosen" in data[0] and "score_rejected" in data[0]
        ):
            return []
        for i, item in enumerate(data):
            item_id = item.get("id") or item.get("item_id") or i
            sc = _as_float(item.get("score_chosen", item.get("score", item.get("result", 0))))
            sr = _as_float(item.get("score_rejected", 0))
            rows.append({"item_id": item_id, "model": model_name,
                         "score_chosen": sc, "score_rejected": sr,
                         "margin": sc - sr,
                         "subset": item.get("subset")})
        return rows

    # ── Fallback E: {"results": [...]} ──
    if isinstance(data, dict) and "results" in data:
        if not include_result_labels:
            return []
        return _result_rows(data["results"], model_name, data.get("subset"))

    print(f"\n  [{model_name}] unsupported structure: "
          f"{type(data).__name__}, "
          f"keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    return rows


def run():
    print("=" * 60)
    print("Downloading RewardBench result data")
    print("=" * 60)

    # ─── Fetch file list ───
    json_files = list_model_files()

    if not json_files:
        print("ERROR: No files found. Please check the HF dataset structure.")
        return

    # ─── Download and parse each model JSON ───
    all_rows = []
    failed = []
    skipped = []
    manifest_rows = []

    for hf_path in tqdm(json_files, desc="Downloading models"):
        model_name = hf_path.replace("eval-set-scores/", "").replace(".json", "")
        local_path = OUT_JSON / hf_path.replace("/", "__")
        try:
            data = download_json(hf_path)
            # Use the "model" field inside JSON if available (e.g. "berkeley-nest/Starling-RM-34B"),
            # because HF path (e.g. "Nexusflow/Starling-RM-34B") can differ from the actual model name.
            if isinstance(data, dict) and "model" in data:
                model_name = data["model"]
            rows = parse_model_json(data, model_name, INCLUDE_RESULT_LABELS)

            if not rows:
                reason = "results_only_skipped" if isinstance(data, dict) and "results" in data else "unsupported_structure"
                skipped.append(model_name)
                manifest_rows.append({
                    "hf_path": hf_path,
                    "model": model_name,
                    "status": "skipped",
                    "reason": reason,
                    "n_rows": 0,
                    "sha256": _sha256_path(local_path),
                })
                continue

            all_rows.extend(rows)
            manifest_rows.append({
                "hf_path": hf_path,
                "model": model_name,
                "status": "parsed",
                "reason": "",
                "n_rows": len(rows),
                "sha256": _sha256_path(local_path),
            })

        except Exception as e:
            print(f"\n  [{model_name}] download failed: {e}")
            failed.append(model_name)
            manifest_rows.append({
                "hf_path": hf_path,
                "model": model_name,
                "status": "failed",
                "reason": str(e),
                "n_rows": 0,
                "sha256": _sha256_path(local_path),
            })
            continue

    # ─── Save DataFrame ───
    if not all_rows:
        print("\nERROR: No valid data found")
        return

    df = pd.DataFrame(all_rows)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  total rows:  {len(df):,}")
    print(f"  models:      {df['model'].nunique()}")
    print(f"  items:       {df['item_id'].nunique()}")
    print(f"  columns:     {list(df.columns)}")
    print(f"  skipped:     {len(skipped)}")

    if failed:
        print(f"  failed models: {len(failed)}")
        for m in failed[:5]:
            print(f"    - {m}")

    # Save CSV (long format with score_chosen, score_rejected, margin, subset)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  -> Saved: {OUT_CSV}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "rewardbench_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  -> Manifest saved: {manifest_path}")

    run_spec = {
        "hf_dataset": "allenai/reward-bench-results",
        "requested_revision": REVISION,
        "tree_path": "eval-set-scores",
        "recursive_tree": True,
        "include_result_labels": INCLUDE_RESULT_LABELS,
        "n_json_files": len(json_files),
        "n_parsed_files": int((manifest["status"] == "parsed").sum()) if not manifest.empty else 0,
        "n_skipped_files": int((manifest["status"] == "skipped").sum()) if not manifest.empty else 0,
        "n_failed_files": int((manifest["status"] == "failed").sum()) if not manifest.empty else 0,
        "n_rows": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_items": int(df["item_id"].nunique()),
    }
    spec_path = OUT_DIR / "rewardbench_run_spec.json"
    spec_path.write_text(json.dumps(run_spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  -> Run spec saved: {spec_path}")

    # Save pivot (item x model, values = score_chosen)
    pivot = df.pivot_table(index="item_id", columns="model", values="score_chosen", aggfunc="first")
    pivot_path = OUT_DIR / "rewardbench_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"  -> Pivot saved: {pivot_path}  shape={pivot.shape}")


if __name__ == "__main__":
    run()
