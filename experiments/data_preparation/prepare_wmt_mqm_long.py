#!/usr/bin/env python3
"""Convert raw WMT MQM TSV files into complete repeated-rating matrices."""

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

SEVERITY_WEIGHTS = {
    "major": -5.0, "minor": -1.0, "no-error": 0.0,
    "neutral": 0.0, "critical": -25.0,
}

TSV_FILES = {
    "wmt20_ende": "newstest2020/ende/mqm_newstest2020_ende.tsv",
    "wmt20_zhen": "newstest2020/zhen/mqm_newstest2020_zhen.tsv",
    "wmt22_zhen": "generalMT2022/zhen/mqm_generalMT2022_zhen.tsv",
    "wmt22_ende_3r": "generalMT2022/ende/mqm_generalMT2022_ende.3ratingsPerSegment.tsv",
}

def score_severity(sev) -> float:
    if pd.isna(sev):
        return 0.0
    s = str(sev).strip().lower()
    for key, val in SEVERITY_WEIGHTS.items():
        if key in s:
            return val
    return 0.0


def load_and_score(tsv_path: Path) -> pd.DataFrame:
    # Source and translation fields may contain literal quote characters.
    # Treating those characters as CSV quoting can silently merge or skip TSV
    # records, so preserve every physical tab-delimited row and fail loudly.
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
    )
    # Normalize columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("globalsegid", "seg_id"):
            col_map[c] = "seg_id"
        elif cl in ("docsegid", "doc_id", "doc_segment_id"):
            col_map[c] = "doc_id"
        elif cl in ("rater_id",):
            col_map[c] = "rater"
        elif cl in ("system_id",):
            col_map[c] = "system"
    df = df.rename(columns=col_map)
    if "seg_id" not in df.columns and "doc_id" in df.columns:
        df["seg_id"] = df["doc_id"]
    if "severity" in df.columns:
        df["penalty"] = df["severity"].apply(score_severity)
    else:
        # 3ratingsPerSegment format: use MQM score if present
        df["penalty"] = 0.0  # will handle below
    return df


def process_standard_tsv(dataset_id: str, tsv_path: Path, min_cluster: int = 200) -> list[pd.DataFrame]:
    df = load_and_score(tsv_path)
    if "penalty" not in df.columns or "system" not in df.columns:
        return []
    if "doc" in df.columns:
        df["seg_key"] = df["doc"].astype(str) + "__" + df["seg_id"].astype(str)
    else:
        df["seg_key"] = df["seg_id"].astype(str)
    agg = (
        df.groupby(["system", "seg_key", "rater"])["penalty"]
        .sum().reset_index()
        .rename(columns={"penalty": "score"})
    )
    agg["item_id"] = agg["system"].astype(str) + "___" + agg["seg_key"].astype(str)

    # Find rater-set clusters
    item_raters = agg.groupby("item_id")["rater"].apply(lambda x: tuple(sorted(set(x))))
    clusters = Counter(item_raters)

    frames = []
    for i, (rater_set, count) in enumerate(clusters.most_common()):
        if count < min_cluster or len(rater_set) < 3:
            continue
        valid_items = set(item_raters[item_raters == rater_set].index)
        sub = agg[agg["item_id"].isin(valid_items)].copy()
        sub = sub[sub["rater"].isin(rater_set)]
        cid = f"{dataset_id}_c{i}" if i > 0 else dataset_id
        sub["dataset_id"] = cid
        n_items = sub["item_id"].nunique()
        n_raters = sub["rater"].nunique()
        print(f"  cluster {cid}: {n_items} items × {n_raters} raters")
        frames.append(sub[["dataset_id", "item_id", "rater", "score"]])
    return frames


def process_3r_tsv(dataset_id: str, tsv_path: Path, min_cluster: int = 200) -> list[pd.DataFrame]:
    """Process the 3ratingsPerSegment format which has different columns."""
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
    )
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "rater_id": col_map[c] = "rater"
        elif cl == "system_id": col_map[c] = "system"
    df = df.rename(columns=col_map)
    df["seg_id"] = df["document_id"].astype(str) + "__" + df["doc_segment_id"].astype(str)

    if "severity" in df.columns:
        df["penalty"] = df["severity"].apply(score_severity)
    elif "mqm_score" in df.columns:
        df["penalty"] = df["mqm_score"]
    else:
        # Check what columns exist
        print(f"  WARNING: no severity or mqm_score column. Columns: {list(df.columns)[:10]}")
        return []

    agg = (
        df.groupby(["system", "seg_id", "rater"])["penalty"]
        .sum().reset_index()
        .rename(columns={"penalty": "score"})
    )
    agg["item_id"] = agg["system"].astype(str) + "___" + agg["seg_id"].astype(str)
    item_raters = agg.groupby("item_id")["rater"].apply(lambda x: tuple(sorted(set(x))))
    clusters = Counter(item_raters)

    frames = []
    for i, (rater_set, count) in enumerate(clusters.most_common()):
        if count < min_cluster or len(rater_set) < 3:
            continue
        valid_items = set(item_raters[item_raters == rater_set].index)
        sub = agg[agg["item_id"].isin(valid_items) & agg["rater"].isin(rater_set)].copy()
        cid = f"{dataset_id}_c{i}" if i > 0 else dataset_id
        sub["dataset_id"] = cid
        print(f"  cluster {cid}: {sub['item_id'].nunique()} items × {sub['rater'].nunique()} raters")
        frames.append(sub[["dataset_id", "item_id", "rater", "score"]])
    return frames


def main():
    import argparse
    ROOT = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=str(ROOT / "data" / "wmt-mqm-human-evaluation"),
                    help="Extracted wmt-mqm-human-evaluation root.")
    ap.add_argument("--outpath",
                    default=str(ROOT / "data" / "wmt_mqm_long.csv"),
                    help="Output CSV path.")
    ap.add_argument("--min-cluster", type=int, default=200,
                    help="Minimum items per rater-set cluster.")
    args = ap.parse_args()
    out_path = Path(args.outpath)
    base = Path(args.root)
    missing = [str(base / relative) for relative in TSV_FILES.values()
               if not (base / relative).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete WMT source checkout; missing: " + ", ".join(missing))
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {out_path}")
    all_frames = []
    for did, rel in TSV_FILES.items():
        p = base / rel
        if p.exists():
            print(f"Processing {did}")
            if "3r" in did:
                frames = process_3r_tsv(did, p, min_cluster=args.min_cluster)
            else:
                frames = process_standard_tsv(did, p, min_cluster=args.min_cluster)
            all_frames.extend(frames)
        else:
            print(f"SKIP {did}")

    if not all_frames:
        raise ValueError("No eligible WMT clusters; no output was written.")
    out = pd.concat(all_frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows")
    print(out.groupby("dataset_id").agg(
        n_items=("item_id", "nunique"), n_raters=("rater", "nunique"),
        score_mean=("score", "mean"), score_std=("score", "std"),
    ).to_string())


if __name__ == "__main__":
    main()
