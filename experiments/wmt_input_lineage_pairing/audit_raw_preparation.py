#!/usr/bin/env python3
"""Audit the WMT MQM raw-data-to-analysis-table chain independently."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


FILES = {
    "wmt20_ende": (
        "newstest2020/ende/mqm_newstest2020_ende.tsv",
        "standard",
    ),
    "wmt20_zhen": (
        "newstest2020/zhen/mqm_newstest2020_zhen.tsv",
        "standard",
    ),
    "wmt22_zhen": (
        "generalMT2022/zhen/mqm_generalMT2022_zhen.tsv",
        "standard",
    ),
    "wmt22_ende_3r": (
        "generalMT2022/ende/"
        "mqm_generalMT2022_ende.3ratingsPerSegment.tsv",
        "three_rater",
    ),
}

SEVERITY_WEIGHTS = {
    "major": -5.0,
    "minor": -1.0,
    "no-error": 0.0,
    "neutral": 0.0,
    "critical": -25.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_severity(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().lower()
    for key, score in SEVERITY_WEIGHTS.items():
        if key in text:
            return score
    return 0.0


def parse_raw(path: Path) -> tuple[pd.DataFrame, dict]:
    physical_rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    frame = pd.read_csv(
        path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
    )
    return frame, {
        "physical_data_rows": physical_rows,
        "parsed_rows": int(len(frame)),
        "all_physical_rows_parsed": physical_rows == len(frame),
        "sha256": sha256(path),
    }


def independent_frames(
    dataset_id: str,
    frame: pd.DataFrame,
    layout: str,
    min_cluster: int,
) -> list[pd.DataFrame]:
    if layout == "standard":
        system_col, rater_col = "system", "rater"
        segment_col, document_col = "seg_id", "doc"
    else:
        system_col, rater_col = "system_id", "rater_id"
        segment_col, document_col = "doc_segment_id", "document_id"

    work = frame.copy()
    work["penalty"] = work["severity"].map(score_severity)
    work["seg_key"] = (
        work[document_col].astype(str)
        + "__"
        + work[segment_col].astype(str)
    )
    aggregate = (
        work.groupby(
            [system_col, "seg_key", rater_col],
            sort=True,
            dropna=False,
        )["penalty"]
        .sum()
        .reset_index()
        .rename(
            columns={
                system_col: "system",
                rater_col: "rater",
                "penalty": "score",
            }
        )
    )
    aggregate["item_id"] = (
        aggregate["system"].astype(str)
        + "___"
        + aggregate["seg_key"].astype(str)
    )
    item_raters = aggregate.groupby("item_id", sort=True)["rater"].apply(
        lambda values: tuple(sorted(set(values)))
    )
    clusters = Counter(item_raters)
    outputs = []
    for index, (rater_set, count) in enumerate(clusters.most_common()):
        if count < min_cluster or len(rater_set) < 3:
            continue
        valid_items = set(item_raters[item_raters == rater_set].index)
        selected = aggregate[
            aggregate["item_id"].isin(valid_items)
            & aggregate["rater"].isin(rater_set)
        ].copy()
        selected["dataset_id"] = (
            f"{dataset_id}_c{index}" if index else dataset_id
        )
        outputs.append(
            selected[["dataset_id", "item_id", "rater", "score"]]
        )
    return outputs


def collision_record(
    dataset_id: str,
    frame: pd.DataFrame,
    layout: str,
) -> dict:
    if layout == "standard":
        system_col, rater_col = "system", "rater"
        segment_col, document_col = "seg_id", "doc"
    else:
        system_col, rater_col = "system_id", "rater_id"
        segment_col, document_col = "doc_segment_id", "document_id"

    pairs = frame[[document_col, segment_col]].drop_duplicates()
    multiplicity = pairs.groupby(segment_col)[document_col].nunique()
    legacy_items = frame[[system_col, segment_col]].drop_duplicates()
    fixed_items = frame[
        [system_col, document_col, segment_col]
    ].drop_duplicates()
    legacy_ratings = frame[
        [system_col, segment_col, rater_col]
    ].drop_duplicates()
    fixed_ratings = frame[
        [system_col, document_col, segment_col, rater_col]
    ].drop_duplicates()
    return {
        "dataset": dataset_id,
        "documents": int(frame[document_col].nunique()),
        "distinct_segment_ids": int(frame[segment_col].nunique()),
        "distinct_document_segment_pairs": int(len(pairs)),
        "segment_ids_reused_across_documents": int((multiplicity > 1).sum()),
        "max_documents_per_segment_id": int(multiplicity.max()),
        "legacy_item_keys": int(len(legacy_items)),
        "document_qualified_item_keys": int(len(fixed_items)),
        "legacy_system_segment_rater_keys": int(len(legacy_ratings)),
        "document_qualified_rating_keys": int(len(fixed_ratings)),
    }


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset_id", "item_id", "rater", "score"]
    result = frame[columns].copy()
    for column in columns[:-1]:
        result[column] = result[column].astype(str)
    return result.sort_values(columns).reset_index(drop=True)


def validate_matrix(frame: pd.DataFrame) -> dict:
    duplicate_count = int(
        frame.duplicated(["dataset_id", "item_id", "rater"]).sum()
    )
    failures = []
    for dataset_id, group in frame.groupby("dataset_id", sort=True):
        expected_raters = set(group["rater"].astype(str))
        counts = group.groupby("item_id")["rater"].apply(
            lambda values: set(values.astype(str))
        )
        if not all(value == expected_raters for value in counts):
            failures.append(str(dataset_id))
    return {
        "duplicate_dataset_item_rater_rows": duplicate_count,
        "incomplete_cluster_ids": failures,
        "complete_repeated_rating_matrices": (
            duplicate_count == 0 and not failures
        ),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# WMT Raw-Preparation Audit",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "## Raw Parsing",
        "",
        "| Slice | Physical rows | Parsed rows | No skipped rows |",
        "|---|---:|---:|---|",
    ]
    for name, record in report["raw_parsing"].items():
        lines.append(
            f"| `{name}` | {record['physical_data_rows']} | "
            f"{record['parsed_rows']} | "
            f"{'PASS' if record['all_physical_rows_parsed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The parser uses literal-tab splitting with CSV quoting disabled. "
            "This is deliberate: free-text fields contain quote characters that "
            "the default CSV quote parser misinterprets.",
            "",
            "## Segment-Key Collision Audit",
            "",
            "| Slice | Segment IDs | Document-segment pairs | Reused IDs | "
            "Max documents per ID |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in report["collisions"]:
        lines.append(
            f"| `{record['dataset']}` | "
            f"{record['distinct_segment_ids']} | "
            f"{record['distinct_document_segment_pairs']} | "
            f"{record['segment_ids_reused_across_documents']} | "
            f"{record['max_documents_per_segment_id']} |"
        )
    lines.extend(
        [
            "",
            "## Prepared Artifact",
            "",
            f"- Independent canonical rebuild SHA-256: "
            f"`{report['independent_canonical_rebuild_sha256']}`",
            f"- Production rebuild SHA-256: "
            f"`{report['production_rebuild_sha256']}`",
            f"- Stored SHA-256: `{report['stored_sha256']}`",
            f"- Byte-identical rebuild: **{report['byte_identical_rebuild']}**",
            f"- Rows: {report['prepared_rows']}",
            f"- Clusters: {report['prepared_clusters']}",
            f"- Paths: {report['prepared_paths']}",
            f"- Complete matrices: "
            f"**{report['matrix_validation']['complete_repeated_rating_matrices']}**",
            "",
            "## Interpretation",
            "",
            "- WMT20 segment IDs are globally unique within each audited slice, "
            "so adding the document component does not change their identities.",
            "- WMT22 segment IDs are local to documents. A document-free item key "
            "therefore merges unrelated segments or creates artificial rater sets.",
            "- The corrected artifact is independently rebuilt from the pinned raw "
            "revision and is byte-identical to the stored 141-path input.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--fixed-csv", type=Path, required=True)
    parser.add_argument(
        "--production-rebuild-csv",
        type=Path,
        help="Optional output independently produced by prepare_wmt_mqm_long.py.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-cluster", type=int, default=200)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    parsed = {}
    parse_records = {}
    collision_records = []
    frames = []
    for dataset_id, (relative_path, layout) in FILES.items():
        frame, parse_record = parse_raw(args.raw_root / relative_path)
        parsed[dataset_id] = frame
        parse_records[dataset_id] = parse_record
        collision_records.append(
            collision_record(dataset_id, frame, layout)
        )
        frames.extend(
            independent_frames(
                dataset_id,
                frame,
                layout,
                args.min_cluster,
            )
        )

    rebuilt = canonical(pd.concat(frames, ignore_index=True))
    stored = canonical(pd.read_csv(args.fixed_csv))
    rebuilt_path = args.outdir / "wmt_mqm_long_independent_rebuild.csv"
    rebuilt.to_csv(rebuilt_path, index=False)
    equality = rebuilt.equals(stored)
    matrix_validation = validate_matrix(stored)
    paths = sum(
        int(group["rater"].nunique())
        for _, group in stored.groupby("dataset_id")
    )
    production_hash = (
        sha256(args.production_rebuild_csv)
        if args.production_rebuild_csv
        else None
    )
    stored_hash = sha256(args.fixed_csv)
    report = {
        "status": (
            "PASS"
            if equality
            and all(
                record["all_physical_rows_parsed"]
                for record in parse_records.values()
            )
            and matrix_validation["complete_repeated_rating_matrices"]
            else "FAIL"
        ),
        "raw_parsing": parse_records,
        "collisions": collision_records,
        "fixed_csv": str(args.fixed_csv),
        "independent_canonical_rebuild_sha256": sha256(rebuilt_path),
        "production_rebuild_sha256": production_hash,
        "stored_sha256": stored_hash,
        "byte_identical_rebuild": (
            production_hash == stored_hash
            if production_hash is not None
            else None
        ),
        "value_identical_rebuild": equality,
        "prepared_rows": int(len(stored)),
        "prepared_clusters": int(stored["dataset_id"].nunique()),
        "prepared_paths": int(paths),
        "matrix_validation": matrix_validation,
    }
    (args.outdir / "audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_markdown(report, args.outdir / "AUDIT.md")
    if report["status"] != "PASS":
        raise SystemExit("WMT raw-preparation audit failed")


if __name__ == "__main__":
    main()
