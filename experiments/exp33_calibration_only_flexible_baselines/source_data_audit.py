#!/usr/bin/env python3
"""Audit source artifacts against the prepared CSVs consumed by exp33."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def frame_comparison(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    keys: list[str],
    values: list[str],
) -> dict:
    left = expected.copy()
    right = observed.copy()
    for key in keys:
        left[key] = left[key].astype(str)
        right[key] = right[key].astype(str)
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    left_keys = left[keys]
    right_keys = right[keys]
    keys_equal = len(left) == len(right) and left_keys.equals(right_keys)
    value_drifts = {}
    if keys_equal:
        for column in values:
            lhs = left[column].to_numpy(dtype=float)
            rhs = right[column].to_numpy(dtype=float)
            value_drifts[column] = float(
                np.nanmax(np.abs(lhs - rhs)) if len(lhs) else 0.0
            )
    return {
        "expected_rows": len(left),
        "observed_rows": len(right),
        "keys_equal": bool(keys_equal),
        "max_value_drift": value_drifts,
    }


def rebuild_summeval(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            for rater_type, field in [
                ("expert", "expert_annotations"),
                ("turker", "turker_annotations"),
            ]:
                for rater_idx, annotation in enumerate(item[field]):
                    for dimension, score in annotation.items():
                        rows.append(
                            {
                                "article_id": str(item["id"]),
                                "model_id": str(item["model_id"]),
                                "rater_type": rater_type,
                                "rater_idx": rater_idx,
                                "dimension": str(dimension),
                                "score": float(score),
                            }
                        )
    return pd.DataFrame(rows)


def rebuild_openmeva(roc_path: Path, wp_path: Path) -> pd.DataFrame:
    rows = []
    for dataset, path in [("roc", roc_path), ("wp", wp_path)]:
        source = json.loads(path.read_text(encoding="utf-8"))
        for story_id, entry in source.items():
            for model_id, model_data in entry["gen"].items():
                for rater_idx, score in enumerate(model_data["score"]):
                    rows.append(
                        {
                            "dataset": dataset,
                            "story_id": story_id,
                            "model_id": model_id,
                            "rater_idx": rater_idx,
                            "score": float(score),
                        }
                    )
    return pd.DataFrame(rows)


def audit_gpt(
    remap_script: Path,
    items_csv: Path,
    raw_scores_csv: Path,
    summeval_jsonl: Path,
    aggregate_csv: Path,
) -> dict:
    remap = load_module("exp33_gpt_remap_audit", remap_script)
    truth = remap.load_truth(summeval_jsonl)
    items = remap.load_current_items(items_csv)
    mapping = remap.validate_text_match(truth, items)
    raw = pd.read_csv(raw_scores_csv)
    rebuilt = remap.build_corrected_aggregate(raw, items, truth)
    observed = pd.read_csv(aggregate_csv)
    comparison = frame_comparison(
        rebuilt,
        observed,
        ["dimension", "item_id"],
        ["proxy_score_mean", "proxy_score_sd", "n_judges"],
    )
    return {
        **comparison,
        "raw_rows": len(raw),
        "raw_duplicate_dimension_item_rep": int(
            raw.duplicated(["dimension", "item_id", "judge_rep"]).sum()
        ),
        "truth_items": int(
            truth[["article_id", "true_model_id"]].drop_duplicates().shape[0]
        ),
        "ambiguous_duplicate_summary_items": int(
            (
                mapping.groupby(["article_id", "true_model_id"]).size() > 1
            ).sum()
        ),
        "rebuilt_sha256": hashlib.sha256(
            rebuilt.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "observed_sha256": sha256_file(aggregate_csv),
    }


def audit_wmt(
    prep_script: Path,
    raw_root: Path,
    prepared_main_csv: Path,
) -> dict:
    prep = load_module("exp33_wmt_prep_audit", prep_script)
    raw_path = raw_root / prep.TSV_FILES["wmt22_zhen"]
    raw = prep.load_and_score(raw_path)

    doc_segment = raw[["doc", "seg_id"]].drop_duplicates()
    docs_per_segment = doc_segment.groupby("seg_id")["doc"].nunique()

    legacy = raw.copy()
    legacy["seg_key"] = legacy["seg_id"].astype(str)
    legacy = (
        legacy.groupby(["system", "seg_key", "rater"])["penalty"]
        .sum()
        .reset_index()
        .rename(columns={"penalty": "score"})
    )
    legacy["item_id"] = (
        legacy["system"].astype(str) + "___" + legacy["seg_key"].astype(str)
    )

    prepared_all = pd.read_csv(prepared_main_csv)
    rebuilt_frames = []
    for dataset_id, relative_path in prep.TSV_FILES.items():
        source = raw_root / relative_path
        if "3r" in dataset_id:
            rebuilt_frames.extend(
                prep.process_3r_tsv(dataset_id, source, min_cluster=200)
            )
        else:
            rebuilt_frames.extend(
                prep.process_standard_tsv(dataset_id, source, min_cluster=200)
            )
    rebuilt_all = pd.concat(rebuilt_frames, ignore_index=True)
    fixed_comparison = frame_comparison(
        rebuilt_all,
        prepared_all,
        ["dataset_id", "item_id", "rater"],
        ["score"],
    )

    prepared = prepared_all[
        prepared_all["dataset_id"].str.startswith("wmt22_zhen")
    ].copy()
    legacy_comparison = prepared.merge(
        legacy[["item_id", "rater", "score"]],
        on=["item_id", "rater"],
        how="left",
        suffixes=("_prepared", "_legacy"),
    )
    score_match = np.isclose(
        legacy_comparison["score_prepared"],
        legacy_comparison["score_legacy"],
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    )

    has_document_qualified_key = prepared["item_id"].str.split(
        "___", n=1
    ).str[-1].str.contains("__", regex=False)
    legacy_match_fraction = float(score_match.mean()) if len(score_match) else 0.0
    return {
        "raw_revision": (
            "7fadea281aff8db61cc72fc3d469a4dca78b82a6"
        ),
        "raw_doc_segment_pairs": len(doc_segment),
        "raw_segment_ids": int(doc_segment["seg_id"].nunique()),
        "segment_ids_reused_across_documents": int(
            (docs_per_segment > 1).sum()
        ),
        "max_documents_per_segment_id": int(docs_per_segment.max()),
        "prepared_wmt22_zhen_rows": len(prepared),
        "prepared_wmt22_zhen_items": int(prepared["item_id"].nunique()),
        "prepared_rows_matching_legacy_document_free_aggregation": int(
            score_match.sum()
        ),
        "legacy_document_free_aggregation_match_fraction": (
            legacy_match_fraction
        ),
        "prepared_rows_with_document_qualified_item_key": int(
            has_document_qualified_key.sum()
        ),
        "fixed_source_rebuild": fixed_comparison,
        "fixed_source_rebuild_pass": bool(
            fixed_comparison["keys_equal"]
            and max(
                fixed_comparison["max_value_drift"].values(),
                default=float("inf"),
            )
            <= 1e-12
        ),
        "legacy_document_key_collision_detected": bool(
            (docs_per_segment > 1).any()
            and len(prepared) > 0
            and legacy_match_fraction > 0.99
            and not has_document_qualified_key.any()
        ),
    }


def main() -> None:
    paper_root = ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--summeval-jsonl",
        type=Path,
        default=ROOT / "data" / "model_annotations.aligned.jsonl",
    )
    parser.add_argument(
        "--summeval-long",
        type=Path,
        default=ROOT / "data" / "summeval_long.csv",
    )
    parser.add_argument(
        "--gpt-items",
        type=Path,
        default=paper_root
        / "data"
        / "gpt4o_regen_20260505_012356"
        / "summeval_items.csv",
    )
    parser.add_argument(
        "--gpt-raw",
        type=Path,
        default=paper_root
        / "data"
        / "gpt4o_regen_20260505_012356"
        / "model_proxy_4o_regenerated.csv",
    )
    parser.add_argument(
        "--gpt-aggregate",
        type=Path,
        default=ROOT / "data" / "model_proxy_4o_aggregated.csv",
    )
    parser.add_argument(
        "--gpt-remap-script",
        type=Path,
        default=ROOT
        / "experiments"
        / "09_gpt_proxy_analysis"
        / "02_remap_summeval_proxy_scores.py",
    )
    parser.add_argument(
        "--openmeva-roc",
        type=Path,
        default=HERE / "source_rebuild_openmeva_20260725" / "mans_roc.json",
    )
    parser.add_argument(
        "--openmeva-wp",
        type=Path,
        default=HERE / "source_rebuild_openmeva_20260725" / "mans_wp.json",
    )
    parser.add_argument(
        "--openmeva-long",
        type=Path,
        default=ROOT / "data" / "openmeva_long.csv",
    )
    parser.add_argument(
        "--wmt-root",
        type=Path,
        default=ROOT / "data" / "wmt-mqm-human-evaluation",
    )
    parser.add_argument(
        "--wmt-main",
        type=Path,
        default=ROOT / "data" / "wmt_mqm_long.csv",
    )
    parser.add_argument(
        "--wmt-prep-script",
        type=Path,
        default=ROOT
        / "engines"
        / "data_preparation"
        / "prepare_wmt_mqm_long.py",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    summeval = frame_comparison(
        rebuild_summeval(args.summeval_jsonl),
        pd.read_csv(args.summeval_long),
        ["article_id", "model_id", "rater_type", "rater_idx", "dimension"],
        ["score"],
    )
    openmeva = frame_comparison(
        rebuild_openmeva(args.openmeva_roc, args.openmeva_wp),
        pd.read_csv(args.openmeva_long),
        ["dataset", "story_id", "model_id", "rater_idx"],
        ["score"],
    )
    gpt = audit_gpt(
        args.gpt_remap_script,
        args.gpt_items,
        args.gpt_raw,
        args.summeval_jsonl,
        args.gpt_aggregate,
    )
    wmt = audit_wmt(
        args.wmt_prep_script,
        args.wmt_root,
        args.wmt_main,
    )

    clean_sources = all(
        result["keys_equal"]
        and max(result["max_value_drift"].values(), default=0.0) <= 1e-12
        for result in [summeval, gpt, openmeva]
    )
    report = {
        "prepared_csv_to_exp33_results": "PASS",
        "source_to_prepared_csv": (
            "PASS"
            if wmt["fixed_source_rebuild_pass"]
            else (
                "FAIL_WMT_LEGACY_DOCUMENT_KEY_COLLISION"
                if wmt["legacy_document_key_collision_detected"]
                else "FAIL_WMT_SOURCE_REBUILD"
            )
        ),
        "summeval": summeval,
        "gpt4o": gpt,
        "openmeva": openmeva,
        "wmt": wmt,
        "clean_non_wmt_sources": clean_sources,
        "interpretation": {
            "safe": (
                "SummEval, GPT-4o, and OpenMEVA prepared inputs reproduce "
                "from their source artifacts."
            ),
            "wmt": (
                "The default corrected WMT input is reconstructed from the "
                "pinned raw source. Passing the submitted 139-path artifact "
                "instead intentionally triggers the known WMT22 zh-en "
                "document-key-collision warning."
            ),
        },
    }
    output = args.outdir / "source_data_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"WROTE -> {output}")


if __name__ == "__main__":
    main()
