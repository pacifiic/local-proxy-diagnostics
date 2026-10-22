"""Repair SummEval GPT-proxy item ids by matching candidate-summary text.

The GPT scoring prompt does not include a model id.  If an item-preparation
artifact has the right article and candidate summary text but the wrong
SummEval model label, we can repair the aggregate proxy artifact without any
new API calls by matching each candidate summary back to
model_annotations.aligned.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def normalize_summary(text: Any) -> str:
    text = str(text).lower()
    text = text.replace("-lrb-", "(").replace("-rrb-", ")")
    return re.sub(r"\s+", " ", text).strip()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_truth(jsonl_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "article_id": item["id"],
                    "true_model_id": item["model_id"],
                    "decoded": item["decoded"],
                    "decoded_norm": normalize_summary(item["decoded"]),
                    "reference0": item.get("references", [""])[0] if item.get("references") else "",
                }
            )
    truth = pd.DataFrame(rows)
    if truth.duplicated(["article_id", "true_model_id"]).any():
        raise ValueError("Truth JSONL has duplicate (article_id, model_id) rows.")
    return truth


def load_current_items(items_csv: Path) -> pd.DataFrame:
    items = pd.read_csv(items_csv)
    required = {"article_id", "model_id", "dimension", "summary", "item_id"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items CSV is missing required columns: {sorted(missing)}")

    base = (
        items[["article_id", "model_id", "summary", "item_id"]]
        .drop_duplicates()
        .copy()
    )
    if base.duplicated(["article_id", "model_id"]).any():
        raise ValueError("items CSV maps one (article_id, model_id) to multiple summaries.")
    base["summary_norm"] = base["summary"].map(normalize_summary)
    base["summary_hash"] = base["summary_norm"].map(short_hash)
    return base


def validate_text_match(truth: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    mapping = truth.merge(
        current[["article_id", "model_id", "item_id", "summary_norm", "summary_hash"]],
        left_on=["article_id", "decoded_norm"],
        right_on=["article_id", "summary_norm"],
        how="left",
        suffixes=("_truth", "_current"),
    )
    missing = mapping["item_id"].isna()
    if missing.any():
        examples = mapping.loc[missing, ["article_id", "true_model_id"]].head(10).to_dict("records")
        raise ValueError(f"Could not match {int(missing.sum())} truth summaries. Examples: {examples}")

    per_truth = mapping.groupby(["article_id", "true_model_id"]).size()
    per_current = mapping.groupby(["article_id", "model_id"]).size()
    print("Truth items:", len(truth))
    print("Current item labels:", len(current))
    print("Matched rows:", len(mapping))
    print("Matches per truth item:", per_truth.value_counts().sort_index().to_dict())
    print("Truth matches per current item:", per_current.value_counts().sort_index().to_dict())
    print("Ambiguous truth items from duplicate summaries:", int((per_truth > 1).sum()))
    print("Ambiguous current items from duplicate summaries:", int((per_current > 1).sum()))
    return mapping


def build_corrected_aggregate(
    raw_scores: pd.DataFrame,
    current: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    required = {"dimension", "item_id", "judge_rep", "proxy_score"}
    missing = required - set(raw_scores.columns)
    if missing:
        raise ValueError(f"raw score CSV is missing required columns: {sorted(missing)}")

    raw_scores = raw_scores.copy()
    duplicate_key = ["dimension", "item_id", "judge_rep"]
    duplicate_mask = raw_scores.duplicated(duplicate_key, keep="last")
    if duplicate_mask.any():
        n_dup = int(duplicate_mask.sum())
        print(f"Dropping {n_dup} duplicate raw score rows before remapping.")
        raw_scores = raw_scores.loc[~duplicate_mask].copy()

    enriched = raw_scores.merge(
        current[["article_id", "model_id", "item_id", "summary_norm", "summary_hash"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    if enriched["article_id"].isna().any():
        raise ValueError("Some raw score rows do not match the current item CSV.")

    text_scores = (
        enriched.groupby(["dimension", "article_id", "summary_norm"], dropna=False)["proxy_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "proxy_score_mean",
                "std": "proxy_score_sd",
                "count": "n_judges",
            }
        )
    )

    truth_expanded = truth.loc[truth.index.repeat(len(DIMENSIONS))].copy()
    truth_expanded["dimension"] = DIMENSIONS * len(truth)
    truth_expanded["item_id"] = (
        truth_expanded["article_id"].astype(str) + "||" + truth_expanded["true_model_id"].astype(str)
    )
    corrected = truth_expanded.merge(
        text_scores,
        left_on=["dimension", "article_id", "decoded_norm"],
        right_on=["dimension", "article_id", "summary_norm"],
        how="left",
        validate="many_to_one",
    )
    if corrected["proxy_score_mean"].isna().any():
        missing = corrected.loc[
            corrected["proxy_score_mean"].isna(), ["dimension", "article_id", "true_model_id"]
        ].head(10)
        raise ValueError(f"Missing corrected proxy scores. Examples: {missing.to_dict('records')}")

    out = corrected[["dimension", "item_id", "proxy_score_mean", "proxy_score_sd", "n_judges"]].copy()
    out = out.sort_values(["dimension", "item_id"]).reset_index(drop=True)
    if out.duplicated(["dimension", "item_id"]).any():
        raise ValueError("Corrected aggregate has duplicate (dimension, item_id) rows.")
    return out


def write_report(path: Path, mapping: pd.DataFrame, corrected: pd.DataFrame) -> None:
    per_truth = mapping.groupby(["article_id", "true_model_id"]).size()
    report = {
        "status": "completed",
        "truth_items": int(mapping[["article_id", "true_model_id"]].drop_duplicates().shape[0]),
        "matched_rows": int(len(mapping)),
        "ambiguous_truth_items": int((per_truth > 1).sum()),
        "corrected_rows": int(len(corrected)),
        "n_judges_values": sorted(int(x) for x in corrected["n_judges"].dropna().unique()),
        "dimensions": sorted(str(x) for x in corrected["dimension"].unique()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-csv", required=True, type=Path)
    parser.add_argument("--raw-scores-csv", required=True, type=Path)
    parser.add_argument("--summeval-jsonl", required=True, type=Path)
    parser.add_argument("--out-aggregate-csv", required=True, type=Path)
    parser.add_argument("--out-mapping-csv", type=Path)
    parser.add_argument("--out-report-json", type=Path)
    args = parser.parse_args()

    truth = load_truth(args.summeval_jsonl)
    current = load_current_items(args.items_csv)
    mapping = validate_text_match(truth, current)
    raw_scores = pd.read_csv(args.raw_scores_csv)
    corrected = build_corrected_aggregate(raw_scores, current, truth)

    args.out_aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.out_aggregate_csv, index=False)
    print(f"Wrote corrected aggregate: {args.out_aggregate_csv}")
    print("Corrected n_judges values:", sorted(corrected["n_judges"].unique().tolist()))

    if args.out_mapping_csv:
        args.out_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(args.out_mapping_csv, index=False)
        print(f"Wrote mapping audit: {args.out_mapping_csv}")
    if args.out_report_json:
        write_report(args.out_report_json, mapping, corrected)
        print(f"Wrote report: {args.out_report_json}")


if __name__ == "__main__":
    main()
