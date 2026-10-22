"""
Parse SummEval model_annotations.aligned.jsonl into long-format CSV files.

Output files:
  data/summeval_long.csv      — all raters (expert + turker), one row per (article, model, rater, dimension)
  data/summeval_wide.csv      — mean scores per (article, model, dimension) by rater group
  data/summeval_scores.csv    — pivoted: one row per (article, model), columns = dim_expert_mean, dim_turker_mean
"""

import argparse
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INPUT_FILE = DATA_DIR / "model_annotations.aligned.jsonl"
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def parse_summeval(filepath: Path) -> pd.DataFrame:
    """Parse JSONL into long-format DataFrame."""
    rows = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            article_id = d["id"]
            model_id = d["model_id"]

            # Expert annotations (3 raters)
            for rater_idx, ann in enumerate(d["expert_annotations"]):
                for dim in DIMENSIONS:
                    rows.append({
                        "article_id": article_id,
                        "model_id": model_id,
                        "rater_type": "expert",
                        "rater_idx": rater_idx,
                        "dimension": dim,
                        "score": ann[dim],
                    })

            # Turker (crowd) annotations (5 raters)
            for rater_idx, ann in enumerate(d["turker_annotations"]):
                for dim in DIMENSIONS:
                    rows.append({
                        "article_id": article_id,
                        "model_id": model_id,
                        "rater_type": "turker",
                        "rater_idx": rater_idx,
                        "dimension": dim,
                        "score": ann[dim],
                    })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INPUT_FILE),
                    help="SummEval model_annotations.aligned.jsonl path.")
    ap.add_argument("--outdir", default=str(DATA_DIR),
                    help="Output directory (default: data/)")
    args = ap.parse_args()
    input_file = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_file} ...")
    df_long = parse_summeval(input_file)
    print(f"Long format: {len(df_long):,} rows")

    # Save long format
    long_path = outdir / "summeval_long.csv"
    df_long.to_csv(long_path, index=False)
    print(f"Saved: {long_path}")

    # Wide format: mean score per (article, model, rater_type, dimension)
    df_wide = (
        df_long
        .groupby(["article_id", "model_id", "rater_type", "dimension"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "mean_score"})
    )
    wide_path = outdir / "summeval_wide.csv"
    df_wide.to_csv(wide_path, index=False)
    print(f"Saved: {wide_path}")

    # Score matrix: one row per (article, model), columns = {dim}_{expert|turker}_mean
    pivot = df_wide.pivot_table(
        index=["article_id", "model_id"],
        columns=["dimension", "rater_type"],
        values="mean_score"
    )
    pivot.columns = [f"{dim}_{rtype}" for dim, rtype in pivot.columns]
    pivot = pivot.reset_index()
    scores_path = outdir / "summeval_scores.csv"
    pivot.to_csv(scores_path, index=False)
    print(f"Saved: {scores_path}")

    # Quick summary
    print("\n--- Score ranges ---")
    for dim in DIMENSIONS:
        for rtype in ["expert", "turker"]:
            col = f"{dim}_{rtype}"
            if col in pivot.columns:
                print(f"  {col}: {pivot[col].min():.2f} – {pivot[col].max():.2f}  "
                      f"(mean={pivot[col].mean():.2f})")

    print("\n--- Row counts ---")
    print(f"  summeval_long.csv  : {len(df_long):,} rows")
    print(f"  summeval_wide.csv  : {len(df_wide):,} rows")
    print(f"  summeval_scores.csv: {len(pivot):,} rows")


if __name__ == "__main__":
    main()
