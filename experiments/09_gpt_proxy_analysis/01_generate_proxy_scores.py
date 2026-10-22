from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from common_experiment_utils import ensure_item_id, mkdir, parse_cols, short_hash


_thread_local = threading.local()


class RatingOutput(BaseModel):
    score: float = Field(description="The final scalar quality score for the candidate output.")
    confidence: float = Field(description="Confidence in [0,1].")
    rationale: str = Field(description="A short rationale, at most 40 words.")


def get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(timeout=60.0)
        _thread_local.client = client
    return client


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_key_value(value: Any) -> str:
    """Normalize resume keys so CSV dtype inference cannot create duplicate calls."""
    if pd.isna(value):
        return "<NA>"
    return str(value)


def render_user_prompt(template: str, row: Dict[str, Any]) -> str:
    try:
        return template.format(**row)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Template refers to missing column: {missing}") from exc


def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    score_min: float,
    score_max: float,
    max_retries: int,
    timeout_sleep: float,
) -> RatingOutput:
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=model,
                store=False,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=RatingOutput,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("Model returned no parsed output.")
            parsed.score = clamp(float(parsed.score), score_min, score_max)
            parsed.confidence = clamp(float(parsed.confidence), 0.0, 1.0)
            parsed.rationale = str(parsed.rationale).strip()
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            sleep_s = timeout_sleep * (2**attempt)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="Rate items with an OpenAI model and save item-level proxy scores.")
    ap.add_argument("--items-csv", required=True, help="CSV with one row per item.")
    ap.add_argument("--out-csv", required=True, help="Output CSV with proxy scores.")
    ap.add_argument("--item-cols", default="item_id", help="Comma-separated columns uniquely identifying an item.")
    ap.add_argument("--group-cols", default="", help="Optional grouping columns to carry through.")
    ap.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model name.")
    ap.add_argument("--system-prompt-file", required=True, help="Path to the system prompt txt file.")
    ap.add_argument("--user-template-file", required=True, help="Path to the user prompt template txt file.")
    ap.add_argument("--score-min", type=float, default=1.0)
    ap.add_argument("--score-max", type=float, default=5.0)
    ap.add_argument("--replicates", type=int, default=1, help="Number of independent calls per item.")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of items (0 = all).")
    ap.add_argument("--workers", type=int, default=1, help="Number of concurrent workers.")
    ap.add_argument("--dry-run", action="store_true", help="Print one rendered prompt and exit.")
    args = ap.parse_args()

    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)

    items = pd.read_csv(args.items_csv)
    items = ensure_item_id(items, item_cols)
    if args.limit > 0:
        items = items.head(args.limit).copy()

    system_prompt = load_text(args.system_prompt_file)
    user_template = load_text(args.user_template_file)

    if args.dry_run:
        row = items.iloc[0].to_dict()
        print("=== SYSTEM PROMPT ===")
        print(system_prompt)
        print("=== USER PROMPT ===")
        print(render_user_prompt(user_template, row))
        return

    out_path = Path(args.out_csv)
    mkdir(out_path.parent)

    key_cols = [*group_cols, "item_id", "judge_rep"]
    done_keys = set()
    existing = None
    if out_path.exists():
        existing = pd.read_csv(out_path)
        missing_key_cols = [c for c in key_cols if c not in existing.columns]
        if missing_key_cols:
            raise ValueError(
                "Existing output cannot be resumed because it is missing key "
                f"columns {missing_key_cols}. Use a fresh --out-csv or regenerate."
            )
        done_keys = {
            tuple(normalize_key_value(row[c]) for c in [*group_cols, "item_id"]) + (int(row["judge_rep"]),)
            for _, row in existing[key_cols].iterrows()
        }

    # Build list of tasks to execute
    tasks: List[Dict[str, Any]] = []
    iterator = list(items.to_dict(orient="records"))
    for row in iterator:
        item_id = str(row["item_id"])
        prompt = render_user_prompt(user_template, row)
        prompt_hash = short_hash(system_prompt + "\n\n" + prompt)
        for rep in range(args.replicates):
            key = tuple(normalize_key_value(row.get(c)) for c in group_cols) + (normalize_key_value(item_id), rep)
            if key in done_keys:
                continue
            tasks.append({
                "item_id": item_id,
                "rep": rep,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "row": row,
            })

    def _process_one(task: Dict[str, Any]) -> Dict[str, Any]:
        result = call_model(
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=task["prompt"],
            score_min=args.score_min,
            score_max=args.score_max,
            max_retries=args.max_retries,
            timeout_sleep=args.retry_sleep,
        )
        out_row: Dict[str, Any] = {
            "item_id": task["item_id"],
            "judge_rep": task["rep"],
            "model": args.model,
            "prompt_hash": task["prompt_hash"],
        }
        for c in group_cols:
            if c in task["row"]:
                out_row[c] = task["row"][c]
        out_row.update({
            "proxy_score": result.score,
            "confidence": result.confidence,
            "rationale": result.rationale,
        })
        return out_row

    rows: List[Dict[str, Any]] = []
    write_lock = threading.Lock()

    def _flush(force: bool = False) -> None:
        nonlocal rows
        if not rows:
            return
        if not force and len(rows) < 100:
            return
        chunk = pd.DataFrame(rows)
        with write_lock:
            if out_path.exists():
                chunk.to_csv(out_path, mode="a", index=False, header=False)
            else:
                chunk.to_csv(out_path, index=False)
        rows = []

    n_workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="rating items"):
            out_row = future.result()
            rows.append(out_row)
            _flush()

    _flush(force=True)

    scored = pd.read_csv(out_path)
    duplicate_mask = scored.duplicated(key_cols, keep="last")
    if duplicate_mask.any():
        n_dup = int(duplicate_mask.sum())
        print(f"Warning: dropping {n_dup} duplicate completed calls before aggregation.")
        scored = scored.loc[~duplicate_mask].copy()
        scored.to_csv(out_path, index=False)
    summary = (
        scored.groupby([*(group_cols or []), "item_id"], dropna=False)["proxy_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "proxy_score_mean", "std": "proxy_score_sd", "count": "n_judges"})
    )
    summary_path = out_path.with_name(out_path.stem + "_aggregated.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved raw scores to {out_path}")
    print(f"Saved aggregated scores to {summary_path}")


if __name__ == "__main__":
    main()
