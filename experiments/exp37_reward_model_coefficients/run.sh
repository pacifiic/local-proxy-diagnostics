#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
: "${ITEMS_CSV:?Set ITEMS_CSV to the canonical SummEval article/summary CSV}"
: "${ALIGNED_JSONL:?Set ALIGNED_JSONL to model_annotations.aligned.jsonl for source validation}"
: "${OUT_ROOT:?Set OUT_ROOT to a new directory}"
: "${CUDA_VISIBLE_DEVICES:?Choose one allocated GPU explicitly}"
if [[ -e "$OUT_ROOT" ]]; then
  printf 'Refusing to overwrite %s\n' "$OUT_ROOT" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT_ROOT")"
mkdir "$OUT_ROOT"
"$PY" "$SCRIPT_DIR/prepare.py" --items-csv "$ITEMS_CSV" --aligned-jsonl "$ALIGNED_JSONL" --outdir "$OUT_ROOT/prepared"
for alias in skywork armorm deberta; do
  "$PY" "$SCRIPT_DIR/score.py" --prepared "$OUT_ROOT/prepared" --alias "$alias" \
    --outdir "$OUT_ROOT/scores/$alias" --allow-remote-code
done
"$PY" "$SCRIPT_DIR/analyze.py" --prepared "$OUT_ROOT/prepared" \
  --scores-root "$OUT_ROOT/scores" --outdir "$OUT_ROOT/analysis"
