#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-python}"
QWEN7_RUN="${QWEN7_RUN:?set QWEN7_RUN to the completed Qwen-7B run}"
OUTDIR="${OUTDIR:?set OUTDIR to a fresh analysis directory}"
QWEN3_POOL="${QWEN3_POOL:-${ROOT}/outputs/exp09_dpo_v4_20260710_051319/pool_scored.jsonl}"
ZEPHYR7_POOL="${ZEPHYR7_POOL:-${ROOT}/outputs/exp11_cross_model_20260710_071053/pool_scored.jsonl}"
optional_args=()
if [[ -n "${QWEN14_RUN:-}" ]]; then
  optional_args+=(--qwen-14b "${QWEN14_RUN}/pool_scored.jsonl")
fi
if [[ -n "${QWEN32_RUN:-}" ]]; then
  optional_args+=(--qwen-32b "${QWEN32_RUN}/pool_scored.jsonl")
fi

export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
"${PY}" "${SCRIPT_DIR}/analyze_size_control.py" \
  --qwen-3b "${QWEN3_POOL}" \
  --qwen-7b "${QWEN7_RUN}/pool_scored.jsonl" \
  "${optional_args[@]}" \
  --zephyr-7b "${ZEPHYR7_POOL}" \
  --random-seed 0 --n-boot 5000 \
  --outdir "${OUTDIR}"
