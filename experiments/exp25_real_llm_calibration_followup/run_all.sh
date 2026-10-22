#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-python}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
WITHIN_OUT="${WITHIN_OUT:-${SCRIPT_DIR}/outputs_within_${DATE_TAG}}"
TRANSFER_OUT="${TRANSFER_OUT:-${SCRIPT_DIR}/outputs_transfer_${DATE_TAG}}"
JOBS="${JOBS:-16}"
N_REPEATS="${N_REPEATS:-200}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

"${PY}" "${SCRIPT_DIR}/01_within_candidate_pool.py" \
  --candidates "${ROOT}/outputs/exp05_real_llm_20260706_032457/candidates_scored.csv" \
  --n-repeats "${N_REPEATS}" --random-seed 0 --jobs "${JOBS}" \
  --outdir "${WITHIN_OUT}"

"${PY}" "${SCRIPT_DIR}/02_cross_model_transfer.py" \
  --qwen-pool "${ROOT}/outputs/exp09_dpo_v4_20260710_051319/pool_scored.jsonl" \
  --zephyr-pool "${ROOT}/outputs/exp11_cross_model_20260710_071053/pool_scored.jsonl" \
  --random-seed 0 --outdir "${TRANSFER_OUT}"

"${PY}" "${SCRIPT_DIR}/03_audit_and_artifacts.py" \
  --within-outdir "${WITHIN_OUT}" \
  --transfer-outdir "${TRANSFER_OUT}"

"${PY}" "${SCRIPT_DIR}/04_data_layer_audit.py" \
  --repo-root "${ROOT}" \
  --within-outdir "${WITHIN_OUT}" \
  --transfer-outdir "${TRANSFER_OUT}" \
  --out-json "${WITHIN_OUT}/data_layer_audit.json"

printf 'WITHIN_OUT=%s\nTRANSFER_OUT=%s\n' "${WITHIN_OUT}" "${TRANSFER_OUT}"
