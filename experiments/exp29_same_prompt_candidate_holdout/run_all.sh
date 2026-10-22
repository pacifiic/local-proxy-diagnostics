#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTDIR="${OUTDIR:-${ROOT}/outputs/exp29_same_prompt_candidate_holdout_20260724}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${ROOT}/outputs/exp29_same_prompt_candidate_holdout_analysis_20260724}"
N_REPEATS="${N_REPEATS:-20}"
N_BOOTSTRAP="${N_BOOTSTRAP:-100}"
JOBS="${JOBS:-12}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" "${ROOT}/experiments/exp29_same_prompt_candidate_holdout/01_same_prompt_candidate_holdout.py" \
  --pool "ministral_seed0=${ROOT}/outputs/exp28_ministral3_8b_seed0_20260724/pool_scored.jsonl" \
  --pool "ministral_seed1=${ROOT}/outputs/exp28_ministral3_8b_seed1_20260724/pool_scored.jsonl" \
  --pool "ministral_seed2=${ROOT}/outputs/exp28_ministral3_8b_seed2_20260724/pool_scored.jsonl" \
  --pool "llama_seed0=${ROOT}/outputs/exp28_llama31_8b_seed0_20260724/pool_scored.jsonl" \
  --pool "llama_seed1=${ROOT}/outputs/exp28_llama31_8b_seed1_20260724/pool_scored.jsonl" \
  --pool "llama_seed2=${ROOT}/outputs/exp28_llama31_8b_seed2_20260724/pool_scored.jsonl" \
  --outdir "${OUTDIR}" \
  --n-repeats "${N_REPEATS}" \
  --n-calibration 10 \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --random-seed 0 \
  --jobs "${JOBS}"

"${PYTHON}" "${ROOT}/experiments/exp29_same_prompt_candidate_holdout/02_analyze.py" \
  --input-dir "${OUTDIR}" \
  --outdir "${ANALYSIS_DIR}"

"${PYTHON}" "${ROOT}/experiments/exp29_same_prompt_candidate_holdout/03_audit.py" \
  --input-dir "${OUTDIR}" \
  --report "${OUTDIR}/integrity_audit.json"

"${PYTHON}" "${ROOT}/experiments/exp29_same_prompt_candidate_holdout/04_audit_source_pipeline.py" \
  --run "${ROOT}/outputs/exp28_ministral3_8b_seed0_20260724" \
  --run "${ROOT}/outputs/exp28_ministral3_8b_seed1_20260724" \
  --run "${ROOT}/outputs/exp28_ministral3_8b_seed2_20260724" \
  --run "${ROOT}/outputs/exp28_llama31_8b_seed0_20260724" \
  --run "${ROOT}/outputs/exp28_llama31_8b_seed1_20260724" \
  --run "${ROOT}/outputs/exp28_llama31_8b_seed2_20260724" \
  --out "${OUTDIR}/source_pipeline_audit.json"
