#!/usr/bin/env bash
# W1 audit (stage 03) with a TRUE SFT->DPO pair, as recommended by the strategy
# Model lineage: mistral-7b-sft-beta is the exact SFT base from
# which zephyr-7b-beta was DPO-trained. Replaces the debug pairing
# (Qwen 0.5B vs 3B) for the R1 "real RLHF dynamics" defense.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REF_MODEL="${REF_MODEL:-HuggingFaceH4/mistral-7b-sft-beta}"
OPT_MODEL="${OPT_MODEL:-HuggingFaceH4/zephyr-7b-beta}"
RM_MODEL="${RM_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
PROMPTS_FILE="${PROMPTS_FILE:-${REPO_ROOT}/outputs/exp01_gpu_pipeline_20260705_201903/summeval_prompts.txt}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp01_w1_zephyr_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

{
  echo "ref_model=${REF_MODEL}"
  echo "opt_model=${OPT_MODEL}"
  echo "rm_model=${RM_MODEL}"
  echo "prompts_file=${PROMPTS_FILE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
} > "${OUT_ROOT}/run_config.txt"

"${PY}" "${SCRIPT_DIR}/03_w1_audit.py" \
  --ref-model "${REF_MODEL}" \
  --opt-model "${OPT_MODEL}" \
  --rm-model "${RM_MODEL}" \
  --prompts-file "${PROMPTS_FILE}" \
  --n-prompts 100 \
  --n-ref 32 \
  --n-opt 32 \
  --max-new-tokens 256 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --outdir "${OUT_ROOT}/w1_audit"

echo "DONE ${OUT_ROOT}"
