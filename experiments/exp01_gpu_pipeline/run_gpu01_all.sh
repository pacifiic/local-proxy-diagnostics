#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ITEMS_CSV="${ITEMS_CSV:-${REPO_ROOT}/data/summeval_items.csv}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp01_gpu_pipeline_$(date -u +%Y%m%d_%H%M%S)}"

RM_MODEL="${RM_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
RM_ALIAS="${RM_ALIAS:-skywork_v02}"
OPEN_JUDGE_MODEL="${OPEN_JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
OPEN_JUDGE_N_JUDGES="${OPEN_JUDGE_N_JUDGES:-5}"
OPEN_JUDGE_MAX_TOKENS="${OPEN_JUDGE_MAX_TOKENS:-200}"
RM_REVISION="${RM_REVISION:-}"
OPEN_JUDGE_REVISION="${OPEN_JUDGE_REVISION:-}"

# 03/04 defaults use cached open models for an end-to-end debug/full run.
# For a stricter DPO/SFT audit, override REF_MODEL and OPT_MODEL with a true
# SFT/DPO pair before launching.
REF_MODEL="${REF_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OPT_MODEL="${OPT_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
REF_REVISION="${REF_REVISION:-}"
OPT_REVISION="${OPT_REVISION:-}"
GEN_REVISION="${GEN_REVISION:-}"
JUDGE_REVISION="${JUDGE_REVISION:-}"

N_PROMPTS="${N_PROMPTS:-100}"
LIMIT_ITEMS="${LIMIT_ITEMS:-}"
N_REF="${N_REF:-32}"
N_OPT="${N_OPT:-32}"
N_CAL="${N_CAL:-24}"
N_POOL="${N_POOL:-48}"
N_LIST="${N_LIST:-2,4,8,16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

mkdir -p "${OUT_ROOT}"
{
  echo "out_root=${OUT_ROOT}"
  echo "python=${PY}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "vllm_worker_multiproc_method=${VLLM_WORKER_MULTIPROC_METHOD}"
  echo "items_csv=${ITEMS_CSV}"
  echo "rm_model=${RM_MODEL}"
  echo "rm_revision=${RM_REVISION}"
  echo "open_judge_model=${OPEN_JUDGE_MODEL}"
  echo "open_judge_revision=${OPEN_JUDGE_REVISION}"
  echo "open_judge_n_judges=${OPEN_JUDGE_N_JUDGES}"
  echo "open_judge_max_tokens=${OPEN_JUDGE_MAX_TOKENS}"
  echo "ref_model=${REF_MODEL}"
  echo "ref_revision=${REF_REVISION}"
  echo "opt_model=${OPT_MODEL}"
  echo "opt_revision=${OPT_REVISION}"
  echo "gen_model=${GEN_MODEL}"
  echo "gen_revision=${GEN_REVISION}"
  echo "judge_model=${JUDGE_MODEL}"
  echo "judge_revision=${JUDGE_REVISION}"
  echo "n_prompts=${N_PROMPTS}"
  echo "limit_items=${LIMIT_ITEMS}"
  echo "n_ref=${N_REF}"
  echo "n_opt=${N_OPT}"
  echo "n_cal=${N_CAL}"
  echo "n_pool=${N_POOL}"
  echo "n_list=${N_LIST}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
} > "${OUT_ROOT}/run_config.txt"

echo "[00] prepare SummEval prompts"
"${PY}" "${SCRIPT_DIR}/00_prepare_summeval_prompts.py" \
  --items-csv "${ITEMS_CSV}" \
  --limit-prompts "${N_PROMPTS}" \
  --out "${OUT_ROOT}/summeval_prompts.txt"

echo "[01] score SummEval summaries with reward model"
LIMIT_ARGS=()
if [[ -n "${LIMIT_ITEMS}" ]]; then
  LIMIT_ARGS=(--limit-items "${LIMIT_ITEMS}")
fi
REV_ARG=()
if [[ -n "${RM_REVISION}" ]]; then
  REV_ARG=(--model-revision "${RM_REVISION}")
fi
"${PY}" "${SCRIPT_DIR}/01_rm_proxy_summeval.py" \
  --items-csv "${ITEMS_CSV}" \
  --model "${RM_MODEL}" \
  --alias "${RM_ALIAS}" \
  --batch-size 16 \
  --outdir "${OUT_ROOT}/rm_proxy" \
  "${REV_ARG[@]}" \
  "${LIMIT_ARGS[@]}"

echo "[02] score SummEval summaries with open judge"
OJ_REV_ARG=()
if [[ -n "${OPEN_JUDGE_REVISION}" ]]; then
  OJ_REV_ARG=(--model-revision "${OPEN_JUDGE_REVISION}")
fi
"${PY}" "${SCRIPT_DIR}/02_open_judge_summeval.py" \
  --items-csv "${ITEMS_CSV}" \
  --model "${OPEN_JUDGE_MODEL}" \
  --n-judges "${OPEN_JUDGE_N_JUDGES}" \
  --max-tokens "${OPEN_JUDGE_MAX_TOKENS}" \
  --tensor-parallel 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --out "${OUT_ROOT}/open_judge/model_proxy_openjudge_aggregated.csv" \
  "${OJ_REV_ARG[@]}" \
  "${LIMIT_ARGS[@]}"

echo "[03] W1 prompt-conditional tilt audit"
W1_REV_ARGS=()
if [[ -n "${REF_REVISION}" ]]; then W1_REV_ARGS+=(--ref-revision "${REF_REVISION}"); fi
if [[ -n "${OPT_REVISION}" ]]; then W1_REV_ARGS+=(--opt-revision "${OPT_REVISION}"); fi
if [[ -n "${RM_REVISION}" ]]; then W1_REV_ARGS+=(--rm-revision "${RM_REVISION}"); fi
"${PY}" "${SCRIPT_DIR}/03_w1_audit.py" \
  --ref-model "${REF_MODEL}" \
  --opt-model "${OPT_MODEL}" \
  --rm-model "${RM_MODEL}" \
  --prompts-file "${OUT_ROOT}/summeval_prompts.txt" \
  --n-prompts "${N_PROMPTS}" \
  --n-ref "${N_REF}" \
  --n-opt "${N_OPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${W1_REV_ARGS[@]}" \
  --outdir "${OUT_ROOT}/w1_audit"

echo "[04] fresh-generation BoN"
BON_REV_ARGS=()
if [[ -n "${GEN_REVISION}" ]]; then BON_REV_ARGS+=(--gen-revision "${GEN_REVISION}"); fi
if [[ -n "${RM_REVISION}" ]]; then BON_REV_ARGS+=(--rm-revision "${RM_REVISION}"); fi
if [[ -n "${JUDGE_REVISION}" ]]; then BON_REV_ARGS+=(--judge-revision "${JUDGE_REVISION}"); fi
"${PY}" "${SCRIPT_DIR}/04_fresh_bon.py" \
  --gen-model "${GEN_MODEL}" \
  --rm-model "${RM_MODEL}" \
  --judge-model "${JUDGE_MODEL}" \
  --prompts-file "${OUT_ROOT}/summeval_prompts.txt" \
  --n-prompts "${N_PROMPTS}" \
  --n-cal "${N_CAL}" \
  --n-pool "${N_POOL}" \
  --N-list "${N_LIST}" \
  --K 200 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --judge-tensor-parallel 1 \
  "${BON_REV_ARGS[@]}" \
  --outdir "${OUT_ROOT}/fresh_bon"

echo "DONE ${OUT_ROOT}"
