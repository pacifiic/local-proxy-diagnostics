#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
E9="${ROOT}/experiments/exp09_dpo_validation"
PY="${PY:-python}"
PROMPTS_SRC="${PROMPTS_SRC:-${ROOT}/outputs/exp09_dpo_validation_20260710_034319}"
RUN="${RUN:?set RUN to a fresh output directory}"
MODEL="${MODEL:?set MODEL to the generator model ID}"
MODEL_LOAD_PATH="${MODEL_LOAD_PATH:-${MODEL}}"
MODEL_REVISION="${MODEL_REVISION:-}"
PROXY_MODEL="${PROXY_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
GOLD_MODEL="${GOLD_MODEL:-OpenAssistant/reward-model-deberta-v3-large-v2}"
PROXY_REVISION="${PROXY_REVISION:-}"
GOLD_REVISION="${GOLD_REVISION:-}"
GPU_GEN="${GPU_GEN:-0}"
GPU_GOLD="${GPU_GOLD:-1}"
BASE_SEED="${BASE_SEED:-0}"
PRETOKENIZE_PROMPTS="${PRETOKENIZE_PROMPTS:-0}"
FIX_MISTRAL_REGEX="${FIX_MISTRAL_REGEX:-0}"
SKIP_VLLM_TOKENIZER="${SKIP_VLLM_TOKENIZER:-0}"
VLLM_TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-auto}"
MISTRAL_COMMON_CHAT="${MISTRAL_COMMON_CHAT:-0}"
TEXT_ONLY="${TEXT_ONLY:-0}"

if [[ -z "${PROXY_REVISION}" ]]; then
  if [[ "${PROXY_MODEL}" == "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2" ]]; then
    PROXY_REVISION="d4117fbfd81b72f41b96341238baa1e3e90a4ce1"
  else
    echo "PROXY_REVISION is required for custom PROXY_MODEL=${PROXY_MODEL}" >&2
    exit 2
  fi
fi
if [[ -z "${GOLD_REVISION}" ]]; then
  if [[ "${GOLD_MODEL}" == "OpenAssistant/reward-model-deberta-v3-large-v2" ]]; then
    GOLD_REVISION="c355404efa9ad2ad069f3a197cae0523c14244fc"
  else
    echo "GOLD_REVISION is required for custom GOLD_MODEL=${GOLD_MODEL}" >&2
    exit 2
  fi
fi
if [[ "${#PROXY_REVISION}" -ne 40 || "${#GOLD_REVISION}" -ne 40 ]]; then
  echo "PROXY_REVISION and GOLD_REVISION must be 40-character commits" >&2
  exit 2
fi

if [[ -e "${RUN}" ]] && [[ -n "$(find "${RUN}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite non-empty RUN=${RUN}" >&2
  exit 2
fi
mkdir -p "${RUN}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
for file in prompts_calibration.jsonl prompts_eval.jsonl prompts_run_spec.json; do
  cp "${PROMPTS_SRC}/${file}" "${RUN}/${file}"
done

echo "[27-01] deterministic candidate generation: ${MODEL}"
model_args=(--model "${MODEL_LOAD_PATH}")
if [[ "${MODEL_LOAD_PATH}" != "${MODEL}" ]]; then
  if [[ -z "${MODEL_REVISION}" ]]; then
    echo "MODEL_REVISION is required with MODEL_LOAD_PATH" >&2
    exit 2
  fi
  model_args+=(--model-id "${MODEL}" --model-revision "${MODEL_REVISION}")
fi
if [[ "${PRETOKENIZE_PROMPTS}" == "1" ]]; then
  model_args+=(--pretokenize-prompts)
elif [[ "${PRETOKENIZE_PROMPTS}" != "0" ]]; then
  echo "PRETOKENIZE_PROMPTS must be 0 or 1" >&2
  exit 2
fi
if [[ "${FIX_MISTRAL_REGEX}" == "1" ]]; then
  model_args+=(--fix-mistral-regex)
elif [[ "${FIX_MISTRAL_REGEX}" != "0" ]]; then
  echo "FIX_MISTRAL_REGEX must be 0 or 1" >&2
  exit 2
fi
if [[ "${SKIP_VLLM_TOKENIZER}" == "1" ]]; then
  model_args+=(--skip-vllm-tokenizer)
elif [[ "${SKIP_VLLM_TOKENIZER}" != "0" ]]; then
  echo "SKIP_VLLM_TOKENIZER must be 0 or 1" >&2
  exit 2
fi
model_args+=(--vllm-tokenizer-mode "${VLLM_TOKENIZER_MODE}")
if [[ "${MISTRAL_COMMON_CHAT}" == "1" ]]; then
  model_args+=(--mistral-common-chat)
elif [[ "${MISTRAL_COMMON_CHAT}" != "0" ]]; then
  echo "MISTRAL_COMMON_CHAT must be 0 or 1" >&2
  exit 2
fi
if [[ "${TEXT_ONLY}" == "1" ]]; then
  model_args+=(--text-only)
elif [[ "${TEXT_ONLY}" != "0" ]]; then
  echo "TEXT_ONLY must be 0 or 1" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="${GPU_GEN}" "${PY}" \
  "${SCRIPT_DIR}/generate_pool_deterministic.py" \
  --prompts-dir "${RUN}" "${model_args[@]}" \
  --n-cal 32 --n-eval 32 --temperature 0.9 --top-p 0.95 \
  --max-tokens 320 --base-seed "${BASE_SEED}" \
  --out-jsonl "${RUN}/pool.jsonl"

echo "[27-02] independent proxy and silver-gold scoring"
CUDA_VISIBLE_DEVICES="${GPU_GEN}" "${PY}" "${E9}/03_score_rm.py" \
  --in-jsonl "${RUN}/pool.jsonl" --model "${PROXY_MODEL}" \
  --revision "${PROXY_REVISION}" --require-user-site-disabled \
  --kind skywork --score-key proxy_score --batch-size 16 \
  --out-jsonl "${RUN}/pool_proxy.jsonl" >"${RUN}/proxy_score.log" 2>&1 &
proxy_pid=$!
CUDA_VISIBLE_DEVICES="${GPU_GOLD}" "${PY}" "${E9}/03_score_rm.py" \
  --in-jsonl "${RUN}/pool.jsonl" --model "${GOLD_MODEL}" \
  --revision "${GOLD_REVISION}" --require-user-site-disabled \
  --kind deberta --score-key gold_score --batch-size 16 \
  --out-jsonl "${RUN}/pool_gold.jsonl" >"${RUN}/gold_score.log" 2>&1 &
gold_pid=$!
status=0
wait "${proxy_pid}" || status=1
wait "${gold_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  tail -n 80 "${RUN}/proxy_score.log" >&2 || true
  tail -n 80 "${RUN}/gold_score.log" >&2 || true
  exit 1
fi

"${PY}" "${SCRIPT_DIR}/merge_and_audit_scores.py" \
  --pool "${RUN}/pool.jsonl" \
  --proxy "${RUN}/pool_proxy.jsonl" \
  --gold "${RUN}/pool_gold.jsonl" \
  --out "${RUN}/pool_scored.jsonl" \
  --audit-json "${RUN}/merge_audit.json"
"${PY}" "${E9}/05_fit_calibration.py" \
  --scored-jsonl "${RUN}/pool_scored.jsonl" \
  --out-json "${RUN}/coefficients.json"
"${PY}" "${SCRIPT_DIR}/write_model_revisions.py" --run "${RUN}"
audit_args=(
  --run "${RUN}"
  --expected-generator "${MODEL}"
  --expected-base-seed "${BASE_SEED}"
  --expected-vllm-tokenizer-mode "${VLLM_TOKENIZER_MODE}"
  --out-json "${RUN}/run_audit.json"
)
if [[ "${PRETOKENIZE_PROMPTS}" == "1" ]]; then
  audit_args+=(--expected-pretokenize-prompts)
fi
if [[ "${FIX_MISTRAL_REGEX}" == "1" ]]; then
  audit_args+=(--expected-fix-mistral-regex)
fi
if [[ "${SKIP_VLLM_TOKENIZER}" == "1" ]]; then
  audit_args+=(--expected-skip-vllm-tokenizer)
fi
if [[ "${MISTRAL_COMMON_CHAT}" == "1" ]]; then
  audit_args+=(--expected-mistral-common-chat)
fi
if [[ "${TEXT_ONLY}" == "1" ]]; then
  audit_args+=(--expected-text-only)
fi
"${PY}" "${SCRIPT_DIR}/audit_model_run.py" "${audit_args[@]}"
printf 'RUN=%s\n' "${RUN}"
