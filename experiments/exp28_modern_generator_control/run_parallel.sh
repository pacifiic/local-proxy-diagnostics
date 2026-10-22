#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT to the directory containing both models}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs}"
TAG="${TAG:-20260724}"

MISTRAL_ID="mistralai/Ministral-3-8B-Instruct-2512-BF16"
MISTRAL_REV="f6fae9795746f63c9be8344932f01275f3c63734"
MISTRAL_PATH="${MODEL_ROOT}/Ministral-3-8B-Instruct-2512-BF16"
LLAMA_ID="meta-llama/Llama-3.1-8B-Instruct"
LLAMA_REV="0e9e39f249a16976918f6564b8830bc894c89659"
LLAMA_PATH="${MODEL_ROOT}/Llama-3.1-8B-Instruct"

for path in "${MISTRAL_PATH}" "${LLAMA_PATH}"; do
  if [[ ! -f "${path}/model.safetensors.index.json" ]]; then
    echo "missing model checkpoint: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${OUT_ROOT}"

pids=()
labels=()

launch() {
  local family="$1"
  local seed="$2"
  local gpu_gen="$3"
  local gpu_gold="$4"
  local model model_path revision run
  local pretokenize tokenizer_mode mistral_chat text_only

  case "${family}" in
    mistral)
      model="${MISTRAL_ID}"
      model_path="${MISTRAL_PATH}"
      revision="${MISTRAL_REV}"
      run="${OUT_ROOT}/exp28_ministral3_8b_seed${seed}_${TAG}"
      pretokenize=1
      tokenizer_mode=mistral
      mistral_chat=1
      text_only=1
      ;;
    llama)
      model="${LLAMA_ID}"
      model_path="${LLAMA_PATH}"
      revision="${LLAMA_REV}"
      run="${OUT_ROOT}/exp28_llama31_8b_seed${seed}_${TAG}"
      pretokenize=0
      tokenizer_mode=auto
      mistral_chat=0
      text_only=0
      ;;
    *)
      echo "unknown family: ${family}" >&2
      exit 2
      ;;
  esac

  if [[ -e "${run}" ]] || [[ -e "${run}.driver.log" ]]; then
    echo "refusing to overwrite ${run} or ${run}.driver.log" >&2
    exit 2
  fi
  echo "launching ${family} seed=${seed} GPUs=${gpu_gen},${gpu_gold}"
  (
    MODEL="${model}" \
    MODEL_LOAD_PATH="${model_path}" \
    MODEL_REVISION="${revision}" \
    BASE_SEED="${seed}" \
    GPU_GEN="${gpu_gen}" \
    GPU_GOLD="${gpu_gold}" \
    PRETOKENIZE_PROMPTS="${pretokenize}" \
    VLLM_TOKENIZER_MODE="${tokenizer_mode}" \
    MISTRAL_COMMON_CHAT="${mistral_chat}" \
    TEXT_ONLY="${text_only}" \
    RUN="${run}" \
    bash "${SCRIPT_DIR}/run_one.sh"
  ) >"${run}.driver.log" 2>&1 &
  pids+=("$!")
  labels+=("${family}/seed${seed}")
}

wait_wave() {
  local status=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "completed ${labels[$i]}"
    else
      echo "failed ${labels[$i]}" >&2
      status=1
    fi
  done
  pids=()
  labels=()
  return "${status}"
}

# Four simultaneous runs use the eight GPUs as four non-overlapping pairs.
launch mistral 0 0 1
launch mistral 1 2 3
launch llama 0 4 5
launch llama 1 6 7
wait_wave

# The third seed for each model runs after the first wave releases its GPUs.
launch mistral 2 0 1
launch llama 2 2 3
wait_wave

echo "all exp28 generator-control runs completed"
