#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
E9="${ROOT}/experiments/exp09_dpo_validation"
PY="${PY:-python}"
PROMPTS_SRC="${PROMPTS_SRC:-${ROOT}/outputs/exp09_dpo_validation_20260710_034319}"
RUN="${RUN:?set RUN to a fresh output directory}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROXY_MODEL="${PROXY_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
GOLD_MODEL="${GOLD_MODEL:-OpenAssistant/reward-model-deberta-v3-large-v2}"
QWEN3_POOL="${QWEN3_POOL:-${ROOT}/outputs/exp09_dpo_v4_20260710_051319/pool_scored.jsonl}"
ZEPHYR7_POOL="${ZEPHYR7_POOL:-${ROOT}/outputs/exp11_cross_model_20260710_071053/pool_scored.jsonl}"
GPU_GEN="${GPU_GEN:-0}"
GPU_GOLD="${GPU_GOLD:-1}"

if [[ -e "${RUN}" ]] && [[ -n "$(find "${RUN}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite non-empty RUN=${RUN}" >&2
  exit 2
fi
mkdir -p "${RUN}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0

for file in prompts_calibration.jsonl prompts_eval.jsonl prompts_run_spec.json; do
  cp "${PROMPTS_SRC}/${file}" "${RUN}/${file}"
done

echo "[27-01] generate Qwen2.5-7B matched candidate pool on GPU ${GPU_GEN}"
CUDA_VISIBLE_DEVICES="${GPU_GEN}" "${PY}" "${E9}/02_generate_pool.py" \
  --prompts-dir "${RUN}" --model "${MODEL}" \
  --n-train 0 --n-cal 32 --n-eval 32 \
  --temperature 0.9 --top-p 0.95 --max-tokens 320 --seed 0 \
  --out-jsonl "${RUN}/pool.jsonl"

echo "[27-02] score proxy and silver-gold models in parallel"
CUDA_VISIBLE_DEVICES="${GPU_GEN}" "${PY}" "${E9}/03_score_rm.py" \
  --in-jsonl "${RUN}/pool.jsonl" --model "${PROXY_MODEL}" \
  --kind skywork --score-key proxy_score --batch-size 16 \
  --out-jsonl "${RUN}/pool_proxy.jsonl" >"${RUN}/proxy_score.log" 2>&1 &
PROXY_PID=$!
CUDA_VISIBLE_DEVICES="${GPU_GOLD}" "${PY}" "${E9}/03_score_rm.py" \
  --in-jsonl "${RUN}/pool.jsonl" --model "${GOLD_MODEL}" \
  --kind deberta --score-key gold_score --batch-size 16 \
  --out-jsonl "${RUN}/pool_gold.jsonl" >"${RUN}/gold_score.log" 2>&1 &
GOLD_PID=$!
status=0
wait "${PROXY_PID}" || status=1
wait "${GOLD_PID}" || status=1
if [[ "${status}" -ne 0 ]]; then
  tail -n 80 "${RUN}/proxy_score.log" >&2 || true
  tail -n 80 "${RUN}/gold_score.log" >&2 || true
  exit 1
fi

echo "[27-03] strict key/text merge and data-layer audit"
"${PY}" "${SCRIPT_DIR}/merge_and_audit_scores.py" \
  --pool "${RUN}/pool.jsonl" \
  --proxy "${RUN}/pool_proxy.jsonl" \
  --gold "${RUN}/pool_gold.jsonl" \
  --out "${RUN}/pool_scored.jsonl" \
  --audit-json "${RUN}/merge_audit.json"

echo "[27-04] fit locked calibration coefficients"
"${PY}" "${E9}/05_fit_calibration.py" \
  --scored-jsonl "${RUN}/pool_scored.jsonl" \
  --out-json "${RUN}/coefficients.json"

echo "[27-05] independent run audit"
"${PY}" "${SCRIPT_DIR}/audit_qwen7_run.py" \
  --run "${RUN}" \
  --qwen-3b "${QWEN3_POOL}" \
  --zephyr-7b "${ZEPHYR7_POOL}" \
  --out-json "${RUN}/run_audit.json"

printf 'RUN=%s\n' "${RUN}"
