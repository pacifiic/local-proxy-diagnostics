#!/usr/bin/env bash
# exp06: complete the R5 defense — feed the open reward-model proxy (Skywork 8B)
# and the open-judge proxy (Qwen2.5-32B, 5-judge mean) produced by exp01 into the
# paper's prospective-prediction pipeline (same protocol as exp02 ROUGE tracks:
# 30% calibration, 50 repeats, seed 0, n_boot 500).
# Also computes the judge–RM Spearman requested by the response-bank checklist
# (fresh BoN item 4) from exp01 fresh_bon outputs.
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp06_rm_judge_proxy_$(date -u +%Y%m%d_%H%M%S)}"
PROSPECT="${REPO_ROOT}/experiments/prospective_prediction_path_macro/01_prospective_prediction.py"
DATA="${REPO_ROOT}/data"
INPUTS="${EXP06_INPUTS:-${DATA}/open_proxy_inputs}"
mkdir -p "${OUT_ROOT}"

echo "[exp06] Skywork RM proxy -> prospective prediction"
"${PY}" "${PROSPECT}" \
  --human-long-csv "${DATA}/summeval_long.csv" \
  --item-cols article_id,model_id --group-cols dimension \
  --proxy-csv "${INPUTS}/proxy_rm_skywork_v02.csv" \
  --proxy-item-col item_id --proxy-score-col proxy_score \
  --n-repeats 50 --n-boot 500 \
  --outdir "${OUT_ROOT}/proxy_rm_skywork"

echo "[exp06] open-judge (Qwen2.5-32B x5) proxy -> prospective prediction"
"${PY}" "${PROSPECT}" \
  --human-long-csv "${DATA}/summeval_long.csv" \
  --item-cols article_id,model_id --group-cols dimension \
  --proxy-csv "${INPUTS}/model_proxy_openjudge_aggregated.csv" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --n-repeats 50 --n-boot 500 \
  --outdir "${OUT_ROOT}/proxy_openjudge"

echo "[exp06] fresh-BoN judge-RM Spearman"
"${PY}" "${SCRIPT_DIR}/judge_rm_spearman.py" \
  --proxy-csv "${INPUTS}/fresh_bon_proxy_scores.csv" \
  --gold-csv "${INPUTS}/fresh_bon_gold_scores.csv" \
  --out-csv "${OUT_ROOT}/fresh_bon_judge_rm_spearman.csv"

echo "DONE ${OUT_ROOT}"
