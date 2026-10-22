#!/usr/bin/env bash
# exp02: CPU metric and filtering audits
#   - ROUGE-1/2/L unaligned-proxy prospective prediction (SummEval)
#   - BoN bridge window audit (SummEval human)
#   - WMT-MQM filter sensitivity (min-items 400 / 100, prepare-side mc100)
# Arguments reproduce the run_spec.json files recorded in
# reference_results/exp02_metric_and_filter/ (random_seed=0 everywhere).
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp02_metric_and_filter_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

PROSPECT="${REPO_ROOT}/experiments/prospective_prediction_path_macro/01_prospective_prediction.py"
BONBRIDGE="${REPO_ROOT}/experiments/08_bon_bridge/01_bon_bridge.py"
DATA="${REPO_ROOT}/data"
WMT_MQM_LONG_CSV="${WMT_MQM_LONG_CSV:-${DATA}/wmt_mqm_long.csv}"

echo "[exp02] ROUGE proxies"
for R in rouge1 rouge2 rougeL; do
  "${PY}" "${PROSPECT}" \
    --human-long-csv "${DATA}/summeval_long.csv" \
    --item-cols article_id,model_id --group-cols dimension \
    --proxy-csv "${DATA}/proxy_${R}.csv" \
    --proxy-item-col item_id --proxy-score-col proxy_score \
    --n-repeats 50 --n-boot 500 \
    --outdir "${OUT_ROOT}/metric_${R}"
done

echo "[exp02] BoN bridge (human)"
"${PY}" "${BONBRIDGE}" \
  --human-long-csv "${DATA}/summeval_long.csv" \
  --item-cols article_id,model_id --group-cols dimension,rater_type \
  --outdir "${OUT_ROOT}/bon_human"

echo "[exp02] WMT filter sensitivity"
"${PY}" "${PROSPECT}" \
  --human-long-csv "${WMT_MQM_LONG_CSV}" \
  --item-cols item_id --group-cols dataset_id --rater-col rater \
  --min-items 400 --n-repeats 8 --n-boot 300 \
  --outdir "${OUT_ROOT}/filt_wmt_mi400"
"${PY}" "${PROSPECT}" \
  --human-long-csv "${WMT_MQM_LONG_CSV}" \
  --item-cols item_id --group-cols dataset_id --rater-col rater \
  --min-items 100 --n-repeats 8 --n-boot 300 \
  --outdir "${OUT_ROOT}/filt_wmt_mi100"
"${PY}" "${PROSPECT}" \
  --human-long-csv "${DATA}/wmt_mqm_long_mc100.csv" \
  --item-cols item_id --group-cols dataset_id --rater-col rater \
  --min-items 100 --n-repeats 8 --n-boot 300 \
  --outdir "${OUT_ROOT}/filt_wmt_mc100"

echo "DONE ${OUT_ROOT}"
