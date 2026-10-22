#!/bin/bash
# Robustness experiments for the local proxy-gain paper.
#
# Outputs are isolated from the main reproduced results by default.  Override
# ROBUSTNESS_OUT to run a scratch reproducibility check without
# touching existing result directories.

set -e
cd "$(cd "$(dirname "$0")" && pwd)"

OUT="${ROBUSTNESS_OUT:-all_results_reproduced/13_robustness}"
DATA="${DATA_ROOT:-data}"
PROXY_CSV="${PROXY_CSV:-$DATA/model_proxy_4o_aggregated.csv}"

if [ -e "$OUT" ] && { [ ! -d "$OUT" ] || [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; } && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
  cat >&2 <<EOF
Refusing to overwrite existing output directory: $OUT

Use a fresh path, for example:
  ROBUSTNESS_OUT=outputs/robustness_\$(date +%Y%m%d_%H%M%S) bash run_robustness_experiments.sh

Or set ALLOW_OVERWRITE=1 if you intentionally want to update $OUT.
EOF
  exit 1
fi

mkdir -p "$OUT"

echo "========================================"
echo "13: Robustness experiments"
echo "Output: $OUT"
echo "Data root: $DATA"
echo "========================================"

echo; echo "── 13.1 BoN bridge: human-heldout SummEval ──"
python3 experiments/08_bon_bridge/01_bon_bridge.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --N-list 1,2,4,8,16,32,64 \
  --shared-fit-max-d 0.10 \
  --bridge-max-d 0.25 \
  --outdir "$OUT/bon_bridge_human"

echo; echo "── 13.1 BoN bridge: GPT-4o proxy ──"
python3 experiments/08_bon_bridge/01_bon_bridge.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id \
  --proxy-score-col proxy_score_mean \
  --N-list 1,2,4,8,16,32,64 \
  --shared-fit-max-d 0.10 \
  --bridge-max-d 0.25 \
  --outdir "$OUT/bon_bridge_gpt4o"

for LAM in 0 0.25 0.5 0.75 1; do
  STEM=$(printf '%s' "$LAM" | sed 's/\./p/g')

  echo; echo "── 13.2 OOD SummEval -> WMT, lambda=$LAM ──"
  python3 experiments/10_ood_transfer/01_ood_transfer_prediction.py \
    --source-human-long-csv "$DATA/summeval_long.csv" \
    --source-name SummEval \
    --source-item-cols article_id,model_id \
    --source-group-cols dimension,rater_type \
    --source-rater-col rater_idx \
    --target-human-long-csv "$DATA/wmt_mqm_long.csv" \
    --target-name WMT_MQM \
    --target-item-cols item_id \
    --target-group-cols dataset_id \
    --target-rater-col rater \
    --calibration-frac 0.30 \
    --n-repeats 50 \
    --source-prior-repeats 50 \
    --shrinkage-lambda "$LAM" \
    --fit-max-d 0.10 \
    --d-targets 0.05,0.10 \
    --standardize-pairs z_cal \
    --outdir "$OUT/ood_summeval_to_wmt_lambda_$STEM"

  echo; echo "── 13.2 OOD WMT -> SummEval, lambda=$LAM ──"
  python3 experiments/10_ood_transfer/01_ood_transfer_prediction.py \
    --source-human-long-csv "$DATA/wmt_mqm_long.csv" \
    --source-name WMT_MQM \
    --source-item-cols item_id \
    --source-group-cols dataset_id \
    --source-rater-col rater \
    --target-human-long-csv "$DATA/summeval_long.csv" \
    --target-name SummEval \
    --target-item-cols article_id,model_id \
    --target-group-cols dimension,rater_type \
    --target-rater-col rater_idx \
    --calibration-frac 0.30 \
    --n-repeats 50 \
    --source-prior-repeats 50 \
    --shrinkage-lambda "$LAM" \
    --fit-max-d 0.10 \
    --d-targets 0.05,0.10 \
    --standardize-pairs z_cal \
    --outdir "$OUT/ood_wmt_to_summeval_lambda_$STEM"

  echo; echo "── 13.2 OOD SummEval -> OpenMEVA, lambda=$LAM ──"
  python3 experiments/10_ood_transfer/01_ood_transfer_prediction.py \
    --source-human-long-csv "$DATA/summeval_long.csv" \
    --source-name SummEval \
    --source-item-cols article_id,model_id \
    --source-group-cols dimension,rater_type \
    --source-rater-col rater_idx \
    --target-human-long-csv "$DATA/openmeva_long.csv" \
    --target-name OpenMEVA \
    --target-item-cols story_id,model_id \
    --target-group-cols dataset \
    --target-rater-col rater_idx \
    --calibration-frac 0.30 \
    --n-repeats 50 \
    --source-prior-repeats 50 \
    --shrinkage-lambda "$LAM" \
    --fit-max-d 0.10 \
    --d-targets 0.05,0.10 \
    --standardize-pairs z_cal \
    --outdir "$OUT/ood_summeval_to_openmeva_lambda_$STEM"
done

echo; echo "── 13.3 Calibration-size robustness: WMT MQM ──"
python3 experiments/12_accept_boost/03_calibration_size_curve.py \
  --human-long-csv "$DATA/wmt_mqm_long.csv" \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater \
  --calibration-fracs 0.10,0.20,0.30,0.50,0.70 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --n-boot 1000 \
  --outdir "$OUT/cal_size_wmt_mqm"

echo; echo "── 13.3 Calibration-size robustness: OpenMEVA ──"
python3 experiments/12_accept_boost/03_calibration_size_curve.py \
  --human-long-csv "$DATA/openmeva_long.csv" \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --rater-col rater_idx \
  --calibration-fracs 0.10,0.20,0.30,0.50,0.70 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --n-boot 1000 \
  --outdir "$OUT/cal_size_openmeva"

echo; echo "========================================"
echo "Robustness experiments complete."
echo "========================================"
