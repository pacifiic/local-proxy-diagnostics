#!/usr/bin/env bash
set -euo pipefail

ROOT="${OUT_ROOT:-outputs/tail_outlier_robustness}"
SCRIPT="experiments/14_tail_outlier_robustness/01_tail_outlier_robustness_audit.py"

python "$SCRIPT" \
  --dataset-name SummEval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --rater-col rater_idx \
  --outdir "$ROOT/summeval"

python "$SCRIPT" \
  --dataset-name OpenMEVA-MAGS \
  --human-long-csv data/openmeva_long.csv \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --rater-col rater_idx \
  --min-items 100 \
  --outdir "$ROOT/openmeva_mags"

python "$SCRIPT" \
  --dataset-name WMT-MQM-clusters \
  --human-long-csv data/wmt_mqm_long.csv \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater \
  --outdir "$ROOT/wmt_fixed_141"

