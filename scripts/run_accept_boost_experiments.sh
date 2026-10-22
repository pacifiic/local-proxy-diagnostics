#!/bin/bash
# Accept-boost experiments for local proxy-gain diagnostics.
#
# These experiments cover:
#   (1) local validity across KL radius d;
#   (2) practical/selective use in GPT-4o proxy settings.
#
# They assume the standard data preparation scripts have already been run.
# The GPT-4o proxy score file is included at data/model_proxy_4o_aggregated.csv.

set -e
# Ensure we run from the repo root regardless of where the script is called from
cd "$(cd "$(dirname "$0")" && pwd)"

R="${OUT_ROOT:-all_results_reproduced}"
DATA="${DATA_ROOT:-data}"
OUT="${ACCEPT_BOOST_OUT:-$R/12_accept_boost}"
PROXY_CSV="${PROXY_CSV:-$DATA/model_proxy_4o_aggregated.csv}"

if [ -e "$OUT" ] && { [ ! -d "$OUT" ] || [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; } && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
  cat >&2 <<EOF
Refusing to overwrite existing output directory: $OUT

Use a fresh path, for example:
  ACCEPT_BOOST_OUT=outputs/accept_boost_\$(date +%Y%m%d_%H%M%S) bash run_accept_boost_experiments.sh

Or set ALLOW_OVERWRITE=1 if you intentionally want to update $OUT.
EOF
  exit 1
fi

mkdir -p "$OUT"

echo "========================================"
echo "12: Accept-boost experiments"
echo "Output: $OUT"
echo "Data root: $DATA"
echo "========================================"

# 1. d-sweep local-validity experiment.
echo; echo "── 12.1 d-sweep: human-heldout SummEval ──"
python3 experiments/12_accept_boost/01_d_sweep_prediction.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --d-grid 0.005,0.01,0.02,0.05,0.10,0.15,0.20 \
  --empfit-max-d 0.10 \
  --n-repeats 50 \
  --n-boot 1000 \
  --outdir "$OUT/d_sweep_human"

echo; echo "── 12.1 d-sweep: GPT-4o proxy ──"
python3 experiments/12_accept_boost/01_d_sweep_prediction.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --d-grid 0.005,0.01,0.02,0.05,0.10,0.15,0.20 \
  --empfit-max-d 0.10 \
  --n-repeats 50 \
  --n-boot 1000 \
  --outdir "$OUT/d_sweep_gpt4o"

# 2. Exact-curve residual/breakdown experiment.
echo; echo "── 12.2 curve residuals: human-heldout SummEval ──"
python3 experiments/12_accept_boost/02_curve_residual_breakdown.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --d-ranges 0.05,0.10,0.20 \
  --outdir "$OUT/residual_human"

echo; echo "── 12.2 curve residuals: GPT-4o proxy ──"
python3 experiments/12_accept_boost/02_curve_residual_breakdown.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --d-ranges 0.05,0.10,0.20 \
  --outdir "$OUT/residual_gpt4o"

# 3. Calibration-size learning curve.
echo; echo "── 12.3 calibration-size curve: human-heldout SummEval ──"
python3 experiments/12_accept_boost/03_calibration_size_curve.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --calibration-fracs 0.10,0.20,0.30,0.50,0.70 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --n-boot 1000 \
  --outdir "$OUT/cal_size_human"

echo; echo "── 12.3 calibration-size curve: GPT-4o proxy ──"
python3 experiments/12_accept_boost/03_calibration_size_curve.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --calibration-fracs 0.10,0.20,0.30,0.50,0.70 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --n-boot 1000 \
  --outdir "$OUT/cal_size_gpt4o"

# 4/5. Selective prediction and optimize/abstain policy gate.
echo; echo "── 12.4/12.5 selective prediction gate: GPT-4o proxy ──"
python3 experiments/12_accept_boost/04_selective_prediction_gate.py \
  --paths-csv "$OUT/d_sweep_gpt4o/d_sweep_paths_long.csv" \
  --input-format long \
  --d-target 0.10 \
  --predictors theory,aonly,empfit,anu \
  --thresholds 0,0.0025,0.005,0.01,0.02,0.05,0.10 \
  --margin-mode predicted \
  --n-boot 1000 \
  --outdir "$OUT/selective_gpt4o"

echo; echo "========================================"
echo "Accept-boost experiments complete."
echo "========================================"
