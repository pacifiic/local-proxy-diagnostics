#!/bin/bash
# Full experiment pipeline reproduction
# - Writes to OUT_ROOT (default: all_results_reproduced)
# - Runs legacy scripts that hard-code results/ inside an isolated temporary results/
#   directory, then restores any pre-existing results/ directory.
# - Does NOT touch all_results/
# - Skips 01_generate_proxy_scores.py (API); uses existing model_proxy_4o_aggregated.csv

set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA_ROOT:-data}"
PROXY_CSV="${PROXY_CSV:-$DATA/model_proxy_4o_aggregated.csv}"
R="${OUT_ROOT:-all_results_reproduced}"
REWARDBENCH_SCORES_CSV="${REWARDBENCH_SCORES_CSV:-$DATA/rewardbench/rewardbench_scores.csv}"
MTBENCH_PARQUET="${MTBENCH_PARQUET:-$DATA/mt_bench_human_judgments/human.parquet}"
if [ ! -f "$MTBENCH_PARQUET" ] && [ -f "$DATA/mt_bench_human_judgments/human-00000-of-00001-25f4910818759289.parquet" ]; then
  MTBENCH_PARQUET="$DATA/mt_bench_human_judgments/human-00000-of-00001-25f4910818759289.parquet"
fi

if [ -e "$R" ] && { [ ! -d "$R" ] || [ -n "$(find "$R" -mindepth 1 -maxdepth 1 -print -quit)" ]; } && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
  cat >&2 <<EOF
Refusing to overwrite existing output directory: $R

Set OUT_ROOT to a fresh scratch directory, for example:
  OUT_ROOT=outputs/proxy_gain_repro_\$(date +%Y%m%d_%H%M%S) bash run_all_reproduced.sh

Or set ALLOW_OVERWRITE=1 if you intentionally want to update $R.
EOF
  exit 1
fi

TMP_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/proxy-gain-run-all.XXXXXX")"
ORIGINAL_RESULTS_BACKUP=""
GENERATED_RESULTS_ACTIVE=0

restore_results() {
  status=$?
  if [ "$GENERATED_RESULTS_ACTIVE" = "1" ]; then
    rm -rf results
    if [ -n "$ORIGINAL_RESULTS_BACKUP" ] && [ -e "$ORIGINAL_RESULTS_BACKUP" ]; then
      mv "$ORIGINAL_RESULTS_BACKUP" results
    fi
  fi
  rm -rf "$TMP_PARENT"
  exit "$status"
}
trap restore_results EXIT

if [ -e results ]; then
  ORIGINAL_RESULTS_BACKUP="$TMP_PARENT/original_results"
  mv results "$ORIGINAL_RESULTS_BACKUP"
fi
mkdir -p results
GENERATED_RESULTS_ACTIVE=1

echo "========================================"
echo "Experiment Pipeline Reproduction"
echo "Output: $R/"
echo "Data root: $DATA"
echo "Legacy scratch: ./results/ (restored/removed on exit)"
echo "RewardBench scores: $REWARDBENCH_SCORES_CSV"
echo "MT-Bench parquet: $MTBENCH_PARQUET"
echo "========================================"

# ── Folder 01: ICC decomposition → results/ ──────────────────────────────
echo; echo "── 01: ICC decomposition ──"
python3 experiments/01_icc_decomposition/01_summeval_icc_table.py
python3 experiments/01_icc_decomposition/02_bon_simulation.py
python3 experiments/01_icc_decomposition/03_pool_perturbation.py
python3 experiments/01_icc_decomposition/04_protocol_perturbation.py
python3 experiments/01_icc_decomposition/05_openmeva_replication.py

# ── Folder 02: Corrected coefficients → results/ ─────────────────────────
echo; echo "── 02: Corrected coefficients ──"
python3 experiments/02_corrected_coefficients/01_corrected_B.py
python3 experiments/02_corrected_coefficients/02_rewardbench_B.py \
  --data-root "$DATA"
python3 experiments/02_corrected_coefficients/03_bon_bootstrap.py
python3 experiments/02_corrected_coefficients/04_gao_comparison.py

# ── Copy 01/02 outputs into all_results_reproduced ───────────────────────
echo; echo "── Copying 01/02 results into $R ──"
mkdir -p "$R/01_icc_decomposition" "$R/02_corrected_coefficients" "$R/builder_cache/figures"
cp results/tables/table1_summeval_decomposition.csv      "$R/01_icc_decomposition/"
cp results/tables/table2_summeval_expert_crowd_lift.csv   "$R/01_icc_decomposition/"
cp results/tables/table3_summeval_protocol_perturbation.csv "$R/01_icc_decomposition/"
cp results/tables/table4_openmeva_replication.csv         "$R/01_icc_decomposition/"
cp results/tables/tableA_bon_gain_summary.csv             "$R/01_icc_decomposition/"
cp results/tables/tableA_bon_secondary.csv                "$R/01_icc_decomposition/"
cp results/intermediate/bootstrap_summeval_icc.json       "$R/01_icc_decomposition/"
cp results/intermediate/bootstrap_summeval_lift.json      "$R/01_icc_decomposition/"
cp results/intermediate/bootstrap_protocol.json           "$R/01_icc_decomposition/"
cp results/intermediate/bootstrap_openmeva.json           "$R/01_icc_decomposition/"
cp results/intermediate/bootstrap_bon_delta.json          "$R/01_icc_decomposition/"
cp results/intermediate/bootstrap_bon_delta_secondary.json "$R/01_icc_decomposition/"
cp results/tables/table_corrected_B.csv                   "$R/02_corrected_coefficients/"
cp results/tables/table07_rewardbench_B.csv               "$R/02_corrected_coefficients/"
cp results/tables/table08v2_bon_bootstrap.csv             "$R/02_corrected_coefficients/"
cp results/tables/table09_gao_comparison.csv              "$R/02_corrected_coefficients/"
cp results/tables/table14v2_bon_bootstrap.csv             "$R/02_corrected_coefficients/" 2>/dev/null || true
cp results/tables/table16_gao_comparison.csv              "$R/02_corrected_coefficients/" 2>/dev/null || true
cp results/tables/table_direct_tilt_summeval.csv          "$R/02_corrected_coefficients/" 2>/dev/null || true
cp results/tables/table_direct_tilt_summeval_summary.csv  "$R/02_corrected_coefficients/" 2>/dev/null || true
cp results/figures/fig*.png                               "$R/builder_cache/figures/" 2>/dev/null || true
cp results/figures/fig*.pdf                               "$R/builder_cache/figures/" 2>/dev/null || true
cp results/figures/fig07_rewardbench_B.pdf                "$R/02_corrected_coefficients/" 2>/dev/null || true
echo "  01/02 results copied to $R"

# ── Folder 03: Direct tilt validation ────────────────────────────────────
echo; echo "── 03: Direct tilt validation ──"
python3 experiments/03_direct_tilt_validation/01_summeval_direct_tilt.py
mkdir -p "$R/builder_cache/tables" "$R/builder_cache/figures"
cp results/tables/table_direct_tilt_summeval.csv          "$R/builder_cache/tables/"
cp results/tables/table_direct_tilt_summeval_summary.csv  "$R/builder_cache/tables/"
cp results/figures/fig_direct_tilt_summeval.png           "$R/builder_cache/figures/"

echo "  03/robustness: loo_corrected_B"
python3 experiments/03_direct_tilt_validation/robustness/01_loo_corrected_B_bootstrap.py \
  --data "$DATA/summeval_long.csv" \
  --outdir "$R/03_direct_tilt_validation/robustness"

echo "  03/robustness: ordinal_artifact"
python3 experiments/03_direct_tilt_validation/robustness/02_ordinal_artifact_control.py \
  --outdir "$R/03_direct_tilt_validation/robustness"

echo "  03/robustness: matched_rater"
python3 experiments/03_direct_tilt_validation/robustness/03_matched_rater_robustness.py \
  --data "$DATA/summeval_long.csv" \
  --outdir "$R/03_direct_tilt_validation/robustness"

echo "  03/robustness: tilt_estimator_sensitivity"
python3 experiments/03_direct_tilt_validation/robustness/04_tilt_estimator_sensitivity.py \
  --data "$DATA/summeval_long.csv" \
  --outdir "$R/03_direct_tilt_validation/robustness"

# ── Folder 04: Cross-dataset ──────────────────────────────────────────────
echo; echo "── 04: Cross-dataset ──"
python3 experiments/04_cross_dataset/01_openmeva_direct_tilt.py \
  --data "$DATA/openmeva_long.csv" \
  --outdir "$R/04_cross_dataset"

python3 experiments/04_cross_dataset/02_qgeval_human_direct_tilt.py \
  --annotator-files "$DATA/QGEval/annotation_result/annotator1.xlsx,$DATA/QGEval/annotation_result/annotator2.xlsx,$DATA/QGEval/annotation_result/annotator3.xlsx" \
  --outdir "$R/04_cross_dataset"

python3 experiments/04_cross_dataset/03_qgeval_metric_proxy_tilt.py \
  --scores-xlsx "$DATA/QGEval/scores.xlsx" \
  --metric-xlsx "$DATA/QGEval/metric_result.xlsx" \
  --outdir "$R/04_cross_dataset"

python3 experiments/04_cross_dataset/04_rewardbench_margin_geometry.py \
  --scores-csv "$REWARDBENCH_SCORES_CSV" \
  --missingness-mode pairwise_available \
  --outdir "$R/04_cross_dataset"

# ── Folder 05: WMT & MT-Bench ────────────────────────────────────────────
echo; echo "── 05: WMT & MT-Bench ──"
python3 experiments/05_wmt_mtbench/01_wmt_mqm_direct_tilt.py \
  --root "$DATA/wmt-mqm-human-evaluation" \
  --outdir "$R/05_wmt_mtbench/exp19_wmt_mqm"

python3 experiments/05_wmt_mtbench/02_wmt_psqm_direct_tilt.py \
  --root "$DATA/wmt-mqm-human-evaluation" \
  --outdir "$R/05_wmt_mtbench/exp20_wmt_psqm"

python3 experiments/05_wmt_mtbench/03_wmt_reliability_summary.py \
  --root "$DATA/wmt-mqm-human-evaluation" \
  --outdir "$R/05_wmt_mtbench/exp21_wmt_inventory"

python3 experiments/05_wmt_mtbench/04_mtbench_split_half_bt.py \
  --parquet "$MTBENCH_PARQUET" \
  --outdir "$R/05_wmt_mtbench/exp22_mtbench_bt"

# ── Folder 06: Supplemental audits ─────────────────────────────────────────
echo; echo "── 06: Supplemental audits ──"

echo "  exp23: monotone (SummEval)"
python3 experiments/06_supplemental_audits/01_monotone_condition_audit.py \
  --data "$DATA/summeval_long.csv" \
  --group-cols dimension,rater_type \
  --score-transform raw \
  --gold-estimator loo_mean \
  --min-items 200 --min-raters 3 --n-bootstrap 200 \
  --outdir "$R/06_supplemental_audits/exp23_summeval"

echo "  exp23: monotone (WMT MQM)"
python3 experiments/06_supplemental_audits/01_monotone_condition_audit.py \
  --data "$DATA/wmt_mqm_long.csv" \
  --group-cols dataset_id \
  --item-col item_id --rater-col rater --score-col score \
  --score-transform raw \
  --gold-estimator loo_mean \
  --min-items 200 --min-raters 3 --n-bootstrap 200 \
  --outdir "$R/06_supplemental_audits/exp23_wmt_mqm"

echo "  exp24: rho scaling (SummEval)"
python3 experiments/06_supplemental_audits/02_rho_scaling_audit.py \
  --data "$DATA/summeval_long.csv" \
  --group-cols dimension,rater_type \
  --score-transform raw \
  --gold-estimator loo_mean \
  --min-items 200 --min-raters 3 --fit-max-d 0.10 \
  --outdir "$R/06_supplemental_audits/exp24_summeval"

echo "  exp24: rho scaling (WMT MQM)"
python3 experiments/06_supplemental_audits/02_rho_scaling_audit.py \
  --data "$DATA/wmt_mqm_long.csv" \
  --group-cols dataset_id \
  --item-col item_id --rater-col rater --score-col score \
  --score-transform raw \
  --gold-estimator loo_mean \
  --min-items 200 --min-raters 3 --fit-max-d 0.10 \
  --outdir "$R/06_supplemental_audits/exp24_wmt_mqm"

echo "  exp25: intervention (SummEval)"
python3 experiments/06_supplemental_audits/03_measurement_intervention.py \
  --data "$DATA/summeval_long.csv" \
  --group-cols dimension,rater_type \
  --score-transform raw \
  --k-list 1,2,3 --gold-size 2 \
  --n-reps 500 --min-items 200 --fit-max-d 0.10 \
  --outdir "$R/06_supplemental_audits/exp25_summeval"

echo "  exp25: intervention (WMT MQM)"
python3 experiments/06_supplemental_audits/03_measurement_intervention.py \
  --data "$DATA/wmt_mqm_long.csv" \
  --group-cols dataset_id \
  --item-col item_id --rater-col rater --score-col score \
  --score-transform raw \
  --k-list 1,2 --gold-size 1 \
  --n-reps 500 --min-items 200 --fit-max-d 0.10 \
  --outdir "$R/06_supplemental_audits/exp25_wmt_mqm"

echo "  exp26: phase map"
python3 experiments/06_supplemental_audits/04_operational_phase_map.py \
  --data "$DATA/phase_inputs.csv" \
  --out "$R/06_supplemental_audits/exp26_phase_assignments.csv" \
  --tau-high 0.90 --tau-mid 0.70 --tau-gap 0.05

# ── Folder 07: Prospective prediction ────────────────────────────────────
echo; echo "── 07: Prospective prediction ──"

echo "  exp27: human_baseline (SummEval)"
python3 experiments/07_prospective_prediction/01_prospective_prediction.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp27_human_baseline"

echo "  exp27: model_proxy_4o (SummEval)"
python3 experiments/07_prospective_prediction/01_prospective_prediction.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp27_model_proxy_4o"

echo "  exp27: openmeva"
python3 experiments/07_prospective_prediction/01_prospective_prediction.py \
  --human-long-csv "$DATA/openmeva_long.csv" \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp27_openmeva"

echo "  exp27: wmt_mqm"
python3 experiments/07_prospective_prediction/01_prospective_prediction.py \
  --human-long-csv "$DATA/wmt_mqm_long.csv" \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater --score-col score \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp27_wmt_mqm"

echo "  exp28: stability human (SummEval)"
python3 experiments/07_prospective_prediction/02_stability_tail_audit.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp28_human"

echo "  exp28: stability model_proxy_4o"
python3 experiments/07_prospective_prediction/02_stability_tail_audit.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp28_model_proxy_4o"

echo "  exp28: stability openmeva"
python3 experiments/07_prospective_prediction/02_stability_tail_audit.py \
  --human-long-csv "$DATA/openmeva_long.csv" \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp28_openmeva"

echo "  exp28: stability wmt_mqm"
python3 experiments/07_prospective_prediction/02_stability_tail_audit.py \
  --human-long-csv "$DATA/wmt_mqm_long.csv" \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater --score-col score \
  --fit-max-d 0.1 \
  --outdir "$R/07_prospective_prediction/exp28_wmt_mqm"

# ── Folder 08: BoN bridge ─────────────────────────────────────────────────
echo; echo "── 08: BoN bridge ──"

echo "  exp29: v2_human"
python3 experiments/08_bon_bridge/01_bon_bridge.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --outdir "$R/08_bon_bridge/v2_human"

echo "  exp29: v2_model_proxy_4o"
python3 experiments/08_bon_bridge/01_bon_bridge.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv "$PROXY_CSV" \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --outdir "$R/08_bon_bridge/v2_model_proxy_4o"

# ── Folder 09: GPT proxy diagnostics (uses existing API data) ─────────────
echo; echo "── 09: GPT proxy diagnostics ──"
python3 experiments/09_gpt_proxy_analysis/03_gpt_proxy_diagnostics.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --proxy-csv "$PROXY_CSV" \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-item-col item_id --proxy-score-col proxy_score_mean \
  --human-compare \
  --exp31-model-paths "$R/07_prospective_prediction/exp27_model_proxy_4o/exp27_prospective_paths.csv" \
  --exp31-human-paths "$R/07_prospective_prediction/exp27_human_baseline/exp27_prospective_paths.csv" \
  --fit-max-d 0.1 \
  --outdir "$R/09_gpt_proxy_analysis/exp30_gpt_proxy_diagnostics"

# ── Folder 10: OOD transfer ───────────────────────────────────────────────
echo; echo "── 10: OOD transfer ──"
python3 experiments/10_ood_transfer/01_ood_transfer_prediction.py \
  --source-human-long-csv "$DATA/summeval_long.csv" \
  --source-name SummEval \
  --source-item-cols article_id,model_id \
  --source-group-cols dimension,rater_type \
  --target-human-long-csv "$DATA/wmt_mqm_long.csv" \
  --target-name WMT_MQM \
  --target-item-cols item_id \
  --target-group-cols dataset_id \
  --target-rater-col rater --target-score-col score \
  --standardize-pairs z_both \
  --fit-max-d 0.1 \
  --outdir "$R/10_ood_transfer"

# ── Folder 11: Proxy perturbation ─────────────────────────────────────────
echo; echo "── 11: Proxy perturbation ──"

echo "  exp32: summeval"
python3 experiments/11_proxy_perturbation/01_proxy_perturbation_baselines.py \
  --human-long-csv "$DATA/summeval_long.csv" \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --fit-max-d 0.1 \
  --outdir "$R/11_proxy_perturbation/exp32_summeval_proxy_baselines"

echo "  exp32: wmt_mqm"
python3 experiments/11_proxy_perturbation/01_proxy_perturbation_baselines.py \
  --human-long-csv "$DATA/wmt_mqm_long.csv" \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater --score-col score \
  --fit-max-d 0.1 \
  --outdir "$R/11_proxy_perturbation/exp32_wmt_mqm"

echo "  exp32: openmeva"
python3 experiments/11_proxy_perturbation/01_proxy_perturbation_baselines.py \
  --human-long-csv "$DATA/openmeva_long.csv" \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --fit-max-d 0.1 \
  --outdir "$R/11_proxy_perturbation/exp32_openmeva"

echo; echo "========================================"
echo "All experiments complete!"
echo "========================================"
