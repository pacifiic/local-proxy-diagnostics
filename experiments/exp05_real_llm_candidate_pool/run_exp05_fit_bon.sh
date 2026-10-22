#!/usr/bin/env bash
# exp05: real-LLM candidate-pool reranking (experiment 19) — reproducibility re-run.
# Re-runs the analysis stages (04 fit gain curves, 05 BoN scope, 06 artifacts) on the
# FIXED synthetic scored-candidate input that produced the reference run
# code_review_synthetic_after_patch_20260611_145920 (random_seed=0, use_existing_split).
# GPU candidate generation for this experiment was superseded by exp04 (conditional
# tilt); the analysis stages here are the deterministic part with a reference to
# compare against.
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp05_candidate_pool_$(date -u +%Y%m%d_%H%M%S)}"
CODE="${SCRIPT_DIR}/code"
INPUT="${SCRIPT_DIR}/candidates_scored_input.csv"
mkdir -p "${OUT_ROOT}"
# The vendored exp19 scripts locate lib/ two levels up from their own path in the
# original repo layout; in the consolidated layout we provide it via PYTHONPATH.
export PYTHONPATH="${REPO_ROOT}/lib${PYTHONPATH:+:${PYTHONPATH}}"

echo "[exp05] fit/predict gain curves"
"${PY}" "${CODE}/04_fit_predict_gain_curves.py" \
  --candidates-csv "${INPUT}" \
  --prompt-col prompt_id --split-col split --use-existing-split \
  --proxy-col proxy_score --gold-col gold_score \
  --calibration-frac 0.3 --n-repeats 1 --random-seed 0 \
  --grid-d-values 0.01,0.05,0.1 --endpoint-d 0.1 --fit-max-d 0.05 --n-grid 100 \
  --mlp-hidden-layers 16,16 --mlp-alpha 0.001 --mlp-max-iter 300 --mlp-seed 0 \
  --outdir "${OUT_ROOT}/gain_curves"

echo "[exp05] BoN scope check"
"${PY}" "${CODE}/05_bon_scope_check.py" \
  --candidates-csv "${INPUT}" \
  --prompt-col prompt_id --split-col split --use-existing-split \
  --proxy-col proxy_score --gold-col gold_score \
  --calibration-frac 0.3 --n-repeats 1 --random-seed 0 \
  --n-list 2,4,8 \
  --outdir "${OUT_ROOT}/bon_scope"

echo "[exp05] analysis artifacts"
"${PY}" "${CODE}/06_make_artifacts.py" \
  --gain-root "${OUT_ROOT}/gain_curves" \
  --bon-root "${OUT_ROOT}/bon_scope" \
  --outdir "${OUT_ROOT}/artifacts" --variant-label "(SYNTHETIC pipeline check)"

echo "DONE ${OUT_ROOT}"
