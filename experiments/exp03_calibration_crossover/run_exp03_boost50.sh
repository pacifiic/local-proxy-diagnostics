#!/usr/bin/env bash
# exp03 boost: same crossover protocol, but 50 resamples per cal_frac on every
# track (response-bank checklist item 3 — uncertainty estimation).
# Reference runs used 12 (human), 30 (gpt, openmeva), 8 (wmt) reps.
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp03_boost50_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"
XOVER="${SCRIPT_DIR}/01_calibration_crossover.py"

for TRACK in human gpt openmeva wmt; do
  echo "[exp03-boost] ${TRACK} track, 50 reps/frac"
  "${PY}" "${XOVER}" --track "${TRACK}" --extended --n-repeats 50 \
    --out-name "merged_${TRACK}3_boost50.csv" \
    --base-out-name "merged_${TRACK}_boost50.csv" \
    --splits-name "splits_${TRACK}_boost50.csv" \
    --outdir "${OUT_ROOT}"
done

echo "DONE ${OUT_ROOT}"
