#!/usr/bin/env bash
# exp03: calibration-size crossover (R2 defense) — full reproduction.
# Reproduces reference_results/exp03_calibration_crossover/:
#   merged_human.csv  merged_human3.csv  (one run: human3 is the same splits + extended predictors)
#   merged_gpt.csv    merged_gpt3.csv    (one run, 30 reps/frac)
#   merged_openmeva.csv merged_wmt.csv   (extended schema, 30 / 8 reps per frac)
#   x50_merged_splits.csv                (human frac 0.3, 25 reps x seeds 0 and 1, merged)
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp03_calibration_crossover_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"
XOVER="${SCRIPT_DIR}/01_calibration_crossover.py"

echo "[exp03] SummEval human held-out track (12 reps/frac) -> merged_human.csv + merged_human3.csv"
"${PY}" "${XOVER}" --track human --extended \
  --out-name merged_human3.csv --base-out-name merged_human.csv \
  --splits-name splits_human.csv --outdir "${OUT_ROOT}"

echo "[exp03] GPT-4o external proxy track (30 reps/frac) -> merged_gpt.csv + merged_gpt3.csv"
"${PY}" "${XOVER}" --track gpt --extended \
  --out-name merged_gpt3.csv --base-out-name merged_gpt.csv \
  --splits-name splits_gpt.csv --outdir "${OUT_ROOT}"

echo "[exp03] OpenMEVA track (30 reps/frac, extended schema) -> merged_openmeva.csv"
"${PY}" "${XOVER}" --track openmeva --extended \
  --out-name merged_openmeva.csv --splits-name splits_openmeva.csv --outdir "${OUT_ROOT}"

echo "[exp03] WMT-MQM track (8 reps/frac, extended schema) -> merged_wmt.csv"
"${PY}" "${XOVER}" --track wmt --extended \
  --out-name merged_wmt.csv --splits-name splits_wmt.csv --outdir "${OUT_ROOT}"

echo "[exp03] x50 per-split file (human, cal_frac 0.3, 25 reps, seeds 0 and 1, merged)"
"${PY}" "${XOVER}" --track human --extended --cal-fracs 0.3 --n-repeats 25 --seed 0 \
  --out-name x50_seed0_summary.csv --splits-name x50_seed0_splits.csv --outdir "${OUT_ROOT}/x50_tmp"
"${PY}" "${XOVER}" --track human --extended --cal-fracs 0.3 --n-repeats 25 --seed 1 \
  --out-name x50_seed1_summary.csv --splits-name x50_seed1_splits.csv --outdir "${OUT_ROOT}/x50_tmp"
"${PY}" - "$OUT_ROOT" <<'EOF'
import sys, pandas as pd
out = sys.argv[1]
a = pd.read_csv(f"{out}/x50_tmp/x50_seed0_splits.csv")
b = pd.read_csv(f"{out}/x50_tmp/x50_seed1_splits.csv")
pd.concat([a, b], ignore_index=True).to_csv(f"{out}/x50_merged_splits.csv", index=False)
print(f"Saved {out}/x50_merged_splits.csv")
EOF

echo "DONE ${OUT_ROOT}"
