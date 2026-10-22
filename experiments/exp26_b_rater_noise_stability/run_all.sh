#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-python}"
OUTDIR="${OUTDIR:?set OUTDIR to a fresh output directory}"
N_REPEATS="${N_REPEATS:-50}"
JOBS="${JOBS:-16}"
RANDOM_SEED="${RANDOM_SEED:-0}"

"$PY" "$SCRIPT_DIR/build_b_rater_noise_stability.py" \
  --data "$REPO_ROOT/data/summeval_long.csv" \
  --outdir "$OUTDIR" \
  --n-repeats "$N_REPEATS" \
  --jobs "$JOBS" \
  --random-seed "$RANDOM_SEED"

"$PY" "$SCRIPT_DIR/make_artifacts.py" --outdir "$OUTDIR"
"$PY" "$SCRIPT_DIR/audit_b_rater_noise_stability.py" --outdir "$OUTDIR"
