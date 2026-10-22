#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${1:?usage: run.sh OUTDIR [JOBS]}"
JOBS="${2:-32}"

python "$ROOT/experiments/exp36_dominant_local_law/build.py" \
  --outdir "$OUTDIR" \
  --jobs "$JOBS"
python "$ROOT/experiments/exp36_dominant_local_law/audit.py" \
  --outdir "$OUTDIR"
python "$ROOT/experiments/exp36_dominant_local_law/data_chain_audit.py" \
  --outdir "$OUTDIR"
