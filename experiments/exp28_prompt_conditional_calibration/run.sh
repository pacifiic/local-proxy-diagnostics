#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-python}"
MULTISEED_ROOT="${MULTISEED_ROOT:?set MULTISEED_ROOT}"
POOL_SCORED="${POOL_SCORED:-${REPO_ROOT}/data/dpo_candidate_pool/pool_scored.jsonl}"
OUTDIR="${OUTDIR:?set OUTDIR to a fresh output directory}"

"$PY" "$SCRIPT_DIR/test_prompt_conditional_calibration.py"
"$PY" "$SCRIPT_DIR/build_prompt_conditional_calibration.py" \
  --multiseed-root "$MULTISEED_ROOT" \
  --pool-scored "$POOL_SCORED" \
  --seeds 0 1 2 \
  --outdir "$OUTDIR"
"$PY" "$SCRIPT_DIR/audit_prompt_conditional_calibration.py" \
  --outdir "$OUTDIR"
"$PY" "$SCRIPT_DIR/plot_prompt_conditional_calibration.py" \
  --outdir "$OUTDIR"
printf 'PASS\n' > "$OUTDIR/WORKFLOW_COMPLETE"
echo "DONE $OUTDIR"
