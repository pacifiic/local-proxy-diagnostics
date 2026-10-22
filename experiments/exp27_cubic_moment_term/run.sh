#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${PY:-python}"
OUTDIR="${OUTDIR:?set OUTDIR to a fresh output directory}"
JOBS="${JOBS:-32}"

if [[ -e "$OUTDIR" ]] && [[ -n "$(find "$OUTDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty OUTDIR=$OUTDIR" >&2
  exit 1
fi

"$PY" "$SCRIPT_DIR/test_exp27_cubic_moment_term.py"
"$PY" "$SCRIPT_DIR/build_cubic_moment_term.py" \
  --outdir "$OUTDIR" --jobs "$JOBS"
"$PY" "$SCRIPT_DIR/audit_cubic_moment_term.py" \
  --outdir "$OUTDIR" --n-boot 2000

printf 'PASS\n' > "$OUTDIR/WORKFLOW_COMPLETE"
echo "DONE $OUTDIR"
