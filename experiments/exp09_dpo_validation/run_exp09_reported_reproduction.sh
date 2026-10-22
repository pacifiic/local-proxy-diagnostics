#!/usr/bin/env bash
# Reproduce the reported six-checkpoint DPO analysis from immutable scored inputs.
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUTS="${INPUTS:-${REPO_ROOT}/data/dpo_candidate_pool}"
RUN="${RUN:?set RUN to a fresh output directory}"
GPUS=(${GPUS:-0 1 2 3 4 5})
STEPS=(${STEPS:-64 90 120 150 180 200})
BETA="${BETA:-0.1}"
LR="${LR:-1e-5}"

export TOKENIZERS_PARALLELISM=false
unset PYTHONNOUSERSITE

if [[ -e "$RUN" ]] && [[ -n "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty RUN=$RUN" >&2
  exit 1
fi
mkdir -p "$RUN"

(
  cd "$INPUTS"
  sha256sum -c SHA256SUMS
)
cp "$INPUTS"/*.json "$INPUTS"/*.jsonl "$RUN"/

"$PY" - "$RUN/environment.json" <<'PYEOF'
import importlib
import json
import platform
import sys
from pathlib import Path

names = [
    "torch", "transformers", "datasets", "trl", "peft", "accelerate",
    "vllm", "tokenizers", "safetensors", "huggingface_hub",
]
expected = {
    "torch": "2.11.0+cu130",
    "transformers": "4.57.6",
    "datasets": "4.8.4",
    "trl": "1.8.0",
    "peft": "0.10.0",
    "accelerate": "1.14.0",
    "vllm": "0.22.1",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
    "huggingface_hub": "0.36.2",
}
packages = {}
for name in names:
    module = importlib.import_module(name)
    module_path = str(Path(module.__file__).resolve())
    version = getattr(module, "__version__", "unknown")
    if version != expected[name]:
        raise RuntimeError(
            f"{name} version {version} does not match reported-run "
            f"environment {expected[name]}"
        )
    packages[name] = {
        "version": version,
        "path": module_path,
    }
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "python": platform.python_version(),
            "executable": sys.executable,
            "packages": packages,
        },
        indent=2,
    ),
    encoding="utf-8",
)
PYEOF

NG=${#GPUS[@]}
for ((start=0; start<${#STEPS[@]}; start+=NG)); do
  PIDS=()
  for ((k=0; k<NG && start+k<${#STEPS[@]}; k++)); do
    STEP=${STEPS[$((start+k))]}
    GPU=${GPUS[$k]}
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$SCRIPT_DIR/07_dpo_worker.py" \
      --beta "$BETA" --lr "$LR" --max-steps "$STEP" \
      --pairs-jsonl "$RUN/pref_pairs.jsonl" \
      --pool-scored-jsonl "$RUN/pool_scored.jsonl" \
      --pool-ref-logp-jsonl "$RUN/pool_eval_reflogp.jsonl" \
      --outdir "$RUN/steps_$STEP" > "$RUN/steps_$STEP.log" 2>&1 &
    PIDS+=($!)
    echo "steps=$STEP GPU=$GPU pid=${PIDS[-1]}"
  done
  STATUS=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=1
  done
  if [[ $STATUS -ne 0 ]]; then
    echo "one or more DPO workers failed; inspect $RUN/steps_*.log" >&2
    exit 1
  fi
done

"$PY" "$SCRIPT_DIR/11_combined_analysis.py" \
  --run-dirs "$RUN" --coefficients-json "$RUN/coefficients.json" \
  --out-prefix "$RUN/exp09_final"
"$PY" "$SCRIPT_DIR/09_frontier_check.py" \
  --run-root "$RUN" --coefficients-json "$RUN/coefficients.json"
"$PY" "$SCRIPT_DIR/10_data_audit.py" --run-root "$RUN"
"$PY" "$SCRIPT_DIR/12_poolsize_robustness.py" \
  --run-dir "$RUN" --reps 200 --out-prefix "$RUN/poolsize"
"$PY" "$SCRIPT_DIR/13_poolsize_audit.py" \
  --run-dir "$RUN" --summary "$RUN/poolsize_summary.csv" \
  --per-rep "$RUN/poolsize_per_rep.csv"
"$PY" "$SCRIPT_DIR/16_length_bias_check.py" \
  --scored-jsonl "$RUN"/steps_*/fresh_samples_scored.jsonl \
  > "$RUN/length_bias_check.txt"
"$PY" "$SCRIPT_DIR/17_efficiency_figure.py" \
  --summary "$RUN/exp09_final_summary.csv" --outdir "$RUN/efficiency"
"$PY" "$SCRIPT_DIR/18_curvature_diagnostic.py" --run-dir "$RUN"

printf 'PASS\n' > "$RUN/REPORTED_REPRODUCTION_COMPLETE"
echo "DONE $RUN"
