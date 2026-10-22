#!/usr/bin/env bash
# Fixed-pool DPO seed sweep for held-out reweight-gain prediction.
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUTS="${INPUTS:-${REPO_ROOT}/data/dpo_candidate_pool}"
ROOT="${ROOT:?set ROOT to a fresh output directory}"
SEEDS=(${SEEDS:-0 1 2})
STEPS=(${STEPS:-64 90 120 150 180 200})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
BETA="${BETA:-0.1}"
LR="${LR:-1e-5}"

export TOKENIZERS_PARALLELISM=false
unset PYTHONNOUSERSITE

if [[ -e "$ROOT" ]] && [[ -n "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty ROOT=$ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"

(
  cd "$INPUTS"
  sha256sum -c <<'EOF'
ed13e5847802b9ba31ddf05c89444e3f4e4f841323b4dc069d5c0d139a01176b  prompts_train.jsonl
98e3b8b4b5da772bb654347bd785e869a32ea3dade3692c524e1900f7be68b55  prompts_calibration.jsonl
05309bb16ee364e2804e73d4b0508c8f55ffb9113a5ccfb0c8b1f1f16ec76f74  prompts_eval.jsonl
fafdb5ef7ce4fa05c13969c61eaf80071dccceaa365ce4f13c5f30f405a78be3  pool.jsonl
cacaec5e7981f3b8da7d6efc835c31d84ff0e19a24413235001e3b4b7a0eb268  pool_p.jsonl
1c6ba78bf5b0e4ddfc1c6b6cc6b4ba2aae51dcdcb3196ef47fba1f2b8ee55ab2  pool_scored.jsonl
38275da384429031cb050fd3d5e4d44376841e85b2859d648a7a36506bfa30e8  pref_pairs.jsonl
c1755a61e77d694b47ff4b275a05b929a2d95c111ca9e60cb8401e7ce2e67f04  pool_eval_reflogp.jsonl
a94aa7d29226834c43125fae7c9af84e2c2c76412d724c74a304d22432603adc  coefficients.json
EOF
)

for seed in "${SEEDS[@]}"; do
  run="$ROOT/seed_$seed"
  mkdir -p "$run"
  cp "$INPUTS"/*.json "$INPUTS"/*.jsonl "$run"/
done

"$PY" - "$ROOT/environment.json" <<'PYEOF'
import importlib
import json
import platform
import sys
from pathlib import Path

names = ["torch", "transformers", "datasets", "trl", "peft", "accelerate"]
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "python": platform.python_version(),
            "packages": {
                name: {
                    "version": getattr(importlib.import_module(name), "__version__", "unknown"),
                }
                for name in names
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)
PYEOF

JOBS=()
for seed in "${SEEDS[@]}"; do
  for step in "${STEPS[@]}"; do
    JOBS+=("$seed:$step")
  done
done

NG=${#GPUS[@]}
for ((start=0; start<${#JOBS[@]}; start+=NG)); do
  PIDS=()
  for ((k=0; k<NG && start+k<${#JOBS[@]}; k++)); do
    IFS=: read -r seed step <<< "${JOBS[$((start+k))]}"
    gpu=${GPUS[$k]}
    run="$ROOT/seed_$seed"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT_DIR/07_dpo_worker.py" \
      --beta "$BETA" --lr "$LR" --max-steps "$step" --seed "$seed" \
      --skip-fresh-evaluation \
      --pairs-jsonl "$run/pref_pairs.jsonl" \
      --pool-scored-jsonl "$run/pool_scored.jsonl" \
      --pool-ref-logp-jsonl "$run/pool_eval_reflogp.jsonl" \
      --outdir "$run/steps_$step" > "$run/steps_$step.log" 2>&1 &
    PIDS+=($!)
    echo "seed=$seed steps=$step GPU=$gpu pid=${PIDS[-1]}"
  done
  status=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || status=1
  done
  if [[ $status -ne 0 ]]; then
    echo "one or more seed-sweep workers failed" >&2
    exit 1
  fi
done

SPLINE_ARGS=()
for seed in "${SEEDS[@]}"; do
  run="$ROOT/seed_$seed"
  "$PY" "$SCRIPT_DIR/11_combined_analysis.py" \
    --run-dirs "$run" --coefficients-json "$run/coefficients.json" \
    --out-prefix "$run/exp09_final"
  "$PY" "$SCRIPT_DIR/10_data_audit.py" --run-root "$run" \
    > "$run/data_audit.txt"
  SPLINE_ARGS+=(
    --source
    "seed_$seed::$run/exp09_final_per_prompt.csv::$run/pool_scored.jsonl"
  )
done

"$PY" "$SCRIPT_DIR/19_spline_baseline.py" \
  "${SPLINE_ARGS[@]}" --outdir "$ROOT/spline"
"$PY" "$SCRIPT_DIR/20_multiseed_summary.py" \
  --root "$ROOT" --seeds "${SEEDS[@]}"

printf 'PASS\n' > "$ROOT/MULTISEED_WORKFLOW_COMPLETE"
echo "DONE $ROOT"
