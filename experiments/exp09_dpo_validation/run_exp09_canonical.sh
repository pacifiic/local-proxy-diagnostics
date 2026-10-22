#!/usr/bin/env bash
# End-to-end reproduction of the six-checkpoint DPO result from the recorded candidate pool.
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN="${RUN:?set RUN to a fresh output directory}"
GPUS=(${GPUS:-0 1 2 3 4 5})
STEPS=(${STEPS:-64 90 120 150 180 200})
BETA="${BETA:-0.1}"
LR="${LR:-1e-5}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

if [[ -e "$RUN" ]] && [[ -n "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty RUN=$RUN" >&2
  exit 1
fi
mkdir -p "$RUN"

echo "[01] pinned UltraFeedback prompt splits"
"$PY" "$SCRIPT_DIR/01_prepare_prompts.py" \
  --outdir "$RUN" --n-train 1000 --n-cal 100 --n-eval 200 --seed 0

echo "[02] pinned Qwen reference pool (GPU ${GPUS[0]})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/02_generate_pool.py" \
  --prompts-dir "$RUN" --out-jsonl "$RUN/pool.jsonl" \
  --n-train 8 --n-cal 32 --n-eval 32 --seed 0

GPU_B=${GPUS[1]:-${GPUS[0]}}
echo "[03] proxy and model-gold scoring (GPUs ${GPUS[0]}, ${GPU_B})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/03_score_rm.py" \
  --in-jsonl "$RUN/pool.jsonl" \
  --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --kind skywork --score-key proxy_score --out-jsonl "$RUN/pool_p.jsonl" &
P1=$!
"$PY" - "$RUN/pool.jsonl" "$RUN/pool_ce.jsonl" <<'PYEOF'
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
with source.open(encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle]
kept = [row for row in rows if row["split"] in {"calibration", "eval"}]
with output.open("w", encoding="utf-8") as handle:
    for row in kept:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"filtered {len(kept)}/{len(rows)} calibration+eval rows")
PYEOF
CUDA_VISIBLE_DEVICES=${GPU_B} "$PY" "$SCRIPT_DIR/03_score_rm.py" \
  --in-jsonl "$RUN/pool_ce.jsonl" \
  --model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --kind deberta --score-key gold_score --out-jsonl "$RUN/pool_ce_g.jsonl" &
P2=$!
wait "$P1"
wait "$P2"

echo "[03b] validated one-to-one score merge"
"$PY" - "$RUN/pool_p.jsonl" "$RUN/pool_ce_g.jsonl" "$RUN/pool_scored.jsonl" <<'PYEOF'
import json
import sys
from pathlib import Path

proxy_path, gold_path, output_path = map(Path, sys.argv[1:])
key_fields = ("prompt_id", "split", "sample_idx")

def load(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]

proxy_rows = load(proxy_path)
gold_rows = load(gold_path)
proxy = {}
for row in proxy_rows:
    key = tuple(row[field] for field in key_fields)
    if key in proxy:
        raise ValueError(f"duplicate proxy key: {key}")
    proxy[key] = row

seen = set()
with output_path.open("w", encoding="utf-8") as handle:
    for row in gold_rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"duplicate gold key: {key}")
        seen.add(key)
        source = proxy.get(key)
        if source is None:
            raise KeyError(f"gold key absent from proxy scores: {key}")
        if row["prompt"] != source["prompt"] or row["response"] != source["response"]:
            raise ValueError(f"text mismatch at key: {key}")
        merged = dict(row)
        merged["proxy_score"] = source["proxy_score"]
        handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
expected = {
    key for key in proxy
    if key[1] in {"calibration", "eval"}
}
if seen != expected:
    raise ValueError(
        f"calibration/eval key mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}"
    )
print(f"merged {len(seen)} rows with exact key/text agreement")
PYEOF

echo "[04-06] preference pairs, locked coefficients, and fp32 reference logp"
"$PY" "$SCRIPT_DIR/04_build_pref_pairs.py" \
  --scored-jsonl "$RUN/pool_p.jsonl" --out-jsonl "$RUN/pref_pairs.jsonl"
"$PY" "$SCRIPT_DIR/05_fit_calibration.py" \
  --scored-jsonl "$RUN/pool_scored.jsonl" --out-json "$RUN/coefficients.json"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/06_logp.py" \
  --in-jsonl "$RUN/pool_scored.jsonl" --filter-split eval \
  --model Qwen/Qwen2.5-3B-Instruct --logp-key logp_ref --dtype float32 \
  --out-jsonl "$RUN/pool_eval_reflogp.jsonl"

echo "[07] six local DPO checkpoints"
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
    echo "  steps=$STEP GPU=$GPU pid=${PIDS[-1]}"
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

echo "[08-18] combined analyses and audits"
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

printf 'PASS\n' > "$RUN/CANONICAL_WORKFLOW_COMPLETE"
echo "DONE $RUN"
