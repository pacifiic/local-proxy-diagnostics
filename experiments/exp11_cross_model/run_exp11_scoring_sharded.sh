#!/usr/bin/env bash
# exp11 stage 2 (sharded): proxy scoring split into 3 deterministic contiguous
# shards (GPUs SHARD_GPUS) + gold scoring in parallel (GPU_GOLD), then key-join
# merge, calibration fit, cross-model analysis. Row scoring is per-row
# independent; shard boundaries only change batch composition (bf16 noise at
# most). Join integrity is hard-checked by 15_cross_model_audit.py.
set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E9="$SCRIPT_DIR/../exp09_dpo_validation"
RUN="${RUN:?}"
QWEN_COEF="${QWEN_COEF:?}"
SHARD_GPUS=(${SHARD_GPUS:-4 6 7})
GPU_GOLD=${GPU_GOLD:-5}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

N=${#SHARD_GPUS[@]}
echo "[11-shard] deterministic contiguous split of pool.jsonl into $N shards"
"$PY" - "$RUN" "$N" <<'PYEOF'
import sys
run, n = sys.argv[1], int(sys.argv[2])
lines = open(f"{run}/pool.jsonl").read().splitlines(True)
per = (len(lines) + n - 1) // n
for i in range(n):
    with open(f"{run}/pool.shard{i}.jsonl", "w") as f:
        f.writelines(lines[i*per:(i+1)*per])
print(f"split {len(lines)} rows into {n} shards of <= {per}")
PYEOF

echo "[11-03s] proxy shards on GPUs ${SHARD_GPUS[*]} + gold on GPU $GPU_GOLD (parallel)"
PIDS=()
for ((i=0; i<N; i++)); do
  CUDA_VISIBLE_DEVICES=${SHARD_GPUS[$i]} "$PY" "$E9/03_score_rm.py" \
    --in-jsonl "$RUN/pool.shard$i.jsonl" --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --kind skywork --score-key proxy_score --out-jsonl "$RUN/pool_p.shard$i.jsonl" \
    > "$RUN/score_p_shard$i.log" 2>&1 &
  PIDS+=($!)
done
CUDA_VISIBLE_DEVICES=$GPU_GOLD "$PY" "$E9/03_score_rm.py" \
  --in-jsonl "$RUN/pool.jsonl" --model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --kind deberta --score-key gold_score --out-jsonl "$RUN/pool_g.jsonl" \
  > "$RUN/score_g.log" 2>&1 &
PIDS+=($!)
ST=0; for p in "${PIDS[@]}"; do wait "$p" || ST=1; done
[ $ST -ne 0 ] && { echo "scoring failed — see score_*.log"; exit 1; }

echo "[11-03c] concat shards (order-preserving) -> pool_p.jsonl"
cat $(for ((i=0; i<N; i++)); do echo "$RUN/pool_p.shard$i.jsonl"; done) > "$RUN/pool_p.jsonl"

echo "[11-03b] merge proxy+gold (same key join as exp09; KeyError = loud fail)"
"$PY" - <<PYEOF
import json
prox = {(r["prompt_id"], r["split"], r["sample_idx"]): r["proxy_score"]
        for r in map(json.loads, open("$RUN/pool_p.jsonl"))}
raw = sum(1 for _ in open("$RUN/pool_p.jsonl"))
assert len(prox) == raw, f"duplicate keys after shard concat: {raw} rows, {len(prox)} keys"
n = 0
with open("$RUN/pool_scored.jsonl", "w") as out:
    for l in open("$RUN/pool_g.jsonl"):
        r = json.loads(l)
        r["proxy_score"] = prox[(r["prompt_id"], r["split"], r["sample_idx"])]
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
assert n == len(prox), f"gold rows {n} != proxy rows {len(prox)}"
print("merged", n)
PYEOF

echo "[11-05] fit zephyr calibration coefficients"
"$PY" "$E9/05_fit_calibration.py" --scored-jsonl "$RUN/pool_scored.jsonl" \
  --out-json "$RUN/coefficients.json"

echo "[11-14] cross-model analysis"
"$PY" "$SCRIPT_DIR/14_cross_model_analysis.py" \
  --zephyr-run "$RUN" --qwen-coefficients "$QWEN_COEF" --out-prefix "$RUN/cross_model"
echo "DONE $RUN"
