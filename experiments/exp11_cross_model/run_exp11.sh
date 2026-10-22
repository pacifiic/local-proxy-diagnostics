#!/usr/bin/env bash
# exp11: cross-model generalization. Mirrors exp09 prep chain EXACTLY (same prompts,
# same seed/sampling params, same scorers, same merge join) with policy swapped to
# zephyr-7b-beta and no train split (no DPO here — frontier replication + transfer).
set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E9="$SCRIPT_DIR/../exp09_dpo_validation"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROMPTS_SRC="${PROMPTS_SRC:?set to exp09 prep dir holding prompts_*.jsonl}"
QWEN_COEF="${QWEN_COEF:?set to exp09 V4 coefficients.json}"
RUN="${RUN:?set RUN output dir}"
GPU_A=${GPU_A:-4}; GPU_B=${GPU_B:-5}
MODEL="${MODEL:-HuggingFaceH4/zephyr-7b-beta}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN"

echo "[11-00] reuse exp09 prompt splits (byte-copy, same prompt_ids)"
for f in prompts_train.jsonl prompts_calibration.jsonl prompts_eval.jsonl prompts_run_spec.json; do
  cp "$PROMPTS_SRC/$f" "$RUN/$f"
done

echo "[11-02] zephyr pool generation (GPU $GPU_A; no train split)"
CUDA_VISIBLE_DEVICES=$GPU_A "$PY" "$E9/02_generate_pool.py" \
  --prompts-dir "$RUN" --model "$MODEL" \
  --n-train 0 --n-cal 32 --n-eval 32 --seed 0 \
  --out-jsonl "$RUN/pool.jsonl"

echo "[11-03] proxy (Skywork, GPU $GPU_A) + gold (deberta, GPU $GPU_B) in parallel"
CUDA_VISIBLE_DEVICES=$GPU_A "$PY" "$E9/03_score_rm.py" \
  --in-jsonl "$RUN/pool.jsonl" --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --kind skywork --score-key proxy_score --out-jsonl "$RUN/pool_p.jsonl" &
P1=$!
CUDA_VISIBLE_DEVICES=$GPU_B "$PY" "$E9/03_score_rm.py" \
  --in-jsonl "$RUN/pool.jsonl" --model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --kind deberta --score-key gold_score --out-jsonl "$RUN/pool_g.jsonl" &
P2=$!
ST=0; wait $P1 || ST=1; wait $P2 || ST=1
[ $ST -ne 0 ] && { echo "scoring failed"; exit 1; }

echo "[11-03b] merge proxy+gold (same key join as exp09)"
"$PY" - <<PYEOF
import json
prox = {(r["prompt_id"], r["split"], r["sample_idx"]): r["proxy_score"]
        for r in map(json.loads, open("$RUN/pool_p.jsonl"))}
n = 0
with open("$RUN/pool_scored.jsonl", "w") as out:
    for l in open("$RUN/pool_g.jsonl"):
        r = json.loads(l)
        r["proxy_score"] = prox[(r["prompt_id"], r["split"], r["sample_idx"])]
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print("merged", n)
PYEOF

echo "[11-05] fit zephyr calibration coefficients"
"$PY" "$E9/05_fit_calibration.py" --scored-jsonl "$RUN/pool_scored.jsonl" \
  --out-json "$RUN/coefficients.json"

echo "[11-14] cross-model analysis (bootstrap CI)"
"$PY" "$SCRIPT_DIR/14_cross_model_analysis.py" \
  --zephyr-run "$RUN" --qwen-coefficients "$QWEN_COEF" \
  --out-prefix "$RUN/cross_model"
echo "DONE $RUN"
