#!/usr/bin/env bash
# exp09: real DPO training validation (preliminary scale, 4 GPUs).
#   shared prep on GPU_A/GPU_B -> per-beta workers on GPUs 4,5,6,7 in parallel.
# All roles are models (proxy=Skywork-8B, gold=OpenAssistant DeBERTa RM):
# no human labels needed;
# cite as model-gold per CITATION_CAVEATS conventions.
set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN="${RUN:-${REPO_ROOT}/outputs/exp09_dpo_validation_$(date -u +%Y%m%d_%H%M%S)}"
GPUS=(${GPUS:-4 5 6 7})
BETAS=(${BETAS:-0.5 0.1 0.05 0.01})
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN"
echo "run=$RUN gpus=${GPUS[*]} betas=${BETAS[*]}"

echo "[01] prompts"
"$PY" "$SCRIPT_DIR/01_prepare_prompts.py" --outdir "$RUN" \
  --n-train "${N_TRAIN:-1000}" --n-cal "${N_CAL:-100}" --n-eval "${N_EVAL:-200}" --seed 0

echo "[02] reference-pool generation (GPU ${GPUS[0]})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/02_generate_pool.py" \
  --prompts-dir "$RUN" --out-jsonl "$RUN/pool.jsonl" \
  --n-train "${NS_TRAIN:-8}" --n-cal "${NS_CAL:-32}" --n-eval "${NS_EVAL:-32}" --seed 0

GPU_B=${GPUS[1]:-${GPUS[0]}}
echo "[03] proxy scoring (Skywork, GPU ${GPUS[0]}) + gold scoring (DeBERTa RM, GPU ${GPU_B}) in parallel"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/03_score_rm.py" \
  --in-jsonl "$RUN/pool.jsonl" --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --kind skywork --score-key proxy_score --out-jsonl "$RUN/pool_p.jsonl" &
P1=$!
CUDA_VISIBLE_DEVICES=${GPU_B} "$PY" - <<PYEOF &
import json, subprocess, sys
# gold scores only needed for calibration+eval rows; filter first to save time
rows=[json.loads(l) for l in open("$RUN/pool.jsonl")]
keep=[r for r in rows if r["split"] in ("calibration","eval")]
with open("$RUN/pool_ce.jsonl","w") as f:
    for r in keep: f.write(json.dumps(r, ensure_ascii=False)+"\n")
subprocess.run([sys.executable, "$SCRIPT_DIR/03_score_rm.py",
  "--in-jsonl","$RUN/pool_ce.jsonl","--model","OpenAssistant/reward-model-deberta-v3-large-v2",
  "--kind","deberta","--score-key","gold_score","--out-jsonl","$RUN/pool_ce_g.jsonl"],check=True)
PYEOF
P2=$!
wait $P1
wait $P2

echo "[03b] merge proxy+gold for cal/eval rows"
"$PY" - <<PYEOF
import json
prox={(r["prompt_id"],r["split"],r["sample_idx"]):r["proxy_score"] for r in map(json.loads,open("$RUN/pool_p.jsonl"))}
out=open("$RUN/pool_scored.jsonl","w")
n=0
for l in open("$RUN/pool_ce_g.jsonl"):
    r=json.loads(l); r["proxy_score"]=prox[(r["prompt_id"],r["split"],r["sample_idx"])]
    out.write(json.dumps(r,ensure_ascii=False)+"\n"); n+=1
print("merged",n)
PYEOF

echo "[04] preference pairs (train split, proxy scores)"
"$PY" "$SCRIPT_DIR/04_build_pref_pairs.py" --scored-jsonl "$RUN/pool_p.jsonl" \
  --out-jsonl "$RUN/pref_pairs.jsonl"

echo "[05] calibration fit (A,B locked before eval)"
"$PY" "$SCRIPT_DIR/05_fit_calibration.py" --scored-jsonl "$RUN/pool_scored.jsonl" \
  --out-json "$RUN/coefficients.json"

echo "[06] ref logp on eval pool (GPU ${GPUS[0]})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/06_logp.py" \
  --in-jsonl "$RUN/pool_scored.jsonl" --filter-split eval \
  --model Qwen/Qwen2.5-3B-Instruct --logp-key logp_ref \
  --out-jsonl "$RUN/pool_eval_reflogp.jsonl"

echo "[07] beta workers in parallel"
PIDS=()
for i in "${!BETAS[@]}"; do
  B=${BETAS[$i]}; G=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$G "$PY" "$SCRIPT_DIR/07_dpo_worker.py" \
    --beta "$B" --pairs-jsonl "$RUN/pref_pairs.jsonl" \
    --pool-scored-jsonl "$RUN/pool_scored.jsonl" \
    --pool-ref-logp-jsonl "$RUN/pool_eval_reflogp.jsonl" \
    --outdir "$RUN/beta_$B" > "$RUN/beta_$B.log" 2>&1 &
  PIDS+=($!)
  echo "  beta=$B on GPU $G (pid ${PIDS[-1]})"
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
[ $STATUS -ne 0 ] && echo "WARNING: at least one worker failed (see beta_*.log)"

echo "[08] summary"
"$PY" "$SCRIPT_DIR/08_summarize.py" --run-root "$RUN" --coefficients-json "$RUN/coefficients.json"
echo "[10] data audit"
"$PY" "$SCRIPT_DIR/10_data_audit.py" --run-root "$RUN"
echo "DONE $RUN"
