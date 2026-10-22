#!/usr/bin/env bash
# Second-wave beta workers reusing an existing run's shared prep
# (pool, pref pairs, locked A/B, ref logp). Waves of len(GPUS).
set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:?set RUN to the existing exp09 run dir}"
GPUS=(${GPUS:-4 5 6 7})
BETAS=(${BETAS:?set BETAS})
NG=${#GPUS[@]}
for ((start=0; start<${#BETAS[@]}; start+=NG)); do
  PIDS=()
  for ((k=0; k<NG && start+k<${#BETAS[@]}; k++)); do
    B=${BETAS[$((start+k))]}; G=${GPUS[$k]}
    if [ -f "$RUN/beta_$B/summary.json" ]; then echo "skip beta=$B (done)"; continue; fi
    echo "wave: beta=$B on GPU $G"
    CUDA_VISIBLE_DEVICES=$G "$PY" "$SCRIPT_DIR/07_dpo_worker.py" \
      --beta "$B" --pairs-jsonl "$RUN/pref_pairs.jsonl" \
      --pool-scored-jsonl "$RUN/pool_scored.jsonl" \
      --pool-ref-logp-jsonl "$RUN/pool_eval_reflogp.jsonl" \
      --outdir "$RUN/beta_$B" > "$RUN/beta_$B.log" 2>&1 &
    PIDS+=($!)
  done
  ST=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || ST=1; }; done
  [ $ST -ne 0 ] && echo "WARNING: failures in this wave (see logs)"
done
echo "ALL WAVES DONE"
