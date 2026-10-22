#!/usr/bin/env bash
# exp09 v2: reuse v1 shared prep (pool/pairs/calibration), recompute ref logp in fp32,
# run stronger DPO (3 epochs, lr 5e-5) with fp32 logp + matched HF-generate fresh baseline.
# Verification gate then decides whether to expand betas.
set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1="${V1:?set V1 to the v1 run dir}"
RUN="${RUN:?set RUN to the new v2 run dir}"
GPUS=(${GPUS:-4 5 6 7})
BETAS=(${BETAS:-0.5 0.1 0.05 0.01})
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN"
# reuse shared prep from v1
for f in pool_scored.jsonl pref_pairs.jsonl coefficients.json prompts_run_spec.json pool.jsonl.run_spec.json; do
  cp "$V1/$f" "$RUN/$f"
done
echo "[v2-06] ref logp on eval pool (fp32, GPU ${GPUS[0]})"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" "$SCRIPT_DIR/06_logp.py" \
  --in-jsonl "$RUN/pool_scored.jsonl" --filter-split eval \
  --model Qwen/Qwen2.5-3B-Instruct --logp-key logp_ref --dtype float32 \
  --out-jsonl "$RUN/pool_eval_reflogp.jsonl"

echo "[v2-07] beta workers (3 epoch / lr 5e-5 / fp32 logp)"
NG=${#GPUS[@]}
for ((start=0; start<${#BETAS[@]}; start+=NG)); do
  PIDS=()
  for ((k=0; k<NG && start+k<${#BETAS[@]}; k++)); do
    B=${BETAS[$((start+k))]}; G=${GPUS[$k]}
    [ -f "$RUN/beta_$B/summary.json" ] && { echo "skip $B"; continue; }
    CUDA_VISIBLE_DEVICES=$G "$PY" "$SCRIPT_DIR/07_dpo_worker.py" \
      --beta "$B" --pairs-jsonl "$RUN/pref_pairs.jsonl" \
      --pool-scored-jsonl "$RUN/pool_scored.jsonl" \
      --pool-ref-logp-jsonl "$RUN/pool_eval_reflogp.jsonl" \
      --outdir "$RUN/beta_$B" > "$RUN/beta_$B.log" 2>&1 &
    PIDS+=($!); echo "  beta=$B GPU $G pid ${PIDS[-1]}"
  done
  ST=0; for p in "${PIDS[@]:-}"; do [ -n "${p:-}" ] && { wait "$p" || ST=1; }; done
  [ $ST -ne 0 ] && echo "WARNING: wave failures"
done

echo "[v2-gate] signal-quality check"
"$PY" - "$RUN" <<'PYEOF'
import sys, glob, json
import pandas as pd, numpy as np
from scipy.stats import spearmanr
run = sys.argv[1]
frames = []
for f in sorted(glob.glob(f"{run}/beta_*/per_prompt.csv")):
    b = float(f.split("beta_")[1].split("/")[0])
    d = pd.read_csv(f); d["beta"] = b; frames.append(d)
    r = spearmanr(d.d_reweight, d.gain_reweight_proxy).statistic
    print(f"beta={b}: d_rw med {d.d_reweight.median():.3f} | corr(d,proxy_gain) {r:+.2f} | "
          f"gain_rw_proxy {d.gain_reweight_proxy.mean():+.3f} gold {d.gain_reweight_gold.mean():+.3f} | "
          f"fresh_proxy {d.gain_fresh_proxy.mean():+.3f} gold {d.gain_fresh_gold.mean():+.3f}")
alld = pd.concat(frames)
med_by_beta = alld.groupby("beta").d_reweight.median().sort_index(ascending=False)
spread = med_by_beta.max() / max(med_by_beta.min(), 1e-9)
pooled_corr = spearmanr(alld.d_reweight, alld.gain_reweight_proxy).statistic
print(f"\nd spread(max/min across beta) = {spread:.2f}  (want > 1.5)")
print(f"pooled corr(d, proxy_gain)    = {pooled_corr:+.3f}  (want > 0.3)")
ok = spread > 1.5 and pooled_corr > 0.3
print("GATE:", "PASS — signal real, expand betas" if ok else "FAIL — still noise-dominated, need stronger training")
PYEOF
echo "DONE $RUN"
