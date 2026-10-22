#!/usr/bin/env bash
# exp09 v3: KL sweep via max_steps at fixed beta to land in the LOCAL window
# (target policy seq_kl ~ 0.1-3 nats). Reuses v1 shared prep + v2 fp32 ref logp.
set -euo pipefail
PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:?set SRC to a run dir with pool_scored/pref_pairs/coefficients/pool_eval_reflogp(fp32)}"
PREP_ROOT="${PREP_ROOT:-$SRC}"
RUN="${RUN:?set RUN}"
GPUS=(${GPUS:-4 5 6 7})
STEPS=(${STEPS:-4 8 16 32})
BETA="${BETA:-0.1}"
LR="${LR:-1e-5}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN"
for f in pool_scored.jsonl pref_pairs.jsonl coefficients.json pool_eval_reflogp.jsonl prompts_run_spec.json pool.jsonl.run_spec.json; do
  cp "$SRC/$f" "$RUN/$f"
done
for f in prompts_train.jsonl prompts_calibration.jsonl prompts_eval.jsonl pool.jsonl pool_p.jsonl; do
  if [ -f "$PREP_ROOT/$f" ]; then
    cp "$PREP_ROOT/$f" "$RUN/$f"
  elif [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$RUN/$f"
  else
    echo "missing required audit input: $f (set PREP_ROOT to the preparation run)" >&2
    exit 1
  fi
done
NG=${#GPUS[@]}
echo "v3 KL sweep: beta=$BETA lr=$LR steps=${STEPS[*]}"
for ((s=0; s<${#STEPS[@]}; s+=NG)); do
  PIDS=()
  for ((k=0; k<NG && s+k<${#STEPS[@]}; k++)); do
    ST=${STEPS[$((s+k))]}; G=${GPUS[$k]}
    OD="$RUN/steps_$ST"
    [ -f "$OD/summary.json" ] && { echo "skip steps=$ST"; continue; }
    CUDA_VISIBLE_DEVICES=$G "$PY" "$SCRIPT_DIR/07_dpo_worker.py" \
      --beta "$BETA" --lr "$LR" --max-steps "$ST" \
      --pairs-jsonl "$RUN/pref_pairs.jsonl" \
      --pool-scored-jsonl "$RUN/pool_scored.jsonl" \
      --pool-ref-logp-jsonl "$RUN/pool_eval_reflogp.jsonl" \
      --outdir "$OD" > "$RUN/steps_$ST.log" 2>&1 &
    PIDS+=($!); echo "  steps=$ST GPU $G pid ${PIDS[-1]}"
  done
  RC=0; for p in "${PIDS[@]:-}"; do [ -n "${p:-}" ] && { wait "$p" || RC=1; }; done
  [ $RC -ne 0 ] && echo "WARNING: wave failure"
done
echo "[v3-analysis] policy-KL local-window fit"
"$PY" - "$RUN" <<'PYEOF'
import sys, glob, json, math
import numpy as np, pandas as pd
run=sys.argv[1]; c=json.load(open(f"{run}/coefficients.json")); A,B=c["A"],c["B"]
rows=[]
for f in sorted(glob.glob(f"{run}/steps_*/summary.json"), key=lambda x:int(x.split('steps_')[1].split('/')[0])):
    s=json.load(open(f))
    d=s["mean_seq_kl_nats"]; drw=s["median_d_reweight"]; ess=s["median_ess"]
    rows.append({"steps":s["max_steps"],"seq_kl":d,"d_reweight":drw,"ess":ess,
                 "fresh_gold":s["mean_gain_fresh_gold"],"fresh_proxy":s["mean_gain_fresh_proxy"],
                 "reweight_gold":s["mean_gain_reweight_gold"],
                 "pred_AB_at_drw":A*math.sqrt(drw)+B*drw if drw>0 else float('nan')})
t=pd.DataFrame(rows); t.to_csv(f"{run}/kl_sweep_summary.csv",index=False)
print(f"locked A={A:.4f} B={B:.4f}\n")
print(t.round(4).to_string(index=False))
loc=t[t.d_reweight<=1.0]
print(f"\nlocal-window points (d_reweight<=1.0): {len(loc)}")
if len(loc)>=2:
    err=(loc.pred_AB_at_drw-loc.reweight_gold).abs().mean()
    print(f"mean |pred_AB - reweight_gold| in local window = {err:.4f}")
PYEOF
echo "DONE $RUN"
