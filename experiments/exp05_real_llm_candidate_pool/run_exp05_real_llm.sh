#!/usr/bin/env bash
# exp05 REAL-LLM run: the pipeline as designed (experiment 19), on actual LLM candidates.
# Generator: Qwen2.5-7B-Instruct (chat template). Proxy: Skywork-Reward-Llama-3.1-8B (chat,
# BOS-fixed). Gold: OpenAssistant deberta-v3-large-v2 (independent encoder family, canonical
# pair encoding). Prompts: 100 SummEval articles (summarization task), 30/70 prompt-level
# calibration/eval split, seed 0. No human labels needed (gold = independent model).
set -euo pipefail

PY="${PY:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE="${SCRIPT_DIR}/code"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/exp05_real_llm_$(date -u +%Y%m%d_%H%M%S)}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
mkdir -p "${OUT_ROOT}"
export PYTHONPATH="${REPO_ROOT}/lib${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

echo "[exp05-real] build summarization prompts from SummEval articles"
"${PY}" - "$REPO_ROOT" "$OUT_ROOT" <<'PYEOF'
import sys
import pandas as pd
root, out = sys.argv[1], sys.argv[2]
items = pd.read_csv(f"{root}/data/summeval_items.csv").drop_duplicates("article_id")
rows = [{"prompt_id": r.article_id,
         "prompt": ("Summarize the following news article in 3-4 sentences.\n\n"
                    f"Article:\n{str(r.article)[:2000]}")}
        for r in items.itertuples()]
pd.DataFrame(rows).to_csv(f"{out}/raw_prompts.csv", index=False)
print(f"wrote {len(rows)} prompts")
PYEOF

"${PY}" "${CODE}/01_prepare_prompts.py" \
  --input-csv "${OUT_ROOT}/raw_prompts.csv" \
  --prompt-col prompt --prompt-id-col prompt_id \
  --calibration-frac 0.3 --random-seed 0 \
  --out-csv "${OUT_ROOT}/prompts.csv" \
  --out-spec-json "${OUT_ROOT}/prompts.run_spec.json"

echo "[exp05-real] generate 16 candidates/prompt with Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${PY}" "${CODE}/02_generate_candidates_vllm.py" \
  --prompts-csv "${OUT_ROOT}/prompts.csv" \
  --model Qwen/Qwen2.5-7B-Instruct --prompt-template chat \
  --n-candidates 16 --temperature 0.8 --top-p 0.95 --max-tokens 256 --seed 0 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --out-csv "${OUT_ROOT}/candidates.csv" \
  --out-spec-json "${OUT_ROOT}/candidates.run_spec.json"

echo "[exp05-real] proxy scores: Skywork-Reward-Llama-3.1-8B (chat)"
CUDA_VISIBLE_DEVICES=0 "${PY}" "${CODE}/03_score_sequence_classifier.py" \
  --candidates-csv "${OUT_ROOT}/candidates.csv" \
  --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --input-format chat --score-col proxy_score --dtype bfloat16 --device cuda --batch-size 16 \
  --out-csv "${OUT_ROOT}/candidates_proxy.csv" \
  --out-spec-json "${OUT_ROOT}/candidates_proxy.run_spec.json"

echo "[exp05-real] gold scores: OpenAssistant deberta-v3-large-v2 (pair encoding)"
CUDA_VISIBLE_DEVICES=1 "${PY}" "${CODE}/03_score_sequence_classifier.py" \
  --candidates-csv "${OUT_ROOT}/candidates_proxy.csv" \
  --model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --input-format pair --score-col gold_score --dtype float32 --device cuda --batch-size 32 \
  --out-csv "${OUT_ROOT}/candidates_scored.csv" \
  --out-spec-json "${OUT_ROOT}/candidates_scored.run_spec.json"

echo "[exp05-real] fit/predict gain curves (prospective, existing prompt split)"
"${PY}" "${CODE}/04_fit_predict_gain_curves.py" \
  --candidates-csv "${OUT_ROOT}/candidates_scored.csv" \
  --prompt-col prompt_id --split-col split --use-existing-split \
  --proxy-col proxy_score --gold-col gold_score \
  --calibration-frac 0.3 --n-repeats 1 --random-seed 0 \
  --grid-d-values 0.01,0.05,0.1 --endpoint-d 0.1 --fit-max-d 0.05 --n-grid 100 \
  --mlp-hidden-layers 16,16 --mlp-alpha 0.001 --mlp-max-iter 300 --mlp-seed 0 \
  --outdir "${OUT_ROOT}/gain_curves"

echo "[exp05-real] BoN scope check"
"${PY}" "${CODE}/05_bon_scope_check.py" \
  --candidates-csv "${OUT_ROOT}/candidates_scored.csv" \
  --prompt-col prompt_id --split-col split --use-existing-split \
  --proxy-col proxy_score --gold-col gold_score \
  --calibration-frac 0.3 --n-repeats 1 --random-seed 0 \
  --n-list 2,4,8,16 \
  --outdir "${OUT_ROOT}/bon_scope"

echo "[exp05-real] artifacts"
"${PY}" "${CODE}/06_make_artifacts.py" \
  --gain-root "${OUT_ROOT}/gain_curves" \
  --bon-root "${OUT_ROOT}/bon_scope" \
  --outdir "${OUT_ROOT}/artifacts" \
  --variant-label "(real LLM candidates)"

echo "DONE ${OUT_ROOT}"
