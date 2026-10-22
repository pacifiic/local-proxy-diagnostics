# exp28: Modern Open-Generator Control

This experiment replaces the older Zephyr generator with recent open
instruction models while keeping the prompts, candidate count, learned reward
proxy, and model-based silver-gold scorer fixed.

- Generator: one public instruction model per run.
- Proxy: `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`.
- Silver gold: `OpenAssistant/reward-model-deberta-v3-large-v2`.
- Data: 100 fixed calibration prompts and 200 disjoint evaluation prompts from
  the existing UltraFeedback prompt manifest.
- Sampling: 32 candidates per prompt, temperature 0.9, top-p 0.95, and
  candidate-level SHA-256-derived seeds.
- Analysis: fixed-prompt-KL gain curves with the same baselines and local
  `d <= 0.10` window as exp25/exp27.

Download the pinned checkpoints after authenticating with Hugging Face:

```bash
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  --revision f6fae9795746f63c9be8344932f01275f3c63734 \
  --local-dir inputs/models/Ministral-3-8B-Instruct-2512-BF16

hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --local-dir inputs/models/Llama-3.1-8B-Instruct
```

The Llama checkpoint requires prior approval on its Hugging Face model page.
No access token is stored in a run directory or run specification.

Run one locally downloaded model:

```bash
MODEL=mistralai/Ministral-3-8B-Instruct-2512-BF16 \
MODEL_LOAD_PATH=inputs/Ministral-3-8B-Instruct-2512-BF16 \
MODEL_REVISION=f6fae9795746f63c9be8344932f01275f3c63734 \
PRETOKENIZE_PROMPTS=1 MISTRAL_COMMON_CHAT=1 TEXT_ONLY=1 \
VLLM_TOKENIZER_MODE=mistral \
BASE_SEED=0 GPU_GEN=0 GPU_GOLD=1 \
RUN=outputs/exp28_ministral3_8b_seed0 \
bash experiments/exp28_modern_generator_control/run_one.sh
```

`MISTRAL_COMMON_CHAT=1` uses the model's pinned `tekken.json` through
`mistral-common`. `TEXT_ONLY=1` prevents vLLM from initializing an unused
Pixtral image path for this text-only experiment.

Repeat with fresh output directories and `BASE_SEED=1,2`, then analyze:

```bash
python experiments/exp28_modern_generator_control/analyze_runs.py \
  --run outputs/exp28_ministral3_8b_seed0 \
  --run outputs/exp28_ministral3_8b_seed1 \
  --run outputs/exp28_ministral3_8b_seed2 \
  --outdir outputs/exp28_ministral3_8b_analysis
```

Every run preserves generated candidates, independent proxy and silver-gold
scores, scorer logs, model revisions, merge hashes, and audit reports. The
scorer checkpoints are pinned by the shared exp27 runner, and new scorer run
specs record the exact input hash and Python package versions. The evaluation
remains model-based rather than a replacement for human ratings.

Run both pinned generators with three seeds on eight GPUs:

```bash
MODEL_ROOT=inputs/models \
OUT_ROOT=outputs TAG=20260724 \
bash experiments/exp28_modern_generator_control/run_parallel.sh
```

The launcher runs four non-overlapping two-GPU jobs in its first wave and two
jobs in its second wave. Each run writes a sibling `*.driver.log` and refuses
to overwrite an existing output directory or log.
