# exp27: Qwen Size Control

This experiment diagnoses whether the previously observed Qwen/Zephyr
difference can be explained by nominal parameter count alone.

- `Qwen2.5-3B` versus `Qwen2.5-7B` changes size within one model family.
- `Qwen2.5-7B` versus `Zephyr-7B` approximately matches nominal size while
  changing the model family and post-training recipe.

All three pools use the same 100 calibration and 200 evaluation UltraFeedback
prompts, 32 sampled responses per prompt, sampling settings, proxy reward
model, and model-based silver-gold scorer. This is a descriptive control with
one checkpoint per cell, not a causal decomposition of size and training.

## Run

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
RUN=outputs/exp27_qwen7_deterministic_20260724 \
GPU_GEN=0 GPU_GOLD=1 \
bash experiments/exp27_qwen_size_control/run_model_pool.sh

QWEN7_RUN=outputs/exp27_qwen7_deterministic_20260724 \
OUTDIR=outputs/exp27_qwen_size_analysis_20260724 \
bash experiments/exp27_qwen_size_control/run_analysis.sh
```

The generation runner retains the generated pool, both independently scored
pools, logs, run specifications, strict merge audit, and merged scored pool.
It also checks model IDs, sampling settings, score finiteness, prompt/candidate
keys, text identity across scoring stages, and non-duplication against the
Qwen-3B and Zephyr-7B pools. It refuses to overwrite a non-empty output
directory. The default proxy and silver-gold checkpoints are pinned to commits
`d4117fbfd81b72f41b96341238baa1e3e90a4ce1` and
`c355404efa9ad2ad069f3a197cae0523c14244fc`, respectively. Scorer run specs
record input/output hashes, package versions, maximum sequence length, and
whether user-site Python packages were disabled.

For byte-level regeneration, use `run_model_pool.sh` for every compared model
and pass those deterministic scored pools to the analysis. Its generator uses
one `n=1` request per candidate and a SHA-256 seed derived from the split,
prompt ID, and candidate index. A smoke rerun is byte-identical on the same
fixed B200 device and software stack. Different GPU devices can still diverge
at sampling boundaries because the underlying GPU inference kernels are not
guaranteed bitwise deterministic; the model revision, environment, and GPU
device should therefore be recorded for strict byte-level comparison.

The analysis reports coefficient geometry, score and response diagnostics,
finite-pool reachability, and all exp25 gain-curve baselines. The “gold” score
is from `OpenAssistant/reward-model-deberta-v3-large-v2`; no new human ratings
are used.
