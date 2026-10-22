# Experiment 16: Prompt-Split LLM-Style BoN/Reranking Workflow

This experiment is a lightweight test bed for an LLM candidate-selection
workflow. It does not train a policy model. Instead, it treats each prompt's
existing candidate outputs as a candidate pool, applies exact prompt-level
Best-of-N selection under a fixed proxy, and evaluates the resulting held-out
gold gain on disjoint evaluation prompts.

The goal is to test the diagnostic in a setting closer to practical LLM
reranking while keeping the experiment reproducible and cheap. It is therefore
a pre-RLHF/reranking audit, not a full PPO/DPO/RLHF run.

## What It Tests

For each admissible proxy path:

1. Split prompts into calibration and evaluation prompts.
2. Estimate `A`, `B`, and `A_nu` on calibration prompts only.
3. On evaluation prompts, compute exact with-replacement Best-of-N selection
   probabilities over the prompt's candidate pool.
4. Measure the realized held-out gold gain.
5. Compare the realized gain against `A+B`, `A-only`, `A_nu`, and zero-gain
   predictions at the prompt-level BoN KL radii.

This directly asks whether coefficients estimated before selection can predict
the gain induced by a fixed proxy in a prompt-conditioned candidate-selection
workflow.

## SummEval Human-Heldout Proxy Example

```bash
cd <repo-root>

OUT_ROOT=outputs/llm_reranking_$(date +%Y%m%d_%H%M%S)

python experiments/16_llm_reranking_workflow/01_prompt_split_bon_reranking.py \
  --dataset-name summeval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --prompt-col-indices 0 \
  --group-cols rater_type,dimension \
  --n-list 1,2,4,8,16,32 \
  --calibration-frac 0.30 \
  --n-repeats 50 \
  --outdir "$OUT_ROOT/summeval_human_proxy"
```

## SummEval GPT-4o External Proxy Example

```bash
cd <repo-root>

OUT_ROOT=outputs/llm_reranking_$(date +%Y%m%d_%H%M%S)

python experiments/16_llm_reranking_workflow/01_prompt_split_bon_reranking.py \
  --dataset-name summeval_gpt4o \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --prompt-col-indices 0 \
  --group-cols dimension \
  --proxy-csv data/model_proxy_4o_aggregated.csv \
  --proxy-item-col item_id \
  --proxy-score-col proxy_score_mean \
  --n-list 1,2,4,8,16,32 \
  --calibration-frac 0.30 \
  --n-repeats 50 \
  --outdir "$OUT_ROOT/summeval_gpt4o_proxy"
```

## OpenMEVA-MAGS ROC/WP Example

OpenMEVA item ids use `dataset||story_id||model_id`, so the prompt is the
`dataset,story_id` pair.

```bash
cd <repo-root>

OUT_ROOT=outputs/llm_reranking_$(date +%Y%m%d_%H%M%S)

python experiments/16_llm_reranking_workflow/01_prompt_split_bon_reranking.py \
  --dataset-name openmeva_mags \
  --human-long-csv data/openmeva_long.csv \
  --item-cols dataset,story_id,model_id \
  --prompt-col-indices 0,1 \
  --group-cols dataset \
  --min-items 100 \
  --n-list 1,2,4,8,16,32 \
  --calibration-frac 0.30 \
  --n-repeats 50 \
  --outdir "$OUT_ROOT/openmeva_human_proxy"
```

## Outputs

The script refuses to overwrite a non-empty output directory unless
`--overwrite` is passed.

Generated files:

- `prompt_split_bon_reranking_run_spec.json`
- `prompt_split_bon_reranking_predictions.csv`
- `prompt_split_bon_reranking_summary.csv`
- `prompt_split_bon_reranking_summary.md`
- `prompt_split_bon_reranking_mae_wide.csv`
- `prompt_split_bon_reranking_mae_wide.md`
- `prompt_split_bon_reranking_skipped.csv`

## Notes

- This experiment uses existing candidate pools and repeated-rating gold
  scores. It does not require GPUs.
- A later full LLM generation experiment can reuse the same evaluation design
  if generated candidates are paired with a reliable held-out gold or judge
  target. Without that target, a full RLHF run would not be directly comparable
  to the paper's repeated-rating diagnostic.
