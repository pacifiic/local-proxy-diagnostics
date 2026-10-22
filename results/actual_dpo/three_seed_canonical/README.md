# Canonical Three-Seed DPO Results

The root result files preserve the original median-KL analysis produced by the
prompt-conditional calibration builder. The reported analysis uses a subsequent
checkpoint-mean-prompt-KL gate at 0.10. In this canonical run, the two rules
select exactly the same ten seed-checkpoints, so the reported gains and MAEs do
not change.

The final gate definition, selected checkpoint list, and the distribution of
the 2,000 prompt-checkpoint radii are recorded under
`prompt_radius_audit/`. Reproduce that audit with:

```bash
python experiments/exp09_dpo_validation/21_prompt_radius_audit.py \
  --checkpoint-predictions \
    results/actual_dpo/three_seed_canonical/checkpoint_predictions.csv \
  --per-prompt-root results/actual_dpo/training_runs \
  --outdir outputs/prompt_radius_audit
```

The command requires the retained per-prompt outputs from the three-seed DPO
run. The compact `prompt_radius_audit/` directory preserves the resulting
per-cell radii, summary, and run specification.
