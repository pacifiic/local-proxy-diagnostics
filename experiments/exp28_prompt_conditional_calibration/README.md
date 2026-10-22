# exp28: Prompt-Conditional Calibration Diagnosis

This experiment explains why raw pooled isotonic/spline calibration behaves
differently in exact-tilt benchmarks and the DPO validation.

It compares raw, prompt-centered, prompt-z-scored, and within-prompt rank proxy
coordinates for isotonic and spline calibration. All main predictors are fit
only on the fixed calibration prompts. Evaluation-prompt cross-fit variants are
descriptive distribution-shift diagnostics and are labeled separately.
Spline hyperparameters are selected by five-fold grouped cross-validation, with
whole prompts kept together.

Two targets are reported:

1. the exact prompt-conditional exponential-tilt frontier;
2. realized held-out DPO reweight gain.

The output separately reports the strict theorem-facing window
(`checkpoint median d <= 0.10`) and the historical six-checkpoint DPO scope.
It also decomposes prediction error into calibration error relative to the
exact prompt-tilt frontier and the gap between that frontier and realized DPO.

Run after the fixed-pool three-seed DPO sweep:

```bash
MULTISEED_ROOT=results/actual_dpo/training_runs \
OUTDIR=outputs/prompt_conditional_calibration \
  bash experiments/exp28_prompt_conditional_calibration/run.sh
```

## Rank A+B follow-up

`rank_ab_followup.py` asks whether the coefficient law also benefits from the
within-prompt rank coordinate. Raw A+B is checked against the raw-score tilt
frontier. Rank A+B is checked against a separate rank-score tilt frontier;
these are different optimization paths and must not be treated as the same
frontier. Both predictions are additionally compared with realized DPO gain.

```bash
python experiments/exp28_prompt_conditional_calibration/rank_ab_followup.py \
  --pool-scored data/dpo_candidate_pool/pool_scored.jsonl \
  --multiseed-root results/actual_dpo/training_runs \
  --seeds 0 1 2 \
  --outdir outputs/rank_ab_followup
```

The script refuses to overwrite a non-empty output directory, verifies the
locked raw coefficients and raw frontiers, and records strict `d <= 0.10` and
historical six-checkpoint summaries separately.

## Binned calibration follow-up

`binned_followup.py` adds equal-frequency binned calibration with shrinkage to
the same three-seed DPO audit. It fits raw, prompt-centered, prompt-z-scored,
and within-prompt rank variants using only labels from the 100 calibration
prompts. Bin count and shrinkage are selected by five-fold `GroupKFold`, so
candidates from one prompt never appear in both training and validation folds.
The fitted maps are then integrated over evaluation-pool proxy scores, so this
is a target-pool-aware comparison rather than the strict pre-pool comparison
implemented by exp34.

The fixed moment-based A+B/A-only/A_nu comparison rows come from exp09, whose
coefficient fit uses 99/100 prompts after excluding one prompt with constant
proxy and silver-target scores.

The command requires the completed exp28 output only to assemble a comparison
table. It does not modify that output.

```bash
python experiments/exp28_prompt_conditional_calibration/binned_followup.py \
  --pool-scored data/dpo_candidate_pool/pool_scored.jsonl \
  --multiseed-root results/actual_dpo/training_runs \
  --existing-exp28-dir results/prompt_conditional_calibration \
  --seeds 0 1 2 \
  --outdir outputs/binned_followup
```

The script refuses to overwrite a non-empty directory, re-derives every saved
exact frontier, avoids evaluation-gold access in proxy transformations, and
records input and code hashes in `run_spec.json`.

## Target-aware A+B versus direct integration

`target_aware_ab_followup.py` gives the prompt-specific coefficient projection
the same fitted conditional mean and evaluation-pool proxy geometry as each
direct calibrator. For every raw, centered, z-scored, and prompt-rank
isotonic/spline/binned model, it compares
`A_h sqrt(d_x) + B_h d_x` with exact integration of the same fitted
`h(r_eval)` under the raw-proxy tilt.

The script locks the hyperparameters selected by the saved canonical direct
runs, verifies that its direct checkpoint predictions reproduce those outputs,
re-derives every idealized frontier, and preserves prompt-level predictions.
Evaluation silver labels are used only to score the frozen predictions.

```bash
python experiments/exp28_prompt_conditional_calibration/target_aware_ab_followup.py \
  --outdir results/actual_dpo/<fresh-target-aware-output>
```

The canonical audited output is
`results/actual_dpo/target_pool_aware_ab_direct/`.
