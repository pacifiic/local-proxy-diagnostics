# Comparison Diagnostics

This analysis evaluates three properties of the saved comparisons:

1. whether lower realized-DPO error is accompanied by signed frontier
   underprediction that opposes the optimizer-realization gap;
2. how closely the closed-form A+B predictor matches a freely fitted
   `a*sqrt(d) + b*d` curve;
3. whether replacing B with zero when its calibration-only confidence interval
   crosses zero improves prediction error.

The script only reads completed experiment outputs and refuses to overwrite a
non-empty output directory.

```bash
python experiments/exp30_positioning_audit/run_positioning_audit.py \
  --dpo-checkpoints path/to/exp28/checkpoint_predictions.csv \
  --dpo-binned-checkpoints path/to/binned/checkpoint_predictions.csv \
  --dpo-fit-metadata path/to/exp28/model_fit_metadata.csv \
  --dpo-binned-fit-metadata path/to/binned/model_fit_metadata.csv \
  --direct-predictions \
    path/to/summeval_gpt4o/direct_calibration_predictions.csv \
    path/to/wmt_openmeva/direct_calibration_predictions.csv \
  --b-stability-rows path/to/b_stability_rows.csv \
  --bootstrap-reps 10000 \
  --random-seed 0 \
  --outdir path/to/fresh_output
```

The B-gate intervals use a path-clustered bootstrap. The DPO report separates
the pre-specified `d <= 0.10` local window from the historical six-checkpoint
stress-test scope.

The report also records model complexity. For closed-form predictors it reports
the number of coefficients. For binned and spline models it reports fitted
representation size, hyperparameter-grid size, and cross-validation scheme.
For isotonic regression it reports the data-adaptive number of fitted
thresholds. Prompt-coordinate variants additionally state which prompt-level
information they use.

The direct-calibration inputs used here are the target-pool-aware exp21
outputs. They are appropriate after a deployment pool has been proxy-scored,
not for the 300-rated-items to future-1,000-candidates pre-pool decision.
Exp33 provides the information-matched comparison; see
`../exp33_calibration_only_flexible_baselines/RESULTS.md`.

The published summaries include MLP baselines and model-complexity accounting.
Run this script into a fresh output directory to reproduce the diagnostics.
