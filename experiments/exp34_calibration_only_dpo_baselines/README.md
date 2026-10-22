# exp34: Calibration-Only DPO Baselines

This experiment gives the DPO comparison the same pre-pool information budget
as `A+B`. The motivating decision is made from 100 rated calibration prompts
before the 200 future evaluation prompts and their candidate responses have
been collected.

The fixed moment-based rows inherited from exp09 use 99/100 prompts because
one calibration prompt has constant proxy and silver-target scores across all
32 candidates. Flexible calibrators use only the same 100-prompt calibration
pool under their declared fitting rules.

Isotonic, binned, spline, and fixed 1-16-16-1 MLP calibrators are fitted on
the 3,200 calibration candidates. Each model is integrated over the
calibration prompt pools to form a frozen gain curve. Evaluation prediction
uses only that curve and the KL radius reached by DPO. Evaluation candidate
proxy scores are not used by these predictors.

The audit is conditional on the measured radius `d`, as is the submitted
coefficient-law comparison. Predicting the radius reached by an optimizer
from DPO hyperparameters is a separate problem.

```bash
python experiments/exp34_calibration_only_dpo_baselines/build.py \
  --pool-scored inputs/pool_scored.jsonl \
  --multiseed-root inputs/dpo_multiseed \
  --outdir experiments/exp34_calibration_only_dpo_baselines/outputs_<timestamp>

python experiments/exp34_calibration_only_dpo_baselines/audit.py \
  --pool-scored inputs/pool_scored.jsonl \
  --multiseed-root inputs/dpo_multiseed \
  --outdir experiments/exp34_calibration_only_dpo_baselines/outputs_<timestamp>

python experiments/exp34_calibration_only_dpo_baselines/compare_target_aware.py \
  --outdir experiments/exp34_calibration_only_dpo_baselines/outputs_<timestamp> \
  --target-aware-primary inputs/exp28/checkpoint_predictions.csv \
  --target-aware-binned inputs/binned/checkpoint_predictions.csv
```

The builder refuses to overwrite an existing output directory.
