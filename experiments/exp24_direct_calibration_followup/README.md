# exp24: Direct-Calibration Follow-up

This experiment tests whether the exp21 comparison changes with calibration
budget, larger KL radius, or cross-dataset transfer. It does not overwrite
exp21 outputs.

Within-dataset run:

```bash
python experiments/exp24_direct_calibration_followup/build_within_dataset.py \
  --jobs 32 \
  --outdir experiments/exp24_direct_calibration_followup/outputs/within_dataset
python experiments/exp24_direct_calibration_followup/audit_within_dataset.py \
  --outdir experiments/exp24_direct_calibration_followup/outputs/within_dataset
```

The calibration fractions are 5%, 10%, 20%, 30%, and 50%, with 50 repeated
item splits per path. The local grid is
`d={.01,.025,.05,.075,.10}`. Values at `d={.15,.20}` are a scope audit and
must not be presented as a theorem-window guarantee.

Cross-dataset run:

```bash
python experiments/exp24_direct_calibration_followup/build_ood_transfer.py \
  --jobs 32 \
  --outdir experiments/exp24_direct_calibration_followup/outputs/ood_transfer
python experiments/exp24_direct_calibration_followup/audit_ood_transfer.py \
  --outdir experiments/exp24_direct_calibration_followup/outputs/ood_transfer
```

The OOD comparison follows a `z_cal` protocol. Source A/B coefficients and
source one-dimensional calibration functions are estimated from source-only
calibration splits. Target proxy and gold units are standardized using target
calibration statistics. The reported errors are therefore standardized-gain
errors. `target_weight` is applied identically to coefficients and functions:
0 is source-only, 1 is target-only, and 0.75 is the fixed target-heavy blend.
