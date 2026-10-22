# Coefficient Precision Audit

This read-only audit produces compact coefficient and local-window diagnostics:

1. how Grid IAE and local-grid MAE differ;
2. calibration-size error in A and B, including the scaled coefficient-error
   terms and evaluation remainder;
3. curvature contribution at `d=0.05` and `d=0.10`;
4. a compact DPO table restricted to the paper's `d <= 0.10` local window.

The coefficient analysis uses calibration fractions `0.05, 0.10, 0.30`, 50
nested split repeats, and radii `0.05, 0.10`. It reuses the audited exp21 path
construction and moment implementation. Existing experiment outputs are never
modified.

```bash
python experiments/exp31_coefficient_precision_audit/run_precision_audit.py \
  --paper-human-d-sweep path/to/d_sweep_human/d_sweep_paths_long.csv \
  --paper-gpt-d-sweep path/to/d_sweep_gpt4o/d_sweep_paths_long.csv \
  --direct-predictions path/to/direct_calibration_predictions.csv \
  --dpo-summary path/to/dpo_signed_bias_summary.csv \
  --exp24-split-metrics path/to/within_split_metrics.csv \
  --outdir path/to/fresh_output
```

The script requires the prior exp24 split-level output and verifies an exact
one-to-one match for every A+B actual gain, prediction, and absolute error used
in the coefficient analysis.
