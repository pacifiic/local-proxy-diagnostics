# Saved Results

`results/` contains the reference outputs used to build the publication tables
and figures. Each major analysis stores predictions or path-level rows together
with summaries and a run specification.

- `results/07_prospective_prediction/`: prospective prediction across datasets.
- `results/direct_calibration_strict/` and `results/evaluation_aware_ab/`:
  calibration-only and target-pool-aware comparisons.
- `results/actual_dpo/`: three-seed DPO summaries and fixed-candidate evaluation.
- `results/bon_scope/` and `results/prompt_bon/`: selection-rule scope audits.
- `results/reward_model_coefficients/`: three-reward-model coefficient example.
- `results/cubic_moment_term/` and `results/dominant_local_law/`: higher-order
  and flexible-functional-form checks, including selected detailed outputs.

The published WMT estimates use the corrected 141-path input. Results based on
the earlier 139-path input are retained only where explicitly labeled as a
lineage control. Use a new output directory for new runs.
