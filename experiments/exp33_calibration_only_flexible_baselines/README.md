# exp33: Calibration-Only Flexible Baselines

This experiment makes the information setting explicit. Isotonic, binned,
spline, and MLP models are fitted on calibration `(proxy, gold)` pairs and
integrated over the calibration proxy distribution. Their predicted gain
curves are then transferred unchanged to the held-out evaluation items.

The motivating decision is concrete: suppose 300 rated calibration items are
available now, but the 1,000 candidates that would later be optimized have not
yet been collected. The question is whether to proceed with that future
collection and optimization. At this decision point, neither the future
candidate proxy distribution nor its human labels are available. This
experiment matches that pre-pool information budget.

Unlike the direct baselines in exp21, these predictors never inspect the
evaluation proxy scores. Evaluation proxy and gold values are used only to
construct the realized held-out gain against which the frozen prediction is
scored.

```bash
python experiments/exp33_calibration_only_flexible_baselines/build.py \
  --jobs 32 \
  --outdir experiments/exp33_calibration_only_flexible_baselines/outputs_<timestamp>

python experiments/exp33_calibration_only_flexible_baselines/audit.py \
  --outdir experiments/exp33_calibration_only_flexible_baselines/outputs_<timestamp> \
  --target-aware-reference \
    inputs/exp21_target_aware_output
```

The builder refuses to write into a non-empty output directory.
The target-aware reference directory must contain
`direct_calibration_predictions.csv`; this large row-level source artifact is
not duplicated in the compact package. Packaged exp21 summaries are under
`results/direct_calibration/`, while packaged exp33 summaries and audits are
under `results/direct_calibration_strict/`.

## Data-chain audit

The prepared-CSV-to-result chain passes independent path, split, coefficient,
and exact-gain re-derivation. Source reconstruction passes for SummEval, the
repaired GPT-4o aggregate, OpenMEVA, and the corrected 141-path WMT input.
The submitted 139-path WMT artifact is retained separately for provenance,
but it is not the default input and should not be used for new reporting.

The source-level audit additionally needs the raw SummEval JSONL, GPT scoring
log and item CSV, OpenMEVA ROC/WP JSON, and pinned WMT checkout. These large
upstream assets are not duplicated in this public repository. Pass
their paths explicitly; `python .../source_data_audit.py --help` lists every
argument. The checked canonical report is:

```text
verification/pipeline/source_data_fixed_141/source_data_audit.json
```

To audit the submitted-input lineage, pass
`--wmt-main data/wmt_mqm_long_main_protocol.csv`; that run is expected to flag
the known document-key collision.
