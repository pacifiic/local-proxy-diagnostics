# Target-Pool-Aware Conditional-Mean A+B Results

This directory packages the canonical Exp32 results. Conditional-mean models
are fitted using calibration labels only. They are then evaluated on the
unlabeled target-pool proxy scores. No target-pool gold label enters any
prediction.

The target-aware projection computes the moments of
`(h(r_target), r_target)` and predicts

```text
Delta_hat(d) = A_h sqrt(d) + B_h d.
```

The direct comparator integrates the same `h(r_target)` exactly under the
target-pool proxy tilt. Their difference therefore measures the truncation
introduced by the two-term local projection after matching the conditional
mean and target-pool proxy geometry.

## Canonical CurveMAE

| Dataset | Raw A+B | EA iso. | Direct iso. | EA spline | Direct spline | EA MLP | Direct MLP | EA binned | Direct binned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SummEval human | 0.011850 | 0.009588 | 0.009086 | 0.010402 | 0.010274 | 0.009724 | 0.009218 | 0.027825 | 0.029140 |
| GPT-4o proxy | 0.008059 | 0.008009 | 0.008109 | 0.007800 | 0.007768 | 0.007929 | 0.007886 | 0.008604 | 0.008646 |
| WMT MQM (`fixed_141`) | 0.059206 | 0.053158 | 0.052110 | 0.056674 | 0.056987 | 0.052863 | 0.052150 | 0.065574 | 0.068429 |
| OpenMEVA-MAGS | 0.009906 | 0.007119 | 0.007058 | 0.007257 | 0.007277 | 0.006954 | 0.006929 | 0.013296 | 0.014286 |

`EA` denotes evaluation-aware, or target-pool-aware, A+B. This is a stronger
information setting than the paper's original calibration-moment estimator.
It is appropriate only after the deployment candidate pool has been collected
and proxy-scored.

## Packaged Files

- `common_support/` contains the canonical comparison-table summary
  generated after applying the 11-cell SummEval exclusion mask shared by all
  displayed calibration comparators.
- `evaluation_aware_ab_summary_fixed_141.csv` and
  `evaluation_aware_ab_paired_bootstrap_fixed_141.csv` are the combined
  four-dataset all-support tables retained for provenance.
- `non_wmt_canonical/` contains SummEval human, GPT-4o, and OpenMEVA rows from
  the byte-identical July 25 canonical runs. The source run also contained the
  submitted-input WMT lineage, which is deliberately filtered out here.
- `wmt_fixed_141/` contains the corrected document-aware WMT result.
- `wmt_fixed_141_repeat/` contains an independent WMT-only
  50-repeat reproduction. It has 564,000 prediction rows
  (141 paths x 50 splits x 5 radii x 16 methods), no skipped records, and
  the same uncompressed prediction SHA-256 as the original source run.
  Its summary, paired bootstrap table, run spec, and results Markdown are
  byte-identical to the canonical files.
- `direct_calibration_predictions.csv.gz` files preserve every
  path/repeat/radius prediction. They use deterministic gzip metadata.
- `manifest.json` records source and packaged SHA-256 hashes.
- `reproduction_verification.json` records the repeat comparison. It can be
  regenerated with
  `experiments/exp32_evaluation_aware_ab/verify_fixed_141_repeat.py`.

The submitted-input `submitted_139` WMT output remains available only in the
lineage-preservation tree and must not be used for new scientific claims.
