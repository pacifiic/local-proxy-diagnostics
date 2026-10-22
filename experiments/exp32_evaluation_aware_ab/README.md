# Evaluation-aware Conditional-Mean A+B

This experiment fits each one-dimensional conditional-mean model using only
calibration labels. It evaluates that model on the unlabeled evaluation proxy
scores and computes the `A,B` moments of `(h(r_eval), r_eval)`.

This is explicitly a **later-stage target-pool-aware diagnostic**. It assumes
the deployment candidate pool has already been collected and proxy-scored.
It is not available in the pre-pool example with 300 rated calibration items
and 1,000 future candidates that do not yet exist. Exp33 is the fair
comparison for that earlier decision.

Run from the repository root into a fresh output directory:

```bash
python experiments/exp32_evaluation_aware_ab/build_evaluation_aware_ab.py \
  --jobs 32 \
  --outdir experiments/exp32_evaluation_aware_ab/outputs_<timestamp>
python experiments/exp32_evaluation_aware_ab/audit_evaluation_aware_ab.py \
  --outdir experiments/exp32_evaluation_aware_ab/outputs_<timestamp> \
  --support-exclusions \
    results/direct_calibration_strict/main_protocol/calibration_only_skipped.csv
```

The direct comparator integrates the same fitted `h(r_eval)` under the exact
evaluation proxy tilt. Evaluation gold labels are used only to calculate final
errors. The support-exclusion file makes every displayed method use the same
path/repeat/radius cells. It removes 11 SummEval `d=0.10` cells at which at
least one calibration-only comparator cannot reach the requested radius.

## Canonical 50-repeat result

The July 25 canonical run used the same 30/70 split plans as Exp. 21. The
non-WMT datasets come from the byte-identical canonical runs. The WMT row
comes from the paired rerun using the corrected document-aware `fixed_141`
input. For isotonic `h`, the local-grid CurveMAE results were:

| Dataset | Raw A+B | Evaluation-aware A+B | Direct isotonic |
|---|---:|---:|---:|
| SummEval human proxy | 0.011850 | 0.009588 | 0.009086 |
| SummEval GPT-4o proxy | 0.008059 | 0.008009 | 0.008109 |
| WMT MQM human proxy (`fixed_141`) | 0.059206 | 0.053158 | 0.052110 |
| OpenMEVA-MAGS human proxy | 0.009906 | 0.007119 | 0.007058 |

Thus, the evaluation-aware projection closes most of the gap between raw
calibration-moment A+B and direct isotonic integration on the human-proxy
slices. It is effectively tied on GPT-4o. This is a stronger information
setting than the paper's raw A+B estimator because it uses the unlabeled
evaluation proxy distribution, although it never uses evaluation gold labels.

Full endpoint results, binned/spline/MLP variants, path-clustered paired
intervals, and compressed path/repeat/radius predictions are packaged under
`results/evaluation_aware_ab/`. The submitted-input `submitted_139` WMT result is
retained only for provenance and is not used in the table above.

The original all-support Exp32 summary is retained for provenance. The common-support comparison uses the common-support output under
`results/evaluation_aware_ab/common_support/`: 7,989 of 8,000
SummEval path/repeat/radius cells are shared by all displayed calibration
comparators.

## Independent corrected-WMT reproduction

The corrected WMT run was independently repeated with:

```bash
python experiments/exp32_evaluation_aware_ab/build_evaluation_aware_ab.py \
  --datasets wmt \
  --jobs 40 \
  --outdir <fresh-output-directory>
python experiments/exp32_evaluation_aware_ab/audit_evaluation_aware_ab.py \
  --outdir <fresh-output-directory>
python experiments/exp32_evaluation_aware_ab/verify_fixed_141_repeat.py \
  --repeat-dir <fresh-output-directory>
```

The repeat produced 564,000 rows from 141 paths, 50 splits, five radii, and
16 methods, with no skipped records. The raw prediction CSV matched the
original source run byte-for-byte (SHA-256
`3ec6becf431a7eed607410c7cf44497f007fdf4edf40929d91480efe623b7bb1`).
The summary, paired bootstrap interval table, run spec, and results Markdown
were also byte-identical. The packaged repeat and machine-readable verification
report are under
`results/evaluation_aware_ab/wmt_fixed_141_repeat/`.

This check also caught a packaging drift in which Exp32 still requested the
MLP projection while its Exp21 dependency lacked the MLP fitter. The
pre-specified 1-16-16-1 implementation used by the canonical run was restored.
The complete rerun then reproduced the canonical values exactly.
