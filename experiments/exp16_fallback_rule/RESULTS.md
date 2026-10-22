# exp16 - Calibration-Only Curvature Fallback

This experiment asks whether a calibration-only uncertainty rule can decide
when to use `A+B` and when to fall back to `A-only`.

## Rule

For each calibration split, prompts are bootstrapped 1,000 times. The rule
uses `A-only` when the 90% bootstrap interval for `B` contains zero, and
`A+B` otherwise. Both the point estimate and bootstrap now use the same
common-temperature, mean-prompt-KL coefficient definition as
`exp14_conditional_tilt`.

## Results

| Dataset | Paths | Use A+B | A+B MAE | A-only MAE | Rule MAE | Ex-post ideal |
|---|---:|---:|---:|---:|---:|---:|
| SummEval | 32 | 0.433 | 0.0191 | 0.0308 | 0.0195 | 0.0169 |
| WMT MQM | 141 | 0.303 | 0.1041 | 0.1041 | 0.1030 | 0.0851 |
| OpenMEVA | 10 | 0.650 | 0.0151 | 0.0158 | 0.0152 | 0.0119 |
| Pooled | 183 | - | 0.0843 | 0.0864 | 0.0836 | 0.0692 |

The rule selects the repeat-level ex-post better fixed predictor 53% of the
time. It is worse than both fixed choices on 42 of 183 path-level aggregates.
It should therefore be interpreted as a weak uncertainty flag, not as a
reliably superior selector.

The WMT row uses the corrected document-aware `fixed_141` input inherited
from exp14. The submitted paper instead used `submitted_139`. Compact artifacts
for both lineages are available under `results/wmt_input_lineages/` and must
remain explicitly labeled.

## Reproducibility

Two independent source runs were byte-identical. The compact package retains
the verified output at:

- a fresh user-selected output directory

The duplicate raw run is omitted. The retained output records the exact-valid
bootstrap accounting, and the core audit verifies equality with the prior
canonical rows.
