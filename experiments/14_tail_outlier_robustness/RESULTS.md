# Tail and Outlier Robustness Results

All three commands in `run_all.sh` completed with no skipped rows:

| Dataset | Paths | Output rows |
|---|---:|---:|
| SummEval | 32 | 1,408 |
| OpenMEVA-MAGS | 10 | 440 |
| WMT MQM `fixed_141` | 141 | 6,204 |

The SummEval and OpenMEVA summaries are byte-identical to the earlier canonical
audit. The WMT summary is byte-identical to the corrected document-aware
141-path audit.

## Proxy-Outlier Condition

| Dataset | corr(B, original B) | B-sign agreement | Median absolute endpoint-gain shift |
|---|---:|---:|---:|
| SummEval | 0.940 | 100.0% | 0.0915 |
| OpenMEVA-MAGS | -0.107 | 20.0% | 0.1518 |
| WMT MQM `fixed_141` | 0.773 | 95.0% | 0.3516 |

The controlled proxy-outlier condition can destabilize `B`, most clearly on
OpenMEVA. This is a sensitivity result, not evidence that the estimator is
robust to arbitrary heavy-tailed distributions.

Gold-side outliers are milder in the same audit:

| Dataset | corr(B, original B) | B-sign agreement | Median absolute endpoint-gain shift |
|---|---:|---:|---:|
| SummEval | 0.997 | 98.6% | 0.0046 |
| OpenMEVA-MAGS | 0.955 | 93.0% | 0.0070 |
| WMT MQM `fixed_141` | 0.979 | 97.5% | 0.0219 |

Use the full summary CSVs when discussing winsorization or rank normalization.
Coefficient correlation, coefficient-sign agreement, and endpoint-gain shift
are distinct estimands and should not be conflated.

