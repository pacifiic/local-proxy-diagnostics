# exp27 Results: Cubic Moment Term

The experiment adds the calibration-only fourth-moment coefficient `C` to the
existing local law. It uses the exact exp21 outer splits and evaluation
protocol.

## CurveMAE

| Dataset | A+B | A+B+C | Isotonic | Spline |
|---|---:|---:|---:|---:|
| SummEval human-heldout | 0.011850 | 0.011701 | 0.009086 | 0.010274 |
| SummEval GPT-4o | 0.008059 | 0.008027 | 0.008109 | 0.007768 |
| WMT MQM human-heldout | 0.059206 | 0.058468 | 0.052110 | 0.056987 |
| OpenMEVA-MAGS human-heldout | 0.009906 | 0.009838 | 0.007058 | 0.007277 |

The SummEval row uses the same 7,989-cell common support as the published comparison
comparison. `A+B+C` improves the point estimate over `A+B` on all four
datasets. Its common-support SummEval difference is `-0.000149` (path-clustered
95% CI `[-0.000376, 0.000006]`). The interval excludes zero only for WMT
(`A+B+C - A+B = -0.000738`, CI `[-0.000942, -0.000541]`). The other three
improvements are not statistically resolved.

The cubic term does not overturn the direct-calibration result. Isotonic and
spline remain better on the three human-proxy datasets. On GPT-4o, `A+B+C`
is numerically better than isotonic and worse than spline, with both paired
intervals containing zero.

## Stability

The split-level standard deviation of `C` is 0.036 for GPT-4o, 0.024 for
OpenMEVA, 0.129 for SummEval human-heldout, and 0.480 for WMT. WMT has a
maximum absolute split estimate of 5.34. This is consistent with the expected
variance cost of estimating fourth-order moments.

## Interpretation

The result supports the mathematical expansion: adding the next moment term
slightly reduces finite-radius bias. It does not support presenting the cubic
law as a remedy for the stronger IID isotonic/spline baselines. The two-term
law remains the cleaner bias-variance and interpretability tradeoff.

The completed outputs are under
`results/cubic_moment_term`. An independent repeat
under `exp27_repeat` produced byte-identical CSV files.
