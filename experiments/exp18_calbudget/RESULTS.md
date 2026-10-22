# exp18 — Common-temperature calibration-budget sweep (RESULTS)

> The WMT rows use the corrected document-aware `fixed_141` input. The
> paper-matched `submitted_139` counterpart is retained in the paired WMT result
> tree.

## Design

For every path, one fixed half of the prompts defines the evaluation frontier at
mean prompt-level KL \(d=0.10\). Calibration subsets of
\(n_{\rm cal}\in\{5,10,20,30\}\) prompts are repeatedly sampled from the disjoint
other half. A+B, A-only, and the empirical two-parameter fit receive identical
calibration prompts.

This run imports the common-temperature policy and context-averaged
coefficients from exp14.

## Results

| Dataset | \(n_{\rm cal}\) | Paths | A+B MAE | Empirical-fit MAE | A-only MAE |
|---|---:|---:|---:|---:|---:|
| SummEval | 5 | 32 | 0.0407 | 0.0402 | 0.0518 |
| SummEval | 10 | 32 | 0.0303 | 0.0297 | 0.0412 |
| SummEval | 20 | 32 | 0.0231 | 0.0222 | 0.0355 |
| SummEval | 30 | 32 | 0.0216 | 0.0202 | 0.0345 |
| OpenMEVA | 5 | 10 | 0.0455 | 0.0451 | 0.0430 |
| OpenMEVA | 10 | 10 | 0.0277 | 0.0274 | 0.0276 |
| OpenMEVA | 20 | 10 | 0.0225 | 0.0223 | 0.0219 |
| OpenMEVA | 30 | 10 | 0.0173 | 0.0172 | 0.0173 |
| WMT-MQM | 5 | 141 | 0.2079 | 0.2042 | 0.1979 |
| WMT-MQM | 10 | 141 | 0.1516 | 0.1492 | 0.1526 |
| WMT-MQM | 20 | 141 | 0.1231 | 0.1215 | 0.1266 |
| WMT-MQM | 30 | 132 | 0.1061 | 0.1052 | 0.1153 |

The empirical fit is tied with or slightly better than A+B at every budget.
A+B is substantially better than A-only on SummEval at moderate budgets, but
that ordering is not universal.

## Path-composition audit

Nine WMT paths do not contain 30 calibration prompts after the fixed half split.
The \(n_{\rm cal}=30\) WMT row therefore has 132 paths rather than 141. This is
sample availability, not stochastic non-reproducibility.

`calbudget_common_path_summary.csv` recomputes all budgets on the common set of
132 WMT paths. The corresponding A+B/empirical-fit MAEs are:

- \(n=5\): 0.2106 / 0.2068
- \(n=10\): 0.1532 / 0.1506
- \(n=20\): 0.1232 / 0.1213
- \(n=30\): 0.1061 / 0.1052

The conclusion is unchanged after controlling path composition.

## Validation

`audit_calbudget.py` passes the following checks:

- calibration and evaluation prompts are disjoint;
- every path has one fixed evaluation frontier across budgets and repeats;
- a stored cell is reproduced from raw data to \(10^{-12}\);
- the \(n_{\rm cal}=30\) coefficient approaches the full calibration-pool value;
- the SummEval realized gain agrees with the corrected exp14 run within the
  expected split difference.

## Reproduction

```bash
python experiments/exp18_calbudget/build_calbudget.py \
  --outdir outputs/exp18_calbudget
python experiments/exp18_calbudget/audit_calbudget.py \
  --outdir outputs/exp18_calbudget
```
