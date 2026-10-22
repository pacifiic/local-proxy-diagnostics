# Results

The full run used six independently generated candidate pools: three
Ministral-3 8B seeds and three Llama-3.1 8B seeds. Each pool contains 300
prompts with 32 candidates per prompt. For each of 20 deterministic repeats,
10 candidates per prompt calibrated the predictors and 22 disjoint candidates
evaluated them.

The primary metric is aggregate gain-curve MAE over
`d = {0.01, 0.025, 0.05, 0.075, 0.10}`.

| Model | A+B | A+B, shrunk B | A-only | A_nu | Empirical curve | Isotonic | Binned | Spline | MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1 8B | 0.00586 | 0.00567 | 0.00563 | **0.00413** | 0.00605 | 0.02881 | 0.01300 | 0.01876 | 0.01972 |
| Ministral-3 8B | 0.00463 | 0.00462 | 0.00463 | 0.01166 | 0.00475 | 0.01030 | **0.00320** | 0.00412 | 0.00490 |

These results clarify, but do not reverse, the modern-generator finding.
Relative to the earlier unseen-prompt split, A+B curve MAE falls from 0.01805
to 0.00586 for Llama and from 0.00967 to 0.00463 for Ministral. Thus a
substantial part of the earlier error was prompt-distribution transfer rather
than failure of the local tilt approximation.

The curvature term is not reliably beneficial with only 10 calibration
candidates per prompt. A+B is essentially tied with A-only for Ministral and
slightly worse for Llama. Calibration-only uncertainty shrinkage moves A+B
toward A-only but does not create a consistent advantage. This matches the
diagnostic: mean B is near zero, while prompt-level bootstrap uncertainty is
much larger.

The black-box comparison is model-dependent. A+B strongly outperforms all four
direct calibration models for Llama. For Ministral, binned calibration is best
and spline is also slightly better than A+B. A_nu reverses similarly: it is
best for Llama and poor for Ministral. There is therefore no defensible claim
that one estimator uniformly dominates across modern generators.

The supported conclusion is narrow: the same-prompt protocol
removes the strongest cross-prompt mismatch and shows that the analytic law is
competitive with substantially more flexible calibrators, but these 32-way
silver candidate pools do not establish an incremental benefit from B over
A-only.
