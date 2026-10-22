# DPO binned-calibration follow-up

This follow-up compares calibration-label-only binned and spline predictors
with A+B on the fixed candidate pool and the same three DPO training seeds.
The fitted calibrators are integrated over evaluation-pool proxy scores, so
the comparison is target-pool-aware rather than strict pre-pool. The current
complete output, including \(A_\nu\), is:

The fixed moment-based A+B/A-only/\(A_\nu\) rows inherit exp09's 99/100
usable-prompt coefficient fit; the one excluded calibration prompt has constant
proxy and silver-target scores across all candidates.

```text
results/actual_dpo/calibration_models/binned
```

## Historical six-checkpoint scope

| Predictor | MAE vs exact frontier | MAE vs realized DPO |
|---|---:|---:|
| Prompt-z-score binned | 0.01667 +/- 0.00142 | **0.01312 +/- 0.00203** |
| Prompt-rank binned | 0.01474 +/- 0.00128 | 0.01505 +/- 0.00216 |
| Prompt-rank spline | 0.01029 +/- 0.00085 | 0.01950 +/- 0.00266 |
| A+B | 0.00686 +/- 0.00065 | 0.02294 +/- 0.00299 |
| Prompt-rank isotonic | 0.00676 +/- 0.00064 | 0.02303 +/- 0.00299 |
| A-only | **0.00207 +/- 0.00065** | 0.02799 +/- 0.00406 |
| Raw spline | 0.00279 +/- 0.00078 | 0.03259 +/- 0.00390 |
| Raw binned | 0.00412 +/- 0.00129 | 0.03391 +/- 0.00445 |
| Zero | 0.06471 +/- 0.00623 | 0.03492 +/- 0.00303 |
| A_nu | 0.00737 +/- 0.00190 | 0.03717 +/- 0.00507 |

The strict `d <= 0.10` scope has the same ordering among A+B, prompt-rank
spline, and the prompt-aware binned variants.

## Interpretation

Prompt-aware binned calibration has the lowest realized-DPO MAE, but it is not
the best model of the exact proxy-tilt frontier. Prompt-z-score and prompt-rank
binned calibration underpredict the frontier at every checkpoint. Their
calibration errors therefore oppose the positive gap between the exact frontier
and realized DPO at every checkpoint.

The mean errors illustrate the cancellation:

```text
prompt-z-score binned:
  prediction - frontier    = -0.01667
  frontier - realized DPO  = +0.02980
  prediction - realized    = +0.01312

A+B:
  MAE vs frontier          = 0.00686
  MAE vs realized DPO      = 0.02294
```

Thus the low realized-DPO error of prompt-aware binning should not be read as
evidence that it better predicts the theoretical local frontier. It partly
models optimizer under-realization through error cancellation. The two targets
answer different questions:

1. A+B is designed to diagnose the idealized local proxy-tilt frontier.
2. Realized-DPO prediction additionally requires a model of how much of that
   frontier the optimizer attains.

All binned fits use only calibration prompts. Hyperparameters are selected by
five-fold prompt-grouped cross-validation, and evaluation gold is used only to
construct the evaluation target.
