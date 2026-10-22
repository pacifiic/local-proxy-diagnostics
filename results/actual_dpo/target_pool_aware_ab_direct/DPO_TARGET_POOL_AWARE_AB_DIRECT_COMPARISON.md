# DPO Target-Pool-Aware A+B and Direct Calibration

All conditional-mean models are fit on the 100 calibration prompts only.
For each of 200 evaluation prompts, the frozen model is evaluated on
that prompt's proxy scores. `Target-aware A+B` computes prompt-specific
moments of `(h(r_eval), r_eval)` and predicts
`A_h sqrt(d_x) + B_h d_x`; `Direct` exactly integrates the same
`h(r_eval)` under the raw-proxy tilt at the same `d_x`. Evaluation
silver scores are used only to compute the idealized frontier and the
DPO-implied fixed-pool gain after predictions are frozen.

The ten checkpoints are selected by checkpoint mean prompt KL <= 0.10.
This is a mean-KL gate, not a guarantee that every prompt is local.

## Original Paper Baselines

| Predictor | Idealized-frontier MAE | DPO-implied-gain MAE |
|---|---:|---:|
| A+B | 0.00395 +/- 0.00047 | 0.01721 +/- 0.00116 |
| A-only | 0.00210 +/- 0.00053 | 0.01905 +/- 0.00120 |
| A_nu | 0.00371 +/- 0.00061 | 0.02487 +/- 0.00129 |
| Zero | 0.04197 +/- 0.00017 | 0.02081 +/- 0.00062 |

## Target-Pool-Aware Projection vs. Direct Integration

Each cell is mean MAE +/- sample SD across the three DPO training seeds.

| Conditional-mean model | Target-aware A+B vs frontier | Direct vs frontier | Target-aware A+B vs DPO-implied gain | Direct vs DPO-implied gain | Projection-direct gap |
|---|---:|---:|---:|---:|---:|
| centered binned | 0.00559 +/- 0.00041 | 0.00573 +/- 0.00040 | 0.01557 +/- 0.00108 | 0.01543 +/- 0.00108 | 0.00014 +/- 0.00001 |
| centered isotonic | 0.00063 +/- 0.00045 | 0.00059 +/- 0.00039 | 0.02179 +/- 0.00112 | 0.02170 +/- 0.00112 | 0.00008 +/- 0.00001 |
| centered spline | 0.00264 +/- 0.00044 | 0.00274 +/- 0.00043 | 0.01852 +/- 0.00111 | 0.01842 +/- 0.00111 | 0.00010 +/- 0.00001 |
| rank binned | 0.00878 +/- 0.00042 | 0.00903 +/- 0.00041 | 0.01238 +/- 0.00111 | 0.01213 +/- 0.00110 | 0.00024 +/- 0.00002 |
| rank isotonic | 0.00381 +/- 0.00047 | 0.00385 +/- 0.00047 | 0.01735 +/- 0.00116 | 0.01731 +/- 0.00116 | 0.00004 +/- 0.00000 |
| rank spline | 0.00614 +/- 0.00044 | 0.00624 +/- 0.00044 | 0.01502 +/- 0.00113 | 0.01492 +/- 0.00113 | 0.00010 +/- 0.00001 |
| raw binned | 0.00262 +/- 0.00056 | 0.00237 +/- 0.00055 | 0.02378 +/- 0.00128 | 0.02353 +/- 0.00127 | 0.00025 +/- 0.00001 |
| raw isotonic | 0.01813 +/- 0.00067 | 0.01786 +/- 0.00067 | 0.03929 +/- 0.00138 | 0.03902 +/- 0.00138 | 0.00027 +/- 0.00002 |
| raw spline | 0.00257 +/- 0.00044 | 0.00239 +/- 0.00045 | 0.02372 +/- 0.00115 | 0.02355 +/- 0.00116 | 0.00017 +/- 0.00001 |
| zscore binned | 0.01007 +/- 0.00043 | 0.01044 +/- 0.00041 | 0.01109 +/- 0.00111 | 0.01072 +/- 0.00110 | 0.00037 +/- 0.00003 |
| zscore isotonic | 0.00085 +/- 0.00050 | 0.00088 +/- 0.00050 | 0.02033 +/- 0.00116 | 0.02031 +/- 0.00116 | 0.00002 +/- 0.00000 |
| zscore spline | 0.00370 +/- 0.00048 | 0.00377 +/- 0.00048 | 0.01746 +/- 0.00116 | 0.01739 +/- 0.00116 | 0.00007 +/- 0.00001 |

The projection-direct gap is computed between the two aggregate
checkpoint predictions before averaging across checkpoints and seeds.
The full prompt-cell discrepancy is retained in the CSV outputs.

## Interpretation

- This is a later-stage target-pool-aware analysis. It is not the
  paper's pre-pool estimator and should not replace the matched
  calibration-only comparison.
- Projection and direct rows use exactly the same fitted conditional
  mean and evaluation proxy geometry. Their difference measures the
  local two-term projection discrepancy for that fitted curve.
- MAE against DPO-implied gain also includes the optimizer-realization
  gap, so a lower value can result from error cancellation.
- Common paper coefficients and prompt-specific target-aware
  coefficients answer different operational questions.

## Integrity Checks

- Selected seed-checkpoints: 10; prompt counts by seed: `{0: 800, 1: 600, 2: 600}`.
- Maximum absolute difference from the previously saved direct
  checkpoint predictions: `3.26e-16`.
- Every stored idealized frontier was re-derived from raw candidate
  scores and matched the original source row.
- The script refuses to overwrite a non-empty output directory.
