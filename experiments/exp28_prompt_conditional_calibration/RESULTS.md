# exp28 Results: Why Raw Isotonic Behaves Differently Under DPO

## Question

Why is direct isotonic calibration competitive in the exact exponential-tilt
benchmarks but poor in the DPO audit?

The answer has two distinct parts:

1. the DPO candidate pool is prompt-conditional, while the original isotonic
   baseline pools raw scores across prompts;
2. realized DPO does not attain the exact prompt-tilt frontier.

The experiment uses the fixed candidate pool and the three DPO training seeds
from `results/actual_dpo/training_runs`. All deployable
calibration models fit their conditional-mean functions using only the 100
calibration prompts. At prediction time, those frozen functions are integrated
over the proxy scores in each of the 200 evaluation candidate pools. Exp28 is
therefore target-pool-aware: evaluation proxy geometry is used for prediction,
while evaluation silver-target scores are used only to score predictions.

The fixed moment-based A+B/A-only/A_nu rows inherit the exp09 coefficient fit:
99/100 prompts are used, with one zero-variance prompt excluded using
calibration information only.

## Data Geometry

| Split | Global Pearson | Within-prompt centered Pearson | Proxy between-prompt variance | Gold between-prompt variance |
|---|---:|---:|---:|---:|
| Calibration | 0.199 | 0.217 | 0.793 | 0.855 |
| Evaluation | 0.119 | 0.266 | 0.795 | 0.871 |

Roughly 79% of proxy variance and 85--87% of gold variance is between prompts.
A raw pooled map therefore spends most of its capacity learning prompt offsets,
although optimization ranks candidates within each prompt.

## Main Results

Mean MAE and sample SD are computed over the three fixed-pool DPO seeds. The
historical scope contains six checkpoints per seed.

| Predictor | MAE vs exact frontier | MAE vs realized DPO |
|---|---:|---:|
| Rank spline | 0.01029 +/- 0.00085 | **0.01950 +/- 0.00266** |
| Spearman | 0.01013 +/- 0.00059 | 0.01966 +/- 0.00315 |
| A+B | 0.00686 +/- 0.00065 | 0.02294 +/- 0.00299 |
| Rank isotonic | 0.00676 +/- 0.00064 | 0.02303 +/- 0.00299 |
| A-only | 0.00207 +/- 0.00065 | 0.02799 +/- 0.00406 |
| Z-score isotonic | 0.00130 +/- 0.00070 | 0.02851 +/- 0.00369 |
| Centered isotonic | **0.00089 +/- 0.00060** | 0.03055 +/- 0.00381 |
| Raw spline | 0.00279 +/- 0.00078 | 0.03259 +/- 0.00390 |
| Zero | 0.06471 +/- 0.00623 | 0.03492 +/- 0.00303 |
| A_nu | 0.00737 +/- 0.00190 | 0.03717 +/- 0.00507 |
| Raw isotonic | 0.02729 +/- 0.00347 | 0.05709 +/- 0.00669 |

Spline hyperparameters were selected with five-fold `GroupKFold` by prompt.
This matters: candidate-row CV gave raw-spline DPO MAE 0.05093, whereas
prompt-grouped CV gives 0.03259.

The strict `d <= 0.10` summary leads to the same qualitative conclusion:
centered isotonic best predicts the exact frontier, while Spearman and rank
spline have the lowest realized-DPO MAE. Depending on the DPO seed, three or
four checkpoints satisfy the strict threshold.

## Exact Error Decomposition

For every model and checkpoint,

```text
prediction - realized DPO
  = (prediction - exact frontier)
  + (exact frontier - realized DPO).
```

The mean mechanism gap is 0.02980. Across the six checkpoints, the observed
DPO gold gain is about 38--59% of exact proxy-tilt gold gain. This ratio is
descriptive, not a formal optimizer-efficiency bound: the exponential tilt
optimizes the proxy objective, so its gold gain is not guaranteed to upper
bound every other policy's gold gain. Centered isotonic nearly eliminates the
first term, so its remaining DPO error is approximately the observed mechanism
gap.
A+B underpredicts the frontier by 0.00686 on average, and rank spline
underpredicts it by 0.01029. Those errors oppose and partially cancel the
positive mechanism gap at every checkpoint.

Therefore, lower realized-DPO MAE does not by itself mean that a predictor is a
better model of the exponential-tilt frontier. In particular, the DPO advantage
of A+B over centered isotonic is partly error cancellation. Rank spline and
Spearman produce still stronger cancellation and slightly outperform A+B on
realized DPO.

## Interpretation

The direct-calibration and DPO findings are compatible:

- The direct benchmarks optimize one item set by the exact exponential tilt,
  so raw `r -> E[J | r]` calibration directly targets the evaluated mechanism.
- DPO performs prompt-conditional optimization. Raw score levels are dominated
  by prompt offsets, making pooled raw isotonic structurally mismatched.
- Prompt centering or z-scoring repairs the calibration model and predicts the
  exact prompt-tilt frontier very accurately.
- DPO is an approximate optimizer and realizes only part of that frontier.
  Predicting the frontier and predicting optimizer efficiency are separate
  tasks.

The supported conclusion is not that A+B universally beats nonparametric
calibration. A+B remains an interpretable, calibration-only frontier
diagnostic and is better than A-only for realized DPO in this audit. However,
prompt-aware rank spline and Spearman are slightly better realized-DPO
predictors, while centered isotonic is substantially better for the exact
frontier.

These paired differences have the same sign in all three seeds. Rank spline
minus A+B realized-DPO MAE ranges from -0.00304 to -0.00366; centered isotonic
minus A+B frontier MAE ranges from -0.00493 to -0.00702. With only three
training seeds, these are descriptive consistency checks rather than
significance claims.

## Reproducibility

The public canonical outputs are in `results/prompt_conditional_calibration/` and
`results/actual_dpo/three_seed_canonical/`. The latter includes the compressed
prompt-level predictions used by the audits. The checks cover:

- exact reproduction of the historical raw-isotonic predictions;
- exact re-derivation of each saved prompt-tilt frontier;
- calibration/evaluation key and support checks;
- the error-decomposition identity;
- prompt-grouped spline model selection.

The canonical output includes the reliability-only \(A_\nu\) in both
the idealized-tilt and realized-DPO summaries.
