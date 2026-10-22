# exp34 Results: Fair Calibration-Only DPO Baselines

## Question

The earlier DPO conditional-mean baselines fitted `h(r)` on 100 calibration
prompts but evaluated `h` on the 200 evaluation prompt pools. They therefore
used evaluation proxy geometry that `A+B` did not use.

Exp34 matches the information budget. It represents a decision made from 100
rated prompts before the 200 future prompts and their 6,400 candidates have
been collected. Every flexible method is fitted and integrated on the 3,200
calibration candidates. The resulting prompt-averaged gain curve is frozen.
At evaluation it receives only the KL radius reached by DPO.

The inherited moment-based A+B/A-only/A_nu rows use 99/100 prompts because one
calibration prompt has zero proxy and silver-target variance. This exclusion
is determined entirely within the calibration pool.

This evaluates gain conditional on the measured radius, matching the original
coefficient-law comparison. It does not predict the optimizer's radius from
training hyperparameters.

## Strict Local Results

The table reports the mean and sample SD across three independent DPO seeds.
Each seed first averages its 3--4 checkpoints whose median prompt-level radius
is at most `0.10`.

| Predictor | Ideal-frontier MAE | Realized-DPO MAE |
|---|---:|---:|
| A+B | 0.00395 +/- 0.00047 | 0.01721 +/- 0.00116 |
| A-only | 0.00210 +/- 0.00053 | 0.01905 +/- 0.00120 |
| A_nu | 0.00371 +/- 0.00061 | 0.02487 +/- 0.00129 |
| Zero | 0.04197 +/- 0.00017 | 0.02081 +/- 0.00062 |
| Calibration empirical curve | 0.00430 +/- 0.00047 | 0.01686 +/- 0.00116 |
| Raw isotonic | 0.01878 +/- 0.00080 | 0.03994 +/- 0.00150 |
| Raw binned | 0.00193 +/- 0.00058 | 0.02309 +/- 0.00126 |
| Raw spline | **0.00131 +/- 0.00055** | 0.02247 +/- 0.00124 |
| Raw MLP | 0.01497 +/- 0.00075 | 0.03612 +/- 0.00144 |
| Prompt-rank isotonic | 0.00433 +/- 0.00047 | 0.01683 +/- 0.00116 |
| Prompt-rank binned | 0.01020 +/- 0.00039 | **0.01096 +/- 0.00107** |
| Prompt-rank spline | 0.00804 +/- 0.00042 | 0.01312 +/- 0.00111 |
| Prompt-rank MLP | 0.00577 +/- 0.00045 | 0.01539 +/- 0.00114 |

## Interpretation

The two error targets answer different questions. Raw spline most faithfully
predicts the matched exponential-tilt frontier. Prompt-rank binned has the
lowest realized-DPO MAE, but it underpredicts that frontier by `0.01020` on
average. DPO itself falls below the matched frontier by `0.02116`. The lower
realized error therefore partly reflects cancellation between frontier
underprediction and optimizer under-realization.

`A+B` remains better than A-only for realized-DPO prediction among the
original paper baselines, while A-only is closer to the ideal frontier in
this audit. Exp34 should be presented as a process-level scope and error
decomposition, not as a general predictor leaderboard or human-rated RLHF
validation.

## Information Boundary and Reproducibility

- The predictor never loads evaluation candidate proxy or gold scores.
- Replacing every evaluation score in the pool file by extreme constants
  leaves all calibration arrays unchanged.
- A+B, A-only, and A_nu exactly match the canonical exp09 rows.
- Exact curve evaluation and the saved interpolation differ by at most
  `1.72e-7`.
- Two complete runs produced 11 byte-identical files.
- Audit status is `PASS`.

Packaged canonical outputs are in
`../../results/dpo_flexible_calibration_only/`. Two complete runs produced 11
byte-identical files; the published directory retains one canonical copy.
