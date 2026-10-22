# exp24: Calibration Efficiency, Radius, and OOD Transfer

> **WMT input lineage (2026-07-25).** The WMT rows below use the corrected
> document-aware `fixed_141` input. The paper-matched `submitted_139` within-
> dataset output was retained in the source working tree. Both WMT lineages
> have compact paired artifacts under `results/wmt_input_lineages/`.
> SummEval, GPT-4o, and OpenMEVA rows are unaffected.

> **Information-setting note (2026-07-25).** The flexible calibrators below
> use the unlabeled evaluation proxy distribution. They describe the stage
> after a target pool has been collected and proxy-scored. They do not match a
> decision made from 300 rated items before the future 1,000 candidates exist.
> The fair pre-pool comparison is exp33 and is summarized in
> `experiments/exp33_calibration_only_flexible_baselines/RESULTS.md`.

## Protocol

The within-dataset experiment uses the same calibration/evaluation construction
as exp21 on SummEval human-heldout, GPT-4o, WMT MQM, and OpenMEVA-MAGS.
Calibration fractions are 5%, 10%, 20%, 30%, and 50%, with 50 nested repeated
splits per path. `CurveMAE` is the mean absolute error on
`d={.01,.025,.05,.075,.10}`. The additional `d={.15,.20}` points are a scope
audit, not a theorem-window claim.

The OOD experiment uses three routes: SummEval→WMT, WMT→SummEval, and
SummEval→OpenMEVA. It follows a `z_cal` protocol and reports standardized target
gain. A/B coefficients and one-dimensional calibration functions receive the
same source/target blending weight. `target_weight=0` is source-only estimation,
0.75 is the fixed target-heavy blend, and 1 is target-only estimation.

## Calibration-size result

Selected local CurveMAE:

| Dataset | Cal. | A+B | Empirical fit | Isotonic | Spline |
|---|---:|---:|---:|---:|---:|
| SummEval human | 5% | 0.0249 | 0.0249 | **0.0202** | 0.0262 |
| SummEval human | 10% | 0.0180 | 0.0181 | **0.0143** | 0.0178 |
| SummEval human | 30% | 0.0119 | 0.0118 | **0.0091** | 0.0103 |
| GPT-4o | 5% | 0.0165 | 0.0165 | **0.0157** | 0.0210 |
| GPT-4o | 10% | 0.0126 | 0.0127 | **0.0119** | 0.0144 |
| GPT-4o | 30% | 0.0081 | 0.0080 | 0.0081 | **0.0078** |
| WMT MQM | 5% | 0.1284 | 0.1268 | **0.1137** | 0.1528 |
| WMT MQM | 10% | 0.0927 | 0.0914 | **0.0820** | 0.1026 |
| WMT MQM | 30% | 0.0592 | 0.0584 | **0.0521** | 0.0570 |
| OpenMEVA | 5% | 0.0203 | 0.0203 | **0.0151** | 0.0181 |
| OpenMEVA | 10% | 0.0154 | 0.0154 | **0.0109** | 0.0125 |
| OpenMEVA | 30% | 0.0099 | 0.0098 | **0.0071** | 0.0073 |

Within this target-pool-aware setting, isotonic is already strong at 5%.
These rows do not establish calibration-only data efficiency because the
future pool's proxy distribution is already available. A narrower stability
claim is supported against the more flexible spline. At 5%, A+B significantly
beats the spline for GPT-4o and WMT; on SummEval the numerical advantage is not
significant. At 10%, A+B still significantly beats the spline for GPT-4o and
WMT. By 30%, spline is better on SummEval, WMT, and OpenMEVA, while GPT-4o is a
tie. Empirical two-parameter fitting is essentially tied with A+B except for a
small but significant WMT advantage.

## Radius result

At 30% calibration, increasing the radius from 0.10 to 0.20 raises A+B MAE:

| Dataset | A+B @.10 | A+B @.20 | Isotonic @.20 | Reachable @.20 |
|---|---:|---:|---:|---:|
| SummEval human | 0.0175 | 0.0216 | **0.0188** | 84.4% |
| GPT-4o | 0.0118 | 0.0165 | 0.0169 | 100% |
| WMT MQM | 0.0846 | 0.1183 | **0.0960** | 99.3% |
| OpenMEVA | 0.0148 | 0.0221 | **0.0157** | 100% |

There is no favorable high-radius crossover that rescues predictive dominance.
The absolute errors grow and the direct calibrators generally remain
competitive. SummEval at 0.20 is a selected reachable subset, so its 0.10 and
0.20 values must not be interpreted as a like-for-like growth estimate.

## OOD result

Selected standardized CurveMAE at 10% target calibration:

| Route | Target weight | A+B | Isotonic | Spline |
|---|---:|---:|---:|---:|
| SummEval→WMT | 0 | 0.0320 | 0.0284 | **0.0281** |
| SummEval→WMT | .75 | 0.0261 | **0.0230** | 0.0285 |
| WMT→SummEval | 0 | **0.0256** | 0.0335 | 0.0398 |
| WMT→SummEval | .75 | 0.0215 | **0.0187** | 0.0249 |
| SummEval→OpenMEVA | 0 | **0.0854** | 0.1208 | 0.1217 |
| SummEval→OpenMEVA | .75 | **0.0263** | 0.0335 | 0.0362 |

Source-only A/B transfer is significantly better than both direct functions
for WMT→SummEval and SummEval→OpenMEVA, but significantly worse for
SummEval→WMT at target fractions of at least 10%. Under the fixed
target-heavy blend, A+B is significantly better than both functions for
SummEval→OpenMEVA, isotonic is numerically and usually significantly better for
the other two routes, and A+B often retains an advantage over the spline at
small target budgets. Transfer is therefore a genuine benefit in some
directions, not a universal A+B advantage.

`target_weight=0` is source-only **estimation**, but the `z_cal` evaluation
still uses target-calibration statistics to define target units. It must not be
described as zero-target-label deployment.

## Interpretation

The defensible statement is:

> After a target pool is proxy-scored, flexible one-dimensional calibration is
> a strong IID predictor, including at small label budgets. In the separate
> information-matched pre-pool comparison, A+B is tied with isotonic/MLP and
> beats binned calibration throughout. The coefficient law is tuning-free,
> provides the theorem-derived alignment/curvature decomposition, and transfers
> more robustly in some cross-dataset directions.

Avoid claiming that A+B is uniformly more data-efficient or uniformly more
transferable than isotonic regression.

## Outputs and reproducibility

The original source working tree retained canonical and byte-identical
reproduction directories for both the within-dataset and OOD runs. Those large
raw directories are not duplicated here. This compact package retains:

- the corrected and submission-reconstruction WMT OOD summaries under
  `results/wmt_input_lineages/`;
- the strict 30% within-dataset comparison under
  `results/direct_calibration_strict/`;
- this result record and the executable builders/audits in the present
  experiment directory.

Both audit scripts produce CSV/LaTeX compact tables and PNG figures.
