# Reward-Model Worked Example

Coherence was chosen before observing the model results. Repeat 0 uses 480 calibration and 1,120 evaluation items.
The coefficients in this worked example use calibration labels only. Held-out gold is used only for evaluation.

| Model | Calibration A | Calibration B | A sqrt(.10) | B(.10) | Predicted gain | Held-out gain | A+B MAE, 50 repeats | A-only MAE, 50 repeats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Skywork | 0.2884 | -0.0054 | 0.0912 | -0.0005 | 0.0907 | 0.0785 | 0.0135 | 0.0137 |
| ArmoRM | 0.2695 | -0.0579 | 0.0852 | -0.0058 | 0.0794 | 0.0775 | 0.0102 | 0.0116 |
| DeBERTa | 0.1974 | -0.0445 | 0.0624 | -0.0044 | 0.0580 | 0.0643 | 0.0102 | 0.0116 |

## Reading the Example

1. Fix the human target, candidate distribution, and proxy input format.
2. Estimate A and B on the rated calibration sample.
3. At a checked radius, compute A sqrt(d) and B d separately, then add them.
4. Use held-out observations to check the prediction, rather than selecting the example using its error.

A measures initial directional alignment. B is the second-order correction to that leading gain.
Negative B reduces the leading prediction at positive d; it is not proof of a global turning point.
The full-pool A/B table is a separate descriptive analysis with article-cluster bootstrap intervals.

## Scope and Manuscript Placement

A short worked example can illustrate how to use the two coefficients. Place the full 12-row coefficient table,
all 50-repeat errors and context details in supplementary material. Do not claim a universal RM ranking or
universal A+B superiority. The supplied articles are prefixes (maximum 2,000 characters), and the common-context
rule may shorten them further. This is a fixed-context, item-split SummEval audit, not a new RLHF training experiment.
In particular, the input preprocessing is frozen for this benchmark before split evaluation; this does not test
generalization to longer future article contexts. The gold averages three expert and five crowd ratings.
Source summary ownership and all human ratings are independently checked against the aligned JSONL.
