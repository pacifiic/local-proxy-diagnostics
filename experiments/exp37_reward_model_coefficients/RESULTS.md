# Three Reward Models: Results and Paper Example

## Valid Run

The completed run is
[`canonical`](../../results/reward_model_coefficients/canonical).
Its name distinguishes a source-text correction during this new experiment.
It does not denote a change to the paper's original SummEval results.

- [All 12 coefficient pairs, bootstrap intervals, and held-out errors](../../results/reward_model_coefficients/canonical/analysis/comparison.md)
- [Calibration-only worked example](../../results/reward_model_coefficients/canonical/paper_example/WORKED_EXAMPLE.md)
- [Worked-example figure (PDF)](../../results/reward_model_coefficients/canonical/paper_example/fig_reward_model_worked_example.pdf)
- [Worked-example table (LaTeX)](../../results/reward_model_coefficients/canonical/paper_example/table_reward_model_worked_example.tex)
- [Full coefficient table (LaTeX)](../../results/reward_model_coefficients/canonical/paper_example/table_reward_model_coefficients.tex)

## Setup

We scored the same 1,600 SummEval summaries with Skywork-Reward-Llama-3.1-8B-v0.2,
ArmoRM-Llama3-8B-v0.1, and OpenAssistant reward-model-deberta-v3-large-v2.
The scalar reward is the proxy. The human target is the mean of three expert
and five crowd ratings, separately for coherence, consistency, fluency, and
relevance. No new human annotation or fine-tuning was performed.

Model revisions, native input templates, output fields and precision are pinned
in `models.json` and recorded in each scoring manifest. Inference used batch size
one and eager attention. Skywork and ArmoRM used bfloat16; DeBERTa used float32.
GPU 0/1/2 scored Skywork/ArmoRM/DeBERTa, respectively. GPU inference was performed
once per model. The statistical analysis was run twice from those cached scores.

All models received the same article prefix for a given article, with every
summary retained in full. The supplied article text was already limited to at
most 2,000 characters. The shared context rule additionally accommodated the
512-token DeBERTa limit and the two 4,096-token limits. Across 100 articles,
the supplied length averaged 1,797.74 characters and the mean retained fraction
was 96.08%. These are fractions of the supplied prefixes, not of full articles.
Preparation uses the benchmark's candidate texts to freeze this context policy
before splitting. This does not test deployment on unseen or longer articles.

Prospective evaluation uses 480 calibration and 1,120 evaluation items, repeated
50 times with seed 0, and radii 0.025, 0.05, 0.075, 0.10. Splits are identical
across models for each dimension and repeat. Splitting is by item, not article.
All 200 dimension/repeat splits were valid for all three models, giving 600
model-splits with no exclusions. Coefficients used for prediction depend only
on calibration gold and proxy scores. Evaluation gold measures prediction error.

Full-pool coefficient estimates are a separate descriptive analysis, with 1,000
paired article-cluster bootstrap draws (seed 1). They are not substituted for
calibration-only coefficients in prospective predictions.

## Results

Endpoint MAE at d=0.10, equally averaged across the four human dimensions and
50 repeats within each dimension:

| Reward model | A+B | A-only | Empirical fit |
|---|---:|---:|---:|
| Skywork | 0.011262 | 0.015533 | 0.011083 |
| ArmoRM | 0.010459 | 0.013607 | 0.010513 |
| DeBERTa | 0.011291 | 0.015081 | 0.011266 |

The empirical-fit baseline fits the same two-term form to the exact
calibration-split tilt curve. All methods use the same splits. These are
point-estimate comparisons, not tests of statistical significance.

A+B has lower endpoint MAE than A-only in all 12 model/dimension combinations.
Empirical fit remains competitive. This supports using B as a useful local
correction in this audit, not universal superiority over fitted baselines.

For a descriptive example, full-pool coherence coefficients are:

| Reward model | A [95% interval] | B [95% interval] |
|---|---:|---:|
| Skywork | 0.26020 [0.20871, 0.31542] | 0.00860 [-0.04031, 0.06233] |
| ArmoRM | 0.26153 [0.21048, 0.31184] | -0.05047 [-0.09725, -0.00373] |
| DeBERTa | 0.22320 [0.16451, 0.28135] | -0.09084 [-0.13753, -0.03765] |

Skywork and ArmoRM have similar estimated initial alignment A, but different
point estimates of the second-order correction B. Skywork's B interval includes
zero, so this table does not establish positive curvature for Skywork. Individual
intervals also are not a paired test of differences between reward models.

## How to Use This in the Paper

Include a small calibration-only worked example in the main text and the full
coefficient/error tables in supplementary material. The preselected example is
coherence, repeat 0, d=0.10. It was not selected for a favorable prediction error.

For ArmoRM in this example, calibration gives A=0.2695 and B=-0.0579.
At d=0.10 the first-order contribution is 0.0852 and the second-order correction
is -0.0058, predicting a gain of 0.0794 human coherence points. The exact
held-out tilt gain is 0.0775. The reader can directly see what each coefficient
contributes and how to check the forecast. Other models are shown alongside it,
including the DeBERTa split where adding B does not improve the individual
example's error. Aggregate claims instead use all 50 repeats.

Suggested manuscript paragraph:

> To illustrate the diagnostic with learned reward models, we score the same
> SummEval candidates with three fixed reward models and retain human ratings
> as the target. In a prespecified coherence example with 480 calibration items,
> ArmoRM gives A=0.2695 and B=-0.0579. At KL radius d=0.10, the first-order gain
> of 0.0852 receives a second-order correction of -0.0058, yielding 0.0794,
> compared with an exact held-out gain of 0.0775 on 1,120 evaluation items.
> Across all three models and four human dimensions, adding B lowers the
> 50-repeat endpoint MAE relative to A-only. These coefficients characterize
> the specified proxy, human target, candidate distribution and input context,
> rather than an intrinsic ranking of reward models.

Suggested figure caption:

> Calibration-only gain predictions for three reward models on SummEval
> coherence. The same preselected 480/1,120-item split is used in all panels.
> A and B are estimated on calibration items; the black curve is the exact
> raw-score exponential-tilt gain on evaluation items using human gold.
> Article prefixes are shared across models and candidate summaries are retained
> in full. This illustrative split is accompanied by 50-repeat errors and
> full-pool descriptive coefficient intervals in the supplementary tables.

The experiment evaluates finite-pool raw-score tilt. It is not actual DPO/RLHF
training and does not show that a larger A guarantees a better optimized model.
A/B are invariant to positive affine transformations of proxy scores, but
depend on the gold scale and candidate distribution. Negative B reduces the
leading prediction locally; it does not establish a global gain maximum.
Training-data overlap with SummEval has not been excluded.

## Validation

The reported run verifies all 1,600 summary-to-item mappings and all 51,200
individual ratings against the aligned source JSONL. Preparation rejects
mismatched ownership. This check covers summaries and ratings, not article
prefixes, which are supplied separately.

The independent numerical audit checks 12 full-pool coefficient pairs, all 600
calibration predictions, exact endpoint gains via a separate KL root solver,
score provenance, split support and aggregate errors. Maximum discrepancies
are below 1e-13. Its report is `independent_audit.json` in the valid run.

Two CPU analyses produced seven byte-identical files, including bootstrap
draws, predictions, summaries and run specifications (`reproducibility_check.json`).
This verifies reanalysis from cached scores; fresh GPU inference may vary across
hardware. CPU tests cover source mapping, joins, context preservation, scalar
model outputs, splits, moment formulas, affine invariance, label isolation and
output preservation.
