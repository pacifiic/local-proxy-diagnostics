# exp25 Results: Direct Calibration on Real LLM Candidate Pools

## Canonical outputs

- Within-pool: `outputs_within_200rep_20260724`
- Cross-model: `outputs_transfer_20260724_retry1`
- Independent reproducibility checks:
  `outputs_within_200rep_reprocheck_20260724` and
  `outputs_transfer_reprocheck_20260724`

The earlier 50-repeat within output and the empty failed transfer directory are
retained as execution history and should not be cited.

## Protocol

The within-pool experiment uses 1,600 Qwen2.5-7B responses generated for 100
SummEval articles, with 16 candidates per prompt. Skywork-8B is the fixed proxy
and an OpenAssistant DeBERTa reward model is the independent model-gold.
Calibration is split by prompt, not candidate. The reported local CurveMAE is
the mean absolute error over `d={.01,.025,.05,.075,.10}`.

The cross-model experiment uses matched UltraFeedback prompt splits, 32
candidates per prompt, and identical proxy/gold scorers for Qwen2.5-3B and
Zephyr-7B. Source-only predictors use no target-model gold labels. These are
real LLM-generated candidates but model-gold, not human-gold, experiments.

## Within-pool calibration budget

Local CurveMAE over 200 nested prompt-level resplits:

| Calibration | A+B | Empirical fit | Isotonic | Spline | MLP | A-only | A_nu |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.0332 | **0.0323** | 0.0429 | 0.0694 | 0.0643 | 0.0378 | **0.0231** |
| 10% | 0.0252 | 0.0237 | 0.0329 | 0.0582 | 0.0465 | 0.0300 | **0.0192** |
| 20% | 0.0183 | **0.0171** | 0.0262 | 0.0544 | 0.0385 | 0.0214 | 0.0183 |
| 30% | 0.0170 | **0.0163** | 0.0246 | 0.0521 | 0.0362 | 0.0205 | 0.0181 |
| 50% | 0.0156 | **0.0149** | 0.0232 | 0.0516 | 0.0337 | 0.0187 | 0.0185 |

The bold entries identify the lowest empirical-curve method and the lowest
overall method when those differ.

### Interpretation

- A+B beats A-only, isotonic, spline, and MLP at every calibration budget.
  Paired split-bootstrap intervals exclude zero for all these comparisons.
- The empirical calibration curve and its two-parameter fit are consistently
  but only slightly better than A+B. At 30%, the difference in CurveMAE is
  0.0007. This directly answers the empirical baseline: direct empirical curve
  fitting is competitive, but does not make the moment law a poor predictor.
- A_nu is unusually strong at 5% and 10%. It is tied with A+B at 20% and 30%,
  while A+B is better at 50%. This is a scorer-scale coincidence in this one
  model-gold pool and should not be generalized as a reliability-baseline win.
- Increasing the radius does not rescue predictive dominance. At 30%
  calibration and `d=.20`, MAE is 0.0338 for A+B, 0.0290 for empirical fit,
  0.0473 for isotonic, 0.0959 for spline, and 0.0657 for MLP.

## Cross-model source-only transfer

Local CurveMAE on matched, held-out UltraFeedback prompts:

| Source to target | A+B | Empirical fit | Isotonic | Spline | MLP | A-only | A_nu |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen to Zephyr | 0.0145 | 0.0149 | 0.0116 | **0.0028** | 0.0057 | 0.0120 | 0.0050 |
| Zephyr to Qwen | **0.0106** | 0.0101 | 0.0415 | 0.0275 | 0.0365 | 0.0111 | 0.0139 |

Transfer is direction-dependent. Qwen-to-Zephyr favors the spline point
estimate, although the paired prompt-bootstrap interval for A+B minus spline
includes zero. Zephyr-to-Qwen strongly favors A+B over isotonic, spline, and
MLP; empirical fit and A-only remain statistically close. Under target
calibration, A+B is best for Zephyr (0.0002), whereas spline is best for Qwen
(0.0008 versus 0.0038 for A+B).

The defensible conclusion is not uniform transfer superiority. The coefficient
law is competitive and robust in one direction, while a flexible score
calibrator transfers very well in the other. This supports describing
cross-model calibration as policy-dependent.

## Reachability

All Zephyr prompts reach every audited radius. Qwen has quantized/tied proxy
scores: at `d=.10`, 99/100 calibration and 196/200 evaluation prompts reach the
requested prompt-level KL. The remaining prompts use the exp09/exp11 saturation
convention and are reported in `transfer_reachability.csv`. The local results
therefore include at most a 2% saturated Qwen subset.

## Data and reproducibility audit

`data_layer_audit.json` records:

- zero calibration/evaluation prompt overlap;
- exact Qwen/Zephyr prompt-ID and prompt-text alignment;
- no duplicate prompt/candidate keys;
- prediction drift of exactly 0 after permuting evaluation gold labels;
- a nontrivial 0.126 change in measured gain under that perturbation;
- reproduction of exp05 A+B predictions within `9.1e-17`;
- reproduction of exp11 A+B predictions/frontier within `2.5e-16`;
- exact expected row counts: 70,000 within-pool and 360 transfer cells.

The canonical and independent rerun directories are byte-identical for every
shared CSV, deterministic gzip file, JSON, PNG, LaTeX table, and PDF figure.
PDF CreationDate/ModDate metadata are intentionally omitted.

## Summary

A concise, accurate response is:

> We added empirical-curve, isotonic, spline, binned, and MLP calibration
> baselines on real LLM-generated candidate pools. Across 200 prompt-level
> resplits, A+B consistently improves over A-only and the flexible
> isotonic/spline/MLP predictors, while matching the direct empirical
> two-parameter fit within about 0.001 CurveMAE. Cross-model transfer is
> direction-dependent: flexible calibration is strongest from Qwen to Zephyr,
> whereas A+B is substantially more robust than the flexible predictors from
> Zephyr to Qwen. These results support the method as an interpretable local
> diagnostic, not a claim of universal predictive dominance. Gold scores in
> this experiment come from an independent reward model; human-gold evidence
> remains the repeated-rating evaluation in the submitted paper.
