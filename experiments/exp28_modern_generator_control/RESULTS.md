# exp28 Results: Modern Open Generators

## Protocol

The experiment used the same 100 calibration prompts, 200 disjoint evaluation
prompts, 32 candidates per prompt, proxy scorer, silver-gold scorer, and local
fixed-prompt-KL analysis for both generators.

| Role | Model | Revision |
|---|---|---|
| Generator | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` |
| Generator | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Proxy | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` | recorded in each run manifest |
| Silver gold | `OpenAssistant/reward-model-deberta-v3-large-v2` | recorded in each run manifest |

Each generator was run with base seeds 0, 1, and 2. Every run contains 9,600
scored candidates and passed the candidate-identity, prompt-count, score,
revision, and merge-hash audit.

## Diagnostics

Values below are the mean and standard deviation across three generation seeds.

| Generator | Proxy-silver Pearson | A | B | A_nu |
|---|---:|---:|---:|---:|
| Ministral 3 8B | 0.1923 +/- 0.0091 | 0.2297 +/- 0.0211 | -0.0013 +/- 0.0111 | 0.1933 +/- 0.0104 |
| Llama 3.1 8B | 0.2509 +/- 0.0080 | 0.2443 +/- 0.0212 | -0.0245 +/- 0.0182 | 0.2613 +/- 0.0064 |

One Ministral calibration prompt had one unique response (`40`) for all 32
candidates in every seed. It is a deterministic addition question, so its
proxy and silver-gold variances are both zero. The coefficient fitter
consistently skipped this one non-informative prompt; no generated row was
deleted.

## Local Curve MAE

Mean CurveMAE on `d <= 0.10`, with standard deviation across three generation
seeds:

| Baseline | Ministral 3 8B | Llama 3.1 8B |
|---|---:|---:|
| A+B | 0.00967 +/- 0.00513 | 0.01805 +/- 0.00530 |
| A-only | 0.00961 +/- 0.00488 | 0.01678 +/- 0.00492 |
| A_nu | 0.01740 +/- 0.00295 | 0.01314 +/- 0.00132 |
| Empirical curve | 0.01017 +/- 0.00500 | 0.01821 +/- 0.00520 |
| Empirical fit | 0.01017 +/- 0.00500 | 0.01820 +/- 0.00520 |
| Binned shrinkage | 0.00484 +/- 0.00209 | 0.01294 +/- 0.00243 |
| Spline ridge | 0.01089 +/- 0.00194 | 0.01936 +/- 0.00120 |
| Isotonic calibration | 0.00485 +/- 0.00315 | 0.02864 +/- 0.00245 |
| MLP calibration | 0.00314 +/- 0.00137 | 0.01913 +/- 0.00192 |
| Zero | 0.05883 +/- 0.00073 | 0.06916 +/- 0.00059 |

## Interpretation

The modern-generator control does not support a predictive-SOTA claim for A+B.
The best flexible calibration method depends on the generator: MLP, binned,
and isotonic calibration are strongest for Ministral, while binned calibration
and the reliability-only baseline are strongest for Llama. A+B nevertheless
improves substantially over zero,
closely tracks the direct empirical curve/fit, and supplies the alignment and
curvature decomposition that the black-box predictors do not provide.

The fitted B values are small relative to A in this setting. Consistent with
that diagnostic, A-only is slightly better than A+B on average, although the
difference is small compared with the variation across generation seeds. This
is evidence for reporting the method as an interpretable local diagnostic,
not as a universally best predictor.

These are model-based silver-gold results. They are useful as a controlled
pipeline and scope check, but they do not replace evaluation with new human
ratings.

## Artifacts

- Per-run outputs: `outputs/exp28_{ministral3_8b,llama31_8b}_seed{0,1,2}_20260724`
- Driver logs: the sibling `*.driver.log` files
- Current combined analysis:
  `outputs/exp28_modern_generators_analysis_current_20260725`
- Historical combined analysis:
  `outputs/exp28_modern_generators_analysis_20260724`
- Figure: `fig_modern_generator_control.pdf`
- Seed-level metrics: `local_curve_summary.csv`
- Across-seed metrics: `across_seed_summary.csv`
- Pool diagnostics: `pool_diagnostics.csv`

The historical combined analysis predates source-hash and package-version
recording and differs in three binned/spline rows. Two independent
current-code reruns are byte-identical; the current directory above is the
canonical analysis.
