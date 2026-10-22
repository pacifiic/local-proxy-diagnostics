# exp21 Protocol Audit (2026-07-25)

## Main-experiment identity

The direct-calibration benchmark imports the paper's canonical path,
coefficient, tilt, and split utilities. It uses a 30% calibration split, 50
repeats, and random seed 0.

On the support retained by the paper's prospective experiment, the `A+B`
prediction agrees split by split:

| Setting | Main paths | exp21 paths | max prediction difference at d=.05/.10 |
|---|---:|---:|---:|
| SummEval human proxy | 32 | 32 | 0 / 0 |
| GPT-4o proxy | 4 | 4 | 0 / 0 |
| WMT MQM human proxy | 139 | 139 | 0 / 0 |
| OpenMEVA human proxy | 10 | 10 | 0 / 0 |

The maximum difference in the realized target is below `1.1e-5` at `d=.05`
and below `4e-15` at `d=.10`. The small `d=.05` difference comes from the
paper code interpolating a precomputed tilt grid while exp21 solves the
requested radius directly.

The paper retained 1,589 of 1,600 SummEval path-splits because its old
all-or-nothing implementation dropped a split when the calibration exact
curve could not bracket the full fit radius. Exp21 retains each baseline at
every evaluation radius it can support. On the common 1,589 splits, its
`A+B` predictions and the paper predictions are identical.

## WMT input lineage

The two prepared WMT files answer different provenance questions:

- `submitted_139`: the submitted paper input, including malformed
  document-free WMT22 zh-en item keys;
- `fixed_141`: the corrected input with document-aware WMT22 keys, seven
  valid en-de 3-rater clusters, and corrected TSV parsing.

This is a deterministic input-version difference, not stochastic variation.
The file `data/wmt_mqm_long_main_protocol.csv`, with SHA-256
`5649ee3041340f7ee96c3079a4f1f859b9ddd625b2d82f731407d32fd88553c0`,
is retained only to reproduce the submitted 139-path result
(`A+B MAE=0.095817` at `d=.10`). The scientifically corrected input is
`data/wmt_mqm_long.csv`, with SHA-256
`6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8`
and 141 paths.

In the source working tree, the paper-matched portable all-dataset rerun was
stored as `outputs_main_protocol_repeat_20260725/`. Its prediction, skip,
endpoint, CurveMAE, and paired-CI CSVs were byte-for-byte identical to the
preceding `submitted_139` run. The redundant raw directory is not copied into this
compact package; its retained summaries are under
`results/direct_calibration/`.

The WMT-dependent calibration-budget and precision audits were also
regenerated from the paper-matched `submitted_139` input in the source working
tree:

- `exp24_direct_calibration_followup/outputs_within_main_protocol_20260725/`;
- `exp31_coefficient_precision_audit/outputs_main_protocol_20260725/`.

The compact paired lineage artifacts and validation report are stored under
`results/wmt_input_lineages/`. New scientific claims and camera-ready values
should use `fixed_141`; submitted-number reproduction should use `submitted_139`.

## Information available to each predictor

Operationally, exp21 corresponds to the stage after a target candidate pool
has been collected and proxy-scored. It does not match a decision made from
300 rated calibration items before the future 1,000 optimization candidates
exist.

- `A+B`, `A-only`, `A_nu`, empirical-curve transfer, and empirical-fit use
  calibration items only.
- Isotonic, binned, spline, and MLP calibrators fit `h(r)` using calibration
  labels, then evaluate `h` on the unlabeled evaluation proxy scores and
  integrate it under the evaluation proxy tilt.
- No evaluation gold labels enter model fitting, CV, hyperparameter
  selection, or prediction.

The flexible baselines are therefore valid deployable predictors, but their
advantage over the original `A+B` estimator combines two effects:

1. flexibility of the estimated conditional mean `h(r)`;
2. adaptation to the evaluation-side proxy distribution.

The calibration-only empirical curve and two-parameter empirical fit are
nearly identical to `A+B`:

| Setting | A+B CurveMAE | empirical curve | empirical fit |
|---|---:|---:|---:|
| SummEval human proxy | .011873 | .011555 | .011767 |
| GPT-4o proxy | .008059 | .008047 | .008047 |
| WMT MQM human proxy | .066745 | .065945 | .065958 |
| OpenMEVA human proxy | .009906 | .009841 | .009841 |

This supports the two-term local form while showing that evaluation-aware
conditional-mean calibration can be a stronger IID predictor.

## Strict pre-pool comparison

Exp33 removes evaluation-side proxy access from isotonic, binned, spline, and
MLP calibration. Each fitted conditional mean is integrated over calibration
proxy values, and the prediction is frozen before evaluation data are read.
Its explicit permutation audit changes evaluation proxy geometry while
leaving every prediction unchanged (`prediction drift = 0`). On paired cells,
calibration-only isotonic and MLP are statistically tied with A+B; A+B beats
binned calibration on all four datasets and spline on SummEval and WMT.

Canonical summary:
`../../CALIBRATION_BASELINE_INFORMATION_AUDIT_20260725.md`.

## Legitimate extensions of A+B

1. **Evaluation-aware conditional-mean moments.** Fit `h(r)` on calibration
   items, evaluate it on unlabeled evaluation proxy scores, and compute the
   `A,B` moments of `(h(r_eval), r_eval)`. This gives the coefficient method
   the same proxy-distribution information as direct calibration. Existing
   calibration-crossover experiments close most of the isotonic gap with
   this estimator. It is a method extension and should be labeled as such.
2. **Third Puiseux term.** The tuning-free `C d^(3/2)` experiment improves
   point estimates slightly on all four datasets, but only the corrected
   `fixed_141` WMT comparison was statistically resolved. Both WMT lineages
   have now been run. The extra fourth-moment variance makes this a
   robustness result, not a replacement for the two-term diagnostic.
3. **Hierarchical or uncertainty-based shrinkage of B.** This is legitimate
   only if the shrinkage rule is selected without evaluation outcomes.
   Existing hard uncertainty gating did not improve aggregate MAE.

Post-hoc tuning of `B`, choosing radii after observing evaluation errors, or
blending with a flexible calibrator using evaluation labels would not be a
valid way to improve the reported method.
