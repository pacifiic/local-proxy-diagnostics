# flex_table2_target_aware: target-pool-aware projection vs direct integration (Table 2)

**Placement.** Main text, Table 2, in the subsection asking whether the two-term form retains the
calibrated curve when information is matched. Body: `tables/flex_table2_target_aware.tex`
(anchor row + {binned, spline, isotonic} x {projection, direct integration}, same dataset
column order as Table 1(b)). `tables/flex_table2_target_aware_main.tex` has the same cells with
shorter labels for the main text. Appendix variant with exploratory MLP rows:
`tables/flex_table2_target_aware_with_mlp.tex`. Data: `data/flex_table2_target_aware.csv` (long
format, including d = 0.10 path-mean endpoint MAE for reference). Paired results: `flex_table2_paired`.

**Information regime.** Target-pool-aware. The available inputs are the calibration proxy scores
and human labels, plus the realized target-pool (evaluation-split) proxy scores. Target human
labels are never used for prediction; they only score the result. This is a legitimate later
deployment scenario, not label leakage. It is a stronger information setting than the pre-pool
Table 1(b).

**Predictors (both use the SAME fitted h).** In each split, h is fitted once on the calibration
items, with the same procedure, splits and seeds as the Table 1(b) calibrators. Both predictors
then use that one fitted h object.
- Projection: insert pseudo-targets h(r) on the target pool into the paper's moment formulas,
  giving A_h, B_h from (h(r_target), r_target), and predict A_h sqrt(d) + B_h d. This is a
  diagnostic extension, not the unchanged paper estimator, and it inherits h's assumptions and
  estimation error.
- Direct integration: evaluate h on the exact target-pool tilt at KL d,
  sum_i w_i(d) h(r_i) - mean_i h(r_i) over target-pool items.
- Projection vs direct for the same h is the information-matched comparison. It isolates the loss
  from the two-term local form.
- **Reference anchor.** The pre-pool A+B row (identical to the Table 1(b) A+B row) sees less
  information: no target-pool proxy scores. It is shown for scale only. It is NOT a matched
  competitor, so rows below it should not be described as "beating" it on equal terms.

## Data, splits and radius grid

- Paths: SummEval human = 32 held-out-rater paths (4 dimensions x 3 expert + 5 crowd held-out
  raters); GPT-4o proxy = 4 dimension paths (GPT-4o judge score as proxy, SummEval human ratings as
  target); WMT MQM = 141 held-out-rater paths from the corrected document-aware `fixed_141` input
  (`data/wmt_mqm_long.csv`, SHA-256 `6f20c0669ff7...`); OpenMEVA-MAGS = 10 held-out-rater
  paths (ROC and WP). The submitted 139-path WMT lineage (6,950 units; A+B CurveMAE 0.06675) is
  excluded everywhere.
- Splits: per path, 50 random 30/70 calibration/evaluation item splits (`random_seed` 0).
  Calibration/evaluation items per split: SummEval and GPT-4o 480/1,120, OpenMEVA 300/700,
  WMT 129-846 / 301-1,974 (median 267/623; varies with cluster size). Every method in both
  information regimes uses the identical split plan (exp21 plan, reused by exp32 and exp33).
- Radius grid: d in {0.01, 0.025, 0.05, 0.075, 0.10}, the KL divergence of the positive raw-score
  exponential tilt from the uniform distribution on the pool.
- Realized gain (the prediction target): the exact finite-pool tilt gain on the held-out evaluation
  split at KL d, computed from evaluation proxy scores and evaluation human labels. These enter
  only the scoring, never a prediction.
- Units: native human-score units of each target. SummEval human and GPT-4o use SummEval 1-5
  ratings, OpenMEVA-MAGS uses 1-5 ratings and WMT uses MQM scores (non-positive error penalties).
  Magnitudes are not comparable across columns, and the columns must not be averaged into a leaderboard.


## Metric: CurveMAE (path-mean over common valid cells)

CurveMAE_m = (1/P) sum_p [ (1/|C_p|) sum_(s,d) in C_p |pred_m(p,s,d) - gain(p,s,d)| ],
where P is the number of paths, s indexes the 50 splits and C_p is the set of common
valid (split, radius) cells of path p. Each path has equal weight, and cells within a path are
equally weighted. Lower is better. This is the convention of the saved paired files
(exp21 `path_clustered_bootstrap_diff`, exp32 support-matched summaries). It is used for every
CurveMAE in the flex displays.

Common valid cells are the cells where every displayed method has a prediction. Only SummEval
loses cells: 11 of 8,000. All are on path `consistency__expert__heldout_2` at d = 0.10, splits 1, 6, 10, 13, 14, 24, 29, 30, 31, 42, 46. At these cells
the calibration split's maximum reachable KL (0.0802-0.0984) is
below 0.10, so the methods that integrate a calibration-pool tilt are undefined there. These are
the stored empirical curve and the binned, spline, isotonic and MLP calibration-only methods.
Source: `direct_calibration_strict/main_protocol/calibration_only_skipped.csv`. The affected path
therefore averages 239 rather than 250 cells. Support is 7,989 SummEval, 1,000 GPT-4o, 35,250 WMT
and 2,500 OpenMEVA cells. WMT has no skipped cells.

The other saved conventions are NOT used:
- `curve_mae.csv` takes the unit-mean over complete five-radius curves. This drops the 11 SummEval
  splits entirely: 1,589 units, so isotonic is 0.01153 rather than 0.01168.
- The all-support numbers, e.g. SummEval A+B 0.01187 on 8,000 cells, are not used either.
- `information_setting_comparison.csv` mixes 1,589-unit and 1,600-unit supports.


## Methods (one sentence each; pre-pool = calibration items only)

- A+B: the paper's estimator, A sqrt(d) + B d, with A and B from calibration proxy/human moments.
- A-only: A sqrt(d), the same A without the second-order term.
- Calibration two-term fit: least-squares (no intercept) fit of A sqrt(d) + B d to the exact
  calibration-pool tilt curve. The curve is sampled at 200 equally spaced tilt values up to KL
  min(0.10, 0.95 x calibration max reachable KL), and the fit is evaluated at each d.
- Stored empirical curve: the exact finite calibration-pool tilt gain at KL d, i.e. calibration
  human labels reweighted by the calibration-proxy tilt. It is undefined if the calibration pool
  cannot reach d.
- Binned / spline / isotonic (and exploratory MLP): fit a conditional mean h(r) ~ E[human | proxy]
  on the calibration items. Then integrate h over the calibration-pool tilt:
  sum_i w_i(d) h(r_i) - mean_i h(r_i) over calibration items.
  - Binned: piecewise-constant means without a monotonicity constraint.
  - Spline: a smooth regularized fit.
  - Isotonic: a nondecreasing fit.

## Tuning grids (calibration-internal only; asserted from the run specs)

- Isotonic: sklearn `IsotonicRegression(increasing=True, out_of_bounds="clip")`, no tuning.
- Binned: equal-frequency proxy bins, with bin count in {3, 5, 8, 12} (only counts with
  3 x bins <= n_cal). Bin means are shrunk toward the calibration mean with pseudo-count
  k in {0, 5, 20}. Both are chosen by 5-fold shuffled KFold CV MSE (random_state 0).
- Spline: cubic B-spline basis (`SplineTransformer`, degree 3, constant extrapolation), with
  knots in {4, 6, 8} chosen by 5-fold KFold CV (random_state 0). The ridge penalty is chosen by
  `RidgeCV` over 13 log-spaced alphas in [1e-3, 1e3] (efficient leave-one-out).
- MLP (exploratory, not in the main tables): 2 hidden ReLU layers of 16 units (321 parameters),
  Adam, L2 alpha 1e-3, learning rate 1e-3, max 1,000 iterations, seed 0, standardized input and
  output. It is pre-specified and untuned.
- Both information regimes fit h with the same code, splits and seeds, using only
  (calibration proxy, calibration human). The pre-pool runs are exp33 and the target-aware runs
  are exp32. Only the pool over which h is integrated differs. Within exp32, projection and
  direct integration use the identical fitted h object.

## Sources

- `evaluation_aware_ab/common_support/evaluation_aware_ab_summary.csv` (non-WMT; the
  11-cell SummEval mask applied).
- `evaluation_aware_ab/wmt_fixed_141/evaluation_aware_ab_summary.csv` (WMT).
- `evaluation_aware_ab_summary_fixed_141.csv` is all-support (SummEval 8,000
  cells) and is NOT used.
- Every displayed value is re-derived from the row-level predictions (`non_wmt_canonical` masked +
  `wmt_fixed_141`) and matches to <= 1e-12. Prediction-file hashes are checked against
  `evaluation_aware_ab/manifest.json` and the common-support report.

## Key results (CurveMAE, 5 decimals)

- SummEval human (anchor 0.01185): Binned proj 0.02782 / direct 0.02914; Spline proj 0.01040 / direct 0.01027; Isotonic proj 0.00959 / direct 0.00909
- GPT-4o proxy (anchor 0.00806): Binned proj 0.00860 / direct 0.00865; Spline proj 0.00780 / direct 0.00777; Isotonic proj 0.00801 / direct 0.00811
- WMT MQM (anchor 0.05921): Binned proj 0.06557 / direct 0.06843; Spline proj 0.05667 / direct 0.05699; Isotonic proj 0.05316 / direct 0.05211
- OpenMEVA-MAGS (anchor 0.00991): Binned proj 0.01330 / direct 0.01429; Spline proj 0.00726 / direct 0.00728; Isotonic proj 0.00712 / direct 0.00706
- Aggregate |projection - direct| CurveMAE gap over the 12 dataset x h cells: 0.00002-0.00286.
- Gridwise (pointwise) mean |projection prediction - direct prediction|, path-mean over common
  cells: 0.00041-0.00850.
- SummEval: the binned-h projection is 0.01824 above the isotonic-h projection.
- Aggregate MAE differences and pointwise projection/direct gaps are different quantities, and
  neither is a uniform theorem bound.

## Do not mix with the original endpoint MAE

The original paper's Table 1(a) endpoint MAE for SummEval (0.0174) uses the identical A+B
predictions but a different metric and protocol: endpoint d = 0.10 only, grid {0.05, 0.10},
whole-split drops and pooling over splits. It must never be placed next to, or compared with, the
CurveMAE 0.01185. Captions must name the metric (CurveMAE over five radii versus endpoint MAE at
d = 0.10).

## Reporting details

1. Target-pool-aware regime: target-pool proxy scores are available and target human labels are
   not used for prediction.
2. Projection and direct integration use the same fitted h. Projection = two-term moment
   formula on (h(r), r); direct = exact tilt integration of h. Define both in full before abbreviating.
3. The pre-pool A+B row is a reference anchor with less information, not a matched competitor.
4. Metric and support as in Table 1(b): CurveMAE over five radii, common valid cells (SummEval
   7,989 of 8,000), native units, lower is better. Paired CIs are in the appendix.
5. MLP is exploratory (appendix).
