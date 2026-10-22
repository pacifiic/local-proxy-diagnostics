# flex_table1b_prepool: pre-pool CurveMAE (Table 1(b))

**Placement.** Main text, Table 1(b). It belongs in the experiments subsection that asks whether
calibration can predict held-out gain before the target pool exists. Body: `tables/flex_table1b_prepool.tex`
(7 method rows x 4 dataset columns, fixed order SummEval human, GPT-4o proxy, WMT MQM, OpenMEVA-MAGS).
Appendix variant with the exploratory MLP row: `tables/flex_table1b_prepool_with_mlp.tex`.
Unrounded data: `data/flex_table1b_prepool.csv`. Paired CIs: `flex_table1b_paired`.

**Information regime.** Pre-pool: every method sees only the rated calibration split (proxy
scores, human labels) and the requested radius d. Target-pool proxy scores and target human
labels are unavailable. Numbers have 5 decimals. The second header row gives paths / common cells.

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

## Sources and provenance of every cell

- Non-WMT rows come from `direct_calibration_strict/main_protocol/paired_diff_curvemae_bootstrap.csv`
  (`mae_baseline`; A+B = `mae_A+B` of the rows paired on common cells). Its WMT rows are the
  superseded 139-path lineage and are dropped.
- WMT rows come from `direct_calibration_strict/wmt_fixed_141/paired_diff_curvemae_bootstrap.csv`.
- **Recomputed cells:** SummEval A-only and calibration two-term fit. The saved rows pair A+B on
  all 8,000 cells (saved 0.01804 / 0.01177, with A+B
  0.01187). Those saved values must NOT be used. flex.py recomputes both from
  `evaluation_aware_ab/non_wmt_canonical/direct_calibration_predictions.csv.gz`, masked by
  `calibration_only_skipped.csv`. These exp32 rows contain the same shared-baseline predictions
  as exp33. The build verifies this: the exp32 rows reproduce the saved exp33 8,000-cell SummEval
  A+B, A-only and two-term-fit values, and every saved common-cell value on the other datasets,
  to <= 1e-12. exp33 `audit.json` also reports zero shared-prediction drift.
- Checks run by the build:
  - A+B, A-only, two-term fit and stored curve, with their saved paired CIs, are checked against
    the row-level predictions on every dataset (max deviation 9.5e-17).
  - A local rerun of exp33 on a different machine changes one WMT split's least-squares
    two-term fit, moving CurveMAE by about 1.8e-8 (the known environment boundary). The audit
    draft `draft_table1b_common_cells.csv` was built from such a rerun. The displayed value is
    the saved one (0.0584275140).
  - The calibration-only binned/spline/isotonic/MLP rows have no row-level predictions in the
    package. Their saved values are on the 7,989 common cells by construction, because exactly
    those methods skip the 11 cells.

## Key results (5 decimals; same common cells within each column)

- A+B CurveMAE: SummEval human 0.01185, GPT-4o proxy 0.00806, WMT MQM 0.05921, OpenMEVA-MAGS 0.00991.
- A-only CurveMAE: SummEval human 0.01803, GPT-4o proxy 0.01030, WMT MQM 0.08039, OpenMEVA-MAGS 0.01058.
- Relative reduction of A+B vs A-only, (A-only - A+B)/A-only: 34.3/21.7/26.4/6.4% (SummEval/GPT-4o/WMT/OpenMEVA).
  These reductions belong to this matched pre-pool CurveMAE analysis, not to the original endpoint table.
- Main-table rows sorted by CurveMAE: SummEval human: isotonic 0.01168, calibration two-term fit 0.01170, stored empirical curve 0.01171, A+B 0.01185, spline 0.01275, A-only 0.01803, binned 0.03010; GPT-4o proxy: spline 0.00803, calibration two-term fit 0.00805, stored empirical curve 0.00805, A+B 0.00806, isotonic 0.00827, binned 0.00878, A-only 0.01030; WMT MQM: stored empirical curve 0.05842, calibration two-term fit 0.05843, isotonic 0.05882, A+B 0.05921, spline 0.06255, binned 0.07356, A-only 0.08039; OpenMEVA-MAGS: isotonic 0.00984, calibration two-term fit 0.00984, stored empirical curve 0.00984, A+B 0.00991, spline 0.01002, A-only 0.01058, binned 0.01583.
- Paired 95% CI outcomes against A+B (from `flex_table1b_paired`; flexible-calibrator
  advantages must be reported in the main text):
  - SummEval human: A+B significantly lower than A-only, binned, spline; significantly lower than A+B: none; CI includes 0: calibration two-term fit, stored empirical curve, isotonic.
  - GPT-4o proxy: A+B significantly lower than A-only, binned, isotonic; significantly lower than A+B: none; CI includes 0: calibration two-term fit, stored empirical curve, spline.
  - WMT MQM: A+B significantly lower than A-only, binned, spline; significantly lower than A+B: calibration two-term fit, stored empirical curve, isotonic; CI includes 0: none.
  - OpenMEVA-MAGS: A+B significantly lower than binned; significantly lower than A+B: none; CI includes 0: A-only, calibration two-term fit, stored empirical curve, spline, isotonic.
- Point-estimate order is not a significance claim. Note that the OpenMEVA A+B vs A-only
  difference (6.4%) has a CI that includes 0.

## Do not mix with the original endpoint MAE

The original paper's Table 1(a) endpoint MAE for SummEval (0.0174) uses the identical A+B
predictions but a different metric and protocol: endpoint d = 0.10 only, grid {0.05, 0.10},
whole-split drops and pooling over splits. It must never be placed next to, or compared with, the
CurveMAE 0.01185. Captions must name the metric (CurveMAE over five radii versus endpoint MAE at
d = 0.10).

## Reporting details

1. Pre-pool regime: only calibration items are used, and no target-pool proxy scores or labels.
2. Metric: CurveMAE over d in {0.01, 0.025, 0.05, 0.075, 0.10}, path-mean of within-path absolute
   errors over common valid cells. Lower is better. Native units; columns are not comparable.
3. Support: 50 random 30/70 splits per path. SummEval uses 7,989 of 8,000 cells, because 11
   d = 0.10 cells are unreachable by calibration-pool methods. WMT is the corrected 141-path input.
4. One-line definitions of the two-term fit and the stored empirical curve. Flexible-calibrator
   tuning is calibration-internal (details in appendix).
5. Paired bootstrap intervals are in the appendix table. No bolding implies significance.
   MLP is exploratory and reported in the appendix.
6. This is not the endpoint MAE of Table 1(a).

## Design notes

- No entry is bolded. If emphasis is wanted, mark cells using the paired CI (see
  `paired_outcome` in `data/flex_table1b_paired.csv`) rather than the raw minimum.
- The second header row (paths / cells) can be dropped if the caption states the support.
