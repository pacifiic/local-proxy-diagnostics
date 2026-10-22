# flex_radius_endpoint: pre-pool endpoint MAE at each radius (appendix)

**Placement.** Appendix, radius-level companion to Table 1(b). Body: `tables/flex_radius_endpoint.tex`.
It has 4 dataset panels x {A+B, A-only, two-term fit, stored curve, binned, spline, isotonic,
MLP (exploratory)}, with columns d = 0.010 ... 0.100, 4 decimals (endpoint-MAE convention). Data:
`data/flex_radius_endpoint.csv`. The data file also has the remaining exp33 pre-pool baselines
(A_nu, Pearson/Spearman correlation scaling, zero) with `in_table = False`.

**Metric.** Endpoint MAE at radius d = mean of |pred - realized gain| over the common valid
(path, split) cells at that radius (pre-pool, native units, lower is better). This is the saved
exp33 `endpoint_mae_by_d.csv` convention: a pooled mean over cells.
- Every path has all 50 splits at every radius, except SummEval at d = 0.10. So the pooled mean
  equals the path-mean at every displayed cell except the SummEval d = 0.10 column.
- In that column, all methods are pooled over the same 1,589 common cells. The path-mean is not
  computable there for the calibration-only binned/spline/isotonic/MLP methods, because their
  row-level predictions are not in the package.
- The data CSV gives the path-mean where it is computable (`endpoint_mae_pathmean`).
- Support per radius: SummEval 1,600 cells (1,589 at d = 0.10), GPT-4o 200, WMT 7,050,
  OpenMEVA 500.

**Sources.**
- `direct_calibration_strict/main_protocol/endpoint_mae_by_d.csv` (non-WMT; its WMT rows are the
  superseded 139-path lineage and are dropped) and `direct_calibration_strict/wmt_fixed_141/endpoint_mae_by_d.csv`.
- **Consistent-support fix:** the saved SummEval d = 0.10 values pool 1,600 cells for A+B, A-only,
  two-term fit, A_nu, Pearson, Spearman and zero, but 1,589 for the calibration-frontier methods.
  flex.py replaces those seven with pooled means over the 1,589 common cells, recomputed from
  the masked exp32 predictions. The 1,600-cell values remain in the CSV as
  `superseded_saved_value_all_cells`.
- Every saved value with a row-level counterpart is checked against the predictions
  (<= 1e-12).

**Link to the original endpoint table.**
- Pooling the identical A+B predictions over the 1,589 valid SummEval d = 0.10 cells gives
  0.017390, which is the original Table 1(a) value 0.0174. The build checks this
  against `results/.../exp27_prospective_paths.csv`. A-only gives
  0.031036, i.e. 0.0310.
- The path-mean on the same cells is 0.017499, the exp32 summary's
  `raw_A+B_endpoint_mae_d0p10`. The all-1,600-cell pooled value is 0.017527.
- The original protocol still differs (d = 0.10 only, grid {0.05, 0.10}, whole-split drops,
  pooled splits). None of these endpoint numbers may be compared with or substituted for CurveMAE
  (SummEval A+B 0.01185).

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

## Key results

A+B endpoint MAE by radius:
- SummEval human: d=0.010 0.0057, d=0.025 0.0088, d=0.050 0.0123, d=0.075 0.0150, d=0.100 0.0174
- GPT-4o proxy: d=0.010 0.0039, d=0.025 0.0060, d=0.050 0.0084, d=0.075 0.0102, d=0.100 0.0118
- WMT MQM: d=0.010 0.0297, d=0.025 0.0454, d=0.050 0.0620, d=0.075 0.0743, d=0.100 0.0846
- OpenMEVA-MAGS: d=0.010 0.0045, d=0.025 0.0072, d=0.050 0.0103, d=0.075 0.0127, d=0.100 0.0148

Relative A+B reduction vs A-only by radius:
- SummEval human: d=0.010 9.2%, d=0.025 18.6%, d=0.050 30.2%, d=0.075 38.1%, d=0.100 44.0%
- GPT-4o proxy: d=0.010 7.2%, d=0.025 12.5%, d=0.050 19.0%, d=0.075 24.4%, d=0.100 28.7%
- WMT MQM: d=0.010 7.9%, d=0.025 14.7%, d=0.050 23.3%, d=0.075 29.8%, d=0.100 34.8%
- OpenMEVA-MAGS: d=0.010 1.9%, d=0.025 3.7%, d=0.050 5.9%, d=0.075 7.4%, d=0.100 8.3%

## Reporting details

1. Pre-pool, endpoint MAE at each radius. This is not CurveMAE, and the d = 0.10 column is
   comparable to Table 1(a).
2. Pooled over common valid cells per radius. SummEval d = 0.10 uses 1,589 of 1,600 cells for
   every method.
3. Native units, lower is better. MLP is exploratory. WMT is the corrected 141-path input.
