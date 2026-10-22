# flex_table1b_paired: pre-pool paired differences against A+B (appendix)

**Placement.** Appendix, next to the full pre-pool comparison. Body: `tables/flex_table1b_paired.tex`
(4 dataset panels, A+B reference row plus 7 methods including the exploratory MLP). Data:
`data/flex_table1b_paired.csv`.

**Quantity.** For each method m: CurveMAE_m - CurveMAE_(A+B) on the same common valid cells,
in native units, 5 decimals. **Positive means the method has larger error than A+B** (A+B
better). The saved exp33 files store the opposite orientation (A+B minus method). The data CSV
keeps both (`*_AplusB_minus_method`). Percentile CIs flip exactly under negation.

## Paired uncertainty protocol

95% percentile intervals come from a path-clustered bootstrap. Paths are resampled with
replacement, 2,000 draws, numpy `default_rng(0)` reseeded for each comparison, and the
difference of path-mean CurveMAEs is recomputed. This is the protocol of exp21
`audit_direct_calibration.path_clustered_bootstrap_diff` (used by exp33) and exp32
`clustered_interval`. flex.py re-implements it and reproduces every saved interval it can check to
<= 1e-12. Splits within a path are correlated, so the path is the resampling unit. GPT-4o has
only 4 paths, so its intervals are coarse and descriptive. Intervals are per comparison and are
not adjusted for multiplicity. An endpoint printed as 0.00000 is not exactly zero. Whether that
CI includes 0 is decided from the unrounded CSV, and those cases are listed below.

**Recomputed intervals (dagger in the table).** The SummEval A-only and calibration two-term-fit
intervals are not saved on common support: the saved rows use 8,000 cells. flex.py computes them
from the exp32 row-level predictions masked to the 7,989 common cells, with the same protocol
(seed 0, 2,000 path draws). All other intervals are the saved exp33 values. Wherever the build
could re-run a saved interval from predictions (A-only, two-term fit, stored curve on all
datasets), the re-run matches (max deviation 9.5e-17).

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

## Citable paired results (method minus A+B, 95% CI, outcome)

- SummEval human: A-only +0.00618 [0.00324, 0.00962] (A+B lower); calibration two-term fit -0.00015 [-0.00043, 0.00004] (CI includes 0); stored empirical curve -0.00014 [-0.00044, 0.00004] (CI includes 0); binned +0.01825 [0.01548, 0.02091] (A+B lower); spline +0.00090 [0.00036, 0.00136] (A+B lower); isotonic -0.00017 [-0.00045, 0.00002] (CI includes 0); MLP (exploratory) -0.00003 [-0.00032, 0.00016] (CI includes 0)
- GPT-4o proxy: A-only +0.00224 [0.00100, 0.00364] (A+B lower); calibration two-term fit -0.00001 [-0.00006, 0.00006] (CI includes 0); stored empirical curve -0.00001 [-0.00006, 0.00007] (CI includes 0); binned +0.00072 [0.00043, 0.00101] (A+B lower); spline -0.00002 [-0.00017, 0.00014] (CI includes 0); isotonic +0.00022 [0.00001, 0.00042] (A+B lower); MLP (exploratory) +0.00008 [-0.00009, 0.00025] (CI includes 0)
- WMT MQM: A-only +0.02118 [0.01826, 0.02451] (A+B lower); calibration two-term fit -0.00078 [-0.00099, -0.00058] (method lower); stored empirical curve -0.00079 [-0.00100, -0.00059] (method lower); binned +0.01435 [0.01220, 0.01680] (A+B lower); spline +0.00334 [0.00248, 0.00427] (A+B lower); isotonic -0.00039 [-0.00070, -0.00009] (method lower); MLP (exploratory) -0.00005 [-0.00047, 0.00041] (CI includes 0)
- OpenMEVA-MAGS: A-only +0.00067 [-0.00001, 0.00136] (CI includes 0); calibration two-term fit -0.00007 [-0.00020, 0.00006] (CI includes 0); stored empirical curve -0.00007 [-0.00021, 0.00006] (CI includes 0); binned +0.00593 [0.00299, 0.00986] (A+B lower); spline +0.00011 [-0.00013, 0.00032] (CI includes 0); isotonic -0.00007 [-0.00021, 0.00006] (CI includes 0); MLP (exploratory) -0.00006 [-0.00024, 0.00013] (CI includes 0)

Outcome labels: "A+B lower" and "method lower" mean the CI excludes 0. "CI includes 0" is not
evidence of equivalence.
- WMT is the only dataset where isotonic is significantly lower than A+B.
- On the 4-path GPT-4o slice, A+B is lower than isotonic with a CI that barely excludes 0.

CI endpoints printed as 0.00000 (nonzero; sign shown):
- none

## Do not mix with the original endpoint MAE

The original paper's Table 1(a) endpoint MAE for SummEval (0.0174) uses the identical A+B
predictions but a different metric and protocol: endpoint d = 0.10 only, grid {0.05, 0.10},
whole-split drops and pooling over splits. It must never be placed next to, or compared with, the
CurveMAE 0.01185. Captions must name the metric (CurveMAE over five radii versus endpoint MAE at
d = 0.10).

## Reporting details

1. Sign convention: method minus A+B, so positive means A+B better.
2. Path-clustered percentile bootstrap: 2,000 draws, paths as clusters, seed 0. GPT-4o has 4 paths.
3. Common valid cells (SummEval 7,989 of 8,000). The dagger marks intervals recomputed with the
   same protocol because they were not saved on common support.
4. No multiplicity adjustment. MLP is exploratory.
