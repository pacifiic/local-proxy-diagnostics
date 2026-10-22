# ab_reduction: A-only versus A+B error by radius and over the curve (appendix)

## Placement
Appendix, A/B contribution section; the numeric companion of Figure 2(a). Two tabular bodies,
to be stacked as panels of one table:
- `ab_reduction.tex`: endpoint MAE by dataset x radius (20 rows).
- `ab_reduction_curvemae.tex`: CurveMAE with paired 95% path-cluster CIs (4 rows).

## Columns
Endpoint panel: Dataset, radius d, Cells, A-only MAE, A+B MAE, difference A+B - A-only
(negative = A+B lower error), relative reduction 1 - A+B/A-only (%), and its 95% path-cluster
bootstrap CI (%). CurveMAE panel: Dataset, A-only, A+B, the paired difference A+B - A-only
("Diff.") with its 95% CI, relative reduction (%), and "Units won": the share of path-split
units whose own CurveMAE (mean over their radius cells) is lower for A+B than for A-only.
Paths and cells are omitted from the CurveMAE panel for width and must go in the caption:
SummEval 32 paths / 7,989 cells, GPT-4o 4 / 1,000, WMT 141 / 35,250, OpenMEVA 10 / 2,500.

## Definitions, units, aggregation
- Predictors and realized gain as in `ab_fig2` (pre-pool, 30% calibration, 50 splits, seed 0,
  exact held-out tilt gain).
- Endpoint MAE: mean of |error| over the cells at that radius (4 decimals). The difference
  uses the same 4 decimals.
- CurveMAE and its paired difference (5 decimals): per-path mean of cell errors over the radius
  grid, then the equal-weight mean over paths. This is exactly the exp33 paired-bootstrap
  convention behind Table 1(b), and it is what the CI resamples.
- CIs: 95% percentile path-cluster bootstrap, 2,000 draws, seed 0. For CurveMAE this is
  draw-for-draw the exp21/exp33 `path_clustered_bootstrap_diff`. Endpoint CIs resample paths
  and keep all their cells.
- Units: MAEs and differences are in each dataset's human-score units. Compare A+B with A-only
  within a dataset, never magnitudes across datasets. Only the relative reduction is unitless.
- Sign convention: differences are A+B minus A-only, as in the exp33 paired files and
  Table 1(b), so negative values favour A+B.

## Support
Common valid cells, identical to Table 1(b). The 11 SummEval cells at d = 0.10 whose
calibration frontier cannot reach d = 0.10 are dropped for both methods. SummEval therefore
has 7,989 CurveMAE cells and 1,589 endpoint cells at d = 0.10, and all other cells are complete.
With this support the SummEval d = 0.10 row matches the original paper's Table 1(a) row
(A+B 0.0174, A-only 0.0310) to within 3e-7. All-cell values (8,000 SummEval cells) are kept
in the CSVs as `allcell_*`: SummEval A-only 0.018035, A+B
0.011873, reduction
34.17%. Do not mix the two
supports in prose.

## Sources
- `results/evaluation_aware_ab/non_wmt_canonical/direct_calibration_predictions.csv.gz`
  (SummEval, GPT-4o, OpenMEVA; exp21 canonical per-cell predictions; WMT deliberately absent)
- `results/evaluation_aware_ab/wmt_fixed_141/direct_calibration_predictions.csv.gz`
  (corrected 141-path WMT)
- exp33 summaries used only for verification: `direct_calibration_strict/main_protocol/`
  (`endpoint_mae_by_d.csv`, `curve_mae.csv`, `paired_diff_curvemae_bootstrap.csv`,
  `calibration_only_skipped.csv`; non-WMT rows only, the WMT rows there are the superseded
  139-path lineage) and `direct_calibration_strict/wmt_fixed_141/` (same four files).
The SummEval common-cell CurveMAE CI is recomputed from the saved per-cell rows. The
procedure is validated by reproducing every saved all-cell A+B vs A-only CI to 1e-12. For the
other three datasets common = all cells, so the values equal the saved exp33 paired rows.

## Key results
Endpoint at d = 0.10:

| Dataset | A-only | A+B | A+B - A-only | Reduction |
|---|---:|---:|---:|---:|
| SummEval | 0.0310 | 0.0174 | -0.0136 (-0.013646) | 44.0% |
| GPT-4o | 0.0165 | 0.0118 | -0.0047 (-0.004742) | 28.7% |
| WMT | 0.1298 | 0.0846 | -0.0452 (-0.045235) | 34.8% |
| OpenMEVA | 0.0162 | 0.0148 | -0.0013 (-0.001338) | 8.3% |

CurveMAE (common valid cells):

| Dataset | A-only | A+B | A+B - A-only [95% CI] | Reduction | Units with A+B lower |
|---|---:|---:|---:|---:|---:|
| SummEval | 0.01803 | 0.01185 | -0.006182 [-0.009619, -0.003243] | 34.28% | 67.0% |
| GPT-4o | 0.01030 | 0.00806 | -0.002237 [-0.003641, -0.001000] | 21.73% | 65.5% |
| WMT | 0.08039 | 0.05921 | -0.021182 [-0.024515, -0.018260] | 26.35% | 65.2% |
| OpenMEVA | 0.01058 | 0.00991 | -0.000673 [-0.001361, +0.000015] | 6.36% | 55.6% |

Rounding traps: the SummEval endpoint difference at d = 0.10 is 0.0136 (0.013646), not 0.0137.
The CurveMAE reductions are 34.28/21.73/26.35/6.36%, or 34.3/21.7/26.4/6.4% at one decimal.

## Caveats the caption MUST include
1. Support is the common valid cells of Table 1(b): 11 SummEval d = 0.10 cells excluded.
2. CIs are 95% path-cluster bootstrap intervals (paths resampled), not per-split intervals.
   The GPT-4o intervals rest on 4 paths and are descriptive.
3. The OpenMEVA CurveMAE CI includes 0 ([-0.00136, +0.000015]),
   and its endpoint-reduction CIs include 0 for d <= 0.05.
4. The improvement is aggregate. A+B wins on only
   55.6-67.0% of path-split units.
5. MAEs are in native human-score units and must not be compared across datasets.
6. The CurveMAE panel's paths/cells (listed under Columns) must be stated in the caption.

## Verification performed by `ab.py`
- `endpoint_vs_exp33`: max |diff| = 9.71e-17
- `curvemae_vs_exp33`: max |diff| = 9.37e-17
- `paired_bootstrap_vs_exp33`: max |diff| = 9.84e-17
- `rb_formula_vs_saved_share`: max |diff| = 4.33e-15
- `exp31_vs_exp33_pred_ab`: max |diff| = 0.00e+00
- `exp31_vs_exp33_pred_aonly`: max |diff| = 8.88e-16
- `exp31_vs_exp33_actual`: max |diff| = 0.00e+00
- `rb_median_vs_exp31_summary`: max |diff| = 8.33e-17
- `summeval_d010_vs_original_dsweep`: max |diff| = 2.64e-07
