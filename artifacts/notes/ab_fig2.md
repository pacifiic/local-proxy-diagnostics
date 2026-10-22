# ab_fig2: what adding B contributes across the local window (main Figure 2)

## Placement
Main text, Experiments, question 3 ("What does B add, and when is its estimate
unreliable?"). Full text width (5.5 in), about 2.2 in high; vector PDF `artifacts/figures/ab_fig2.pdf`
(PNG preview alongside). Appendix companions: `ab_reduction` (native-unit MAEs and paired
CurveMAE CIs), `ab_bshare` (table with IQRs), `ab_b_stability`, `ab_rater_noise`, `ab_skewness`.

## Design choice
Panel (a) plots the within-dataset relative endpoint-MAE reduction
1 - MAE_A+B(d) / MAE_A-only(d), not native-unit MAEs. Both MAEs are in the same dataset's
human-score units, so the ratio is unitless, and the four settings can share one axis without
comparing native-unit MAEs across different human-score scales (plan Section 6).
Native-unit small multiples were rejected for three reasons. Four MAE panels plus a B-share
panel do not fit legibly at 5.5 x 2.2 in. Separate y-scales still invite cross-dataset
comparison by eye. The ratio answers the question directly: how much of A-only's error
adding B removes at each radius. The native-unit values are in `ab_reduction`.
Panel (b) uses the same radius axis and dataset encoding. Readers can then see that B is a
small part of the predicted gain magnitude (1-13%) but removes a large part of A-only's
error (up to 44%). These are different quantities, and the caption must say so.
Dataset colours avoid the paper's A+B blue and A-only orange, because neither panel plots a
single method. Each dataset also has its own marker. The palette was checked for
colour-vision deficiency (all pairs). The yellow OpenMEVA line is below 3:1 contrast, which
is acceptable because markers, the legend and the appendix tables also identify it.

## What it shows
(a) At every checked radius, adding the B term lowers the aggregate endpoint MAE in all four
settings (point estimates). The relative reduction grows monotonically with d. At d = 0.10 it is
about 44% (SummEval), 29% (GPT-4o), 35% (WMT) and 8% (OpenMEVA). The OpenMEVA intervals include
0 for d <= 0.05 and barely exclude it at d >= 0.075. (b) The median magnitude share of the B
term in the calibration-predicted gain also grows with d, roughly like sqrt(d), because
|B d| / |A sqrt(d)| = (|B|/|A|) sqrt(d). At d = 0.10 it is 4.4% (SummEval), 13.3% (GPT-4o),
8.3% (WMT) and 3.1% (OpenMEVA). B is therefore a modest share of the predicted gain. Omitting
it still leaves a systematic curvature error that grows with d, and adding it removes a large
fraction of A-only's aggregate endpoint error.

## Definitions
- Predictors (pre-pool, calibration-only): moment coefficients A-hat, B-hat are computed on
  the 30% calibration split. A-only predicts A-hat sqrt(d). A+B predicts
  A-hat sqrt(d) + B-hat d, with B = E[J-bar Z^2] - (2/3) E[J-bar Z] E[Z^3].
- Realized gain Delta(d): exact exponential-tilt gain of the held-out 70% evaluation split at
  KL radius d (evaluation human labels used only here).
- Endpoint MAE at d: mean over (path, split) cells of |prediction - Delta(d)|.
- Panel (a): 1 - MAE_A+B(d) / MAE_A-only(d) on identical cells. Positive means adding B lowers MAE.
- Panel (b): R_B(d) = |B-hat d| / (|A-hat sqrt(d)| + |B-hat d|) per path-split, then the dataset
  median. R_B is a magnitude share of the *predicted gain*. It is not the net sign or size of
  B, not a share of the realized gain, and not an MAE improvement. It is computed from the
  calibration coefficients only and says nothing about accuracy.

## Units
Both panels are percentages (unitless). Native-unit MAEs (human-score units of each dataset)
are not displayed or compared across datasets.

## Aggregation and support
- Grid d in {0.01, 0.025, 0.05, 0.075, 0.10}; 50 splits per path; split seed 0.
- Paths / path-splits: SummEval 32 / 1,600, GPT-4o 4 / 200, WMT 141 / 7,050 (corrected
  fixed_141 lineage), OpenMEVA 10 / 500.
- Common-valid-cell support, identical to Table 1(b). The 11 SummEval d = 0.10 cells whose
  calibration frontier cannot reach d = 0.10 are excluded for every method, so the SummEval
  endpoint at d = 0.10 uses 1,589 cells and reproduces the original Table 1(a) row
  (A+B 0.0174, A-only 0.0310; agreement within 3e-7 of the independent original d-sweep
  code path). All other radius/dataset cells are complete.
- Panel (a) error bars: 95% percentile path-cluster bootstrap intervals of the ratio. Paths
  are resampled with replacement and all their cells are kept, using 2,000 draws with seed 0
  per dataset x radius. They are path-cluster intervals for the aggregate ratio, not per-split
  intervals. GPT-4o has only 4 paths and OpenMEVA 10, so their intervals are coarse and
  descriptive. Markers are dodged horizontally by up to +/-0.002 in d so the bars do not
  overlap. The plotted radii are the five grid values.
- Panel (b): median over all path-splits (B share does not depend on reachability). The
  interquartile ranges are in `ab_bshare`.

## Protocol and information
Pre-pool: predictions use only calibration proxy scores and calibration human labels. The
held-out evaluation split (70%) supplies realized gains only. The exp31 30%-calibration
coefficients used in (b) are the coefficients of the exp33 A+B predictor in (a). The build
asserts that the predictions are identical to 1e-12 (same split schedule, RNG seed 0).

## Sources
- `results/evaluation_aware_ab/non_wmt_canonical/direct_calibration_predictions.csv.gz`
  (SummEval, GPT-4o, OpenMEVA; exp21 canonical per-cell predictions; WMT deliberately absent)
- `results/evaluation_aware_ab/wmt_fixed_141/direct_calibration_predictions.csv.gz`
  (corrected 141-path WMT)
- exp33 summaries used only for verification: `direct_calibration_strict/main_protocol/`
  (`endpoint_mae_by_d.csv`, `curve_mae.csv`, `paired_diff_curvemae_bootstrap.csv`,
  `calibration_only_skipped.csv`; non-WMT rows only, the WMT rows there are the superseded
  139-path lineage) and `direct_calibration_strict/wmt_fixed_141/` (same four files).
- `results/coefficient_precision/coefficient_error_rows.csv`
  (exp31; 30%-calibration rows) and `coefficient_error_summary.csv` (verification).
- `results/12_accept_boost/d_sweep_human/d_sweep_summary_overall.csv`
  (original SummEval endpoint row; verification only).
Data: `artifacts/data/ab_fig2.csv` holds the plotted values unrounded, and
`artifacts/data/ab_fig2_path_sums.csv` holds the per-path cell sums that the bootstrap resamples.

## Key results
Relative endpoint-MAE reduction [95% path-cluster CI]:

| Dataset | d = 0.01 | d = 0.05 | d = 0.10 |
|---|---:|---:|---:|
| SummEval | 9.2% [5.9, 13.3] | 30.2% [19.8, 38.2] | 44.0% [30.8, 52.8] |
| GPT-4o | 7.2% [4.1, 10.7] | 19.0% [10.3, 27.4] | 28.7% [14.9, 40.0] |
| WMT | 7.9% [6.9, 8.9] | 23.3% [20.7, 25.9] | 34.8% [31.5, 38.2] |
| OpenMEVA | 1.9% [-1.1, 4.6] | 5.9% [-0.2, 11.3] | 8.3% [0.1, 15.5] |

Median R_B(d):

| Dataset | d = 0.01 | d = 0.05 | d = 0.10 |
|---|---:|---:|---:|
| SummEval | 1.5% | 3.2% | 4.4% |
| GPT-4o | 4.6% | 9.7% | 13.3% |
| WMT | 2.8% | 6.0% | 8.3% |
| OpenMEVA | 1.0% | 2.2% | 3.1% |

SummEval's d = 0.10 median share is 4.4% at one decimal (4.447%), not 4.5%.

Curve-level context (Table 1(b) support, from `ab_reduction`): CurveMAE reductions are
SummEval 34.28%, GPT-4o 21.73%, WMT 26.35%, OpenMEVA 6.36%.

## Caveats the caption MUST include
1. Panel (a) is a within-dataset ratio of MAEs on identical cells. Native-unit MAEs are not
   comparable across datasets and are not shown.
2. The improvement is an aggregate effect. A+B has lower CurveMAE than A-only on only
   55.6-67.0% of path-split units (OpenMEVA 55.6%, SummEval 67.0%).
3. R_B(d) is a magnitude share of the predicted gain, |B-hat d| / (|A-hat sqrt(d)| + |B-hat d|),
   not net B and not an MAE improvement.
4. Error bars are 95% path-cluster bootstrap intervals (paths resampled). GPT-4o has 4 paths
   and OpenMEVA 10, so their intervals are descriptive. OpenMEVA's intervals include 0 at
   d <= 0.05.
5. Pre-pool setting: 30% calibration, 50 splits, held-out exact tilt gain. Support is the common
   valid cells of Table 1(b), so SummEval d = 0.10 uses 1,589 of 1,600 cells.

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
