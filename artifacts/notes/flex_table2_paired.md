# flex_table2_paired: paired target-pool-aware differences (appendix)

**Placement.** Appendix, next to Table 2. Body: `tables/flex_table2_paired.tex`. Each dataset x h
uses two lines: point differences, then 95% CIs. Data: `data/flex_table2_paired.csv` (wide; also
contains the MLP rows, the d = 0.10 endpoint path-means and both gap measures).

**Quantities** (CurveMAE differences in native units, 5 decimals; negative = first-named predictor
has lower error):
- Projection - direct: same h, same target-pool information. This is the two-term truncation
  loss in aggregate. Saved (exp32 "evaluation-aware A+B minus direct calibrator").
- Projection - anchor: target-pool-aware projection minus pre-pool A+B. These have different
  information and are not a matched comparison. Saved ("... minus raw A+B").
- Direct - anchor (dagger): this is not saved on common support. flex.py computes it from the
  row-level predictions with the identical protocol. Audit example: SummEval isotonic -0.00276
  [-0.00432, -0.00148]. It also has different information from the anchor.
- Mean pointwise gap: path-mean over common cells of |projection prediction - direct
  prediction| (saved `mean_abs_projection_vs_direct`). It measures prediction disagreement, not error.

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

All saved differences and intervals were re-run from predictions and match to <= 1e-12.

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

## Citable ranges (binned/spline/isotonic, 12 dataset x h cells)

- Aggregate projection-direct gap |CurveMAE_proj - CurveMAE_direct|: 0.00002-0.00286.
- Gridwise mean pointwise gap: 0.00041-0.00850.
- These two quantities are different, and neither is a uniform bound from the theorem.

## All rows

- SummEval human, Binned: proj-direct -0.00132 [-0.00156, -0.00108]; proj-anchor +0.01598 [0.01250, 0.01909]; direct-anchor +0.01729 [0.01388, 0.02039]; pointwise gap 0.00210
- SummEval human, Spline: proj-direct +0.00013 [-0.00023, 0.00058]; proj-anchor -0.00145 [-0.00279, -0.00025]; direct-anchor -0.00158 [-0.00332, -0.00007]; pointwise gap 0.00185
- SummEval human, Isotonic: proj-direct +0.00050 [0.00008, 0.00103]; proj-anchor -0.00226 [-0.00336, -0.00132]; direct-anchor -0.00276 [-0.00432, -0.00148]; pointwise gap 0.00171
- GPT-4o proxy, Binned: proj-direct -0.00004 [-0.00009, 0.00000]; proj-anchor +0.00055 [0.00024, 0.00075]; direct-anchor +0.00059 [0.00027, 0.00084]; pointwise gap 0.00041
- GPT-4o proxy, Spline: proj-direct +0.00003 [-0.00005, 0.00013]; proj-anchor -0.00026 [-0.00108, 0.00037]; direct-anchor -0.00029 [-0.00120, 0.00037]; pointwise gap 0.00042
- GPT-4o proxy, Isotonic: proj-direct -0.00010 [-0.00022, -0.00003]; proj-anchor -0.00005 [-0.00094, 0.00048]; direct-anchor +0.00005 [-0.00072, 0.00053]; pointwise gap 0.00044
- WMT MQM, Binned: proj-direct -0.00286 [-0.00358, -0.00223]; proj-anchor +0.00637 [0.00419, 0.00874]; direct-anchor +0.00922 [0.00673, 0.01191]; pointwise gap 0.00850
- WMT MQM, Spline: proj-direct -0.00031 [-0.00056, -0.00004]; proj-anchor -0.00253 [-0.00426, -0.00099]; direct-anchor -0.00222 [-0.00404, -0.00057]; pointwise gap 0.00570
- WMT MQM, Isotonic: proj-direct +0.00105 [0.00083, 0.00128]; proj-anchor -0.00605 [-0.00750, -0.00467]; direct-anchor -0.00710 [-0.00860, -0.00573]; pointwise gap 0.00456
- OpenMEVA-MAGS, Binned: proj-direct -0.00099 [-0.00125, -0.00075]; proj-anchor +0.00339 [0.00008, 0.00795]; direct-anchor +0.00438 [0.00093, 0.00904]; pointwise gap 0.00180
- OpenMEVA-MAGS, Spline: proj-direct -0.00002 [-0.00015, 0.00011]; proj-anchor -0.00265 [-0.00336, -0.00184]; direct-anchor -0.00263 [-0.00342, -0.00177]; pointwise gap 0.00124
- OpenMEVA-MAGS, Isotonic: proj-direct +0.00006 [-0.00007, 0.00019]; proj-anchor -0.00279 [-0.00342, -0.00210]; direct-anchor -0.00285 [-0.00353, -0.00212]; pointwise gap 0.00120

CI endpoints printed as 0.00000 (nonzero; sign shown):
- GPT-4o proxy, Binned, projection - direct: projection_minus_direct_ci_hi = +3.13e-06

## Reporting details

1. Sign convention: first-named minus second-named CurveMAE, so negative means the first is better.
2. Anchor comparisons mix information regimes. Only projection - direct is information-matched.
3. Path-clustered bootstrap: 2,000 draws, seed 0, paths as clusters. The dagger marks intervals
   computed by flex.py with the same protocol. GPT-4o has only 4 paths.
4. Common valid cells (SummEval 7,989 of 8,000). WMT is the corrected 141-path input.
