# bon_softened_rules: smooth BoN-direction rules inside the checked window (exp23)

**Intended display:** appendix (supports the short main-text scope paragraph on "smooth mechanisms").

## Rules
- **Mixture:** p_alpha = (1 - alpha) p0 + alpha p_BoN,N=8, with p_BoN,8 the exact tie-corrected
  BoN distribution on the evaluation items; alpha in {0, .002, .005, .01, .02, .05, .1, .2, .4, .7, 1}.
- **Rank tilt:** q_lambda(i) proportional to p0(i) F(r_i)^lambda, F = mid-rank empirical CDF in (0, 1]
  (ties get the average rank); lambda in {0, .05, .1, .2, .4, .8, 1.5, 3, 6, 12, 24}. In the atomless
  limit lambda = N - 1 reproduces BoN-N (see `bon_kl_ess_definitions`).
- d = KL(q || p0) computed exactly on the evaluation items (no closed form used).

## Protocol
- exp23 pooled finite-item-set stress test, **not prompt-conditional**.
- 30 random calibration/evaluation splits per path, **30% calibration / 70% evaluation**
  (seed 0). A, B, A_nu and the empirical fit are estimated on the calibration 30%; gains, d and
  errors are evaluated on the held-out 70% (n_eval: SummEval 1,120; OpenMEVA 700; WMT 301-1,974).
- Empirical fit: least-squares a sqrt(d) + b d on the calibration exact tilt curve over
  d <= min(0.10, 0.95 log(m_cal / n_top,cal)).
- WMT: fixed_141 lineage (141 paths; `require_wmt_fixed` passed).

## Aggregation
Nontrivial in-window points 0.005 < d <= 0.10. For each (dataset, rule, path) the absolute
errors are averaged over all in-window (split, grid value) rows; the table reports the unweighted
mean over paths ("path-macro"; each path weight 1, regardless of how many grid values land in the
window). "A+B < A-only (paths)" counts paths whose path-level A+B error is below A-only.
Units: native human-score units (SummEval/OpenMEVA 1-5, WMT MQM points); do not compare across datasets.

## Verification
Rebuilt from `mixture_ranktilt_rows.csv` and compared with the saved
`inwindow_baseline_path_macro.csv`: all 40 cells (4 datasets x 2 rules x 5 baselines) match,
max |diff| = 9.2e-17; path counts match.

Table columns: "Mixture" is the mixture with BoN N = 8. "Paths: A+B wins" = k/n paths whose
path-level A+B error is below A-only (n = number of paths). The in-window grid values per split
(2-4) are given below rather than as a column. Fits 5.5 in at \small.

## Support caveat
Only 2-4 grid values per path and split land in 0.005 < d <= 0.10 (coarse
grids). Median in-window d per cell is 0.023-0.035, so the
comparison mostly probes the lower half of the window; the grid values and d ranges are listed below.

| Dataset | Rule | Paths | A+B | A-only | A_nu | Emp | Zero | A+B<A-only paths | pts/split | in-window grid values | d range (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SummEval human | alpha_mixture | 32 | 0.0189 | 0.0244 | 0.0556 | 0.0186 | 0.0939 | 32 | 2-3 | 0.1;0.2;0.4;0.7 | 0.0054-0.0908 (0.0247) |
| SummEval human | rank_tilt | 32 | 0.0103 | 0.0154 | 0.0458 | 0.0101 | 0.1163 | 31 | 2-4 | 0.2;0.4;0.8;1.5 | 0.0094-0.0995 (0.0330) |
| SummEval GPT-4o | alpha_mixture | 4 | 0.0168 | 0.0209 | 0.0976 | 0.0169 | 0.0241 | 4 | 2-2 | 0.1;0.2 | 0.0139-0.0530 (0.0331) |
| SummEval GPT-4o | rank_tilt | 4 | 0.0064 | 0.0065 | 0.0784 | 0.0063 | 0.0414 | 3 | 2-2 | 0.2;0.4 | 0.0146-0.0492 (0.0315) |
| WMT MQM | alpha_mixture | 141 | 0.1207 | 0.1432 | 0.3917 | 0.1198 | 0.3140 | 135 | 2-3 | 0.1;0.2;0.4;0.7 | 0.0050-0.1000 (0.0232) |
| WMT MQM | rank_tilt | 141 | 0.0492 | 0.0607 | 0.3197 | 0.0488 | 0.4794 | 132 | 2-3 | 0.2;0.4;0.8 | 0.0113-0.0998 (0.0346) |
| OpenMEVA-MAGS | alpha_mixture | 10 | 0.0174 | 0.0156 | 0.0235 | 0.0169 | 0.1552 | 2 | 2-2 | 0.1;0.2 | 0.0113-0.0495 (0.0280) |
| OpenMEVA-MAGS | rank_tilt | 10 | 0.0156 | 0.0139 | 0.0211 | 0.0153 | 0.1421 | 3 | 2-2 | 0.2;0.4 | 0.0089-0.0408 (0.0227) |

## Key results
- A+B is below zero and A_nu in every cell (closest: GPT-4o mixture, 0.0168 vs zero
  0.0241) and within 0.001 of the empirical fit in every cell (asserted).
- A+B improves on A-only for SummEval human (mixture 0.0189 vs 0.0244;
  rank tilt 0.0103 vs 0.0154) and WMT
  (0.1207 vs 0.1432; 0.0492 vs 0.0607).
- GPT-4o mixture favours A+B (0.0168 vs 0.0209); GPT-4o rank tilt is a
  **near tie** (0.0064 vs 0.0065; 4 paths).
- OpenMEVA-MAGS is **worse with B** (mixture 0.0174 vs 0.0156;
  rank tilt 0.0156 vs 0.0139).
- Wording: "supports an informative local approximation for smooth rules inside the window, not
  universal dominance of the curvature term".

## Reporting details
"Pooled finite-item-set rules (not prompt-conditional); coefficients from 30% calibration splits,
errors on the 70% evaluation items, 30 splits; path-macro MAE over 0.005 < d <= 0.10 (each path
weighted equally); only 2-4 grid values per split fall in that range; native units, not comparable
across datasets."

## Sources
- `results/rank_selection_scope/mixture_ranktilt_rows.csv` (rebuild), `results/rank_selection_scope/inwindow_baseline_path_macro.csv` (comparison), `results/rank_selection_scope/mixture_ranktilt_run_spec.json`
- Generator: `experiments/exp23_mixture_ranktilt/build_mixture_ranktilt.py`
