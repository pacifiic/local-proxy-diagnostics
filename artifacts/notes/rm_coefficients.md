# rm_coefficients: exp37 full-pool descriptive coefficients

**Placement.** Appendix (CAMERA_READY_PLAN.md §8.4); referenced from the worked example.

**What it shows.** A and B for each of 12 RM/dimension pairs, estimated on all 1,600 SummEval items,
with 95% percentile intervals from 1,000 paired article-cluster bootstrap draws (seed 1; same article
draw for every model and dimension). B = tail term + skewness term (checked < 1e-12).

**Units.** A, B in human-score units of the dimension per sqrt(KL) and per KL respectively; invariant to
positive affine changes of the proxy score, not to target rescaling or population shift.

**Sources.** `results/reward_model_coefficients/canonical/analysis/coefficients.csv`;
intervals re-checked as percentiles of `.../analysis/coefficient_bootstrap.csv` (< 1e-12). The regenerated
body equals `.../paper_example/table_reward_model_coefficients.tex` cell for cell (only `$-$` minus signs differ).

**Key results.** Skywork coherence B = 0.00860 [-0.04031, 0.06233] (includes zero);
the other 11 B intervals exclude zero (all negative). Coherence A: Skywork 0.26020,
ArmoRM 0.26153, DeBERTa
0.22320.

**Reporting details.** Full-pool descriptive estimates (all 1,600 items), not the calibration-only
predictors of the worked example; 1,000-draw paired article-cluster bootstrap, percentile 95% intervals;
same shared article context as the worked example.

**Caveats.** Do not describe Skywork coherence curvature as positive (its interval includes zero).
Marginal intervals are not a paired test of differences between reward models. Descriptive full-pool
values differ from the worked example's calibration-only values because the estimation sample differs.
These A/B characterize proxy, target, candidate distribution and context, not intrinsic RM quality.
