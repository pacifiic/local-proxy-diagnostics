# exp26 Results

The 50-repeat run is in
`outputs/exp26_b_rater_noise_50rep_reachability_20260724`. An independent
rerun with half as many workers is in
`outputs/exp26_b_rater_noise_50rep_reprocheck_20260724`. Every CSV, JSON, PNG,
and PDF output is byte-identical across the two runs.

## Main findings

At 30% calibration with clean ratings, increasing the number of gold raters
reduces mean absolute B error:

| Rating pool | Gold raters | Mean absolute B error | B-sign agreement with clean evaluation | CurveMAE A+B | CurveMAE A-only |
|---|---:|---:|---:|---:|---:|
| Expert | 1 | 0.0618 | 97.3% | 0.0269 | 0.0393 |
| Expert | 2 | 0.0428 | 100.0% | 0.0144 | 0.0278 |
| Crowd | 1 | 0.0659 | 81.6% | 0.0122 | 0.0135 |
| Crowd | 2 | 0.0612 | 84.0% | 0.0108 | 0.0122 |
| Crowd | 3 | 0.0596 | 85.8% | 0.0106 | 0.0121 |
| Crowd | 4 | 0.0590 | 85.4% | 0.0104 | 0.0119 |

With all available gold raters and 30% calibration, B-sign agreement with the
corresponding clean estimate remains at least 99.3% for expert paths and 93.0%
for crowd paths across the audited Gaussian and variance-normalized
Student-t noise conditions. Predictive usefulness is less stable than sign:
A+B remains better than A-only under moderate expert noise, but the advantage
disappears at one rating-SD noise and erodes earlier for crowd ratings.

This supports a scoped claim: B's direction is fairly stable under the
audited rater-budget and noise interventions, while accurate second-order
gain prediction still benefits from repeated, reasonably clean ratings.

## Reachability and interpretation

One of 6,400 path/repeat/calibration-fraction combinations had a tied proxy
maximum with maximum reachable KL 0.099820, just below d=0.10. The code
retains its four lower-radius observations and marks only its d=0.10 exact
endpoint as unavailable. The event is
`consistency__expert__heldout_2`, repeat 13, at 50% calibration.

Noise scales are nominal pre-clipping scales. Student-t noise is normalized to
unit variance before scaling, then all perturbed ratings are clipped to the
observed rating range, so realized post-clipping variance can be smaller.
Evaluation gold remains the clean mean of all non-proxy raters by design.
