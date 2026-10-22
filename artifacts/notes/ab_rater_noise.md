# ab_rater_noise: sensitivity of B to rater count and rating noise (appendix)

## Placement
Appendix, rater count and noise. The main text gets one sentence in the "when is B unreliable"
paragraph. Outputs: table `ab_rater_noise.tex` and figure `ab_rater_noise.pdf` (5.5 x 1.95 in).

## Figure provenance
The figure is restyled from `results/rater_noise/fig_b_rater_noise_stability.pdf`
(registered as an input; produced by `experiments/exp26_b_rater_noise_stability/make_artifacts.py`).
It plots the same summary rows and the same panel filters (artifact_spec: {'panel_a': 'clean, 30% calibration, all available gold-rater counts', 'panels_b_c': '30% calibration, maximum gold raters, nonzero noise'}).
Changes: camera-ready font sizes and width; "Turker" relabelled "Crowd"; the sign panel's y-axis
starts at 90% instead of 0%, because the old axis hid the differences; panel (c) adds the saved
95% path-cluster CIs, which the original omitted; and colours avoid the A+B/A-only method hues.

## Design (exp26)
- Data: SummEval (32 paths = 4 dimensions x {expert, crowd} x held-out proxy rater), 50 nested
  item splits, seed 0 with SHA-256-derived streams. Only the 30% calibration rows are shown;
  10/20/50% rows are in the source.
- Proxy: one fixed held-out rater (a *held-out human rater as proxy*, not a model).
- Calibration gold J: mean of a nested subset of non-proxy raters, with optional synthetic noise. Expert pools have up to 2 non-proxy
  raters and crowd pools up to 4.
- Evaluation gold: clean mean of every non-proxy rater, independent of calibration rater count and noise.
- Noise: independent per-rating Gaussian or variance-normalized t(df=3) noise, scaled by group rating SD and clipped to the observed rating range. Scales of 0.25/0.5/1.0 group rating SD are nominal
  pre-clipping scales, so the realized variance can be smaller.
- Table columns: "B-hat error" = mean |B-hat - B_eval|, the absolute error of the calibration
  moment B against the clean evaluation moment B. "Sign (%)" uses a reference that differs by
  block, as each block header states. Clean block: agreement of sign(B-hat) with the clean
  evaluation B_eval. Noise block: agreement with the clean calibration estimate at the same
  rater count, which is the quantity behind the >= 99.3% / >= 93.0% claims. Both sign rates
  for every row are in `artifacts/data/ab_rater_noise.csv` (`b_sign_matches_eval`,
  `b_sign_matches_same_k_clean`). CurveMAE is over d in {0.01, 0.025, 0.05, 0.075, 0.10}.
  "A+B - A-only" is the paired CurveMAE difference with a 5,000-draw path-cluster bootstrap
  95% CI over path-level means. Noise rows are labelled by family (Gauss., t3) and nominal
  scale in group rating SD. "k of K" is the number of gold raters averaged out of those available.
- Aggregation: mean over repeats within path, then equal-weight mean over paths.

## Key results (30% calibration)
- Expert, 1 to 2 gold raters: B error 0.0618, CurveMAE A+B 0.0269 (A-only 0.0393), sign vs B_eval 97.3% -> B error 0.0428, CurveMAE A+B 0.0144 (A-only 0.0278), sign vs B_eval 100.0%.
- Crowd, 1 to 4 gold raters: B error 0.0659, CurveMAE A+B 0.0122 (A-only 0.0135), sign vs B_eval 81.6% -> B error 0.0590, CurveMAE A+B 0.0104 (A-only 0.0119), sign vs B_eval 85.4%.
- Under injected noise with all gold raters, sign agreement with the clean estimate is at least
  99.3% (expert) and
  93.0% (crowd).
- A+B - A-only CurveMAE, expert: Gaussian 0.25 SD -0.00865 [-0.01379, -0.00354];
  0.5 SD -0.00343 [-0.00731, +0.00031] (CI includes 0); 1.0 SD +0.00657 [+0.00465, +0.00857]
  (A+B worse). The t3 noise results are 0.5 SD -0.00551 [-0.01011, -0.00113] and 1.0 SD
  +0.00236 [+0.00041, +0.00427].
- Crowd: Gaussian 0.25 SD -0.00031 [-0.00068, +0.00008]; 0.5 SD +0.00110 [+0.00065, +0.00163];
  t3 0.5 SD +0.00054 [+0.00010, +0.00101]. A+B's advantage is gone at 1.0 SD for experts
  and at 0.5 SD for crowd ratings.

## Sources
- `results/rater_noise/condition_summary.csv` (all values).
- `results/rater_noise/selected_conditions.csv` (asserted equal, max |diff| 0.0e+00).
- `results/rater_noise/run_spec.json`, `artifact_spec.json`,
  `fig_b_rater_noise_stability.pdf` (provenance).

## Caveats the caption MUST include
1. The proxy is a held-out human rater, not a model, and gold labels come from the remaining
   raters.
2. exp26 CurveMAE values come from their own split streams and rater-subset construction. They
   are not comparable with the exp33 CurveMAE of Table 1(b) or `ab_reduction`. Similarly, the
   exp35 tail/outlier analyses use full-pool coefficients, which are a different estimand.
3. Noise scales are nominal (pre-clipping), and evaluation gold is always the clean mean of
   all non-proxy raters.
4. B's sign is stable under the audited noise, but its predictive usefulness is less stable:
   A+B loses its advantage at 1.0 SD (experts) and 0.5 SD (crowd).
