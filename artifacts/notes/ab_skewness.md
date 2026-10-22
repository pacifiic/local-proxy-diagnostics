# ab_skewness: when the skewness correction in B matters (appendix)

## Placement
Appendix, skewness interpretation. The main text gets one short explanation beside the
coefficient formula. Outputs: table `ab_skewness.tex` and figure `ab_skewness.pdf` (redrawn).

## Definitions
B = E[J-bar Z^2] - (2/3) E[J-bar Z] E[Z^3], with Z the standardized proxy and J-bar the
centred target. The correction share is

    |-(2/3) E[J-bar Z] E[Z^3]| / (|E[J-bar Z^2]| + |-(2/3) E[J-bar Z] E[Z^3]|).

The denominator is the sum of the two components' absolute values, not |B|. This avoids
shares above 100% when the components cancel. Skewness here is skew(r) = E[Z^3].

## Data and protocol (exp19)
- Synthetic family: Z ~ standardized skew-normal(a), a in {-30, -8, -3, -1, 0, 1, 3, 8, 30},
  20,000 items each. The fixed conditional mean is J = 0 + 1.3 Z + 0.6 (Z^2 - 1), and
  r = 5 + 2 Z. Per-setting seeds are in `synthetic_manifest.json`.
- Real: SummEval coherence, full pool of 1,600 summaries, ROUGE-1 or GPT-4o as proxy. These
  are descriptive full-pool coefficients, not calibration-split predictors.
- Closed-form moments were cross-checked against an independent exact-tilt local fit
  (d_max = 5e-4). The max relative B error is 3.1% (< 5%).

## Figure
Redrawn from `results/skewness/skewcorr_summary.csv`. The original
`fig_skewcorr.png` (registered for provenance) rendered J-bar as a broken glyph ("E[]Z]").
This version uses mathtext E[J-bar Z^2]. It also plots the two SummEval proxies as points
rather than vertical lines, and it uses the paper's width and fonts.
(a) Correction share vs skewness. (b) The two components and their sum B for the synthetic
family.

## Key results

| Setting | E[Z^3] | E[J-bar Z^2] | Skew correction | B | Share |
|---|---:|---:|---:|---:|---:|
| $a=-30$ | -0.992 | +0.4308 | +0.4661 | +0.8969 | 51.97% |
| $a=-8$ | -0.938 | +0.4692 | +0.4609 | +0.9301 | 49.56% |
| $a=-3$ | -0.650 | +0.6095 | +0.3944 | +1.0039 | 39.29% |
| $a=-1$ | -0.130 | +1.0605 | +0.1062 | +1.1667 | 9.10% |
| $a=0$ | +0.004 | +1.2256 | -0.0031 | +1.2225 | 0.25% |
| $a=+1$ | +0.138 | +1.4217 | -0.1270 | +1.2947 | 8.20% |
| $a=+3$ | +0.679 | +2.3698 | -0.7732 | +1.5966 | 24.60% |
| $a=+8$ | +0.946 | +2.9262 | -1.1780 | +1.7482 | 28.70% |
| $a=+30$ | +0.958 | +2.8736 | -1.1974 | +1.6762 | 29.41% |
| ROUGE-1 | +0.082 | +0.0304 | -0.0108 | +0.0196 | 26.24% |
| GPT-4o | -0.240 | -0.1533 | +0.0802 | -0.0731 | 34.35% |

Summary: the share is about 0.25% for a symmetric (Gaussian, a = 0) proxy, 29% at a = +30 and
52% at a = -30. For the real SummEval coherence proxies it is 26% (ROUGE-1) and 34% (GPT-4o).
The 0.25%/29%/52% values are synthetic-family settings, not SummEval. The table uses one decimal,
following the build convention (0.3%/29.4%/52.0%). If the prose quotes "0.25%", say it
is the Gaussian synthetic setting.
Sign: the correction is -(2/3) E[J-bar Z] E[Z^3]. For a positively aligned proxy
(E[J-bar Z] > 0, all rows here), positive skew makes it negative (it lowers B), and negative
skew makes it positive.

## Caveats the caption MUST include
1. The share is |correction| / (|E[J-bar Z^2]| + |correction|), not |correction| / |B|.
2. Changing the skew-normal shape also changes higher moments, so this is a family-level
   intuition audit, not an isolated causal effect of skewness.
3. There are only two real proxy paths (SummEval coherence, full pool), so the observed range
   of skewness across datasets is not generalized.

## Verification performed by `ab.py`
- share recomputed from components: 5.0e-16; B = term1 + term2: 4.4e-16;
  term2 = -(2/3) E[J-bar Z] E[Z^3]: 2.2e-16; A = sqrt(2) E[J-bar Z]: 8.9e-16;
  skew vs manifest: 1.4e-15.
