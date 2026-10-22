# bon_inwindow_cases: the four nontrivial exact-BoN points inside d <= 0.10

**Intended display:** appendix, next to `bon_nladder`. The main text should state the exception in one
sentence (e.g. "four SummEval held-out-human paths are inside the window at N = 2; there A+B
overestimates the BoN gain by 0.020 on average versus 0.088 for A-only").

## What the four cases are
All are SummEval held-out-human paths at N = 2 with m = 1,600 items (100 articles x 16
systems): consistency with expert held-out rater 0, 1 and 2, and fluency with expert held-out
rater 2. The proxy is the held-out expert's integer 1-5 score; gold is the mean of the other
two experts (1-5 scale). These are the only nontrivial (N >= 2) points with d_BoN <= 0.10 among
2,101 nontrivial points in 191 paths.

## Why they are in the window
85.4-89.1% of the 1,600 items carry the top proxy score 5 (n_top = 1367, 1376, 1425, 1394).
BoN and the pooled exponential tilt weight items only through their proxy score, so tied items
get equal weight; as N (or t) grows both converge to the uniform distribution on the top block,
and their KL is capped at log(m/n_top) = 0.157, 0.151, 0.116, 0.138. N = 2 already puts
1-(1-s)^2 = 97.9%, 98.0%, 98.8%, 98.3% of its mass on that block, which yields
d = 0.096, 0.093, 0.078, 0.088, well below the atomless value log 2 - 1/2 = 0.193.
At N = 3 the radii are 0.143, 0.138, 0.109, 0.127 and at N = 64 they reach the cap
(0.157, 0.151, 0.116, 0.138).

## Definitions
- d_BoN = KL(p||p0) in nats with exact with-replacement BoN and uniform tie-breaking (A.7).
- ESS = 1/sum p_i^2 (items); RelESS = ESS/m.
- Items at top proxy score = n_top/m, n_top = number of items whose proxy equals the path maximum.
- Actual BoN gain = sum_i p_i j_i - mean(j), SummEval 1-5 units.
- A+B = A sqrt(d) + B d; A-only = A sqrt(d); A_nu = A_nu sqrt(d); empirical fit = descriptive
  least-squares a sqrt(d) + b d on the exact tilt curve over d <= 0.10 of the same pool;
  zero = 0 (its error equals the gain). |error| = |prediction - actual gain|.
- Mean column: unweighted mean of the four cases (= MAE for the error rows).

## Protocol and caveats
- exp20 coefficients are **full-pool**: A, B, A_nu and the empirical fit are computed on the same
  1,600 items on which BoN is evaluated, with no calibration split. This is a same-pool,
  descriptive comparison, not a held-out prediction.
- BoN is **pooled** over the whole item set, not prompt-conditional (contrast with
  `bon_prompt_conditional_scope`).
- **Every predictor overestimates** all four gains (signed errors > 0 for A+B, A-only, A_nu and the
  empirical fit; asserted in bon.py). The empirical fit is closest (MAE 0.00432).
- n = 4 points from 2 dimensions of one dataset; these are scope exceptions, not evidence that
  BoN generally lands in the window.

## Per-case values
| Path | d | ESS | RelESS | top share | log(m/n_top) | gain | A+B pred | err A+B | err A-only | err A_nu | err emp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| consistency__expert__heldout_0 | 0.0959 | 1421.8 | 0.8886 | 85.4% | 0.1574 | 0.2806 | 0.3020 | 0.0213 | 0.0999 | 0.1617 | 0.0050 |
| consistency__expert__heldout_1 | 0.0930 | 1427.0 | 0.8919 | 86.0% | 0.1508 | 0.2510 | 0.2775 | 0.0265 | 0.0987 | 0.0490 | 0.0087 |
| consistency__expert__heldout_2 | 0.0781 | 1457.5 | 0.9109 | 89.1% | 0.1158 | 0.2761 | 0.2941 | 0.0180 | 0.0837 | 0.1422 | 0.0015 |
| fluency__expert__heldout_2 | 0.0880 | 1437.8 | 0.8986 | 87.1% | 0.1378 | 0.1881 | 0.2010 | 0.0129 | 0.0711 | 0.1424 | 0.0020 |

## Key results (asserted)
- Mean d = 0.0887; RelESS 0.889-0.911; ESS 1422-1457.
- MAE: A+B 0.01968, A-only 0.08836, A_nu 0.12381,
  zero 0.24896, empirical fit 0.00432.
- A+B has lower error than A-only in 4/4 cases.

## Units and decimals
d, RelESS and the KL cap: 3 decimals; ESS: integer items; share: % with 1 decimal; gains,
predictions and errors: 4 decimals (SummEval 1-5 units).

## Sources
- `results/bon_scope/bon_gate_points.csv` (d, ESS, gain, coefficients, predictions, errors).
- `data/summeval_long.csv` via `lib/common_experiment_utils.py` `generate_human_paths` (top-score ties, m);
  the same rebuild re-verifies d, ESS, gain, A and B to <= 1e-10.
