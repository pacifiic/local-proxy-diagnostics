# rm_endpoint_mae: exp37 endpoint MAE at d = 0.10

**Placement.** Appendix; main-eligible as a one-line or small-table support for the worked example
(CAMERA_READY_PLAN.md §8.4). `rm_endpoint_mae_by_dimension.tex` is the 12-row appendix variant.

**What it shows.** Prospective endpoint MAE at KL d = 0.10, for predictions made from 480 calibration
items and scored on the 1,120 held-out items, in human-score units of each dimension.

**Definitions.** A-only = A sqrt(d); A+B = A sqrt(d) + B d (moment coefficients from calibration items);
calibration empirical fit = the same two-term form least-squares fitted to the exact calibration-split
tilt curve. Target = exact raw-score exponential-tilt gain on evaluation items with human gold (mean of
three expert and five crowd ratings, per dimension). This is endpoint MAE, not CurveMAE.

**Aggregation.** Per model/dimension: mean absolute error over 50 item splits (seed 0, 480/1,120,
identical across models). Main table: unweighted mean over the four dimensions (= mean over 200 splits).
All 200 dimension/repeat splits are valid for all three models (600 model-splits, no exclusions). The
saved summary means are re-checked against the saved split rows (< 1e-12); nothing is re-analysed.

**Sources.** `results/reward_model_coefficients/canonical/analysis/summary.csv`
and `.../analysis/prospective_paths.csv`; `.../analysis/run_spec.json` (NumPy 2.3.5 environment).

**Key results.**
- Skywork: A-only 0.015533, A+B 0.011262, empirical fit 0.011083; reduction 27.5%
- ArmoRM: A-only 0.013607, A+B 0.010459, empirical fit 0.010513; reduction 23.1%
- DeBERTa: A-only 0.015081, A+B 0.011291, empirical fit 0.011266; reduction 25.1%
- A+B < A-only in all 12 model/dimension endpoint cells, and in 48/48 cells including radii
  0.025/0.05/0.075/0.10.
- Calibration empirical fit < A+B in 7/12 endpoint cells (Skywork coherence, ArmoRM coherence, DeBERTa coherence, Skywork consistency, DeBERTa consistency, Skywork fluency, DeBERTa relevance).

**Reporting details.** d = 0.10 endpoint MAE (not CurveMAE), equally averaged over four SummEval
dimensions and 50 item splits per dimension; 480 calibration / 1,120 evaluation items; item-held-out,
not article-held-out; shared truncated article context; human-score units; raw-score exponential tilt,
not RL training. Point estimates only.

**Caveats.**
- Not a significance claim and not a universal ranking; empirical fit remains competitive (wins 7/12).
  Split SDs (in `rm_endpoint_mae_by_dimension.csv`) overlap and are not independent-sample confidence
  intervals: splits reuse the same 1,600 items.
- A_nu is omitted from the tables. Its d = 0.10 MAE is in the CSV (Skywork 0.0445, ArmoRM 6.3655, DeBERTa 0.0214); the ArmoRM value is a
  proxy-scale artifact (ArmoRM's scalar reward has SD 0.022, so the variance-ratio surrogate is not
  comparable), not a meaningful forecast. If shown, it needs that caveat.
- These shared-context Skywork scores are not the earlier open-proxy Skywork scoring run
  (`proxy_open_judge_rm`); the near-equal Skywork A+B values there (0.01126) are a coincidence of two
  different runs.
- The 4-decimal display ties A+B and empirical fit for ArmoRM/DeBERTa; differences are below 0.0001.
