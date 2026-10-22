# dpo_table3: LoRA-DPO bridge (Table 3)

**Placement.** Main text, Table 3 (CAMERA_READY_PLAN.md §5). If space is tight, move the table to
the appendix but keep the setup, both gains, the A+B vs A-only numbers and the estimand caveat in
the main text. Explain the realized vs idealized-tilt distinction in adjacent prose. Link the full
comparator table (`dpo_comparators`); this restricted four-predictor table must not be read as a
claim that A+B is the best realized-gain predictor.

**What it shows.** Three LoRA-DPO training seeds; checkpoints gated by mean prompt KL <= 0.10
(4/3/3 checkpoints for seeds 0/1/2). Top block: KL radius and the two silver-gain targets.
Bottom block: MAE of four calibration-only predictors against each target. Lower is better.

**Setup.** Policy Qwen2.5-3B-Instruct, LoRA-DPO (beta 0.1, lr 1e-5, effective batch 16, LoRA
rank/alpha/dropout 16/32/0.05) on 968 nonzero-gap preference pairs from 1,000 UltraFeedback prompts
(8 candidates each); chosen/rejected = highest/lowest Skywork-Reward-Llama-3.1-8B-v0.2 score (the
optimized proxy r). Frozen silver target J = OpenAssistant/reward-model-deberta-v3-large-v2, never
used in training. Training, calibration (100 prompts, 99 usable) and evaluation (200 prompts) prompt
sets are disjoint. Calibration and evaluation pools: 32 fixed candidates per prompt from the reference
policy (temperature 0.9, top-p 0.95, <= 320 new tokens). Seeds share data and pools; only training
randomness differs.. The numbers here are recomputed from the saved CSVs.

**Definitions and units.** All gains are in silver-target (DeBERTa reward) units.
- Per evaluation prompt x: w_x,i proportional to exp(log p_DPO(y_i|x) - log p_ref(y_i|x)) over its
  32 candidates; d_x = KL(w_x || Unif_32).
- Realized (likelihood-ratio) gain: sum_i w_x,i J_i minus the uniform pool mean of J, averaged over
  the 200 evaluation prompts (`dpo_realized`).
- Idealized tilt gain: silver gain of the proxy-score exponential tilt of the same pool at the same
  d_x (`frontier`).
- Predictors: A+B = mean_x(A sqrt(d_x) + B d_x); A-only = mean_x(A sqrt(d_x));
  A_nu = mean_x(A_nu sqrt(d_x)) (variance-ratio surrogate, sqrt(2) Var(J)/SD(r) per calibration prompt,
  averaged; no rater reliability exists for a deterministic silver target); Zero = 0. Common
  calibration-only coefficients A = 0.2268, B = -0.0465, A_nu = 0.2599
  (`actual_dpo/coefficients.json`, 99/100 prompts; one calibration prompt has zero proxy and silver
  variance). This builder re-derives every saved checkpoint prediction from these coefficients and
  the saved prompt radii (max deviation < 1e-12).
- KL radius row: per checkpoint, the mean of the 200 prompt radii d_x; then averaged within seed.

**Aggregation.** Per checkpoint, error = |prompt-averaged prediction - prompt-averaged target|.
Average over the retained checkpoints within each seed (4/3/3), then report mean and sample SD
(ddof = 1) across the three seed means. The SD is seed-to-seed spread, not a confidence interval.

**Protocol / gate.** Selected seed-checkpoints: seed 0 steps 64/90/120/150; seeds 1 and 2 steps
64/90/120. Gate = checkpoint mean prompt KL <= 0.10 (`prompt_radius_audit/run_spec.json`). The
`three_seed_canonical/run_spec.json` and exp34 run_spec state "checkpoint median d <= 0.10"; this
builder verifies that the median rule selects exactly the same ten checkpoints (the eight excluded
checkpoints have median d 0.111-0.248). The gate uses no
silver label, gain or error, but it is transductive in the reported radius. Not every prompt is
local: across the 2,000 retained prompt-checkpoint cells, d_x median/p90/p95/p99/max =
0.0257/0.0919/0.1224/0.1867/0.4668, and 8.7% exceed 0.10.

**Sources.** `results/actual_dpo/three_seed_canonical/checkpoint_predictions.csv`
(checkpoint rows, A+B / A-only / A_nu / Zero, `coefficient_or_fixed_baseline`);
`.../prompt_radius_audit/{run_spec.json, prompt_radius_cells.csv, prompt_radius_summary.csv}`;
`.../window_summary_mean_sd.csv` (`strict_local_d_le_0p10` rows, used only as an exact cross-check);
`results/actual_dpo/coefficients.json`. NOT used: the root `summary_mean_sd.csv`,
`seed_level_mae.csv`, `error_decomposition_summary.csv` (historical six-checkpoint scope, A+B
0.02294), `exp09_final_summary.csv`, `kl_sweep*.csv` (old single runs).

**Key results** (mean +/- sample SD over 3 seed means).
- Mean KL radius 0.03968 +/- 0.00136; idealized-tilt gain 0.04197 +/- 0.00017; likelihood-ratio gain 0.02081 +/- 0.00062;
  ratio of means 0.496; idealized minus realized 0.02116 +/- 0.00074.
- Realized-gain MAE: A+B 0.01721 +/- 0.00116, A-only
  0.01905 +/- 0.00120, zero 0.02081 +/- 0.00062,
  A_nu 0.02487 +/- 0.00129.
- Frontier MAE: A+B 0.00395 +/- 0.00047, A-only
  0.00210 +/- 0.00053, A_nu 0.00371 +/- 0.00061,
  zero 0.04197 +/- 0.00017.
- A-only minus A+B realized MAE per seed 0/1/2: 0.00187/0.00177/0.00189; A+B < A-only at 10/10 checkpoints.
- A-only has lower frontier MAE than A+B in 3/3 seeds.

**Reporting details.**
1. Actual LoRA-DPO training, but the reported gain is a self-normalized DPO/reference likelihood-ratio reweighting of FIXED candidate pools (32 per prompt) relative to uniform, scored by a frozen silver target model. It is not fresh generation, not human-rated gain and not an exact full-policy expectation. Checkpoints are selected by checkpoint mean prompt KL <= 0.10; the exp28/exp34 run specifications state the earlier rule "checkpoint median d <= 0.10", which selects the same ten checkpoints.
2. Mean +/- sample SD across three training seeds after averaging 4/3/3 retained checkpoints per seed;
   gains in silver-target units; lower MAE is better.
3. The two target columns: realized = likelihood-ratio reweighting; idealized = proxy exponential tilt
   of the same pool at the same d_x.
4. Restricted predictor set; pointer to the full comparator table (`dpo_comparators`).

**Caveats for adjacent prose.**
- A-only is closer to the idealized frontier (0.00210 vs 0.00395). At all 10 checkpoints the ordering is
  realized gain < A+B prediction < A-only prediction < idealized gain (asserted here). Both predictors
  underpredict the frontier; A+B's larger underprediction (its negative B term) moves it toward the
  realized gain, so its realized-gain advantage partly reflects error cancellation against the
  optimizer shortfall (realized gain is about half the idealized gain), not a better frontier model.
- Other comparators reach lower realized MAE (Spearman A-only variant
  0.01378, exp34 prompt-rank binned 0.01096); see `dpo_comparators`. No universal-best claim.
- Three seeds: descriptive consistency, not significance. Common coefficients applied to promptwise
  radii are an empirical pooled-coefficient approximation; 8.7% of prompt cells lie beyond 0.10.
- Do not merge numerically with the exp37 reward-model results (different target and mechanism).

**Formatting.** Five decimals for every gain, KL radius and MAE, fixed within the table, matching the saved results; the README's 4-decimal endpoint rule targets human-score endpoint MAE, and
the 3-decimal KL rule would show the radius as 0.040 +/- 0.001. `dpo_table3.csv` holds unrounded
per-seed values; `dpo_table3_checkpoints.csv` the ten checkpoint rows.
