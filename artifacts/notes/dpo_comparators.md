# dpo_comparators: full DPO comparator table

**Placement.** Appendix, linked from Table 3 (`dpo_table3`). Information settings differ by panel;
compare rows within a panel, and across panels only with the information difference stated.

**What it shows.** MAE of 46 predictor variants against the idealized tilt frontier and against the
realized likelihood-ratio gain, on the same ten mean-KL-gated checkpoints as Table 3. Cells are
mean +/- sample SD over three seed means (4/3/3 checkpoints per seed). The last column applies only to
panel (c): the aggregate projection-minus-direct prediction gap for the same fitted h(r) (`--` elsewhere
because those rows have no paired projection).

**Panels (information settings).**
- (a) Pre-pool: uses only the 100 rated calibration prompts (3,200 candidates) and each evaluation
  prompt's measured d_x; evaluation candidates are never loaded. A+B, A-only, A_nu, Zero: common moment
  coefficients (exp09 fit, 99/100 prompts). Spearman A-only variant: A_S sqrt(d_x) with
  A_S = mean over calibration prompts of sqrt(2) SD(J) Spearman(r, J) (an A-only variant using Spearman
  correlation; exp09 `11_combined_analysis.py`). Frozen curves (exp34): flexible calibrator fitted and
  integrated on calibration candidates, prompt-averaged gain curve frozen, evaluated at d_x. Prompt-rank =
  within-prompt rank coordinate.
- (b) Target-pool-aware direct: conditional mean h(r) fit on calibration prompts only, then exactly
  integrated under raw-proxy tilt over each evaluation prompt's proxy scores at d_x (evaluation proxy
  geometry used; evaluation silver labels not used).
- (c) Target-pool-aware projection: prompt-specific moments of (h(r_eval), r_eval) give A_h, B_h;
  prediction A_h sqrt(d_x) + B_h d_x. Same h and geometry as (b).
- (d) Cross-fit diagnostics: h fitted by cross-fitting across evaluation prompts, i.e. using evaluation
  silver labels; descriptive distribution-shift diagnostics, not deployable predictors.

**Sources and de-duplication.**
- exp28 strict rows: `results/actual_dpo/three_seed_canonical/window_summary_mean_sd.csv` (`strict_local_d_le_0p10`, 21 rows: 5 coefficient, 8 `calibration_only`, 8 cross-fit).
- Pool-aware follow-up: `results/actual_dpo/target_pool_aware_ab_direct/summary_mean_sd.csv` (12 models x projection/direct).
- exp34 pre-pool: `results/dpo_flexible_calibration_only/summary_mean_sd.csv` (`strict_local_d_le_0p10`, 13 rows; original outputs restored 2026-10-12, audit PASS).
- A+B/A-only/A_nu/Zero are identical in exp28 and exp34 (checked < 1e-12) and shown once in (a). The
  8 exp28 `calibration_only` isotonic/spline rows equal the follow-up's direct rows (< 1e-12) and are
  shown once in (b). `dpo_comparators.csv` records the duplicate source in `identical_row_in`.
- The follow-up's `selected_checkpoints.csv` is the same 10 checkpoints as Table 3. exp34's run_spec
  states "checkpoint median d <= 0.10" (same ten checkpoints).

**Key results.**
- Best realized MAE in exp28 strict rows: Spearman A-only variant
  0.01378 +/- 0.00113.
- Best realized MAE in exp34 pre-pool rows: prompt-rank binned
  0.01096 +/- 0.00107 (frontier MAE
  0.01020).
- Lowest realized MAE in the whole table: Prompt-z binned, panel (b),
  0.01072; lowest frontier MAE: Prompt-centered isotonic, panel
  (b), 0.00059.
- 14 of 46 rows have lower realized MAE than A+B (0.01721); by panel
  a: 6/14, b: 4/12, c: 4/12, d: 0/8.
- Projection-direct aggregate gap <= 0.00037 for every conditional-mean model (max: z-score binned).

**Reporting details.**
1. Actual LoRA-DPO training, but the reported gain is a self-normalized DPO/reference likelihood-ratio reweighting of FIXED candidate pools (32 per prompt) relative to uniform, scored by a frozen silver target model. It is not fresh generation, not human-rated gain and not an exact full-policy expectation. Checkpoints are selected by checkpoint mean prompt KL <= 0.10; the exp28/exp34 run specifications state the earlier rule "checkpoint median d <= 0.10", which selects the same ten checkpoints.
2. Mean +/- sample SD over three seed means; same ten checkpoints as Table 3; lower is better.
3. The information setting of each panel, especially that (b)/(c) use evaluation proxy scores and (d)
   uses evaluation silver labels.
4. Realized-gain MAE mixes predictor error with the optimizer shortfall (realized gain is about half the
   idealized gain, gap 0.02116), so a lower realized MAE can arise from cancellation; frontier MAE and
   realized MAE answer different questions. No universal-best claim.

**Caveats.**
- The CV-tuned binned and spline rows use GroupKFold over 100 prompt groups. Reproduce them
  with the pinned NumPy and scikit-learn versions in `docs/ENVIRONMENT.md`; the published
  reference outputs are retained in `results/`.
- Three seeds: descriptive; per-seed SDs are not confidence intervals.
- The MLP rows are exploratory (exp34: fixed 1-16-16-1 ReLU MLP, alpha 1e-3, 1,000 iterations, seed 0;
  no tuning).

**Formatting.** Five decimals throughout (as in `dpo_table3`). Row order is by method family, not by
MAE; no bolding, to avoid implying a winner.
