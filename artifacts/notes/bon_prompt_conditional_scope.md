# bon_prompt_conditional_scope: prompt-conditional exact BoN

## Publication use
Appendix table and figure for the prompt-conditional BoN scope audit. The table reports
path-macro summaries, and the figure shows mean prompt KL and pooled raw-score tilt
reachability as Best-of-N increases.

## Definitions
- Prompt-conditional exact BoN uses replacement sampling and uniform tie-breaking within
  each prompt. The prompt reference distribution is uniform over its candidates.
- Mean and median prompt KL are first computed within each path and then summarized with
  equal path weight.
- Prompts KL <= .10 is the path-macro share of prompts within the checked local window.
- Pooled tilt reachable is the share of paths whose mean prompt KL is below both 0.25 and
  the pooled raw-score tilt supremum log(m/n_top).
- Mean BoN gain is measured in each dataset's native target-score units.

## Main observations
At N = 2, mean prompt KL is 0.144 for SummEval and
0.131 for OpenMEVA-MAGS. Only
12.5% and
11.7% of prompts, respectively,
remain inside d <= 0.10. Pooled raw-score tilt reachability is
100.0% and
100.0% at N = 2. At N = 4,
only 6 of 32 SummEval paths remain reachable;
no path is reachable for N >= 8.

Fully tied prompts remain at zero KL for every N and explain the small in-window plateau
at larger N. This is a descriptive comparison between prompt-conditional selection and a
pooled raw-score tilt, not a theorem validation for BoN.

## Sources and checks
- Prepared scores: `data/summeval_long.csv`, `data/openmeva_long.csv`.
- Experiment implementation: `experiments/15_prompt_conditional_bon_scope/01_prompt_conditional_bon_audit.py` and `lib/common_experiment_utils.py`.
- Current saved results: `results/prompt_bon/{summeval,openmeva_mags}/`.
- The builder re-derives reachability from prepared scores and checks all unaffected fields
  against a compact regression fixture in `verification/fixtures/prompt_bon_reachability_reference`.
