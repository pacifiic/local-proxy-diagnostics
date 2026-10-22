# Experiment 29: Same-Prompt Candidate Holdout

This experiment separates within-prompt estimation from unseen-prompt
transfer. For every prompt in each real-LLM candidate pool, 10 of 32 candidates
are used for calibration and the remaining 22 are held out for evaluation.
Candidate roles are assigned from stable hashes of prompt and candidate IDs,
without consulting proxy or gold scores.

The experiment compares `A+B`, `A-only`, reliability-only `A_nu`, empirical
curve and curve-fit baselines, isotonic regression, binned calibration, spline
regression, an MLP, and zero gain. It also reports `A+B-shrunk`, whose `B`
coefficient is shrunk using calibration-only bootstrap uncertainty.

All methods use the same prompt set in each repeat. Evaluation proxy scores are
used to construct the fixed-prompt-KL tilt. Evaluation gold scores are used
only after predictions have been frozen.

## Run

```bash
cd /path/to/local-proxy-diagnostics
bash experiments/exp29_same_prompt_candidate_holdout/run_all.sh
```

Defaults are 20 deterministic candidate-split repeats, 10 calibration
candidates per prompt, 100 bootstrap resamples for `B`, and 12 CPU workers.
Override them without modifying the script:

```bash
N_REPEATS=2 N_BOOTSTRAP=20 JOBS=2 \
OUTDIR=outputs/exp29_smoke \
ANALYSIS_DIR=outputs/exp29_smoke_analysis \
bash experiments/exp29_same_prompt_candidate_holdout/run_all.sh
```

Both output directories must be absent or empty. The runner refuses to
overwrite an existing non-empty experiment directory.

## Main Outputs

- `candidate_split_manifest.csv.gz`: every prompt/candidate role in every repeat
- `aggregate_predictions.csv`: aggregate gain prediction by radius and baseline
- `prompt_predictions.csv.gz`: prompt-level frozen predictions and actual gains
- `prompt_coefficients.csv.gz`: prompt coefficients, bootstrap uncertainty, and skips
- `fit_metadata.csv`: model settings, skip accounting, shrinkage, and leakage audit
- `run_spec.json`: complete input hashes and run specification
- `integrity_audit.json`: independent split, coefficient, prediction, input-hash,
  and raw-gain recomputation audit
- `source_pipeline_audit.json`: prompt-to-generation-to-scoring lineage audit,
  including exact candidate-grid and score-propagation checks
- `table_same_prompt_summary.csv`: local curve MAE summary
- `table_paired_differences.csv`: paired comparisons against `A+B`
- `fig_same_prompt_candidate_holdout.pdf`: compact comparison figure

This is a silver evaluation because the gold score is an independent reward
model, not newly collected human ratings.

The end-to-end prompt, generation, scoring, merge, and execution audit is
documented in `DATA_PIPELINE_AUDIT.md`.
