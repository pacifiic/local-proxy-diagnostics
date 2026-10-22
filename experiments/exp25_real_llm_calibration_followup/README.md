# exp25: Real-LLM Direct-Calibration Follow-up

This CPU-only follow-up compares the theorem-derived `A+B` predictor with
empirical curve fitting and direct one-dimensional calibration on saved,
real LLM-generated candidate pools. It does not regenerate candidates or
rescore them.

## Experiments

`01_within_candidate_pool.py` uses the Qwen2.5-7B SummEval candidate pool from
exp05. It evaluates nested prompt-level calibration fractions of
5%, 10%, 20%, 30%, and 50% over 200 repeated splits. The local curve grid is
`d={.01,.025,.05,.075,.10}`; `d={.15,.20}` is an extended-radius scope audit.
The geometry is the exp05 common-temperature prompt-conditional tilt.

`02_cross_model_transfer.py` uses matched UltraFeedback prompt splits from
exp09 and exp11. It evaluates Qwen2.5-3B to Zephyr-7B and the reverse direction.
`source_only` fits every predictor using source-model calibration candidates
only. `target_calibrated` is the within-target reference. The geometry matches
exp09/exp11: every prompt is evaluated at the same prompt-level KL radius.
Prompts with tied proxy maxima that cannot attain a requested radius saturate
at the exp09/exp11 cap, and `transfer_reachability.csv` reports this explicitly.

All spline and binned-calibration hyperparameters are selected by
calibration-only `GroupKFold`, with `prompt_id` as the group. Isotonic and MLP
models likewise receive no evaluation gold labels. Evaluation proxy scores are
used only to construct the requested KL tilt.

The independent gold scorer in these experiments is a reward model from a
different family, not a human annotator. These results must be described as
model-gold validation on real LLM-generated candidates.

## Run

```bash
JOBS=16 N_REPEATS=200 DATE_TAG=rerun \
  bash experiments/exp25_real_llm_calibration_followup/run_all.sh
```

The scripts refuse to overwrite non-empty output directories. Override
`WITHIN_OUT` and `TRANSFER_OUT` to choose fresh locations.

## Outputs

Each run records raw prediction cells, prompt-level cells, fit metadata, input
SHA-256 hashes, split manifests, run specifications, compact summaries, and
PDF/PNG figures. `03_audit_and_artifacts.py` hard-fails on duplicate keys,
missing baselines, non-finite predictions, split-accounting errors, or an
unexpected number of rows.

`04_data_layer_audit.py` additionally permutes evaluation gold labels and
requires every frozen prediction to remain unchanged while the measured gain
changes. It also reproduces the original exp05 A+B predictions and exp11
Zephyr frontier to numerical precision.
