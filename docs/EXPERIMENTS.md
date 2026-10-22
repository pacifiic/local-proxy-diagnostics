# Experiment Guide

All experiment implementations live under `experiments/`, use the shared code
in `lib/`, and read prepared inputs from `data/`. Stable experiment IDs are kept
because saved run specifications refer to them.

Run `python tools/restore_artifacts.py` before accessing expanded inputs or large
prediction files. Write new runs to a fresh directory rather than into
`results/`, which contains the published reference outputs.

## Analysis Map

| Question | Implementation | Saved results |
|---|---|---|
| Reliability and coefficient construction | `experiments/01_icc_decomposition/`, `experiments/02_corrected_coefficients/` | `results/01_icc_decomposition/`, `results/02_corrected_coefficients/` |
| Direct tilt validation | `experiments/03_direct_tilt_validation/` | `results/03_direct_tilt_validation/` |
| Prospective prediction | `experiments/07_prospective_prediction/` | `results/07_prospective_prediction/` |
| Calibration baselines | `experiments/17_calibration_baseline_benchmark/`, `experiments/exp21_direct_calibration/`, `experiments/exp32_evaluation_aware_ab/`, `experiments/exp33_calibration_only_flexible_baselines/` | `results/direct_calibration_strict/`, `results/evaluation_aware_ab/` |
| OOD transfer | `experiments/10_ood_transfer/`, `experiments/exp24_direct_calibration_followup/` | `results/10_ood_transfer/`, `results/direct_calibration/` |
| Proxy perturbation and tail robustness | `experiments/11_proxy_perturbation/`, `experiments/14_tail_outlier_robustness/` | `results/11_proxy_perturbation/`, `results/tail_outlier_robustness/` |
| Exact and prompt-conditional Best-of-N | `experiments/08_bon_bridge/`, `experiments/exp20_bon_window_gate/`, `experiments/exp23_mixture_ranktilt/`, `experiments/15_prompt_conditional_bon_scope/` | `results/bon_scope/`, `results/rank_selection_scope/`, `results/prompt_bon/` |
| LoRA-DPO | `experiments/exp09_dpo_validation/`, `experiments/exp28_prompt_conditional_calibration/`, `experiments/exp34_calibration_only_dpo_baselines/` | `results/actual_dpo/`, `results/dpo_flexible_calibration_only/` |
| External learned proxies | `experiments/exp01_gpu_pipeline/`, `experiments/exp06_rm_judge_proxy_prediction/`, `experiments/exp37_reward_model_coefficients/` | `results/open_proxy/`, `results/reward_model_coefficients/` |
| B stability, rater noise, and skewness | `experiments/exp19_skewness_intuition/`, `experiments/exp22_b_stability/`, `experiments/exp26_b_rater_noise_stability/` | `results/skewness/`, `results/b_stability/`, `results/rater_noise/` |
| Richer local functional forms | `experiments/exp27_cubic_moment_term/`, `experiments/exp36_dominant_local_law/` | `results/cubic_moment_term/`, `results/dominant_local_law/` |
| WMT input lineage | `experiments/wmt_input_lineage_pairing/` | `results/wmt_input_lineages/`, `verification/pipeline/` |

## CPU Verification

```bash
python tools/restore_artifacts.py
python tools/verify_release.py --fresh-splits
python verification/pipeline/verify_pipeline_data_integrity.py \
  --report verification/pipeline_integrity.json
```

## Full CPU Workflow

The retained runners cover the primary analyses and robustness checks:

```bash
OUT_ROOT=outputs/full_run bash scripts/run_all_reproduced.sh
ACCEPT_BOOST_OUT=outputs/local_radius bash scripts/run_accept_boost_experiments.sh
ROBUSTNESS_OUT=outputs/robustness bash scripts/run_robustness_experiments.sh
```

Inspect each script before running because some auxiliary datasets must be
obtained from their providers. Generated publication artifacts can be rebuilt
from the included saved results without those raw downloads.

## Three-Seed DPO

The DPO study uses UltraFeedback prompts, a Qwen2.5-3B-Instruct generator, a
Skywork reward-model proxy, and a separate OpenAssistant DeBERTa silver target.
The scored pool and preference pairs are in `data/dpo_candidate_pool/`.

```bash
conda env create -f environments/gpu.yml
conda activate local-proxy-gpu
ROOT=outputs/dpo PY=python GPUS="0 1" SEEDS="0 1 2" \
  bash experiments/exp09_dpo_validation/run_exp09_multiseed.sh
```

This performs real LoRA-DPO training. Evaluation uses a finite-candidate
likelihood-ratio diagnostic rather than fresh-generation human ratings.

## WMT Inputs

`data/wmt_mqm_long.csv` is the corrected 141-path input.
`data/wmt_mqm_long_main_protocol.csv` is the retained 139-path control. The
raw-data direct-validation slice is a separate analysis and has different path
counts.
