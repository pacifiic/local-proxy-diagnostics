# Experiments

This directory contains the experiment scripts for the local proxy-gain
diagnostic paper. The top-level `README.md` gives the repository overview; this
file maps claims and artifacts to executable scripts and output files.

## Reproduction Quick Start

Run experiments into scratch directories so that reference outputs are not
modified:

```bash
cd <repo-root>

RUN_ROOT=outputs/proxy_gain_repro_$(date +%Y%m%d_%H%M%S)
OUT_ROOT="$RUN_ROOT/all_results_reproduced"
ACCEPT_BOOST_OUT="$OUT_ROOT/12_accept_boost"
ROBUSTNESS_OUT="$OUT_ROOT/13_robustness"
export OUT_ROOT ACCEPT_BOOST_OUT ROBUSTNESS_OUT

bash scripts/run_all_reproduced.sh
bash scripts/run_accept_boost_experiments.sh
bash scripts/run_robustness_experiments.sh
```

Large restored outputs are stored in losslessly compressed form.
The runners refuse to overwrite an existing output directory unless
`ALLOW_OVERWRITE=1` is set, so scratch output paths are the safest reproduction
workflow.

No new LLM API calls are needed to reproduce the paper tables and figures. The
GPT-4o proxy analysis uses `data/model_proxy_4o_aggregated.csv`.

After experiments finish, regenerate publication tables and figures by
pointing the artifact builder at the scratch output roots:

```bash
python make_artifacts.py all \
  --results-root "$OUT_ROOT" \
  --accept-root "$OUT_ROOT/12_accept_boost" \
  --robustness-root "$OUT_ROOT/13_robustness"
```

Use `python make_artifacts.py all --dry-run` to see the individual builder
commands without writing files.

## Result Map

| Result | Main script or runner | Saved output |
|---|---|---|
| Multi-dataset prospective prediction | `run_all_reproduced.sh` -> `07_prospective_prediction/01_prospective_prediction.py`; table assembled by `artifact_tools/make_robustness_tables.py` | `$OUT_ROOT/07_prospective_prediction/exp27_*`; `artifacts/tables/table_multidataset_prospective.*` |
| OOD transfer | `run_robustness_experiments.sh` -> `10_ood_transfer/01_ood_transfer_prediction.py`; table assembled by `artifact_tools/make_robustness_tables.py` | `$OUT_ROOT/13_robustness/ood_*_lambda_*`; `artifacts/tables/table_ood_lambda_sweep.*` |
| BoN scope/bridge table | `run_robustness_experiments.sh` -> `08_bon_bridge/01_bon_bridge.py` | `$OUT_ROOT/13_robustness/bon_bridge_*`; `artifacts/tables/table_bon_scope.*` |
| Prompt-conditional BoN scope audit | `15_prompt_conditional_bon_scope/01_prompt_conditional_bon_audit.py` | user-chosen output root with `prompt_conditional_bon_*` summary files |
| Direct coefficient-law validation | `run_all_reproduced.sh` -> `03_direct_tilt_validation`, `04_cross_dataset`, `05_wmt_mtbench` | `$OUT_ROOT/03_direct_tilt_validation/`; `04_cross_dataset/`; `05_wmt_mtbench/` |
| GPT-4o external-proxy diagnostics | `run_all_reproduced.sh` -> `09_gpt_proxy_analysis/03_gpt_proxy_diagnostics.py` | `$OUT_ROOT/09_gpt_proxy_analysis/exp30_gpt_proxy_diagnostics/` |
| Local-radius d-sweep | `run_accept_boost_experiments.sh` -> `12_accept_boost/01_d_sweep_prediction.py` | `$OUT_ROOT/12_accept_boost/d_sweep_*`; `artifacts/tables/table_d_sweep.*` |
| Curve residual decomposition | `run_accept_boost_experiments.sh` -> `12_accept_boost/02_curve_residual_breakdown.py` | `$OUT_ROOT/12_accept_boost/residual_*`; `artifacts/tables/table_curve_residuals.*` |
| Calibration-size sensitivity | `run_accept_boost_experiments.sh`; extended WMT/OpenMEVA runs in `run_robustness_experiments.sh` | `$OUT_ROOT/12_accept_boost/cal_size_*`; `13_robustness/cal_size_*`; `artifacts/tables/*calibration_size*` |
| GPT-4o selective prediction gate | `run_accept_boost_experiments.sh` -> `12_accept_boost/04_selective_prediction_gate.py` | `$OUT_ROOT/12_accept_boost/selective_gpt4o/`; `artifacts/tables/table_selective_gpt4o.*` |
| Proxy perturbation controls | `run_all_reproduced.sh` -> `11_proxy_perturbation/01_proxy_perturbation_baselines.py` | `$OUT_ROOT/11_proxy_perturbation/` |
| Tail/outlier robustness audit | `14_tail_outlier_robustness/01_tail_outlier_robustness_audit.py` | user-chosen output root with `tail_outlier_robustness_*` summary files |
| Prompt-split candidate-pool BoN/reranking | `16_llm_reranking_workflow/01_prompt_split_bon_reranking.py` | user-chosen output root with `prompt_split_bon_reranking_*` files |
| Calibration-baseline benchmark | `17_calibration_baseline_benchmark/01_calibration_baseline_benchmark.py` | user-chosen output root with `calibration_baseline_*` files |
| Strong-baseline calibration-size sweep | `18_strong_baseline_calibration_sweep/01_calibration_size_strong_baselines.py` | user-chosen output root with `calibration_size_strong_baselines_*` files |

## Directory Guide

| Directory | Role | Notes |
|---|---|---|
| `data_preparation/` | Converts public/raw sources into analysis-ready CSV files | Produces admitted repeated-rating slices; this is not full-benchmark coverage for every source dataset. |
| `01_icc_decomposition/` | Reliability and BoN utility background | Appendix support. Some scripts write intermediate `results/` paths; `run_all_reproduced.sh` isolates and copies these outputs safely. |
| `02_corrected_coefficients/` | Corrected coefficient and related comparisons | Appendix support and historical coefficient checks. |
| `03_direct_tilt_validation/` | SummEval exact-tilt validation and robustness | Direct validation of the coefficient-law approximation. |
| `04_cross_dataset/` | OpenMEVA, QGEval, RewardBench direct checks | Appendix cross-source evidence; not all are prospective prediction tasks. |
| `05_wmt_mtbench/` | WMT MQM/pSQM and MT-Bench utility analyses | WMT repeated-rating paths and split-half Bradley-Terry checks. |
| `06_supplemental_audits/` | Monotonicity, scaling, and intervention audits | Supplemental monotonicity and measurement audits. |
| `07_prospective_prediction/` | Main prospective prediction experiment | Produces the per-path records used for the main multi-dataset prospective table. |
| `08_bon_bridge/` | Best-of-N vs exponential-tilt bridge | Used both in the main runner and in robustness scope checks. |
| `09_gpt_proxy_analysis/` | GPT-4o external-proxy diagnostics | `01_generate_proxy_scores.py` is optional/API-based; `02_remap_summeval_proxy_scores.py` repairs raw-call item ids when raw logs are available; paper reproduction uses the included aggregate CSV. |
| `10_ood_transfer/` | Cross-slice transfer prediction | Used for OOD transfer and lambda-sweep robustness. |
| `11_proxy_perturbation/` | Shuffle/random/noisy proxy controls | Appendix robustness. |
| `12_accept_boost/` | Local-radius and selective-use stress tests | d-sweep, residuals, calibration-size curves, and GPT-4o selective gate. |
| `14_tail_outlier_robustness/` | Boundedness and tail/outlier stress test | Empirical audit of coefficient stability under winsorization, rank-normalization, and controlled score outliers. |
| `15_prompt_conditional_bon_scope/` | Prompt-conditional Best-of-N scope audit | Quantifies how discrete per-prompt BoN differs from the smooth raw-score exponential tilt target. |
| `16_llm_reranking_workflow/` | Prompt-split candidate-pool BoN/reranking | Estimates coefficients on calibration prompts and evaluates fixed-proxy BoN gains on held-out prompts. |
| `17_calibration_baseline_benchmark/` | Calibration baseline comparison | Compares `A+B` with empirical curve, curve-fit, correlation, isotonic, reliability, and zero baselines. |
| `18_strong_baseline_calibration_sweep/` | Strong-baseline calibration-size sweep | Tests whether flexible calibration baselines remain strongest as calibration data becomes scarce. |

## Data and Scope Notes

The experiments use only the slices that satisfy the repeated-rating design
requirements of the diagnostic. In particular, the paper should be read as
using admitted repeated-rating slices from SummEval, WMT MQM, and OpenMEVA-MAGS,
not as claiming full benchmark coverage over every available item in those
datasets.

The GPT-4o proxy artifact is aggregated at `data/model_proxy_4o_aggregated.csv`.
Regenerating raw model calls requires API access and is intentionally not part
of the default reproduction path. If raw GPT call logs and the prepared
SummEval prompt-item CSV are available, `09_gpt_proxy_analysis/02_remap_summeval_proxy_scores.py`
repairs item ids by matching candidate-summary text back to
`model_annotations.aligned.jsonl`.

Prepare raw data from the top-level `README.md` before running the full
pipeline. Two inputs are manual: SummEval's public
`model_annotations.aligned.jsonl` file must be placed at
`data/model_annotations.aligned.jsonl`, and QGEval xlsx files must be placed at
`data/QGEval/annotation_result/annotator{1,2,3}.xlsx`,
`data/QGEval/scores.xlsx`, and `data/QGEval/metric_result.xlsx`. OpenMEVA,
RewardBench, WMT MQM, and MT-Bench are prepared by the documented scripts or
clone command.

## Output and Metadata

Key prospective and accept-boost scripts write run-spec JSON files recording the
main randomization and protocol choices. Most output tables are CSV plus
Markdown/LaTeX table exports. The figure builder fixes `SOURCE_DATE_EPOCH=0`
and normalizes PDF `CreationDate`/`ModDate` fields after writing figures.

Data-preparation scripts for OpenMEVA, MT-Bench, and RewardBench accept explicit
HuggingFace revisions and write manifest/run-spec files. RewardBench margin
geometry is evaluated in pairwise-available mode so a missing margin for one
model does not force every other model in the subset to be dropped.
