# Experiment 17: Calibration-Baseline Benchmark

This experiment compares the theorem-based `A+B` predictor against natural
calibration baselines under the same prospective split protocol used by the
main prediction experiment.

## Baselines

- `A+B`: moment-formula predictor from the paper.
- `A-only`: first-order moment predictor without curvature.
- `A_nu`: reliability-only surrogate.
- `empirical_curve`: exact finite-tilt gain curve measured on calibration
  items and copied directly to evaluation items.
- `empirical_fit`: two-parameter `alpha sqrt(d) + beta d` fit to the
  calibration exact curve.
- `pearson_corr`: correlation-scaled first-order predictor,
  `sqrt(2) sigma_J corr(J,r) sqrt(d)`.
- `spearman_corr`: same form using Spearman rank correlation.
- `isotonic_calibration`: fits an increasing isotonic calibration map
  `h(r) ~= E[J | r]` on calibration items, then integrates `h` under the
  evaluation proxy tilt.
- `zero`: zero-gain baseline.

The two empirical curve baselines test whether moment
formulas instead of directly estimating the curve on the calibration split? The
correlation baselines test whether average proxy-human correlation is enough.
The isotonic baseline tests a stronger nonparametric metric-calibration
alternative.

## Example Commands

SummEval held-out human proxy:

```bash
python experiments/17_calibration_baseline_benchmark/01_calibration_baseline_benchmark.py \
  --dataset-name summeval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --fit-max-d 0.10 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --outdir outputs/calibration_baseline_benchmark/summeval_human
```

SummEval GPT-4o external proxy:

```bash
python experiments/17_calibration_baseline_benchmark/01_calibration_baseline_benchmark.py \
  --dataset-name summeval_gpt4o \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension \
  --proxy-csv data/model_proxy_4o_aggregated.csv \
  --proxy-item-col item_id \
  --proxy-score-col proxy_score_mean \
  --fit-max-d 0.10 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --outdir outputs/calibration_baseline_benchmark/summeval_gpt4o
```

OpenMEVA-MAGS ROC/WP:

```bash
python experiments/17_calibration_baseline_benchmark/01_calibration_baseline_benchmark.py \
  --dataset-name openmeva \
  --human-long-csv data/openmeva_long.csv \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --fit-max-d 0.10 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --outdir outputs/calibration_baseline_benchmark/openmeva
```

WMT MQM:

```bash
python experiments/17_calibration_baseline_benchmark/01_calibration_baseline_benchmark.py \
  --dataset-name wmt_mqm \
  --human-long-csv data/wmt_mqm_long.csv \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater \
  --score-col score \
  --fit-max-d 0.10 \
  --d-targets 0.05,0.10 \
  --n-repeats 50 \
  --outdir outputs/calibration_baseline_benchmark/wmt_mqm
```

## Outputs

The script refuses to overwrite non-empty output directories unless
`--overwrite` is passed.

- `calibration_baseline_run_spec.json`
- `calibration_baseline_predictions.csv`
- `calibration_baseline_summary.csv`
- `calibration_baseline_summary.md`
- `calibration_baseline_mae_wide.csv`
- `calibration_baseline_mae_wide.md`
- `calibration_baseline_skipped.csv`
