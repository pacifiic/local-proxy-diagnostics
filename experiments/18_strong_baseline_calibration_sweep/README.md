# Experiment 18: Calibration-Size Sweep with Strong Baselines

This experiment asks whether flexible calibration baselines remain strongest
when the calibration split is small. It extends Experiment 17 by sweeping the
calibration fraction while keeping the same prospective evaluation protocol.

## Question

Flexible baselines such as direct empirical curve transfer, empirical curve
fitting, isotonic calibration, and a small MLP calibrator can be strong when
enough calibration data are available. This experiment checks whether the
theory-derived `A+B` predictor is more stable or more sample-efficient in
smaller calibration regimes.

## Baselines

- `A+B`
- `A-only`
- `A_nu`
- `empirical_curve`
- `empirical_fit`
- `pearson_corr`
- `spearman_corr`
- `isotonic_calibration`
- `mlp_calibration`
- `zero`

The MLP baseline is a black-box one-dimensional score calibrator: it fits
`proxy score -> human gold` on the calibration split only, then integrates the
learned function under the evaluation split tilt weights. The default MLP is a
small deterministic network with hidden layers `16,16`, `alpha=1e-3`,
`max_iter=300`, and `random_seed=0`. Use `--skip-mlp` for a faster run without
this baseline.

The main plot shows `A+B`, `empirical_fit`, `isotonic_calibration`,
`mlp_calibration`, and `A-only`. Full CSV/Markdown tables include all
baselines.

## Example

```bash
python experiments/18_strong_baseline_calibration_sweep/01_calibration_size_strong_baselines.py \
  --dataset-name summeval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --calibration-fracs 0.05,0.10,0.20,0.30,0.50 \
  --n-repeats 50 \
  --mlp-hidden-layers 16,16 \
  --mlp-alpha 1e-3 \
  --mlp-max-iter 300 \
  --mlp-seed 0 \
  --outdir outputs/strong_baseline_calibration_sweep/summeval_human
```

Use `02_collect_and_plot.py` after running one or more dataset directories:

```bash
python experiments/18_strong_baseline_calibration_sweep/02_collect_and_plot.py \
  --root outputs/strong_baseline_calibration_sweep \
  --outdir outputs/strong_baseline_calibration_sweep/combined
```

## Outputs

Per dataset:

- `calibration_size_strong_baselines_run_spec.json`
- `calibration_size_strong_baselines_predictions.csv`
- `calibration_size_strong_baselines_endpoint_summary.csv`
- `calibration_size_strong_baselines_endpoint_summary.md`
- `calibration_size_strong_baselines_curve_summary.csv`
- `calibration_size_strong_baselines_curve_summary.md`
- `calibration_size_strong_baselines_skipped.csv`

Combined:

- `all_calibration_size_strong_baselines_endpoint_summary.csv`
- `all_calibration_size_strong_baselines_curve_summary.csv`
- `compact_endpoint_mae_wide.csv`
- `fig_calibration_size_strong_baselines.pdf`
- `fig_calibration_size_strong_baselines.png`
