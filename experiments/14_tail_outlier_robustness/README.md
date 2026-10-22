# Tail and Outlier Robustness Audit

Audits coefficient stability under bounded score transformations and controlled
score outliers.  The script runs on one repeated-scalar dataset at a time.

Example:

```bash
python experiments/14_tail_outlier_robustness/01_tail_outlier_robustness_audit.py \
  --dataset-name SummEval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --rater-col rater_idx \
  --outdir outputs/tail_robustness_summeval
```

Default conditions are:

- original
- winsorize both proxy and gold at 1/99%
- winsorize both proxy and gold at 5/95%
- rank-normalize both proxy and gold
- inject positive proxy outliers
- inject positive gold outliers

Outlier conditions use stable SHA-256-derived seeds.

Reference runs:

```bash
ROOT=outputs/tail_outlier_robustness

python experiments/14_tail_outlier_robustness/01_tail_outlier_robustness_audit.py \
  --dataset-name SummEval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --rater-col rater_idx \
  --outdir "$ROOT/summeval"

python experiments/14_tail_outlier_robustness/01_tail_outlier_robustness_audit.py \
  --dataset-name OpenMEVA-MAGS \
  --human-long-csv data/openmeva_long.csv \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --rater-col rater_idx \
  --outdir "$ROOT/openmeva_mags"

python experiments/14_tail_outlier_robustness/01_tail_outlier_robustness_audit.py \
  --dataset-name WMT-MQM-clusters \
  --human-long-csv data/wmt_mqm_long.csv \
  --item-cols item_id \
  --group-cols dataset_id \
  --rater-col rater \
  --outdir "$ROOT/wmt_mqm_clusters_outlier20"
```

These runs use `outlier_repeats=20`, `outlier_frac=0.01`,
`outlier_magnitude=8.0`, `outlier_direction=positive`, `d_target=0.10`,
`fit_max_d=0.10`, `min_items=200`, `min_raters=3`, and `random_seed=0`.
Run specs are saved in each output subdirectory.
