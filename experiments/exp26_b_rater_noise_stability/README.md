# exp26: B Stability Under Rater Budget and Rating Noise

This CPU experiment answers the metareview request to characterize the
second-order coefficient under fewer repeated ratings and noisier calibration
labels.

## Design

- Dataset: complete SummEval expert and crowd rating matrices.
- Proxy: one fixed held-out human rater.
- Calibration gold: a nested subset of the remaining raters.
  Expert paths use 1 or 2 gold raters; crowd paths use 1 through 4.
- Evaluation gold: the clean mean of all remaining raters, regardless of the
  calibration condition.
- Calibration fractions: 10%, 20%, 30%, and 50%, with nested item splits.
- Noise: clean ratings, clipped Gaussian noise, and clipped
  variance-normalized Student-t noise with 3 degrees of freedom. Noise scales
  are 0.25, 0.5, and 1.0 times the group-level rating standard deviation.
- Repeats: 50 deterministic item splits by default.
- Local curve: `d={.01,.025,.05,.075,.10}`.
- Reachability: all path/repeat conditions are retained. Curve error is
  averaged over the requested radii reachable by the evaluation proxy. If a
  tied proxy maximum makes `d=.10` unreachable, only the `.10` exact-gain
  endpoint fields are marked missing and the event is recorded separately.

The primary outputs report B error and sign stability, the relative size of
the `Bd` contribution, and paired CurveMAE for A+B versus A-only. All random
streams are derived from SHA-256 keys. The script refuses to overwrite a
non-empty output directory.

## Run

```bash
OUTDIR=outputs/exp26_b_rater_noise_50rep \
N_REPEATS=50 JOBS=16 RANDOM_SEED=0 \
bash experiments/exp26_b_rater_noise_stability/run_all.sh
```

For a quick code and data smoke test:

```bash
OUTDIR=outputs/exp26_smoke N_REPEATS=2 JOBS=2 \
bash experiments/exp26_b_rater_noise_stability/run_all.sh
```

## Outputs

- `b_rater_noise_rows.csv`: complete paired condition-level results.
- `condition_summary.csv`: path-macro summaries and path-bootstrap intervals.
- `reachability_accounting.csv`: one row per path, repeat, and calibration
  fraction, including the finite-pool maximum reachable KL.
- `selected_conditions.csv`: compact 30% calibration subset.
- `fig_b_rater_noise_stability.{pdf,png}`: three-panel summary.
- `run_spec.json` and `artifact_spec.json`: complete run metadata.
