# Exp36: Dominant Local-Law and Residual Audit

This experiment tests whether a flexible regression in optimization radius
materially improves held-out gain prediction beyond

\[
\Delta(d)=A\sqrt d+Bd.
\]

It uses the paper's 30% calibration / 70% evaluation item split, 50 repeats,
seed 0, and the local grid \(d\in\{0.01,0.025,0.05,0.075,0.10\}\). All
prospective methods are fit using calibration items only. Evaluation proxy and
gold values are used only to calculate the exact held-out gain target.

## Methods

- `A+B_moment`: theorem coefficients estimated from calibration moments.
- `empirical_two_term`: two coefficients fit directly to the exact calibration
  curve.
- `empirical_cubic`: adds a fitted \(d^{3/2}\) term.
- `direct_spline`: fixed six-knot cubic spline fit to the calibration curve,
  with ridge smoothing selected by `RidgeCV`.
- `A+B_residual_spline`: moment \(A+B\) plus a spline fit to the calibration
  residual curve.
- `calibration_curve`: the exact calibration gain transferred at each radius.
- `eval_oracle_A+B`: evaluation moments used retrospectively to isolate
  truncation error. This is not a deployable predictor.

The flexible models therefore receive more curve-fitting freedom than the
moment law, but no target-pool information.

## Run

Choose a fresh output directory:

```bash
bash experiments/exp36_dominant_local_law/run.sh \
  outputs/exp36_run1 32
```

The build refuses to reuse an existing output directory. Intermediate
predictions, skipped rows, run metadata, unit-level errors, paired bootstrap
intervals, and the compact Markdown table are all preserved. The final
data-chain audit independently rebuilds paths from the raw input tables,
recomputes every held-out exact gain with a separate KL solver, and refits the
flexible models for one repeat of every path.
