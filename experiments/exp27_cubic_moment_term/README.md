# exp27: Cubic Moment Term

This experiment adds the next tuning-free Puiseux term to the local law:

```text
Delta(d) = A sqrt(d) + B d + C d^(3/2).
```

`C` is computed only from calibration-split fourth-order proxy and mixed
moments. The experiment reuses exp21's datasets, outer 30/70 splits, 50
repeats, radii, exact held-out frontier, and isotonic/spline baselines.

Run into a fresh directory:

```bash
OUTDIR=outputs/cubic_moment_term JOBS=32 \
  bash experiments/exp27_cubic_moment_term/run.sh
```

The audit reports endpoint MAE, CurveMAE, path-clustered bootstrap comparisons,
and the split-to-split distribution of `C`. A lower approximation bias is not
assumed to imply lower prediction error because `C` uses fourth-order moments.
