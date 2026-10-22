# OpenMEVA direct tilt experiment

- score_transform: `raw`

- gold_estimator: `loo_mean`

- fit_max_d: `0.1`


## Summary

| dataset_eval   |   n_paths |   corr_A |   corr_B |   B_sign_agree |   median_r2 |   mean_icc_a1 |   frac_B_theory_neg |   frac_B_fit_neg |
|:---------------|----------:|---------:|---------:|---------------:|------------:|--------------:|--------------------:|-----------------:|
| pooled         |         5 | 0.99996  | 0.999336 |              4 |    1        |      0.755335 |                 0   |              0.2 |
| roc            |         5 | 0.999623 | 0.941136 |              5 |    0.999999 |      0.775762 |                 0   |              0   |
| wp             |         5 | 0.99999  | 0.998899 |              4 |    1        |      0.715597 |                 0.4 |              0.6 |


## Notes

- `roc`, `wp`, and `pooled` are treated as separate task slices.

- Each path uses held-out rater one-vs-rest exact tilt.

- OpenMEVA has a single overall score, so dataset decomposition is the focus instead of dimension decomposition.
