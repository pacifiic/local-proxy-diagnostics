# QGEval human direct tilt

- score_transform: `raw`

- gold_estimator: `loo_mean`

- fit_max_d: `0.1`


## Summary

| dataset_eval   | dimension          |   n_paths |   corr_A |   corr_B |   B_sign_agree |   median_r2 |   mean_icc_a1 |   frac_B_theory_neg |   frac_B_fit_neg |
|:---------------|:-------------------|----------:|---------:|---------:|---------------:|------------:|--------------:|--------------------:|-----------------:|
| HotpotQA       | answer_consistency |         3 | 0.999619 | 0.99357  |              3 |    0.999992 |      0.85668  |            1        |         1        |
| HotpotQA       | answerability      |         3 | 0.9923   | 0.857278 |              3 |    0.999967 |      0.732964 |            1        |         1        |
| HotpotQA       | clarity            |         3 | 0.993548 | 0.880975 |              3 |    0.99988  |      0.669824 |            1        |         1        |
| HotpotQA       | conciseness        |         3 | 0.99711  | 0.999859 |              3 |    0.999743 |      0.77121  |            1        |         1        |
| HotpotQA       | consistency        |         3 | 0.999828 | 0.997765 |              3 |    0.999773 |      0.47831  |            1        |         1        |
| HotpotQA       | fluency            |         3 | 0.999995 | 0.990621 |              3 |    0.999859 |      0.377296 |            1        |         1        |
| HotpotQA       | relevance          |         3 | 0.999871 | 0.999898 |              3 |    0.999693 |      0.559811 |            1        |         1        |
| SQuAD          | answer_consistency |         3 | 0.999977 | 0.943192 |              3 |    0.999989 |      0.719868 |            1        |         1        |
| SQuAD          | answerability      |         3 | 0.999478 | 0.9968   |              3 |    0.99994  |      0.586069 |            1        |         1        |
| SQuAD          | clarity            |         3 | 0.999335 | 0.995967 |              3 |    0.999936 |      0.46315  |            1        |         1        |
| SQuAD          | conciseness        |         3 | 1        | 1        |              3 |    0.999949 |      0.450183 |            1        |         1        |
| SQuAD          | consistency        |         3 | 0.997065 | 0.974987 |              3 |    0.999743 |      0.398636 |            1        |         1        |
| SQuAD          | fluency            |         3 | 0.993386 | 0.979542 |              3 |    0.999823 |      0.580206 |            1        |         1        |
| SQuAD          | relevance          |         3 | 1        | 0.999997 |              3 |    0.999758 |      0.158156 |            0.666667 |         0.666667 |
| pooled         | answer_consistency |         3 | 0.99998  | 0.989536 |              3 |    0.999991 |      0.799775 |            1        |         1        |
| pooled         | answerability      |         3 | 0.999581 | 0.974322 |              3 |    0.999959 |      0.662066 |            1        |         1        |
| pooled         | clarity            |         3 | 0.99809  | 0.982202 |              3 |    0.999927 |      0.576951 |            1        |         1        |
| pooled         | conciseness        |         3 | 0.999735 | 0.948003 |              3 |    0.999702 |      0.755245 |            1        |         1        |
| pooled         | consistency        |         3 | 0.997583 | 0.979115 |              3 |    0.999762 |      0.44582  |            1        |         1        |
| pooled         | fluency            |         3 | 0.999499 | 0.921923 |              3 |    0.999886 |      0.427587 |            1        |         1        |
| pooled         | relevance          |         3 | 0.997254 | 0.994232 |              3 |    0.999879 |      0.43741  |            1        |         1        |


## Notes

- QGEval has 3 annotators; each path uses one-vs-rest (average of 2 as gold).

- All three slices (`SQuAD`, `HotpotQA`, `pooled`) are saved.

- The main finding is the per-dimension ICC / B-sign pattern rather than cross-dataset differences.
