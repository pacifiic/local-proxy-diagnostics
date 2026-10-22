# QGEval metric-proxy tilt

- score_transform: `z`

- fit_max_d: `0.1`


## Summary

| dataset_eval   | dimension          |   n_metrics |   frac_B_theory_neg |   frac_B_fit_neg |   corr_theory_fit_B |   median_r2 | best_corr_metric   | largest_B_metric   | most_negative_B_metric   |
|:---------------|:-------------------|------------:|--------------------:|-----------------:|--------------------:|------------:|:-------------------|:-------------------|:-------------------------|
| HotpotQA       | answer_consistency |          15 |            0.8      |         0.8      |            0.885629 |    0.999999 | RQUGE              | BARTScore-ref      | RQUGE                    |
| HotpotQA       | answerability      |          15 |            0.866667 |         0.933333 |            0.626794 |    0.999993 | UniEval            | QSTS               | BARTScore-src            |
| HotpotQA       | clarity            |          15 |            1        |         1        |            0.63981  |    0.999994 | UniEval            | RQUGE              | BARTScore-src            |
| HotpotQA       | conciseness        |          15 |            0.6      |         0.666667 |            0.890374 |    0.999997 | BERTScore          | METEOR             | BARTScore-src            |
| HotpotQA       | consistency        |          15 |            0.866667 |         0.866667 |            0.810551 |    0.999958 | GPTScore-src       | QSTS               | GPTScore-src             |
| HotpotQA       | fluency            |          15 |            0.8      |         0.866667 |            0.726379 |    0.999983 | UniEval            | RQUGE              | UniEval                  |
| HotpotQA       | relevance          |          15 |            1        |         1        |            0.954555 |    0.999983 | GPTScore-src       | Q-BLEU4            | GPTScore-src             |
| SQuAD          | answer_consistency |          15 |            0.8      |         0.866667 |            0.707466 |    0.999999 | RQUGE              | QRelScore          | RQUGE                    |
| SQuAD          | answerability      |          15 |            0.666667 |         0.666667 |            0.97352  |    0.999999 | RQUGE              | GPTScore-src       | RQUGE                    |
| SQuAD          | clarity            |          15 |            0.4      |         0.466667 |            0.971514 |    0.999998 | BARTScore-src      | GPTScore-src       | RQUGE                    |
| SQuAD          | conciseness        |          15 |            0.533333 |         0.6      |            0.532511 |    0.999996 | GPTScore-src       | QSTS               | BARTScore-src            |
| SQuAD          | consistency        |          15 |            0.8      |         0.8      |            0.963227 |    0.999996 | RQUGE              | QRelScore          | RQUGE                    |
| SQuAD          | fluency            |          15 |            0.333333 |         0.4      |            0.999603 |    0.999996 | UniEval            | BARTScore-src      | UniEval                  |
| SQuAD          | relevance          |          15 |            0.8      |         0.8      |            0.920556 |    0.999993 | GPTScore-src       | RQUGE              | UniEval                  |
| pooled         | answer_consistency |          15 |            0.866667 |         0.8      |            0.979908 |    0.999999 | RQUGE              | UniEval            | RQUGE                    |
| pooled         | answerability      |          15 |            0.866667 |         1        |            0.374211 |    0.999998 | RQUGE              | BARTScore-ref      | BARTScore-src            |
| pooled         | clarity            |          15 |            1        |         1        |            0.432175 |    0.999996 | UniEval            | GPTScore-ref       | BARTScore-src            |
| pooled         | conciseness        |          15 |            0.666667 |         0.6      |            0.980343 |    0.999996 | BERTScore          | QSTS               | BARTScore-src            |
| pooled         | consistency        |          15 |            0.866667 |         0.866667 |            0.351083 |    0.999996 | RQUGE              | QRelScore          | UniEval                  |
| pooled         | fluency            |          15 |            0.733333 |         0.8      |            0.557948 |    0.999992 | UniEval            | RQUGE              | UniEval                  |
| pooled         | relevance          |          15 |            1        |         1        |            0.900569 |    0.999992 | GPTScore-src       | QRelScore          | GPTScore-src             |


## Notes

- This experiment uses the human average score as gold and automatic metrics as proxies.

- The key finding is that a metric with high correlation does not necessarily have a good local `B`.

- Metrics that consistently yield a negative `B_fit` are the closest reward-model analogues.
