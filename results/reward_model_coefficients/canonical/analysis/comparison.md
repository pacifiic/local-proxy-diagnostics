# Reward-Model Coefficients

Full-pool coefficients are descriptive. Intervals are paired article-cluster bootstrap 95% intervals.
Held-out MAE uses calibration-only coefficients and common item-level splits across models.

| Model | Human dimension | A [95% CI] | B [95% CI] | B tail | B skewness | A+B MAE | A-only MAE | Empirical-fit MAE | Valid splits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skywork | coherence | 0.26020 [0.20871, 0.31542] | 0.00860 [-0.04031, 0.06233] | -0.08162 | 0.09022 | 0.01350 | 0.01375 | 0.01288 | 50 |
| armorm | coherence | 0.26153 [0.21048, 0.31184] | -0.05047 [-0.09725, -0.00373] | -0.11532 | 0.06485 | 0.01024 | 0.01165 | 0.01011 | 50 |
| deberta | coherence | 0.22320 [0.16451, 0.28135] | -0.09084 [-0.13753, -0.03765] | -0.12299 | 0.03215 | 0.01025 | 0.01165 | 0.01013 | 50 |
| skywork | consistency | 0.27219 [0.22298, 0.32232] | -0.19906 [-0.25599, -0.13841] | -0.29343 | 0.09437 | 0.01339 | 0.02250 | 0.01338 | 50 |
| armorm | consistency | 0.22262 [0.16625, 0.28113] | -0.13225 [-0.19057, -0.07446] | -0.18745 | 0.05520 | 0.01222 | 0.01656 | 0.01240 | 50 |
| deberta | consistency | 0.19058 [0.13488, 0.24934] | -0.15581 [-0.21294, -0.09923] | -0.18326 | 0.02745 | 0.01347 | 0.01987 | 0.01335 | 50 |
| skywork | fluency | 0.16944 [0.11927, 0.22574] | -0.12287 [-0.19050, -0.05278] | -0.18161 | 0.05875 | 0.01029 | 0.01618 | 0.01018 | 50 |
| armorm | fluency | 0.18772 [0.14162, 0.23926] | -0.11415 [-0.17225, -0.05792] | -0.16070 | 0.04655 | 0.00971 | 0.01509 | 0.00983 | 50 |
| deberta | fluency | 0.14095 [0.08829, 0.19565] | -0.11452 [-0.17417, -0.05637] | -0.13482 | 0.02031 | 0.01109 | 0.01718 | 0.01123 | 50 |
| skywork | relevance | 0.27497 [0.22955, 0.32568] | -0.06444 [-0.11391, -0.01609] | -0.15978 | 0.09534 | 0.00787 | 0.00970 | 0.00789 | 50 |
| armorm | relevance | 0.26753 [0.22405, 0.31184] | -0.07490 [-0.11744, -0.03382] | -0.14123 | 0.06634 | 0.00967 | 0.01112 | 0.00972 | 50 |
| deberta | relevance | 0.21154 [0.16247, 0.26279] | -0.07447 [-0.11451, -0.03783] | -0.10494 | 0.03048 | 0.01035 | 0.01162 | 0.01035 | 50 |

Endpoint radius: d=0.1. Full radius/baseline results are in summary.csv.
The split SD is descriptive variability over overlapping splits, not an independent-sample CI.
Coefficients depend on proxy, human target, item distribution and text-context policy.
