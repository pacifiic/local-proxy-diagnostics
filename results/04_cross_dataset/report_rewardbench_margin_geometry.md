# RewardBench margin geometry

- pseudo_gold: `loo_mean`

- score_transform: `model_z`

- missingness_mode: `pairwise_available`

- fit_max_d: `0.1`


## Summary

| subset_eval           |   n_models |   frac_B_theory_neg |   frac_B_fit_neg |   corr_theory_fit_B |   median_r2 |   mean_corr_J_r |
|:----------------------|-----------:|--------------------:|-----------------:|--------------------:|------------:|----------------:|
| alpacaeval-easy       |        115 |          0.791304   |        0.782609  |            0.921005 |    1        |        0.537866 |
| alpacaeval-hard       |        115 |          0.73913    |        0.721739  |            0.962955 |    0.999999 |        0.49963  |
| alpacaeval-length     |        115 |          0.243478   |        0.278261  |            0.95073  |    0.999999 |        0.457546 |
| donotanswer           |        115 |          0.747826   |        0.756522  |            0.942488 |    1        |        0.540406 |
| hep-cpp               |        115 |          0.0956522  |        0.104348  |            0.942649 |    0.999999 |        0.487471 |
| hep-go                |        115 |          0.121739   |        0.130435  |            0.867555 |    0.999998 |        0.486284 |
| hep-java              |        115 |          0.121739   |        0.173913  |            0.953309 |    0.999999 |        0.514515 |
| hep-js                |        115 |          0.0695652  |        0.121739  |            0.925112 |    0.999999 |        0.538613 |
| hep-python            |        115 |          0.0695652  |        0.104348  |            0.878141 |    0.999999 |        0.520209 |
| hep-rust              |        115 |          0.147826   |        0.173913  |            0.909187 |    0.999999 |        0.482344 |
| llmbar-adver-GPTInst  |        115 |          0.573913   |        0.591304  |            0.902153 |    0.999999 |        0.486263 |
| llmbar-adver-GPTOut   |        115 |          0.782609   |        0.834783  |            0.95906  |    0.999996 |        0.473124 |
| llmbar-adver-manual   |        115 |          0.365217   |        0.304348  |            0.957446 |    0.999999 |        0.448972 |
| llmbar-adver-neighbor |        115 |          0.982609   |        0.973913  |            0.865827 |    0.999999 |        0.488841 |
| llmbar-natural        |        115 |          0.278261   |        0.313043  |            0.914292 |    0.999999 |        0.484208 |
| math-prm              |        100 |          0.57       |        0.54      |            0.872006 |    1        |        0.369419 |
| mt-bench-med          |        115 |          0.521739   |        0.547826  |            0.97274  |    0.999998 |        0.461973 |
| pooled                |        115 |          0.00869565 |        0.0434783 |            0.691825 |    1        |        0.549256 |
| refusals-dangerous    |        115 |          0.695652   |        0.756522  |            0.940977 |    0.999999 |        0.516461 |
| refusals-offensive    |        115 |          0.843478   |        0.834783  |            0.970338 |    0.999995 |        0.42971  |
| xstest-should-refuse  |        115 |          0.869565   |        0.886957  |            0.9724   |    0.999999 |        0.528674 |
| xstest-should-respond |        115 |          0.434783   |        0.530435  |            0.935951 |    0.999999 |        0.507137 |


## Notes

- This experiment is cross-model ensemble geometry, not human-gold theorem validation.

- This is better suited for the appendix or robustness section than the main text.

- `other_family_mean` is more conservative than `loo_mean` and reduces the same-family leakage objection.
