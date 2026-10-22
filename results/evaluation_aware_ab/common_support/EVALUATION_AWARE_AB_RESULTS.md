# Evaluation-aware conditional-mean A+B

> **Information setting.** This analysis assumes the target candidate pool
> has already been collected and proxy-scored. It is not a fair pre-pool
> comparison for a decision made from 300 rated items before the future 1,000
> candidates exist; use exp33 for that setting.

> **Evaluation support.** All displayed methods use the shared radius cells defined by `--support-exclusions`.

| dataset                     | conditional_mean_model   |   n_paths |   n_path_repeats |   raw_A+B_curve_mae |   evaluation_aware_A+B_curve_mae |   direct_calibrator_curve_mae |   raw_A+B_endpoint_mae_d0p10 |   evaluation_aware_A+B_endpoint_mae_d0p10 |   direct_calibrator_endpoint_mae_d0p10 |   mean_abs_projection_vs_direct |
|:----------------------------|:-------------------------|----------:|-----------------:|--------------------:|---------------------------------:|------------------------------:|-----------------------------:|------------------------------------------:|---------------------------------------:|--------------------------------:|
| SummEval-human_heldout      | isotonic                 |        32 |             1600 |            0.011850 |                         0.009588 |                      0.009086 |                     0.017499 |                                  0.014611 |                               0.013047 |                        0.001710 |
| SummEval-human_heldout      | binned                   |        32 |             1600 |            0.011850 |                         0.027825 |                      0.029140 |                     0.017499 |                                  0.042348 |                               0.045091 |                        0.002100 |
| SummEval-human_heldout      | spline                   |        32 |             1600 |            0.011850 |                         0.010402 |                      0.010274 |                     0.017499 |                                  0.015412 |                               0.014636 |                        0.001846 |
| SummEval-human_heldout      | MLP                      |        32 |             1600 |            0.011850 |                         0.009724 |                      0.009218 |                     0.017499 |                                  0.014802 |                               0.013236 |                        0.001722 |
| SummEval-GPT4o              | isotonic                 |         4 |              200 |            0.008059 |                         0.008009 |                      0.008109 |                     0.011757 |                                  0.011627 |                               0.011849 |                        0.000436 |
| SummEval-GPT4o              | binned                   |         4 |              200 |            0.008059 |                         0.008604 |                      0.008646 |                     0.011757 |                                  0.012335 |                               0.012412 |                        0.000414 |
| SummEval-GPT4o              | spline                   |         4 |              200 |            0.008059 |                         0.007800 |                      0.007768 |                     0.011757 |                                  0.011346 |                               0.011278 |                        0.000423 |
| SummEval-GPT4o              | MLP                      |         4 |              200 |            0.008059 |                         0.007929 |                      0.007886 |                     0.011757 |                                  0.011639 |                               0.011551 |                        0.000356 |
| OpenMEVA-MAGS-human_heldout | isotonic                 |        10 |              500 |            0.009906 |                         0.007119 |                      0.007058 |                     0.014816 |                                  0.010795 |                               0.010615 |                        0.001198 |
| OpenMEVA-MAGS-human_heldout | binned                   |        10 |              500 |            0.009906 |                         0.013296 |                      0.014286 |                     0.014816 |                                  0.020476 |                               0.022808 |                        0.001799 |
| OpenMEVA-MAGS-human_heldout | spline                   |        10 |              500 |            0.009906 |                         0.007257 |                      0.007277 |                     0.014816 |                                  0.010920 |                               0.010930 |                        0.001241 |
| OpenMEVA-MAGS-human_heldout | MLP                      |        10 |              500 |            0.009906 |                         0.006954 |                      0.006929 |                     0.014816 |                                  0.010446 |                               0.010353 |                        0.001255 |

## Path-clustered paired differences

| dataset                     | conditional_mean_model   | comparison                                   |   difference |     ci_lo |     ci_hi |
|:----------------------------|:-------------------------|:---------------------------------------------|-------------:|----------:|----------:|
| SummEval-human_heldout      | isotonic                 | evaluation-aware A+B minus raw A+B           |    -0.002262 | -0.003365 | -0.001318 |
| SummEval-human_heldout      | isotonic                 | evaluation-aware A+B minus direct calibrator |     0.000502 |  0.000081 |  0.001026 |
| SummEval-human_heldout      | binned                   | evaluation-aware A+B minus raw A+B           |     0.015975 |  0.012496 |  0.019092 |
| SummEval-human_heldout      | binned                   | evaluation-aware A+B minus direct calibrator |    -0.001316 | -0.001557 | -0.001082 |
| SummEval-human_heldout      | spline                   | evaluation-aware A+B minus raw A+B           |    -0.001448 | -0.002789 | -0.000253 |
| SummEval-human_heldout      | spline                   | evaluation-aware A+B minus direct calibrator |     0.000128 | -0.000228 |  0.000579 |
| SummEval-human_heldout      | MLP                      | evaluation-aware A+B minus raw A+B           |    -0.002126 | -0.003232 | -0.001169 |
| SummEval-human_heldout      | MLP                      | evaluation-aware A+B minus direct calibrator |     0.000506 |  0.000105 |  0.001030 |
| SummEval-GPT4o              | isotonic                 | evaluation-aware A+B minus raw A+B           |    -0.000050 | -0.000936 |  0.000482 |
| SummEval-GPT4o              | isotonic                 | evaluation-aware A+B minus direct calibrator |    -0.000100 | -0.000217 | -0.000030 |
| SummEval-GPT4o              | binned                   | evaluation-aware A+B minus raw A+B           |     0.000545 |  0.000236 |  0.000754 |
| SummEval-GPT4o              | binned                   | evaluation-aware A+B minus direct calibrator |    -0.000043 | -0.000088 |  0.000003 |
| SummEval-GPT4o              | spline                   | evaluation-aware A+B minus raw A+B           |    -0.000259 | -0.001081 |  0.000372 |
| SummEval-GPT4o              | spline                   | evaluation-aware A+B minus direct calibrator |     0.000032 | -0.000046 |  0.000130 |
| SummEval-GPT4o              | MLP                      | evaluation-aware A+B minus raw A+B           |    -0.000130 | -0.000607 |  0.000267 |
| SummEval-GPT4o              | MLP                      | evaluation-aware A+B minus direct calibrator |     0.000042 |  0.000036 |  0.000049 |
| OpenMEVA-MAGS-human_heldout | isotonic                 | evaluation-aware A+B minus raw A+B           |    -0.002787 | -0.003422 | -0.002098 |
| OpenMEVA-MAGS-human_heldout | isotonic                 | evaluation-aware A+B minus direct calibrator |     0.000061 | -0.000071 |  0.000190 |
| OpenMEVA-MAGS-human_heldout | binned                   | evaluation-aware A+B minus raw A+B           |     0.003390 |  0.000080 |  0.007947 |
| OpenMEVA-MAGS-human_heldout | binned                   | evaluation-aware A+B minus direct calibrator |    -0.000990 | -0.001247 | -0.000751 |
| OpenMEVA-MAGS-human_heldout | spline                   | evaluation-aware A+B minus raw A+B           |    -0.002649 | -0.003355 | -0.001843 |
| OpenMEVA-MAGS-human_heldout | spline                   | evaluation-aware A+B minus direct calibrator |    -0.000020 | -0.000147 |  0.000109 |
| OpenMEVA-MAGS-human_heldout | MLP                      | evaluation-aware A+B minus raw A+B           |    -0.002952 | -0.003589 | -0.002219 |
| OpenMEVA-MAGS-human_heldout | MLP                      | evaluation-aware A+B minus direct calibrator |     0.000025 | -0.000070 |  0.000131 |

Negative differences favor evaluation-aware A+B.
