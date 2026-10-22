# Evaluation-aware conditional-mean A+B

> **Information setting.** This analysis assumes the target candidate pool
> has already been collected and proxy-scored. It is not a fair pre-pool
> comparison for a decision made from 300 rated items before the future 1,000
> candidates exist; use exp33 for that setting.

| dataset               | conditional_mean_model   |   n_paths |   n_path_repeats |   raw_A+B_curve_mae |   evaluation_aware_A+B_curve_mae |   direct_calibrator_curve_mae |   raw_A+B_endpoint_mae_d0p10 |   evaluation_aware_A+B_endpoint_mae_d0p10 |   direct_calibrator_endpoint_mae_d0p10 |   mean_abs_projection_vs_direct |
|:----------------------|:-------------------------|----------:|-----------------:|--------------------:|---------------------------------:|------------------------------:|-----------------------------:|------------------------------------------:|---------------------------------------:|--------------------------------:|
| WMT-MQM-human_heldout | isotonic                 |       141 |             7050 |            0.059206 |                         0.053158 |                      0.052110 |                     0.084592 |                                  0.075654 |                               0.073031 |                        0.004558 |
| WMT-MQM-human_heldout | binned                   |       141 |             7050 |            0.059206 |                         0.065574 |                      0.068429 |                     0.084592 |                                  0.090504 |                               0.096732 |                        0.008500 |
| WMT-MQM-human_heldout | spline                   |       141 |             7050 |            0.059206 |                         0.056674 |                      0.056987 |                     0.084592 |                                  0.079392 |                               0.079799 |                        0.005700 |
| WMT-MQM-human_heldout | MLP                      |       141 |             7050 |            0.059206 |                         0.052863 |                      0.052150 |                     0.084592 |                                  0.075038 |                               0.073125 |                        0.004749 |

## Path-clustered paired differences

| dataset               | conditional_mean_model   | comparison                                   |   difference |     ci_lo |     ci_hi |
|:----------------------|:-------------------------|:---------------------------------------------|-------------:|----------:|----------:|
| WMT-MQM-human_heldout | isotonic                 | evaluation-aware A+B minus raw A+B           |    -0.006048 | -0.007497 | -0.004674 |
| WMT-MQM-human_heldout | isotonic                 | evaluation-aware A+B minus direct calibrator |     0.001047 |  0.000832 |  0.001279 |
| WMT-MQM-human_heldout | binned                   | evaluation-aware A+B minus raw A+B           |     0.006368 |  0.004192 |  0.008736 |
| WMT-MQM-human_heldout | binned                   | evaluation-aware A+B minus direct calibrator |    -0.002856 | -0.003585 | -0.002228 |
| WMT-MQM-human_heldout | spline                   | evaluation-aware A+B minus raw A+B           |    -0.002531 | -0.004263 | -0.000989 |
| WMT-MQM-human_heldout | spline                   | evaluation-aware A+B minus direct calibrator |    -0.000313 | -0.000562 | -0.000038 |
| WMT-MQM-human_heldout | MLP                      | evaluation-aware A+B minus raw A+B           |    -0.006343 | -0.007881 | -0.005023 |
| WMT-MQM-human_heldout | MLP                      | evaluation-aware A+B minus direct calibrator |     0.000713 |  0.000507 |  0.000972 |

Negative differences favor evaluation-aware A+B.
