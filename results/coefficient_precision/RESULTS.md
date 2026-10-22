# Coefficient Precision Audit

## Metric reconciliation

| setting                | protocol                | radius_grid               |   n_complete_curves |   a_plus_b_error | reachability_policy                                        |
|:-----------------------|:------------------------|:--------------------------|--------------------:|-----------------:|:-----------------------------------------------------------|
| SummEval-human_heldout | paper Grid IAE          | 0.005,0.01,0.02,0.05,0.1  |                1589 |       0.00939027 | d-sweep retained 1,589/1,600 SummEval path-splits          |
| SummEval-GPT4o         | paper Grid IAE          | 0.005,0.01,0.02,0.05,0.1  |                 200 |       0.00644302 | all 200 path-splits                                        |
| WMT-MQM-human_heldout  | local-grid MAE | 0.01,0.025,0.05,0.075,0.1 |                7050 |       0.0592059  | retain each reachable radius; all A+B path-splits complete |

The two metrics are both means of pointwise absolute error, but they use different radius grids and reachability accounting. They should not be presented as the same numerical experiment.

## Calibration-size coefficient error at d=0.10

| dataset                     |   calibration_frac |   median_abs_b_error |   b_sign_agreement |   median_scaled_a_error |   median_scaled_b_error |   median_eval_remainder_abs |
|:----------------------------|-------------------:|---------------------:|-------------------:|------------------------:|------------------------:|----------------------------:|
| OpenMEVA-MAGS-human_heldout |               0.05 |            0.0560178 |           0.826    |               0.023804  |              0.00560178 |                 0.00268418  |
| OpenMEVA-MAGS-human_heldout |               0.1  |            0.0438077 |           0.852    |               0.0187623 |              0.00438077 |                 0.00270783  |
| OpenMEVA-MAGS-human_heldout |               0.3  |            0.0292646 |           0.9      |               0.012574  |              0.00292646 |                 0.00272159  |
| SummEval-GPT4o              |               0.05 |            0.0690256 |           0.86     |               0.0213528 |              0.00690256 |                 0.000318622 |
| SummEval-GPT4o              |               0.1  |            0.0494676 |           0.935    |               0.0159889 |              0.00494676 |                 0.000320315 |
| SummEval-GPT4o              |               0.3  |            0.0367168 |           0.97     |               0.0115889 |              0.00367168 |                 0.000399821 |
| SummEval-human_heldout      |               0.05 |            0.092726  |           0.784375 |               0.0317743 |              0.0092726  |                 0.00172712  |
| SummEval-human_heldout      |               0.1  |            0.0679604 |           0.819375 |               0.0226452 |              0.00679604 |                 0.00172353  |
| SummEval-human_heldout      |               0.3  |            0.0437438 |           0.894375 |               0.0156506 |              0.00437438 |                 0.00177646  |
| WMT-MQM-human_heldout       |               0.05 |            0.545666  |           0.795319 |               0.16209   |              0.0545666  |                 0.0081518   |
| WMT-MQM-human_heldout       |               0.1  |            0.431039  |           0.840426 |               0.1187    |              0.0431039  |                 0.00826985  |
| WMT-MQM-human_heldout       |               0.3  |            0.316645  |           0.896596 |               0.0777115 |              0.0316645  |                 0.00872163  |

The scaled A/B columns are the first two coefficient-shift terms in the paper's prediction-error decomposition. The remainder is computed on the evaluation split using its own A and B.

## Curvature contribution over radius

| dataset                     |    d |   median_curvature_share |
|:----------------------------|-----:|-------------------------:|
| OpenMEVA-MAGS-human_heldout | 0.05 |                0.0222936 |
| OpenMEVA-MAGS-human_heldout | 0.1  |                0.0312395 |
| SummEval-GPT4o              | 0.05 |                0.0974875 |
| SummEval-GPT4o              | 0.1  |                0.132517  |
| SummEval-human_heldout      | 0.05 |                0.0318628 |
| SummEval-human_heldout      | 0.1  |                0.0444738 |
| WMT-MQM-human_heldout       | 0.05 |                0.0603196 |
| WMT-MQM-human_heldout       | 0.1  |                0.0832253 |

## Strict-local DPO comparison

| predictor             |   n_seeds |   n_checkpoints_total |   min_checkpoints_per_seed |   max_checkpoints_per_seed |   frontier_mae |   frontier_mae_sd |   realized_mae |   realized_mae_sd |   frontier_signed_bias |   optimizer_gap_mean |
|:----------------------|----------:|----------------------:|---------------------------:|---------------------------:|---------------:|------------------:|---------------:|------------------:|-----------------------:|---------------------:|
| prompt-z-score binned |         3 |                    10 |                          3 |                          4 |     0.0104379  |       0.000405917 |      0.0107207 |       0.00109849  |            -0.0104379  |            0.0211585 |
| A+B                   |         3 |                    10 |                          3 |                          4 |     0.003947   |       0.000472143 |      0.0172115 |       0.00116275  |            -0.003947   |            0.0211585 |
| A-only                |         3 |                    10 |                          3 |                          4 |     0.00210373 |       0.000527313 |      0.0190548 |       0.00120341  |            -0.00210373 |            0.0211585 |
| zero                  |         3 |                    10 |                          3 |                          4 |     0.0419695  |       0.000170027 |      0.0208109 |       0.000616739 |            -0.0419695  |            0.0211585 |

The SD columns are sample standard deviations across three DPO seeds. The strict window contains 10 seed-checkpoint observations: three or four checkpoints per seed because checkpoint median KL differs by seed.
