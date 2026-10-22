# Recommended table placement

## Main text

1. **Table 1: Human direct-tilt coefficient-law validation.**
   Use this as the main theorem-validation table. Keep it restricted to repeated human scalar ratings and admitted analysis slices: SummEval, QGEval-human, OpenMEVA-MAGS ROC/WP, WMT MQM admitted paths, and WMT pSQM admitted paths. This avoids overclaiming and makes the sign-regime heterogeneity clear.

2. **Table 2: Prospective prediction of held-out local geometry.**
   This should be the most important empirical table. It directly supports the paper's strongest claim: small calibration matrices predict held-out gain curves better than reliability-only A_nu baselines. Include the GPT/4o margin columns to separate curvature-sign and fixed-gain sign diagnostics.

3. **Table 3a/3b: Scope checks.**
   Put finite-range stability and BoN bridge as two small adjacent tables or one two-panel table. These tables define the scope of the local analysis.

4. **Table 4: Measurement-intervention summary.**
   Use this to support the corrected intervention message: measurement improves signal A reliably, but headroom requires curvature cooperation and is not guaranteed.

## Appendix

A1. **Feedback-geometry extensions.** MT-Bench split-half BT, QGEval metric proxies, RewardBench margin geometry. This shows breadth without treating them as dataset-wide theorem validation.

A2. **Monotone error-amplification audit.** Keep as assumption audit, not headline evidence.

A3. **Exact A vs normalized A_nu scaling audit.** This justifies de-emphasizing the old nu^2 headline.

A4. **Prospective prediction by group/dimension.** Useful for inspection and failure localization.

A5. **GPT/4o external-proxy diagnostics.** Use to show the external-proxy ranking/top-tail alignment pattern, not just score compression or tilt instability.

A6. **OOD transfer prediction.** Exp35: source-only transfer vs target calibration vs shrinkage prior. Defends against "same-dataset split is too easy."

A7. **Proxy perturbation baselines.** Exp36: shuffle/random/noisy/sign-flip controls. Shows exact A,B is alignment-sensitive, while A_nu is not.

A8. Optionally include detailed WMT split admissibility and per-dataset tables if space permits.


## main_table_1_direct_validation

| Dataset                | Setting                     |   Paths |   corr(A) |   corr(B) | sign(B) agree   |   B_th<0 % |   B_fit<0 % |   median R2 |
|:-----------------------|:----------------------------|--------:|----------:|----------:|:----------------|-----------:|------------:|------------:|
| SummEval               | human scalar LOO            |      32 |    0.9997 |    0.9934 | 32/32           |      100   |       100   |    1        |
| QGEval-human           | admitted human scalar paths |      63 |    0.9998 |    0.9302 | 63/63           |       98.4 |        98.4 |    0.999874 |
| OpenMEVA-MAGS ROC/WP   | admitted human scalar paths |      15 |    1      |    0.9975 | 13/15           |       13.3 |        26.7 |    1        |
| WMT MQM admitted paths | human scalar LOO            |      61 |    0.9999 |    0.9816 | 61/61           |      100   |       100   |    0.999999 |
| WMT pSQM               | human scalar LOO            |      12 |    1      |    0.9928 | 12/12           |       91.7 |        91.7 |    1        |


## main_table_2_prospective_prediction

| Setting       |   Splits |   sign acc. theory |   sign acc. emp-fit | MAE theory @0.10        | MAE A-only @0.10        | MAE A_nu @0.10          | MAE emp-fit @0.10       | MAE zero @0.10          | MAE oracle @0.10        |   mean Delta(0.10) |   abs(Delta)>0.01 % |   sign acc. if abs(Delta)>0.01 |
|:--------------|---------:|-------------------:|--------------------:|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|-------------------:|--------------------:|-------------------------------:|
| Human-heldout |     1589 |               90.7 |                97.4 | 0.0174 [0.0158, 0.0193] | 0.0310 [0.0229, 0.0409] | 0.0880 [0.0698, 0.1115] | 0.0168 [0.0155, 0.0183] | 0.2147 [0.2002, 0.2314] | 0.0041 [0.0023, 0.0064] |             0.2147 |                 100 |                           90.7 |
| GPT/4o proxy  |      200 |               97.5 |                98   | 0.0118 [0.0104, 0.0125] | 0.0165 [0.0130, 0.0200] | 0.1508 [0.1234, 0.1782] | 0.0117 [0.0103, 0.0126] | 0.0680 [0.0517, 0.0818] | 0.0005 [0.0003, 0.0009] |             0.068  |                 100 |                           97.5 |


## main_table_3a_finite_range_stability

| Setting       |   Paths |   median min Var_pi(r) |   median min ESS |   median max abs k4(r) |   median max abs k4(J,r) |   median tail-baseline |   tail<baseline % |
|:--------------|--------:|-----------------------:|-----------------:|-----------------------:|-------------------------:|-----------------------:|------------------:|
| Human-heldout |      32 |                  0.631 |             1338 |                  0.258 |                     0.15 |                  0.571 |                 0 |
| GPT/4o proxy  |       4 |                  0.375 |             1327 |                  0.035 |                     0.04 |                 -0.054 |                50 |


## main_table_3b_bon_bridge

| Setting       |   Paths |   median d_BoN(N=2) |   nontriv. BoN in d<=0.1 |   nontriv. BoN in d<=0.25 |   MAE BoN vs theory |   MAE BoN vs tilt |
|:--------------|--------:|--------------------:|-------------------------:|--------------------------:|--------------------:|------------------:|
| Human-heldout |      22 |               0.172 |                        0 |                     0.143 |              0.0045 |            0.0027 |
| GPT/4o proxy  |       4 |               0.189 |                        0 |                     0.143 |              0.0034 |            0.0033 |


## main_table_4_measurement_intervention

| Dataset          | Transition   |   Groups |   Pair reps |   Delta nu>0 % |   Delta A>0 % |   Delta B<0 % |   Delta H>0 % |   q_0.025(Delta H)>0 % |   median Delta A |   median Delta B |   median Delta H |
|:-----------------|:-------------|---------:|------------:|---------------:|--------------:|--------------:|--------------:|-----------------------:|-----------------:|-----------------:|-----------------:|
| SummEval         | k=1 to 2     |        4 |        2000 |          100   |           100 |         100   |           0   |                    0   |            0.1   |           -0.069 |           -0.569 |
| SummEval         | k=2 to 3     |        4 |        2000 |          100   |           100 |         100   |          75   |                    0   |            0.045 |           -0.018 |            0.002 |
| WMT MQM clusters | k=1 to 2     |       47 |       23500 |           95.7 |           100 |          72.3 |          72.3 |                   12.8 |            0.364 |           -0.089 |            0.331 |


## appendix_table_a1_feedback_extensions

| Dataset                | Setting                         |   Paths |   corr(A) |   corr(B) | sign(B) agree   |   B_th<0 % |   B_fit<0 % |   median R2 |
|:-----------------------|:--------------------------------|--------:|----------:|----------:|:----------------|-----------:|------------:|------------:|
| MT-Bench BT split-half | split-half BT utilities         |     200 |    0.9999 |    0.9831 | 186/200         |       93   |       100   |    1        |
| QGEval metrics         | automatic metrics vs human gold |     315 |    0.9991 |    0.6915 | 301/315         |       77.5 |        80   |    0.999996 |
| RewardBench margins    | cross-model margin geometry     |    2515 |    0.9999 |    0.9585 | 2359/2515       |       45.7 |        47.7 |    0.999999 |


## appendix_table_a2_monotone_audit

| Dataset          |   Paths |   pos. bin trend % |   CI-pos. trend % |   zero violations % |   median viol. frac |   median isotonic R2 |
|:-----------------|--------:|-------------------:|------------------:|--------------------:|--------------------:|---------------------:|
| SummEval         |      32 |               96.9 |              84.4 |                90.6 |                 0   |                0.424 |
| WMT MQM clusters |     141 |               97.2 |              91.5 |                39   |                 0.2 |                0.358 |


## appendix_table_a3_scaling_a_vs_nu

| Dataset          |   Paths |   median abs(rho)/sigma_J^2 |   abs(rho) small % |   median headroom error A_nu |   corr(A_exact,A_fit) |   corr(A_nu,A_fit) |   corr(B_th,B_fit) |
|:-----------------|--------:|----------------------------:|-------------------:|-----------------------------:|----------------------:|-------------------:|-------------------:|
| SummEval         |      32 |                       0.212 |                3.1 |                         22.2 |                0.9997 |             0.9821 |             0.9934 |
| WMT MQM clusters |     141 |                       0.338 |               13.5 |                         31   |                1      |             0.9814 |             0.9768 |


## appendix_table_a4_prediction_by_group

| Setting       | Group               |   Splits |   sign acc. theory |   sign acc. emp-fit |   MAE theory @0.10 |   MAE A_nu @0.10 |   mean Delta(0.10) |
|:--------------|:--------------------|---------:|-------------------:|--------------------:|-------------------:|-----------------:|-------------------:|
| human-heldout | coherence__expert   |      150 |              100   |               100   |             0.0137 |           0.1574 |             0.3319 |
| human-heldout | coherence__turker   |      250 |               91.6 |                95.6 |             0.0164 |           0.063  |             0.1949 |
| human-heldout | consistency__expert |      139 |              100   |               100   |             0.0325 |           0.125  |             0.2817 |
| human-heldout | consistency__turker |      250 |               76   |                95.6 |             0.0148 |           0.0594 |             0.2052 |
| human-heldout | fluency__expert     |      150 |              100   |               100   |             0.0229 |           0.1212 |             0.1943 |
| human-heldout | fluency__turker     |      250 |               82   |                93.2 |             0.0154 |           0.0597 |             0.1925 |
| human-heldout | relevance__expert   |      150 |              100   |               100   |             0.0161 |           0.1346 |             0.2016 |
| human-heldout | relevance__turker   |      250 |               91.6 |                99.2 |             0.0142 |           0.0596 |             0.1787 |
| GPT/4o proxy  | coherence           |       50 |               90   |                92   |             0.0125 |           0.1691 |             0.0865 |
| GPT/4o proxy  | consistency         |       50 |              100   |               100   |             0.0123 |           0.1356 |             0.0731 |
| GPT/4o proxy  | fluency             |       50 |              100   |               100   |             0.0126 |           0.1873 |             0.0446 |
| GPT/4o proxy  | relevance           |       50 |              100   |               100   |             0.0097 |           0.1113 |             0.0676 |


## appendix_table_a5_gpt_diagnostics

| Diagnostic                               |   Overall |
|:-----------------------------------------|----------:|
| median proxy std.                        |    0.678  |
| median Spearman(proxy,human gold)        |    0.285  |
| median Pearson(proxy,human gold)         |    0.329  |
| median top-10% overlap                   |    0.15   |
| median top-10% gold delta                |    0.1666 |
| median top-10% oracle gap                |    0.682  |
| std ratio model/human, median over dims  |    0.909  |
| Delta Spearman model-human, median       |   -0.288  |
| Delta top-10 overlap model-human, median |   -0.256  |


## appendix_table_a6_ood_transfer

| Predictor    |   MAE @0.10 |   sign acc. |
|:-------------|------------:|------------:|
| source_mean  |      0.0407 |         100 |
| target_cal   |      0.0258 |         100 |
| source_prior |      0.0235 |         100 |
| target_anu   |      0.2053 |         100 |
| source_anu   |      0.2031 |         100 |
| zero         |      0.2432 |           0 |


## appendix_table_a7_proxy_perturbation

| Mode          | sigma   |   actual Delta(0.1) |   A_exact |   A_nu |   Spearman |   MAE theory |   MAE A_nu |
|:--------------|:--------|--------------------:|----------:|-------:|-----------:|-------------:|-----------:|
| noisy         | 0.5     |              0.1893 |     0.678 |  0.855 |      0.5   |       0.017  |     0.0828 |
| noisy         | 1.0     |              0.1503 |     0.537 |  0.676 |      0.403 |       0.0165 |     0.0646 |
| noisy         | 2.0     |              0.0974 |     0.338 |  0.428 |      0.258 |       0.0156 |     0.0392 |
| oracle        | --      |              0.3246 |     1.111 |  0.954 |      1     |       0.0133 |     0.0664 |
| original      | --      |              0.2159 |     0.754 |  0.955 |      0.606 |       0.0158 |     0.0878 |
| random_normal | --      |              0.0009 |     0.003 |  0.957 |      0.003 |       0.0158 |     0.3017 |
| shuffle       | --      |             -0      |    -0.001 |  0.959 |     -0.001 |       0.0147 |     0.3033 |
| sign_flip     | --      |             -0.2577 |    -0.758 |  0.957 |     -0.605 |       0.0222 |     0.5602 |
