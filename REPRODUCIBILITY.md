# Reproduction Notes

## Active WMT Input

The release uses the corrected WMT preparation, which parses literal-tab files
without quote interpretation and includes document IDs in document-local segment
keys. The resulting input contains 47 complete rater-set clusters, 141 proxy
paths, and 7,050 valid 30/70 splits. Its SHA-256 is
`6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8`.
The endpoint A+B MAE at d=0.10 is 0.0846 with a descriptive 95% interval of
[0.0793, 0.0901]. Input hashes and result sources are recorded in the run
specifications and artifact manifests.

## Estimators and Reporting

- Endpoint and calibration-size intervals bootstrap paths while retaining all
  split rows. The recorded runs use 1,000 draws and seed 0.
- OOD figure intervals use 2,000 draws and seed 0, matching the supporting table.
- CurveMAE, DPO seed summaries, and reward-model article-cluster intervals use
  the aggregation units stated in their table and figure notes.
- Prompt-conditional BoN reachability is evaluated at the requested KL radius.
- Reward-model audits reject nonfinite values and incomplete model/dimension
  cells. B-sign labels distinguish curve-fitted and moment-derived coefficients.
- The operational reliability-only coefficient is
  `sqrt(2) * Var(J_cal) / SD(r_cal)`.

## Verification

`tools/verify_release.py` checks saved inputs, preparation guards, and
prediction-to-summary arithmetic. `--fresh-splits` additionally recomputes
selected paths and repeats while preserving the recorded RNG sequence. The
pinned NumPy 1.26.4 CPU environment is used for empirical fits near the checked
radius boundary; moment predictions and exact gains are stable across the
verified environments.

Saved-data verification does not rerun external API calls or GPU training.
Those environments and model revisions are documented in `docs/ENVIRONMENT.md`
and the experiment run specifications.
