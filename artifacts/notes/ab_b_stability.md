# ab_b_stability: how reliably B is estimated (appendix)

## Placement
Appendix, B-stability section. The main text gets one paragraph on when B becomes
unreliable. Two tabular bodies, to be stacked as panels (a) and (b) of one table:
- `ab_b_stability.tex`, (a) calibration-size coefficient error at d = 0.10 (exp31).
- `ab_b_stability_bootstrap.tex`, (b) bootstrap sign/CI stability and decision
  disagreement (exp22).
The panels come from different split schedules (exp31 seed-0 sequential, exp22 SHA-256 streams).
Keep them as separate panels and do not merge them row-wise.

## Panel (a) definitions (exp31; medians over path-splits at d = 0.10)
- sqrt(d)|A-hat - A_eval| and d|B-hat - B_eval|: A_eval and B_eval are the *moment-formula*
  coefficients computed on the held-out evaluation split, used only retrospectively. These are
  the first two coefficient-shift terms of the paper's prediction-error decomposition. They
  combine calibration estimation error with calibration-to-evaluation shift, and they are
  triangle-bound components, not a causal decomposition of each split's error.
- Remainder: |Delta_eval(d) - (A_eval sqrt(d) + B_eval d)|, the held-out exact tilt gain minus
  its own evaluation-moment two-term curve (finite-range local-law remainder).
- Sign agr.: share of path-splits with sign(B-hat_cal) = sign(B_eval,moment). This is a
  **moment-based** sign agreement. The earlier curve-fit audit instead compared B-hat_cal
  with a B *fitted to the held-out exact curve* (curve-fit sign). The two definitions answer different questions and must remain separate. This table does not report curve-fit signs.
- Calibration fractions 5/10/30%, 50 nested splits per path, seed 0. Path-splits: 1,600 /
  200 / 7,050 / 500. These are the same as panel (b), so the column is omitted from (a). Values are in native human-score units (4 decimals). Compare them across
  calibration sizes within a dataset, not across datasets.

## Panel (b) definitions (exp22; 30% calibration, 50 splits, 300 prompt-cluster bootstrap draws within each calibration split)
- Clusters: SummEval articles, OpenMEVA stories, WMT source segments.
- Median SE(B-hat): bootstrap standard error of B-hat, with the median over path-splits (native units).
- Sign kept: per path-split, the share of bootstrap draws with the same sign as B-hat. The
  table gives the dataset median.
- CI includes 0: share of path-splits whose 95% percentile bootstrap CI for B contains 0.
  These are *path-split* CIs (one per calibration split), not dataset-level CIs. Sign stability
  and zero-inclusion are different summaries of the same bootstrap distribution, and high
  sign stability does not establish nonzero population curvature.
- Disagree: share of path-splits where A-only and A+B make different proceed/abstain
  decisions at d = 0.10. The rule is: proceed if the 2.5th bootstrap percentile of the
  predicted gain exceeds gamma = 0.1 sigma_J,cal. Gamma is an operational, deployment-specific
  threshold. For GPT-4o, the A+B proceed rate is 83.0/27.5/0% at gamma/sigma_J = 0.05/0.10/0.20
  (`gamma_sensitivity.csv`).

## Sources
- `results/coefficient_precision/coefficient_error_rows.csv`
  (recomputed medians) and `coefficient_error_summary.csv` (asserted equal, max |diff| 5.6e-17).
- `results/b_stability/coefficient_level.csv`, `decision_level.csv`,
  `b_stability_run_spec.json` (WMT = 7,050 fixed_141 path-splits).

## Key results
At 30% calibration (d = 0.10):

| Dataset | sqrt(d)|dA| | d|dB| | Remainder | Moment sign agr. |
|---|---:|---:|---:|---:|
| SummEval | 0.0157 | 0.0044 | 0.0018 | 89.4% |
| GPT-4o | 0.0116 | 0.0037 | 0.0004 | 97.0% |
| WMT | 0.0777 | 0.0317 | 0.0087 | 89.7% |
| OpenMEVA | 0.0126 | 0.0029 | 0.0027 | 90.0% |

Ranges at 30%: sqrt(d)|dA| 0.0116-0.0777, d|dB| 0.0029-0.0317, remainder 0.0004-0.0087.
d|dB| decreases monotonically from 5% to 10% to 30% calibration in every dataset.

| Dataset | Bootstrap sign kept (median) | Path-split CIs including 0 | Decision disagreement | Proceed A+B / A-only |
|---|---:|---:|---:|---:|
| SummEval | 96.7% | 52.6% | 0.0% | 100.0% / 100.0% |
| GPT-4o | 100.0% | 25.0% | 25.5% | 27.5% / 53.0% |
| WMT | 97.3% | 49.8% | 2.5% | 93.1% / 94.4% |
| OpenMEVA | 100.0% | 35.2% | 0.0% | 100.0% / 100.0% |

## Caveats the caption MUST include
1. B_eval and A_eval are held-out *moment* coefficients, and the sign agreement is
   moment-based, not curve-fit.
2. "CI includes 0" counts per-path-split bootstrap CIs. High bootstrap sign stability coexists
   with 25-53% of CIs including 0.
3. The decision columns depend on the operational threshold gamma = 0.1 sigma_J,cal.
4. Panels (a) and (b) use different split schedules (exp31 vs exp22).
5. Error columns are in native human-score units and must not be compared across datasets.
