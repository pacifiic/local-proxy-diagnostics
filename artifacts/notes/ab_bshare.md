# ab_bshare: median magnitude share of the B term in the predicted gain (appendix)

## Placement
Appendix, A/B contribution section; the numeric companion of Figure 2(b). One tabular body
(`ab_bshare.tex`) with rows = datasets and columns = radius.

## Definition (must appear in the caption or adjacent text)
For each path-split, with calibration moment coefficients A-hat and B-hat:

    R_B(d) = |B-hat d| / (|A-hat sqrt(d)| + |B-hat d|).

The table reports the median over path-splits, and the IQR column gives the 25th-75th
percentiles at d = 0.10. R_B is a magnitude share of the *predicted* gain. It is not net B
(sign and cancellation are discarded), not a share of the realized gain, and not an MAE
improvement. For the latter see `ab_reduction`. Because R_B is a ratio, it is unitless and
comparable across datasets as a share. It still reflects each setting's own A/B ratio and is
not a quality score.

## Aggregation, support, protocol
- Coefficients: exp31 30%-calibration moment coefficients (seed-0 sequential split schedule,
  50 splits per path). These are bit-identical to the exp33 A+B predictor (asserted to 1e-12).
- Path-splits: SummEval 1,600, GPT-4o 200, WMT 7,050 (fixed_141), OpenMEVA 500. All
  path-splits are used at every radius, because R_B needs no reachable evaluation radius.
- exp31 stores rows at d in {0.05, 0.10} only. R_B at d = 0.01, 0.025 and 0.075 is computed
  from the same (A-hat, B-hat), and the build checks the saved d = 0.05/0.10 medians to 1e-12.
- Percentages: 1 decimal in the table; unrounded values in `artifacts/data/ab_bshare.csv`.

## Sources
- `results/coefficient_precision/coefficient_error_rows.csv`
- `results/coefficient_precision/coefficient_error_summary.csv`
- The exp21-canonical per-cell prediction files (equality check against the A+B predictor).
- Not used: `results/b_stability/contribution_level.csv` (exp22). It uses
  independent SHA-256 split streams and gives slightly different medians, for example
  SummEval 4.5% and GPT-4o 13.8% at d = 0.10. The publication table and these summaries use exp31.

## Key results (median R_B, two decimals; IQR at d = 0.10; share of path-splits with B-hat < 0)

| Dataset | d = 0.01 | 0.025 | 0.05 | 0.075 | 0.10 | IQR at 0.10 | B-hat < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SummEval | 1.45% | 2.27% | 3.19% | 3.87% | 4.45% | [2.2, 8.4]% | 90.8% |
| GPT-4o | 4.61% | 7.10% | 9.75% | 11.68% | 13.25% | [7.2, 18.6]% | 98.0% |
| WMT | 2.79% | 4.34% | 6.03% | 7.29% | 8.32% | [4.8, 12.3]% | 90.4% |
| OpenMEVA | 1.01% | 1.59% | 2.23% | 2.72% | 3.12% | [1.8, 4.0]% | 23.2% |

Rounding trap: SummEval at d = 0.10 is 4.4% at one decimal (4.447%), not 4.5%.

## Caveats the caption MUST include
1. R_B is the magnitude share of B-hat d in the calibration-predicted gain. It is not net B
   and not an MAE improvement.
2. Medians are over path-splits of the 30%-calibration pre-pool protocol. The share is
   heterogeneous across path-splits (see the IQR column).
3. B's share grows like sqrt(d) inside the window. The values say nothing about d > 0.10.
