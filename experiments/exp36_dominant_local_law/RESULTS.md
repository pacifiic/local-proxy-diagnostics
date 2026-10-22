# Exp36 Results: Is the Two-Term Local Law Dominant?

## Question

The experiment asks two distinct questions.

1. Does a flexible function of radius \(d\), fit only on the 30% calibration
   item split, materially improve prediction of the exact gain curve on the
   held-out 70% item split?
2. After removing \(A\sqrt d+Bd\), is there a large and transferable residual
   function left to learn?

This is a pre-pool comparison. No prospective method sees evaluation proxy
scores or evaluation human gold. The `eval_oracle_A+B` diagnostic is the sole
exception and is labeled retrospective because it uses evaluation moments only
to isolate truncation error.

## CurveMAE

| Dataset | Moment \(A+B\) | Fitted two-term | Fitted cubic | Direct spline | \(A+B\) + residual spline |
|---|---:|---:|---:|---:|---:|
| SummEval | 0.01185 | 0.01170 | 0.01170 | 0.01171 | 0.01170 |
| GPT-4o proxy | 0.00806 | 0.00805 | 0.00805 | 0.00805 | 0.00805 |
| WMT MQM, corrected 141 paths | 0.05921 | 0.05843 | 0.05842 | 0.05842 | 0.05842 |
| OpenMEVA-MAGS | 0.00991 | 0.00984 | 0.00984 | 0.00984 | 0.00984 |

Relative to moment \(A+B\), the direct spline changes CurveMAE by:

- SummEval: \(-0.0001434\), path-bootstrap 95% CI
  \([-0.0004414, 0.0000409]\).
- GPT-4o proxy: \(-0.0000116\), CI
  \([-0.0000622, 0.0000671]\).
- OpenMEVA-MAGS: \(-0.0000653\), CI
  \([-0.0002090, 0.0000600]\).
- WMT MQM: \(-0.0007874\), CI
  \([-0.0010077, -0.0005856]\).

Only WMT shows a clear difference from moment \(A+B\), and its absolute
improvement is about 1.33% of the \(A+B\) CurveMAE.

## Does General Functional Flexibility Matter?

The cleaner functional-form comparison is the direct spline against the
empirical two-term fit. Both are fit to the same exact calibration curve; they
differ mainly in functional flexibility.

- SummEval: spline minus two-term \(=0.0000032\), CI
  \([-0.0000117, 0.0000162]\).
- GPT-4o proxy: \(0.0000007\), CI
  \([-0.0000007, 0.0000034]\).
- OpenMEVA-MAGS: \(-0.00000008\), CI
  \([-0.0000039, 0.0000033]\).
- WMT MQM: \(-0.0000078\), CI
  \([-0.0000147, -0.0000009]\).

Thus, even where the WMT difference is statistically resolved, replacing the
two-term calibration-curve fit with the spline improves CurveMAE by only about
0.013% relative to the two-term fit. The fitted cubic and residual-spline
results lead to the same conclusion.

The SummEval entries above use the same 7,989-cell common support as the
calibration-baseline comparison.

## Retrospective Truncation Diagnostic

Using evaluation moments retrospectively removes coefficient transfer error
and isolates the finite-radius truncation residual. Across path/repeat units,
the median fraction of gain energy explained by evaluation-oracle \(A+B\) is:

- SummEval: 99.9949%.
- GPT-4o proxy: 99.9982%.
- WMT MQM: 99.9931%.
- OpenMEVA-MAGS: 99.9959%.

The corresponding median max absolute residuals over \(d\leq0.10\) are
0.00178, 0.00040, 0.00872, and 0.00272, respectively.

This diagnostic is not a predictive result. It shows that the two leading
terms account for nearly all of the exact local curve shape once coefficient
estimation and item-split transfer are removed.

## Interpretation

The experiment supports a measured version of the meeting hypothesis:

> A much more flexible radius regression does not reveal a materially different
> local curve shape. Most of its apparent advantage over moment \(A+B\) comes
> from fitting coefficients directly to the calibration curve, not from needing
> a substantially richer function of \(d\).

This does not establish global optimality of the two-term law, and it does not
turn the empirical spline into an invalid baseline. It shows that within the
validated \(d\leq0.10\) window, higher-order functional flexibility contributes
very little beyond the theoretically derived leading terms.

## Reproducibility

Two complete runs were byte-for-byte identical. Compact summaries and one
canonical set of detailed prediction rows are published under
`results/dominant_local_law/`.

The first audit found that interpolating a 201-point \(t\)-grid onto the
\(d\)-grid could introduce a maximum pointwise calibration-curve error of
\(9.07\times10^{-5}\). The final implementation instead solves the KL root
independently at all 41 fit radii. This changed the largest aggregate CurveMAE
by only \(2.65\times10^{-7}\), so the rounded table and interpretation did not
change.

The data-chain audit independently:

- reconstructs all 187 paths from the four prepared input tables;
- checks all 9,350 calibration/evaluation splits;
- recomputes all 46,750 held-out exact gain targets with a separate KL solver;
- refits 4,675 flexible-model predictions from raw path arrays; and
- verifies that only the explicitly retrospective oracle accesses evaluation
  moments.

The maximum exact-gain discrepancy is \(4.44\times10^{-16}\). The maximum
independent flexible-model discrepancy is \(2.53\times10^{-9}\), attributable
to floating-point linear algebra.

Eleven of 1,600 SummEval path/repeat units cannot reach \(d=0.10\) on the
calibration pool. The fitted regressions can extrapolate, whereas the exact
calibration-curve row is unavailable there. A common-support sensitivity that
excludes those 11 units gives SummEval CurveMAE 0.01174 for moment \(A+B\),
0.01155 for the fitted two-term model, and 0.01156 for the direct spline. The
interpretation is unchanged.

The GPT-4o aggregate has 3, 6, 9, or 12 contributing scores per canonical
item. This is not a resume duplication: the raw table has exactly 19,200
unique `(dimension, item, judge_rep)` rows, or three passes per original
prompt. The larger counts arise when duplicate article-summary texts produced
by multiple SummEval model ids are pooled during the text-based item-id repair.
The prepared aggregate reproduces exactly from those raw rows.
