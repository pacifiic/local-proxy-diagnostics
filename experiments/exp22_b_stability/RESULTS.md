# Exp22: B Stability and Decision Audit

## Pipeline

The experiment uses 50 deterministic calibration/evaluation splits. Within each
calibration draw, 300 bootstrap samples resample prompt clusters rather than
individual rows:

- SummEval: article cluster;
- OpenMEVA-MAGS: story cluster;
- WMT-MQM: source-segment cluster.

Outer split and inner bootstrap use separate SHA-256-derived RNG streams for
each dataset/path/repeat. Changing the bootstrap count therefore cannot change
later calibration/evaluation splits.

An independent rerun produced byte-identical row-level and run-specification files.
The joint \(A,B\) bootstrap uses one shared finite-value mask, and dataset-level
decision statistics are computed separately within each dataset.

## Coefficient stability

| Dataset | Path-repeat units | Median SE(A) | Median SE(B) | B CI includes 0 | Median B-sign stability |
|---|---:|---:|---:|---:|---:|
| SummEval human proxy | 1,600 | 0.0593 | 0.0570 | 52.6% | 0.967 |
| SummEval GPT-4o proxy | 200 | 0.0401 | 0.0376 | 25.0% | 1.000 |
| WMT-MQM human proxy | 7,050 | 0.4129 | 0.3870 | 49.8% | 0.973 |
| OpenMEVA-MAGS human proxy | 500 | 0.0461 | 0.0333 | 35.2% | 1.000 |

The bootstrap CI often includes zero even when the median bootstrap sign
agreement is high. These are different statistics and should not be conflated.

## Contribution and coefficient-shift bounds

At \(d=0.10\), the median curvature share
\(|B d|/(|A\sqrt d|+|Bd|)\) is approximately 4.5% for SummEval-human,
13.8% for GPT-4o, 8.1% for WMT, and 3.1% for OpenMEVA.
These are exp22-specific values from independent SHA-256-derived outer split
streams. Public curvature-share reporting instead uses exp31's
seed-0 sequential split schedule, which matches the submitted prospective
protocol; exp22 is retained for bootstrap uncertainty and decision stability.

The quantities
\[
E_A=\sqrt d\,|\hat A-A_{\rm eval}|,\qquad
E_B=d\,|\hat B-B_{\rm eval}|
\]
are coefficient-shift upper-bound components. Their sum is not the actual
prediction error because it omits cancellation and the finite-range remainder.
At \(d=0.10\), the median \(E_A/(E_A+E_B)\) bound share ranges from 0.72 to
0.79 across datasets.

## Decision audit

The operational threshold is \(\gamma=0.1\sigma_{J,\mathrm{cal}}\). The reported
\(D_{\rm safe}\) is the largest passing value on the finite grid
\(\{0.01,0.025,0.05,0.075,0.10\}\), not a continuous supremum.

| Dataset | A-only/A+B disagreement | Grid \(D_{\rm safe}\) median | Stable when B CI includes 0 |
|---|---:|---:|---:|
| SummEval human proxy | 0.0% | 0.10 | 100.0% |
| SummEval GPT-4o proxy | 25.5% | 0.00 | 88.0% |
| WMT-MQM human proxy | 2.5% | 0.10 | 98.1% |
| OpenMEVA-MAGS human proxy | 0.0% | 0.10 | 100.0% |

The decision is sensitive to the chosen operational threshold. For GPT-4o, the
A+B proceed rate at \(d=0.10\) is 83.0%, 27.5%, and 0% for
\(\gamma/\sigma_J=0.05,0.10,0.20\), respectively. WMT also changes materially
at 0.20. The threshold must therefore be presented as deployment-specific rather
than universal.

## Jackknife audit

The separate closed-form leave-one-out/leave-top-k calculation remains valid.
Its internal direct-recomputation checks pass. Those results are not regenerated
by the corrected bootstrap command and should be reported as a separate
sensitivity analysis.

## Scope

This experiment supports a cautious conclusion:

1. \(B\) can be uncertain at calibration scale.
2. Multiplication by \(d\) often limits its contribution inside the local window.
3. It can still alter operational decisions, especially for GPT-4o.
4. Decision conclusions depend on \(\gamma\).

The full E3 plan is not complete. Separate calibration-size, proxy-rater count,
gold-rater count, and injected-rating-noise sweeps remain future work.

## Reproduction

```bash
python experiments/exp22_b_stability/build_b_stability.py \
  --outdir experiments/exp22_b_stability/outputs/b_stability
python experiments/exp22_b_stability/audit_b_stability.py \
  --outdir experiments/exp22_b_stability/outputs/b_stability
```
