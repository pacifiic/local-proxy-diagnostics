# exp23 — Pooled soft-selection scope audit (RESULTS)

> The WMT rows use the corrected document-aware `fixed_141` input. The
> paper-matched `submitted_139` counterpart is retained in the paired WMT result
> tree.

## Scope

This experiment applies alpha-mixture and rank-tilt distributions to the pooled
evaluation item set of each proxy/gold path. It is a valid finite-item-set
stress test, but it is not prompt-conditional reranking and must not be described
as the E1-A deployment policy.

The prompt-conditional common-temperature experiment is exp14. Exp23 asks the
narrower question of whether the local law remains informative for smooth
rank-based selection mechanisms inside the checked radius.

## Mechanisms

1. Alpha mixture:
   \(q_\alpha=(1-\alpha)p_0+\alpha q_{\mathrm{BoN},N=8}\).
2. Rank tilt:
   \(q_\lambda(i)\propto p_0(i)F(r_i)^\lambda\), with average-rank tie handling.

Both induced distributions and their KL values are computed exactly on the
finite item set. The atomless analytic BoN-radius formula is not used.

## Data and checks

The run contains 123,420 rows from SummEval human/GPT-4o paths, WMT-MQM, and
OpenMEVA-MAGS, with 30 deterministic calibration splits and 11 parameter values
per mechanism. Self-checks verify the reference-distribution limits, the pure
BoN endpoint, tie handling, and an independent discrete-KL calculation.

Canonical output: `results/rank_selection_scope/`.

## Baseline-complete results

The table below reports path-macro MAE for nontrivial in-window points
\(0.005<d\le0.10\).

| Dataset | Mechanism | A+B | A-only | \(A_\nu\) | Empirical fit | Zero |
|---|---|---:|---:|---:|---:|---:|
| SummEval human | Alpha mixture | 0.0189 | 0.0244 | 0.0556 | 0.0186 | 0.0939 |
| SummEval human | Rank tilt | 0.0103 | 0.0154 | 0.0458 | 0.0101 | 0.1163 |
| SummEval GPT-4o | Alpha mixture | 0.0168 | 0.0209 | 0.0976 | 0.0169 | 0.0241 |
| SummEval GPT-4o | Rank tilt | 0.0064 | 0.0065 | 0.0784 | 0.0063 | 0.0414 |
| WMT-MQM | Alpha mixture | 0.1207 | 0.1432 | 0.3917 | 0.1198 | 0.3140 |
| WMT-MQM | Rank tilt | 0.0492 | 0.0607 | 0.3197 | 0.0488 | 0.4794 |
| OpenMEVA-MAGS | Alpha mixture | 0.0174 | 0.0156 | 0.0235 | 0.0169 | 0.1552 |
| OpenMEVA-MAGS | Rank tilt | 0.0156 | 0.0139 | 0.0211 | 0.0153 | 0.1421 |

A+B is much better than zero and \(A_\nu\), and is competitive with the
empirical fit. It improves over A-only for SummEval-human and WMT, is nearly tied
for GPT-4o rank tilt, and is worse than A-only on OpenMEVA. The result therefore
supports an informative local approximation, not universal dominance of the
curvature term.

## Correlation reporting

Row-level correlations across many monotone radius points are 0.94--0.98, but
those rows are dependent and should not be presented as independent evidence.
`inwindow_correlation_by_path.csv` reports within-path correlations, while
`inwindow_baseline_path_macro.csv` gives the primary equal-path-weighted error
comparison.

## Summary

> As a separate pooled finite-item-set stress test, we evaluated smooth mixtures
> with a BoN distribution and a rank-based tilt. Inside the checked KL window,
> A+B is substantially better than zero and reliability-only baselines and is
> close to the empirical two-parameter fit. A-only comparisons are
> dataset-dependent, including an unfavorable OpenMEVA result. This audit does
> not substitute for the common-temperature prompt-conditional experiment.

## Reproduction

```bash
python experiments/exp23_mixture_ranktilt/build_mixture_ranktilt.py \
  --outdir results/rank_selection_scope_reproduced
python experiments/exp23_mixture_ranktilt/audit_mixture_ranktilt.py \
  --outdir results/rank_selection_scope_reproduced
```

The current regeneration produced 123,420 rows. Its four compact CSV summaries
are byte-identical to the previously audited corrected run.
