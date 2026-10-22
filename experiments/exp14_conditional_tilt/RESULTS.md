# exp14 — Common-temperature prompt-conditional tilt (RESULTS)

## Question

Does the local law continue to work when the prompt marginal is held fixed and
one shared temperature is applied within every prompt response set?

The implemented policy is

\[
q_t(y\mid x)\propto p_0(y\mid x)\exp\{t r(x,y)\},\qquad
d(t)=\mathbb E_x[\mathrm{KL}(q_t(\cdot\mid x)\Vert p_0(\cdot\mid x))].
\]

The same \(t\) is used for every prompt. Prompt-level KL values therefore vary;
only their mean is fixed at the requested radius. This differs from the earlier
equal-KL-per-prompt implementation.

## Protocol

- Prompt-level 30/70 calibration/evaluation split, 20 deterministic repeats.
- SummEval: article prompts with 16 system summaries.
- OpenMEVA-MAGS: story prompts with 5 continuations.
- WMT-MQM: source-segment prompts with 10--15 systems.
- Human-proxy paths hold one rater out as the proxy and average the remaining
  raters as gold.
- Target radius: mean prompt-level KL \(d=0.10\).
- \(A_c,B_c\) use context-averaged within-prompt cumulants, not averages of
  separately normalized per-prompt coefficients.

## Results

| Dataset | Paths | Mean realized gain | A+B MAE | A-only MAE | \(A_\nu\) MAE | Empirical-fit MAE | Zero MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SummEval | 32 | 0.1997 | 0.0198 | 0.0321 | 0.0844 | 0.0186 | 0.1997 |
| WMT-MQM | 141 | 0.4738 | 0.1033 | 0.1016 | 0.4029 | 0.1002 | 0.4738 |
| OpenMEVA-MAGS | 10 | 0.2695 | 0.0161 | 0.0152 | 0.0211 | 0.0156 | 0.2695 |

A+B beats zero on every path in all three datasets. It also improves strongly
over the reliability-only surrogate in aggregate. The A-only comparison is
dataset-dependent: A+B is clearly better on SummEval, statistically tied on WMT,
and slightly worse on OpenMEVA. The deployable empirical two-parameter fit is
slightly better than A+B on all three datasets. These results support the
functional local form and the usefulness of the moment diagnostic, but not a
predictive-SOTA claim over flexible calibration baselines.

## Pooled comparison

The common-temperature conditional gains remain substantial relative to the
previous pooled global-tilt gains. This is evidence that the headline signal is
not solely produced by moving mass across prompts. It is not an additive
within/between decomposition: pooled and conditional tilts are different
policies with different mass-allocation rules, so their gain ratio must not be
reported as a causal share.

## Reachability

Discrete and tied human scores can make the calibration frontier unable to
reach \(d=0.10\). The common-temperature implementation computes the exact
maximum reachable mean KL. The empirical fit is estimated only over the
reachable calibration radius and may then predict at \(d=0.10\). A split is
excluded only when the evaluation policy itself cannot reach the requested
radius. All retained paths have all 20 repeats in the reported run.

## Validation

`audit_conditional.py` independently verifies:

- exact path counts: 32 SummEval, 10 OpenMEVA, and 141 WMT paths;
- system-aligned proxy/gold pivots and held-out-rater exclusion;
- lossless WMT item-ID parsing;
- one shared temperature reaches the target mean KL while per-prompt KLs differ;
- \(A_c,B_c\) equal an independent context-averaged-cumulant calculation;
- deterministic end-to-end reconstruction of stored path results.

The audit passes. Write regenerated outputs to a fresh directory.

## Summary

> We additionally evaluated the prompt-conditional policy
> \(q_t(y\mid x)\propto p_0(y\mid x)e^{t r(x,y)}\), using one shared temperature
> and defining the radius as average prompt-level KL. Coefficients estimated
> from disjoint calibration prompts predicted held-out conditional gain with
> MAE 0.020 on SummEval, 0.103 on WMT-MQM, and 0.016 on OpenMEVA-MAGS, compared
> with zero-baseline MAE 0.200, 0.474, and 0.270. A-only and empirical-fit
> comparisons are dataset-dependent and are reported without claiming
> predictive superiority. This experiment shows that the local diagnostic is
> not limited to a pooled policy that changes the prompt marginal.
