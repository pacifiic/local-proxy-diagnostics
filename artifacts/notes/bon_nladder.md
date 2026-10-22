# bon_nladder: exact pooled Best-of-N ladder by dataset

This artifact summarizes exact pooled Best-of-N behavior by dataset. "SummEval human",
"WMT MQM" and "OpenMEVA-MAGS" are held-out-human-proxy paths; "SummEval ROUGE-1"
and "SummEval GPT-4o" use the named model proxy. The generated tables provide a
per-dataset summary and a compact view across N.

## Definitions
- **Path:** one proxy/gold pairing on a fixed finite item set of size m. Reference
  p0 = uniform over the m items. Held-out-human paths use one rater as proxy and the
  mean of the remaining raters of the same group as gold; ROUGE-1 / GPT-4o paths use the
  model proxy and the mean of all SummEval ratings for that dimension as gold.
- **Exact BoN (with replacement, uniform tie-breaking; Appendix A.7):**
  p_i = w_N(r_i)/m, w_N(u) = (F(u)^N - F_-(u)^N)/(F(u) - F_-(u)), F and F_- the right/left
  proxy CDFs under p0. Pooled over the path's whole item set (not prompt-conditional).
- **d_BoN** = KL(p || p0) = sum_i p_i log(m p_i), in nats.
- **ESS** = 1/sum_i p_i^2; **RelESS** = (sum_i p_i^2/p0_i)^(-1) = ESS/m (uniform p0).
- **Actual BoN gain** = sum_i p_i j_i - mean(j), in native human-score units.
- **Predictions at d = d_BoN:** A+B = A sqrt(d) + B d; A-only = A sqrt(d); A_nu = A_nu sqrt(d);
  empirical fit = a sqrt(d) + b d least-squares fitted to the exact exponential-tilt curve on
  d <= 0.10 of the same pool; zero = 0.
- **Nontrivial point:** a (path, N) pair with N >= 2. The ladder has 11 nontrivial N values
  per path: N in {2,3,4,6,8,12,16,24,32,48,64} (N = 1 is the reference, d = 0).
- **In-window:** d_BoN <= 0.10 (the checked local window). **Bridge:** d_BoN <= 0.25.

## Aggregation, units, support
- Medians are over paths (each path weight 1) at fixed N. The "All settings" row pools the
  191 paths, so its medians are path-weighted and dominated by WMT (141/191).
- Point counts are over (path, N) pairs; shares divide by paths x 11.
- MAE = mean absolute error over in-window nontrivial points (each point weight 1). `--`: the
  setting has no nontrivial point with d <= 0.10, so there is nothing to score in-window.
- d in nats; RelESS dimensionless; MAE in native human units (SummEval and OpenMEVA 1-5
  Likert, WMT MQM points). Do not compare MAE across settings.
- Support: SummEval 32 held-out-human paths (4 dimensions x {3 expert, 5 turker} raters),
  ROUGE-1 4, GPT-4o 4 (one per dimension), m = 1,600; WMT MQM 141 paths (fixed_141 lineage,
  checked with `require_wmt_fixed`), m = 430-2820; OpenMEVA-MAGS 10 paths, m = 1,000.
  Total 191 paths x 12 N = 2,292 points.

## Protocol
- Pooled, full-pool: A, B, A_nu and the empirical fit are computed on the same full item pool
  on which BoN is evaluated (exp20; no calibration split). These comparisons are descriptive
  same-pool audits, not held-out predictions.
- Outside d <= 0.10 the predictions are extrapolations; they are reported only in the data CSV.

## Verification performed by bon.py
- Path counts per dataset and the 12-value N ladder; `d_bon == d_bon_recheck` everywhere.
- All 2,292 (path, N) points re-derived from the prepared score files with an independent
  implementation of w_N: max |d diff| = 1.5e-14, max |ESS diff|/m = 1.3e-15,
  max |gain diff| = 1.2e-14, max |A diff| = 4.4e-16, max |B diff| = 4.4e-16.
- ROUGE-1 (almost untied proxy) medians match the atomless closed forms
  d_N = log N - (N-1)/N and RelESS_N = (2N-1)/N^2 at N = 2, 4, 8 within 5e-4.
- Audited numbers asserted with `check()`: N=2 medians .1718/.1931/.1889/.1860/.1656;
  exactly 4 nontrivial in-window points; in-window MAE A+B .01968, A-only .08836.

## Key results
| Setting | Paths | median d (N=2) | median RelESS (N=2) | median d (N=4) | median d (N=8) | median d (N=64) | points d<=.10 | points d<=.25 |
|---|---|---|---|---|---|---|---|---|
| SummEval, held-out human | 32 | 0.1718 | 0.7716 | 0.549 | 0.988 | 1.266 | 4/352 | 92/352 |
| SummEval, ROUGE-1 | 4 | 0.1931 | 0.7500 | 0.636 | 1.204 | 3.174 | 0/44 | 4/44 |
| SummEval, GPT-4o | 4 | 0.1889 | 0.7540 | 0.619 | 1.171 | 3.000 | 0/44 | 4/44 |
| WMT MQM, held-out human | 141 | 0.1860 | 0.7600 | 0.582 | 0.959 | 1.131 | 0/1551 | 154/1551 |
| OpenMEVA-MAGS, held-out human | 10 | 0.1656 | 0.7715 | 0.560 | 1.059 | 2.563 | 0/110 | 10/110 |

- Nontrivial in-window points: 4 of 2101 (0.19%),
  all SummEval held-out-human at N=2 (consistency expert held-out raters 0/1/2 and fluency
  expert held-out rater 2). 187 of 191 paths have no nontrivial in-window point.
- Smallest N=2 radius outside SummEval: WMT 0.1013 (just above 0.10).
  SummEval held-out-human N=2 radii range 0.0781-0.1819.
- In-window MAE (n = 4 points): A+B 0.01968, A-only 0.08836,
  A_nu 0.12381, zero 0.24896, empirical fit 0.00432.
- d <= 0.25 counts include saturated paths: 6 SummEval expert paths and 1 WMT
  path have reachable KL log(m/n_top) <= 0.25, so BoN stays below 0.25 for every N. In SummEval,
  60 of the 92 d <= 0.25 points are such N >= 3 points; in WMT,
  13 of 154.
- Descriptive N=2 MAE over all paths (mostly outside the window; data CSV `N2_mae_*`), e.g.
  SummEval held-out human A+B 0.0107 vs A-only 0.0356.


## Sources
- `results/bon_scope/bon_gate_points.csv` (exp20, `experiments/exp20_bon_window_gate/build_bon_gate.py`), `results/bon_scope/run_spec.json`
- Prepared scores for re-verification and tie counts: `data/summeval_long.csv`, `data/wmt_mqm_long.csv`,
  `data/openmeva_long.csv`, `data/proxy_rouge1.csv`, `data/model_proxy_4o_aggregated.csv`; path builders `lib/common_experiment_utils.py`
