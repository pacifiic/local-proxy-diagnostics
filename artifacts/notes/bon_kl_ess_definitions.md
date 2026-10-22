# bon_kl_ess_definitions: induced KL, ESS and relative ESS for selection rules

**Intended display:** main text (definitions + applicability rule, ~1 paragraph); proofs/derivations stay
in Appendix A.7. Note only (no table); `bon_kl_ess_definitions.csv` records the numerical checks.

## Definitions (finite reference pool of m items, reference p0, rule-induced distribution p)
- **Induced KL radius:** d = KL(p || p0) = sum_i p_i log(p_i / p0_i) (nats). This is the radius that
  the local law A sqrt(d) + B d is indexed by; for uniform p0, d = sum_i p_i log(m p_i).
- **Effective sample size:** ESS = 1 / sum_i p_i^2 (items).
- **Relative ESS:** RelESS = (sum_i p_i^2 / p0_i)^(-1) = 1 / (1 + chi^2(p || p0)) in (0, 1]; for
  uniform p0, RelESS = ESS/m. RelESS = 1 means no reweighting.
- **Exact Best-of-N with replacement and ties (Appendix A.7):** draw N items i.i.d. from p0, keep
  the highest proxy score, break ties uniformly. With F(u) = p0(r <= u), F_-(u) = p0(r < u),
  p_i = p0_i w_N(r_i), w_N(u) = (F(u)^N - F_-(u)^N) / (F(u) - F_-(u)) (atom case) and
  w_N(u) = N F(u)^(N-1) (atomless). Computable from unlabeled proxy scores alone.

## Closed forms
- Atomless BoN: d_N = log N - (N-1)/N and RelESS_N = (2N-1)/N^2
  (E_p0[w_N^2] = int_0^1 N^2 u^(2N-2) du = N^2/(2N-1)). N = 2, 4, 8: d = 0.193, 0.636, 1.204;
  RelESS = 0.750, 0.438, 0.234. Measured ROUGE-1 medians (near-untied proxy): RelESS
  0.750/0.438/0.234, d 0.193/0.636/1.204.
- Rank tilt q_lambda proportional to p0 F(r)^lambda has atomless density (lambda+1) F^lambda = w_(lambda+1):
  **rank tilt with lambda equals BoN with N = lambda + 1** (atomless; with ties the two differ and
  both must be computed exactly).
- Mixture p_alpha = (1-alpha) p0 + alpha p_BoN: chi^2(p_alpha || p0) = alpha^2 chi^2(p_BoN || p0), so
  **RelESS(p_alpha) = 1 / (1 + alpha^2 chi^2(p_BoN || p0))**. KL(p_alpha || p0) has no closed form and
  is computed exactly.
- Ties cap the radius: as N -> infinity BoN (and the pooled exponential tilt as t -> infinity)
  converge to the uniform distribution on the n_top items tied at the maximum, so
  d <= log(m / n_top). Saturated proxies therefore give small radii even at moderate N.

## Numerical checks (all asserted in bon.py)
| Check | computed | reference | abs diff |
|---|---|---|---|
| tie formula = exhaustive enumeration, m=7 tied pool, N=1 (exact rational) | 0 | 0 | 0.0e+00 |
| tie formula = exhaustive enumeration, m=7 tied pool, N=2 (exact rational) | 0 | 0 | 0.0e+00 |
| tie formula = exhaustive enumeration, m=7 tied pool, N=3 (exact rational) | 0 | 0 | 0.0e+00 |
| tie formula = exhaustive enumeration, m=7 tied pool, N=4 (exact rational) | 0 | 0 | 0.0e+00 |
| A.7 closed form vs binomial tie sum, 32 SummEval paths, max |p diff| | 1.30104e-18 | 0 | 1.3e-18 |
| independent exact BoN vs saved d_BoN, 2,292 points, max |diff| | 1.46549e-14 | 0 | 1.5e-14 |
| independent exact BoN vs saved ESS/m, 2,292 points, max |diff| | 1.25056e-15 | 0 | 1.3e-15 |
| independent exact BoN vs saved gain, 2,292 points, max |diff| | 1.24484e-14 | 0 | 1.2e-14 |
| RelESS = ESS/m under uniform p0 (BoN N=4, 5 paths per setting) | 2.22045e-16 | 0 | 2.2e-16 |
| untied m=200k: d_N vs log N-(N-1)/N, N=2 | 0.193147 | 0.193147 | 3.0e-11 |
| untied m=200k: RelESS_N vs (2N-1)/N^2, N=2 | 0.75 | 0.75 | 4.7e-12 |
| untied m=200k: rank tilt lambda=1 KL vs BoN N=2 KL | 0.193145 | 0.193147 | 2.5e-06 |
| untied m=200k: rank tilt lambda=1 RelESS vs BoN N=2 RelESS | 0.750002 | 0.75 | 1.9e-06 |
| untied m=200k: d_N vs log N-(N-1)/N, N=4 | 0.636294 | 0.636294 | 1.9e-11 |
| untied m=200k: RelESS_N vs (2N-1)/N^2, N=4 | 0.4375 | 0.4375 | 1.1e-11 |
| untied m=200k: rank tilt lambda=3 KL vs BoN N=4 KL | 0.636292 | 0.636294 | 2.5e-06 |
| untied m=200k: rank tilt lambda=3 RelESS vs BoN N=4 RelESS | 0.437501 | 0.4375 | 1.1e-06 |
| untied m=200k: d_N vs log N-(N-1)/N, N=8 | 1.20444 | 1.20444 | 6.8e-11 |
| untied m=200k: RelESS_N vs (2N-1)/N^2, N=8 | 0.234375 | 0.234375 | 2.8e-11 |
| untied m=200k: rank tilt lambda=7 KL vs BoN N=8 KL | 1.20444 | 1.20444 | 2.5e-06 |
| untied m=200k: rank tilt lambda=7 RelESS vs BoN N=8 RelESS | 0.234376 | 0.234375 | 5.9e-07 |
| ROUGE-1 median RelESS N=2 vs (2N-1)/N^2 | 0.750001 | 0.75 | 1.1e-06 |
| ROUGE-1 median d N=2 vs log N-(N-1)/N | 0.193145 | 0.193147 | 1.7e-06 |
| ROUGE-1 median RelESS N=4 vs (2N-1)/N^2 | 0.437503 | 0.4375 | 2.5e-06 |
| ROUGE-1 median d N=4 vs log N-(N-1)/N | 0.63629 | 0.636294 | 4.7e-06 |
| ROUGE-1 median RelESS N=8 vs (2N-1)/N^2 | 0.234378 | 0.234375 | 3.0e-06 |
| ROUGE-1 median d N=8 vs log N-(N-1)/N | 1.20443 | 1.20444 | 1.4e-05 |
| mixture RelESS identity, BoN N=8, 191 paths x 6 alphas, max |diff| | 4.44089e-16 | 0 | 4.4e-16 |
| saturation: d_BoN(N=64) -> log(m/n_top), consistency__expert__heldout_0 | 0.157385 | 0.157385 | 5.6e-17 |
| saturation: d_BoN(N=64) -> log(m/n_top), consistency__expert__heldout_1 | 0.150823 | 0.150823 | 0.0e+00 |
| saturation: d_BoN(N=64) -> log(m/n_top), consistency__expert__heldout_2 | 0.115832 | 0.115832 | 4.2e-17 |
| saturation: d_BoN(N=64) -> log(m/n_top), fluency__expert__heldout_2 | 0.137826 | 0.137826 | 1.4e-16 |
| exact tilt hits d=0.10 on every path, max |d-0.10| | 1.36002e-15 | 0 | 1.4e-15 |

## Orientation: RelESS of the exact pooled exponential tilt at d = 0.10
| Setting | median | range |
|---|---|---|
| SummEval, held-out human | 0.836 | 0.819-0.898 |
| SummEval, ROUGE-1 | 0.814 | 0.814-0.814 |
| SummEval, GPT-4o | 0.829 | 0.824-0.834 |
| WMT MQM, held-out human | 0.855 | 0.840-0.891 |
| OpenMEVA-MAGS, held-out human | 0.821 | 0.817-0.824 |

(For Gaussian scores an exponential tilt has RelESS = exp(-2d), 0.819 at d = 0.10.) Different rule
families reach different RelESS at the same KL, so KL remains the gate quantity and RelESS is a
companion concentration diagnostic.

## Suggested applicability-rule wording (main text)
"Before using A and B for a planned selection rule, translate the rule into the distribution it
induces on the reference pool. For Best-of-N this is computable from unlabeled proxy scores
(Appendix A.7, including ties). Report the induced radius d = KL(p || p0) and relative ESS
(sum_i p_i^2/p0_i)^(-1), and compare d with the checked window (here d <= 0.10). Use A sqrt(d) + B d
only inside that window; otherwise lower the selection pressure (smaller N, a mixture weight alpha, or a
rank-tilt exponent lambda), validate the rule directly, or abstain. Hard BoN usually leaves the window
already at N = 2 (median d = 0.17-0.19 across our settings), but proxies with many tied top scores can
keep it inside (four SummEval paths at N = 2)."

## Key results
- N=2 median d: SummEval held-out human 0.1718, ROUGE-1 0.1931,
  GPT-4o 0.1889, WMT 0.1860, OpenMEVA 0.1656
  (from `bon_nladder`).
- 4 of 2,101 nontrivial exact-BoN points (191 paths x 11 N) have d <= 0.10.

## Caveats for the main text
- The KL is computed on the finite reference pool actually used (pooled or per prompt); prompt-
  conditional BoN gives a different, per-prompt radius (`bon_prompt_conditional_scope`).
- "Inside the window" is necessary, not sufficient: equal-KL discrete rules can have different
  gains (plan section 6).

## Sources
`results/bon_scope/bon_gate_points.csv`; prepared scores `data/summeval_long.csv`, `data/wmt_mqm_long.csv`, `data/openmeva_long.csv`,
`data/proxy_rouge1.csv`, `data/model_proxy_4o_aggregated.csv`; `lib/common_experiment_utils.py` (path builders, tilt helpers);
Appendix A.7 of `paper/appendix.tex` for the w_N derivation.
