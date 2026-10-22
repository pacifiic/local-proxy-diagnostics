# proxy_open_judge_rm: SummEval open proxies at d = 0.10

**Placement.** Appendix table; one short sentence in main next to the GPT-4o proxy result
(CAMERA_READY_PLAN.md §3, "Learned RM / open judge on SummEval"). `proxy_open_judge_rm_by_dimension.tex`
is the per-dimension variant.

**What it shows.** Prospective endpoint MAE at KL d = 0.10 when an open reward model or an open LLM judge
replaces GPT-4o as the SummEval proxy, using the paper's original prospective protocol.

**Proxies (scoring run specs).**
- Skywork RM: `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`, resolved commit `d4117fbfd81b72f41b96341238baa1e3e90a4ce1`, bfloat16, max length
  4096, batch 16.
- Qwen judge: `Qwen/Qwen2.5-32B-Instruct`, resolved commit `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, 5 judge passes
  averaged, temperature 0.3, max 200 tokens, prompt
  `v2_rationale_float_rubric`.
- Both read the repaired SummEval item CSV (`4a29f598...`). Proxy CSV hashes
  match `data/open_proxy_inputs/README.md`.

**Definitions and protocol.** Same as the original SummEval prospective evaluation: per dimension, 30%
calibration (480 items) / 70% evaluation (1,120), 50 repeats, seed 0; A+B = A sqrt(d) + B d and A-only =
A sqrt(d) from calibration moments; A_nu = reliability-based variance-ratio surrogate; empirical fit =
two-term form fitted to the calibration-split exact tilt curve (fit up to d = 0.10); zero = 0. Target =
exact raw-score exponential-tilt gain on evaluation items with human gold, human-score units. MAE = mean
over 4 dimensions x 50 splits (200 rows, none skipped).

**Paired interval (computed here, not saved by the experiment).** The experiment saved only marginal
path-bootstrap intervals and an A_nu - A+B interval. This builder reproduces the engine's
path-clustered bootstrap (`experiments/prospective_prediction_path_macro/01_prospective_prediction.py`: resample the
4 dimension paths, 500 draws, `default_rng(0)`), checks that it reproduces every saved interval
exactly (< 1e-12), and then reports the paired A-only - A+B and empirical fit - A+B intervals from the same
draws. With only 4 clusters the interval is coarse (at most 35 distinct resamples) and reflects
between-dimension variation only.

**Sources.** `results/open_proxy/{proxy_rm_skywork,proxy_openjudge}/exp27_prospective_summary.csv`,
`.../exp27_prospective_paths.csv`, `.../exp27_run_spec.json`;
`data/open_proxy_inputs/scoring_run_specs/*.run_spec.json`; proxy CSVs (hash check only).

**Key results (d = 0.10).**
- Skywork RM: A+B 0.01126, A-only 0.01545, A_nu 0.04376, empirical fit
  0.01113, zero 0.06646; A+B reduction vs A-only 27.1%;
  A-only - A+B 0.00419 [0.00106, 0.00732]; empirical fit - A+B
  -0.00013 [-0.00037, 0.00009].
- Qwen judge: A+B 0.01120, A-only 0.01485, A_nu 0.09636, empirical fit
  0.01099, zero 0.08167; reduction 24.5%; A-only - A+B
  0.00364 [0.00026, 0.00992]; empirical fit - A+B
  -0.00021 [-0.00050, -0.00003].
- Split-level shares: A+B beats A-only in 66.0% / 60.5% of the 200
  splits (Skywork / Qwen) and beats empirical fit in 48.5% / 47.5%.

**Reporting details.** SummEval, d = 0.10 endpoint MAE averaged over four dimensions x 50 item splits
(480/1,120); model IDs; the Qwen judge averages five passes at T = 0.3; the paired interval is a
path-clustered bootstrap over only four dimension paths, computed for this paper from the saved split rows.

**Caveats.**
- Qwen coherence: A-only beats A+B (0.01197 vs 0.01244); the only one of the
  8 proxy/dimension cells to do so.
- Empirical fit has slightly lower MAE than A+B for both proxies and wins about half the splits; A+B is
  not better than empirical fit here. For Qwen the path-bootstrap interval of empirical fit - A+B
  ([-0.00050, -0.00003]) lies entirely below zero.
- A+B beats A-only in only 66.0%/60.5% of individual splits:
  the aggregate reduction is not split-by-split dominance.
- These Skywork scores come from the original open-proxy scoring run, NOT the exp37 shared-context
  Skywork scores: same model revision (`d4117fbf`), but exp37 re-scored with shared, further
  truncated article prefixes common to all three RMs; do not mix with `rm_endpoint_mae`. The near-identical
  Skywork A+B MAE in both (0.01126 vs 0.011262) is coincidental.
- Training-data overlap of these models with SummEval is not excluded.
