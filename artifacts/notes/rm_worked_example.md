# rm_worked_example: three-RM calibration-only worked example (exp37)

**Placement.** Appendix (CAMERA_READY_PLAN.md §8.1, §8.3). Space was short, so Section 5.1 states all
three decompositions in prose, including the DeBERTa row where adding B moves away from the held-out
gain, and points to this table.

**What it shows.** For one preselected split, the calibration-only coefficients, the separate
first-order (A sqrt(d)) and second-order (B d) contributions at d = 0.10, their sum, and the exact
held-out gain.

**Definitions and units.** Proxy r = frozen scalar reward of Skywork-Reward-Llama-3.1-8B-v0.2,
ArmoRM-Llama3-8B-v0.1 (final gated scalar) or OpenAssistant reward-model-deberta-v3-large-v2. Target
J = mean of three expert and five crowd SummEval coherence ratings (human coherence-score units).
A, B estimated on the 480 calibration items only. Predicted gain = A sqrt(.10) + B(.10). Held-out
gain = exact raw-score exponential-tilt gain at KL d = 0.10 on the 1,120 evaluation items using human
gold. A-only prediction = A sqrt(.10).

**Protocol.** SummEval coherence, repeat 0, d = 0.10, chosen before inspecting this run's results
(`paper_example/report_spec.json`: "Preselected coherence, repeat 0, d=0.10; no outcome-based example selection"). Item-level split (480/1,120), not
article-held-out; identical across the three models. Shared context: every model sees the same article
prefix; supplied prefixes are already capped at 2,000 characters and are further shortened to fit all
three tokenizers (4,096/4,096/512 tokens), keeping each summary in full; mean retained fraction
96.08% of the supplied prefixes (not of full articles). No fine-tuning, no new annotation.
Exp37 was analysed with NumPy 2.3.5; nothing is recomputed here.
Pinned models (`prepared/models.json`): `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` @ `d4117fbfd81b72f41b96341238baa1e3e90a4ce1` (bfloat16, max 4096 tokens); `RLHFlow/ArmoRM-Llama3-8B-v0.1` @ `eb2676d20da2f2d41082289d23c59b9f7427f955` (bfloat16, max 4096 tokens); `OpenAssistant/reward-model-deberta-v3-large-v2` @ `c355404efa9ad2ad069f3a197cae0523c14244fc` (float32, max 512 tokens).

**Sources.** `results/reward_model_coefficients/canonical/paper_example/worked_example.csv`
(values), `.../paper_example/table_reward_model_worked_example.tex` (checked: the regenerated body equals
it cell for cell; the only textual change is `$-$` for minus signs), `.../paper_example/report_spec.json`.

**Key results.** ArmoRM: A = 0.2695, B = -0.0579,
A sqrt(.10) = 0.0852, B(.10) = -0.0058, predicted
0.0794, held-out exact 0.0775. Split errors
|pred - held-out| A+B vs A-only: Skywork 0.0122 vs 0.0127;
ArmoRM 0.0019 vs 0.0077; DeBERTa
0.0063 vs 0.0018. Coherence 50-repeat endpoint MAE
A+B vs A-only: 0.0135/0.0137 (Skywork),
0.0102/0.0116 (ArmoRM),
0.0102/0.0116 (DeBERTa).

**Reporting details.**
1. SummEval coherence, repeat 0, d = 0.10 preselected before inspecting results; one illustrative split,
   not the aggregate performance claim (that uses all 50 repeats: `rm_endpoint_mae`).
2. A, B from the 480 calibration items only; these calibration-only values differ from the full-pool
   descriptive coefficients (`rm_coefficients`), which use all 1,600 items and are not held-out predictors.
3. Held-out gain = exact raw-score exponential-tilt gain on 1,120 evaluation items, in human coherence
   units. Item-held-out, not article-held-out. Shared, truncated article context across models.
4. For DeBERTa, adding B worsens the error on this split (0.0063 vs 0.0018).

**Caveats.** Raw-score tilt on a fixed pool, not DPO/RLHF training; do not merge with the DPO table.
A, B describe this proxy/target/distribution/context, not intrinsic RM quality. Negative B is a local
correction, not a global turning point. Training-data overlap of the RMs with SummEval is not excluded.
Context preprocessing was frozen on this benchmark's texts before splitting.
