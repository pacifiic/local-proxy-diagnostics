# Recommended table placement

## Main text

1. **Table 1: Human direct-tilt coefficient-law validation.**
   Use this as the main theorem-validation table. Keep it restricted to repeated human scalar ratings and admitted analysis slices: SummEval, QGEval-human, OpenMEVA-MAGS ROC/WP, WMT MQM admitted paths, and WMT pSQM admitted paths. This avoids overclaiming and makes the sign-regime heterogeneity clear.

2. **Table 2: Prospective prediction of held-out local geometry.**
   This should be the most important empirical table. It directly supports the paper's strongest claim: small calibration matrices predict held-out gain curves better than reliability-only A_nu baselines. Include the GPT/4o margin columns to separate curvature-sign and fixed-gain sign diagnostics.

3. **Table 3a/3b: Scope checks.**
   Put finite-range stability and BoN bridge as two small adjacent tables or one two-panel table. These tables define the scope of the local analysis.

4. **Table 4: Measurement-intervention summary.**
   Use this to support the corrected intervention message: measurement improves signal A reliably, but headroom requires curvature cooperation and is not guaranteed.

## Appendix

A1. **Feedback-geometry extensions.** MT-Bench split-half BT, QGEval metric proxies, RewardBench margin geometry. This shows breadth without treating them as dataset-wide theorem validation.

A2. **Monotone error-amplification audit.** Keep as assumption audit, not headline evidence.

A3. **Exact A vs normalized A_nu scaling audit.** This justifies de-emphasizing the old nu^2 headline.

A4. **Prospective prediction by group/dimension.** Useful for inspection and failure localization.

A5. **GPT/4o external-proxy diagnostics.** Use to show the external-proxy ranking/top-tail alignment pattern, not just score compression or tilt instability.

A6. **OOD transfer prediction.** Exp35: source-only transfer vs target calibration vs shrinkage prior. Defends against "same-dataset split is too easy."

A7. **Proxy perturbation baselines.** Exp36: shuffle/random/noisy/sign-flip controls. Shows exact A,B is alignment-sensitive, while A_nu is not.

A8. Optionally include detailed WMT split admissibility and per-dataset tables if space permits.
