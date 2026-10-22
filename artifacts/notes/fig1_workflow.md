# fig1_workflow: Figure 1

**Placement.** Main text, Figure 1. It summarizes the workflow, information visibility, and local gain-law interpretation. No empirical example is shown, so
the §5 selection rule for examples does not apply. Following §8.1, the main text instead states the
three-reward-model decomposition in prose (Section 5.1), and the panels are in the appendix
(`rm_worked_example`, `rm_fig_worked_example`).

**Panel (a): what each step uses.**
- Calibration items carry a proxy score r and k >= 2 human ratings Y_1..Y_k (Section 3 notation
  Y_ij = J_i + eta_ij). Only these items are human-rated.
- Estimate: plug-in moments give A_hat (alignment) and B_hat (curvature).
- Forecast: A_hat sqrt(d) + B_hat d on the checked window d <= d_max (0.10 in the experiments, not a
  universal constant).
- Translate rule (dashed): a planned rule that is not a raw-score tilt, such as Best-of-N, is translated
  into its induced KL radius d and RelESS (Eq. kl-ess) on an unlabeled, proxy-scored pilot (Section 6,
  step iii). A raw-score tilt needs no translation, because d is chosen directly.
- Decide: proceed, abstain, or rate more. Section 6 also lists reducing pressure and direct validation.
- The target-pool-aware regime (proxy scores of the target pool feed flexible calibrators) is a
  comparison setting of Table 2 and is described in Section 4, not drawn here. No forecast uses human
  labels of the target pool.

**Panel (b): schematic.** Illustrative curves, not data: A = 1, B = -0.8 (forecast), and a realized
curve with an added -0.8 d^{3/2} remainder that saturates and declines beyond the window. The caption
must say "schematic" and that no forecast is made beyond the checked window.

**Source.** TikZ/pgfplots source `figures/fig1_workflow.tex`, compiled with pdflatex
(SOURCE_DATE_EPOCH=0, FORCE_SOURCE_DATE=1); the PNG is a 200-dpi pdftoppm preview.
