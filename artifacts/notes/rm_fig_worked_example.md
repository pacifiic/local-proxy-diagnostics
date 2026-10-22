# rm_fig_worked_example: three-panel worked-example figure (exp37)

**Placement.** Appendix, beside `rm_worked_example`. Following CAMERA_READY_PLAN.md §8.1, the main
text states the three decompositions in prose (Section 5.1) and points here; Figure 1 is a schematic
(`fig1_workflow`).

**What it shows.** For the preselected SummEval coherence split (repeat 0; 480 calibration / 1,120
evaluation items), the exact held-out raw-score exponential-tilt gain (black) and the calibration-only
A+B (solid blue) and A-only (dashed orange) predictions for d in [0, 0.10]. Panel titles give the
calibration A and B (3 decimals). The right endpoints equal the `rm_worked_example` values (< 1e-12).

**Units.** x: KL radius d of the tilt on the evaluation items. y: gain in human coherence score (mean
of eight raters).

**Regeneration.** Redrawn at 5.5 in width with `common.figure_style()` and `COLORS` from
`.../paper_example/worked_example_curves.csv` (201 grid points per model; A+B curve re-checked against
A sqrt(d) + B d). No numbers were recomputed. The original exp37 figure
(`results/reward_model_coefficients/canonical/paper_example/fig_reward_model_worked_example.pdf`, 10.6 in wide, different palette) remains available. A-only is dashed and drawn on top,
so it stays visible where it coincides with A+B (Skywork) and in grayscale / colour-vision deficiency.

**Reporting details.** Calibration-only A/B on the preselected coherence / repeat-0 split (chosen
before inspecting results); black = exact raw-score exponential-tilt gain on the 1,120 evaluation items
with human gold; item-held-out, not article-held-out; shared, truncated article context across models;
illustrative split with aggregate 50-repeat errors in `rm_endpoint_mae` and full-pool intervals in
`rm_coefficients`. For DeBERTa the A-only curve is closer at d = 0.10 on this split.

**Caveats.** Same as `rm_worked_example`: not DPO/RLHF, not an RM ranking, negative B is local
curvature only.
