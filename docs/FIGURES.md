# Figure and Table Builders

Run from the release root using the pinned CPU environment:

```bash
python tools/restore_artifacts.py
python tools/build_paper_artifacts.py --outdir build/paper-artifacts
python artifact_tools/build_all.py --outdir build/publication
python artifact_tools/sync_paper.py --outdir build/publication
```

The first command rebuilds the main experiment displays. The second builds the comparison and worked-example displays. The sync places these in
`build/paper-artifacts`. Saved publication figures are also collected in
`artifacts/figures`. Inputs and individual experimental
predictions are not refitted or replaced. CI aggregation uses the corrected
pooled-split estimator described in `REPRODUCIBILITY.md`.

| Display | Builder |
|---|---|
| Workflow | `artifact_tools/fig1.py` (requires pdflatex/TikZ) |
| B contribution, rater noise, skewness | `artifact_tools/ab.py` |
| Prompt-conditional BoN | `artifact_tools/bon.py` |
| Reward-model tables and 50-split coherence figure | `artifact_tools/dpo.py`, `artifact_tools/reward_model_figure.py` |
| Reliability, exact-tilt example, main prediction/OOD, perturbations, stability | `artifact_tools/make_paper_figures.py` |
| KL radius and reachability | `artifact_tools/make_kl_radius_audit_figure.py` |
| Calibration-size and same-pool residual composites | `artifact_tools/make_appendix_composite_figures.py` |
| Tail/outlier comparison | `artifact_tools/make_tail_bon_artifacts.py` |
| Validated pooled BoN reference | `verification/figure_label_reference/` |
| Eight-group fit-range table | `artifact_tools/make_fit_range_table.py` |

The display output directory includes unrounded data, table bodies,
definitions, aggregation details, and SHA-256 manifests. Figure PDFs contain
embedded Type 1 or TrueType fonts, not Type 3 fonts.

The pooled BoN panel is preserved with corrected labels. Current scope results use
the separate exact-BoN and prompt-conditional audits.

## Reward-Model Figure

To build Figure B.8 alone into a new directory:

```bash
python artifact_tools/reward_model_figure.py --out build/reward-model-figure
```

The upper row averages gain forecasts across 50 matched SummEval coherence
splits. The lower row averages within-split absolute errors, not the error of
the mean forecast. Zoomed axes expose the differences; shading marks the
between-method gap, not a confidence interval. The single-split coefficient
example remains a separate table. No model inference is rerun.

Inter font files and their SIL Open Font License are bundled in
`artifact_tools/fonts/inter`. The standalone build exports PDF, PNG, SVG,
unrounded data, and input hashes. The full publication builder registers this
figure and its mean/MAE CSV under the existing `rm_fig_worked_example` filename.
