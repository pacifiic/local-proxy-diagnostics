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
| Reward-model coefficient example | `artifact_tools/dpo.py` |
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
