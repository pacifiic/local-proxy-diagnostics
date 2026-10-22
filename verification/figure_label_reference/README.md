# Validated Pooled-BoN Figure Labels

`fig_bon_bridge_bootstrap.pdf` preserves the plotted values while correcting the
labels to match the underlying pooled-item analysis. The horizontal coordinate
is `u = log N`, Panel F reports a gold-score level, and `nu` denotes an SD ratio.

Regenerate the labeled overlay in this directory with:

```bash
pdflatex -halt-on-error -jobname=fig_bon_bridge_bootstrap \
  '\def\cleanreview{1}\input{fig_bon_bridge_bootstrap_labels.tex}'
```

The prompt-conditional BoN scope figure is generated separately by
`artifact_tools/bon.py`.
