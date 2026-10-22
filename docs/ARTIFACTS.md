# Publication Artifact Builders

This directory builds the publication tables and figures from **saved results only**. It does not retrain models, call APIs,
or modify any existing result file. Use a new output directory to keep earlier
builds.

```bash
python artifact_tools/build_all.py --outdir build/publication
python artifact_tools/sync_paper.py --outdir build/publication
```

Run `tools/restore_artifacts.py` first if using a fresh checkout. Use the root CPU pins
(NumPy 1.26.4, pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8).
`pdflatex` with TikZ is needed for the workflow figure. PyMuPDF renders its preview.
The default sync destination is `build/paper-artifacts`; `--paper-dir PATH`
selects another directory. If manuscript sources are present there, the sync
checks their table and figure inputs. Existing source text is not modified.

## Layout

| Path | Contents |
|---|---|
| `common.py` | Shared paths, formatting, figure style, input/output registry |
| `<group>.py` | One builder per display group (`bon`, `flex`, `ab`, `dpo`, `fig1`) |
| `sync_paper.py` | Copies built tables and figures; derives the row block of Table 1(b) |
| `artifacts/tables/` | LaTeX `tabular` bodies (no float, caption or label) |
| `artifacts/figures/` | Vector PDFs, plus PNG previews |
| `artifacts/data/` | Unrounded CSV behind every table/figure |
| `artifacts/notes/` | One Markdown note per display: definitions, units, aggregation, support, sources, caption caveats |
| `artifacts/manifest_<group>.json` | Input and output SHA-256 for each display |
| `artifacts/MANIFEST.md` | Merged claim-to-artifact index written by `build_all.py` |

## Conventions

- Every output filename starts with its group prefix: `bon_`, `flex_`, `ab_`, `dpo_`, `rm_`, `proxy_` or `fig1_`.
- Tables are `booktabs` tabular bodies, so captions can be written once in the
  manuscript. The matching note states what the caption must say.
- Decimal places are fixed within a table:
  - Endpoint MAE: 4.
  - CurveMAE and paired differences: 5.
  - Coefficients: 4.
  - KL radius and relative ESS: 3.
  - Percentages: 1.
- Negative numbers use `$-$`. Missing cells use `--`, and the note says why.
- WMT inputs must come from the corrected 141-path lineage. `common.require_wmt_fixed()`
  rejects 139-path or 6,950-unit rows.
- DPO main numbers use the `strict_local_d_le_0p10` rows. Root summaries in
  `three_seed_canonical/` cover the historical six-checkpoint scope.
- Each builder asserts the audited headline numbers it reproduces, so silent
  drift fails the build.
- Builders write only to their chosen output directory (default `artifacts/`).
- DPO comparator rows use the preserved cross-validated calibration output; artifact generation does not refit them.
