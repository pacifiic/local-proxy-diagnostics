# Artifact Builders

The scripts in this directory generate the publication tables, figures,
unrounded display data, notes, and SHA-256 manifests in `artifacts/`.
They read saved files from `results/` and do not retrain models or call APIs.

```bash
python tools/restore_artifacts.py
python tools/build_paper_artifacts.py --outdir build/paper-artifacts
python artifact_tools/build_all.py --outdir build/displays
python artifact_tools/sync_paper.py \
  --outdir build/displays \
  --paper-dir build/paper-artifacts
```

`make_artifacts.py` and the `make_*` modules build the primary experiment
displays. `build_all.py` runs the grouped comparison builders (`bon`, `flex`,
`ab`, `dpo`, and `fig1`). Every grouped builder records its exact inputs and
outputs in a manifest. See `docs/FIGURES.md` and `docs/ARTIFACTS.md` for the
claim-to-builder map and formatting conventions.
