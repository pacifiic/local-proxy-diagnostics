# Predicting Human-Gain Curves under Local Proxy Optimization from Repeated Ratings

Code, prepared inputs, saved results, and publication artifacts for PACE (Proxy
Alignment and Curvature Expansion). PACE predicts local human-gain curves from
repeated ratings before optimizing against a proxy.

## Quick Start

Run commands from the repository root. Saved-result verification is CPU-only.

```bash
conda env create -f environments/cpu.yml
conda activate local-proxy-cpu
python tools/restore_artifacts.py
python tools/verify_release.py
python verification/pipeline/verify_pipeline_data_integrity.py \
  --report verification/pipeline_integrity.json
pytest -q
```

For an existing Python 3.11 environment, use `pip install -r requirements.txt`.
Restoring the losslessly compressed files requires about 2 GB of free space and
never overwrites a different file. Benchmark text is preserved verbatim and may
include multilingual prompts.

## Repository Layout

| Path | Contents |
|---|---|
| `data/` | Prepared repeated-rating data, proxy scores, and the scored DPO candidate pool |
| `experiments/` | All experiment and data-preparation code, indexed by stable experiment IDs |
| `lib/` | Shared estimators, diagnostics, reliability, and bootstrap utilities |
| `results/` | Saved predictions, summaries, run specifications, and selected detailed outputs |
| `artifacts/` | Publication tables, figures, unrounded display data, notes, and manifests |
| `artifact_tools/` | Builders for the files in `artifacts/` |
| `tools/` | Restoration, release verification, and end-to-end artifact commands |
| `verification/` | Data-lineage and numerical integrity reports |
| `docs/` | Experiment, environment, figure, and third-party asset documentation |
| `provenance/` | Checksums and records for corrected inputs and publication artifacts |

## Reproduce Publication Artifacts

```bash
python tools/build_paper_artifacts.py --outdir build/paper-artifacts
python artifact_tools/build_all.py --outdir build/displays
python artifact_tools/sync_paper.py \
  --outdir build/displays \
  --paper-dir build/paper-artifacts
python tools/verify_release.py --fresh-splits
```

These commands rebuild tables and figures from saved predictions. They do not
retrain models, call external APIs, or replace saved results. `pdflatex` with
TikZ is needed only for the workflow figure.

See [Experiment Guide](docs/EXPERIMENTS.md) for claim-to-code mappings and full
run settings, [Figure and Table Builders](docs/FIGURES.md) for display sources,
and [Environments](docs/ENVIRONMENT.md) for CPU and GPU stacks.

## Data and Reporting Conventions

The active WMT analysis uses the corrected 141-path input. Tables distinguish
endpoint MAE from CurveMAE and pre-pool prediction from target-pool-aware
prediction. New runs should always use a fresh output directory.

The software is MIT licensed. External datasets and models retain their own
terms; see [Third-Party Assets](docs/THIRD_PARTY_ASSETS.md).
