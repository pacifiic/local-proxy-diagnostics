# Data Preparation and Full Runs

This guide covers source-data preparation and full experiment runners. Start with the [repository README](../README.md) for saved-result verification and artifact generation.
The active saved output is `results/`, and `data/wmt_mqm_long.csv` is the corrected 141-path
input. Prepared CSVs and saved numerical results are included for verification.
Run `python tools/restore_artifacts.py` from the repository root first.
Raw benchmark checkouts are obtained separately from their providers.

The instructions below describe the experiment organization. For external
asset sources and terms, see [Third-Party Assets](../docs/THIRD_PARTY_ASSETS.md).

## Contents

```text
.
├── artifact_tools/       # builders for tables and figures
├── data/                 # prepared matrices and cached proxy scores
├── experiments/          # data preparation and experiment scripts
├── artifacts/figures/    # generated publication figures
├── lib/                  # shared analysis utilities
├── artifacts/tables/     # generated publication tables
├── make_artifacts.py     # thin wrapper around artifact_tools
├── scripts/run_all_reproduced.sh
├── scripts/run_accept_boost_experiments.sh
└── scripts/run_robustness_experiments.sh
```

## Quick Start

Create an environment with the core dependencies. The included artifacts were
tested with Python 3.11.11 and the direct package versions pinned in
`requirements.txt`.

```bash
conda create -n proxy-gain python=3.11 -y
conda activate proxy-gain

pip install -r requirements.txt
```

The included figure and table artifacts are already present under `artifacts/figures/`
and `artifacts/tables/`. To regenerate them from experiment outputs, first run the
experiments into a chosen output directory, then point the artifact builder at
that directory:

```bash
RUN_ROOT=outputs/proxy_gain_repro_$(date +%Y%m%d_%H%M%S)
OUT_ROOT="$RUN_ROOT/all_results_reproduced"
ACCEPT_BOOST_OUT="$OUT_ROOT/12_accept_boost"
ROBUSTNESS_OUT="$OUT_ROOT/13_robustness"
export OUT_ROOT ACCEPT_BOOST_OUT ROBUSTNESS_OUT

bash scripts/run_all_reproduced.sh
bash scripts/run_accept_boost_experiments.sh
bash scripts/run_robustness_experiments.sh

python make_artifacts.py all \
  --results-root "$OUT_ROOT" \
  --accept-root "$OUT_ROOT/12_accept_boost" \
  --robustness-root "$OUT_ROOT/13_robustness"
```

The runners refuse to overwrite an existing output directory unless
`ALLOW_OVERWRITE=1` is set. Reproduction reruns should use scratch output paths.

## Data Preparation

The experiments use admitted repeated-rating slices, not full-benchmark coverage
for every source dataset. The main repeated-rating analyses use:

- SummEval human annotations
- WMT MQM repeated-rating clusters
- OpenMEVA-MAGS ROC/WP manual annotations

The auxiliary checks use QGEval paths, RewardBench margin geometry, MT-Bench
split-half Bradley-Terry utilities, and a cached GPT-4o aggregate proxy file.

The main prepared rating matrices and `data/model_proxy_4o_aggregated.csv`
are included. The commands below are optional source-data regeneration steps
or preparation for auxiliary datasets. Use a separate data directory when
checking preparation, rather than replacing the included prepared matrices.

### SummEval

Download the public human-annotation JSONL from the SummEval repository and run
the parser:

```bash
mkdir -p data
curl -L \
  -o data/model_annotations.aligned.jsonl \
  https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl

python experiments/data_preparation/parse_summeval.py
```

### OpenMEVA-MAGS ROC/WP

```bash
python experiments/data_preparation/parse_openmeva.py \
  --revision 5ee4fe59caafce625670f91d3983f5c1ddab8c1d
```

This script prepares only the manually annotated MAGS ROC/WP slices used by the
diagnostic.

### RewardBench Margins

```bash
python experiments/data_preparation/download_rewardbench.py \
  --revision 96302cc604e4f257dc526272c803c4eff27e3469
```

RewardBench is used as an auxiliary cross-model margin-geometry check, not as a
repeated human-rating dataset.

### WMT MQM

```bash
git clone https://github.com/google/wmt-mqm-human-evaluation data/wmt-mqm-human-evaluation
git -C data/wmt-mqm-human-evaluation checkout 7fadea281aff8db61cc72fc3d469a4dca78b82a6
python experiments/data_preparation/prepare_wmt_mqm_long.py \
  --outpath data/wmt_mqm_long_rebuilt.csv
```

The experiments use admitted repeated-rating clusters from the WMT MQM
resources. The checkout above matches the source snapshot used for the included
artifacts.

### MT-Bench

```bash
python experiments/data_preparation/download_mtbench.py \
  --revision f7d2896d2cc5d80f8b55c2bbc722613555233c25
```

MT-Bench is used only for split-half Bradley-Terry utility checks.

### QGEval

QGEval files are not downloaded automatically. Place the raw files at:

```text
data/QGEval/annotation_result/annotator1.xlsx
data/QGEval/annotation_result/annotator2.xlsx
data/QGEval/annotation_result/annotator3.xlsx
data/QGEval/scores.xlsx
data/QGEval/metric_result.xlsx
```

Then run the relevant scripts through `scripts/run_all_reproduced.sh`.

## Experiment Workflow

The top-level runners are the recommended entry points:

- `scripts/run_all_reproduced.sh`: core experiments and cross-source checks
- `scripts/run_accept_boost_experiments.sh`: local-radius, residual, calibration-size,
  and selective-use stress tests
- `scripts/run_robustness_experiments.sh`: BoN scope checks, OOD lambda
  sweeps, and additional calibration-size robustness

See `experiments/README.md` for a mapping from claims to scripts and output
files.

## Artifact Generation

The tracked artifact directories are:

- `artifacts/figures/`: generated publication figure PDFs
- `artifacts/tables/`: generated table exports in LaTeX, CSV, and Markdown formats

From the repository root:

```bash
python make_artifacts.py tables
python make_artifacts.py figures
```

Use `python make_artifacts.py --help` for output-directory overrides. To build
from the current saved results, run `tools/build_paper_artifacts.py` from the repository
root; it selects the corrected result roots explicitly.

## GPT-4o Artifact

The GPT-4o external-proxy diagnostic uses the included cached aggregate file:

```text
data/model_proxy_4o_aggregated.csv
```

The current aggregate was repaired by matching each scored candidate summary
back to the SummEval model id in `model_annotations.aligned.jsonl`; duplicated
candidate summaries share the same text-level GPT-4o average. Its SHA-256
checksum is recorded in `data/README.md`.

Regenerating model-proxy scores is optional and requires API access configured
locally by the user. New model calls may differ from the cached artifact if
the provider changes model behavior or defaults. Install the optional API
dependencies with:

```bash
pip install -r experiments/09_gpt_proxy_analysis/requirements.txt
```

The optional API scoring entry point is
`experiments/09_gpt_proxy_analysis/01_generate_proxy_scores.py`.

If raw GPT score logs and the prompt-item CSV are available, the SummEval
item-id repair step can be reproduced without new API calls:

```bash
python experiments/09_gpt_proxy_analysis/02_remap_summeval_proxy_scores.py \
  --items-csv path/to/summeval_items.csv \
  --raw-scores-csv path/to/model_proxy_4o_raw_scores.csv \
  --summeval-jsonl data/model_annotations.aligned.jsonl \
  --out-aggregate-csv data/model_proxy_4o_aggregated.csv \
  --out-mapping-csv path/to/model_proxy_4o_item_remap_audit.csv \
  --out-report-json path/to/remap_report.json
```

## Reproducibility Notes

- Stochastic experiment scripts use explicit seeds and write local output
  directories that can be regenerated from the commands above.
- OpenMEVA, RewardBench, MT-Bench, and WMT MQM are pinned by explicit source
  revisions or Git commits. Figure PDFs are generated with normalized timestamp
  metadata.
