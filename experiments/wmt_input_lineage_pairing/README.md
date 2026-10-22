# WMT Input-Lineage Pairing

This workflow reruns every packaged WMT-dependent analysis
against both preserved WMT inputs:

- `submitted_139`: the 139-path submitted snapshot using document-free WMT22 item identifiers.
- `fixed_141`: the corrected 141-path preparation with document-aware WMT22
  item identifiers.

The workflow never overwrites an existing output directory. Each command,
input hash, log, runtime, and completion status is stored below its lineage
directory.

```bash
python experiments/wmt_input_lineage_pairing/run_paired.py \
  --out-root outputs/wmt_input_lineage_pair \
  --jobs 40 \
  --max-parallel 6

python experiments/wmt_input_lineage_pairing/run_paper_addons.py \
  --root outputs/wmt_input_lineage_pair \
  --jobs 40 \
  --max-parallel 8

python experiments/wmt_input_lineage_pairing/validate_paired.py \
  --root outputs/wmt_input_lineage_pair
```

The paired workflow covers the WMT portions of the filter-sensitivity,
calibration-crossover, paired-statistics, prompt-conditional, fallback,
calibration-budget, BoN-scope, direct-calibration, B-stability,
mixture/rank-tilt, within-dataset robustness, OOD transfer,
evaluation-aware, strict calibration-only, and positioning analyses. The paper
addons also preserve paired results for direct coefficient validation,
reliability, monotonicity, rho scaling, measurement intervention, prospective
prediction, stability/tail summaries, proxy perturbation, calibration-size,
tail/outlier robustness, cubic moments, and the OOD variants.

DPO training is not rerun. The positioning analysis reads the same immutable
DPO summaries for both WMT lineages because the DPO candidate pool and training
data do not use WMT.
