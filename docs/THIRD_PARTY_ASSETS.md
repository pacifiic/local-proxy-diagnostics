# Third-Party Assets

The root MIT license covers this project's software. It does not replace the
terms of external data, model weights, or upstream material. This package
provides prepared score matrices and derived experimental artifacts; complete
raw benchmark repositories and model weights are not vendored.

| Asset | Source and scope |
|---|---|
| SummEval | [Upstream repository](https://github.com/Yale-LILY/SummEval). Repeated-rating prepared matrices are included. Obtain original annotations from the upstream source when rebuilding them. |
| WMT MQM | [Upstream repository](https://github.com/google/wmt-mqm-human-evaluation), revision `7fadea281aff8db61cc72fc3d469a4dca78b82a6`. The upstream repository supplies its Apache-2.0 license. Corrected prepared matrices and derived numerical results are included here. |
| OpenMEVA | [Upstream repository](https://github.com/thu-coai/OpenMEVA). Prepared ROC/WP repeated-rating scores are included. No new assertion about unspecified raw-asset licensing is made by this software license. |
| QGEval | Raw spreadsheets are not bundled. The data-preparation code documents the expected files; obtain them from the benchmark provider under its terms. |
| RewardBench / MT-Bench | Raw datasets and repository checkouts are not bundled. Preserved numerical results are included; see the original experiment scripts for source datasets. |
| UltraFeedback | [HuggingFaceH4 dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), labeled MIT. DPO prompt preparation pins revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`. The saved prompt subset and locally generated/scored candidate artifacts retain their upstream provenance. |
| GPT-4o | The aggregate proxy-score artifact used by the analyses is included. Model-service terms apply to API access. Cached downstream reproduction is not a guarantee of repeatable future API outputs. |
| Generator and reward models | Models are downloaded separately. Their IDs and pinned revisions are in the experiment scripts/run specifications; model-specific access requirements and licenses still apply. |

## Optional Raw WMT Rebuild

After obtaining the upstream checkout at the revision above, run:

```bash
python experiments/data_preparation/prepare_wmt_mqm_long.py \
  --root inputs/wmt-mqm-human-evaluation \
  --outpath outputs/wmt_mqm_rebuilt.csv
```

Use a fresh output file. The expected corrected prepared-input checksum is
recorded in `REPRODUCIBILITY.md`. The source TSV parsing and document-aware item
keys are independently audited under `experiments/wmt_input_lineage_pairing/`.

Prepared matrices are sufficient for the advertised saved-result checks.
Additional raw sources are required for a source-to-prepared-data regeneration
or some auxiliary experiments.
