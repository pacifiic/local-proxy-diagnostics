# exp33 Source-to-Result Audit

## Result

The exp33 implementation itself passes the prepared-data-to-result audit:

- all 185 paths were independently reconstructed from the four prepared CSVs;
- all 9,250 path/repeat splits have the expected calibration and evaluation
  sizes;
- `A+B`, `A-only`, and `A_nu` predictions were independently re-derived for
  every split and radius;
- 2,775 held-out exact-gain cells were recomputed with a separate root solver;
- calibration-only predictions are invariant to evaluation-proxy permutation;
- two full runs produced byte-identical outputs.

No evaluation proxy or evaluation gold value enters the exp33 prediction
function.

## Source artifacts

| Input | Source reconstruction | Finding |
|---|---|---|
| SummEval | Rebuilt 51,200 ratings from `model_annotations.aligned.jsonl` | All keys and scores match |
| GPT-4o aggregate | Re-ran text-based item remapping from 19,200 raw scores | The 6,400-row aggregate is byte-identical |
| OpenMEVA | Downloaded the pinned ROC/WP files and parsed 10,000 ratings | Output is byte-identical |
| WMT MQM | Rebuilt the corrected 141-path input from pinned raw TSV files | All 161,715 keys and scores match |

The WMT raw file has document-local segment IDs. The corrected preparation
uses document-qualified keys, literal-tab parsing, and fail-loud row handling.
The rebuilt 161,715-row file matches `data/wmt_mqm_long.csv` exactly. Passing
the submitted 139-path artifact instead triggers the separately documented
document-key-collision warning.

## Interpretation

The fair information-setting comparison is source-reconstructed for all four
datasets. Corrected WMT `fixed_141` is the default result. The submitted
`submitted_139` artifact is retained only as labeled provenance.

Machine-readable reports:

- `data_chain_audit.json`
- `verification/pipeline/source_data_fixed_141/source_data_audit.json`
