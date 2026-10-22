# Prepared Data

The prepared SummEval, OpenMEVA and WMT score matrices and repaired GPT-4o
aggregate scores are included for reproducing the analyses. Large files are
gzip-packed. Run `python tools/restore_artifacts.py` from the repository root
before starting an experiment.

`wmt_mqm_long.csv` is the corrected 141-path input.
`wmt_mqm_long_main_protocol.csv` retains the submitted 139-path input as a
historical control. Do not replace the corrected file with that control.

The GPT-4o aggregate is already mapped to the SummEval item IDs. Ordinary
downstream reproduction requires neither a mapping repair nor a new API call.
Checksums are verified by the release and pipeline audit tools.

Complete raw benchmark repositories are not bundled. See the component README
for optional preparation commands and `../../docs/THIRD_PARTY_ASSETS.md` for
sources and the distinction between software and external-asset terms.
