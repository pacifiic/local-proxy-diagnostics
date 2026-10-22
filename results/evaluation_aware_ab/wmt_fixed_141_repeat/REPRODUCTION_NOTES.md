# Independent fixed_141 WMT Reproduction

This directory is an independent WMT-only reproduction of Exp32.

## Protocol

- input: `data/wmt_mqm_long.csv`
- input SHA-256:
  `6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8`
- paths: 141
- calibration/evaluation split: 30/70
- repeats: 50
- random seed: 0
- radii: 0.01, 0.025, 0.05, 0.075, 0.10
- methods: 16
- prediction rows: 564,000
- skipped records: 0

## Reproduction result

The raw prediction CSV had SHA-256
`3ec6becf431a7eed607410c7cf44497f007fdf4edf40929d91480efe623b7bb1`,
which is byte-identical to the original source run. The summary, paired
bootstrap table, run spec, and results Markdown are also byte-identical to the
canonical files.

`reproduction_verification.json` contains the machine-readable comparison.

## Packaging drift found by the preflight

The first preflight run exposed that Exp32 requested `A+B (MLP h)`, while the
packaged Exp21 dependency lacked the MLP fitter used by the canonical run.
That incomplete preflight produced no MLP predictions and failed the Exp32
audit. The pre-specified 1-16-16-1 ReLU implementation recorded by the
canonical run was restored to Exp21. The complete rerun then reproduced the
canonical artifacts exactly.

This was source-packaging drift, not a change to the WMT input, split plan, or
reported result.
