# Data Pipeline Audit (2026-07-24)

## Scope

This audit follows the full real-LLM lineage:

1. pinned UltraFeedback prompt preparation;
2. deterministic candidate generation;
3. independent proxy and silver-gold scoring;
4. strict score merging;
5. same-prompt calibration/evaluation splitting;
6. coefficient, prediction, and realized-gain computation.

The six audited full pools contain 57,600 candidate rows in total: three
Ministral seeds and three Llama-3.1 seeds, each with 300 prompts and 32
candidates per prompt. Existing experiment outputs were read but not replaced.

## Result

No candidate-mapping, score-label, split-leakage, partial-resume, or arithmetic
bug was found.

- All prompt IDs equal `SHA256(prompt)[:16]`.
- Calibration and evaluation prompt files are disjoint and byte-identical
  across all six runs.
- Every run has the complete 100 calibration + 200 evaluation prompt grid and
  candidate indices `0..31`.
- `pool.jsonl`, proxy output, gold output, and merged output have exactly the
  same candidate identities.
- Every merged proxy and gold value exactly equals its corresponding scorer
  output.
- Candidate-level generation seeds have no collisions in the six stored runs.
- No run log contains a traceback, OOM, killed process, or scoring failure.

The machine-readable lineage report is
`source_pipeline_audit.json`.

## Independent Execution Audit

`03_audit.py` independently recomputes the saved results from the raw scored
pools and candidate split manifest. The deep audit passed every check:

- calibration/evaluation candidate overlap: `0`;
- raw realized-gain cells recomputed: `179,060`;
- maximum raw realized-gain difference: `2.22e-16`;
- maximum coefficient difference: `8.88e-16`;
- maximum analytic-prediction difference: `4.44e-16`;
- maximum aggregate difference: `9.71e-17`;
- empirical-fit prompts independently refit: `120` (one per run/repeat).

The report is
`integrity_audit.json`.
The full experiment also passed a separate byte-for-byte repeat with the same
inputs and settings.

## Source and Scorer Replay

The prompt source is now explicitly pinned to UltraFeedback commit
`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`. Regenerating all 1,300 prompt
records produced the original hashes:

- train: `ed13e5847802b9ba31ddf05c89444e3f4e4f841323b4dc069d5c0d139a01176b`;
- calibration: `98e3b8b4b5da772bb654347bd785e869a32ea3dade3692c524e1900f7be68b55`;
- evaluation: `05309bb16ee364e2804e73d4b0508c8f55ffb9113a5ccfb0c8b1f1f16ec76f74`.

One full 9,600-row Llama pool was rescored under the original isolated
environment (`PYTHONNOUSERSITE=1`, Transformers 5.13.0):

- proxy replay SHA-256:
  `9e127308aa1e1bdf59244898854860b155ea8cb6c87ae50712d60db1c000e025`;
- gold replay SHA-256:
  `f7fdf6c083f55344926d1041fbba0aee1f651524351c9fa8ad58db76b4cfec57`.

Both replay files are byte-identical to the stored scorer outputs. A
post-patch candidate-generation smoke run and its repeat were also
byte-identical.

## Hardening Applied

- Prompt preparation pins and records the dataset commit and output hashes.
- Candidate generation validates prompt hashes and records candidate seed,
  generated-token count, finish reason, and prompt-file hashes.
- Known scorer checkpoints are pinned by commit.
- Scorer outputs refuse overwrite and record input/output hashes, sequence
  length, package versions, Python executable, and user-site isolation.
- The shared runner requires user-site isolation.
- Run audits verify exact score propagation and prompt hashes.
- Exp29 validates prompt text, responses, source split counts, and the complete
  candidate index grid before fitting.

## Residual Robustness Item

The DeBERTa silver-gold scorer was intentionally run with
`max_length=2048`. Its tokenizer is effectively unbounded and the model uses
relative attention, so this executes successfully; however, its configuration
lists 512 maximum position embeddings. In one Llama pool, 586 of 9,600 pairs
(6.1%) exceeded 512 pair tokens. A 512-token sensitivity rescore changed the
aggregate exact gain at `d=0.10` by `0.00339` and the mean per-prompt gain by
`0.00489` in absolute value.

This is not evidence of a mapping or resume bug, and it does not invalidate the
stored 2048-token experiment. It is a scorer-length robustness choice. Any
change to 512 should be treated as a new experiment and followed by rescoring
all pools and rerunning downstream analyses.

## Benign Degeneracy

Two short arithmetic prompts in every Ministral seed produce one identical
response across all 32 candidates. Their proxy/gold scores are consequently
constant. This behavior is stable across seeds and is not a row-mapping error.
The downstream experiment reports and skips the affected prompt/repeat cases
under its nondegeneracy checks.
