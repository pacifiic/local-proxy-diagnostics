# Release Verification

The release was checked in a clean Python 3.11 CPU environment on 2026-10-22.

| Check | Result |
|---|---|
| Saved inputs, selected fresh splits, endpoint summaries and intervals | 905 checks passed (`release_checks.json`) |
| Data keys, split disjointness, input hashes and result arithmetic | 149 checks passed (`pipeline_integrity.json`) |
| Unit and regression tests | 51 passed, including 9 subtests |
| Table and figure generation from saved results | Completed |
| Exp37 cached-score repeat in its pinned environment | Exact numeric agreement: 12 coefficient, 600 prediction, 48 summary and 12,000 bootstrap rows |

Fresh-split checks sample three paths and three repeats per setting, preserving
the full RNG sequence. These are CPU checks of prepared inputs and saved results,
not fresh GPU training. Environments and numerical details are documented in
`docs/ENVIRONMENT.md` and `REPRODUCIBILITY.md` at the repository root.

Benchmark text is preserved. Historical execution metadata use portable paths.
