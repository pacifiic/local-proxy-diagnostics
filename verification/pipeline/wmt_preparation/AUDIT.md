# WMT Raw-Preparation Audit

Overall status: **PASS**

## Raw Parsing

| Slice | Physical rows | Parsed rows | No skipped rows |
|---|---:|---:|---|
| `wmt20_ende` | 79024 | 79024 | PASS |
| `wmt20_zhen` | 124312 | 124312 | PASS |
| `wmt22_zhen` | 46988 | 46988 | PASS |
| `wmt22_ende_3r` | 78559 | 78559 | PASS |

The parser uses literal-tab splitting with CSV quoting disabled. This is deliberate: free-text fields contain quote characters that the default CSV quote parser misinterprets.

## Segment-Key Collision Audit

| Slice | Segment IDs | Document-segment pairs | Reused IDs | Max documents per ID |
|---|---:|---:|---:|---:|
| `wmt20_ende` | 1418 | 1418 | 0 | 1 |
| `wmt20_zhen` | 2000 | 2000 | 0 | 1 |
| `wmt22_zhen` | 469 | 1875 | 469 | 4 |
| `wmt22_ende_3r` | 10 | 1315 | 10 | 181 |

## Prepared Artifact

- Independent canonical rebuild SHA-256: `efce1a53715f680e8d7c4d64bd9b4ed9fdaedf48eda6810ebc76740412459057`
- Production rebuild SHA-256: `6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8`
- Stored SHA-256: `6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8`
- Byte-identical rebuild: **True**
- Rows: 161715
- Clusters: 47
- Paths: 141
- Complete matrices: **True**

## Interpretation

- WMT20 segment IDs are globally unique within each audited slice, so adding the document component does not change their identities.
- WMT22 segment IDs are local to documents. A document-free item key therefore merges unrelated segments or creates artificial rater sets.
- The corrected artifact is independently rebuilt from the pinned raw revision and is byte-identical to the stored 141-path input.
