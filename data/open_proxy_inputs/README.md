# Open-Proxy Experiment Inputs

These four compact CSVs are the final scored inputs consumed by
`experiments/exp06_rm_judge_proxy_prediction/run_exp06.sh`. They preserve the
outputs needed from the upstream open-proxy scoring workflow without bundling
raw model generations or judge logs.

| File | Role | SHA256 |
|---|---|---|
| `proxy_rm_skywork_v02.csv` | Skywork reward-model proxy on SummEval | `1e7b09507c62daedc7e52ccf9cdd2643855be5f1df15497f773bfe99b8f71598` |
| `model_proxy_openjudge_aggregated.csv` | aggregated Qwen2.5-32B open-judge proxy on SummEval | `9083b173dedab9661de439f61fb1722b36b143fc253a5304167af1d39c2ae455` |
| `fresh_bon_proxy_scores.csv` | proxy scores for the fresh-BoN candidate pool | `44a19accb23c0a30cbf8fb112b6a047d70406a0e01e48d746c289eddb455e13f` |
| `fresh_bon_gold_scores.csv` | independent judge scores for the same candidate pool | `a4f6ad1baa2e49265a17a4677a75d34bffc619218eb265822458f181fd11d4f3` |

The prospective experiment validates item coverage and records input hashes in
its run specification. The fresh-BoN correlation script requires exact,
duplicate-free candidate-key agreement before merging proxy and gold scores.
