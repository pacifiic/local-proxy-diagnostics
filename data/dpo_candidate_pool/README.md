# Canonical DPO Inputs

These are the immutable prepared artifacts shared by the DPO analyses under
`results/actual_dpo/`. They are retained because stochastic LLM candidate
generation is not guaranteed to be bit-identical across vLLM, PyTorch, CUDA,
or GPU-kernel versions even when prompts, model revision, and sampling seed
are fixed.

`run_exp09_reported_reproduction.sh` verifies every hash before training.
The raw and scored pools are both kept so `10_data_audit.py` can independently
check split membership, candidate keys, score joins, preference-pair direction,
and reference-log-probability alignment.

The primary local-window summary selects checkpoints by mean prompt-level KL at most 0.10. In the
canonical three-seed run this selects the same ten checkpoints as the earlier
median-KL rule. The selected support and prompt-radius distribution are recorded
in `results/actual_dpo/three_seed_canonical/prompt_radius_audit/`.

The canonical hashes are stored once in `SHA256SUMS`. Verify them with:

```bash
(cd data/dpo_candidate_pool && sha256sum -c SHA256SUMS)
```
