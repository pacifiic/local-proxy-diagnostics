# Open Reward and Judge Scoring

Use the root GPU environment and run scripts from this directory. Set
`CUDA_VISIBLE_DEVICES` explicitly to GPUs available for the run.

| Script | Task |
|---|---|
| `00_prepare_summeval_prompts.py` | Prepare article/summary inputs |
| `01_rm_proxy_summeval.py` | Score summaries with a learned reward model |
| `02_open_judge_summeval.py` | Score summaries with an open judge |
| `03_w1_audit.py` | Compare supplied reference and optimized models |
| `04_fresh_bon.py` | Fresh-generation Best-of-N scope audit |

The reported SummEval proxies are
`Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` and
`Qwen/Qwen2.5-32B-Instruct` (five judge passes). See each script's `--help`
for input/output arguments. Use a new output directory and pin the model
revision to the commit recorded with the saved run.

Saved scoring outputs are connected to prospective prediction through
`../exp06_rm_judge_proxy_prediction/`. Actual LoRA-DPO training is in
`../exp09_dpo_validation/`. An arbitrary pair of model sizes is not a
before/after optimization comparison. Generated or model-scored targets
are not human ratings.

GPU sampling need not be byte-identical across hardware and runtime versions.
Preserve model commits, input hashes, score files, and run specifications.
