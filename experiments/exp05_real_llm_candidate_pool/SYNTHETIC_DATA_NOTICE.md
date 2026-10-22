# Data Provenance

- `run_exp05_real_llm.sh` generates 16 Qwen2.5-7B-Instruct candidates for each
  of 100 SummEval articles. Skywork is the proxy and OpenAssistant DeBERTa is
  the model-based silver target. The prospective split is by prompt, 30/70%.
- `run_exp05_fit_bon.sh` also supports Gaussian toy inputs for deterministic
  pipeline checks. Synthetic outputs must not be reported as real-LLM or
  human-evaluated evidence.

Inspect each run specification before combining results. Fresh sampling and
cached-pool analysis are different reproduction levels.
