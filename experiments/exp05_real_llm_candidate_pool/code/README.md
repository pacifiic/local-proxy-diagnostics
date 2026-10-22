# Experiment 19: Real LLM Candidate-Pool Reranking

This experiment tests the local proxy-gain diagnostic on
actual LLM-generated candidate pools. It is intentionally separate from the
main CPU-only artifact because candidate generation and reward-model scoring
can require GPUs and gated Hugging Face models.

## Question

Given prompts, a generator, a fixed proxy scorer, and an independent gold
scorer, can calibration prompts predict the held-out human-preference gain
curve induced by locally optimizing the proxy over generated candidates?

The experiment targets prompt-conditional soft reranking:

```text
pi_t(candidate | prompt) proportional to exp(t * proxy_score)
```

The reported KL radius is the average prompt-level KL relative to the uniform
candidate pool. Best-of-N is included only as a discrete scope check.

## Pipeline

1. Prepare prompts and prompt-level splits.

```bash
python experiments/exp05_real_llm_candidate_pool/code/01_prepare_prompts.py \
  --input-jsonl path/to/prompts.jsonl \
  --prompt-col prompt \
  --limit 1000 \
  --calibration-frac 0.30 \
  --out-csv outputs/19_llm/prompts.csv
```

2. Generate candidate responses with vLLM.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python experiments/exp05_real_llm_candidate_pool/code/02_generate_candidates_vllm.py \
  --prompts-csv outputs/19_llm/prompts.csv \
  --model <generator-model> \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --n-candidates 16 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-tokens 512 \
  --out-csv outputs/19_llm/candidates.csv
```

3. Score the same candidates with a fixed proxy and independent gold scorer.

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/exp05_real_llm_candidate_pool/code/03_score_sequence_classifier.py \
  --candidates-csv outputs/19_llm/candidates.csv \
  --model <proxy-reward-model> \
  --score-col proxy_score \
  --input-format chat \
  --out-csv outputs/19_llm/candidates_proxy.csv

CUDA_VISIBLE_DEVICES=1 \
python experiments/exp05_real_llm_candidate_pool/code/03_score_sequence_classifier.py \
  --candidates-csv outputs/19_llm/candidates_proxy.csv \
  --model <independent-gold-reward-model> \
  --score-col gold_score \
  --input-format chat \
  --out-csv outputs/19_llm/candidates_scored.csv
```

If a selected Hugging Face model requires authentication, either log in with
the Hugging Face CLI or pass `--hf-token-file path/to/token` to the generation
or scoring scripts. The token value and token-file path are not written to run
specs; the specs record only whether a token file was used. On a shared GPU,
lower `--gpu-memory-utilization` to avoid vLLM reserving most of the device.

4. Fit calibration predictors and evaluate held-out gain curves.

```bash
python experiments/exp05_real_llm_candidate_pool/code/04_fit_predict_gain_curves.py \
  --candidates-csv outputs/19_llm/candidates_scored.csv \
  --calibration-frac 0.30 \
  --n-repeats 20 \
  --outdir outputs/19_llm/gain_curves
```

5. Optional Best-of-N scope check.

```bash
python experiments/exp05_real_llm_candidate_pool/code/05_bon_scope_check.py \
  --candidates-csv outputs/19_llm/candidates_scored.csv \
  --calibration-frac 0.30 \
  --n-repeats 20 \
  --n-list 2,4,8,16 \
  --outdir outputs/19_llm/bon_scope
```

6. Build compact analysis artifacts.

```bash
python experiments/exp05_real_llm_candidate_pool/code/06_make_artifacts.py \
  --gain-root outputs/19_llm/gain_curves \
  --bon-root outputs/19_llm/bon_scope \
  --outdir outputs/19_llm/artifacts
```

## Human Actions Usually Needed

- Install GPU dependencies separately, for example `vllm`, `torch`, and
  `transformers`.
- Log in to Hugging Face if the generator, proxy, or gold model is gated.
- Accept model licenses or access terms on the provider website if required.
- Choose models that are legally usable for research experiments.
- Provide a prompt dataset. The code accepts CSV or JSONL with a prompt column.

## Outputs

The main analysis writes:

- `llm_gain_curve_predictions.csv`
- `llm_gain_curve_endpoint_summary.csv`
- `llm_gain_curve_curve_summary.csv`
- `llm_gain_curve_coefficients.csv`
- `llm_gain_curve_skipped.csv`
- `llm_gain_curve_run_spec.json`

The BoN scope check writes:

- `llm_bon_scope_rows.csv`
- `llm_bon_scope_summary.csv`
- `llm_bon_scope_run_spec.json`
