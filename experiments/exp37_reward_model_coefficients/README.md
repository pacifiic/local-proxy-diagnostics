# Three Reward Models: Alignment and Curvature

Compare the same SummEval items under three fixed scalar reward proxies:

| Alias | Model | Input / output |
|---|---|---|
| `skywork` | [Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) | Chat template / scalar logit |
| `armorm` | [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) | Chat template / gated `score`, not its 19-objective vector |
| `deberta` | [OpenAssistant DeBERTa-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2) | Prompt-response pair / scalar logit |

`models.json` pins model and tokenizer commits. These are illustrative models,
not a current-leaderboard selection. Human ratings, not another reward model,
are the target. This does not certify absence of overlap with RM training data.

## Protocol

- 1,600 summaries, four human dimensions, three expert plus five crowd ratings
  per item/dimension. Gold is their eight-rating mean, matching the existing
  external-proxy engine. No fine-tuning or new human ratings.
- The same scalar RM score is compared with each of the four human dimensions.
  A/B describe a proxy-target-distribution combination, not an intrinsic RM rank.
- Context limits are 4,096 / 4,096 / 512 tokens. Preparation finds one common
  article prefix that fits all three native input formats and all summaries of
  that article. Every model and every summary of the article uses that same source
  text, with the entire summary retained. No silent summary truncation
  or model-specific article truncation is allowed. The retained character counts
  and all three token lengths are saved.
- This common-context experiment is **not** a bitwise rerun of the earlier
  full-context Skywork experiment. Do not substitute the earlier cached scores.
- Prospective evaluation: 30% calibration, 70% evaluation, 50 repeats, seed 0,
  item-level splitting, radii 0.025 / 0.05 / 0.075 / 0.10. All three models use
  the exact same saved IDs for each dimension/repeat. This is not an unseen-article
  split: summaries of one article can appear in both partitions, as in the
  original external-proxy protocol.
- The existing prospective engine supplies A+B, A-only, empirical fit, A_nu and
  zero. Evaluation gold is used only to score predictions and retrospective
  coefficient diagnostics. A split is kept only when all models are valid across
  the full grid; failures are reported, not silently inner-joined away.
- Full-pool descriptive A/B get 1,000 paired article-cluster bootstrap draws
  (seed 1), preserving correlations between summaries of the same article.
  Split-to-split MAE SD is descriptive, not an independent-sample confidence interval.

With `Z=(r-mean(r))/std(r)` and `Jc=J-mean(J)`, the existing moment code computes
`A=sqrt(2)*mean(Jc*Z)` and
`B=mean(Jc*Z^2)-(2/3)*mean(Jc*Z)*mean(Z^3)`.
The report separates the two terms of B. A/B are invariant to positive affine
changes of proxy units. A_nu is not, so raw-score A_nu differences are not a
scale-neutral reward-model ranking. Negative B alone does not establish a global
turning point or failure outside the checked local window.

## Reproduce

Use `environments/reward-model-cpu.yml` for cached-score analysis
(NumPy 2.3.5, matching the recorded run). From the
repository root, a complete CPU-only repeat is:

```bash
conda activate local-proxy-rm-cpu
RUN=results/reward_model_coefficients/canonical
python experiments/exp37_reward_model_coefficients/analyze.py \
  --prepared "$RUN/prepared" --scores-root "$RUN/scores" \
  --outdir outputs/reward_model_analysis_repeat --n-boot 1000 --bootstrap-seed 1
```

For fresh scoring, first create `environments/gpu.yml`, then run from
this experiment directory in the `local-proxy-gpu` environment (Torch 2.11.0, Transformers 4.57.6,
huggingface-hub 0.36.2, sentencepiece 0.2.1). All stages record actual versions.
The pinned ArmoRM module needs a compatibility alias for a removed documentation
constant in Transformers; `score.py` supplies that alias without changing its
forward computation. All three pinned models completed GPU inference on B200
for the validated run recorded in [RESULTS.md](RESULTS.md).

```bash
conda activate local-proxy-gpu
export ITEMS_CSV=inputs/summeval_items.csv
export ALIGNED_JSONL=inputs/model_annotations.aligned.jsonl
export OUT_ROOT=outputs/reward_model_comparison
export CUDA_VISIBLE_DEVICES=0   # only an allocated, free GPU
bash run.sh
```

No output directory may already exist. The runner evaluates one model at a time
on the selected device, with batch size one, eager attention, no sampling, and
no automatic multi-GPU dispatch. Keep all partial journals if interrupted; use
a new scoring output directory for a retry. ArmoRM's pinned custom code is loaded
with explicit consent in the runner. Review it before running untrusted models.
The model weights may require downloads and access approval. Set `HF_HUB_CACHE`
to a large-volume directory if downloads should not use the default home cache.

Stages can also be run separately, with `--local-files-only` when assets are cached:

```bash
python prepare.py --items-csv "$ITEMS_CSV" --aligned-jsonl "$ALIGNED_JSONL" --outdir "$OUT_ROOT/prepared"
python score.py --prepared "$OUT_ROOT/prepared" --alias skywork --outdir "$OUT_ROOT/scores/skywork"
python score.py --prepared "$OUT_ROOT/prepared" --alias armorm --outdir "$OUT_ROOT/scores/armorm" --allow-remote-code
python score.py --prepared "$OUT_ROOT/prepared" --alias deberta --outdir "$OUT_ROOT/scores/deberta"
python analyze.py --prepared "$OUT_ROOT/prepared" --scores-root "$OUT_ROOT/scores" --outdir "$OUT_ROOT/analysis"
```

Prepared outputs: validated texts, complete human ratings, split IDs, immutable
model registry and hashes. Preparation requires exact summary ownership and all
human ratings to match the original aligned JSONL, not merely valid item IDs.
Use a source item CSV whose summary ownership matches the aligned annotations.
A historical permuted CSV is not a valid preparation input. The source audit
does not authenticate article text, which is absent from the aligned JSONL.
Scoring outputs: per-item score, exact text hash,
token count, append-only partial journal and completion hashes. Analysis outputs:
`coefficients.csv`, `coefficient_bootstrap.csv`, `prospective_paths.csv`,
`summary.csv`, `split_accounting.json`, `comparison.md`, and `run_spec.json`.
The Markdown table reports the d=0.10 endpoint, not a mean over radii.

Independently verify source ownership, human labels, score provenance, moment
coefficients, exact endpoint gains and summary aggregation:

```bash
python audit_run.py --run "$OUT_ROOT" --aligned-jsonl "$ALIGNED_JSONL" --out "$OUT_ROOT/independent_audit.json"
python analyze.py --prepared "$OUT_ROOT/prepared" --scores-root "$OUT_ROOT/scores" --outdir "$OUT_ROOT/analysis_repeat"
diff -r "$OUT_ROOT/analysis" "$OUT_ROOT/analysis_repeat"
```

The second command repeats the CPU analysis from cached model scores, not GPU
inference. Both analysis directories are retained.

## Tests

```bash
python -B -m unittest discover -s . -p test_exp37.py
```

These synthetic tests do not download models or run GPU inference. They check ID
mapping, text/score joins, missing/duplicate ratings, context preservation,
scalar-output adapters, shared splits, unchanged outputs, the A/B formula,
affine invariance, paired bootstrap and held-out-label isolation.

## Worked Example Artifacts

After analysis, build a table and figure for the preselected coherence target,
repeat 0, and d=0.10. These use calibration-only A/B, with the evaluation gold
shown only to check the prediction. The complete 12 model/dimension combinations
remain in the descriptive coefficient table.

```bash
python report.py --run "$OUT_ROOT" --outdir "$OUT_ROOT/paper_example"
```

The default fixed-split example is illustrative, not a new performance estimate.
Use all 50 repeats for prediction-error comparisons. All outputs are written to
a new directory. The bounded, source-prefix-conditioned setting should be named
in the paper rather than presented as a universal reward-model leaderboard.

## Saved Run in the Code Release

`../../results/reward_model_coefficients/canonical/`
contains the validated prepared slice, three cached score files, both CPU
analyses, independent audit reports, and the worked-example figure and tables.
From the repository root, run `python tools/restore_artifacts.py` first to
restore the losslessly compressed prepared CSV/JSONL files. Then use the
analysis command above with the saved `prepared/` and `scores/` directories and
a new output directory. This step requires no GPU or model download.

Model weights and transient scoring logs are not included. Rebuilding or
independently source-auditing the prepared slice still requires
the upstream inputs specified above. The supplied SummEval texts and annotations
retain their upstream terms; the repository software license does not replace
them. Model versions and context limitations are unchanged from the saved run.
