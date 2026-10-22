# Prompt-Conditional Best-of-N Scope Audit

This experiment measures the prompt-level KL radius induced by exact
with-replacement Best-of-N (BoN). It is a scope audit, not a validation of the
paper's pooled raw-score exponential-tilt law.

The audit groups candidates by prompt, computes exact BoN selection
probabilities within each prompt, and reports the fraction of prompt-level KL
values within the local `d <= 0.10` window. The pooled raw-score tilt comparison
is attempted only within the audited `d <= 0.25` bridge range.
Reachability is checked at the requested mean prompt KL, including paths
whose tied top scores cap the pooled tilt below 0.25. The pooled comparison
solves directly for that radius. Verified camera-ready outputs are in
`results/prompt_bon/`.

Run from the repository root into fresh output directories:

```bash
python experiments/15_prompt_conditional_bon_scope/01_prompt_conditional_bon_audit.py \
  --dataset-name SummEval \
  --human-long-csv data/summeval_long.csv \
  --item-cols article_id,model_id \
  --group-cols dimension,rater_type \
  --min-items 200 \
  --min-raters 3 \
  --n-list 2,4,8,16,32,64 \
  --tilt-max-d 0.25 \
  --outdir outputs/prompt_conditional_bon_scope/summeval

python experiments/15_prompt_conditional_bon_scope/01_prompt_conditional_bon_audit.py \
  --dataset-name OpenMEVA-MAGS \
  --human-long-csv data/openmeva_long.csv \
  --item-cols story_id,model_id \
  --group-cols dataset \
  --min-items 200 \
  --min-raters 3 \
  --n-list 2,4,8,16,32,64 \
  --tilt-max-d 0.25 \
  --outdir outputs/prompt_conditional_bon_scope/openmeva_mags
```

At `N=2`, the canonical summaries report mean within-window prompt shares of
`0.1246875` for SummEval and `0.1165` for OpenMEVA-MAGS. These are the sources
of the rounded `12.5%` and `11.7%` values reported in the accompanying analysis.
