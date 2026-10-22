# Prompt-Radius Audit

This directory audits the prompt-level KL radii inside the ten canonical
three-seed DPO checkpoints selected by checkpoint **mean** prompt KL at most
0.10.

The gate does not imply that every prompt is local. Across 2,000
prompt-checkpoint cells, the prompt radius has median/p90/p95/p99/max
0.0257/0.0919/0.1224/0.1867/0.4668; 8.7% exceed 0.10, and the mean of
\(d_x^{3/2}\) is 0.0107. Applying common calibration coefficients to these
promptwise radii is therefore an empirical pooled-coefficient approximation,
not a direct theorem-level guarantee for conditional DPO.

Reproduce from the repository root:

```bash
python experiments/exp09_dpo_validation/21_prompt_radius_audit.py
```

Files:

- `prompt_radius_cells.csv`: one row per retained prompt-checkpoint cell.
- `prompt_radius_summary.csv`: pooled and per-seed distribution summaries.
- `run_spec.json`: source paths, gate definition, and selected checkpoints.
