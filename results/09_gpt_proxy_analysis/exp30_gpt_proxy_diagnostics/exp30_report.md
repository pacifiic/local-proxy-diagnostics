# GPT proxy failure analysis report

## Overall model-proxy profile
- Median proxy std: **0.6775**
- Median Spearman(proxy, gold): **0.2847**
- Median top-10% gold delta: **0.1666**
- Median top-10% overlap with gold top set: **0.1500**

## Model vs human comparison (group-level)

- **coherence**: std ratio=0.751, Δspearman=-0.213, Δtop10 overlap=-0.112
- **consistency**: std ratio=0.998, Δspearman=-0.338, Δtop10 overlap=-0.344
- **fluency**: std ratio=0.823, Δspearman=-0.346, Δtop10 overlap=-0.306
- **relevance**: std ratio=0.995, Δspearman=-0.238, Δtop10 overlap=-0.206

## Predictive-margin analysis

- **model_proxy** overall: sign acc=0.975, median |actual Δ(0.1)|=0.0707, sign acc when |actual Δ|>0.01 = 0.975
- **human_baseline** overall: sign acc=0.907, median |actual Δ(0.1)|=0.1988, sign acc when |actual Δ|>0.01 = 0.907
