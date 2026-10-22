# Interpretation

The proxy is Skywork and the silver target is OpenAssistant DeBERTa. These
are different model families, not human ratings; independence of their training
data is not established. Treat results as model-based candidate-pool audits.

The 20-resplit outputs quantify split sensitivity. Single-split outcomes should
not replace them, and repeats reuse the same pool. Best-of-N outside the checked
KL window is a scope diagnostic, not validation of the local theorem.

Synthetic pipeline checks are not real-generation evidence. Keep them distinct
from model-generated candidate pools as described in `SYNTHETIC_DATA_NOTICE.md`.
