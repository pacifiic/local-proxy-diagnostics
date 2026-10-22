# Environments

## CPU Verification and Analysis

Use Python 3.11. `environments/cpu.yml` installs the resolved package set in
`environments/cpu-lock.txt`, which was used for the release checks. Direct
dependencies are listed separately in `requirements.txt`. To install that exact package set in
an existing Python 3.11 environment:

```bash
python -m pip install -r environments/cpu-lock.txt
python -m pip check
```

The numerical stack uses NumPy 1.26.4, pandas 2.3.3, SciPy 1.17.1,
scikit-learn 1.8.0, and Matplotlib 3.10.8. Set BLAS threads explicitly for
CPU-only runs when sharing a machine, for example `OMP_NUM_THREADS=4` and
`OPENBLAS_NUM_THREADS=4`. A GPU is not needed for saved-score analysis.

## Reward-Model Coefficient Example

Exp37 used NumPy 2.3.5. Its cached-score analysis has a separate CPU environment
to preserve empirical-fit boundary behavior:

```bash
conda env create -f environments/reward-model-cpu.yml
conda activate local-proxy-rm-cpu
```

Use this environment for the exp37 repeat command in its README. Moment A/B
and the main prediction estimates agree between the two CPU stacks; some
empirical-fit diagnostics are sensitive to the NumPy version.

## GPU Training and Scoring

```bash
conda env create -f environments/gpu.yml
conda activate local-proxy-gpu
python -m pip check
```

The reported DPO stack uses NumPy 2.3.5 and pins torch 2.11.0, transformers 4.57.6, TRL 1.8.0,
PEFT 0.10.0, Accelerate 1.14.0, and vLLM 0.22.1. Use an NVIDIA driver
compatible with the installed torch CUDA build. Model revisions and scoring
settings are recorded beside the saved outputs. Model access may require
accepting the model license and authenticating with a read-only Hugging Face
token. Do not put tokens into scripts or result files.

`environments/gpu-current-requirements.txt` records a separate robustness
rerun stack. It is not a replacement for the reported DPO environment.
GPU training and scoring are separate from the CPU verification commands.

## Reproduction Levels

- **Saved data:** restoration hashes, split keys, score alignment, and numerical
  audits validate the included inputs and outputs.
- **CPU reruns:** scripts use recorded seeds and splits. Fresh fit coefficients
  near a floating-point window boundary may differ; see `REPRODUCIBILITY.md`.
- **GPU reruns:** fixed seeds and model revisions constrain the run, but sampling,
  kernels, and hardware can prevent byte-identical outputs. The cached candidate
  pools and scores remain the reference for analysis reproduction.

Run metadata retain scientific settings and model commits, with portable paths
for bundled files and explicit references for external inputs.
