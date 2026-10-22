"""Small reproducibility helpers for the GPU external-proxy experiments."""
from __future__ import annotations

import os
from pathlib import Path


def clean_path(path: str | Path) -> str:
    """Return a stable, user-readable path string without resolving symlinks."""
    return str(Path(path).expanduser())


def resolve_hf_revision(model_id: str, revision: str | None = None) -> str | None:
    """Resolve a Hugging Face model reference to the current commit SHA.

    The experiments can run without network access if the model is cached, so
    this helper is best-effort. A missing SHA is recorded as null rather than
    causing the run to fail.
    """
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, revision=revision)
        return getattr(info, "sha", None)
    except Exception as exc:
        print(f"warning: could not resolve HF revision for {model_id}@{revision or 'default'}: {exc}")
        return None


def env_snapshot() -> dict[str, str | None]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "vllm_worker_multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
    }
