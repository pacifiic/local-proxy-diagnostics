#!/usr/bin/env python3
"""Small invariance tests for exp28's prompt-coordinate transforms."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name(
    "build_prompt_conditional_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("exp28_builder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)  # type: ignore[union-attr]


def test_prompt_transforms() -> None:
    proxy = np.array([10.0, 12.0, 14.0, 20.0])
    gold = np.array([-2.0, 1.0, 4.0, 5.0])
    shifted_gold = gold + 1000.0

    raw, _ = MODULE.transform(proxy, gold, "raw")
    centered, centered_gold = MODULE.transform(proxy, gold, "centered")
    zscore, _ = MODULE.transform(proxy, gold, "zscore")
    rank, _ = MODULE.transform(proxy, gold, "rank")

    np.testing.assert_allclose(raw, proxy)
    np.testing.assert_allclose(centered.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(centered_gold.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(zscore.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(zscore.std(), 1.0, atol=1e-12)
    np.testing.assert_allclose(rank, [0.125, 0.375, 0.625, 0.875])

    # Evaluation proxy coordinates cannot depend on held-out gold labels.
    for mode in MODULE.MODES:
        original_x, _ = MODULE.transform(proxy, gold, mode)
        shifted_x, _ = MODULE.transform(proxy, shifted_gold, mode)
        np.testing.assert_allclose(original_x, shifted_x)


if __name__ == "__main__":
    test_prompt_transforms()
    print("TRANSFORM TESTS PASSED")
