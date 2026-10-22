#!/usr/bin/env python3
"""Small unit checks for the exp36 curve regressors."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp36_build_test", HERE / "build.py")
build = importlib.util.module_from_spec(SPEC)
sys.modules["exp36_build_test"] = build
SPEC.loader.exec_module(build)


def test_origin_polynomial_recovers_three_terms() -> None:
    d = np.linspace(0.0, 0.10, 41)
    expected = np.array([0.7, -0.4, 0.2])
    x = np.sqrt(d)
    gain = expected[0] * x + expected[1] * x**2 + expected[2] * x**3
    observed = build.fit_origin_polynomial(d, gain, order=3)
    assert np.allclose(observed, expected, atol=1e-10)


def test_centered_spline_is_zero_at_origin() -> None:
    d = np.linspace(0.0, 0.10, 41)
    values = np.sin(3.0 * np.sqrt(d))
    predict, metadata = build.fit_centered_spline(d, values)
    assert abs(predict(0.0)) < 1e-12
    assert metadata["n_basis"] > 3


if __name__ == "__main__":
    test_origin_polynomial_recovers_three_terms()
    test_centered_spline_is_zero_at_origin()
    print("exp36 unit checks passed")
