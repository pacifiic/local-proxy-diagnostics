#!/usr/bin/env python3
"""Small numerical checks for the C moment formula."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_cubic_moment_term.py"
_spec = importlib.util.spec_from_file_location("cubic_builder", BUILD)
builder = importlib.util.module_from_spec(_spec)
sys.modules["cubic_builder"] = builder
_spec.loader.exec_module(builder)  # type: ignore[union-attr]


def run_checks() -> None:
    rng = np.random.default_rng(123)
    r = rng.normal(size=5000) + 0.2 * rng.exponential(size=5000)
    j = 0.4 * r - 0.15 * r**2 + 0.03 * r**3 + rng.normal(scale=0.2, size=5000)

    c = builder.cubic_moment_coefficient(j, r)
    coeff = builder.exp21.calib.theory_coefficients(j, r)
    if not np.isfinite(c):
        raise AssertionError("C is not finite")

    # A+B+C must approach the exact finite-tilt gain faster than A+B as d -> 0
    # on this deterministic smooth synthetic distribution.
    ds = np.array([2e-6, 5e-6, 1e-5, 2e-5])
    err_ab = []
    err_abc = []
    for d in ds:
        exact = builder.exp21.calib.exact_delta_at_d(r, j, float(d))
        ab = coeff.a_exact * np.sqrt(d) + coeff.b_theory * d
        abc = ab + c * d**1.5
        err_ab.append(abs(ab - exact))
        err_abc.append(abs(abc - exact))
    if not np.mean(err_abc) < np.mean(err_ab):
        raise AssertionError(
            f"cubic term did not improve local synthetic approximation: "
            f"{err_abc} vs {err_ab}"
        )

    # Affine proxy transformations leave the positive-tilt path unchanged.
    c_affine = builder.cubic_moment_coefficient(j, 3.7 * r + 11.0)
    if not np.isclose(c, c_affine, rtol=1e-10, atol=1e-10):
        raise AssertionError(f"C is not proxy-affine invariant: {c} vs {c_affine}")
    print(f"PASS C={c:.8f}, mean local error {np.mean(err_ab):.3e} -> {np.mean(err_abc):.3e}")


def test_cubic_moment_formula() -> None:
    run_checks()


def main() -> None:
    run_checks()


if __name__ == "__main__":
    main()
