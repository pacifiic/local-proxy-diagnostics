"""Compute item-jackknife and top-proxy-removal sensitivity in linear time.

Additive power sums allow leave-one-out moment updates without quadratic work.
Self-check each closed-form update against direct coefficient recomputation
on small random arrays before processing experimental paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT_DIR = HERE / "outputs"

sys.path.insert(0, str(REPO_ROOT / "lib"))
from common_experiment_utils import theory_coefficients, split_indices  # noqa: E402

sys.path.insert(0, str(HERE))
import build_b_stability as bs  # noqa: E402


def power_sums(r: np.ndarray, j: np.ndarray) -> dict:
    return {
        "n": len(r),
        "S_J": float(np.sum(j)), "S_R": float(np.sum(r)),
        "S_JJ": float(np.sum(j * j)), "S_RR": float(np.sum(r * r)),
        "S_JR": float(np.sum(j * r)),
        "S_RRR": float(np.sum(r ** 3)), "S_JRR": float(np.sum(j * r * r)),
    }


def ab_from_sums(s: dict) -> tuple[float, float]:
    n = s["n"]
    if n < 2:
        return float("nan"), float("nan")
    j_bar = s["S_J"] / n
    r_bar = s["S_R"] / n
    sigma_j2 = s["S_JJ"] / n - j_bar ** 2
    sigma_r2 = s["S_RR"] / n - r_bar ** 2
    if sigma_r2 <= 0:
        return float("nan"), float("nan")
    sigma_r = np.sqrt(sigma_r2)
    cov_jr = s["S_JR"] / n - j_bar * r_bar
    rho = cov_jr - sigma_j2  # Cov(J,r) == sigma_J^2 + rho identity (see build_b_stability docstring)
    kappa_r = s["S_RRR"] / n - 3 * r_bar * s["S_RR"] / n + 2 * r_bar ** 3
    kappa_jrr = s["S_JRR"] / n - 2 * r_bar * s["S_JR"] / n - j_bar * s["S_RR"] / n + 2 * j_bar * r_bar ** 2
    a_exact = np.sqrt(2.0) * (sigma_j2 + rho) / sigma_r
    b_theory = kappa_jrr / sigma_r2 - 2.0 * (sigma_j2 + rho) * kappa_r / (3.0 * sigma_r2 ** 2)
    return float(a_exact), float(b_theory)


def remove_point_sums(s: dict, r_i: float, j_i: float) -> dict:
    return {
        "n": s["n"] - 1,
        "S_J": s["S_J"] - j_i, "S_R": s["S_R"] - r_i,
        "S_JJ": s["S_JJ"] - j_i * j_i, "S_RR": s["S_RR"] - r_i * r_i,
        "S_JR": s["S_JR"] - j_i * r_i,
        "S_RRR": s["S_RRR"] - r_i ** 3, "S_JRR": s["S_JRR"] - j_i * r_i * r_i,
    }


def remove_subset_sums(s: dict, r_sub: np.ndarray, j_sub: np.ndarray) -> dict:
    return {
        "n": s["n"] - len(r_sub),
        "S_J": s["S_J"] - float(np.sum(j_sub)), "S_R": s["S_R"] - float(np.sum(r_sub)),
        "S_JJ": s["S_JJ"] - float(np.sum(j_sub * j_sub)), "S_RR": s["S_RR"] - float(np.sum(r_sub * r_sub)),
        "S_JR": s["S_JR"] - float(np.sum(j_sub * r_sub)),
        "S_RRR": s["S_RRR"] - float(np.sum(r_sub ** 3)), "S_JRR": s["S_JRR"] - float(np.sum(j_sub * r_sub * r_sub)),
    }


def jackknife_ab_vectorized(r: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact O(n) leave-one-out (A_loo_i, B_loo_i) for every item i."""
    n = len(r)
    s = power_sums(r, j)
    n_arr = np.full(n, s["n"] - 1, dtype=float)
    j_bar = (s["S_J"] - j) / n_arr
    r_bar = (s["S_R"] - r) / n_arr
    sigma_j2 = (s["S_JJ"] - j * j) / n_arr - j_bar ** 2
    sigma_r2 = (s["S_RR"] - r * r) / n_arr - r_bar ** 2
    cov_jr = (s["S_JR"] - j * r) / n_arr - j_bar * r_bar
    rho = cov_jr - sigma_j2
    kappa_r = (s["S_RRR"] - r ** 3) / n_arr - 3 * r_bar * (s["S_RR"] - r * r) / n_arr + 2 * r_bar ** 3
    kappa_jrr = ((s["S_JRR"] - j * r * r) / n_arr - 2 * r_bar * (s["S_JR"] - j * r) / n_arr
                 - j_bar * (s["S_RR"] - r * r) / n_arr + 2 * j_bar * r_bar ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_loo = np.sqrt(2.0) * (sigma_j2 + rho) / np.sqrt(sigma_r2)
        b_loo = kappa_jrr / sigma_r2 - 2.0 * (sigma_j2 + rho) * kappa_r / (3.0 * sigma_r2 ** 2)
    return a_loo, b_loo


def _selfcheck(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 60
    r = rng.normal(size=n) * 2 + 5
    j = 0.5 * r + rng.normal(size=n) * 0.7

    coeff_full = theory_coefficients(j, r)
    s = power_sums(r, j)
    a_closed, b_closed = ab_from_sums(s)
    assert np.isclose(a_closed, coeff_full.a_exact, rtol=1e-9), (a_closed, coeff_full.a_exact)
    assert np.isclose(b_closed, coeff_full.b_theory, rtol=1e-9), (b_closed, coeff_full.b_theory)

    a_loo, b_loo = jackknife_ab_vectorized(r, j)
    for i in [0, 5, 30, n - 1]:
        mask = np.arange(n) != i
        ref = theory_coefficients(j[mask], r[mask])
        assert np.isclose(a_loo[i], ref.a_exact, rtol=1e-8), (i, a_loo[i], ref.a_exact)
        assert np.isclose(b_loo[i], ref.b_theory, rtol=1e-8), (i, b_loo[i], ref.b_theory)

    # leave-top-5% self-check
    k = max(1, int(0.05 * n))
    top_idx = np.argsort(r)[-k:]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[top_idx] = False
    s_reduced = remove_subset_sums(s, r[top_idx], j[top_idx])
    a_top, b_top = ab_from_sums(s_reduced)
    ref = theory_coefficients(j[keep_mask], r[keep_mask])
    assert np.isclose(a_top, ref.a_exact, rtol=1e-8), (a_top, ref.a_exact)
    assert np.isclose(b_top, ref.b_theory, rtol=1e-8), (b_top, ref.b_theory)
    print("Self-check passed: closed-form full-sample, leave-one-out, and leave-top-k all match theory_coefficients exactly.")


def run() -> pd.DataFrame:
    _selfcheck()
    rows = []
    for cfg in bs.CONFIGS:
        paths = bs.build_paths(cfg)
        for path in paths:
            r = np.asarray(path.proxy_scores, dtype=float)
            j = np.asarray(path.gold_scores, dtype=float)
            if len(np.unique(r)) < 2:
                continue
            coeff_full = theory_coefficients(j, r)
            a_loo, b_loo = jackknife_ab_vectorized(r, j)
            valid = np.isfinite(a_loo) & np.isfinite(b_loo)
            max_a_influence = float(np.max(np.abs(a_loo[valid] - coeff_full.a_exact))) if valid.any() else float("nan")
            max_b_influence = float(np.max(np.abs(b_loo[valid] - coeff_full.b_theory))) if valid.any() else float("nan")

            s_full = power_sums(r, j)
            row = {
                "dataset": cfg["dataset_name"], "group_key": path.group_key, "path_id": path.path_id,
                "n_items": len(r), "a_full": coeff_full.a_exact, "b_full": coeff_full.b_theory,
                "jk_max_a_influence": max_a_influence, "jk_max_b_influence": max_b_influence,
                "jk_max_b_influence_rel": max_b_influence / max(abs(coeff_full.b_theory), 1e-8),
            }
            for pct in (0.01, 0.05):
                k = max(1, int(round(pct * len(r))))
                top_idx = np.argsort(r)[-k:]
                keep_mask = np.ones(len(r), dtype=bool)
                keep_mask[top_idx] = False
                s_reduced = remove_subset_sums(s_full, r[top_idx], j[top_idx])
                a_top, b_top = ab_from_sums(s_reduced)
                tag = f"top{int(pct * 100)}pct"
                row[f"a_excl_{tag}"] = a_top
                row[f"b_excl_{tag}"] = b_top
                row[f"b_shift_excl_{tag}"] = b_top - coeff_full.b_theory
                row[f"b_sign_flip_excl_{tag}"] = bool(np.sign(b_top) != np.sign(coeff_full.b_theory))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = run()
    out_path = args.outdir / "jackknife_leave_top_k.csv"
    df.to_csv(out_path, index=False)
    print(f"-> {out_path} ({len(df)} paths)")
    print(df.groupby("dataset")[["jk_max_b_influence_rel", "b_sign_flip_excl_top5pct"]].mean().to_string())


if __name__ == "__main__":
    main()
