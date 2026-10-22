#!/usr/bin/env python3
"""Utilities for verifiability experiments.

Designed for local execution with only numpy/pandas/matplotlib.
Expected long-format input columns (SummEval-style):
    article_id, model_id, dimension, rater_type, rater_idx, score

All functions are standalone and intentionally dependency-light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DIMENSIONS_DEFAULT = ["coherence", "consistency", "fluency", "relevance"]
POOL_ALIASES = {
    "expert": {"expert", "experts", "exp"},
    "crowd": {"crowd", "turker", "turkers", "worker", "workers", "mturk"},
    "all": {"all"},
}


def _canonical_pool_name(name: str) -> str:
    s = str(name).strip().lower()
    for canon, aliases in POOL_ALIASES.items():
        if s in aliases:
            return canon
    return s


def add_item_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "item_id" not in df.columns:
        if {"article_id", "model_id"}.issubset(df.columns):
            df["item_id"] = df["article_id"].astype(str) + "___" + df["model_id"].astype(str)
        else:
            raise ValueError("Need either item_id or article_id/model_id columns.")
    return df


def normalize_rater_type(df: pd.DataFrame) -> pd.DataFrame:
    if "rater_type" not in df.columns:
        raise ValueError("Input must contain a rater_type column.")
    df = df.copy()
    df["rater_type_norm"] = df["rater_type"].map(_canonical_pool_name)
    return df


def load_long_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_item_id(df)
    df = normalize_rater_type(df)
    required = {"item_id", "dimension", "rater_type_norm", "rater_idx", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def zscore_within_rater(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Rater-wise z-score transform within each dimension × rater_type × rater_idx."""
    df = df.copy()
    group_cols = ["dimension", "rater_type_norm", "rater_idx"]
    grouped = df.groupby(group_cols)[score_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    df[score_col] = (df[score_col] - mean) / std
    # keep degenerate groups centered at 0 instead of NaN
    df[score_col] = df[score_col].fillna(0.0)
    return df


def make_matrix(
    df: pd.DataFrame,
    *,
    dimension: str,
    pool: str,
    score_col: str = "score",
    complete_cases: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, item_ids, rater_ids) for one dimension × pool.

    X shape: n_items × n_raters.
    """
    pool = _canonical_pool_name(pool)
    sub = df[df["dimension"] == dimension].copy()
    if pool != "all":
        sub = sub[sub["rater_type_norm"] == pool]
    if sub.empty:
        raise ValueError(f"No rows for dimension={dimension!r}, pool={pool!r}")

    sub["rater_key"] = sub["rater_type_norm"].astype(str) + "_" + sub["rater_idx"].astype(str)
    X = sub.pivot_table(index="item_id", columns="rater_key", values=score_col, aggfunc="first")
    if complete_cases:
        X = X.dropna(axis=0, how="any")
    X = X.dropna(axis=1, how="all")
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 items and 2 raters after filtering; got {X.shape}"
        )
    return X.to_numpy(dtype=float), X.index.to_numpy(), X.columns.to_numpy()


def compute_ab(J: Sequence[float], r: Sequence[float]) -> dict[str, float]:
    """Corrected Theorem 2 coefficients for one gold/proxy pair."""
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float)
    Jc = J - J.mean()
    rc = r - r.mean()
    sigma_J_sq = float(np.mean(Jc ** 2))
    sigma_r_sq = float(np.mean(rc ** 2))
    sigma_r = float(np.sqrt(max(sigma_r_sq, 1e-18)))
    eps = r - J
    rho = float(np.mean(Jc * ((eps) - eps.mean())))
    kappa_r = float(np.mean(rc ** 3))
    kappa_Jrr = float(np.mean(Jc * (rc ** 2)))
    A = float(np.sqrt(2.0) * (sigma_J_sq + rho) / sigma_r)
    B = float(kappa_Jrr / sigma_r_sq - 2.0 * (sigma_J_sq + rho) * kappa_r / (3.0 * sigma_r_sq ** 2))
    nu_tilde_sq = float(sigma_J_sq / sigma_r_sq)
    return {
        "A": A,
        "B": B,
        "rho": rho,
        "nu_tilde": float(np.sqrt(max(nu_tilde_sq, 0.0))),
        "sigma_J_sq": sigma_J_sq,
        "sigma_r_sq": sigma_r_sq,
        "kappa_r": kappa_r,
        "kappa_Jrr": kappa_Jrr,
    }


def tilt_curve(
    J: Sequence[float],
    r: Sequence[float],
    t_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact exponential tilt trajectory under uniform reference."""
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    n = len(r)
    d_vals = np.empty_like(t_grid)
    q_vals = np.empty_like(t_grid)

    for idx, t in enumerate(t_grid):
        logits = t * r
        m = float(np.max(logits))
        w = np.exp(logits - m)
        pi = w / w.sum()
        log_pi = np.log(pi + 1e-300)
        d_vals[idx] = float(np.dot(pi, log_pi + np.log(n)))  # KL(pi || uniform)
        q_vals[idx] = float(np.dot(pi, J))
    return d_vals, q_vals


def fit_local_ab(d: Sequence[float], q: Sequence[float], max_d: float) -> tuple[float, float, float]:
    """Fit Q(d)-Q(0) = A sqrt(d) + B d over 0 < d <= max_d."""
    d = np.asarray(d, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (d > 1e-12) & (d <= max_d)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.sqrt(d[mask]), d[mask]])
    y = q[mask] - q[0]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0]), float(coef[1]), r2


def gold_from_matrix(X: np.ndarray, j: int, estimator: str = "loo_mean") -> tuple[np.ndarray, np.ndarray]:
    """Return (gold, proxy) for held-out rater j."""
    X = np.asarray(X, dtype=float)
    estimator = estimator.lower()
    r = X[:, j]
    others = np.delete(X, j, axis=1)
    if others.shape[1] < 1:
        raise ValueError("Need at least one non-held-out rater.")
    if estimator == "loo_mean":
        J = others.mean(axis=1)
    elif estimator == "same_mean":
        J = X.mean(axis=1)
    elif estimator == "median_others":
        J = np.median(others, axis=1)
    elif estimator == "trimmed_mean_others":
        if others.shape[1] < 3:
            J = others.mean(axis=1)
        else:
            sorted_others = np.sort(others, axis=1)
            J = sorted_others[:, 1:-1].mean(axis=1)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return np.asarray(J, dtype=float), np.asarray(r, dtype=float)


def bootstrap_index_samples(n: int, n_boot: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    for _ in range(n_boot):
        yield rng.integers(0, n, size=n)


def icc_a1(X: np.ndarray) -> float:
    """Two-way random, absolute-agreement, single-rater ICC(A,1)."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    grand = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    msr = k * np.var(row_means, ddof=1)
    msc = n * np.var(col_means, ddof=1)
    sse = np.sum((X - row_means[:, None] - col_means[None, :] + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else np.nan


def icc_c1(X: np.ndarray) -> float:
    """Two-way mixed/random consistency ICC(C,1)."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    row_means = X.mean(axis=1)
    msr = k * np.var(row_means, ddof=1)
    sse = np.sum((X - row_means[:, None] - X.mean(axis=0)[None, :] + X.mean()) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse
    return float((msr - mse) / denom) if denom != 0 else np.nan


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]
