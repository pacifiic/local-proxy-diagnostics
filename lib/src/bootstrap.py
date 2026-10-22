"""
Bootstrap utilities for ICC, gap, lift, and arbitrary metrics.

All bootstrap functions resample **targets** (not individual ratings) to respect
the clustered structure of the data.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .reliability import compute_icc_a1, compute_icc_c1


def _resample_targets(df: pd.DataFrame, target_col: str, rng: np.random.RandomState):
    """Resample targets with replacement, keeping all raters per target."""
    targets = df[target_col].unique()
    sampled = rng.choice(targets, size=len(targets), replace=True)

    frames = []
    for i, t in enumerate(sampled):
        chunk = df[df[target_col] == t].copy()
        chunk[target_col] = f"{t}__boot{i}"  # make unique
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Generic bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metric(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    target_col: str = "target",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap any metric_fn(df) → float by resampling targets."""
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_boot):
        boot_df = _resample_targets(df, target_col, rng)
        val = metric_fn(boot_df)
        if not np.isnan(val):
            values.append(val)
    values = np.array(values)
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "ci95_lower": np.percentile(values, 2.5),
        "ci95_upper": np.percentile(values, 97.5),
        "n_valid": len(values),
    }


# ---------------------------------------------------------------------------
# Specific bootstrap wrappers
# ---------------------------------------------------------------------------

def bootstrap_icc(
    df: pd.DataFrame,
    icc_type: str = "a1",
    target_col: str = "target",
    rater_col: str = "rater",
    score_col: str = "score",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap ICC(A,1) or ICC(C,1)."""
    fn = compute_icc_a1 if icc_type == "a1" else compute_icc_c1

    def metric(boot_df):
        res = fn(boot_df, target_col, rater_col, score_col)
        return res["ICC"]

    return bootstrap_metric(df, metric, target_col, n_boot, seed)


def bootstrap_gap(
    df: pd.DataFrame,
    target_col: str = "target",
    rater_col: str = "rater",
    score_col: str = "score",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap G_pers = ICC(C,1) - ICC(A,1)."""
    def metric(boot_df):
        a1 = compute_icc_a1(boot_df, target_col, rater_col, score_col)["ICC"]
        c1 = compute_icc_c1(boot_df, target_col, rater_col, score_col)["ICC"]
        if np.isnan(a1) or np.isnan(c1):
            return np.nan
        return c1 - a1

    return bootstrap_metric(df, metric, target_col, n_boot, seed)


def bootstrap_lift(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    icc_type: str = "a1",
    target_col: str = "target",
    rater_col: str = "rater",
    score_col: str = "score",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap Lift = ICC(pool_b) - ICC(pool_a).

    Typically pool_a = crowd, pool_b = expert.
    Both pools are resampled on shared targets.
    """
    fn = compute_icc_a1 if icc_type == "a1" else compute_icc_c1
    rng = np.random.RandomState(seed)

    # Shared target set
    shared_targets = np.intersect1d(
        df_a[target_col].unique(), df_b[target_col].unique()
    )
    df_a = df_a[df_a[target_col].isin(shared_targets)]
    df_b = df_b[df_b[target_col].isin(shared_targets)]

    values = []
    for _ in range(n_boot):
        sampled = rng.choice(shared_targets, size=len(shared_targets), replace=True)
        frames_a, frames_b = [], []
        for i, t in enumerate(sampled):
            ca = df_a[df_a[target_col] == t].copy()
            cb = df_b[df_b[target_col] == t].copy()
            ca[target_col] = f"{t}__boot{i}"
            cb[target_col] = f"{t}__boot{i}"
            frames_a.append(ca)
            frames_b.append(cb)
        boot_a = pd.concat(frames_a, ignore_index=True)
        boot_b = pd.concat(frames_b, ignore_index=True)
        icc_a = fn(boot_a, target_col, rater_col, score_col)["ICC"]
        icc_b = fn(boot_b, target_col, rater_col, score_col)["ICC"]
        if not (np.isnan(icc_a) or np.isnan(icc_b)):
            values.append(icc_b - icc_a)

    values = np.array(values)
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "ci95_lower": np.percentile(values, 2.5),
        "ci95_upper": np.percentile(values, 97.5),
        "n_valid": len(values),
    }
