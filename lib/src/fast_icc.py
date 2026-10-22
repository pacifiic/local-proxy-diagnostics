"""
Fast ICC computation from a pivot matrix (targets × raters).

Formulas follow Shrout & Fleiss (1979) using one-way / two-way ANOVA decomposition.
This is ~100x faster than calling pingouin because it avoids long-format reshaping
and pandas groupby on every bootstrap iteration.

ICC(A,1) — two-way random, absolute agreement, single rater:
    ICC_A1 = (BMS - EMS) / (BMS + (k-1)*EMS + k*(JMS - EMS)/n)

ICC(C,1) — two-way mixed, consistency, single rater:
    ICC_C1 = (BMS - EMS) / (BMS + (k-1)*EMS)

where:
    n   = number of targets (rows)
    k   = number of raters (columns)
    BMS = between-target mean square
    WMS = within-target mean square
    EMS = error (residual) mean square
    JMS = between-rater (judge) mean square
"""

from __future__ import annotations
import numpy as np


def _anova_components(X: np.ndarray):
    """
    X: (n_targets, n_raters) array, may contain NaN (handled by ignoring).
    Returns BMS, WMS, JMS, EMS, n, k.
    """
    n, k = X.shape

    grand_mean = np.nanmean(X)
    target_means = np.nanmean(X, axis=1, keepdims=True)  # (n,1)
    rater_means  = np.nanmean(X, axis=0, keepdims=True)  # (1,k)

    SSb = k * np.nansum((target_means - grand_mean) ** 2)
    SSj = n * np.nansum((rater_means - grand_mean) ** 2)
    SSt = np.nansum((X - grand_mean) ** 2)
    SSe = SSt - SSb - SSj

    dfb = n - 1
    dfj = k - 1
    dfe = (n - 1) * (k - 1)

    BMS = SSb / dfb if dfb > 0 else np.nan
    JMS = SSj / dfj if dfj > 0 else np.nan
    EMS = SSe / dfe if dfe > 0 else np.nan
    WMS = (SSt - SSb) / (n * (k - 1)) if (n * (k - 1)) > 0 else np.nan

    return BMS, WMS, JMS, EMS, n, k


def icc_a1(X: np.ndarray) -> float:
    """ICC(A,1): two-way random, absolute agreement, single rater."""
    BMS, WMS, JMS, EMS, n, k = _anova_components(X)
    denom = BMS + (k - 1) * EMS + k * (JMS - EMS) / n
    if denom <= 0 or np.isnan(denom):
        return np.nan
    return (BMS - EMS) / denom


def icc_c1(X: np.ndarray) -> float:
    """ICC(C,1): two-way mixed, consistency, single rater."""
    BMS, WMS, JMS, EMS, n, k = _anova_components(X)
    denom = BMS + (k - 1) * EMS
    if denom <= 0 or np.isnan(denom):
        return np.nan
    return (BMS - EMS) / denom


def df_to_matrix(
    df,
    target_col: str = "target",
    rater_col: str = "rater",
    score_col: str = "score",
) -> np.ndarray:
    """Convert long-format DataFrame to (n_targets, n_raters) matrix."""
    pivot = df.pivot_table(
        index=target_col, columns=rater_col, values=score_col, aggfunc="mean"
    )
    return pivot.values  # (n_targets, n_raters)


def bootstrap_icc_fast(
    X: np.ndarray,
    icc_fn=icc_a1,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap ICC by resampling rows (targets) of the pivot matrix.

    X: (n_targets, n_raters) — no NaN preferred (pad with column mean if needed)
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        val = icc_fn(X[idx])
        if not np.isnan(val):
            values.append(val)
    values = np.array(values)
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
        "n_valid": len(values),
    }


def bootstrap_gap_fast(
    X: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap G_pers = ICC(C,1) - ICC(A,1)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xb = X[idx]
        a1 = icc_a1(Xb)
        c1 = icc_c1(Xb)
        if not (np.isnan(a1) or np.isnan(c1)):
            values.append(c1 - a1)
    values = np.array(values)
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
        "n_valid": len(values),
    }


def bootstrap_lift_fast(
    X_a: np.ndarray,
    X_b: np.ndarray,
    icc_fn=icc_a1,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap Lift = ICC(X_b) - ICC(X_a), resampling shared row indices."""
    assert X_a.shape[0] == X_b.shape[0], "X_a and X_b must have same number of targets"
    rng = np.random.RandomState(seed)
    n = X_a.shape[0]
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        va = icc_fn(X_a[idx])
        vb = icc_fn(X_b[idx])
        if not (np.isnan(va) or np.isnan(vb)):
            values.append(vb - va)
    values = np.array(values)
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
        "n_valid": len(values),
    }
