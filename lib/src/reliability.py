"""
Core reliability metrics: ICC(A,1), ICC(C,1), personalization gap, split-half stability.

All functions expect a **long-format** DataFrame with columns:
  target  — item being rated  (e.g. article_id × model_id composite key)
  rater   — rater identifier
  score   — numeric rating
"""

from __future__ import annotations

import warnings
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import pingouin as pg


# ---------------------------------------------------------------------------
# ICC wrappers
# ---------------------------------------------------------------------------

def compute_icc(
    df: pd.DataFrame,
    target_col: str = "target",
    rater_col: str = "rater",
    score_col: str = "score",
    icc_type: str = "ICC2",
) -> dict:
    """Compute a single ICC value using pingouin.

    Parameters
    ----------
    icc_type : str
        One of ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k.
        - ICC2  = two-way random, absolute agreement, single rater  → ICC(A,1)
        - ICC3  = two-way mixed, consistency, single rater          → ICC(C,1)

    Returns
    -------
    dict with keys: ICC, F, df1, df2, pval, CI95_lower, CI95_upper
    """
    sub = df[[target_col, rater_col, score_col]].dropna()
    if sub[target_col].nunique() < 3 or sub[rater_col].nunique() < 2:
        return {"ICC": np.nan, "CI95_lower": np.nan, "CI95_upper": np.nan}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        icc_table = pg.intraclass_corr(
            data=sub,
            targets=target_col,
            raters=rater_col,
            ratings=score_col,
        )
    row = icc_table[icc_table["Type"] == icc_type].iloc[0]
    ci = row["CI95"]  # pingouin returns list [lower, upper]
    return {
        "ICC": row["ICC"],
        "F": row["F"],
        "df1": row["df1"],
        "df2": row["df2"],
        "pval": row["pval"],
        "CI95_lower": ci[0],
        "CI95_upper": ci[1],
    }


def compute_icc_a1(df, target_col="target", rater_col="rater", score_col="score"):
    """ICC(A,1): two-way random, absolute agreement, single rater."""
    return compute_icc(df, target_col, rater_col, score_col, icc_type="ICC(A,1)")


def compute_icc_c1(df, target_col="target", rater_col="rater", score_col="score"):
    """ICC(C,1): two-way mixed, consistency, single rater."""
    return compute_icc(df, target_col, rater_col, score_col, icc_type="ICC(C,1)")


def compute_personalization_gap(
    df, target_col="target", rater_col="rater", score_col="score"
) -> dict:
    """G_pers = ICC(C,1) - ICC(A,1)."""
    a1 = compute_icc_a1(df, target_col, rater_col, score_col)
    c1 = compute_icc_c1(df, target_col, rater_col, score_col)
    return {
        "ICC_A1": a1["ICC"],
        "ICC_C1": c1["ICC"],
        "G_pers": c1["ICC"] - a1["ICC"] if not (np.isnan(a1["ICC"]) or np.isnan(c1["ICC"])) else np.nan,
        "ICC_A1_CI": (a1["CI95_lower"], a1["CI95_upper"]),
        "ICC_C1_CI": (c1["CI95_lower"], c1["CI95_upper"]),
    }


# ---------------------------------------------------------------------------
# Split-half stability (Experiment 3)
# ---------------------------------------------------------------------------

_SPLIT_DF = None
_SPLIT_TARGET_COL = None
_SPLIT_RATER_COL = None
_SPLIT_SCORE_COL = None
_SPLIT_AGG_FUNC = None


def _init_split_half_worker(df, target_col, rater_col, score_col, agg_func):
    global _SPLIT_DF, _SPLIT_TARGET_COL, _SPLIT_RATER_COL, _SPLIT_SCORE_COL, _SPLIT_AGG_FUNC
    _SPLIT_DF = df
    _SPLIT_TARGET_COL = target_col
    _SPLIT_RATER_COL = rater_col
    _SPLIT_SCORE_COL = score_col
    _SPLIT_AGG_FUNC = agg_func


def _split_half_one(perm):
    half = len(perm) // 2
    group_a, group_b = perm[:half], perm[half: half * 2]

    df_a = _SPLIT_DF[_SPLIT_DF[_SPLIT_RATER_COL].isin(group_a)]
    df_b = _SPLIT_DF[_SPLIT_DF[_SPLIT_RATER_COL].isin(group_b)]

    agg_a = _aggregate(df_a, _SPLIT_TARGET_COL, _SPLIT_SCORE_COL, _SPLIT_AGG_FUNC)
    agg_b = _aggregate(df_b, _SPLIT_TARGET_COL, _SPLIT_SCORE_COL, _SPLIT_AGG_FUNC)

    merged = agg_a.merge(agg_b, on=_SPLIT_TARGET_COL, suffixes=("_a", "_b"))
    if len(merged) < 5:
        return np.nan
    return merged[f"{_SPLIT_SCORE_COL}_a"].corr(merged[f"{_SPLIT_SCORE_COL}_b"])

def compute_split_half_stability(
    df: pd.DataFrame,
    target_col: str = "target",
    rater_col: str = "rater",
    score_col: str = "score",
    agg_func: str = "mean",
    n_splits: int = 200,
    seed: int = 42,
) -> dict:
    """Split raters randomly in half, aggregate each half, correlate across targets.

    Parameters
    ----------
    agg_func : str
        "mean", "median", or "trimmed_mean"

    Returns
    -------
    dict with mean_corr, std_corr, ci95
    """
    rng = np.random.RandomState(seed)
    raters = df[rater_col].unique()
    n_raters = len(raters)
    if n_raters < 4:
        return {"mean_corr": np.nan, "std_corr": np.nan, "ci95": (np.nan, np.nan)}

    half = n_raters // 2
    perms = [rng.permutation(raters).tolist() for _ in range(n_splits)]
    n_jobs = int(os.environ.get("SPLIT_HALF_JOBS", "1"))
    if n_jobs > 1 and n_splits > 1:
        chunksize = max(1, n_splits // (n_jobs * 4))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_split_half_worker,
            initargs=(df, target_col, rater_col, score_col, agg_func),
        ) as pool:
            corrs = list(pool.map(_split_half_one, perms, chunksize=chunksize))
    else:
        _init_split_half_worker(df, target_col, rater_col, score_col, agg_func)
        corrs = [_split_half_one(perm) for perm in perms]

    corrs = np.array(corrs)
    corrs = corrs[np.isfinite(corrs)]
    if len(corrs) == 0:
        return {"mean_corr": np.nan, "std_corr": np.nan, "ci95": (np.nan, np.nan)}
    return {
        "mean_corr": np.mean(corrs),
        "std_corr": np.std(corrs),
        "ci95": (np.percentile(corrs, 2.5), np.percentile(corrs, 97.5)),
    }


def _aggregate(df, target_col, score_col, method):
    """Aggregate scores per target using the given method."""
    if method == "mean":
        return df.groupby(target_col)[score_col].mean().reset_index()
    elif method == "median":
        return df.groupby(target_col)[score_col].median().reset_index()
    elif method == "trimmed_mean":
        from scipy.stats import trim_mean
        return (
            df.groupby(target_col)[score_col]
            .apply(lambda x: trim_mean(x, proportiontocut=0.1))
            .reset_index()
        )
    else:
        raise ValueError(f"Unknown aggregation: {method}")
