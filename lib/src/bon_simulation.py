"""
Best-of-N (BoN) simulation at the system level.

Given a proxy/oracle annotator split, for each article we:
  1. Sample K systems uniformly
  2. Select the system with the highest mean proxy score
  3. Record the oracle score of that system

Sweeping K from 1..N_systems produces a BoN curve whose saturation gap
ΔJ_max should scale with ICC(A,1) per Theorem 2.
"""

from __future__ import annotations

import random
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Proxy / oracle score computation
# ---------------------------------------------------------------------------

def compute_proxy_oracle_scores(
    df: pd.DataFrame,
    proxy_raters: list,
    oracle_raters: list,
    dimension: str,
    article_col: str = "article_id",
    model_col: str = "model_id",
    rater_type_col: str = "rater_type",
    rater_idx_col: str = "rater_idx",
    dim_col: str = "dimension",
    score_col: str = "score",
) -> Tuple[Dict, Dict]:
    """Compute mean proxy and oracle scores per (article, system).

    Parameters
    ----------
    proxy_raters, oracle_raters : list of (rater_type, rater_idx) tuples
        e.g. [("turker", 0), ("turker", 1), ("turker", 2)]

    Returns
    -------
    proxy_scores : dict  {article_id: {model_id: float}}
    oracle_scores : dict {article_id: {model_id: float}}
    """
    df_dim = df[df[dim_col] == dimension].copy()

    def _build_scores(rater_list):
        mask = pd.Series(False, index=df_dim.index)
        for rtype, ridx in rater_list:
            mask |= (df_dim[rater_type_col] == rtype) & (df_dim[rater_idx_col] == ridx)
        sub = df_dim[mask]
        grouped = sub.groupby([article_col, model_col])[score_col].mean()
        scores = defaultdict(dict)
        for (art, mod), val in grouped.items():
            scores[art][mod] = val
        return dict(scores)

    return _build_scores(proxy_raters), _build_scores(oracle_raters)


# ---------------------------------------------------------------------------
# BoN simulation
# ---------------------------------------------------------------------------

def simulate_bon_curve(
    proxy_scores: Dict[str, Dict[str, float]],
    oracle_scores: Dict[str, Dict[str, float]],
    K_values: List[int] | None = None,
    n_trials: int = 300,
    seed: int = 42,
) -> Dict[int, float]:
    """Simulate BoN selection for each K and return mean oracle quality.

    Returns
    -------
    bon_curve : {K: mean_oracle_score}
    """
    rng = random.Random(seed)

    articles = sorted(set(proxy_scores.keys()) & set(oracle_scores.keys()))
    if not articles:
        return {}

    # Determine available systems (intersection across articles)
    all_systems = None
    for art in articles:
        systems_art = set(proxy_scores[art].keys()) & set(oracle_scores[art].keys())
        if all_systems is None:
            all_systems = systems_art
        else:
            all_systems &= systems_art
    systems = sorted(all_systems)
    N_systems = len(systems)

    if K_values is None:
        K_values = list(range(1, N_systems + 1))

    bon_curve = {}
    for K in K_values:
        bon_scores = []
        for art in articles:
            for _ in range(n_trials):
                sampled = rng.sample(systems, min(K, N_systems))
                best = max(sampled, key=lambda s: proxy_scores[art][s])
                bon_scores.append(oracle_scores[art][best])
        bon_curve[K] = np.mean(bon_scores)

    return bon_curve


def compute_delta_J_max(bon_curve: Dict[int, float]) -> float:
    """ΔJ_max = peak oracle quality - baseline (K=1).  Always >= 0."""
    if not bon_curve:
        return np.nan
    baseline = bon_curve.get(1, bon_curve[min(bon_curve.keys())])
    peak = max(bon_curve.values())
    return peak - baseline


def compute_delta_at_Kmax(bon_curve: Dict[int, float]) -> float:
    """Δ@K_max = oracle quality at largest K minus baseline (K=1).

    Unlike compute_delta_J_max, this CAN be negative — a negative value
    means proxy-based selection actively *hurts* oracle quality (anti-selection).
    """
    if not bon_curve:
        return np.nan
    baseline = bon_curve.get(1, bon_curve[min(bon_curve.keys())])
    K_max = max(bon_curve.keys())
    return bon_curve[K_max] - baseline


def compute_proxy_oracle_corr(
    proxy_scores: Dict[str, Dict[str, float]],
    oracle_scores: Dict[str, Dict[str, float]],
) -> float:
    """Mean within-article Pearson r between proxy and oracle scores.

    For each article, computes Pearson r between proxy and oracle system scores,
    then returns the mean r across articles.  This directly measures how well
    the proxy identifies the best system *within* each article — exactly the
    mechanism BoN selection relies on.

    A value near 0 means the proxy carries no information about oracle quality.
    A negative value means proxy actively anti-selects.
    """
    articles = sorted(set(proxy_scores.keys()) & set(oracle_scores.keys()))
    within_rs = []
    for art in articles:
        systems = sorted(set(proxy_scores[art].keys()) & set(oracle_scores[art].keys()))
        if len(systems) < 3:
            continue
        pv = [proxy_scores[art][s] for s in systems]
        ov = [oracle_scores[art][s] for s in systems]
        if np.std(pv) > 1e-12 and np.std(ov) > 1e-12:
            within_rs.append(float(np.corrcoef(pv, ov)[0, 1]))
    if not within_rs:
        return np.nan
    return float(np.mean(within_rs))


def compute_oracle_score_range(
    oracle_scores: Dict[str, Dict[str, float]],
) -> float:
    """Range of mean oracle scores across systems (averaged over articles).

    Large range = BoN has room to differentiate; small range = ceiling effect.
    """
    articles = sorted(oracle_scores.keys())
    if not articles:
        return np.nan
    # collect per-system means across articles
    system_totals: Dict[str, List[float]] = defaultdict(list)
    for art in articles:
        for s, v in oracle_scores[art].items():
            system_totals[s].append(v)
    system_means = [np.mean(vals) for vals in system_totals.values()]
    return max(system_means) - min(system_means) if system_means else np.nan


# ---------------------------------------------------------------------------
# Bootstrap for ΔJ_max
# ---------------------------------------------------------------------------

_BOOT_PROXY_FULL = None
_BOOT_ORACLE_FULL = None
_BOOT_N_TRIALS = None
_BOOT_SEED = None


def _init_bon_bootstrap_worker(proxy_full, oracle_full, n_trials_per_boot, seed):
    global _BOOT_PROXY_FULL, _BOOT_ORACLE_FULL, _BOOT_N_TRIALS, _BOOT_SEED
    _BOOT_PROXY_FULL = proxy_full
    _BOOT_ORACLE_FULL = oracle_full
    _BOOT_N_TRIALS = n_trials_per_boot
    _BOOT_SEED = seed


def _bon_bootstrap_one(task):
    b, sampled_arts = task
    proxy_boot = {f"{a}__b{i}": _BOOT_PROXY_FULL[a] for i, a in enumerate(sampled_arts)}
    oracle_boot = {f"{a}__b{i}": _BOOT_ORACLE_FULL[a] for i, a in enumerate(sampled_arts)}
    curve = simulate_bon_curve(
        proxy_boot, oracle_boot,
        n_trials=_BOOT_N_TRIALS, seed=_BOOT_SEED + b,
    )
    return compute_delta_J_max(curve), compute_delta_at_Kmax(curve)

def bootstrap_bon_delta(
    df: pd.DataFrame,
    dimension: str,
    proxy_raters: list,
    oracle_raters: list,
    article_col: str = "article_id",
    model_col: str = "model_id",
    rater_type_col: str = "rater_type",
    rater_idx_col: str = "rater_idx",
    dim_col: str = "dimension",
    score_col: str = "score",
    n_boot: int = 2000,
    n_trials_per_boot: int = 100,
    seed: int = 42,
) -> dict:
    """Bootstrap ΔJ_max AND Δ@K_max by resampling articles.

    Returns dict with keys for both metrics:
      - mean, std, ci95_lower, ci95_upper  (ΔJ_max — always >= 0)
      - raw_mean, raw_std, raw_ci95_lower, raw_ci95_upper  (Δ@K_max — can be negative)
    """
    rng_np = np.random.RandomState(seed)

    # Full proxy/oracle scores
    proxy_full, oracle_full = compute_proxy_oracle_scores(
        df, proxy_raters, oracle_raters, dimension,
        article_col, model_col, rater_type_col, rater_idx_col, dim_col, score_col,
    )
    articles = sorted(set(proxy_full.keys()) & set(oracle_full.keys()))

    tasks = [
        (b, rng_np.choice(articles, size=len(articles), replace=True).tolist())
        for b in range(n_boot)
    ]

    n_jobs = int(os.environ.get("BON_BOOTSTRAP_JOBS", "1"))
    if n_jobs > 1 and n_boot > 1:
        chunksize = max(1, n_boot // (n_jobs * 4))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_bon_bootstrap_worker,
            initargs=(proxy_full, oracle_full, n_trials_per_boot, seed),
        ) as pool:
            pairs = list(pool.map(_bon_bootstrap_one, tasks, chunksize=chunksize))
    else:
        _init_bon_bootstrap_worker(proxy_full, oracle_full, n_trials_per_boot, seed)
        pairs = [_bon_bootstrap_one(task) for task in tasks]

    deltas = [p[0] for p in pairs]
    deltas_raw = [p[1] for p in pairs]

    deltas = np.array(deltas)
    deltas_raw = np.array(deltas_raw)
    return {
        "mean": np.mean(deltas),
        "std": np.std(deltas),
        "ci95_lower": np.percentile(deltas, 2.5),
        "ci95_upper": np.percentile(deltas, 97.5),
        "raw_mean": np.mean(deltas_raw),
        "raw_std": np.std(deltas_raw),
        "raw_ci95_lower": np.percentile(deltas_raw, 2.5),
        "raw_ci95_upper": np.percentile(deltas_raw, 97.5),
    }
