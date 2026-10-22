#!/usr/bin/env python3
"""Utilities for supplemental-audit experiments.

These scripts are designed to test three diagnostic claims:
1) empirical monotone-error amplification audits for Theorem 1;
2) empirical checks of rho / nu^2 scaling normalization conditions;
3) measurement-improvement interventions using repeated human ratings.

Supported input:
- generic CSV/TSV/parquet in long format with at least:
    item identifier
    rater identifier
    score
- optional group columns (e.g. dataset_id, dimension, rater_type)

Examples
--------
SummEval long CSV:
    article_id, model_id, dimension, rater_type, rater_idx, score

WMT MQM long CSV:
    dataset_id, item_id, rater, score
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# -----------------------------
# IO / canonicalization
# -----------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def infer_item_col(df: pd.DataFrame) -> str:
    if "item_id" in df.columns:
        return "item_id"
    if {"article_id", "model_id"}.issubset(df.columns):
        df["item_id"] = df["article_id"].astype(str) + "___" + df["model_id"].astype(str)
        return "item_id"
    if {"doc", "seg_id", "system"}.issubset(df.columns):
        df["item_id"] = (
            df["doc"].astype(str) + "___" + df["seg_id"].astype(str) + "___" + df["system"].astype(str)
        )
        return "item_id"
    raise ValueError("Could not infer item column. Provide --item-col explicitly.")


def infer_rater_col(df: pd.DataFrame) -> str:
    if "rater" in df.columns:
        return "rater"
    if "judge" in df.columns:
        return "judge"
    if {"rater_type", "rater_idx"}.issubset(df.columns):
        df["rater"] = df["rater_type"].astype(str) + "_" + df["rater_idx"].astype(str)
        return "rater"
    if "rater_idx" in df.columns:
        df["rater"] = df["rater_idx"].astype(str)
        return "rater"
    raise ValueError("Could not infer rater column. Provide --rater-col explicitly.")


def infer_group_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if "dataset_id" in df.columns:
        cols.append("dataset_id")
    if "dimension" in df.columns:
        cols.append("dimension")
    if "rater_type" in df.columns:
        cols.append("rater_type")
    return cols


def canonicalize_long(
    df: pd.DataFrame,
    *,
    item_col: str | None = None,
    rater_col: str | None = None,
    score_col: str = "score",
    group_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    if item_col is None:
        item_col = infer_item_col(df)
    elif item_col not in df.columns:
        raise ValueError(f"item_col={item_col!r} not found in dataframe")

    if rater_col is None:
        rater_col = infer_rater_col(df)
    elif rater_col not in df.columns:
        raise ValueError(f"rater_col={rater_col!r} not found in dataframe")

    if score_col not in df.columns:
        raise ValueError(f"score_col={score_col!r} not found in dataframe")

    if group_cols is None:
        group_cols = infer_group_cols(df)
    else:
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"group columns missing from dataframe: {missing}")

    out = df.copy()
    out["item_id"] = out[item_col].astype(str)
    out["rater"] = out[rater_col].astype(str)
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna(subset=["item_id", "rater", "score"]).copy()

    if len(group_cols) == 0:
        out["group_id"] = "all"
    else:
        out["group_id"] = out[list(group_cols)].astype(str).agg("___".join, axis=1)

    keep = ["group_id", "item_id", "rater", "score"] + list(group_cols)
    out = out[keep].copy()
    return out, list(group_cols)


def zscore_within_rater(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(["group_id", "rater"])["score"]
    mu = grouped.transform("mean")
    sd = grouped.transform("std").replace(0, np.nan)
    df["score"] = (df["score"] - mu) / sd
    df["score"] = df["score"].fillna(0.0)
    return df


# -----------------------------
# Core local-law utilities
# -----------------------------

def compute_ab(J: Sequence[float], r: Sequence[float]) -> dict[str, float]:
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float)
    Jc = J - J.mean()
    rc = r - r.mean()
    sigma_J_sq = float(np.mean(Jc ** 2))
    sigma_r_sq = float(np.mean(rc ** 2))
    sigma_r_sq = max(sigma_r_sq, 1e-18)
    sigma_r = float(np.sqrt(sigma_r_sq))
    eps = r - J
    epsc = eps - eps.mean()
    rho = float(np.mean(Jc * epsc))
    kappa_r = float(np.mean(rc ** 3))
    kappa_Jrr = float(np.mean(Jc * (rc ** 2)))
    A = float(np.sqrt(2.0) * (sigma_J_sq + rho) / sigma_r)
    B = float(
        kappa_Jrr / sigma_r_sq
        - 2.0 * (sigma_J_sq + rho) * kappa_r / (3.0 * sigma_r_sq ** 2)
    )
    nu_tilde_sq = sigma_J_sq / sigma_r_sq
    return {
        "A": A,
        "B": B,
        "rho": rho,
        "sigma_J_sq": sigma_J_sq,
        "sigma_r_sq": sigma_r_sq,
        "sigma_J": float(np.sqrt(max(sigma_J_sq, 0.0))),
        "sigma_r": float(np.sqrt(sigma_r_sq)),
        "kappa_r": kappa_r,
        "kappa_Jrr": kappa_Jrr,
        "nu_tilde_sq": float(nu_tilde_sq),
        "nu_tilde": float(np.sqrt(max(nu_tilde_sq, 0.0))),
    }


def tilt_curve(J: Sequence[float], r: Sequence[float], t_grid: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
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
        d_vals[idx] = float(np.dot(pi, log_pi + np.log(n)))
        q_vals[idx] = float(np.dot(pi, J))
    return d_vals, q_vals


def fit_local_ab(d: Sequence[float], q: Sequence[float], max_d: float) -> tuple[float, float, float]:
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


# -----------------------------
# Matrix / ICC helpers
# -----------------------------

def make_complete_matrix(df_group: pd.DataFrame, min_raters: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df_group.pivot_table(index="item_id", columns="rater", values="score", aggfunc="first")
    X = X.dropna(axis=0, how="any")
    if X.shape[0] < 2 or X.shape[1] < min_raters:
        raise ValueError(f"Need at least 2 items and {min_raters} raters; got {X.shape}")
    return X.to_numpy(dtype=float), X.index.to_numpy(), X.columns.to_numpy()


def icc_a1(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan
    grand = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    msr = k * np.var(row_means, ddof=1)
    msc = n * np.var(col_means, ddof=1)
    sse = np.sum((X - row_means[:, None] - col_means[None, :] + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else np.nan


def spearman_brown(icc1: float, k: int) -> float:
    if np.isnan(icc1):
        return np.nan
    denom = 1 + (k - 1) * icc1
    return float(k * icc1 / denom) if denom != 0 else np.nan


def bootstrap_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    lo = np.quantile(x, alpha / 2)
    hi = np.quantile(x, 1 - alpha / 2)
    return float(lo), float(hi)


# -----------------------------
# Path constructors
# -----------------------------

def gold_from_matrix(X: np.ndarray, j: int, estimator: str = "loo_mean") -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    r = X[:, j]
    others = np.delete(X, j, axis=1)
    if others.shape[1] < 1:
        raise ValueError("Need at least one non-held-out rater")
    est = estimator.lower()
    if est == "loo_mean":
        J = others.mean(axis=1)
    elif est == "median_others":
        J = np.median(others, axis=1)
    elif est == "same_mean":
        J = X.mean(axis=1)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return np.asarray(J, dtype=float), np.asarray(r, dtype=float)


def iter_heldout_paths(
    df: pd.DataFrame,
    *,
    min_items: int = 50,
    min_raters: int = 3,
    score_transform: str = "raw",
    gold_estimator: str = "loo_mean",
) -> Iterable[dict[str, object]]:
    # Pre-scan to report skipped groups
    _all_groups = list(df.groupby("group_id"))
    _skipped_matrix = []
    for _gid, _g in _all_groups:
        _gg = _g.copy()
        if score_transform == "rater_z":
            _gg = zscore_within_rater(_gg)
        try:
            make_complete_matrix(_gg, min_raters=min_raters)
        except Exception:
            _skipped_matrix.append(str(_gid))
    if _skipped_matrix:
        print(f"iter_heldout_paths: {len(_skipped_matrix)}/{len(_all_groups)} groups skipped (matrix assembly failed, need >={min_raters} raters):")
        for s in _skipped_matrix:
            print(f"  {s}")

    # Also pre-scan for min_items to report
    _skipped_items = []
    for _gid, _g in _all_groups:
        _gg = _g.copy()
        if score_transform == "rater_z":
            _gg = zscore_within_rater(_gg)
        try:
            _X, _, _raters = make_complete_matrix(_gg, min_raters=min_raters)
        except Exception:
            continue
        for _j, _rater in enumerate(_raters):
            _J, _ = gold_from_matrix(_X, _j, estimator=gold_estimator)
            if len(_J) < min_items:
                _skipped_items.append(f"{_gid}/{_rater}: {len(_J)} items (need >={min_items})")
    if _skipped_items:
        print(f"iter_heldout_paths: {len(_skipped_items)} paths skipped (below min_items={min_items}):")
        for s in _skipped_items[:10]:
            print(f"  {s}")
        if len(_skipped_items) > 10:
            print(f"  ... and {len(_skipped_items) - 10} more")

    for group_id, g in df.groupby("group_id"):
        gg = g.copy()
        if score_transform == "rater_z":
            gg = zscore_within_rater(gg)
        try:
            X, item_ids, raters = make_complete_matrix(gg, min_raters=min_raters)
        except Exception:
            continue
        icc1 = icc_a1(X)
        for j, rater in enumerate(raters):
            J, r = gold_from_matrix(X, j, estimator=gold_estimator)
            if len(J) < min_items:
                continue
            yield {
                "group_id": group_id,
                "heldout_rater": str(rater),
                "n_items": int(len(J)),
                "n_raters_total": int(X.shape[1]),
                "icc_a1": icc1,
                "J": J,
                "r": r,
            }


# -----------------------------
# Monotone-condition helpers
# -----------------------------

def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def quantile_bin_summary(r: np.ndarray, eps: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    r = np.asarray(r, dtype=float)
    eps = np.asarray(eps, dtype=float)
    # Use quantile bins but keep unique edges only.
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(r, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin_id", "r_mean", "eps_mean", "count"])
    bins = np.digitize(r, edges[1:-1], right=True)
    rows = []
    for b in sorted(np.unique(bins)):
        mask = bins == b
        rows.append({
            "bin_id": int(b),
            "r_mean": float(r[mask].mean()),
            "eps_mean": float(eps[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    levels = y.copy()
    weights = w.copy()
    starts = list(range(len(y)))
    ends = list(range(len(y)))
    i = 0
    while i < len(levels) - 1:
        if levels[i] <= levels[i + 1] + 1e-15:
            i += 1
            continue
        new_w = weights[i] + weights[i + 1]
        new_level = (weights[i] * levels[i] + weights[i + 1] * levels[i + 1]) / new_w
        levels[i] = new_level
        weights[i] = new_w
        ends[i] = ends[i + 1]
        del starts[i + 1]
        del ends[i + 1]
        levels = np.delete(levels, i + 1)
        weights = np.delete(weights, i + 1)
        if i > 0:
            i -= 1
    out = np.empty(len(y), dtype=float)
    for s, e, lvl in zip(starts, ends, levels, strict=False):
        out[s : e + 1] = lvl
    return out


def isotonic_increasing_fit(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    yfit = pava(ys)
    return xs, yfit


def monotone_metrics(
    J: Sequence[float],
    r: Sequence[float],
    n_bins: int = 10,
    n_bootstrap: int = 0,
    random_state: int = 0,
) -> tuple[dict[str, float], pd.DataFrame]:
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float)
    eps = r - J
    bins = quantile_bin_summary(r, eps, n_bins=n_bins)

    if not bins.empty and len(bins) >= 2:
        diffs = np.diff(bins["eps_mean"].to_numpy(dtype=float))
        violation_frac = float(np.mean(diffs < -1e-12))
        bin_spearman = spearman_corr(
            bins["r_mean"].to_numpy(dtype=float),
            bins["eps_mean"].to_numpy(dtype=float),
        )
        zero_viol = float(violation_frac == 0.0)
    else:
        violation_frac = np.nan
        bin_spearman = np.nan
        zero_viol = np.nan

    xs, yiso = isotonic_increasing_fit(r, eps)
    y_sorted = eps[np.argsort(r, kind="mergesort")]
    sse_iso = float(np.sum((y_sorted - yiso) ** 2))
    sse_const = float(np.sum((y_sorted - y_sorted.mean()) ** 2))
    isotonic_r2 = 1.0 - sse_iso / sse_const if sse_const > 0 else np.nan

    boot_stats = []
    if n_bootstrap > 0 and len(J) >= 20:
        rng = np.random.default_rng(random_state)
        n = len(J)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bb = quantile_bin_summary(r[idx], eps[idx], n_bins=n_bins)
            if len(bb) >= 3:
                boot_stats.append(
                    spearman_corr(
                        bb["r_mean"].to_numpy(dtype=float),
                        bb["eps_mean"].to_numpy(dtype=float),
                    )
                )
        ci_lo, ci_hi = bootstrap_ci(boot_stats)
        frac_boot_pos = float(np.mean(np.asarray(boot_stats, dtype=float) > 0)) if boot_stats else np.nan
    else:
        ci_lo, ci_hi, frac_boot_pos = np.nan, np.nan, np.nan

    metrics = {
        "bin_spearman": bin_spearman,
        "bin_spearman_ci_lo": ci_lo,
        "bin_spearman_ci_hi": ci_hi,
        "frac_boot_bin_spearman_pos": frac_boot_pos,
        "bin_violation_frac": violation_frac,
        "bin_zero_violations": zero_viol,
        "isotonic_r2": isotonic_r2,
        "eps_mean": float(eps.mean()),
        "eps_std": float(eps.std()),
    }
    return metrics, bins


# -----------------------------
# Intervention helpers
# -----------------------------

@dataclass
class InterventionPath:
    group_id: str
    rep_id: int
    k_proxy: int
    proxy_raters: list[str]
    gold_raters: list[str]
    n_items: int
    J: np.ndarray
    r: np.ndarray
    icc_a1_full: float


def build_nested_intervention_paths(
    df_group: pd.DataFrame,
    *,
    k_list: Sequence[int],
    gold_size: int | None = None,
    n_reps: int = 200,
    random_state: int = 0,
    score_transform: str = "raw",
    min_items: int = 50,
) -> list[InterventionPath]:
    g = df_group.copy()
    if score_transform == "rater_z":
        g = zscore_within_rater(g)
    Xdf = g.pivot_table(index="item_id", columns="rater", values="score", aggfunc="first")
    Xdf = Xdf.dropna(axis=0, how="any")
    if Xdf.shape[0] < min_items or Xdf.shape[1] < 3:
        return []
    X = Xdf.to_numpy(dtype=float)
    raters = list(Xdf.columns.astype(str))
    m = len(raters)
    k_list = sorted(set(int(k) for k in k_list))
    max_k = max(k_list)
    if gold_size is None:
        gold_size = max(1, m - max_k)
    if gold_size + max_k > m:
        # shrink the largest usable k
        k_list = [k for k in k_list if k + gold_size <= m]
    if len(k_list) == 0:
        return []

    rng = np.random.default_rng(random_state)
    icc1 = icc_a1(X)
    out: list[InterventionPath] = []

    for rep in range(n_reps):
        perm = rng.permutation(m)
        gold_idx = perm[:gold_size]
        proxy_order = perm[gold_size:]
        if len(proxy_order) < min(k_list):
            continue
        G = X[:, gold_idx].mean(axis=1)
        for k in k_list:
            if k > len(proxy_order):
                continue
            P_idx = proxy_order[:k]
            R = X[:, P_idx].mean(axis=1)
            if len(G) < min_items:
                continue
            out.append(
                InterventionPath(
                    group_id=str(df_group["group_id"].iloc[0]),
                    rep_id=rep,
                    k_proxy=k,
                    proxy_raters=[raters[i] for i in P_idx],
                    gold_raters=[raters[i] for i in gold_idx],
                    n_items=int(len(G)),
                    J=np.asarray(G, dtype=float),
                    r=np.asarray(R, dtype=float),
                    icc_a1_full=icc1,
                )
            )
    return out
