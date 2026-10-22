from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PathData:
    group_key: str
    path_id: str
    item_ids: np.ndarray
    proxy_scores: np.ndarray
    gold_scores: np.ndarray
    meta: Dict[str, object]


def parse_cols(text: str | None) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def stable_group_key(values: Sequence[object]) -> str:
    return "__".join(str(v) for v in values)


def ensure_item_id(df: pd.DataFrame, item_cols: Sequence[str], out_col: str = "item_id") -> pd.DataFrame:
    if out_col in df.columns:
        return df.copy()
    if not item_cols:
        raise ValueError("Need item_cols if item_id is not already present.")
    out = df.copy()
    out[out_col] = out[item_cols].astype(str).agg("||".join, axis=1)
    return out


def pop_var(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        m = float(np.mean(x))
        return float(np.mean((x - m) ** 2))
    w = np.asarray(w, dtype=float)
    m = float(np.sum(w * x))
    return float(np.sum(w * (x - m) ** 2))


def pop_cov(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        mx = float(np.mean(x))
        my = float(np.mean(y))
        return float(np.mean((x - mx) * (y - my)))
    w = np.asarray(w, dtype=float)
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    return float(np.sum(w * (x - mx) * (y - my)))


def third_central_moment(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        m = float(np.mean(x))
        return float(np.mean((x - m) ** 3))
    w = np.asarray(w, dtype=float)
    m = float(np.sum(w * x))
    return float(np.sum(w * (x - m) ** 3))


def mixed_kappa_jrr(j: np.ndarray, r: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        jm = float(np.mean(j))
        rm = float(np.mean(r))
        return float(np.mean((j - jm) * (r - rm) ** 2))
    w = np.asarray(w, dtype=float)
    jm = float(np.sum(w * j))
    rm = float(np.sum(w * r))
    return float(np.sum(w * (j - jm) * (r - rm) ** 2))


def fourth_cumulant_rrrr(r: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    var = pop_var(r, w)
    if w is None:
        m = float(np.mean(r))
        mu4 = float(np.mean((r - m) ** 4))
    else:
        w = np.asarray(w, dtype=float)
        m = float(np.sum(w * r))
        mu4 = float(np.sum(w * (r - m) ** 4))
    return mu4 - 3.0 * (var ** 2)


def fourth_cumulant_jrrr(j: np.ndarray, r: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    cov = pop_cov(j, r, w)
    var = pop_var(r, w)
    if w is None:
        jm = float(np.mean(j))
        rm = float(np.mean(r))
        mu = float(np.mean((j - jm) * (r - rm) ** 3))
    else:
        w = np.asarray(w, dtype=float)
        jm = float(np.sum(w * j))
        rm = float(np.sum(w * r))
        mu = float(np.sum(w * (j - jm) * (r - rm) ** 3))
    return mu - 3.0 * cov * var


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xs = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ys = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    xv = xs - xs.mean()
    yv = ys - ys.mean()
    denom = math.sqrt(float(np.sum(xv**2) * np.sum(yv**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)


def log_mean_exp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.mean(np.exp(x - m))))


def tilt_weights(r: np.ndarray, t: float) -> np.ndarray:
    a = t * r
    m = float(np.max(a))
    w = np.exp(a - m)
    w /= float(np.sum(w))
    return w


def kl_to_uniform(r: np.ndarray, t: float) -> float:
    w = tilt_weights(r, t)
    er = float(np.sum(w * r))
    return t * er - log_mean_exp(t * r)


def find_t_for_dmax(r: np.ndarray, d_max: float, t_hi_init: float = 1.0) -> float:
    if d_max <= 0:
        return 0.0
    lo, hi = 0.0, max(t_hi_init, 1e-6)
    while kl_to_uniform(r, hi) < d_max:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket t for requested d_max.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kl_to_uniform(r, mid) < d_max:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class TiltCurve:
    t: np.ndarray
    d: np.ndarray
    q: np.ndarray
    mu_r: np.ndarray
    var_r: np.ndarray
    ess: np.ndarray


def exact_tilt_curve(
    r: np.ndarray,
    j: np.ndarray,
    d_max: float = 0.10,
    n_grid: int = 200,
) -> TiltCurve:
    t_hi = find_t_for_dmax(r, d_max)
    t_grid = np.linspace(0.0, t_hi, n_grid)
    d_vals = np.zeros_like(t_grid)
    q_vals = np.zeros_like(t_grid)
    mu_r_vals = np.zeros_like(t_grid)
    var_r_vals = np.zeros_like(t_grid)
    ess_vals = np.zeros_like(t_grid)
    for i, t in enumerate(t_grid):
        w = tilt_weights(r, float(t))
        mu_r = float(np.sum(w * r))
        q = float(np.sum(w * j))
        d_vals[i] = float(t) * mu_r - log_mean_exp(float(t) * r)
        q_vals[i] = q
        mu_r_vals[i] = mu_r
        var_r_vals[i] = pop_var(r, w)
        ess_vals[i] = 1.0 / float(np.sum(w**2))
    return TiltCurve(t=t_grid, d=d_vals, q=q_vals, mu_r=mu_r_vals, var_r=var_r_vals, ess=ess_vals)


@dataclass
class FitResult:
    a_fit: float
    b_fit: float
    r2: float
    d_fit: np.ndarray
    y_fit: np.ndarray


def fit_local_law(d: np.ndarray, q: np.ndarray, fit_max_d: float = 0.10) -> FitResult:
    d = np.asarray(d, dtype=float)
    q = np.asarray(q, dtype=float)
    y = q - float(q[0])
    mask = (d > 0) & (d <= fit_max_d)
    d_fit = d[mask]
    y_fit = y[mask]
    if len(d_fit) < 3:
        raise ValueError("Need at least 3 positive d points in fit window.")
    X = np.column_stack([np.sqrt(d_fit), d_fit])
    coef, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
    y_pred = X @ coef
    sse = float(np.sum((y_fit - y_pred) ** 2))
    sst = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    r2 = 1.0 if sst == 0 else 1.0 - sse / sst
    return FitResult(a_fit=float(coef[0]), b_fit=float(coef[1]), r2=r2, d_fit=d_fit, y_fit=y_fit)


@dataclass
class TheoryCoefficients:
    a_exact: float
    b_theory: float
    a_nu: float
    sigma_j2: float
    sigma_r2: float
    rho: float
    kappa_r: float
    kappa_jrr: float


def theory_coefficients(j: np.ndarray, r: np.ndarray) -> TheoryCoefficients:
    j = np.asarray(j, dtype=float)
    r = np.asarray(r, dtype=float)
    eps = r - j
    sigma_j2 = pop_var(j)
    sigma_r2 = pop_var(r)
    sigma_j = math.sqrt(max(sigma_j2, 0.0))
    sigma_r = math.sqrt(max(sigma_r2, 0.0))
    if sigma_r == 0.0:
        raise ValueError("Proxy variance is zero.")
    rho = pop_cov(j, eps)
    kappa_r = third_central_moment(r)
    kappa_jrr = mixed_kappa_jrr(j, r)
    a_exact = math.sqrt(2.0) * (sigma_j2 + rho) / sigma_r
    a_nu = math.sqrt(2.0) * sigma_j * (sigma_j / sigma_r)
    b_theory = kappa_jrr / sigma_r2 - 2.0 * (sigma_j2 + rho) * kappa_r / (3.0 * sigma_r2**2)
    return TheoryCoefficients(
        a_exact=a_exact,
        b_theory=b_theory,
        a_nu=a_nu,
        sigma_j2=sigma_j2,
        sigma_r2=sigma_r2,
        rho=rho,
        kappa_r=kappa_r,
        kappa_jrr=kappa_jrr,
    )


def delta_at_d(curve: TiltCurve, d_target: float) -> float:
    if d_target < 0 or d_target > float(curve.d[-1]):
        return float("nan")
    q0 = float(curve.q[0])
    q_target = float(np.interp(d_target, curve.d, curve.q))
    return q_target - q0


def local_peak_from_coefficients(a: float, b: float) -> Tuple[float, float]:
    if a <= 0 or b >= 0:
        return float("nan"), float("nan")
    c = -b
    d_peak = a * a / (4.0 * c * c)
    headroom = a * a / (4.0 * c)
    return d_peak, headroom


def complete_groups_from_long(
    df: pd.DataFrame,
    item_cols: Sequence[str],
    rater_col: str,
    score_col: str,
    group_cols: Sequence[str],
    min_items: int = 200,
    min_raters: int = 3,
) -> Dict[str, pd.DataFrame]:
    df = ensure_item_id(df, item_cols)
    out: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []
    grouped = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]
    for gkey, gdf in grouped:
        gkey_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
        mat_raw = gdf.pivot_table(index="item_id", columns=rater_col, values=score_col, aggfunc="mean")
        n_dropped = int(mat_raw.isna().any(axis=1).sum())
        mat = mat_raw.dropna(axis=0, how="any").sort_index()
        key = stable_group_key(gkey_tuple)
        if mat.shape[0] < min_items or mat.shape[1] < min_raters:
            skipped.append(f"{key}: {mat.shape[0]} items, {mat.shape[1]} raters (need >={min_items} items, >={min_raters} raters)")
            continue
        if n_dropped > 0:
            print(f"[{key}] dropna removed {n_dropped}/{len(mat_raw)} items with incomplete ratings")
        out[stable_group_key(gkey_tuple)] = mat
    if skipped:
        print(f"complete_groups_from_long: skipped {len(skipped)} groups below threshold:")
        for s in skipped:
            print(f"  {s}")
    return out


def generate_human_paths(
    df: pd.DataFrame,
    item_cols: Sequence[str],
    rater_col: str,
    score_col: str,
    group_cols: Sequence[str],
    min_items: int = 200,
    min_raters: int = 3,
) -> Iterator[PathData]:
    groups = complete_groups_from_long(df, item_cols, rater_col, score_col, group_cols, min_items, min_raters)
    for group_key, mat in groups.items():
        arr = mat.to_numpy(dtype=float)
        item_ids = mat.index.to_numpy(dtype=str)
        raters = list(mat.columns)
        for idx, rater in enumerate(raters):
            proxy = arr[:, idx].astype(float)
            gold = np.mean(np.delete(arr, idx, axis=1), axis=1)
            yield PathData(
                group_key=group_key,
                path_id=f"{group_key}__heldout_{rater}",
                item_ids=item_ids,
                proxy_scores=proxy,
                gold_scores=gold,
                meta={"proxy_source": "heldout_human", "heldout_rater": rater, "n_items": len(item_ids)},
            )


def generate_model_paths(
    human_long: pd.DataFrame,
    proxy_df: pd.DataFrame,
    item_cols: Sequence[str],
    score_col: str,
    group_cols: Sequence[str],
    proxy_item_col: str = "item_id",
    proxy_score_col: str = "proxy_score",
    min_items: int = 200,
) -> Iterator[PathData]:
    human_long = ensure_item_id(human_long, item_cols)
    gold_df = human_long.groupby([*group_cols, "item_id"], dropna=False)[score_col].mean().reset_index(name="gold_score") if group_cols else human_long.groupby(["item_id"], dropna=False)[score_col].mean().reset_index(name="gold_score")
    proxy_df = proxy_df.copy()
    if proxy_item_col != "item_id":
        proxy_df = proxy_df.rename(columns={proxy_item_col: "item_id"})
    if proxy_score_col != "proxy_score":
        proxy_df = proxy_df.rename(columns={proxy_score_col: "proxy_score"})
    merge_cols = ["item_id", *group_cols]
    proxy_dedup = proxy_df[merge_cols + ["proxy_score"]].drop_duplicates()
    n_proxy_dedup = len(proxy_df) - len(proxy_dedup)
    if n_proxy_dedup > 0:
        print(f"generate_model_paths: dropped {n_proxy_dedup} duplicate proxy rows")
    conflicting = (
        proxy_dedup.groupby(merge_cols, dropna=False)["proxy_score"]
        .nunique(dropna=False)
        .reset_index(name="n_proxy_scores")
    )
    conflicting = conflicting[conflicting["n_proxy_scores"] > 1]
    if not conflicting.empty:
        examples = conflicting.head(5)[merge_cols].to_dict(orient="records")
        raise ValueError(
            "Proxy table is not item-level unique after exact duplicate removal: "
            f"{len(conflicting)} keys have multiple proxy_score values. "
            "Pass an already aggregated proxy CSV or include additional grouping columns. "
            f"Example keys: {examples}"
        )
    proxy_unique = proxy_dedup.drop_duplicates(subset=merge_cols)
    merged = gold_df.merge(proxy_unique, on=merge_cols, how="inner")
    n_unmatched = len(gold_df) - len(merged)
    if n_unmatched > 0:
        print(f"generate_model_paths: {n_unmatched}/{len(gold_df)} gold items had no matching proxy (inner join)")
    grouped_items = list(merged.groupby(list(group_cols), dropna=False)) if group_cols else [((), merged)]
    skipped_groups = [
        f"{stable_group_key(gk if isinstance(gk, tuple) else (gk,))}: {len(gd)} items (need >={min_items})"
        for gk, gd in grouped_items if len(gd) < min_items
    ]
    if skipped_groups:
        print(f"generate_model_paths: skipped {len(skipped_groups)} groups below min_items={min_items}:")
        for s in skipped_groups:
            print(f"  {s}")
    for gkey, gdf in grouped_items:
        gkey_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
        gdf = gdf.sort_values("item_id")
        if len(gdf) < min_items:
            continue
        yield PathData(
            group_key=stable_group_key(gkey_tuple),
            path_id=f"{stable_group_key(gkey_tuple)}__model_proxy",
            item_ids=gdf["item_id"].to_numpy(dtype=str),
            proxy_scores=gdf["proxy_score"].to_numpy(dtype=float),
            gold_scores=gdf["gold_score"].to_numpy(dtype=float),
            meta={"proxy_source": "model_proxy", "n_items": len(gdf)},
        )


def split_indices(n: int, calibration_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < calibration_frac < 1.0):
        raise ValueError("calibration_frac must be in (0,1).")
    idx = np.arange(n)
    rng.shuffle(idx)
    n_cal = max(10, int(round(calibration_frac * n)))
    n_cal = min(n_cal, n - 10)
    cal = np.sort(idx[:n_cal])
    eva = np.sort(idx[n_cal:])
    return cal, eva


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
