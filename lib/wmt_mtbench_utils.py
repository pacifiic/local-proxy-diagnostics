#!/usr/bin/env python3
from __future__ import annotations

import io
import math
import os
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------
# Core local-law utilities
# -----------------------------

def compute_ab(J: np.ndarray, r: np.ndarray) -> dict[str, float]:
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float)
    Jc = J - J.mean()
    rc = r - r.mean()
    sigma_J_sq = float(np.mean(Jc ** 2))
    sigma_r_sq = float(np.mean(rc ** 2))
    sigma_r_sq = max(sigma_r_sq, 1e-18)
    sigma_r = float(np.sqrt(sigma_r_sq))
    eps = r - J
    rho = float(np.mean(Jc * (eps - eps.mean())))
    kappa_r = float(np.mean(rc ** 3))
    kappa_Jrr = float(np.mean(Jc * (rc ** 2)))
    A = float(np.sqrt(2.0) * (sigma_J_sq + rho) / sigma_r)
    B = float(
        kappa_Jrr / sigma_r_sq
        - 2.0 * (sigma_J_sq + rho) * kappa_r / (3.0 * sigma_r_sq ** 2)
    )
    return {
        "A": A,
        "B": B,
        "rho": rho,
        "sigma_J_sq": sigma_J_sq,
        "sigma_r_sq": sigma_r_sq,
        "kappa_r": kappa_r,
        "kappa_Jrr": kappa_Jrr,
    }


def tilt_curve(J: np.ndarray, r: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def fit_local_ab(d: np.ndarray, q: np.ndarray, max_d: float) -> tuple[float, float, float]:
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


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = float(np.std(x))
    if s <= 1e-12:
        return x - x.mean()
    return (x - x.mean()) / s


# -----------------------------
# ICC utilities
# -----------------------------

def icc1_from_balanced_scores(X: np.ndarray) -> float:
    """One-way random single-rater ICC(1,1) for balanced n_items x k arrays."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan
    row_means = X.mean(axis=1)
    grand = row_means.mean()
    ms_between = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_within = np.sum((X - row_means[:, None]) ** 2) / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom != 0 else np.nan


def icc1_from_long(long_df: pd.DataFrame, item_col: str, score_col: str) -> tuple[float, int, int]:
    """Compute ICC(1,1) using the modal number of ratings per item.

    Returns (icc, n_items_used, k_modal).
    """
    counts = long_df.groupby(item_col)[score_col].size()
    if counts.empty:
        return np.nan, 0, 0
    k_modal = Counter(counts.tolist()).most_common(1)[0][0]
    if k_modal < 2:
        return np.nan, 0, k_modal
    eligible = counts[counts == k_modal].index
    sub = long_df[long_df[item_col].isin(eligible)].copy()
    # Arbitrary within-item order is fine for ICC(1,1) because raters are exchangeable.
    sub["_ord"] = sub.groupby(item_col).cumcount()
    X = sub.pivot(index=item_col, columns="_ord", values=score_col).dropna(axis=0, how="any")
    if X.empty:
        return np.nan, 0, k_modal
    return icc1_from_balanced_scores(X.to_numpy(dtype=float)), int(X.shape[0]), int(k_modal)


# -----------------------------
# WMT helpers
# -----------------------------

def _severity_weight(category: str, severity: str) -> float:
    cat = str(category).strip().lower()
    sev = str(severity).strip().lower()
    if sev in {"no-error", "no_error", "no error", "neutral"}:
        return 0.0
    if "non-translation" in cat or cat == "nontranslation":
        return 25.0
    if sev == "major":
        return 5.0
    if sev == "minor":
        if "punctuation" in cat:
            return 0.1
        return 1.0
    # conservative default for unexpected labels
    return 0.0


def discover_wmt_files(root: str | Path, mode: str = "mqm") -> list[Path]:
    root = Path(root)
    if mode == "mqm":
        files = []
        for p in root.rglob("*.tsv"):
            name = p.name.lower()
            if not name.startswith("mqm_"):
                continue
            if ".avg_seg_scores" in name or ".source-errors" in name or ".no-postedits" in name:
                continue
            files.append(p)
        for p in root.rglob("*.tsv.tar.gz"):
            name = p.name.lower()
            if "mqm_" in name and "3ratingspersegment" in name:
                files.append(p)
        return sorted(files)
    if mode == "psqm":
        return sorted(root.rglob("psqm_*.tsv"))
    raise ValueError(f"Unknown mode: {mode}")


def _robust_tsv_from_text(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].rstrip("\n\r").split("\t")
    rows = []
    n = len(header)
    for line in lines[1:]:
        if not line:
            continue
        parts = line.rstrip("\n\r").split("\t")
        if len(parts) < n:
            parts = parts + [""] * (n - len(parts))
        elif len(parts) > n:
            parts = parts[: n - 1] + ["\t".join(parts[n - 1 :])]
        rows.append(parts)
    return pd.DataFrame(rows, columns=header)


def _read_tsv_or_tar_tsv(path: Path) -> pd.DataFrame:
    if path.suffixes[-3:] == [".tsv", ".tar", ".gz"] or str(path).endswith(".tsv.tar.gz"):
        with tarfile.open(path, mode="r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith(".tsv")), None)
            if member is None:
                raise ValueError(f"No TSV found inside {path}")
            f = tf.extractfile(member)
            assert f is not None
            return _robust_tsv_from_text(f.read().decode("utf-8", "ignore"))
    return _robust_tsv_from_text(path.read_text(encoding="utf-8", errors="ignore"))




def _read_psqm_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def dataset_id_from_path(path: Path, mode: str) -> str:
    challenge = path.parent.parent.name
    lang = path.parent.name
    name = path.name.lower()
    tag = "raw"
    if "3ratingspersegment" in name:
        tag = "3r"
    if "sxs" in name:
        tag = f"sxs_{tag}"
    return f"{challenge}__{lang}__{mode}__{tag}"


def canonicalize_wmt_mqm(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "system_id": "system",
        "document_id": "doc",
        "doc_segment_id": "seg_id",
        "docSegId": "seg_id",
        "globalSegId": "global_seg_id",
        "rater_id": "rater",
        "candidate": "target",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    needed = {"system", "doc", "seg_id", "rater", "category", "severity"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing MQM columns in {dataset_id}: {sorted(missing)}")
    # Drop documentation column if present.
    doc_cols = [c for c in df.columns if c.startswith("# Documentation")]
    if doc_cols:
        df = df.drop(columns=doc_cols)
    df["penalty"] = [
        _severity_weight(c, s) for c, s in zip(df["category"], df["severity"], strict=False)
    ]
    agg = (
        df.groupby(["system", "doc", "seg_id", "rater"], as_index=False)["penalty"]
        .sum()
        .rename(columns={"penalty": "total_penalty"})
    )
    agg["score"] = -agg["total_penalty"]
    agg["dataset_id"] = dataset_id
    agg["item_id"] = (
        agg["dataset_id"].astype(str)
        + "___"
        + agg["doc"].astype(str)
        + "___"
        + agg["seg_id"].astype(str)
        + "___"
        + agg["system"].astype(str)
    )
    return agg[["dataset_id", "item_id", "system", "doc", "seg_id", "rater", "score"]]


def canonicalize_wmt_psqm(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    needed = {"system", "doc", "seg_id", "rater", "score"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing pSQM columns in {dataset_id}: {sorted(missing)}")
    out = df[["system", "doc", "seg_id", "rater", "score"]].copy()
    out["dataset_id"] = dataset_id
    out["item_id"] = (
        out["dataset_id"].astype(str)
        + "___"
        + out["doc"].astype(str)
        + "___"
        + out["seg_id"].astype(str)
        + "___"
        + out["system"].astype(str)
    )
    return out[["dataset_id", "item_id", "system", "doc", "seg_id", "rater", "score"]]


def load_wmt_long(root: str | Path, mode: str = "mqm") -> pd.DataFrame:
    rows = []
    for path in discover_wmt_files(root, mode=mode):
        dataset_id = dataset_id_from_path(path, mode)
        if mode == "mqm":
            raw = _read_tsv_or_tar_tsv(path)
            can = canonicalize_wmt_mqm(raw, dataset_id)
        else:
            raw = _read_psqm_tsv(path)
            can = canonicalize_wmt_psqm(raw, dataset_id)
        rows.append(can)
    if not rows:
        raise FileNotFoundError(f"No {mode} files found under {root}")
    return pd.concat(rows, ignore_index=True)


def filter_ref_like_systems(df: pd.DataFrame) -> pd.DataFrame:
    pat = r"(^ref\b|^ref[.-]|^human\b|^human[.-]|\bref\b|\bhuman\b)"
    return df[~df["system"].astype(str).str.lower().str.contains(pat, regex=True, na=False)].copy()


def build_heldout_paths_from_long(
    long_df: pd.DataFrame,
    dataset_id: str,
    *,
    gold_estimator: str = "loo_mean",
    score_transform: str = "raw",
    min_items: int = 100,
    min_other_raters: int = 2,
) -> list[dict[str, object]]:
    sub = long_df[long_df["dataset_id"] == dataset_id].copy()
    if sub.empty:
        return []
    if score_transform == "rater_z":
        sub["score"] = sub.groupby("rater")["score"].transform(zscore)
    item_counts = sub.groupby("item_id")["rater"].nunique().rename("n_raters")
    sub = sub.merge(item_counts, left_on="item_id", right_index=True, how="left")
    raters = sorted(sub["rater"].astype(str).unique().tolist())
    out: list[dict[str, object]] = []
    for rater in raters:
        my = sub[sub["rater"].astype(str) == rater][["item_id", "score"]].rename(columns={"score": "r"})
        others = sub[sub["rater"].astype(str) != rater]
        stats = others.groupby("item_id")["score"].agg(["mean", "median", "count"]).reset_index()
        if gold_estimator == "loo_mean":
            stats = stats.rename(columns={"mean": "J", "count": "n_others"})
        elif gold_estimator == "median_others":
            stats = stats.rename(columns={"median": "J", "count": "n_others"})
        else:
            raise ValueError(f"Unsupported gold estimator: {gold_estimator}")
        merged = my.merge(stats[["item_id", "J", "n_others"]], on="item_id", how="inner")
        merged = merged[merged["n_others"] >= min_other_raters]
        if merged.shape[0] < min_items:
            continue
        J = merged["J"].to_numpy(dtype=float)
        r = merged["r"].to_numpy(dtype=float)
        out.append({
            "dataset_id": dataset_id,
            "heldout_rater": str(rater),
            "n_items": int(len(merged)),
            "J": J,
            "r": r,
        })
    return out


# -----------------------------
# MT-Bench / Bradley-Terry helpers
# -----------------------------

def read_mtbench_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # JSONL support
    if path.suffix in {".jsonl", ".json"}:
        import json
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        # keep only human split if column exists
        if "split" in df.columns:
            df = df[df["split"] == "human"].reset_index(drop=True)
        return df
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover - depends on local parquet engine
        raise RuntimeError(
            "Could not read parquet. Install pyarrow or fastparquet locally, then rerun."
        ) from e


def parse_winner(label: object, model_a: str, model_b: str) -> tuple[float, float] | None:
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    s = str(label).strip().lower()
    ma = str(model_a).strip().lower()
    mb = str(model_b).strip().lower()
    if s in {"model_a", "a", "left", "1", ma} or s.endswith(ma):
        return 1.0, 0.0
    if s in {"model_b", "b", "right", "2", mb} or s.endswith(mb):
        return 0.0, 1.0
    if "tie" in s or "draw" in s or s in {"both", "equal"}:
        return 0.5, 0.5
    return None


def fit_bt_mm(
    pair_df: pd.DataFrame,
    model_a_col: str = "model_a",
    model_b_col: str = "model_b",
    score_a_col: str = "score_a",
    score_b_col: str = "score_b",
    prior: float = 1e-2,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> dict[str, float]:
    models = sorted(set(pair_df[model_a_col].astype(str)).union(set(pair_df[model_b_col].astype(str))))
    m_to_i = {m: i for i, m in enumerate(models)}
    K = len(models)
    W = np.zeros((K, K), dtype=float)
    N = np.zeros((K, K), dtype=float)
    for row in pair_df.itertuples(index=False):
        a = getattr(row, model_a_col)
        b = getattr(row, model_b_col)
        sa = float(getattr(row, score_a_col))
        sb = float(getattr(row, score_b_col))
        i, j = m_to_i[str(a)], m_to_i[str(b)]
        W[i, j] += sa
        W[j, i] += sb
        N[i, j] += sa + sb
        N[j, i] += sa + sb
    # tiny symmetric prior to ensure connectivity / shrinkage
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            W[i, j] += prior
            N[i, j] += 2 * prior
    lam = np.ones(K, dtype=float)
    for _ in range(max_iter):
        old = lam.copy()
        wins = W.sum(axis=1)
        denom = np.zeros(K, dtype=float)
        for i in range(K):
            s = 0.0
            for j in range(K):
                if i == j:
                    continue
                nij = N[i, j]
                if nij <= 0:
                    continue
                s += nij / max(lam[i] + lam[j], 1e-12)
            denom[i] = s
        lam = wins / np.maximum(denom, 1e-12)
        lam = np.maximum(lam, 1e-12)
        lam /= np.exp(np.mean(np.log(lam)))  # fix scale
        if np.max(np.abs(np.log(lam) - np.log(old))) < tol:
            break
    util = np.log(lam)
    util -= util.mean()
    return {m: float(util[m_to_i[m]]) for m in models}


_MTBENCH_GROUPS = None
_MTBENCH_STANDARDIZE = None
_MTBENCH_MIN_ROWS_PER_SPLIT_GROUP = None


def _init_mtbench_split_worker(groups, standardize_within_group, min_rows_per_split_group):
    global _MTBENCH_GROUPS, _MTBENCH_STANDARDIZE, _MTBENCH_MIN_ROWS_PER_SPLIT_GROUP
    _MTBENCH_GROUPS = groups
    _MTBENCH_STANDARDIZE = standardize_within_group
    _MTBENCH_MIN_ROWS_PER_SPLIT_GROUP = min_rows_per_split_group


def _mtbench_one_split(task):
    split_id, perms = task
    pooled = []
    n_groups_used = 0
    for (qid, turn, g, _judges), perm in zip(_MTBENCH_GROUPS, perms):
        half = len(perm) // 2
        if half < 2:
            continue
        judges_a = set(perm[:half])
        judges_b = set(perm[half:])
        ga = g[g["judge"].isin(judges_a)]
        gb = g[g["judge"].isin(judges_b)]
        if len(ga) < _MTBENCH_MIN_ROWS_PER_SPLIT_GROUP or len(gb) < _MTBENCH_MIN_ROWS_PER_SPLIT_GROUP:
            continue
        ua = fit_bt_mm(ga)
        ub = fit_bt_mm(gb)
        models = sorted(set(ua).intersection(set(ub)))
        if len(models) < 4:
            continue
        Ja = np.array([ua[m] for m in models], dtype=float)
        rb = np.array([ub[m] for m in models], dtype=float)
        if _MTBENCH_STANDARDIZE:
            Ja = zscore(Ja)
            rb = zscore(rb)
        pooled.append(pd.DataFrame({
            "split_id": split_id,
            "question_id": qid,
            "turn": turn,
            "model": models,
            "J": Ja,
            "r": rb,
        }))
        n_groups_used += 1
    if not pooled:
        return None
    pooled_df = pd.concat(pooled, ignore_index=True)
    J = pooled_df["J"].to_numpy(dtype=float)
    r = pooled_df["r"].to_numpy(dtype=float)
    d = compute_ab(J, r)
    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, 4.0, 400)])
    dvals, qvals = tilt_curve(J, r, t_grid)
    A_fit, B_fit, r2 = fit_local_ab(dvals, qvals, max_d=0.10)
    summary = {
        "split_id": split_id,
        "n_items": int(len(pooled_df)),
        "n_groups_used": int(n_groups_used),
        "A_theory": d["A"],
        "B_theory": d["B"],
        "A_fit": A_fit,
        "B_fit": B_fit,
        "r2": r2,
        "corr_J_r": float(np.corrcoef(J, r)[0, 1]) if len(J) > 1 else np.nan,
        "frac_B_theory_neg": float(d["B"] < 0),
        "frac_B_fit_neg": float(B_fit < 0) if not np.isnan(B_fit) else np.nan,
    }
    return pooled_df, summary


def split_half_bt_items(
    df: pd.DataFrame,
    *,
    n_splits: int = 100,
    random_state: int = 0,
    standardize_within_group: bool = True,
    min_judges_per_group: int = 4,
    min_rows_per_split_group: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    required = {"question_id", "model_a", "model_b", "winner", "judge", "turn"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"MT-Bench data missing columns: {sorted(missing)}")

    pair_rows = []
    for row in df.itertuples(index=False):
        parsed = parse_winner(getattr(row, "winner"), getattr(row, "model_a"), getattr(row, "model_b"))
        if parsed is None:
            continue
        sa, sb = parsed
        pair_rows.append({
            "question_id": getattr(row, "question_id"),
            "turn": getattr(row, "turn"),
            "judge": str(getattr(row, "judge")),
            "model_a": str(getattr(row, "model_a")),
            "model_b": str(getattr(row, "model_b")),
            "score_a": sa,
            "score_b": sb,
        })
    pair_df = pd.DataFrame(pair_rows)
    groups = []
    for (qid, turn), g in pair_df.groupby(["question_id", "turn"]):
        judges = sorted(g["judge"].unique().tolist())
        if len(judges) < min_judges_per_group:
            continue
        groups.append((qid, turn, g.reset_index(drop=True), judges))

    tasks = []
    for split_id in range(n_splits):
        perms = [rng.permutation(judges).tolist() for _qid, _turn, _g, judges in groups]
        tasks.append((split_id, perms))

    n_jobs = int(os.environ.get("MTBENCH_BT_JOBS", "1"))
    if n_jobs > 1 and n_splits > 1:
        chunksize = max(1, n_splits // (n_jobs * 4))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_mtbench_split_worker,
            initargs=(groups, standardize_within_group, min_rows_per_split_group),
        ) as pool:
            results = list(pool.map(_mtbench_one_split, tasks, chunksize=chunksize))
    else:
        _init_mtbench_split_worker(groups, standardize_within_group, min_rows_per_split_group)
        results = [_mtbench_one_split(task) for task in tasks]

    split_item_rows = []
    split_summary_rows = []
    for result in results:
        if result is None:
            continue
        pooled_df, summary = result
        split_item_rows.append(pooled_df)
        split_summary_rows.append(summary)
    items_df = pd.concat(split_item_rows, ignore_index=True) if split_item_rows else pd.DataFrame()
    summary_df = pd.DataFrame(split_summary_rows)
    return items_df, summary_df
