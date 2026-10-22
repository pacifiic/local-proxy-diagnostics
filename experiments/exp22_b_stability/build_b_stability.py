"""Measure coefficient stability, contribution, and prediction diagnostics.

Use the existing moment estimator. Report coefficient bootstrap uncertainty,
the A*sqrt(d) and B*d contributions, and calibration/evaluation coefficient
mismatches. The latter are termwise discrepancies, not causal error attribution.
Exploratory decision summaries use gamma=0.1*sigma_J_cal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = HERE / "outputs"

sys.path.insert(0, str(REPO_ROOT / "lib"))
from common_experiment_utils import (  # noqa: E402
    generate_human_paths,
    generate_model_paths,
    parse_cols,
    split_indices,
    theory_coefficients,
)

D_GRID = [0.01, 0.025, 0.05, 0.075, 0.10]
D_VALID = 0.10
CALIBRATION_FRAC = 0.30
N_SPLIT_REPEATS = 50     # outer: fresh cal/eval item splits (captures split-to-split variability)
N_ITEM_BOOT = 300        # inner: prompt-cluster bootstrap WITHIN one calibration draw (SE/CI)
GAMMA_FRACTION_OF_SIGMA_J = 0.1
RANDOM_SEED = 0


CONFIGS = [
    dict(
        dataset_name="SummEval-human_heldout",
        human_long_csv=str(DATA_DIR / "summeval_long.csv"),
        item_cols="article_id,model_id",
        group_cols="dimension,rater_type",
        rater_col="rater_idx",
        score_col="score",
        proxy_csv="",
        min_items=200,
        min_raters=3,
    ),
    dict(
        dataset_name="SummEval-GPT4o",
        human_long_csv=str(DATA_DIR / "summeval_long.csv"),
        item_cols="article_id,model_id",
        group_cols="dimension",
        rater_col="rater_idx",
        score_col="score",
        proxy_csv=str(DATA_DIR / "model_proxy_4o_aggregated.csv"),
        proxy_item_col="item_id",
        proxy_score_col="proxy_score_mean",
        min_items=200,
        min_raters=3,
    ),
    dict(
        dataset_name="WMT-MQM-human_heldout",
        human_long_csv=str(DATA_DIR / "wmt_mqm_long.csv"),
        item_cols="item_id",
        group_cols="dataset_id",
        rater_col="rater",
        score_col="score",
        proxy_csv="",
        min_items=200,
        min_raters=3,
    ),
    dict(
        dataset_name="OpenMEVA-MAGS-human_heldout",
        human_long_csv=str(DATA_DIR / "openmeva_long.csv"),
        item_cols="story_id,model_id",
        group_cols="dataset",
        rater_col="rater_idx",
        score_col="score",
        proxy_csv="",
        min_items=100,
        min_raters=3,
    ),
]


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def build_paths(cfg: dict):
    human = pd.read_csv(cfg["human_long_csv"])
    item_cols = parse_cols(cfg["item_cols"])
    group_cols = parse_cols(cfg["group_cols"])
    if cfg.get("proxy_csv"):
        proxy = pd.read_csv(cfg["proxy_csv"])
        proxy = (
            proxy[[*group_cols, cfg["proxy_item_col"], cfg["proxy_score_col"]]]
            .groupby([*group_cols, cfg["proxy_item_col"]], dropna=False)[cfg["proxy_score_col"]]
            .mean()
            .reset_index()
        )
        return list(generate_model_paths(
            human_long=human, proxy_df=proxy, item_cols=item_cols, score_col=cfg["score_col"],
            group_cols=group_cols, proxy_item_col=cfg["proxy_item_col"], proxy_score_col=cfg["proxy_score_col"],
            min_items=cfg["min_items"],
        ))
    return list(generate_human_paths(
        df=human, item_cols=item_cols, rater_col=cfg["rater_col"], score_col=cfg["score_col"],
        group_cols=group_cols, min_items=cfg["min_items"], min_raters=cfg["min_raters"],
    ))


def prompt_cluster_ids(dataset_name: str, item_ids: np.ndarray) -> np.ndarray:
    """Recover the shared prompt/article/story unit encoded in canonical item IDs."""
    ids = np.asarray(item_ids, dtype=str)
    if dataset_name.startswith("WMT"):
        return np.array([x.split("___", 1)[1] if "___" in x else x for x in ids])
    if dataset_name.startswith(("SummEval", "OpenMEVA")):
        return np.array([x.split("||", 1)[0] if "||" in x else x for x in ids])
    return ids.copy()


def cluster_bootstrap_ab(
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
):
    """Resample prompt clusters, retaining every calibration item in each cluster."""
    n = len(r_cal)
    if len(j_cal) != n or len(cluster_ids) != n:
        raise ValueError("r_cal, j_cal, and cluster_ids must have identical lengths")
    clusters = np.unique(cluster_ids)
    members = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    a_boot = np.empty(n_boot)
    b_boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([members[cluster] for cluster in sampled])
        if len(np.unique(r_cal[idx])) < 2:
            a_boot[i] = np.nan
            b_boot[i] = np.nan
            continue
        try:
            c = theory_coefficients(j_cal[idx], r_cal[idx])
            a_boot[i] = c.a_exact
            b_boot[i] = c.b_theory
        except (ValueError, ZeroDivisionError):
            a_boot[i] = np.nan
            b_boot[i] = np.nan
    return a_boot, b_boot


def predicted_curve_ci(a_boot: np.ndarray, b_boot: np.ndarray, d_grid, alpha: float = 0.05):
    """Bootstrap LCB/UCB of Delta_pred(d) = A*sqrt(d) + B*d, propagating the joint
    (A,B) bootstrap draws (not independent per-coefficient CIs, which would ignore
    the negative A/B correlation the theorem implies)."""
    finite = np.isfinite(a_boot) & np.isfinite(b_boot)
    a_boot = a_boot[finite]
    b_boot = b_boot[finite]
    out = {}
    for d in d_grid:
        draws = a_boot * np.sqrt(d) + b_boot * d
        out[d] = {
            "lcb": float(np.percentile(draws, 100 * alpha / 2)) if len(draws) else float("nan"),
            "ucb": float(np.percentile(draws, 100 * (1 - alpha / 2))) if len(draws) else float("nan"),
            "mean": float(np.mean(draws)) if len(draws) else float("nan"),
        }
    return out


def run_config(cfg: dict) -> pd.DataFrame:
    paths = build_paths(cfg)
    rows = []
    for path in paths:
        n = len(path.item_ids)
        for repeat in range(N_SPLIT_REPEATS):
            split_rng = np.random.default_rng(
                stable_seed(RANDOM_SEED, cfg["dataset_name"], path.path_id, repeat, "split")
            )
            boot_rng = np.random.default_rng(
                stable_seed(RANDOM_SEED, cfg["dataset_name"], path.path_id, repeat, "bootstrap")
            )
            cal_idx, eval_idx = split_indices(n, CALIBRATION_FRAC, split_rng)
            r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
            r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
            j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
            if len(np.unique(r_cal)) < 2 or len(np.unique(r_eval)) < 2:
                continue

            coeff_cal = theory_coefficients(j_cal, r_cal)
            coeff_eval = theory_coefficients(j_eval, r_eval)
            sigma_j_cal = float(np.std(j_cal))
            gamma = GAMMA_FRACTION_OF_SIGMA_J * sigma_j_cal

            clusters_cal = prompt_cluster_ids(
                cfg["dataset_name"], np.asarray(path.item_ids)[cal_idx]
            )
            a_boot, b_boot = cluster_bootstrap_ab(
                r_cal, j_cal, clusters_cal, N_ITEM_BOOT, boot_rng
            )
            finite_boot = np.isfinite(a_boot) & np.isfinite(b_boot)
            a_boot_f = a_boot[finite_boot]
            b_boot_f = b_boot[finite_boot]
            curve_ci = predicted_curve_ci(a_boot, b_boot, D_GRID)
            aonly_ci = {d: {
                "lcb": float(np.percentile(a_boot_f * np.sqrt(d), 2.5)) if len(a_boot_f) else float("nan"),
            } for d in D_GRID}

            # Decision at d = D_VALID: proceed if LCB[Delta(d)] > gamma.
            d_ref = D_VALID
            proceed_ab = curve_ci[d_ref]["lcb"] > gamma
            proceed_aonly = aonly_ci[d_ref]["lcb"] > gamma
            b_ci_lo = float(np.percentile(b_boot_f, 2.5)) if len(b_boot_f) else float("nan")
            b_ci_hi = float(np.percentile(b_boot_f, 97.5)) if len(b_boot_f) else float("nan")
            b_ci_includes_zero = bool(b_ci_lo <= 0 <= b_ci_hi)

            # D_safe: largest d in the grid (<= D_VALID) where LCB[Delta(d)] > gamma.
            safe_ds = [d for d in D_GRID if d <= D_VALID and curve_ci[d]["lcb"] > gamma]
            d_safe = max(safe_ds) if safe_ds else 0.0

            for d in D_GRID:
                g_a = coeff_cal.a_exact * np.sqrt(d)
                g_b = coeff_cal.b_theory * d
                r_b = abs(g_b) / max(abs(g_a) + abs(g_b), 1e-12)
                e_a = np.sqrt(d) * abs(coeff_cal.a_exact - coeff_eval.a_exact)
                e_b = d * abs(coeff_cal.b_theory - coeff_eval.b_theory)
                rows.append({
                    "dataset": cfg["dataset_name"],
                    "group_key": path.group_key,
                    "path_id": path.path_id,
                    "repeat": repeat,
                    "d": d,
                    "n_cal": len(cal_idx),
                    "n_eval": len(eval_idx),
                    "a_cal": coeff_cal.a_exact,
                    "b_cal": coeff_cal.b_theory,
                    "a_eval": coeff_eval.a_exact,
                    "b_eval": coeff_eval.b_theory,
                    "a_boot_se": float(np.std(a_boot_f)) if len(a_boot_f) else float("nan"),
                    "b_boot_se": float(np.std(b_boot_f)) if len(b_boot_f) else float("nan"),
                    "b_ci_lo": b_ci_lo,
                    "b_ci_hi": b_ci_hi,
                    "b_ci_includes_zero": b_ci_includes_zero,
                    "b_sign_stability": float(np.mean(np.sign(b_boot_f) == np.sign(coeff_cal.b_theory))) if len(b_boot_f) else float("nan"),
                    "G_A": g_a,
                    "G_B": g_b,
                    "R_B": r_b,
                    "E_A": e_a,
                    "E_B": e_b,
                    "delta_pred_lcb": curve_ci[d]["lcb"],
                    "delta_pred_ucb": curve_ci[d]["ucb"],
                    "delta_pred_mean": curve_ci[d]["mean"],
                    "gamma": gamma,
                    "sigma_j_cal": sigma_j_cal,
                    "d_safe": d_safe,
                    "d_safe_grid": d_safe,
                    "n_cal_clusters": int(len(np.unique(clusters_cal))),
                    "proceed_ab_at_dvalid": proceed_ab,
                    "proceed_aonly_at_dvalid": proceed_aonly,
                    "decision_disagree_at_dvalid": proceed_ab != proceed_aonly,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for cfg in CONFIGS:
        print(f"=== {cfg['dataset_name']} ===")
        df = run_config(cfg)
        print(f"  rows: {len(df)}, paths: {df['path_id'].nunique() if not df.empty else 0}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "b_stability_rows.csv"
    combined.to_csv(out_path, index=False)
    print(f"-> {out_path} ({len(combined)} rows)")

    spec = {
        "d_grid": D_GRID, "d_valid": D_VALID, "calibration_frac": CALIBRATION_FRAC,
        "n_split_repeats": N_SPLIT_REPEATS, "n_item_boot": N_ITEM_BOOT,
        "bootstrap_unit": "prompt_cluster_within_calibration_split",
        "gamma_fraction_of_sigma_j": GAMMA_FRACTION_OF_SIGMA_J,
        "random_seed": RANDOM_SEED,
        "seed_policy": (
            "independent SHA-256-derived streams per dataset/path/repeat for "
            "outer split and inner prompt-cluster bootstrap"
        ),
        "configs": [c["dataset_name"] for c in CONFIGS],
    }
    (out_dir / "b_stability_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
