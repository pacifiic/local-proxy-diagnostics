"""Build continuously softened selection rules on finite pools.

Use a reference/BoN mixture and a rank-based tilt. Compute KL directly from
each discrete distribution rather than using atomless closed forms in tied
pools. Reuse the tie-aware BoN engine for the mixture component.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = HERE / "outputs"
WMT_MQM_LONG_CSV = Path(
    os.environ.get("WMT_MQM_LONG_CSV", DATA_DIR / "wmt_mqm_long.csv")
)

sys.path.insert(0, str(REPO_ROOT / "lib"))
from common_experiment_utils import (  # noqa: E402
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
    parse_cols,
    split_indices,
    theory_coefficients,
)

BON_BRIDGE_PATH = (
    REPO_ROOT / "engines" / "08_bon_bridge" / "01_bon_bridge.py"
)
assert BON_BRIDGE_PATH.exists()
_spec = importlib.util.spec_from_file_location("bon_bridge_engine", BON_BRIDGE_PATH)
bon_bridge = importlib.util.module_from_spec(_spec)
sys.modules["bon_bridge_engine"] = bon_bridge
_spec.loader.exec_module(bon_bridge)  # type: ignore[union-attr]


def kl_discrete(q: np.ndarray, p0: np.ndarray) -> float:
    mask = q > 0
    return float(np.sum(q[mask] * np.log(q[mask] / p0[mask])))


def alpha_mixture_distribution(r: np.ndarray, N: int, alpha: float) -> np.ndarray:
    n = len(r)
    p0 = np.ones(n) / n
    p_bon = bon_bridge.bon_distribution_uniform_with_ties(r, N)
    q = (1 - alpha) * p0 + alpha * p_bon
    q = q / q.sum()
    return q


def rank_tilt_distribution(r: np.ndarray, lam: float) -> np.ndarray:
    n = len(r)
    p0 = np.ones(n) / n
    # Empirical CDF under p0 with average-rank tie handling (F in [1/n, 1], not 0),
    # so F^lambda stays finite and well-defined for the minimum-score item too.
    order = np.argsort(r, kind="mergesort")
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    # average rank for ties
    _, inv, counts = np.unique(r, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank_per_group = cum - (counts - 1) / 2.0
    ranks = avg_rank_per_group[inv]
    F = ranks / n
    w = p0 * (F ** lam)
    q = w / w.sum()
    return q


ALPHA_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0, 24.0]
BON_N_FOR_MIXTURE = 8  # base BoN mechanism the mixture dials into/out of
D_VALID = 0.10
CALIBRATION_FRAC = 0.30
N_REPEATS = 30
RANDOM_SEED = 0


CONFIGS = [
    dict(
        dataset_name="SummEval-human_heldout",
        human_long_csv=str(DATA_DIR / "summeval_long.csv"),
        item_cols="article_id,model_id", group_cols="dimension,rater_type",
        rater_col="rater_idx", score_col="score", proxy_csv="", min_items=200, min_raters=3,
    ),
    dict(
        dataset_name="SummEval-GPT4o",
        human_long_csv=str(DATA_DIR / "summeval_long.csv"),
        item_cols="article_id,model_id", group_cols="dimension",
        rater_col="rater_idx", score_col="score",
        proxy_csv=str(DATA_DIR / "model_proxy_4o_aggregated.csv"),
        proxy_item_col="item_id", proxy_score_col="proxy_score_mean",
        min_items=200, min_raters=3,
    ),
    dict(
        dataset_name="WMT-MQM-human_heldout",
        human_long_csv=str(WMT_MQM_LONG_CSV),
        item_cols="item_id", group_cols="dataset_id",
        rater_col="rater", score_col="score", proxy_csv="", min_items=200, min_raters=3,
    ),
    dict(
        dataset_name="OpenMEVA-MAGS-human_heldout",
        human_long_csv=str(DATA_DIR / "openmeva_long.csv"),
        item_cols="story_id,model_id", group_cols="dataset",
        rater_col="rater_idx", score_col="score", proxy_csv="", min_items=100, min_raters=3,
    ),
]


def build_paths(cfg: dict):
    human = pd.read_csv(cfg["human_long_csv"])
    item_cols = parse_cols(cfg["item_cols"])
    group_cols = parse_cols(cfg["group_cols"])
    if cfg.get("proxy_csv"):
        proxy = pd.read_csv(cfg["proxy_csv"])
        proxy = (
            proxy[[*group_cols, cfg["proxy_item_col"], cfg["proxy_score_col"]]]
            .groupby([*group_cols, cfg["proxy_item_col"]], dropna=False)[cfg["proxy_score_col"]]
            .mean().reset_index()
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


def _selfcheck_kl(seed: int = 0) -> None:
    """alpha=0 / lambda=0 must reduce to the reference (d=0, delta=0); a manual small-n
    KL recomputation must match kl_discrete. Silent-bug guard for the mixture/rank-tilt
    machinery itself, independent of any real dataset."""
    rng = np.random.default_rng(seed)
    n = 12
    r = rng.normal(size=n)
    j = 0.5 * r + rng.normal(size=n) * 0.3
    p0 = np.ones(n) / n

    q0 = alpha_mixture_distribution(r, N=4, alpha=0.0)
    assert np.allclose(q0, p0), "alpha=0 mixture must equal the reference distribution"
    assert abs(kl_discrete(q0, p0)) < 1e-12

    q1 = alpha_mixture_distribution(r, N=4, alpha=1.0)
    q1_direct = bon_bridge.bon_distribution_uniform_with_ties(r, 4)
    assert np.allclose(q1, q1_direct), "alpha=1 mixture must equal pure BoN_N"

    ql0 = rank_tilt_distribution(r, lam=0.0)
    assert np.allclose(ql0, p0), "lambda=0 rank-tilt must equal the reference distribution"

    # manual KL cross-check for one nontrivial point
    q_test = alpha_mixture_distribution(r, N=4, alpha=0.3)
    manual_kl = float(np.sum(q_test * np.log(q_test * n)))  # log(q/p0) = log(q*n) since p0=1/n
    assert np.isclose(manual_kl, kl_discrete(q_test, p0), rtol=1e-10)
    print("Self-check passed: alpha=0/lambda=0 reduce to reference; alpha=1 matches pure BoN; KL formula verified manually.")


def max_reachable_d(proxy: np.ndarray) -> float:
    n_max = int(np.sum(np.isclose(proxy, np.max(proxy), rtol=0.0, atol=1e-12)))
    return float(np.log(len(proxy) / n_max))


def run_config(cfg: dict) -> pd.DataFrame:
    paths = build_paths(cfg)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for path in paths:
        n = len(path.item_ids)
        for repeat in range(N_REPEATS):
            cal_idx, eval_idx = split_indices(n, CALIBRATION_FRAC, rng)
            r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
            r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
            j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
            if len(np.unique(r_cal)) < 2 or len(np.unique(r_eval)) < 2 or len(r_eval) < 4:
                continue
            coeff = theory_coefficients(j_cal, r_cal)
            cal_fit_max_d = min(D_VALID, 0.95 * max_reachable_d(r_cal))
            empirical_fit = None
            if cal_fit_max_d > 1e-8:
                cal_curve = exact_tilt_curve(
                    r_cal, j_cal, d_max=cal_fit_max_d
                )
                empirical_fit = fit_local_law(
                    cal_curve.d, cal_curve.q, fit_max_d=cal_fit_max_d
                )
            p0_eval = np.ones(len(r_eval)) / len(r_eval)
            j0 = float(np.mean(j_eval))

            for mech, grid in (("alpha_mixture", ALPHA_GRID), ("rank_tilt", LAMBDA_GRID)):
                for g in grid:
                    if mech == "alpha_mixture":
                        q = alpha_mixture_distribution(r_eval, BON_N_FOR_MIXTURE, g)
                        param_name, param_val = "alpha", g
                    else:
                        q = rank_tilt_distribution(r_eval, g)
                        param_name, param_val = "lambda", g
                    d = kl_discrete(q, p0_eval)
                    delta_realized = float(np.sum(q * j_eval) - j0)
                    theory_pred = coeff.a_exact * np.sqrt(max(d, 0.0)) + coeff.b_theory * d
                    aonly_pred = coeff.a_exact * np.sqrt(max(d, 0.0))
                    anu_pred = coeff.a_nu * np.sqrt(max(d, 0.0))
                    empfit_pred = (
                        empirical_fit.a_fit * np.sqrt(max(d, 0.0))
                        + empirical_fit.b_fit * d
                        if empirical_fit is not None else float("nan")
                    )
                    rows.append({
                        "dataset": cfg["dataset_name"], "group_key": path.group_key,
                        "path_id": path.path_id, "repeat": repeat,
                        "mechanism": mech, "param_name": param_name, "param_value": param_val,
                        "d": d, "in_window": bool(d <= D_VALID),
                        "delta_realized": delta_realized, "theory_pred": theory_pred,
                        "abs_err_theory": abs(theory_pred - delta_realized),
                        "aonly_pred": aonly_pred,
                        "abs_err_aonly": abs(aonly_pred - delta_realized),
                        "anu_pred": anu_pred,
                        "abs_err_anu": abs(anu_pred - delta_realized),
                        "empfit_pred": empfit_pred,
                        "abs_err_empfit": abs(empfit_pred - delta_realized),
                        "zero_pred": 0.0,
                        "abs_err_zero": abs(delta_realized),
                        "a_cal": coeff.a_exact, "b_cal": coeff.b_theory,
                        "n_eval": len(r_eval),
                        "empfit_fit_max_d": cal_fit_max_d,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[cfg["dataset_name"] for cfg in CONFIGS],
        default=[cfg["dataset_name"] for cfg in CONFIGS],
    )
    args = parser.parse_args()
    out_dir = args.outdir
    _selfcheck_kl()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    selected = [
        cfg for cfg in CONFIGS
        if cfg["dataset_name"] in set(args.datasets)
    ]
    for cfg in selected:
        print(f"=== {cfg['dataset_name']} ===")
        df = run_config(cfg)
        print(f"  rows: {len(df)}, paths: {df['path_id'].nunique() if not df.empty else 0}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out_path = out_dir / "mixture_ranktilt_rows.csv"
    combined.to_csv(out_path, index=False)
    print(f"-> {out_path} ({len(combined)} rows)")

    spec = {
        "alpha_grid": ALPHA_GRID, "lambda_grid": LAMBDA_GRID, "bon_n_for_mixture": BON_N_FOR_MIXTURE,
        "d_valid": D_VALID, "calibration_frac": CALIBRATION_FRAC, "n_repeats": N_REPEATS,
        "random_seed": RANDOM_SEED,
        "scope": "pooled_global_item_set_stress_test_not_prompt_conditional",
        "baselines": ["A+B", "A-only", "A_nu", "empirical_fit", "zero"],
        "configs": [c["dataset_name"] for c in selected],
    }
    (out_dir / "mixture_ranktilt_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
