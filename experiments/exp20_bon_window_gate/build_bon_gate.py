"""Construct exact finite-pool Best-of-N distributions and local-law diagnostics.

Reuse the tie-aware selection engine. Compute BoN gains and KL directly from
its distribution, independently of the reachable local curve used for fitting.
This prevents saturated, tied-score paths from being dropped merely because
the nominal outer fitting radius is unreachable.
"""
from __future__ import annotations

import argparse
import importlib.util
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

BON_BRIDGE_PATH = (
    REPO_ROOT
    / "engines"
    / "08_bon_bridge"
    / "01_bon_bridge.py"
)
assert BON_BRIDGE_PATH.exists(), f"canonical BoN bridge engine not found at {BON_BRIDGE_PATH}"

_spec = importlib.util.spec_from_file_location("bon_bridge_engine", BON_BRIDGE_PATH)
bon_bridge = importlib.util.module_from_spec(_spec)
sys.modules["bon_bridge_engine"] = bon_bridge  # needed before exec: @dataclass looks this up
_spec.loader.exec_module(bon_bridge)  # type: ignore[union-attr]

# Fine N grid: dense near N=1..8 (where the "first nontrivial step already outside window"
# issue lives, per Appendix A.6), sparser beyond.
N_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
SHARED_FIT_MAX_D = 0.10   # the paper's checked local window


CONFIGS = [
    dict(
        dataset_label="SummEval-human_heldout",
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
        dataset_label="SummEval-ROUGE1",
        human_long_csv=str(DATA_DIR / "summeval_long.csv"),
        item_cols="article_id,model_id",
        group_cols="dimension",
        rater_col="rater_idx",
        score_col="score",
        proxy_csv=str(DATA_DIR / "proxy_rouge1.csv"),
        proxy_item_col="item_id",
        proxy_score_col="proxy_score",
        min_items=200,
        min_raters=3,
    ),
    dict(
        dataset_label="SummEval-GPT4o",
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
        dataset_label="WMT-MQM-human_heldout",
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
        dataset_label="OpenMEVA-MAGS-human_heldout",
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


def _make_args(cfg: dict) -> "argparse.Namespace":
    import argparse
    return argparse.Namespace(
        human_long_csv=cfg["human_long_csv"],
        item_cols=cfg["item_cols"],
        group_cols=cfg["group_cols"],
        rater_col=cfg.get("rater_col", "rater_idx"),
        score_col=cfg.get("score_col", "score"),
        proxy_csv=cfg.get("proxy_csv", ""),
        proxy_item_col=cfg.get("proxy_item_col", "item_id"),
        proxy_score_col=cfg.get("proxy_score_col", "proxy_score"),
        min_items=cfg.get("min_items", 200),
        min_raters=cfg.get("min_raters", 3),
    )


# Curve d-window used ONLY to fit (A,B)/(a_fit,b_fit) -- decoupled from how far BoN's own
# d_bon happens to reach at N=64. This is what actually fixed the RuntimeError skips: the
# original path_to_rows() forces the curve to span max(bridge_max_d, max_bon_d), and for
# discrete/skewed proxies max_bon_d at N=64 can sit right at that path's true KL supremum
# (uniform-over-ties-at-argmax limit), which requires t->inf and is not reachable by the
# doubling bracket search in find_t_for_dmax. The fit itself only ever needs d up to
# SHARED_FIT_MAX_D; BoN's realized delta_bon (the "ground truth" we predict) never needs a
# curve at all -- it comes straight from bon_points_for_path.
FIT_CURVE_D_MAX_LADDER = [0.30, 0.15, SHARED_FIT_MAX_D, 0.05]


def _fit_shared_law(proxy: np.ndarray, gold: np.ndarray):
    """Try a ladder of decreasing curve caps; return (coeff, fit_shared) or (coeff, None)."""
    coeff = bon_bridge.theory_coefficients(gold, proxy)
    for d_cap in FIT_CURVE_D_MAX_LADDER:
        try:
            curve = bon_bridge.exact_tilt_curve(proxy, gold, d_max=d_cap, n_grid=400)
            fit_shared = bon_bridge.fit_local_law(curve.d, curve.q, fit_max_d=SHARED_FIT_MAX_D)
            return coeff, fit_shared
        except (RuntimeError, ValueError):
            continue
    return coeff, None


def _oracle_bon_fit(bon_points) -> tuple[float, float]:
    """In-sample (cheating) two-coefficient fit directly to this path's own realized BoN
    points (d_bon, delta_bon) -- NOT the same object as exp29's "oracle" (which refits
    theory_coefficients on eval-labeled *continuous tilt curve* data). This oracle instead
    asks: what is the best any A*sqrt(d)+B*d curve could do specifically against the
    *discrete* BoN target, using the eval points themselves? It upper-bounds both theory and
    empfit_shared for this experiment's actual prediction target. Named oracle_bon_fit to keep
    the distinction explicit and avoid silently conflating two different "oracle" definitions
    across experiments.
    """
    d = np.array([bp.d_bon for bp in bon_points], dtype=float)
    y = np.array([bp.delta_bon for bp in bon_points], dtype=float)
    mask = d > 1e-12
    if mask.sum() < 2:
        return float("nan"), float("nan")
    X = np.column_stack([np.sqrt(d[mask]), d[mask]])
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    return float(coef[0]), float(coef[1])


def run_config(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    args = _make_args(cfg)
    paths = bon_bridge.build_paths(args)
    frames = []
    skipped = []
    for path in paths:
        proxy = np.asarray(path.proxy_scores, dtype=float)
        gold = np.asarray(path.gold_scores, dtype=float)

        if float(np.var(proxy)) <= 1e-12 or pd.Series(proxy).nunique() < 2:
            skipped.append(f"{cfg['dataset_label']}/{path.group_key}/{path.path_id}: degenerate_proxy_variance")
            continue

        coeff, fit_shared = _fit_shared_law(proxy, gold)
        if fit_shared is None:
            skipped.append(f"{cfg['dataset_label']}/{path.group_key}/{path.path_id}: fit_unreachable_even_at_d0.05")
            continue

        bon_points = bon_bridge.bon_points_for_path(proxy, gold, N_LIST)
        a_oracle, b_oracle = _oracle_bon_fit(bon_points)
        rows = []
        for bp in bon_points:
            d = bp.d_bon
            sqrt_d = float(np.sqrt(max(d, 0.0)))
            theory_pred = coeff.a_exact * sqrt_d + coeff.b_theory * d
            empfit_pred = fit_shared.a_fit * sqrt_d + fit_shared.b_fit * d
            aonly_pred = coeff.a_exact * sqrt_d
            anu_pred = coeff.a_nu * sqrt_d
            zero_pred = 0.0
            oracle_pred = a_oracle * sqrt_d + b_oracle * d
            # Independent re-derivation of d_bon and a fresh ESS computation from the same
            # selection distribution p, used as a merge/alignment self-check in the audit step.
            p = bon_bridge.bon_distribution_uniform_with_ties(proxy, int(bp.N))
            d_recheck = float(bon_bridge.kl_to_uniform_from_distribution(p))
            ess = float(1.0 / np.sum(p**2))
            rows.append({
                "dataset_label": cfg["dataset_label"],
                "group_key": path.group_key,
                "path_id": path.path_id,
                "N": int(bp.N),
                "d_bon": d,
                "d_bon_recheck": d_recheck,
                "ess_bon": ess,
                "delta_bon": bp.delta_bon,
                "a_exact": coeff.a_exact,
                "b_theory": coeff.b_theory,
                "a_nu": coeff.a_nu,
                "a_fit_shared": fit_shared.a_fit,
                "b_fit_shared": fit_shared.b_fit,
                "a_oracle_bon": a_oracle,
                "b_oracle_bon": b_oracle,
                "theory_local_delta_at_dbon": theory_pred,
                "empfit_shared_delta_at_dbon": empfit_pred,
                "aonly_delta_at_dbon": aonly_pred,
                "anu_delta_at_dbon": anu_pred,
                "zero_delta_at_dbon": zero_pred,
                "oracle_bon_delta_at_dbon": oracle_pred,
                "abs_err_theory_vs_bon": abs(theory_pred - bp.delta_bon),
                "abs_err_empfit_shared_vs_bon": abs(empfit_pred - bp.delta_bon),
                "abs_err_aonly_vs_bon": abs(aonly_pred - bp.delta_bon),
                "abs_err_anu_vs_bon": abs(anu_pred - bp.delta_bon),
                "abs_err_zero_vs_bon": abs(zero_pred - bp.delta_bon),
                "abs_err_oracle_bon_vs_bon": abs(oracle_pred - bp.delta_bon),
                "in_shared_window": int(d <= SHARED_FIT_MAX_D + 1e-12),
            })
        frames.append(pd.DataFrame(rows))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    all_skipped = []
    for cfg in CONFIGS:
        print(f"=== {cfg['dataset_label']} ===")
        df, skipped = run_config(cfg)
        print(f"  paths kept: {df['path_id'].nunique() if not df.empty else 0}, "
              f"skipped: {len(skipped)}")
        all_frames.append(df)
        all_skipped.extend(skipped)

    combined = pd.concat(all_frames, ignore_index=True)
    out_path = out_dir / "bon_gate_points.csv"
    combined.to_csv(out_path, index=False)
    n_path_configs = combined[["dataset_label", "path_id"]].drop_duplicates().shape[0]
    print(f"-> {out_path}  ({len(combined)} rows, {n_path_configs} path-configurations)")

    skip_path = out_dir / "bon_gate_skipped.txt"
    if all_skipped:
        skip_path.write_text("\n".join(all_skipped), encoding="utf-8")
        print(f"-> {skip_path}  ({len(all_skipped)} skipped paths)")
    elif skip_path.exists():
        skip_path.unlink()
        print(f"removed stale zero-skip artifact: {skip_path}")

    (out_dir / "run_spec.json").write_text(json.dumps({
        "n_path_configurations": n_path_configs,
        "n_values": N_LIST,
        "shared_fit_max_d": SHARED_FIT_MAX_D,
        "note": "Predictions outside shared_fit_max_d are descriptive extrapolation audits, not deployment recommendations.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
