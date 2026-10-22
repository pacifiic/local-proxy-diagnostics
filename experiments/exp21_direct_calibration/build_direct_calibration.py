"""Compare moment prediction with direct proxy-conditional calibration.

Reuse the established split engine and add quantile-binned shrinkage and
penalized cubic-spline predictors alongside isotonic and empirical-curve fits.
Fits use calibration labels only. Direct integration uses evaluation proxy
scores but not evaluation labels. Eligible slices include SummEval, GPT-4o,
WMT MQM, and OpenMEVA-MAGS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = HERE / "outputs"

sys.path.insert(0, str(REPO_ROOT / "lib"))

CANON_PATH = (
    REPO_ROOT
    / "experiments"
    / "17_calibration_baseline_benchmark"
    / "01_calibration_baseline_benchmark.py"
)
assert CANON_PATH.exists(), f"canonical calibration-baseline engine not found at {CANON_PATH}"

_spec = importlib.util.spec_from_file_location("calib_engine", CANON_PATH)
calib = importlib.util.module_from_spec(_spec)
sys.modules["calib_engine"] = calib
_spec.loader.exec_module(calib)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# New predictors: quantile-binned + shrinkage, and spline + ridge (spline/GAM
# stand-in with automatic smoothing via internal cross-validation).
# Both follow the exact same "fit h(r) on calibration only, then integrate
# under the evaluation proxy's exact tilt weights at d" pattern as the
# existing isotonic baseline (predict_isotonic_gain), so they slot into the
# same comparison table.
# ---------------------------------------------------------------------------

BIN_CANDIDATES = (3, 5, 8, 12)
SHRINK_K_CANDIDATES = (0.0, 5.0, 20.0)
SPLINE_KNOT_CANDIDATES = (4, 6, 8)
MLP_HIDDEN_LAYERS = (16, 16)
MLP_ALPHA = 1e-3
MLP_MAX_ITER = 1000
MLP_LEARNING_RATE_INIT = 1e-3
MLP_RANDOM_SEED = 0


def fit_binned_shrinkage(r_cal: np.ndarray, j_cal: np.ndarray, n_folds: int = 5, seed: int = 0):
    """Equal-frequency (quantile) binning with James-Stein-style shrinkage toward
    the global calibration mean. Bin count and shrinkage strength are chosen by
    calibration-internal K-fold CV (never touches evaluation labels)."""
    r_cal = np.asarray(r_cal, dtype=float)
    j_cal = np.asarray(j_cal, dtype=float)
    n = len(r_cal)
    bin_candidates = tuple(b for b in BIN_CANDIDATES if b * 3 <= n) or (2,)
    kf = KFold(n_splits=min(n_folds, max(2, n // 5)), shuffle=True, random_state=seed)

    def _fit_bins(r, j, n_bins: int, shrink_k: float):
        edges = np.quantile(r, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([r.min() - 1e-9, np.median(r), r.max() + 1e-9])
        edges_inner = edges[1:-1]
        bin_id = np.digitize(r, edges_inner)
        n_actual_bins = len(edges) - 1
        global_mean = float(np.mean(j))
        bin_means = np.full(n_actual_bins, global_mean)
        bin_counts = np.zeros(n_actual_bins)
        for b in range(n_actual_bins):
            mask = bin_id == b
            bin_counts[b] = mask.sum()
            if mask.any():
                bin_means[b] = float(np.mean(j[mask]))
        denom = bin_counts + shrink_k
        # Guard: an empty bin (count=0) with shrink_k=0 gives 0/0 -- fall back to the
        # calibration global mean rather than letting NaN silently propagate into
        # downstream predictions/errors.
        shrunk = np.where(denom > 0, (bin_counts * bin_means + shrink_k * global_mean) / np.where(denom > 0, denom, 1.0), global_mean)
        return edges_inner, shrunk, global_mean

    best = None
    for n_bins in bin_candidates:
        for shrink_k in SHRINK_K_CANDIDATES:
            fold_mses = []
            for train_idx, val_idx in kf.split(r_cal):
                if len(train_idx) < n_bins * 2:
                    continue
                edges_inner, shrunk, global_mean = _fit_bins(r_cal[train_idx], j_cal[train_idx], n_bins, shrink_k)
                bid_val = np.digitize(r_cal[val_idx], edges_inner)
                bid_val = np.clip(bid_val, 0, len(shrunk) - 1)
                pred = shrunk[bid_val]
                fold_mses.append(float(np.mean((pred - j_cal[val_idx]) ** 2)))
            if fold_mses:
                mse = float(np.mean(fold_mses))
                if best is None or mse < best[0]:
                    best = (mse, n_bins, shrink_k)

    if best is None:
        n_bins, shrink_k = 2, 0.0
    else:
        _, n_bins, shrink_k = best

    edges_inner, shrunk, global_mean = _fit_bins(r_cal, j_cal, n_bins, shrink_k)

    def predict(r_new: np.ndarray) -> np.ndarray:
        bid = np.digitize(np.asarray(r_new, dtype=float), edges_inner)
        bid = np.clip(bid, 0, len(shrunk) - 1)
        return shrunk[bid]

    return predict, {"n_bins": int(n_bins), "shrink_k": float(shrink_k)}


def fit_spline_ridge(r_cal: np.ndarray, j_cal: np.ndarray, n_folds: int = 5, seed: int = 0):
    """B-spline basis (clamped/natural-like boundary extrapolation) + ridge
    regression, with knot count AND ridge alpha chosen by calibration-internal CV."""
    r_cal = np.asarray(r_cal, dtype=float).reshape(-1, 1)
    j_cal = np.asarray(j_cal, dtype=float)
    n = len(r_cal)
    n_splits = min(n_folds, max(2, n // 5))

    knot_candidates = tuple(k for k in SPLINE_KNOT_CANDIDATES if k + 3 < n) or (2,)
    best = None
    for n_knots in knot_candidates:
        pipe = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant"),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        try:
            scores = cross_val_score(
                pipe, r_cal, j_cal, cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed),
                scoring="neg_mean_squared_error",
            )
        except ValueError:
            continue
        mse = float(-np.mean(scores))
        if best is None or mse < best[0]:
            best = (mse, n_knots)

    n_knots = best[1] if best is not None else 2
    pipe = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant"),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    pipe.fit(r_cal, j_cal)

    def predict(r_new: np.ndarray) -> np.ndarray:
        return pipe.predict(np.asarray(r_new, dtype=float).reshape(-1, 1))

    return predict, {"n_knots": int(n_knots), "ridge_alpha": float(pipe.named_steps["ridgecv"].alpha_)}


def fit_mlp_calibrator(r_cal: np.ndarray, j_cal: np.ndarray):
    """Fit the pre-specified one-dimensional MLP calibrator."""
    x = np.asarray(r_cal, dtype=float).reshape(-1, 1)
    y = np.asarray(j_cal, dtype=float)
    if len(x) < 4 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        mean = float(np.mean(y))

        def predict_constant(r_new: np.ndarray) -> np.ndarray:
            return np.full(len(r_new), mean, dtype=float)

        return predict_constant, {
            "n_parameters": 0,
            "n_iter": 0,
            "converged": True,
        }

    x_scaler = StandardScaler().fit(x)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    x_scaled = x_scaler.transform(x)
    y_scaled = (y - y_mean) / y_std
    model = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        learning_rate_init=MLP_LEARNING_RATE_INIT,
        max_iter=MLP_MAX_ITER,
        random_state=MLP_RANDOM_SEED,
        early_stopping=False,
        n_iter_no_change=max(20, MLP_MAX_ITER // 10),
        tol=1e-4,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_scaled, y_scaled)
    converged = not any(
        issubclass(warning.category, ConvergenceWarning)
        for warning in caught
    )

    def predict(r_new: np.ndarray) -> np.ndarray:
        x_new = x_scaler.transform(
            np.asarray(r_new, dtype=float).reshape(-1, 1)
        )
        return y_mean + y_std * model.predict(x_new)

    n_parameters = int(
        sum(weights.size for weights in model.coefs_)
        + sum(bias.size for bias in model.intercepts_)
    )
    return predict, {
        "n_parameters": n_parameters,
        "n_iter": int(model.n_iter_),
        "converged": bool(converged),
    }


def predict_gain_from_h(h_predict_fn, eval_proxy: np.ndarray, d: float) -> float:
    h = h_predict_fn(eval_proxy)
    w = calib.exact_weights_at_d(eval_proxy, d)
    return float(np.sum(w * h) - np.mean(h))


NEW_BASELINES = ["binned_shrinkage", "spline_ridge", "mlp_calibration"]
ALL_BASELINES = calib.BASELINES + NEW_BASELINES


def max_reachable_d(proxy: np.ndarray) -> float:
    proxy = np.asarray(proxy, dtype=float)
    n_max = int(np.sum(np.isclose(proxy, np.max(proxy), rtol=0.0, atol=1e-12)))
    return float(np.log(len(proxy) / n_max))


def fit_split_models(
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    r_eval: np.ndarray,
    fit_max_d: float,
) -> dict:
    """Fit all calibration-only models once for one outer split."""
    coeff = calib.theory_coefficients(j_cal, r_cal)
    sigma_j = float(np.std(j_cal))
    rho_pearson = calib.safe_corr(r_cal, j_cal, rank=False)
    rho_spearman = calib.safe_corr(r_cal, j_cal, rank=True)
    errors: dict[str, str] = {}

    h_eval: dict[str, np.ndarray] = {}
    try:
        iso = calib.fit_isotonic(r_cal, j_cal)
        h_eval["isotonic_calibration"] = np.asarray(
            iso.predict(r_eval), dtype=float
        )
    except Exception as exc:  # noqa: BLE001
        errors["isotonic_calibration"] = f"{type(exc).__name__}: {exc}"

    bs_meta = {"n_bins": np.nan, "shrink_k": np.nan}
    try:
        bs_predict, bs_meta = fit_binned_shrinkage(r_cal, j_cal)
        h_eval["binned_shrinkage"] = np.asarray(
            bs_predict(r_eval), dtype=float
        )
    except Exception as exc:  # noqa: BLE001
        errors["binned_shrinkage"] = f"{type(exc).__name__}: {exc}"

    sp_meta = {"n_knots": np.nan, "ridge_alpha": np.nan}
    try:
        sp_predict, sp_meta = fit_spline_ridge(r_cal, j_cal)
        h_eval["spline_ridge"] = np.asarray(sp_predict(r_eval), dtype=float)
    except Exception as exc:  # noqa: BLE001
        errors["spline_ridge"] = f"{type(exc).__name__}: {exc}"

    mlp_meta = {
        "n_parameters": np.nan,
        "n_iter": np.nan,
        "converged": False,
    }
    try:
        mlp_predict, mlp_meta = fit_mlp_calibrator(r_cal, j_cal)
        h_eval["mlp_calibration"] = np.asarray(
            mlp_predict(r_eval), dtype=float
        )
    except Exception as exc:  # noqa: BLE001
        errors["mlp_calibration"] = f"{type(exc).__name__}: {exc}"

    cal_max_d = max_reachable_d(r_cal)
    empirical_fit_max_d = min(float(fit_max_d), 0.95 * cal_max_d)
    fit_coeff = None
    if empirical_fit_max_d > 1e-8:
        curve_cal = calib.exact_tilt_curve(
            r_cal, j_cal, d_max=empirical_fit_max_d
        )
        fit_coeff = calib.fit_local_law(
            curve_cal.d, curve_cal.q, fit_max_d=empirical_fit_max_d
        )
    else:
        errors["empirical_fit"] = "calibration frontier has zero reachable KL"

    return {
        "coeff": coeff,
        "sigma_j": sigma_j,
        "rho_pearson": rho_pearson,
        "rho_spearman": rho_spearman,
        "h_eval": h_eval,
        "fit_coeff": fit_coeff,
        "fit_errors": errors,
        "bs_meta": bs_meta,
        "sp_meta": sp_meta,
        "mlp_meta": mlp_meta,
        "cal_max_d": cal_max_d,
        "empirical_fit_max_d": empirical_fit_max_d,
    }


def predictions_at_d(
    split_models: dict,
    r_cal: np.ndarray,
    j_cal: np.ndarray,
    eval_weights: np.ndarray,
    d: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Evaluate fitted split models at one radius without refitting."""
    coeff = split_models["coeff"]
    sqrt_d = math.sqrt(max(0.0, d))
    preds = {
        "A+B": coeff.a_exact * sqrt_d + coeff.b_theory * d,
        "A-only": coeff.a_exact * sqrt_d,
        "A_nu": coeff.a_nu * sqrt_d,
        "pearson_corr": (
            math.sqrt(2.0)
            * split_models["sigma_j"]
            * split_models["rho_pearson"]
            * sqrt_d
        ),
        "spearman_corr": (
            math.sqrt(2.0)
            * split_models["sigma_j"]
            * split_models["rho_spearman"]
            * sqrt_d
        ),
        "zero": 0.0,
    }
    errors = dict(split_models["fit_errors"])

    for baseline, h in split_models["h_eval"].items():
        preds[baseline] = float(np.sum(eval_weights * h) - np.mean(h))

    cal_max_d = split_models["cal_max_d"]
    if d < cal_max_d - 1e-10:
        preds["empirical_curve"] = calib.exact_delta_at_d(r_cal, j_cal, d)
    else:
        errors["empirical_curve"] = (
            f"calibration frontier max KL={cal_max_d:.6g} cannot reach d={d:g}"
        )

    fit_coeff = split_models["fit_coeff"]
    if fit_coeff is not None:
        preds["empirical_fit"] = (
            fit_coeff.a_fit * sqrt_d + fit_coeff.b_fit * d
        )
    return preds, errors


CONFIGS = [
    dict(
        config_key="summeval",
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
        config_key="gpt4o",
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
        config_key="wmt",
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
        config_key="openmeva",
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

D_TARGETS = [0.01, 0.025, 0.05, 0.075, 0.10]
CALIBRATION_FRAC = 0.30
N_REPEATS = 50
FIT_MAX_D = 0.10
RANDOM_SEED = 0


def _make_args(cfg: dict):
    import argparse
    return argparse.Namespace(
        dataset_name=cfg["dataset_name"],
        human_long_csv=cfg["human_long_csv"],
        item_cols=cfg["item_cols"],
        group_cols=cfg["group_cols"],
        rater_col=cfg.get("rater_col", "rater_idx"),
        score_col=cfg.get("score_col", "score"),
        proxy_csv=cfg.get("proxy_csv", ""),
        proxy_item_col=cfg.get("proxy_item_col", "item_id"),
        proxy_score_col=cfg.get("proxy_score_col", "proxy_score_mean"),
        min_items=cfg.get("min_items", 200),
        min_raters=cfg.get("min_raters", 3),
    )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def run_path_task(task: tuple[dict, object, list[tuple[np.ndarray, np.ndarray]]]):
    """Run all repeats for one path; suitable for process parallelism."""
    cfg, path, split_plan = task
    rows: list[dict] = []
    skipped: list[dict] = []
    try:
        from threadpoolctl import threadpool_limits

        thread_limit = threadpool_limits(limits=1)
    except ImportError:
        from contextlib import nullcontext

        thread_limit = nullcontext()

    with thread_limit:
        for repeat, (cal_idx, eval_idx) in enumerate(split_plan):
            r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
            r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
            j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
            try:
                split_models = fit_split_models(
                    r_cal, j_cal, r_eval, FIT_MAX_D
                )
            except Exception as exc:  # noqa: BLE001
                for d in D_TARGETS:
                    skipped.append({
                        "dataset": cfg["dataset_name"],
                        "path_id": path.path_id,
                        "repeat": repeat,
                        "d": d,
                        "baseline": "ALL",
                        "reason": (
                            f"calibration_fit_failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    })
                continue

            for d in D_TARGETS:
                try:
                    eval_weights = calib.exact_weights_at_d(r_eval, d)
                    actual = float(
                        np.sum(eval_weights * j_eval) - np.mean(j_eval)
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped.append({
                        "dataset": cfg["dataset_name"],
                        "path_id": path.path_id,
                        "repeat": repeat,
                        "d": d,
                        "baseline": "ALL",
                        "reason": (
                            f"evaluation_unreachable: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    })
                    continue

                preds, errors = predictions_at_d(
                    split_models, r_cal, j_cal, eval_weights, d
                )
                for baseline, reason in errors.items():
                    skipped.append({
                        "dataset": cfg["dataset_name"],
                        "path_id": path.path_id,
                        "repeat": repeat,
                        "d": d,
                        "baseline": baseline,
                        "reason": reason,
                    })
                for baseline in ALL_BASELINES:
                    if baseline not in preds or not np.isfinite(preds[baseline]):
                        continue
                    pred = preds[baseline]
                    rows.append({
                        "dataset": cfg["dataset_name"],
                        "proxy_mode": (
                            "external_model_proxy"
                            if cfg.get("proxy_csv")
                            else "heldout_human_proxy"
                        ),
                        "group_key": path.group_key,
                        "path_id": path.path_id,
                        "repeat": repeat,
                        "d": float(d),
                        "baseline": baseline,
                        "actual_gain": actual,
                        "predicted_gain": pred,
                        "abs_error": abs(pred - actual),
                        "n_cal": int(len(cal_idx)),
                        "n_eval": int(len(eval_idx)),
                        "bs_n_bins": split_models["bs_meta"]["n_bins"],
                        "bs_shrink_k": split_models["bs_meta"]["shrink_k"],
                        "sp_n_knots": split_models["sp_meta"]["n_knots"],
                        "sp_ridge_alpha": split_models["sp_meta"][
                            "ridge_alpha"
                        ],
                        "mlp_n_parameters": split_models["mlp_meta"][
                            "n_parameters"
                        ],
                        "mlp_n_iter": split_models["mlp_meta"]["n_iter"],
                        "mlp_converged": split_models["mlp_meta"][
                            "converged"
                        ],
                        "cal_max_reachable_d": split_models["cal_max_d"],
                        "empirical_fit_max_d": split_models[
                            "empirical_fit_max_d"
                        ],
                    })
    return rows, skipped


def run_config(cfg: dict, n_jobs: int) -> tuple[pd.DataFrame, list[dict]]:
    args = _make_args(cfg)
    paths = list(calib.build_paths(args))
    rng = np.random.default_rng(RANDOM_SEED)
    tasks = []
    for path in paths:
        split_plan = [
            calib.split_indices(
                len(path.item_ids), CALIBRATION_FRAC, rng
            )
            for _ in range(N_REPEATS)
        ]
        tasks.append((cfg, path, split_plan))

    if n_jobs == 1:
        results = [run_path_task(task) for task in tasks]
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=context
        ) as executor:
            results = list(
                executor.map(run_path_task, tasks, chunksize=1)
            )

    rows = [row for path_rows, _ in results for row in path_rows]
    skipped = [
        skip for _, path_skips in results for skip in path_skips
    ]
    return pd.DataFrame(rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[cfg["config_key"] for cfg in CONFIGS],
        default=[cfg["config_key"] for cfg in CONFIGS],
        help="Dataset configurations to run (default: all).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Parallel path workers (default: min(32, CPU count)).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    out_dir = args.outdir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = set(args.datasets)
    selected = [
        cfg for cfg in CONFIGS if cfg["config_key"] in selected_keys
    ]
    all_frames = []
    all_skipped = []
    for cfg in selected:
        print(f"=== {cfg['dataset_name']} ===")
        df, skipped = run_config(cfg, n_jobs=args.jobs)
        print(f"  rows: {len(df)}, skipped: {len(skipped)}, "
              f"paths: {df['path_id'].nunique() if not df.empty else 0}")
        all_frames.append(df)
        all_skipped.extend(skipped)

    combined = pd.concat(all_frames, ignore_index=True)
    out_path = out_dir / "direct_calibration_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"-> {out_path}  ({len(combined)} rows)")

    if all_skipped:
        skip_df = pd.DataFrame(all_skipped)
        skip_path = out_dir / "direct_calibration_skipped.csv"
        skip_df.to_csv(skip_path, index=False)
        print(f"-> {skip_path}  ({len(skip_df)} skipped)")

    spec = {
        "configs": [
            {
                "config_key": c["config_key"],
                "dataset_name": c["dataset_name"],
                "human_long_csv": _portable_path(c["human_long_csv"]),
                "human_long_sha256": _sha256(c["human_long_csv"]),
                "proxy_csv": (
                    _portable_path(c["proxy_csv"])
                    if c.get("proxy_csv")
                    else None
                ),
                "proxy_sha256": (
                    _sha256(c["proxy_csv"]) if c.get("proxy_csv") else None
                ),
                "item_cols": c["item_cols"],
                "group_cols": c["group_cols"],
                "min_items": c["min_items"],
                "min_raters": c["min_raters"],
            }
            for c in selected
        ],
        "d_targets": D_TARGETS,
        "calibration_frac": CALIBRATION_FRAC,
        "n_repeats": N_REPEATS,
        "random_seed": RANDOM_SEED,
        "n_jobs": args.jobs,
        "parallelization_unit": "path",
        "calibration_model_fit_frequency": "once per path/repeat split",
        "fit_max_d": FIT_MAX_D,
        "reachability_policy": {
            "evaluation": "skip only the requested d when unreachable",
            "empirical_curve": "unavailable when calibration cannot reach requested d",
            "empirical_fit": "fit through 95% of calibration max reachable KL, capped at fit_max_d",
        },
        "all_baselines": ALL_BASELINES,
        "new_baselines": NEW_BASELINES,
        "bin_candidates": BIN_CANDIDATES,
        "shrink_k_candidates": SHRINK_K_CANDIDATES,
        "spline_knot_candidates": SPLINE_KNOT_CANDIDATES,
        "mlp": {
            "hidden_layers": MLP_HIDDEN_LAYERS,
            "n_parameters": 321,
            "activation": "relu",
            "solver": "adam",
            "alpha": MLP_ALPHA,
            "max_iter": MLP_MAX_ITER,
            "learning_rate_init": MLP_LEARNING_RATE_INIT,
            "random_seed": MLP_RANDOM_SEED,
            "hyperparameter_selection": "none; pre-specified",
        },
    }
    (out_dir / "direct_calibration_run_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
