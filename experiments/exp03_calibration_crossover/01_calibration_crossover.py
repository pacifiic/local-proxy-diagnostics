#!/usr/bin/env python3
"""[exp03] Calibration-size crossover experiment (R2 defense).

Reconstruction of the lost `experiments/15_calibration_crossover/01_calibration_crossover.py`.
Validated against reference_results/exp03_calibration_crossover/:
per-split rows reproduce x50_merged_splits.csv to machine precision, and the aggregated
outputs reproduce merged_human/gpt/openmeva/wmt(.3) CSVs.

Protocol (identical to the paper's prospective-prediction protocol):
  - paths: group x held-out-rater proxy (or external model proxy),
    built with lib/common_experiment_utils.generate_{human,model}_paths
  - per cal_frac: fresh np.random.default_rng(seed); for each path (generation order),
    n_repeats item splits drawn sequentially from that single stream
    (a skipped/degenerate split still consumes its draw)
  - exact finite-item exponential tilt curves on both halves (n_grid=120, d_max=0.10)
  - predictors evaluated on a 40-point d-grid (0.0025..0.10); abs_err at d=0.10,
    iae = mean absolute error over the grid
  - aggregation: mean over path means ("path-clustered"), path-clustered bootstrap
    95% CI (n_boot=1000, fresh default_rng(boot_seed) per cal_frac, paths sorted by id)

Predictors:
  theory      : plug-in A*sqrt(d) + B*d from calibration (r, J) moments
  aonly       : A*sqrt(d) only
  empfit      : 2-term LS fit of the calibration tilt curve (sqrt(d), d)
  poly3       : 3-term LS fit (sqrt(d), d, d^1.5)
  transfer    : nonparametric calibration-curve transfer (interp cal curve at d)
  zero        : predict 0
  extended (--extended):
    theory_semi : isotonic h(r)=E[J|r] fit on cal, plug-in moments of (h(r_eval), r_eval)
    iso_cal     : isotonic h integrated under the CAL tilt curve (cal info only)
    iso_eval    : isotonic h applied to eval r, integrated under the EVAL tilt curve
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from common_experiment_utils import (  # noqa: E402
    PathData,
    exact_tilt_curve,
    fit_local_law,
    generate_human_paths,
    generate_model_paths,
    mkdir,
    parse_cols,
    split_indices,
    theory_coefficients,
    tilt_weights,
)

try:
    from sklearn.isotonic import IsotonicRegression

    def isotonic_fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(np.asarray(x_fit, dtype=float), np.asarray(y_fit, dtype=float))
        return np.asarray(model.predict(np.asarray(x_pred, dtype=float)), dtype=float)

except ImportError:  # pragma: no cover - PAVA fallback if sklearn is unavailable

    def isotonic_fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
        x = np.asarray(x_fit, dtype=float)
        y = np.asarray(y_fit, dtype=float)
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        # collapse ties on x (secondary averaging), then PAVA with counts as weights
        ux, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
        sums = np.zeros(len(ux))
        np.add.at(sums, inv, ys)
        level_y = sums / cnt
        vals: List[float] = []
        wts: List[float] = []
        for v, w in zip(level_y, cnt.astype(float)):
            vals.append(float(v))
            wts.append(float(w))
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2 = vals.pop(), wts.pop()
                v1, w1 = vals.pop(), wts.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
                wts.append(w1 + w2)
        # expand back to level fits
        fitted = np.empty(len(ux))
        i = 0
        blocks: List[int] = []
        # reconstruct block sizes by re-running pooling on counts
        # (simple approach: nearest-isotonic via np.interp over block boundaries)
        # rebuild: assign each pooled value to consecutive levels
        # recompute pooling with block tracking
        vals2: List[float] = []
        wts2: List[float] = []
        sizes: List[int] = []
        for v, w in zip(level_y, cnt.astype(float)):
            vals2.append(float(v)); wts2.append(float(w)); sizes.append(1)
            while len(vals2) > 1 and vals2[-2] > vals2[-1]:
                v2, w2, s2 = vals2.pop(), wts2.pop(), sizes.pop()
                v1, w1, s1 = vals2.pop(), wts2.pop(), sizes.pop()
                vals2.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts2.append(w1 + w2); sizes.append(s1 + s2)
        for v, s in zip(vals2, sizes):
            fitted[i : i + s] = v
            i += s
        xp = np.clip(np.asarray(x_pred, dtype=float), ux[0], ux[-1])
        return np.interp(xp, ux, fitted)


BASE_PREDICTORS = ["theory", "aonly", "empfit", "poly3", "transfer", "zero"]
EXT_PREDICTORS = ["theory", "theory_semi", "aonly", "empfit", "poly3", "transfer", "iso_cal", "iso_eval", "zero"]
BASE_DIFFS = ["empfit", "poly3", "transfer"]
EXT_DIFFS = ["empfit", "poly3", "transfer", "iso_cal", "iso_eval", "theory_semi"]

TRACKS: Dict[str, Dict[str, Any]] = {
    "human": dict(
        human_long_csv="summeval_long.csv", item_cols="article_id,model_id",
        group_cols="dimension,rater_type", rater_col="rater_idx", score_col="score",
        proxy_csv="", n_repeats=12, cal_fracs="0.05,0.1,0.15,0.2,0.3,0.5,0.7",
        min_items=200, setting="human_heldout",
    ),
    "gpt": dict(
        human_long_csv="summeval_long.csv", item_cols="article_id,model_id",
        group_cols="dimension", rater_col="rater_idx", score_col="score",
        proxy_csv="model_proxy_4o_aggregated.csv", proxy_item_col="item_id",
        proxy_score_col="proxy_score_mean", n_repeats=30,
        cal_fracs="0.05,0.1,0.15,0.2,0.3,0.5,0.7", min_items=200, setting="model_proxy",
    ),
    "openmeva": dict(
        human_long_csv="openmeva_long.csv", item_cols="story_id,model_id",
        group_cols="dataset", rater_col="rater_idx", score_col="score",
        proxy_csv="", n_repeats=30, cal_fracs="0.05,0.1,0.2,0.3,0.5",
        min_items=200, setting="human_heldout",
    ),
    "wmt": dict(
        human_long_csv="wmt_mqm_long.csv", item_cols="item_id",
        group_cols="dataset_id", rater_col="rater", score_col="score",
        proxy_csv="", n_repeats=8, cal_fracs="0.05,0.1,0.2,0.3,0.5",
        min_items=100, setting="human_heldout", n_boot=500,
    ),
}


def curve_interp_deltas(curve, d_grid: np.ndarray, q_values: Optional[np.ndarray] = None) -> np.ndarray:
    """Gain at each d via linear interpolation of a (d, q) tilt curve, relative to q(0)."""
    q = curve.q if q_values is None else q_values
    return np.interp(d_grid, curve.d, q) - float(q[0])


def iso_curve_q(curve, r: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Gain curve of a calibrated score h(r) evaluated at the tilt curve's own t grid."""
    out = np.empty(len(curve.t))
    for i, t in enumerate(curve.t):
        w = tilt_weights(r, float(t))
        out[i] = float(np.sum(w * h))
    return out


def evaluate_one_split(
    path: PathData,
    cal_idx: np.ndarray,
    eva_idx: np.ndarray,
    d_grid: np.ndarray,
    fit_max_d: float,
    n_grid: int,
    extended: bool,
) -> Dict[str, Any]:
    r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
    j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
    r_eva = np.asarray(path.proxy_scores[eva_idx], dtype=float)
    j_eva = np.asarray(path.gold_scores[eva_idx], dtype=float)

    curve_eva = exact_tilt_curve(r_eva, j_eva, d_max=fit_max_d, n_grid=n_grid)
    curve_cal = exact_tilt_curve(r_cal, j_cal, d_max=fit_max_d, n_grid=n_grid)
    coeff = theory_coefficients(j_cal, r_cal)
    fit2 = fit_local_law(curve_cal.d, curve_cal.q, fit_max_d=fit_max_d)

    # 3-term fit: sqrt(d), d, d^1.5 on the calibration tilt curve
    dd = curve_cal.d
    yy = curve_cal.q - float(curve_cal.q[0])
    mask = (dd > 0) & (dd <= fit_max_d)
    X3 = np.column_stack([np.sqrt(dd[mask]), dd[mask], dd[mask] ** 1.5])
    coef3, *_ = np.linalg.lstsq(X3, yy[mask], rcond=None)

    actual = curve_interp_deltas(curve_eva, d_grid)
    sqrt_d = np.sqrt(d_grid)

    preds: Dict[str, np.ndarray] = {
        "theory": coeff.a_exact * sqrt_d + coeff.b_theory * d_grid,
        "aonly": coeff.a_exact * sqrt_d,
        "empfit": fit2.a_fit * sqrt_d + fit2.b_fit * d_grid,
        "poly3": coef3[0] * sqrt_d + coef3[1] * d_grid + coef3[2] * d_grid ** 1.5,
        "transfer": curve_interp_deltas(curve_cal, d_grid),
        "zero": np.zeros_like(d_grid),
    }
    if extended:
        h_cal = isotonic_fit_predict(r_cal, j_cal, r_cal)
        h_eva = isotonic_fit_predict(r_cal, j_cal, r_eva)
        coeff_semi = theory_coefficients(h_eva, r_eva)
        preds["theory_semi"] = coeff_semi.a_exact * sqrt_d + coeff_semi.b_theory * d_grid
        preds["iso_cal"] = curve_interp_deltas(curve_cal, d_grid, iso_curve_q(curve_cal, r_cal, h_cal))
        preds["iso_eval"] = curve_interp_deltas(curve_eva, d_grid, iso_curve_q(curve_eva, r_eva, h_eva))

    names = EXT_PREDICTORS if extended else BASE_PREDICTORS
    row: Dict[str, Any] = {
        "group_key": path.group_key,
        "path_id": path.path_id,
        "n_cal": int(len(cal_idx)),
        "n_eval": int(len(eva_idx)),
        "actual_delta": float(actual[-1]),
    }
    for name in names:
        err = np.abs(preds[name] - actual)
        row[f"iae_{name}"] = float(np.mean(err))
        row[f"abs_err_{name}"] = float(err[-1])
    return row


def aggregate_frac(
    df: pd.DataFrame,
    setting: str,
    cal_frac: float,
    n_skipped: int,
    predictors: Sequence[str],
    diffs: Sequence[str],
    n_boot: int,
    boot_seed: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "setting": setting,
        "cal_frac": cal_frac,
        "n_splits": int(len(df)),
        "n_skipped": int(n_skipped),
        "median_n_cal": float(df["n_cal"].median()),
    }
    metric_cols = [f"abs_err_{p}" for p in predictors] + [f"iae_{p}" for p in predictors]
    path_df = df.groupby("path_id")[metric_cols].mean()  # sorted by path_id
    mat = path_df.values
    col_index = {c: i for i, c in enumerate(metric_cols)}
    n_paths = len(mat)

    rng = np.random.default_rng(boot_seed)
    boot_means = np.empty((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n_paths, size=n_paths)
        boot_means[b] = mat[idx].mean(axis=0)

    means = mat.mean(axis=0)
    for kind in ("abs_err", "iae"):
        for p in predictors:
            c = f"{kind}_{p}"
            i = col_index[c]
            row[f"{c}_mean"] = float(means[i])
            row[f"{c}_lo"] = float(np.percentile(boot_means[:, i], 2.5))
            row[f"{c}_hi"] = float(np.percentile(boot_means[:, i], 97.5))
    it = col_index["abs_err_theory"]
    for p in diffs:
        ix = col_index[f"abs_err_{p}"]
        d = boot_means[:, it] - boot_means[:, ix]
        row[f"diff_theory_minus_{p}_mean"] = float(means[it] - means[ix])
        row[f"diff_theory_minus_{p}_lo"] = float(np.percentile(d, 2.5))
        row[f"diff_theory_minus_{p}_hi"] = float(np.percentile(d, 97.5))
    return row


def build_paths(args: argparse.Namespace) -> List[PathData]:
    human = pd.read_csv(args.human_long_csv)
    item_cols = parse_cols(args.item_cols)
    group_cols = parse_cols(args.group_cols)
    if args.proxy_csv:
        proxy = pd.read_csv(args.proxy_csv)
        return list(
            generate_model_paths(
                human_long=human,
                proxy_df=proxy,
                item_cols=item_cols,
                score_col=args.score_col,
                group_cols=group_cols,
                proxy_item_col=args.proxy_item_col,
                proxy_score_col=args.proxy_score_col,
                min_items=args.min_items,
            )
        )
    return list(
        generate_human_paths(
            df=human,
            item_cols=item_cols,
            rater_col=args.rater_col,
            score_col=args.score_col,
            group_cols=group_cols,
            min_items=args.min_items,
            min_raters=args.min_raters,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration-size crossover experiment (exp03 reconstruction).")
    ap.add_argument("--track", required=True, choices=sorted(TRACKS))
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--human-long-csv", default=None)
    ap.add_argument("--proxy-csv", default=None)
    ap.add_argument("--item-cols", default=None)
    ap.add_argument("--group-cols", default=None)
    ap.add_argument("--rater-col", default=None)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--proxy-item-col", default=None)
    ap.add_argument("--proxy-score-col", default=None)
    ap.add_argument("--cal-fracs", default=None, help="Comma-separated calibration fractions.")
    ap.add_argument("--n-repeats", type=int, default=None, help="Resamples per path per cal_frac.")
    ap.add_argument("--min-items", type=int, default=None)
    ap.add_argument("--min-raters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed (fresh stream per cal_frac).")
    ap.add_argument("--extended", action="store_true", help="Add theory_semi / iso_cal / iso_eval predictors.")
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--n-grid", type=int, default=120, help="Tilt-curve t-grid points.")
    ap.add_argument("--iae-points", type=int, default=40, help="d-grid points for the IAE metric.")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="Bootstrap replicates (default 1000; 500 for the wmt track, matching the reference run).")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out-name", default=None, help="Merged summary CSV name (default merged_<track>.csv).")
    ap.add_argument("--base-out-name", default=None,
                    help="With --extended: also write a base-schema (non-extended) summary under this name.")
    ap.add_argument("--splits-name", default=None, help="Per-split CSV name (default splits_<track>.csv).")
    ap.add_argument("--no-splits", action="store_true", help="Skip writing the per-split CSV.")
    args = ap.parse_args()

    defaults = TRACKS[args.track]
    for key in ("human_long_csv", "proxy_csv", "item_cols", "group_cols", "rater_col", "score_col",
                "proxy_item_col", "proxy_score_col", "cal_fracs", "n_repeats", "min_items"):
        if getattr(args, key, None) is None:
            setattr(args, key, defaults.get(key, ""))
    if args.n_boot is None:
        args.n_boot = int(defaults.get("n_boot", 1000))
    # resolve data paths
    data_root = Path(args.data_root)
    if args.human_long_csv and not Path(args.human_long_csv).is_absolute() and not Path(args.human_long_csv).exists():
        args.human_long_csv = str(data_root / args.human_long_csv)
    if args.proxy_csv and not Path(args.proxy_csv).is_absolute() and not Path(args.proxy_csv).exists():
        args.proxy_csv = str(data_root / args.proxy_csv)

    setting = defaults["setting"]
    cal_fracs = [float(x) for x in parse_cols(args.cal_fracs)]
    predictors = EXT_PREDICTORS if args.extended else BASE_PREDICTORS
    diffs = EXT_DIFFS if args.extended else BASE_DIFFS
    d_grid = np.arange(1, args.iae_points + 1) * args.fit_max_d / args.iae_points

    outdir = mkdir(args.outdir)
    paths = build_paths(args)
    if not paths:
        raise RuntimeError("No admissible paths were generated.")
    print(f"[exp03:{args.track}] {len(paths)} paths, fracs={cal_fracs}, n_repeats={args.n_repeats}, "
          f"extended={args.extended}, seed={args.seed}")

    summary_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    skipped_total = 0
    for frac in cal_fracs:
        rng = np.random.default_rng(args.seed)  # fresh stream per cal_frac
        rows: List[Dict[str, Any]] = []
        n_skipped = 0
        for path in paths:
            n = len(path.item_ids)
            if n < 30:
                continue
            for rep in range(args.n_repeats):
                cal_idx, eva_idx = split_indices(n, frac, rng)
                try:
                    row = evaluate_one_split(path, cal_idx, eva_idx, d_grid,
                                             args.fit_max_d, args.n_grid, args.extended)
                except (RuntimeError, ValueError):
                    n_skipped += 1
                    continue
                row["cal_frac"] = frac
                row["rep"] = rep
                row["seed"] = args.seed
                rows.append(row)
        skipped_total += n_skipped
        df = pd.DataFrame(rows)
        split_rows.extend(rows)
        summary_rows.append(
            aggregate_frac(df, setting, frac, n_skipped, predictors, diffs, args.n_boot, args.boot_seed)
        )
        print(f"  cal_frac={frac}: n_splits={len(df)} n_skipped={n_skipped} "
              f"mae_theory={summary_rows[-1]['abs_err_theory_mean']:.4f}")

    summary = pd.DataFrame(summary_rows)
    out_name = args.out_name or f"merged_{args.track}.csv"
    summary.to_csv(outdir / out_name, index=False)
    print(f"Saved {outdir / out_name}")

    if args.extended and args.base_out_name:
        # base (non-extended) schema: all abs_err first, then iae, then diffs
        ordered = (["setting", "cal_frac", "n_splits", "n_skipped", "median_n_cal"]
                   + [f"abs_err_{p}_{s}" for p in BASE_PREDICTORS for s in ("mean", "lo", "hi")]
                   + [f"iae_{p}_{s}" for p in BASE_PREDICTORS for s in ("mean", "lo", "hi")]
                   + [f"diff_theory_minus_{p}_{s}" for p in BASE_DIFFS for s in ("mean", "lo", "hi")])
        summary[ordered].to_csv(outdir / args.base_out_name, index=False)
        print(f"Saved {outdir / args.base_out_name}")

    if not args.no_splits:
        splits_df = pd.DataFrame(split_rows)
        front = ["group_key", "path_id", "n_cal", "n_eval", "actual_delta"]
        metric = []
        for p in predictors:
            metric += [f"iae_{p}", f"abs_err_{p}"]
        splits_df = splits_df[front + metric + ["cal_frac", "rep", "seed"]]
        splits_name = args.splits_name or f"splits_{args.track}.csv"
        splits_df.to_csv(outdir / splits_name, index=False)
        print(f"Saved {outdir / splits_name}")

    spec = {
        "script": "experiments/exp03_calibration_crossover/01_calibration_crossover.py",
        "track": args.track,
        "setting": setting,
        "human_long_csv": args.human_long_csv,
        "proxy_csv": args.proxy_csv or None,
        "item_cols": parse_cols(args.item_cols),
        "group_cols": parse_cols(args.group_cols),
        "rater_col": args.rater_col,
        "score_col": args.score_col,
        "proxy_item_col": args.proxy_item_col,
        "proxy_score_col": args.proxy_score_col,
        "cal_fracs": cal_fracs,
        "n_repeats": args.n_repeats,
        "min_items": args.min_items,
        "min_raters": args.min_raters,
        "seed": args.seed,
        "extended": args.extended,
        "fit_max_d": args.fit_max_d,
        "n_grid": args.n_grid,
        "iae_points": args.iae_points,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "n_paths": len(paths),
        "n_split_rows": len(split_rows),
        "n_skipped_total": skipped_total,
        "predictors": list(predictors),
    }
    (outdir / f"run_spec_{args.track}.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
