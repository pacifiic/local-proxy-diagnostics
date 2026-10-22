#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verif_experiment_utils import compute_ab, fit_local_ab, tilt_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RewardBench margin-geometry experiment")
    p.add_argument("--scores-csv", required=True, help="Path to rewardbench_scores.csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--pseudo-gold", default="loo_mean", choices=["loo_mean", "other_family_mean"])
    p.add_argument("--score-transform", default="model_z", choices=["raw", "model_z"])
    p.add_argument("--fit-max-d", type=float, default=0.10)
    p.add_argument("--t-max", type=float, default=4.0)
    p.add_argument("--n-grid", type=int, default=241)
    p.add_argument("--min-items", type=int, default=40)
    p.add_argument(
        "--missingness-mode",
        default="complete_case",
        choices=["complete_case", "pairwise_available"],
        help="complete_case preserves the legacy behavior; pairwise_available keeps each model's available margins.",
    )
    p.add_argument(
        "--include-missingness-counts",
        action="store_true",
        help="Include raw/matrix/used item-count diagnostics in output tables.",
    )
    return p.parse_args()


def zscore_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        mu = X[c].mean()
        sd = X[c].std()
        if pd.isna(sd) or sd == 0:
            X[c] = X[c].where(X[c].isna(), 0.0)
        else:
            X[c] = (X[c] - mu) / sd
    return X


def family_of_model(model: str) -> str:
    return str(model).split("/")[0]


def build_subset_pivot(df: pd.DataFrame, subset: str, missingness_mode: str) -> tuple[pd.DataFrame, dict[str, int]]:
    sub = df if subset == "pooled" else df[df["subset"] == subset]
    X = sub.pivot_table(index="item_id", columns="model", values="margin", aggfunc="first")
    raw_items, raw_models = X.shape
    if missingness_mode == "complete_case":
        X = X.dropna(axis=0, how="any").dropna(axis=1, how="all")
    elif missingness_mode == "pairwise_available":
        X = X.dropna(axis=0, how="all").dropna(axis=1, how="all")
    else:
        raise ValueError(f"Unknown missingness_mode: {missingness_mode}")
    return X, {
        "n_items_raw": int(raw_items),
        "n_models_raw": int(raw_models),
        "n_items_matrix": int(X.shape[0]),
        "n_models_matrix": int(X.shape[1]),
    }


def get_gold_and_proxy(
    X: pd.DataFrame,
    model: str,
    pseudo_gold: str,
    families: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    others = [m for m in X.columns if m != model]
    if pseudo_gold == "loo_mean":
        if not others:
            return None
        other_X = X[others]
        min_gold_models = 1
    else:
        fam = families[model]
        other_fam_models = [m for m in others if families[m] != fam]
        if len(other_fam_models) < 2:
            return None
        other_X = X[other_fam_models]
        min_gold_models = 2
    J_series = other_X.mean(axis=1, skipna=True)
    n_gold_models = other_X.notna().sum(axis=1)
    r_series = X[model]
    mask = (
        r_series.notna()
        & J_series.notna()
        & (n_gold_models >= min_gold_models)
    )
    if not bool(mask.any()):
        return None
    J = J_series[mask].to_numpy(dtype=float)
    r = r_series[mask].to_numpy(dtype=float)
    counts = n_gold_models[mask]
    meta = {
        "n_items_used": int(mask.sum()),
        "min_gold_models_per_item": int(counts.min()),
        "median_gold_models_per_item": float(counts.median()),
    }
    return J, r, meta


def summarize(df: pd.DataFrame, include_counts: bool = False) -> pd.DataFrame:
    out = []
    for subset, g in df.groupby("subset_eval"):
        row = {
            "subset_eval": subset,
            "n_models": len(g),
            "frac_B_theory_neg": float((g["B_theory"] < 0).mean()),
            "frac_B_fit_neg": float((g["B_fit"] < 0).mean()),
            "corr_theory_fit_B": np.corrcoef(g["B_theory"], g["B_fit"])[0, 1] if len(g) > 1 else np.nan,
            "median_r2": float(g["local_r2"].median()),
            "mean_corr_J_r": float(g["corr_J_r"].mean()),
        }
        if include_counts and "n_items_used" in g.columns:
            row.update({
                "median_n_items_used": float(g["n_items_used"].median()),
                "min_n_items_used": int(g["n_items_used"].min()),
                "max_n_items_used": int(g["n_items_used"].max()),
                "n_items_matrix": int(g["n_items_matrix"].max()),
                "n_models_matrix": int(g["n_models_matrix"].max()),
            })
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores_csv)
    req = {"item_id", "model", "margin", "subset"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.copy()
    df["family"] = df["model"].map(family_of_model)

    subsets = sorted(df["subset"].unique().tolist()) + ["pooled"]
    t_grid = np.linspace(0.0, args.t_max, args.n_grid)
    rows = []
    trajs = []

    for subset in subsets:
        X, matrix_meta = build_subset_pivot(df, subset, args.missingness_mode)
        if X.shape[0] < args.min_items or X.shape[1] < 3:
            continue
        if args.score_transform == "model_z":
            X = zscore_columns(X)
        families = {m: family_of_model(m) for m in X.columns}
        for model in X.columns:
            pair = get_gold_and_proxy(X, model, args.pseudo_gold, families)
            if pair is None:
                continue
            J, r, used_meta = pair
            if len(J) < args.min_items:
                continue
            ab = compute_ab(J, r)
            d_vals, q_vals = tilt_curve(J, r, t_grid)
            A_fit, B_fit, r2 = fit_local_ab(d_vals, q_vals, max_d=args.fit_max_d)
            row = {
                "subset_eval": subset,
                "model": model,
                "family": families[model],
                "n_items": len(J),
                "fit_max_d": args.fit_max_d,
                "A_theory": ab["A"],
                "B_theory": ab["B"],
                "A_fit": A_fit,
                "B_fit": B_fit,
                "local_r2": r2,
                "corr_J_r": float(np.corrcoef(J, r)[0, 1]) if len(J) > 1 else np.nan,
                "rho": ab["rho"],
                "nu_tilde": ab["nu_tilde"],
            }
            if args.include_missingness_counts:
                row.update(matrix_meta)
                row.update(used_meta)
            rows.append(row)
            for t, d, q in zip(t_grid, d_vals, q_vals):
                trajs.append({
                    "subset_eval": subset,
                    "model": model,
                    "t": t,
                    "d": d,
                    "q": q,
                    "q_delta": q - q_vals[0],
                })

    df_rows = pd.DataFrame(rows).sort_values(["subset_eval", "model"])
    df_traj = pd.DataFrame(trajs)
    df_summary = summarize(df_rows, include_counts=args.include_missingness_counts)

    df_rows.to_csv(outdir / "table_rewardbench_margin_geometry.csv", index=False)
    df_traj.to_csv(outdir / "table_rewardbench_margin_geometry_traj.csv", index=False)
    df_summary.to_csv(outdir / "table_rewardbench_margin_geometry_summary.csv", index=False)
    spec = {
        "scores_csv": str(Path(args.scores_csv)),
        "pseudo_gold": args.pseudo_gold,
        "score_transform": args.score_transform,
        "fit_max_d": args.fit_max_d,
        "t_max": args.t_max,
        "n_grid": args.n_grid,
        "min_items": args.min_items,
        "missingness_mode": args.missingness_mode,
        "include_missingness_counts": bool(args.include_missingness_counts),
        "n_rows": int(len(df_rows)),
        "n_trajectory_rows": int(len(df_traj)),
    }
    (outdir / "rewardbench_margin_geometry_run_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = [
        "# RewardBench margin geometry\n",
        f"- pseudo_gold: `{args.pseudo_gold}`\n",
        f"- score_transform: `{args.score_transform}`\n",
        f"- missingness_mode: `{args.missingness_mode}`\n",
        f"- fit_max_d: `{args.fit_max_d}`\n",
        "\n## Summary\n",
        df_summary.to_markdown(index=False),
        "\n\n## Notes\n",
        "- This experiment is cross-model ensemble geometry, not human-gold theorem validation.\n",
        "- This is better suited for the appendix or robustness section than the main text.\n",
        "- `other_family_mean` is more conservative than `loo_mean` and reduces the same-family leakage objection.\n",
    ]
    (outdir / "report_rewardbench_margin_geometry.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
