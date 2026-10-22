#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from verif_experiment_utils import compute_ab, fit_local_ab, gold_from_matrix, icc_a1, tilt_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMEVA human direct-tilt experiment")
    p.add_argument("--data", required=True, help="Path to openmeva_long.csv")
    p.add_argument("--outdir", required=True)
    p.add_argument("--gold-estimator", default="loo_mean", choices=["loo_mean", "same_mean", "median_others", "trimmed_mean_others"])
    p.add_argument("--fit-max-d", type=float, default=0.10)
    p.add_argument("--score-transform", default="raw", choices=["raw", "rater_z"])
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--n-grid", type=int, default=401)
    return p.parse_args()


def zscore_within_rater(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(["dataset", "rater_idx"])["score"]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    df["score"] = ((df["score"] - mu) / sd).fillna(0.0)
    return df


def build_matrix(df: pd.DataFrame, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df.copy()
    if dataset != "pooled":
        sub = sub[sub["dataset"] == dataset].copy()
    sub["item_id"] = sub["dataset"].astype(str) + "___" + sub["story_id"].astype(str) + "___" + sub["model_id"].astype(str)
    X = sub.pivot_table(index="item_id", columns="rater_idx", values="score", aggfunc="first")
    X = X.dropna(axis=0, how="any").dropna(axis=1, how="all")
    return X.to_numpy(dtype=float), X.index.to_numpy(), X.columns.to_numpy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dataset, g in df.groupby("dataset_eval"):
        out.append({
            "dataset_eval": dataset,
            "n_paths": len(g),
            "corr_A": np.corrcoef(g["A_theory"], g["A_fit"])[0, 1] if len(g) > 1 else np.nan,
            "corr_B": np.corrcoef(g["B_theory"], g["B_fit"])[0, 1] if len(g) > 1 else np.nan,
            "B_sign_agree": int((np.sign(g["B_theory"]) == np.sign(g["B_fit"])).sum()),
            "median_r2": float(g["local_r2"].median()),
            "mean_icc_a1": float(g["icc_a1"].mean()),
            "frac_B_theory_neg": float((g["B_theory"] < 0).mean()),
            "frac_B_fit_neg": float((g["B_fit"] < 0).mean()),
        })
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    required = {"dataset", "story_id", "model_id", "rater_idx", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if args.score_transform == "rater_z":
        df = zscore_within_rater(df)

    datasets = sorted(df["dataset"].unique().tolist()) + ["pooled"]
    t_grid = np.linspace(0.0, args.t_max, args.n_grid)

    rows = []
    trajs = []
    for dataset in datasets:
        X, item_ids, rater_ids = build_matrix(df, dataset)
        icc = icc_a1(X)
        for j, rater_id in enumerate(rater_ids):
            J, r = gold_from_matrix(X, j, estimator=args.gold_estimator)
            ab = compute_ab(J, r)
            d_vals, q_vals = tilt_curve(J, r, t_grid)
            A_fit, B_fit, r2 = fit_local_ab(d_vals, q_vals, max_d=args.fit_max_d)
            row = {
                "dataset_eval": dataset,
                "heldout_rater": rater_id,
                "n_items": len(J),
                "icc_a1": icc,
                "fit_max_d": args.fit_max_d,
                "A_theory": ab["A"],
                "B_theory": ab["B"],
                "A_fit": A_fit,
                "B_fit": B_fit,
                "local_r2": r2,
                "rho": ab["rho"],
                "nu_tilde": ab["nu_tilde"],
                "sigma_J_sq": ab["sigma_J_sq"],
                "sigma_r_sq": ab["sigma_r_sq"],
                "kappa_r": ab["kappa_r"],
                "kappa_Jrr": ab["kappa_Jrr"],
            }
            rows.append(row)
            for t, d, q in zip(t_grid, d_vals, q_vals):
                trajs.append({
                    "dataset_eval": dataset,
                    "heldout_rater": rater_id,
                    "t": t,
                    "d": d,
                    "q": q,
                    "q_delta": q - q_vals[0],
                })

    df_rows = pd.DataFrame(rows).sort_values(["dataset_eval", "heldout_rater"])
    df_traj = pd.DataFrame(trajs)
    df_summary = summarize(df_rows)

    df_rows.to_csv(outdir / "table_openmeva_direct_tilt.csv", index=False)
    df_traj.to_csv(outdir / "table_openmeva_direct_tilt_traj.csv", index=False)
    df_summary.to_csv(outdir / "table_openmeva_direct_tilt_summary.csv", index=False)

    report = [
        "# OpenMEVA direct tilt experiment\n",
        f"- score_transform: `{args.score_transform}`\n",
        f"- gold_estimator: `{args.gold_estimator}`\n",
        f"- fit_max_d: `{args.fit_max_d}`\n",
        "\n## Summary\n",
        df_summary.to_markdown(index=False),
        "\n\n## Notes\n",
        "- `roc`, `wp`, and `pooled` are treated as separate task slices.\n",
        "- Each path uses held-out rater one-vs-rest exact tilt.\n",
        "- OpenMEVA has a single overall score, so dataset decomposition is the focus instead of dimension decomposition.\n",
    ]
    (outdir / "report_openmeva_direct_tilt.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
