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

DIMS = [
    "fluency",
    "clarity",
    "conciseness",
    "relevance",
    "consistency",
    "answerability",
    "answer_consistency",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QGEval human direct-tilt experiment")
    p.add_argument("--annotator-files", required=True, help="Comma-separated annotator xlsx paths")
    p.add_argument("--outdir", required=True)
    p.add_argument("--gold-estimator", default="loo_mean", choices=["loo_mean", "same_mean", "median_others", "trimmed_mean_others"])
    p.add_argument("--fit-max-d", type=float, default=0.10)
    p.add_argument("--score-transform", default="raw", choices=["raw", "rater_z"])
    p.add_argument("--t-max", type=float, default=5.0)
    p.add_argument("--n-grid", type=int, default=401)
    return p.parse_args()


def load_qgeval_long(paths: list[str]) -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(paths, start=1):
        df = pd.read_excel(path)
        df["item_id"] = df["id"].astype(str) + "___" + df["source"].astype(str)
        df["dataset"] = df["source"].astype(str).str.split("_").str[0]
        df["rater_idx"] = idx
        long = df.melt(
            id_vars=["item_id", "dataset", "source", "rater_idx"],
            value_vars=DIMS,
            var_name="dimension",
            value_name="score",
        )
        rows.append(long)
    out = pd.concat(rows, ignore_index=True)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"])
    return out


def zscore_within_rater(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(["dataset", "dimension", "rater_idx"])["score"]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    df["score"] = ((df["score"] - mu) / sd).fillna(0.0)
    return df


def build_matrix(df: pd.DataFrame, dataset: str, dimension: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["dimension"] == dimension].copy()
    if dataset != "pooled":
        sub = sub[sub["dataset"] == dataset].copy()
    X = sub.pivot_table(index="item_id", columns="rater_idx", values="score", aggfunc="first")
    X = X.dropna(axis=0, how="any").dropna(axis=1, how="all")
    return X.to_numpy(dtype=float), X.index.to_numpy(), X.columns.to_numpy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, dimension), g in df.groupby(["dataset_eval", "dimension"]):
        out.append({
            "dataset_eval": dataset,
            "dimension": dimension,
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

    paths = [x.strip() for x in args.annotator_files.split(",") if x.strip()]
    if len(paths) < 2:
        raise ValueError("Need at least two annotator files.")
    df = load_qgeval_long(paths)
    if args.score_transform == "rater_z":
        df = zscore_within_rater(df)

    datasets = sorted(df["dataset"].unique().tolist()) + ["pooled"]
    t_grid = np.linspace(0.0, args.t_max, args.n_grid)

    rows = []
    trajs = []
    for dataset in datasets:
        for dim in DIMS:
            X, item_ids, rater_ids = build_matrix(df, dataset, dim)
            if X.shape[0] < 10 or X.shape[1] < 2:
                continue
            icc = icc_a1(X)
            for j, rater_id in enumerate(rater_ids):
                J, r = gold_from_matrix(X, j, estimator=args.gold_estimator)
                ab = compute_ab(J, r)
                d_vals, q_vals = tilt_curve(J, r, t_grid)
                A_fit, B_fit, r2 = fit_local_ab(d_vals, q_vals, max_d=args.fit_max_d)
                rows.append({
                    "dataset_eval": dataset,
                    "dimension": dim,
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
                })
                for t, d, q in zip(t_grid, d_vals, q_vals):
                    trajs.append({
                        "dataset_eval": dataset,
                        "dimension": dim,
                        "heldout_rater": rater_id,
                        "t": t,
                        "d": d,
                        "q": q,
                        "q_delta": q - q_vals[0],
                    })

    df_rows = pd.DataFrame(rows).sort_values(["dataset_eval", "dimension", "heldout_rater"])
    df_traj = pd.DataFrame(trajs)
    df_summary = summarize(df_rows)

    df_rows.to_csv(outdir / "table_qgeval_human_direct_tilt.csv", index=False)
    df_traj.to_csv(outdir / "table_qgeval_human_direct_tilt_traj.csv", index=False)
    df_summary.to_csv(outdir / "table_qgeval_human_direct_tilt_summary.csv", index=False)

    report = [
        "# QGEval human direct tilt\n",
        f"- score_transform: `{args.score_transform}`\n",
        f"- gold_estimator: `{args.gold_estimator}`\n",
        f"- fit_max_d: `{args.fit_max_d}`\n",
        "\n## Summary\n",
        df_summary.to_markdown(index=False),
        "\n\n## Notes\n",
        "- QGEval has 3 annotators; each path uses one-vs-rest (average of 2 as gold).\n",
        "- All three slices (`SQuAD`, `HotpotQA`, `pooled`) are saved.\n",
        "- The main finding is the per-dimension ICC / B-sign pattern rather than cross-dataset differences.\n",
    ]
    (outdir / "report_qgeval_human_direct_tilt.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
