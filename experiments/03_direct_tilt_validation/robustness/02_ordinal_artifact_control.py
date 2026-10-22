#!/usr/bin/env python3
"""  [exp12]Experiment 16: Ordinal/ceiling artifact control.

Goal:
Test whether negative local B can arise purely from quantizing a benign continuous
latent-quality model onto bounded Likert/ordinal scales.

Model family (null control):
    J* ~ Normal(0, latent_sd)
    score_cont = J* + e,   e ~ Normal(0, noise_sd)
    observed score = quantize(score_cont)

We compute corrected Theorem-2 A/B and exact-tilt fitted B under both continuous and
quantized observations using the same LOO construction as the main human-data analysis.

If quantization alone systematically makes B<0, that weakens the interpretation of human
B<0. If it does not, the empirical human result is much stronger.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from pathlib import Path

import numpy as np
import pandas as pd

from verif_experiment_utils import compute_ab, fit_local_ab, gold_from_matrix, tilt_curve


def make_thresholds(k: int, span: float = 2.5) -> np.ndarray:
    # k categories -> k-1 thresholds; symmetric fixed thresholds
    return np.linspace(-span, span, k - 1)


def quantize(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(x, thresholds) + 1


def simulate_one(
    *,
    n_items: int,
    n_raters: int,
    latent_sd: float,
    noise_sd: float,
    scale_points: int,
    fit_max_d: float,
    t_grid: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    J_star = rng.normal(0.0, latent_sd, size=n_items)
    E = rng.normal(0.0, noise_sd, size=(n_items, n_raters))
    X_cont = J_star[:, None] + E
    thresholds = make_thresholds(scale_points)
    X_ord = quantize(X_cont, thresholds).astype(float)

    rows = []
    for obs_name, X in [("continuous", X_cont), ("ordinal", X_ord)]:
        B_theories, B_fits, A_theories, A_fits, rhos = [], [], [], [], []
        for j in range(n_raters):
            J, r = gold_from_matrix(X, j, estimator="loo_mean")
            th = compute_ab(J, r)
            d, q = tilt_curve(J, r, t_grid)
            A_fit, B_fit, _ = fit_local_ab(d, q, fit_max_d)
            B_theories.append(th["B"])
            B_fits.append(B_fit)
            A_theories.append(th["A"])
            A_fits.append(A_fit)
            rhos.append(th["rho"])
        rows.append({
            "obs": obs_name,
            "mean_B_theory": float(np.mean(B_theories)),
            "mean_B_fit": float(np.mean(B_fits)),
            "mean_A_theory": float(np.mean(A_theories)),
            "mean_A_fit": float(np.mean(A_fits)),
            "mean_rho": float(np.mean(rhos)),
            "frac_B_theory_lt_0": float(np.mean(np.array(B_theories) < 0)),
            "frac_B_fit_lt_0": float(np.mean(np.array(B_fits) < 0)),
        })

    out = {
        "continuous_mean_B_theory": rows[0]["mean_B_theory"],
        "continuous_mean_B_fit": rows[0]["mean_B_fit"],
        "continuous_frac_B_theory_lt_0": rows[0]["frac_B_theory_lt_0"],
        "continuous_frac_B_fit_lt_0": rows[0]["frac_B_fit_lt_0"],
        "ordinal_mean_B_theory": rows[1]["mean_B_theory"],
        "ordinal_mean_B_fit": rows[1]["mean_B_fit"],
        "ordinal_frac_B_theory_lt_0": rows[1]["frac_B_theory_lt_0"],
        "ordinal_frac_B_fit_lt_0": rows[1]["frac_B_fit_lt_0"],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-items", type=int, default=1600)
    ap.add_argument("--n-raters", type=int, default=5)
    ap.add_argument("--latent-sd", type=float, default=1.0)
    ap.add_argument("--noise-sds", default="0.25,0.5,0.75,1.0,1.25")
    ap.add_argument("--scale-points", default="5,9")
    ap.add_argument("--n-reps", type=int, default=500)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    ap.add_argument("--t-max", type=float, default=4.0)
    ap.add_argument("--n-t", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    noise_sds = [float(x) for x in args.noise_sds.split(",") if x.strip()]
    scale_points_list = [int(x) for x in args.scale_points.split(",") if x.strip()]
    t_grid = np.concatenate([[0.0], np.geomspace(1e-5, args.t_max, args.n_t)])

    rng = np.random.default_rng(args.seed)
    rows = []
    for k in scale_points_list:
        for noise_sd in noise_sds:
            print(f"[scale={k}] [noise_sd={noise_sd}]")
            rep_rows = []
            for rep in range(args.n_reps):
                rep_out = simulate_one(
                    n_items=args.n_items,
                    n_raters=args.n_raters,
                    latent_sd=args.latent_sd,
                    noise_sd=noise_sd,
                    scale_points=k,
                    fit_max_d=args.fit_max_d,
                    t_grid=t_grid,
                    rng=rng,
                )
                rep_out.update({
                    "rep": rep,
                    "scale_points": k,
                    "noise_sd": noise_sd,
                    "n_items": args.n_items,
                    "n_raters": args.n_raters,
                })
                rep_rows.append(rep_out)
            rep_df = pd.DataFrame(rep_rows)
            rows.append(rep_df)

    all_df = pd.concat(rows, ignore_index=True)
    summary_df = (
        all_df.groupby(["scale_points", "noise_sd"], as_index=False)
        .agg(
            cont_B_theory_mean=("continuous_mean_B_theory", "mean"),
            cont_B_fit_mean=("continuous_mean_B_fit", "mean"),
            cont_Pr_B_theory_lt_0=("continuous_frac_B_theory_lt_0", "mean"),
            cont_Pr_B_fit_lt_0=("continuous_frac_B_fit_lt_0", "mean"),
            ord_B_theory_mean=("ordinal_mean_B_theory", "mean"),
            ord_B_fit_mean=("ordinal_mean_B_fit", "mean"),
            ord_Pr_B_theory_lt_0=("ordinal_frac_B_theory_lt_0", "mean"),
            ord_Pr_B_fit_lt_0=("ordinal_frac_B_fit_lt_0", "mean"),
        )
    )

    all_file = outdir / "table_exp16_ordinal_artifact_reps.csv"
    summ_file = outdir / "table_exp16_ordinal_artifact_summary.csv"
    all_df.to_csv(all_file, index=False)
    summary_df.to_csv(summ_file, index=False)
    print(f"→ saved {all_file}")
    print(f"→ saved {summ_file}")


if __name__ == "__main__":
    main()
