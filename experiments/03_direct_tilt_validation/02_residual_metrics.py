"""  [Table 1 supplement]
Compute residual metrics for direct coefficient validation.

Reads trajectory CSVs (point-by-point tilt curves) and computes:
- A+B vs A-only R² comparison (showing B's contribution)
- Max and mean absolute residuals for both models
- Residual breakdown by d range

Usage:
    python 02_residual_metrics.py \
        --results-dir RESULTS \
        --outdir RESULTS/03_direct_tilt_validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_ab(d: np.ndarray, q_delta: np.ndarray):
    """Fit Q(d)-Q(0) = A*sqrt(d) + B*d. Returns (A, B, r2, residuals, d_used)."""
    mask = (d > 0) & np.isfinite(q_delta)
    d, q = d[mask], q_delta[mask]
    X = np.column_stack([np.sqrt(d), d])
    coeff, _, _, _ = np.linalg.lstsq(X, q, rcond=None)
    pred = X @ coeff
    ss_res = ((q - pred) ** 2).sum()
    ss_tot = ((q - q.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coeff[0], coeff[1], r2, q - pred, d


def fit_a_only(d: np.ndarray, q_delta: np.ndarray):
    """Fit Q(d)-Q(0) = A*sqrt(d). Returns (A, r2, residuals)."""
    mask = (d > 0) & np.isfinite(q_delta)
    d, q = d[mask], q_delta[mask]
    X = np.sqrt(d).reshape(-1, 1)
    coeff, _, _, _ = np.linalg.lstsq(X, q, rcond=None)
    pred = (X @ coeff).ravel()
    ss_res = ((q - pred) ** 2).sum()
    ss_tot = ((q - q.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coeff[0], r2, q - pred


def analyze_traj(tdf: pd.DataFrame, ds_name: str, path_cols: list[str],
                 fit_max_d: float = 0.10) -> list[dict]:
    q_col = "q_delta"
    if q_col not in tdf.columns:
        for key, gdf in tdf.groupby(path_cols):
            q0 = gdf.loc[gdf["d"].idxmin(), "q"]
            tdf.loc[gdf.index, q_col] = gdf["q"] - q0

    rows = []
    for key, gdf in tdf.groupby(path_cols):
        d = gdf["d"].values
        q_delta = gdf[q_col].values
        mask = d <= fit_max_d
        d_local, q_local = d[mask], q_delta[mask]
        if len(d_local) < 5:
            continue

        A_ab, B_ab, r2_ab, resid_ab, d_fit = fit_ab(d_local, q_local)
        A_a, r2_a, resid_a = fit_a_only(d_local, q_local)

        rows.append({
            "dataset": ds_name,
            "path": str(key),
            "n_points": len(d_fit),
            "R2_AB": r2_ab,
            "R2_A_only": r2_a,
            "R2_drop": r2_ab - r2_a,
            "max_abs_resid_AB": np.abs(resid_ab).max(),
            "max_abs_resid_A_only": np.abs(resid_a).max(),
            "mean_abs_resid_AB": np.abs(resid_ab).mean(),
            "mean_abs_resid_A_only": np.abs(resid_a).mean(),
            "A_fit": A_ab,
            "B_fit": B_ab,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Root results directory (e.g. all_results_reproduced)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--fit-max-d", type=float, default=0.10)
    args = ap.parse_args()

    R = args.results_dir
    traj_specs = [
        ("OpenMEVA", R / "04_cross_dataset" / "table_openmeva_direct_tilt_traj.csv",
         ["dataset_eval", "heldout_rater"]),
        ("WMT MQM", R / "05_wmt_mtbench" / "exp19_wmt_mqm" / "wmt_mqm_traj.csv",
         ["dataset_id", "heldout_rater"]),
        ("QGEval-human", R / "04_cross_dataset" / "table_qgeval_human_direct_tilt_traj.csv",
         ["dataset_eval", "heldout_rater"]),
    ]

    all_rows = []
    for ds_name, traj_path, path_cols in traj_specs:
        if not traj_path.exists():
            print(f"  {ds_name}: {traj_path} not found, skipping")
            continue
        tdf = pd.read_csv(traj_path)
        rows = analyze_traj(tdf, ds_name, path_cols, fit_max_d=args.fit_max_d)
        all_rows.extend(rows)
        print(f"  {ds_name}: {len(rows)} paths analyzed")

    df = pd.DataFrame(all_rows)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "direct_validation_residual_metrics.csv"
    df.to_csv(out_csv, index=False)

    # Print summary
    print(f"\nSummary (d <= {args.fit_max_d}):")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"  {ds}: {len(sub)} paths, "
              f"med R2(A+B)={sub['R2_AB'].median():.6f}, "
              f"med R2(A)={sub['R2_A_only'].median():.6f}, "
              f"med max|resid|={sub['max_abs_resid_AB'].median():.6f}")

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
