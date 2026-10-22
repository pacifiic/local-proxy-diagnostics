#!/usr/bin/env python3
"""Build compact tables for the robustness experiments.

The script is intentionally read-only with respect to experiment outputs.  It
summarizes existing prospective prediction runs plus the local-only robustness
runs written under all_results_reproduced/13_robustness.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_LABELS = {
    "exp27_human_baseline": "SummEval human",
    "exp27_wmt_mqm": "WMT MQM clusters",
    "exp27_openmeva": "OpenMEVA-MAGS",
    "exp27_model_proxy_4o": "GPT-4o proxy",
}

OOD_LABELS = {
    "summeval_to_wmt": "SummEval -> WMT clusters",
    "wmt_to_summeval": "WMT clusters -> SummEval",
    "summeval_to_openmeva": "SummEval -> OpenMEVA-MAGS",
}


def fmt(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def pct(x: Any, digits: int = 1) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{100.0 * val:.{digits}f}%"


def ci(row: pd.Series, stem: str, digits: int = 4) -> str:
    val = fmt(row.get(stem), digits)
    lo = row.get(f"{stem.replace('mae_', 'abs_err_')}_ci_lo")
    hi = row.get(f"{stem.replace('mae_', 'abs_err_')}_ci_hi")
    if pd.notna(lo) and pd.notna(hi):
        return f"{val} [{fmt(lo, digits)}, {fmt(hi, digits)}]"
    return val


def latex_cell(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    return x.replace(" -> ", r" $\rightarrow$ ").replace("%", r"\%")


def write_table(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    latex_df = df.copy()
    for col in latex_df.columns:
        if latex_df[col].dtype == object:
            latex_df[col] = latex_df[col].map(latex_cell)
    (outdir / f"{stem}.tex").write_text(
        latex_df.to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def overall_row(csv_path: Path, subset_cols: list[str]) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=subset_cols, keep="first")
    mask = np.ones(len(df), dtype=bool)
    for col in subset_cols:
        mask &= df[col].eq("__OVERALL__").to_numpy()
    return df.loc[mask].iloc[0]


def build_multidataset_prospective(results_root: Path) -> pd.DataFrame:
    base = results_root / "07_prospective_prediction"
    rows: list[dict[str, str]] = []
    for run, label in DATASET_LABELS.items():
        row = overall_row(base / run / "exp27_prospective_summary.csv", ["group_key"])
        rows.append(
            {
                "Setting": label,
                "Splits": str(int(row["n_splits"])),
                "A+B MAE @0.10": ci(row, "mae_theory_0p1"),
                "A-only MAE": ci(row, "mae_aonly_0p1"),
                "Zero MAE": ci(row, "mae_zero_0p1"),
                "$A_\\nu$ MAE": ci(row, "mae_anu_0p1"),
                "Sign acc.": pct(row["theory_sign_acc"]),
                "Mean gain": fmt(row["actual_delta_mean_0p1"]),
            }
        )
    return pd.DataFrame(rows)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    tried = "\n  ".join(str(p) for p in paths)
    raise FileNotFoundError(f"None of the expected files exists:\n  {tried}")


def build_bon_scope(results_root: Path, robustness_root: Path) -> pd.DataFrame:
    rows = []
    for candidates, label in [
        (
            [
                results_root / "08_bon_bridge" / "v2_human" / "exp29_bon_bridge_summary.csv",
                robustness_root / "bon_bridge_human" / "exp29_bon_bridge_summary.csv",
            ],
            "SummEval human",
        ),
        (
            [
                results_root / "08_bon_bridge" / "v2_model_proxy_4o" / "exp29_bon_bridge_summary.csv",
                robustness_root / "bon_bridge_gpt4o" / "exp29_bon_bridge_summary.csv",
            ],
            "GPT-4o proxy",
        ),
    ]:
        row = overall_row(first_existing(candidates), ["group_key"])
        rows.append(
            {
                "Setting": label,
                "Paths": str(int(row["n_paths"])),
                "median $d_{BoN}(N=2)$": fmt(row["median_d_bon_at_N2"], 3),
                "Nontriv. BoN in $d\\le0.1$": pct(row["median_share_nontrivial_in_shared_window"]),
                "MAE BoN vs tilt ($d\\le0.25$)": fmt(row["median_mae_bon_vs_tilt_bridge"]),
                "MAE BoN vs A+B ($d\\le0.25$)": fmt(row["median_mae_bon_vs_theory_bridge"]),
            }
        )
    return pd.DataFrame(rows)


def build_ood_lambda(robustness_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(robustness_root.glob("ood_*_lambda_*/exp31_ood_transfer_summary.csv")):
        m = re.search(r"ood_(.*)_lambda_(.*)/", str(path))
        if not m:
            continue
        route = m.group(1)
        lam = m.group(2).replace("p", ".")
        row = overall_row(path, ["target_dataset", "target_group_key"])
        rows.append(
            {
                "Route": OOD_LABELS.get(route, route),
                "$\\lambda$": lam,
                "Splits": str(int(row["n_splits"])),
                "Source-only": fmt(row["mae_source_mean_0p1"]),
                "Target cal.": fmt(row["mae_target_cal_0p1"]),
                "Source-prior": fmt(row["mae_source_prior_0p1"]),
                "$A_\\nu$": fmt(row["mae_target_anu_0p1"]),
                "Zero": fmt(row["mae_zero_0p1"]),
                "Prior sign": pct(row["sign_acc_source_prior_0p1"]),
            }
        )
    order = {v: i for i, v in enumerate(OOD_LABELS.values())}
    out = pd.DataFrame(rows)
    out["_route_order"] = out["Route"].map(order)
    out["_lambda_num"] = out["$\\lambda$"].astype(float)
    out = out.sort_values(["_route_order", "_lambda_num"]).drop(columns=["_route_order", "_lambda_num"])
    return out.reset_index(drop=True)


def build_extended_calibration(robustness_root: Path) -> pd.DataFrame:
    rows = []
    for run, label in [
        ("cal_size_wmt_mqm", "WMT MQM clusters"),
        ("cal_size_openmeva", "OpenMEVA-MAGS"),
    ]:
        df = pd.read_csv(robustness_root / run / "calibration_size_summary_overall.csv")
        df = df[df["d"].eq(0.10)].sort_values("calibration_frac")
        for _, row in df.iterrows():
            rows.append(
                {
                    "Setting": label,
                    "Calibration": f"{100 * float(row['calibration_frac']):.0f}%",
                    "Splits": str(int(row["n_rows"])),
                    "Paths": str(int(row["n_paths"])),
                    "A+B MAE": ci(row, "mae_theory"),
                    "A-only MAE": ci(row, "mae_aonly"),
                    "$A_\\nu$ MAE": ci(row, "mae_anu"),
                    "Zero MAE": ci(row, "mae_zero"),
                }
            )
    return pd.DataFrame(rows)


def build_reachability_audit(results_root: Path) -> pd.DataFrame:
    df = pd.read_csv(results_root / "12_accept_boost" / "d_sweep_human" / "d_sweep_paths_long.csv")
    base = df[df["d"].eq(0.10)].copy()
    rows = []
    for d in [0.15, 0.20]:
        valid = df[df["d"].eq(d) & df["actual_delta"].notna()].copy()
        paths = set(valid["path_id"].unique())
        sub = base[base["path_id"].isin(paths)]
        rows.append(
            {
                "Stress radius": fmt(d, 2),
                "Valid paths": str(len(paths)),
                "Valid rows": str(len(valid)),
                "Mean gain full @0.10": fmt(base["actual_delta"].mean()),
                "Mean gain reachable @0.10": fmt(sub["actual_delta"].mean()),
                "$\\Delta$ gain": fmt(sub["actual_delta"].mean() - base["actual_delta"].mean()),
                "$\\Delta$ cal. A": fmt(sub["cal_a_exact"].mean() - base["cal_a_exact"].mean()),
                "$\\Delta$ cal. B": fmt(sub["cal_b_theory"].mean() - base["cal_b_theory"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="all_results_reproduced")
    ap.add_argument("--robustness-root", default="all_results_reproduced/13_robustness")
    ap.add_argument("--out", default="paper_tables_robustness")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    robustness_root = Path(args.robustness_root)
    outdir = Path(args.out)

    tables = {
        "table_multidataset_prospective": build_multidataset_prospective(results_root),
        "table_bon_scope": build_bon_scope(results_root, robustness_root),
        "table_ood_lambda_sweep": build_ood_lambda(robustness_root),
        "table_calibration_size_extended": build_extended_calibration(robustness_root),
        "table_reachability_audit": build_reachability_audit(results_root),
    }
    for stem, df in tables.items():
        write_table(df, outdir, stem)
        print(f"wrote {outdir / (stem + '.csv')} ({len(df)} rows)")


if __name__ == "__main__":
    main()
