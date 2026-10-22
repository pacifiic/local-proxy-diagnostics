"""Generate table exports for the accept-boost experiments.

Run after ``bash run_accept_boost_experiments.sh``:

    python make_accept_boost_tables.py \
      --root all_results_reproduced/12_accept_boost \
      --out tables

The script is intentionally tolerant of missing result directories: it writes
what it can find and skips the rest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fmt(x, nd=4):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def fmt_ci(row, col, nd=4):
    val = row.get(col, float("nan"))
    lo = row.get(f"{col}_ci_lo", float("nan"))
    hi = row.get(f"{col}_ci_hi", float("nan"))
    if pd.isna(val):
        return "--"
    if pd.isna(lo) or pd.isna(hi):
        return fmt(val, nd)
    return f"{float(val):.{nd}f} [{float(lo):.{nd}f}, {float(hi):.{nd}f}]"


def pct(x, nd=1):
    if pd.isna(x):
        return "--"
    return f"{100*float(x):.{nd}f}%"


def latex_cell(x):
    if not isinstance(x, str):
        return x
    # Preserve intentional math fragments such as $A_\nu$ and $d$ while still
    # escaping percent signs in formatted numeric cells.
    if "$" in x:
        return x.replace("%", r"\%")
    return (
        x.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def write_table(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    try:
        (outdir / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        (outdir / f"{stem}.md").write_text(df.to_string(index=False), encoding="utf-8")
    latex_df = df.copy()
    latex_df.columns = [latex_cell(str(col)) for col in latex_df.columns]
    for col in latex_df.columns:
        if latex_df[col].dtype == object:
            latex_df[col] = latex_df[col].map(latex_cell)
    (outdir / f"{stem}.tex").write_text(
        latex_df.to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def d_sweep_table(path: Path, setting: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values("d").iterrows():
        valid_paths = r.get("valid_paths", r.get("n_paths", float("nan")))
        valid_rows = r.get("valid_rows", r.get("n_rows", float("nan")))
        rows.append({
            "Setting": setting,
            "$d$": fmt(r["d"], 3),
            "Valid paths": "--" if pd.isna(valid_paths) else str(int(valid_paths)),
            "Valid rows": "--" if pd.isna(valid_rows) else str(int(valid_rows)),
            "A+B MAE": fmt_ci(r, "mae_theory"),
            "A-only MAE": fmt_ci(r, "mae_aonly"),
            "Zero MAE": fmt_ci(r, "mae_zero"),
            "$A_\\nu$ MAE": fmt_ci(r, "mae_anu"),
            "A+B sign acc.": pct(r.get("sign_acc_theory")),
        })
    return pd.DataFrame(rows)


def residual_table(path: Path, setting: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    predictor_order = ["theory_ab", "aonly", "fit_ab", "anu", "zero", "fit_abc"]
    df = df[df["group_key"].eq("__OVERALL__") & df["predictor"].isin(predictor_order)].copy()
    df["predictor"] = pd.Categorical(df["predictor"], categories=predictor_order, ordered=True)
    rows = []
    for _, r in df.sort_values(["D", "predictor"]).iterrows():
        rows.append({
            "Setting": setting,
            "$D$": fmt(r["D"], 2),
            "Paths": str(int(r["n_paths"])) if not pd.isna(r.get("n_paths")) else "--",
            "Predictor": r["predictor"],
            "Median max abs resid": fmt(r["median_max_abs_resid"]),
            "90% max abs resid": fmt(r["q90_max_abs_resid"]),
            "Median RMSE": fmt(r["median_rmse_resid"]),
            "Median $R^2$": fmt(r["median_r2"], 5),
        })
    return pd.DataFrame(rows)


def cal_size_table(path: Path, setting: str, d: float = 0.10) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["d"].round(8).eq(round(d, 8))].copy()
    rows = []
    for _, r in df.sort_values("calibration_frac").iterrows():
        rows.append({
            "Setting": setting,
            "Calibration": pct(r["calibration_frac"], 0),
            "A+B MAE": fmt_ci(r, "mae_theory"),
            "A-only MAE": fmt_ci(r, "mae_aonly"),
            "Zero MAE": fmt_ci(r, "mae_zero"),
            "$A_\\nu$ MAE": fmt_ci(r, "mae_anu"),
        })
    return pd.DataFrame(rows)


def selective_table(path: Path, setting: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["predictor"].eq("theory")].copy()
    rows = []
    for _, r in df.sort_values("threshold").iterrows():
        rows.append({
            "Setting": setting,
            "$\\tau$": fmt(r["threshold"], 4),
            "Coverage": pct(r["coverage"]),
            "Sign acc.": pct(r["sign_acc_covered"]),
            "MAE covered": fmt_ci(r, "mae_covered"),
            "Policy gain": fmt_ci(r, "policy_realized_gain"),
            "Bad opt. rate": pct(r["bad_optimization_rate"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("all_results_reproduced/12_accept_boost"))
    ap.add_argument("--out", type=Path, default=Path("paper_tables_accept_boost"))
    args = ap.parse_args()
    outdir = mkdir(args.out)
    R = args.root

    tables = []
    for setting, rel in [
        ("Human-heldout", "d_sweep_human/d_sweep_summary_overall.csv"),
        ("GPT-4o proxy", "d_sweep_gpt4o/d_sweep_summary_overall.csv"),
    ]:
        t = d_sweep_table(R / rel, setting)
        if t is not None:
            tables.append(t)
    if tables:
        write_table(pd.concat(tables, ignore_index=True), outdir, "table_d_sweep")

    tables = []
    for setting, rel in [
        ("Human-heldout", "residual_human/curve_residual_summary.csv"),
        ("GPT-4o proxy", "residual_gpt4o/curve_residual_summary.csv"),
    ]:
        t = residual_table(R / rel, setting)
        if t is not None:
            tables.append(t)
    if tables:
        write_table(pd.concat(tables, ignore_index=True), outdir, "table_curve_residuals")

    tables = []
    for setting, rel in [
        ("Human-heldout", "cal_size_human/calibration_size_summary_overall.csv"),
        ("GPT-4o proxy", "cal_size_gpt4o/calibration_size_summary_overall.csv"),
    ]:
        t = cal_size_table(R / rel, setting, d=0.10)
        if t is not None:
            tables.append(t)
    if tables:
        write_table(pd.concat(tables, ignore_index=True), outdir, "table_calibration_size_d0p1")

    t = selective_table(R / "selective_gpt4o/selective_prediction_gate_summary.csv", "GPT-4o proxy")
    if t is not None:
        write_table(t, outdir, "table_selective_gpt4o")

    print(f"Wrote accept-boost tables to: {outdir}")


if __name__ == "__main__":
    main()
