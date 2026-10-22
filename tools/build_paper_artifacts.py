#!/usr/bin/env python3
"""Reaggregate prediction intervals and rebuild artifacts; preserve source predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
PAPER = ROOT / "build/paper-artifacts"
GENERATED_TABLES = PAPER / "generated/tables"
GENERATED_FIGURES = PAPER / "generated/figures"
RESULTS = ROOT / "results"
FIGURES = [
    "fig_main_results", "fig_prediction_mae_multidataset",
    "fig_proxy_perturbation_multidataset", "fig_stability_multidataset",
    "fig_tail_outlier_robustness", "fig6_dataset_replication",
    "fig_kl_radius_audit", "fig_radius_calibration_sensitivity",
    "fig_prompt_conditional_bon_scope",
    "fig1_summeval_icc_by_dimension", "fig3_expert_vs_crowd_lift",
    "fig5_protocol_split_half_stability", "fig_direct_tilt_summeval_tight",
    "fig_bon_bridge_bootstrap", "fig_proxy_perturbation",
    "fig_residual_selective_diagnostics",
]
TABLES = [
    "table_sign_metric_audit", "table_multidataset_prospective",
    "table_calibration_size_extended", "table_ood_lambda_sweep",
    "table_tail_outlier_robustness", "table_d_sweep",
    "table_calibration_size_d0p1", "table_curve_level_metrics",
    "table_prompt_conditional_bon_scope",
    "table_fit_range_sensitivity",
]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=CORE, check=True)


def body(text):
    return text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]


def ood_detail():
    source = RESULTS / "extended_analyses/ood_summeval_to_wmt_lambda_0p75/exp31_ood_transfer_paths.csv"
    data = pd.read_csv(source)
    groups = data.groupby("target_path_id", sort=True)
    assert groups.ngroups == 141 and groups.size().eq(50).all()
    rng = np.random.default_rng(0)
    index = rng.integers(0, groups.ngroups, size=(2000, groups.ngroups))
    rows, tex = [], []
    for key, label in [
        ("source_mean", "source mean"), ("target_cal", "target calibration"),
        ("source_prior", "source prior"), ("target_anu", r"$A_\nu$"),
        ("zero", "zero"),
    ]:
        values = groups[f"abs_err_{key}_0p1"].mean().to_numpy()
        lo, hi = np.quantile(values[index].mean(axis=1), [.025, .975])
        mean05 = data[f"abs_err_{key}_0p05"].mean()
        mean10 = data[f"abs_err_{key}_0p1"].mean()
        sign05 = 100 * data[f"sign_correct_{key}_0p05"].mean()
        sign10 = 100 * data[f"sign_correct_{key}_0p1"].mean()
        rows.append(dict(predictor=key, mae_005=mean05, sign_005=sign05,
                         mae_010=mean10, ci_lo=lo, ci_hi=hi, sign_010=sign10))
        tex.append(
            f"{label} & {mean05:.4f} & {sign05:.1f}\\% & "
            f"{mean10:.4f} [{lo:.4f}, {hi:.4f}] & {sign10:.1f}\\% " + r"\\"
        )
    pd.DataFrame(rows).to_csv(PAPER / "tables/arxiv_ood_detail.csv", index=False)
    (PAPER / "tables/arxiv_ood_detail_rows.tex").write_text(
        "\n".join(tex) + "\n" + r"\bottomrule" + "\n")
    return {"source": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bootstrap_draws": 2000, "seed": 0, "cluster_unit": "target_path_id",
            "interval": "percentile 95%; equal-weight path means over 50 splits"}


def main():
    global PAPER, GENERATED_TABLES, GENERATED_FIGURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--outdir", type=Path, default=PAPER)
    args = parser.parse_args()
    PAPER = args.outdir.resolve()
    GENERATED_TABLES = PAPER / "generated/tables"
    GENERATED_FIGURES = PAPER / "generated/figures"
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(exist_ok=True)
    GENERATED_TABLES.mkdir(parents=True, exist_ok=True)
    GENERATED_FIGURES.mkdir(parents=True, exist_ok=True)
    refresh = [sys.executable, str(ROOT / "tools/recompute_prediction_intervals.py")]
    subprocess.run(refresh if args.sync_only else refresh + ["--apply"], check=True)
    if not args.sync_only:
        run("make_artifacts.py", "all", "--results-root", str(RESULTS),
            "--robustness-root", str(RESULTS / "extended_analyses"),
            "--core-tables-out", str(GENERATED_TABLES),
            "--accept-out", str(GENERATED_TABLES),
            "--metric-audit-out", str(GENERATED_TABLES),
            "--robustness-out", str(GENERATED_TABLES),
            "--figures-out", str(GENERATED_FIGURES))
        run("artifact_tools/make_tail_bon_artifacts.py", "--tail-root",
            "results/tail_outlier", "--bon-root", "results/prompt_bon",
            "--tables-out", str(GENERATED_TABLES), "--figures-out", str(GENERATED_FIGURES))
    for name in TABLES:
        if name == "table_fit_range_sensitivity":
            run("artifact_tools/make_fit_range_table.py", "--outdir", str(GENERATED_TABLES))
        dest = PAPER / "tables" / f"{name}.tex"
        source = (GENERATED_TABLES / f"{name}.tex").read_text()
        old = dest.read_text() if dest.exists() else source
        assert old.count(r"\midrule") == source.count(r"\midrule") == 1
        if name == "table_curve_level_metrics":
            dest.write_text(source)
        else:
            dest.write_text(old.replace(body(old), body(source), 1))
    for name in FIGURES:
        shutil.copy2(GENERATED_FIGURES / f"{name}.pdf",
                     PAPER / "figures" / f"{name}.pdf")
    record = ood_detail()
    manifest = ROOT / "provenance/artifact_build.json"
    manifest.write_text(json.dumps({
        "active_results": str(RESULTS.relative_to(ROOT)),
        "updated_tables": TABLES, "updated_figures": FIGURES,
        "ood_interval": record,
    }, indent=2) + "\n")
    print("Updated tables, figures, and provenance/artifact_build.json")


if __name__ == "__main__":
    main()
