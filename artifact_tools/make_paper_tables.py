#!/usr/bin/env python3
"""
Generate compact paper tables from a reproduced experiment-output root.

Usage:
  python make_paper_tables.py --root outputs/all_results_reproduced --out tables

Outputs:
  - *.tex  : LaTeX booktabs tables
  - *.md   : Markdown mirrors for quick inspection
  - table_plan.md : recommended placement plan

Dependencies: pandas, numpy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------- formatting helpers ----------

def pct(x: float | int | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{100 * float(x):.{digits}f}"


def fnum(x: float | int | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fsmall(x: float | int | None, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fint(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "--"
    return str(int(round(float(x))))


def latex_escape(s: object) -> str:
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def tex_table(df: pd.DataFrame, caption: str, label: str, note: str | None = None) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + "l" * len(df.columns) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(v) for v in row.tolist()) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(r"\vspace{0.25em}")
        lines.append(r"\caption{" + latex_escape(caption) + r" " + latex_escape(note) + r"}")
    else:
        lines.append(r"\caption{" + latex_escape(caption) + r"}")
    lines.append(r"\label{" + latex_escape(label) + r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(out: Path, name: str, df: pd.DataFrame, caption: str, label: str, note: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(md_table(df) + "\n", encoding="utf-8")
    (out / f"{name}.tex").write_text(tex_table(df, caption, label, note), encoding="utf-8")


def read(root: Path, rel: str) -> pd.DataFrame:
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing expected file: {p}")
    return pd.read_csv(p)


def latest_rewardbench_margin_geometry(root: Path) -> str:
    candidates = sorted(
        root.glob("14_rewardbench_latest_snapshot_*/04_cross_dataset/table_rewardbench_margin_geometry.csv")
    )
    if candidates:
        return str(candidates[-1].relative_to(root))
    return "04_cross_dataset/table_rewardbench_margin_geometry.csv"


def corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y)))


# ---------- main table builders ----------

def summarize_direct_validation(root: Path) -> pd.DataFrame:
    specs = [
        ("SummEval", "human scalar LOO", "03_direct_tilt_validation/robustness/table_exp15_loo_corrected_B_paths.csv", "r2"),
        ("QGEval-human", "admitted human scalar paths", "04_cross_dataset/table_qgeval_human_direct_tilt.csv", "local_r2"),
        ("OpenMEVA-MAGS ROC/WP", "admitted human scalar paths", "04_cross_dataset/table_openmeva_direct_tilt.csv", "local_r2"),
        ("WMT MQM admitted paths", "human scalar LOO", "05_wmt_mtbench/exp19_wmt_mqm/wmt_mqm_paths.csv", "r2"),
        ("WMT pSQM", "human scalar LOO", "05_wmt_mtbench/exp20_wmt_psqm/wmt_psqm_paths.csv", "r2"),
    ]
    rows = []
    for dataset, setting, rel, r2_col in specs:
        df = read(root, rel)
        rows.append({
            "Dataset": dataset,
            "Setting": setting,
            "Paths": fint(len(df)),
            "corr(A)": fnum(corr(df["A_theory"], df["A_fit"]), 4),
            "corr(B)": fnum(corr(df["B_theory"], df["B_fit"]), 4),
            "sign(B) agree": f"{int(((df['B_theory'] < 0) == (df['B_fit'] < 0)).sum())}/{len(df)}",
            "B_th<0 %": pct((df["B_theory"] < 0).mean(), 1),
            "B_fit<0 %": pct((df["B_fit"] < 0).mean(), 1),
            "median R2": fnum(df[r2_col].median(), 6),
        })
    return pd.DataFrame(rows)


def summarize_feedback_extensions(root: Path) -> pd.DataFrame:
    specs = [
        ("MT-Bench BT split-half", "split-half BT utilities", "05_wmt_mtbench/exp22_mtbench_bt/mtbench_bt_split_summary.csv", "r2"),
        ("QGEval metrics", "automatic metrics vs human gold", "04_cross_dataset/table_qgeval_metric_proxy_tilt.csv", "local_r2"),
        ("RewardBench margins", "cross-model margin geometry", latest_rewardbench_margin_geometry(root), "local_r2"),
    ]
    rows = []
    for dataset, setting, rel, r2_col in specs:
        df = read(root, rel)
        rows.append({
            "Dataset": dataset,
            "Setting": setting,
            "Paths": fint(len(df)),
            "corr(A)": fnum(corr(df["A_theory"], df["A_fit"]), 4) if "A_theory" in df else "--",
            "corr(B)": fnum(corr(df["B_theory"], df["B_fit"]), 4),
            "sign(B) agree": f"{int(((df['B_theory'] < 0) == (df['B_fit'] < 0)).sum())}/{len(df)}",
            "B_th<0 %": pct((df["B_theory"] < 0).mean(), 1),
            "B_fit<0 %": pct((df["B_fit"] < 0).mean(), 1),
            "median R2": fnum(df[r2_col].median(), 6),
        })
    return pd.DataFrame(rows)


def _mae_with_ci(s: pd.Series, name: str, suffix: str = "0p1") -> str:
    """Format MAE as 'value [lo, hi]' if CI columns exist, else just 'value'."""
    val = s.get(f"mae_{name}_{suffix}")
    if val is None or pd.isna(val):
        return "--"
    lo = s.get(f"abs_err_{name}_{suffix}_ci_lo")
    hi = s.get(f"abs_err_{name}_{suffix}_ci_hi")
    if lo is not None and not pd.isna(lo):
        return f"{val:.4f} [{lo:.4f}, {hi:.4f}]"
    return f"{val:.4f}"


def summarize_prediction(root: Path) -> pd.DataFrame:
    human = read(root, "07_prospective_prediction/exp27_human_baseline/exp27_prospective_summary.csv")
    model = read(root, "07_prospective_prediction/exp27_model_proxy_4o/exp27_prospective_summary.csv")
    margin = read(root, "09_gpt_proxy_analysis/exp30_gpt_proxy_diagnostics/exp30_margin_summary.csv")

    human_over = human.loc[human["group_key"].eq("__OVERALL__")].iloc[0]
    model_over = model.loc[model["group_key"].eq("__OVERALL__")].iloc[0]
    margin_h = margin[(margin["source"].eq("human_baseline")) & (margin["group_key"].eq("__OVERALL__"))].iloc[0]
    margin_m = margin[(margin["source"].eq("model_proxy")) & (margin["group_key"].eq("__OVERALL__"))].iloc[0]

    rows = []
    for setting, s, m in [
        ("Human-heldout", human_over, margin_h),
        ("GPT/4o proxy", model_over, margin_m),
    ]:
        rows.append({
            "Setting": setting,
            "Splits": fint(s["n_splits"]),
            "sign acc. theory": pct(s["theory_sign_acc"], 1),
            "sign acc. emp-fit": pct(s["empfit_sign_acc"], 1),
            "MAE theory @0.10": _mae_with_ci(s, "theory"),
            "MAE A-only @0.10": _mae_with_ci(s, "aonly"),
            "MAE A_nu @0.10": _mae_with_ci(s, "anu"),
            "MAE emp-fit @0.10": _mae_with_ci(s, "empfit"),
            "MAE zero @0.10": _mae_with_ci(s, "zero"),
            "MAE oracle @0.10": _mae_with_ci(s, "oracle"),
            "mean Delta(0.10)": fsmall(s["actual_delta_mean_0p1"], 4),
            "abs(Delta)>0.01 %": pct(m["frac_abs_actual_gt_0p01"], 1),
            "sign acc. if abs(Delta)>0.01": pct(m["theory_sign_acc_abs_actual_gt_0p01"], 1),
        })
    return pd.DataFrame(rows)


def summarize_prediction_by_group(root: Path) -> pd.DataFrame:
    human = read(root, "07_prospective_prediction/exp27_human_baseline/exp27_prospective_summary.csv")
    model = read(root, "07_prospective_prediction/exp27_model_proxy_4o/exp27_prospective_summary.csv")
    human = human[~human["group_key"].eq("__OVERALL__")].copy()
    model = model[~model["group_key"].eq("__OVERALL__")].copy()
    human.insert(0, "source", "human-heldout")
    model.insert(0, "source", "GPT/4o proxy")
    df = pd.concat([human, model], ignore_index=True)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Setting": r["source"],
            "Group": r["group_key"],
            "Splits": fint(r["n_splits"]),
            "sign acc. theory": pct(r["theory_sign_acc"], 1),
            "sign acc. emp-fit": pct(r["empfit_sign_acc"], 1),
            "MAE theory @0.10": fsmall(r["mae_theory_0p1"], 4),
            "MAE A_nu @0.10": fsmall(r["mae_anu_0p1"], 4),
            "mean Delta(0.10)": fsmall(r["actual_delta_mean_0p1"], 4),
        })
    return pd.DataFrame(rows)


def summarize_stability(root: Path) -> pd.DataFrame:
    specs = [
        ("Human-heldout", "07_prospective_prediction/exp28_human/exp28_stability_paths.csv"),
        ("GPT/4o proxy",  "07_prospective_prediction/exp28_model_proxy_4o/exp28_stability_paths.csv"),
    ]
    rows = []
    for setting, rel in specs:
        df = read(root, rel)
        rows.append({
            "Setting": setting,
            "Paths": fint(len(df)),
            "median min Var_pi(r)": fnum(df["min_var_pi_t_d_le_max"].median(), 3),
            "median min ESS": fint(df["min_ess_d_le_max"].median()),
            "median max abs k4(r)": fnum(df["max_abs_kappa4_rrrr_d_le_max"].median(), 3),
            "median max abs k4(J,r)": fnum(df["max_abs_kappa4_jrrr_d_le_max"].median(), 3),
            "median tail-baseline": fnum(df["tail_mean_at_max_proxy_minus_baseline"].median(), 3),
            "tail<baseline %": pct((df["tail_mean_at_max_proxy_minus_baseline"] < 0).mean(), 1),
        })
    return pd.DataFrame(rows)


def summarize_bon_bridge(root: Path) -> pd.DataFrame:
    specs = [
        ("Human-heldout", "08_bon_bridge/v2_human/exp29_bon_bridge_path_summary.csv"),
        ("GPT/4o proxy",  "08_bon_bridge/v2_model_proxy_4o/exp29_bon_bridge_path_summary.csv"),
    ]
    rows = []
    for setting, rel in specs:
        df = read(root, rel)
        if "path_status" in df:
            df = df[df["path_status"].eq("ok")].copy()
        rows.append({
            "Setting": setting,
            "Paths": fint(len(df)),
            "median d_BoN(N=2)": fnum(df["d_bon_at_N2"].median(), 3),
            "nontriv. BoN in d<=0.1": fnum(df["share_nontrivial_in_shared_window"].median(), 3),
            "nontriv. BoN in d<=0.25": fnum(df["share_nontrivial_in_bridge_window"].median(), 3),
            "MAE BoN vs theory": fsmall(df["mae_bon_vs_theory_bridge"].median(), 4),
            "MAE BoN vs tilt": fsmall(df["mae_bon_vs_tilt_bridge"].median(), 4),
        })
    return pd.DataFrame(rows)


def summarize_intervention(root: Path) -> pd.DataFrame:
    specs = [
        ("SummEval", "06_supplemental_audits/exp25_summeval/exp25_intervention_pair_summary.csv"),
        ("WMT MQM clusters",  "06_supplemental_audits/exp25_wmt_mqm/exp25_intervention_pair_summary.csv"),
    ]
    rows = []
    for dataset, rel in specs:
        df = read(root, rel)
        for (kb, ka), g in df.groupby(["k_before", "k_after"], sort=True):
            rows.append({
                "Dataset": dataset,
                "Transition": f"k={int(kb)} to {int(ka)}",
                "Groups": fint(len(g)),
                "Pair reps": fint(g["n_pairs"].sum()),
                "Delta nu>0 %": pct((g["median_delta_nu_tilde"] > 0).mean(), 1),
                "Delta A>0 %": pct((g["median_delta_A_exact"] > 0).mean(), 1),
                "Delta B<0 %": pct((g["median_delta_B_theory"] < 0).mean(), 1),
                "Delta H>0 %": pct((g["median_delta_headroom_exact"] > 0).mean(), 1),
                "q_0.025(Delta H)>0 %": pct((g["delta_headroom_exact_ci_lo"] > 0).mean(), 1),
                "median Delta A": fnum(g["median_delta_A_exact"].median(), 3),
                "median Delta B": fnum(g["median_delta_B_theory"].median(), 3),
                "median Delta H": fnum(g["median_delta_headroom_exact"].median(), 3),
            })
    return pd.DataFrame(rows)


def summarize_monotone(root: Path) -> pd.DataFrame:
    specs = [
        ("SummEval", "06_supplemental_audits/exp23_summeval/exp23_monotone_summary.csv"),
        ("WMT MQM clusters",  "06_supplemental_audits/exp23_wmt_mqm/exp23_monotone_summary.csv"),
    ]
    rows = []
    for dataset, rel in specs:
        s = read(root, rel).iloc[0]
        rows.append({
            "Dataset": dataset,
            "Paths": fint(s["n_paths_total"]),
            "pos. bin trend %": pct(s["frac_bin_spearman_pos_pooled"], 1),
            "CI-pos. trend %": pct(s["frac_bin_spearman_ci_pos_pooled"], 1),
            "zero violations %": pct(s["frac_zero_violations_pooled"], 1),
            "median viol. frac": fnum(s["median_violation_frac_pooled"], 3),
            "median isotonic R2": fnum(s["median_isotonic_r2_pooled"], 3),
        })
    return pd.DataFrame(rows)


def summarize_scaling(root: Path) -> pd.DataFrame:
    specs = [
        ("SummEval", "06_supplemental_audits/exp24_summeval/exp24_scaling_summary.csv"),
        ("WMT MQM clusters",  "06_supplemental_audits/exp24_wmt_mqm/exp24_scaling_summary.csv"),
    ]
    rows = []
    for dataset, rel in specs:
        s = read(root, rel).iloc[0]
        rows.append({
            "Dataset": dataset,
            "Paths": fint(s["n_paths_total"]),
            "median abs(rho)/sigma_J^2": fnum(s["median_abs_rho_rel_pooled"], 3),
            "abs(rho) small %": pct(s["frac_abs_rho_rel_le_thresh_pooled"], 1),
            "median headroom error A_nu": pct(s["median_headroom_rel_error_nu_pooled"], 1),
            "corr(A_exact,A_fit)": fnum(s["corr_A_exact_fit_pooled"], 4),
            "corr(A_nu,A_fit)": fnum(s["corr_A_nu_fit_pooled"], 4),
            "corr(B_th,B_fit)": fnum(s["corr_B_fit_pooled"], 4),
        })
    return pd.DataFrame(rows)


def summarize_gpt_diagnostics(root: Path) -> pd.DataFrame:
    stats = read(root, "09_gpt_proxy_analysis/exp30_gpt_proxy_diagnostics/exp30_model_proxy_stats_summary.csv")
    compare = read(root, "09_gpt_proxy_analysis/exp30_gpt_proxy_diagnostics/exp30_model_vs_human_compare.csv")
    over = stats[stats["group_key"].eq("__OVERALL__")].iloc[0]
    rows = [{
        "Diagnostic": "median proxy std.",
        "Overall": fnum(over["median_proxy_std"], 3),
    }, {
        "Diagnostic": "median Spearman(proxy,human gold)",
        "Overall": fnum(over["median_spearman_r_j"], 3),
    }, {
        "Diagnostic": "median Pearson(proxy,human gold)",
        "Overall": fnum(over["median_pearson_r_j"], 3),
    }, {
        "Diagnostic": "median top-10% overlap",
        "Overall": fnum(over["median_tail_q_0p1_overlap"], 3),
    }, {
        "Diagnostic": "median top-10% gold delta",
        "Overall": fnum(over["median_tail_q_0p1_gold_delta"], 4),
    }, {
        "Diagnostic": "median top-10% oracle gap",
        "Overall": fnum(over["median_tail_q_0p1_oracle_gap"], 3),
    }, {
        "Diagnostic": "std ratio model/human, median over dims",
        "Overall": fnum(compare["proxy_std_ratio_model_over_human"].median(), 3),
    }, {
        "Diagnostic": "Delta Spearman model-human, median",
        "Overall": fnum(compare["delta_spearman_model_minus_human"].median(), 3),
    }, {
        "Diagnostic": "Delta top-10 overlap model-human, median",
        "Overall": fnum(compare["delta_top10_overlap_model_minus_human"].median(), 3),
    }]
    return pd.DataFrame(rows)


def summarize_ood_transfer(root: Path) -> pd.DataFrame:
    df = read(root, "10_ood_transfer/exp31_ood_transfer_summary.csv")
    overall = df[df["target_group_key"].eq("__OVERALL__")].iloc[0]
    predictors = ["source_mean", "target_cal", "source_prior", "target_anu", "source_anu", "zero"]
    rows = []
    for pred in predictors:
        mae_col = f"mae_{pred}_0p1"
        sign_col = f"sign_acc_{pred}_0p1"
        rows.append({
            "Predictor": pred,
            "MAE @0.10": fsmall(overall.get(mae_col), 4),
            "sign acc.": pct(overall.get(sign_col), 1),
        })
    return pd.DataFrame(rows)


def summarize_proxy_perturbation(root: Path) -> pd.DataFrame:
    df = read(root, "11_proxy_perturbation/exp32_summeval_proxy_baselines/exp32_proxy_baseline_summary.csv")
    df = df[~df["mode"].eq("__OVERALL__")].copy()
    rows = []
    for _, r in df.iterrows():
        sigma = f"{r['noise_sigma']:.1f}" if pd.notna(r["noise_sigma"]) and r["noise_sigma"] > 0 else "--"
        rows.append({
            "Mode": r["mode"],
            "sigma": sigma,
            "actual Delta(0.1)": fsmall(r["actual_delta_mean_0p1"], 4),
            "A_exact": fnum(r["cal_a_exact_mean"], 3),
            "A_nu": fnum(r["cal_a_nu_mean"], 3),
            "Spearman": fnum(r["eval_spearman_mean"], 3),
            "MAE theory": fsmall(r.get("mae_theory_0p1"), 4),
            "MAE A_nu": fsmall(r.get("mae_anu_0p1"), 4),
        })
    return pd.DataFrame(rows)


def build_plan() -> str:
    return """# Recommended table placement

## Main text

1. **Table 1: Human direct-tilt coefficient-law validation.**
   Use this as the main theorem-validation table. Keep it restricted to repeated human scalar ratings and admitted analysis slices: SummEval, QGEval-human, OpenMEVA-MAGS ROC/WP, WMT MQM admitted paths, and WMT pSQM admitted paths. This avoids overclaiming and makes the sign-regime heterogeneity clear.

2. **Table 2: Prospective prediction of held-out local geometry.**
   This should be the most important empirical table. It directly supports the paper's strongest claim: small calibration matrices predict held-out gain curves better than reliability-only A_nu baselines. Include the GPT/4o margin columns to separate curvature-sign and fixed-gain sign diagnostics.

3. **Table 3a/3b: Scope checks.**
   Put finite-range stability and BoN bridge as two small adjacent tables or one two-panel table. These tables define the scope of the local analysis.

4. **Table 4: Measurement-intervention summary.**
   Use this to support the corrected intervention message: measurement improves signal A reliably, but headroom requires curvature cooperation and is not guaranteed.

## Appendix

A1. **Feedback-geometry extensions.** MT-Bench split-half BT, QGEval metric proxies, RewardBench margin geometry. This shows breadth without treating them as dataset-wide theorem validation.

A2. **Monotone error-amplification audit.** Keep as assumption audit, not headline evidence.

A3. **Exact A vs normalized A_nu scaling audit.** This justifies de-emphasizing the old nu^2 headline.

A4. **Prospective prediction by group/dimension.** Useful for inspection and failure localization.

A5. **GPT/4o external-proxy diagnostics.** Use to show the external-proxy ranking/top-tail alignment pattern, not just score compression or tilt instability.

A6. **OOD transfer prediction.** Exp35: source-only transfer vs target calibration vs shrinkage prior. Defends against "same-dataset split is too easy."

A7. **Proxy perturbation baselines.** Exp36: shuffle/random/noisy/sign-flip controls. Shows exact A,B is alignment-sensitive, while A_nu is not.

A8. Optionally include detailed WMT split admissibility and per-dataset tables if space permits.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("all_results_reproduced"))
    ap.add_argument("--out", type=Path, default=Path("paper_tables"))
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    table_specs = [
        ("main_table_1_direct_validation", summarize_direct_validation(root),
         "Human direct-tilt validation of the local coefficient law.", "tab:direct-validation",
         "All fits use exact exponential tilt and LOO/disjoint human gold where applicable."),
        ("main_table_2_prospective_prediction", summarize_prediction(root),
         "Prospective prediction of held-out local optimization geometry.", "tab:prospective-prediction",
         "A_nu denotes the normalized reliability-only baseline."),
        ("main_table_3a_finite_range_stability", summarize_stability(root),
         "Finite-range stability diagnostics on the shared local range.", "tab:finite-range-stability",
         "The shared range is d <= 0.1."),
        ("main_table_3b_bon_bridge", summarize_bon_bridge(root),
         "Best-of-N bridge diagnostics.", "tab:bon-bridge",
         "The bridge window is d <= 0.25; the shared local window is d <= 0.1."),
        ("main_table_4_measurement_intervention", summarize_intervention(root),
         "Measurement-intervention summary.", "tab:measurement-intervention",
         "H is the extrapolated surrogate vertex height -A^2/(4B) for A>0 and B<0, not a checked-window maximum."),
        ("appendix_table_a1_feedback_extensions", summarize_feedback_extensions(root),
         "Feedback-geometry and proxy-regime extensions.", "tab:feedback-extensions",
         "These are not the clean human direct-rating theorem-validation regime."),
        ("appendix_table_a2_monotone_audit", summarize_monotone(root),
         "Monotone error-amplification audit.", "tab:monotone-audit",
         "The condition is treated as an empirical tendency, not a universal law."),
        ("appendix_table_a3_scaling_a_vs_nu", summarize_scaling(root),
         "Exact A coefficient versus normalized A_nu approximation.", "tab:scaling-a-vs-nu",
         "Small-rho threshold is |rho|/sigma_J^2 <= 0.1."),
        ("appendix_table_a4_prediction_by_group", summarize_prediction_by_group(root),
         "Prospective prediction diagnostics by group.", "tab:prediction-by-group",
         None),
        ("appendix_table_a5_gpt_diagnostics", summarize_gpt_diagnostics(root),
         "GPT/4o external-proxy ranking and tail diagnostics.", "tab:gpt-diagnostics",
         "Overall rows are medians over SummEval dimensions unless otherwise stated."),
        ("appendix_table_a6_ood_transfer", summarize_ood_transfer(root),
         "Cross-dataset transfer prediction (SummEval to admitted WMT MQM clusters).", "tab:ood-transfer",
         "Source-only uses SummEval mean coefficients. Target calibration uses 30\\% held-out split."),
        ("appendix_table_a7_proxy_perturbation", summarize_proxy_perturbation(root),
         "Proxy perturbation baselines.", "tab:proxy-perturbation",
         "Shuffle and random\\_normal destroy proxy--gold alignment; A\\_nu remains spuriously high."),
    ]

    preview_chunks = [build_plan()]
    for name, df, caption, label, note in table_specs:
        write_table(out, name, df, caption, label, note)
        preview_chunks.append(f"\n## {name}\n\n" + md_table(df) + "\n")

    (out / "table_plan.md").write_text(build_plan(), encoding="utf-8")
    (out / "all_tables_preview.md").write_text("\n".join(preview_chunks), encoding="utf-8")
    print(f"Wrote {len(table_specs)} tables to {out}")


if __name__ == "__main__":
    main()
