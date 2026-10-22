#!/usr/bin/env python3
"""Flexible conditional-calibration baselines: Table 1(b), Table 2 and appendix displays.

Builds five displays from saved results only (no refitting):

  flex_table1b_prepool      main      pre-pool CurveMAE, seven methods x four datasets
  flex_table1b_paired       appendix  method minus A+B, path-clustered bootstrap CIs
  flex_table2_target_aware  main      target-pool-aware projection/direct, plus pre-pool anchor
  flex_table2_paired        appendix  projection-direct, projection-anchor, direct-anchor
  flex_radius_endpoint      appendix  endpoint MAE at each radius for the pre-pool methods

Conventions shared by every display:
  * CurveMAE = mean over paths of the within-path mean absolute error over the
    common valid (split, radius) cells (exp21/exp32 paired convention). It is NOT
    the unit-mean over complete curves in `curve_mae.csv`.
  * Common valid cells: SummEval loses 11 d=0.10 cells (7,989 of 8,000) that a
    calibration-pool tilt cannot reach; all other datasets keep every cell.
  * WMT comes only from the corrected fixed_141 lineage.
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True  # keep artifact_tools/ free of __pycache__

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import CORE, RR, Group, check, fmt, fmt_ci, require_wmt_fixed, sha256  # noqa: E402

# --------------------------------------------------------------------------- inputs
S = RR / "results" / "direct_calibration_strict"   # exp33 pre-pool (calibration-only)
E = RR / "results" / "evaluation_aware_ab"         # exp32 target-pool-aware
P = {
    "s_paired": S / "main_protocol" / "paired_diff_curvemae_bootstrap.csv",
    "s_paired_wmt": S / "wmt_fixed_141" / "paired_diff_curvemae_bootstrap.csv",
    "s_endpoint": S / "main_protocol" / "endpoint_mae_by_d.csv",
    "s_endpoint_wmt": S / "wmt_fixed_141" / "endpoint_mae_by_d.csv",
    "s_skipped": S / "main_protocol" / "calibration_only_skipped.csv",
    "s_skipped_wmt": S / "wmt_fixed_141" / "calibration_only_skipped.csv",
    "s_spec": S / "main_protocol" / "run_spec.json",
    "s_spec_wmt": S / "wmt_fixed_141" / "run_spec.json",
    "e_pred": E / "non_wmt_canonical" / "direct_calibration_predictions.csv.gz",
    "e_pred_wmt": E / "wmt_fixed_141" / "direct_calibration_predictions.csv.gz",
    "e_spec": E / "non_wmt_canonical" / "source_run_spec.json",
    "e_spec_wmt": E / "wmt_fixed_141" / "direct_calibration_run_spec.json",
    "e_manifest": E / "manifest.json",
    "e_summary": E / "common_support" / "evaluation_aware_ab_summary.csv",
    "e_paired": E / "common_support" / "evaluation_aware_ab_paired_bootstrap.csv",
    "e_support": E / "common_support" / "evaluation_aware_ab_support.json",
    "e_summary_wmt": E / "wmt_fixed_141" / "evaluation_aware_ab_summary.csv",
    "e_paired_wmt": E / "wmt_fixed_141" / "evaluation_aware_ab_paired_bootstrap.csv",
    "original": CORE / "results" / "07_prospective_prediction"
    / "exp27_human_baseline" / "exp27_prospective_paths.csv",
}
WMT_FIXED_SHA = "6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8"

# ------------------------------------------------------------------- design constants
SE, GPT, WMT, OM = ("SummEval-human_heldout", "SummEval-GPT4o",
                    "WMT-MQM-human_heldout", "OpenMEVA-MAGS-human_heldout")
DATASETS = [SE, GPT, WMT, OM]  # column order fixed by the publication plan
LABEL = {SE: "SummEval human", GPT: "GPT-4o proxy", WMT: "WMT MQM", OM: "OpenMEVA-MAGS"}
SUPPORT = {SE: (32, 7989), GPT: (4, 1000), WMT: (141, 35250), OM: (10, 2500)}  # paths, cells
GRID = (0.01, 0.025, 0.05, 0.075, 0.1)
N_REPEATS, N_BOOT, SEED = 50, 2000, 0
KEYS = ["dataset", "path_id", "repeat", "d"]

# key, LaTeX label, plain label, exp33 baseline name, exp32 baseline name (None: no rows saved)
PREPOOL = [
    ("A+B", r"$A{+}B$", "A+B", "A+B", "A+B"),
    ("A-only", r"$A$-only", "A-only", "A-only", "A-only"),
    ("two_term_fit", "Calibration two-term fit", "calibration two-term fit", "empirical_fit", "empirical_fit"),
    ("empirical_curve", "Stored empirical curve", "stored empirical curve", "empirical_curve", "empirical_curve"),
    ("binned", "Binned", "binned", "binned_calibration_only", None),
    ("spline", "Spline", "spline", "spline_calibration_only", None),
    ("isotonic", "Isotonic", "isotonic", "isotonic_calibration_only", None),
    ("mlp", "MLP (exploratory)", "MLP (exploratory)", "mlp_calibration_only", None),
]
MAIN_PREPOOL = [m for m in PREPOOL if m[0] != "mlp"]
# Remaining exp33 pre-pool baselines: radius data CSV only (not Table 1(b) rows).
EXTRA_PREPOOL = [
    ("A_nu", r"$A_\nu$", "A_nu", "A_nu", "A_nu"),
    ("pearson_corr", "Pearson-correlation scaling", "pearson_corr", "pearson_corr", "pearson_corr"),
    ("spearman_corr", "Spearman-correlation scaling", "spearman_corr", "spearman_corr", "spearman_corr"),
    ("zero", "Zero gain", "zero", "zero", "zero"),
]
# Methods that integrate a calibration-pool tilt and are undefined where it cannot reach d.
FRONTIER = {"empirical_curve", "binned_calibration_only", "spline_calibration_only",
            "isotonic_calibration_only", "mlp_calibration_only"}
# Target-pool-aware families: key, label, summary key, direct baseline, projection baseline.
# Same h order as Table 1(b).
FLEX = [
    ("binned", "Binned", "binned", "binned_shrinkage", "A+B (binned h)"),
    ("spline", "Spline", "spline", "spline_ridge", "A+B (spline h)"),
    ("isotonic", "Isotonic", "isotonic", "isotonic_calibration", "A+B (isotonic h)"),
]
MLP_TA = ("mlp", "MLP (exploratory)", "MLP", "mlp_calibration", "A+B (MLP h)")


def load_csv(path, displays, **kwargs) -> pd.DataFrame:
    frame = displays[0].read_csv(path, **kwargs)
    for display in displays[1:]:
        display.input(path)
    return frame


def load_json(path, displays) -> dict:
    for display in displays:
        display.input(path)
    return json.loads(path.read_text())


def p5(value) -> str:
    return f"{value:.5f}"


def ci5(low, high) -> str:
    return f"[{low:.5f}, {high:.5f}]"


# ------------------------------------------------------------------------ protocol
def check_specs(displays) -> dict:
    """Assert the split/radius/tuning protocol the notes describe, from the saved run specs."""
    s_main, s_wmt = load_json(P["s_spec"], displays), load_json(P["s_spec_wmt"], displays)
    e_main, e_wmt = load_json(P["e_spec"], displays), load_json(P["e_spec_wmt"], displays)
    for spec in (s_main, s_wmt):
        assert spec["experiment"] == "exp33_calibration_only_flexible_baselines"
        assert spec["calibration_fraction"] == 0.3 and spec["n_repeats"] == N_REPEATS
        assert spec["d_targets"] == list(GRID) and spec["random_seed"] == 0
        assert spec["fit_max_d"] == 0.1
        assert spec["information_setting"]["target_pool_available_at_prediction_time"] is False
    wmt_main = [x for x in s_main["datasets"] if x["dataset"] == WMT][0]
    assert wmt_main["human_long_csv"].endswith("wmt_mqm_long_main_protocol.csv")  # superseded
    assert s_wmt["datasets"][0]["human_long_sha256"] == WMT_FIXED_SHA
    assert e_wmt["configs"][0]["human_long_sha256"] == WMT_FIXED_SHA
    for spec in (e_main, e_wmt):
        assert spec["calibration_frac"] == 0.3 and spec["n_repeats"] == N_REPEATS
        assert spec["d_targets"] == list(GRID) and spec["random_seed"] == 0
        assert spec["bin_candidates"] == [3, 5, 8, 12]
        assert spec["shrink_k_candidates"] == [0.0, 5.0, 20.0]
        assert spec["spline_knot_candidates"] == [4, 6, 8]
        assert spec["mlp"]["hidden_layers"] == [16, 16] and spec["mlp"]["n_parameters"] == 321
    ea = e_wmt["evaluation_aware_conditional_mean_ab"]
    assert ea["fit_labels"] == "calibration gold labels only"
    assert ea["evaluation_inputs"] == "evaluation proxy scores only; no evaluation gold"
    return {"empirical_fit_policy": e_wmt["reachability_policy"]["empirical_fit"], "mlp": e_wmt["mlp"]}


def load_predictions(displays) -> tuple[pd.DataFrame, dict]:
    """exp32 row-level predictions on the common valid cells (shared baselines + target-aware)."""
    cols = ["dataset", "path_id", "repeat", "d", "baseline", "predicted_gain", "abs_error",
            "cal_max_reachable_d", "n_cal", "n_eval"]
    nonwmt = load_csv(P["e_pred"], displays, usecols=cols)
    wmt = load_csv(P["e_pred_wmt"], displays, usecols=cols)
    assert not nonwmt["dataset"].eq(WMT).any(), "non-WMT package must not carry WMT rows"
    assert set(wmt["dataset"]) == {WMT}
    require_wmt_fixed(wmt, label="exp32 fixed_141 predictions", path_col="path_id")
    manifest = load_json(P["e_manifest"], displays)["canonical_lineages"]
    assert sha256(P["e_pred"]) == manifest["non_wmt"]["packaged_predictions_gzip_sha256"]
    assert sha256(P["e_pred_wmt"]) == manifest["wmt"]["packaged_predictions_gzip_sha256"]
    assert manifest["wmt"]["lineage"] == "fixed_141" and manifest["wmt"]["input_sha256"] == WMT_FIXED_SHA

    skipped = load_csv(P["s_skipped"], displays)
    cells = skipped[KEYS].drop_duplicates()
    assert len(cells) == 11 and set(cells["dataset"]) == {SE} and np.allclose(cells["d"], 0.1)
    assert cells["path_id"].nunique() == 1 and set(skipped["baseline"]) == FRONTIER
    for display in displays:
        display.input(P["s_skipped_wmt"])
    assert P["s_skipped_wmt"].read_text().strip() == "", "WMT fixed_141 has no skipped cells"
    support = load_json(P["e_support"], displays)
    assert support["predictions_sha256"] == sha256(P["e_pred"])
    assert support["support_exclusions_sha256"] == sha256(P["s_skipped"])

    marked = nonwmt.merge(cells.assign(_x=True), on=KEYS, how="left", validate="many_to_one")
    excluded = marked["_x"].eq(True)
    assert len(marked.loc[excluded, KEYS].drop_duplicates()) == 11
    assert len(nonwmt) == support["prediction_rows_before"] == 183989
    assert int((~excluded).sum()) == support["prediction_rows_after"] == 183824
    reach = marked.loc[excluded & marked["baseline"].eq("A+B"), "cal_max_reachable_d"]
    full = (nonwmt[nonwmt["dataset"].eq(SE)].groupby(["baseline", "path_id"])["abs_error"].mean()
            .groupby(level="baseline").mean())
    mask = dict(path=cells["path_id"].iloc[0], repeats=sorted(cells["repeat"].astype(int)),
                reach_min=float(reach.min()), reach_max=float(reach.max()),
                se_all_cells={b: float(full[b]) for b in ("A+B", "A-only", "empirical_fit")})

    pred = pd.concat([marked.loc[~excluded].drop(columns="_x"), wmt], ignore_index=True)
    pred["dk"] = pred["d"].round(6)
    assert not pred.duplicated(KEYS + ["baseline"]).any()
    counts = pred.groupby(["dataset", "baseline"]).size()
    for ds, (n_paths, n_cells) in SUPPORT.items():
        assert n_cells == n_paths * N_REPEATS * len(GRID) - (11 if ds == SE else 0)
        assert counts.loc[ds].eq(n_cells).all(), f"{ds}: baselines do not share the common cells"
        assert pred.loc[pred["dataset"].eq(ds), "path_id"].nunique() == n_paths
    per_path = pred.groupby(["dataset", "path_id"])[["n_cal", "n_eval"]].agg(["min", "max"])
    assert (per_path.xs("min", axis=1, level=1) == per_path.xs("max", axis=1, level=1)).all().all()
    per_path = per_path.xs("min", axis=1, level=1)
    mask["sizes"] = {}
    for ds in DATASETS:
        x = per_path.loc[ds]
        lo, hi, med = x.min(), x.max(), x.median()
        mask["sizes"][ds] = (f"{lo['n_cal']:,}/{lo['n_eval']:,}" if lo.equals(hi) else
                             f"{lo['n_cal']:,}-{hi['n_cal']:,} / {lo['n_eval']:,}-{hi['n_eval']:,} "
                             f"(median {med['n_cal']:,.0f}/{med['n_eval']:,.0f}; varies with cluster size)")
    return pred, mask


class Aggregates:
    """Path-mean and pooled aggregations of the common-cell predictions."""

    def __init__(self, pred: pd.DataFrame):
        self.path = pred.groupby(["dataset", "baseline", "path_id"])["abs_error"].mean()
        self.path_d = pred.groupby(["dataset", "baseline", "dk", "path_id"])["abs_error"].mean()
        self.pooled_d = pred.groupby(["dataset", "baseline", "dk"])["abs_error"].agg(["mean", "size"])
        ta = [b for *_, d, p in FLEX + [MLP_TA] for b in (d, p)]
        wide = (pred[pred["baseline"].isin(ta)]
                .set_index(KEYS + ["baseline"])["predicted_gain"].unstack("baseline"))
        assert not wide.isna().any().any()
        self.gap = {
            key: (wide[proj] - wide[direct]).abs().groupby(level=["dataset", "path_id"]).mean()
            for key, _, _, direct, proj in FLEX + [MLP_TA]
        }

    def pm(self, ds: str, baseline: str) -> pd.Series:
        return self.path.loc[(ds, baseline)]

    def curve(self, ds: str, baseline: str) -> float:
        return float(self.pm(ds, baseline).mean())

    def endpoint_path_mean(self, ds: str, baseline: str, d: float) -> float:
        return float(self.path_d.loc[(ds, baseline, round(d, 6))].mean())

    def gridwise_gap(self, ds: str, key: str) -> float:
        return float(self.gap[key].loc[ds].mean())


def path_bootstrap(left: pd.Series, right: pd.Series) -> tuple[float, float, float]:
    """mean(left) - mean(right) over paths with a path-clustered percentile CI.

    Reproduces exp21 `path_clustered_bootstrap_diff` / exp32 `clustered_interval`:
    resample paths with replacement, 2,000 draws, `default_rng(0)` reseeded per comparison.
    """
    assert left.index.equals(right.index)
    a, b = left.to_numpy(float), right.to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(a), size=len(a))
        draws[i] = a[idx].mean() - b[idx].mean()
    return float(a.mean() - b.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------- Table 1(b)
def build_prepool(d1, d2, agg: Aggregates, se_all_cells: dict) -> tuple[pd.DataFrame, dict]:
    main = load_csv(P["s_paired"], [d1, d2])
    superseded = main[main["dataset"].eq(WMT)]
    assert len(superseded) and superseded["n_paths"].eq(139).all()  # dropped: 139-path lineage
    paired = pd.concat([main[~main["dataset"].eq(WMT)], load_csv(P["s_paired_wmt"], [d1, d2])],
                       ignore_index=True)
    require_wmt_fixed(paired, label="exp33 paired CurveMAE", unit_col="n_paths")
    assert set(paired["dataset"]) == set(DATASETS)
    check(float((paired["mae_A+B"] - paired["mae_baseline"]
                 - paired["diff_AplusB_minus_baseline"]).abs().max()), 0.0, 1e-15, "saved diff identity")
    rows, info = [], {"superseded_all_support": {}, "max_saved_vs_rerun": 0.0}
    for ds in DATASETS:
        n_paths, n_cells = SUPPORT[ds]
        saved = paired[paired["dataset"].eq(ds)].set_index("baseline")
        assert saved["n_paths"].eq(n_paths).all()
        ab_values = saved.loc[saved["n_paired_cells"].eq(n_cells), "mae_A+B"]
        assert ab_values.max() - ab_values.min() < 1e-15
        ab = float(ab_values.iloc[0])
        check(ab, agg.curve(ds, "A+B"), 1e-12, f"{ds} A+B saved vs common-cell predictions")
        for key, tex, plain, s_name, e_name in PREPOOL:
            row = dict(dataset=ds, dataset_label=LABEL[ds], method=key, method_label=plain,
                       exp33_baseline=s_name, in_main_table=key != "mlp", n_paths=n_paths,
                       n_common_cells=n_cells, aplusb_curve_mae_same_cells=ab)
            if key == "A+B":
                rows.append(dict(row, curve_mae=ab, diff_AplusB_minus_method=np.nan, ci_lo=np.nan,
                                 ci_hi=np.nan, value_source="saved paired file (mae_A+B on common cells)",
                                 ci_source="", superseded_all_support_value=np.nan))
                continue
            r = saved.loc[s_name]
            if int(r["n_paired_cells"]) == n_cells:
                value, diff, lo, hi = (float(r[c]) for c in
                                       ("mae_baseline", "diff_AplusB_minus_baseline", "ci_lo", "ci_hi"))
                assert abs(float(r["mae_A+B"]) - ab) < 1e-15
                if e_name is not None:  # cross-check saved value and CI against row-level predictions
                    again = path_bootstrap(agg.pm(ds, "A+B"), agg.pm(ds, e_name))
                    check(value, agg.curve(ds, e_name), 1e-12, f"{ds} {key} vs predictions")
                    for got, ref, name in zip((diff, lo, hi), again, ("diff", "lo", "hi")):
                        check(got, ref, 1e-12, f"{ds} {key} saved {name} vs bootstrap re-run")
                    info["max_saved_vs_rerun"] = max(info["max_saved_vs_rerun"], abs(value - agg.curve(ds, e_name)),
                                                     *(abs(g - r) for g, r in zip((diff, lo, hi), again)))
                rows.append(dict(row, curve_mae=value, diff_AplusB_minus_method=diff, ci_lo=lo, ci_hi=hi,
                                 value_source="saved paired file", ci_source="saved",
                                 superseded_all_support_value=np.nan))
            else:
                # Saved SummEval A-only / two-term-fit rows pair A+B on all 8,000 cells.
                assert ds == SE and int(r["n_paired_cells"]) == 8000 and e_name in ("A-only", "empirical_fit")
                # Same predictions in exp32 and exp33: the 8,000-cell saved values are reproduced.
                check(float(r["mae_baseline"]), se_all_cells[e_name], 1e-12, f"SummEval {key} 8,000-cell identity")
                check(float(r["mae_A+B"]), se_all_cells["A+B"], 1e-12, "SummEval A+B 8,000-cell identity")
                value = agg.curve(ds, e_name)
                diff, lo, hi = path_bootstrap(agg.pm(ds, "A+B"), agg.pm(ds, e_name))
                info["superseded_all_support"][key] = (float(r["mae_baseline"]), float(r["mae_A+B"]))
                rows.append(dict(row, curve_mae=value, diff_AplusB_minus_method=diff, ci_lo=lo, ci_hi=hi,
                                 value_source="recomputed: exp32 predictions masked to common cells",
                                 ci_source="recomputed (same protocol)",
                                 superseded_all_support_value=float(r["mae_baseline"])))
    frame = pd.DataFrame(rows)
    frame["diff_method_minus_AplusB"] = -frame["diff_AplusB_minus_method"]
    frame["ci_lo_method_minus_AplusB"] = -frame["ci_hi"]
    frame["ci_hi_method_minus_AplusB"] = -frame["ci_lo"]
    excl = ~((frame["ci_lo"] <= 0) & (frame["ci_hi"] >= 0))
    frame["paired_outcome"] = np.where(frame["method"].eq("A+B"), "reference",
                                       np.where(~excl, "CI includes 0",
                                                np.where(frame["diff_AplusB_minus_method"] < 0,
                                                         "A+B lower", "method lower")))
    frame["aplusb_reduction_vs_method_pct"] = np.where(
        frame["method"].eq("A-only"),
        100 * (frame["curve_mae"] - frame["aplusb_curve_mae_same_cells"]) / frame["curve_mae"], np.nan)
    frame["rank_in_dataset_main_rows"] = (frame[frame["in_main_table"]]
                                          .groupby("dataset")["curve_mae"].rank(method="min"))
    return frame, info


def write_prepool_tables(d1, d2, frame: pd.DataFrame) -> None:
    value = frame.set_index(["dataset", "method"])
    header = [["Method"] + [LABEL[ds] for ds in DATASETS],
              ["Paths / cells"] + [f"{SUPPORT[ds][0]} / {SUPPORT[ds][1]:,}" for ds in DATASETS]]
    rows = [[tex] + [fmt(value.loc[(ds, key), "curve_mae"], 5) for ds in DATASETS]
            for key, tex, *_ in PREPOOL]
    d1.write_table("lrrrr", header, rows[:7], midrules_after=[1, 3])
    d1.write_table("lrrrr", header, rows, suffix="_with_mlp", midrules_after=[1, 3, 6])
    d1.write_data(frame[["dataset", "dataset_label", "method", "method_label", "exp33_baseline",
                         "in_main_table", "curve_mae", "n_paths", "n_common_cells",
                         "aplusb_curve_mae_same_cells", "aplusb_reduction_vs_method_pct",
                         "rank_in_dataset_main_rows", "value_source", "superseded_all_support_value"]])

    body, breaks = [], []
    for ds in DATASETS:
        for key, tex, *_ in PREPOOL:
            r = value.loc[(ds, key)]
            first = LABEL[ds] if key == "A+B" else ""
            if key == "A+B":
                body.append([first, tex + " (reference)", fmt(r["curve_mae"], 5), "--", "--"])
                continue
            dagger = r"$^{\dagger}$" if r["ci_source"].startswith("recomputed") else ""
            body.append([first, tex, fmt(r["curve_mae"], 5), fmt(r["diff_method_minus_AplusB"], 5),
                         fmt_ci(r["ci_lo_method_minus_AplusB"], r["ci_hi_method_minus_AplusB"], 5) + dagger])
        breaks.append(len(body) - 1)
    d2.write_table("llrrl", ["Dataset", "Method", "CurveMAE", r"Method $-$ $A{+}B$", r"95\% CI"],
                   body, midrules_after=breaks[:-1])
    d2.write_data(frame[["dataset", "dataset_label", "method", "method_label", "exp33_baseline",
                         "in_main_table", "curve_mae", "aplusb_curve_mae_same_cells",
                         "diff_method_minus_AplusB", "ci_lo_method_minus_AplusB",
                         "ci_hi_method_minus_AplusB", "diff_AplusB_minus_method", "ci_lo", "ci_hi",
                         "paired_outcome", "ci_source", "n_paths", "n_common_cells"]]
                  .rename(columns={"ci_lo": "ci_lo_AplusB_minus_method",
                                   "ci_hi": "ci_hi_AplusB_minus_method"})
                  .assign(n_boot=N_BOOT, seed=SEED))


# ------------------------------------------------------------------------ Table 2
def build_target_aware(d3, d4, agg: Aggregates, prepool: pd.DataFrame) -> pd.DataFrame:
    summary = pd.concat([load_csv(P["e_summary"], [d3, d4]), load_csv(P["e_summary_wmt"], [d3, d4])],
                        ignore_index=True)
    paired = pd.concat([load_csv(P["e_paired"], [d3, d4]), load_csv(P["e_paired_wmt"], [d3, d4])],
                       ignore_index=True)
    require_wmt_fixed(summary, label="exp32 summary", unit_col="n_path_repeats", path_col=None)
    assert set(summary["dataset"]) == set(DATASETS) and set(paired["dataset"]) == set(DATASETS)
    assert not summary.duplicated(["dataset", "conditional_mean_model"]).any()
    prepool_ab = prepool[prepool["method"].eq("A+B")].set_index("dataset")["curve_mae"]
    # The combined all-support summary is all-support (SummEval A+B on 8,000 cells): not displayed.
    all_support = load_csv(E / "evaluation_aware_ab_summary_fixed_141.csv", [d3])
    se_all = all_support.loc[all_support["dataset"].eq(SE), "raw_A+B_curve_mae"]
    check(float(se_all.iloc[0]), 0.0118727205973004, 1e-12, "all-support summary is the 8,000-cell lineage")
    assert abs(float(se_all.iloc[0]) - prepool_ab[SE]) > 1e-5
    rows = []
    for ds in DATASETS:
        n_paths, n_cells = SUPPORT[ds]
        for key, label, skey, direct, proj in FLEX + [MLP_TA]:
            s = summary[summary["dataset"].eq(ds) & summary["conditional_mean_model"].eq(skey)]
            assert len(s) == 1
            s = s.iloc[0]
            assert int(s["n_paths"]) == n_paths and int(s["n_path_repeats"]) == n_paths * N_REPEATS
            anchor, p_val, d_val = (float(s[c]) for c in ("raw_A+B_curve_mae", "evaluation_aware_A+B_curve_mae",
                                                          "direct_calibrator_curve_mae"))
            check(anchor, agg.curve(ds, "A+B"), 1e-12, f"{ds} anchor vs predictions")
            check(anchor, prepool_ab[ds], 1e-12, f"{ds} anchor vs Table 1(b) A+B")
            check(p_val, agg.curve(ds, proj), 1e-12, f"{ds} {key} projection vs predictions")
            check(d_val, agg.curve(ds, direct), 1e-12, f"{ds} {key} direct vs predictions")
            endpoints = {}
            for col, base in (("raw_A+B_endpoint_mae_d0p10", "A+B"),
                              ("evaluation_aware_A+B_endpoint_mae_d0p10", proj),
                              ("direct_calibrator_endpoint_mae_d0p10", direct)):
                endpoints[base] = float(s[col])
                check(endpoints[base], agg.endpoint_path_mean(ds, base, 0.1), 1e-12, f"{ds} {col}")
            gap = float(s["mean_abs_projection_vs_direct"])
            check(gap, agg.gridwise_gap(ds, key), 1e-12, f"{ds} {key} gridwise gap")
            comps = paired[paired["dataset"].eq(ds) & paired["conditional_mean_model"].eq(skey)].set_index("comparison")
            saved_pa = comps.loc["evaluation-aware A+B minus raw A+B", ["difference", "ci_lo", "ci_hi"]]
            saved_pd = comps.loc["evaluation-aware A+B minus direct calibrator", ["difference", "ci_lo", "ci_hi"]]
            pa = path_bootstrap(agg.pm(ds, proj), agg.pm(ds, "A+B"))
            pdiff = path_bootstrap(agg.pm(ds, proj), agg.pm(ds, direct))
            da = path_bootstrap(agg.pm(ds, direct), agg.pm(ds, "A+B"))  # not saved on common support
            for got, ref in zip(saved_pa.to_numpy(float), pa):
                check(got, ref, 1e-12, f"{ds} {key} projection-anchor saved vs re-run")
            for got, ref in zip(saved_pd.to_numpy(float), pdiff):
                check(got, ref, 1e-12, f"{ds} {key} projection-direct saved vs re-run")
            rows.append(dict(
                dataset=ds, dataset_label=LABEL[ds], h=key, h_label=label, in_main_table=key != "mlp",
                n_paths=n_paths, n_common_cells=n_cells,
                anchor_prepool_AplusB_curve_mae=anchor, projection_curve_mae=p_val, direct_curve_mae=d_val,
                anchor_endpoint_mae_d0p10_pathmean=endpoints["A+B"],
                projection_endpoint_mae_d0p10_pathmean=endpoints[proj],
                direct_endpoint_mae_d0p10_pathmean=endpoints[direct],
                projection_minus_direct=float(saved_pd["difference"]),
                projection_minus_direct_ci_lo=float(saved_pd["ci_lo"]),
                projection_minus_direct_ci_hi=float(saved_pd["ci_hi"]),
                projection_minus_anchor=float(saved_pa["difference"]),
                projection_minus_anchor_ci_lo=float(saved_pa["ci_lo"]),
                projection_minus_anchor_ci_hi=float(saved_pa["ci_hi"]),
                direct_minus_anchor=da[0], direct_minus_anchor_ci_lo=da[1], direct_minus_anchor_ci_hi=da[2],
                aggregate_abs_gap=abs(p_val - d_val), gridwise_mean_abs_gap=gap,
                exp32_direct_baseline=direct, exp32_projection_baseline=proj,
                ci_sources="projection-direct, projection-anchor: saved; direct-anchor: recomputed (same protocol)",
            ))
    return pd.DataFrame(rows)


def write_target_aware_tables(d3, d4, ta: pd.DataFrame) -> None:
    v = ta.set_index(["dataset", "h"])
    header = [[r"Conditional mean $h$", "Predictor"] + [LABEL[ds] for ds in DATASETS],
              ["", "Paths / cells"] + [f"{SUPPORT[ds][0]} / {SUPPORT[ds][1]:,}" for ds in DATASETS]]
    anchor = ["", r"Pre-pool $A{+}B$ (reference anchor)"] + [
        fmt(v.loc[(ds, "isotonic"), "anchor_prepool_AplusB_curve_mae"], 5) for ds in DATASETS]
    rows = [anchor]
    for key, label, *_ in FLEX + [MLP_TA]:
        rows.append([label, "Projection"] + [fmt(v.loc[(ds, key), "projection_curve_mae"], 5) for ds in DATASETS])
        rows.append(["", "Direct integration"] + [fmt(v.loc[(ds, key), "direct_curve_mae"], 5) for ds in DATASETS])
    d3.write_table("llrrrr", header, rows[:7], midrules_after=[0, 2, 4])
    d3.write_table("llrrrr", header, rows, suffix="_with_mlp", midrules_after=[0, 2, 4, 6])
    # Main-text variant: identical cells, shorter labels so the table fits at \small without scaling.
    short = {SE: "SummEval", GPT: "GPT-4o", WMT: "WMT", OM: "OpenMEVA"}
    compact_header = [[r"$\hat h$", "Predictor"] + [short[ds] for ds in DATASETS], header[1]]
    compact = [["", r"Pre-pool $A{+}B$ (reference)"] + rows[0][2:]] + rows[1:7]
    d3.write_table("llrrrr", compact_header, compact, suffix="_main", midrules_after=[0, 2, 4])
    long = []
    for _, r in ta.iterrows():
        base = dict(dataset=r["dataset"], dataset_label=r["dataset_label"], n_paths=r["n_paths"],
                    n_common_cells=r["n_common_cells"])
        if r["h"] == "isotonic":
            long.append(dict(base, h="none", predictor="pre-pool A+B (reference anchor)", in_main_table=True,
                             curve_mae=r["anchor_prepool_AplusB_curve_mae"],
                             endpoint_mae_d0p10_pathmean=r["anchor_endpoint_mae_d0p10_pathmean"],
                             exp32_baseline="A+B"))
        for kind in ("projection", "direct"):
            long.append(dict(base, h=r["h"], predictor="projection" if kind == "projection" else "direct integration",
                             in_main_table=r["in_main_table"], curve_mae=r[f"{kind}_curve_mae"],
                             endpoint_mae_d0p10_pathmean=r[f"{kind}_endpoint_mae_d0p10_pathmean"],
                             exp32_baseline=r[f"exp32_{kind}_baseline"]))
    d3.write_data(pd.DataFrame(long))

    body, breaks = [], []
    for ds in DATASETS:
        for i, (key, label, *_) in enumerate(FLEX):
            r = v.loc[(ds, key)]
            body.append([LABEL[ds] if i == 0 else "", label, fmt(r["projection_minus_direct"], 5),
                         fmt(r["projection_minus_anchor"], 5), fmt(r["direct_minus_anchor"], 5),
                         fmt(r["gridwise_mean_abs_gap"], 5)])
            body.append(["", "", fmt_ci(r["projection_minus_direct_ci_lo"], r["projection_minus_direct_ci_hi"], 5),
                         fmt_ci(r["projection_minus_anchor_ci_lo"], r["projection_minus_anchor_ci_hi"], 5),
                         fmt_ci(r["direct_minus_anchor_ci_lo"], r["direct_minus_anchor_ci_hi"], 5), ""])
        breaks.append(len(body) - 1)
    d4.write_table("llrrrr", ["Dataset", r"$h$", r"Projection $-$ direct", r"Projection $-$ anchor",
                              r"Direct $-$ anchor$^{\dagger}$", "Mean pointwise gap"],
                   body, midrules_after=breaks[:-1])
    d4.write_data(ta.assign(n_boot=N_BOOT, seed=SEED))


# ----------------------------------------------------------------- radius endpoint
def build_radius(d5, agg: Aggregates, d1) -> tuple[pd.DataFrame, dict]:
    main = load_csv(P["s_endpoint"], [d5])
    assert main["dataset"].eq(WMT).any()  # superseded 139-path rows, dropped below
    ep = pd.concat([main[~main["dataset"].eq(WMT)], load_csv(P["s_endpoint_wmt"], [d5])], ignore_index=True)
    ep["dk"] = ep["d"].round(6)
    assert not ep.duplicated(["dataset", "dk", "baseline"]).any()
    saved = ep.set_index(["dataset", "dk", "baseline"])["mae"]
    rows = []
    for ds in DATASETS:
        n_paths = SUPPORT[ds][0]
        for key, tex, plain, s_name, e_name in PREPOOL + EXTRA_PREPOOL:
            for d in GRID:
                dk = round(d, 6)
                n_cells = int(agg.pooled_d.loc[(ds, "A+B", dk), "size"])
                value, superseded = float(saved.loc[(ds, dk, s_name)]), np.nan
                source = "saved endpoint_mae_by_d"
                balanced = n_cells == n_paths * N_REPEATS
                path_mean = value if balanced else np.nan
                if e_name is not None:
                    pooled = float(agg.pooled_d.loc[(ds, e_name, dk), "mean"])
                    assert int(agg.pooled_d.loc[(ds, e_name, dk), "size"]) == n_cells
                    path_mean = agg.endpoint_path_mean(ds, e_name, d)
                    if not balanced and s_name not in FRONTIER:
                        superseded, value = value, pooled  # saved value pools all 1,600 cells
                        source = "recomputed: exp32 predictions masked to common cells"
                    else:
                        check(value, pooled, 1e-12, f"{ds} {key} d={d} saved vs predictions")
                rows.append(dict(dataset=ds, dataset_label=LABEL[ds], method=key, method_label=plain,
                                 exp33_baseline=s_name, in_table=key in {m[0] for m in PREPOOL}, d=d,
                                 endpoint_mae=value, endpoint_mae_pathmean=path_mean, n_cells=n_cells,
                                 n_paths=n_paths, value_source=source,
                                 superseded_saved_value_all_cells=superseded))
    frame = pd.DataFrame(rows)
    require_wmt_fixed(frame, label="radius endpoint", unit_col="n_paths")

    # The original paper's SummEval endpoint (Table 1(a)) pools the same 1,589 valid d=0.10 cells.
    original = load_csv(P["original"], [d5, d1], usecols=["path_id", "abs_err_theory_0p1", "abs_err_aonly_0p1"])
    se10 = frame[frame["dataset"].eq(SE) & frame["d"].eq(0.1)].set_index("method")
    assert len(original) == int(se10.loc["A+B", "n_cells"]) == 1589
    check(float(original["abs_err_theory_0p1"].mean()), se10.loc["A+B", "endpoint_mae"], 1e-12, "0.0174 link")
    check(float(original["abs_err_aonly_0p1"].mean()), se10.loc["A-only", "endpoint_mae"], 1e-12, "A-only link")
    link = dict(original_ab=float(original["abs_err_theory_0p1"].mean()),
                original_aonly=float(original["abs_err_aonly_0p1"].mean()),
                pathmean_ab=float(se10.loc["A+B", "endpoint_mae_pathmean"]),
                all_cells_ab=float(se10.loc["A+B", "superseded_saved_value_all_cells"]))
    return frame, link


def write_radius_table(d5, frame: pd.DataFrame) -> None:
    v = frame.set_index(["dataset", "method", "d"])["endpoint_mae"]
    body, breaks = [], []
    for ds in DATASETS:
        for i, (key, tex, *_) in enumerate(PREPOOL):
            body.append([LABEL[ds] if i == 0 else "", tex] + [fmt(v.loc[(ds, key, d)], 4) for d in GRID])
        breaks.append(len(body) - 1)
    header = ["Dataset", "Method"] + [f"$d={d:.3f}$" for d in GRID]
    d5.write_table("llrrrrr", header, body, midrules_after=breaks[:-1])
    d5.write_data(frame)


# ------------------------------------------------------------------ audited values
def assert_audited(pre: pd.DataFrame, ta: pd.DataFrame, radius: pd.DataFrame, info: dict, link: dict) -> dict:
    v = pre.set_index(["dataset", "method"])
    half5 = 0.5e-5 + 1e-12
    for ds, exp in zip(DATASETS, (0.01185, 0.00806, 0.05921, 0.00991)):
        check(v.loc[(ds, "A+B"), "curve_mae"], exp, half5, f"A+B CurveMAE {ds}")
    for ds, exp in zip(DATASETS, (0.01803, 0.01030, 0.08039, 0.01058)):
        check(v.loc[(ds, "A-only"), "curve_mae"], exp, half5, f"A-only CurveMAE {ds}")
    for ds, exp in zip(DATASETS, (34.3, 21.7, 26.4, 6.4)):
        check(v.loc[(ds, "A-only"), "aplusb_reduction_vs_method_pct"], exp, 0.05 + 1e-9, f"reduction {ds}")
    check(v.loc[(SE, "two_term_fit"), "curve_mae"], 0.01170, half5, "SummEval two-term fit (common cells)")
    sup = info["superseded_all_support"]
    check(sup["A-only"][0], 0.01804, half5, "superseded 8,000-cell A-only")
    check(sup["two_term_fit"][0], 0.01177, half5, "superseded 8,000-cell two-term fit")
    r = v.loc[(WMT, "isotonic")]
    for got, exp in zip(r[["diff_AplusB_minus_method", "ci_lo", "ci_hi"]], (0.00039, 0.00009, 0.00070)):
        check(got, exp, half5, "WMT A+B - isotonic")
    r = v.loc[(GPT, "isotonic")]
    for got, exp in zip(r[["diff_AplusB_minus_method", "ci_lo", "ci_hi"]], (-0.00022, -0.00042, -0.00001)):
        check(got, exp, half5, "GPT-4o A+B - isotonic")
    iso = pre[pre["method"].eq("isotonic")].set_index("dataset")["paired_outcome"]
    assert list(iso[iso.eq("method lower")].index) == [WMT], "isotonic significantly lower only on WMT"
    main = ta[ta["in_main_table"]]
    check(main["aggregate_abs_gap"].min(), 0.00002, half5, "aggregate gap min")
    check(main["aggregate_abs_gap"].max(), 0.00286, half5, "aggregate gap max")
    check(main["gridwise_mean_abs_gap"].min(), 0.00041, half5, "gridwise gap min")
    check(main["gridwise_mean_abs_gap"].max(), 0.00850, half5, "gridwise gap max")
    t = ta.set_index(["dataset", "h"])
    check(t.loc[(SE, "binned"), "projection_curve_mae"] - t.loc[(SE, "isotonic"), "projection_curve_mae"],
          0.01824, half5, "SummEval binned-h minus isotonic-h projection")
    r = t.loc[(SE, "isotonic")]
    for got, exp in zip(r[["direct_minus_anchor", "direct_minus_anchor_ci_lo", "direct_minus_anchor_ci_hi"]],
                        (-0.00276, -0.00432, -0.00148)):
        check(got, exp, half5, "SummEval isotonic direct - anchor")
    check(link["original_ab"], 0.0174, 0.5e-4, "original SummEval endpoint 0.0174")
    return {"agg_gap": (main["aggregate_abs_gap"].min(), main["aggregate_abs_gap"].max()),
            "grid_gap": (main["gridwise_mean_abs_gap"].min(), main["gridwise_mean_abs_gap"].max())}


# --------------------------------------------------------------------------- notes
def common_protocol(mask: dict) -> str:
    n = mask["sizes"]
    return f"""
## Data, splits and radius grid

- Paths: SummEval human = 32 held-out-rater paths (4 dimensions x 3 expert + 5 crowd held-out
  raters); GPT-4o proxy = 4 dimension paths (GPT-4o judge score as proxy, SummEval human ratings as
  target); WMT MQM = 141 held-out-rater paths from the corrected document-aware `fixed_141` input
  (`data/wmt_mqm_long.csv`, SHA-256 `{WMT_FIXED_SHA[:12]}...`); OpenMEVA-MAGS = 10 held-out-rater
  paths (ROC and WP). The submitted 139-path WMT lineage (6,950 units; A+B CurveMAE 0.06675) is
  excluded everywhere.
- Splits: per path, {N_REPEATS} random 30/70 calibration/evaluation item splits (`random_seed` 0).
  Calibration/evaluation items per split: SummEval and GPT-4o {n[SE]}, OpenMEVA {n[OM]},
  WMT {n[WMT]}. Every method in both
  information regimes uses the identical split plan (exp21 plan, reused by exp32 and exp33).
- Radius grid: d in {{0.01, 0.025, 0.05, 0.075, 0.10}}, the KL divergence of the positive raw-score
  exponential tilt from the uniform distribution on the pool.
- Realized gain (the prediction target): the exact finite-pool tilt gain on the held-out evaluation
  split at KL d, computed from evaluation proxy scores and evaluation human labels. These enter
  only the scoring, never a prediction.
- Units: native human-score units of each target. SummEval human and GPT-4o use SummEval 1-5
  ratings, OpenMEVA-MAGS uses 1-5 ratings and WMT uses MQM scores (non-positive error penalties).
  Magnitudes are not comparable across columns, and the columns must not be averaged into a leaderboard.
"""


def curve_def(mask: dict) -> str:
    reps = ", ".join(str(r) for r in mask["repeats"])
    return f"""
## Metric: CurveMAE (path-mean over common valid cells)

CurveMAE_m = (1/P) sum_p [ (1/|C_p|) sum_(s,d) in C_p |pred_m(p,s,d) - gain(p,s,d)| ],
where P is the number of paths, s indexes the {N_REPEATS} splits and C_p is the set of common
valid (split, radius) cells of path p. Each path has equal weight, and cells within a path are
equally weighted. Lower is better. This is the convention of the saved paired files
(exp21 `path_clustered_bootstrap_diff`, exp32 support-matched summaries). It is used for every
CurveMAE in the flex displays.

Common valid cells are the cells where every displayed method has a prediction. Only SummEval
loses cells: 11 of 8,000. All are on path `{mask['path']}` at d = 0.10, splits {reps}. At these cells
the calibration split's maximum reachable KL ({mask['reach_min']:.4f}-{mask['reach_max']:.4f}) is
below 0.10, so the methods that integrate a calibration-pool tilt are undefined there. These are
the stored empirical curve and the binned, spline, isotonic and MLP calibration-only methods.
Source: `direct_calibration_strict/main_protocol/calibration_only_skipped.csv`. The affected path
therefore averages 239 rather than 250 cells. Support is 7,989 SummEval, 1,000 GPT-4o, 35,250 WMT
and 2,500 OpenMEVA cells. WMT has no skipped cells.

The other saved conventions are NOT used:
- `curve_mae.csv` takes the unit-mean over complete five-radius curves. This drops the 11 SummEval
  splits entirely: 1,589 units, so isotonic is 0.01153 rather than 0.01168.
- The all-support numbers, e.g. SummEval A+B 0.01187 on 8,000 cells, are not used either.
- `information_setting_comparison.csv` mixes 1,589-unit and 1,600-unit supports.
"""


METHODS = """
## Methods (one sentence each; pre-pool = calibration items only)

- A+B: the paper's estimator, A sqrt(d) + B d, with A and B from calibration proxy/human moments.
- A-only: A sqrt(d), the same A without the second-order term.
- Calibration two-term fit: least-squares (no intercept) fit of A sqrt(d) + B d to the exact
  calibration-pool tilt curve. The curve is sampled at 200 equally spaced tilt values up to KL
  min(0.10, 0.95 x calibration max reachable KL), and the fit is evaluated at each d.
- Stored empirical curve: the exact finite calibration-pool tilt gain at KL d, i.e. calibration
  human labels reweighted by the calibration-proxy tilt. It is undefined if the calibration pool
  cannot reach d.
- Binned / spline / isotonic (and exploratory MLP): fit a conditional mean h(r) ~ E[human | proxy]
  on the calibration items. Then integrate h over the calibration-pool tilt:
  sum_i w_i(d) h(r_i) - mean_i h(r_i) over calibration items.
  - Binned: piecewise-constant means without a monotonicity constraint.
  - Spline: a smooth regularized fit.
  - Isotonic: a nondecreasing fit.

## Tuning grids (calibration-internal only; asserted from the run specs)

- Isotonic: sklearn `IsotonicRegression(increasing=True, out_of_bounds="clip")`, no tuning.
- Binned: equal-frequency proxy bins, with bin count in {3, 5, 8, 12} (only counts with
  3 x bins <= n_cal). Bin means are shrunk toward the calibration mean with pseudo-count
  k in {0, 5, 20}. Both are chosen by 5-fold shuffled KFold CV MSE (random_state 0).
- Spline: cubic B-spline basis (`SplineTransformer`, degree 3, constant extrapolation), with
  knots in {4, 6, 8} chosen by 5-fold KFold CV (random_state 0). The ridge penalty is chosen by
  `RidgeCV` over 13 log-spaced alphas in [1e-3, 1e3] (efficient leave-one-out).
- MLP (exploratory, not in the main tables): 2 hidden ReLU layers of 16 units (321 parameters),
  Adam, L2 alpha 1e-3, learning rate 1e-3, max 1,000 iterations, seed 0, standardized input and
  output. It is pre-specified and untuned.
- Both information regimes fit h with the same code, splits and seeds, using only
  (calibration proxy, calibration human). The pre-pool runs are exp33 and the target-aware runs
  are exp32. Only the pool over which h is integrated differs. Within exp32, projection and
  direct integration use the identical fitted h object.
"""

BOOT = f"""
## Paired uncertainty protocol

95% percentile intervals come from a path-clustered bootstrap. Paths are resampled with
replacement, {N_BOOT:,} draws, numpy `default_rng({SEED})` reseeded for each comparison, and the
difference of path-mean CurveMAEs is recomputed. This is the protocol of exp21
`audit_direct_calibration.path_clustered_bootstrap_diff` (used by exp33) and exp32
`clustered_interval`. flex.py re-implements it and reproduces every saved interval it can check to
<= 1e-12. Splits within a path are correlated, so the path is the resampling unit. GPT-4o has
only 4 paths, so its intervals are coarse and descriptive. Intervals are per comparison and are
not adjusted for multiplicity. An endpoint printed as 0.00000 is not exactly zero. Whether that
CI includes 0 is decided from the unrounded CSV, and those cases are listed below.
"""


def near_zero_endpoints(frame: pd.DataFrame, label_cols, pairs) -> str:
    """List CI endpoints that print as 0.00000 / -0.00000 but are nonzero."""
    hits = []
    for r in frame.itertuples():
        for name, lo_col, hi_col in pairs:
            for col in (lo_col, hi_col):
                x = getattr(r, col)
                if pd.notna(x) and x != 0 and abs(x) < 0.5e-5:
                    hits.append(f"{', '.join(str(getattr(r, c)) for c in label_cols)}, {name}: "
                                f"{col} = {x:+.2e}")
    return "\n".join(f"- {h}" for h in hits) if hits else "- none"

ORIGINAL_ENDPOINT = """
## Do not mix with the original endpoint MAE

The original paper's Table 1(a) endpoint MAE for SummEval (0.0174) uses the identical A+B
predictions but a different metric and protocol: endpoint d = 0.10 only, grid {0.05, 0.10},
whole-split drops and pooling over splits. It must never be placed next to, or compared with, the
CurveMAE 0.01185. Captions must name the metric (CurveMAE over five radii versus endpoint MAE at
d = 0.10).
"""


def note_prepool(d1, pre: pd.DataFrame, info: dict, mask: dict) -> None:
    v = pre.set_index(["dataset", "method"])
    ab = ", ".join(f"{LABEL[ds]} {p5(v.loc[(ds, 'A+B'), 'curve_mae'])}" for ds in DATASETS)
    ao = ", ".join(f"{LABEL[ds]} {p5(v.loc[(ds, 'A-only'), 'curve_mae'])}" for ds in DATASETS)
    red = "/".join(f"{v.loc[(ds, 'A-only'), 'aplusb_reduction_vs_method_pct']:.1f}" for ds in DATASETS)
    lowest = []
    for ds in DATASETS:
        sub = pre[pre["dataset"].eq(ds) & pre["in_main_table"]].sort_values("curve_mae")
        lowest.append(f"{LABEL[ds]}: {', '.join(f'{m} {p5(x)}' for m, x in zip(sub['method_label'], sub['curve_mae']))}")
    sup = info["superseded_all_support"]
    outcome = []
    for ds in DATASETS:
        sub = pre[pre["dataset"].eq(ds) & pre["in_main_table"] & ~pre["method"].eq("A+B")]
        groups = {k: ", ".join(sub.loc[sub["paired_outcome"].eq(k), "method_label"]) or "none"
                  for k in ("A+B lower", "method lower", "CI includes 0")}
        outcome.append(f"  - {LABEL[ds]}: A+B significantly lower than {groups['A+B lower']}; "
                       f"significantly lower than A+B: {groups['method lower']}; "
                       f"CI includes 0: {groups['CI includes 0']}.")
    d1.write_note(f"""
# flex_table1b_prepool: pre-pool CurveMAE (Table 1(b))

**Placement.** Main text, Table 1(b). It belongs in the experiments subsection that asks whether
calibration can predict held-out gain before the target pool exists. Body: `tables/flex_table1b_prepool.tex`
(7 method rows x 4 dataset columns, fixed order SummEval human, GPT-4o proxy, WMT MQM, OpenMEVA-MAGS).
Appendix variant with the exploratory MLP row: `tables/flex_table1b_prepool_with_mlp.tex`.
Unrounded data: `data/flex_table1b_prepool.csv`. Paired CIs: `flex_table1b_paired`.

**Information regime.** Pre-pool: every method sees only the rated calibration split (proxy
scores, human labels) and the requested radius d. Target-pool proxy scores and target human
labels are unavailable. Numbers have 5 decimals. The second header row gives paths / common cells.
{common_protocol(mask)}
{curve_def(mask)}
{METHODS}
## Sources and provenance of every cell

- Non-WMT rows come from `direct_calibration_strict/main_protocol/paired_diff_curvemae_bootstrap.csv`
  (`mae_baseline`; A+B = `mae_A+B` of the rows paired on common cells). Its WMT rows are the
  superseded 139-path lineage and are dropped.
- WMT rows come from `direct_calibration_strict/wmt_fixed_141/paired_diff_curvemae_bootstrap.csv`.
- **Recomputed cells:** SummEval A-only and calibration two-term fit. The saved rows pair A+B on
  all 8,000 cells (saved {p5(sup['A-only'][0])} / {p5(sup['two_term_fit'][0])}, with A+B
  {p5(sup['A-only'][1])}). Those saved values must NOT be used. flex.py recomputes both from
  `evaluation_aware_ab/non_wmt_canonical/direct_calibration_predictions.csv.gz`, masked by
  `calibration_only_skipped.csv`. These exp32 rows contain the same shared-baseline predictions
  as exp33. The build verifies this: the exp32 rows reproduce the saved exp33 8,000-cell SummEval
  A+B, A-only and two-term-fit values, and every saved common-cell value on the other datasets,
  to <= 1e-12. exp33 `audit.json` also reports zero shared-prediction drift.
- Checks run by the build:
  - A+B, A-only, two-term fit and stored curve, with their saved paired CIs, are checked against
    the row-level predictions on every dataset (max deviation {info['max_saved_vs_rerun']:.1e}).
  - A local rerun of exp33 on a different machine changes one WMT split's least-squares
    two-term fit, moving CurveMAE by about 1.8e-8 (the known environment boundary). The audit
    draft `draft_table1b_common_cells.csv` was built from such a rerun. The displayed value is
    the saved one (0.0584275140).
  - The calibration-only binned/spline/isotonic/MLP rows have no row-level predictions in the
    package. Their saved values are on the 7,989 common cells by construction, because exactly
    those methods skip the 11 cells.

## Key results (5 decimals; same common cells within each column)

- A+B CurveMAE: {ab}.
- A-only CurveMAE: {ao}.
- Relative reduction of A+B vs A-only, (A-only - A+B)/A-only: {red}% (SummEval/GPT-4o/WMT/OpenMEVA).
  These reductions belong to this matched pre-pool CurveMAE analysis, not to the original endpoint table.
- Main-table rows sorted by CurveMAE: {'; '.join(lowest)}.
- Paired 95% CI outcomes against A+B (from `flex_table1b_paired`; flexible-calibrator
  advantages must be reported in the main text):
{chr(10).join(outcome)}
- Point-estimate order is not a significance claim. Note that the OpenMEVA A+B vs A-only
  difference (6.4%) has a CI that includes 0.
{ORIGINAL_ENDPOINT}
## Reporting details

1. Pre-pool regime: only calibration items are used, and no target-pool proxy scores or labels.
2. Metric: CurveMAE over d in {{0.01, 0.025, 0.05, 0.075, 0.10}}, path-mean of within-path absolute
   errors over common valid cells. Lower is better. Native units; columns are not comparable.
3. Support: 50 random 30/70 splits per path. SummEval uses 7,989 of 8,000 cells, because 11
   d = 0.10 cells are unreachable by calibration-pool methods. WMT is the corrected 141-path input.
4. One-line definitions of the two-term fit and the stored empirical curve. Flexible-calibrator
   tuning is calibration-internal (details in appendix).
5. Paired bootstrap intervals are in the appendix table. No bolding implies significance.
   MLP is exploratory and reported in the appendix.
6. This is not the endpoint MAE of Table 1(a).

## Design notes

- No entry is bolded. If emphasis is wanted, mark cells using the paired CI (see
  `paired_outcome` in `data/flex_table1b_paired.csv`) rather than the raw minimum.
- The second header row (paths / cells) can be dropped if the caption states the support.
""")


def note_prepool_paired(d2, pre: pd.DataFrame, info: dict, mask: dict) -> None:
    lines = []
    for ds in DATASETS:
        sub = pre[pre["dataset"].eq(ds) & ~pre["method"].eq("A+B")]
        parts = [f"{r.method_label} {r.diff_method_minus_AplusB:+.5f} "
                 f"{ci5(r.ci_lo_method_minus_AplusB, r.ci_hi_method_minus_AplusB)} ({r.paired_outcome})"
                 for r in sub.itertuples()]
        lines.append(f"- {LABEL[ds]}: " + "; ".join(parts))
    d2.write_note(f"""
# flex_table1b_paired: pre-pool paired differences against A+B (appendix)

**Placement.** Appendix, next to the full pre-pool comparison. Body: `tables/flex_table1b_paired.tex`
(4 dataset panels, A+B reference row plus 7 methods including the exploratory MLP). Data:
`data/flex_table1b_paired.csv`.

**Quantity.** For each method m: CurveMAE_m - CurveMAE_(A+B) on the same common valid cells,
in native units, 5 decimals. **Positive means the method has larger error than A+B** (A+B
better). The saved exp33 files store the opposite orientation (A+B minus method). The data CSV
keeps both (`*_AplusB_minus_method`). Percentile CIs flip exactly under negation.
{BOOT}
**Recomputed intervals (dagger in the table).** The SummEval A-only and calibration two-term-fit
intervals are not saved on common support: the saved rows use 8,000 cells. flex.py computes them
from the exp32 row-level predictions masked to the 7,989 common cells, with the same protocol
(seed 0, 2,000 path draws). All other intervals are the saved exp33 values. Wherever the build
could re-run a saved interval from predictions (A-only, two-term fit, stored curve on all
datasets), the re-run matches (max deviation {info['max_saved_vs_rerun']:.1e}).
{curve_def(mask)}
{common_protocol(mask)}
## Citable paired results (method minus A+B, 95% CI, outcome)

{chr(10).join(lines)}

Outcome labels: "A+B lower" and "method lower" mean the CI excludes 0. "CI includes 0" is not
evidence of equivalence.
- WMT is the only dataset where isotonic is significantly lower than A+B.
- On the 4-path GPT-4o slice, A+B is lower than isotonic with a CI that barely excludes 0.

CI endpoints printed as 0.00000 (nonzero; sign shown):
{near_zero_endpoints(pre[~pre["method"].eq("A+B")], ["dataset_label", "method_label"],
                     [("method - A+B", "ci_lo_method_minus_AplusB", "ci_hi_method_minus_AplusB")])}
{ORIGINAL_ENDPOINT}
## Reporting details

1. Sign convention: method minus A+B, so positive means A+B better.
2. Path-clustered percentile bootstrap: 2,000 draws, paths as clusters, seed 0. GPT-4o has 4 paths.
3. Common valid cells (SummEval 7,989 of 8,000). The dagger marks intervals recomputed with the
   same protocol because they were not saved on common support.
4. No multiplicity adjustment. MLP is exploratory.
""")


def note_target_aware(d3, ta: pd.DataFrame, gaps: dict, mask: dict) -> None:
    t = ta.set_index(["dataset", "h"])
    rows = []
    for ds in DATASETS:
        parts = [f"{lab} proj {p5(t.loc[(ds, k), 'projection_curve_mae'])} / direct {p5(t.loc[(ds, k), 'direct_curve_mae'])}"
                 for k, lab, *_ in FLEX]
        rows.append(f"- {LABEL[ds]} (anchor {p5(t.loc[(ds, 'isotonic'), 'anchor_prepool_AplusB_curve_mae'])}): "
                    + "; ".join(parts))
    binned_iso = t.loc[(SE, "binned"), "projection_curve_mae"] - t.loc[(SE, "isotonic"), "projection_curve_mae"]
    d3.write_note(f"""
# flex_table2_target_aware: target-pool-aware projection vs direct integration (Table 2)

**Placement.** Main text, Table 2, in the subsection asking whether the two-term form retains the
calibrated curve when information is matched. Body: `tables/flex_table2_target_aware.tex`
(anchor row + {{binned, spline, isotonic}} x {{projection, direct integration}}, same dataset
column order as Table 1(b)). `tables/flex_table2_target_aware_main.tex` has the same cells with
shorter labels for the main text. Appendix variant with exploratory MLP rows:
`tables/flex_table2_target_aware_with_mlp.tex`. Data: `data/flex_table2_target_aware.csv` (long
format, including d = 0.10 path-mean endpoint MAE for reference). Paired results: `flex_table2_paired`.

**Information regime.** Target-pool-aware. The available inputs are the calibration proxy scores
and human labels, plus the realized target-pool (evaluation-split) proxy scores. Target human
labels are never used for prediction; they only score the result. This is a legitimate later
deployment scenario, not label leakage. It is a stronger information setting than the pre-pool
Table 1(b).

**Predictors (both use the SAME fitted h).** In each split, h is fitted once on the calibration
items, with the same procedure, splits and seeds as the Table 1(b) calibrators. Both predictors
then use that one fitted h object.
- Projection: insert pseudo-targets h(r) on the target pool into the paper's moment formulas,
  giving A_h, B_h from (h(r_target), r_target), and predict A_h sqrt(d) + B_h d. This is a
  diagnostic extension, not the unchanged paper estimator, and it inherits h's assumptions and
  estimation error.
- Direct integration: evaluate h on the exact target-pool tilt at KL d,
  sum_i w_i(d) h(r_i) - mean_i h(r_i) over target-pool items.
- Projection vs direct for the same h is the information-matched comparison. It isolates the loss
  from the two-term local form.
- **Reference anchor.** The pre-pool A+B row (identical to the Table 1(b) A+B row) sees less
  information: no target-pool proxy scores. It is shown for scale only. It is NOT a matched
  competitor, so rows below it should not be described as "beating" it on equal terms.
{common_protocol(mask)}
{curve_def(mask)}
{METHODS}
## Sources

- `evaluation_aware_ab/common_support/evaluation_aware_ab_summary.csv` (non-WMT; the
  11-cell SummEval mask applied).
- `evaluation_aware_ab/wmt_fixed_141/evaluation_aware_ab_summary.csv` (WMT).
- `evaluation_aware_ab_summary_fixed_141.csv` is all-support (SummEval 8,000
  cells) and is NOT used.
- Every displayed value is re-derived from the row-level predictions (`non_wmt_canonical` masked +
  `wmt_fixed_141`) and matches to <= 1e-12. Prediction-file hashes are checked against
  `evaluation_aware_ab/manifest.json` and the common-support report.

## Key results (CurveMAE, 5 decimals)

{chr(10).join(rows)}
- Aggregate |projection - direct| CurveMAE gap over the 12 dataset x h cells: {p5(gaps['agg_gap'][0])}-{p5(gaps['agg_gap'][1])}.
- Gridwise (pointwise) mean |projection prediction - direct prediction|, path-mean over common
  cells: {p5(gaps['grid_gap'][0])}-{p5(gaps['grid_gap'][1])}.
- SummEval: the binned-h projection is {p5(binned_iso)} above the isotonic-h projection.
- Aggregate MAE differences and pointwise projection/direct gaps are different quantities, and
  neither is a uniform theorem bound.
{ORIGINAL_ENDPOINT}
## Reporting details

1. Target-pool-aware regime: target-pool proxy scores are available and target human labels are
   not used for prediction.
2. Projection and direct integration use the same fitted h. Projection = two-term moment
   formula on (h(r), r); direct = exact tilt integration of h. Define both in full before abbreviating.
3. The pre-pool A+B row is a reference anchor with less information, not a matched competitor.
4. Metric and support as in Table 1(b): CurveMAE over five radii, common valid cells (SummEval
   7,989 of 8,000), native units, lower is better. Paired CIs are in the appendix.
5. MLP is exploratory (appendix).
""")


def note_target_paired(d4, ta: pd.DataFrame, gaps: dict, mask: dict) -> None:
    lines = []
    for r in ta[ta["in_main_table"]].itertuples():
        lines.append(
            f"- {r.dataset_label}, {r.h_label}: proj-direct {r.projection_minus_direct:+.5f} "
            f"{ci5(r.projection_minus_direct_ci_lo, r.projection_minus_direct_ci_hi)}; proj-anchor "
            f"{r.projection_minus_anchor:+.5f} {ci5(r.projection_minus_anchor_ci_lo, r.projection_minus_anchor_ci_hi)}; "
            f"direct-anchor {r.direct_minus_anchor:+.5f} {ci5(r.direct_minus_anchor_ci_lo, r.direct_minus_anchor_ci_hi)}; "
            f"pointwise gap {r.gridwise_mean_abs_gap:.5f}")
    d4.write_note(f"""
# flex_table2_paired: paired target-pool-aware differences (appendix)

**Placement.** Appendix, next to Table 2. Body: `tables/flex_table2_paired.tex`. Each dataset x h
uses two lines: point differences, then 95% CIs. Data: `data/flex_table2_paired.csv` (wide; also
contains the MLP rows, the d = 0.10 endpoint path-means and both gap measures).

**Quantities** (CurveMAE differences in native units, 5 decimals; negative = first-named predictor
has lower error):
- Projection - direct: same h, same target-pool information. This is the two-term truncation
  loss in aggregate. Saved (exp32 "evaluation-aware A+B minus direct calibrator").
- Projection - anchor: target-pool-aware projection minus pre-pool A+B. These have different
  information and are not a matched comparison. Saved ("... minus raw A+B").
- Direct - anchor (dagger): this is not saved on common support. flex.py computes it from the
  row-level predictions with the identical protocol. Audit example: SummEval isotonic -0.00276
  [-0.00432, -0.00148]. It also has different information from the anchor.
- Mean pointwise gap: path-mean over common cells of |projection prediction - direct
  prediction| (saved `mean_abs_projection_vs_direct`). It measures prediction disagreement, not error.
{BOOT}
All saved differences and intervals were re-run from predictions and match to <= 1e-12.
{curve_def(mask)}
## Citable ranges (binned/spline/isotonic, 12 dataset x h cells)

- Aggregate projection-direct gap |CurveMAE_proj - CurveMAE_direct|: {p5(gaps['agg_gap'][0])}-{p5(gaps['agg_gap'][1])}.
- Gridwise mean pointwise gap: {p5(gaps['grid_gap'][0])}-{p5(gaps['grid_gap'][1])}.
- These two quantities are different, and neither is a uniform bound from the theorem.

## All rows

{chr(10).join(lines)}

CI endpoints printed as 0.00000 (nonzero; sign shown):
{near_zero_endpoints(ta[ta["in_main_table"]], ["dataset_label", "h_label"],
                     [("projection - direct", "projection_minus_direct_ci_lo", "projection_minus_direct_ci_hi"),
                      ("projection - anchor", "projection_minus_anchor_ci_lo", "projection_minus_anchor_ci_hi"),
                      ("direct - anchor", "direct_minus_anchor_ci_lo", "direct_minus_anchor_ci_hi")])}

## Reporting details

1. Sign convention: first-named minus second-named CurveMAE, so negative means the first is better.
2. Anchor comparisons mix information regimes. Only projection - direct is information-matched.
3. Path-clustered bootstrap: 2,000 draws, seed 0, paths as clusters. The dagger marks intervals
   computed by flex.py with the same protocol. GPT-4o has only 4 paths.
4. Common valid cells (SummEval 7,989 of 8,000). WMT is the corrected 141-path input.
""")


def note_radius(d5, radius: pd.DataFrame, link: dict, mask: dict) -> None:
    v = radius.set_index(["dataset", "method", "d"])["endpoint_mae"]
    ab_rows = [f"- {LABEL[ds]}: " + ", ".join(f"d={d:.3f} {v.loc[(ds, 'A+B', d)]:.4f}" for d in GRID)
               for ds in DATASETS]
    red = [f"- {LABEL[ds]}: " + ", ".join(
        f"d={d:.3f} {100 * (v.loc[(ds, 'A-only', d)] - v.loc[(ds, 'A+B', d)]) / v.loc[(ds, 'A-only', d)]:.1f}%"
        for d in GRID) for ds in DATASETS]
    d5.write_note(f"""
# flex_radius_endpoint: pre-pool endpoint MAE at each radius (appendix)

**Placement.** Appendix, radius-level companion to Table 1(b). Body: `tables/flex_radius_endpoint.tex`.
It has 4 dataset panels x {{A+B, A-only, two-term fit, stored curve, binned, spline, isotonic,
MLP (exploratory)}}, with columns d = 0.010 ... 0.100, 4 decimals (endpoint-MAE convention). Data:
`data/flex_radius_endpoint.csv`. The data file also has the remaining exp33 pre-pool baselines
(A_nu, Pearson/Spearman correlation scaling, zero) with `in_table = False`.

**Metric.** Endpoint MAE at radius d = mean of |pred - realized gain| over the common valid
(path, split) cells at that radius (pre-pool, native units, lower is better). This is the saved
exp33 `endpoint_mae_by_d.csv` convention: a pooled mean over cells.
- Every path has all 50 splits at every radius, except SummEval at d = 0.10. So the pooled mean
  equals the path-mean at every displayed cell except the SummEval d = 0.10 column.
- In that column, all methods are pooled over the same 1,589 common cells. The path-mean is not
  computable there for the calibration-only binned/spline/isotonic/MLP methods, because their
  row-level predictions are not in the package.
- The data CSV gives the path-mean where it is computable (`endpoint_mae_pathmean`).
- Support per radius: SummEval 1,600 cells (1,589 at d = 0.10), GPT-4o 200, WMT 7,050,
  OpenMEVA 500.

**Sources.**
- `direct_calibration_strict/main_protocol/endpoint_mae_by_d.csv` (non-WMT; its WMT rows are the
  superseded 139-path lineage and are dropped) and `direct_calibration_strict/wmt_fixed_141/endpoint_mae_by_d.csv`.
- **Consistent-support fix:** the saved SummEval d = 0.10 values pool 1,600 cells for A+B, A-only,
  two-term fit, A_nu, Pearson, Spearman and zero, but 1,589 for the calibration-frontier methods.
  flex.py replaces those seven with pooled means over the 1,589 common cells, recomputed from
  the masked exp32 predictions. The 1,600-cell values remain in the CSV as
  `superseded_saved_value_all_cells`.
- Every saved value with a row-level counterpart is checked against the predictions
  (<= 1e-12).

**Link to the original endpoint table.**
- Pooling the identical A+B predictions over the 1,589 valid SummEval d = 0.10 cells gives
  {link['original_ab']:.6f}, which is the original Table 1(a) value 0.0174. The build checks this
  against `results/.../exp27_prospective_paths.csv`. A-only gives
  {link['original_aonly']:.6f}, i.e. 0.0310.
- The path-mean on the same cells is {link['pathmean_ab']:.6f}, the exp32 summary's
  `raw_A+B_endpoint_mae_d0p10`. The all-1,600-cell pooled value is {link['all_cells_ab']:.6f}.
- The original protocol still differs (d = 0.10 only, grid {{0.05, 0.10}}, whole-split drops,
  pooled splits). None of these endpoint numbers may be compared with or substituted for CurveMAE
  (SummEval A+B 0.01185).
{common_protocol(mask)}
## Key results

A+B endpoint MAE by radius:
{chr(10).join(ab_rows)}

Relative A+B reduction vs A-only by radius:
{chr(10).join(red)}

## Reporting details

1. Pre-pool, endpoint MAE at each radius. This is not CurveMAE, and the d = 0.10 column is
   comparable to Table 1(a).
2. Pooled over common valid cells per radius. SummEval d = 0.10 uses 1,589 of 1,600 cells for
   every method.
3. Native units, lower is better. MLP is exploratory. WMT is the corrected 141-path input.
""")


# ---------------------------------------------------------------------------- main
def main() -> None:
    group = Group("flex")
    d1 = group.display("flex_table1b_prepool", "Main text, Table 1(b)",
                       "Pre-pool CurveMAE: A+B vs A-only, calibration-curve fits and flexible calibrators")
    d2 = group.display("flex_table1b_paired", "Appendix",
                       "Pre-pool CurveMAE differences (method minus A+B) with path-clustered bootstrap CIs")
    d3 = group.display("flex_table2_target_aware", "Main text, Table 2",
                       "Target-pool-aware projection vs direct integration of the same h, with pre-pool A+B anchor")
    d4 = group.display("flex_table2_paired", "Appendix",
                       "Target-pool-aware paired differences: projection-direct, projection-anchor, direct-anchor")
    d5 = group.display("flex_radius_endpoint", "Appendix", "Pre-pool endpoint MAE at each radius d")
    every = [d1, d2, d3, d4, d5]

    check_specs(every)
    pred, mask = load_predictions(every)
    agg = Aggregates(pred)

    pre, info = build_prepool(d1, d2, agg, mask["se_all_cells"])
    ta = build_target_aware(d3, d4, agg, pre)
    radius, link = build_radius(d5, agg, d1)
    gaps = assert_audited(pre, ta, radius, info, link)

    write_prepool_tables(d1, d2, pre)
    write_target_aware_tables(d3, d4, ta)
    write_radius_table(d5, radius)
    note_prepool(d1, pre, info, mask)
    note_prepool_paired(d2, pre, info, mask)
    note_target_aware(d3, ta, gaps, mask)
    note_target_paired(d4, ta, gaps, mask)
    note_radius(d5, radius, link, mask)
    print(f"flex: {len(group.displays)} displays -> {group.write_manifest()}")


if __name__ == "__main__":
    main()
