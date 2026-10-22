#!/usr/bin/env python3
"""Group `ab`: A-only versus A+B contribution and B-stability displays.

Run from this directory (``python ab.py``). Everything is built from saved
results; no model is refit and no experiment code is imported.

* ``ab_fig2``         main Figure 2: relative endpoint-MAE reduction and B share
* ``ab_reduction``    appendix: endpoint MAE by radius and paired CurveMAE
* ``ab_bshare``       appendix: median B share by dataset and radius
* ``ab_b_stability``  appendix: calibration-size and bootstrap/decision stability
* ``ab_rater_noise``  appendix: rater-count/noise stress table and restyled figure
* ``ab_skewness``     appendix: skewness-correction share table and redrawn figure
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # keep artifact_tools/ free of __pycache__
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # fixed PDF CreationDate -> stable hashes

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    CORE, RR, TEXT_WIDTH_IN, Group, check, figure_style, fmt, fmt_ci, plt,
    require_wmt_fixed,
)

# --------------------------------------------------------------------------- inputs
RES = RR / "results"
DCS = RES / "direct_calibration_strict"
MAIN, WMT141 = DCS / "main_protocol", DCS / "wmt_fixed_141"
EAB = RES / "evaluation_aware_ab"
CELLS_NON_WMT = EAB / "non_wmt_canonical" / "direct_calibration_predictions.csv.gz"
CELLS_WMT = EAB / "wmt_fixed_141" / "direct_calibration_predictions.csv.gz"
EXP31 = RES / "coefficient_precision"
EXP22 = RES / "b_stability"
EXP26 = RES / "rater_noise"
EXP19 = RES / "skewness"
DSWEEP = (CORE / "results" / "12_accept_boost" / "d_sweep_human"
          / "d_sweep_summary_overall.csv")

# Source dataset name -> display label, in the paper's column order.
DATASETS = {
    "SummEval-human_heldout": "SummEval",
    "SummEval-GPT4o": "GPT-4o",
    "WMT-MQM-human_heldout": "WMT",
    "OpenMEVA-MAGS-human_heldout": "OpenMEVA",
}
ORDER = list(DATASETS)
N_PATHS = {"SummEval-human_heldout": 32, "SummEval-GPT4o": 4,
           "WMT-MQM-human_heldout": 141, "OpenMEVA-MAGS-human_heldout": 10}
N_REPEATS = 50
D_GRID = (0.01, 0.025, 0.05, 0.075, 0.10)
N_BOOT, BOOT_SEED = 2000, 0  # exp21/exp33 path-cluster bootstrap settings
EXACT = 1e-12

# Dataset identity: categorical hues chosen to avoid the A+B (blue) / A-only
# (orange) method colours used elsewhere in the paper; validated all-pairs for
# colour-vision deficiency (worst CVD dE 13.0). Markers give a second encoding.
DS_STYLE = {
    "SummEval-human_heldout": dict(color="#6250d6", marker="o"),
    "SummEval-GPT4o": dict(color="#d55181", marker="s"),
    "WMT-MQM-human_heldout": dict(color="#008300", marker="^"),
    "OpenMEVA-MAGS-human_heldout": dict(color="#eda100", marker="D"),
}
POOL_STYLE = {"expert": dict(color="#6250d6", marker="o", label="Expert"),
              "turker": dict(color="#008300", marker="s", label="Crowd")}
INK, MUTED = "#222222", "#6b6b6b"


# --------------------------------------------------------------------------- helpers
def label(dataset: str) -> str:
    return DATASETS[dataset]


def p1(x: float) -> str:
    """Percentage cell (value is a fraction), one decimal, no % sign."""
    return fmt(100 * x, 1)


def radius_ticks(ax) -> None:
    ax.set_xticks(D_GRID)
    ax.set_xticklabels(["0.01", "0.025", "0.05", "0.075", "0.10"])
    ax.set_xlim(0.004, 0.106)
    ax.set_xlabel(r"KL radius $d$")


def max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def assert_close(a, b, tol: float, what: str) -> float:
    gap = max_abs(a, b)
    if not gap <= tol:
        raise AssertionError(f"{what}: max |diff| {gap} > {tol}")
    return gap


def round_check(value: float, digits: int, expected: float, what: str) -> None:
    """Assert that `value` rounds (half-up on the decimal string) to `expected`."""
    if round(float(value), digits) != round(expected, digits):
        raise AssertionError(f"{what}: {value} does not round to {expected:.{digits}f}")


def path_bootstrap_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """exp21 `path_clustered_bootstrap_diff`, reproduced draw for draw.

    `a`, `b` are per-path mean errors (paths in sorted path_id order).
    Also returns the bootstrap interval of the relative reduction 1 - mean(a)/mean(b)
    from the same draws.
    """
    rng = np.random.default_rng(BOOT_SEED)
    n = len(a)
    diffs, rels = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ma, mb = np.mean(a[idx]), np.mean(b[idx])
        diffs[i], rels[i] = ma - mb, 1 - ma / mb
    return dict(mae_a=float(np.mean(a)), mae_b=float(np.mean(b)),
                diff=float(np.mean(a) - np.mean(b)),
                ci_lo=float(np.percentile(diffs, 2.5)), ci_hi=float(np.percentile(diffs, 97.5)),
                rel=float(1 - np.mean(a) / np.mean(b)),
                rel_lo=float(np.percentile(rels, 2.5)), rel_hi=float(np.percentile(rels, 97.5)))


def cell_bootstrap(sums_ab: np.ndarray, sums_ao: np.ndarray, counts: np.ndarray) -> dict:
    """Path-cluster bootstrap of cell-mean MAEs (resample paths, keep their cells)."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(counts), size=(N_BOOT, len(counts)))
    s_ab, s_ao, n = sums_ab[idx].sum(1), sums_ao[idx].sum(1), counts[idx].sum(1)
    rel, diff = 1 - s_ab / s_ao, (s_ab - s_ao) / n
    return dict(rel_lo=float(np.percentile(rel, 2.5)), rel_hi=float(np.percentile(rel, 97.5)),
                diff_lo=float(np.percentile(diff, 2.5)), diff_hi=float(np.percentile(diff, 97.5)))


# --------------------------------------------------------------------------- loaders
def load_cells(display) -> pd.DataFrame:
    """Saved exp21-canonical per-cell A+B and A-only predictions (identical to exp33's)."""
    cols = ["dataset", "path_id", "repeat", "d", "baseline", "actual_gain",
            "predicted_gain", "abs_error"]
    non_wmt = display.read_csv(CELLS_NON_WMT, usecols=cols)
    if non_wmt["dataset"].str.contains("WMT").any():
        raise AssertionError("non_wmt_canonical unexpectedly contains WMT rows")
    wmt = display.read_csv(CELLS_WMT, usecols=cols)
    require_wmt_fixed(wmt, label="fixed_141 WMT cells", path_col="path_id")
    frame = pd.concat([non_wmt, wmt], ignore_index=True)
    frame["d"] = frame["d"].round(6)
    keys = ["dataset", "path_id", "repeat", "d"]
    ab = frame[frame["baseline"] == "A+B"].drop(columns="baseline")
    ao = frame[frame["baseline"] == "A-only"].drop(columns="baseline")
    cells = ab.merge(ao, on=keys, suffixes=("_ab", "_ao"), validate="one_to_one")
    if len(cells) != len(ab) or len(cells) != len(ao):
        raise AssertionError("A+B and A-only cells do not pair one-to-one")
    assert_close(cells["actual_gain_ab"], cells["actual_gain_ao"], 0.0, "realized gain A+B vs A-only")
    cells = cells.rename(columns={"actual_gain_ab": "actual", "predicted_gain_ab": "pred_ab",
                                  "predicted_gain_ao": "pred_ao", "abs_error_ab": "err_ab",
                                  "abs_error_ao": "err_ao"}).drop(columns="actual_gain_ao")
    assert_close(cells["err_ab"], (cells["pred_ab"] - cells["actual"]).abs(), 1e-15, "A+B error identity")
    assert_close(cells["err_ao"], (cells["pred_ao"] - cells["actual"]).abs(), 1e-15, "A-only error identity")
    for dataset in ORDER:
        sub = cells[cells["dataset"] == dataset]
        expected = N_PATHS[dataset] * N_REPEATS * len(D_GRID)
        if len(sub) != expected or sub["path_id"].nunique() != N_PATHS[dataset]:
            raise AssertionError(f"{dataset}: {len(sub)} cells / {sub['path_id'].nunique()} paths")
    if set(cells["dataset"]) != set(ORDER):
        raise AssertionError(f"unexpected datasets {sorted(set(cells['dataset']))}")
    return cells


def load_exp33_summaries(display) -> dict:
    """exp33 saved summaries: non-WMT rows from main_protocol, WMT from wmt_fixed_141 only."""
    out = {}
    for name in ("endpoint_mae_by_d", "curve_mae", "paired_diff_curvemae_bootstrap"):
        main = display.read_csv(MAIN / f"{name}.csv")
        main = main[~main["dataset"].str.contains("WMT")]  # main_protocol WMT = superseded 139-path rows
        wmt = display.read_csv(WMT141 / f"{name}.csv")
        unit_col = {"curve_mae": "n_complete_units",
                    "paired_diff_curvemae_bootstrap": "n_paths"}.get(name)
        require_wmt_fixed(wmt, label=f"wmt_fixed_141/{name}", unit_col=unit_col)
        both = pd.concat([main, wmt], ignore_index=True)
        if "d" in both:
            both["d"] = both["d"].round(6)
        out[name] = both
    skipped = display.read_csv(MAIN / "calibration_only_skipped.csv")
    skipped = skipped[~skipped["dataset"].str.contains("WMT")]
    wmt_skipped = display.input(WMT141 / "calibration_only_skipped.csv")
    if wmt_skipped.stat().st_size > 1:
        raise AssertionError("fixed_141 WMT skipped-cell file is no longer empty")
    skipped["d"] = skipped["d"].round(6)
    out["skipped"] = skipped
    return out


def load_exp31(display) -> pd.DataFrame:
    cols = ["dataset", "path_id", "repeat", "calibration_frac", "d", "a_cal", "b_cal",
            "abs_a_error", "abs_b_error", "b_sign_agreement", "scaled_a_error",
            "scaled_b_error", "eval_remainder_abs", "actual_gain", "predicted_ab",
            "curvature_share"]
    rows = display.read_csv(EXP31 / "coefficient_error_rows.csv", usecols=cols)
    require_wmt_fixed(rows, label="exp31 coefficient rows", path_col="path_id")
    rows["d"] = rows["d"].round(6)
    return rows


# --------------------------------------------------------------------------- Figure 2 + reduction
def endpoint_and_curve(cells: pd.DataFrame, exp33: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Verify saved per-cell rows against exp33 summaries, then summarize on common cells."""
    checks = {}
    # (1) all-cell endpoint MAE and CurveMAE reproduce the saved exp33 summaries.
    ep = exp33["endpoint_mae_by_d"]
    ep = ep[ep["baseline"].isin(["A+B", "A-only"])].pivot_table(
        index=["dataset", "d"], columns="baseline", values="mae")
    mine = cells.groupby(["dataset", "d"])[["err_ab", "err_ao"]].mean()
    joined = mine.join(ep, how="inner")
    if len(joined) != len(ORDER) * len(D_GRID):
        raise AssertionError("endpoint summary rows missing")
    checks["endpoint_vs_exp33"] = max(assert_close(joined["err_ab"], joined["A+B"], EXACT, "endpoint A+B"),
                                      assert_close(joined["err_ao"], joined["A-only"], EXACT, "endpoint A-only"))
    unit = cells.groupby(["dataset", "path_id", "repeat"])[["err_ab", "err_ao"]].mean()
    curve = unit.groupby("dataset").mean()
    saved = exp33["curve_mae"].pivot_table(index="dataset", columns="baseline", values="curve_mae")
    checks["curvemae_vs_exp33"] = max(
        assert_close(curve["err_ab"], saved.loc[curve.index, "A+B"], EXACT, "CurveMAE A+B"),
        assert_close(curve["err_ao"], saved.loc[curve.index, "A-only"], EXACT, "CurveMAE A-only"))

    # (2) the exp21 path-cluster bootstrap is reproduced exactly on all cells.
    paired = exp33["paired_diff_curvemae_bootstrap"]
    paired = paired[paired["baseline"] == "A-only"].set_index("dataset")
    gap = 0.0
    for dataset in ORDER:
        by_path = cells[cells["dataset"] == dataset].groupby("path_id")[["err_ab", "err_ao"]].mean()
        res = path_bootstrap_diff(by_path["err_ab"].to_numpy(), by_path["err_ao"].to_numpy())
        row = paired.loc[dataset]
        for mine_key, saved_key in (("mae_a", "mae_A+B"), ("mae_b", "mae_baseline"),
                                    ("diff", "diff_AplusB_minus_baseline"),
                                    ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi")):
            gap = max(gap, assert_close(res[mine_key], row[saved_key], EXACT,
                                        f"{dataset} paired bootstrap {saved_key}"))
    checks["paired_bootstrap_vs_exp33"] = gap

    # (3) common-valid-cell support: drop radius cells any displayed calibrator skipped.
    skip = exp33["skipped"][["dataset", "path_id", "repeat", "d"]].drop_duplicates()
    if len(skip) != 11 or set(skip["dataset"]) != {"SummEval-human_heldout"} or set(skip["d"]) != {0.1}:
        raise AssertionError("common-support exclusion is no longer the 11 SummEval d=0.10 cells")
    flagged = cells.merge(skip.assign(_skip=True), on=["dataset", "path_id", "repeat", "d"], how="left")
    common = flagged[flagged["_skip"].isna()].drop(columns="_skip")
    if len(cells) - len(common) != 11:
        raise AssertionError("exclusion mask did not remove 11 cells")
    iso = exp33["paired_diff_curvemae_bootstrap"]
    iso = iso[iso["baseline"] == "isotonic_calibration_only"].set_index("dataset")

    curve_rows, path_rows = [], []
    for dataset in ORDER:
        sub = common[common["dataset"] == dataset]
        by_path = sub.groupby("path_id")[["err_ab", "err_ao"]].mean()
        res = path_bootstrap_diff(by_path["err_ab"].to_numpy(), by_path["err_ao"].to_numpy())
        # Common-cell A+B path mean equals the saved Table 1(b) value (isotonic pairing row).
        assert_close(res["mae_a"], iso.loc[dataset, "mae_A+B"], EXACT, f"{dataset} common-cell A+B")
        if int(iso.loc[dataset, "n_paired_cells"]) != len(sub):
            raise AssertionError(f"{dataset}: common cells {len(sub)} != saved pairing")
        all_cells = paired.loc[dataset]
        u = sub.groupby(["path_id", "repeat"])[["err_ab", "err_ao"]].mean()
        curve_rows.append(dict(
            dataset=dataset, label=label(dataset), support="common valid cells",
            n_paths=len(by_path), n_units=len(u), n_cells=len(sub),
            curvemae_aonly=res["mae_b"], curvemae_ab=res["mae_a"],
            diff_ab_minus_aonly=res["diff"], diff_ci_lo=res["ci_lo"], diff_ci_hi=res["ci_hi"],
            rel_reduction=res["rel"], rel_ci_lo=res["rel_lo"], rel_ci_hi=res["rel_hi"],
            share_units_ab_lower=float((u["err_ab"] < u["err_ao"]).mean()),
            allcell_curvemae_aonly=float(all_cells["mae_baseline"]),
            allcell_curvemae_ab=float(all_cells["mae_A+B"]),
            allcell_diff=float(all_cells["diff_AplusB_minus_baseline"]),
            allcell_ci_lo=float(all_cells["ci_lo"]), allcell_ci_hi=float(all_cells["ci_hi"]),
            allcell_n_cells=int(all_cells["n_paired_cells"])))

    endpoint_rows = []
    for dataset in ORDER:
        for d in D_GRID:
            sub = common[(common["dataset"] == dataset) & (common["d"] == round(d, 6))]
            g = sub.groupby("path_id").agg(s_ab=("err_ab", "sum"), s_ao=("err_ao", "sum"),
                                           n=("err_ab", "size"))
            boot = cell_bootstrap(g["s_ab"].to_numpy(), g["s_ao"].to_numpy(), g["n"].to_numpy())
            m_ab, m_ao = sub["err_ab"].mean(), sub["err_ao"].mean()
            endpoint_rows.append(dict(
                dataset=dataset, label=label(dataset), d=d, n_paths=len(g), n_cells=len(sub),
                mae_aonly=m_ao, mae_ab=m_ab, diff_ab_minus_aonly=m_ab - m_ao,
                diff_ci_lo=boot["diff_lo"], diff_ci_hi=boot["diff_hi"],
                rel_reduction=1 - m_ab / m_ao, rel_ci_lo=boot["rel_lo"], rel_ci_hi=boot["rel_hi"],
                allcell_mae_aonly=float(joined.loc[(dataset, round(d, 6)), "A-only"]),
                allcell_mae_ab=float(joined.loc[(dataset, round(d, 6)), "A+B"])))
            for path_id, rec in g.iterrows():
                path_rows.append(dict(dataset=dataset, d=d, path_id=path_id, n_cells=int(rec["n"]),
                                      sum_abs_error_ab=rec["s_ab"], sum_abs_error_aonly=rec["s_ao"]))
    return pd.DataFrame(endpoint_rows), pd.DataFrame(curve_rows), pd.DataFrame(path_rows), checks


def b_share(rows31: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Median R_B(d) from exp31 30%-calibration moment coefficients."""
    checks = {}
    shares = (np.abs(rows31["b_cal"]) * rows31["d"]
              / (np.abs(rows31["a_cal"]) * np.sqrt(rows31["d"]) + np.abs(rows31["b_cal"]) * rows31["d"]))
    checks["rb_formula_vs_saved_share"] = assert_close(shares, rows31["curvature_share"], EXACT,
                                                      "R_B formula vs exp31 curvature_share")
    r30 = rows31[rows31["calibration_frac"] == 0.3]
    # exp31's 30% coefficients are the exp33 A+B predictor: same splits, bit-identical predictions.
    merged = r30.merge(cells, on=["dataset", "path_id", "repeat", "d"], how="inner", validate="one_to_one")
    if len(merged) != len(r30) or len(r30) != 2 * sum(N_PATHS[k] * N_REPEATS for k in ORDER):
        raise AssertionError("exp31 30% rows do not pair one-to-one with exp33 cells")
    checks["exp31_vs_exp33_pred_ab"] = assert_close(merged["predicted_ab"], merged["pred_ab"], EXACT,
                                                    "exp31 vs exp33 A+B prediction")
    checks["exp31_vs_exp33_pred_aonly"] = assert_close(
        merged["a_cal"] * np.sqrt(merged["d"]), merged["pred_ao"], EXACT, "exp31 A vs exp33 A-only")
    checks["exp31_vs_exp33_actual"] = assert_close(merged["actual_gain"], merged["actual"], EXACT,
                                                   "exp31 vs exp33 realized gain")
    coef = r30.drop_duplicates(["dataset", "path_id", "repeat", "a_cal", "b_cal"])
    if coef.duplicated(["dataset", "path_id", "repeat"]).any():
        raise AssertionError("exp31 coefficients differ across radii")
    long = []
    for d in D_GRID:
        part = coef[["dataset", "path_id", "repeat", "a_cal", "b_cal"]].copy()
        part["d"] = d
        gb, ga = np.abs(part["b_cal"]) * d, np.abs(part["a_cal"]) * np.sqrt(d)
        part["r_b"] = gb / (ga + gb)
        long.append(part)
    long = pd.concat(long, ignore_index=True)
    summary = (long.groupby(["dataset", "d"])["r_b"]
               .agg(n_path_splits="size", r_b_median="median",
                    r_b_p25=lambda s: s.quantile(0.25), r_b_p75=lambda s: s.quantile(0.75))
               .reset_index())
    summary["label"] = summary["dataset"].map(DATASETS)
    summary["frac_b_negative"] = summary["dataset"].map(coef.groupby("dataset")["b_cal"].apply(lambda s: (s < 0).mean()))
    saved = pd.read_csv(EXP31 / "coefficient_error_summary.csv")  # registered by the caller
    saved = saved[saved["calibration_frac"] == 0.3].assign(d=lambda f: f["d"].round(6))
    cmp = summary.merge(saved, on=["dataset", "d"])
    if len(cmp) != 2 * len(ORDER):
        raise AssertionError("exp31 summary rows missing for d in {0.05, 0.10}")
    checks["rb_median_vs_exp31_summary"] = assert_close(cmp["r_b_median"], cmp["median_curvature_share"],
                                                        EXACT, "median R_B vs exp31 summary")
    for dataset in ORDER:
        if summary.loc[summary["dataset"] == dataset, "n_path_splits"].iloc[0] != N_PATHS[dataset] * N_REPEATS:
            raise AssertionError(f"{dataset}: R_B path-split count")
    summary["sort"] = summary["dataset"].map(ORDER.index)
    summary = summary.sort_values(["sort", "d"]).drop(columns="sort").reset_index(drop=True)
    return summary, long, checks


def draw_fig2(endpoint: pd.DataFrame, share: pd.DataFrame):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.15))
    dodge = dict(zip(ORDER, (-0.0021, -0.0007, 0.0007, 0.0021)))  # keep CI bars apart
    handles = []
    for dataset in ORDER:
        style = DS_STYLE[dataset]
        e = endpoint[endpoint["dataset"] == dataset]
        y = 100 * e["rel_reduction"]
        ax_a.errorbar(e["d"] + dodge[dataset], y,
                      yerr=[y - 100 * e["rel_ci_lo"], 100 * e["rel_ci_hi"] - y],
                      fmt="none", ecolor=style["color"], elinewidth=0.8, alpha=0.55, zorder=2)
        line, = ax_a.plot(e["d"] + dodge[dataset], y, color=style["color"], marker=style["marker"],
                          ms=3.6, lw=1.3, zorder=3, markeredgecolor="white", markeredgewidth=0.5,
                          label=label(dataset))
        handles.append(line)
        s = share[share["dataset"] == dataset]
        ax_b.plot(s["d"], 100 * s["r_b_median"], color=style["color"], marker=style["marker"],
                  ms=3.6, lw=1.3, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
    ax_a.axhline(0, color=MUTED, lw=0.7, zorder=1)
    ax_a.set_title(r"(a) Error reduction from adding $\hat B d$", loc="left")
    ax_a.set_ylabel("Endpoint-MAE reduction (%)")
    ax_a.set_ylim(-3, 55)
    ax_a.set_yticks([0, 10, 20, 30, 40, 50])
    ax_b.set_title(r"(b) Share of $|\hat B d|$ in predicted gain", loc="left")
    ax_b.set_ylabel(r"Median $R_B(d)$ (%)")
    ax_b.set_ylim(0, 15)
    ax_b.set_yticks([0, 5, 10, 15])
    for ax in (ax_a, ax_b):
        radius_ticks(ax)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(w_pad=2.0)
    return fig


def build_fig2_reduction_bshare(group: Group) -> dict:
    fig2 = group.display(
        "ab_fig2", "Main text, Figure 2 (Experiments: what does B add?)",
        "Relative endpoint-MAE reduction from A-only to A+B and median B share R_B(d), by radius")
    red = group.display(
        "ab_reduction", "Appendix (A/B contribution)",
        "Endpoint MAE by radius and paired CurveMAE, A-only vs A+B, common valid cells")
    bsh = group.display(
        "ab_bshare", "Appendix (A/B contribution)",
        "Median calibration-coefficient B share R_B(d) by dataset and radius (exp31, 30% calibration)")

    cells = load_cells(fig2)
    exp33 = load_exp33_summaries(fig2)
    rows31 = load_exp31(fig2)
    fig2.input(EXP31 / "coefficient_error_summary.csv")
    dsweep = fig2.read_csv(DSWEEP)
    for path in (CELLS_NON_WMT, CELLS_WMT, MAIN / "endpoint_mae_by_d.csv", MAIN / "curve_mae.csv",
                 MAIN / "paired_diff_curvemae_bootstrap.csv", MAIN / "calibration_only_skipped.csv",
                 WMT141 / "endpoint_mae_by_d.csv", WMT141 / "curve_mae.csv",
                 WMT141 / "paired_diff_curvemae_bootstrap.csv", WMT141 / "calibration_only_skipped.csv",
                 DSWEEP):
        red.input(path)
    for path in (EXP31 / "coefficient_error_rows.csv", EXP31 / "coefficient_error_summary.csv",
                 CELLS_NON_WMT, CELLS_WMT):
        bsh.input(path)

    endpoint, curve, path_means, checks = endpoint_and_curve(cells, exp33)
    share, share_long, share_checks = b_share(rows31, cells)
    checks.update(share_checks)

    # SummEval d=0.10 on common cells is exactly the original paper's endpoint row (1,589 rows).
    sweep = dsweep[np.isclose(dsweep["d"], 0.10)].iloc[0]
    se10 = endpoint[(endpoint["dataset"] == "SummEval-human_heldout") & (endpoint["d"] == 0.10)].iloc[0]
    if int(sweep["n_rows"]) != se10["n_cells"]:
        raise AssertionError("SummEval d=0.10 support differs from the original d-sweep")
    # Independent code lineage (original paper d-sweep), so agreement is ~1e-7, not bitwise.
    checks["summeval_d010_vs_original_dsweep"] = max(
        assert_close(se10["mae_ab"], sweep["mae_theory"], 1e-6, "SummEval d=.10 A+B vs d-sweep"),
        assert_close(se10["mae_aonly"], sweep["mae_aonly"], 1e-6, "SummEval d=.10 A-only vs d-sweep"))

    # ---- audited numbers
    cur = curve.set_index("dataset")
    for dataset, pct in zip(ORDER, (34.28, 21.73, 26.35, 6.36)):
        round_check(100 * cur.loc[dataset, "rel_reduction"], 2, pct, f"{dataset} CurveMAE reduction %")
    check(cur.loc["SummEval-human_heldout", "curvemae_aonly"], 0.018032, 5e-7, "SummEval common-cell A-only")
    check(cur.loc["SummEval-human_heldout", "curvemae_ab"], 0.01185, 5e-6, "SummEval common-cell A+B")
    om = cur.loc["OpenMEVA-MAGS-human_heldout"]
    round_check(om["diff_ci_lo"], 5, -0.00136, "OpenMEVA CI low")
    round_check(om["diff_ci_hi"], 6, 0.000015, "OpenMEVA CI high")
    if not om["diff_ci_lo"] < 0 < om["diff_ci_hi"]:
        raise AssertionError("OpenMEVA CurveMAE CI should include 0")
    for dataset in ORDER[:3]:
        if not cur.loc[dataset, "diff_ci_hi"] < 0:
            raise AssertionError(f"{dataset}: CurveMAE CI should exclude 0")
    end10 = endpoint[endpoint["d"] == 0.10].set_index("dataset")
    for dataset, value in zip(ORDER, (0.0136, 0.0047, 0.0452, 0.0013)):
        round_check(-end10.loc[dataset, "diff_ab_minus_aonly"], 4, value, f"{dataset} endpoint diff d=.10")
    share_ix = share.set_index(["dataset", "d"])
    for dataset, lo, hi in zip(ORDER, (1.45, 4.61, 2.79, 1.01), (4.45, 13.25, 8.32, 3.12)):
        round_check(100 * share_ix.loc[(dataset, 0.01), "r_b_median"], 2, lo, f"{dataset} R_B d=.01")
        round_check(100 * share_ix.loc[(dataset, 0.10), "r_b_median"], 2, hi, f"{dataset} R_B d=.10")
    round_check(100 * share_ix.loc[("SummEval-human_heldout", 0.10), "r_b_median"], 1, 4.4,
                "SummEval R_B d=.10 at one decimal")
    units_lower = curve["share_units_ab_lower"]
    if not (0.555 <= units_lower.min() and units_lower.max() <= 0.675):
        raise AssertionError(f"share of units with A+B lower CurveMAE out of 56-67%: {units_lower.tolist()}")
    if not (endpoint["rel_reduction"] > 0).all():
        raise AssertionError("A+B should lower endpoint MAE at every checked radius")
    for dataset in ORDER:
        r = endpoint[endpoint["dataset"] == dataset]["rel_reduction"].to_numpy()
        if not np.all(np.diff(r) > 0):
            raise AssertionError(f"{dataset}: relative reduction not increasing in d")

    # ---- ab_fig2 outputs
    fig_data = endpoint.merge(share.drop(columns="label"), on=["dataset", "d"], how="left")
    fig_data = fig_data[["dataset", "label", "d", "n_paths", "n_cells", "mae_aonly", "mae_ab",
                         "diff_ab_minus_aonly", "rel_reduction", "rel_ci_lo", "rel_ci_hi",
                         "n_path_splits", "r_b_median", "r_b_p25", "r_b_p75"]]
    fig2.write_data(fig_data)
    fig2.write_data(path_means, suffix="_path_sums")
    figure_style()
    fig2.save_figure(draw_fig2(endpoint, share))
    fig2.write_note(note_fig2(endpoint, curve, share, checks))

    # ---- ab_reduction outputs
    red.write_data(endpoint, suffix="_endpoint")
    red.write_data(curve, suffix="_curvemae")
    rows, breaks = [], []
    for i, dataset in enumerate(ORDER):
        e = endpoint[endpoint["dataset"] == dataset]
        for j, rec in enumerate(e.itertuples()):
            rows.append([label(dataset) if j == 0 else "", fmt(rec.d, 3), f"{rec.n_cells:,}",
                         fmt(rec.mae_aonly, 4), fmt(rec.mae_ab, 4), fmt(rec.diff_ab_minus_aonly, 4),
                         p1(rec.rel_reduction), fmt_ci(100 * rec.rel_ci_lo, 100 * rec.rel_ci_hi, 1)])
        if i < len(ORDER) - 1:
            breaks.append(len(rows) - 1)
    red.write_table("llrrrrrr",
                    ["Dataset", "$d$", "Cells", r"$A$-only", r"$A{+}B$", r"$A{+}B-A$-only",
                     r"Red.\ (\%)", r"95\% CI (\%)"], rows, midrules_after=breaks)
    rows = []
    for rec in curve.itertuples():
        rows.append([rec.label, fmt(rec.curvemae_aonly, 5), fmt(rec.curvemae_ab, 5),
                     fmt(rec.diff_ab_minus_aonly, 5), fmt_ci(rec.diff_ci_lo, rec.diff_ci_hi, 5),
                     p1(rec.rel_reduction), p1(rec.share_units_ab_lower)])
    red.write_table("lrrrcrr",
                    [[r"", r"\multicolumn{2}{c}{CurveMAE}", r"\multicolumn{2}{c}{$A{+}B-A$-only}",
                      r"", r"Units"],
                     [r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}Dataset", r"$A$-only", r"$A{+}B$", "Diff.",
                      r"95\% CI", r"Red.\ (\%)", r"won (\%)"]], rows, suffix="_curvemae")
    red.write_note(note_reduction(endpoint, curve, checks))

    # ---- ab_bshare outputs
    bsh.write_data(share)
    header = [[r"\multicolumn{2}{c}{}", r"\multicolumn{5}{c}{Median $R_B(d)$ (\%) at $d=$}",
               r"\multicolumn{1}{c}{IQR (\%)}"],
              [r"\cmidrule(lr){3-7}Dataset", "Path-splits"] + [fmt(d, 3) for d in D_GRID]
              + [r"at $d=0.100$"]]
    rows = []
    for dataset in ORDER:
        s = share[share["dataset"] == dataset]
        last = s[s["d"] == 0.10].iloc[0]
        rows.append([label(dataset), f"{int(last['n_path_splits']):,}"]
                    + [p1(v) for v in s["r_b_median"]]
                    + [fmt_ci(100 * last["r_b_p25"], 100 * last["r_b_p75"], 1)])
    bsh.write_table("lrrrrrrc", header, rows)
    bsh.write_note(note_bshare(share, curve, endpoint))
    return dict(rows31=rows31, endpoint=endpoint, curve=curve, share=share)


# --------------------------------------------------------------------------- B stability
def build_b_stability(group: Group, rows31: pd.DataFrame) -> None:
    disp = group.display(
        "ab_b_stability", "Appendix (B stability)",
        "Calibration-size coefficient error (exp31) and bootstrap sign/CI/decision stability (exp22)")
    disp.input(EXP31 / "coefficient_error_rows.csv")
    saved = disp.read_csv(EXP31 / "coefficient_error_summary.csv")
    require_wmt_fixed(saved, label="exp31 summary", unit_col="n_path_splits")
    saved["d"] = saved["d"].round(6)
    coef = disp.read_csv(EXP22 / "coefficient_level.csv")
    decision = disp.read_csv(EXP22 / "decision_level.csv")
    spec = json.loads(disp.input(EXP22 / "b_stability_run_spec.json").read_text())
    require_wmt_fixed(coef, label="exp22 coefficient_level", unit_col="n_path_repeat_units")
    require_wmt_fixed(decision, label="exp22 decision_level", unit_col="n")
    if spec["calibration_frac"] != 0.3 or spec["n_item_boot"] != 300 or spec["d_valid"] != 0.1:
        raise AssertionError("exp22 run spec changed")

    at10 = rows31[rows31["d"] == 0.10]
    mine = (at10.groupby(["dataset", "calibration_frac"])
            .agg(n_path_splits=("a_cal", "size"), median_scaled_a_error=("scaled_a_error", "median"),
                 median_scaled_b_error=("scaled_b_error", "median"),
                 median_eval_remainder_abs=("eval_remainder_abs", "median"),
                 b_sign_agreement=("b_sign_agreement", "mean"),
                 median_abs_a_error=("abs_a_error", "median"),
                 median_abs_b_error=("abs_b_error", "median"))
            .reset_index())
    cmp = mine.merge(saved[saved["d"] == 0.10], on=["dataset", "calibration_frac"], suffixes=("", "_saved"))
    if len(cmp) != 12:
        raise AssertionError("exp31 summary rows missing at d=0.10")
    gap = 0.0
    for col in ("n_path_splits", "median_scaled_a_error", "median_scaled_b_error",
                "median_eval_remainder_abs", "b_sign_agreement", "median_abs_a_error",
                "median_abs_b_error"):
        gap = max(gap, assert_close(cmp[col], cmp[f"{col}_saved"], EXACT, f"exp31 {col}"))
    mine["sort"] = mine["dataset"].map(ORDER.index)
    mine = mine.sort_values(["sort", "calibration_frac"]).drop(columns="sort").reset_index(drop=True)
    mine.insert(1, "label", mine["dataset"].map(DATASETS))
    mine.insert(3, "d", 0.10)

    # audited numbers (30% calibration)
    m30 = mine[mine["calibration_frac"] == 0.3].set_index("dataset")
    round_check(m30["median_scaled_a_error"].min(), 4, 0.0116, "min sqrt(d)|dA| at 30%")
    round_check(m30["median_scaled_a_error"].max(), 4, 0.0777, "max sqrt(d)|dA| at 30%")
    round_check(m30["median_scaled_b_error"].min(), 4, 0.0029, "min d|dB| at 30%")
    round_check(m30["median_scaled_b_error"].max(), 4, 0.0317, "max d|dB| at 30%")
    round_check(m30["median_eval_remainder_abs"].min(), 4, 0.0004, "min remainder at 30%")
    round_check(m30["median_eval_remainder_abs"].max(), 4, 0.0087, "max remainder at 30%")
    for dataset, value in zip(ORDER, (89.4, 97.0, 89.7, 90.0)):
        round_check(100 * m30.loc[dataset, "b_sign_agreement"], 1, value, f"{dataset} sign agreement 30%")
    for dataset in ORDER:
        series = mine[mine["dataset"] == dataset]["median_scaled_b_error"].to_numpy()
        if not np.all(np.diff(series) < 0):
            raise AssertionError(f"{dataset}: d|dB| not decreasing with calibration size")

    boot = coef.merge(decision[["dataset", "n", "decision_disagree_rate", "proceed_ab_rate",
                                "proceed_aonly_rate", "decision_stable_despite_b_uncertain_rate",
                                "b_ci_includes_zero_rate"]],
                      on="dataset", suffixes=("", "_decision"))
    assert_close(boot["b_ci_includes_zero_rate"], boot["b_ci_includes_zero_rate_decision"], 0.0,
                 "exp22 CI-includes-zero rate (coefficient vs decision file)")
    if not (boot["n_path_repeat_units"] == boot["n"]).all():
        raise AssertionError("exp22 unit counts disagree")
    boot = boot.drop(columns=["b_ci_includes_zero_rate_decision", "n"])
    boot["sort"] = boot["dataset"].map(ORDER.index)
    boot = boot.sort_values("sort").drop(columns="sort").reset_index(drop=True)
    boot.insert(1, "label", boot["dataset"].map(DATASETS))
    bix = boot.set_index("dataset")
    for dataset, sign, ci0, dis in zip(ORDER, (96.7, 100.0, 97.3, 100.0), (52.6, 25.0, 49.8, 35.2),
                                       (0.0, 25.5, 2.5, 0.0)):
        round_check(100 * bix.loc[dataset, "b_sign_stability_median"], 1, sign, f"{dataset} sign stability")
        round_check(100 * bix.loc[dataset, "b_ci_includes_zero_rate"], 1, ci0, f"{dataset} CI includes 0")
        round_check(100 * bix.loc[dataset, "decision_disagree_rate"], 1, dis, f"{dataset} decision disagree")

    disp.write_data(mine, suffix="_calibration_size")
    disp.write_data(boot, suffix="_bootstrap")
    rows, breaks = [], []
    for i, dataset in enumerate(ORDER):
        block = mine[mine["dataset"] == dataset]
        for j, rec in enumerate(block.itertuples()):
            rows.append([label(dataset) if j == 0 else "", fmt(100 * rec.calibration_frac, 0) + r"\%",
                         fmt(rec.median_scaled_a_error, 4), fmt(rec.median_scaled_b_error, 4),
                         fmt(rec.median_eval_remainder_abs, 4), p1(rec.b_sign_agreement)])
        if i < len(ORDER) - 1:
            breaks.append(len(rows) - 1)
    disp.write_table("lrrrrr",
                     ["Dataset", "Cal.", r"$\sqrt{d}\,|\hat A-A_{\rm eval}|$",
                      r"$d\,|\hat B-B_{\rm eval}|$", "Remainder", r"Sign agr.\ (\%)"],
                     rows, midrules_after=breaks)
    rows = [[rec.label, f"{rec.n_path_repeat_units:,}", fmt(rec.b_se_median, 4),
             p1(rec.b_sign_stability_median), p1(rec.b_ci_includes_zero_rate),
             p1(rec.decision_disagree_rate)] for rec in boot.itertuples()]
    disp.write_table("lrrrrr",
                     ["Dataset", "Path-splits", r"Median SE$(\hat B)$", r"Sign kept (\%)",
                      r"CI $\ni 0$ (\%)", r"Disagree (\%)"], rows, suffix="_bootstrap")
    disp.write_note(note_b_stability(mine, boot, gap))


# --------------------------------------------------------------------------- rater noise
def draw_rater_noise(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.95))
    clean = summary[(summary["calibration_frac"] == 0.3) & (summary["noise_family"] == "clean")]
    noisy = summary[(summary["calibration_frac"] == 0.3)
                    & (summary["gold_rater_count"] == summary["max_gold_raters"])
                    & (summary["noise_family"] != "clean")]
    for pool, style in POOL_STYLE.items():
        c = clean[clean["rater_type"] == pool]
        axes[0].plot(c["gold_rater_count"], c["b_abs_error_vs_eval"], color=style["color"],
                     marker=style["marker"], ms=3.6, lw=1.3)
        for family, ls, offset in (("gaussian", "-", -0.015), ("student_t3", "--", 0.015)):
            n = noisy[(noisy["rater_type"] == pool) & (noisy["noise_family"] == family)]
            axes[1].plot(n["noise_scale_sd"], 100 * n["b_sign_matches_same_k_clean"], color=style["color"],
                         marker=style["marker"], ms=3.6, lw=1.2, ls=ls)
            x = n["noise_scale_sd"] + offset
            axes[2].errorbar(x, n["ab_minus_aonly_curve_mae"],
                             yerr=[n["ab_minus_aonly_curve_mae"] - n["ab_minus_aonly_ci_lo"],
                                   n["ab_minus_aonly_ci_hi"] - n["ab_minus_aonly_curve_mae"]],
                             color=style["color"], marker=style["marker"], ms=3.4, lw=1.1, ls=ls,
                             elinewidth=0.8, capsize=1.5)
    axes[0].set_title("(a) Gold raters (clean)", loc="left")
    axes[0].set_xlabel("Gold raters per item")
    axes[0].set_ylabel(r"Mean $|\hat B-B_{\rm eval}|$")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[1].set_title(r"(b) Sign of $\hat B$ vs. clean", loc="left")
    axes[1].set_xlabel("Noise (rating SD)")
    axes[1].set_ylabel("Sign agreement (%)")
    axes[1].set_ylim(90, 100.6)
    axes[2].set_title(r"(c) $A{+}B$ minus $A$-only", loc="left")
    axes[2].set_xlabel("Noise (rating SD)")
    axes[2].set_ylabel(r"$\Delta$ CurveMAE")
    axes[2].axhline(0, color=MUTED, lw=0.7, zorder=0)
    for ax in axes[1:]:
        ax.set_xticks([0.25, 0.5, 1.0])
        ax.set_xlim(0.15, 1.1)
    for ax in axes:
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s["color"], marker=s["marker"], ms=3.6, lw=1.3, label=s["label"])
               for s in POOL_STYLE.values()]
    handles += [Line2D([], [], color=MUTED, lw=1.2, ls="-", label="Gaussian noise"),
                Line2D([], [], color=MUTED, lw=1.2, ls="--", label=r"Student-$t_3$ noise")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09),
               handlelength=2.0, columnspacing=1.4)
    fig.tight_layout(w_pad=1.0)
    return fig


def build_rater_noise(group: Group) -> None:
    disp = group.display(
        "ab_rater_noise", "Appendix (rater count and rating noise)",
        "exp26 SummEval rater-count and injected-noise stress of B (table + restyled figure)")
    summary = disp.read_csv(EXP26 / "condition_summary.csv")
    selected = disp.read_csv(EXP26 / "selected_conditions.csv")
    spec = json.loads(disp.input(EXP26 / "run_spec.json").read_text())
    art = json.loads(disp.input(EXP26 / "artifact_spec.json").read_text())
    disp.input(EXP26 / "fig_b_rater_noise_stability.pdf")  # provenance of the restyled figure
    if spec["n_repeats"] != 50 or spec["n_paths"] != 32 or "held-out rater" not in spec["human_proxy_rule"]:
        raise AssertionError("exp26 run spec changed")
    keys = ["rater_type", "calibration_frac", "gold_rater_count", "noise_family", "noise_scale_sd"]
    cmp = selected.merge(summary, on=keys, suffixes=("_sel", ""))
    if len(cmp) != len(selected):
        raise AssertionError("selected exp26 rows not found in condition summary")
    gap = 0.0
    for col in ("b_abs_error_vs_eval", "b_sign_matches_eval", "curve_mae_ab", "curve_mae_aonly",
                "ab_minus_aonly_ci_lo", "ab_minus_aonly_ci_hi"):
        gap = max(gap, assert_close(cmp[f"{col}_sel"], cmp[col], EXACT, f"exp26 selected {col}"))

    s30 = summary[summary["calibration_frac"] == 0.3]
    clean = s30[s30["noise_family"] == "clean"].sort_values(["rater_type", "gold_rater_count"])
    noisy = s30[(s30["gold_rater_count"] == s30["max_gold_raters"]) & (s30["noise_family"] != "clean")]
    noisy = noisy.sort_values(["rater_type", "noise_family", "noise_scale_sd"])
    ci = clean.set_index(["rater_type", "gold_rater_count"])
    for (pool, k), (berr, cmae, sign) in {("expert", 1): (0.0618, 0.0269, 97.3), ("expert", 2): (0.0428, 0.0144, 100.0),
                                          ("turker", 1): (0.0659, 0.0122, 81.6), ("turker", 4): (0.0590, 0.0104, 85.4)}.items():
        round_check(ci.loc[(pool, k), "b_abs_error_vs_eval"], 4, berr, f"{pool} k={k} B error")
        round_check(ci.loc[(pool, k), "curve_mae_ab"], 4, cmae, f"{pool} k={k} CurveMAE A+B")
        round_check(100 * ci.loc[(pool, k), "b_sign_matches_eval"], 1, sign, f"{pool} k={k} sign")
    nmin = noisy.groupby("rater_type")["b_sign_matches_same_k_clean"].min()
    round_check(100 * nmin["expert"], 1, 99.3, "expert min noisy sign agreement")
    round_check(100 * nmin["turker"], 1, 93.0, "crowd min noisy sign agreement")
    nx = noisy.set_index(["rater_type", "noise_family", "noise_scale_sd"])
    for pool, scale in (("expert", 1.0), ("turker", 0.5)):
        for family in ("gaussian", "student_t3"):
            rec = nx.loc[(pool, family, scale)]
            if not rec["ab_minus_aonly_ci_lo"] > 0:
                raise AssertionError(f"{pool} {family} {scale}: A+B advantage should be gone (CI lo > 0)")
    for family in ("gaussian", "student_t3"):
        if not nx.loc[("expert", family, 0.25), "ab_minus_aonly_ci_hi"] < 0:
            raise AssertionError("expert 0.25 SD: A+B should still beat A-only")

    table = pd.concat([clean.assign(block="clean_rater_count"), noisy.assign(block="noise_all_gold_raters")])
    table = table.assign(pool=table["rater_type"].map({"expert": "Expert", "turker": "Crowd"}))
    disp.write_data(table)
    # One sign column whose reference is stated in each block header (both are in the CSV):
    # clean rows -> held-out clean B_eval; noise rows -> same-rater-count clean estimate.
    pool = {"expert": "Expert", "turker": "Crowd"}
    rows = [[r"\multicolumn{6}{l}{\textit{Clean ratings, $k$ of $K$ gold raters; sign vs.\ $B_{\rm eval}$}}"]]
    for rec in clean.itertuples():
        rows.append([f"{pool[rec.rater_type]}, {rec.gold_rater_count} of {rec.max_gold_raters}",
                     fmt(rec.b_abs_error_vs_eval, 4), p1(rec.b_sign_matches_eval), fmt(rec.curve_mae_ab, 5),
                     fmt(rec.ab_minus_aonly_curve_mae, 5),
                     fmt_ci(rec.ab_minus_aonly_ci_lo, rec.ab_minus_aonly_ci_hi, 5)])
    first_noise = len(rows) - 1
    rows.append([r"\multicolumn{6}{l}{\textit{Noise (rating SD), all $K$ gold raters; sign vs.\ clean $\hat B$}}"])
    fam = {"gaussian": "Gauss.", "student_t3": "$t_3$"}
    for rec in noisy.itertuples():
        rows.append([f"{pool[rec.rater_type]}, {fam[rec.noise_family]} {rec.noise_scale_sd:.2f}",
                     fmt(rec.b_abs_error_vs_eval, 4), p1(rec.b_sign_matches_same_k_clean),
                     fmt(rec.curve_mae_ab, 5), fmt(rec.ab_minus_aonly_curve_mae, 5),
                     fmt_ci(rec.ab_minus_aonly_ci_lo, rec.ab_minus_aonly_ci_hi, 5)])
    disp.write_table("lrrrrc",
                     [[r"", r"", r"", r"\multicolumn{3}{c}{CurveMAE}"],
                      [r"\cmidrule(lr){4-6}Setting", r"$\hat B$ error", r"Sign (\%)", r"$A{+}B$",
                       r"$A{+}B-A$-only", r"95\% CI"]],
                     rows, midrules_after=[first_noise])
    figure_style()
    disp.save_figure(draw_rater_noise(summary))
    disp.write_note(note_rater_noise(clean, noisy, spec, art, gap))


# --------------------------------------------------------------------------- skewness
def draw_skewness(frame: pd.DataFrame):
    syn = frame[frame["source"] == "synthetic"].sort_values("skew_r")
    real = frame[frame["source"] == "real"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0))
    ax_a.plot(syn["skew_r"], 100 * syn["correction_fraction_abs_components"], color="#6250d6",
              marker="o", ms=3.4, lw=1.3, label="Synthetic skew-normal family")
    real_name = {"real:SummEval-coherence-ROUGE1": "ROUGE-1", "real:SummEval-coherence-GPT4o": "GPT-4o"}
    for rec in real.itertuples():
        y = 100 * rec.correction_fraction_abs_components
        ax_a.plot(rec.skew_r, y, ls="none", marker="D", ms=4.2, color="#d55181",
                  markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax_a.annotate(real_name[rec.label], (rec.skew_r, y), xytext=(0, 4), textcoords="offset points",
                      ha="center", va="bottom", fontsize=6.5, color=INK)
    ax_a.set_title("(a) Size of the skewness correction", loc="left")
    ax_a.set_xlabel(r"Proxy skewness $\mathbb{E}[Z^3]$")
    ax_a.set_ylabel("Absolute-component\nshare (%)")
    ax_a.set_ylim(0, 56)
    from matplotlib.lines import Line2D
    ax_a.legend(handles=[Line2D([], [], color="#6250d6", marker="o", ms=3.4, lw=1.3, label="Synthetic family"),
                         Line2D([], [], color="#d55181", marker="D", ms=4.0, ls="none",
                                label="SummEval coherence proxies")],
                loc="upper center", bbox_to_anchor=(0.5, -0.48),
                fontsize=6.5, handlelength=1.6)
    ax_b.axhline(0, color=MUTED, lw=0.7, zorder=0)
    ax_b.plot(syn["skew_r"], syn["b_term1_e_jz2"], color="#6250d6", marker="o", ms=3.2, lw=1.2,
              label=r"$\mathbb{E}[\bar J Z^2]$")
    ax_b.plot(syn["skew_r"], syn["b_term2_skew_correction"], color="#d55181", marker="s", ms=3.2,
              lw=1.2, ls="--", label=r"$-\frac{2}{3}\mathbb{E}[\bar J Z]\,\mathbb{E}[Z^3]$")
    ax_b.plot(syn["skew_r"], syn["b_moment"], color=INK, lw=1.0, ls=":", marker="x", ms=3.0,
              label=r"$B$ (both terms vary)")
    ax_b.set_title("(b) Components of $B$ (synthetic)", loc="left")
    ax_b.set_xlabel(r"Proxy skewness $\mathbb{E}[Z^3]$")
    ax_b.set_ylabel("Coefficient value")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.48), fontsize=6.5, handlelength=2.0)
    for ax in (ax_a, ax_b):
        ax.set_xlim(-1.08, 1.08)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.40, wspace=0.50)
    return fig


def build_skewness(group: Group) -> None:
    disp = group.display(
        "ab_skewness", "Appendix (skewness interpretation of B)",
        "exp19 share of the skewness correction in B: synthetic skew-normal family and two SummEval proxies")
    frame = disp.read_csv(EXP19 / "skewcorr_summary.csv")
    manifest = json.loads(disp.input(EXP19 / "synthetic_manifest.json").read_text())
    disp.input(EXP19 / "fig_skewcorr.png")  # provenance: original figure (broken J-bar glyph)
    t1, t2 = frame["b_term1_e_jz2"], frame["b_term2_skew_correction"]
    gaps = dict(
        share=assert_close(frame["correction_fraction_abs_components"], t2.abs() / (t1.abs() + t2.abs()),
                           EXACT, "correction share definition"),
        b_sum=assert_close(frame["b_moment"], t1 + t2, 1e-12, "B = term1 + term2"),
        term2=assert_close(t2, -(2 / 3) * frame["e_jz"] * frame["e_z3"], 1e-12, "term2 formula"),
        a=assert_close(frame["a_moment"], np.sqrt(2) * frame["e_jz"], 1e-12, "A = sqrt(2) E[JZ]"),
        skew=assert_close(frame["skew_r"], frame["e_z3"], 1e-12, "skew(r) = E[Z^3]"),
        b_numeric_rel=float(frame["b_rel_err"].max()))
    by_setting = {f"synthetic:{m['setting']}": m for m in manifest}
    syn = frame[frame["source"] == "synthetic"]
    if set(syn["label"]) != set(by_setting) or len(syn) != 9:
        raise AssertionError("synthetic settings differ from manifest")
    gaps["manifest_skew"] = assert_close(
        syn["skew_r"], [by_setting[x]["empirical_skew_z_at_build_time"] for x in syn["label"]], 1e-9,
        "manifest skewness")
    if gaps["b_numeric_rel"] > 0.05:
        raise AssertionError("closed-form B vs numerical fit cross-check exceeds 5%")
    ix = frame.set_index("label")["correction_fraction_abs_components"]
    round_check(100 * ix["synthetic:ap0_0"], 2, 0.25, "Gaussian (a=0) share")
    round_check(100 * ix["synthetic:ap30_0"], 0, 29, "a=+30 share")
    round_check(100 * ix["synthetic:am30_0"], 0, 52, "a=-30 share")
    round_check(100 * ix["real:SummEval-coherence-ROUGE1"], 0, 26, "ROUGE-1 share")
    round_check(100 * ix["real:SummEval-coherence-GPT4o"], 0, 34, "GPT-4o share")

    order = syn.sort_values("skewnorm_a")["label"].tolist() + [
        "real:SummEval-coherence-ROUGE1", "real:SummEval-coherence-GPT4o"]
    out = frame.set_index("label").loc[order].reset_index()
    names = {"real:SummEval-coherence-ROUGE1": "ROUGE-1", "real:SummEval-coherence-GPT4o": "GPT-4o"}
    out.insert(1, "display_name", [names.get(x, f"$a={int(a):+d}$" if a else "$a=0$")
                                   for x, a in zip(out["label"], out["skewnorm_a"].fillna(0))])
    disp.write_data(out)
    rows = [[r"\multicolumn{6}{l}{\textit{Synthetic skew-normal family, shape $a$ (20,000 items each)}}"]]
    for rec in out.itertuples():
        if rec.source == "real" and rec.display_name == "ROUGE-1":
            rows.append([r"\multicolumn{6}{l}{\textit{SummEval coherence, full pool (1,600 items), proxy:}}"])
        rows.append([rec.display_name, fmt(rec.skew_r, 3), fmt(rec.b_term1_e_jz2, 4),
                     fmt(rec.b_term2_skew_correction, 4), fmt(rec.b_moment, 4),
                     p1(rec.correction_fraction_abs_components)])
    disp.write_table("lrrrrr",
                     ["Setting", r"$\mathbb{E}[Z^3]$", r"$\mathbb{E}[\bar J Z^2]$",
                      r"$-\tfrac23\mathbb{E}[\bar J Z]\mathbb{E}[Z^3]$", r"$B$", r"Share (\%)"],
                     rows, midrules_after=[9])
    figure_style()
    disp.save_figure(draw_skewness(frame))
    disp.write_note(note_skewness(out, gaps))


# --------------------------------------------------------------------------- notes
SOURCES_CELLS = """\
- `results/evaluation_aware_ab/non_wmt_canonical/direct_calibration_predictions.csv.gz`
  (SummEval, GPT-4o, OpenMEVA; exp21 canonical per-cell predictions; WMT deliberately absent)
- `results/evaluation_aware_ab/wmt_fixed_141/direct_calibration_predictions.csv.gz`
  (corrected 141-path WMT)
- exp33 summaries used only for verification: `direct_calibration_strict/main_protocol/`
  (`endpoint_mae_by_d.csv`, `curve_mae.csv`, `paired_diff_curvemae_bootstrap.csv`,
  `calibration_only_skipped.csv`; non-WMT rows only, the WMT rows there are the superseded
  139-path lineage) and `direct_calibration_strict/wmt_fixed_141/` (same four files)."""


def fmt_checks(checks: dict) -> str:
    return "\n".join(f"- `{k}`: max |diff| = {v:.2e}" for k, v in checks.items())


def note_fig2(endpoint, curve, share, checks) -> str:
    def rel(dataset, d):
        r = endpoint[(endpoint["dataset"] == dataset) & (endpoint["d"] == d)].iloc[0]
        return f"{100 * r.rel_reduction:.1f}% [{100 * r.rel_ci_lo:.1f}, {100 * r.rel_ci_hi:.1f}]"

    def rb(dataset, d):
        return f"{100 * share[(share['dataset'] == dataset) & (share['d'] == d)]['r_b_median'].iloc[0]:.1f}%"

    lines_a = "\n".join(f"| {label(k)} | {rel(k, 0.01)} | {rel(k, 0.05)} | {rel(k, 0.10)} |" for k in ORDER)
    curve_red = ", ".join(f"{label(r.dataset)} {100 * r.rel_reduction:.2f}%" for r in curve.itertuples())
    lines_b = "\n".join(f"| {label(k)} | {rb(k, 0.01)} | {rb(k, 0.05)} | {rb(k, 0.10)} |" for k in ORDER)
    units = curve.set_index("dataset")["share_units_ab_lower"]
    return f"""
# ab_fig2: what adding B contributes across the local window (main Figure 2)

## Placement
Main text, Experiments, question 3 ("What does B add, and when is its estimate
unreliable?"). Full text width (5.5 in), about 2.2 in high; vector PDF `artifacts/figures/ab_fig2.pdf`
(PNG preview alongside). Appendix companions: `ab_reduction` (native-unit MAEs and paired
CurveMAE CIs), `ab_bshare` (table with IQRs), `ab_b_stability`, `ab_rater_noise`, `ab_skewness`.

## Design choice
Panel (a) plots the within-dataset relative endpoint-MAE reduction
1 - MAE_A+B(d) / MAE_A-only(d), not native-unit MAEs. Both MAEs are in the same dataset's
human-score units, so the ratio is unitless, and the four settings can share one axis without
comparing native-unit MAEs across different human-score scales (plan Section 6).
Native-unit small multiples were rejected for three reasons. Four MAE panels plus a B-share
panel do not fit legibly at 5.5 x 2.2 in. Separate y-scales still invite cross-dataset
comparison by eye. The ratio answers the question directly: how much of A-only's error
adding B removes at each radius. The native-unit values are in `ab_reduction`.
Panel (b) uses the same radius axis and dataset encoding. Readers can then see that B is a
small part of the predicted gain magnitude (1-13%) but removes a large part of A-only's
error (up to 44%). These are different quantities, and the caption must say so.
Dataset colours avoid the paper's A+B blue and A-only orange, because neither panel plots a
single method. Each dataset also has its own marker. The palette was checked for
colour-vision deficiency (all pairs). The yellow OpenMEVA line is below 3:1 contrast, which
is acceptable because markers, the legend and the appendix tables also identify it.

## What it shows
(a) At every checked radius, adding the B term lowers the aggregate endpoint MAE in all four
settings (point estimates). The relative reduction grows monotonically with d. At d = 0.10 it is
about 44% (SummEval), 29% (GPT-4o), 35% (WMT) and 8% (OpenMEVA). The OpenMEVA intervals include
0 for d <= 0.05 and barely exclude it at d >= 0.075. (b) The median magnitude share of the B
term in the calibration-predicted gain also grows with d, roughly like sqrt(d), because
|B d| / |A sqrt(d)| = (|B|/|A|) sqrt(d). At d = 0.10 it is 4.4% (SummEval), 13.3% (GPT-4o),
8.3% (WMT) and 3.1% (OpenMEVA). B is therefore a modest share of the predicted gain. Omitting
it still leaves a systematic curvature error that grows with d, and adding it removes a large
fraction of A-only's aggregate endpoint error.

## Definitions
- Predictors (pre-pool, calibration-only): moment coefficients A-hat, B-hat are computed on
  the 30% calibration split. A-only predicts A-hat sqrt(d). A+B predicts
  A-hat sqrt(d) + B-hat d, with B = E[J-bar Z^2] - (2/3) E[J-bar Z] E[Z^3].
- Realized gain Delta(d): exact exponential-tilt gain of the held-out 70% evaluation split at
  KL radius d (evaluation human labels used only here).
- Endpoint MAE at d: mean over (path, split) cells of |prediction - Delta(d)|.
- Panel (a): 1 - MAE_A+B(d) / MAE_A-only(d) on identical cells. Positive means adding B lowers MAE.
- Panel (b): R_B(d) = |B-hat d| / (|A-hat sqrt(d)| + |B-hat d|) per path-split, then the dataset
  median. R_B is a magnitude share of the *predicted gain*. It is not the net sign or size of
  B, not a share of the realized gain, and not an MAE improvement. It is computed from the
  calibration coefficients only and says nothing about accuracy.

## Units
Both panels are percentages (unitless). Native-unit MAEs (human-score units of each dataset)
are not displayed or compared across datasets.

## Aggregation and support
- Grid d in {{0.01, 0.025, 0.05, 0.075, 0.10}}; 50 splits per path; split seed 0.
- Paths / path-splits: SummEval 32 / 1,600, GPT-4o 4 / 200, WMT 141 / 7,050 (corrected
  fixed_141 lineage), OpenMEVA 10 / 500.
- Common-valid-cell support, identical to Table 1(b). The 11 SummEval d = 0.10 cells whose
  calibration frontier cannot reach d = 0.10 are excluded for every method, so the SummEval
  endpoint at d = 0.10 uses 1,589 cells and reproduces the original Table 1(a) row
  (A+B 0.0174, A-only 0.0310; agreement within 3e-7 of the independent original d-sweep
  code path). All other radius/dataset cells are complete.
- Panel (a) error bars: 95% percentile path-cluster bootstrap intervals of the ratio. Paths
  are resampled with replacement and all their cells are kept, using 2,000 draws with seed 0
  per dataset x radius. They are path-cluster intervals for the aggregate ratio, not per-split
  intervals. GPT-4o has only 4 paths and OpenMEVA 10, so their intervals are coarse and
  descriptive. Markers are dodged horizontally by up to +/-0.002 in d so the bars do not
  overlap. The plotted radii are the five grid values.
- Panel (b): median over all path-splits (B share does not depend on reachability). The
  interquartile ranges are in `ab_bshare`.

## Protocol and information
Pre-pool: predictions use only calibration proxy scores and calibration human labels. The
held-out evaluation split (70%) supplies realized gains only. The exp31 30%-calibration
coefficients used in (b) are the coefficients of the exp33 A+B predictor in (a). The build
asserts that the predictions are identical to 1e-12 (same split schedule, RNG seed 0).

## Sources
{SOURCES_CELLS}
- `results/coefficient_precision/coefficient_error_rows.csv`
  (exp31; 30%-calibration rows) and `coefficient_error_summary.csv` (verification).
- `results/12_accept_boost/d_sweep_human/d_sweep_summary_overall.csv`
  (original SummEval endpoint row; verification only).
Data: `artifacts/data/ab_fig2.csv` holds the plotted values unrounded, and
`artifacts/data/ab_fig2_path_sums.csv` holds the per-path cell sums that the bootstrap resamples.

## Key results
Relative endpoint-MAE reduction [95% path-cluster CI]:

| Dataset | d = 0.01 | d = 0.05 | d = 0.10 |
|---|---:|---:|---:|
{lines_a}

Median R_B(d):

| Dataset | d = 0.01 | d = 0.05 | d = 0.10 |
|---|---:|---:|---:|
{lines_b}

SummEval's d = 0.10 median share is 4.4% at one decimal (4.447%), not 4.5%.

Curve-level context (Table 1(b) support, from `ab_reduction`): CurveMAE reductions are
{curve_red}.

## Caveats the caption MUST include
1. Panel (a) is a within-dataset ratio of MAEs on identical cells. Native-unit MAEs are not
   comparable across datasets and are not shown.
2. The improvement is an aggregate effect. A+B has lower CurveMAE than A-only on only
   {100 * units.min():.1f}-{100 * units.max():.1f}% of path-split units (OpenMEVA {100 * units['OpenMEVA-MAGS-human_heldout']:.1f}%, SummEval {100 * units['SummEval-human_heldout']:.1f}%).
3. R_B(d) is a magnitude share of the predicted gain, |B-hat d| / (|A-hat sqrt(d)| + |B-hat d|),
   not net B and not an MAE improvement.
4. Error bars are 95% path-cluster bootstrap intervals (paths resampled). GPT-4o has 4 paths
   and OpenMEVA 10, so their intervals are descriptive. OpenMEVA's intervals include 0 at
   d <= 0.05.
5. Pre-pool setting: 30% calibration, 50 splits, held-out exact tilt gain. Support is the common
   valid cells of Table 1(b), so SummEval d = 0.10 uses 1,589 of 1,600 cells.

## Verification performed by `ab.py`
{fmt_checks(checks)}
"""


def note_reduction(endpoint, curve, checks) -> str:
    c = curve.set_index("dataset")
    crow = "\n".join(
        f"| {label(k)} | {c.loc[k, 'curvemae_aonly']:.5f} | {c.loc[k, 'curvemae_ab']:.5f} | "
        f"{c.loc[k, 'diff_ab_minus_aonly']:+.6f} [{c.loc[k, 'diff_ci_lo']:+.6f}, {c.loc[k, 'diff_ci_hi']:+.6f}] | "
        f"{100 * c.loc[k, 'rel_reduction']:.2f}% | {100 * c.loc[k, 'share_units_ab_lower']:.1f}% |" for k in ORDER)
    e10 = endpoint[endpoint["d"] == 0.10].set_index("dataset")
    erow = "\n".join(
        f"| {label(k)} | {e10.loc[k, 'mae_aonly']:.4f} | {e10.loc[k, 'mae_ab']:.4f} | "
        f"{e10.loc[k, 'diff_ab_minus_aonly']:+.4f} ({e10.loc[k, 'diff_ab_minus_aonly']:+.6f}) | "
        f"{100 * e10.loc[k, 'rel_reduction']:.1f}% |" for k in ORDER)
    se = c.loc["SummEval-human_heldout"]
    return f"""
# ab_reduction: A-only versus A+B error by radius and over the curve (appendix)

## Placement
Appendix, A/B contribution section; the numeric companion of Figure 2(a). Two tabular bodies,
to be stacked as panels of one table:
- `ab_reduction.tex`: endpoint MAE by dataset x radius (20 rows).
- `ab_reduction_curvemae.tex`: CurveMAE with paired 95% path-cluster CIs (4 rows).

## Columns
Endpoint panel: Dataset, radius d, Cells, A-only MAE, A+B MAE, difference A+B - A-only
(negative = A+B lower error), relative reduction 1 - A+B/A-only (%), and its 95% path-cluster
bootstrap CI (%). CurveMAE panel: Dataset, A-only, A+B, the paired difference A+B - A-only
("Diff.") with its 95% CI, relative reduction (%), and "Units won": the share of path-split
units whose own CurveMAE (mean over their radius cells) is lower for A+B than for A-only.
Paths and cells are omitted from the CurveMAE panel for width and must go in the caption:
SummEval 32 paths / 7,989 cells, GPT-4o 4 / 1,000, WMT 141 / 35,250, OpenMEVA 10 / 2,500.

## Definitions, units, aggregation
- Predictors and realized gain as in `ab_fig2` (pre-pool, 30% calibration, 50 splits, seed 0,
  exact held-out tilt gain).
- Endpoint MAE: mean of |error| over the cells at that radius (4 decimals). The difference
  uses the same 4 decimals.
- CurveMAE and its paired difference (5 decimals): per-path mean of cell errors over the radius
  grid, then the equal-weight mean over paths. This is exactly the exp33 paired-bootstrap
  convention behind Table 1(b), and it is what the CI resamples.
- CIs: 95% percentile path-cluster bootstrap, 2,000 draws, seed 0. For CurveMAE this is
  draw-for-draw the exp21/exp33 `path_clustered_bootstrap_diff`. Endpoint CIs resample paths
  and keep all their cells.
- Units: MAEs and differences are in each dataset's human-score units. Compare A+B with A-only
  within a dataset, never magnitudes across datasets. Only the relative reduction is unitless.
- Sign convention: differences are A+B minus A-only, as in the exp33 paired files and
  Table 1(b), so negative values favour A+B.

## Support
Common valid cells, identical to Table 1(b). The 11 SummEval cells at d = 0.10 whose
calibration frontier cannot reach d = 0.10 are dropped for both methods. SummEval therefore
has 7,989 CurveMAE cells and 1,589 endpoint cells at d = 0.10, and all other cells are complete.
With this support the SummEval d = 0.10 row matches the original paper's Table 1(a) row
(A+B 0.0174, A-only 0.0310) to within 3e-7. All-cell values (8,000 SummEval cells) are kept
in the CSVs as `allcell_*`: SummEval A-only {se['allcell_curvemae_aonly']:.6f}, A+B
{se['allcell_curvemae_ab']:.6f}, reduction
{100 * (1 - se['allcell_curvemae_ab'] / se['allcell_curvemae_aonly']):.2f}%. Do not mix the two
supports in prose.

## Sources
{SOURCES_CELLS}
The SummEval common-cell CurveMAE CI is recomputed from the saved per-cell rows. The
procedure is validated by reproducing every saved all-cell A+B vs A-only CI to 1e-12. For the
other three datasets common = all cells, so the values equal the saved exp33 paired rows.

## Key results
Endpoint at d = 0.10:

| Dataset | A-only | A+B | A+B - A-only | Reduction |
|---|---:|---:|---:|---:|
{erow}

CurveMAE (common valid cells):

| Dataset | A-only | A+B | A+B - A-only [95% CI] | Reduction | Units with A+B lower |
|---|---:|---:|---:|---:|---:|
{crow}

Rounding traps: the SummEval endpoint difference at d = 0.10 is 0.0136 (0.013646), not 0.0137.
The CurveMAE reductions are 34.28/21.73/26.35/6.36%, or 34.3/21.7/26.4/6.4% at one decimal.

## Caveats the caption MUST include
1. Support is the common valid cells of Table 1(b): 11 SummEval d = 0.10 cells excluded.
2. CIs are 95% path-cluster bootstrap intervals (paths resampled), not per-split intervals.
   The GPT-4o intervals rest on 4 paths and are descriptive.
3. The OpenMEVA CurveMAE CI includes 0 ([{c.loc['OpenMEVA-MAGS-human_heldout', 'diff_ci_lo']:.5f}, {c.loc['OpenMEVA-MAGS-human_heldout', 'diff_ci_hi']:+.6f}]),
   and its endpoint-reduction CIs include 0 for d <= 0.05.
4. The improvement is aggregate. A+B wins on only
   {100 * c['share_units_ab_lower'].min():.1f}-{100 * c['share_units_ab_lower'].max():.1f}% of path-split units.
5. MAEs are in native human-score units and must not be compared across datasets.
6. The CurveMAE panel's paths/cells (listed under Columns) must be stated in the caption.

## Verification performed by `ab.py`
{fmt_checks(checks)}
"""


def note_bshare(share, curve, endpoint) -> str:
    rows = []
    for k in ORDER:
        s = share[share["dataset"] == k].set_index("d")
        rows.append(f"| {label(k)} | " + " | ".join(f"{100 * s.loc[d, 'r_b_median']:.2f}%" for d in D_GRID)
                    + f" | [{100 * s.loc[0.10, 'r_b_p25']:.1f}, {100 * s.loc[0.10, 'r_b_p75']:.1f}]% |"
                    + f" {100 * s.loc[0.10, 'frac_b_negative']:.1f}% |")
    return f"""
# ab_bshare: median magnitude share of the B term in the predicted gain (appendix)

## Placement
Appendix, A/B contribution section; the numeric companion of Figure 2(b). One tabular body
(`ab_bshare.tex`) with rows = datasets and columns = radius.

## Definition (must appear in the caption or adjacent text)
For each path-split, with calibration moment coefficients A-hat and B-hat:

    R_B(d) = |B-hat d| / (|A-hat sqrt(d)| + |B-hat d|).

The table reports the median over path-splits, and the IQR column gives the 25th-75th
percentiles at d = 0.10. R_B is a magnitude share of the *predicted* gain. It is not net B
(sign and cancellation are discarded), not a share of the realized gain, and not an MAE
improvement. For the latter see `ab_reduction`. Because R_B is a ratio, it is unitless and
comparable across datasets as a share. It still reflects each setting's own A/B ratio and is
not a quality score.

## Aggregation, support, protocol
- Coefficients: exp31 30%-calibration moment coefficients (seed-0 sequential split schedule,
  50 splits per path). These are bit-identical to the exp33 A+B predictor (asserted to 1e-12).
- Path-splits: SummEval 1,600, GPT-4o 200, WMT 7,050 (fixed_141), OpenMEVA 500. All
  path-splits are used at every radius, because R_B needs no reachable evaluation radius.
- exp31 stores rows at d in {{0.05, 0.10}} only. R_B at d = 0.01, 0.025 and 0.075 is computed
  from the same (A-hat, B-hat), and the build checks the saved d = 0.05/0.10 medians to 1e-12.
- Percentages: 1 decimal in the table; unrounded values in `artifacts/data/ab_bshare.csv`.

## Sources
- `results/coefficient_precision/coefficient_error_rows.csv`
- `results/coefficient_precision/coefficient_error_summary.csv`
- The exp21-canonical per-cell prediction files (equality check against the A+B predictor).
- Not used: `results/b_stability/contribution_level.csv` (exp22). It uses
  independent SHA-256 split streams and gives slightly different medians, for example
  SummEval 4.5% and GPT-4o 13.8% at d = 0.10. The publication table and these summaries use exp31.

## Key results (median R_B, two decimals; IQR at d = 0.10; share of path-splits with B-hat < 0)

| Dataset | d = 0.01 | 0.025 | 0.05 | 0.075 | 0.10 | IQR at 0.10 | B-hat < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Rounding trap: SummEval at d = 0.10 is 4.4% at one decimal (4.447%), not 4.5%.

## Caveats the caption MUST include
1. R_B is the magnitude share of B-hat d in the calibration-predicted gain. It is not net B
   and not an MAE improvement.
2. Medians are over path-splits of the 30%-calibration pre-pool protocol. The share is
   heterogeneous across path-splits (see the IQR column).
3. B's share grows like sqrt(d) inside the window. The values say nothing about d > 0.10.
"""


def note_b_stability(mine, boot, gap) -> str:
    m30 = mine[mine["calibration_frac"] == 0.3].set_index("dataset")
    b = boot.set_index("dataset")
    rows30 = "\n".join(
        f"| {label(k)} | {m30.loc[k, 'median_scaled_a_error']:.4f} | {m30.loc[k, 'median_scaled_b_error']:.4f} | "
        f"{m30.loc[k, 'median_eval_remainder_abs']:.4f} | {100 * m30.loc[k, 'b_sign_agreement']:.1f}% |" for k in ORDER)
    rowsb = "\n".join(
        f"| {label(k)} | {100 * b.loc[k, 'b_sign_stability_median']:.1f}% | {100 * b.loc[k, 'b_ci_includes_zero_rate']:.1f}% | "
        f"{100 * b.loc[k, 'decision_disagree_rate']:.1f}% | {100 * b.loc[k, 'proceed_ab_rate']:.1f}% / "
        f"{100 * b.loc[k, 'proceed_aonly_rate']:.1f}% |" for k in ORDER)
    return f"""
# ab_b_stability: how reliably B is estimated (appendix)

## Placement
Appendix, B-stability section. The main text gets one paragraph on when B becomes
unreliable. Two tabular bodies, to be stacked as panels (a) and (b) of one table:
- `ab_b_stability.tex`, (a) calibration-size coefficient error at d = 0.10 (exp31).
- `ab_b_stability_bootstrap.tex`, (b) bootstrap sign/CI stability and decision
  disagreement (exp22).
The panels come from different split schedules (exp31 seed-0 sequential, exp22 SHA-256 streams).
Keep them as separate panels and do not merge them row-wise.

## Panel (a) definitions (exp31; medians over path-splits at d = 0.10)
- sqrt(d)|A-hat - A_eval| and d|B-hat - B_eval|: A_eval and B_eval are the *moment-formula*
  coefficients computed on the held-out evaluation split, used only retrospectively. These are
  the first two coefficient-shift terms of the paper's prediction-error decomposition. They
  combine calibration estimation error with calibration-to-evaluation shift, and they are
  triangle-bound components, not a causal decomposition of each split's error.
- Remainder: |Delta_eval(d) - (A_eval sqrt(d) + B_eval d)|, the held-out exact tilt gain minus
  its own evaluation-moment two-term curve (finite-range local-law remainder).
- Sign agr.: share of path-splits with sign(B-hat_cal) = sign(B_eval,moment). This is a
  **moment-based** sign agreement. The earlier curve-fit audit instead compared B-hat_cal
  with a B *fitted to the held-out exact curve* (curve-fit sign). The two definitions answer different questions and must remain separate. This table does not report curve-fit signs.
- Calibration fractions 5/10/30%, 50 nested splits per path, seed 0. Path-splits: 1,600 /
  200 / 7,050 / 500. These are the same as panel (b), so the column is omitted from (a). Values are in native human-score units (4 decimals). Compare them across
  calibration sizes within a dataset, not across datasets.

## Panel (b) definitions (exp22; 30% calibration, 50 splits, 300 prompt-cluster bootstrap draws within each calibration split)
- Clusters: SummEval articles, OpenMEVA stories, WMT source segments.
- Median SE(B-hat): bootstrap standard error of B-hat, with the median over path-splits (native units).
- Sign kept: per path-split, the share of bootstrap draws with the same sign as B-hat. The
  table gives the dataset median.
- CI includes 0: share of path-splits whose 95% percentile bootstrap CI for B contains 0.
  These are *path-split* CIs (one per calibration split), not dataset-level CIs. Sign stability
  and zero-inclusion are different summaries of the same bootstrap distribution, and high
  sign stability does not establish nonzero population curvature.
- Disagree: share of path-splits where A-only and A+B make different proceed/abstain
  decisions at d = 0.10. The rule is: proceed if the 2.5th bootstrap percentile of the
  predicted gain exceeds gamma = 0.1 sigma_J,cal. Gamma is an operational, deployment-specific
  threshold. For GPT-4o, the A+B proceed rate is 83.0/27.5/0% at gamma/sigma_J = 0.05/0.10/0.20
  (`gamma_sensitivity.csv`).

## Sources
- `results/coefficient_precision/coefficient_error_rows.csv`
  (recomputed medians) and `coefficient_error_summary.csv` (asserted equal, max |diff| {gap:.1e}).
- `results/b_stability/coefficient_level.csv`, `decision_level.csv`,
  `b_stability_run_spec.json` (WMT = 7,050 fixed_141 path-splits).

## Key results
At 30% calibration (d = 0.10):

| Dataset | sqrt(d)|dA| | d|dB| | Remainder | Moment sign agr. |
|---|---:|---:|---:|---:|
{rows30}

Ranges at 30%: sqrt(d)|dA| {m30['median_scaled_a_error'].min():.4f}-{m30['median_scaled_a_error'].max():.4f}, d|dB| {m30['median_scaled_b_error'].min():.4f}-{m30['median_scaled_b_error'].max():.4f}, remainder {m30['median_eval_remainder_abs'].min():.4f}-{m30['median_eval_remainder_abs'].max():.4f}.
d|dB| decreases monotonically from 5% to 10% to 30% calibration in every dataset.

| Dataset | Bootstrap sign kept (median) | Path-split CIs including 0 | Decision disagreement | Proceed A+B / A-only |
|---|---:|---:|---:|---:|
{rowsb}

## Caveats the caption MUST include
1. B_eval and A_eval are held-out *moment* coefficients, and the sign agreement is
   moment-based, not curve-fit.
2. "CI includes 0" counts per-path-split bootstrap CIs. High bootstrap sign stability coexists
   with 25-53% of CIs including 0.
3. The decision columns depend on the operational threshold gamma = 0.1 sigma_J,cal.
4. Panels (a) and (b) use different split schedules (exp31 vs exp22).
5. Error columns are in native human-score units and must not be compared across datasets.
"""


def note_rater_noise(clean, noisy, spec, art, gap) -> str:
    c = clean.set_index(["rater_type", "gold_rater_count"])
    n = noisy.set_index(["rater_type", "noise_family", "noise_scale_sd"])

    def cm(pool, k):
        r = c.loc[(pool, k)]
        return (f"B error {r.b_abs_error_vs_eval:.4f}, CurveMAE A+B {r.curve_mae_ab:.4f} "
                f"(A-only {r.curve_mae_aonly:.4f}), sign vs B_eval {100 * r.b_sign_matches_eval:.1f}%")

    def nz(pool, fam, s):
        r = n.loc[(pool, fam, s)]
        return f"{r.ab_minus_aonly_curve_mae:+.5f} [{r.ab_minus_aonly_ci_lo:+.5f}, {r.ab_minus_aonly_ci_hi:+.5f}]"

    return f"""
# ab_rater_noise: sensitivity of B to rater count and rating noise (appendix)

## Placement
Appendix, rater count and noise. The main text gets one sentence in the "when is B unreliable"
paragraph. Outputs: table `ab_rater_noise.tex` and figure `ab_rater_noise.pdf` (5.5 x 1.95 in).

## Figure provenance
The figure is restyled from `results/rater_noise/fig_b_rater_noise_stability.pdf`
(registered as an input; produced by `experiments/exp26_b_rater_noise_stability/make_artifacts.py`).
It plots the same summary rows and the same panel filters (artifact_spec: {art['figure_filters']}).
Changes: camera-ready font sizes and width; "Turker" relabelled "Crowd"; the sign panel's y-axis
starts at 90% instead of 0%, because the old axis hid the differences; panel (c) adds the saved
95% path-cluster CIs, which the original omitted; and colours avoid the A+B/A-only method hues.

## Design (exp26)
- Data: SummEval (32 paths = 4 dimensions x {{expert, crowd}} x held-out proxy rater), 50 nested
  item splits, seed 0 with SHA-256-derived streams. Only the 30% calibration rows are shown;
  10/20/50% rows are in the source.
- Proxy: {spec['human_proxy_rule']} (a *held-out human rater as proxy*, not a model).
- Calibration gold J: {spec['calibration_gold_rule']}. Expert pools have up to 2 non-proxy
  raters and crowd pools up to 4.
- Evaluation gold: {spec['evaluation_gold_rule']}.
- Noise: {spec['noise_policy']}. Scales of 0.25/0.5/1.0 group rating SD are nominal
  pre-clipping scales, so the realized variance can be smaller.
- Table columns: "B-hat error" = mean |B-hat - B_eval|, the absolute error of the calibration
  moment B against the clean evaluation moment B. "Sign (%)" uses a reference that differs by
  block, as each block header states. Clean block: agreement of sign(B-hat) with the clean
  evaluation B_eval. Noise block: agreement with the clean calibration estimate at the same
  rater count, which is the quantity behind the >= 99.3% / >= 93.0% claims. Both sign rates
  for every row are in `artifacts/data/ab_rater_noise.csv` (`b_sign_matches_eval`,
  `b_sign_matches_same_k_clean`). CurveMAE is over d in {{0.01, 0.025, 0.05, 0.075, 0.10}}.
  "A+B - A-only" is the paired CurveMAE difference with a 5,000-draw path-cluster bootstrap
  95% CI over path-level means. Noise rows are labelled by family (Gauss., t3) and nominal
  scale in group rating SD. "k of K" is the number of gold raters averaged out of those available.
- Aggregation: {art['summary_unit']}.

## Key results (30% calibration)
- Expert, 1 to 2 gold raters: {cm('expert', 1)} -> {cm('expert', 2)}.
- Crowd, 1 to 4 gold raters: {cm('turker', 1)} -> {cm('turker', 4)}.
- Under injected noise with all gold raters, sign agreement with the clean estimate is at least
  {100 * noisy[noisy.rater_type == 'expert']['b_sign_matches_same_k_clean'].min():.1f}% (expert) and
  {100 * noisy[noisy.rater_type == 'turker']['b_sign_matches_same_k_clean'].min():.1f}% (crowd).
- A+B - A-only CurveMAE, expert: Gaussian 0.25 SD {nz('expert', 'gaussian', 0.25)};
  0.5 SD {nz('expert', 'gaussian', 0.5)} (CI includes 0); 1.0 SD {nz('expert', 'gaussian', 1.0)}
  (A+B worse). The t3 noise results are 0.5 SD {nz('expert', 'student_t3', 0.5)} and 1.0 SD
  {nz('expert', 'student_t3', 1.0)}.
- Crowd: Gaussian 0.25 SD {nz('turker', 'gaussian', 0.25)}; 0.5 SD {nz('turker', 'gaussian', 0.5)};
  t3 0.5 SD {nz('turker', 'student_t3', 0.5)}. A+B's advantage is gone at 1.0 SD for experts
  and at 0.5 SD for crowd ratings.

## Sources
- `results/rater_noise/condition_summary.csv` (all values).
- `results/rater_noise/selected_conditions.csv` (asserted equal, max |diff| {gap:.1e}).
- `results/rater_noise/run_spec.json`, `artifact_spec.json`,
  `fig_b_rater_noise_stability.pdf` (provenance).

## Caveats the caption MUST include
1. The proxy is a held-out human rater, not a model, and gold labels come from the remaining
   raters.
2. exp26 CurveMAE values come from their own split streams and rater-subset construction. They
   are not comparable with the exp33 CurveMAE of Table 1(b) or `ab_reduction`. Similarly, the
   exp35 tail/outlier analyses use full-pool coefficients, which are a different estimand.
3. Noise scales are nominal (pre-clipping), and evaluation gold is always the clean mean of
   all non-proxy raters.
4. B's sign is stable under the audited noise, but its predictive usefulness is less stable:
   A+B loses its advantage at 1.0 SD (experts) and 0.5 SD (crowd).
"""


def note_skewness(out, gaps) -> str:
    rows = "\n".join(
        f"| {r.display_name} | {r.skew_r:+.3f} | {r.b_term1_e_jz2:+.4f} | {r.b_term2_skew_correction:+.4f} | "
        f"{r.b_moment:+.4f} | {100 * r.correction_fraction_abs_components:.2f}% |" for r in out.itertuples())
    return f"""
# ab_skewness: when the skewness correction in B matters (appendix)

## Placement
Appendix, skewness interpretation. The main text gets one short explanation beside the
coefficient formula. Outputs: table `ab_skewness.tex` and figure `ab_skewness.pdf` (redrawn).

## Definitions
B = E[J-bar Z^2] - (2/3) E[J-bar Z] E[Z^3], with Z the standardized proxy and J-bar the
centred target. The correction share is

    |-(2/3) E[J-bar Z] E[Z^3]| / (|E[J-bar Z^2]| + |-(2/3) E[J-bar Z] E[Z^3]|).

The denominator is the sum of the two components' absolute values, not |B|. This avoids
shares above 100% when the components cancel. Skewness here is skew(r) = E[Z^3].

## Data and protocol (exp19)
- Synthetic family: Z ~ standardized skew-normal(a), a in {{-30, -8, -3, -1, 0, 1, 3, 8, 30}},
  20,000 items each. The fixed conditional mean is J = 0 + 1.3 Z + 0.6 (Z^2 - 1), and
  r = 5 + 2 Z. Per-setting seeds are in `synthetic_manifest.json`.
- Real: SummEval coherence, full pool of 1,600 summaries, ROUGE-1 or GPT-4o as proxy. These
  are descriptive full-pool coefficients, not calibration-split predictors.
- Closed-form moments were cross-checked against an independent exact-tilt local fit
  (d_max = 5e-4). The max relative B error is {100 * gaps['b_numeric_rel']:.1f}% (< 5%).

## Figure
Redrawn from `results/skewness/skewcorr_summary.csv`. The original
`fig_skewcorr.png` (registered for provenance) rendered J-bar as a broken glyph ("E[]Z]").
This version uses mathtext E[J-bar Z^2]. It also plots the two SummEval proxies as points
rather than vertical lines, and it uses the paper's width and fonts.
(a) Correction share vs skewness. (b) The two components and their sum B for the synthetic
family.

## Key results

| Setting | E[Z^3] | E[J-bar Z^2] | Skew correction | B | Share |
|---|---:|---:|---:|---:|---:|
{rows}

Summary: the share is about 0.25% for a symmetric (Gaussian, a = 0) proxy, 29% at a = +30 and
52% at a = -30. For the real SummEval coherence proxies it is 26% (ROUGE-1) and 34% (GPT-4o).
The 0.25%/29%/52% values are synthetic-family settings, not SummEval. The table uses one decimal,
following the build convention (0.3%/29.4%/52.0%). If the prose quotes "0.25%", say it
is the Gaussian synthetic setting.
Sign: the correction is -(2/3) E[J-bar Z] E[Z^3]. For a positively aligned proxy
(E[J-bar Z] > 0, all rows here), positive skew makes it negative (it lowers B), and negative
skew makes it positive.

## Caveats the caption MUST include
1. The share is |correction| / (|E[J-bar Z^2]| + |correction|), not |correction| / |B|.
2. Changing the skew-normal shape also changes higher moments, so this is a family-level
   intuition audit, not an isolated causal effect of skewness.
3. There are only two real proxy paths (SummEval coherence, full pool), so the observed range
   of skewness across datasets is not generalized.

## Verification performed by `ab.py`
- share recomputed from components: {gaps['share']:.1e}; B = term1 + term2: {gaps['b_sum']:.1e};
  term2 = -(2/3) E[J-bar Z] E[Z^3]: {gaps['term2']:.1e}; A = sqrt(2) E[J-bar Z]: {gaps['a']:.1e};
  skew vs manifest: {gaps['manifest_skew']:.1e}.
"""


# --------------------------------------------------------------------------- main
def main() -> None:
    group = Group("ab")
    shared = build_fig2_reduction_bshare(group)
    build_b_stability(group, shared["rows31"])
    build_rater_noise(group)
    build_skewness(group)
    path = group.write_manifest()
    print(f"ab: {len(group.displays)} displays -> {path}")


if __name__ == "__main__":
    main()
