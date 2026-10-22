#!/usr/bin/env python3
"""Best-of-N (BoN) scope displays for the publication, built from saved results only.

Displays (all appendix except the definitions note, which feeds the main text):

1. bon_nladder                   exact-BoN N-ladder per dataset
2. bon_inwindow_cases            the four nontrivial in-window BoN points
3. bon_softened_rules            exp23 mixture / rank-tilt rules inside the checked window
4. bon_prompt_conditional_scope  prompt-conditional scope table + figure
5. bon_kl_ess_definitions        KL / ESS / relative-ESS definitions and closed-form checks

Nothing here retrains, calls an API or writes outside ``out/``.  The prepared
human/proxy score files are read only to recompute deterministic quantities that
the saved CSVs do not store (top-score ties and pooled-tilt reachability)
and to re-verify the saved BoN points independently.
"""
from __future__ import annotations

import sys

# Importing the experiment module and shared lib must not leave
# __pycache__ directories in experiment or result trees.
sys.dont_write_bytecode = True

import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from fractions import Fraction  # noqa: E402
from itertools import product  # noqa: E402
from pathlib import Path  # noqa: E402

# Pin the PDF CreationDate so figure hashes in the manifest are reproducible.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    CORE, RR, TEXT_WIDTH_IN, Group, check, figure_style, fmt, rel, require_wmt_fixed,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
DATA_DIR = RR / "data"
SUMMEVAL_LONG = DATA_DIR / "summeval_long.csv"
WMT_LONG = DATA_DIR / "wmt_mqm_long.csv"
OPENMEVA_LONG = DATA_DIR / "openmeva_long.csv"
ROUGE1_PROXY = DATA_DIR / "proxy_rouge1.csv"
GPT4O_PROXY = DATA_DIR / "model_proxy_4o_aggregated.csv"
LIB_UTILS = RR / "lib" / "common_experiment_utils.py"

BON_DIR = RR / "results" / "bon_scope"
BON_POINTS = BON_DIR / "bon_gate_points.csv"
BON_SPEC = BON_DIR / "run_spec.json"
EXP20_BUILDER = RR / "experiments" / "exp20_bon_window_gate" / "build_bon_gate.py"


MIX_DIR = RR / "results" / "rank_selection_scope"
MIX_ROWS = MIX_DIR / "mixture_ranktilt_rows.csv"
MIX_MACRO = MIX_DIR / "inwindow_baseline_path_macro.csv"
MIX_SPEC = MIX_DIR / "mixture_ranktilt_run_spec.json"
EXP23_BUILDER = RR / "experiments" / "exp23_mixture_ranktilt" / "build_mixture_ranktilt.py"

PC_DIR = RR / "verification" / "fixtures" / "prompt_bon_reachability_reference"
PC_SCRIPT = RR / "experiments" / "15_prompt_conditional_bon_scope" / "01_prompt_conditional_bon_audit.py"
PC_FIGURE_GENERATOR = CORE / "artifact_tools" / "make_tail_bon_artifacts.py"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW = 0.10          # checked local KL window
BRIDGE = 0.25          # audited bridge window of the prompt-conditional audit
N_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
N_SHOW = [2, 3, 4, 8, 16, 64]
PC_N = [2, 4, 8, 16, 32, 64]

# (exp20 label, display label, short label)
DATASETS = [
    ("SummEval-human_heldout", "SummEval, held-out human", "SummEval human"),
    ("SummEval-ROUGE1", "SummEval, ROUGE-1", "SummEval ROUGE-1"),
    ("SummEval-GPT4o", "SummEval, GPT-4o", "SummEval GPT-4o"),
    ("WMT-MQM-human_heldout", "WMT MQM, held-out human", "WMT MQM"),
    ("OpenMEVA-MAGS-human_heldout", "OpenMEVA-MAGS, held-out human", "OpenMEVA-MAGS"),
]
LABEL = {key: label for key, label, _ in DATASETS}
SHORT = {key: short for key, _, short in DATASETS}
EXPECTED_PATHS = {"SummEval-human_heldout": 32, "SummEval-ROUGE1": 4, "SummEval-GPT4o": 4,
                  "WMT-MQM-human_heldout": 141, "OpenMEVA-MAGS-human_heldout": 10}
INWINDOW_IDS = ["consistency__expert__heldout_0", "consistency__expert__heldout_1",
                "consistency__expert__heldout_2", "fluency__expert__heldout_2"]

# Error columns of bon_gate_points.csv and their display names.
GATE_ERRORS = {
    "A+B": "abs_err_theory_vs_bon",
    "A-only": "abs_err_aonly_vs_bon",
    "A_nu": "abs_err_anu_vs_bon",
    "empirical": "abs_err_empfit_shared_vs_bon",
    "zero": "abs_err_zero_vs_bon",
}
GATE_PREDS = {
    "A+B": "theory_local_delta_at_dbon",
    "A-only": "aonly_delta_at_dbon",
    "A_nu": "anu_delta_at_dbon",
    "empirical": "empfit_shared_delta_at_dbon",
}
MIX_ERRORS = {
    "A+B": "abs_err_theory",
    "A-only": "abs_err_aonly",
    "A_nu": "abs_err_anu",
    "empirical_fit": "abs_err_empfit",
    "zero": "abs_err_zero",
}
MIX_DATASETS = ["SummEval-human_heldout", "SummEval-GPT4o", "WMT-MQM-human_heldout",
                "OpenMEVA-MAGS-human_heldout"]
MECH_LABEL = {"alpha_mixture": "Mixture", "rank_tilt": "Rank tilt"}

# Dataset colours for the prompt-conditional figure (datasets, not methods, so the
# method palette in common.COLORS is deliberately not reused).
DATASET_COLORS = {"SummEval": "#3949AB", "OpenMEVA-MAGS": "#2E7D32"}
DATASET_MARKERS = {"SummEval": "o", "OpenMEVA-MAGS": "s"}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_LIB = None


def lib():
    """lib/common_experiment_utils, imported once."""
    global _LIB
    if _LIB is None:
        sys.path.insert(0, str(RR / "lib"))
        import common_experiment_utils as module  # noqa: E402
        _LIB = module
    return _LIB


def quiet(fn, *args, **kwargs):
    """Run a repo helper while discarding its progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def fmt_int(value) -> str:
    """Integer with LaTeX thousands separator."""
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def fmt_pct_cell(count: int, total: int) -> str:
    return f"{count} ({fmt(count / total, 1, percent=True)})"


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def bon_probs(r: np.ndarray, n_best: int) -> np.ndarray:
    """Exact BoN selection probabilities (Appendix A.7 w_N), uniform reference.

    Draws are i.i.d. uniform with replacement; ties at the maximum are broken
    uniformly.  p_i = w_N(r_i) / m with w_N(u) = (F(u)^N - F_-(u)^N) / (F(u) - F_-(u)).
    Written independently of the repository engines.
    """
    r = np.asarray(r, dtype=float)
    m = r.size
    _, inv, counts = np.unique(r, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts) / m
    lower = upper - counts / m
    w = (upper ** n_best - lower ** n_best) / (upper - lower)
    return w[inv] / m


def bon_probs_binomial(r: np.ndarray, n_best: int) -> np.ndarray:
    """Same probabilities from the explicit tie-count binomial sum of Appendix A.7."""
    r = np.asarray(r, dtype=float)
    m = r.size
    _, inv, counts = np.unique(r, return_inverse=True, return_counts=True)
    lower = (np.cumsum(counts) - counts) / m
    atom = counts / m
    sel = np.array([sum(math.comb(n_best - 1, j) * lower[g] ** (n_best - 1 - j) * atom[g] ** j / (j + 1)
                        for j in range(n_best)) for g in range(len(counts))])
    return sel[inv] / m * n_best


def kl_uniform(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] * p.size)))


def rel_ess(p: np.ndarray, p0: np.ndarray | None = None) -> float:
    p = np.asarray(p, dtype=float)
    p0 = np.full(p.size, 1.0 / p.size) if p0 is None else np.asarray(p0, dtype=float)
    return float(1.0 / np.sum(p ** 2 / p0))


def tilt_at_kl(r: np.ndarray, j: np.ndarray, target: float) -> tuple[float, float]:
    """Exact pooled exponential tilt at KL = target by bisection on t (independent check).

    Returns (gain q(t*) - q(0), t*).  Requires target < log(m / n_top).
    """
    r = np.asarray(r, dtype=float)
    j = np.asarray(j, dtype=float)

    def stats(t):
        a = t * r
        w = np.exp(a - a.max())
        w /= w.sum()
        return kl_uniform(w), float(w @ j)

    lo, hi = 0.0, 1.0
    while stats(hi)[0] < target:
        lo, hi = hi, hi * 2.0
        if hi > 1e9:
            raise RuntimeError("target KL not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if stats(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return stats(hi)[1] - float(np.mean(j)), hi


# ---------------------------------------------------------------------------
# Shared data: BoN points and rebuilt paths
# ---------------------------------------------------------------------------
_PATHS: dict | None = None
_POINTS: pd.DataFrame | None = None
_RECHECK: dict = {}


def path_inputs() -> list[Path]:
    return [SUMMEVAL_LONG, WMT_LONG, OPENMEVA_LONG, ROUGE1_PROXY, GPT4O_PROXY, LIB_UTILS, EXP20_BUILDER]


def get_paths(display) -> dict:
    """Rebuild the 191 exp20 paths from the prepared score files (cached)."""
    global _PATHS
    for path in path_inputs():
        display.input(path)
    if _PATHS is not None:
        return _PATHS
    u = lib()
    se = pd.read_csv(SUMMEVAL_LONG)
    wmt = pd.read_csv(WMT_LONG)
    om = pd.read_csv(OPENMEVA_LONG)
    paths = {
        "SummEval-human_heldout": quiet(lambda: list(u.generate_human_paths(
            se, ["article_id", "model_id"], "rater_idx", "score", ["dimension", "rater_type"], 200, 3))),
        "WMT-MQM-human_heldout": quiet(lambda: list(u.generate_human_paths(
            wmt, ["item_id"], "rater", "score", ["dataset_id"], 200, 3))),
        "OpenMEVA-MAGS-human_heldout": quiet(lambda: list(u.generate_human_paths(
            om, ["story_id", "model_id"], "rater_idx", "score", ["dataset"], 100, 3))),
    }
    for key, source, column in [("SummEval-ROUGE1", ROUGE1_PROXY, "proxy_score"),
                                ("SummEval-GPT4o", GPT4O_PROXY, "proxy_score_mean")]:
        proxy = pd.read_csv(source).groupby(["dimension", "item_id"])[column].mean().reset_index()
        paths[key] = quiet(lambda: list(u.generate_model_paths(
            se, proxy, ["article_id", "model_id"], "score", ["dimension"], "item_id", column, 200)))
    for key, expected in EXPECTED_PATHS.items():
        check(len(paths[key]), expected, 0, f"rebuilt {key} path count")
    _PATHS = {key: {p.path_id: p for p in plist} for key, plist in paths.items()}
    return _PATHS


def path_info(paths: dict) -> pd.DataFrame:
    rows = []
    for key, by_id in paths.items():
        for pid, p in by_id.items():
            r = np.asarray(p.proxy_scores, dtype=float)
            n_top = int(np.sum(r == r.max()))
            rows.append(dict(dataset_label=key, path_id=pid, m=r.size, n_unique_proxy=int(np.unique(r).size),
                             top_proxy_score=float(r.max()), n_top=n_top, share_top=n_top / r.size,
                             sup_kl=math.log(r.size / n_top)))
    return pd.DataFrame(rows)


def get_points(display, paths: dict) -> pd.DataFrame:
    """Read, validate and independently re-verify bon_gate_points.csv."""
    global _POINTS
    pts = display.read_csv(BON_POINTS)
    display.input(BON_SPEC)
    if _POINTS is not None:
        return _POINTS.copy()
    require_wmt_fixed(pts, label="bon_gate_points", path_col="path_id")
    counts = pts.groupby("dataset_label")["path_id"].nunique().to_dict()
    if counts != EXPECTED_PATHS:
        raise AssertionError(f"bon_gate_points path counts {counts}")
    if not (pts.groupby(["dataset_label", "path_id"])["N"].apply(sorted).apply(lambda x: x == N_LADDER)).all():
        raise AssertionError("bon_gate_points: unexpected N ladder")
    check((pts["d_bon"] - pts["d_bon_recheck"]).abs().max(), 0.0, 0.0, "d_bon vs d_bon_recheck")
    # Independent recomputation of every (path, N) point from the prepared scores.
    worst = dict(d=0.0, ess_rel=0.0, gain=0.0, a=0.0, b=0.0)
    m_by_path = {}
    indexed = pts.set_index(["dataset_label", "path_id", "N"]).sort_index()
    u = lib()
    for key, by_id in paths.items():
        for pid, p in by_id.items():
            r = np.asarray(p.proxy_scores, dtype=float)
            j = np.asarray(p.gold_scores, dtype=float)
            m_by_path[(key, pid)] = r.size
            coeff = u.theory_coefficients(j, r)
            for n_best in N_LADDER:
                prob = bon_probs(r, n_best)
                saved = indexed.loc[(key, pid, n_best)]
                worst["d"] = max(worst["d"], abs(kl_uniform(prob) - saved["d_bon"]))
                worst["ess_rel"] = max(worst["ess_rel"], abs(1.0 / np.sum(prob ** 2) - saved["ess_bon"]) / r.size)
                worst["gain"] = max(worst["gain"], abs(float(prob @ j - j.mean()) - saved["delta_bon"]))
                worst["a"] = max(worst["a"], abs(coeff.a_exact - saved["a_exact"]))
                worst["b"] = max(worst["b"], abs(coeff.b_theory - saved["b_theory"]))
    for name, value in worst.items():
        check(value, 0.0, 1e-10, f"independent BoN recomputation max |diff| {name}")
    _RECHECK.update(worst)
    _RECHECK["n_points"] = len(pts)
    pts = pts.copy()
    pts["m"] = [m_by_path[(k, p)] for k, p in zip(pts["dataset_label"], pts["path_id"])]
    n1 = pts[pts["N"] == 1]
    check((n1["ess_bon"] - n1["m"]).abs().max(), 0.0, 1e-6, "ESS at N=1 equals m")
    pts["rel_ess"] = pts["ess_bon"] / pts["m"]
    # Stored predictions are A sqrt(d) + B d etc.; recheck the algebra.
    sd = np.sqrt(pts["d_bon"].clip(lower=0))
    check((pts["a_exact"] * sd + pts["b_theory"] * pts["d_bon"] - pts["theory_local_delta_at_dbon"]).abs().max(),
          0.0, 1e-12, "A+B prediction algebra")
    check((pts["a_exact"] * sd - pts["aonly_delta_at_dbon"]).abs().max(), 0.0, 1e-12, "A-only prediction algebra")
    window_flag = (pts["d_bon"] <= WINDOW + 1e-12).astype(int)
    check((window_flag != pts["in_shared_window"]).sum(), 0, 0, "in_shared_window flag")
    _POINTS = pts
    return pts.copy()


# ---------------------------------------------------------------------------
# Display 1: N-ladder
# ---------------------------------------------------------------------------
def build_nladder(group: Group) -> dict:
    disp = group.display("bon_nladder", "appendix",
                         "Exact pooled BoN N-ladder per dataset: induced KL, relative ESS, in-window "
                         "counts and A+B/A-only errors")
    paths = get_paths(disp)
    pts = get_points(disp, paths)
    info = path_info(paths)
    pts = pts.merge(info[["dataset_label", "path_id", "sup_kl", "n_top", "share_top"]],
                    on=["dataset_label", "path_id"], how="left")
    n_nontrivial_per_path = len(N_LADDER) - 1

    summary_rows, byn_rows = [], []
    scopes = [(key, pts[pts["dataset_label"] == key]) for key, _, _ in DATASETS] + [("All", pts)]
    for key, g in scopes:
        nt = g[g["N"] > 1]
        iw = nt[nt["d_bon"] <= WINDOW]
        n2 = g[g["N"] == 2]
        pinfo = g.drop_duplicates(["dataset_label", "path_id"])
        sat_paths = pinfo[pinfo["sup_kl"] <= BRIDGE]
        row = dict(dataset_label=key, n_paths=len(pinfo),
                   m_min=int(pinfo["m"].min()), m_max=int(pinfo["m"].max()),
                   n_nontrivial_points=len(nt),
                   n_le_010=int((nt["d_bon"] <= WINDOW).sum()),
                   n_le_025=int((nt["d_bon"] <= BRIDGE).sum()),
                   n_paths_any_le_010=len(iw.drop_duplicates(["dataset_label", "path_id"])),
                   n_paths_sup_kl_le_025=len(sat_paths),
                   n_le_025_from_N_ge3=int(((nt["N"] >= 3) & (nt["d_bon"] <= BRIDGE)).sum()),
                   min_d_N2=float(n2["d_bon"].min()), max_d_N2=float(n2["d_bon"].max()),
                   n_inwindow_points=len(iw))
        if len(nt) != row["n_paths"] * n_nontrivial_per_path:
            raise AssertionError(f"{key}: nontrivial point count")
        row["share_le_010"] = row["n_le_010"] / len(nt)
        row["share_le_025"] = row["n_le_025"] / len(nt)
        for name, column in GATE_ERRORS.items():
            row[f"inwindow_mae_{name}"] = float(iw[column].mean()) if len(iw) else float("nan")
            row[f"N2_mae_{name}"] = float(n2[column].mean())
        for n_best in N_LADDER:
            gn = g[g["N"] == n_best]
            row[f"median_d_N{n_best}"] = float(gn["d_bon"].median())
            row[f"median_rel_ess_N{n_best}"] = float(gn["rel_ess"].median())
            byn_rows.append(dict(
                dataset_label=key, N=n_best, n_paths=len(gn),
                median_d=float(gn["d_bon"].median()), q25_d=float(gn["d_bon"].quantile(0.25)),
                q75_d=float(gn["d_bon"].quantile(0.75)), min_d=float(gn["d_bon"].min()),
                max_d=float(gn["d_bon"].max()), median_rel_ess=float(gn["rel_ess"].median()),
                min_rel_ess=float(gn["rel_ess"].min()), max_rel_ess=float(gn["rel_ess"].max()),
                median_ess=float(gn["ess_bon"].median()),
                n_le_010=int((gn["d_bon"] <= WINDOW).sum()), n_le_025=int((gn["d_bon"] <= BRIDGE).sum()),
                median_gain=float(gn["delta_bon"].median()),
                mae_A_plus_B=float(gn[GATE_ERRORS["A+B"]].mean()),
                mae_A_only=float(gn[GATE_ERRORS["A-only"]].mean()),
                mae_empirical=float(gn[GATE_ERRORS["empirical"]].mean())))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    byn = pd.DataFrame(byn_rows)
    s = summary.set_index("dataset_label")

    # Audited numbers.
    for key, value in [("SummEval-human_heldout", 0.1718), ("SummEval-ROUGE1", 0.1931),
                       ("SummEval-GPT4o", 0.1889), ("WMT-MQM-human_heldout", 0.1860),
                       ("OpenMEVA-MAGS-human_heldout", 0.1656)]:
        check(s.loc[key, "median_d_N2"], value, 5e-5, f"N=2 median d {key}")
    check(s.loc["All", "n_le_010"], 4, 0, "nontrivial in-window BoN points")
    check(s.loc["SummEval-human_heldout", "n_le_010"], 4, 0, "SummEval-human in-window points")
    check(s.loc["All", "n_paths_any_le_010"], 4, 0, "paths with an in-window nontrivial point")
    for key, value in [("A+B", 0.01968), ("A-only", 0.08836), ("A_nu", 0.12381),
                       ("zero", 0.24896), ("empirical", 0.00432)]:
        check(s.loc["All", f"inwindow_mae_{key}"], value, 5e-6, f"in-window MAE {key}")
    atomless = {n: (math.log(n) - (n - 1) / n, (2 * n - 1) / n ** 2) for n in (2, 4, 8)}
    for n_best, (d_closed, rel_closed) in atomless.items():
        check(s.loc["SummEval-ROUGE1", f"median_rel_ess_N{n_best}"], rel_closed, 5e-4,
              f"ROUGE-1 median RelESS N={n_best} vs atomless closed form")
        check(s.loc["SummEval-ROUGE1", f"median_d_N{n_best}"], d_closed, 5e-4,
              f"ROUGE-1 median d N={n_best} vs atomless closed form")

    disp.write_data(summary)
    disp.write_data(byn, suffix="_byN")

    # Table (a): per-dataset summary.
    header = [
        ["", "", r"\multicolumn{2}{c}{Median at $N=2$}",
         r"\multicolumn{2}{c}{Nontrivial points ($N\ge2$)}", r"\multicolumn{2}{c}{In-window MAE}"],
        [r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}Setting", "Paths",
         r"$d_{\mathrm{BoN}}$", "RelESS", r"$d\le0.10$", r"$d\le0.25$", "A+B", "A-only"],
    ]
    rows = []
    for key, _ in scopes:
        r = s.loc[key]
        label = "All settings" if key == "All" else SHORT[key]
        rows.append([label, str(int(r["n_paths"])), fmt(r["median_d_N2"], 3), fmt(r["median_rel_ess_N2"], 3),
                     fmt_pct_cell(int(r["n_le_010"]), int(r["n_nontrivial_points"])),
                     fmt_pct_cell(int(r["n_le_025"]), int(r["n_nontrivial_points"])),
                     fmt(r["inwindow_mae_A+B"], 4), fmt(r["inwindow_mae_A-only"], 4)])
    disp.write_table("lrrrrrrr", header, rows, midrules_after=[len(DATASETS) - 1])

    # Table (b): compact view across N.
    header_b = ["Setting"] + [f"$N={n}$" for n in N_SHOW]
    rows_b = [[rf"\multicolumn{{{len(N_SHOW) + 1}}}{{l}}{{\textit{{Median $d_{{\mathrm{{BoN}}}}$ (nats)}}}}"]]
    rows_b += [[SHORT[key]] + [fmt(s.loc[key, f"median_d_N{n}"], 3) for n in N_SHOW] for key, _, _ in DATASETS]
    rows_b += [[rf"\multicolumn{{{len(N_SHOW) + 1}}}{{l}}{{\textit{{Median relative ESS}}}}"]]
    rows_b += [[SHORT[key]] + [fmt(s.loc[key, f"median_rel_ess_N{n}"], 3) for n in N_SHOW] for key, _, _ in DATASETS]
    disp.write_table("l" + "r" * len(N_SHOW), header_b, rows_b, suffix="_byN", midrules_after=[len(DATASETS)])

    se_h = s.loc["SummEval-human_heldout"]
    wmt = s.loc["WMT-MQM-human_heldout"]
    allr = s.loc["All"]
    sat_human = info[(info["dataset_label"] == "SummEval-human_heldout") & (info["sup_kl"] <= BRIDGE)]
    sat_wmt = info[(info["dataset_label"] == "WMT-MQM-human_heldout") & (info["sup_kl"] <= BRIDGE)]
    med_lines = "\n".join(
        f"| {LABEL[key]} | {int(s.loc[key, 'n_paths'])} | {s.loc[key, 'median_d_N2']:.4f} | "
        f"{s.loc[key, 'median_rel_ess_N2']:.4f} | {s.loc[key, 'median_d_N4']:.3f} | {s.loc[key, 'median_d_N8']:.3f} | "
        f"{s.loc[key, 'median_d_N64']:.3f} | {int(s.loc[key, 'n_le_010'])}/{int(s.loc[key, 'n_nontrivial_points'])} | "
        f"{int(s.loc[key, 'n_le_025'])}/{int(s.loc[key, 'n_nontrivial_points'])} |"
        for key, _, _ in DATASETS)
    note = f"""
# bon_nladder: exact pooled Best-of-N ladder by dataset

This artifact summarizes exact pooled Best-of-N behavior by dataset. "SummEval human",
"WMT MQM" and "OpenMEVA-MAGS" are held-out-human-proxy paths; "SummEval ROUGE-1"
and "SummEval GPT-4o" use the named model proxy. The generated tables provide a
per-dataset summary and a compact view across N.

## Definitions
- **Path:** one proxy/gold pairing on a fixed finite item set of size m. Reference
  p0 = uniform over the m items. Held-out-human paths use one rater as proxy and the
  mean of the remaining raters of the same group as gold; ROUGE-1 / GPT-4o paths use the
  model proxy and the mean of all SummEval ratings for that dimension as gold.
- **Exact BoN (with replacement, uniform tie-breaking; Appendix A.7):**
  p_i = w_N(r_i)/m, w_N(u) = (F(u)^N - F_-(u)^N)/(F(u) - F_-(u)), F and F_- the right/left
  proxy CDFs under p0. Pooled over the path's whole item set (not prompt-conditional).
- **d_BoN** = KL(p || p0) = sum_i p_i log(m p_i), in nats.
- **ESS** = 1/sum_i p_i^2; **RelESS** = (sum_i p_i^2/p0_i)^(-1) = ESS/m (uniform p0).
- **Actual BoN gain** = sum_i p_i j_i - mean(j), in native human-score units.
- **Predictions at d = d_BoN:** A+B = A sqrt(d) + B d; A-only = A sqrt(d); A_nu = A_nu sqrt(d);
  empirical fit = a sqrt(d) + b d least-squares fitted to the exact exponential-tilt curve on
  d <= 0.10 of the same pool; zero = 0.
- **Nontrivial point:** a (path, N) pair with N >= 2. The ladder has 11 nontrivial N values
  per path: N in {{2,3,4,6,8,12,16,24,32,48,64}} (N = 1 is the reference, d = 0).
- **In-window:** d_BoN <= 0.10 (the checked local window). **Bridge:** d_BoN <= 0.25.

## Aggregation, units, support
- Medians are over paths (each path weight 1) at fixed N. The "All settings" row pools the
  191 paths, so its medians are path-weighted and dominated by WMT (141/191).
- Point counts are over (path, N) pairs; shares divide by paths x 11.
- MAE = mean absolute error over in-window nontrivial points (each point weight 1). `--`: the
  setting has no nontrivial point with d <= 0.10, so there is nothing to score in-window.
- d in nats; RelESS dimensionless; MAE in native human units (SummEval and OpenMEVA 1-5
  Likert, WMT MQM points). Do not compare MAE across settings.
- Support: SummEval 32 held-out-human paths (4 dimensions x {{3 expert, 5 turker}} raters),
  ROUGE-1 4, GPT-4o 4 (one per dimension), m = 1,600; WMT MQM 141 paths (fixed_141 lineage,
  checked with `require_wmt_fixed`), m = {int(wmt['m_min'])}-{int(wmt['m_max'])}; OpenMEVA-MAGS 10 paths, m = 1,000.
  Total 191 paths x 12 N = 2,292 points.

## Protocol
- Pooled, full-pool: A, B, A_nu and the empirical fit are computed on the same full item pool
  on which BoN is evaluated (exp20; no calibration split). These comparisons are descriptive
  same-pool audits, not held-out predictions.
- Outside d <= 0.10 the predictions are extrapolations; they are reported only in the data CSV.

## Verification performed by bon.py
- Path counts per dataset and the 12-value N ladder; `d_bon == d_bon_recheck` everywhere.
- All 2,292 (path, N) points re-derived from the prepared score files with an independent
  implementation of w_N: max |d diff| = {_RECHECK['d']:.1e}, max |ESS diff|/m = {_RECHECK['ess_rel']:.1e},
  max |gain diff| = {_RECHECK['gain']:.1e}, max |A diff| = {_RECHECK['a']:.1e}, max |B diff| = {_RECHECK['b']:.1e}.
- ROUGE-1 (almost untied proxy) medians match the atomless closed forms
  d_N = log N - (N-1)/N and RelESS_N = (2N-1)/N^2 at N = 2, 4, 8 within 5e-4.
- Audited numbers asserted with `check()`: N=2 medians .1718/.1931/.1889/.1860/.1656;
  exactly 4 nontrivial in-window points; in-window MAE A+B .01968, A-only .08836.

## Key results
| Setting | Paths | median d (N=2) | median RelESS (N=2) | median d (N=4) | median d (N=8) | median d (N=64) | points d<=.10 | points d<=.25 |
|---|---|---|---|---|---|---|---|---|
{med_lines}

- Nontrivial in-window points: {int(allr['n_le_010'])} of {int(allr['n_nontrivial_points'])} ({100 * allr['share_le_010']:.2f}%),
  all SummEval held-out-human at N=2 (consistency expert held-out raters 0/1/2 and fluency
  expert held-out rater 2). {191 - int(allr['n_paths_any_le_010'])} of 191 paths have no nontrivial in-window point.
- Smallest N=2 radius outside SummEval: WMT {wmt['min_d_N2']:.4f} (just above 0.10).
  SummEval held-out-human N=2 radii range {se_h['min_d_N2']:.4f}-{se_h['max_d_N2']:.4f}.
- In-window MAE (n = 4 points): A+B {allr['inwindow_mae_A+B']:.5f}, A-only {allr['inwindow_mae_A-only']:.5f},
  A_nu {allr['inwindow_mae_A_nu']:.5f}, zero {allr['inwindow_mae_zero']:.5f}, empirical fit {allr['inwindow_mae_empirical']:.5f}.
- d <= 0.25 counts include saturated paths: {len(sat_human)} SummEval expert paths and {len(sat_wmt)} WMT
  path{'s' if len(sat_wmt) != 1 else ''} have reachable KL log(m/n_top) <= 0.25, so BoN stays below 0.25 for every N. In SummEval,
  {int(se_h['n_le_025_from_N_ge3'])} of the {int(se_h['n_le_025'])} d <= 0.25 points are such N >= 3 points; in WMT,
  {int(wmt['n_le_025_from_N_ge3'])} of {int(wmt['n_le_025'])}.
- Descriptive N=2 MAE over all paths (mostly outside the window; data CSV `N2_mae_*`), e.g.
  SummEval held-out human A+B {se_h['N2_mae_A+B']:.4f} vs A-only {se_h['N2_mae_A-only']:.4f}.


## Sources
- `{rel(BON_POINTS)}` (exp20, `{rel(EXP20_BUILDER)}`), `{rel(BON_SPEC)}`
- Prepared scores for re-verification and tie counts: `{rel(SUMMEVAL_LONG)}`, `{rel(WMT_LONG)}`,
  `{rel(OPENMEVA_LONG)}`, `{rel(ROUGE1_PROXY)}`, `{rel(GPT4O_PROXY)}`; path builders `{rel(LIB_UTILS)}`

"""
    disp.write_note(note)
    return dict(summary=s, info=info, pts=pts)


# ---------------------------------------------------------------------------
# Display 2: the four in-window cases
# ---------------------------------------------------------------------------
def build_inwindow_cases(group: Group, ctx: dict) -> None:
    disp = group.display("bon_inwindow_cases", "appendix",
                         "The four nontrivial in-window exact-BoN points (SummEval held-out human, N=2): "
                         "radius, ESS, top-score ties, gain and predictor errors")
    paths = get_paths(disp)
    pts = get_points(disp, paths)
    info = ctx["info"]
    nt = pts[(pts["N"] > 1) & (pts["d_bon"] <= WINDOW)]
    if sorted(nt["path_id"]) != INWINDOW_IDS or set(nt["N"]) != {2} or set(nt["dataset_label"]) != {"SummEval-human_heldout"}:
        raise AssertionError(f"unexpected in-window set: {nt[['dataset_label', 'path_id', 'N']].values.tolist()}")
    cases = nt.merge(info, on=["dataset_label", "path_id", "m"]).set_index("path_id").loc[INWINDOW_IDS]
    later = pts[pts["path_id"].isin(INWINDOW_IDS) & (pts["dataset_label"] == "SummEval-human_heldout")]
    cases["d_bon_N3"] = later[later["N"] == 3].set_index("path_id")["d_bon"]
    cases["d_bon_N64"] = later[later["N"] == 64].set_index("path_id")["d_bon"]
    for name, column in GATE_PREDS.items():
        cases[f"signed_err_{name}"] = cases[column] - cases["delta_bon"]
        if not (cases[f"signed_err_{name}"] > 0).all():
            raise AssertionError(f"{name} does not overestimate every in-window case")
    for pid in INWINDOW_IDS:
        proxy = np.asarray(paths["SummEval-human_heldout"][pid].proxy_scores, dtype=float)
        if not (np.isin(proxy, [1, 2, 3, 4, 5]).all() and proxy.max() == 5):
            raise AssertionError(f"{pid}: proxy is not a single-rater 1-5 score with top value 5")
    if not ((cases["share_top"] >= 0.85) & (cases["share_top"] <= 0.89 + 0.005)).all():
        raise AssertionError("top-score share outside 85-89%")
    # Atom check: with the top atom of mass s, N=2 BoN puts (1 - (1 - s)^2) on the tied top block.
    top_mass = 1 - (1 - cases["share_top"]) ** 2
    cases["bon_mass_on_top_block"] = top_mass
    means = {"A+B": 0.01968, "A-only": 0.08836, "A_nu": 0.12381, "zero": 0.24896, "empirical": 0.00432}
    for name, value in means.items():
        check(cases[GATE_ERRORS[name]].mean(), value, 5e-6, f"in-window cases MAE {name}")
    check(cases["d_bon"].mean(), 0.0887, 5e-5, "in-window cases mean d")
    keep = ["dataset_label", "group_key", "N", "m", "d_bon", "ess_bon", "rel_ess", "top_proxy_score", "n_top",
            "share_top", "sup_kl", "bon_mass_on_top_block", "d_bon_N3", "d_bon_N64", "delta_bon",
            "a_exact", "b_theory", "a_nu", "a_fit_shared", "b_fit_shared"]
    keep += list(GATE_PREDS.values()) + list(GATE_ERRORS.values()) + [f"signed_err_{k}" for k in GATE_PREDS]
    data = cases[keep].reset_index()
    mean_row = data.select_dtypes("number").mean().to_dict()
    mean_row.update(path_id="mean_of_4", dataset_label="SummEval-human_heldout", group_key="")
    data = pd.concat([data, pd.DataFrame([mean_row])], ignore_index=True)
    disp.write_data(data)

    cols = INWINDOW_IDS
    mean = data.set_index("path_id").loc["mean_of_4"]
    c = data.set_index("path_id")

    def line(label, column, digits, scale=1.0, integer=False, show_mean=True):
        vals = [c.loc[p, column] * scale for p in cols]
        cells = [fmt_int(v) if integer else fmt(v, digits) for v in vals]
        mean_cell = (fmt_int(mean[column] * scale) if integer else fmt(mean[column] * scale, digits)) if show_mean else "--"
        return [label] + cells + [mean_cell]

    rows = [
        line(r"$d_{\mathrm{BoN}}$ (nats)", "d_bon", 3),
        line("ESS", "ess_bon", 0, integer=True),
        line("RelESS", "rel_ess", 3),
        line(r"Items at top proxy score (\%)", "share_top", 1, scale=100),
        line(r"KL cap $\log(m/n_{\mathrm{top}})$", "sup_kl", 3),
        line("Actual BoN gain", "delta_bon", 4),
        line("A+B prediction", GATE_PREDS["A+B"], 4),
        line(r"$|$error$|$, A+B", GATE_ERRORS["A+B"], 4),
        line(r"$|$error$|$, A-only", GATE_ERRORS["A-only"], 4),
        line(r"$|$error$|$, $A_\nu$", GATE_ERRORS["A_nu"], 4),
        line(r"$|$error$|$, empirical fit", GATE_ERRORS["empirical"], 4),
        line(r"$|$error$|$, zero", GATE_ERRORS["zero"], 4),
    ]
    header = [
        ["", r"\multicolumn{3}{c}{Consistency, expert}", "Fluency, expert", ""],
        [r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}Quantity", "rater 0", "rater 1", "rater 2", "rater 2", "Mean"],
    ]
    disp.write_table("lrrrrr", header, rows, midrules_after=[4, 6])

    per_case = "\n".join(
        f"| {p} | {c.loc[p, 'd_bon']:.4f} | {c.loc[p, 'ess_bon']:.1f} | {c.loc[p, 'rel_ess']:.4f} | "
        f"{100 * c.loc[p, 'share_top']:.1f}% | {c.loc[p, 'sup_kl']:.4f} | {c.loc[p, 'delta_bon']:.4f} | "
        f"{c.loc[p, GATE_PREDS['A+B']]:.4f} | {c.loc[p, GATE_ERRORS['A+B']]:.4f} | {c.loc[p, GATE_ERRORS['A-only']]:.4f} | "
        f"{c.loc[p, GATE_ERRORS['A_nu']]:.4f} | {c.loc[p, GATE_ERRORS['empirical']]:.4f} |" for p in cols)
    note = f"""
# bon_inwindow_cases: the four nontrivial exact-BoN points inside d <= 0.10

**Intended display:** appendix, next to `bon_nladder`. The main text should state the exception in one
sentence (e.g. "four SummEval held-out-human paths are inside the window at N = 2; there A+B
overestimates the BoN gain by 0.020 on average versus 0.088 for A-only").

## What the four cases are
All are SummEval held-out-human paths at N = 2 with m = 1,600 items (100 articles x 16
systems): consistency with expert held-out rater 0, 1 and 2, and fluency with expert held-out
rater 2. The proxy is the held-out expert's integer 1-5 score; gold is the mean of the other
two experts (1-5 scale). These are the only nontrivial (N >= 2) points with d_BoN <= 0.10 among
2,101 nontrivial points in 191 paths.

## Why they are in the window
{100 * c['share_top'].iloc[:4].min():.1f}-{100 * c['share_top'].iloc[:4].max():.1f}% of the 1,600 items carry the top proxy score 5 (n_top = {', '.join(str(int(c.loc[p, 'n_top'])) for p in cols)}).
BoN and the pooled exponential tilt weight items only through their proxy score, so tied items
get equal weight; as N (or t) grows both converge to the uniform distribution on the top block,
and their KL is capped at log(m/n_top) = {', '.join(f'{c.loc[p, "sup_kl"]:.3f}' for p in cols)}. N = 2 already puts
1-(1-s)^2 = {', '.join(f'{100 * c.loc[p, "bon_mass_on_top_block"]:.1f}%' for p in cols)} of its mass on that block, which yields
d = {', '.join(f'{c.loc[p, "d_bon"]:.3f}' for p in cols)}, well below the atomless value log 2 - 1/2 = 0.193.
At N = 3 the radii are {', '.join(f'{c.loc[p, "d_bon_N3"]:.3f}' for p in cols)} and at N = 64 they reach the cap
({', '.join(f'{c.loc[p, "d_bon_N64"]:.3f}' for p in cols)}).

## Definitions
- d_BoN = KL(p||p0) in nats with exact with-replacement BoN and uniform tie-breaking (A.7).
- ESS = 1/sum p_i^2 (items); RelESS = ESS/m.
- Items at top proxy score = n_top/m, n_top = number of items whose proxy equals the path maximum.
- Actual BoN gain = sum_i p_i j_i - mean(j), SummEval 1-5 units.
- A+B = A sqrt(d) + B d; A-only = A sqrt(d); A_nu = A_nu sqrt(d); empirical fit = descriptive
  least-squares a sqrt(d) + b d on the exact tilt curve over d <= 0.10 of the same pool;
  zero = 0 (its error equals the gain). |error| = |prediction - actual gain|.
- Mean column: unweighted mean of the four cases (= MAE for the error rows).

## Protocol and caveats
- exp20 coefficients are **full-pool**: A, B, A_nu and the empirical fit are computed on the same
  1,600 items on which BoN is evaluated, with no calibration split. This is a same-pool,
  descriptive comparison, not a held-out prediction.
- BoN is **pooled** over the whole item set, not prompt-conditional (contrast with
  `bon_prompt_conditional_scope`).
- **Every predictor overestimates** all four gains (signed errors > 0 for A+B, A-only, A_nu and the
  empirical fit; asserted in bon.py). The empirical fit is closest (MAE {mean[GATE_ERRORS['empirical']]:.5f}).
- n = 4 points from 2 dimensions of one dataset; these are scope exceptions, not evidence that
  BoN generally lands in the window.

## Per-case values
| Path | d | ESS | RelESS | top share | log(m/n_top) | gain | A+B pred | err A+B | err A-only | err A_nu | err emp |
|---|---|---|---|---|---|---|---|---|---|---|---|
{per_case}

## Key results (asserted)
- Mean d = {mean['d_bon']:.4f}; RelESS {c['rel_ess'].iloc[:4].min():.3f}-{c['rel_ess'].iloc[:4].max():.3f}; ESS {c['ess_bon'].iloc[:4].min():.0f}-{c['ess_bon'].iloc[:4].max():.0f}.
- MAE: A+B {mean[GATE_ERRORS['A+B']]:.5f}, A-only {mean[GATE_ERRORS['A-only']]:.5f}, A_nu {mean[GATE_ERRORS['A_nu']]:.5f},
  zero {mean[GATE_ERRORS['zero']]:.5f}, empirical fit {mean[GATE_ERRORS['empirical']]:.5f}.
- A+B has lower error than A-only in 4/4 cases.

## Units and decimals
d, RelESS and the KL cap: 3 decimals; ESS: integer items; share: % with 1 decimal; gains,
predictions and errors: 4 decimals (SummEval 1-5 units).

## Sources
- `{rel(BON_POINTS)}` (d, ESS, gain, coefficients, predictions, errors).
- `{rel(SUMMEVAL_LONG)}` via `{rel(LIB_UTILS)}` `generate_human_paths` (top-score ties, m);
  the same rebuild re-verifies d, ESS, gain, A and B to <= 1e-10.
"""
    disp.write_note(note)


# ---------------------------------------------------------------------------
# Display 3: softened rules (exp23)
# ---------------------------------------------------------------------------
def build_softened_rules(group: Group) -> None:
    disp = group.display("bon_softened_rules", "appendix",
                         "Mixture-with-BoN and rank-tilt rules inside 0.005<d<=0.10 (exp23, 30/70 splits): "
                         "path-macro MAE of A+B and baselines")
    rows_all = disp.read_csv(MIX_ROWS)
    saved = disp.read_csv(MIX_MACRO)
    disp.input(MIX_SPEC)
    disp.input(EXP23_BUILDER)
    require_wmt_fixed(rows_all, label="mixture_ranktilt_rows", unit_col="n_eval", path_col="path_id")
    check(len(rows_all), 123420, 0, "exp23 row count")
    iw = rows_all[(rows_all["d"] > 0.005) & (rows_all["d"] <= WINDOW)].copy()

    per_path = {}
    out = []
    for (dataset, mech), g in iw.groupby(["dataset", "mechanism"]):
        rec = dict(dataset=dataset, mechanism=mech, n_paths=g["path_id"].nunique(), n_rows=len(g),
                   n_splits_per_path=int(g.groupby("path_id")["repeat"].nunique().min()))
        per_split = g.groupby(["path_id", "repeat"]).size()
        all_splits = rows_all[(rows_all["dataset"] == dataset) & (rows_all["mechanism"] == mech)].groupby(
            ["path_id", "repeat"]).size()
        if len(per_split) != len(all_splits):
            raise AssertionError(f"{dataset}/{mech}: some splits have no in-window point")
        rec.update(points_per_split_min=int(per_split.min()), points_per_split_max=int(per_split.max()),
                   points_per_split_mean=float(per_split.mean()), d_min=float(g["d"].min()),
                   d_median=float(g["d"].median()), d_max=float(g["d"].max()),
                   inwindow_param_values=";".join(f"{v:g}" for v in sorted(g["param_value"].unique())),
                   n_eval_min=int(g["n_eval"].min()), n_eval_max=int(g["n_eval"].max()))
        for name, column in MIX_ERRORS.items():
            pm = g[np.isfinite(g[column])].groupby("path_id")[column].mean()
            per_path[(dataset, mech, name)] = pm
            rec[f"mae_{name}"] = float(pm.mean())
            rec[f"n_nonfinite_{name}"] = int((~np.isfinite(g[column])).sum())
        rec["paths_AB_lt_Aonly"] = int((per_path[(dataset, mech, "A+B")] < per_path[(dataset, mech, "A-only")]).sum())
        rec["paths_AB_lt_empirical"] = int((per_path[(dataset, mech, "A+B")] < per_path[(dataset, mech, "empirical_fit")]).sum())
        rec["mean_realized_gain"] = float(g["delta_realized"].mean())
        out.append(rec)
    res = pd.DataFrame(out)
    # Compare all 40 cells with the saved path-macro table.
    merged = saved.merge(res.melt(id_vars=["dataset", "mechanism"],
                                  value_vars=[f"mae_{k}" for k in MIX_ERRORS], var_name="baseline", value_name="rebuilt")
                         .assign(baseline=lambda f: f["baseline"].str.replace("mae_", "", regex=False)),
                         on=["dataset", "mechanism", "baseline"], how="outer", indicator=True)
    check((merged["_merge"] != "both").sum(), 0, 0, "macro table cell alignment")
    check(len(merged), 40, 0, "macro table cells")
    max_cell_diff = float((merged["path_macro_mae"] - merged["rebuilt"]).abs().max())
    check(max_cell_diff, 0.0, 1e-12, "rebuilt vs saved path-macro MAE")
    paths_ok = merged.merge(res[["dataset", "mechanism", "n_paths"]], on=["dataset", "mechanism"], suffixes=("_saved", ""))
    check((paths_ok["n_paths_saved"] != paths_ok["n_paths"]).sum(), 0, 0, "macro table path counts")
    r = res.set_index(["dataset", "mechanism"])
    check(r.loc[("SummEval-GPT4o", "rank_tilt"), "mae_A+B"], 0.0064, 5e-5, "GPT-4o rank tilt A+B")
    check(r.loc[("SummEval-GPT4o", "rank_tilt"), "mae_A-only"], 0.0065, 5e-5, "GPT-4o rank tilt A-only")
    check(res["points_per_split_min"].min(), 2, 0, "min in-window grid points per split")
    check(res["points_per_split_max"].max(), 4, 0, "max in-window grid points per split")
    check(r.loc[("WMT-MQM-human_heldout", "alpha_mixture"), "n_paths"], 141, 0, "exp23 WMT paths")
    check((res["mae_A+B"] - res["mae_empirical_fit"]).abs().max(), 0.0, 1e-3, "A+B within 0.001 of empirical fit")
    if not ((res["mae_A+B"] < res["mae_zero"]) & (res["mae_A+B"] < res["mae_A_nu"])).all():
        raise AssertionError("A+B not below zero and A_nu in every cell")
    disp.write_data(res)

    header = [
        ["", "", r"\multicolumn{5}{c}{Path-macro MAE, $0.005<d\le0.10$}", "Paths:"],
        [r"\cmidrule(lr){3-7}Dataset", "Rule", "A+B", "A-only", r"$A_\nu$", "Emp.\\ fit", "Zero", "A+B wins"],
    ]
    rows, breaks = [], []
    for i, dataset in enumerate(MIX_DATASETS):
        for k, mech in enumerate(["alpha_mixture", "rank_tilt"]):
            rec = r.loc[(dataset, mech)]
            rows.append([SHORT[dataset] if k == 0 else "", MECH_LABEL[mech],
                         fmt(rec["mae_A+B"], 4), fmt(rec["mae_A-only"], 4), fmt(rec["mae_A_nu"], 4),
                         fmt(rec["mae_empirical_fit"], 4), fmt(rec["mae_zero"], 4),
                         f"{int(rec['paths_AB_lt_Aonly'])}/{int(rec['n_paths'])}"])
        if i < len(MIX_DATASETS) - 1:
            breaks.append(len(rows) - 1)
    disp.write_table("llrrrrrr", header, rows, midrules_after=breaks)

    lines = "\n".join(
        f"| {SHORT[d]} | {m} | {int(r.loc[(d, m), 'n_paths'])} | {r.loc[(d, m), 'mae_A+B']:.4f} | "
        f"{r.loc[(d, m), 'mae_A-only']:.4f} | {r.loc[(d, m), 'mae_A_nu']:.4f} | {r.loc[(d, m), 'mae_empirical_fit']:.4f} | "
        f"{r.loc[(d, m), 'mae_zero']:.4f} | {int(r.loc[(d, m), 'paths_AB_lt_Aonly'])} | "
        f"{int(r.loc[(d, m), 'points_per_split_min'])}-{int(r.loc[(d, m), 'points_per_split_max'])} | "
        f"{r.loc[(d, m), 'inwindow_param_values']} | {r.loc[(d, m), 'd_min']:.4f}-{r.loc[(d, m), 'd_max']:.4f} ({r.loc[(d, m), 'd_median']:.4f}) |"
        for d in MIX_DATASETS for m in ["alpha_mixture", "rank_tilt"])
    note = f"""
# bon_softened_rules: smooth BoN-direction rules inside the checked window (exp23)

**Intended display:** appendix (supports the short main-text scope paragraph on "smooth mechanisms").

## Rules
- **Mixture:** p_alpha = (1 - alpha) p0 + alpha p_BoN,N=8, with p_BoN,8 the exact tie-corrected
  BoN distribution on the evaluation items; alpha in {{0, .002, .005, .01, .02, .05, .1, .2, .4, .7, 1}}.
- **Rank tilt:** q_lambda(i) proportional to p0(i) F(r_i)^lambda, F = mid-rank empirical CDF in (0, 1]
  (ties get the average rank); lambda in {{0, .05, .1, .2, .4, .8, 1.5, 3, 6, 12, 24}}. In the atomless
  limit lambda = N - 1 reproduces BoN-N (see `bon_kl_ess_definitions`).
- d = KL(q || p0) computed exactly on the evaluation items (no closed form used).

## Protocol
- exp23 pooled finite-item-set stress test, **not prompt-conditional**.
- 30 random calibration/evaluation splits per path, **30% calibration / 70% evaluation**
  (seed 0). A, B, A_nu and the empirical fit are estimated on the calibration 30%; gains, d and
  errors are evaluated on the held-out 70% (n_eval: SummEval 1,120; OpenMEVA 700; WMT 301-1,974).
- Empirical fit: least-squares a sqrt(d) + b d on the calibration exact tilt curve over
  d <= min(0.10, 0.95 log(m_cal / n_top,cal)).
- WMT: fixed_141 lineage (141 paths; `require_wmt_fixed` passed).

## Aggregation
Nontrivial in-window points 0.005 < d <= 0.10. For each (dataset, rule, path) the absolute
errors are averaged over all in-window (split, grid value) rows; the table reports the unweighted
mean over paths ("path-macro"; each path weight 1, regardless of how many grid values land in the
window). "A+B < A-only (paths)" counts paths whose path-level A+B error is below A-only.
Units: native human-score units (SummEval/OpenMEVA 1-5, WMT MQM points); do not compare across datasets.

## Verification
Rebuilt from `mixture_ranktilt_rows.csv` and compared with the saved
`inwindow_baseline_path_macro.csv`: all 40 cells (4 datasets x 2 rules x 5 baselines) match,
max |diff| = {max_cell_diff:.1e}; path counts match.

Table columns: "Mixture" is the mixture with BoN N = 8. "Paths: A+B wins" = k/n paths whose
path-level A+B error is below A-only (n = number of paths). The in-window grid values per split
(2-4) are given below rather than as a column. Fits 5.5 in at \\small.

## Support caveat
Only {int(res['points_per_split_min'].min())}-{int(res['points_per_split_max'].max())} grid values per path and split land in 0.005 < d <= 0.10 (coarse
grids). Median in-window d per cell is {res['d_median'].min():.3f}-{res['d_median'].max():.3f}, so the
comparison mostly probes the lower half of the window; the grid values and d ranges are listed below.

| Dataset | Rule | Paths | A+B | A-only | A_nu | Emp | Zero | A+B<A-only paths | pts/split | in-window grid values | d range (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{lines}

## Key results
- A+B is below zero and A_nu in every cell (closest: GPT-4o mixture, {r.loc[('SummEval-GPT4o', 'alpha_mixture'), 'mae_A+B']:.4f} vs zero
  {r.loc[('SummEval-GPT4o', 'alpha_mixture'), 'mae_zero']:.4f}) and within 0.001 of the empirical fit in every cell (asserted).
- A+B improves on A-only for SummEval human (mixture {r.loc[('SummEval-human_heldout', 'alpha_mixture'), 'mae_A+B']:.4f} vs {r.loc[('SummEval-human_heldout', 'alpha_mixture'), 'mae_A-only']:.4f};
  rank tilt {r.loc[('SummEval-human_heldout', 'rank_tilt'), 'mae_A+B']:.4f} vs {r.loc[('SummEval-human_heldout', 'rank_tilt'), 'mae_A-only']:.4f}) and WMT
  ({r.loc[('WMT-MQM-human_heldout', 'alpha_mixture'), 'mae_A+B']:.4f} vs {r.loc[('WMT-MQM-human_heldout', 'alpha_mixture'), 'mae_A-only']:.4f}; {r.loc[('WMT-MQM-human_heldout', 'rank_tilt'), 'mae_A+B']:.4f} vs {r.loc[('WMT-MQM-human_heldout', 'rank_tilt'), 'mae_A-only']:.4f}).
- GPT-4o mixture favours A+B ({r.loc[('SummEval-GPT4o', 'alpha_mixture'), 'mae_A+B']:.4f} vs {r.loc[('SummEval-GPT4o', 'alpha_mixture'), 'mae_A-only']:.4f}); GPT-4o rank tilt is a
  **near tie** ({r.loc[('SummEval-GPT4o', 'rank_tilt'), 'mae_A+B']:.4f} vs {r.loc[('SummEval-GPT4o', 'rank_tilt'), 'mae_A-only']:.4f}; 4 paths).
- OpenMEVA-MAGS is **worse with B** (mixture {r.loc[('OpenMEVA-MAGS-human_heldout', 'alpha_mixture'), 'mae_A+B']:.4f} vs {r.loc[('OpenMEVA-MAGS-human_heldout', 'alpha_mixture'), 'mae_A-only']:.4f};
  rank tilt {r.loc[('OpenMEVA-MAGS-human_heldout', 'rank_tilt'), 'mae_A+B']:.4f} vs {r.loc[('OpenMEVA-MAGS-human_heldout', 'rank_tilt'), 'mae_A-only']:.4f}).
- Wording: "supports an informative local approximation for smooth rules inside the window, not
  universal dominance of the curvature term".

## Reporting details
"Pooled finite-item-set rules (not prompt-conditional); coefficients from 30% calibration splits,
errors on the 70% evaluation items, 30 splits; path-macro MAE over 0.005 < d <= 0.10 (each path
weighted equally); only 2-4 grid values per split fall in that range; native units, not comparable
across datasets."

## Sources
- `{rel(MIX_ROWS)}` (rebuild), `{rel(MIX_MACRO)}` (comparison), `{rel(MIX_SPEC)}`
- Generator: `{rel(EXP23_BUILDER)}`
"""
    disp.write_note(note)


# ---------------------------------------------------------------------------
# Display 4: prompt-conditional scope (corrected reachability)
# ---------------------------------------------------------------------------
def load_pc_module():
    spec = importlib.util.spec_from_file_location("prompt_conditional_bon_audit", PC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_conditional_bon_audit"] = module
    spec.loader.exec_module(module)
    return module


def build_prompt_conditional(group: Group) -> None:
    disp = group.display("bon_prompt_conditional_scope", "appendix",
                         "Prompt-conditional exact BoN radius and pooled raw-score tilt reachability")
    pc = load_pc_module()
    disp.input(PC_SCRIPT)
    disp.input(LIB_UTILS)
    disp.input(PC_FIGURE_GENERATOR)
    configs = {
        "summeval": dict(name="SummEval", source=SUMMEVAL_LONG, items="article_id,model_id", groups="dimension,rater_type"),
        "openmeva_mags": dict(name="OpenMEVA-MAGS", source=OPENMEVA_LONG, items="story_id,model_id", groups="dataset"),
    }
    corrected_frames, summary_frames, saved_summaries, path_stats = [], [], [], []
    regression = {}
    interp_gap = []
    for key, cfg in configs.items():
        run_spec = disp.input(PC_DIR / key / "prompt_conditional_bon_run_spec.json")
        spec = json.loads(run_spec.read_text())
        if list(spec["n_list"]) != PC_N or float(spec["tilt_max_d"]) != BRIDGE or int(spec["min_items"]) != 200:
            raise AssertionError(f"{key}: unexpected run spec")
        saved = disp.read_csv(PC_DIR / key / "prompt_conditional_bon_paths.csv")
        saved_summary = disp.read_csv(PC_DIR / key / "prompt_conditional_bon_summary.csv")
        long = pd.read_csv(disp.input(cfg["source"]))
        paths = quiet(lambda: list(pc.generate_human_paths(
            df=long, item_cols=pc.parse_cols(cfg["items"]), rater_col="rater_idx", score_col="score",
            group_cols=pc.parse_cols(cfg["groups"]), min_items=200, min_raters=3)))
        check(len(paths), int(spec["n_paths"]), 0, f"{key} path count")
        regen, path_rows = [], []
        for p in paths:
            prompt_keys = pc.prompt_keys_from_item_ids(p.item_ids, "||", 0)
            for row in pc.evaluate_path(p, prompt_keys, PC_N, BRIDGE):
                row["dataset"] = cfg["name"]
                regen.append(row)
            r = np.asarray(p.proxy_scores, dtype=float)
            j = np.asarray(p.gold_scores, dtype=float)
            n_top = int(np.sum(r == r.max()))
            try:
                lib().exact_tilt_curve(r, j, d_max=BRIDGE)
                curve_ok, err = True, ""
            except RuntimeError as exc:
                curve_ok, err = False, f"{type(exc).__name__}: {exc}"
            prompts = sorted(pd.unique(prompt_keys))
            sizes = {int(np.sum(prompt_keys == q)) for q in prompts}
            if sizes != {16 if key == "summeval" else 5}:
                raise AssertionError(f"{key}: unexpected prompt sizes {sizes}")
            tied, saturated = [], []
            for prompt in prompts:
                idx = np.flatnonzero(prompt_keys == prompt)
                if len(idx) < 2:
                    continue
                rp = r[idx]
                tied.append(np.unique(rp).size == 1)
                saturated.append(math.log(len(idx) / np.sum(rp == rp.max())) <= WINDOW)
            path_rows.append(dict(dataset=cfg["name"], path_id=p.path_id, m=r.size, n_top=n_top,
                                  pooled_sup_kl=math.log(r.size / n_top), tilt_curve_d025_ok=curve_ok,
                                  tilt_curve_error=err, n_prompts_with_2plus=len(tied),
                                  share_prompts_fully_tied=float(np.mean(tied)),
                                  share_prompts_sup_kl_le_010=float(np.mean(saturated)),
                                  r=r, j=j))
        regen = pd.DataFrame(regen)
        # The engine now fixes pooled reachability and adds its KL supremum.
        # Compare unchanged fields to the historical snapshot; validate the
        # three corrected fields independently below.
        corrected_columns = {"global_tilt_reachable", "global_tilt_at_mean_kl", "abs_err_global_tilt"}
        if not set(saved.columns).issubset(regen.columns) or len(regen) != len(saved):
            raise AssertionError(f"{key}: regenerated frame shape/columns differ")
        max_diff = 0.0
        for column in saved.columns:
            if column in corrected_columns:
                continue
            a, b = regen[column], saved[column]
            if a.dtype.kind == "f":
                same_nan = a.isna() == b.isna()
                if not same_nan.all():
                    raise AssertionError(f"{key}/{column}: NaN pattern differs")
                max_diff = max(max_diff, float(np.nanmax(np.abs(a - b))) if a.notna().any() else 0.0)
            elif not (a.astype(str) == b.astype(str)).all():
                raise AssertionError(f"{key}/{column}: values differ")
        check(max_diff, 0.0, 1e-12, f"{key} regenerated vs saved rows")
        regression[cfg["name"]] = max_diff

        pinfo = pd.DataFrame(path_rows)
        bad = pinfo[~pinfo["tilt_curve_d025_ok"]]
        if not (bad["pooled_sup_kl"] < BRIDGE).all() or not (pinfo[pinfo["tilt_curve_d025_ok"]]["pooled_sup_kl"] >= BRIDGE).all():
            raise AssertionError(f"{key}: curve failures are not exactly the paths with sup KL < 0.25")
        if not bad["tilt_curve_error"].str.startswith("RuntimeError: Could not bracket").all():
            raise AssertionError(f"{key}: unexpected curve failure type")
        corrected = saved.merge(pinfo.drop(columns=["r", "j", "tilt_curve_error"]), on=["dataset", "path_id"], how="left")
        corrected["global_tilt_reachable_original"] = corrected["global_tilt_reachable"]
        corrected["global_tilt_at_mean_kl_original"] = corrected["global_tilt_at_mean_kl"]
        corrected["row_corrected"] = False
        by_id = pinfo.set_index("path_id")
        for idx, row in corrected.iterrows():
            if row["tilt_curve_d025_ok"]:
                # Unaffected: keep saved values; record interpolation accuracy of the original curve.
                if row["global_tilt_reachable"]:
                    exact, _ = tilt_at_kl(by_id.loc[row["path_id"], "r"], by_id.loc[row["path_id"], "j"], row["prompt_mean_kl"])
                    interp_gap.append(abs(exact - row["global_tilt_at_mean_kl"]))
                if bool(row["global_tilt_reachable"]) != bool(row["prompt_mean_kl"] <= BRIDGE):
                    raise AssertionError("unaffected reachability is not d_mean <= 0.25")
                continue
            r, j = by_id.loc[row["path_id"], "r"], by_id.loc[row["path_id"], "j"]
            target = float(row["prompt_mean_kl"])
            reachable = target < row["pooled_sup_kl"]
            corrected.at[idx, "row_corrected"] = True
            corrected.at[idx, "global_tilt_reachable"] = bool(reachable)
            if reachable:
                # Bracketed exact curve: d_max = target < sup KL, so find_t_for_dmax succeeds.
                curve = lib().exact_tilt_curve(r, j, d_max=target)
                tilt = float(lib().delta_at_d(curve, target))
                exact, _ = tilt_at_kl(r, j, target)
                check(tilt, exact, 1e-9, f"{row['path_id']} N={row['n_best']} tilt vs independent bisection")
                corrected.at[idx, "global_tilt_at_mean_kl"] = tilt
                corrected.at[idx, "abs_err_global_tilt"] = abs(tilt - row["conditional_bon_gain"])
            else:
                corrected.at[idx, "global_tilt_at_mean_kl"] = float("nan")
                corrected.at[idx, "abs_err_global_tilt"] = float("nan")
        # Every column except the three patched ones equals the saved file, row by row.
        untouched = [c for c in saved.columns if c not in ("global_tilt_reachable", "global_tilt_at_mean_kl", "abs_err_global_tilt")]
        if not corrected[untouched].equals(saved[untouched]):
            raise AssertionError(f"{key}: untouched columns changed")
        unaffected = ~corrected["row_corrected"]
        if not corrected.loc[unaffected, saved.columns].equals(saved.loc[unaffected, saved.columns]):
            raise AssertionError(f"{key}: unaffected rows changed")
        corrected["global_tilt_reachable"] = corrected["global_tilt_reachable"].astype(bool)
        if not corrected["global_tilt_reachable"].equals(regen["global_tilt_reachable"]):
            raise AssertionError(f"{key}: corrected reachability differs from current engine")
        finite_new = regen["global_tilt_reachable"].to_numpy(bool)
        if not np.isfinite(regen.loc[finite_new, "global_tilt_at_mean_kl"]).all():
            raise AssertionError(f"{key}: current engine has nonfinite reachable gains")
        reach_rule = (corrected["prompt_mean_kl"] < corrected["pooled_sup_kl"]) & (corrected["prompt_mean_kl"] <= BRIDGE)
        if not (reach_rule == corrected["global_tilt_reachable"]).all():
            raise AssertionError(f"{key}: corrected reachability != (d_mean < sup KL and d_mean <= 0.25)")
        summ = pc.summarize(corrected[saved.columns])
        summ.insert(0, "dataset", cfg["name"])
        # Summary regression: only the reachability-dependent columns may change.
        for column in saved_summary.columns:
            if column in ("share_paths_global_tilt_reachable", "mae_global_tilt_at_mean_kl"):
                continue
            diff = np.nanmax(np.abs(summ[column].to_numpy(float) - saved_summary[column].to_numpy(float)))
            check(diff, 0.0, 1e-12, f"{key} summary {column}")
        summ["share_paths_global_tilt_reachable_original"] = saved_summary["share_paths_global_tilt_reachable"].to_numpy()
        summ["mae_global_tilt_at_mean_kl_original"] = saved_summary["mae_global_tilt_at_mean_kl"].to_numpy()
        summ["n_paths_reachable"] = corrected.groupby("n_best")["global_tilt_reachable"].sum().to_numpy()
        summ["n_paths_reachable_original"] = corrected.groupby("n_best")["global_tilt_reachable_original"].sum().to_numpy()
        summ["mean_share_prompts_fully_tied"] = float(pinfo["share_prompts_fully_tied"].mean())
        summ["mean_share_prompts_sup_kl_le_010"] = float(pinfo["share_prompts_sup_kl_le_010"].mean())
        summ["n_paths_sup_kl_lt_025"] = int((pinfo["pooled_sup_kl"] < BRIDGE).sum())
        corrected_frames.append(corrected)
        summary_frames.append(summ)
        saved_summaries.append(saved_summary.assign(dataset=cfg["name"]))
        path_stats.append(pinfo.drop(columns=["r", "j"]))

    corrected = pd.concat(corrected_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    pstats = pd.concat(path_stats, ignore_index=True)
    sm = summary.set_index(["dataset", "n_best"])
    check(sm.loc[("SummEval", 2), "mean_prompt_share_kl_le_0p10"], 0.1247, 5e-5, "SummEval N=2 prompts in window")
    check(sm.loc[("OpenMEVA-MAGS", 2), "mean_prompt_share_kl_le_0p10"], 0.1165, 5e-5, "OpenMEVA N=2 prompts in window")
    check(sm.loc[("SummEval", 2), "share_paths_global_tilt_reachable"], 1.0, 1e-12, "SummEval N=2 reachable (fixed)")
    check(sm.loc[("SummEval", 4), "share_paths_global_tilt_reachable"], 0.1875, 1e-12, "SummEval N=4 reachable (fixed)")
    check(sm.loc[("SummEval", 2), "share_paths_global_tilt_reachable_original"], 0.8125, 1e-12, "SummEval N=2 reachable (saved)")
    check(sm.loc[("SummEval", 4), "share_paths_global_tilt_reachable_original"], 0.0, 0, "SummEval N=4 reachable (saved)")
    for name in ("SummEval", "OpenMEVA-MAGS"):
        for n_best in PC_N:
            if name == "OpenMEVA-MAGS" or n_best >= 8:
                check(sm.loc[(name, n_best), "share_paths_global_tilt_reachable"],
                      sm.loc[(name, n_best), "share_paths_global_tilt_reachable_original"], 0, f"{name} N={n_best} unchanged")
    check(sm.loc[("SummEval", 2), "mean_share_prompts_fully_tied"], 0.020, 1e-12, "SummEval fully tied prompts")
    check(sm.loc[("SummEval", 64), "mean_prompt_share_kl_le_0p10"], 0.0584375, 1e-12, "SummEval prompts <= .10 at N=64")
    check(sm.loc[("SummEval", 2), "mean_share_prompts_sup_kl_le_010"], 0.0584375, 1e-12, "SummEval prompts saturated <= .10")
    check(sm.loc[("OpenMEVA-MAGS", 2), "mean_share_prompts_fully_tied"], 0.030, 1e-12, "OpenMEVA fully tied prompts")
    check(sm.loc[("OpenMEVA-MAGS", 64), "mean_prompt_share_kl_le_0p10"], 0.030, 1e-12, "OpenMEVA prompts <= .10 at N=64")
    disp.write_data(summary)
    disp.write_data(corrected, suffix="_paths")

    header = [["", "", "mean", "median", "prompts", "pooled tilt", "mean"],
              ["", "", "prompt", "prompt", r"KL$\le.10$", "reachable,", "BoN"],
              ["Dataset", "$N$", "KL", "KL", r"(\%)", r"$d\le0.25$ (\%)", "gain"]]
    rows = []
    for name in ("SummEval", "OpenMEVA-MAGS"):
        for n_best in PC_N:
            rec = sm.loc[(name, n_best)]
            rows.append([name, str(n_best), fmt(rec["mean_prompt_kl"], 3), fmt(rec["median_prompt_kl"], 3),
                         fmt(100 * rec["mean_prompt_share_kl_le_0p10"], 1),
                         fmt(100 * rec["share_paths_global_tilt_reachable"], 1),
                         fmt(rec["mean_conditional_bon_gain"], 3)])
    disp.write_table("lrrrrrr", header, rows, midrules_after=[len(PC_N) - 1])

    # Figure: same two panels as make_tail_bon_artifacts.plot_bon, corrected values.
    figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.15), constrained_layout=True)
    for name in ("SummEval", "OpenMEVA-MAGS"):
        g = summary[summary["dataset"] == name].sort_values("n_best")
        style = dict(color=DATASET_COLORS[name], marker=DATASET_MARKERS[name], label=name)
        axes[0].plot(g["n_best"], g["mean_prompt_kl"], **style)
        axes[1].plot(g["n_best"], 100 * g["share_paths_global_tilt_reachable"], **style)
    axes[0].axhline(WINDOW, color="#616161", linestyle="--", linewidth=0.9)
    axes[0].text(64, WINDOW + 0.025, "local window $d=0.10$", fontsize=7, color="#616161", ha="right", va="bottom")
    axes[0].set_ylim(0, 1.32)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(PC_N)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Best-of-$N$")
        ax.grid(True, color="#E0E0E0", linewidth=0.6)
    axes[0].set_ylabel("Mean prompt KL (nats)")
    axes[1].set_ylabel(r"Paths reachable, $d\leq0.25$ (%)")
    axes[1].set_ylim(-3, 103)
    axes[0].set_title("Prompt-conditional BoN radius")
    axes[1].set_title("Pooled raw-score tilt reachability")
    axes[1].legend(loc="upper right")
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
    disp.save_figure(fig)

    ssum = sm.loc["SummEval"]
    osum = sm.loc["OpenMEVA-MAGS"]
    affected = pstats[~pstats["tilt_curve_d025_ok"]]
    sat_lo = 100 * float((affected["n_top"] / affected["m"]).min())
    sat_hi = 100 * float((affected["n_top"] / affected["m"]).max())
    check(len(affected), 6, 0, "paths hit by the d_max=0.25 bracket failure")
    n2_fixed = corrected[(corrected["n_best"] == 2) & corrected["row_corrected"]]
    n4_fixed = corrected[(corrected["n_best"] == 4) & corrected["row_corrected"]]
    table_lines = "\n".join(
        f"| {name} | {n} | {sm.loc[(name, n), 'mean_prompt_kl']:.4f} | {sm.loc[(name, n), 'median_prompt_kl']:.4f} | "
        f"{100 * sm.loc[(name, n), 'mean_prompt_share_kl_le_0p10']:.2f} | "
        f"{100 * sm.loc[(name, n), 'share_paths_global_tilt_reachable_original']:.1f} -> "
        f"{100 * sm.loc[(name, n), 'share_paths_global_tilt_reachable']:.1f} | "
        f"{sm.loc[(name, n), 'mae_global_tilt_at_mean_kl_original']:.4f} -> {sm.loc[(name, n), 'mae_global_tilt_at_mean_kl']:.4f} | "
        f"{sm.loc[(name, n), 'mae_local_law_at_mean_kl']:.4f} | {sm.loc[(name, n), 'mean_conditional_bon_gain']:.4f} |"
        for name in ("SummEval", "OpenMEVA-MAGS") for n in PC_N)
    affected_lines = "\n".join(
        f"| {row.path_id} | {int(row.n_top)} | {row.pooled_sup_kl:.4f} | "
        + " | ".join(f"{v:.4f}" for v in corrected[(corrected['path_id'] == row.path_id)].sort_values('n_best')['prompt_mean_kl'][:2])
        + " |" for row in affected.itertuples())
    note = f"""
# bon_prompt_conditional_scope: prompt-conditional exact BoN

## Publication use
Appendix table and figure for the prompt-conditional BoN scope audit. The table reports
path-macro summaries, and the figure shows mean prompt KL and pooled raw-score tilt
reachability as Best-of-N increases.

## Definitions
- Prompt-conditional exact BoN uses replacement sampling and uniform tie-breaking within
  each prompt. The prompt reference distribution is uniform over its candidates.
- Mean and median prompt KL are first computed within each path and then summarized with
  equal path weight.
- Prompts KL <= .10 is the path-macro share of prompts within the checked local window.
- Pooled tilt reachable is the share of paths whose mean prompt KL is below both 0.25 and
  the pooled raw-score tilt supremum log(m/n_top).
- Mean BoN gain is measured in each dataset's native target-score units.

## Main observations
At N = 2, mean prompt KL is {ssum.loc[2].mean_prompt_kl:.3f} for SummEval and
{osum.loc[2].mean_prompt_kl:.3f} for OpenMEVA-MAGS. Only
{100 * ssum.loc[2].mean_prompt_share_kl_le_0p10:.1f}% and
{100 * osum.loc[2].mean_prompt_share_kl_le_0p10:.1f}% of prompts, respectively,
remain inside d <= 0.10. Pooled raw-score tilt reachability is
{100 * ssum.loc[2].share_paths_global_tilt_reachable:.1f}% and
{100 * osum.loc[2].share_paths_global_tilt_reachable:.1f}% at N = 2. At N = 4,
only {int(ssum.loc[4].n_paths_reachable)} of 32 SummEval paths remain reachable;
no path is reachable for N >= 8.

Fully tied prompts remain at zero KL for every N and explain the small in-window plateau
at larger N. This is a descriptive comparison between prompt-conditional selection and a
pooled raw-score tilt, not a theorem validation for BoN.

## Sources and checks
- Prepared scores: `{rel(SUMMEVAL_LONG)}`, `{rel(OPENMEVA_LONG)}`.
- Experiment implementation: `{rel(PC_SCRIPT)}` and `{rel(LIB_UTILS)}`.
- Current saved results: `results/prompt_bon/{{summeval,openmeva_mags}}/`.
- The builder re-derives reachability from prepared scores and checks all unaffected fields
  against a compact regression fixture in `{rel(PC_DIR)}`.
"""
    disp.write_note(note)


# ---------------------------------------------------------------------------
# Display 5: definitions and closed-form checks (note + check CSV)
# ---------------------------------------------------------------------------
def build_definitions(group: Group, ctx: dict) -> None:
    disp = group.display("bon_kl_ess_definitions", "main (definitions; note only)",
                         "Definitions of induced KL, ESS and relative ESS for selection rules, exact tied BoN, "
                         "closed forms with numerical checks, and an applicability-rule wording")
    paths = get_paths(disp)
    pts = get_points(disp, paths)
    s = ctx["summary"]
    checks = []

    def record(name, computed, reference, tol):
        diff = abs(float(computed) - float(reference))
        check(computed, reference, tol, name)
        checks.append(dict(check=name, computed=float(computed), reference=float(reference), abs_diff=diff, tolerance=tol))

    # 1. Exact tie formula vs brute-force enumeration (exact rationals) on a tiny tied pool.
    r_small = [1, 1, 2, 2, 2, 3, 3]
    m_small = len(r_small)
    for n_best in (1, 2, 3, 4):
        brute = [Fraction(0)] * m_small
        for draw in product(range(m_small), repeat=n_best):
            top = max(r_small[i] for i in draw)
            winners = [i for i in draw if r_small[i] == top]
            for i in winners:  # uniform choice among tied draws (positions)
                brute[i] += Fraction(1, len(winners)) / m_small ** n_best
        vals = sorted(set(r_small))
        closed = []
        for i in range(m_small):
            upper = Fraction(sum(1 for x in r_small if x <= r_small[i]), m_small)
            lower = Fraction(sum(1 for x in r_small if x < r_small[i]), m_small)
            closed.append((upper ** n_best - lower ** n_best) / (upper - lower) / m_small)
        if brute != closed or sum(closed) != 1:
            raise AssertionError(f"tie formula vs enumeration failed at N={n_best} ({vals})")
        record(f"tie formula = exhaustive enumeration, m=7 tied pool, N={n_best} (exact rational)", 0, 0, 0)
    # 2. Closed form vs binomial-sum form on every SummEval held-out-human path, N=2..8.
    worst = 0.0
    for p in paths["SummEval-human_heldout"].values():
        for n_best in (2, 3, 4, 8):
            worst = max(worst, float(np.max(np.abs(bon_probs(p.proxy_scores, n_best) - bon_probs_binomial(p.proxy_scores, n_best)))))
    record("A.7 closed form vs binomial tie sum, 32 SummEval paths, max |p diff|", worst, 0, 1e-14)
    # 3. Independent recomputation of all saved points (done in get_points).
    record("independent exact BoN vs saved d_BoN, 2,292 points, max |diff|", _RECHECK["d"], 0, 1e-10)
    record("independent exact BoN vs saved ESS/m, 2,292 points, max |diff|", _RECHECK["ess_rel"], 0, 1e-10)
    record("independent exact BoN vs saved gain, 2,292 points, max |diff|", _RECHECK["gain"], 0, 1e-10)
    # 4. RelESS = (sum p^2/p0)^-1 = ESS/m under uniform p0.
    worst = 0.0
    for key, by_id in paths.items():
        for p in list(by_id.values())[:5]:
            prob = bon_probs(p.proxy_scores, 4)
            worst = max(worst, abs(rel_ess(prob) - 1.0 / np.sum(prob ** 2) / prob.size))
    record("RelESS = ESS/m under uniform p0 (BoN N=4, 5 paths per setting)", worst, 0, 1e-12)
    # 5. Atomless closed forms on an untied pool (ranks 1..m), m = 200,000.
    m_big = 200_000
    ranks = np.arange(1, m_big + 1, dtype=float)
    for n_best in (2, 4, 8):
        prob = bon_probs(ranks, n_best)
        record(f"untied m=200k: d_N vs log N-(N-1)/N, N={n_best}", kl_uniform(prob), math.log(n_best) - (n_best - 1) / n_best, 1e-4)
        record(f"untied m=200k: RelESS_N vs (2N-1)/N^2, N={n_best}", rel_ess(prob), (2 * n_best - 1) / n_best ** 2, 1e-4)
        # Rank tilt lambda = N-1 with mid-rank F (exp23 definition) vs BoN-N.
        q = ranks ** (n_best - 1)
        q /= q.sum()
        record(f"untied m=200k: rank tilt lambda={n_best - 1} KL vs BoN N={n_best} KL", kl_uniform(q), kl_uniform(prob), 1e-4)
        record(f"untied m=200k: rank tilt lambda={n_best - 1} RelESS vs BoN N={n_best} RelESS", rel_ess(q), rel_ess(prob), 1e-4)
    # 6. ROUGE-1 medians (near-atomless proxy) vs closed forms.
    for n_best in (2, 4, 8):
        record(f"ROUGE-1 median RelESS N={n_best} vs (2N-1)/N^2", s.loc["SummEval-ROUGE1", f"median_rel_ess_N{n_best}"],
               (2 * n_best - 1) / n_best ** 2, 5e-4)
        record(f"ROUGE-1 median d N={n_best} vs log N-(N-1)/N", s.loc["SummEval-ROUGE1", f"median_d_N{n_best}"],
               math.log(n_best) - (n_best - 1) / n_best, 5e-4)
    # 7. Mixture identity RelESS(p_alpha) = 1/(1 + alpha^2 chi2(p_BoN||p0)) on all 191 paths.
    worst = 0.0
    for key, by_id in paths.items():
        for p in by_id.values():
            prob = bon_probs(p.proxy_scores, 8)
            m = prob.size
            p0 = np.full(m, 1.0 / m)
            chi2 = float(np.sum(prob ** 2 / p0) - 1.0)
            for alpha in (0.002, 0.01, 0.05, 0.2, 0.7, 1.0):
                mix = (1 - alpha) * p0 + alpha * prob
                worst = max(worst, abs(rel_ess(mix, p0) - 1.0 / (1.0 + alpha ** 2 * chi2)))
    record("mixture RelESS identity, BoN N=8, 191 paths x 6 alphas, max |diff|", worst, 0, 1e-12)
    # 8. Saturation: BoN N=64 approaches log(m/n_top) on the four in-window paths.
    info = ctx["info"].set_index(["dataset_label", "path_id"])
    for pid in INWINDOW_IDS:
        d64 = pts[(pts["dataset_label"] == "SummEval-human_heldout") & (pts["path_id"] == pid) & (pts["N"] == 64)]["d_bon"].iloc[0]
        record(f"saturation: d_BoN(N=64) -> log(m/n_top), {pid}", d64, info.loc[("SummEval-human_heldout", pid), "sup_kl"], 1e-3)
    # 9. Orientation: RelESS of the exact pooled exponential tilt at d = 0.10 on every path.
    u = lib()
    tilt_rel = []
    for key, by_id in paths.items():
        for pid, p in by_id.items():
            r = np.asarray(p.proxy_scores, dtype=float)
            t = u.find_t_for_dmax(r, WINDOW)
            w = u.tilt_weights(r, t)
            tilt_rel.append(dict(dataset_label=key, path_id=pid, d=kl_uniform(w), rel_ess=rel_ess(w)))
    tilt_rel = pd.DataFrame(tilt_rel)
    record("exact tilt hits d=0.10 on every path, max |d-0.10|", (tilt_rel["d"] - WINDOW).abs().max(), 0, 1e-9)
    tilt_summary = tilt_rel.groupby("dataset_label")["rel_ess"].agg(["median", "min", "max"])
    checks_df = pd.DataFrame(checks)
    tilt_out = tilt_summary.reset_index().rename(columns={"median": "tilt_rel_ess_at_d010_median",
                                                          "min": "tilt_rel_ess_at_d010_min",
                                                          "max": "tilt_rel_ess_at_d010_max"})
    disp.write_data(checks_df)
    disp.write_data(tilt_out, suffix="_tilt_relESS_at_window")

    tilt_lines = "\n".join(f"| {LABEL[k]} | {tilt_summary.loc[k, 'median']:.3f} | {tilt_summary.loc[k, 'min']:.3f}-{tilt_summary.loc[k, 'max']:.3f} |"
                           for k, _, _ in DATASETS)
    check_lines = "\n".join(f"| {row.check} | {row.computed:.6g} | {row.reference:.6g} | {row.abs_diff:.1e} |"
                            for row in checks_df.itertuples())
    note = f"""
# bon_kl_ess_definitions: induced KL, ESS and relative ESS for selection rules

**Intended display:** main text (definitions + applicability rule, ~1 paragraph); proofs/derivations stay
in Appendix A.7. Note only (no table); `bon_kl_ess_definitions.csv` records the numerical checks.

## Definitions (finite reference pool of m items, reference p0, rule-induced distribution p)
- **Induced KL radius:** d = KL(p || p0) = sum_i p_i log(p_i / p0_i) (nats). This is the radius that
  the local law A sqrt(d) + B d is indexed by; for uniform p0, d = sum_i p_i log(m p_i).
- **Effective sample size:** ESS = 1 / sum_i p_i^2 (items).
- **Relative ESS:** RelESS = (sum_i p_i^2 / p0_i)^(-1) = 1 / (1 + chi^2(p || p0)) in (0, 1]; for
  uniform p0, RelESS = ESS/m. RelESS = 1 means no reweighting.
- **Exact Best-of-N with replacement and ties (Appendix A.7):** draw N items i.i.d. from p0, keep
  the highest proxy score, break ties uniformly. With F(u) = p0(r <= u), F_-(u) = p0(r < u),
  p_i = p0_i w_N(r_i), w_N(u) = (F(u)^N - F_-(u)^N) / (F(u) - F_-(u)) (atom case) and
  w_N(u) = N F(u)^(N-1) (atomless). Computable from unlabeled proxy scores alone.

## Closed forms
- Atomless BoN: d_N = log N - (N-1)/N and RelESS_N = (2N-1)/N^2
  (E_p0[w_N^2] = int_0^1 N^2 u^(2N-2) du = N^2/(2N-1)). N = 2, 4, 8: d = 0.193, 0.636, 1.204;
  RelESS = 0.750, 0.438, 0.234. Measured ROUGE-1 medians (near-untied proxy): RelESS
  {s.loc['SummEval-ROUGE1', 'median_rel_ess_N2']:.3f}/{s.loc['SummEval-ROUGE1', 'median_rel_ess_N4']:.3f}/{s.loc['SummEval-ROUGE1', 'median_rel_ess_N8']:.3f}, d {s.loc['SummEval-ROUGE1', 'median_d_N2']:.3f}/{s.loc['SummEval-ROUGE1', 'median_d_N4']:.3f}/{s.loc['SummEval-ROUGE1', 'median_d_N8']:.3f}.
- Rank tilt q_lambda proportional to p0 F(r)^lambda has atomless density (lambda+1) F^lambda = w_(lambda+1):
  **rank tilt with lambda equals BoN with N = lambda + 1** (atomless; with ties the two differ and
  both must be computed exactly).
- Mixture p_alpha = (1-alpha) p0 + alpha p_BoN: chi^2(p_alpha || p0) = alpha^2 chi^2(p_BoN || p0), so
  **RelESS(p_alpha) = 1 / (1 + alpha^2 chi^2(p_BoN || p0))**. KL(p_alpha || p0) has no closed form and
  is computed exactly.
- Ties cap the radius: as N -> infinity BoN (and the pooled exponential tilt as t -> infinity)
  converge to the uniform distribution on the n_top items tied at the maximum, so
  d <= log(m / n_top). Saturated proxies therefore give small radii even at moderate N.

## Numerical checks (all asserted in bon.py)
| Check | computed | reference | abs diff |
|---|---|---|---|
{check_lines}

## Orientation: RelESS of the exact pooled exponential tilt at d = 0.10
| Setting | median | range |
|---|---|---|
{tilt_lines}

(For Gaussian scores an exponential tilt has RelESS = exp(-2d), 0.819 at d = 0.10.) Different rule
families reach different RelESS at the same KL, so KL remains the gate quantity and RelESS is a
companion concentration diagnostic.

## Suggested applicability-rule wording (main text)
"Before using A and B for a planned selection rule, translate the rule into the distribution it
induces on the reference pool. For Best-of-N this is computable from unlabeled proxy scores
(Appendix A.7, including ties). Report the induced radius d = KL(p || p0) and relative ESS
(sum_i p_i^2/p0_i)^(-1), and compare d with the checked window (here d <= 0.10). Use A sqrt(d) + B d
only inside that window; otherwise lower the selection pressure (smaller N, a mixture weight alpha, or a
rank-tilt exponent lambda), validate the rule directly, or abstain. Hard BoN usually leaves the window
already at N = 2 (median d = 0.17-0.19 across our settings), but proxies with many tied top scores can
keep it inside (four SummEval paths at N = 2)."

## Key results
- N=2 median d: SummEval held-out human {s.loc['SummEval-human_heldout', 'median_d_N2']:.4f}, ROUGE-1 {s.loc['SummEval-ROUGE1', 'median_d_N2']:.4f},
  GPT-4o {s.loc['SummEval-GPT4o', 'median_d_N2']:.4f}, WMT {s.loc['WMT-MQM-human_heldout', 'median_d_N2']:.4f}, OpenMEVA {s.loc['OpenMEVA-MAGS-human_heldout', 'median_d_N2']:.4f}
  (from `bon_nladder`).
- 4 of 2,101 nontrivial exact-BoN points (191 paths x 11 N) have d <= 0.10.

## Caveats for the main text
- The KL is computed on the finite reference pool actually used (pooled or per prompt); prompt-
  conditional BoN gives a different, per-prompt radius (`bon_prompt_conditional_scope`).
- "Inside the window" is necessary, not sufficient: equal-KL discrete rules can have different
  gains (plan section 6).

## Sources
`{rel(BON_POINTS)}`; prepared scores `{rel(SUMMEVAL_LONG)}`, `{rel(WMT_LONG)}`, `{rel(OPENMEVA_LONG)}`,
`{rel(ROUGE1_PROXY)}`, `{rel(GPT4O_PROXY)}`; `{rel(LIB_UTILS)}` (path builders, tilt helpers);
Appendix A.7 of `paper/appendix.tex` for the w_N derivation.
"""
    disp.write_note(note)


# ---------------------------------------------------------------------------
def main() -> None:
    group = Group("bon")
    ctx = build_nladder(group)
    build_inwindow_cases(group, ctx)
    build_softened_rules(group)
    build_prompt_conditional(group)
    build_definitions(group, ctx)
    path = group.write_manifest()
    print(f"bon: {len(group.displays)} displays -> {path}")


if __name__ == "__main__":
    main()
