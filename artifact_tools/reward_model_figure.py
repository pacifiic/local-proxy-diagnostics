"""Figure B.8: PACE (A+B) vs A-only on SummEval coherence, averaged over 50 matched splits.

Canonical inputs are read-only. The bundled static Inter fonts use the SIL OFL.
The output directory must not already exist. No models are rescored or fitted.

    python artifact_tools/reward_model_figure.py --out build/reward-model-figure
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

MODELS = ["skywork", "armorm", "deberta"]
LABELS = dict(zip(MODELS, ["Skywork", "ArmoRM", "DeBERTa"]))
RADII = [.025, .05, .075, .1]
FONT = "Inter"
FONT_FILES = ["Inter-Regular.ttf", "Inter-Italic.ttf", "Inter-SemiBold.ttf"]
FONT_DIR = Path(__file__).resolve().parent / "fonts" / "inter"
DIMS = ["coherence", "consistency", "fluency", "relevance"]
INK, AXIS = "#1E1E1E", "#4A4A4A"
PACE, AONLY = "#1765A5", "#C15B21"
STYLES = {
    "exact": dict(color=INK, lw=1.2, ls="-"),
    "ab": dict(color=PACE, lw=1.55, ls="-"),
    "aonly": dict(color=AONLY, lw=1.1, ls=(0, (3.4, 1.7))),
}
MARKERS = {"exact": "o", "ab": "s", "aonly": "^"}
# Preblend against white so SVG-to-PDF renderers preserve the pale fill.
SHADE = dict(color="#E5EEF5", lw=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), message + ": nonfinite")
    require(np.allclose(actual, expected, rtol=1e-9, atol=1e-11), message)


def suffix(d):
    return str(d).replace(".", "p")


def load_font(font_dir=FONT_DIR):
    for name in FONT_FILES:
        path = font_dir / name
        require(path.exists(), f"Missing font file: {path}")
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": FONT, "mathtext.fontset": "custom",
        "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:semibold",
        "mathtext.cal": FONT,
        "font.size": 6.8, "axes.titlesize": 7.4, "axes.labelsize": 6.8,
        "xtick.labelsize": 6.1, "ytick.labelsize": 6.1, "legend.fontsize": 6.6,
        "axes.edgecolor": AXIS, "axes.linewidth": .6, "axes.labelcolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": "#333333", "ytick.labelcolor": "#333333",
        "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
    })


def coherence_curves(run, repo, paths):
    """Recompute dense exact and forecast curves for every coherence split, checked against saved rows."""
    sys.path.insert(0, str(repo / "lib"))
    from common_experiment_utils import exact_tilt_curve, theory_coefficients
    items = pd.read_csv(run / "prepared/items.csv.gz").sort_values("item_id").reset_index(drop=True)
    require(len(items) == 1600 and items.item_id.is_unique, "Item identity")
    human = pd.read_csv(run / "prepared/human.csv.gz")
    human = human[human.dimension == "coherence"]
    require(set(human.item_id) == set(items.item_id), "Human item identity")
    require(not human.duplicated(["item_id", "rater_type", "rater_idx"]).any(), "Duplicate ratings")
    require(human.groupby("item_id").size().eq(8).all(), "Rating count")
    counts = human.groupby(["item_id", "rater_type"]).size().unstack(fill_value=0)
    require(set(counts.columns) == {"expert", "crowd"} or set(counts.columns) == {"expert", "turker"}, "Rater types")
    crowd = "crowd" if "crowd" in counts else "turker"
    require(counts.expert.eq(3).all() and counts[crowd].eq(5).all(), "Expert/crowd balance")
    require(np.isfinite(human.score.to_numpy()).all(), "Nonfinite human ratings")
    gold = human.groupby("item_id").score.mean().reindex(items.item_id).to_numpy()
    require(np.isfinite(gold).all(), "Missing gold")
    with gzip.open(run / "prepared/splits.jsonl.gz", "rt") as f:
        splits = [json.loads(line) for line in f if line.strip()]
    splits = [s for s in splits if s["dimension"] == "coherence"]
    require(len(splits) == 50 and {s["repeat"] for s in splits} == set(range(50)), "Split manifest completeness")
    positions = dict(zip(items.item_id, range(len(items))))
    proxies = {}
    for m in MODELS:
        scores = pd.read_csv(run / "scores" / m / "scores.csv")
        require(scores.item_id.is_unique and set(scores.item_id) == set(items.item_id), "Score identity")
        scores = scores.set_index("item_id").loc[items.item_id]
        require((scores.text_sha256.to_numpy() == items.text_sha256.to_numpy()).all(), "Scored text mismatch")
        proxies[m] = scores.proxy_score.to_numpy()
        require(np.isfinite(proxies[m]).all(), "Nonfinite scores")
    grid = np.unique(np.r_[np.linspace(0, .1, 201), RADII])
    dense, diffs = [], []
    for split in sorted(splits, key=lambda s: s["repeat"]):
        ca, ev = split["calibration"], split["evaluation"]
        require(len(ca) == len(set(ca)) == 480 and len(ev) == len(set(ev)) == 1120 and not set(ca) & set(ev), "Split overlap/duplicates")
        require(set(ca) | set(ev) == set(items.item_id), "Split coverage")
        cal, eva = np.array([positions[k] for k in ca]), np.array([positions[k] for k in ev])
        for m in MODELS:
            r = paths[(paths.alias == m) & (paths.dimension == "coherence") & (paths.repeat == split["repeat"])].iloc[0]
            coeff = theory_coefficients(gold[cal], proxies[m][cal])
            close([coeff.a_exact, coeff.b_theory], [r.cal_a_exact, r.cal_b_theory], "Recomputed calibration coefficients")
            curve = exact_tilt_curve(proxies[m][eva], gold[eva], d_max=.1, n_grid=200)
            require(curve.d[-1] >= .1 - 1e-10 and np.all(np.diff(curve.d) >= 0), "Unreachable/nonmonotone radius")
            exact = np.interp(grid, curve.d, curve.q) - curve.q[0]
            for d in RADII:
                v = exact[np.flatnonzero(grid == d)[0]]
                close(v, r[f"actual_eval_delta_{suffix(d)}"], "Recomputed exact gain")
                diffs.append(abs(v - r[f"actual_eval_delta_{suffix(d)}"]))
            dense.extend(dict(alias=m, repeat=split["repeat"], d=float(d), exact=float(q),
                              ab=float(coeff.a_exact * np.sqrt(d) + coeff.b_theory * d),
                              aonly=float(coeff.a_exact * np.sqrt(d))) for d, q in zip(grid, exact))
    curves = pd.DataFrame(dense)
    require(np.isfinite(curves[["d", "exact", "ab", "aonly"]]).all().all(), "Nonfinite dense curves")
    return curves, max(diffs)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7EAED", lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlim(.02, .1035)
    ax.set_xticks(RADII, ["0.025", "0.05", "0.075", "0.10"])


def plot(ax, x, y, key, z):
    ax.plot(x, y, **STYLES[key], zorder=z, solid_capstyle="round", dash_capstyle="round")
    ax.plot(RADII, y.loc[RADII], ls="none", marker=MARKERS[key], ms=3.0 if key == "ab" else 2.8, mew=.8,
            color=STYLES[key]["color"], mfc="white", zorder=z + .5)


def aggregate(curves):
    """Average predictions and within-split errors separately."""
    df = curves.assign(mae_ab=(curves.ab-curves.exact).abs(),
                       mae_aonly=(curves.aonly-curves.exact).abs())
    return df.groupby(["alias", "d"], sort=True)[["exact", "ab", "aonly", "mae_ab", "mae_aonly"]].mean().reset_index()


def figure(curves, summary, out=None):
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.1), sharex=True, sharey="row")
    fig.subplots_adjust(left=.085, right=.99, bottom=.12, top=.85, wspace=.17, hspace=.24)
    for i, alias in enumerate(MODELS):
        df = curves[(curves.alias == alias) & (curves.d >= .025 - 1e-12)]
        mean = df.groupby("d", sort=True)[["exact", "ab", "aonly"]].mean()
        ax = axes[0, i]
        # The shaded band between the two forecasts is the mean B-hat d correction.
        ax.fill_between(mean.index, mean.ab, mean.aonly, **SHADE, zorder=1)
        # A-only is drawn last so it stays visible where it coincides with PACE (Skywork).
        plot(ax, mean.index, mean.exact, "exact", 2)
        plot(ax, mean.index, mean.ab, "ab", 3)
        plot(ax, mean.index, mean.aonly, "aonly", 4)
        ax.set_title(LABELS[alias], color=INK, pad=3.5, weight="semibold")
        style(ax)
        ax.set_ylim(.03, .09)
        ax.yaxis.set_major_locator(MultipleLocator(.02))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax = axes[1, i]
        # Absolute error within each split, then the mean; never the error of the mean curve.
        err = df.assign(ab=(df.ab - df.exact).abs(), aonly=(df.aonly - df.exact).abs())
        err = err.groupby("d", sort=True)[["ab", "aonly"]].mean()
        ax.fill_between(err.index, err.ab, err.aonly, **SHADE, zorder=1)
        plot(ax, err.index, err.ab, "ab", 3)
        plot(ax, err.index, err.aonly, "aonly", 4)
        a, b = err.loc[.1, "aonly"], err.loc[.1, "ab"]
        old = summary[(summary.alias == alias) & (summary.dimension == "coherence") & np.isclose(summary.radius, .1)]
        require(len(old) == 1, "Summary key missing")
        close([a, b], [old.iloc[0].mae_aonly, old.iloc[0].mae_theory], f"Plotted endpoint MAE {alias}")
        change = f"{100 * (b - a) / a:.1f}%".replace("-", "−")
        ax.text(.035, .95, f"MAE vs A-only: {change} at $d = 0.10$", transform=ax.transAxes, va="top",
                color=PACE, fontsize=6.3)
        style(ax)
        ax.set_ylim(.004, .016)
        ax.yaxis.set_major_locator(MultipleLocator(.004))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0, 0].set_ylabel("Mean human gain", labelpad=3)
    axes[1, 0].set_ylabel("Mean absolute error", labelpad=3)
    fig.supxlabel(r"KL radius $d$", y=.012, fontsize=6.8, color=INK)
    handles = [Line2D([], [], **STYLES[k], marker=MARKERS[k], ms=2.9, mew=.8, mfc="white", label=v)
               for k, v in [("exact", "Held-out exact tilt"), ("ab", r"PACE ($A{+}B$) forecast"),
                            ("aonly", r"$A$-only forecast")]]
    handles.append(Patch(**SHADE, label="Between-method gap"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0), ncol=4, frameon=False,
               handlelength=2.4, columnspacing=1.5, handletextpad=.6)
    if out is not None:
        fig.savefig(out / "rm_fig_worked_example.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / "rm_fig_worked_example.png", dpi=300)
        fig.savefig(out / "rm_fig_worked_example.svg", metadata={"Date": None})
    return fig


def validate_tables(paths, summary):
    keys = ["alias", "dimension", "repeat"]
    require(not paths.duplicated(keys).any(), "Duplicate split key")
    expected = {(m, d, r) for m in MODELS for d in DIMS for r in range(50)}
    require(set(paths[keys].itertuples(index=False, name=None)) == expected, "Split key completeness")
    require((paths.n_cal == 480).all() and (paths.n_eval == 1120).all(), "Unexpected split sizes")
    keys = ["alias", "dimension", "radius"]
    require(not summary.duplicated(keys).any(), "Duplicate summary key")
    require(set(summary[keys].itertuples(index=False, name=None)) ==
            {(m, d, r) for m in MODELS for d in DIMS for r in RADII}, "Summary completeness")
    for (m, dim), rows in paths.groupby(["alias", "dimension"]):
        for radius in RADII:
            s = suffix(radius)
            actual = rows[f"actual_eval_delta_{s}"]
            old = summary[(summary.alias == m) & (summary.dimension == dim) & (summary.radius == radius)].iloc[0]
            for method in ["theory", "aonly"]:
                prediction = rows.cal_a_exact*np.sqrt(radius)
                if method == "theory":
                    prediction = prediction + rows.cal_b_theory*radius
                close(prediction, rows[f"pred_{method}_delta_{s}"], "Prediction formula")
                errors = (prediction-actual).abs()
                close(errors, rows[f"abs_err_{method}_{s}"], "Stored absolute errors")
                close(errors.mean(), old[f"mae_{method}"], "Stored summary MAE")


def build_data(repo):
    repo = Path(repo).resolve()
    run = repo / "results/reward_model_coefficients/canonical"
    files = [run / f for f in ["analysis/prospective_paths.csv", "analysis/summary.csv",
             "prepared/items.csv.gz", "prepared/human.csv.gz", "prepared/splits.jsonl.gz"]]
    files += [run / "scores" / m / "scores.csv" for m in MODELS] + [repo / "lib/common_experiment_utils.py"]
    hashes = {str(f.relative_to(repo)): sha(f) for f in files}
    paths = pd.read_csv(run / "analysis/prospective_paths.csv")
    summary = pd.read_csv(run / "analysis/summary.csv")
    validate_tables(paths, summary)
    curves, max_diff = coherence_curves(run, repo, paths)
    require(all(sha(repo / k) == v for k, v in hashes.items()), "Source input modified during build")
    return curves, summary, max_diff, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--font-dir", type=Path, default=FONT_DIR)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    require(not args.out.exists(), "Output exists; choose a fresh directory")
    load_font(args.font_dir)
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "pace-reward-model"})
    repo = args.repo.resolve()
    curves, summary, max_diff, hashes = build_data(repo)
    out = args.out
    out.mkdir(parents=True)
    fig = figure(curves, summary, out)
    plt.close(fig)
    curves.to_csv(out / "coherence_dense_per_split.csv", index=False)
    aggregate(curves).to_csv(out / "rm_fig_worked_example.csv", index=False)
    spec = dict(status="PASS", source_hashes=hashes, script_sha256=sha(Path(__file__)),
                font_hashes={name: sha(args.font_dir / name) for name in FONT_FILES},
                font=FONT, python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                matplotlib=matplotlib.__version__,
                data="SummEval coherence, 3 reward models x 50 matched splits (480 calibration / 1,120 held-out)",
                max_exact_gain_reproduction_difference=max_diff,
                aggregation="Absolute error within each split, then arithmetic mean.",
                axes="Zoomed axes; curves shown from d=0.025 to 0.10; y-axes do not start at zero",
                shading="Between-method gaps, not confidence intervals; top: mean B*d correction; bottom: MAE difference")
    (out / "audit_and_run_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({k: spec[k] for k in ["status", "font", "max_exact_gain_reproduction_difference"]}, indent=2))


if __name__ == "__main__":
    main()
