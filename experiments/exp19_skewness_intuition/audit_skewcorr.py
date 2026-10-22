"""Audit the skewness correction using moments and an independent local tilt fit.

The moment decomposition and numerical fit must agree within the stated
local-approximation tolerance. The audit raises on failed checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
from common_experiment_utils import theory_coefficients, exact_tilt_curve, fit_local_law  # noqa: E402

REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
SYN_DIR = HERE / "outputs" / "synthetic"
OUT_DIR = HERE / "outputs"

CROSS_CHECK_D_MAX = 0.0005  # d-window for the moment-vs-numeric cross-check itself.
# NOTE: this must be much smaller than the d<=0.1 window used elsewhere in the paper.
# fit_local_law regresses on [sqrt(d), d] which becomes ill-conditioned as d_max shrinks,
# and the O(d^{3/2}) remainder contaminates the fit as d_max grows -- both push in opposite
# directions. Empirically (see notes below), for the most extreme synthetic skew setting
# (skewnorm a=-30) the two estimates only agree within 5% once d_max <~ 5e-4; at the paper's
# own d<=0.02-0.1 windows the naive numeric fit is off by up to ~8-9% purely from remainder
# contamination, NOT a bug -- this is itself a mild illustration of Prop D (extreme proxy
# skew shrinks the radius over which the two-coefficient law is a good local approximation).
CROSS_CHECK_TOL_REL = 0.05  # 5% relative tolerance between the two independent computations
CROSS_CHECK_N_GRID = 600


def cross_checked_ab(r: np.ndarray, j: np.ndarray, label: str) -> dict:
    tc = theory_coefficients(j, r)
    curve = exact_tilt_curve(r, j, d_max=CROSS_CHECK_D_MAX, n_grid=CROSS_CHECK_N_GRID)
    fit = fit_local_law(curve.d, curve.q, fit_max_d=CROSS_CHECK_D_MAX)

    a_moment, b_moment = tc.a_exact, tc.b_theory
    a_numeric, b_numeric = fit.a_fit, fit.b_fit

    def rel_err(x, y):
        denom = max(abs(x), abs(y), 1e-8)
        return abs(x - y) / denom

    a_err = rel_err(a_moment, a_numeric)
    b_err = rel_err(b_moment, b_numeric)
    ok = (a_err < CROSS_CHECK_TOL_REL) and (b_err < CROSS_CHECK_TOL_REL)

    print(f"[{label}] A: moment={a_moment:+.5f} numeric={a_numeric:+.5f} rel_err={a_err:.4f}")
    print(f"[{label}] B: moment={b_moment:+.5f} numeric={b_numeric:+.5f} rel_err={b_err:.4f}  "
          f"(fit R^2={fit.r2:.5f})")
    assert ok, (
        f"[{label}] moment-vs-numeric cross-check FAILED "
        f"(A rel_err={a_err:.4f}, B rel_err={b_err:.4f}, tol={CROSS_CHECK_TOL_REL}) "
        f"-- likely a silent bug in data generation or coefficient computation."
    )

    # sigma_J2 + rho must equal Cov(J, r) exactly (algebraic identity used by theory_coefficients).
    cov_jr = float(np.cov(j, r, ddof=0)[0, 1])
    identity_err = abs((tc.sigma_j2 + tc.rho) - cov_jr)
    assert identity_err < 1e-6 * max(1.0, abs(cov_jr)), (
        f"[{label}] sigma_J^2+rho == Cov(J,r) identity broken (err={identity_err}) "
        f"-- theory_coefficients assumption violated."
    )

    z = (r - np.mean(r)) / np.std(r, ddof=0)
    j_bar = j - np.mean(j)
    e_jz = float(np.mean(j_bar * z))
    e_jz2 = float(np.mean(j_bar * z**2))
    e_z3 = float(np.mean(z**3))
    b_term1 = e_jz2
    b_term2 = -(2.0 / 3.0) * e_jz * e_z3
    b_recon = b_term1 + b_term2
    assert abs(b_recon - b_moment) < 1e-6 * max(1.0, abs(b_moment)), (
        f"[{label}] E[Jbar Z^2] - (2/3)E[Jbar Z]E[Z^3] decomposition does not match "
        f"theory_coefficients.b_theory (recon={b_recon}, b_theory={b_moment})."
    )

    return {
        "label": label,
        "n_items": len(r),
        "skew_r": float(stats.skew(r)),
        "e_z3": e_z3,
        "e_jz": e_jz,
        "a_moment": a_moment,
        "b_moment": b_moment,
        "a_numeric": a_numeric,
        "b_numeric": b_numeric,
        "a_rel_err": a_err,
        "b_rel_err": b_err,
        "b_term1_e_jz2": b_term1,
        "b_term2_skew_correction": b_term2,
        "correction_fraction_abs_components": (
            abs(b_term2) / max(abs(b_term1) + abs(b_term2), 1e-8)
        ),
        "fit_r2": fit.r2,
    }


def load_synthetic_settings() -> list[dict]:
    manifest = json.loads((SYN_DIR / "manifest.json").read_text())
    rows = []
    for m in manifest:
        source_path = Path(m["path"])
        if not source_path.is_absolute():
            source_path = SYN_DIR / source_path
        df = pd.read_csv(source_path)
        res = cross_checked_ab(df["r"].to_numpy(), df["J"].to_numpy(), label=f"synthetic:{m['setting']}")
        res["skewnorm_a"] = m["skewnorm_a"]
        res["source"] = "synthetic"
        rows.append(res)
    return rows


def load_real_dataset_points() -> list[dict]:
    """Real (proxy, gold) pairs, for overlay only -- computed independently here
    from the raw long-format CSVs already used elsewhere in this repo, not copied
    from any other experiment's cached table."""
    rows = []

    # SummEval: expert-mean gold vs ROUGE-1 proxy, dimension=coherence.
    long_df = pd.read_csv(DATA_DIR / "summeval_long.csv")
    sub = long_df[(long_df["rater_type"] == "expert") & (long_df["dimension"] == "coherence")].copy()
    sub["item_id"] = sub["article_id"] + "||" + sub["model_id"]
    gold = sub.groupby("item_id")["score"].mean().rename("J")

    rouge = pd.read_csv(DATA_DIR / "proxy_rouge1.csv")
    rouge = rouge[rouge["dimension"] == "coherence"].set_index("item_id")["proxy_score"].rename("r")
    merged = pd.concat([gold, rouge], axis=1).dropna()
    if len(merged) >= 50:
        res = cross_checked_ab(merged["r"].to_numpy(), merged["J"].to_numpy(),
                                label="real:SummEval-coherence-ROUGE1")
        res["skewnorm_a"] = float("nan")
        res["source"] = "real"
        rows.append(res)

    gpt4o = pd.read_csv(DATA_DIR / "model_proxy_4o_aggregated.csv")
    gpt4o = gpt4o[gpt4o["dimension"] == "coherence"].set_index("item_id")["proxy_score_mean"].rename("r")
    merged2 = pd.concat([gold, gpt4o], axis=1).dropna()
    if len(merged2) >= 50:
        res = cross_checked_ab(merged2["r"].to_numpy(), merged2["J"].to_numpy(),
                                label="real:SummEval-coherence-GPT4o")
        res["skewnorm_a"] = float("nan")
        res["source"] = "real"
        rows.append(res)

    return rows


def plot(df: pd.DataFrame, out_path: Path) -> None:
    syn = df[df["source"] == "synthetic"].sort_values("skew_r")
    real = df[df["source"] == "real"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(
        syn["skew_r"],
        syn["correction_fraction_abs_components"] * 100.0,
        "o-",
        color="steelblue",
            label="synthetic (fixed conditional mean, varying proxy skew)")
    for _, row in real.iterrows():
        ax.axvline(row["skew_r"], color="tab:red", linestyle="--", alpha=0.6)
        ax.annotate(row["label"].replace("real:", ""), (row["skew_r"], 5),
                    rotation=90, fontsize=7, color="tab:red", va="bottom")
    ax.set_xlabel("Proxy score skewness  skew(r)")
    ax.set_ylabel("|skew correction| / (|term1| + |term2|)  (%)")
    ax.set_title("When does the -2/3 E[J̄Z]E[Z³] term matter?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(syn["skew_r"], syn["b_term1_e_jz2"], "o-", label="term1: E[J̄Z²]", color="steelblue")
    ax.plot(syn["skew_r"], syn["b_term2_skew_correction"], "o-", label="term2: skew correction", color="salmon")
    ax.plot(syn["skew_r"], syn["b_moment"], "k--", label="B (net)")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Proxy score skewness  skew(r)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("B decomposition vs proxy skewness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


def main() -> None:
    global SYN_DIR, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, default=SYN_DIR)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    SYN_DIR = args.synthetic_dir
    OUT_DIR = args.outdir
    rows = load_synthetic_settings() + load_real_dataset_points()
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "skewcorr_summary.csv"
    df.to_csv(table_path, index=False)
    print(f"-> {table_path}")

    plot(df, OUT_DIR / "fig_skewcorr.png")

    print("\nAll cross-checks passed (moment formula and exact numeric tilting agree "
          f"within {CROSS_CHECK_TOL_REL:.0%}).")


if __name__ == "__main__":
    main()
