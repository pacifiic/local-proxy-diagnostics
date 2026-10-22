"""Summarize coefficient, contribution, error, and exploratory decision audits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"


def coefficient_level(df: pd.DataFrame) -> pd.DataFrame:
    units = df.sort_values("d").drop_duplicates(["dataset", "path_id", "repeat"])
    g = units.groupby("dataset").agg(
        n_path_repeat_units=("b_boot_se", "size"),
        n_paths=("path_id", "nunique"),
        a_se_median=("a_boot_se", "median"),
        b_se_median=("b_boot_se", "median"),
        b_ci_includes_zero_rate=("b_ci_includes_zero", "mean"),
        b_sign_stability_median=("b_sign_stability", "median"),
    ).reset_index()
    return g


def contribution_level(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "d"])
        .agg(R_B_median=("R_B", "median"), R_B_p25=("R_B", lambda s: s.quantile(0.25)),
             R_B_p75=("R_B", lambda s: s.quantile(0.75)))
        .reset_index()
    )


def error_attribution_level(df: pd.DataFrame) -> pd.DataFrame:
    d2 = df.copy()
    d2["coefficient_shift_bound"] = d2["E_A"] + d2["E_B"]
    d2["E_A_bound_share"] = (
        d2["E_A"] / d2["coefficient_shift_bound"].replace(0, np.nan)
    )
    return (
        d2.groupby(["dataset", "d"])
        .agg(E_A_median=("E_A", "median"), E_B_median=("E_B", "median"),
             E_A_bound_share_median=("E_A_bound_share", "median"))
        .reset_index()
    )


def decision_level(df: pd.DataFrame) -> pd.DataFrame:
    at_dvalid = df[np.isclose(df["d"], 0.10)]
    g = at_dvalid.groupby("dataset").agg(
        n=("decision_disagree_at_dvalid", "size"),
        decision_disagree_rate=("decision_disagree_at_dvalid", "mean"),
        d_safe_grid_median=("d_safe_grid", "median"),
        d_safe_grid_p10=("d_safe_grid", lambda s: s.quantile(0.10)),
        proceed_ab_rate=("proceed_ab_at_dvalid", "mean"),
        proceed_aonly_rate=("proceed_aonly_at_dvalid", "mean"),
        b_ci_includes_zero_rate=("b_ci_includes_zero", "mean"),
    ).reset_index()
    # Among rows where B's CI includes 0, how often is the overall Delta(d) decision
    # (proceed via A+B) still identical to what A-only alone would have said? This is
    # the "B uncertain but decision unaffected" statistic from the plan.
    stable = (
        at_dvalid[at_dvalid["b_ci_includes_zero"]]
        .groupby("dataset")["decision_disagree_at_dvalid"]
        .apply(lambda s: float((~s.astype(bool)).mean()))
        .rename("decision_stable_despite_b_uncertain_rate")
        .reset_index()
    )
    return g.merge(stable, on="dataset", how="left")


def gamma_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Decision sensitivity to the operational gain threshold gamma=c*sigma_J."""
    at_dvalid = df[np.isclose(df["d"], 0.10)].copy()
    rows = []
    for fraction in (0.05, 0.10, 0.20):
        threshold = fraction * at_dvalid["sigma_j_cal"]
        proceed_ab = at_dvalid["delta_pred_lcb"] > threshold
        # A-only LCB is not retained separately, so this table deliberately reports
        # only the full A+B decision rate.
        for dataset, mask in at_dvalid.groupby("dataset").groups.items():
            idx = list(mask)
            rows.append({
                "dataset": dataset,
                "gamma_fraction_of_sigma_j": fraction,
                "n_path_repeat_units": len(idx),
                "proceed_ab_rate": float(proceed_ab.loc[idx].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    df = pd.read_csv(out_dir / "b_stability_rows.csv")

    coef = coefficient_level(df)
    contrib = contribution_level(df)
    err = error_attribution_level(df)
    decision = decision_level(df)
    gamma = gamma_sensitivity(df)

    coef.to_csv(out_dir / "coefficient_level.csv", index=False)
    contrib.to_csv(out_dir / "contribution_level.csv", index=False)
    err.to_csv(out_dir / "error_attribution_level.csv", index=False)
    decision.to_csv(out_dir / "decision_level.csv", index=False)
    gamma.to_csv(out_dir / "gamma_sensitivity.csv", index=False)

    print("=== Coefficient level ===")
    print(coef.to_string(index=False))
    print("\n=== Contribution level (R_B by d) ===")
    print(contrib.to_string(index=False))
    print("\n=== Coefficient-shift bound components (E_A vs E_B by d) ===")
    print(err.to_string(index=False))
    print("\n=== Decision level (at d=D_valid=0.10) ===")
    print(decision.to_string(index=False))
    print("\n=== Gamma sensitivity (A+B decision at d=D_valid=0.10) ===")
    print(gamma.to_string(index=False))

    # Jackknife / leave-top-k (separate script's output, if present)
    jk_path = out_dir / "jackknife_leave_top_k.csv"
    if jk_path.exists():
        jk = pd.read_csv(jk_path)
        jk_summary = jk.groupby("dataset").agg(
            n_paths=("jk_max_b_influence_rel", "size"),
            jk_max_b_influence_rel_median=("jk_max_b_influence_rel", "median"),
            frac_b_signflip_top1pct=("b_sign_flip_excl_top1pct", "mean"),
            frac_b_signflip_top5pct=("b_sign_flip_excl_top5pct", "mean"),
        ).reset_index()
        jk_summary.to_csv(out_dir / "jackknife_summary.csv", index=False)
        print("\n=== Jackknife / leave-top-k summary ===")
        print(jk_summary.to_string(index=False))

    plot(contrib, err, out_dir / "fig_b_stability.png")


def plot(contrib: pd.DataFrame, err: pd.DataFrame, out_path: Path) -> None:
    datasets = contrib["dataset"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    for dataset in datasets:
        sub = contrib[contrib["dataset"] == dataset].sort_values("d")
        ax.plot(sub["d"], sub["R_B_median"], "o-", label=dataset.replace("-human_heldout", ""))
    ax.set_xlabel("KL radius d")
    ax.set_ylabel("R_B(d) = |G_B| / (|G_A| + |G_B|)")
    ax.set_title("A. B's share of the total predicted gain")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for dataset in datasets:
        sub = err[err["dataset"] == dataset].sort_values("d")
        ax.plot(sub["d"], sub["E_A_bound_share_median"], "o-", label=dataset.replace("-human_heldout", ""))
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("KL radius d")
    ax.set_ylabel("E_A / (E_A + E_B) bound share (median)")
    ax.set_title("B. Coefficient-shift upper-bound components")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
