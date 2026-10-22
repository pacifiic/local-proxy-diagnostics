"""Summarize descriptive same-KL BoN prediction errors and crossing points.

Thresholds are selected retrospectively on these data, not validated deployment
rules. Outside the checked local window, abstain or validate the intended rule
directly rather than automatically switching to an empirical fit.
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
SHARED_FIT_MAX_D = 0.10


BASELINE_ERR_COLS = {
    "zero": "abs_err_zero_vs_bon",
    "anu": "abs_err_anu_vs_bon",
    "aonly": "abs_err_aonly_vs_bon",
    "theory": "abs_err_theory_vs_bon",       # A+B, moment-based
    "empfit": "abs_err_empfit_shared_vs_bon",  # same-path local empirical fit
    "oracle_bon": "abs_err_oracle_bon_vs_bon",  # in-sample cheating upper bound (see build script)
}


def pooled_mae_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    agg = {"n_rows": ("delta_bon", "size"), "median_d": ("d_bon", "median"), "median_ess": ("ess_bon", "median")}
    for name, col in BASELINE_ERR_COLS.items():
        agg[f"mae_{name}"] = (col, "mean")
    g = df.groupby(key).agg(**agg).reset_index()
    g["theory_wins"] = g["mae_theory"] < g["mae_empfit"]
    return g.sort_values(key)


def full_baseline_table(df: pd.DataFrame, key) -> pd.DataFrame:
    """Full A-only vs A+B vs empfit vs anu vs zero vs oracle_bon comparison, pooled or by key."""
    if key is None:
        row = {"group": "__POOLED__", "n_rows": len(df)}
        for name, col in BASELINE_ERR_COLS.items():
            row[f"mae_{name}"] = float(df[col].mean())
        return pd.DataFrame([row])
    rows = []
    for gval, gdf in df.groupby(key):
        row = {"group": gval, "n_rows": len(gdf)}
        for name, col in BASELINE_ERR_COLS.items():
            row[f"mae_{name}"] = float(gdf[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def find_crossover(by_n: pd.DataFrame) -> dict:
    """First N at/after which empfit stays >= as good as theory for all subsequent N
    (a monotonic 'permanent crossover', not just a single noisy flip)."""
    by_n = by_n.sort_values("N").reset_index(drop=True)
    theory_wins = by_n["theory_wins"].to_numpy()
    crossover_idx = None
    for i in range(len(theory_wins)):
        if not theory_wins[i:].any():  # theory never wins again from here on
            crossover_idx = i
            break
    if crossover_idx is None:
        return {"crossover_N": None, "crossover_d": None, "crossover_ess": None}
    row = by_n.iloc[crossover_idx]
    return {
        "crossover_N": int(row["N"]),
        "crossover_d": float(row["median_d"]),
        "crossover_ess": float(row["median_ess"]),
    }


def retrospective_switch_mae(df: pd.DataFrame, d_threshold: float) -> dict:
    use_theory = df["d_bon"] <= d_threshold
    err = np.where(use_theory, df["abs_err_theory_vs_bon"], df["abs_err_empfit_shared_vs_bon"])
    return {
        "switch_d_threshold": d_threshold,
        "mae_retrospective_switch": float(np.mean(err)),
        "mae_always_theory": float(df["abs_err_theory_vs_bon"].mean()),
        "mae_always_empfit": float(df["abs_err_empfit_shared_vs_bon"].mean()),
        "mae_oracle": float(np.minimum(df["abs_err_theory_vs_bon"], df["abs_err_empfit_shared_vs_bon"]).mean()),
        "frac_using_theory": float(use_theory.mean()),
    }


def plot_gate(df: pd.DataFrame, by_n_pooled: pd.DataFrame, crossover: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(by_n_pooled["median_d"], by_n_pooled["mae_theory"], "o-", label="A+B (theory)", color="steelblue")
    ax.plot(by_n_pooled["median_d"], by_n_pooled["mae_empfit"], "o-", label="same-path local empirical fit", color="salmon")
    ax.axvline(SHARED_FIT_MAX_D, color="gray", linestyle="--", label=f"checked window d={SHARED_FIT_MAX_D}")
    if crossover["crossover_d"] is not None:
        ax.axvline(crossover["crossover_d"], color="black", linestyle=":", label=f"empirical crossover d≈{crossover['crossover_d']:.2f}")
    n2_row = by_n_pooled.loc[by_n_pooled["N"] == 2]
    if not n2_row.empty:
        ax.axvline(float(n2_row["median_d"].iloc[0]), color="green", linestyle="-.", alpha=0.6, label="BoN N=2")
    ax.set_xscale("log")
    ax.set_xlabel("KL radius d_bon (log scale)")
    ax.set_ylabel("Pooled MAE vs realized BoN gain")
    ax.set_title("A. When does the empirical fit overtake A+B?")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(by_n_pooled["N"], by_n_pooled["mae_theory"], "o-", label="A+B (theory)", color="steelblue")
    ax.plot(by_n_pooled["N"], by_n_pooled["mae_empfit"], "o-", label="empirical fit", color="salmon")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Best-of-N  (N, log2 scale)")
    ax.set_ylabel("Pooled MAE")
    ax.set_title("B. Same, vs N directly")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


def plot_full_baselines(by_n_baselines: pd.DataFrame, by_dataset_baselines: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = {
        "mae_zero": "zero gain", "mae_anu": "A_nu (reliability-only)", "mae_aonly": "A-only",
        "mae_theory": "A+B", "mae_empfit": "empirical fit", "mae_oracle_bon": "oracle (in-sample, cheating)",
    }
    colors = {
        "mae_zero": "gray", "mae_anu": "purple", "mae_aonly": "orange",
        "mae_theory": "steelblue", "mae_empfit": "salmon", "mae_oracle_bon": "black",
    }

    ax = axes[0]
    by_n = by_n_baselines.sort_values("group")
    for col, lab in labels.items():
        ax.plot(
            by_n["group"], by_n[col],
            marker=None if col == "mae_oracle_bon" else "o",
            linestyle="--" if col == "mae_oracle_bon" else "-",
            label=lab, color=colors[col],
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Best-of-N (N, log2 scale)")
    ax.set_ylabel("Pooled MAE vs realized BoN gain")
    ax.set_title("A. Full baseline comparison vs N")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bd = by_dataset_baselines.set_index("group")
    x = np.arange(len(bd))
    width = 0.13
    for i, (col, lab) in enumerate(labels.items()):
        ax.bar(x + (i - 2.5) * width, bd[col], width, label=lab, color=colors[col])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("-human_heldout", "") for s in bd.index], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Pooled MAE (all N)")
    ax.set_title("B. Full baseline comparison by dataset")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    df = pd.read_csv(out_dir / "bon_gate_points.csv")

    # sanity re-check carried over from build step (cheap, catches a stale/edited CSV too)
    max_mismatch = (df["d_bon"] - df["d_bon_recheck"]).abs().max()
    assert max_mismatch < 1e-9, f"d_bon vs d_bon_recheck mismatch: {max_mismatch}"

    by_n_pooled = pooled_mae_by_key(df, "N")
    by_n_pooled.to_csv(out_dir / "gate_by_N_pooled.csv", index=False)
    print("Pooled (all datasets) MAE by N:")
    print(by_n_pooled.to_string(index=False))

    # Full baseline comparison: zero / A_nu (reliability-only) / A-only / A+B / empfit / oracle_bon.
    pooled_baselines = full_baseline_table(df, key=None)
    by_dataset_baselines = full_baseline_table(df, key="dataset_label")
    by_n_baselines = full_baseline_table(df, key="N")
    pooled_baselines.to_csv(out_dir / "baselines_pooled.csv", index=False)
    by_dataset_baselines.to_csv(out_dir / "baselines_by_dataset.csv", index=False)
    by_n_baselines.to_csv(out_dir / "baselines_by_N.csv", index=False)
    print("\nFull baseline comparison (pooled):")
    print(pooled_baselines.to_string(index=False))
    print("\nFull baseline comparison (by dataset):")
    print(by_dataset_baselines.to_string(index=False))
    print("\nFull baseline comparison (by N):")
    print(by_n_baselines.to_string(index=False))

    crossover = find_crossover(by_n_pooled)
    print(f"\nPooled permanent crossover: N={crossover['crossover_N']}, "
          f"d≈{crossover['crossover_d']}, ESS≈{crossover['crossover_ess']}")

    def first_n_above_window(by_n: pd.DataFrame) -> float:
        above = by_n.loc[by_n["median_d"] > SHARED_FIT_MAX_D, "N"]
        return float(above.min()) if not above.empty else float("nan")

    per_dataset_rows = []
    for label, gdf in df.groupby("dataset_label"):
        by_n = pooled_mae_by_key(gdf, "N")
        co = find_crossover(by_n)
        per_dataset_rows.append({
            "dataset_label": label,
            "first_nontrivial_N_above_window": first_n_above_window(by_n),
            **co,
        })
    per_dataset_df = pd.DataFrame(per_dataset_rows)
    per_dataset_df.to_csv(out_dir / "gate_crossover_by_dataset.csv", index=False)
    print("\nPer-dataset permanent crossover:")
    print(per_dataset_df.to_string(index=False))

    # Retrospective diagnostic only: quantify what a threshold-based switch would
    # have done on these same observations. This is not a held-out deployment rule;
    # outside the checked window the recommendation remains abstention/direct evaluation.
    switch_at_window = retrospective_switch_mae(df, SHARED_FIT_MAX_D)
    switch_at_crossover = (
        retrospective_switch_mae(df, crossover["crossover_d"])
        if crossover["crossover_d"] is not None
        else None
    )
    print(
        f"\nRetrospective switch @ checked window (d<={SHARED_FIT_MAX_D}): "
        f"{switch_at_window}"
    )
    if switch_at_crossover is not None:
        print(
            f"Retrospective switch @ empirical crossover "
            f"(d<={crossover['crossover_d']:.3f}): {switch_at_crossover}"
        )

    summary = {
        "pooled_crossover": crossover,
        "retrospective_switch_at_checked_window": switch_at_window,
        "retrospective_switch_at_empirical_crossover": switch_at_crossover,
        "deployment_note": (
            "This switch is a same-data descriptive audit, not a validated fallback rule. "
            "Outside the checked window, abstain or evaluate directly."
        ),
    }
    import json
    (out_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_gate(df, by_n_pooled, crossover, out_dir / "fig_bon_gate.png")
    plot_full_baselines(by_n_baselines, by_dataset_baselines, out_dir / "fig_bon_gate_baselines.png")


if __name__ == "__main__":
    main()
