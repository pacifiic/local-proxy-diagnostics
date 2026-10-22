"""Build an auditable worked example and full coefficient table, without selecting a favorable split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DIMENSIONS, digest, new_directory, read_csv, sha256, validate_scores,
                    verify_prepared, write_json)
from analyze import PathData, describe, prospective
from common_experiment_utils import exact_tilt_curve

LABELS = {"skywork": "Skywork", "armorm": "ArmoRM", "deberta": "DeBERTa"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    prepared, analysis = args.run / "prepared", args.run / "analysis"
    meta = verify_prepared(prepared)
    run_spec = json.loads((analysis / "run_spec.json").read_text())
    if run_spec["prepared_manifest_sha256"] != sha256(prepared / "manifest.json"):
        raise ValueError("Analysis/preparation mismatch")
    items = read_csv(prepared / "items.csv").sort_values("item_id").reset_index(drop=True)
    human = read_csv(prepared / "human.csv")
    models = json.loads((prepared / "models.json").read_text())
    splits = [json.loads(line) for line in (prepared / "splits.jsonl").read_text().splitlines()]
    split = next(s for s in splits if s["dimension"] == "coherence" and s["repeat"] == 0)
    ids = {key: i for i, key in enumerate(items.item_id)}
    cal = np.array([ids[k] for k in split["calibration"]])
    eva = np.array([ids[k] for k in split["evaluation"]])
    gold = human[human.dimension == "coherence"].groupby("item_id").score.mean().reindex(items.item_id).to_numpy()
    summary = pd.read_csv(analysis / "summary.csv")
    coeff = pd.read_csv(analysis / "coefficients.csv")
    rows, curves = [], []
    for alias in models:
        score_file = args.run / "scores" / alias / "scores.csv"
        if run_spec["scores"][alias]["scores"] != sha256(score_file):
            raise ValueError(f"Analysis/score mismatch: {alias}")
        proxy = validate_scores(read_csv(score_file), items)
        path = PathData("coherence", "coherence__model_proxy", items.item_id.to_numpy(), proxy, gold, {})
        result = prospective.evaluate_one_split(path, cal, eva, .1, [.025, .05, .075, .1])
        a, b = result["cal_a_exact"], result["cal_b_theory"]
        checks = describe(gold, proxy)
        stored = coeff[(coeff.alias == alias) & (coeff.dimension == "coherence")].iloc[0]
        for name in ["A", "B", "B_tail", "B_skewness"]:
            if not np.isclose(checks[name], stored[name], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Stored coefficient mismatch: {alias}/{name}")
        end = summary[(summary.alias == alias) & (summary.dimension == "coherence") & np.isclose(summary.radius, .1)].iloc[0]
        rows.append(dict(alias=alias, target="coherence", repeat=0, n_cal=len(cal), n_eval=len(eva),
                         A_cal=a, B_cal=b, first_order=a * np.sqrt(.1), second_order=b * .1,
                         predicted_gain=result["pred_theory_delta_0p1"],
                         heldout_gain=result["actual_eval_delta_0p1"],
                         mae_ab_50=end.mae_theory, mae_aonly_50=end.mae_aonly,
                         mae_empfit_50=end.mae_empfit, n_valid_repeats=int(end.n_common_splits)))
        curve = exact_tilt_curve(proxy[eva], gold[eva], d_max=.1, n_grid=201)
        curves.extend(dict(alias=alias, d=d, exact_gain=q - curve.q[0],
                           ab=a * np.sqrt(max(d, 0)) + b * d,
                           aonly=a * np.sqrt(max(d, 0))) for d, q in zip(curve.d, curve.q))
    out = new_directory(args.outdir)
    worked = pd.DataFrame(rows)
    curves = pd.DataFrame(curves)
    worked.to_csv(out / "worked_example.csv", index=False)
    curves.to_csv(out / "worked_example_curves.csv", index=False)
    tex = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Reward model & $\widehat A$ & $\widehat B$ & $\widehat A\sqrt{.10}$ & $\widehat B(.10)$ & Predicted gain & Held-out gain \\", r"\midrule"]
    for r in worked.itertuples():
        tex.append(f"{LABELS[r.alias]} & {r.A_cal:.4f} & {r.B_cal:.4f} & {r.first_order:.4f} & "
                   f"{r.second_order:.4f} & {r.predicted_gain:.4f} & {r.heldout_gain:.4f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_reward_model_worked_example.tex").write_text("\n".join(tex) + "\n")
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Reward model & Human dimension & $A$ [95\% CI] & $B$ [95\% CI] & Tail term & Skewness term \\", r"\midrule"]
    for r in coeff.itertuples():
        tex.append(f"{LABELS[r.alias]} & {r.dimension} & {r.A:.4f} [{r.A_lo:.4f}, {r.A_hi:.4f}] & "
                   f"{r.B:.4f} [{r.B_lo:.4f}, {r.B_hi:.4f}] & {r.B_tail:.4f} & {r.B_skewness:.4f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_reward_model_coefficients.tex").write_text("\n".join(tex) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), sharex=True, sharey=True)
    for ax, alias in zip(axes, models):
        df = curves[curves.alias == alias]
        r = worked[worked.alias == alias].iloc[0]
        ax.plot(df.d, df.exact_gain, color="#222222", label="Held-out exact tilt", linewidth=2)
        ax.plot(df.d, df.ab, color="#007f8b", label=r"$A+B$ prediction", linewidth=1.8)
        ax.plot(df.d, df.aonly, color="#a74c5e", label=r"$A$-only", linewidth=1.6, linestyle="--")
        ax.set_title(f"{LABELS[alias]}\n" + rf"$\widehat A={r.A_cal:.3f},\ \widehat B={r.B_cal:.3f}$", fontsize=10)
        ax.set_xlabel(r"KL radius $d$")
        ax.set_xlim(0, .1)
        ax.set_xticks([0, .05, .1])
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Gain in human coherence")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(out / "fig_reward_model_worked_example.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / "fig_reward_model_worked_example.png", dpi=180)
    plt.close(fig)
    article = items.drop_duplicates("article_id")
    context = dict(n_articles=len(article), mean_supplied_chars=float(article.article_chars_original.mean()),
                   max_supplied_chars=int(article.article_chars_original.max()),
                   mean_retained_fraction=float((article.article_chars_used / article.article_chars_original).mean()))
    write_json(out / "report_spec.json", dict(
        selection="Preselected coherence, repeat 0, d=0.10; no outcome-based example selection",
        prepared_manifest_sha256=sha256(prepared / "manifest.json"),
        analysis_spec_sha256=sha256(analysis / "run_spec.json"), context=context,
        split_sha256=digest(split), report_code_sha256=sha256(Path(__file__)),
        files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    repeats = int(meta["repeats"])
    valid_counts = sorted(worked.n_valid_repeats.unique())
    repeats_label = str(valid_counts[0]) if len(valid_counts) == 1 else f"{valid_counts[0]}-{valid_counts[-1]}"
    text = ["# Reward-Model Worked Example", "",
            f"Coherence was chosen before observing the model results. Repeat 0 uses {len(cal):,} calibration and {len(eva):,} evaluation items.",
            "The coefficients in this worked example use calibration labels only. Held-out gold is used only for evaluation.",
            "", f"| Model | Calibration A | Calibration B | A sqrt(.10) | B(.10) | Predicted gain | Held-out gain | A+B MAE, {repeats_label} valid repeats | A-only MAE, {repeats_label} valid repeats |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in worked.itertuples():
        text.append(f"| {LABELS[r.alias]} | {r.A_cal:.4f} | {r.B_cal:.4f} | {r.first_order:.4f} | {r.second_order:.4f} | "
                    f"{r.predicted_gain:.4f} | {r.heldout_gain:.4f} | {r.mae_ab_50:.4f} | {r.mae_aonly_50:.4f} |")
    text += ["", "## Reading the Example", "",
             "1. Fix the human target, candidate distribution, and proxy input format.",
             "2. Estimate A and B on the rated calibration sample.",
             "3. At a checked radius, compute A sqrt(d) and B d separately, then add them.",
             "4. Use held-out observations to check the prediction, rather than selecting the example using its error.", "",
             "A measures initial directional alignment. B is the second-order correction to that leading gain.",
             "Negative B reduces the leading prediction at positive d; it is not proof of a global turning point.",
             "The full-pool A/B table is a separate descriptive analysis with article-cluster bootstrap intervals.", "",
             "## Scope and Manuscript Placement", "",
             "A short worked example can illustrate how to use the two coefficients. Place the full 12-row coefficient table,",
             f"all {repeats}-repeat errors and context details in supplementary material. Do not claim a universal RM ranking or",
             "universal A+B superiority. The supplied articles are prefixes (maximum 2,000 characters), and the common-context",
             "rule may shorten them further. This is a fixed-context, item-split SummEval audit, not a new RLHF training experiment.",
             "In particular, the input preprocessing is frozen for this benchmark before split evaluation; this does not test",
             "generalization to longer future article contexts. The gold averages three expert and five crowd ratings.",
             "Source summary ownership and all human ratings are independently checked against the aligned JSONL."]
    (out / "WORKED_EXAMPLE.md").write_text("\n".join(text) + "\n")
    print(out)


if __name__ == "__main__":
    main()
