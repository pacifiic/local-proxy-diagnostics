#!/usr/bin/env python3
"""LoRA-DPO bridge, three-reward-model (exp37) and open-proxy displays.

Built from saved results only; nothing is retrained, rescored or re-analysed.
Exp37 was analysed with NumPy 2.3.5, so its saved CSVs are read, never recomputed.

All displays register in one manifest, `out/manifest_dpo.json`, because
`build_all.py` merges only that file for this group. The display names keep
their own prefixes (`dpo_`, `rm_`, `proxy_`) through `CombinedGroup`, a thin
subclass of `common.Group` that accepts all three prefixes.

    cd artifact_tools && python dpo.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ["SOURCE_DATE_EPOCH"] = "0"  # fixed PDF CreationDate, so figure hashes are stable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COLORS, RR, TEXT_WIDTH_IN, Display, Group, check, figure_style,  # noqa: E402
                    fmt, fmt_ci, rel, sha256)
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- sources
DPO = RR / "results" / "actual_dpo"
CANON = DPO / "three_seed_canonical"
AUDIT = CANON / "prompt_radius_audit"
POOL = DPO / "target_pool_aware_ab_direct"
EXP34 = RR / "results" / "dpo_flexible_calibration_only"
RM = RR / "results" / "reward_model_coefficients" / "canonical"
PROXY = RR / "results" / "open_proxy"
PROXY_INPUTS = RR / "data" / "open_proxy_inputs"
STRICT = "strict_local_d_le_0p10"

RM_LABEL = {"skywork": "Skywork", "armorm": "ArmoRM", "deberta": "DeBERTa"}
RM_ORDER = ["skywork", "armorm", "deberta"]
DIMS = ["coherence", "consistency", "fluency", "relevance"]
PRED_LABEL = {"A+B": r"$A{+}B$", "A-only": r"$A$-only", "Zero": "Zero", "A_nu": r"$A_\nu$"}
TABLE3_MODELS = ["A+B", "A-only", "Zero", "A_nu"]
MODE_LABEL = {"raw": "Raw", "centered": "Prompt-centered", "zscore": r"Prompt-$z$", "rank": "Prompt-rank"}
FAMILY_LABEL = {"isotonic": "isotonic", "binned": "binned", "spline": "spline", "mlp": "MLP"}

DPO_ESTIMAND = (
    "Actual LoRA-DPO training, but the reported gain is a self-normalized DPO/reference "
    "likelihood-ratio reweighting of FIXED candidate pools (32 per prompt) relative to uniform, "
    "scored by a frozen silver target model. It is not fresh generation, not human-rated gain and "
    "not an exact full-policy expectation. Checkpoints are selected by checkpoint mean prompt "
    "KL <= 0.10; the exp28/exp34 run specifications state the earlier rule \"checkpoint median "
    "d <= 0.10\", which selects the same ten checkpoints.")


# ---------------------------------------------------------------- registry and helpers
class CombinedGroup(Group):
    """The `dpo` registry, also accepting `rm_` and `proxy_` display names.

    Everything is written to `manifest_dpo.json`, the only manifest of this
    group that `build_all.py` merges. common.py is left unchanged.
    """

    def __init__(self, prefix: str, extra_prefixes: tuple[str, ...]):
        super().__init__(prefix)
        self.prefixes = (prefix, *extra_prefixes)

    def display(self, name: str, placement: str, description: str) -> Display:
        if not name.startswith(tuple(p + "_" for p in self.prefixes)):
            raise ValueError(f"{name} must start with one of {self.prefixes}")
        if name in self.displays:
            raise ValueError(f"duplicate display {name}")
        record = dict(placement=placement, description=description, inputs={}, outputs={})
        self.displays[name] = record
        return Display(name, record)


def pm(mean: float, sd: float, digits: int = 5) -> str:
    return rf"{fmt(mean, digits)} $\pm$ {fmt(sd, digits)}"


def pmt(mean: float, sd: float, digits: int = 5) -> str:
    """Plain-text mean +/- SD for Markdown notes."""
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def section(text: str, ncols: int) -> list[str]:
    return [rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{text}}}}}"]


def check_all(values, expected, tol: float, label: str) -> None:
    values, expected = np.asarray(values, float), np.asarray(expected, float)
    if values.shape != expected.shape:
        raise AssertionError(f"{label}: shape {values.shape} != {expected.shape}")
    worst = np.max(np.abs(values - expected)) if values.size else 0.0
    if not worst <= tol:
        raise AssertionError(f"{label}: max deviation {worst} > {tol}")


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def tex_body_cells(path: Path) -> list[list[str]]:
    """Cells of a booktabs tabular body (rows between the first midrule and bottomrule)."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    body = lines[lines.index(r"\midrule") + 1: lines.index(r"\bottomrule")]
    rows = []
    for line in body:
        if line == r"\midrule":
            continue
        line = line[:-2].strip() if line.endswith("\\\\") else line
        rows.append([cell.strip() for cell in line.split("&")])
    return rows


def plain_minus(rows) -> list[list[str]]:
    return [[cell.replace("$-$", "-") for cell in row] for row in rows]


def number_list(values, digits: int) -> str:
    return "/".join(f"{v:.{digits}f}" for v in values)


# ================================================================ DPO
def dpo_checkpoints(disp: Display):
    """Selected checkpoints and their checkpoint-level quantities, with gate checks."""
    cp = disp.read_csv(CANON / "checkpoint_predictions.csv")
    spec = json.loads(disp.input(AUDIT / "run_spec.json").read_text())
    cells = disp.read_csv(AUDIT / "prompt_radius_cells.csv")
    coef = json.loads(disp.input(DPO / "coefficients.json").read_text())

    # Gate: checkpoint mean prompt KL <= 0.10 (the reported local-window rule) ...
    require(spec["selection"]["checkpoint_mean_prompt_kl_max"] == 0.1, "mean-KL gate threshold")
    sel = pd.DataFrame(spec["selected_seed_checkpoints"])
    require(len(sel) == 10 and (sel.checkpoint_mean_prompt_kl <= 0.10).all(), "10 checkpoints, mean KL <= .10")
    require(sel.groupby("seed").size().to_dict() == {0: 4, 1: 3, 2: 3}, "checkpoints per seed 4/3/3")
    keys = set(zip(sel.seed, sel.tag))
    # ... and the earlier median rule written in the run specs selects the same ten.
    allck = cp[["seed", "tag", "d_med"]].drop_duplicates()
    require(len(allck) == 18, "18 seed-checkpoints in the canonical sweep")
    require(set(zip(allck.seed[allck.d_med <= 0.10], allck.tag[allck.d_med <= 0.10])) == keys,
            "median-d gate selects the same checkpoints as the mean-KL gate")
    excluded_median = allck[~allck.d_med.le(0.10)].d_med

    # Prompt-level radii reproduce the gate quantities.
    radii = cells.groupby(["seed", "tag"]).d_reweight.agg(["mean", "median", "size"]).reset_index()
    merged = sel.merge(radii, on=["seed", "tag"], validate="1:1")
    require(len(merged) == 10 and (merged["size"] == 200).all(), "200 evaluation prompts per checkpoint")
    check_all(merged["mean"], merged.checkpoint_mean_prompt_kl, 1e-12, "cell mean = checkpoint mean KL")
    check_all(merged["median"], merged.checkpoint_median_prompt_kl, 1e-12, "cell median = checkpoint median KL")

    rows = cp[(cp.fit_scope == "coefficient_or_fixed_baseline") & cp.model.isin(TABLE3_MODELS)]
    rows = rows[[k in keys for k in zip(rows.seed, rows.tag)]]
    require(len(rows) == 40, "4 predictors x 10 checkpoints")
    spread = rows.groupby(["seed", "tag"])[["frontier", "dpo_realized"]].agg(lambda s: s.max() - s.min())
    require(float(spread.to_numpy().max()) == 0.0, "targets identical across predictors")
    check_all(rows.abs_error_dpo, (rows.prediction - rows.dpo_realized).abs(), 1e-15, "|pred - realized|")
    check_all(rows.abs_error_frontier, (rows.prediction - rows.frontier).abs(), 1e-15, "|pred - frontier|")

    # The saved common calibration coefficients reproduce every prediction.
    A, B, A_nu = coef["A"], coef["B"], coef["A_nu"]
    require(coef["n_prompts_used"] == 99 and coef["n_prompts_skipped"] == 1, "99/100 calibration prompts")
    d = cells.d_reweight
    cells = cells.assign(**{"A+B": A * np.sqrt(d) + B * d, "A-only": A * np.sqrt(d), "A_nu": A_nu * np.sqrt(d)})
    recomputed = cells.groupby(["seed", "tag"])[["A+B", "A-only", "A_nu"]].mean()
    for model in ["A+B", "A-only", "A_nu"]:
        saved = rows[rows.model == model].set_index(["seed", "tag"]).prediction
        check_all(recomputed[model].reindex(saved.index), saved, 1e-12, f"{model} prediction from coefficients")
    zero = rows[rows.model == "Zero"].prediction
    require(bool((zero == 0).all()), "zero predictor is 0")

    wide = rows.pivot_table(index=["seed", "tag"], columns="model",
                            values=["prediction", "abs_error_frontier", "abs_error_dpo"])
    wide.columns = [f"{value}__{model}" for value, model in wide.columns]
    base = rows.drop_duplicates(["seed", "tag"]).set_index(["seed", "tag"])[["d_med", "frontier", "dpo_realized"]]
    ck = base.join(sel.set_index(["seed", "tag"])[["checkpoint_mean_prompt_kl"]]).join(wide).reset_index()
    ck = ck.rename(columns={"d_med": "median_prompt_kl", "checkpoint_mean_prompt_kl": "mean_prompt_kl",
                            "frontier": "idealized_tilt_gain", "dpo_realized": "likelihood_ratio_gain"})
    ck["idealized_minus_realized"] = ck.idealized_tilt_gain - ck.likelihood_ratio_gain
    ck["step"] = ck.tag.str.replace("steps_", "").astype(int)
    ck = ck.sort_values(["seed", "step"]).drop(columns="step").reset_index(drop=True)
    return ck, keys, coef, excluded_median


def build_dpo_table3(group: Group):
    disp = group.display(
        "dpo_table3", "main text",
        "Three-seed LoRA-DPO bridge: KL radius, likelihood-ratio vs idealized-tilt silver gain, and MAE of "
        "A+B / A-only / zero / A_nu against both targets (mean +/- sample SD over seed means).")
    ck, keys, coef, excluded_median = dpo_checkpoints(disp)
    window = disp.read_csv(CANON / "window_summary_mean_sd.csv")
    radius = disp.read_csv(AUDIT / "prompt_radius_summary.csv")
    canon_spec = json.loads(disp.input(CANON / "run_spec.json").read_text())

    per_seed = ck.groupby("seed").agg(
        n_checkpoints=("tag", "size"), mean_prompt_kl=("mean_prompt_kl", "mean"),
        idealized_tilt_gain=("idealized_tilt_gain", "mean"), likelihood_ratio_gain=("likelihood_ratio_gain", "mean"),
        idealized_minus_realized=("idealized_minus_realized", "mean"),
        **{f"mae_realized__{m}": (f"abs_error_dpo__{m}", "mean") for m in TABLE3_MODELS},
        **{f"mae_frontier__{m}": (f"abs_error_frontier__{m}", "mean") for m in TABLE3_MODELS})
    require(per_seed.n_checkpoints.tolist() == [4, 3, 3], "4/3/3 checkpoints")
    per_seed["realization_ratio"] = per_seed.likelihood_ratio_gain / per_seed.idealized_tilt_gain

    def summary_row(quantity, column, predictor="", target=""):
        values = per_seed[column].to_numpy()
        return dict(quantity=quantity, predictor=predictor, target=target,
                    seed_0=values[0], seed_1=values[1], seed_2=values[2],
                    mean=values.mean(), sd=values.std(ddof=1), n_seeds=3,
                    checkpoints_per_seed="4/3/3")

    rows = [summary_row("mean_prompt_kl_radius", "mean_prompt_kl"),
            summary_row("silver_gain", "likelihood_ratio_gain", target="likelihood_ratio_realized"),
            summary_row("silver_gain", "idealized_tilt_gain", target="idealized_tilt_frontier"),
            summary_row("idealized_minus_realized_gain", "idealized_minus_realized"),
            summary_row("realization_ratio_per_seed", "realization_ratio")]
    for model in TABLE3_MODELS:
        rows.append(summary_row("mae", f"mae_realized__{model}", model, "likelihood_ratio_realized"))
        rows.append(summary_row("mae", f"mae_frontier__{model}", model, "idealized_tilt_frontier"))
    summary = pd.DataFrame(rows)

    def get(quantity, predictor="", target=""):
        hit = summary[(summary.quantity == quantity) & (summary.predictor == predictor) & (summary.target == target)]
        require(len(hit) == 1, f"summary lookup {quantity}/{predictor}/{target}")
        return hit.iloc[0]

    kl = get("mean_prompt_kl_radius")
    lr = get("silver_gain", target="likelihood_ratio_realized")
    ideal = get("silver_gain", target="idealized_tilt_frontier")
    gap = get("idealized_minus_realized_gain")
    ratio = lr["mean"] / ideal["mean"]
    mae = {(m, t): get("mae", m, t) for m in TABLE3_MODELS
           for t in ("likelihood_ratio_realized", "idealized_tilt_frontier")}

    # ---- audited numbers
    check(kl["mean"], 0.03968, 5e-6, "KL radius mean"); check(kl["sd"], 0.00136, 5e-6, "KL radius SD")
    check(ideal["mean"], 0.04197, 5e-6, "idealized gain"); check(ideal["sd"], 0.00017, 5e-6, "idealized SD")
    check(lr["mean"], 0.02081, 5e-6, "LR gain"); check(lr["sd"], 0.00062, 5e-6, "LR gain SD")
    check(ratio, 0.496, 5e-4, "LR / idealized ratio")
    check(gap["mean"], 0.02116, 5e-6, "gap"); check(gap["sd"], 0.00074, 5e-6, "gap SD")
    audited = {"A+B": (0.01721, 0.00116, 0.00395), "A-only": (0.01905, 0.00120, 0.00210),
               "Zero": (0.02081, 0.00062, None), "A_nu": (0.02487, 0.00129, None)}
    for model, (m, s, f) in audited.items():
        check(mae[model, "likelihood_ratio_realized"]["mean"], m, 5e-6, f"{model} realized MAE")
        check(mae[model, "likelihood_ratio_realized"]["sd"], s, 5e-6, f"{model} realized MAE SD")
        if f is not None:
            check(mae[model, "idealized_tilt_frontier"]["mean"], f, 5e-6, f"{model} frontier MAE")
    # Saved strict-window summary agrees exactly with this seed-level recomputation.
    strict = window[window.scope == STRICT].set_index(["model", "fit_scope"])
    for model in TABLE3_MODELS:
        saved = strict.loc[(model, "coefficient_or_fixed_baseline")]
        check(saved.mean_mae_dpo, mae[model, "likelihood_ratio_realized"]["mean"], 1e-12, f"{model} window mean")
        check(saved.sd_mae_dpo, mae[model, "likelihood_ratio_realized"]["sd"], 1e-12, f"{model} window SD")
        check(saved.mean_mae_frontier, mae[model, "idealized_tilt_frontier"]["mean"], 1e-12, f"{model} window frontier")
        check(saved.sd_mae_frontier, mae[model, "idealized_tilt_frontier"]["sd"], 1e-12, f"{model} window frontier SD")
        require(saved.min_checkpoints_per_seed == 3 and saved.max_checkpoints_per_seed == 4, "window 3-4 checkpoints")
    # A+B beats A-only on realized gain in every seed and every checkpoint; A-only is closer to the frontier.
    seed_diff = per_seed["mae_realized__A-only"] - per_seed["mae_realized__A+B"]
    check_all(seed_diff, [0.00187, 0.00177, 0.00189], 5e-6, "per-seed A-only minus A+B")
    n_ck_win = int((ck["abs_error_dpo__A+B"] < ck["abs_error_dpo__A-only"]).sum())
    require(n_ck_win == 10, "A+B < A-only at all 10 checkpoints")
    n_seed_front = int((per_seed["mae_frontier__A-only"] < per_seed["mae_frontier__A+B"]).sum())
    ordered = ((ck.likelihood_ratio_gain < ck["prediction__A+B"]) & (ck["prediction__A+B"] < ck["prediction__A-only"])
               & (ck["prediction__A-only"] < ck.idealized_tilt_gain))
    require(int(ordered.sum()) == 10, "realized < A+B < A-only < idealized at all 10 checkpoints")
    require(n_seed_front == 3, "A-only closer to frontier in every seed")
    # Prompt-radius tail of the retained cells.
    pooled = radius[radius.scope == "pooled"].iloc[0]
    tail = [pooled["median"], pooled.p90, pooled.p95, pooled.p99, pooled["max"]]
    check_all(tail, [0.0257, 0.0919, 0.1224, 0.1867, 0.4668], 5e-5, "prompt-radius quantiles")
    check(pooled.fraction_gt_0p10, 0.087, 1e-12, "fraction of prompt cells above .10")
    require(int(pooled.n_prompt_checkpoint_cells) == 2000, "2000 prompt-checkpoint cells")
    require(canon_spec["strict_local_window"] == "checkpoint median d <= 0.10", "canonical run_spec wording")

    disp.write_data(summary)
    disp.write_data(ck, suffix="_checkpoints")

    body = [
        ["Mean KL radius $d$", rf"\multicolumn{{2}}{{c}}{{{pm(kl['mean'], kl['sd'])}}}"],
        ["Mean silver gain", pm(lr["mean"], lr["sd"]), pm(ideal["mean"], ideal["sd"])],
        ["Idealized minus realized gain", rf"\multicolumn{{2}}{{c}}{{{pm(gap['mean'], gap['sd'])}}}"],
        section("MAE of calibration-only prediction", 3),
    ]
    for model in TABLE3_MODELS:
        body.append([PRED_LABEL[model],
                     pm(mae[model, "likelihood_ratio_realized"]["mean"], mae[model, "likelihood_ratio_realized"]["sd"]),
                     pm(mae[model, "idealized_tilt_frontier"]["mean"], mae[model, "idealized_tilt_frontier"]["sd"])])
    disp.write_table("lcc", [["", "Realized", "Idealized"],
                             [r"Mean $\pm$ SD over 3 seeds", "likelihood-ratio gain", "tilt gain"]],
                     body, midrules_after=[2])

    s = lambda q: pmt(q["mean"], q["sd"])  # noqa: E731
    seed_wins = number_list(seed_diff, 5)
    disp.write_note(f"""
# dpo_table3: LoRA-DPO bridge (Table 3)

**Placement.** Main text, Table 3 (CAMERA_READY_PLAN.md §5). If space is tight, move the table to
the appendix but keep the setup, both gains, the A+B vs A-only numbers and the estimand caveat in
the main text. Explain the realized vs idealized-tilt distinction in adjacent prose. Link the full
comparator table (`dpo_comparators`); this restricted four-predictor table must not be read as a
claim that A+B is the best realized-gain predictor.

**What it shows.** Three LoRA-DPO training seeds; checkpoints gated by mean prompt KL <= 0.10
(4/3/3 checkpoints for seeds 0/1/2). Top block: KL radius and the two silver-gain targets.
Bottom block: MAE of four calibration-only predictors against each target. Lower is better.

**Setup.** Policy Qwen2.5-3B-Instruct, LoRA-DPO (beta 0.1, lr 1e-5, effective batch 16, LoRA
rank/alpha/dropout 16/32/0.05) on 968 nonzero-gap preference pairs from 1,000 UltraFeedback prompts
(8 candidates each); chosen/rejected = highest/lowest Skywork-Reward-Llama-3.1-8B-v0.2 score (the
optimized proxy r). Frozen silver target J = OpenAssistant/reward-model-deberta-v3-large-v2, never
used in training. Training, calibration (100 prompts, 99 usable) and evaluation (200 prompts) prompt
sets are disjoint. Calibration and evaluation pools: 32 fixed candidates per prompt from the reference
policy (temperature 0.9, top-p 0.95, <= 320 new tokens). Seeds share data and pools; only training
randomness differs.. The numbers here are recomputed from the saved CSVs.

**Definitions and units.** All gains are in silver-target (DeBERTa reward) units.
- Per evaluation prompt x: w_x,i proportional to exp(log p_DPO(y_i|x) - log p_ref(y_i|x)) over its
  32 candidates; d_x = KL(w_x || Unif_32).
- Realized (likelihood-ratio) gain: sum_i w_x,i J_i minus the uniform pool mean of J, averaged over
  the 200 evaluation prompts (`dpo_realized`).
- Idealized tilt gain: silver gain of the proxy-score exponential tilt of the same pool at the same
  d_x (`frontier`).
- Predictors: A+B = mean_x(A sqrt(d_x) + B d_x); A-only = mean_x(A sqrt(d_x));
  A_nu = mean_x(A_nu sqrt(d_x)) (variance-ratio surrogate, sqrt(2) Var(J)/SD(r) per calibration prompt,
  averaged; no rater reliability exists for a deterministic silver target); Zero = 0. Common
  calibration-only coefficients A = {coef['A']:.4f}, B = {coef['B']:.4f}, A_nu = {coef['A_nu']:.4f}
  (`actual_dpo/coefficients.json`, 99/100 prompts; one calibration prompt has zero proxy and silver
  variance). This builder re-derives every saved checkpoint prediction from these coefficients and
  the saved prompt radii (max deviation < 1e-12).
- KL radius row: per checkpoint, the mean of the 200 prompt radii d_x; then averaged within seed.

**Aggregation.** Per checkpoint, error = |prompt-averaged prediction - prompt-averaged target|.
Average over the retained checkpoints within each seed (4/3/3), then report mean and sample SD
(ddof = 1) across the three seed means. The SD is seed-to-seed spread, not a confidence interval.

**Protocol / gate.** Selected seed-checkpoints: seed 0 steps 64/90/120/150; seeds 1 and 2 steps
64/90/120. Gate = checkpoint mean prompt KL <= 0.10 (`prompt_radius_audit/run_spec.json`). The
`three_seed_canonical/run_spec.json` and exp34 run_spec state "checkpoint median d <= 0.10"; this
builder verifies that the median rule selects exactly the same ten checkpoints (the eight excluded
checkpoints have median d {excluded_median.min():.3f}-{excluded_median.max():.3f}). The gate uses no
silver label, gain or error, but it is transductive in the reported radius. Not every prompt is
local: across the 2,000 retained prompt-checkpoint cells, d_x median/p90/p95/p99/max =
{number_list(tail, 4)}, and {pooled.fraction_gt_0p10 * 100:.1f}% exceed 0.10.

**Sources.** `results/actual_dpo/three_seed_canonical/checkpoint_predictions.csv`
(checkpoint rows, A+B / A-only / A_nu / Zero, `coefficient_or_fixed_baseline`);
`.../prompt_radius_audit/{{run_spec.json, prompt_radius_cells.csv, prompt_radius_summary.csv}}`;
`.../window_summary_mean_sd.csv` (`{STRICT}` rows, used only as an exact cross-check);
`results/actual_dpo/coefficients.json`. NOT used: the root `summary_mean_sd.csv`,
`seed_level_mae.csv`, `error_decomposition_summary.csv` (historical six-checkpoint scope, A+B
0.02294), `exp09_final_summary.csv`, `kl_sweep*.csv` (old single runs).

**Key results** (mean +/- sample SD over 3 seed means).
- Mean KL radius {s(kl)}; idealized-tilt gain {s(ideal)}; likelihood-ratio gain {s(lr)};
  ratio of means {ratio:.3f}; idealized minus realized {s(gap)}.
- Realized-gain MAE: A+B {s(mae['A+B', 'likelihood_ratio_realized'])}, A-only
  {s(mae['A-only', 'likelihood_ratio_realized'])}, zero {s(mae['Zero', 'likelihood_ratio_realized'])},
  A_nu {s(mae['A_nu', 'likelihood_ratio_realized'])}.
- Frontier MAE: A+B {s(mae['A+B', 'idealized_tilt_frontier'])}, A-only
  {s(mae['A-only', 'idealized_tilt_frontier'])}, A_nu {s(mae['A_nu', 'idealized_tilt_frontier'])},
  zero {s(mae['Zero', 'idealized_tilt_frontier'])}.
- A-only minus A+B realized MAE per seed 0/1/2: {seed_wins}; A+B < A-only at {n_ck_win}/10 checkpoints.
- A-only has lower frontier MAE than A+B in {n_seed_front}/3 seeds.

**Reporting details.**
1. {DPO_ESTIMAND}
2. Mean +/- sample SD across three training seeds after averaging 4/3/3 retained checkpoints per seed;
   gains in silver-target units; lower MAE is better.
3. The two target columns: realized = likelihood-ratio reweighting; idealized = proxy exponential tilt
   of the same pool at the same d_x.
4. Restricted predictor set; pointer to the full comparator table (`dpo_comparators`).

**Caveats for adjacent prose.**
- A-only is closer to the idealized frontier (0.00210 vs 0.00395). At all 10 checkpoints the ordering is
  realized gain < A+B prediction < A-only prediction < idealized gain (asserted here). Both predictors
  underpredict the frontier; A+B's larger underprediction (its negative B term) moves it toward the
  realized gain, so its realized-gain advantage partly reflects error cancellation against the
  optimizer shortfall (realized gain is about half the idealized gain), not a better frontier model.
- Other comparators reach lower realized MAE (Spearman A-only variant
  0.01378, exp34 prompt-rank binned 0.01096); see `dpo_comparators`. No universal-best claim.
- Three seeds: descriptive consistency, not significance. Common coefficients applied to promptwise
  radii are an empirical pooled-coefficient approximation; 8.7% of prompt cells lie beyond 0.10.
- Do not merge numerically with the exp37 reward-model results (different target and mechanism).

**Formatting.** Five decimals for every gain, KL radius and MAE, fixed within the table, matching the saved results; the README's 4-decimal endpoint rule targets human-score endpoint MAE, and
the 3-decimal KL rule would show the radius as 0.040 +/- 0.001. `dpo_table3.csv` holds unrounded
per-seed values; `dpo_table3_checkpoints.csv` the ten checkpoint rows.
""")
    return keys, mae


def build_dpo_comparators(group: Group, keys: set):
    disp = group.display(
        "dpo_comparators", "appendix",
        "Full DPO comparator table by information setting: pre-pool coefficient and frozen-curve predictors "
        "(exp09/exp28/exp34), target-pool-aware direct and two-term projection (exp28 follow-up), "
        "and evaluation-prompt cross-fit diagnostics; frontier and realized MAE, mean +/- SD over 3 seeds.")
    window = disp.read_csv(CANON / "window_summary_mean_sd.csv")
    w = window[window.scope == STRICT].set_index("model")
    require(len(w) == 21, "21 strict exp28 rows")
    pool = disp.read_csv(POOL / "summary_mean_sd.csv").set_index("model")
    require(len(pool) == 12, "12 pool-aware rows")
    pool_sel = disp.read_csv(POOL / "selected_checkpoints.csv")
    require(set(zip(pool_sel.seed, pool_sel.tag)) == keys, "follow-up uses the same 10 checkpoints")
    pool_spec = json.loads(disp.input(POOL / "run_spec.json").read_text())
    require(pool_spec["selection"]["checkpoint_mean_prompt_kl_max"] == 0.1, "follow-up mean-KL gate")
    e34 = disp.read_csv(EXP34 / "summary_mean_sd.csv")
    e = e34[e34.scope == STRICT].set_index("model")
    require(len(e) == 13, "13 strict exp34 rows")
    e_spec = json.loads(disp.input(EXP34 / "run_spec.json").read_text())
    e_audit = json.loads(disp.input(EXP34 / "audit.json").read_text())
    require(e_audit["status"] == "PASS", "exp34 audit PASS")
    require(e_spec["strict_local_scope"] == "checkpoint median d <= 0.10", "exp34 strict scope wording")
    require(((e.min_checkpoints == 3) & (e.max_checkpoints == 4) & (e.n_seeds == 3)).all(), "exp34 3-4 checkpoints")

    rows = []

    def add(panel, label, source, key, fm, fs, dm, ds, gm=np.nan, gs=np.nan, duplicate=""):
        rows.append(dict(panel=panel, predictor=label, source=source, source_model=key,
                         frontier_mae_mean=fm, frontier_mae_sd=fs, realized_mae_mean=dm, realized_mae_sd=ds,
                         projection_direct_gap_mean=gm, projection_direct_gap_sd=gs, identical_row_in=duplicate))

    src28 = rel(CANON / "window_summary_mean_sd.csv")
    src34 = rel(EXP34 / "summary_mean_sd.csv")
    srcpool = rel(POOL / "summary_mean_sd.csv")
    panel_a = "a_pre_pool"
    # Coefficient predictors: identical in exp28 and exp34 (inherited exp09 fit).
    for model in ["A+B", "A-only", "Spearman", "A_nu", "Zero"]:
        r28 = w.loc[model]
        require(r28.fit_scope == "coefficient_or_fixed_baseline", f"{model} fit scope")
        dup = ""
        if model != "Spearman":
            r34 = e.loc[model]
            check_all([r34.mean_mae_frontier, r34.sd_mae_frontier, r34.mean_mae_dpo, r34.sd_mae_dpo],
                      [r28.mean_mae_frontier, r28.sd_mae_frontier, r28.mean_mae_dpo, r28.sd_mae_dpo],
                      1e-12, f"{model} exp28 = exp34")
            dup = f"{src34}:{model}"
        label = r"Spearman $A$-only variant" if model == "Spearman" else PRED_LABEL[model]
        add(panel_a, label, src28, model, r28.mean_mae_frontier, r28.sd_mae_frontier,
            r28.mean_mae_dpo, r28.sd_mae_dpo, duplicate=dup)
    exp34_curves = ["cal_empirical_curve"] + [f"cal_{m}_{f}" for m in ("raw", "rank")
                                              for f in ("isotonic", "binned", "spline", "mlp")]
    for model in exp34_curves:
        r = e.loc[model]
        require(r.information_setting == "calibration_only_frozen_curve", f"{model} setting")
        if model == "cal_empirical_curve":
            label = "Calibration empirical curve"
        else:
            _, mode, family = model.split("_")
            label = f"{MODE_LABEL[mode]} {FAMILY_LABEL[family]} (frozen curve)"
        add(panel_a, label, src34, model, r.mean_mae_frontier, r.sd_mae_frontier, r.mean_mae_dpo, r.sd_mae_dpo)
    require(set(e.index) == {"A+B", "A-only", "A_nu", "Zero", *exp34_curves}, "all 13 exp34 rows used")

    grid = [(m, f) for m in ("raw", "centered", "zscore", "rank") for f in ("isotonic", "binned", "spline")]
    for mode, family in grid:  # (b) direct integration
        key = f"cal_{mode}_{family}"
        r = pool.loc[key]
        dup = ""
        if family != "binned":  # exp28 calibration_only rows are the same direct predictions
            r28 = w.loc[key]
            require(r28.fit_scope == "calibration_only", f"{key} exp28 scope")
            check_all([r28.mean_mae_frontier, r28.sd_mae_frontier, r28.mean_mae_dpo, r28.sd_mae_dpo],
                      [r.direct_frontier_mae_mean, r.direct_frontier_mae_sd, r.direct_dpo_mae_mean, r.direct_dpo_mae_sd],
                      1e-12, f"{key} exp28 = follow-up direct")
            dup = f"{src28}:{key}"
        add("b_pool_aware_direct", f"{MODE_LABEL[mode]} {family}", srcpool, f"{key}:direct",
            r.direct_frontier_mae_mean, r.direct_frontier_mae_sd, r.direct_dpo_mae_mean, r.direct_dpo_mae_sd,
            duplicate=dup)
    for mode, family in grid:  # (c) two-term projection
        key = f"cal_{mode}_{family}"
        r = pool.loc[key]
        add("c_pool_aware_projection", f"{MODE_LABEL[mode]} {family}", srcpool, f"{key}:projection",
            r.projection_frontier_mae_mean, r.projection_frontier_mae_sd, r.projection_dpo_mae_mean,
            r.projection_dpo_mae_sd, r.aggregate_projection_direct_gap_mean, r.aggregate_projection_direct_gap_sd)
    for mode in ("raw", "centered", "zscore", "rank"):  # (d) cross-fit diagnostics
        for family in ("isotonic", "spline"):
            key = f"eval_xfit_{mode}_{family}"
            r = w.loc[key]
            require(r.fit_scope == "eval_prompt_crossfit_diagnostic", f"{key} scope")
            add("d_eval_crossfit_diagnostic", f"{MODE_LABEL[mode]} {family} (cross-fit)", src28, key,
                r.mean_mae_frontier, r.sd_mae_frontier, r.mean_mae_dpo, r.sd_mae_dpo)
    table = pd.DataFrame(rows)
    require(len(table) == 46, "46 display rows")
    used28 = {"A+B", "A-only", "Spearman", "A_nu", "Zero"} | {f"cal_{m}_{f}" for m, f in grid if f != "binned"} \
        | {f"eval_xfit_{m}_{f}" for m in ("raw", "centered", "zscore", "rank") for f in ("isotonic", "spline")}
    require(used28 == set(w.index), "all 21 exp28 strict rows used")

    # ---- audited numbers
    check(w.loc["Spearman"].mean_mae_dpo, 0.01378, 5e-6, "Spearman realized MAE")
    require(w.mean_mae_dpo.idxmin() == "Spearman", "Spearman best realized among exp28 rows")
    check(e.loc["cal_rank_binned"].mean_mae_dpo, 0.01096, 5e-6, "exp34 rank binned realized MAE")
    require(e.mean_mae_dpo.idxmin() == "cal_rank_binned", "rank binned best realized among exp34 rows")
    max_gap = pool.aggregate_projection_direct_gap_mean.max()
    check(max_gap, 0.00037, 5e-6, "max projection-direct gap")
    require(max_gap <= 0.000372, "projection-direct gap <= 0.00037")
    require(pool.aggregate_projection_direct_gap_mean.idxmax() == "cal_zscore_binned", "largest gap: z-score binned")
    best_realized = table.loc[table.realized_mae_mean.idxmin()]
    best_frontier = table.loc[table.frontier_mae_mean.idxmin()]
    ab = table[(table.panel == panel_a) & (table.source_model == "A+B")].iloc[0]
    n_below_ab = int((table.realized_mae_mean < ab.realized_mae_mean).sum())
    below_by_panel = ", ".join(f"{panel[0]}: {int((sub.realized_mae_mean < ab.realized_mae_mean).sum())}/{len(sub)}"
                               for panel, sub in table.groupby("panel"))
    plain = lambda text: text.replace("$", "").replace("{+}", "+").replace(r"\nu", "nu")  # noqa: E731

    disp.write_data(table)
    body, breaks, panels = [], [], [
        ("a_pre_pool", r"(a) Pre-pool: calibration prompts and measured $d_x$ only"),
        ("b_pool_aware_direct", r"(b) Target-pool-aware: direct integration of $\hat h(r)$"),
        ("c_pool_aware_projection", r"(c) Target-pool-aware: two-term projection of the same $\hat h(r)$"),
        ("d_eval_crossfit_diagnostic", "(d) Evaluation-prompt cross-fit diagnostic (uses evaluation silver labels)")]
    for panel, title in panels:
        if body:
            breaks.append(len(body) - 1)
        body.append(section(title, 4))
        for r in table[table.panel == panel].itertuples():
            gap = pm(r.projection_direct_gap_mean, r.projection_direct_gap_sd) if panel.startswith("c_") else "--"
            body.append([r.predictor, pm(r.frontier_mae_mean, r.frontier_mae_sd),
                         pm(r.realized_mae_mean, r.realized_mae_sd), gap])
    disp.write_table("lccc", [["", "MAE vs idealized", "MAE vs realized", r"Projection$-$direct"],
                              ["Predictor", "tilt frontier", "likelihood-ratio gain", "gap"]],
                     body, midrules_after=breaks)

    disp.write_note(f"""
# dpo_comparators: full DPO comparator table

**Placement.** Appendix, linked from Table 3 (`dpo_table3`). Information settings differ by panel;
compare rows within a panel, and across panels only with the information difference stated.

**What it shows.** MAE of 46 predictor variants against the idealized tilt frontier and against the
realized likelihood-ratio gain, on the same ten mean-KL-gated checkpoints as Table 3. Cells are
mean +/- sample SD over three seed means (4/3/3 checkpoints per seed). The last column applies only to
panel (c): the aggregate projection-minus-direct prediction gap for the same fitted h(r) (`--` elsewhere
because those rows have no paired projection).

**Panels (information settings).**
- (a) Pre-pool: uses only the 100 rated calibration prompts (3,200 candidates) and each evaluation
  prompt's measured d_x; evaluation candidates are never loaded. A+B, A-only, A_nu, Zero: common moment
  coefficients (exp09 fit, 99/100 prompts). Spearman A-only variant: A_S sqrt(d_x) with
  A_S = mean over calibration prompts of sqrt(2) SD(J) Spearman(r, J) (an A-only variant using Spearman
  correlation; exp09 `11_combined_analysis.py`). Frozen curves (exp34): flexible calibrator fitted and
  integrated on calibration candidates, prompt-averaged gain curve frozen, evaluated at d_x. Prompt-rank =
  within-prompt rank coordinate.
- (b) Target-pool-aware direct: conditional mean h(r) fit on calibration prompts only, then exactly
  integrated under raw-proxy tilt over each evaluation prompt's proxy scores at d_x (evaluation proxy
  geometry used; evaluation silver labels not used).
- (c) Target-pool-aware projection: prompt-specific moments of (h(r_eval), r_eval) give A_h, B_h;
  prediction A_h sqrt(d_x) + B_h d_x. Same h and geometry as (b).
- (d) Cross-fit diagnostics: h fitted by cross-fitting across evaluation prompts, i.e. using evaluation
  silver labels; descriptive distribution-shift diagnostics, not deployable predictors.

**Sources and de-duplication.**
- exp28 strict rows: `{src28}` (`{STRICT}`, 21 rows: 5 coefficient, 8 `calibration_only`, 8 cross-fit).
- Pool-aware follow-up: `{srcpool}` (12 models x projection/direct).
- exp34 pre-pool: `{src34}` (`{STRICT}`, 13 rows; original outputs restored 2026-10-12, audit PASS).
- A+B/A-only/A_nu/Zero are identical in exp28 and exp34 (checked < 1e-12) and shown once in (a). The
  8 exp28 `calibration_only` isotonic/spline rows equal the follow-up's direct rows (< 1e-12) and are
  shown once in (b). `dpo_comparators.csv` records the duplicate source in `identical_row_in`.
- The follow-up's `selected_checkpoints.csv` is the same 10 checkpoints as Table 3. exp34's run_spec
  states "checkpoint median d <= 0.10" (same ten checkpoints).

**Key results.**
- Best realized MAE in exp28 strict rows: Spearman A-only variant
  {pmt(w.loc['Spearman'].mean_mae_dpo, w.loc['Spearman'].sd_mae_dpo)}.
- Best realized MAE in exp34 pre-pool rows: prompt-rank binned
  {pmt(e.loc['cal_rank_binned'].mean_mae_dpo, e.loc['cal_rank_binned'].sd_mae_dpo)} (frontier MAE
  {e.loc['cal_rank_binned'].mean_mae_frontier:.5f}).
- Lowest realized MAE in the whole table: {plain(best_realized.predictor)}, panel ({best_realized.panel[0]}),
  {best_realized.realized_mae_mean:.5f}; lowest frontier MAE: {plain(best_frontier.predictor)}, panel
  ({best_frontier.panel[0]}), {best_frontier.frontier_mae_mean:.5f}.
- {n_below_ab} of 46 rows have lower realized MAE than A+B ({ab.realized_mae_mean:.5f}); by panel
  {below_by_panel}.
- Projection-direct aggregate gap <= {max_gap:.5f} for every conditional-mean model (max: z-score binned).

**Reporting details.**
1. {DPO_ESTIMAND}
2. Mean +/- sample SD over three seed means; same ten checkpoints as Table 3; lower is better.
3. The information setting of each panel, especially that (b)/(c) use evaluation proxy scores and (d)
   uses evaluation silver labels.
4. Realized-gain MAE mixes predictor error with the optimizer shortfall (realized gain is about half the
   idealized gain, gap 0.02116), so a lower realized MAE can arise from cancellation; frontier MAE and
   realized MAE answer different questions. No universal-best claim.

**Caveats.**
- The CV-tuned binned and spline rows use GroupKFold over 100 prompt groups. Reproduce them
  with the pinned NumPy and scikit-learn versions in `docs/ENVIRONMENT.md`; the published
  reference outputs are retained in `results/`.
- Three seeds: descriptive; per-seed SDs are not confidence intervals.
- The MLP rows are exploratory (exp34: fixed 1-16-16-1 ReLU MLP, alpha 1e-3, 1,000 iterations, seed 0;
  no tuning).

**Formatting.** Five decimals throughout (as in `dpo_table3`). Row order is by method family, not by
MAE; no bolding, to avoid implying a winner.
""")


# ================================================================ exp37 reward models
def rm_preamble(disp: Display) -> dict:
    run_spec = json.loads(disp.input(RM / "analysis" / "run_spec.json").read_text())
    require(run_spec["env"]["numpy"] == "2.3.5", "exp37 NumPy 2.3.5 environment recorded")
    require(run_spec["repeats"] == 50 and run_spec["radii"] == [0.025, 0.05, 0.075, 0.1], "exp37 protocol")
    return run_spec


def build_rm_worked_example(group: Group) -> pd.DataFrame:
    disp = group.display(
        "rm_worked_example", "appendix (worked example; the main text states all three rows in prose)",
        "Exp37 preselected SummEval coherence split (repeat 0, d=0.10): calibration-only A, B, their "
        "contributions, predicted and held-out exact gain for Skywork, ArmoRM and DeBERTa.")
    rm_preamble(disp)
    worked = disp.read_csv(RM / "paper_example" / "worked_example.csv")
    existing = disp.input(RM / "paper_example" / "table_reward_model_worked_example.tex")
    report = json.loads(disp.input(RM / "paper_example" / "report_spec.json").read_text())
    for name in ("worked_example.csv", "table_reward_model_worked_example.tex"):
        require(report["files"][name] == sha256(RM / "paper_example" / name), f"{name} matches report_spec hash")
    require("Preselected coherence, repeat 0, d=0.10" in report["selection"], "preselection recorded")
    retained = report["context"]["mean_retained_fraction"]
    check(retained, 0.9608, 5e-5, "mean retained context fraction")
    models = json.loads(disp.input(RM / "prepared" / "models.json").read_text())
    require(list(models) == RM_ORDER and all(len(m["revision"]) == 40 for m in models.values()), "pinned revisions")
    pins = "; ".join(f"`{m['model_id']}` @ `{m['revision']}` ({m['dtype']}, max {m['max_length']} tokens)"
                     for m in models.values())

    worked = worked.set_index("alias").loc[RM_ORDER].reset_index()
    require((worked.target == "coherence").all() and (worked["repeat"] == 0).all(), "coherence repeat 0")
    require((worked.n_cal == 480).all() and (worked.n_eval == 1120).all(), "480/1120 split")
    check_all(worked.first_order, worked.A_cal * np.sqrt(0.1), 1e-15, "A sqrt(.1)")
    check_all(worked.second_order, worked.B_cal * 0.1, 1e-15, "B (.1)")
    check_all(worked.predicted_gain, worked.first_order + worked.second_order, 1e-15, "predicted = sum")
    worked["abs_err_ab"] = (worked.predicted_gain - worked.heldout_gain).abs()
    worked["abs_err_aonly"] = (worked.first_order - worked.heldout_gain).abs()
    audited = {"skywork": [0.2884, -0.0054, 0.0912, -0.0005, 0.0907, 0.0785],
               "armorm": [0.2695, -0.0579, 0.0852, -0.0058, 0.0794, 0.0775],
               "deberta": [0.1974, -0.0445, 0.0624, -0.0044, 0.0580, 0.0643]}
    cols = ["A_cal", "B_cal", "first_order", "second_order", "predicted_gain", "heldout_gain"]
    for r in worked.itertuples():
        for col, value in zip(cols, audited[r.alias]):
            check(getattr(r, col), value, 5e-5, f"{r.alias} {col}")
    deb = worked[worked.alias == "deberta"].iloc[0]
    check(deb.abs_err_ab, 0.0063, 5e-5, "DeBERTa A+B split error")
    check(deb.abs_err_aonly, 0.0018, 5e-5, "DeBERTa A-only split error")
    require(deb.abs_err_ab > deb.abs_err_aonly, "DeBERTa: adding B worsens this split")

    rows = [[RM_LABEL[r.alias]] + [fmt(getattr(r, c), 4) for c in cols] for r in worked.itertuples()]
    old = tex_body_cells(existing)
    require(plain_minus(rows) == old, "regenerated worked-example body equals the exp37 table body")
    disp.write_data(worked)
    disp.write_table("lrrrrrr", [["Reward model", r"$\widehat A$", r"$\widehat B$", r"$\widehat A\sqrt{.10}$",
                                  r"$\widehat B(.10)$", "Predicted gain", "Held-out gain"]], rows)
    w = worked.set_index("alias")
    disp.write_note(f"""
# rm_worked_example: three-RM calibration-only worked example (exp37)

**Placement.** Appendix (CAMERA_READY_PLAN.md §8.1, §8.3). Space was short, so Section 5.1 states all
three decompositions in prose, including the DeBERTa row where adding B moves away from the held-out
gain, and points to this table.

**What it shows.** For one preselected split, the calibration-only coefficients, the separate
first-order (A sqrt(d)) and second-order (B d) contributions at d = 0.10, their sum, and the exact
held-out gain.

**Definitions and units.** Proxy r = frozen scalar reward of Skywork-Reward-Llama-3.1-8B-v0.2,
ArmoRM-Llama3-8B-v0.1 (final gated scalar) or OpenAssistant reward-model-deberta-v3-large-v2. Target
J = mean of three expert and five crowd SummEval coherence ratings (human coherence-score units).
A, B estimated on the 480 calibration items only. Predicted gain = A sqrt(.10) + B(.10). Held-out
gain = exact raw-score exponential-tilt gain at KL d = 0.10 on the 1,120 evaluation items using human
gold. A-only prediction = A sqrt(.10).

**Protocol.** SummEval coherence, repeat 0, d = 0.10, chosen before inspecting this run's results
(`paper_example/report_spec.json`: "{report['selection']}"). Item-level split (480/1,120), not
article-held-out; identical across the three models. Shared context: every model sees the same article
prefix; supplied prefixes are already capped at 2,000 characters and are further shortened to fit all
three tokenizers (4,096/4,096/512 tokens), keeping each summary in full; mean retained fraction
{retained * 100:.2f}% of the supplied prefixes (not of full articles). No fine-tuning, no new annotation.
Exp37 was analysed with NumPy 2.3.5; nothing is recomputed here.
Pinned models (`prepared/models.json`): {pins}.

**Sources.** `results/reward_model_coefficients/canonical/paper_example/worked_example.csv`
(values), `.../paper_example/table_reward_model_worked_example.tex` (checked: the regenerated body equals
it cell for cell; the only textual change is `$-$` for minus signs), `.../paper_example/report_spec.json`.

**Key results.** ArmoRM: A = {w.loc['armorm'].A_cal:.4f}, B = {w.loc['armorm'].B_cal:.4f},
A sqrt(.10) = {w.loc['armorm'].first_order:.4f}, B(.10) = {w.loc['armorm'].second_order:.4f}, predicted
{w.loc['armorm'].predicted_gain:.4f}, held-out exact {w.loc['armorm'].heldout_gain:.4f}. Split errors
|pred - held-out| A+B vs A-only: Skywork {w.loc['skywork'].abs_err_ab:.4f} vs {w.loc['skywork'].abs_err_aonly:.4f};
ArmoRM {w.loc['armorm'].abs_err_ab:.4f} vs {w.loc['armorm'].abs_err_aonly:.4f}; DeBERTa
{w.loc['deberta'].abs_err_ab:.4f} vs {w.loc['deberta'].abs_err_aonly:.4f}. Coherence 50-repeat endpoint MAE
A+B vs A-only: {w.loc['skywork'].mae_ab_50:.4f}/{w.loc['skywork'].mae_aonly_50:.4f} (Skywork),
{w.loc['armorm'].mae_ab_50:.4f}/{w.loc['armorm'].mae_aonly_50:.4f} (ArmoRM),
{w.loc['deberta'].mae_ab_50:.4f}/{w.loc['deberta'].mae_aonly_50:.4f} (DeBERTa).

**Reporting details.**
1. SummEval coherence, repeat 0, d = 0.10 preselected before inspecting results; one illustrative split,
   not the aggregate performance claim (that uses all 50 repeats: `rm_endpoint_mae`).
2. A, B from the 480 calibration items only; these calibration-only values differ from the full-pool
   descriptive coefficients (`rm_coefficients`), which use all 1,600 items and are not held-out predictors.
3. Held-out gain = exact raw-score exponential-tilt gain on 1,120 evaluation items, in human coherence
   units. Item-held-out, not article-held-out. Shared, truncated article context across models.
4. For DeBERTa, adding B worsens the error on this split ({deb.abs_err_ab:.4f} vs {deb.abs_err_aonly:.4f}).

**Caveats.** Raw-score tilt on a fixed pool, not DPO/RLHF training; do not merge with the DPO table.
A, B describe this proxy/target/distribution/context, not intrinsic RM quality. Negative B is a local
correction, not a global turning point. Training-data overlap of the RMs with SummEval is not excluded.
Context preprocessing was frozen on this benchmark's texts before splitting.
""")
    return worked


def build_rm_endpoint_mae(group: Group):
    disp = group.display(
        "rm_endpoint_mae", "appendix (main-eligible)",
        "Exp37 d=0.10 endpoint MAE of A-only, A+B and calibration empirical fit for three RMs, averaged over "
        "4 SummEval dimensions x 50 item splits, plus a 12-row per-dimension variant.")
    rm_preamble(disp)
    summary = disp.read_csv(RM / "analysis" / "summary.csv")
    paths = disp.read_csv(RM / "analysis" / "prospective_paths.csv")
    require(len(summary) == 48 and (summary.n_common_splits == 50).all(), "48 rows x 50 common splits")
    require(len(paths) == 600, "600 model-splits")
    # Saved summary means equal the saved split-level rows (consistency, not re-analysis).
    for radius in (0.025, 0.05, 0.075, 0.1):
        suffix = str(radius).replace(".", "p")
        means = paths.groupby(["alias", "dimension"])[[f"abs_err_{n}_{suffix}" for n in
                                                       ("theory", "aonly", "empfit", "anu", "zero")]].mean()
        sub = summary[np.isclose(summary.radius, radius)].set_index(["alias", "dimension"]).loc[means.index]
        check_all(sub[["mae_theory", "mae_aonly", "mae_empfit", "mae_anu", "mae_zero"]].to_numpy(),
                  means.to_numpy(), 1e-12, f"summary = split rows at d={radius}")
    n_b_helps_all = int((summary.mae_theory < summary.mae_aonly).sum())
    require(n_b_helps_all == 48, "A+B < A-only in 48/48 model/dimension/radius cells")

    end = summary[np.isclose(summary.radius, 0.1)].copy()
    require(len(end) == 12, "12 endpoint rows")
    n_b_helps = int((end.mae_theory < end.mae_aonly).sum())
    n_emp_beats = int((end.mae_empfit < end.mae_theory).sum())
    require(n_b_helps == 12, "12/12 decrease with B at d=.10")
    require(n_emp_beats == 7, "empirical fit beats A+B in 7/12")
    agg = end.groupby("alias")[["mae_aonly", "mae_theory", "mae_empfit", "mae_anu", "mae_zero",
                                "realized_gain_mean"]].mean().loc[RM_ORDER]
    agg["reduction_vs_aonly"] = 1 - agg.mae_theory / agg.mae_aonly
    audited = {"skywork": (0.015533, 0.011262, 0.011083), "armorm": (0.013607, 0.010459, 0.010513),
               "deberta": (0.015081, 0.011291, 0.011266)}
    for alias, (a_only, ab, emp) in audited.items():
        check(agg.loc[alias, "mae_aonly"], a_only, 5e-7, f"{alias} A-only endpoint MAE")
        check(agg.loc[alias, "mae_theory"], ab, 5e-7, f"{alias} A+B endpoint MAE")
        check(agg.loc[alias, "mae_empfit"], emp, 5e-7, f"{alias} empirical fit endpoint MAE")
    check(agg.loc["armorm", "mae_anu"], 6.37, 5e-3, "ArmoRM A_nu artifact")

    out = agg.reset_index().rename(columns={"mae_theory": "mae_ab"})
    out.insert(1, "radius", 0.1)
    out["n_dimensions"], out["n_splits_per_dimension"] = 4, 50
    disp.write_data(out)
    by_dim = end[["alias", "dimension", "realized_gain_mean", "mae_aonly", "split_sd_aonly", "mae_theory",
                  "split_sd_theory", "mae_empfit", "split_sd_empfit", "mae_anu", "mae_zero",
                  "cal_A_mean", "cal_B_mean"]].rename(columns={"mae_theory": "mae_ab", "split_sd_theory": "split_sd_ab"})
    by_dim = by_dim.assign(dim_order=by_dim.dimension.map(DIMS.index), rm_order=by_dim.alias.map(RM_ORDER.index))
    by_dim = by_dim.sort_values(["dim_order", "rm_order"]).drop(columns=["dim_order", "rm_order"]).reset_index(drop=True)
    disp.write_data(by_dim, suffix="_by_dimension")
    disp.write_data(summary.rename(columns={"mae_theory": "mae_ab", "split_sd_theory": "split_sd_ab"}),
                    suffix="_all_radii")

    rows = [[RM_LABEL[a], fmt(r.mae_aonly, 4), fmt(r.mae_theory, 4), fmt(r.mae_empfit, 4),
             fmt(r.reduction_vs_aonly, 1, percent=True)] for a, r in agg.iterrows()]
    disp.write_table("lrrrr", [["", "", "", "Calibration", r"$A{+}B$ reduction"],
                               ["Reward model", r"$A$-only", r"$A{+}B$", "empirical fit", r"vs $A$-only"]], rows)
    dim_rows = [[r.dimension if i % 3 == 0 else "", RM_LABEL[r.alias], fmt(r.realized_gain_mean, 4),
                 fmt(r.mae_aonly, 4), fmt(r.mae_ab, 4), fmt(r.mae_empfit, 4)]
                for i, r in enumerate(by_dim.itertuples())]
    disp.write_table("llrrrr", [["", "", "Mean held-out", "", "", "Calibration"],
                                ["Dimension", "Reward model", "gain", r"$A$-only", r"$A{+}B$", "empirical fit"]],
                     dim_rows, suffix="_by_dimension", midrules_after=[2, 5, 8])
    anu = ", ".join(f"{RM_LABEL[a]} {agg.loc[a, 'mae_anu']:.4f}" for a in RM_ORDER)
    lines = "\n".join(f"- {RM_LABEL[a]}: A-only {agg.loc[a, 'mae_aonly']:.6f}, A+B {agg.loc[a, 'mae_theory']:.6f}, "
                      f"empirical fit {agg.loc[a, 'mae_empfit']:.6f}; reduction {agg.loc[a, 'reduction_vs_aonly'] * 100:.1f}%"
                      for a in RM_ORDER)
    emp_wins = ", ".join(f"{RM_LABEL[r.alias]} {r.dimension}" for r in by_dim.itertuples() if r.mae_empfit < r.mae_ab)
    disp.write_note(f"""
# rm_endpoint_mae: exp37 endpoint MAE at d = 0.10

**Placement.** Appendix; main-eligible as a one-line or small-table support for the worked example
(CAMERA_READY_PLAN.md §8.4). `rm_endpoint_mae_by_dimension.tex` is the 12-row appendix variant.

**What it shows.** Prospective endpoint MAE at KL d = 0.10, for predictions made from 480 calibration
items and scored on the 1,120 held-out items, in human-score units of each dimension.

**Definitions.** A-only = A sqrt(d); A+B = A sqrt(d) + B d (moment coefficients from calibration items);
calibration empirical fit = the same two-term form least-squares fitted to the exact calibration-split
tilt curve. Target = exact raw-score exponential-tilt gain on evaluation items with human gold (mean of
three expert and five crowd ratings, per dimension). This is endpoint MAE, not CurveMAE.

**Aggregation.** Per model/dimension: mean absolute error over 50 item splits (seed 0, 480/1,120,
identical across models). Main table: unweighted mean over the four dimensions (= mean over 200 splits).
All 200 dimension/repeat splits are valid for all three models (600 model-splits, no exclusions). The
saved summary means are re-checked against the saved split rows (< 1e-12); nothing is re-analysed.

**Sources.** `results/reward_model_coefficients/canonical/analysis/summary.csv`
and `.../analysis/prospective_paths.csv`; `.../analysis/run_spec.json` (NumPy 2.3.5 environment).

**Key results.**
{lines}
- A+B < A-only in all 12 model/dimension endpoint cells, and in 48/48 cells including radii
  0.025/0.05/0.075/0.10.
- Calibration empirical fit < A+B in {n_emp_beats}/12 endpoint cells ({emp_wins}).

**Reporting details.** d = 0.10 endpoint MAE (not CurveMAE), equally averaged over four SummEval
dimensions and 50 item splits per dimension; 480 calibration / 1,120 evaluation items; item-held-out,
not article-held-out; shared truncated article context; human-score units; raw-score exponential tilt,
not RL training. Point estimates only.

**Caveats.**
- Not a significance claim and not a universal ranking; empirical fit remains competitive (wins 7/12).
  Split SDs (in `rm_endpoint_mae_by_dimension.csv`) overlap and are not independent-sample confidence
  intervals: splits reuse the same 1,600 items.
- A_nu is omitted from the tables. Its d = 0.10 MAE is in the CSV ({anu}); the ArmoRM value is a
  proxy-scale artifact (ArmoRM's scalar reward has SD 0.022, so the variance-ratio surrogate is not
  comparable), not a meaningful forecast. If shown, it needs that caveat.
- These shared-context Skywork scores are not the earlier open-proxy Skywork scoring run
  (`proxy_open_judge_rm`); the near-equal Skywork A+B values there (0.01126) are a coincidence of two
  different runs.
- The 4-decimal display ties A+B and empirical fit for ArmoRM/DeBERTa; differences are below 0.0001.
""")


def build_rm_coefficients(group: Group):
    disp = group.display(
        "rm_coefficients", "appendix",
        "Exp37 full-pool descriptive A and B for 3 RMs x 4 SummEval dimensions with 1,000-draw paired "
        "article-cluster bootstrap 95% intervals, plus the tail and skewness parts of B.")
    run_spec = rm_preamble(disp)
    coef = disp.read_csv(RM / "analysis" / "coefficients.csv")
    boot = disp.read_csv(RM / "analysis" / "coefficient_bootstrap.csv")
    existing = disp.input(RM / "paper_example" / "table_reward_model_coefficients.tex")
    require(len(coef) == 12 and (coef.n_items == 1600).all(), "12 pairs on 1,600 items")
    require(run_spec["n_boot"] == 1000 and run_spec["bootstrap_seed"] == 1, "1000 draws, seed 1")
    require((boot.groupby(["alias", "dimension"]).size() == 1000).all(), "1000 draws per pair")
    check_all(coef.B, coef.B_tail + coef.B_skewness, 1e-12, "B = tail + skewness")
    for r in coef.itertuples():  # intervals are percentiles of the saved draws
        b = boot[(boot.alias == r.alias) & (boot.dimension == r.dimension)]
        check_all(np.percentile(b.A, [2.5, 97.5]), [r.A_lo, r.A_hi], 1e-12, f"{r.alias}/{r.dimension} A interval")
        check_all(np.percentile(b.B, [2.5, 97.5]), [r.B_lo, r.B_hi], 1e-12, f"{r.alias}/{r.dimension} B interval")
    sky = coef[(coef.alias == "skywork") & (coef.dimension == "coherence")].iloc[0]
    check(sky.B, 0.00860, 5e-6, "Skywork coherence B")
    check(sky.B_lo, -0.04031, 5e-6, "Skywork coherence B low")
    check(sky.B_hi, 0.06233, 5e-6, "Skywork coherence B high")
    n_b_excl_zero = int(((coef.B_hi < 0) | (coef.B_lo > 0)).sum())
    require(n_b_excl_zero == 11, "11/12 B intervals exclude zero")

    rows = [[RM_LABEL[r.alias], r.dimension, f"{fmt(r.A, 4)} {fmt_ci(r.A_lo, r.A_hi, 4)}",
             f"{fmt(r.B, 4)} {fmt_ci(r.B_lo, r.B_hi, 4)}", fmt(r.B_tail, 4), fmt(r.B_skewness, 4)]
            for r in coef.itertuples()]
    require(plain_minus(rows) == tex_body_cells(existing), "regenerated coefficient body equals the exp37 table")
    disp.write_data(coef)
    disp.write_table("llrrrr", [["Reward model", "Human dimension", r"$A$ [95\% CI]", r"$B$ [95\% CI]",
                                 "Tail term", "Skewness term"]], rows, midrules_after=[2, 5, 8])
    disp.write_note(f"""
# rm_coefficients: exp37 full-pool descriptive coefficients

**Placement.** Appendix (CAMERA_READY_PLAN.md §8.4); referenced from the worked example.

**What it shows.** A and B for each of 12 RM/dimension pairs, estimated on all 1,600 SummEval items,
with 95% percentile intervals from 1,000 paired article-cluster bootstrap draws (seed 1; same article
draw for every model and dimension). B = tail term + skewness term (checked < 1e-12).

**Units.** A, B in human-score units of the dimension per sqrt(KL) and per KL respectively; invariant to
positive affine changes of the proxy score, not to target rescaling or population shift.

**Sources.** `results/reward_model_coefficients/canonical/analysis/coefficients.csv`;
intervals re-checked as percentiles of `.../analysis/coefficient_bootstrap.csv` (< 1e-12). The regenerated
body equals `.../paper_example/table_reward_model_coefficients.tex` cell for cell (only `$-$` minus signs differ).

**Key results.** Skywork coherence B = {sky.B:.5f} [{sky.B_lo:.5f}, {sky.B_hi:.5f}] (includes zero);
the other {n_b_excl_zero} B intervals exclude zero (all negative). Coherence A: Skywork {sky.A:.5f},
ArmoRM {coef.set_index(['alias', 'dimension']).loc[('armorm', 'coherence')].A:.5f}, DeBERTa
{coef.set_index(['alias', 'dimension']).loc[('deberta', 'coherence')].A:.5f}.

**Reporting details.** Full-pool descriptive estimates (all 1,600 items), not the calibration-only
predictors of the worked example; 1,000-draw paired article-cluster bootstrap, percentile 95% intervals;
same shared article context as the worked example.

**Caveats.** Do not describe Skywork coherence curvature as positive (its interval includes zero).
Marginal intervals are not a paired test of differences between reward models. Descriptive full-pool
values differ from the worked example's calibration-only values because the estimation sample differs.
These A/B characterize proxy, target, candidate distribution and context, not intrinsic RM quality.
""")


def build_rm_figure(group: Group, worked: pd.DataFrame):
    disp = group.display(
        "rm_fig_worked_example", "appendix (worked example; the main text states it in prose)",
        "Exp37 worked example as three panels: held-out exact coherence gain vs calibration-only A+B and "
        "A-only predictions over d in [0, 0.10], regenerated in the publication style.")
    curves = disp.read_csv(RM / "paper_example" / "worked_example_curves.csv")
    original = disp.input(RM / "paper_example" / "fig_reward_model_worked_example.pdf")
    require(curves.groupby("alias").size().loc[RM_ORDER].tolist() == [201] * 3, "201 grid points per model")
    w = worked.set_index("alias")
    end = curves.groupby("alias").tail(1).set_index("alias").loc[RM_ORDER]
    check_all(end.d, [0.1] * 3, 1e-15, "curves end at d=0.10")
    check_all(end.ab, w.predicted_gain.loc[RM_ORDER], 1e-12, "A+B endpoint = predicted gain")
    check_all(end.aonly, w.first_order.loc[RM_ORDER], 1e-12, "A-only endpoint = A sqrt(.10)")
    check_all(end.exact_gain, w.heldout_gain.loc[RM_ORDER], 1e-12, "exact endpoint = held-out gain")
    for alias in RM_ORDER:
        sub = curves[curves.alias == alias]
        a, b = w.loc[alias, "A_cal"], w.loc[alias, "B_cal"]
        dd = sub.d.clip(lower=0)
        check_all(sub.ab, a * np.sqrt(dd) + b * dd, 1e-12, f"{alias} A+B curve")

    figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.05), sharex=True, sharey=True)
    for ax, alias in zip(axes, RM_ORDER):
        sub = curves[curves.alias == alias].sort_values("d")
        ax.plot(sub.d, sub.exact_gain, color="#222222", lw=1.6, label="Held-out exact tilt", zorder=3)
        ax.plot(sub.d, sub.ab, color=COLORS["A+B"], lw=1.4, label=r"$A{+}B$ prediction", zorder=4)
        ax.plot(sub.d, sub.aonly, color=COLORS["A-only"], lw=1.4, ls=(0, (4, 2)), label=r"$A$-only", zorder=5)
        a, b = w.loc[alias, "A_cal"], w.loc[alias, "B_cal"]
        bs = f"{b:.3f}".replace("-", "{-}")
        ax.set_title(f"{RM_LABEL[alias]}\n" + rf"$\widehat A={a:.3f},\ \widehat B={bs}$", pad=3)
        ax.set_xlim(0, 0.1)
        ax.set_xticks([0, 0.05, 0.1], ["0", "0.05", "0.10"])
        ax.set_xlabel(r"KL radius $d$")
        ax.grid(axis="y", color="#E0E0E0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Gain in human coherence")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.8)
    disp.write_data(curves)
    disp.save_figure(fig)
    disp.write_note(f"""
# rm_fig_worked_example: three-panel worked-example figure (exp37)

**Placement.** Appendix, beside `rm_worked_example`. Following CAMERA_READY_PLAN.md §8.1, the main
text states the three decompositions in prose (Section 5.1) and points here; Figure 1 is a schematic
(`fig1_workflow`).

**What it shows.** For the preselected SummEval coherence split (repeat 0; 480 calibration / 1,120
evaluation items), the exact held-out raw-score exponential-tilt gain (black) and the calibration-only
A+B (solid blue) and A-only (dashed orange) predictions for d in [0, 0.10]. Panel titles give the
calibration A and B (3 decimals). The right endpoints equal the `rm_worked_example` values (< 1e-12).

**Units.** x: KL radius d of the tilt on the evaluation items. y: gain in human coherence score (mean
of eight raters).

**Regeneration.** Redrawn at {TEXT_WIDTH_IN} in width with `common.figure_style()` and `COLORS` from
`.../paper_example/worked_example_curves.csv` (201 grid points per model; A+B curve re-checked against
A sqrt(d) + B d). No numbers were recomputed. The original exp37 figure
(`{rel(original)}`, 10.6 in wide, different palette) remains available. A-only is dashed and drawn on top,
so it stays visible where it coincides with A+B (Skywork) and in grayscale / colour-vision deficiency.

**Reporting details.** Calibration-only A/B on the preselected coherence / repeat-0 split (chosen
before inspecting results); black = exact raw-score exponential-tilt gain on the 1,120 evaluation items
with human gold; item-held-out, not article-held-out; shared, truncated article context across models;
illustrative split with aggregate 50-repeat errors in `rm_endpoint_mae` and full-pool intervals in
`rm_coefficients`. For DeBERTa the A-only curve is closer at d = 0.10 on this split.

**Caveats.** Same as `rm_worked_example`: not DPO/RLHF, not an RM ranking, negative B is local
curvature only.
""")


# ================================================================ open proxies
def path_bootstrap(paths: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Path-clustered bootstrap draws exactly as experiments/prospective_prediction_path_macro (seed 0)."""
    err_cols = [c for c in paths.columns if c.startswith("abs_err_")] + ["theory_sign_correct"]
    by_path = paths.groupby("path_id")[err_cols].mean()
    rng = np.random.default_rng(seed)
    mat = by_path.to_numpy()
    draws = np.empty((n_boot, mat.shape[1]))
    for b in range(n_boot):
        draws[b] = mat[rng.integers(0, len(mat), size=len(mat))].mean(axis=0)
    return pd.DataFrame(draws, columns=by_path.columns)


def build_proxy(group: Group):
    disp = group.display(
        "proxy_open_judge_rm", "appendix (short external-proxy summary sentence in main)",
        "SummEval open proxies (Skywork-Reward-Llama-3.1-8B-v0.2 RM; Qwen2.5-32B-Instruct 5-pass judge): "
        "d=0.10 endpoint MAE of A+B, A-only, A_nu, empirical fit and zero, with a path-clustered paired interval.")
    readme = disp.input(PROXY_INPUTS / "README.md").read_text()
    listed = dict(re.findall(r"\| `([^`]+\.csv)` \|[^|]*\| `([0-9a-f]{64})` \|", readme))
    proxies = {
        "skywork_rm": dict(dir="proxy_rm_skywork", spec="proxy_rm_skywork_v02.run_spec.json",
                           csv="proxy_rm_skywork_v02.csv", label="Skywork RM v0.2", kind="reward model"),
        "qwen_judge": dict(dir="proxy_openjudge", spec="model_proxy_openjudge_aggregated.run_spec.json",
                           csv="model_proxy_openjudge_aggregated.csv", label="Qwen2.5-32B judge", kind="LLM judge"),
    }
    audited = {"skywork_rm": dict(ab=0.01126, aonly=0.01545, anu=0.04376, empfit=0.01113, zero=0.06646,
                                  red=0.271, win_aonly=0.66, win_empfit=0.485,
                                  model="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", sha="d4117fbf"),
               "qwen_judge": dict(ab=0.01120, aonly=0.01485, anu=0.09636, empfit=0.01099, zero=0.08167,
                                  red=0.245, win_aonly=0.605, win_empfit=0.475,
                                  model="Qwen/Qwen2.5-32B-Instruct", sha="5ede1c97")}
    main_rows, dim_rows, boot_frames, meta = [], [], [], {}
    for key, p in proxies.items():
        summary = disp.read_csv(PROXY / p["dir"] / "exp27_prospective_summary.csv").set_index("group_key")
        paths = disp.read_csv(PROXY / p["dir"] / "exp27_prospective_paths.csv")
        exp_spec = json.loads(disp.input(PROXY / p["dir"] / "exp27_run_spec.json").read_text())
        score_spec = json.loads(disp.input(PROXY_INPUTS / "scoring_run_specs" / p["spec"]).read_text())
        require(listed[p["csv"]] == sha256(disp.input(PROXY_INPUTS / p["csv"])), f"{p['csv']} hash = README")
        a = audited[key]
        require(score_spec["model"] == a["model"] and score_spec["model_resolved_sha"].startswith(a["sha"]),
                f"{key} model id / commit")
        require(len(score_spec["model_resolved_sha"]) == 40, f"{key} full commit")
        require(score_spec["items_csv_sha256"].startswith("4a29f598"), f"{key} repaired SummEval item CSV")
        if key == "qwen_judge":
            require(score_spec["n_judges"] == 5 and score_spec["temperature"] == 0.3, "5 judge passes, T=0.3")
        require(exp_spec["n_repeats"] == 50 and exp_spec["calibration_frac"] == 0.3 and exp_spec["random_seed"] == 0
                and exp_spec["n_skipped_splits"] == 0 and exp_spec["n_rows"] == 200, f"{key} protocol")
        require(len(paths) == 200 and paths.path_id.nunique() == 4, f"{key} 200 split rows on 4 paths")
        require((paths.n_cal == 480).all() and (paths.n_eval == 1120).all(), f"{key} 480/1120")

        ov = summary.loc["__OVERALL__"]
        names = ["theory", "aonly", "anu", "empfit", "zero"]
        for n in names:  # saved summary = mean of saved split rows
            check(ov[f"mae_{n}_0p1"], paths[f"abs_err_{n}_0p1"].mean(), 1e-12, f"{key} {n} overall mean")
            per_dim = paths.groupby("group_key")[f"abs_err_{n}_0p1"].mean()
            check_all(summary.loc[DIMS, f"mae_{n}_0p1"], per_dim.loc[DIMS], 1e-12, f"{key} {n} per-dimension")
        for n, label in zip(names, ["ab", "aonly", "anu", "empfit", "zero"]):
            check(ov[f"mae_{n}_0p1"], a[label], 5e-6, f"{key} {label} MAE")
        reduction = 1 - ov.mae_theory_0p1 / ov.mae_aonly_0p1
        check(reduction, a["red"], 5e-4, f"{key} reduction")
        win_aonly = float((paths.abs_err_theory_0p1 < paths.abs_err_aonly_0p1).mean())
        win_empfit = float((paths.abs_err_theory_0p1 < paths.abs_err_empfit_0p1).mean())
        check(win_aonly, a["win_aonly"], 1e-12, f"{key} A+B beats A-only share")
        check(win_empfit, a["win_empfit"], 1e-12, f"{key} A+B beats empirical-fit share")

        # Path-clustered paired interval: reproduce the engine's saved intervals, then reuse its draws.
        draws = path_bootstrap(paths, int(exp_spec["n_boot"]))
        for n in names:
            check_all(np.percentile(draws[f"abs_err_{n}_0p1"], [2.5, 97.5]),
                      [ov[f"abs_err_{n}_0p1_ci_lo"], ov[f"abs_err_{n}_0p1_ci_hi"]], 1e-12, f"{key} {n} saved CI")
        check_all(np.percentile(draws.abs_err_anu_0p1 - draws.abs_err_theory_0p1, [2.5, 97.5]),
                  [ov.mae_diff_anu_minus_theory_0p1_ci_lo, ov.mae_diff_anu_minus_theory_0p1_ci_hi], 1e-12,
                  f"{key} saved A_nu-A+B CI")
        d_aonly = draws.abs_err_aonly_0p1 - draws.abs_err_theory_0p1
        d_emp = draws.abs_err_empfit_0p1 - draws.abs_err_theory_0p1
        ci_aonly, ci_emp = np.percentile(d_aonly, [2.5, 97.5]), np.percentile(d_emp, [2.5, 97.5])
        boot_frames.append(pd.DataFrame({"proxy": key, "draw": np.arange(len(draws)),
                                         "mae_ab": draws.abs_err_theory_0p1, "mae_aonly": draws.abs_err_aonly_0p1,
                                         "mae_empfit": draws.abs_err_empfit_0p1,
                                         "aonly_minus_ab": d_aonly, "empfit_minus_ab": d_emp}))
        meta[key] = dict(model=score_spec["model"], commit=score_spec["model_resolved_sha"], spec=score_spec,
                         n_boot=int(exp_spec["n_boot"]), ci_aonly=ci_aonly, ci_emp=ci_emp,
                         win_aonly=win_aonly, win_empfit=win_empfit, reduction=reduction)
        main_rows.append(dict(proxy=key, label=p["label"], kind=p["kind"], model_id=score_spec["model"],
                              resolved_commit=score_spec["model_resolved_sha"], radius=0.1, n_splits=200,
                              mae_ab=ov.mae_theory_0p1, mae_aonly=ov.mae_aonly_0p1, mae_empfit=ov.mae_empfit_0p1,
                              mae_anu=ov.mae_anu_0p1, mae_zero=ov.mae_zero_0p1, reduction_vs_aonly=reduction,
                              share_ab_beats_aonly=win_aonly, share_ab_beats_empfit=win_empfit,
                              aonly_minus_ab=ov.mae_aonly_0p1 - ov.mae_theory_0p1,
                              aonly_minus_ab_ci_lo=ci_aonly[0], aonly_minus_ab_ci_hi=ci_aonly[1],
                              empfit_minus_ab=ov.mae_empfit_0p1 - ov.mae_theory_0p1,
                              empfit_minus_ab_ci_lo=ci_emp[0], empfit_minus_ab_ci_hi=ci_emp[1],
                              bootstrap="path-clustered (4 dimension paths), 500 draws, seed 0; computed here"))
        for dim in DIMS:
            r = summary.loc[dim]
            sub = paths[paths.group_key == dim]
            dim_rows.append(dict(proxy=key, label=p["label"], dimension=dim, n_splits=int(r.n_splits),
                                 mae_ab=r.mae_theory_0p1, mae_aonly=r.mae_aonly_0p1, mae_empfit=r.mae_empfit_0p1,
                                 mae_anu=r.mae_anu_0p1, mae_zero=r.mae_zero_0p1,
                                 share_ab_beats_aonly=float((sub.abs_err_theory_0p1 < sub.abs_err_aonly_0p1).mean()),
                                 share_ab_beats_empfit=float((sub.abs_err_theory_0p1 < sub.abs_err_empfit_0p1).mean())))
    main = pd.DataFrame(main_rows)
    dims = pd.DataFrame(dim_rows)
    qcoh = dims[(dims.proxy == "qwen_judge") & (dims.dimension == "coherence")].iloc[0]
    require(qcoh.mae_aonly < qcoh.mae_ab, "Qwen coherence: A-only beats A+B")
    require(int((dims.mae_aonly < dims.mae_ab).sum()) == 1, "only Qwen coherence favors A-only")

    disp.write_data(main)
    disp.write_data(dims, suffix="_by_dimension")
    disp.write_data(pd.concat(boot_frames, ignore_index=True), suffix="_bootstrap")
    rows = [[r.label, fmt(r.mae_aonly, 4), fmt(r.mae_ab, 4), fmt(r.mae_empfit, 4), fmt(r.mae_anu, 4),
             fmt(r.mae_zero, 4), fmt(r.reduction_vs_aonly, 1, percent=True),
             f"{fmt(r.aonly_minus_ab, 5)} {fmt_ci(r.aonly_minus_ab_ci_lo, r.aonly_minus_ab_ci_hi, 5)}"]
            for r in main.itertuples()]
    disp.write_table("lrrrrrrc", [["", "", "", "Emp.", "", "", r"$A{+}B$", r"$A$-only $-$ $A{+}B$"],
                                  ["Proxy", r"$A$-only", r"$A{+}B$", "fit", r"$A_\nu$", "Zero", "reduction",
                                   r"[path-bootstrap 95\%]"]], rows)
    drows = [[r.label if i % 4 == 0 else "", r.dimension, fmt(r.mae_aonly, 4), fmt(r.mae_ab, 4),
              fmt(r.mae_empfit, 4), fmt(r.mae_anu, 4), fmt(r.mae_zero, 4)] for i, r in enumerate(dims.itertuples())]
    disp.write_table("llrrrrr", [["Proxy", "Dimension", r"$A$-only", r"$A{+}B$", "Emp. fit", r"$A_\nu$", "Zero"]],
                     drows, suffix="_by_dimension", midrules_after=[3])

    s, q = meta["skywork_rm"], meta["qwen_judge"]
    ms, mq = main.set_index("proxy").loc["skywork_rm"], main.set_index("proxy").loc["qwen_judge"]
    require(q["ci_emp"][1] < 0, "Qwen empirical fit - A+B interval below zero")
    exp37_models = json.loads(disp.input(RM / "prepared" / "models.json").read_text())
    require(exp37_models["skywork"]["revision"] == s["commit"], "exp37 Skywork uses the same model revision")
    disp.write_note(f"""
# proxy_open_judge_rm: SummEval open proxies at d = 0.10

**Placement.** Appendix table; one short sentence in main next to the GPT-4o proxy result
(CAMERA_READY_PLAN.md §3, "Learned RM / open judge on SummEval"). `proxy_open_judge_rm_by_dimension.tex`
is the per-dimension variant.

**What it shows.** Prospective endpoint MAE at KL d = 0.10 when an open reward model or an open LLM judge
replaces GPT-4o as the SummEval proxy, using the paper's original prospective protocol.

**Proxies (scoring run specs).**
- Skywork RM: `{s['model']}`, resolved commit `{s['commit']}`, bfloat16, max length
  {s['spec']['max_length']}, batch {s['spec']['batch_size']}.
- Qwen judge: `{q['model']}`, resolved commit `{q['commit']}`, {q['spec']['n_judges']} judge passes
  averaged, temperature {q['spec']['temperature']}, max {q['spec']['max_tokens']} tokens, prompt
  `{q['spec']['prompt_version']}`.
- Both read the repaired SummEval item CSV (`{q['spec']['items_csv_sha256'][:8]}...`). Proxy CSV hashes
  match `data/open_proxy_inputs/README.md`.

**Definitions and protocol.** Same as the original SummEval prospective evaluation: per dimension, 30%
calibration (480 items) / 70% evaluation (1,120), 50 repeats, seed 0; A+B = A sqrt(d) + B d and A-only =
A sqrt(d) from calibration moments; A_nu = reliability-based variance-ratio surrogate; empirical fit =
two-term form fitted to the calibration-split exact tilt curve (fit up to d = 0.10); zero = 0. Target =
exact raw-score exponential-tilt gain on evaluation items with human gold, human-score units. MAE = mean
over 4 dimensions x 50 splits (200 rows, none skipped).

**Paired interval (computed here, not saved by the experiment).** The experiment saved only marginal
path-bootstrap intervals and an A_nu - A+B interval. This builder reproduces the engine's
path-clustered bootstrap (`experiments/prospective_prediction_path_macro/01_prospective_prediction.py`: resample the
4 dimension paths, {s['n_boot']} draws, `default_rng(0)`), checks that it reproduces every saved interval
exactly (< 1e-12), and then reports the paired A-only - A+B and empirical fit - A+B intervals from the same
draws. With only 4 clusters the interval is coarse (at most 35 distinct resamples) and reflects
between-dimension variation only.

**Sources.** `results/open_proxy/{{proxy_rm_skywork,proxy_openjudge}}/exp27_prospective_summary.csv`,
`.../exp27_prospective_paths.csv`, `.../exp27_run_spec.json`;
`data/open_proxy_inputs/scoring_run_specs/*.run_spec.json`; proxy CSVs (hash check only).

**Key results (d = 0.10).**
- Skywork RM: A+B {ms.mae_ab:.5f}, A-only {ms.mae_aonly:.5f}, A_nu {ms.mae_anu:.5f}, empirical fit
  {ms.mae_empfit:.5f}, zero {ms.mae_zero:.5f}; A+B reduction vs A-only {ms.reduction_vs_aonly * 100:.1f}%;
  A-only - A+B {ms.aonly_minus_ab:.5f} [{s['ci_aonly'][0]:.5f}, {s['ci_aonly'][1]:.5f}]; empirical fit - A+B
  {ms.empfit_minus_ab:.5f} [{s['ci_emp'][0]:.5f}, {s['ci_emp'][1]:.5f}].
- Qwen judge: A+B {mq.mae_ab:.5f}, A-only {mq.mae_aonly:.5f}, A_nu {mq.mae_anu:.5f}, empirical fit
  {mq.mae_empfit:.5f}, zero {mq.mae_zero:.5f}; reduction {mq.reduction_vs_aonly * 100:.1f}%; A-only - A+B
  {mq.aonly_minus_ab:.5f} [{q['ci_aonly'][0]:.5f}, {q['ci_aonly'][1]:.5f}]; empirical fit - A+B
  {mq.empfit_minus_ab:.5f} [{q['ci_emp'][0]:.5f}, {q['ci_emp'][1]:.5f}].
- Split-level shares: A+B beats A-only in {s['win_aonly'] * 100:.1f}% / {q['win_aonly'] * 100:.1f}% of the 200
  splits (Skywork / Qwen) and beats empirical fit in {s['win_empfit'] * 100:.1f}% / {q['win_empfit'] * 100:.1f}%.

**Reporting details.** SummEval, d = 0.10 endpoint MAE averaged over four dimensions x 50 item splits
(480/1,120); model IDs; the Qwen judge averages five passes at T = 0.3; the paired interval is a
path-clustered bootstrap over only four dimension paths, computed for this paper from the saved split rows.

**Caveats.**
- Qwen coherence: A-only beats A+B ({qcoh.mae_aonly:.5f} vs {qcoh.mae_ab:.5f}); the only one of the
  8 proxy/dimension cells to do so.
- Empirical fit has slightly lower MAE than A+B for both proxies and wins about half the splits; A+B is
  not better than empirical fit here. For Qwen the path-bootstrap interval of empirical fit - A+B
  ([{q['ci_emp'][0]:.5f}, {q['ci_emp'][1]:.5f}]) lies entirely below zero.
- A+B beats A-only in only {s['win_aonly'] * 100:.1f}%/{q['win_aonly'] * 100:.1f}% of individual splits:
  the aggregate reduction is not split-by-split dominance.
- These Skywork scores come from the original open-proxy scoring run, NOT the exp37 shared-context
  Skywork scores: same model revision (`{s['commit'][:8]}`), but exp37 re-scored with shared, further
  truncated article prefixes common to all three RMs; do not mix with `rm_endpoint_mae`. The near-identical
  Skywork A+B MAE in both (0.01126 vs 0.011262) is coincidental.
- Training-data overlap of these models with SummEval is not excluded.
""")


# ================================================================ main
def main() -> None:
    group = CombinedGroup("dpo", ("rm", "proxy"))
    keys, _ = build_dpo_table3(group)
    build_dpo_comparators(group, keys)
    worked = build_rm_worked_example(group)
    build_rm_endpoint_mae(group)
    build_rm_coefficients(group)
    build_rm_figure(group, worked)
    build_proxy(group)
    path = group.write_manifest()
    print(f"{len(group.displays)} displays -> {rel(path)}")
    for name, record in group.displays.items():
        print(f"  {name}: {len(record['inputs'])} inputs, {len(record['outputs'])} outputs")


if __name__ == "__main__":
    main()
