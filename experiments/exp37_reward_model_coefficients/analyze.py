"""Compare A/B and held-out gain predictions on exactly matched three-RM support."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (COMPONENT, DIMENSIONS, HERE, environment, load_models, new_directory,
                    read_csv, sha256, validate_inputs, validate_scores, verify_prepared, write_json)
from prepare import make_splits

sys.path.insert(0, str(COMPONENT / "lib"))
from common_experiment_utils import PathData, theory_coefficients

ENGINE = COMPONENT / "experiments/prospective_prediction_path_macro/01_prospective_prediction.py"
spec = importlib.util.spec_from_file_location("exp37_prospective_engine", ENGINE)
prospective = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prospective)
PREDICTORS = ["theory", "aonly", "empfit", "anu", "zero"]


def describe(j, r):
    j, r = np.asarray(j, float), np.asarray(r, float)
    if not np.isfinite(j).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite moment input")
    c = theory_coefficients(j, r)
    tail = c.kappa_jrr / c.sigma_r2
    correction = c.b_theory - tail
    return dict(A=c.a_exact, B=c.b_theory, B_tail=tail, B_skewness=correction,
                proxy_std=np.sqrt(c.sigma_r2), gold_std=np.sqrt(c.sigma_j2))


def bootstrap_coefficients(j, proxies, articles, n_boot, seed):
    groups = [np.flatnonzero(articles == x) for x in np.unique(articles)]
    if len(groups) < 2 or n_boot < 1:
        raise ValueError("Need >=2 articles and positive bootstrap draws")
    rng = np.random.default_rng(seed)
    rows, valid, attempts = [], 0, 0
    while valid < n_boot and attempts < 100 * n_boot:
        attempts += 1
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        try:
            values = {m: describe(j[idx], r[idx]) for m, r in proxies.items()}
        except ValueError:
            continue
        if not all(np.isfinite(list(v.values())).all() for v in values.values()):
            continue
        rows.extend(dict(alias=m, draw=valid, **v) for m, v in values.items())
        valid += 1
    if valid != n_boot:
        raise RuntimeError(f"Only {valid}/{n_boot} valid paired article-bootstrap draws")
    return pd.DataFrame(rows), attempts


def evaluate_matched(paths, splits, radii):
    rows, accounting = [], []
    for split in splits:
        dim, repeat = split["dimension"], split["repeat"]
        fitted, reasons = {}, {}
        for alias, by_dimension in paths.items():
            path = by_dimension[dim]
            positions = {key: i for i, key in enumerate(path.item_ids)}
            cal = np.array([positions[k] for k in split["calibration"]])
            eva = np.array([positions[k] for k in split["evaluation"]])
            try:
                result = prospective.evaluate_one_split(path, cal, eva, max(radii), radii)
                for radius in radii:
                    suffix = str(radius).replace(".", "p")
                    cols = [f"actual_eval_delta_{suffix}"] + [f"abs_err_{p}_{suffix}" for p in PREDICTORS]
                    if not all(np.isfinite(result.get(col, np.nan)) for col in cols):
                        raise ValueError("Unreachable radius or nonfinite prediction/error")
                fitted[alias] = dict(alias=alias, dimension=dim, repeat=repeat, **result)
            except (ValueError, RuntimeError) as exc:
                reasons[alias] = str(exc)
        keep = not reasons
        accounting.append(dict(dimension=dim, repeat=repeat, kept_for_all_models=keep, reasons=reasons))
        if keep:
            rows.extend(fitted.values())
    return pd.DataFrame(rows), accounting


def summarize(paths, radii):
    rows = []
    for (alias, dimension), df in paths.groupby(["alias", "dimension"], sort=True):
        for radius in radii:
            suffix = str(radius).replace(".", "p")
            row = dict(alias=alias, dimension=dimension, radius=radius, n_common_splits=len(df),
                       cal_A_mean=df.cal_a_exact.mean(), cal_B_mean=df.cal_b_theory.mean(),
                       moment_B_sign_agreement=float((np.sign(df.cal_b_theory) == np.sign(df.eval_b_theory)).mean()),
                       curve_fit_B_sign_agreement=df.theory_sign_correct.mean(),
                       realized_gain_mean=df[f"actual_eval_delta_{suffix}"].mean())
            for name in PREDICTORS:
                errs = df[f"abs_err_{name}_{suffix}"]
                row[f"mae_{name}"] = errs.mean()
                row[f"split_sd_{name}"] = errs.std(ddof=1) if len(errs) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def write_table(out, coeff, summary, endpoint):
    text = ["# Reward-Model Coefficients", "",
            "Full-pool coefficients are descriptive. Intervals are paired article-cluster bootstrap 95% intervals.",
            "Held-out MAE uses calibration-only coefficients and common item-level splits across models.",
            "", "| Model | Human dimension | A [95% CI] | B [95% CI] | B tail | B skewness | A+B MAE | A-only MAE | Empirical-fit MAE | Valid splits |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in coeff.itertuples():
        m = summary[(summary.alias == row.alias) & (summary.dimension == row.dimension) &
                    np.isclose(summary.radius, endpoint)].iloc[0]
        text.append(f"| {row.alias} | {row.dimension} | {row.A:.5f} [{row.A_lo:.5f}, {row.A_hi:.5f}] | "
                    f"{row.B:.5f} [{row.B_lo:.5f}, {row.B_hi:.5f}] | {row.B_tail:.5f} | "
                    f"{row.B_skewness:.5f} | {m.mae_theory:.5f} | {m.mae_aonly:.5f} | "
                    f"{m.mae_empfit:.5f} | {int(m.n_common_splits)} |")
    text += ["", f"Endpoint radius: d={endpoint:g}. Full radius/baseline results are in summary.csv.",
             "The split SD is descriptive variability over overlapping splits, not an independent-sample CI.",
             "Coefficients depend on proxy, human target, item distribution and text-context policy."]
    (out / "comparison.md").write_text("\n".join(text) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--scores-root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--radii", default="0.025,0.05,0.075,0.10")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=1)
    args = p.parse_args()
    radii = [float(x) for x in args.radii.split(",")]
    if not radii or not np.isfinite(radii).all() or sorted(set(radii)) != radii or radii[0] <= 0 or radii[-1] > .1:
        raise ValueError("Use unique increasing positive radii within d <= 0.10")
    if args.n_boot < 1 or args.outdir.exists():
        raise ValueError("Positive bootstrap count and a new output directory required")
    manifest = verify_prepared(args.prepared)
    items, human = validate_inputs(read_csv(args.prepared / "items.csv"), read_csv(args.prepared / "human.csv"))
    models = load_models(args.prepared / "models.json")
    splits = [json.loads(line) for line in (args.prepared / "splits.jsonl").read_text().splitlines()]
    expected = make_splits(items.item_id.tolist(), manifest["repeats"], manifest["calibration_frac"], manifest["seed"])
    if splits != expected:
        raise ValueError("Split IDs do not reproduce from the recorded seed/configuration")
    proxies, source_hashes = {}, {}
    for alias, model in models.items():
        folder = args.scores_root / alias
        record = json.loads((folder / "run_spec.json").read_text())
        done = json.loads((folder / "complete.json").read_text())
        if (record["alias"] != alias or record["model"] != model or
                record["prepared_manifest_sha256"] != sha256(args.prepared / "manifest.json") or
                done["scores_sha256"] != sha256(folder / "scores.csv") or
                done["spec_sha256"] != sha256(folder / "run_spec.json") or done["n_items"] != len(items)):
            raise ValueError(f"Score provenance mismatch: {alias}")
        proxies[alias] = validate_scores(read_csv(folder / "scores.csv"), items)
        source_hashes[alias] = dict(scores=sha256(folder / "scores.csv"), spec=sha256(folder / "run_spec.json"))
    paths = {alias: {} for alias in models}
    descriptors, bootstraps, bootstrap_attempts = [], [], {}
    for dim in DIMENSIONS:
        target = human[human.dimension == dim].groupby("item_id").score.mean().reindex(items.item_id).to_numpy()
        boot, attempts = bootstrap_coefficients(target, proxies, items.article_id.to_numpy(), args.n_boot, args.bootstrap_seed)
        bootstrap_attempts[dim] = attempts
        bootstraps.append(boot.assign(dimension=dim))
        for alias, r in proxies.items():
            paths[alias][dim] = PathData(dim, f"{dim}__model_proxy", items.item_id.to_numpy(), r, target,
                                        {"proxy_source": "model_proxy"})
            desc = dict(alias=alias, dimension=dim, n_items=len(items), **describe(target, r))
            for name in ["A", "B"]:
                desc[f"{name}_lo"], desc[f"{name}_hi"] = np.quantile(boot.loc[boot.alias == alias, name], [.025, .975])
            descriptors.append(desc)
    out = new_directory(args.outdir)
    coeff = pd.DataFrame(descriptors)
    coeff.to_csv(out / "coefficients.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(out / "coefficient_bootstrap.csv", index=False)
    prediction, accounting = evaluate_matched(paths, splits, radii)
    write_json(out / "split_accounting.json", accounting)
    if prediction.empty or set(prediction.dimension) != set(DIMENSIONS):
        raise RuntimeError("A dimension has no complete common support; see split_accounting.json")
    prediction.to_csv(out / "prospective_paths.csv", index=False)
    summary = summarize(prediction, radii)
    summary.to_csv(out / "summary.csv", index=False)
    write_table(out, coeff, summary, radii[-1])
    write_json(out / "run_spec.json", dict(
        prepared_manifest_sha256=sha256(args.prepared / "manifest.json"), scores=source_hashes,
        radii=radii, repeats=manifest["repeats"], n_boot=args.n_boot, bootstrap_seed=args.bootstrap_seed,
        bootstrap_unit="article_id; same draw for every model and dimension", bootstrap_attempts=bootstrap_attempts,
        split_unit="item", gold="Mean of all eight human raters, separately per dimension",
        information="Calibration-only predictions; evaluation gold is retrospective scoring only",
        shared_support="Keep a split only if all models have finite predictions at every radius",
        engine_sha256=sha256(ENGINE), math_sha256=sha256(COMPONENT / "lib/common_experiment_utils.py"),
        code_sha256={n: sha256(HERE / n) for n in ["analyze.py", "prepare.py", "common.py"]}, env=environment()))
    print(out)


if __name__ == "__main__":
    main()
