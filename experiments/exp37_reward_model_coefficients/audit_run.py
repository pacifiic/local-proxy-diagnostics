"""Independent score/key/moment/endpoint audit of a completed three-RM run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from common import (DIMENSIONS, read_csv, sha256, validate_inputs, validate_scores,
                    verify_aligned_source, verify_prepared, write_json)


def independent_ab(j, r):
    z = (r - r.mean()) / np.sqrt(np.mean((r - r.mean()) ** 2))
    centered = j - j.mean()
    alignment = np.mean(centered * z)
    return np.sqrt(2) * alignment, np.mean(centered * z ** 2) - (2 / 3) * alignment * np.mean(z ** 3)


def independent_gain(j, r, radius=.1):
    z = (r - r.mean()) / r.std()

    def distribution(s):
        logs = s * z - logsumexp(s * z)
        weights = np.exp(logs)
        return weights, float(weights @ (logs + np.log(len(r))))

    hi = 1.
    while distribution(hi)[1] < radius:
        hi *= 2
        if hi > 1e6:
            raise ValueError("Unreachable audit radius")
    s = brentq(lambda x: distribution(x)[1] - radius, 0, hi, xtol=1e-13)
    return float(distribution(s)[0] @ j - j.mean())


def independent_grid_gains(j, r, radii, n_grid=200):
    # Stored non-endpoint gains interpolate a 200-point temperature grid.
    z = (r - r.mean()) / r.std()
    def radius_at(s):
        logs = s * z - logsumexp(s * z)
        return float(np.exp(logs) @ (logs + np.log(len(r))))
    hi = 1.0
    while radius_at(hi) < max(radii):
        hi *= 2
        if hi > 1e6:
            raise ValueError("Unreachable audit radius")
    end = brentq(lambda x: radius_at(x) - max(radii), 0, hi, xtol=1e-13)
    logits = np.linspace(0, end, n_grid)[:, None] * z[None, :]
    logs = logits - logsumexp(logits, axis=1, keepdims=True)
    weights = np.exp(logs)
    kl = (weights * (logs + np.log(len(r)))).sum(axis=1)
    gain = weights @ j - j.mean()
    return dict(zip(radii, np.interp(radii, kl, gain)))


def validate_result_frames(coeff, predictions, summary, models, splits, radii, accounting):
    def complete(frame, keys, expected, label):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"Missing or duplicate {label} key")
        if set(frame[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Incomplete {label} support")

    def finite(frame, columns, label):
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {label}")

    complete(coeff, ["alias", "dimension"],
             {(a, d) for a in models for d in DIMENSIONS}, "coefficient")
    finite(coeff, ["A", "B", "A_lo", "A_hi", "B_lo", "B_hi", "B_tail", "B_skewness"], "coefficient")
    records = pd.DataFrame(accounting)
    complete(records, ["dimension", "repeat"], set(splits), "split accounting")
    kept = set()
    for row in accounting:
        if not isinstance(row["kept_for_all_models"], bool):
            raise ValueError("Invalid kept-split flag")
        if row["kept_for_all_models"]:
            if row["reasons"]:
                raise ValueError("Kept split has failure reasons")
            kept.add((row["dimension"], row["repeat"]))
        elif not row["reasons"] or not set(row["reasons"]).issubset(models):
            raise ValueError("Skipped split lacks a model failure reason")
    complete(predictions, ["alias", "dimension", "repeat"],
             {(a, d, r) for a in models for d, r in kept}, "prediction")
    prediction_cols = ["cal_a_exact", "cal_b_theory"]
    for radius in radii:
        suffix = str(float(radius)).replace(".", "p")
        prediction_cols += [f"actual_eval_delta_{suffix}", f"pred_theory_delta_{suffix}"]
        prediction_cols += [f"abs_err_{name}_{suffix}" for name in ["theory", "aonly", "empfit", "anu", "zero"]]
    finite(predictions, prediction_cols, "prediction")
    complete(summary, ["alias", "dimension", "radius"],
             {(a, d, float(r)) for a in models for d in DIMENSIONS for r in radii}, "summary")
    finite(summary, ["n_common_splits", "cal_A_mean", "cal_B_mean", "realized_gain_mean"] +
           [f"mae_{name}" for name in ["theory", "aonly", "empfit", "anu", "zero"]], "summary")


def audit(run, aligned):
    prepared, analysis = run / "prepared", run / "analysis"
    manifest = verify_prepared(prepared)
    items, human = validate_inputs(read_csv(prepared / "items.csv"), read_csv(prepared / "human.csv"))
    source = verify_aligned_source(items, human, aligned)
    if source != json.loads((prepared / "source_validation.json").read_text()):
        raise ValueError("Source-audit discrepancy")
    for row in items.itertuples():
        from common import digest
        if digest([row.article, row.summary]) != row.text_sha256:
            raise ValueError("Item text differs from its stored fingerprint")
    models = json.loads((prepared / "models.json").read_text())
    proxies = {}
    for alias in models:
        folder = run / "scores" / alias
        done = json.loads((folder / "complete.json").read_text())
        spec = json.loads((folder / "run_spec.json").read_text())
        if (done["scores_sha256"] != sha256(folder / "scores.csv") or
                done["spec_sha256"] != sha256(folder / "run_spec.json") or
                spec["model"] != models[alias] or
                spec["prepared_manifest_sha256"] != sha256(prepared / "manifest.json")):
            raise ValueError("Model/source provenance differs")
        proxies[alias] = validate_scores(read_csv(folder / "scores.csv"), items)
    targets = {dim: human[human.dimension == dim].groupby("item_id").score.mean().reindex(items.item_id).to_numpy()
               for dim in DIMENSIONS}
    coeff = pd.read_csv(analysis / "coefficients.csv")
    predictions = pd.read_csv(analysis / "prospective_paths.csv")
    splits = {(s["dimension"], s["repeat"]): s for s in
              [json.loads(line) for line in (prepared / "splits.jsonl").read_text().splitlines()]}
    summary = pd.read_csv(analysis / "summary.csv")
    analysis_spec = json.loads((analysis / "run_spec.json").read_text())
    accounting = json.loads((analysis / "split_accounting.json").read_text())
    radii = analysis_spec["radii"]
    validate_result_frames(coeff, predictions, summary, models, splits, radii, accounting)
    positions = {k: i for i, k in enumerate(items.item_id)}
    coeff_error = 0.
    endpoint_error = 0.
    for row in coeff.itertuples():
        ab = independent_ab(targets[row.dimension], proxies[row.alias])
        coeff_error = max(coeff_error, *np.abs(np.array(ab) - [row.A, row.B]))
    if predictions.duplicated(["alias", "dimension", "repeat"]).any():
        raise ValueError("Duplicate model/split prediction")
    for row in predictions.itertuples():
        split = splits[(row.dimension, row.repeat)]
        if set(split["calibration"]) & set(split["evaluation"]):
            raise ValueError("Calibration/evaluation overlap")
        cal = np.array([positions[k] for k in split["calibration"]])
        eva = np.array([positions[k] for k in split["evaluation"]])
        j, r = targets[row.dimension], proxies[row.alias]
        ab = independent_ab(j[cal], r[cal])
        coeff_error = max(coeff_error, *np.abs(np.array(ab) - [row.cal_a_exact, row.cal_b_theory]))
        gains = independent_grid_gains(j[eva], r[eva], radii)
        for radius in radii:
            suffix = str(float(radius)).replace(".", "p")
            pred = ab[0] * np.sqrt(radius) + ab[1] * radius
            actual = gains[radius]
            endpoint_error = max(endpoint_error, abs(actual - getattr(row, f"actual_eval_delta_{suffix}")),
                                 abs(pred - getattr(row, f"pred_theory_delta_{suffix}")),
                                 abs(abs(pred - actual) - getattr(row, f"abs_err_theory_{suffix}")))
    if coeff_error > 1e-10 or endpoint_error > 1e-9:
        raise ValueError(f"Independent numerical audit failed: coefficient={coeff_error}, endpoint={endpoint_error}")
    common = predictions.groupby(["dimension", "repeat"]).alias.agg(set)
    if not common.map(lambda values: values == set(models)).all():
        raise ValueError("Comparison uses unmatched model support")
    summary = pd.read_csv(analysis / "summary.csv")
    max_summary_error = 0.
    for row in summary.itertuples():
        block = predictions[(predictions.alias == row.alias) & (predictions.dimension == row.dimension)]
        suffix = str(float(row.radius)).replace(".", "p")
        for name in ["theory", "aonly", "empfit", "anu", "zero"]:
            expected = block[f"abs_err_{name}_{suffix}"].mean()
            max_summary_error = max(max_summary_error, abs(expected - getattr(row, f"mae_{name}")))
        if row.n_common_splits != len(block):
            raise ValueError("Summary split count mismatch")
    if max_summary_error > 1e-12:
        raise ValueError("Summary aggregation mismatch")
    return dict(status="PASS", n_verified_summaries=source["n_verified_summaries"],
                n_verified_ratings=source["n_verified_ratings"], n_full_pool_coefficients=len(coeff),
                n_model_splits=len(predictions), n_joint_splits=len(common),
                n_planned_joint_splits=4 * manifest["repeats"],
                max_coefficient_error=coeff_error, max_independent_endpoint_error=endpoint_error,
                max_summary_error=max_summary_error,
                analysis_spec_sha256=sha256(analysis / "run_spec.json"), audit_code_sha256=sha256(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--aligned-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.aligned_jsonl)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
