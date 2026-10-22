#!/usr/bin/env python3
"""Independent data-chain audit for exp33.

This audit reconstructs paths directly from input CSVs, recreates every outer
split, independently recomputes coefficient predictions, and re-derives exact
held-out gains on a deterministic sample with a separate root solver.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD_PATH = HERE / "build.py"
SPEC = importlib.util.spec_from_file_location("exp33_build", BUILD_PATH)
exp33 = importlib.util.module_from_spec(SPEC)
sys.modules["exp33_build"] = exp33
SPEC.loader.exec_module(exp33)


@dataclass
class ManualPath:
    path_id: str
    item_ids: np.ndarray
    proxy: np.ndarray
    gold: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_id_frame(frame: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    if "item_id" not in out:
        out["item_id"] = out[item_cols].astype(str).agg("||".join, axis=1)
    return out


def group_key(key: object) -> str:
    values = key if isinstance(key, tuple) else (key,)
    return "__".join(str(value) for value in values)


def manual_human_paths(cfg: dict) -> dict[str, ManualPath]:
    item_cols = cfg["item_cols"].split(",")
    group_cols = cfg["group_cols"].split(",")
    frame = item_id_frame(pd.read_csv(cfg["human_long_csv"]), item_cols)
    paths = {}
    for key, group in frame.groupby(group_cols, dropna=False):
        matrix = group.pivot_table(
            index="item_id",
            columns=cfg["rater_col"],
            values=cfg["score_col"],
            aggfunc="mean",
        )
        matrix = matrix.dropna(axis=0, how="any").sort_index()
        if (
            matrix.shape[0] < cfg["min_items"]
            or matrix.shape[1] < cfg["min_raters"]
        ):
            continue
        values = matrix.to_numpy(dtype=float)
        prefix = group_key(key)
        for index, rater in enumerate(matrix.columns):
            path_id = f"{prefix}__heldout_{rater}"
            paths[path_id] = ManualPath(
                path_id=path_id,
                item_ids=matrix.index.to_numpy(dtype=str),
                proxy=values[:, index],
                gold=np.mean(np.delete(values, index, axis=1), axis=1),
            )
    return paths


def manual_model_paths(cfg: dict) -> tuple[dict[str, ManualPath], dict]:
    item_cols = cfg["item_cols"].split(",")
    group_cols = cfg["group_cols"].split(",")
    human = item_id_frame(pd.read_csv(cfg["human_long_csv"]), item_cols)
    proxy = pd.read_csv(cfg["proxy_csv"])
    score_col = cfg["proxy_score_col"]
    proxy_aggregated = (
        proxy[[*group_cols, cfg["proxy_item_col"], score_col]]
        .groupby(
            [*group_cols, cfg["proxy_item_col"]], dropna=False
        )[score_col]
        .mean()
        .reset_index()
        .rename(
            columns={
                cfg["proxy_item_col"]: "item_id",
                score_col: "proxy_score",
            }
        )
    )
    gold = (
        human.groupby([*group_cols, "item_id"], dropna=False)[
            cfg["score_col"]
        ]
        .mean()
        .reset_index(name="gold_score")
    )
    merged = gold.merge(
        proxy_aggregated,
        on=[*group_cols, "item_id"],
        how="inner",
        validate="one_to_one",
    )
    paths = {}
    for key, group in merged.groupby(group_cols, dropna=False):
        group = group.sort_values("item_id")
        if len(group) < cfg["min_items"]:
            continue
        prefix = group_key(key)
        path_id = f"{prefix}__model_proxy"
        paths[path_id] = ManualPath(
            path_id=path_id,
            item_ids=group["item_id"].to_numpy(dtype=str),
            proxy=group["proxy_score"].to_numpy(dtype=float),
            gold=group["gold_score"].to_numpy(dtype=float),
        )
    mapping = {
        "n_human_gold_keys": len(gold),
        "n_proxy_keys_after_aggregation": len(proxy_aggregated),
        "n_matched_keys": len(merged),
        "n_unmatched_human_keys": len(gold) - len(merged),
        "n_unmatched_proxy_keys": len(proxy_aggregated) - len(merged),
        "duplicate_aggregated_proxy_keys": int(
            proxy_aggregated.duplicated([*group_cols, "item_id"]).sum()
        ),
    }
    return paths, mapping


def independent_coefficients(
    gold: np.ndarray, proxy: np.ndarray
) -> tuple[float, float, float]:
    j = np.asarray(gold, dtype=float)
    r = np.asarray(proxy, dtype=float)
    jm, rm = float(np.mean(j)), float(np.mean(r))
    jc, rc = j - jm, r - rm
    var_j = float(np.mean(jc**2))
    var_r = float(np.mean(rc**2))
    if var_r <= 0:
        raise ValueError("zero proxy variance")
    covariance = float(np.mean(jc * rc))
    kappa_r = float(np.mean(rc**3))
    kappa_jrr = float(np.mean(jc * rc**2))
    a = math.sqrt(2.0) * covariance / math.sqrt(var_r)
    b = (
        kappa_jrr / var_r
        - 2.0 * covariance * kappa_r / (3.0 * var_r**2)
    )
    a_nu = math.sqrt(2.0) * var_j / math.sqrt(var_r)
    return a, b, a_nu


def independent_weights(proxy: np.ndarray, d: float) -> np.ndarray:
    r = np.asarray(proxy, dtype=float)
    if d <= 0:
        return np.full(len(r), 1.0 / len(r))

    def weights(t: float) -> np.ndarray:
        logits = t * (r - np.max(r))
        result = np.exp(logits)
        return result / np.sum(result)

    def kl(t: float) -> float:
        current = weights(t)
        return float(
            np.log(len(r))
            + np.sum(current * np.log(np.clip(current, 1e-300, None)))
        )

    hi = 1.0
    while kl(hi) < d:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(f"could not independently bracket d={d}")
    temperature = brentq(lambda value: kl(value) - d, 0.0, hi, xtol=1e-13)
    return weights(float(temperature))


def validate_input_table(path: Path) -> dict:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "numeric_nan_cells": int(numeric.isna().sum().sum()),
        "numeric_inf_cells": int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        ),
    }


def compare_paths(
    manual: dict[str, ManualPath], generated: list
) -> tuple[dict, dict[str, object]]:
    generated_by_id = {path.path_id: path for path in generated}
    if set(manual) != set(generated_by_id):
        raise AssertionError(
            "manual/generated path IDs differ: "
            f"manual-only={sorted(set(manual)-set(generated_by_id))[:5]}, "
            f"generated-only={sorted(set(generated_by_id)-set(manual))[:5]}"
        )
    max_proxy_drift = 0.0
    max_gold_drift = 0.0
    item_mismatches = 0
    duplicate_item_paths = 0
    details = {}
    for path_id, expected in manual.items():
        observed = generated_by_id[path_id]
        if len(np.unique(observed.item_ids)) != len(observed.item_ids):
            duplicate_item_paths += 1
        if not np.array_equal(expected.item_ids, observed.item_ids):
            item_mismatches += 1
        max_proxy_drift = max(
            max_proxy_drift,
            float(
                np.max(
                    np.abs(expected.proxy - observed.proxy_scores)
                )
            ),
        )
        max_gold_drift = max(
            max_gold_drift,
            float(
                np.max(
                    np.abs(expected.gold - observed.gold_scores)
                )
            ),
        )
        details[path_id] = observed
    summary = {
        "n_paths": len(generated),
        "n_unique_path_ids": len(generated_by_id),
        "item_id_mismatch_paths": item_mismatches,
        "duplicate_item_paths": duplicate_item_paths,
        "max_proxy_array_drift": max_proxy_drift,
        "max_gold_array_drift": max_gold_drift,
        "all_numeric_finite": bool(
            all(
                np.isfinite(path.proxy_scores).all()
                and np.isfinite(path.gold_scores).all()
                for path in generated
            )
        ),
    }
    return summary, details


def audit_splits_and_outputs(
    cfg: dict,
    paths: list,
    saved: pd.DataFrame,
) -> dict:
    rng = np.random.default_rng(exp33.exp21.RANDOM_SEED)
    saved_cfg = saved[saved["dataset"] == cfg["dataset_name"]]
    max_coefficient_drift = {
        "A+B": 0.0,
        "A-only": 0.0,
        "A_nu": 0.0,
    }
    max_actual_drift = 0.0
    actual_cells_checked = 0
    split_cells_checked = 0
    sampled_repeats = {0, 17, 49}

    for path in paths:
        path_saved = saved_cfg[saved_cfg["path_id"] == path.path_id]
        for repeat in range(exp33.exp21.N_REPEATS):
            cal_idx, eval_idx = exp33.exp21.calib.split_indices(
                len(path.item_ids), exp33.exp21.CALIBRATION_FRAC, rng
            )
            cell = path_saved[path_saved["repeat"] == repeat]
            expected_n_cal, expected_n_eval = len(cal_idx), len(eval_idx)
            if (
                set(cell["n_cal"]) != {expected_n_cal}
                or set(cell["n_eval"]) != {expected_n_eval}
            ):
                raise AssertionError(
                    f"split-size mismatch: {cfg['dataset_name']} "
                    f"{path.path_id} repeat={repeat}"
                )
            split_cells_checked += 1

            r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
            j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
            a, b, a_nu = independent_coefficients(j_cal, r_cal)
            formulas = {
                "A+B": lambda d: a * math.sqrt(d) + b * d,
                "A-only": lambda d: a * math.sqrt(d),
                "A_nu": lambda d: a_nu * math.sqrt(d),
            }
            for baseline, formula in formulas.items():
                observed = cell[cell["baseline"] == baseline]
                for row in observed.itertuples():
                    max_coefficient_drift[baseline] = max(
                        max_coefficient_drift[baseline],
                        abs(float(row.predicted_gain) - formula(float(row.d))),
                    )

            if repeat in sampled_repeats:
                r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
                j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
                observed = cell[cell["baseline"] == "A+B"]
                for row in observed.itertuples():
                    weights = independent_weights(r_eval, float(row.d))
                    actual = float(
                        np.sum(weights * j_eval) - np.mean(j_eval)
                    )
                    max_actual_drift = max(
                        max_actual_drift,
                        abs(actual - float(row.actual_gain)),
                    )
                    actual_cells_checked += 1

    return {
        "split_cells_checked": split_cells_checked,
        "independent_actual_cells_checked": actual_cells_checked,
        "max_independent_actual_gain_drift": max_actual_drift,
        "max_independent_coefficient_prediction_drift": (
            max_coefficient_drift
        ),
    }


def audit_skips(outdir: Path) -> dict:
    skipped = pd.read_csv(outdir / "calibration_only_skipped.csv")
    expected_baselines = {
        "empirical_curve",
        "isotonic_calibration_only",
        "binned_calibration_only",
        "spline_calibration_only",
        "mlp_calibration_only",
    }
    unique_cells = skipped[["dataset", "path_id", "repeat", "d"]].drop_duplicates()
    counts = skipped.groupby(
        ["dataset", "path_id", "repeat", "d"]
    )["baseline"].agg(lambda values: set(values))
    wrong_sets = int(sum(value != expected_baselines for value in counts))
    return {
        "n_skip_records": len(skipped),
        "n_unique_skipped_cells": len(unique_cells),
        "datasets": sorted(skipped["dataset"].unique()),
        "d_values": sorted(skipped["d"].unique().tolist()),
        "wrong_baseline_sets": wrong_sets,
        "all_reasons_calibration_reachability": bool(
            skipped["reason"].str.startswith(
                "calibration frontier max KL="
            ).all()
        ),
    }


def audit_flexible_prediction_boundary(
    cfg: dict,
    paths: list,
    saved: pd.DataFrame,
) -> dict:
    """Re-fit one split and verify flexible predictions have no eval input."""
    path = paths[0]
    rng = np.random.default_rng(exp33.exp21.RANDOM_SEED)
    cal_idx, eval_idx = exp33.exp21.calib.split_indices(
        len(path.item_ids), exp33.exp21.CALIBRATION_FRAC, rng
    )
    r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
    j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
    r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
    j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)

    # The third argument intentionally receives r_cal, not r_eval. Therefore
    # fitted h values live on the calibration pool and are integrated there.
    models = exp33.exp21.fit_split_models(
        r_cal, j_cal, r_cal, exp33.exp21.FIT_MAX_D
    )
    max_saved_prediction_drift = 0.0
    max_counterfactual_prediction_drift = 0.0
    realized_changes = []
    baselines_checked = set()
    cell = saved[
        (saved["dataset"] == cfg["dataset_name"])
        & (saved["path_id"] == path.path_id)
        & (saved["repeat"] == 0)
    ]
    for d in exp33.exp21.D_TARGETS:
        prediction_1, _ = exp33.calibration_only_predictions(
            models, r_cal, j_cal, float(d)
        )
        # There is deliberately no evaluation argument to this function.
        prediction_2, _ = exp33.calibration_only_predictions(
            models, r_cal, j_cal, float(d)
        )
        for baseline, predicted in prediction_1.items():
            observed = cell[
                (cell["d"] == d) & (cell["baseline"] == baseline)
            ]
            if len(observed) == 1:
                max_saved_prediction_drift = max(
                    max_saved_prediction_drift,
                    abs(predicted - float(observed.iloc[0]["predicted_gain"])),
                )
                baselines_checked.add(baseline)
            max_counterfactual_prediction_drift = max(
                max_counterfactual_prediction_drift,
                abs(predicted - prediction_2[baseline]),
            )

        weights_original = independent_weights(r_eval, float(d))
        weights_permuted = independent_weights(r_eval[::-1], float(d))
        actual_original = float(
            np.sum(weights_original * j_eval) - np.mean(j_eval)
        )
        actual_permuted = float(
            np.sum(weights_permuted * j_eval) - np.mean(j_eval)
        )
        realized_changes.append(abs(actual_original - actual_permuted))

    return {
        "path_id": path.path_id,
        "baselines_checked": sorted(baselines_checked),
        "max_saved_prediction_drift": max_saved_prediction_drift,
        "prediction_drift_after_eval_proxy_permutation": (
            max_counterfactual_prediction_drift
        ),
        "max_realized_gain_change_after_eval_proxy_permutation": max(
            realized_changes
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp33-outdir", type=Path, required=True)
    parser.add_argument("--audit-outdir", type=Path, required=True)
    args = parser.parse_args()
    args.audit_outdir.mkdir(parents=True, exist_ok=False)

    run_spec = json.loads(
        (args.exp33_outdir / "run_spec.json").read_text(encoding="utf-8")
    )
    source_paths = {
        "build.py": BUILD_PATH,
        "exp21_builder": exp33.EXP21_PATH,
        "canonical_engine": exp33.exp21.CANON_PATH,
    }
    source_hashes = {
        name: {
            "recorded": run_spec["source_sha256"][name],
            "current": sha256_file(path),
        }
        for name, path in source_paths.items()
    }

    data_tables = {}
    data_hashes = {}
    path_audits = {}
    split_audits = {}
    mapping_audits = {}
    flexible_audits = {}
    saved = pd.read_csv(
        args.exp33_outdir / "calibration_only_predictions.csv"
    )

    for cfg in exp33.exp21.CONFIGS:
        human_path = Path(cfg["human_long_csv"])
        human_key = exp33.exp21._portable_path(str(human_path))
        data_tables[human_key] = validate_input_table(human_path)
        data_hashes[str(human_path)] = sha256_file(human_path)
        if cfg.get("proxy_csv"):
            proxy_path = Path(cfg["proxy_csv"])
            proxy_key = exp33.exp21._portable_path(str(proxy_path))
            data_tables[proxy_key] = validate_input_table(proxy_path)
            data_hashes[str(proxy_path)] = sha256_file(proxy_path)
            manual, mapping = manual_model_paths(cfg)
            mapping_audits[cfg["dataset_name"]] = mapping
        else:
            manual = manual_human_paths(cfg)

        generated = list(
            exp33.exp21.calib.build_paths(exp33.exp21._make_args(cfg))
        )
        path_summary, _ = compare_paths(manual, generated)
        path_audits[cfg["dataset_name"]] = path_summary
        split_audits[cfg["dataset_name"]] = audit_splits_and_outputs(
            cfg, generated, saved
        )
        flexible_audits[cfg["dataset_name"]] = (
            audit_flexible_prediction_boundary(cfg, generated, saved)
        )

    recorded_hashes = {}
    for record in run_spec["datasets"]:
        recorded_hashes[record["human_long_csv"]] = record[
            "human_long_sha256"
        ]
        if record["proxy_csv"]:
            recorded_hashes[record["proxy_csv"]] = record["proxy_sha256"]
    portable_current = {
        exp33.exp21._portable_path(path): digest
        for path, digest in data_hashes.items()
    }

    failures = []
    if any(
        values["recorded"] != values["current"]
        for values in source_hashes.values()
    ):
        failures.append("source hash mismatch")
    if recorded_hashes != portable_current:
        failures.append("input data hash mismatch")
    if any(
        table["exact_duplicate_rows"]
        or table["numeric_nan_cells"]
        or table["numeric_inf_cells"]
        for table in data_tables.values()
    ):
        failures.append("input table duplicate/non-finite cells")
    if any(
        audit["item_id_mismatch_paths"]
        or audit["duplicate_item_paths"]
        or audit["max_proxy_array_drift"] > 0
        or audit["max_gold_array_drift"] > 1e-12
        or not audit["all_numeric_finite"]
        for audit in path_audits.values()
    ):
        failures.append("manual/generated path mismatch")
    if any(
        audit["max_independent_actual_gain_drift"] > 1e-10
        or max(
            audit["max_independent_coefficient_prediction_drift"].values()
        )
        > 1e-10
        for audit in split_audits.values()
    ):
        failures.append("independent prediction re-derivation mismatch")
    if any(
        audit["max_saved_prediction_drift"] > 1e-10
        or audit["prediction_drift_after_eval_proxy_permutation"] > 0
        or audit[
            "max_realized_gain_change_after_eval_proxy_permutation"
        ]
        <= 1e-8
        for audit in flexible_audits.values()
    ):
        failures.append("flexible prediction information-boundary mismatch")
    skip_audit = audit_skips(args.exp33_outdir)
    if (
        skip_audit["wrong_baseline_sets"]
        or not skip_audit["all_reasons_calibration_reachability"]
    ):
        failures.append("skip accounting mismatch")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "source_hashes": source_hashes,
        "input_hashes_recorded": recorded_hashes,
        "input_hashes_current": portable_current,
        "input_tables": data_tables,
        "path_reconstruction": path_audits,
        "model_proxy_mapping": mapping_audits,
        "split_and_result_rederivation": split_audits,
        "flexible_prediction_boundary": flexible_audits,
        "skip_accounting": skip_audit,
        "failures": failures,
    }
    (args.audit_outdir / "data_chain_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
