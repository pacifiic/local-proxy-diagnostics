#!/usr/bin/env python3
"""Independent input-to-prediction audit for exp36.

The audit intentionally reimplements path construction, coefficient formulas,
and exact KL root solving instead of trusting the experiment helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD_SPEC = importlib.util.spec_from_file_location(
    "exp36_build_audit", HERE / "build.py"
)
build = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules["exp36_build_audit"] = build
BUILD_SPEC.loader.exec_module(build)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_id(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].astype(str).agg("||".join, axis=1)


def manual_coefficients(gold: np.ndarray, proxy: np.ndarray) -> tuple[float, float]:
    j = np.asarray(gold, dtype=float)
    r = np.asarray(proxy, dtype=float)
    j_centered = j - np.mean(j)
    r_centered = r - np.mean(r)
    epsilon = r - j
    epsilon_centered = epsilon - np.mean(epsilon)
    sigma_j2 = float(np.mean(j_centered**2))
    sigma_r2 = float(np.mean(r_centered**2))
    if sigma_r2 <= 0:
        raise ValueError("zero proxy variance")
    rho = float(np.mean(j_centered * epsilon_centered))
    kappa_r = float(np.mean(r_centered**3))
    kappa_jrr = float(np.mean(j_centered * r_centered**2))
    a = math.sqrt(2.0) * (sigma_j2 + rho) / math.sqrt(sigma_r2)
    b = (
        kappa_jrr / sigma_r2
        - 2.0 * (sigma_j2 + rho) * kappa_r / (3.0 * sigma_r2**2)
    )
    return a, b


def log_mean_exp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.mean(np.exp(values - maximum))))


def independent_weights(proxy: np.ndarray, d: float) -> np.ndarray:
    r = np.asarray(proxy, dtype=float)
    if d <= 0:
        return np.full(len(r), 1.0 / len(r))

    def weights_at(t: float) -> np.ndarray:
        logits = t * r
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        return weights / float(np.sum(weights))

    def kl_at(t: float) -> float:
        weights = weights_at(t)
        return t * float(np.sum(weights * r)) - log_mean_exp(t * r)

    low, high = 0.0, 1.0
    while kl_at(high) < d:
        low, high = high, high * 2.0
        if high > 1e6:
            raise RuntimeError("independent KL root could not be bracketed")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if kl_at(middle) < d:
            low = middle
        else:
            high = middle
    return weights_at(high)


def independent_gain(proxy: np.ndarray, gold: np.ndarray, d: float) -> float:
    weights = independent_weights(proxy, d)
    return float(np.sum(weights * gold) - np.mean(gold))


def independent_split_indices(
    n_items: int,
    calibration_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_items)
    rng.shuffle(indices)
    n_calibration = max(10, int(round(calibration_fraction * n_items)))
    n_calibration = min(n_calibration, n_items - 10)
    return (
        np.sort(indices[:n_calibration]),
        np.sort(indices[n_calibration:]),
    )


def manual_human_paths(cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    frame = pd.read_csv(cfg["human_long_csv"])
    item_columns = [x.strip() for x in cfg["item_cols"].split(",")]
    if "item_id" not in frame.columns:
        frame["item_id"] = item_id(frame, item_columns)
    group_columns = [x.strip() for x in cfg["group_cols"].split(",")]
    grouped = frame.groupby(group_columns, dropna=False)
    paths = {}
    for group_value, group in grouped:
        values = group_value if isinstance(group_value, tuple) else (group_value,)
        group_key = "__".join(str(value) for value in values)
        matrix = group.pivot(
            index="item_id",
            columns=cfg["rater_col"],
            values=cfg["score_col"],
        ).sort_index()
        if matrix.isna().any(axis=None):
            matrix = matrix.dropna(axis=0, how="any")
        if len(matrix) < cfg["min_items"] or matrix.shape[1] < cfg["min_raters"]:
            continue
        array = matrix.to_numpy(dtype=float)
        ids = matrix.index.to_numpy(dtype=str)
        for index, rater in enumerate(matrix.columns):
            proxy = array[:, index]
            gold = np.mean(np.delete(array, index, axis=1), axis=1)
            paths[f"{group_key}__heldout_{rater}"] = (ids, proxy, gold)
    return paths


def manual_model_paths(cfg: dict) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    human = pd.read_csv(cfg["human_long_csv"])
    item_columns = [x.strip() for x in cfg["item_cols"].split(",")]
    human["item_id"] = item_id(human, item_columns)
    group_columns = [x.strip() for x in cfg["group_cols"].split(",")]
    gold = (
        human.groupby([*group_columns, "item_id"], dropna=False)[cfg["score_col"]]
        .mean()
        .reset_index(name="gold")
    )
    proxy = pd.read_csv(cfg["proxy_csv"]).rename(
        columns={
            cfg["proxy_item_col"]: "item_id",
            cfg["proxy_score_col"]: "proxy",
        }
    )
    merged = gold.merge(
        proxy[[*group_columns, "item_id", "proxy"]],
        on=[*group_columns, "item_id"],
        how="inner",
        validate="one_to_one",
    )
    paths = {}
    for group_value, group in merged.groupby(group_columns, dropna=False):
        values = group_value if isinstance(group_value, tuple) else (group_value,)
        group_key = "__".join(str(value) for value in values)
        group = group.sort_values("item_id")
        if len(group) < cfg["min_items"]:
            continue
        paths[f"{group_key}__model_proxy"] = (
            group["item_id"].to_numpy(dtype=str),
            group["proxy"].to_numpy(dtype=float),
            group["gold"].to_numpy(dtype=float),
        )
    return paths


def independent_curve_models(proxy: np.ndarray, gold: np.ndarray) -> dict:
    n_at_maximum = int(
        np.sum(
            np.isclose(
                proxy,
                np.max(proxy),
                rtol=0.0,
                atol=1e-12,
            )
        )
    )
    max_d = float(np.log(len(proxy) / n_at_maximum))
    fit_max_d = min(float(build.exp21.FIT_MAX_D), 0.95 * max_d)
    d_fit = build.FIT_D_GRID[build.FIT_D_GRID <= fit_max_d + 1e-12]
    if len(d_fit) < 8:
        d_fit = np.linspace(0.0, fit_max_d, 21)
    gain_fit = np.array(
        [independent_gain(proxy, gold, float(d)) if d > 0 else 0.0 for d in d_fit]
    )
    x = np.sqrt(d_fit)
    two_design = np.column_stack([x, x**2])
    cubic_design = np.column_stack([x, x**2, x**3])
    two, *_ = np.linalg.lstsq(two_design, gain_fit, rcond=None)
    cubic, *_ = np.linalg.lstsq(cubic_design, gain_fit, rcond=None)
    a, b = manual_coefficients(gold, proxy)
    moment = a * x + b * x**2

    def fit_spline(values: np.ndarray):
        model = make_pipeline(
            SplineTransformer(
                n_knots=build.SPLINE_N_KNOTS,
                degree=build.SPLINE_DEGREE,
                knots="quantile",
                extrapolation="constant",
                include_bias=False,
            ),
            RidgeCV(alphas=build.SPLINE_ALPHAS, fit_intercept=True),
        )
        model.fit(x.reshape(-1, 1), values)
        origin = float(model.predict(np.zeros((1, 1)))[0])

        def predict(d: float) -> float:
            value = model.predict(np.array([[math.sqrt(max(0.0, d))]]))[0]
            return float(value - origin)

        return predict

    return {
        "A": a,
        "B": b,
        "max_d": max_d,
        "two": two,
        "cubic": cubic,
        "direct_spline": fit_spline(gain_fit),
        "residual_spline": fit_spline(gain_fit - moment),
    }


def validate_input_tables() -> dict:
    checks = {}
    summeval = pd.read_csv(ROOT / "data" / "summeval_long.csv")
    sum_key = [
        "article_id",
        "model_id",
        "rater_type",
        "rater_idx",
        "dimension",
    ]
    assert len(summeval) == 51200
    assert not summeval.duplicated(sum_key).any()
    summeval["item_id"] = item_id(summeval, ["article_id", "model_id"])
    rating_count = summeval.groupby(
        ["item_id", "dimension", "rater_type"]
    )["rater_idx"].nunique().reset_index(name="n_raters")
    assert set(
        rating_count.loc[
            rating_count["rater_type"] == "expert", "n_raters"
        ]
    ) == {3}
    assert set(
        rating_count.loc[
            rating_count["rater_type"] == "turker", "n_raters"
        ]
    ) == {5}
    assert np.isfinite(summeval["score"]).all()
    assert summeval["score"].between(1, 5).all()
    checks["summeval"] = {
        "rows": len(summeval),
        "items": summeval["item_id"].nunique(),
        "duplicate_full_keys": 0,
    }

    proxy = pd.read_csv(ROOT / "data" / "model_proxy_4o_aggregated.csv")
    assert len(proxy) == 6400
    assert not proxy.duplicated(["dimension", "item_id"]).any()
    assert set(proxy["item_id"]) == set(summeval["item_id"])
    assert set(proxy["dimension"]) == set(summeval["dimension"])
    # Duplicate candidate summaries within the same article are genuinely
    # indistinguishable to the scoring prompt. The remapping pipeline groups
    # their independent three-pass score sets by article-summary text before
    # assigning the score to each canonical model id. Hence duplicate summaries
    # intentionally have 6/9/12 rather than exactly three contributing passes.
    expected_judge_distribution = {3: 6016, 6: 304, 9: 48, 12: 32}
    observed_judge_distribution = {
        int(key): int(value)
        for key, value in proxy["n_judges"].value_counts().sort_index().items()
    }
    assert observed_judge_distribution == expected_judge_distribution
    per_dimension_judges = (
        proxy.groupby("dimension")["n_judges"]
        .value_counts()
        .unstack(fill_value=0)
    )
    assert per_dimension_judges.nunique(axis=0).eq(1).all()
    assert np.isfinite(proxy["proxy_score_mean"]).all()
    checks["gpt4o_proxy"] = {
        "rows": len(proxy),
        "items": proxy["item_id"].nunique(),
        "exact_summeval_item_coverage": True,
        "n_judges_distribution": observed_judge_distribution,
        "variable_n_judges_reason": (
            "three-pass score sets are pooled for duplicate article-summary "
            "texts that map to multiple canonical model ids"
        ),
    }

    wmt = pd.read_csv(ROOT / "data" / "wmt_mqm_long.csv")
    assert len(wmt) == 161715
    assert not wmt.duplicated(["dataset_id", "item_id", "rater"]).any()
    assert wmt["dataset_id"].nunique() == 47
    assert np.isfinite(wmt["score"]).all()
    for _, group in wmt.groupby("dataset_id"):
        matrix = group.pivot(index="item_id", columns="rater", values="score")
        assert matrix.shape[1] == 3
        assert not matrix.isna().any(axis=None)
        assert len(matrix) >= 200
    checks["wmt"] = {
        "rows": len(wmt),
        "clusters": wmt["dataset_id"].nunique(),
        "eligible_paths": 141,
        "complete_three_rater_matrices": True,
    }

    openmeva = pd.read_csv(ROOT / "data" / "openmeva_long.csv")
    open_key = ["dataset", "story_id", "model_id", "rater_idx"]
    assert len(openmeva) == 10000
    assert not openmeva.duplicated(open_key).any()
    count = openmeva.groupby(["dataset", "story_id", "model_id"])[
        "rater_idx"
    ].nunique()
    assert set(count) == {5}
    assert len(count) == 2000
    assert np.isfinite(openmeva["score"]).all()
    assert openmeva["score"].between(1, 5).all()
    checks["openmeva"] = {
        "rows": len(openmeva),
        "dataset_aware_items": len(count),
        "ratings_per_item": 5,
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--flexible-check-repeats",
        type=int,
        nargs="*",
        default=[0],
        help="Repeat indices independently refit for every path.",
    )
    args = parser.parse_args()
    predictions = pd.read_csv(args.outdir / "predictions.csv")
    run_spec = json.loads((args.outdir / "run_spec.json").read_text())

    upstream_report_path = (
        ROOT
        / "audits"
        / "source_data_fixed_141"
        / "source_data_audit.json"
    )
    upstream_report = json.loads(upstream_report_path.read_text())
    assert upstream_report["prepared_csv_to_exp33_results"] == "PASS"
    assert upstream_report["source_to_prepared_csv"] == "PASS"
    assert upstream_report["gpt4o"]["raw_rows"] == 19200
    assert upstream_report["gpt4o"]["raw_duplicate_dimension_item_rep"] == 0
    assert upstream_report["gpt4o"]["ambiguous_duplicate_summary_items"] == 96
    assert upstream_report["gpt4o"]["keys_equal"]
    assert upstream_report["wmt"]["fixed_source_rebuild_pass"]
    assert upstream_report["summeval"]["keys_equal"]
    assert upstream_report["openmeva"]["keys_equal"]

    hash_checks = {}
    for record in run_spec["datasets"]:
        human_path = ROOT / record["human_long_csv"]
        observed = sha256(human_path)
        assert observed == record["human_long_sha256"]
        hash_checks[record["human_long_csv"]] = observed
        if record["proxy_csv"]:
            proxy_path = ROOT / record["proxy_csv"]
            observed_proxy = sha256(proxy_path)
            assert observed_proxy == record["proxy_sha256"]
            hash_checks[record["proxy_csv"]] = observed_proxy

    input_checks = validate_input_tables()
    stored_unit = (
        predictions.sort_values(["dataset", "path_id", "repeat", "d", "method"])
        .groupby(["dataset", "path_id", "repeat", "d"], as_index=False)
        .first()
    )
    unit_lookup = {
        (row.dataset, row.path_id, int(row.repeat), float(row.d)): row
        for row in stored_unit.itertuples(index=False)
    }
    prediction_lookup = {
        (
            row.dataset,
            row.path_id,
            int(row.repeat),
            float(row.d),
            row.method,
        ): row
        for row in predictions.itertuples(index=False)
    }

    max_path_deviation = 0.0
    max_cal_coefficient_deviation = 0.0
    max_eval_coefficient_deviation = 0.0
    max_actual_gain_deviation = 0.0
    max_flexible_prediction_deviation = 0.0
    worst_flexible_key = None
    n_units_checked = 0
    n_targets_checked = 0
    n_flexible_targets_checked = 0
    path_counts = {}

    for cfg in build.exp21.CONFIGS:
        canonical = list(build.exp21.calib.build_paths(build.exp21._make_args(cfg)))
        manual = (
            manual_model_paths(cfg)
            if cfg.get("proxy_csv")
            else manual_human_paths(cfg)
        )
        assert len(canonical) == len(manual)
        path_counts[cfg["dataset_name"]] = len(canonical)
        for path in canonical:
            assert path.path_id in manual
            ids, proxy, gold = manual[path.path_id]
            assert np.array_equal(path.item_ids, ids)
            max_path_deviation = max(
                max_path_deviation,
                float(np.max(np.abs(path.proxy_scores - proxy))),
                float(np.max(np.abs(path.gold_scores - gold))),
            )

        rng = np.random.default_rng(build.exp21.RANDOM_SEED)
        for path in canonical:
            split_plan = [
                independent_split_indices(
                    len(path.item_ids), build.exp21.CALIBRATION_FRAC, rng
                )
                for _ in range(build.exp21.N_REPEATS)
            ]
            for repeat, (cal_idx, eval_idx) in enumerate(split_plan):
                r_cal = np.asarray(path.proxy_scores[cal_idx], dtype=float)
                j_cal = np.asarray(path.gold_scores[cal_idx], dtype=float)
                r_eval = np.asarray(path.proxy_scores[eval_idx], dtype=float)
                j_eval = np.asarray(path.gold_scores[eval_idx], dtype=float)
                a_cal, b_cal = manual_coefficients(j_cal, r_cal)
                a_eval, b_eval = manual_coefficients(j_eval, r_eval)
                n_units_checked += 1

                for d in build.D_TARGETS:
                    key = (cfg["dataset_name"], path.path_id, repeat, d)
                    stored = unit_lookup[key]
                    assert int(stored.n_cal) == len(cal_idx)
                    assert int(stored.n_eval) == len(eval_idx)
                    max_cal_coefficient_deviation = max(
                        max_cal_coefficient_deviation,
                        abs(float(stored.A_cal) - a_cal),
                        abs(float(stored.B_cal) - b_cal),
                    )
                    max_eval_coefficient_deviation = max(
                        max_eval_coefficient_deviation,
                        abs(float(stored.A_eval_diagnostic) - a_eval),
                        abs(float(stored.B_eval_diagnostic) - b_eval),
                    )
                    exact = independent_gain(r_eval, j_eval, d)
                    max_actual_gain_deviation = max(
                        max_actual_gain_deviation,
                        abs(float(stored.actual_gain) - exact),
                    )
                    n_targets_checked += 1

                    ab = prediction_lookup[
                        (*key, "A+B_moment")
                    ].predicted_gain
                    aonly = prediction_lookup[
                        (*key, "A-only_moment")
                    ].predicted_gain
                    oracle = prediction_lookup[
                        (*key, "eval_oracle_A+B")
                    ].predicted_gain
                    assert abs(ab - (a_cal * math.sqrt(d) + b_cal * d)) < 1e-11
                    assert abs(aonly - a_cal * math.sqrt(d)) < 1e-11
                    assert abs(
                        oracle - (a_eval * math.sqrt(d) + b_eval * d)
                    ) < 1e-11

                if repeat not in args.flexible_check_repeats:
                    continue
                models = independent_curve_models(r_cal, j_cal)
                for d in build.D_TARGETS:
                    x = math.sqrt(d)
                    independent_predictions = {
                        "empirical_two_term": (
                            models["two"][0] * x + models["two"][1] * x**2
                        ),
                        "empirical_cubic": (
                            models["cubic"][0] * x
                            + models["cubic"][1] * x**2
                            + models["cubic"][2] * x**3
                        ),
                        "direct_spline": models["direct_spline"](d),
                        "A+B_residual_spline": (
                            models["A"] * x
                            + models["B"] * d
                            + models["residual_spline"](d)
                        ),
                    }
                    if d < models["max_d"] - 1e-10:
                        independent_predictions["calibration_curve"] = (
                            independent_gain(r_cal, j_cal, d)
                        )
                    for method, expected in independent_predictions.items():
                        observed = prediction_lookup[
                            (
                                cfg["dataset_name"],
                                path.path_id,
                                repeat,
                                d,
                                method,
                            )
                        ].predicted_gain
                        deviation = abs(float(observed) - float(expected))
                        if deviation > max_flexible_prediction_deviation:
                            max_flexible_prediction_deviation = deviation
                            worst_flexible_key = (
                                cfg["dataset_name"],
                                path.path_id,
                                repeat,
                                d,
                                method,
                                float(observed),
                                float(expected),
                            )
                        n_flexible_targets_checked += 1

    expected_path_counts = {
        "SummEval-human_heldout": 32,
        "SummEval-GPT4o": 4,
        "WMT-MQM-human_heldout": 141,
        "OpenMEVA-MAGS-human_heldout": 10,
    }
    assert path_counts == expected_path_counts
    assert max_path_deviation < 1e-12
    assert max_cal_coefficient_deviation < 1e-11
    assert max_eval_coefficient_deviation < 1e-11
    assert max_actual_gain_deviation < 1e-11
    assert max_flexible_prediction_deviation < 1e-8, (
        "flexible prediction deviation "
        f"{max_flexible_prediction_deviation:.12g}; "
        f"worst={worst_flexible_key}"
    )

    prospective = predictions[
        predictions["information_setting"] == "calibration_only_pre_pool"
    ]
    assert "eval_oracle_A+B" not in set(prospective["method"])
    assert run_spec["information_settings"]["prospective"].startswith(
        "30% calibration"
    )
    assert "no interpolation" in run_spec["curve_fit_evaluation"]
    skipped = pd.read_csv(args.outdir / "skipped.csv")
    assert set(skipped["method"]) <= {"calibration_curve"}

    report = {
        "status": "PASS",
        "input_hashes": hash_checks,
        "upstream_source_to_prepared_audit": {
            "status": "PASS",
            "report": str(upstream_report_path.relative_to(ROOT)),
            "gpt_raw_rows": upstream_report["gpt4o"]["raw_rows"],
            "gpt_raw_duplicate_dimension_item_rep": upstream_report[
                "gpt4o"
            ]["raw_duplicate_dimension_item_rep"],
            "gpt_ambiguous_duplicate_summary_items": upstream_report[
                "gpt4o"
            ]["ambiguous_duplicate_summary_items"],
            "wmt_fixed_source_rebuild": upstream_report["wmt"][
                "fixed_source_rebuild_pass"
            ],
        },
        "input_table_checks": input_checks,
        "path_counts": path_counts,
        "path_array_max_abs_deviation": max_path_deviation,
        "split_units_checked": n_units_checked,
        "exact_targets_recomputed": n_targets_checked,
        "flexible_predictions_independently_refit": n_flexible_targets_checked,
        "max_calibration_coefficient_deviation": max_cal_coefficient_deviation,
        "max_evaluation_coefficient_deviation": max_eval_coefficient_deviation,
        "max_exact_gain_deviation": max_actual_gain_deviation,
        "max_flexible_prediction_deviation": max_flexible_prediction_deviation,
        "worst_flexible_prediction_key": worst_flexible_key,
        "prospective_information_check": "PASS",
        "reachability_skip_check": {
            "status": "PASS",
            "n_skipped": len(skipped),
            "methods": sorted(skipped["method"].unique().tolist()),
        },
        "notes": [
            "All held-out exact gains were recomputed with an independent KL solver.",
            "Flexible methods were independently refit for the selected repeat of every path.",
            "The evaluation-oracle row is excluded from prospective methods.",
        ],
    }
    (args.outdir / "data_chain_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
