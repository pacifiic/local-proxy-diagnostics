#!/usr/bin/env python3
"""Audit silent failures between data generation, artifacts, and analyses.

This audit deliberately checks keys, cardinalities, text identity, split
membership, input hashes, and independently recomputed quantities. It does not
replace each experiment's numerical audit; it checks the data contracts that
can be violated even when every individual script exits successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_INPUT_HASHES = {
    "data/summeval_long.csv":
        "d4c4dd815cc851920c796343f9952d0ec60c124db63d913d2826998881c1531e",
    "data/model_proxy_4o_aggregated.csv":
        "9440315ccb82939e360e8bbac802f31bb3a17d6af244997cfbfa34eeeeeb4599",
    "data/openmeva_long.csv":
        "4e205f20b5183ab21b3caa1a5270e89068f35ae2aa14356e3b425f18f314a10b",
    "data/wmt_mqm_long.csv":
        "6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8",
    "data/wmt_mqm_long_main_protocol.csv":
        "5649ee3041340f7ee96c3079a4f1f859b9ddd625b2d82f731407d32fd88553c0",
}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, condition: bool, name: str, detail: object = None) -> None:
        passed = bool(condition)
        self.checks.append(
            {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )
        print(f"  {'OK  ' if passed else 'FAIL'} {name}"
              + (f": {detail}" if detail is not None else ""))

    @property
    def failures(self) -> list[dict]:
        return [row for row in self.checks if row["status"] == "FAIL"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(np.isfinite(frame[columns].to_numpy(dtype=float)).all())


def audit_hashes(root: Path, audit: Audit) -> None:
    print("== Pinned input hashes ==")
    for relative, expected in EXPECTED_INPUT_HASHES.items():
        observed = sha256_file(root / relative)
        audit.check(observed == expected, relative, observed)

    manifest = root / "data/dpo_candidate_pool/SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        filename = filename.strip()
        observed = sha256_file(manifest.parent / filename)
        audit.check(observed == expected, f"DPO hash {filename}", observed)


def audit_base_data(root: Path, audit: Audit) -> None:
    print("== Prepared data tables ==")
    summeval = pd.read_csv(
        root / "data/summeval_long.csv",
        dtype={"article_id": str, "model_id": str},
    )
    sum_key = [
        "article_id", "model_id", "rater_type", "rater_idx", "dimension"
    ]
    audit.check(len(summeval) == 51200, "SummEval row count", len(summeval))
    audit.check(not summeval.duplicated(sum_key).any(), "SummEval full key unique")
    audit.check(
        summeval[["article_id", "model_id"]].drop_duplicates().shape[0] == 1600,
        "SummEval has 1,600 items",
    )
    audit.check(
        set(summeval["dimension"]) ==
        {"coherence", "consistency", "fluency", "relevance"},
        "SummEval dimensions exact",
    )
    rater_counts = (
        summeval.groupby(
            ["article_id", "model_id", "dimension", "rater_type"]
        )["rater_idx"].nunique()
    )
    expected_raters = (
        summeval["rater_type"].map({"expert": 3, "turker": 5})
    )
    observed_raters = summeval.set_index(
        ["article_id", "model_id", "dimension", "rater_type"]
    ).index.map(rater_counts)
    audit.check(
        np.array_equal(observed_raters, expected_raters.to_numpy()),
        "SummEval every item has 3 expert and 5 crowd ratings",
    )
    audit.check(
        finite(summeval, ["score"]) and summeval["score"].between(1, 5).all(),
        "SummEval scores finite and within [1,5]",
    )

    item_ids = set(
        summeval[["article_id", "model_id"]]
        .drop_duplicates()
        .astype(str)
        .agg("||".join, axis=1)
    )
    proxy_files = [
        root / "data/model_proxy_4o_aggregated.csv",
        root / "data/open_proxy_inputs/model_proxy_openjudge_aggregated.csv",
        root / "data/open_proxy_inputs/proxy_rm_skywork_v02.csv",
    ]
    for path in proxy_files:
        proxy = pd.read_csv(path, dtype={"item_id": str})
        key = ["dimension", "item_id"]
        score_columns = [
            column for column in ["proxy_score", "proxy_score_mean"]
            if column in proxy
        ]
        audit.check(len(proxy) == 6400, f"{path.name} row count", len(proxy))
        audit.check(
            not proxy.duplicated(key).any(), f"{path.name} key unique"
        )
        audit.check(
            set(proxy["item_id"]) == item_ids,
            f"{path.name} exact SummEval item coverage",
        )
        audit.check(
            set(proxy["dimension"]) == set(summeval["dimension"]),
            f"{path.name} exact dimension coverage",
        )
        audit.check(finite(proxy, score_columns), f"{path.name} scores finite")

    rm = pd.read_csv(
        root / "data/open_proxy_inputs/proxy_rm_skywork_v02.csv",
        dtype={"item_id": str},
    )
    audit.check(
        rm.groupby("item_id")["proxy_score"].nunique().eq(1).all(),
        "Skywork item-level RM score is intentionally repeated across dimensions",
    )

    openmeva = pd.read_csv(
        root / "data/openmeva_long.csv",
        dtype={"dataset": str, "story_id": str, "model_id": str,
               "rater_idx": str},
    )
    open_key = ["dataset", "story_id", "model_id", "rater_idx"]
    audit.check(len(openmeva) == 10000, "OpenMEVA row count", len(openmeva))
    audit.check(not openmeva.duplicated(open_key).any(), "OpenMEVA full key unique")
    full_items = openmeva[["dataset", "story_id", "model_id"]].drop_duplicates()
    short_items = openmeva[["story_id", "model_id"]].drop_duplicates()
    audit.check(len(full_items) == 2000, "OpenMEVA dataset-aware item count", len(full_items))
    audit.check(
        len(short_items) == 1000,
        "OpenMEVA ROC/WP IDs overlap when dataset is omitted",
        len(short_items),
    )
    audit.check(
        openmeva.groupby(["dataset", "story_id", "model_id"])["rater_idx"]
        .nunique().eq(5).all(),
        "OpenMEVA every item has five ratings",
    )

    for filename, n_rows, n_datasets, fixed in [
        ("wmt_mqm_long.csv", 161715, 47, True),
        ("wmt_mqm_long_main_protocol.csv", 130026, 45, False),
    ]:
        wmt = pd.read_csv(
            root / "data" / filename,
            dtype={"dataset_id": str, "item_id": str, "rater": str},
        )
        key = ["dataset_id", "item_id", "rater"]
        audit.check(len(wmt) == n_rows, f"{filename} row count", len(wmt))
        audit.check(not wmt.duplicated(key).any(), f"{filename} full key unique")
        audit.check(
            wmt["dataset_id"].nunique() == n_datasets,
            f"{filename} dataset count",
            wmt["dataset_id"].nunique(),
        )
        complete = all(
            len(group) ==
            group["item_id"].nunique() * group["rater"].nunique()
            for _, group in wmt.groupby("dataset_id")
        )
        audit.check(complete, f"{filename} complete item-by-rater matrices")
        if fixed:
            audit.check(
                wmt.groupby("dataset_id")["rater"].nunique().eq(3).all(),
                "fixed_141 contains 47 valid three-rater clusters",
            )
        else:
            n_four = int(
                wmt.groupby("dataset_id")["rater"].nunique().eq(4).sum()
            )
            audit.check(
                n_four == 4,
                "submitted_139 retains four malformed four-rater clusters for provenance",
                n_four,
            )


def audit_dpo(root: Path, audit: Audit) -> None:
    print("== DPO generation -> artifacts -> results ==")
    prep = root / "data/dpo_candidate_pool"
    split_rows = {
        name: read_jsonl(prep / f"prompts_{name}.jsonl")
        for name in ["train", "calibration", "eval"]
    }
    split_ids = {
        name: {row["prompt_id"] for row in rows}
        for name, rows in split_rows.items()
    }
    all_prompts = [row for rows in split_rows.values() for row in rows]
    audit.check(
        all(
            hashlib.sha256(row["prompt"].encode()).hexdigest()[:16] ==
            row["prompt_id"]
            for row in all_prompts
        ),
        "prompt IDs rederive from exact prompt text",
    )
    audit.check(
        len({row["prompt"] for row in all_prompts}) == len(all_prompts),
        "prompt text unique across all splits",
    )
    audit.check(
        not any(
            split_ids[a] & split_ids[b]
            for a, b in [
                ("train", "calibration"), ("train", "eval"),
                ("calibration", "eval"),
            ]
        ),
        "DPO prompt splits disjoint",
    )

    pool = read_jsonl(prep / "pool.jsonl")
    pool_spec = json.loads((prep / "pool.jsonl.run_spec.json").read_text())
    pool_frame = pd.DataFrame(pool)
    pool_key = ["prompt_id", "split", "sample_idx"]
    audit.check(not pool_frame.duplicated(pool_key).any(), "DPO pool key unique")
    for split, expected_prompts in [
        ("train", 1000), ("calibration", 100), ("eval", 200)
    ]:
        expected_candidates = pool_spec[f"n_{'cal' if split == 'calibration' else split}"]
        subset = pool_frame[pool_frame["split"].eq(split)]
        audit.check(
            len(subset) == expected_prompts * expected_candidates,
            f"DPO {split} pool cardinality",
            len(subset),
        )
        sample_sets = subset.groupby("prompt_id")["sample_idx"].apply(set)
        audit.check(
            all(values == set(range(expected_candidates)) for values in sample_sets),
            f"DPO {split} sample_idx is complete and contiguous",
        )
        audit.check(
            set(subset["prompt_id"]) == split_ids[split],
            f"DPO {split} pool covers exact prompt split",
        )

    pool_by_key = {
        tuple(row[column] for column in pool_key): row for row in pool
    }
    proxy = read_jsonl(prep / "pool_p.jsonl")
    proxy_by_key = {
        tuple(row[column] for column in pool_key): row for row in proxy
    }
    audit.check(set(proxy_by_key) == set(pool_by_key), "DPO proxy keys equal pool keys")
    audit.check(
        all(
            proxy_by_key[key]["prompt"] == pool_by_key[key]["prompt"]
            and proxy_by_key[key]["response"] == pool_by_key[key]["response"]
            for key in pool_by_key
        ),
        "DPO proxy scoring preserves exact texts",
    )

    scored = read_jsonl(prep / "pool_scored.jsonl")
    scored_by_key = {
        tuple(row[column] for column in pool_key): row for row in scored
    }
    expected_scored = {
        key for key in pool_by_key if key[1] in {"calibration", "eval"}
    }
    audit.check(set(scored_by_key) == expected_scored, "DPO gold keys equal cal+eval keys")
    audit.check(
        all(
            scored_by_key[key]["prompt"] == pool_by_key[key]["prompt"]
            and scored_by_key[key]["response"] == pool_by_key[key]["response"]
            and scored_by_key[key]["proxy_score"] ==
            proxy_by_key[key]["proxy_score"]
            for key in scored_by_key
        ),
        "DPO score merge preserves text and proxy values",
    )

    by_train: dict[str, list[dict]] = defaultdict(list)
    for row in proxy:
        if row["split"] == "train":
            by_train[row["prompt_id"]].append(row)
    pairs = read_jsonl(prep / "pref_pairs.jsonl")
    exact_pairs = 0
    for pair in pairs:
        candidates = by_train[pair["prompt_id"]]
        max_score = max(row["proxy_score"] for row in candidates)
        min_score = min(row["proxy_score"] for row in candidates)
        best_responses = {
            row["response"] for row in candidates
            if np.isclose(row["proxy_score"], max_score, rtol=0, atol=1e-12)
        }
        worst_responses = {
            row["response"] for row in candidates
            if np.isclose(row["proxy_score"], min_score, rtol=0, atol=1e-12)
        }
        if (
            pair["chosen"] in best_responses
            and pair["rejected"] in worst_responses
            and max_score > min_score
        ):
            exact_pairs += 1
    audit.check(
        exact_pairs == len(pairs) == 968,
        "DPO pairs independently rederive as proxy argmax/argmin",
        exact_pairs,
    )

    sys.path.insert(0, str(root / "lib"))
    from common_experiment_utils import theory_coefficients

    by_cal: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        if row["split"] == "calibration":
            by_cal[row["prompt_id"]].append(row)
    coefficients = []
    for prompt_id, rows in by_cal.items():
        proxy_values = np.array([row["proxy_score"] for row in rows], dtype=float)
        gold_values = np.array([row["gold_score"] for row in rows], dtype=float)
        if np.var(proxy_values) == 0 or np.var(gold_values) == 0:
            continue
        coef = theory_coefficients(gold_values, proxy_values)
        coefficients.append([coef.a_exact, coef.b_theory, coef.a_nu])
    recomputed = np.mean(np.asarray(coefficients), axis=0)
    saved = json.loads((prep / "coefficients.json").read_text())
    audit.check(
        len(coefficients) == saved["n_prompts_used"] == 99,
        "DPO coefficient prompt accounting",
        len(coefficients),
    )
    audit.check(
        np.allclose(
            recomputed, [saved["A"], saved["B"], saved["A_nu"]],
            rtol=0, atol=1e-12,
        ),
        "DPO coefficients independently recompute from calibration only",
    )

    reflogp = read_jsonl(prep / "pool_eval_reflogp.jsonl")
    ref_keys = {tuple(row[column] for column in pool_key) for row in reflogp}
    expected_eval = {key for key in pool_by_key if key[1] == "eval"}
    audit.check(ref_keys == expected_eval, "DPO reference-logp keys equal eval pool")

    final = pd.read_csv(root / "results/actual_dpo/exp09_final_per_prompt.csv")
    audit.check(
        not final.duplicated(["tag", "prompt_id"]).any(),
        "DPO final checkpoint key unique",
    )
    audit.check(
        set(final["prompt_id"]) == split_ids["eval"],
        "DPO final table uses evaluation prompts only",
    )
    legacy_tags = {
        "steps_64", "steps_90", "steps_120", "steps_150",
        "steps_180", "steps_200", "steps_600", "steps_1200",
    }
    audit.check(
        final.groupby("tag").size().eq(200).all()
        and set(final["tag"]) == legacy_tags,
        "legacy DPO table has eight labeled checkpoints x 200 eval prompts",
    )

    three_seed_root = root / "results/actual_dpo/three_seed_canonical"
    three_seed_spec = json.loads((three_seed_root / "run_spec.json").read_text())
    source_root = root / "results/actual_dpo/training_runs"
    expected_seed_hashes = three_seed_spec["input_sha256"]
    audit.check(
        expected_seed_hashes["pool_scored"] == sha256_file(prep / "pool_scored.jsonl"),
        "three-seed DPO uses the canonical scored pool",
    )
    for seed in [0, 1, 2]:
        seed_path = source_root / f"seed_{seed}/exp09_final_per_prompt.csv"
        observed_hash = sha256_file(seed_path)
        audit.check(
            observed_hash == expected_seed_hashes[f"seed_{seed}_per_prompt"],
            f"three-seed DPO seed {seed} raw-table hash",
            observed_hash,
        )
        seed_frame = pd.read_csv(seed_path)
        audit.check(
            set(seed_frame["prompt_id"]) == split_ids["eval"]
            and seed_frame.groupby("tag").size().eq(200).all()
            and seed_frame["tag"].nunique() == 6,
            f"three-seed DPO seed {seed}: six checkpoints x exact eval prompts",
        )

    checkpoint = pd.read_csv(three_seed_root / "checkpoint_predictions.csv")
    checkpoint_key = ["seed", "tag", "model"]
    audit.check(
        len(checkpoint) == 378 and not checkpoint.duplicated(checkpoint_key).any(),
        "three-seed derived checkpoint table key/cardinality",
        len(checkpoint),
    )
    audit.check(
        set(checkpoint["seed"]) == {0, 1, 2}
        and checkpoint.groupby(["seed", "model"]).size().eq(6).all(),
        "three-seed derived table has six checkpoints per seed/model",
    )
    frontier_drift = np.max(
        np.abs(
            np.abs(checkpoint["prediction"] - checkpoint["frontier"])
            - checkpoint["abs_error_frontier"]
        )
    )
    dpo_drift = np.max(
        np.abs(
            np.abs(checkpoint["prediction"] - checkpoint["dpo_realized"])
            - checkpoint["abs_error_dpo"]
        )
    )
    audit.check(
        frontier_drift < 1e-12,
        "three-seed frontier-error arithmetic",
        frontier_drift,
    )
    audit.check(
        dpo_drift < 1e-12,
        "three-seed realized-DPO-error arithmetic",
        dpo_drift,
    )


def audit_results(root: Path, audit: Audit) -> None:
    print("== Published result contracts ==")
    main_spec = json.loads(
        (root / "results/direct_calibration_strict/main_protocol/run_spec.json")
        .read_text()
    )
    fixed_spec = json.loads(
        (root / "results/direct_calibration_strict/wmt_fixed_141/run_spec.json")
        .read_text()
    )
    main_wmt = next(
        row for row in main_spec["datasets"]
        if row["dataset"] == "WMT-MQM-human_heldout"
    )
    fixed_wmt = fixed_spec["datasets"][0]
    audit.check(
        main_wmt["human_long_sha256"] ==
        EXPECTED_INPUT_HASHES["data/wmt_mqm_long_main_protocol.csv"],
        "submitted-number main_protocol is explicitly submitted_139",
    )
    audit.check(
        fixed_wmt["human_long_sha256"] ==
        EXPECTED_INPUT_HASHES["data/wmt_mqm_long.csv"],
        "new WMT scientific result is explicitly fixed_141",
    )

    curve_main = pd.read_csv(
        root / "results/direct_calibration_strict/main_protocol/curve_mae.csv"
    )
    curve_fixed = pd.read_csv(
        root / "results/direct_calibration_strict/wmt_fixed_141/curve_mae.csv"
    )
    attempts = {
        "SummEval-human_heldout": 1600,
        "SummEval-GPT4o": 200,
        "OpenMEVA-MAGS-human_heldout": 500,
        "WMT-MQM-human_heldout": 6950,
    }
    for dataset, expected in attempts.items():
        rows = curve_main[curve_main["dataset"].eq(dataset)]
        audit.check(
            not rows.empty and rows["n_attempted_units"].eq(expected).all(),
            f"direct-calibration attempts {dataset}",
            expected,
        )
    audit.check(
        curve_fixed["n_attempted_units"].eq(7050).all(),
        "fixed_141 direct-calibration attempts",
        7050,
    )

    evaluation_aware_root = root / "results/evaluation_aware_ab"
    evaluation_aware_manifest = json.loads(
        (evaluation_aware_root / "manifest.json").read_text()
    )
    audit.check(
        evaluation_aware_manifest["canonical_lineages"]["wmt"]["input_sha256"]
        == EXPECTED_INPUT_HASHES["data/wmt_mqm_long.csv"],
        "target-aware WMT input is fixed_141",
    )
    evaluation_aware_summary = pd.read_csv(
        evaluation_aware_root / "evaluation_aware_ab_summary_fixed_141.csv"
    )
    audit.check(
        len(evaluation_aware_summary) == 16
        and set(evaluation_aware_summary["dataset"]) == {
            "SummEval-human_heldout",
            "SummEval-GPT4o",
            "WMT-MQM-human_heldout",
            "OpenMEVA-MAGS-human_heldout",
        },
        "target-aware summary has four datasets x four conditional means",
        len(evaluation_aware_summary),
    )
    common_support_root = (
        evaluation_aware_root / "common_support"
    )
    common_support = json.loads(
        (common_support_root / "evaluation_aware_ab_support.json").read_text()
    )
    common_summary = pd.read_csv(
        common_support_root / "evaluation_aware_ab_summary.csv"
    )
    common_summeval = common_summary.loc[
        (common_summary["dataset"] == "SummEval-human_heldout")
        & (common_summary["conditional_mean_model"] == "isotonic")
    ]
    audit.check(
        common_support["mode"] == "common_support"
        and common_support["n_excluded_radius_cells"] == 11
        and common_support["excluded_cells_by_dataset"]
        == {"SummEval-human_heldout": 11},
        "target-aware common-support mask",
        common_support["n_excluded_radius_cells"],
    )
    audit.check(
        common_support["predictions_sha256"]
        == sha256_file(
            evaluation_aware_root
            / "non_wmt_canonical/direct_calibration_predictions.csv.gz"
        )
        and common_support["support_exclusions_sha256"]
        == sha256_file(
            root
            / "results/direct_calibration_strict/main_protocol/"
            "calibration_only_skipped.csv"
        ),
        "target-aware common-support input hashes",
    )
    audit.check(
        len(common_summeval) == 1
        and np.isclose(
            float(common_summeval.iloc[0]["raw_A+B_curve_mae"]),
            0.011849763157716409,
            atol=1e-15,
            rtol=0,
        ),
        "target-aware common-support SummEval A+B",
        (
            None
            if common_summeval.empty
            else float(common_summeval.iloc[0]["raw_A+B_curve_mae"])
        ),
    )
    common_manifest = evaluation_aware_manifest["common_support"]
    audit.check(
        common_manifest["excluded_radius_cells"] == 11
        and common_manifest["summary_sha256"]
        == sha256_file(
            common_support_root / "evaluation_aware_ab_summary.csv"
        )
        and common_manifest["paired_bootstrap_sha256"]
        == sha256_file(
            common_support_root
            / "evaluation_aware_ab_paired_bootstrap.csv"
        )
        and common_manifest["support_report_sha256"]
        == sha256_file(
            common_support_root / "evaluation_aware_ab_support.json"
        ),
        "target-aware common-support artifact hashes",
    )
    target_baselines = {
        "A+B (isotonic h)",
        "A+B (binned h)",
        "A+B (spline h)",
        "A+B (MLP h)",
    }
    for label, relative, expected_rows, expected_datasets in [
        (
            "non-WMT",
            "non_wmt_canonical/direct_calibration_predictions.csv.gz",
            183989,
            {
                "SummEval-human_heldout",
                "SummEval-GPT4o",
                "OpenMEVA-MAGS-human_heldout",
            },
        ),
        (
            "fixed_141 WMT",
            "wmt_fixed_141/direct_calibration_predictions.csv.gz",
            564000,
            {"WMT-MQM-human_heldout"},
        ),
    ]:
        predictions = pd.read_csv(evaluation_aware_root / relative)
        key = ["dataset", "path_id", "repeat", "d", "baseline"]
        audit.check(
            len(predictions) == expected_rows
            and set(predictions["dataset"]) == expected_datasets,
            f"target-aware {label} row and dataset accounting",
            len(predictions),
        )
        audit.check(
            not predictions.duplicated(key).any(),
            f"target-aware {label} prediction key unique",
        )
        audit.check(
            target_baselines.issubset(set(predictions["baseline"])),
            f"target-aware {label} A+B variants present",
        )
        error_drift = np.max(
            np.abs(
                np.abs(
                    predictions["predicted_gain"] - predictions["actual_gain"]
                )
                - predictions["abs_error"]
            )
        )
        audit.check(
            error_drift < 1e-12,
            f"target-aware {label} error arithmetic",
            error_drift,
        )
    target_wmt_spec = json.loads(
        (
            evaluation_aware_root
            / "wmt_fixed_141/direct_calibration_run_spec.json"
        ).read_text()
    )
    target_contract = target_wmt_spec["evaluation_aware_conditional_mean_ab"]
    audit.check(
        target_contract["evaluation_inputs"]
        == "evaluation proxy scores only; no evaluation gold"
        and target_contract["coefficient_distribution"]
        == "(h(r_eval), r_eval)",
        "target-aware A+B run spec excludes evaluation gold from prediction",
    )
    target_repeat_root = (
        evaluation_aware_root / "wmt_fixed_141_repeat"
    )
    target_canonical_predictions = pd.read_csv(
        evaluation_aware_root
        / "wmt_fixed_141/direct_calibration_predictions.csv.gz"
    )
    target_repeat_predictions = pd.read_csv(
        target_repeat_root / "direct_calibration_predictions.csv.gz"
    )
    audit.check(
        target_canonical_predictions.equals(target_repeat_predictions),
        "independent fixed_141 Exp32 predictions equal canonical values/order",
        len(target_repeat_predictions),
    )
    repeat_exact_files = [
        "direct_calibration_run_spec.json",
        "evaluation_aware_ab_summary.csv",
        "evaluation_aware_ab_paired_bootstrap.csv",
        "EVALUATION_AWARE_AB_RESULTS.md",
    ]
    audit.check(
        all(
            sha256_file(target_repeat_root / name)
            == sha256_file(evaluation_aware_root / "wmt_fixed_141" / name)
            for name in repeat_exact_files
        ),
        "independent fixed_141 Exp32 small artifacts byte-identical",
    )
    repeat_verification = json.loads(
        (target_repeat_root / "reproduction_verification.json").read_text()
    )
    audit.check(
        repeat_verification["status"] == "PASS"
        and repeat_verification["repeat_predictions_payload_sha256"]
        == evaluation_aware_manifest["canonical_lineages"]["wmt"][
            "source_predictions_sha256"
        ]
        and repeat_verification["dimensions"]
        == {
            "rows": 564000,
            "paths": 141,
            "repeats": 50,
            "radii": 5,
            "baselines": 16,
        },
        "independent fixed_141 Exp32 verification report",
        repeat_verification["status"],
    )

    conditional = pd.read_csv(
        root / "results/prompt_conditional/conditional_per_path.csv"
    )
    cond_counts = conditional.groupby("dataset")["path"].nunique().to_dict()
    audit.check(
        cond_counts == {"OpenMEVA": 10, "SummEval": 32, "WMT-MQM": 141},
        "prompt-conditional path counts",
        cond_counts,
    )
    fixed_conditional = pd.read_csv(
        root /
        "results/wmt_input_lineages/fixed_141/conditional/conditional_per_path.csv"
    )
    combined_wmt = (
        conditional[conditional["dataset"].eq("WMT-MQM")]
        .sort_values("path").reset_index(drop=True)
    )
    fixed_conditional = fixed_conditional.sort_values("path").reset_index(drop=True)
    audit.check(
        combined_wmt.equals(fixed_conditional),
        "combined prompt-conditional WMT block equals fixed_141 artifact",
    )

    rank = pd.read_csv(
        root / "results/rank_selection_scope/mixture_ranktilt_rows.csv"
    )
    rank_key = [
        "dataset", "path_id", "repeat", "mechanism", "param_name", "param_value"
    ]
    audit.check(len(rank) == 123420, "rank-selection row count", len(rank))
    audit.check(not rank.duplicated(rank_key).any(), "rank-selection key unique")
    for prediction, error in [
        ("theory_pred", "abs_err_theory"),
        ("aonly_pred", "abs_err_aonly"),
        ("anu_pred", "abs_err_anu"),
        ("empfit_pred", "abs_err_empfit"),
        ("zero_pred", "abs_err_zero"),
    ]:
        drift = np.max(
            np.abs(
                np.abs(rank["delta_realized"] - rank[prediction]) - rank[error]
            )
        )
        audit.check(drift < 1e-12, f"rank-selection arithmetic {error}", drift)

    bon = pd.read_csv(root / "results/bon_scope/bon_gate_points.csv")
    audit.check(len(bon) == 191 * 12, "BoN scope row count", len(bon))
    audit.check(
        not bon.duplicated(["dataset_label", "path_id", "N"]).any(),
        "BoN scope key unique",
    )
    audit.check(
        np.allclose(bon["d_bon"], bon["d_bon_recheck"], atol=1e-12),
        "BoN KL independently stored/rechecked values agree",
    )

    stability = pd.read_csv(root / "results/b_stability/coefficient_level.csv")
    expected_units = {
        "SummEval-human_heldout": (32, 1600),
        "SummEval-GPT4o": (4, 200),
        "WMT-MQM-human_heldout": (141, 7050),
        "OpenMEVA-MAGS-human_heldout": (10, 500),
    }
    observed_units = {
        row.dataset: (int(row.n_paths), int(row.n_path_repeat_units))
        for row in stability.itertuples()
    }
    audit.check(
        observed_units == expected_units,
        "B-stability path/repeat accounting",
        observed_units,
    )

    for directory, expected_paths, expected_rows in [
        ("summeval", 32, 1408),
        ("openmeva_mags", 10, 440),
        ("wmt_fixed_141", 141, 6204),
    ]:
        path = (
            root / "results/tail_outlier_robustness" / directory /
            "tail_outlier_robustness_paths.csv"
        )
        frame = pd.read_csv(path)
        key = ["dataset", "path_id", "condition", "rep"]
        audit.check(len(frame) == expected_rows, f"tail audit {directory} rows", len(frame))
        audit.check(
            frame["path_id"].nunique() == expected_paths,
            f"tail audit {directory} paths",
            frame["path_id"].nunique(),
        )
        audit.check(not frame.duplicated(key).any(), f"tail audit {directory} key unique")
        audit.check(frame["status"].eq("ok").all(), f"tail audit {directory} all status ok")
        drift = np.max(
            np.abs(
                np.abs(frame["actual_delta"] - frame["pred_delta"])
                - frame["abs_err_theory"]
            )
        )
        audit.check(drift < 1e-12, f"tail audit {directory} error arithmetic", drift)

    for proxy_name in ["proxy_openjudge", "proxy_rm_skywork"]:
        paths = pd.read_csv(
            root / "results/open_proxy" / proxy_name /
            "exp27_prospective_paths.csv"
        )
        audit.check(len(paths) == 200, f"{proxy_name} prospective rows", len(paths))
        audit.check(
            not paths.duplicated(["group_key", "split_rep"]).any(),
            f"{proxy_name} prospective split key unique",
        )
        audit.check(
            paths.groupby("group_key").size().eq(50).all(),
            f"{proxy_name} has 50 splits per dimension",
        )

    source_audit = json.loads(
        (root / "verification/pipeline/source_data_fixed_141/"
         "source_data_audit.json").read_text()
    )
    audit.check(
        source_audit["prepared_csv_to_exp33_results"] == "PASS",
        "exp33 prepared-data-to-result source audit",
    )
    audit.check(
        source_audit["source_to_prepared_csv"] == "PASS",
        "exp33 raw-source-to-prepared-data audit",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    report = args.report or (
        root / "verification/pipeline_integrity.json"
    )

    audit = Audit()
    audit_hashes(root, audit)
    audit_base_data(root, audit)
    audit_dpo(root, audit)
    audit_results(root, audit)

    payload = {
        "status": "PASS" if not audit.failures else "FAIL",
        "n_checks": len(audit.checks),
        "n_failures": len(audit.failures),
        "checks": audit.checks,
    }
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n{payload['status']}: {len(audit.checks)} checks, "
          f"{len(audit.failures)} failures")
    print(f"Report: {report}")
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
