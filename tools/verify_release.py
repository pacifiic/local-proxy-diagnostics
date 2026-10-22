#!/usr/bin/env python3
"""Check release data contracts, saved summaries, and optional fresh split predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
RESULTS = ROOT / "results"
REPORT = ROOT / "verification/release_checks.json"
checks = []


def check(ok, name, detail=None):
    checks.append(dict(name=name, passed=bool(ok), detail=detail))
    if not ok:
        raise AssertionError(name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-splits', action='store_true',
                        help='Also recompute selected splits, including empirical fits.')
    args = parser.parse_args()
    sys.path.insert(0, str(CORE / "lib"))
    from common_experiment_utils import generate_human_paths, generate_model_paths, split_indices
    engine = load("prospective", CORE / "experiments/07_prospective_prediction/01_prospective_prediction.py")
    for name in ["summeval_long.csv", "openmeva_long.csv", "wmt_mqm_long.csv",
                 "model_proxy_4o_aggregated.csv"]:
        check((ROOT / "data" / name).is_file(), f"prepared input present: {name}")
    check(sha(CORE / "data/wmt_mqm_long.csv") ==
          "6f20c0669ff7e76bef71a584f6c627dadab063d560d931cae74acff78d8cbbf8",
          "corrected WMT pinned hash")
    check(sha(ROOT / "verification/wmt_mqm_rebuilt.csv") == sha(CORE / "data/wmt_mqm_long.csv"),
          "fresh raw WMT rebuild byte identity")
    check(not [p for p in ROOT.rglob("*") if p.is_symlink() and not p.exists()],
          "no broken internal symlinks")
    for script in [ROOT / "experiments/data_preparation/prepare_wmt_mqm_long.py"]:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "partial.csv"
            proc = subprocess.run([sys.executable, str(script), "--root", temp,
                                   "--outpath", str(out)], capture_output=True, text=True)
            check(proc.returncode != 0 and not out.exists() and "Incomplete WMT" in proc.stderr,
                  f"missing-input guard: {script.relative_to(ROOT)}")
        before = sha(CORE / "data/wmt_mqm_long.csv")
        with tempfile.TemporaryDirectory() as temp:
            # The overwrite check runs before parsing. Empty source fixtures
            # exercise that guard without redistributing the raw benchmark.
            preparation = load("preparation_guard", script)
            for relative in preparation.TSV_FILES.values():
                fixture = Path(temp) / relative
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.touch()
            proc = subprocess.run([sys.executable, str(script), "--root", temp,
                                   "--outpath", str(CORE / "data/wmt_mqm_long.csv")],
                                  capture_output=True, text=True)
            check(proc.returncode != 0 and "Refusing to overwrite" in proc.stderr
                  and before == sha(CORE / "data/wmt_mqm_long.csv"),
                  f"overwrite guard: {script.relative_to(ROOT)}")
    runs = ["exp27_human_baseline", "exp27_model_proxy_4o",
            "exp27_wmt_mqm", "exp27_openmeva"]
    for run in runs:
        folder = RESULTS / "07_prospective_prediction" / run
        spec = json.loads((folder / "exp27_run_spec.json").read_text())
        human = pd.read_csv(CORE / "data" / Path(spec["human_long_csv"]).name)
        if spec["proxy_csv"]:
            proxy = pd.read_csv(CORE / "data/model_proxy_4o_aggregated.csv")
            paths = list(generate_model_paths(human_long=human, proxy_df=proxy,
                item_cols=spec["item_cols"], score_col=spec["score_col"],
                group_cols=spec["group_cols"], proxy_item_col=spec["proxy_item_col"],
                proxy_score_col=spec["proxy_score_col"], min_items=spec["min_items"]))
        else:
            paths = list(generate_human_paths(df=human, item_cols=spec["item_cols"],
                rater_col=spec["rater_col"], score_col=spec["score_col"],
                group_cols=spec["group_cols"], min_items=spec["min_items"],
                min_raters=spec["min_raters"]))
        saved = pd.read_csv(folder / "exp27_prospective_paths.csv")
        check(len(paths) == spec["n_paths"], f"path cardinality: {run}", len(paths))
        check(not saved.duplicated(["path_id", "split_rep"]).any(), f"unique split keys: {run}")
        if args.fresh_splits:
            rng = np.random.default_rng(spec["random_seed"])
            largest, count = 0., 0
            for index, path in enumerate(paths):
                for rep in range(spec["n_repeats"]):
                    cal, eva = split_indices(len(path.item_ids), spec["calibration_frac"], rng)
                    if index not in {0, len(paths)//2, len(paths)-1} or rep not in {0, 1, 49}:
                        continue
                    old = saved[(saved.path_id == path.path_id) & (saved.split_rep == rep)]
                    try:
                        row = engine.evaluate_one_split(path, cal, eva, spec["fit_max_d"], spec["d_targets"])
                    except (RuntimeError, ValueError):
                        check(old.empty, f"expected skipped split: {run}/{index}/{rep}")
                        continue
                    check(len(old) == 1, f"fresh split has saved row: {run}/{index}/{rep}")
                    for key, value in row.items():
                        if isinstance(value, (int, float, np.number)) and key in old:
                            reference = float(old.iloc[0][key])
                            if np.isnan(value) and np.isnan(reference):
                                continue
                            if value == reference:
                                continue
                            check(np.isclose(value, reference, rtol=1e-10, atol=1e-11, equal_nan=True),
                                  f"fresh numerical equality: {run}/{index}/{rep}/{key}")
                            if np.isfinite(value) and np.isfinite(reference):
                                largest = max(largest, abs(value-reference))
                    count += 1
            check(count > 0, f"fresh endpoint recomputation: {run}",
                  {"rows": count, "max_abs_difference": largest})
        summary = pd.read_csv(folder / "exp27_prospective_summary.csv")
        overall = summary[summary.group_key == "__OVERALL__"].iloc[0]
        for method in ["theory", "aonly", "anu", "zero"]:
            column = f"abs_err_{method}_0p1"
            check(np.isclose(saved[column].mean(), overall[f"mae_{method}_0p1"]),
                  f"raw rows -> overall MAE: {run}/{method}")
    # Recompute every endpoint/sweep CI from saved predictions, not cached summaries.
    subprocess.run([sys.executable, str(ROOT / "tools/recompute_prediction_intervals.py")], check=True)
    check("evaluation exact coefficient" not in
          (CORE / "artifacts/tables/table_sign_metric_audit.tex").read_text(), "fitted B label")



if __name__ == "__main__":
    try:
        main()
    finally:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(checks, indent=2) + "\n")
    print(f"{len(checks)} release checks passed")
