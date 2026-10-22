#!/usr/bin/env python3
"""Reaggregate saved prediction CIs without rerunning or changing predictions.

Use --apply to update current summaries after all point statistics and support
checks pass. The default is a read-only consistency check. Both modes use the
experiment implementations, the original path rows, and the recorded seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
RESULTS = ROOT / "results"
sys.path.insert(0, str(CORE / "lib"))
sys.path.insert(0, str(CORE / "experiments/12_accept_boost"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.allclose(a, b, atol=5e-13, rtol=5e-13, equal_nan=True))
    return a.fillna("<NA>").astype(str).equals(b.fillna("<NA>").astype(str))


def merged_intervals(old, new, keys):
    """Keep every stored point statistic; replace only interval columns."""
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError(f"Duplicate summary keys: {keys}")
    left, right = old.set_index(keys), new.set_index(keys)
    if len(left) != len(right) or set(left.index) != set(right.index):
        raise ValueError("Reaggregation changed the summary support")
    right = right.reindex(left.index)
    if set(left.columns) != set(right.columns):
        raise ValueError("Reaggregation changed the summary columns")
    merged, changed = left.copy(), []
    for col in left.columns:
        if col.endswith(("_ci_lo", "_ci_hi")):
            if np.isinf(right[col].to_numpy(dtype=float)).any() or not left[col].isna().equals(right[col].isna()):
                raise ValueError(f"Invalid interval support: {col}")
            if not same(left[col], right[col]):
                merged[col] = right[col]
                changed.append(col)
        elif not same(left[col], right[col]):
            raise ValueError(f"Point statistic changed: {col}")
    return merged.reset_index()[old.columns], changed


def validate_rows(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate prediction key: {keys}")
    errors = frame[[c for c in frame if c.startswith("abs_err_")]]
    if np.isinf(errors.to_numpy(dtype=float)).any():
        raise ValueError("Infinite prediction error")
    if (errors < 0).any().any():
        raise ValueError("Negative absolute prediction error")
    for col in errors:
        name = col.removeprefix("abs_err_")
        if "actual_delta" in frame:
            pred, actual = f"pred_{name}", "actual_delta"
        else:
            method, suffix = name.rsplit("_", 1)
            pred, actual = f"pred_{method}_delta_{suffix}", f"actual_eval_delta_{suffix}"
        if pred in frame and actual in frame:
            expected = (frame[pred] - frame[actual]).abs()
            if not np.allclose(expected, frame[col], rtol=5e-11, atol=5e-12, equal_nan=True):
                raise ValueError(f"Prediction/target/error mismatch: {col}")


def candidates():
    endpoint = load("interval_endpoint", CORE / "experiments/07_prospective_prediction/01_prospective_prediction.py")
    sweep = load("interval_sweep", CORE / "experiments/12_accept_boost/accept_boost_utils.py")
    for folder in sorted((RESULTS / "07_prospective_prediction").glob("exp27_*")):
        source = folder / "exp27_prospective_paths.csv"
        summary = folder / "exp27_prospective_summary.csv"
        if not summary.exists():
            continue
        frame = pd.read_csv(source)
        validate_rows(frame, ["path_id", "split_rep"])
        radii = sorted(float(c.removeprefix("actual_eval_delta_").replace("p", "."))
                       for c in frame if c.startswith("actual_eval_delta_"))
        spec = json.loads((folder / "exp27_run_spec.json").read_text())
        n_boot = int(spec.get("n_boot", 1000))
        # The endpoint implementation uses bootstrap seed 0 independently of split seeds.
        yield source, summary, endpoint.summarize_results(frame, radii, n_boot=n_boot), ["group_key"], n_boot, 0

    for stem, raw, spec_name, groups in [
        ("d_sweep", "d_sweep_paths_long.csv", "d_sweep_run_spec.json", ["group_key", "d"]),
        ("calibration_size", "calibration_size_paths_long.csv", "calibration_size_run_spec.json",
         ["group_key", "calibration_frac", "d"]),
    ]:
        for source in sorted(RESULTS.rglob(raw)):
            folder = source.parent
            spec = json.loads((folder / spec_name).read_text())
            n_boot, seed = int(spec.get("n_boot", 1000)), int(spec["random_seed"])
            frame = pd.read_csv(source)
            validate_rows(frame, ["path_id", "repeat"] + groups[1:])
            predictors = spec["predictors"]
            for scope in ["by_group", "overall"]:
                data = frame if scope == "by_group" else frame.assign(group_key="__OVERALL__")
                fresh = sweep.summarize_prediction_long(data, group_cols=groups, predictors=predictors,
                                                       n_boot=n_boot, seed=seed)
                yield source, folder / f"{stem}_summary_{scope}.csv", fresh, groups, n_boot, seed


def refresh_sweep_figures():
    import os
    import tempfile
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    dplot = load("interval_d_plot", CORE / "experiments/12_accept_boost/01_d_sweep_prediction.py")
    cplot = load("interval_cal_plot", CORE / "experiments/12_accept_boost/03_calibration_size_curve.py")
    for stem, spec_name in [("d_sweep", "d_sweep_run_spec.json"),
                            ("calibration_size", "calibration_size_run_spec.json")]:
        for summary in sorted(RESULTS.rglob(f"{stem}_summary_overall.csv")):
            spec = json.loads((summary.parent / spec_name).read_text())
            predictors = [p for p in spec["predictors"] if p != "oracle"]
            data = pd.read_csv(summary)
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                if stem == "d_sweep":
                    dplot.plot_mae(data, out, predictors)
                    dplot.plot_sign(data, out, predictors)
                else:
                    for radius in sorted(data.d.unique()):
                        cplot.plot_calibration_curve(data, out, float(radius), predictors)
                for generated in out.iterdir():
                    target = summary.parent / generated.name
                    if not target.exists() or sha(target) != sha(generated):
                        shutil.copy2(generated, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Replace changed interval summaries after validation.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    jobs, records = [], []
    for source, summary, fresh, keys, n_boot, seed in candidates():
        old = pd.read_csv(summary)
        merged, columns = merged_intervals(old, fresh, keys)
        entry = {"source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
                 "summary": str(summary.relative_to(ROOT)), "bootstrap_draws": n_boot,
                 "seed": seed, "interval_columns_changed": columns}
        records.append(entry)
        jobs.append((source, summary, old, merged, entry))
    # Validate all inputs before updating any output.
    for source, summary, old, merged, entry in jobs:
        if args.apply and entry["interval_columns_changed"]:
            merged.to_csv(summary, index=False)
            if summary.with_suffix(".md").exists():
                summary.with_suffix(".md").write_text(merged.to_markdown(index=False) + "\n")
        entry["summary_sha256"] = sha(summary)
        if sha(source) != entry["source_sha256"]:
            raise RuntimeError(f"Prediction source changed during reaggregation: {source}")
    report = {"estimand": "mean over valid split rows", "cluster_unit": "path",
              "resampling": "sample paths with replacement, retaining all split rows per sampled path",
              "point_estimates_unchanged": True, "prediction_files_unchanged": True,
              "numpy": np.__version__, "pandas": pd.__version__, "summaries": records}
    if args.apply:
        refresh_sweep_figures()
    changed = sum(bool(r["interval_columns_changed"]) for r in records)
    destination = args.report
    if args.apply:
        destination = destination or ROOT / "provenance/prediction_interval_reaggregation.json"
        # Idempotent invocations must not erase the original before/after record.
        if changed == 0 and destination.exists():
            destination = None
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Checked {len(records)} summaries; {changed} {'updated' if args.apply else 'stale'}; predictions unchanged.")
    if changed and not args.apply:
        raise SystemExit("Stale intervals: run this command with --apply before rebuilding artifacts.")


if __name__ == "__main__":
    main()
