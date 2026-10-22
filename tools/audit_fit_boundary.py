#!/usr/bin/env python3
"""Read-only reproduction of the WMT empirical-fit endpoint inclusion issue."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
sys.path.insert(0, str(CORE / "lib"))
from common_experiment_utils import exact_tilt_curve, fit_local_law, generate_human_paths, split_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    folder = CORE / "results/07_prospective_prediction/exp27_wmt_mqm"
    spec = json.loads((folder / "exp27_run_spec.json").read_text())
    human = pd.read_csv(CORE / "data/wmt_mqm_long.csv")
    path = next(generate_human_paths(df=human, item_cols=spec["item_cols"],
        rater_col=spec["rater_col"], score_col=spec["score_col"],
        group_cols=spec["group_cols"], min_items=spec["min_items"], min_raters=spec["min_raters"]))
    cal, eva = split_indices(len(path.item_ids), spec["calibration_frac"], np.random.default_rng(spec["random_seed"]))
    source = CORE / "experiments/07_prospective_prediction/01_prospective_prediction.py"
    module_spec = importlib.util.spec_from_file_location("boundary_endpoint", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    fresh = module.evaluate_one_split(path, cal, eva, spec["fit_max_d"], spec["d_targets"])
    saved = pd.read_csv(folder / "exp27_prospective_paths.csv")
    old = saved[(saved.path_id == path.path_id) & (saved.split_rep == 0)].iloc[0]
    curve = exact_tilt_curve(path.proxy_scores[eva], path.gold_scores[eva], d_max=spec["fit_max_d"])
    inclusive = fit_local_law(curve.d, curve.q, fit_max_d=spec["fit_max_d"] + 1e-12)
    fields = ["eval_a_fit", "eval_b_fit", "eval_r2", "eval_peak_d_fit", "eval_headroom_fit"]
    payload = {
        "diagnostic_only": True,
        "path_id": path.path_id, "split_rep": 0,
        "fit_max_d": spec["fit_max_d"], "last_curve_d": float(curve.d[-1]),
        "strict_mask_retains_endpoint": bool(curve.d[-1] <= spec["fit_max_d"]),
        "numpy": np.__version__,
        "saved_vs_fresh": {k: {"saved": float(old[k]), "fresh": fresh[k]} for k in fields},
        "inclusive_endpoint_fit": {"a": inclusive.a_fit, "b": inclusive.b_fit, "r2": inclusive.r2},
        "inclusive_fit_matches_saved": bool(np.allclose(
            [inclusive.a_fit, inclusive.b_fit, inclusive.r2],
            [old.eval_a_fit, old.eval_b_fit, old.eval_r2], rtol=1e-10, atol=1e-11)),
        "moment_prediction_and_gain_match": bool(all(np.isclose(fresh[k],old[k], rtol=1e-10, atol=1e-11)
            for k in fresh if k.startswith(("pred_theory_delta_", "actual_eval_delta_")))),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
