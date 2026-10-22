#!/usr/bin/env python3
"""[exp13/R1'] Paired path-level MAE differences with sign tests and cluster
bootstrap CIs — a path-clustered uncertainty analysis.

Recorded paired-comparison protocol:
  unit      = independent proxy-gold path (SummEval 32, WMT 139, OpenMEVA 10, GPT-4o 4,
              model proxies 4 each); random resplits are AVERAGED WITHIN path first,
              never counted as independent samples.
  cal_frac  = 0.30 for the exp03 tracks (paper's main setting); d = 0.1 for exp27 files.
  contrasts = theory(A+B) minus {aonly, empfit, anu/zero where available} on per-path MAE.
  stats     = per-path win/tie/loss + exact binomial sign test (ties dropped) +
              path-bootstrap percentile CI (B=10000, seed 0). Sources with <8 paths get
              the CI flagged SMALL-N: interpret via the per-path table, not the CI.
  honesty   = CIs covering 0 are reported as-is ("value of A+B is decomposition, not
              dominance"); no source is excluded post hoc.

DATA-LAYER SELF-CHECKS (hard-fail):
  exp27 files : re-derive pred_aonly == cal_a_exact*sqrt(d) and
                pred_theory == cal_a_exact*sqrt(d)+cal_b_theory*d row-wise — proves our
                reading of the columns, not just that code runs.
  splits files: abs_err_zero == |actual_delta| row-wise (semantic anchor of the
                abs_err_* columns); path counts match the reported analysis; and our
                within-cal_frac aggregation reproduces the published merged_*.csv
                abs_err_theory_mean for the same setting (provenance of splits->merged).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

B_BOOT = 10000
SEED = 0
SMALL_N = 8

FAIL = []


def ck(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def paired_stats(per_path: pd.DataFrame, base: str, other: str, rng) -> dict:
    """per_path: index=path_id, columns = MAE per predictor. diff = base - other (<0 => base better)."""
    d = (per_path[base] - per_path[other]).dropna()
    n = len(d)
    wins = int((d < 0).sum()); losses = int((d > 0).sum()); ties = n - wins - losses
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses > 0 else np.nan
    idx = rng.integers(0, n, (B_BOOT, n))
    boots = d.values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"contrast": f"{base}-vs-{other}", "n_paths": n,
            "mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "wins": wins, "ties": ties, "losses": losses,
            "sign_p": float(p), "small_n": n < SMALL_N,
            "significant": bool(hi < 0 or lo > 0) and n >= SMALL_N}


def load_exp03_track(path: str, cal_frac: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    ck("cal_frac" in df.columns, f"{Path(path).name}: has cal_frac")
    sub = df[np.isclose(df.cal_frac, cal_frac)].copy()
    ck(len(sub) > 0, f"{Path(path).name}: rows at cal_frac={cal_frac} ({len(sub)})")
    # semantic anchor: zero predictor's error must equal |actual gain| row-wise
    if "abs_err_zero" in sub.columns:
        ck(bool(np.allclose(sub.abs_err_zero, sub.actual_delta.abs(), atol=1e-9)),
           f"{Path(path).name}: abs_err_zero == |actual_delta| (column semantics)")
    cols = {"theory": "abs_err_theory", "aonly": "abs_err_aonly",
            "empfit": "abs_err_empfit", "zero": "abs_err_zero"}
    per_path = sub.groupby("path_id").agg(**{k: (v, "mean") for k, v in cols.items()
                                             if v in sub.columns})
    per_path["n_reps"] = sub.groupby("path_id").size()
    return per_path


def load_exp27(path: str, d: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    tag = str(d).replace(".", "p").replace("0p", "0p")
    sfx = f"delta_{str(d).replace('.', 'p')}"
    need = [f"pred_theory_{sfx}", f"pred_aonly_{sfx}", f"pred_empfit_{sfx}",
            f"pred_anu_{sfx}", f"actual_eval_{sfx}"]
    for c in need:
        ck(c in df.columns, f"{Path(path).name}: has {c}")
    # column-semantics re-derivation (the data-layer check):
    re_aonly = df.cal_a_exact * np.sqrt(d)
    re_theory = df.cal_a_exact * np.sqrt(d) + df.cal_b_theory * d
    ck(bool(np.allclose(df[f"pred_aonly_{sfx}"], re_aonly, atol=1e-9)),
       f"{Path(path).name}: pred_aonly == A_cal*sqrt(d) (re-derived)")
    ck(bool(np.allclose(df[f"pred_theory_{sfx}"], re_theory, atol=1e-9)),
       f"{Path(path).name}: pred_theory == A*sqrt(d)+B*d (re-derived)")
    act = df[f"actual_eval_{sfx}"]
    err = pd.DataFrame({
        "path_id": df.path_id,
        "theory": (df[f"pred_theory_{sfx}"] - act).abs(),
        "aonly": (df[f"pred_aonly_{sfx}"] - act).abs(),
        "empfit": (df[f"pred_empfit_{sfx}"] - act).abs(),
        "anu": (df[f"pred_anu_{sfx}"] - act).abs(),
        "zero": act.abs(),
    })
    per_path = err.groupby("path_id").mean()
    per_path["n_reps"] = err.groupby("path_id").size()
    return per_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--cal-frac", type=float, default=0.30)
    ap.add_argument("--d", type=float, default=0.1)
    ap.add_argument(
        "--wmt-splits",
        default=None,
        help="Optional WMT exp03 split CSV; defaults to the original paper run.",
    )
    ap.add_argument("--expected-wmt-paths", type=int, default=139)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    R = Path(args.repo_root); O = R / "outputs"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    SOURCES = [
        # (label, kind, path, expected_paths)
        ("SummEval human-heldout", "exp03", O / "exp03_boost50_20260706_004007/splits_human_boost50.csv", 32),
        (
            "WMT-MQM human-heldout",
            "exp03",
            (
                Path(args.wmt_splits)
                if args.wmt_splits
                else O / "exp03_boost50_20260706_004007/splits_wmt_boost50.csv"
            ),
            args.expected_wmt_paths,
        ),
        ("OpenMEVA human-heldout", "exp03", O / "exp03_boost50_20260706_004007/splits_openmeva_boost50.csv", 10),
        # NOTE: the exp03 'gpt' track and exp06 exp27 gpt4o file are the SAME computation
        # (per-path MAE identical to 4e-7; verified in the loop-2 audit) — including both
        # would double-count. Keep the exp27 version (has the A_nu column).
        ("GPT-4o judge (exp27)", "exp27", O / "exp06_final_20260706_022627/proxy_gpt4o/exp27_prospective_paths.csv", 4),
        ("Open judge (exp27)", "exp27", O / "exp06_final_20260706_022627/proxy_openjudge/exp27_prospective_paths.csv", 4),
        ("Skywork RM (exp27)", "exp27", O / "exp06_final_20260706_022627/proxy_rm_skywork/exp27_prospective_paths.csv", 4),
        ("ROUGE-1 (exp27)", "exp27", O / "exp02_fixed_rouge_20260706_020516/metric_rouge1/exp27_prospective_paths.csv", 4),
        ("ROUGE-2 (exp27)", "exp27", O / "exp02_fixed_rouge_20260706_020516/metric_rouge2/exp27_prospective_paths.csv", 4),
        ("ROUGE-L (exp27)", "exp27", O / "exp02_fixed_rouge_20260706_020516/metric_rougeL/exp27_prospective_paths.csv", 4),
    ]

    print("== data-layer self-checks ==")
    tables, rows = {}, []
    for label, kind, path, n_expect in SOURCES:
        per_path = (load_exp03_track(str(path), args.cal_frac) if kind == "exp03"
                    else load_exp27(str(path), args.d))
        ck(len(per_path) == n_expect,
           f"{label}: path count {len(per_path)} == reported path count {n_expect}")
        tables[label] = per_path
        for other in [c for c in ("aonly", "empfit", "anu", "zero") if c in per_path.columns]:
            rows.append({"source": label, **paired_stats(per_path, "theory", other, rng)})
        per_path.to_csv(out / f"per_path_{label.split(' ')[0].lower().replace('-','')}_{kind}.csv")

    # provenance check: the published merged summary is the PATH-BALANCED mean (audit
    # finding) — exactly our unit convention. Reproduce it exactly, not within tolerance.
    merged = pd.read_csv(O / "exp03_boost50_20260706_004007/merged_human_boost50.csv")
    pub = float(merged[np.isclose(merged.cal_frac, args.cal_frac)].abs_err_theory_mean.iloc[0])
    splits = pd.read_csv(O / "exp03_boost50_20260706_004007/splits_human_boost50.csv")
    sub = splits[np.isclose(splits.cal_frac, args.cal_frac)]
    ours_pb = float(sub.groupby("path_id").abs_err_theory.mean().mean())
    ck(abs(ours_pb - pub) < 1e-9,
       f"provenance: path-balanced mean {ours_pb:.6f} == merged summary {pub:.6f} "
       f"(published aggregation IS path-balanced — matches our unit)")

    res = pd.DataFrame(rows)
    res.to_csv(out / "paired_stats_summary.csv", index=False)
    print("\n== paired path-level results (diff = MAE_theory - MAE_other; <0 favors A+B) ==")
    with pd.option_context("display.width", 200):
        show = res[["source", "contrast", "n_paths", "mean_diff", "ci_lo", "ci_hi",
                    "wins", "ties", "losses", "sign_p", "small_n", "significant"]].copy()
        for c in ("mean_diff", "ci_lo", "ci_hi"):
            show[c] = show[c].round(4)
        show["sign_p"] = show["sign_p"].round(3)
        print(show.to_string(index=False))

    json.dump({"cal_frac": args.cal_frac, "d": args.d, "B_boot": B_BOOT, "seed": SEED,
               "small_n_threshold": SMALL_N,
               "sources": {l: len(t) for l, (t) in tables.items()}},
              open(out / "run_spec.json", "w"), indent=2)
    print(f"\nwrote {out}/paired_stats_summary.csv, per_path_*.csv, run_spec.json")
    if FAIL:
        raise SystemExit(f"SELF-CHECKS FAILED ({len(FAIL)}): " + "; ".join(FAIL))


if __name__ == "__main__":
    main()
