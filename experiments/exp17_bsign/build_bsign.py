#!/usr/bin/env python3
"""[exp17/R5'] B-sign prediction: balanced accuracy + majority baseline + |B| margin
strata. Raw B-sign accuracy is inflated when the sign distribution is
imbalanced (mostly negative). We report balanced accuracy (mean of per-class recall),
the majority-class baseline it must beat, and accuracy stratified by |eval B| margin.
Task: predict sign(eval_b_theory) from sign(cal_b_theory), per proxy path. Pure CPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = sorted(
    (ROOT / "outputs" / "exp06_final_20260706_022627").glob(
        "*/exp27_prospective_paths.csv"
    )
) + sorted(
    (ROOT / "outputs" / "exp02_fixed_rouge_20260706_020516").glob(
        "*/exp27_prospective_paths.csv"
    )
)


def balanced_acc(true_sign, pred_sign):
    accs = []
    for cls in (-1, 1):
        m = true_sign == cls
        if m.sum() > 0:
            accs.append((pred_sign[m] == cls).mean())
    return float(np.mean(accs)) if accs else np.nan


def analyze(df, label):
    d = df[(df.cal_b_theory != 0) & (df.eval_b_theory != 0)].copy()
    ts = np.sign(d.eval_b_theory.values).astype(int)
    ps = np.sign(d.cal_b_theory.values).astype(int)
    raw = float((ts == ps).mean())
    bal = balanced_acc(ts, ps)
    maj_cls = 1 if (ts == 1).sum() >= (ts == -1).sum() else -1
    maj_acc = float((ts == maj_cls).mean())              # always-majority raw accuracy
    maj_bal = 0.5                                          # balanced acc of a constant predictor
    out = {"setting": label, "n": len(d), "pos": int((ts == 1).sum()), "neg": int((ts == -1).sum()),
           "raw_acc": round(raw, 3), "balanced_acc": round(bal, 3),
           "majority_raw": round(maj_acc, 3), "majority_balanced": maj_bal,
           "beats_majority_balanced": bal > 0.5 + 1e-9}
    # |eval B| margin strata
    mag = d.eval_b_theory.abs().values
    for name, mask in [("|B|<p33", mag <= np.quantile(mag, 1/3)),
                       ("p33-p66", (mag > np.quantile(mag, 1/3)) & (mag <= np.quantile(mag, 2/3))),
                       ("|B|>p66", mag > np.quantile(mag, 2/3))]:
        if mask.sum() > 0:
            out[f"bal_{name}"] = round(balanced_acc(ts[mask], ps[mask]), 3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path).assign(src=path.parent.name) for path in FILES]
    if not frames:
        raise FileNotFoundError(
            "No prospective prediction inputs matched the configured output globs."
        )
    allp = pd.concat(frames, ignore_index=True)
    rows = [analyze(allp, "POOLED (all proxies)")]
    for src, g in allp.groupby("src"):
        rows.append(analyze(g, src))
    res = pd.DataFrame(rows)
    out = args.outdir
    out.mkdir(parents=True, exist_ok=False)
    res.to_csv(out / "bsign_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.to_string(index=False))
    (out / "bsign_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out}/bsign_summary.csv")


if __name__ == "__main__":
    main()
