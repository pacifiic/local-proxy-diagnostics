#!/usr/bin/env python3
"""[exp15/R3'] Concede empfit's MAE parity, then differentiate on TRANSFER + what
empfit cannot provide. The comparison tests whether an empirical two-parameter fit ties
A+B and is deployable — why the moment formula?"

We concede: on a single policy's own frontier, empfit (LS fit of A√d+Bd to the
calibration conditional-tilt curve) matches the moment law's MAE. The moment law's
value is not lower single-policy MAE; it is:
  (1) TRANSFER — moment coefficients are policy-conditional MOMENTS with a definite
      meaning (A=√2 cov(J,r)/σ_r); empfit coefficients are curve summaries with no
      cross-policy meaning. We test Qwen-calibrated coefficients predicting the
      Zephyr frontier, moment vs empfit head to head.
  (2) DECOMPOSITION — A vs A_nu isolates the alignment cross-term ρ; empfit gives no
      such readout.
  (3) A DATA-DRIVEN FALLBACK RULE (R4') and a CURVATURE DIAGNOSTIC (exp09-18), both
      of which need the moment structure; empfit supports neither.

Pure CPU. Self-check: moment coefficients recomputed on calibration reproduce the
locked coefficients.json byte-for-byte.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
from common_experiment_utils import theory_coefficients  # noqa: E402
_s = importlib.util.spec_from_file_location(
    "ps12", HERE.parent / "exp09_dpo_validation" / "12_poolsize_robustness.py")
ps = importlib.util.module_from_spec(_s); _s.loader.exec_module(ps)

D_GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])


def moment_AB(pool):
    a, b, anu = [], [], []
    for pid, (r, j) in pool.items():
        if len(r) < 8 or np.var(r) == 0 or np.var(j) == 0:
            continue
        c = theory_coefficients(j, r)
        a.append(c.a_exact); b.append(c.b_theory); anu.append(c.a_nu)
    return float(np.mean(a)), float(np.mean(b)), float(np.mean(anu))


def frontier_curve(pool, d_grid):
    Q = np.zeros(len(d_grid)); n = 0
    for pid, (r, j) in pool.items():
        n += 1; base = j.mean()
        for k, d in enumerate(d_grid):
            t = ps.find_t(r, d)
            Q[k] += float(np.sum(ps.tilt_w(r, t) * j) - base)
    return Q / max(n, 1)


def empfit_AB(pool, d_grid):
    """LS fit of A*sqrt(d)+B*d to the calibration conditional-tilt frontier curve."""
    Q = frontier_curve(pool, d_grid)
    X = np.column_stack([np.sqrt(d_grid), d_grid])
    a, b = np.linalg.lstsq(X, Q, rcond=None)[0]
    return float(a), float(b)


def mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-run", required=True)
    ap.add_argument("--zephyr-run", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    qcal, qev = ps.load_pool(Path(args.qwen_run) / "pool_scored.jsonl")
    zcal, zev = ps.load_pool(Path(args.zephyr_run) / "pool_scored.jsonl")
    locked_q = json.load(open(Path(args.qwen_run) / "coefficients.json"))

    # self-check: moment on Qwen cal reproduces the lock
    Aq, Bq, Anuq = moment_AB(qcal)
    if abs(Aq - locked_q["A"]) > 1e-9 or abs(Bq - locked_q["B"]) > 1e-9:
        raise SystemExit(f"SELF-CHECK: moment {Aq:.4f},{Bq:.4f} != locked "
                         f"{locked_q['A']:.4f},{locked_q['B']:.4f}")
    print(f"[self-check] moment reproduces locked Qwen coefficients (A={Aq:.4f} B={Bq:.4f}) — PASS\n")
    Az, Bz, Anuz = moment_AB(zcal)
    Aef_q, Bef_q = empfit_AB(qcal, D_GRID)
    Aef_z, Bef_z = empfit_AB(zcal, D_GRID)

    def pred(A, B):
        return A * np.sqrt(D_GRID) + B * D_GRID

    # (a) within-policy parity: empfit vs moment on each policy's OWN eval frontier
    q_front = frontier_curve(qev, D_GRID)
    z_front = frontier_curve(zev, D_GRID)
    print("=== (a) within-policy parity (MAE vs own eval frontier) ===")
    print(f"Qwen  : moment A+B {mae(pred(Aq,Bq), q_front):.4f} | empfit {mae(pred(Aef_q,Bef_q), q_front):.4f} "
          f"| A-only {mae(pred(Aq,0), q_front):.4f} | A_nu {mae(pred(Anuq,0), q_front):.4f}")
    print(f"Zephyr: moment A+B {mae(pred(Az,Bz), z_front):.4f} | empfit {mae(pred(Aef_z,Bef_z), z_front):.4f} "
          f"| A-only {mae(pred(Az,0), z_front):.4f} | A_nu {mae(pred(Anuz,0), z_front):.4f}")

    # (b) TRANSFER duel: Qwen-calibrated coefficients predicting the ZEPHYR frontier
    print("\n=== (b) transfer duel: Qwen-calibrated -> Zephyr eval frontier (MAE) ===")
    rows = {
        "Zephyr-within moment": mae(pred(Az, Bz), z_front),
        "Zephyr-within empfit": mae(pred(Aef_z, Bef_z), z_front),
        "Qwen-transfer moment": mae(pred(Aq, Bq), z_front),
        "Qwen-transfer empfit": mae(pred(Aef_q, Bef_q), z_front),
        "zero": mae(np.zeros(len(D_GRID)), z_front),
    }
    for k, v in rows.items():
        print(f"  {k:22s}: {v:.4f}")
    tm, te = rows["Qwen-transfer moment"], rows["Qwen-transfer empfit"]
    print(f"\n  transfer moment vs empfit: {tm:.4f} vs {te:.4f} "
          f"({'moment better' if tm < te else 'empfit better'} by {abs(tm-te):.4f})")

    coeffs = {"Qwen": {"moment": [Aq, Bq], "empfit": [Aef_q, Bef_q], "A_nu": Anuq},
              "Zephyr": {"moment": [Az, Bz], "empfit": [Aef_z, Bef_z], "A_nu": Anuz}}
    print("\n=== coefficients ===")
    for pol, c in coeffs.items():
        print(f"  {pol}: moment A={c['moment'][0]:.4f} B={c['moment'][1]:+.4f} | "
              f"empfit A={c['empfit'][0]:.4f} B={c['empfit'][1]:+.4f} | A_nu={c['A_nu']:.4f}")
    json.dump({"within": {"qwen": {"moment": mae(pred(Aq,Bq), q_front), "empfit": mae(pred(Aef_q,Bef_q), q_front)},
                          "zephyr": {"moment": mae(pred(Az,Bz), z_front), "empfit": mae(pred(Aef_z,Bef_z), z_front)}},
               "transfer": rows, "coeffs": coeffs, "d_grid": D_GRID.tolist()},
              open(out / "empfit_duel.json", "w"), indent=2)
    print(f"\nwrote {out}/empfit_duel.json")


if __name__ == "__main__":
    main()
