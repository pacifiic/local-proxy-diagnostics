"""Generate standardized skew-normal proxy/target pairs for a skewness audit.

The target is alpha + beta*Z + gamma*(Z**2 - 1), without added noise.
Changing the skew-normal shape also changes higher moments, so comparisons
are family-level diagnostics, not isolated causal effects of skewness.
Location/positive-scale transformations check coefficient invariance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "synthetic"

N_ITEMS = 20000
SEED = 20260723

# skewnorm shape parameter 'a' grid: a=0 -> symmetric(Gaussian), |a| large -> highly skewed.
# a values chosen to span the empirical skewness range we later measure on real datasets
# (see audit_skewcorr.py real-dataset overlay), plus symmetric/extreme anchors.
SKEWNORM_A_GRID = [-30.0, -8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0, 30.0]

# Fixed conditional-mean shape (quadratic in standardized Z). The distribution
# of Z still changes in higher moments across settings.
ALPHA = 0.0
BETA = 1.3      # linear (first-order alignment) term
GAMMA = 0.6     # curvature term feeding E[J̄Z^2]

# Arbitrary non-zero location/scale for r, to also test that (A,B) estimation is invariant
# to affine reparametrization of the proxy (a basic sanity property, not the paper's main claim).
MU_R = 5.0
SIGMA_R = 2.0


def standardize(z: np.ndarray) -> np.ndarray:
    z = z - np.mean(z)
    z = z / np.std(z, ddof=0)
    return z


def make_setting(a: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z_raw = stats.skewnorm.rvs(a, size=n, random_state=rng)
    z = standardize(z_raw)  # exact E[Z]=0, Var[Z]=1 in this finite sample by construction
    j = ALPHA + BETA * z + GAMMA * (z**2 - 1.0)
    r = MU_R + SIGMA_R * z
    df = pd.DataFrame({
        "item_id": np.arange(n),
        "z_latent": z,
        "r": r,
        "J": j,
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for a in SKEWNORM_A_GRID:
        df = make_setting(a, N_ITEMS, seed=SEED + int(round(a * 100)))
        tag = f"a{a:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
        path = out_dir / f"synthetic_{tag}.csv"
        df.to_csv(path, index=False)
        manifest.append({
            "setting": tag,
            "skewnorm_a": a,
            "n_items": N_ITEMS,
            "seed": SEED + int(round(a * 100)),
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "mu_r": MU_R,
            "sigma_r": SIGMA_R,
            "path": path.name,
            # Recorded here for a quick manifest-level sanity read; audit_skewcorr.py
            # recomputes this independently from the saved CSV rather than trusting this value.
            "empirical_skew_z_at_build_time": float(stats.skew(df["z_latent"].to_numpy())),
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} settings -> {out_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
