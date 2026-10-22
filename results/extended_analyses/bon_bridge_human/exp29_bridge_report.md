# BoN bridge report

## Executive summary

This run uses two KL windows:
- shared local-law window: `d <= 0.100`
- BoN bridge window: `d <= 0.250`

Across all processed paths:
- median share of **all** BoN points in the shared window: `0.143`
- median share of **nontrivial** BoN points in the shared window: `0.000`
- median share of **nontrivial** BoN points in the bridge window: `0.143`
- fraction of paths with **no nontrivial** BoN points inside the shared window: `1.000`
- median KL of the first nontrivial BoN step (`N=2`): `0.172`

## Interpretation

If the first nontrivial BoN step already lands above the shared local-law window, then a lack of shared-window coverage is best interpreted as a **protocol mismatch between discrete BoN selection and the chosen local KL window**, rather than as evidence against the local law itself.

The bridge-window columns are designed to make this distinction explicit.
