# exp33 Results: Fair Calibration-Only Flexible Baselines

## Question

The exp21 direct isotonic, binned, spline, and MLP baselines fit `h(r)` on
calibration labels but integrate it over the held-out evaluation proxy
distribution. They are target-pool-aware and therefore use more information
than the original calibration-only `A+B` estimator.

This experiment removes that advantage. Each flexible model is fitted and
integrated on the calibration split only. The resulting gain curve is frozen
before evaluation proxy scores or gold labels are observed.

An operational example is a decision made from 300 rated items about whether
to collect and optimize a future pool of 1,000 candidates. Because those
1,000 candidates do not yet exist, a fair pre-pool predictor cannot use their
proxy-score distribution. The strict comparison below gives every method only
the 300-item calibration information.

## Protocol

- datasets: SummEval human-heldout, GPT-4o proxy, corrected 141-path WMT MQM,
  and OpenMEVA-MAGS;
- 30% calibration and 70% evaluation;
- 50 fixed splits per path;
- `d in {0.01, 0.025, 0.05, 0.075, 0.10}`;
- identical item splits and shared-baseline predictions to canonical exp21;
- path-clustered bootstrap with 2,000 draws.

## Fair Paired CurveMAE

The table uses the exact cells shared by `A+B` and each calibration-only
baseline. SummEval has 11 unreachable calibration-radius cells for methods
that must integrate a calibration frontier.

| Dataset | A+B | Isotonic | Binned | Spline | MLP | Empirical curve |
|---|---:|---:|---:|---:|---:|---:|
| SummEval human | 0.01185 | 0.01168 | 0.03010 | 0.01275 | 0.01182 | 0.01171 |
| GPT-4o proxy | 0.00806 | 0.00827 | 0.00878 | 0.00803 | 0.00814 | 0.00805 |
| WMT MQM (`fixed_141`) | 0.05921 | 0.05882 | 0.07356 | 0.06255 | 0.05916 | 0.05842 |
| OpenMEVA-MAGS | 0.00991 | 0.00984 | 0.01583 | 0.01002 | 0.00985 | 0.00984 |

## Paired Uncertainty

`A+B minus baseline` CurveMAE differences:

| Dataset | Isotonic difference [95% CI] | MLP difference [95% CI] |
|---|---:|---:|
| SummEval human | +0.00017 [-0.00002, +0.00045] | +0.00003 [-0.00016, +0.00032] |
| GPT-4o proxy | -0.00022 [-0.00042, -0.00001] | -0.00008 [-0.00025, +0.00009] |
| WMT MQM (`fixed_141`) | +0.00039 [+0.00009, +0.00070] | +0.00005 [-0.00041, +0.00047] |
| OpenMEVA-MAGS | +0.00007 [-0.00006, +0.00021] | +0.00006 [-0.00013, +0.00024] |

Calibration-only MLP is statistically tied with `A+B` on all four datasets.
Isotonic is tied on SummEval and OpenMEVA, `A+B` is slightly better on the
four-path GPT-4o setting, and isotonic is lower on corrected WMT. `A+B` is
significantly better than binned calibration on all four datasets and better
than spline on SummEval and corrected WMT. The empirical calibration curve is
slightly better on WMT and tied on the other datasets.

## Information-Setting Effect

Using the held-out proxy distribution improves the flexible models, especially
on the human-proxy datasets:

| Dataset | Isotonic calibration-only | Isotonic target-aware |
|---|---:|---:|
| SummEval human | 0.01153 | 0.00909 |
| GPT-4o proxy | 0.00827 | 0.00811 |
| WMT MQM (`fixed_141`) | 0.05882 | 0.05211 |
| OpenMEVA-MAGS | 0.00984 | 0.00706 |

For SummEval, the first paired table averages all 7,989 radius cells shared
with A+B, including lower-radius cells from 11 curves that cannot reach
`d=.10`. The information-setting table instead averages the 1,589 complete
five-radius curves, which explains `0.01168` versus `0.01153`. The other
datasets have complete support.

The first table is the fair comparison for a diagnostic run before the future
candidate pool is available. The target-aware result answers a different
question: prediction after that pool has been collected and proxy-scored.

## Integrity and Reproducibility

- Every shared exp21 baseline prediction and realized gain matches the
  canonical exp21 output exactly.
- Permuting evaluation proxy scores changes realized gain but leaves every
  calibration-only prediction unchanged.
- Evaluation gold and proxy values are absent from the prediction function.
- Two full independent runs produced eight byte-identical output files.
- The run spec records input and source hashes and package versions.

Packaged canonical outputs:

- `../../results/direct_calibration_strict/main_protocol/`
- `../../results/direct_calibration_strict/wmt_fixed_141/`
- `../../results/direct_calibration_strict/wmt_submitted_139/` (submission
  reconstruction only)

The two original full runs produced eight byte-identical files. The redundant
repeat directory is not copied into this public package.

## Source-data qualification

An independent source audit reconstructs SummEval ratings, the repaired
GPT-4o aggregate, OpenMEVA, and the corrected WMT input. The submitted
139-path WMT artifact remains available as a labeled provenance lineage. Its
strict calibration-only A+B/isotonic CurveMAE is 0.06675/0.06661, but it is
not used for the primary result above.
