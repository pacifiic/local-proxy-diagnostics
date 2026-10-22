# Prompt-Conditional BoN Results

These outputs were regenerated inside this repository from
`data/summeval_long.csv` and `data/openmeva_long.csv`.

The regenerated path-level and summary CSVs are byte-identical to the canonical
paper outputs:

| Dataset | Paths CSV SHA-256 | Summary CSV SHA-256 |
|---|---|---|
| SummEval | `1c7b5b019f53a47734372cf72088a3a03b7b6ca856696b6bde7920687f03dd14` | `ee1695b03dc6394ffbd7e8e5a2013d2ae6668806058d30c43681dd2b1aa33381` |
| OpenMEVA-MAGS | `9d733a0d0406a128a93d2840601f42f3832f7eb3917fa9acc714b6065d0e38bb` | `d5e12d20358959b2e41be673d37fa281cbe1a6ca386a99c36cf34c538f906750` |

At `N=2`, the mean fractions of prompt-level KL values at or below `0.10`
are `0.1246875` for SummEval and `0.1165` for OpenMEVA-MAGS.
