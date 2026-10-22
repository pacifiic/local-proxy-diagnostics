"""
Diagnosis / intervention mapping (Experiment 5).

Given a results DataFrame with columns:
  dimension, ICC_A1, G_pers, Lift_expert, Lift_protocol

assign one of:
  D1. personalization-limited  — G_pers is the dominant issue
  D2. measurement-limited      — expert lift is large, crowd just can't do it
  D3. deeply-sparse            — even experts can't agree, fundamental difficulty
  D4. mixed                    — no single dominant factor

Thresholds use absolute values grounded in ICC literature (Cicchetti 1994,
Koo & Li 2016) rather than data-driven terciles, because n=4 dimensions
makes percentile-based cutoffs degenerate.

ICC(A,1) levels:
  - good:     >= 0.75  (good-to-excellent agreement)
  - moderate: 0.50-0.75
  - poor:     < 0.50

G_pers (personalization gap = ICC(C,1) - ICC(A,1)):
  - high: >= 0.03   (meaningful rater bias component)
  - low:  < 0.01

Lift_expert (expert ICC - crowd ICC):
  - high: >= 0.15   (substantial expert advantage)
  - low:  < 0.05
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Absolute thresholds (domain-grounded)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "ICC_A1": {"low": 0.50, "high": 0.75},
    "G_pers": {"low": 0.01, "high": 0.03},
    "Lift_expert": {"low": 0.05, "high": 0.15},
    "Lift_protocol": {"low": 0.02, "high": 0.05},
}


def assign_diagnosis(row: pd.Series, thresholds: dict | None = None) -> str:
    """Assign a diagnosis label based on threshold levels.

    Parameters
    ----------
    row : Series with keys ICC_A1, G_pers, Lift_expert, Lift_protocol
    thresholds : dict with keys {metric: {"low": float, "high": float}}
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    def level(val, metric):
        t = thresholds[metric]
        if val <= t["low"]:
            return "low"
        elif val >= t["high"]:
            return "high"
        return "medium"

    nu = level(row["ICC_A1"], "ICC_A1")
    gp = level(row["G_pers"], "G_pers")
    le = level(row["Lift_expert"], "Lift_expert")
    lp_val = row.get("Lift_protocol", np.nan)
    lp = level(lp_val, "Lift_protocol") if not np.isnan(lp_val) else "unknown"

    # Priority 1: already high ICC → evaluation is working
    if nu == "high":
        return "high-verifiability"

    # Priority 2: low ICC + experts don't help → fundamental difficulty
    # Even if G_pers is high, if expert lift is non-positive the dimension
    # is intractable through rater-pool improvements alone
    if nu == "low" and le == "low":
        return "deeply-sparse"

    # Priority 3: expert lift is high → problem is solvable with better raters
    if le == "high":
        return "measurement-limited"

    # Priority 4: high G_pers with moderate lift → rater bias dominates
    if gp == "high":
        return "personalization-limited"

    return "mixed"


INTERVENTION_MAP = {
    "personalization-limited": [
        "user conditioning",
        "personalized evaluator",
        "preference clusters",
    ],
    "measurement-limited": [
        "better rubric",
        "expert evaluation",
        "adjudication",
        "more context",
    ],
    "deeply-sparse": [
        "task decomposition",
        "step-level checking",
        "target reformulation",
    ],
    "high-verifiability": [
        "standard evaluation sufficient",
        "focus on scale and efficiency",
    ],
    "mixed": [
        "further investigation needed",
    ],
}


def make_intervention_table(df_results: pd.DataFrame,
                            thresholds: dict | None = None) -> pd.DataFrame:
    """Build the final intervention map table.

    df_results must have columns:
        dimension, ICC_A1, G_pers, Lift_expert, Lift_protocol (optional)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    df = df_results.copy()
    df["diagnosis"] = df.apply(lambda r: assign_diagnosis(r, thresholds), axis=1)
    df["recommended_intervention"] = df["diagnosis"].map(
        lambda d: "; ".join(INTERVENTION_MAP.get(d, ["N/A"]))
    )
    return df


# ---------------------------------------------------------------------------
# Sensitivity analysis — sweep over threshold variants
# ---------------------------------------------------------------------------

SENSITIVITY_VARIANTS = {
    "lenient": {
        "ICC_A1": {"low": 0.40, "high": 0.60},
        "G_pers": {"low": 0.005, "high": 0.02},
        "Lift_expert": {"low": 0.03, "high": 0.10},
        "Lift_protocol": {"low": 0.01, "high": 0.03},
    },
    "default": DEFAULT_THRESHOLDS,
    "strict": {
        "ICC_A1": {"low": 0.60, "high": 0.80},
        "G_pers": {"low": 0.02, "high": 0.05},
        "Lift_expert": {"low": 0.08, "high": 0.20},
        "Lift_protocol": {"low": 0.03, "high": 0.08},
    },
}


def run_sensitivity_analysis(df_combined: pd.DataFrame) -> pd.DataFrame:
    """Run diagnoses under lenient / default / strict thresholds.

    Returns a wide DataFrame with columns:
        dimension, diag_lenient, diag_default, diag_strict, stable
    where `stable` is True if the diagnosis is the same across all three.
    """
    results = {"dimension": df_combined["dimension"].tolist()}
    for variant_name, thresholds in SENSITIVITY_VARIANTS.items():
        diags = df_combined.apply(
            lambda r: assign_diagnosis(r, thresholds), axis=1
        ).tolist()
        results[f"diag_{variant_name}"] = diags

    df_sens = pd.DataFrame(results)
    df_sens["stable"] = (
        (df_sens["diag_lenient"] == df_sens["diag_default"])
        & (df_sens["diag_default"] == df_sens["diag_strict"])
    )
    return df_sens
