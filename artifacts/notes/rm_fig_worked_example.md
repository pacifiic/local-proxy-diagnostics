# Reward-model forecasts across 50 matched splits

The upper row shows mean exact held-out tilt gain and mean calibration-only
PACE (A+B) and A-only predictions for SummEval coherence. Each of the 50 splits
uses 480 calibration and 1,120 evaluation items, shared across the three models.
The lower row averages absolute prediction errors computed within each split.
It is not the absolute error of the mean forecast.

Axes are zoomed; curves are shown for d=0.025 to 0.10. Shading marks the gap
between the two methods, not uncertainty: the mean B*d correction above and
the MAE difference below. Percentages compare PACE with A-only at d=0.10.
The repeats reuse the same 1,600 items and do not define independent-sample CIs.

`reward_model_figure.py` reconstructs curves from saved model scores and item
splits, and verifies coefficients, original-radius gains, and MAE against the
stored analysis. Static Inter fonts are bundled under the SIL Open Font License.
The exported CSV contains mean curves and within-split MAE, not single-split values.

The preselected repeat-0 example remains in `rm_worked_example` and in the
original `results/reward_model_coefficients/canonical/paper_example/` outputs.
This figure summarizes coherence only; the four-dimension endpoint comparison
is `rm_endpoint_mae`. These are fixed-pool tilt forecasts, not DPO or an RM ranking.
