"""Guard the matched-split inputs and order of MAE aggregation."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from artifact_tools import reward_model_figure as rm


class RewardModelFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        run = root / "results/reward_model_coefficients/canonical/analysis"
        cls.paths = pd.read_csv(run / "prospective_paths.csv")
        cls.summary = pd.read_csv(run / "summary.csv")

    def test_saved_inputs(self):
        rm.validate_tables(self.paths, self.summary)

    def test_missing_split(self):
        with self.assertRaisesRegex(ValueError, "completeness"):
            rm.validate_tables(self.paths.iloc[1:], self.summary)

    def test_duplicate_split(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rm.validate_tables(pd.concat([self.paths, self.paths.iloc[:1]]), self.summary)

    def test_nonfinite_prediction(self):
        data = self.paths.copy()
        data.loc[0, "cal_a_exact"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            rm.validate_tables(data, self.summary)

    def test_bad_intermediate_radius(self):
        data = self.paths.copy()
        data.loc[0, "actual_eval_delta_0p05"] += .01
        with self.assertRaisesRegex(ValueError, "absolute errors"):
            rm.validate_tables(data, self.summary)

    def test_missing_summary(self):
        with self.assertRaisesRegex(ValueError, "completeness"):
            rm.validate_tables(self.paths, self.summary.iloc[1:])

    def test_mae_is_not_error_of_mean(self):
        data = pd.DataFrame(dict(alias=["armorm"]*2, repeat=[0, 1], d=[.1]*2,
                                 exact=[0., 0.], ab=[1., -1.], aonly=[2., -2.]))
        row = rm.aggregate(data).iloc[0]
        self.assertEqual(row.ab, 0.)
        self.assertEqual(row.mae_ab, 1.)
        self.assertEqual(row.mae_aonly, 2.)


if __name__ == "__main__":
    unittest.main()
