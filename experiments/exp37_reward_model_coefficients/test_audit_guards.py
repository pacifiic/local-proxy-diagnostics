"""Adversarial result-file checks without model scoring."""
import copy
import unittest

import numpy as np
import pandas as pd

from audit_run import validate_result_frames
from common import DIMENSIONS


class ResultGuards(unittest.TestCase):
    def setUp(self):
        self.models = ['one', 'two', 'three']
        self.splits = {(d, 0): {} for d in DIMENSIONS}
        self.accounting = [dict(dimension=d, repeat=0, kept_for_all_models=True, reasons={}) for d in DIMENSIONS]
        self.radii = [.1]
        coeff, predictions, summary = [], [], []
        for a in self.models:
            for d in DIMENSIONS:
                coeff.append(dict(alias=a, dimension=d, **{c: .1 for c in ['A', 'B', 'A_lo', 'A_hi', 'B_lo', 'B_hi', 'B_tail', 'B_skewness']}))
                predictions.append(dict(alias=a, dimension=d, repeat=0, cal_a_exact=.1, cal_b_theory=.1,
                                        actual_eval_delta_0p1=.1, pred_theory_delta_0p1=.1,
                                        **{f'abs_err_{p}_0p1': .1 for p in ['theory', 'aonly', 'empfit', 'anu', 'zero']}))
                summary.append(dict(alias=a, dimension=d, radius=.1, n_common_splits=1, cal_A_mean=.1,
                                    cal_B_mean=.1, realized_gain_mean=.1,
                                    **{f'mae_{p}': .1 for p in ['theory', 'aonly', 'empfit', 'anu', 'zero']}))
        self.coeff, self.predictions, self.summary = map(pd.DataFrame, [coeff, predictions, summary])

    def check(self):
        validate_result_frames(self.coeff, self.predictions, self.summary, self.models,
                               self.splits, self.radii, self.accounting)

    def test_valid(self):
        self.check()

    def test_nonfinite_coefficient_gain_and_mae(self):
        for frame_name, column in [('coeff', 'A'), ('predictions', 'actual_eval_delta_0p1'), ('summary', 'mae_theory')]:
            for value in [np.nan, np.inf]:
                with self.subTest(frame=frame_name, column=column, value=value):
                    frame = getattr(self, frame_name)
                    previous = frame.loc[0, column]
                    frame.loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                        self.check()
                    frame.loc[0, column] = previous

    def test_missing_coefficient(self):
        self.coeff = self.coeff.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'coefficient'):
            self.check()

    def test_missing_whole_joint_split(self):
        self.predictions = self.predictions[self.predictions.dimension != DIMENSIONS[0]]
        with self.assertRaisesRegex(ValueError, 'prediction'):
            self.check()

    def test_missing_summary(self):
        self.summary = self.summary.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'summary'):
            self.check()

    def test_unexplained_skip(self):
        self.accounting[0]['kept_for_all_models'] = False
        with self.assertRaisesRegex(ValueError, 'failure reason'):
            self.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
