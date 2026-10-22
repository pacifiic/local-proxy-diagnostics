"""CI aggregation and reaggregation contracts; CPU only."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT
sys.path.insert(0, str(CORE / 'lib'))
sys.path.insert(0, str(CORE / 'experiments/12_accept_boost'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


endpoint = load('ci_endpoint', CORE / 'experiments/07_prospective_prediction/01_prospective_prediction.py')
sweep = load('ci_sweep', CORE / 'experiments/12_accept_boost/accept_boost_utils.py')
figures = load('ci_figures', CORE / 'artifact_tools/make_paper_figures.py')
refresh = load('ci_refresh', ROOT / 'tools/recompute_prediction_intervals.py')


class Intervals(unittest.TestCase):
    def compare(self, counts, missing=False):
        frame = pd.DataFrame([(str(i), float(i + j / 10)) for i, n in enumerate(counts) for j in range(n)],
                             columns=['path_id', 'value'])
        if missing:
            frame.loc[0, 'value'] = np.nan
        frame['abs_err_theory_0p1'] = frame.value
        frame['abs_err_theory'] = frame.value
        groups = [g for _, g in frame.groupby('path_id')]
        rng = np.random.default_rng(0)
        draws = [pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]).value.mean()
                 for _ in range(1000)]
        expected = np.nanquantile(draws, [.025, .975])
        wide = endpoint._bootstrap_mae_ci(frame, [.1])
        long = sweep._cluster_bootstrap_for_group(frame, ['theory'], 'path_id', 1000, 0)
        plot = figures._cluster_bootstrap_mean_ci(frame, 'value', 'path_id')
        np.testing.assert_allclose([wide['abs_err_theory_0p1_ci_lo'], wide['abs_err_theory_0p1_ci_hi']], expected)
        np.testing.assert_allclose([long['mae_theory_ci_lo'], long['mae_theory_ci_hi']], expected)
        np.testing.assert_allclose(plot[1:], expected)
        self.assertAlmostEqual(plot[0], frame.value.mean())

    def test_unequal_counts(self):
        self.compare([2, 5, 7, 3])

    def test_equal_counts(self):
        self.compare([5, 5, 5, 5])

    def test_missing_values(self):
        self.compare([2, 5, 7, 3], missing=True)

    def test_saved_endpoint(self):
        data = pd.read_csv(CORE / 'results/07_prospective_prediction/exp27_human_baseline/exp27_prospective_paths.csv')
        ci = endpoint._bootstrap_mae_ci(data, [.1])
        self.assertEqual(len(data), 1589)
        self.assertEqual(f'{data.abs_err_theory_0p1.mean():.4f}', '0.0174')
        self.assertEqual(f"{ci['abs_err_theory_0p1_ci_hi']:.4f}", '0.0193')

    def test_point_statistic_change_rejected(self):
        old = pd.DataFrame({'key': ['a'], 'mae': [.1], 'mae_ci_lo': [.05]})
        new = old.assign(mae=.2)
        with self.assertRaisesRegex(ValueError, 'Point statistic'):
            refresh.merged_intervals(old, new, ['key'])

    def test_duplicate_summary_rejected(self):
        old = pd.DataFrame({'key': ['a', 'a'], 'mae': [.1, .2]})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            refresh.merged_intervals(old, old, ['key'])

    def test_duplicate_prediction_rejected(self):
        data = pd.DataFrame({'path_id': ['a', 'a'], 'split_rep': [0, 0], 'abs_err_theory': [.1, .2]})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            refresh.validate_rows(data, ['path_id', 'split_rep'])

    def test_nonfinite_and_negative_errors_rejected(self):
        for value in [np.inf, -np.inf, -.1]:
            data = pd.DataFrame({'path_id': ['a'], 'abs_err_theory': [value]})
            with self.subTest(value=value), self.assertRaises(ValueError):
                refresh.validate_rows(data, ['path_id'])

    def test_missing_support_rejected(self):
        old = pd.DataFrame({'key': ['a', 'b'], 'mae': [.1, .2]})
        with self.assertRaisesRegex(ValueError, 'support'):
            refresh.merged_intervals(old, old.iloc[:1], ['key'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
