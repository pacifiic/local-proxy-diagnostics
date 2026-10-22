"""Regression tests for manuscript-audit fixes; no GPU or source-data writes."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT
RR = ROOT
sys.path.insert(0, str(CORE / 'lib'))
sys.path.insert(0, str(CORE / 'experiments/12_accept_boost'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bon = load('audit_bon', RR / 'experiments/15_prompt_conditional_bon_scope/01_prompt_conditional_bon_audit.py')


class BoNTests(unittest.TestCase):
    def path(self, r=None, gold=None):
        r = np.tile([1.] + [5.] * 7, 2) if r is None else np.asarray(r, dtype=float)
        gold = np.arange(len(r), dtype=float) ** 1.3 if gold is None else np.asarray(gold)
        return SimpleNamespace(proxy_scores=r, gold_scores=gold, path_id='test', group_key='test')

    def evaluate(self, path, ns=(1, 2, 4, 64)):
        prompts = np.repeat(['a', 'b'], len(path.proxy_scores) // 2)
        return bon.evaluate_path(path, prompts, list(ns), .25)

    def test_saturated_pool_below_outer_cutoff(self):
        rows = self.evaluate(self.path())
        self.assertLess(rows[1]['pooled_kl_supremum'], .25)
        self.assertTrue(rows[1]['global_tilt_reachable'])
        self.assertTrue(np.isfinite(rows[1]['global_tilt_at_mean_kl']))

    def test_reference_is_zero(self):
        first = self.evaluate(self.path())[0]
        self.assertTrue(first['global_tilt_reachable'])
        self.assertEqual(first['global_tilt_at_mean_kl'], 0)
        self.assertAlmostEqual(first['conditional_bon_gain'], 0)

    def test_outside_audited_window(self):
        rows = self.evaluate(self.path(np.arange(20)), [4])
        self.assertGreater(rows[0]['prompt_mean_kl'], .25)
        self.assertFalse(rows[0]['global_tilt_reachable'])
        self.assertTrue(np.isnan(rows[0]['global_tilt_at_mean_kl']))

    def test_positive_affine_proxy_invariance(self):
        path = self.path()
        other = self.path(7 * path.proxy_scores - 100, path.gold_scores)
        a, b = self.evaluate(path, [2]), self.evaluate(other, [2])
        self.assertAlmostEqual(a[0]['global_tilt_at_mean_kl'], b[0]['global_tilt_at_mean_kl'], places=11)

    def test_pooled_gold_not_last_prompt(self):
        path = self.path()
        rows = self.evaluate(path, [2])
        # Identical score compositions in both prompts imply identical pooled
        # and within-prompt selection distributions for this two-score example.
        self.assertAlmostEqual(rows[0]['global_tilt_at_mean_kl'], rows[0]['conditional_bon_gain'], places=9)

    def test_within_prompt_permutation(self):
        path = self.path()
        perm = np.r_[np.arange(8)[::-1], np.arange(8, 16)[::-1]]
        other = self.path(path.proxy_scores[perm], path.gold_scores[perm])
        a,b = self.evaluate(path,[2])[0], self.evaluate(other,[2])[0]
        self.assertAlmostEqual(a['global_tilt_at_mean_kl'],b['global_tilt_at_mean_kl'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
