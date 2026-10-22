"""Guard the fit-range table's group-level aggregation and input completeness."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "fit_range_table", ROOT / "artifact_tools/make_fit_range_table.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FitRangeArtifact(unittest.TestCase):
    def test_saved_group_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder.build(builder.SOURCE, Path(tmp))
            table = pd.read_csv(Path(tmp)/"table_fit_range_sensitivity.csv")
        self.assertEqual(table.groups.tolist(), [8]*6)
        self.assertEqual([f"{x:.6f}" for x in table.mean_r2],
                         ["0.999999","0.999993","0.999968","0.999962","0.999933","0.999912"])
        self.assertEqual([f"{x:.4f}" for x in table.corr_B],
                         ["0.9983","0.9953","0.9909","0.9891","0.9846","0.9825"])

    def test_missing_group_rejected(self):
        frame = pd.read_csv(builder.SOURCE)
        index = frame.index[(frame.score_transform == "raw") & (frame.gold_estimator == "loo_mean")][0]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"partial.csv"
            frame.drop(index).to_csv(source,index=False)
            with self.assertRaisesRegex(ValueError, "eight"):
                builder.build(source,Path(tmp)/"out")

    def test_nonfinite_rejected(self):
        frame = pd.read_csv(builder.SOURCE)
        mask = (frame.score_transform == "raw") & (frame.gold_estimator == "loo_mean")
        frame.loc[mask, "mean_r2"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"invalid.csv"
            frame.to_csv(source,index=False)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                builder.build(source,Path(tmp)/"out")


if __name__ == "__main__":
    unittest.main()
