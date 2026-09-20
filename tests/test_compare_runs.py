import tempfile
import unittest
from pathlib import Path

import numpy as np

from aft8.compare_runs import compare_runs


class ComparisonTests(unittest.TestCase):
    def test_tolerance_and_nan_mask_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            for root in (first, second):
                (root / "gxa_vectors").mkdir(parents=True)
            np.savez(
                first / "gxa_vectors/A001.npz",
                metric=np.array([np.nan, 1.0]),
                tokens=np.array(["a", "b"]),
            )
            np.savez(
                second / "gxa_vectors/A001.npz",
                metric=np.array([np.nan, 1.000001]),
                tokens=np.array(["a", "b"]),
            )
            self.assertEqual(compare_runs(first, second)["failures"], [])
            np.savez(
                second / "gxa_vectors/A001.npz",
                metric=np.array([0.0, 1.0]),
                tokens=np.array(["a", "b"]),
            )
            self.assertTrue(compare_runs(first, second)["failures"])

    def test_missing_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                compare_runs(Path(directory) / "a", Path(directory) / "b")
