"""Propiedades de los dos soportes de redistribución GxA."""

import unittest

import numpy as np

from aft8.metrics.gxa import redistribution_series


class RedistributionTests(unittest.TestCase):
    def test_new_token_mass_changes_only_padded_support(self):
        salience = np.array([[1.0, np.nan, np.nan], [0.25, 0.75, np.nan], [np.nan] * 3])
        shared, padded = redistribution_series(salience)
        self.assertEqual(shared[1], 0.0)
        self.assertGreater(padded[1], 0.0)
        self.assertTrue(np.isnan(shared[[0, 2]]).all())
        self.assertTrue(np.isnan(padded[[0, 2]]).all())

    def test_identical_shared_distribution_without_new_mass_has_zero_change(self):
        salience = np.array([[1.0, np.nan, np.nan], [1.0, 0.0, np.nan], [np.nan] * 3])
        shared, padded = redistribution_series(salience)
        self.assertEqual(shared[1], 0.0)
        self.assertEqual(padded[1], 0.0)

    def test_incomplete_causal_row_is_undefined_in_both_supports(self):
        salience = np.array([[1.0, np.nan, np.nan], [np.nan, 1.0, np.nan], [np.nan] * 3])
        shared, padded = redistribution_series(salience)
        self.assertTrue(np.isnan(shared[1]))
        self.assertTrue(np.isnan(padded[1]))
