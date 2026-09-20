import unittest

import numpy as np
import pandas as pd

from aft8.statistics import (
    conditional_permutation,
    document_bootstrap,
    fdr_bh,
    paired_comparison,
    score_peaks,
)


class StatisticsTests(unittest.TestCase):
    def test_bh_known_values_and_input_order(self):
        np.testing.assert_allclose(
            fdr_bh([0.2, 0.01, 0.04, 0.03]), [0.2, 0.04, 0.0533333333333, 0.0533333333333]
        )
        with self.assertRaises(ValueError):
            fdr_bh([np.nan])

    def test_exact_paired_test_uses_discordant_pairs(self):
        a = pd.DataFrame(
            {
                "record_id": [str(i) for i in range(18)],
                "expected_verse": 6,
                "hit": [1] * 9 + [0] * 9,
            }
        )
        b = a.copy()
        b["hit"] = [1] * 2 + [0] * 7 + [1] * 3 + [0] * 6
        result = paired_comparison(a, b)
        self.assertEqual((result["metric_only"], result["baseline_only"]), (7, 3))
        self.assertEqual(result["p_raw"], 0.34375)

    def test_missing_pair_and_changed_target_are_rejected(self):
        a = pd.DataFrame({"record_id": ["A", "B"], "expected_verse": [5, 6], "hit": [1, 0]})
        with self.assertRaises(ValueError):
            paired_comparison(a, a.iloc[:1])
        b = a.copy()
        b.loc[0, "expected_verse"] = 6
        with self.assertRaises(ValueError):
            paired_comparison(a, b)

    def test_identical_methods_have_p_one(self):
        a = pd.DataFrame({"record_id": ["A", "B"], "expected_verse": [5, 6], "hit": [1, 0]})
        self.assertEqual(paired_comparison(a, a)["p_raw"], 1.0)

    def test_peak_ties_are_explicit_and_rank_is_average(self):
        verses = pd.DataFrame(
            {
                "record_id": ["A"] * 5,
                "line_id": [2, 3, 4, 5, 6],
                "expected_verse": 6,
                "design_family": "P1",
                "x": [0, 0, 0, 1, 1],
            }
        )
        result = score_peaks(verses, "x").iloc[0]
        self.assertEqual(result.argmax_verse, 5)
        self.assertEqual(result.max_ties, 2)
        self.assertEqual(result.rank_of_expected, 1.5)

    def test_permutation_preserves_document_cluster_and_family_blocks(self):
        table = pd.DataFrame(
            [
                {
                    "record_id": identifier,
                    "design_family": family,
                    "expected_verse": verse,
                    "argmax_verse": verse,
                    "hit": 1,
                    "metric": metric,
                }
                for identifier, family, verse in [("A", "P1", 5), ("B", "P2", 6)]
                for metric in ("delta_p", "delta_e", "gxa")
            ]
        )
        fixed = conditional_permutation(table, n_permutations=99, stratify_family=True)
        self.assertTrue((fixed.p_raw == 1).all())
        shuffled = conditional_permutation(table, n_permutations=99)
        self.assertTrue((shuffled.null_mean == shuffled.null_mean.iloc[0]).all())
        self.assertTrue((shuffled.p_raw >= 0.01).all())
        self.assertEqual(shuffled.n_texts.iloc[0], 2)
        self.assertEqual(shuffled.n_score_rows.iloc[0], 6)

    def test_document_bootstrap_constant_effect(self):
        result = document_bootstrap([0.25] * 30, n_resamples=100)
        self.assertEqual(result["mean_document_weighted"], 0.25)
        self.assertEqual(result["ci95_low"], 0.25)
        self.assertEqual(result["ci95_high"], 0.25)


if __name__ == "__main__":
    unittest.main()
