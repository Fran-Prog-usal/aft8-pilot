import unittest

import numpy as np
import pandas as pd

from aft8.spans import evaluate_span, matching_spans, span_mass, tokens_in_span


class SpanTests(unittest.TestCase):
    def test_accented_whole_word_does_not_match_substring(self):
        self.assertEqual(matching_spans("sin responder. Dije sí.", "sí"), [(20, 22)])
        self.assertEqual(matching_spans("silla sin si", "sí"), [])

    def test_matching_retains_offsets_and_all_occurrences(self):
        self.assertEqual(matching_spans("Mano y mano", "mano"), [(0, 4), (7, 11)])
        self.assertEqual(matching_spans("mi nombre", "mi nombre"), [(0, 9)])

    def test_subword_mapping_uses_overlap_not_token_spelling(self):
        tokens = pd.DataFrame({"token_idx": [0, 1, 2], "char_start": [0, 0, 3],
                               "char_end": [0, 3, 8], "is_special": [True, False, False]})
        self.assertEqual(tokens_in_span(tokens, (0, 8)), [1, 2])

    def test_future_span_has_zero_mass_but_missing_query_is_nan(self):
        matrix = np.array([[1., np.nan, np.nan], [.2, .8, np.nan], [np.nan] * 3])
        self.assertEqual(span_mass(matrix, 0, [1]), 0.)
        self.assertEqual(span_mass(matrix, 1, [1]), .8)
        self.assertTrue(np.isnan(span_mass(matrix, 2, [1])))

    def test_transfer_windows_and_source_change(self):
        text = "madre pan cuchillo fin"
        tokens = pd.DataFrame({"token_idx": [0, 1, 2, 3], "char_start": [0, 6, 10, 19],
                               "char_end": [5, 9, 18, 22], "is_special": False, "line_id": 1})
        matrix = np.array([[1., np.nan, np.nan, np.nan], [.8, .2, np.nan, np.nan],
                           [.2, .1, .7, np.nan], [np.nan] * 4])
        row, trajectory = evaluate_span(text, tokens, matrix, {"source": "madre", "target": "cuchillo"}, set())
        self.assertEqual(row["window_before_start"], 0)
        self.assertEqual(row["window_before_end"], 1)
        self.assertEqual(row["window_after_start"], 2)
        self.assertAlmostEqual(row["source_mass_change"], -.7)
        self.assertEqual(trajectory[0]["target_mass"], 0)
        self.assertTrue(row["target_exceeds_controls"])

    def test_missing_source_is_not_a_successful_transfer(self):
        tokens = pd.DataFrame({"token_idx": [0, 1], "char_start": [0, 4], "char_end": [3, 7],
                               "is_special": False, "line_id": 1})
        matrix = np.array([[1., np.nan], [np.nan, np.nan]])
        row, _ = evaluate_span("pan fin", tokens, matrix, {"source": "madre", "target": "pan"}, set())
        self.assertEqual(row["status"], "source_not_found")


if __name__ == "__main__":
    unittest.main()
