import copy
import json
import unittest
from pathlib import Path

from aft8.controls import build_controls


class ControlTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "config/control_design.json"
        self.design = json.loads(path.read_text(encoding="utf-8"))["records"]
        self.corpus = []
        for item in self.design:
            if item["variant"] == "D1":
                original_target = item["permutation"][item["expected_verse"] - 1]
                self.corpus.append(
                    {
                        "record_id": item["source_record_id"],
                        "text_content": "uno\ndos\ntres\ncuatro\ncinco\nseis",
                        "expected_transition_span": f"v.{original_target}",
                        "design_family": "P1",
                        "target_dimensions": "ΔP",
                    }
                )

    def test_saved_design_preserves_all_verses_and_target_content(self):
        records = build_controls(self.corpus, self.design)
        self.assertEqual(len(records), 60)
        verses = self.corpus[0]["text_content"].split("\n")
        for record in records:
            actual = record["text_content"].split("\n")
            self.assertCountEqual(actual, verses)
            if record["variant"] == "D1":
                self.assertEqual(
                    actual[record["expected_verse"] - 1],
                    verses[record["original_expected_verse"] - 1],
                )
            else:
                self.assertTrue(all(a != b for a, b in zip(actual, verses)))
                self.assertIsNone(record["expected_verse"])

    def test_missing_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Faltan"):
            build_controls(self.corpus, self.design[:-1])

    def test_repeated_verse_is_rejected(self):
        design = copy.deepcopy(self.design)
        design[0]["permutation"] = [1] * 6
        with self.assertRaisesRegex(ValueError, "Permutación"):
            build_controls(self.corpus, design)

    def test_derangement_cannot_receive_a_target(self):
        design = copy.deepcopy(self.design)
        next(item for item in design if item["variant"] == "D2")["expected_verse"] = 5
        with self.assertRaisesRegex(ValueError, "Desarreglo"):
            build_controls(self.corpus, design)


if __name__ == "__main__":
    unittest.main()
