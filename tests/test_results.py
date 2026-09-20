import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aft8.results import ResultArchive
from aft8.validation import aggregate_lines, check_offsets


class ResultTests(unittest.TestCase):
    def archive(self, root, name, blob):
        path = Path(root) / "package.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_wrong_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.archive(root, "data.txt", b"data")
            with self.assertRaisesRegex(ValueError, "hash"):
                ResultArchive(path, "0" * 64)

    def test_unsafe_member_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path, digest = self.archive(root, "../outside", b"data")
            with self.assertRaisesRegex(ValueError, "Miembro"):
                ResultArchive(path, digest)
            self.assertFalse((Path(root).parent / "outside").exists())

    def test_vectors_do_not_require_object_deserialization(self):
        blob = io.BytesIO()
        np.savez(blob, tokens=np.array(["text"], dtype=object), input_ids=np.array([42]))
        with tempfile.TemporaryDirectory() as root:
            path, digest = self.archive(
                root, "results/pilot30_aft8_v1__mistral/gxa_vectors/A001.npz", blob.getvalue()
            )
            with ResultArchive(path, digest) as archive:
                values = archive.vectors("original", "A001")
                self.assertEqual(list(values), ["input_ids"])
                self.assertEqual(values["input_ids"].tolist(), [42])

    def test_offsets_distinguish_missing_space_from_missing_content(self):
        tokens = pd.DataFrame(
            {
                "token_idx": [0, 1],
                "char_start": [0, 2],
                "char_end": [1, 3],
                "is_special": [False, False],
                "line_id": [1, 1],
                "is_newline": [False, False],
            }
        )
        result = check_offsets("a b", tokens)
        self.assertEqual(result["errors"], [])
        self.assertFalse(result["literal_offset_coverage"])
        self.assertTrue(check_offsets("axb", tokens)["errors"])

    def test_aggregation_excludes_specials_includes_newline_and_preserves_sign(self):
        tokens = pd.DataFrame(
            {
                "token_idx": [0, 1, 2],
                "line_id": [0, 1, 1],
                "is_special": [True, False, False],
                "is_newline": [False, False, True],
            }
        )
        for name in [
            "delta_P",
            "delta_E",
            "gxa_redistribution",
            "gxa_redist_pad",
            "deltaP_top100_T10",
        ]:
            tokens[name] = [100.0, 2.0, -3.0]
        salience = np.array([[1.0, np.nan, np.nan], [0.25, 0.75, np.nan], [np.nan] * 3])
        row = aggregate_lines(tokens, salience).loc[1]
        self.assertEqual(row.n_tokens, 2)
        self.assertEqual(row.deltaE_sum, -1.0)
        self.assertEqual(row.deltaE_absmax, 3.0)
        self.assertEqual(row.deltaP_max, 2.0)
        self.assertEqual(row.gxa_mass_received, 0.375)


if __name__ == "__main__":
    unittest.main()
