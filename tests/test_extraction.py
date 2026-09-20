import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aft8.extraction import assert_smoke, extract_record, read_inputs
from aft8.model import MistralExtractor, tokenize_exact, validate_attention


class ExtractionGuards(unittest.TestCase):
    def test_rejects_hypotheses_in_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            text = "a\nb\nc\nd\ne\nf"
            record = {
                "record_id": "A001",
                "text_content": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "expected_verse": 6,
            }
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "hipótesis"):
                read_inputs(path)

    def test_checks_every_attention_layer(self):
        good = torch.eye(3).reshape(1, 1, 3, 3).requires_grad_()
        bad = good.detach().clone()
        with self.assertRaisesRegex(ValueError, "capa 1"):
            validate_attention([good, bad])

    def test_rejects_future_attention(self):
        attention = torch.full((1, 1, 3, 3), 1 / 3, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "causal"):
            validate_attention([attention])

    def test_old_smoke_cannot_unlock_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            path.write_text(
                json.dumps(
                    {"status": "complete", "cohort": "smoke", "record_ids": ["A001", "A002"]}
                )
            )
            with self.assertRaisesRegex(ValueError, "smoke"):
                assert_smoke(path, "original", "test", {})


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Requiere el extra extraction")
class TinyMistralTest(unittest.TestCase):
    def test_teacher_forcing_attention_gradients_and_serialization(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

        text = "Uno.\nDos.\nTres.\nCuatro.\nCinco.\nSeis."
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            [text],
            trainers.BpeTrainer(
                vocab_size=300,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<unk>", "<s>", "</s>"],
            ),
        )
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(7)
            config = MistralConfig(
                vocab_size=len(fast),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=256,
                sliding_window=None,
            )
            config._attn_implementation = "eager"
            model = MistralForCausalLM(config)
            adapter = MistralExtractor(model, fast, device="cpu")
            tokens, lines, arrays, report, attention = extract_record(
                adapter, {"record_id": "A001", "text_content": text}, "synthetic"
            )
            self.assertTrue(report["decode_exact"])
            self.assertTrue(report["ok"])
            self.assertEqual(len(lines), 6)
            self.assertNotIn("expected_verse", tokens)
            self.assertEqual(attention.shape[:2], (2, 2))
            self.assertGreater(np.nansum(arrays["salience_cruda"]), 0)
            self.assertTrue(np.isnan(arrays["salience"][-1]).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "vectors.npz"
                np.savez_compressed(path, **arrays)
                with np.load(path, allow_pickle=False) as stored:
                    for key in stored.files:
                        self.assertFalse(stored[key].dtype.hasobject)
            encoding = tokenize_exact(fast, text)
            self.assertEqual(sum(encoding["special_mask"]), 1)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
