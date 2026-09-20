"""Propiedades matemáticas de las métricas predictivas."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from aft8.metrics import (
    delta_e_series,
    delta_p_series,
    entropy_norm,
    entropy_series,
    reconstruye_probabilidades,
    surprisal_series,
    topk_logits,
)


def logits_of(probabilities):
    """Construye logits a partir de probabilidades, admitiendo ceros."""
    return torch.log(torch.tensor(probabilities, dtype=torch.float64) + 1e-300)


class PredictiveMetricTests(unittest.TestCase):
    def test_equal_distributions_have_zero_divergence(self):
        values = delta_p_series(logits_of([[0.8, 0.2], [0.8, 0.2]]))
        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1], 0.0, places=7)

    def test_disjoint_supports_reach_theoretical_ceiling(self):
        values = delta_p_series(logits_of([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(values[1], math.log(2), places=6)

    def test_divergence_depends_on_overlap_at_equal_concentration(self):
        same = delta_p_series(logits_of([[0.999, 0.001], [0.999, 0.001]]))[1]
        swapped = delta_p_series(logits_of([[0.999, 0.001], [0.001, 0.999]]))[1]
        self.assertAlmostEqual(same, 0.0, places=7)
        self.assertGreater(swapped, 0.68)

    def test_divergence_is_symmetric_and_bounded(self):
        generator = torch.Generator().manual_seed(17)
        pairs = torch.randn(40, 2, 16, generator=generator) * 3
        for pair in pairs:
            forward = delta_p_series(pair)[1]
            backward = delta_p_series(pair.flip(0))[1]
            self.assertAlmostEqual(forward, backward, places=7)
            self.assertGreaterEqual(forward, 0.0)
            self.assertLessEqual(forward, math.log(2) + 1e-6)

    def test_entropy_uniform_and_point_mass(self):
        values = entropy_series(logits_of([[0.25]*4, [1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(values, [math.log(4), 0.0], atol=1e-6)
        np.testing.assert_allclose(entropy_norm(values, 4), [1.0, 0.0], atol=1e-6)

    def test_entropy_difference_telescopes(self):
        values = np.array([0.2, 1.4, 0.6, 0.1, 0.8])
        differences = delta_e_series(values)
        self.assertTrue(np.isnan(differences[0]))
        self.assertAlmostEqual(np.nansum(differences), values[-1] - values[0])

    def test_surprisal_uses_previous_position(self):
        logits = logits_of([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
        values = surprisal_series(logits, torch.tensor([0, 1]))
        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1], -math.log(0.25), places=6)

    def test_chunking_preserves_predictive_metrics(self):
        logits = torch.randn(9, 31, generator=torch.Generator().manual_seed(29))
        ids = torch.arange(9)
        for fn, args in [(delta_p_series, (logits,)),
                         (entropy_series, (logits,)),
                         (surprisal_series, (logits, ids))]:
            np.testing.assert_array_equal(fn(*args, chunk=1), fn(*args, chunk=4))

    def test_single_token_has_no_transition(self):
        logits = torch.zeros(1, 3)
        self.assertTrue(np.isnan(delta_p_series(logits)[0]))
        self.assertTrue(np.isnan(surprisal_series(logits, torch.tensor([1]))[0]))
        self.assertTrue(np.isnan(delta_e_series(entropy_series(logits))[0]))

    def test_topk_probabilities_match_full_distribution(self):
        logits = torch.randn(5, 17, generator=torch.Generator().manual_seed(41))
        stored = topk_logits(logits, torch.arange(5), k=4)
        restored = reconstruye_probabilidades(
            stored['logits_topk_idx'], stored['logits_topk_val'],
            stored['logits_logsumexp'])
        expected = np.take_along_axis(torch.softmax(logits, -1).numpy(),
                                      stored['logits_topk_idx'], axis=1)
        np.testing.assert_allclose(restored, expected, rtol=1e-5, atol=1e-7)
        self.assertTrue(np.all(restored.sum(axis=1) < 1))

    def test_topk_caps_width_and_preserves_observed_rank(self):
        logits = torch.tensor([[2., 1., 0.], [0., 1., 2.], [1., 1., 0.]])
        stored = topk_logits(logits, torch.tensor([0, 2, 2]), k=50)
        self.assertEqual(stored['logits_topk_idx'].shape, (3, 3))
        np.testing.assert_array_equal(stored['observed_rank'], [-1, 2, 0])


if __name__ == '__main__':
    unittest.main()
