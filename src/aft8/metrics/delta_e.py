"""Entropía predictiva y su primera diferencia, expresadas en nats.

Un incremento de entropía indica mayor dispersión de la distribución
predictiva. La interpretación describe incertidumbre probabilística y
no atribuye estados emocionales al modelo.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


def entropy_series(logits: torch.Tensor, chunk: int = 32) -> np.ndarray:
    """Calcula H = -Σ p ln(p) por token, con probabilidades en float32."""
    T = logits.shape[0]
    out = np.empty(T, dtype=np.float64)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        p = torch.softmax(logits[start:end].float(), dim=-1)
        h = -(p * torch.log(p + EPS)).sum(-1)
        out[start:end] = h.double().cpu().numpy()
    return out


def entropy_norm(entropy: np.ndarray, vocab_size: int) -> np.ndarray:
    """Expresa H como fracción del máximo ln(V), para V mayor que uno."""
    return entropy / float(np.log(vocab_size))


def delta_e_series(entropy: np.ndarray) -> np.ndarray:
    """Obtiene la primera diferencia con signo; la posición inicial es NaN.

    La entrada debe ser un array de punto flotante para representar NaN.
    """
    out = np.full_like(entropy, np.nan)
    out[1:] = entropy[1:] - entropy[:-1]
    return out
