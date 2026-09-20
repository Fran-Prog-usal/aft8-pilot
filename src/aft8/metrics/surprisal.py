"""Surprisal del token observado, utilizado como medida de referencia.

El token x_t se evalúa con la distribución predicha en t-1. Esta cantidad
se conserva separada de la divergencia entre distribuciones consecutivas.
"""

from __future__ import annotations

import numpy as np
import torch


def surprisal_series(logits: torch.Tensor, input_ids: torch.Tensor, chunk: int = 64) -> np.ndarray:
    """Devuelve -ln p_(t-1)(x_t), en nats y con NaN en la primera posición.

    Los logits tienen forma (tokens, vocabulario); los identificadores tienen
    forma (tokens,) y deben estar en el mismo dispositivo que los logits.
    """
    T = logits.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    ids = input_ids.to(torch.long)
    for start in range(1, T, chunk):
        end = min(start + chunk, T)
        lp = torch.log_softmax(logits[start - 1 : end - 1].float(), dim=-1)
        out[start:end] = (
            (-lp.gather(-1, ids[start:end].unsqueeze(-1)).squeeze(-1)).double().cpu().numpy()
        )
    return out
