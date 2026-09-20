"""Cambio predictivo entre distribuciones de posiciones consecutivas.

ΔP es la divergencia de Jensen–Shannon sobre el vocabulario completo, con
logaritmos naturales. Su rango teórico es [0, ln(2)]. El primer token carece
de distribución anterior y se representa mediante NaN.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12
TECHO_NATS = float(np.log(2.0))


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Calcula JS(p, q) sobre la última dimensión de tensores de probabilidades."""
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + EPS) - torch.log(m + EPS))).sum(-1)
    kl_qm = (q * (torch.log(q + EPS) - torch.log(m + EPS))).sum(-1)
    return torch.clamp(0.5 * (kl_pm + kl_qm), min=0.0)


def delta_p_series(logits: torch.Tensor, chunk: int = 32) -> np.ndarray:
    """Convierte logits (tokens, vocabulario) en una serie de ΔP en nats.

    Las probabilidades y divergencias se calculan en float32. El resultado
    se almacena como float64, sin aumentar la precisión del cálculo previo.
    Los logits deben estar separados del grafo de atribución.
    """
    T = logits.shape[0]
    out = np.full(T, np.nan, dtype=np.float64)
    for start in range(1, T, chunk):
        end = min(start + chunk, T)
        cur = torch.softmax(logits[start:end].float(), dim=-1)
        prv = torch.softmax(logits[start - 1:end - 1].float(), dim=-1)
        out[start:end] = js_divergence(prv, cur).double().cpu().numpy()
    return out
