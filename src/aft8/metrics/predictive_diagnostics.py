"""Diagnósticos predictivos con temperaturas fijas y unión top-k más masa residual."""

from __future__ import annotations

import numpy as np
import torch

from .delta_p import js_divergence

EPS = 1e-12
DIAGNOSTIC_K = 100
TEMPERATURES = (0.5, 1.0, 2.0)


def _label(T: float) -> str:
    return "T" + f"{T:.1f}".replace(".", "")


def probabilities(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    z = logits.float()
    if T != 1.0:
        z = z / float(T)
    return torch.softmax(z, dim=-1)


def js_topk_residual(p: torch.Tensor, q: torch.Tensor, k: int = DIAGNOSTIC_K) -> torch.Tensor:
    k = min(int(k), p.shape[-1])
    idx = torch.cat([p.topk(k, dim=-1).indices, q.topk(k, dim=-1).indices], dim=-1)
    idx, _ = idx.sort(dim=-1)
    repeated = torch.zeros_like(idx, dtype=torch.bool)
    repeated[:, 1:] = idx[:, 1:] == idx[:, :-1]
    pk = torch.gather(p, -1, idx).masked_fill(repeated, 0.0)
    qk = torch.gather(q, -1, idx).masked_fill(repeated, 0.0)
    residual_p = (1.0 - pk.sum(-1, keepdim=True)).clamp(min=0.0)
    residual_q = (1.0 - qk.sum(-1, keepdim=True)).clamp(min=0.0)
    return js_divergence(torch.cat([pk, residual_p], dim=-1), torch.cat([qk, residual_q], dim=-1))


def delta_p_topk_series(
    logits: torch.Tensor, k: int = DIAGNOSTIC_K, T: float = 1.0, chunk: int = 32
) -> np.ndarray:
    n = logits.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for start in range(1, n, chunk):
        end = min(start + chunk, n)
        cur = probabilities(logits[start:end], T)
        prv = probabilities(logits[start - 1 : end - 1], T)
        out[start:end] = js_topk_residual(prv, cur, k).double().cpu().numpy()
    return out


def temperature_series(
    logits: torch.Tensor, T: float, k: int = DIAGNOSTIC_K, chunk: int = 32
) -> dict[str, np.ndarray]:
    n = logits.shape[0]
    dp = np.full(n, np.nan, dtype=np.float64)
    dpk = np.full(n, np.nan, dtype=np.float64)
    H = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        p = probabilities(logits[start:end], T)
        H[start:end] = (-(p * torch.log(p + EPS)).sum(-1)).double().cpu().numpy()
    for start in range(1, n, chunk):
        end = min(start + chunk, n)
        cur = probabilities(logits[start:end], T)
        prv = probabilities(logits[start - 1 : end - 1], T)
        dp[start:end] = js_divergence(prv, cur).double().cpu().numpy()
        dpk[start:end] = js_topk_residual(prv, cur, k).double().cpu().numpy()
    dE = np.full(n, np.nan, dtype=np.float64)
    dE[1:] = H[1:] - H[:-1]
    return {"delta_P": dp, "delta_P_topk": dpk, "entropy_H": H, "delta_E": dE}


def diagnostic_series(
    logits: torch.Tensor,
    k: int = DIAGNOSTIC_K,
    temperatures: tuple[float, ...] = TEMPERATURES,
    chunk: int = 32,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for T in temperatures:
        et = _label(T)
        s = temperature_series(logits, T, k=k, chunk=chunk)
        output[f"deltaP_full_{et}"] = s["delta_P"]
        output[f"deltaP_top{k}_{et}"] = s["delta_P_topk"]
        output[f"entropy_H_{et}"] = s["entropy_H"]
        output[f"deltaE_{et}"] = s["delta_E"]
    return output


# Alias conservados para clientes de la API inicial.
K_DIAGNOSTICO = DIAGNOSTIC_K
TEMPERATURAS = TEMPERATURES
probabilidades = probabilities
series_a_temperatura = temperature_series


def capa_diagnostica(logits, k=DIAGNOSTIC_K, temperaturas=TEMPERATURES, chunk=32):
    """Compatibilidad con el nombre y argumento de temperatura originales."""
    return diagnostic_series(logits, k=k, temperatures=temperaturas, chunk=chunk)
