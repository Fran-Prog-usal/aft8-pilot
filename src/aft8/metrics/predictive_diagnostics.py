"""Diagnósticos predictivos con temperaturas fijas y unión top-k más masa residual."""
from __future__ import annotations
import numpy as np
import torch
from .delta_p import js_divergence
EPS = 1e-12
K_DIAGNOSTICO = 100
TEMPERATURAS = (0.5, 1.0, 2.0)

def _etiqueta(T: float) -> str:
    return 'T' + f'{T:.1f}'.replace('.', '')

def probabilidades(logits: torch.Tensor, T: float=1.0) -> torch.Tensor:
    z = logits.float()
    if T != 1.0:
        z = z / float(T)
    return torch.softmax(z, dim=-1)

def js_topk_residual(p: torch.Tensor, q: torch.Tensor, k: int=K_DIAGNOSTICO) -> torch.Tensor:
    k = min(int(k), p.shape[-1])
    idx = torch.cat([p.topk(k, dim=-1).indices, q.topk(k, dim=-1).indices], dim=-1)
    idx, _ = idx.sort(dim=-1)
    repetido = torch.zeros_like(idx, dtype=torch.bool)
    repetido[:, 1:] = idx[:, 1:] == idx[:, :-1]
    pk = torch.gather(p, -1, idx).masked_fill(repetido, 0.0)
    qk = torch.gather(q, -1, idx).masked_fill(repetido, 0.0)
    resto_p = (1.0 - pk.sum(-1, keepdim=True)).clamp(min=0.0)
    resto_q = (1.0 - qk.sum(-1, keepdim=True)).clamp(min=0.0)
    return js_divergence(torch.cat([pk, resto_p], dim=-1), torch.cat([qk, resto_q], dim=-1))

def delta_p_topk_series(logits: torch.Tensor, k: int=K_DIAGNOSTICO, T: float=1.0, chunk: int=32) -> np.ndarray:
    n = logits.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for ini in range(1, n, chunk):
        fin = min(ini + chunk, n)
        cur = probabilidades(logits[ini:fin], T)
        prv = probabilidades(logits[ini - 1:fin - 1], T)
        out[ini:fin] = js_topk_residual(prv, cur, k).double().cpu().numpy()
    return out

def series_a_temperatura(logits: torch.Tensor, T: float, k: int=K_DIAGNOSTICO, chunk: int=32) -> dict[str, np.ndarray]:
    n = logits.shape[0]
    dp = np.full(n, np.nan, dtype=np.float64)
    dpk = np.full(n, np.nan, dtype=np.float64)
    H = np.empty(n, dtype=np.float64)
    for ini in range(0, n, chunk):
        fin = min(ini + chunk, n)
        p = probabilidades(logits[ini:fin], T)
        H[ini:fin] = (-(p * torch.log(p + EPS)).sum(-1)).double().cpu().numpy()
    for ini in range(1, n, chunk):
        fin = min(ini + chunk, n)
        cur = probabilidades(logits[ini:fin], T)
        prv = probabilidades(logits[ini - 1:fin - 1], T)
        dp[ini:fin] = js_divergence(prv, cur).double().cpu().numpy()
        dpk[ini:fin] = js_topk_residual(prv, cur, k).double().cpu().numpy()
    dE = np.full(n, np.nan, dtype=np.float64)
    dE[1:] = H[1:] - H[:-1]
    return {'delta_P': dp, 'delta_P_topk': dpk, 'entropy_H': H, 'delta_E': dE}

def capa_diagnostica(logits: torch.Tensor, k: int=K_DIAGNOSTICO, temperaturas: tuple[float, ...]=TEMPERATURAS, chunk: int=32) -> dict[str, np.ndarray]:
    salida: dict[str, np.ndarray] = {}
    for T in temperaturas:
        et = _etiqueta(T)
        s = series_a_temperatura(logits, T, k=k, chunk=chunk)
        salida[f'deltaP_full_{et}'] = s['delta_P']
        salida[f'deltaP_top{k}_{et}'] = s['delta_P_topk']
        salida[f'entropy_H_{et}'] = s['entropy_H']
        salida[f'deltaE_{et}'] = s['delta_E']
    return salida
