"""Saliencia por posición y redistribución en dos soportes, conservados por separado."""
from __future__ import annotations
import numpy as np
import torch
EPS = 1e-12

def salience_matrix(attentions: list[torch.Tensor], model_out, targets: torch.Tensor, logits: torch.Tensor, devolver_cruda: bool=False):
    T = logits.shape[0]
    S = np.full((T, T), np.nan, dtype=np.float64)
    S_cruda = np.full((T, T), np.nan, dtype=np.float64) if devolver_cruda else None
    logprobs = torch.log_softmax(logits, dim=-1)
    for t in range(T - 1):
        y_t = logprobs[t, targets[t]]
        grads = torch.autograd.grad(y_t, attentions, retain_graph=True, allow_unused=False)
        acc = None
        for A, G in zip(attentions, grads):
            a = A[0] if A.dim() == 4 else A
            g = G[0] if G.dim() == 4 else G
            fila = (a[:, t, :t + 1] * g[:, t, :t + 1]).abs().mean(dim=0)
            acc = fila if acc is None else acc + fila
        s = (acc / len(attentions)).double()
        if devolver_cruda:
            S_cruda[t, :t + 1] = s.detach().cpu().numpy()
        S[t, :t + 1] = (s / (s.sum() + EPS)).detach().cpu().numpy()
    return (S, S_cruda) if devolver_cruda else S

def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log(a + EPS) - np.log(b + EPS))))
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))

def redistribution_series(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = S.shape[0]
    comun = np.full(T, np.nan)
    pad = np.full(T, np.nan)
    for t in range(1, T - 1):
        prev, cur = (S[t - 1], S[t])
        if not np.isfinite(prev[:t]).all() or not np.isfinite(cur[:t + 1]).all():
            continue
        a = prev[:t]
        b = cur[:t]
        sa, sb = (a.sum(), b.sum())
        if sa > 0 and sb > 0:
            comun[t] = _js(a / sa, b / sb)
        ap = np.concatenate([prev[:t], [0.0]])
        pad[t] = _js(ap, cur[:t + 1])
    return (comun, pad)

def mass_by_verse(S: np.ndarray, line_id: np.ndarray, n_versos: int) -> np.ndarray:
    masa = np.zeros(n_versos + 1)
    validas = [t for t in range(S.shape[0]) if np.isfinite(S[t]).any()]
    for t in validas:
        fila = S[t]
        for j in range(t + 1):
            v = int(line_id[j])
            if 0 < v <= n_versos and np.isfinite(fila[j]):
                masa[v] += fila[j]
    return masa[1:] / max(1, len(validas))

def raw_attention_received(attentions: list[torch.Tensor]) -> np.ndarray:
    acc = None
    for A in attentions:
        a = A[0] if A.dim() == 4 else A
        recibida = a.sum(dim=1).mean(dim=0)
        acc = recibida if acc is None else acc + recibida
    return (acc / len(attentions)).detach().double().cpu().numpy()
