"""Saliencia por posición y redistribución en dos soportes, conservados por separado."""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


def salience_matrix(
    attentions: list[torch.Tensor],
    model_out,
    targets: torch.Tensor,
    logits: torch.Tensor,
    return_raw: bool = False,
    *,
    devolver_cruda: bool | None = None,
):
    """Promedia |atención × gradiente| por cabeza/capa y normaliza cada fila.

    `return_raw` incluye la saliencia anterior a la normalización.
    `devolver_cruda` se acepta para compatibilidad con la API inicial.
    """
    if devolver_cruda is not None:
        return_raw = devolver_cruda
    T = logits.shape[0]
    S = np.full((T, T), np.nan, dtype=np.float64)
    raw_salience = np.full((T, T), np.nan, dtype=np.float64) if return_raw else None
    logprobs = torch.log_softmax(logits, dim=-1)
    for t in range(T - 1):
        y_t = logprobs[t, targets[t]]
        grads = torch.autograd.grad(y_t, attentions, retain_graph=True, allow_unused=False)
        acc = None
        for A, G in zip(attentions, grads):
            a = A[0] if A.dim() == 4 else A
            g = G[0] if G.dim() == 4 else G
            row = (a[:, t, : t + 1] * g[:, t, : t + 1]).abs().mean(dim=0)
            acc = row if acc is None else acc + row
        s = (acc / len(attentions)).double()
        if return_raw:
            raw_salience[t, : t + 1] = s.detach().cpu().numpy()
        S[t, : t + 1] = (s / (s.sum() + EPS)).detach().cpu().numpy()
    return (S, raw_salience) if return_raw else S


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * (np.log(a + EPS) - np.log(b + EPS))))

    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))


def redistribution_series(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (soporte compartido renormalizado, soporte ampliado con cero).

    La segunda variante añade cero a la fila anterior para incluir el token
    nuevo. Los extremos y las filas no definidas conservan NaN. Se mantienen
    ambas variantes: la selección metodológica sigue abierta (docs/metodo.md).
    """
    T = S.shape[0]
    shared = np.full(T, np.nan)
    pad = np.full(T, np.nan)
    for t in range(1, T - 1):
        prev, cur = (S[t - 1], S[t])
        if not np.isfinite(prev[:t]).all() or not np.isfinite(cur[: t + 1]).all():
            continue
        a = prev[:t]
        b = cur[:t]
        sa, sb = (a.sum(), b.sum())
        if sa > 0 and sb > 0:
            shared[t] = _js(a / sa, b / sb)
        ap = np.concatenate([prev[:t], [0.0]])
        pad[t] = _js(ap, cur[: t + 1])
    return (shared, pad)


def mass_by_verse(S: np.ndarray, line_id: np.ndarray, n_verses: int) -> np.ndarray:
    mass = np.zeros(n_verses + 1)
    valid_rows = [t for t in range(S.shape[0]) if np.isfinite(S[t]).any()]
    for t in valid_rows:
        row = S[t]
        for j in range(t + 1):
            v = int(line_id[j])
            if 0 < v <= n_verses and np.isfinite(row[j]):
                mass[v] += row[j]
    return mass[1:] / max(1, len(valid_rows))


def raw_attention_received(attentions: list[torch.Tensor]) -> np.ndarray:
    acc = None
    for A in attentions:
        a = A[0] if A.dim() == 4 else A
        received = a.sum(dim=1).mean(dim=0)
        acc = received if acc is None else acc + received
    return (acc / len(attentions)).detach().double().cpu().numpy()
