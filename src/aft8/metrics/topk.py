"""Exportación compacta de logits para inspección de predicciones.

El logsumexp del vocabulario completo permite reconstruir las probabilidades
de los elementos guardados. El top-k no permite recuperar por sí solo todas
las distribuciones ni recalcular entropía o JS sobre el vocabulario completo.
"""
from __future__ import annotations

import numpy as np
import torch


def topk_logits(logits: torch.Tensor, input_ids: torch.Tensor, k: int = 50,
                chunk: int = 64) -> dict[str, np.ndarray]:
    """Guarda índices, logits, normalizador y rango del siguiente token real.

    Las matrices top-k tienen forma (tokens, min(k, vocabulario)). El rango
    observado empieza en cero; -1 representa la posición sin predicción
    anterior. En caso de empate se cuentan solo logits estrictamente mayores.
    Los identificadores y logits deben estar en el mismo dispositivo.
    """
    T, V = logits.shape
    k = int(min(k, V))
    idx = np.zeros((T, k), dtype=np.int32)
    val = np.zeros((T, k), dtype=np.float32)
    lse = np.zeros(T, dtype=np.float32)
    obs = np.full(T, -1, dtype=np.int32)
    ids = input_ids.to(torch.long)

    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        # El normalizador se evalúa en float32 incluso con pesos en bfloat16.
        bloque = logits[a:b].float()
        v, i = torch.topk(bloque, k, dim=-1)
        idx[a:b] = i.cpu().numpy().astype(np.int32)
        val[a:b] = v.cpu().numpy().astype(np.float32)
        lse[a:b] = torch.logsumexp(bloque, dim=-1).cpu().numpy().astype(np.float32)

    # La observación en t corresponde a la predicción emitida en t-1.
    for a in range(1, T, chunk):
        b = min(a + chunk, T)
        bloque = logits[a - 1:b - 1].float()
        objetivo = ids[a:b].unsqueeze(-1)
        propio = bloque.gather(-1, objetivo)
        obs[a:b] = (bloque > propio).sum(dim=-1).cpu().numpy().astype(np.int32)

    return {"logits_topk_idx": idx, "logits_topk_val": val,
            "logits_logsumexp": lse, "observed_rank": obs}


def reconstruye_probabilidades(idx: np.ndarray, val: np.ndarray,
                               lse: np.ndarray) -> np.ndarray:
    """Recupera las probabilidades en el orden de los índices top-k guardados.

    Los índices identifican las columnas del vocabulario; la transformación
    numérica usa los logits y el normalizador, sin renormalizar el subconjunto.
    """
    return np.exp(val - lse[:, None])
