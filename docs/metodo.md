# Método

El piloto observa distribuciones de siguiente token mediante teacher forcing. No entrena el modelo ni genera continuaciones. El análisis se limita a las métricas, corpus y condiciones expresamente definidos para el experimento.

El modelo del estudio es Mistral-7B-v0.3 base. El alcance de este TFM es de un único modelo.

## Métricas predictivas implementadas

| Cantidad | Definición | Unidad |
|---|---|---|
| ΔP | JS entre predictivas consecutivas, sobre el vocabulario completo | Nats; techo ln(2) |
| H | Entropía de la distribución predictiva | Nats |
| ΔE | H(t) − H(t−1) | Nats, con signo |
| Surprisal | −ln p(t−1, token observado en t) | Nats |

Probabilidades y métricas se calculan en float32. Los arrays de salida pueden almacenarse en float64; esa conversión no aumenta la precisión previa. El valor inicial de las series de transición se marca como ausente mediante NaN.

El top-k conserva índices, logits, logsumexp completo y rango del token observado. Permite inspeccionar las probabilidades guardadas; no permite reconstruir el vocabulario completo ni recalcular arbitrariamente las transformaciones de temperatura.

## Especificación pendiente de completar

El extractor implementa saliencia por posición sobre el logaritmo de probabilidad del siguiente token observado, promediando el valor absoluto de atención por gradiente entre capas y cabezas. Conserva saliencia cruda, normalizada y ambas variantes de soporte. Las agregaciones, alineación y tratamiento de tokens especiales están implementados; la ejecución científica del nuevo extractor en GPU sigue pendiente.

Permanecen abiertos dos puntos de validación:

- **Alineación de soportes GxA:** pendiente de resolución entre soporte común renormalizado y relleno de cero. Las dos variantes se conservarán identificadas por separado.
- **Smoke test:** el paquete disponible contiene A001/A002 y la plantilla requiere A001/A013. La conformidad con ese requisito queda pendiente de resolución documentada.

Los diagnósticos numéricos, las comparaciones estadísticas y las comprobaciones exploratorias deberán distinguirse en los informes. Un umbral descriptivo no establece por sí mismo validez instrumental; la ausencia de significación tampoco demuestra ausencia de efecto.
