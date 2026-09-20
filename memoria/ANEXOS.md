# Material complementario de la memoria

Los anexos reúnen definiciones, código y evidencia detallada. Los apuntes del máster se citan como fuentes docentes; no se redistribuyen dentro del proyecto.

## A. Método y ejecución

- [Definiciones y alcance](../docs/metodo.md).
- [Validación de la extracción recibida](../docs/validacion_runpod.md).
- [Ejecución del extractor](../docs/runpod.md).
- [Condiciones de reproducción](../docs/reproducibilidad.md).
- [Código Python](../src/aft8/) y [pruebas](../tests/).

## B. Datos y decisiones de evaluación

- [Procedencia y organización de datos](../data/README.md).
- [Diseño de controles](../config/control_design.json).
- [Anotaciones candidatas](../config/gxa_annotations_candidate.json).
- [Pares de transferencia archivados](../config/gxa_transfers_archived.json).
- [Reglas para spans y controles](../docs/spans.md).

La autorización para difundir el material de AFT8-15 ha sido confirmada por el autor. La publicación debe conservar la procedencia y las licencias de componentes de terceros. La elección de soporte GxA y la discrepancia del smoke de referencia permanecen abiertas; las anotaciones candidatas no constituyen aprobación metodológica.

## C. Correspondencia entre memoria y resultados

Las rutas de esta tabla son salidas del análisis del paquete con SHA-256 `1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b`. Los comandos para generarlas están en el anexo A. Los resultados ignorados por Git deberán entregarse como artefactos adicionales: una ruta de esta tabla no implica que ya exista una descarga pública.

| Apartado | Evidencia relativa a la raíz del repositorio |
|---|---|
| 4. Corpus | `data/processed/pilot30.jsonl`, `controls.jsonl` y `config/control_design.json` |
| 7.1. Aciertos | `outputs/analysis_runpod/global_descriptive.csv`, `nominal_binomial.csv` |
| 7.1. Comparaciones | `outputs/analysis_runpod/paired_baselines.csv`, `conditional_permutations.csv` |
| 7.2. Distribuciones | `outputs/diagnostics_runpod/predictive_distribution.csv`, `resolution_ratios.csv` |
| 7.2. Factorial | `outputs/analysis_runpod/factorial_document_bootstrap.csv` |
| 7.3. Perfiles | `outputs/analysis_runpod/control_profile_sensitivity.csv` |
| 8.1. GxA | `outputs/analysis_runpod/gxa_distributions.csv`, `gxa_per_text.csv`, `gxa_support_sensitivity.csv` |
| 8.2. Spans | `outputs/diagnostics_runpod/span_results.csv`, `span_summary.csv`, `span_matching_audit.csv`, `span_trajectories.csv` |
| 9. Integridad | `outputs/validation_runpod/summary.json`, `original_validation.json`, `control_validation.json`, `smoke_validation.json` |
| 9. Reproducción | `outputs/validation_runpod/previous_arrays_comparison.json`, `analysis_comparison.json`, `smoke_vs_full_arrays.json` |
| 10. Clustering | `outputs/diagnostics_runpod/clustering_descriptive.json`, `states.parquet` |

Los 30 mapas individuales y las siete trayectorias se encuentran en `outputs/diagnostics_runpod/figures/`. La memoria incluye un ejemplo y conserva los denominadores globales para evitar que ese ejemplo sustituya la evidencia conjunta.

## D. Fuentes docentes de la conexión con el máster

Numeración correspondiente a páginas físicas de los PDF del curso, incluida su portada:

| Fuente | Páginas | Aplicación explicada |
|---|---|---|
| NLP, Tema 1 - Basics | 2-5 | Tokens, subtokens, representación numérica y alineación |
| NLP, Tema 2 - Embeddings | 2-3 | Vectores latentes y límites de interpretación |
| NLP, Tema 4 - Attention | 10, 19, 31 y 35 | Q/K/V, softmax, decoder y causalidad |
| NLP, Tema 5 - Hugging Face | 3-6 | Transformers, modelos preentrenados y licencias |
| Estadística descriptiva | 3-5 | Descripción frente a inferencia y tipos de variables |
| Cálculo de probabilidades | 2 | Modelos probabilísticos y su relación con la incertidumbre |
| Inferencia paramétrica | 2-3 y 38-41 | Muestra, estimación, intervalos, contrastes y p-valores |
| Inferencia no paramétrica | 3 | Evaluación sin una distribución poblacional completamente especificada |

Permutación, bootstrap por documento y ajuste BH se presentan como ampliaciones aplicadas. No se atribuye a los apuntes un desarrollo específico que no se haya comprobado en ellos.
