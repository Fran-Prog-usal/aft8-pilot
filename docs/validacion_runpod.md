# Validación de la extracción Mistral

El paquete `results-mistral.tar.gz` tiene SHA-256 `1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b` y 264.600.175 bytes. El hash calculado coincide con el archivo SHA entregado. Contiene 113 archivos; los 110 artefactos enumerados por los tres manifiestos coinciden con sus hashes. Los otros tres archivos son los propios manifiestos, cubiertos por el hash del contenedor.

## Datos y entorno

| Conjunto | Textos | Tokens, incluido BOS | Versos | Tiempo registrado (s) |
|---|---:|---:|---:|---:|
| Smoke A001/A013 | 2 | 127 | 12 | 39,736 |
| Originales | 30 | 2.046 | 180 | 199,035 |
| Controles | 60 | 4.095 | 360 | 395,034 |

Los tiempos son los registrados por el extractor; no representan toda la duración facturable del pod, instalación o descarga. Los tres manifiestos indican estado completo. Modelo y tokenizador: `mistralai/Mistral-7B-v0.3`, revisión `caa1feb0e54d415e2df31207e5f4e273e33509b1`, bfloat16 y atención eager. Entorno: NVIDIA A40, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Transformers 5.17.0, Tokenizers 0.23.1, NumPy 2.1.2, pandas 3.0.5 y PyArrow 25.0.1.

Los hashes de entrada coinciden con la proyección sin hipótesis del corpus autorizado y sus controles. El Excel fuente tiene SHA-256 `7dc1af0a69226ee03d92d8eb7a930f53f7fcc67f6ec956ebea10e11e87f6816c`. Se han comprobado los IDs previstos, cobertura textual, agregaciones por verso y decode, IDs y offsets con el tokenizador auténtico: 92 registros contando el smoke, sin incidencias. Todos los campos NPZ de vectores se leen sin pickle.

Los hashes de código del manifiesto coinciden con los archivos usados para la auditoría, conservados en `executed-source.zip` en la release. El código revisado del repositorio se verifica por separado y tiene otros hashes. Frente al ZIP de preparación existían diferencias de finales de línea y líneas vacías al final de algunos archivos. Se comprobó igualdad del texto tras normalizar exclusivamente esos elementos y del árbol sintáctico completo de los 22 módulos; no se afirma identidad binaria del código del ZIP de preparación.

## Comparación numérica y análisis

Frente al paquete de referencia de SHA-256 `2656d5c571da2a86102537ccf0a07301234367d1efb69134e9d668769c32418c`, los 3.240 arrays numéricos comunes de los 90 textos son exactamente iguales, incluidas las posiciones NaN. La comparación aplica también las tolerancias predefinidas rtol=1e-5 y atol=1e-7, sin diferencias fuera de ellas. El alcance es la intersección de campos de vectores; no se equipara con igualdad binaria de todos los archivos o de las muestras de atención.

Los vectores de A001/A013 del smoke coinciden con sus correspondientes extracciones del conjunto completo. Es una repetición interna de dos textos en procesos separados, no una segunda ejecución independiente completa del paquete limpio.

Se regeneraron 16 tablas CSV estadísticas y siete de diagnósticos con este paquete: sus valores y bytes coinciden con los del reanálisis sobre la referencia en el mismo entorno local. Se mantienen 6/48 aciertos crudos y 15/48 residuales. El menor p BH de los ocho contrastes nominales residuales es 0,13568. Los resultados siguen siendo exploratorios y no acreditan validación general de AFT8.

También se generaron 30 mapas GxA, siete trayectorias, tres figuras estadísticas y una proyección PCA. El clustering local conserva siete grupos, 694/1.986 estados de ruido y silhouette −0,1019036. El paquete de extracción limpio no contiene etiquetas previas de clustering. La discrepancia ya documentada respecto a las etiquetas del paquete de referencia sigue siendo una cuestión separada.

## Repetir el análisis del paquete

Desde la raíz del proyecto, con el paquete en `outputs/results-mistral.tar.gz` y los corpus ya importados:

```bash
aft8-validate-results outputs/results-mistral.tar.gz data/processed/pilot30.jsonl --run original --archive-sha256 1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b --output outputs/validation_runpod/original.json
aft8-validate-results outputs/results-mistral.tar.gz data/processed/controls.jsonl --run control --archive-sha256 1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b --output outputs/validation_runpod/control.json
aft8-analyze outputs/results-mistral.tar.gz data/processed/pilot30.jsonl data/processed/controls.jsonl --archive-sha256 1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b --output outputs/analysis_runpod
aft8-diagnostics outputs/results-mistral.tar.gz data/processed/pilot30.jsonl --config config --archive-sha256 1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b --output outputs/diagnostics_runpod
```

Los comandos de validación recalculan agregaciones y offsets; la auditoría de recepción añadió la comprobación independiente con el tokenizador. Los informes generados se encuentran en `outputs/analysis_runpod/report.md` y `outputs/diagnostics_runpod/report.md`. Ambos corresponden al nuevo hash; el enlace relativo a diagnósticos del informe estadístico genérico apunta a la carpeta convencional `diagnostics`, por lo que para esta ejecución debe consultarse `diagnostics_runpod`.

## Decisiones abiertas

- C02/D04: selección de soporte GxA. Se mantienen ambas variantes sin atribuir aprobación metodológica.
- Smoke histórico A001/A002 frente a A001/A013: se conserva abierto y documentado; la ejecución nueva aporta el smoke solicitado sin alterar el archivo de referencia.
- Anotaciones exploratorias A004/A016/A020 y tratamiento del clustering en la memoria.
- La distribución y la instalación nueva se documentan en [la revisión técnica](revision_tecnica.md), sin convertir estas comprobaciones en cierre metodológico.
