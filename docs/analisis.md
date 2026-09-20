# Análisis estadístico

El reanálisis utiliza exclusivamente Mistral-7B-v0.3 y conserva los artefactos originales. Su estado es exploratorio. Las hipótesis textuales del libro y los procedimientos estadísticos añadidos se distinguen: la existencia de una columna llamada `preregistered_hypothesis` no acredita la preespecificación de este reanálisis.

## Ejecución

Después de importar el corpus y reproducir los controles según `resultados.md`:

```bash
python -m pip install -e ".[data,analysis]"
aft8-analyze data/raw/results-mistral.tar.gz data/processed/pilot30.jsonl data/processed/controls.jsonl --output outputs/analysis
aft8-diagnostics data/raw/results-mistral.tar.gz data/processed/pilot30.jsonl --config config --output outputs/diagnostics
```

El primer comando genera tablas, tres figuras y `report.md`. El segundo genera spans, transferencias, mapas individuales y diagnósticos instrumentales. Los manifiestos registran hashes de datos y código, semillas y versiones instaladas. Documentan el entorno utilizado; el entorno de extracción GPU se fijará por separado.

## Definiciones y unidades

- Máximos: v.2–v.6; ΔP máximo, ΔE suma con signo y GxA máximo. El desempate elige el primer verso; se informa del número de empates y se usa rango medio. Un máximo de ΔE no contrasta por sí solo hipótesis de dirección negativa. El resumen de máximos tampoco prueba focalización o transferencia GxA.
- Modos: valores originales y residuos respecto a la media por posición de los 30 textos. El ajuste utiliza el mismo corpus evaluado; no mide generalización ni prueba que eliminar un perfil sea una corrección suficiente.
- Familias objetivo: P1→ΔP, P2→ΔE, P3→GxA, P4→ΔP/ΔE, P5→las tres. Son 48 puntuaciones procedentes de 30 textos. El acierto global se informa descriptivamente y no recibe un contraste binomial que trate esas 48 filas como independientes.
- Baselines: entropía, atención cruda, PCA, ICA y norma de una proyección PCA de ocho componentes, todos comparados sobre los textos objetivo de cada métrica. La capa oculta es 32. Las proyecciones se ajustan sobre el corpus completo con semilla 0. El baseline GxA sin ajuste se compara solo con GxA; en el modo crudo es una identidad, no una prueba de valor añadido.
- McNemar: exacto bilateral sobre pares discordantes. Estas comparaciones son exploratorias; la dependencia inducida por ajustar proyecciones y perfil sobre todo el corpus limita su interpretación inferencial.
- Referencia aleatoria: una predicción uniforme por documento, compartida entre las métricas de ese documento. Se informa cada denominador y se conserva la ponderación del resumen global.

## Multiplicidad y permutaciones

Los p brutos y ajustados tienen columnas diferentes. Las familias BH son ocho contrastes binomiales por modo, dieciséis comparaciones emparejadas por modo y cuatro ámbitos de permutación por modo y esquema. No se afirma control de FDR sobre toda la exploración. La referencia binomial uniforme p=0,2 es nominal y no incorpora automáticamente las preferencias posicionales ni el diseño del corpus.

Se ejecutan 20.000 permutaciones con semilla 0, tanto entre todos los textos como dentro de cada familia. El objetivo de un texto se mantiene compartido entre sus métricas. El p Monte Carlo usa `(b + 1)/(B + 1)`. Ambos esquemas requieren intercambiabilidad de etiquetas en sus respectivos bloques; la estratificación se incorpora como sensibilidad, no como aprobación de ese supuesto. No se elige un esquema por su p ni se denomina a su media «azar real» universal.

## Diagnósticos e interpretación

La matriz factorial conserva medias y medianas por token claramente separadas. Los intervalos del 95 % remuestrean 30 documentos, con 10.000 réplicas y semilla 0, tras calcular medias de diferencias por documento. Estiman una media con igual peso por texto; la media ponderada por tokens se conserva como cantidad diferente. Los intervalos no establecen representatividad del corpus diseñado.

Se presentan las correlaciones entre perfiles medios original/control para v.1–v.6 y v.2–v.6. Una correlación de cinco o seis medias no demuestra igualdad por texto ni ausencia de sensibilidad al contenido. Los controles D1 pueden alterar la preparación contextual del giro y D2 altera el orden; ninguno justifica por sí solo una atribución causal exclusiva.

GxA dispone de resumen de 30 textos y del subconjunto de 12 con hipótesis explícita, distribución de ambas variantes, máximos y percentil del verso objetivo. Los rangos ascendentes medios divididos entre seis tienen esperanza 7/12 bajo objetivo uniforme entre versos, no 0,5. Los objetivos reales no son uniformes, por lo que esa referencia es ilustrativa. La diferencia objetivo–resto incluida aquí se refiere a redistribución por verso, no a masa sobre spans.

## Spans, instrumentación y espacio de estados

`aft8-diagnostics` calcula masas sobre spans, sus trayectorias, ventanas y controles por palabras de contenido. Se versionan los pares archivados con controles actuales, las anotaciones candidatas y las alternativas. Genera 30 mapas de saliencia con escala común y siete trayectorias origen–destino. Consultar `spans.md` para las convenciones y decisiones pendientes.

Los cocientes de resolución y fracciones de techo aparecen separados. La sensibilidad a normalización aplica transformaciones afines después de agregar por verso, con parámetros ajustados sobre todos los tokens no especiales finitos de cada métrica: desviación típica muestral y escala robusta 1,4826·MAD. No equivale a normalizar tokens antes de sumarlos; en ΔE esa operación puede cambiar los máximos por la distinta cantidad de tokens de cada verso.

El clustering es descriptivo, con HDBSCAN de scikit-learn y PCA explícitos. Se conservan las etiquetas archivadas y una comparación de particiones; el recálculo local no las reproduce exactamente. No se reutiliza el p de permutación por columnas, pues la independencia temporal de tokens no está justificada. Su inclusión final en la memoria y la explicación de la diferencia de particiones quedan pendientes; ninguna de las dos versiones identifica categorías emocionales.

Falta enlazar todas las cifras y figuras elegidas con la memoria, confirmar las anotaciones y fijar el entorno final. La estabilidad numérica GPU y un control de sensibilidad semántica específica no pueden deducirse de estos diagnósticos.

La alineación de soportes GxA y el smoke A001/A002 frente a A001/A013 permanecen abiertos por decisión del proyecto.
