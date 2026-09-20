# Acceso y validación de resultados

El paquete vigente contiene resultados de Mistral-7B-v0.3: treinta originales, sesenta controles y el smoke A001/A013. Su SHA-256 es `1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b`. Es el paquete esperado por defecto por los comandos de lectura, validación y análisis.

Descargar los archivos de la [release de reproducción](https://github.com/Fran-Prog-usal/aft8-pilot/releases/tag/v0.1.1-reproducible) en `data/raw/`: paquete, Excel autorizado, código ejecutado y evidencia de análisis. `config/release_assets.json` identifica sus hashes y tamaños. `SHA256SUMS.txt` permite comprobarlos también con `sha256sum -c SHA256SUMS.txt` desde esa carpeta. Los manifiestos internos originales se conservan sin modificaciones.

El paquete histórico de referencia tiene SHA-256 `2656d5c571da2a86102537ccf0a07301234367d1efb69134e9d668769c32418c`; `config/result_inventory.json` corresponde exclusivamente a ese paquete. Para leerlo se debe indicar ese hash mediante `--archive-sha256`. No se intercambian sus nombres, inventarios ni smoke con los del paquete vigente.

## Ejecución local

Tras disponer del libro autorizado y del paquete, ejecutar desde la raíz del repositorio:

```bash
python -m pip install -r requirements/analysis-windows.lock.txt
python -m pip install --no-deps -e .
aft8-import-corpus data/raw/AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx data/processed/pilot30.jsonl
aft8-build-controls data/processed/pilot30.jsonl config/control_design.json data/processed/controls.jsonl
aft8-validate-results data/raw/results-mistral.tar.gz data/processed/pilot30.jsonl --run original --output outputs/validation/original.json
aft8-validate-results data/raw/results-mistral.tar.gz data/processed/controls.jsonl --run control --output outputs/validation/control.json
python -m unittest discover -s tests -v
```

El diseño contiene las permutaciones exactas realizadas, con índices basados en 1. D1 intercambia dos versos y equilibra los objetivos entre v.2–v.6; D2 es un desarreglo completo sin verso objetivo. Su reproducción no vuelve a resolver empates de un optimizador ni modifica el diseño observado. La semilla de generación se conserva como metadato del diseño.

`ResultArchive` verifica el hash antes de leer y proporciona tablas, manifiestos y arrays numéricos. Omite el array `tokens` de tipo object; los textos de tokens están disponibles en Parquet. No requiere `allow_pickle=True` ni extrae rutas del contenedor. El archivo temporal de lectura se elimina al cerrar el contexto.

## Alcance de las comprobaciones

Se han regenerado las agregaciones de los 30 originales y los 60 controles con tolerancia relativa `1e-6` y absoluta `1e-8`, sin discrepancias. Los controles reproducen exactamente los textos, orígenes, permutaciones y versos objetivo empleados en la extracción.

La agregación excluye tokens especiales, incluye saltos de línea y conserva el signo de ΔE. La masa por verso promedia la saliencia recibida sobre todas las consultas definidas; la masa asignada a posiciones especiales no se redistribuye a los versos. Ambos soportes GxA se comprueban por separado.

Los offsets cubren todos los caracteres de los 90 textos y coinciden con la asignación por máximo solapamiento y la regla de salto de línea. Además, el tokenizador de la revisión fijada reproduce exactamente el texto, los IDs y los offsets. Esta comprobación también pasa para el paquete de la [extracción limpia en RunPod](validacion_runpod.md), incluidos sus dos textos de smoke.

El resultado satisfactorio de estas comprobaciones no valida las conclusiones estadísticas ni resuelve la elección metodológica del soporte GxA. La discrepancia del smoke A001/A002 frente a A001/A013 sigue abierta.

## Anotaciones de GxA

Se distinguen los pares de transferencia de la salida archivada (`config/gxa_transfers_archived.json`) de la propuesta de anotaciones (`config/gxa_annotations_candidate.json`). Ambos incluyen hashes de procedencia. Las anotaciones son exploratorias y no se presentan como hipótesis registradas previamente.

La salida archivada contiene siete pares, de los que cinco pudieron calcularse: `cuerpo` y `monitor` no aparecen literalmente en A004 y A016. La propuesta utiliza `corazón` y `pantalla`, respectivamente. En A004 deben conservarse también las alternativas `pecho` y `mano`. Estas decisiones siguen pendientes de confirmación; importar las anotaciones no las valida.

El análisis presenta mapas y resumen de los treinta textos, separando los doce con hipótesis GxA explícita. Incluye distribución, máximos, percentil objetivo con su referencia nula, masa objetivo frente al resto y transferencias origen–destino con ventanas y controles definidos. Las cifras archivadas 5/5 no se mezclan con resultados recalculados con otras anotaciones y controles. El recorrido completo se ejecuta con `python scripts/reproduce.py`; véase el README para instalar el entorno comprobado.
