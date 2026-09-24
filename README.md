# AImotional Field Theory AFT8-15

## Entrega del TFM

Autor: **Francisco Jose Martinez Fernandez**.

- [Memoria en PDF](memoria/Francisco_Jose_Martinez_Fernandez_TFM_AFT8_15.pdf).
- [Vídeo con voz en off en MP4](presentacion/Francisco_Jose_Martinez_Fernandez_Diapositivas_AFT8_15.mp4) (3 min 41 s, 33,1 MB).
- [Presentación editable](presentacion/Francisco_Jose_Martinez_Fernandez_Diapositivas_AFT8_15.pptx).
- [Anexos y correspondencia con los resultados](memoria/ANEXOS.md).
- [Código fuente](src/aft8/) y [artefactos de reproducción](https://github.com/Fran-Prog-usal/aft8-pilot/releases/tag/v0.1.1-reproducible).

El repositorio es público. Los paquetes de datos y resultados se encuentran en los archivos adjuntos de la release enlazada.

Código para estudiar cambios en las distribuciones predictivas y en la saliencia de un modelo de lenguaje durante la lectura de textos mediante teacher forcing.

El alcance científico es un estudio instrumental de tres dimensiones de AFT8 sobre Mistral-7B-v0.3. Los resultados deberán interpretarse respecto al corpus y a los procedimientos evaluados; el piloto no constituye una validación de AFT8 completo.

## Estado del proyecto

El paquete incluye métricas, corpus verificable, controles, extractor Mistral, validación de resultados y análisis con figuras. La extracción en una NVIDIA A40 está verificada para 30 originales, 60 controles y el smoke A001/A013. La [validación de la ejecución](docs/validacion_runpod.md) identifica el paquete, las comprobaciones y sus límites. Las decisiones metodológicas abiertas se documentan expresamente. La preparación de la ejecución se describe en [RunPod](docs/runpod.md).

El Excel autorizado, los resultados y la instantánea del código ejecutado se distribuyen en la [release de reproducción](https://github.com/Fran-Prog-usal/aft8-pilot/releases/tag/v0.1.1-reproducible). Sus hashes están versionados en `config/release_assets.json`.

## Reproducir el análisis sin GPU

Entorno comprobado: Windows x64 y Python 3.11. Descargar los artefactos de la release en `data/raw/` y ejecutar desde la raíz, en PowerShell:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements/analysis-windows.lock.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/reproduce.py
```

El script verifica los hashes, importa los 30 textos, construye los 60 controles, valida el paquete y genera análisis y diagnósticos en `outputs/reproduction/`. No descarga pesos ni ejecuta la extracción GPU. Exige un destino nuevo para no sobrescribir resultados. Véase [la verificación técnica](docs/revision_tecnica.md).

## Instalación de desarrollo

Se requiere Python 3.11 o posterior. Crear un entorno virtual e instalar el paquete desde la raíz:

```bash
python -m pip install -e ".[data,analysis]"
python -m unittest discover -s tests -v
```

La instalación anterior sirve para desarrollo; para reproducir el entorno comprobado debe usarse `requirements/analysis-windows.lock.txt`. El entorno de extracción Linux/CUDA está separado en `requirements/runpod.txt`.

## Estilo

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

## Organización

| Carpeta | Contenido |
|---|---|
| `src/aft8/` | Cálculo de métricas |
| `tests/` | Propiedades matemáticas y contratos de las funciones |
| `docs/` | Método y requisitos de reproducción |
| `data/` | Organización prevista para datos originales y derivados |

Consultar [el método](docs/metodo.md), [los requisitos de reproducción](docs/reproducibilidad.md), [la organización de datos](data/README.md), [el acceso y validación de resultados](docs/resultados.md) y [el análisis estadístico](docs/analisis.md).
