# Piloto instrumental AFT8

Código para estudiar cambios en las distribuciones predictivas y en la saliencia de un modelo de lenguaje durante la lectura de textos mediante teacher forcing.

El alcance científico es un estudio instrumental de tres dimensiones de AFT8 sobre Mistral-7B-v0.3. Los resultados deberán interpretarse respecto al corpus y a los procedimientos evaluados; el piloto no constituye una validación de AFT8 completo.

## Estado del proyecto

El paquete incluye métricas, corpus verificable, controles, extractor Mistral, validación de resultados y análisis con figuras. La integración del extractor se ha probado con un modelo diminuto en CPU; su ejecución científica en GPU sigue pendiente. Las decisiones metodológicas abiertas se documentan expresamente. La preparación de la ejecución se describe en [RunPod](docs/runpod.md).

No se distribuyen resultados científicos con esta base. La ejecución completa y el entorno fijado se incorporarán tras completar la validación de datos y de los módulos de análisis.

## Instalación de desarrollo

Se requiere Python 3.11 o posterior. Crear un entorno virtual e instalar el paquete desde la raíz:

```bash
python -m pip install -e ".[data,analysis]"
python -m unittest discover -s tests -v
```

La instalación anterior sirve para desarrollo. Los rangos de dependencias de `pyproject.toml` no constituyen todavía un entorno de reproducción científica fijado.

## Organización

| Carpeta | Contenido |
|---|---|
| `src/aft8/` | Cálculo de métricas |
| `tests/` | Propiedades matemáticas y contratos de las funciones |
| `docs/` | Método y requisitos de reproducción |
| `data/` | Organización prevista para datos originales y derivados |

Consultar [el método](docs/metodo.md), [los requisitos de reproducción](docs/reproducibilidad.md), [la organización de datos](data/README.md), [el acceso y validación de resultados](docs/resultados.md) y [el análisis estadístico](docs/analisis.md).
