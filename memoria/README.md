# Memoria y anexos

`TFM_memoria_revisada.md` es la fuente editable. `Francisco_Jose_Martinez_Fernandez_TFM_AFT8_15.pdf` es la versión paginada de entrega. `ANEXOS.md` relaciona definiciones, código, datos, evidencia y fuentes docentes.

La memoria se compone en A4, Arial 11, interlineado de 14,3 puntos y márgenes laterales de 24 mm. El límite de 20 caras incluye bibliografía; portada e índice quedan fuera. El recuento real, los hashes de fuente y PDF y las páginas de cada sección están en `paginacion.json`. Los anexos se entregan por separado.

## Regenerar el PDF

Desde la raíz del repositorio:

```bash
python -m pip install -r memoria/requirements.txt
python memoria/generar_pdf.py
```

El generador actualiza directamente el PDF de entrega y `paginacion.json`.

El generador usa las fuentes Arial instaladas en Windows. En otro sistema se puede indicar un directorio que contenga `arial.ttf`, `arialbd.ttf`, `ariali.ttf` y `arialbi.ttf`:

```bash
python memoria/generar_pdf.py --font-dir /ruta/a/fuentes
```

Las fuentes deben obtenerse con su licencia correspondiente; no se incluyen en el repositorio. Las tres figuras copiadas en `figuras/` proceden del análisis del paquete de resultados identificado en los anexos. El generador recalcula el índice y falla si el cuerpo supera 20 páginas. La inspección visual sigue siendo necesaria tras cambiar texto, fuentes o figuras.

Los puntos metodológicos pendientes se conservan dentro de la memoria. La versión no presupone su aprobación ni la publicación de los artefactos complementarios. La denominación de AFT8-15 como proyecto y la autorización de publicación reflejan la información proporcionada por el autor.
