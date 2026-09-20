# Datos

`raw/` se reserva para la fuente original autorizada y `processed/` para los datos derivados. La conversión deberá conservar el contenido textual, identificar cada registro y verificar los hashes esperados.

La fuente autorizada es `AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx`, hoja `Hypotheses`, registros A001–A030. Su SHA-256 es `7dc1af0a69226ee03d92d8eb7a930f53f7fcc67f6ec956ebea10e11e87f6816c`. El hash de dataset mencionado dentro de la plantilla corresponde a otra referencia y no al archivo XLSX autorizado; se conserva esa distinción.

Con el libro en `data/raw/`, ejecutar desde la raíz del proyecto:

```bash
python -m pip install -e ".[data]"
aft8-import-corpus data/raw/AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx data/processed/pilot30.jsonl
```

El importador comprueba el libro, los treinta identificadores y los hashes de los textos. La salida conserva los campos científicos de la fuente; todavía no es el formato de entrada de un extractor migrado. El nombre de columna `preregistered_hypothesis` procede del libro y no acredita por sí solo el registro previo de los análisis posteriores.

Los objetivos de evaluación se mantendrán separados de los campos entregados al extractor. Cada anotación tendrá una especificación y versión para que el análisis pueda repetirse sin decisiones manuales implícitas.

La distribución del corpus aún no está configurada. La versión de entrega deberá incluir los datos autorizados o un procedimiento de obtención verificable; una carpeta vacía o una ruta privada no será suficiente para reproducir el TFM.
