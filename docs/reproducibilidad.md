# Reproducción del experimento

La reproducción científica requiere un paquete completo de código, configuración, datos y artefactos. La instalación de desarrollo disponible no acredita todavía la reproducción del piloto completo.

## Contenido requerido para una versión de resultados

1. Identificador del modelo y revisión inmutable del checkpoint y del tokenizador.
2. Dataset identificado por hash, procedencia y condiciones de uso.
3. Configuración de extracción: precisión, atención, tokens especiales y semillas.
4. Entorno de Python y dependencias fijadas, junto con CUDA, controlador y hardware cuando corresponda.
5. Resultados por token y por verso, matrices requeridas y control de cobertura por documento.
6. Código y configuración del análisis identificados separadamente de la extracción.
7. Anotaciones de evaluación versionadas y separadas de las entradas de extracción.
8. Tablas, figuras y cifras de la memoria generadas desde los mismos resultados.

Cada ejecución tendrá un identificador único y un manifiesto. Los resultados crudos serán inmutables; un análisis distinto producirá un nuevo conjunto de salidas. El manifiesto registrará hashes y condiciones reales de ejecución, incluidas marcas de tiempo operativas cuando sean necesarias para auditar los artefactos.

## Niveles de comprobación

- **Pruebas unitarias:** verifican propiedades y contratos del código.
- **Repetición del análisis:** produce las mismas tablas a partir de los mismos datos guardados.
- **Repetición de extracción:** compara métricas bajo condiciones registradas y tolerancias predefinidas.

La igualdad de tablas numéricas no implica identidad binaria de archivos comprimidos o contenedores con metadatos variables. Las tolerancias y el criterio de aceptación deberán fijarse antes de la repetición. No se afirmará reproducibilidad completa hasta terminar estas comprobaciones.
