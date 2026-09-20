# Verificación de la revisión técnica

La revisión conserva el cálculo científico y normaliza el formato, los imports y los identificadores internos. Las funciones públicas renombradas mantienen alias de compatibilidad. Las claves de los resultados originales no se traducen. Ruff comprueba estilo e imports con una configuración versionada.

## Comprobaciones realizadas

- Instalación nueva en Windows x64 y Python 3.11, sin heredar los paquetes del entorno anterior. Dependencias completas en `requirements/analysis-windows.lock.txt`; `pip check` sin conflictos.
- 43 pruebas superadas, incluida la extracción con Mistral diminuto en CPU y tres propiedades de los soportes GxA.
- Importación del Excel autorizado: 30 textos y 60 controles con los mismos hashes de corpus.
- Validación de agregaciones y offsets: cero incidencias en originales y controles.
- Reproducción del análisis y diagnósticos: 23 CSV idénticos en bytes a los resultados verificados que sustentan la memoria.
- Comparación del extractor anterior y revisado sobre un Mistral diminuto con los mismos pesos aleatorios: igualdad de los 34 arrays generados, incluidas las posiciones NaN.

La comprobación sintética no equivale a una nueva extracción Mistral-7B en GPU ni demuestra igualdad entre dispositivos. La ejecución A40 recibida mantiene sus manifiestos originales. La elección del soporte GxA y las anotaciones exploratorias continúan abiertas.

## Procedencia del código

`executed-source.zip`, adjunto a la release, conserva los 22 módulos Python cuyos hashes coincidían con el código ejecutado. Incluye `code_sha256.json` y se identifica por su propio hash en `config/release_assets.json`. El código revisado del repositorio tiene otros hashes: no debe compararse directamente con un manifiesto anterior como si fuese la misma versión.

El paquete `analysis-evidence.zip` conserva tablas, figuras y comprobaciones de la recepción original. La reproducción nueva se escribe en `outputs/reproduction/`; sus propios manifiestos identifican el código revisado y el entorno que realmente se ha utilizado.

## Lectura del contraste nominal P5/ΔP

En modo residual hay 4 aciertos de 6 textos: el contraste binomial con referencia uniforme de 0,2 da p bruto 0,01696 y p ajustado BH 0,13568 dentro de ocho contrastes familia/métrica. No alcanza el umbral de 0,05 después de ese ajuste. La corrección por multiplicidad tampoco demuestra que la referencia uniforme represente adecuadamente el diseño: por eso se conservan las comparaciones emparejadas y las permutaciones con sus supuestos explícitos.
