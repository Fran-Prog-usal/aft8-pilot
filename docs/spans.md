# Saliencia sobre spans

El análisis de spans es exploratorio y usa matrices completas de saliencia normalizada de Mistral. Sus resultados se mantienen separados de los contrastes de máximos por verso.

## Localización

Se buscan frases completas sin distinguir mayúsculas, conservando acentos y offsets del texto original. No se lematiza ni se cambia el corpus. Si una frase aparece varias veces se selecciona la primera y se registra el número de apariciones, los offsets y los tokens que solapan con ella. Esta convención debe tenerse en cuenta al interpretar palabras repetidas.

La búsqueda por subcadenas anterior localizaba «sí» dentro de «sin» en A011. Ahora se selecciona «sí» en el último verso. En A020, «devolver» no aparece como palabra completa: existe «devolverlo». Se conserva la anotación literal como no localizada y se calcula la forma completa como sensibilidad pendiente de confirmación. No se sustituye silenciosamente una anotación por otra.

## Masa y controles

La masa del span suma la saliencia de sus tokens causalmente visibles. Antes de que exista el destino, su masa es cero por construcción; la última consulta sin objetivo siguiente queda indefinida. Para comparar destino y resto se utilizan las mismas consultas, desde el destino completo hasta la última consulta definida; el resto excluye tokens especiales y el propio destino. Los conjuntos pueden tener tamaños distintos, por lo que esa diferencia de masas es descriptiva.

Las transferencias usan una ventana anterior desde que se completa el origen hasta justo antes de completarse el destino, y otra posterior desde el destino completo hasta la última consulta válida. El verso hipotetizado se conserva en la tabla: estas ventanas de aparición no equivalen automáticamente a la transición prevista por el diseño.

La caída del origen se compara con la mediana del cambio de palabras de contenido previas. Cada control se evalúa solo donde tiene tokens causalmente visibles, de modo que puede disponer de menos consultas que el origen. La cuota del destino en la última consulta se compara con otras palabras de contenido de su verso. La lista funcional fija se publica en `config/span_controls.json`; filtra palabras de menos de tres letras y no sustituye a un etiquetado lingüístico. Los controles no constituyen una prueba definitiva de transferencia causal.

## Versiones

- `candidate-1`: treinta anotaciones propuestas, siete transferencias calculables con control de destino. Dos destinos superan al control y seis orígenes caen más que sus controles; son recuentos descriptivos.
- `archived-pairs-current-controls`: siete pares archivados, cinco calculables, con los controles actuales por palabras. Dos destinos superan al control. No representa el antiguo resultado 5/5, que utilizaba otros controles.
- `source-sensitivity`: A004 con «pecho» y «mano», además del candidato «corazón». Ningún destino supera al control; la caída de origen supera al control solo con «corazón». La elección del origen sigue abierta.
- `surface-sensitivity`: A020 con «devolverlo», pendiente de confirmar frente al objetivo literal «devolver».

A016 conserva «monitor» como origen archivado no localizado y «pantalla» como candidato. Los resúmenes diferencian todos los textos y el subconjunto con hipótesis GxA explícita. Los mapas y las trayectorias están en `outputs/diagnostics/figures/`; las tablas incluyen los casos no calculables.

Los cálculos se completan sobre estas convenciones sin aprobar las anotaciones candidatas ni elegir una variante de soporte GxA. El soporte y el smoke test permanecen abiertos.
