# Extracción en RunPod

El extractor implementa exclusivamente Mistral-7B-v0.3 base, con revisión de modelo y tokenizador `caa1feb0e54d415e2df31207e5f4e273e33509b1`. Usa teacher forcing sin generación ni chat template, un BOS explícito, atención eager, pesos bfloat16 y probabilidades float32. No trunca textos.

## Estado de validación

La integración completa por texto se ha probado con un Mistral diminuto de pesos aleatorios en CPU: forward, gradientes por posición, métricas, agregaciones y NPZ sin pickle. Esta prueba no valida resultados científicos ni compatibilidad GPU del entorno objetivo.

Se ha comprobado por separado el tokenizador auténtico de la revisión fijada: decode exacto e igualdad de IDs y offsets con los artefactos archivados en los 90 textos. No se han descargado aquí los pesos de 7B ni ejecutado una extracción CUDA.

El entorno objetivo reproduce las versiones principales registradas en la extracción de referencia: Python 3.12, PyTorch 2.8.0/CUDA 12.8, Transformers 5.17.0 y NumPy 2.1.2. Las versiones auxiliares están fijadas en `requirements/runpod.txt`. Su instalación conjunta en Linux y la extracción GPU siguen pendientes. No es todavía un lock transitivo validado; cada ejecución guardará `pip freeze`. Si la resolución de dependencias falla, conservar el error y revisar el entorno, sin cambiar versiones silenciosamente.

## Preparación y ejecución

Utilizar una GPU con soporte bfloat16. La referencia se obtuvo en una NVIDIA A40; para comparar entornos conviene utilizar el mismo modelo de GPU. Preparar Python 3.12 y espacio persistente para pesos, entorno y resultados. El script utiliza `/workspace/hf_cache` salvo que `HF_HOME` esté definido.

Descomprimir el paquete de entrega o situarse en la raíz de este proyecto. El libro autorizado debe estar en `data/raw/AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx`.

```bash
bash scripts/setup_runpod.sh
bash scripts/run_runpod.sh
```

El segundo script importa el Excel, reproduce controles y crea dos entradas sin hipótesis. Ejecuta primero A001/A013, luego los 30 originales y los 60 controles. Cada proceso carga el modelo por separado. La ejecución completa exige el manifiesto y los hashes de un smoke terminado con el mismo código y revisión. No se aceptan A001/A002 como sustituto del nuevo smoke.

El destino por defecto es `outputs/extraction`. No se sobrescriben ejecuciones. Si una ejecución falla, sus resultados parciales y un manifiesto `failed` se conservan; para otra tentativa elegir otra carpeta:

```bash
bash scripts/run_runpod.sh outputs/extraction_retry
```

Cada extracción guarda datos por token/verso, saliencia cruda y normalizada, ambas redistribuciones GxA, top-k, diagnósticos y estados ocultos. Una muestra determinista conserva atención completa. Los NPZ usan cadenas Unicode y no requieren pickle. Los manifiestos incluyen datos, código, versiones, GPU, tiempos e inventario de archivos. Los pesos y el corpus no se modifican durante el análisis.

## Repetición y recogida

Para comprobar una repetición independiente del smoke:

```bash
source .venv/bin/activate
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
aft8-extract data/processed/extraction_original.jsonl --cohort smoke --output outputs/smoke_repeat
aft8-compare-runs outputs/extraction/results/smoke__mistral outputs/smoke_repeat --output outputs/smoke_comparison.json
```

Se comparan arrays, formas, ausencias y cadenas; tolerancias iniciales rtol=1e-5 y atol=1e-7. Las diferencias se informan, no se ocultan aumentando tolerancias automáticamente. Esto no exige identidad de bytes de ZIP/NPZ ni acredita por sí solo repetibilidad entre hardware distinto.

El script produce `outputs/extraction/results-mistral.tar.gz` y muestra su SHA-256. Conservar también la repetición de smoke si se realiza. Devolver el paquete, el hash y cualquier error de instalación/ejecución. Para validar o analizar un paquete nuevo, pasar su hash explícito mediante `--archive-sha256`; el valor por defecto de los analizadores corresponde al paquete de referencia.

```bash
aft8-validate-results outputs/extraction/results-mistral.tar.gz data/processed/pilot30.jsonl --archive-sha256 HASH_DEL_PAQUETE --output outputs/new_validation.json
```

La elección de soporte GxA permanece abierta: se calculan ambas variantes. Una nueva ejecución correcta permite documentar el smoke solicitado; no borra la discrepancia del paquete archivado ni implica que se haya resuelto antes de realizarla.

## Clonar desde GitHub en RunPod

Repositorio privado: https://github.com/Fran-Prog-usal/aft8-pilot.
El Excel, los pesos y los resultados están excluidos de Git.

### Crear el token de lectura

En GitHub: Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token.

1. Nombre: `runpod-aft8-read`; caducidad: 30 días.
2. Resource owner: `Fran-Prog-usal`.
3. Repository access: **Only select repositories → aft8-pilot**.
4. Repository permissions: **Contents → Read-only**. Metadata queda en lectura. No añadir escritura ni administración.
5. Generar y guardar el token en un gestor de contraseñas. No pegarlo en mensajes, archivos, comandos ni URL.

El token permite clonar y descargar cambios, no publicar commits. La publicación desde el ordenador utiliza la autenticación local de Git. [Documentación oficial de tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

### Primera ejecución desde un Pod vacío

Elegir Python 3.12 y una GPU compatible con bfloat16; la referencia utilizó A40. No se han medido aún los requisitos máximos de disco/VRAM ni la duración de esta implementación. Mantener pesos, entorno y resultados en almacenamiento persistente. Un network volume sobrevive a la terminación del Pod; descargar también una copia de los resultados antes de terminarlo. [Almacenamiento RunPod](https://docs.runpod.io/pods/storage/types).

En la terminal del Pod:

```bash
python --version  # Debe ser 3.12.x
nvidia-smi
cd /workspace
git -c credential.helper= clone https://github.com/Fran-Prog-usal/aft8-pilot.git
cd aft8-pilot
git rev-parse HEAD
df -h /workspace
```

En Username introducir `Fran-Prog-usal`; en Password pegar el token (no se muestra al escribir), no la contraseña de GitHub. El comando desactiva el almacenamiento de credenciales para esta operación.

Subir el Excel autorizado mediante el explorador de archivos de Jupyter del Pod a `/workspace/aft8-pilot/data/raw/`, con el nombre exacto `AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx`. Clonar no descarga el Excel.

```bash
test -f data/raw/AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx && echo 'Excel disponible'
```

Si `tmux` está disponible, iniciar `tmux new -s aft8` para que cerrar la terminal no interrumpa el proceso. Después:

```bash
bash scripts/setup_runpod.sh
bash scripts/run_runpod.sh
```

Para separarse de tmux: Ctrl+B y después D. Para volver: `tmux attach -t aft8`.
La instalación necesita Python 3.12 con venv, PyPI, el índice CUDA de PyTorch y Hugging Face. Si falla, guardar el error y no cambiar versiones silenciosamente. El token GitHub no autentica en Hugging Face.

Descargar `outputs/extraction/results-mistral.tar.gz` desde Jupyter y conservar el SHA-256 mostrado en terminal. Descargar también cualquier smoke repetido. No terminar el Pod antes de comprobar la copia descargada. Si falla la extracción, conservar los parciales y reintentar con un destino nuevo como se indica arriba.
