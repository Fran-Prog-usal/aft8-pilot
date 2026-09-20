#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info[:2] == (3,12), "Se requiere Python 3.12"'
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/runpod.txt
python -m pip install -e '.[data,extraction]'
python -m pip check
python - <<'PY'
import torch, transformers, numpy
assert torch.__version__.split('+')[0] == '2.8.0'
assert torch.version.cuda == '12.8'
assert transformers.__version__ == '5.17.0'
assert numpy.__version__ == '2.1.2'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('GPU:', torch.cuda.get_device_name())
print('Entorno preparado; la validación científica requiere ejecutar el smoke.')
PY
