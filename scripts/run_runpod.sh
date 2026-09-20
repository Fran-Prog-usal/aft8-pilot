#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
run_root="${1:-outputs/extraction}"
if [[ -e "$run_root" ]]; then
  echo "El destino ya existe: $run_root. Elige una carpeta nueva." >&2
  exit 1
fi
if [[ ! -f data/raw/AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx ]]; then
  echo "Falta el Excel autorizado en data/raw; no se distribuye mediante Git." >&2
  exit 1
fi
aft8-import-corpus data/raw/AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx data/processed/pilot30.jsonl
aft8-build-controls data/processed/pilot30.jsonl config/control_design.json data/processed/controls.jsonl
aft8-prepare-input data/processed/pilot30.jsonl data/processed/extraction_original.jsonl
aft8-prepare-input data/processed/controls.jsonl data/processed/extraction_controls.jsonl
aft8-extract data/processed/extraction_original.jsonl --cohort smoke --output "$run_root/results/smoke__mistral"
aft8-extract data/processed/extraction_original.jsonl --cohort original --smoke-manifest "$run_root/results/smoke__mistral/run_manifest.json" --output "$run_root/results/pilot30_aft8_v1__mistral"
aft8-extract data/processed/extraction_controls.jsonl --cohort control --smoke-manifest "$run_root/results/smoke__mistral/run_manifest.json" --output "$run_root/results/pilot30_aft8_D_v1__mistral"
tar -czf "$run_root/results-mistral.tar.gz" -C "$run_root" results
sha256sum "$run_root/results-mistral.tar.gz"
