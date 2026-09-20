"""Extracción Mistral con entradas sin hipótesis y salidas numéricas auditables."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .alignment import check_alignment, covers_non_whitespace, map_tokens_to_verses
from .metrics.delta_e import delta_e_series, entropy_norm, entropy_series
from .metrics.delta_p import delta_p_series
from .metrics.gxa import (
    mass_by_verse,
    raw_attention_received,
    redistribution_series,
    salience_matrix,
)
from .metrics.predictive_diagnostics import diagnostic_series
from .metrics.surprisal import surprisal_series
from .metrics.topk import topk_logits
from .model import MODEL_ID, REVISION, MistralExtractor
from .validation import aggregate_lines

SMOKE_IDS = ("A001", "A013")
INPUT_KEYS = {"record_id", "text_content", "text_sha256"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob("*.py"))}


def read_inputs(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not records or len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Corpus vacío o identificadores duplicados.")
    for record in records:
        if set(record) != INPUT_KEYS:
            raise ValueError(
                "La entrada debe contener solo identificador, texto y hash; no admite hipótesis."
            )
        if not record["record_id"].isalnum():
            raise ValueError("Identificador no permitido.")
        if len(record["text_content"].split("\n")) != 6:
            raise ValueError("Se requieren seis versos.")
        if (
            hashlib.sha256(record["text_content"].encode("utf-8")).hexdigest()
            != record["text_sha256"]
        ):
            raise ValueError("Hash textual incorrecto.")
    return records


def extract_record(adapter, record, run_id):
    """Calcula un texto sin consultar sus objetivos de evaluación."""
    text = record["text_content"]
    encoding, ids, logits, attentions, hidden = adapter.forward(text)
    alignment = map_tokens_to_verses(
        text, encoding["offsets"], encoding["tokens"], encoding["special_mask"]
    )
    report = check_alignment(text, text.split("\n"), alignment)
    report.update(
        decode_exact=encoding["decode_exact"],
        offset_coverage=covers_non_whitespace(text, alignment),
    )
    if not report["ok"] or not report["offset_coverage"] or not report["decode_exact"]:
        raise ValueError("Alineación o reconstrucción textual incorrecta.")
    cpu_logits, cpu_ids = logits.detach().cpu(), ids.detach().cpu()
    entropy = entropy_series(cpu_logits)
    delta_p, delta_e = delta_p_series(cpu_logits), delta_e_series(entropy)
    surprise = surprisal_series(cpu_logits, cpu_ids)
    diagnostic = diagnostic_series(cpu_logits)
    np.testing.assert_allclose(
        diagnostic["deltaP_full_T10"], delta_p, rtol=1e-6, atol=1e-8, equal_nan=True
    )
    salience, unnormalized = salience_matrix(attentions, None, ids[1:], logits, return_raw=True)
    common, padded = redistribution_series(salience)
    mass = mass_by_verse(salience, np.array(alignment["line_id"]), 6)
    n = len(cpu_ids)
    rows = pd.DataFrame(
        {
            "run_id": run_id,
            "model_key": "mistral",
            "record_id": record["record_id"],
            "token_idx": np.arange(n),
            "token_id": cpu_ids.numpy(),
            "token_text": encoding["tokens"],
            **alignment,
            "entropy_H": entropy,
            "entropy_H_norm": entropy_norm(entropy, logits.shape[-1]),
            "delta_E": delta_e,
            "delta_P": delta_p,
            "delta_P_norm": delta_p / np.log(2),
            "surprisal_observed": surprise,
            "gxa_redistribution": common,
            "gxa_redist_pad": padded,
            "status": "OK",
            "notes": "",
            **diagnostic,
        }
    )
    lines = aggregate_lines(rows, salience).reset_index()
    lines["run_id"], lines["model_key"], lines["record_id"] = run_id, "mistral", record["record_id"]
    lines["expected_flag"], lines["status"], lines["notes"] = None, "OK", ""
    arrays = {
        "tokens": np.asarray(encoding["tokens"], dtype=str),
        "line_id": np.asarray(alignment["line_id"]),
        "is_special": np.asarray(alignment["is_special"]),
        "is_newline": np.asarray(alignment["is_newline"]),
        "input_ids": cpu_ids.numpy(),
        "salience": salience.astype(np.float32),
        "salience_cruda": unnormalized.astype(np.float32),
        "gxa_redistribution": common,
        "gxa_redist_pad": padded,
        "gxa_mass_por_verso": mass,
        "attn_cruda": raw_attention_received(attentions),
        "delta_P": delta_p,
        "delta_P_norm": delta_p / np.log(2),
        "entropy_H": entropy,
        "delta_E": delta_e,
        "surprisal": surprise,
        **diagnostic,
        **topk_logits(cpu_logits, cpu_ids, k=50),
    }
    for index in (0, 8, 16, 24, len(hidden) - 1):
        if index < len(hidden):
            arrays[f"hidden_{index}"] = hidden[index][0].detach().float().cpu().numpy()
    audit = np.stack(
        [attention[0].detach().float().cpu().numpy() for attention in attentions]
    ).astype(np.float16)
    return rows, lines, arrays, report, audit


def assert_smoke(path, cohort, inputs_hash, code_hashes):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("cohort") != "smoke"
        or manifest.get("record_ids") != list(SMOKE_IDS)
        or manifest.get("model_revision") != REVISION
        or manifest.get("code_sha256") != code_hashes
    ):
        raise ValueError("El smoke no acredita esta versión del extractor y modelo.")
    if cohort == "original" and manifest.get("input_sha256") != inputs_hash:
        raise ValueError("El corpus original no coincide con el utilizado para smoke.")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        if digest(Path(path).parent / name) != expected:
            raise ValueError("Artefacto de smoke modificado.")
    if not manifest.get("artifact_sha256"):
        raise ValueError("El smoke no incluye inventario de artefactos.")


def run_extraction(inputs, output, cohort, smoke_manifest=None):
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("El entorno científico requiere Python 3.12.")
    for package, expected_version in {
        "torch": "2.8.0",
        "transformers": "5.17.0",
        "numpy": "2.1.2",
    }.items():
        if importlib.metadata.version(package).split("+")[0] != expected_version:
            raise ValueError(f"Se requiere {package}=={expected_version} para esta especificación.")
    if torch.version.cuda != "12.8":
        raise ValueError("Se requiere la compilación CUDA 12.8 de PyTorch.")
    records = read_inputs(inputs)
    expected = (
        {f"A{i:03d}" for i in range(1, 31)}
        if cohort != "control"
        else {f"A{i:03d}D{j}" for i in range(1, 31) for j in (1, 2)}
    )
    if {record["record_id"] for record in records} != expected:
        raise ValueError("Los identificadores no corresponden al conjunto solicitado.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("El destino debe estar vacío; no se sobrescriben ejecuciones.")
    hashes = source_hashes()
    input_hash = digest(inputs)
    if cohort != "smoke":
        if smoke_manifest is None:
            raise ValueError("La ejecución completa requiere un manifiesto de smoke válido.")
        assert_smoke(smoke_manifest, cohort, input_hash, hashes)
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("Iniciar Python con PYTHONHASHSEED=42.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if cohort == "smoke":
        records = sorted(
            [record for record in records if record["record_id"] in SMOKE_IDS],
            key=lambda r: r["record_id"],
        )
    else:
        random.Random(20260902).shuffle(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gxa_vectors").mkdir()
    (output / "attention_audit").mkdir()
    manifest = {
        "status": "running",
        "cohort": cohort,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "dtype": "bfloat16",
        "attention_backend": "eager",
        "input_sha256": input_hash,
        "code_sha256": hashes,
        "seed": 42,
        "order_seed": 20260902,
        "python": platform.python_version(),
        "open_issues": [
            "GxA support alignment remains open",
            "Historical smoke mismatch remains documented",
        ],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "numpy", "pandas", "pyarrow")
        },
    }
    manifest_path = output / "run_manifest.json"

    def save():
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    save()
    start = time.perf_counter()
    try:
        adapter = MistralExtractor.load()
        manifest.update(
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            model_config={"hf_id": MODEL_ID, "revision": REVISION},
        )
        (output / "environment.txt").write_text(
            subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True),
            encoding="utf-8",
        )
        audit_ids = set(
            sorted(
                [r["record_id"] for r in records],
                key=lambda rid: hashlib.sha256(f"42:{rid}".encode()).hexdigest(),
            )[: int(np.ceil(len(records) * 0.05))]
        )
        token_frames, line_frames, alignment = [], [], {}
        for record in records:
            tokens, lines, arrays, check, attention = extract_record(adapter, record, output.name)
            identifier = record["record_id"]
            np.savez_compressed(output / f"gxa_vectors/{identifier}.npz", **arrays)
            if identifier in audit_ids:
                np.savez_compressed(
                    output / f"attention_audit/{identifier}.npz",
                    attention=attention,
                    tokens=arrays["tokens"],
                    line_id=arrays["line_id"],
                )
            token_frames.append(tokens)
            line_frames.append(lines)
            alignment[identifier] = check
            pd.concat(token_frames).to_parquet(output / "token_metrics.parquet", index=False)
            pd.concat(line_frames).to_parquet(output / "line_metrics.parquet", index=False)
            (output / "alignment_report.json").write_text(
                json.dumps(alignment, indent=2), encoding="utf-8"
            )
            manifest["completed_record_ids"] = list(alignment)
            save()
            print(f"{len(alignment)}/{len(records)}: {identifier}", flush=True)
        manifest.update(
            status="complete",
            record_ids=sorted(alignment),
            n_texts=len(records),
            n_tokens=sum(len(frame) for frame in token_frames),
            elapsed_seconds=time.perf_counter() - start,
            artifact_sha256={
                p.relative_to(output).as_posix(): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p != manifest_path
            },
        )
        save()
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("smoke", "original", "control"), required=True)
    parser.add_argument("--smoke-manifest", type=Path)
    args = parser.parse_args()
    run_extraction(args.inputs, args.output, args.cohort, args.smoke_manifest)


if __name__ == "__main__":
    main()
