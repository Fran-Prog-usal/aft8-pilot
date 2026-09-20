"""Importación verificable del corpus y sus hipótesis desde el libro de referencia."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

SOURCE_SHA256 = "7dc1af0a69226ee03d92d8eb7a930f53f7fcc67f6ec956ebea10e11e87f6816c"
FIELDS = (
    "record_id",
    "title",
    "design_family",
    "target_dimensions",
    "preregistered_hypothesis",
    "expected_transition_span",
    "expected_delta_p",
    "expected_delta_e",
    "expected_gxa",
    "text_content",
    "text_sha256",
)


def read_corpus(path: Path) -> list[dict]:
    """Verifica la fuente y conserva literalmente los treinta registros del piloto.

    Las notas del diccionario de campos al final de la hoja no son registros.
    No se normalizan espacios, Unicode ni saltos de línea del texto.
    """
    import openpyxl

    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("El SHA-256 del libro no coincide con la fuente autorizada.")
    expected = {f"A{i:03d}" for i in range(1, 31)}
    records = {}
    workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=False)
    try:
        rows = iter(workbook["Hypotheses"].values)
        headers = next(rows)
        if any(headers.count(field) != 1 for field in FIELDS):
            raise ValueError("La hoja Hypotheses no contiene el esquema esperado.")
        for row in rows:
            values = dict(zip(headers, row))
            identifier = values.get("record_id")
            if identifier not in expected:
                continue
            if identifier in records:
                raise ValueError(f"Registro duplicado: {identifier}")
            record = {field: values.get(field) for field in FIELDS}
            text = record["text_content"]
            if not isinstance(text, str) or len(text.splitlines()) != 6:
                raise ValueError(f"Texto inválido: {identifier}")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["text_sha256"]:
                raise ValueError(f"Hash textual incorrecto: {identifier}")
            records[identifier] = record
    finally:
        workbook.close()
    if set(records) != expected:
        raise ValueError(f"Faltan registros: {sorted(expected - set(records))}")
    return [records[identifier] for identifier in sorted(records)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = read_corpus(args.workbook)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload.encode("utf-8"))
    print(
        f"{len(records)} registros; SHA-256: {hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    )


def prepare_input():
    """Proyecta el corpus a una entrada de extracción sin hipótesis ni etiquetas."""
    parser = argparse.ArgumentParser(description=prepare_input.__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines()]
    payload = "".join(
        json.dumps(
            {key: record[key] for key in ("record_id", "text_content", "text_sha256")},
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload.encode("utf-8"))
    print(f"{len(records)} entradas sin hipótesis")


if __name__ == "__main__":
    main()
