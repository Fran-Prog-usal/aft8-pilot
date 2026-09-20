"""Comprobaciones independientes de alineación y agregaciones de resultados."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .results import ARCHIVE_SHA256, ResultArchive


def check_offsets(text: str, tokens: pd.DataFrame) -> dict:
    """Comprueba cobertura y asignación de offsets sin sustituir un decode.

    Los offsets pueden omitir espacios. Se comprueba por separado la cobertura
    de caracteres no blancos y la cobertura literal; ninguna equivale a volver
    a tokenizar y decodificar con el tokenizer del modelo.
    """
    starts = np.cumsum([0] + [len(line) + 1 for line in text.split("\n")[:-1]])
    spans = [(int(start), int(start) + len(line)) for start, line in zip(starts, text.split("\n"))]
    covered = np.zeros(len(text), dtype=bool)
    errors = []
    previous = 1
    for row in tokens.sort_values("token_idx").itertuples():
        start, end = int(row.char_start), int(row.char_end)
        if not 0 <= start <= end <= len(text):
            errors.append(f"offset fuera de rango: {row.token_idx}")
            continue
        if row.is_special:
            if row.line_id != 0:
                errors.append(f"verso de token especial: {row.token_idx}")
            continue
        covered[start:end] = True
        fragment = text[start:end]
        overlaps = [max(0, min(end, b) - max(start, a)) for a, b in spans]
        newline_only = "\n" in fragment and not fragment.strip("\r\n \t")
        if newline_only or max(overlaps) == 0:
            expected_line, expected_newline = previous, True
        else:
            expected_line = int(np.argmax(overlaps)) + 1
            previous = expected_line
            expected_newline = "\n" in fragment
        if row.line_id != expected_line or bool(row.is_newline) != expected_newline:
            errors.append(f"asignación de verso: {row.token_idx}")
    missing = [i for i, flag in enumerate(covered) if not flag]
    nonspace = [i for i in missing if not text[i].isspace()]
    if nonspace:
        errors.append("caracteres no blancos sin cobertura")
    return {
        "errors": errors,
        "uncovered_characters": len(missing),
        "uncovered_nonspace": len(nonspace),
        "literal_offset_coverage": not missing,
    }


def aggregate_lines(tokens: pd.DataFrame, salience: np.ndarray) -> pd.DataFrame:
    """Agrega tokens no especiales, incluidos saltos de línea, por verso.

    La masa recibida se divide por todas las consultas con saliencia definida.
    Las posiciones especiales no reciben masa en la salida por verso.
    """
    tokens = tokens.sort_values("token_idx")
    line_ids = tokens.line_id.to_numpy()
    if salience.shape != (len(tokens), len(tokens)):
        raise ValueError("Dimensiones de saliencia incompatibles con los tokens.")
    valid_queries = np.isfinite(salience).any(axis=1)
    rows = []
    for line, frame in tokens[~tokens.is_special].groupby("line_id", sort=True):
        record = {"line_id": int(line), "n_tokens": len(frame)}
        for column, source, operation in (
            ("deltaP_max", "delta_P", "max"),
            ("deltaP_mean", "delta_P", "mean"),
            ("deltaE_mean", "delta_E", "mean"),
            ("deltaE_sum", "delta_E", "sum"),
            ("deltaE_absmax", "delta_E", "absmax"),
            ("gxa_redistribution_max", "gxa_redistribution", "max"),
            ("gxa_redist_pad_max", "gxa_redist_pad", "max"),
            ("deltaP_top100_max", "deltaP_top100_T10", "max"),
        ):
            values = frame[source].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"Valor infinito en {source}.")
            values = values[np.isfinite(values)]
            record[column] = (
                (
                    float(np.max(np.abs(values)))
                    if operation == "absmax"
                    else float(getattr(np, operation)(values))
                )
                if len(values)
                else np.nan
            )
        selected = salience[np.ix_(valid_queries, line_ids == line)]
        record["gxa_mass_received"] = float(np.nansum(selected) / max(1, valid_queries.sum()))
        rows.append(record)
    return pd.DataFrame(rows).set_index("line_id")


def validate_run(archive: ResultArchive, run: str, corpus: dict[str, dict]) -> dict:
    tokens, lines = archive.table(run, "token"), archive.table(run, "line")
    manifest = archive.manifest(run)
    errors, details = [], []
    if len(tokens) != manifest["n_tokens"] or tokens.record_id.nunique() != manifest["n_texts"]:
        errors.append("Recuentos distintos del manifiesto")
    if manifest["model_config"]["hf_id"] != "mistralai/Mistral-7B-v0.3":
        errors.append("Modelo distinto de Mistral-7B-v0.3")
    if (
        tokens.duplicated(["record_id", "token_idx"]).any()
        or lines.duplicated(["record_id", "line_id"]).any()
    ):
        errors.append("Claves duplicadas")
    if set(tokens.record_id) != set(lines.record_id):
        errors.append("Identificadores distintos entre tablas")
    if run != "smoke" and set(tokens.record_id) != set(corpus):
        errors.append("Identificadores distintos del corpus")
    for identifier, frame in tokens.groupby("record_id", sort=True):
        if identifier not in corpus:
            errors.append(f"Texto ausente del corpus: {identifier}")
            continue
        frame = frame.sort_values("token_idx")
        if not np.array_equal(frame.token_idx.to_numpy(), np.arange(len(frame))):
            errors.append(identifier + ": índices de token no consecutivos")
            continue
        offset = check_offsets(corpus[identifier]["text_content"], frame)
        arrays = archive.vectors(run, identifier)
        calculated = aggregate_lines(frame, arrays["salience"])
        saved = lines[lines.record_id == identifier].set_index("line_id")
        differences = {}
        if set(calculated.index) != set(saved.index):
            errors.append(identifier + ": versos diferentes")
            continue
        for column in calculated.columns:
            left = calculated[column].to_numpy(dtype=float)
            right = saved.loc[calculated.index, column].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=True):
                differences[column] = float(np.nanmax(np.abs(left - right)))
        if offset["errors"] or differences:
            errors.append(identifier + ": discrepancias de offsets o agregaciones")
        details.append(
            {"record_id": identifier, "offsets": offset, "aggregation_differences": differences}
        )
    return {
        "run": run,
        "texts": len(details),
        "errors": errors,
        "records": details,
        "rtol": 1e-6,
        "atol": 1e-8,
        "tokenizer_decode_verified": False,
        "open_issues": ["GxA: alineación de soportes", "Smoke: A001/A002 frente a A001/A013"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--run", choices=("original", "control", "smoke"), default="original")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines()]
    corpus = {record["record_id"]: record for record in records}
    if len(records) != len(corpus):
        raise ValueError("Identificadores duplicados en el corpus.")
    with ResultArchive(args.archive, args.archive_sha256) as archive:
        report = validate_run(archive, args.run, corpus)
        report["archive_sha256"] = archive.sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{report['texts']} textos; {len(report['errors'])} incidencias.")
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
