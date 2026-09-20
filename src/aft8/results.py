"""Lectura de resultados de Mistral con verificación de contenido y sin pickle."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ARCHIVE_SHA256 = "1b8cdbe8ed7b83815fc61751ed6446266ffe3989e463604da76a3130e226087b"
REFERENCE_ARCHIVE_SHA256 = "2656d5c571da2a86102537ccf0a07301234367d1efb69134e9d668769c32418c"
RUNS = {
    "original": "results/pilot30_aft8_v1__mistral",
    "control": "results/pilot30_aft8_D_v1__mistral",
    "smoke": "results/smoke__mistral",
}


class ResultArchive:
    """Accede al paquete identificado por su hash sin extraer rutas al disco.

    Los miembros se guardan en un archivo temporal indexado para evitar
    descomprimir repetidamente el contenedor. No se evalúan los arrays object.
    """

    def __init__(self, path: Path, expected_sha256: str = ARCHIVE_SHA256):
        self._store = tempfile.TemporaryFile()
        self._index = {}
        self.inventory = []
        try:
            with Path(path).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != expected_sha256:
                    raise ValueError("El hash del paquete no coincide con el esperado.")
                source.seek(0)
                with tarfile.open(fileobj=source, mode="r|gz") as archive:
                    seen = set()
                    for member in archive:
                        name = member.name
                        parts = PurePosixPath(name)
                        if (
                            name in seen
                            or parts.is_absolute()
                            or ".." in parts.parts
                            or "\\" in name
                            or ":" in name
                            or not (member.isfile() or member.isdir())
                        ):
                            raise ValueError(f"Miembro de archivo no admitido: {name}")
                        seen.add(name)
                        if not member.isfile():
                            continue
                        offset = self._store.tell()
                        digest = hashlib.sha256()
                        length = 0
                        with archive.extractfile(member) as stream:
                            while block := stream.read(1024 * 1024):
                                self._store.write(block)
                                digest.update(block)
                                length += len(block)
                        if length != member.size:
                            raise ValueError(f"Miembro truncado: {name}")
                        self._index[name] = offset, length
                        self.inventory.append(
                            {"path": name, "bytes": length, "sha256": digest.hexdigest()}
                        )
            self.sha256 = actual
        except BaseException:
            self.close()
            raise

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self, run: str, relative_path: str) -> bytes:
        name = RUNS[run] + "/" + relative_path
        offset, length = self._index[name]
        self._store.seek(offset)
        return self._store.read(length)

    def table(self, run: str, level: str) -> pd.DataFrame:
        if level not in ("token", "line"):
            raise ValueError("El nivel debe ser token o line.")
        return pd.read_parquet(io.BytesIO(self.read(run, level + "_metrics.parquet")))

    def manifest(self, run: str) -> dict:
        return json.loads(self.read(run, "run_manifest.json"))

    def vectors(self, run: str, record_id: str) -> dict[str, np.ndarray]:
        with np.load(
            io.BytesIO(self.read(run, f"gxa_vectors/{record_id}.npz")), allow_pickle=False
        ) as data:
            # El texto de los tokens se obtiene de token_metrics.parquet.
            return {key: data[key] for key in data.files if key != "tokens"}
