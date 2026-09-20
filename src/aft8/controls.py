"""Reproduce los controles a partir de las permutaciones fijadas del experimento."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def build_controls(corpus: list[dict], design: list[dict]) -> list[dict]:
    base = {record["record_id"]: record for record in corpus}
    if len(base) != len(corpus):
        raise ValueError("Identificadores originales duplicados.")
    result, seen = [], set()
    for item in design:
        source = base[item["source_record_id"]]
        identifier = item["record_id"]
        if identifier in seen:
            raise ValueError("Identificador de control duplicado.")
        seen.add(identifier)
        verses = source["text_content"].split("\n")
        permutation = item["permutation"]
        if sorted(permutation) != list(range(1, len(verses) + 1)):
            raise ValueError(f"Permutación inválida: {identifier}")
        variant = item["variant"]
        if identifier != source["record_id"] + variant or variant not in ("D1", "D2"):
            raise ValueError(f"Identificador o variante incompatible: {identifier}")
        original_target = re.fullmatch(r"v\.(\d+)", source["expected_transition_span"])
        if original_target is None:
            raise ValueError("Objetivo original no reconocido.")
        original_target = int(original_target[1])
        moved = sum(old != new for new, old in enumerate(permutation, start=1))
        target = item["expected_verse"]
        if variant == "D1":
            if moved != 2 or target == original_target or not isinstance(target, int) or not 2 <= target <= 6 or permutation[target - 1] != original_target:
                raise ValueError(f"Transposición u objetivo inválido: {identifier}")
        elif moved != len(verses) or target is not None:
            raise ValueError(f"Desarreglo u objetivo inválido: {identifier}")
        text = "\n".join(verses[index - 1] for index in permutation)
        result.append({"record_id": identifier, "source_record_id": source["record_id"],
                       "variant": variant, "design_family": source["design_family"],
                       "target_dimensions": source["target_dimensions"],
                       "permutation": permutation, "expected_verse": target,
                       "original_expected_verse": original_target,
                       "text_content": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    expected = {identifier + variant for identifier in base for variant in ("D1", "D2")}
    if seen != expected:
        raise ValueError("Faltan controles D1 o D2 para algún original.")
    counts = Counter(record["expected_verse"] for record in result if record["variant"] == "D1")
    if len(base) % 5 or counts != {line: len(base) // 5 for line in range(2, 7)}:
        raise ValueError("La distribución de objetivos D1 no está equilibrada.")
    return sorted(result, key=lambda record: record["record_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("design", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    corpus = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines()]
    design = json.loads(args.design.read_text(encoding="utf-8"))
    records = build_controls(corpus, design["records"])
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"{len(records)} controles; SHA-256: {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
