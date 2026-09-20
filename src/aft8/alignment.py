"""Correspondencia entre offsets de caracteres y versos por máximo solapamiento."""

from __future__ import annotations


def verse_char_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = ([], 0)
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def map_tokens_to_verses(
    text: str,
    offsets: list[tuple[int, int]],
    tokens: list[str] | None = None,
    special_mask: list[bool] | None = None,
) -> dict[str, list]:
    """Asigna por máximo solapamiento; empates favorecen el verso anterior.

    Los saltos aislados pertenecen al último verso observado (o al primero).
    Se conservan las claves del formato de resultados; véase docs/metodo.md.
    """
    spans = verse_char_spans(text)
    line_id: list[int] = []
    is_special: list[bool] = []
    is_newline: list[bool] = []
    last_verse = 0
    for k, (s, e) in enumerate(offsets):
        special = bool(special_mask[k]) if special_mask else e <= s
        if special:
            line_id.append(0)
            is_special.append(True)
            is_newline.append(False)
            continue
        fragment = text[s:e]
        newline = "\n" in fragment and fragment.strip("\r\n \t") == ""
        overlaps = [(_overlap((s, e), sp), i) for i, sp in enumerate(spans, start=1)]
        best, idx = max(overlaps, key=lambda x: (x[0], -x[1]))
        if newline or best == 0:
            v = last_verse if last_verse > 0 else 1
            line_id.append(v)
            is_special.append(False)
            is_newline.append(True)
        else:
            last_verse = idx
            line_id.append(idx)
            is_special.append(False)
            is_newline.append("\n" in fragment)
    return {
        "line_id": line_id,
        "is_special": is_special,
        "is_newline": is_newline,
        "char_start": [o[0] for o in offsets],
        "char_end": [o[1] for o in offsets],
    }


def check_alignment(text: str, verses: list[str], al: dict) -> dict:
    n = len(verses)
    counts = {i: 0 for i in range(1, n + 1)}
    for v, esp in zip(al["line_id"], al["is_special"]):
        if not esp and v > 0:
            counts[v] = counts.get(v, 0) + 1
    empty = [i for i, c in counts.items() if c == 0]
    return {
        "tokens_por_verso": counts,
        "versos_vacios": empty,
        "n_especiales": sum(al["is_special"]),
        "n_saltos_de_linea": sum(al["is_newline"]),
        "n_versos_detectados": len([c for c in counts.values() if c > 0]),
        "ok": len(empty) == 0 and len(counts) == n,
    }


def covers_non_whitespace(text: str, al: dict) -> bool:
    covered = bytearray(len(text))
    for s, e, esp in zip(al["char_start"], al["char_end"], al["is_special"]):
        if not esp:
            for i in range(s, min(e, len(text))):
                covered[i] = 1
    missing = [i for i, c in enumerate(covered) if not c and (not text[i].isspace())]
    return len(missing) == 0


# Compatibilidad con clientes de la primera versión del extractor.
reconstruye = covers_non_whitespace
