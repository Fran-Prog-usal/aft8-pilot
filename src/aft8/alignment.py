"""Correspondencia entre offsets de caracteres y versos por máximo solapamiento."""
from __future__ import annotations

def verse_char_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = ([], 0)
    for line in text.split('\n'):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans

def _solape(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def map_tokens_to_verses(text: str, offsets: list[tuple[int, int]], tokens: list[str] | None=None, special_mask: list[bool] | None=None) -> dict[str, list]:
    spans = verse_char_spans(text)
    n = len(offsets)
    line_id: list[int] = []
    is_special: list[bool] = []
    is_newline: list[bool] = []
    ultimo_verso = 0
    for k, (s, e) in enumerate(offsets):
        especial = bool(special_mask[k]) if special_mask else e <= s
        if especial:
            line_id.append(0)
            is_special.append(True)
            is_newline.append(False)
            continue
        trozo = text[s:e]
        salto = '\n' in trozo and trozo.strip('\r\n \t') == ''
        solapes = [(_solape((s, e), sp), i) for i, sp in enumerate(spans, start=1)]
        mejor, idx = max(solapes, key=lambda x: (x[0], -x[1]))
        if salto or mejor == 0:
            v = ultimo_verso if ultimo_verso > 0 else 1
            line_id.append(v)
            is_special.append(False)
            is_newline.append(True)
        else:
            ultimo_verso = idx
            line_id.append(idx)
            is_special.append(False)
            is_newline.append('\n' in trozo)
    return {'line_id': line_id, 'is_special': is_special, 'is_newline': is_newline, 'char_start': [o[0] for o in offsets], 'char_end': [o[1] for o in offsets]}

def check_alignment(text: str, verses: list[str], al: dict) -> dict:
    n = len(verses)
    conteo = {i: 0 for i in range(1, n + 1)}
    for v, esp in zip(al['line_id'], al['is_special']):
        if not esp and v > 0:
            conteo[v] = conteo.get(v, 0) + 1
    vacios = [i for i, c in conteo.items() if c == 0]
    return {'tokens_por_verso': conteo, 'versos_vacios': vacios, 'n_especiales': sum(al['is_special']), 'n_saltos_de_linea': sum(al['is_newline']), 'n_versos_detectados': len([c for c in conteo.values() if c > 0]), 'ok': len(vacios) == 0 and len(conteo) == n}

def reconstruye(text: str, al: dict) -> bool:
    cubierto = bytearray(len(text))
    for s, e, esp in zip(al['char_start'], al['char_end'], al['is_special']):
        if not esp:
            for i in range(s, min(e, len(text))):
                cubierto[i] = 1
    faltan = [i for i, c in enumerate(cubierto) if not c and (not text[i].isspace())]
    return len(faltan) == 0
