"""Saliencia sobre spans y transferencias exploratorias entre palabras."""

from __future__ import annotations

import re
import unicodedata

import numpy as np


def matching_spans(text, phrase):
    """Encuentra frases completas, sin confundir un objetivo con una subcadena.

    Se conservan acentos y offsets del texto original. La búsqueda no distingue
    mayúsculas. Si hay varias apariciones, el análisis utiliza la primera y
    registra su número para que esa convención sea revisable.
    """
    if not phrase:
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.IGNORECASE
        )
    ]


def tokens_in_span(tokens, span):
    if span is None:
        return []
    start, end = span
    selected = tokens[(tokens.char_start < end) & (tokens.char_end > start) & ~tokens.is_special]
    return sorted(selected.token_idx.astype(int).tolist())


def span_mass(matrix, query, indices):
    """Masa causal; cero si el span aún no existe y NaN si la consulta no está definida."""
    if query < 0 or query >= len(matrix) or not np.isfinite(matrix[query]).any():
        return float("nan")
    present = [index for index in indices if index <= query]
    if not present:
        return 0.0
    values = matrix[query, present]
    return float(values.sum()) if np.isfinite(values).all() else float("nan")


def normalized_word(word):
    return "".join(
        char
        for char in unicodedata.normalize("NFD", word.lower())
        if unicodedata.category(char) != "Mn"
    )


def content_words(text, tokens, stopwords):
    output = []
    for match in re.finditer(r"\w+", text, flags=re.UNICODE):
        word = normalized_word(match.group())
        if word.isalpha() and len(word) >= 3 and word not in stopwords:
            indices = tokens_in_span(tokens, match.span())
            if indices:
                output.append((match.group(), indices))
    return output


def evaluate_span(text, tokens, matrix, annotation, stopwords):
    """Devuelve medidas y trayectoria sin convertir umbrales descriptivos en validación.

    Las ventanas reproducen la convención exploratoria existente: antes desde
    que se completa el origen hasta que se completa el destino; después desde
    el destino completo hasta la última consulta definida. No se afirma que
    esa ventana coincida con la transición hipotetizada por verso.
    """
    target_matches = matching_spans(text, annotation.get("target"))
    source_matches = matching_spans(text, annotation.get("source"))
    target = tokens_in_span(tokens, target_matches[0] if target_matches else None)
    source = tokens_in_span(tokens, source_matches[0] if source_matches else None)
    row = {
        "source": annotation.get("source"),
        "target": annotation.get("target"),
        "target_occurrences": len(target_matches),
        "source_occurrences": len(source_matches),
        "target_char_start": target_matches[0][0] if target_matches else None,
        "target_char_end": target_matches[0][1] if target_matches else None,
        "source_char_start": source_matches[0][0] if source_matches else None,
        "source_char_end": source_matches[0][1] if source_matches else None,
        "target_token_indices": ",".join(map(str, target)),
        "source_token_indices": ",".join(map(str, source)),
        "status": "no_target",
    }
    if not target:
        row["status"] = "target_not_found" if annotation.get("target") else "no_target"
        return row, []
    queries = np.flatnonzero(np.isfinite(matrix).any(axis=1)).tolist()
    visible = [query for query in queries if query >= max(target)]
    if not visible:
        row["status"] = "no_query_after_target"
        return row, []
    row["target_line"] = int(tokens.set_index("token_idx").loc[max(target), "line_id"])
    row["target_first_complete_query"] = max(target)
    row["last_query"] = visible[-1]
    row["target_mass_mean_after_complete"] = float(
        np.mean([span_mass(matrix, query, target) for query in visible])
    )
    rest_masses = []
    for query in visible:
        rest = [
            int(item.token_idx)
            for item in tokens.itertuples()
            if not item.is_special and item.token_idx <= query and item.token_idx not in target
        ]
        rest_masses.append(span_mass(matrix, query, rest))
    row["rest_mass_mean_same_queries"] = float(np.mean(rest_masses))
    row["target_minus_rest_mass"] = (
        row["target_mass_mean_after_complete"] - row["rest_mass_mean_same_queries"]
    )
    trajectory = [
        {
            "query_token_idx": query,
            "query_line": int(tokens.set_index("token_idx").loc[query, "line_id"]),
            "target_mass": span_mass(matrix, query, target),
            "source_mass": span_mass(matrix, query, source) if source else np.nan,
        }
        for query in queries
    ]
    row["status"] = "focus_calculated"
    if not annotation.get("source"):
        return row, trajectory
    if not source:
        row["status"] = "source_not_found"
        return row, trajectory
    before = [query for query in queries if max(source) <= query < max(target)]
    if not before:
        row["status"] = "no_comparable_window"
        return row, trajectory
    mean_before = np.mean([span_mass(matrix, query, source) for query in before])
    mean_after = np.mean([span_mass(matrix, query, source) for query in visible])
    words = content_words(text, tokens, stopwords)
    prior = [
        (word, indices)
        for word, indices in words
        if max(indices) < max(target) and not set(indices) & set(source)
    ]
    changes = []
    for _, indices in prior:
        # Una palabra de control puede no existir en todas las consultas de la
        # ventana; se promedia solo donde ya tiene posiciones causales visibles.
        a = [span_mass(matrix, query, indices) for query in before if min(indices) <= query]
        b = [span_mass(matrix, query, indices) for query in visible if min(indices) <= query]
        if a and b:
            changes.append(float(np.mean(b) - np.mean(a)))
    last = visible[-1]
    source_last, target_last = span_mass(matrix, last, source), span_mass(matrix, last, target)
    quota = target_last / (source_last + target_last) if source_last + target_last > 0 else np.nan
    line_tokens = set(tokens.loc[tokens.line_id == row["target_line"], "token_idx"].astype(int))
    controls = [
        (word, indices)
        for word, indices in words
        if set(indices) <= line_tokens and not set(indices) & set(target) and min(indices) <= last
    ]
    quotas = [
        span_mass(matrix, last, indices) / (source_last + span_mass(matrix, last, indices))
        for _, indices in controls
        if source_last + span_mass(matrix, last, indices) > 0
    ]
    row.update(
        {
            "status": "transfer_calculated" if quotas else "transfer_no_target_control",
            "window_before_start": before[0],
            "window_before_end": before[-1],
            "window_after_start": visible[0],
            "window_after_end": last,
            "source_mass_before": float(mean_before),
            "source_mass_after": float(mean_after),
            "source_mass_change": float(mean_after - mean_before),
            "source_decreases": bool(mean_after < mean_before),
            "source_control_median_change": float(np.median(changes)) if changes else np.nan,
            "n_source_controls": len(changes),
            "source_decreases_more_than_controls": bool(
                mean_after - mean_before < np.median(changes)
            )
            if changes
            else None,
            "target_quota_last_query": float(quota),
            "target_control_median_quota": float(np.median(quotas)) if quotas else np.nan,
            "n_target_controls": len(quotas),
            "target_control_words": ", ".join(word for word, _ in controls),
            "target_exceeds_controls": bool(quota > np.median(quotas)) if quotas else None,
        }
    )
    return row, trajectory
