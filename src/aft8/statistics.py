"""Contrastes descriptivos y exploratorios con unidades y familias explícitas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

FAMILY_TARGETS = {
    "P1": ("delta_p",), "P2": ("delta_e",), "P3": ("gxa",),
    "P4": ("delta_p", "delta_e"), "P5": ("delta_p", "delta_e", "gxa"),
}
METRICS = ("delta_p", "delta_e", "gxa")
VERSE_SET = (2, 3, 4, 5, 6)


def fdr_bh(values):
    """Benjamini–Hochberg para una familia explícita de valores p finitos."""
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Valores p inválidos.")
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    output = np.empty_like(p)
    output[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return output


def score_peaks(verses, column):
    """Máximos en v.2–v.6; desempate por primer verso y rango medio en empates."""
    rows = []
    for identifier, frame in verses.groupby("record_id", sort=True):
        frame = frame[frame.line_id.isin(VERSE_SET)].sort_values("line_id")
        if frame.line_id.tolist() != list(VERSE_SET):
            raise ValueError(f"Versos incompletos o duplicados: {identifier}")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Valores no finitos: {identifier}/{column}")
        expected = frame.expected_verse.unique()
        if len(expected) != 1 or expected[0] not in VERSE_SET:
            raise ValueError(f"Objetivo no válido: {identifier}")
        target = int(expected[0])
        winner = VERSE_SET[int(np.argmax(values))]
        rows.append({"record_id": identifier, "design_family": frame.design_family.iloc[0],
                     "expected_verse": target, "argmax_verse": winner,
                     "hit": int(winner == target), "max_ties": int(np.sum(values == values.max())),
                     "rank_of_expected": float(rankdata(-values, method="average")[VERSE_SET.index(target)])})
    return pd.DataFrame(rows)


def nominal_summary(table):
    """Ocho binomiales nominales por familia y métrica, sin tratar 48 filas como independientes."""
    rows = []
    for (family, metric), group in table.groupby(["design_family", "metric"], sort=True):
        n, hits = len(group), int(group.hit.sum())
        rows.append({"design_family": family, "metric": metric, "n_texts": n, "hits": hits,
                     "hit_rate": hits / n, "p0_uniform_reference": .2,
                     "p_raw": binomtest(hits, n, .2, alternative="greater").pvalue})
    result = pd.DataFrame(rows)
    result["p_bh"] = fdr_bh(result.p_raw)
    result["correction_family"] = "eight_family_metric_tests_within_mode"
    return result


def paired_comparison(metric_scores, baseline_scores):
    """McNemar exacto bilateral sobre textos emparejados; nunca sobre tasas globales distintas."""
    if metric_scores.record_id.duplicated().any() or baseline_scores.record_id.duplicated().any():
        raise ValueError("Identificadores duplicados en la comparación.")
    paired = metric_scores.merge(baseline_scores, on="record_id", suffixes=("_metric", "_baseline"),
                                 how="left", validate="one_to_one")
    if paired.hit_baseline.isna().any() or not (paired.expected_verse_metric == paired.expected_verse_baseline).all():
        raise ValueError("Faltan pares o los objetivos son diferentes.")
    a, b = paired.hit_metric.to_numpy(), paired.hit_baseline.to_numpy()
    wins, losses = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"n_texts": len(paired), "metric_hits": int(a.sum()), "baseline_hits": int(b.sum()),
            "metric_only": wins, "baseline_only": losses,
            "hit_rate_difference": float(np.mean(a - b)),
            "p_raw": binomtest(wins, wins + losses, .5).pvalue if wins + losses else 1.0}


def conditional_permutation(table, n_permutations=20000, seed=0, stratify_family=False):
    """Permuta objetivos por documento, compartiéndolos entre sus métricas.

    La validez inferencial requiere intercambiabilidad dentro de los bloques.
    La versión por familia es una sensibilidad exploratoria, no una selección
    del contraste con menor p. Se usa la corrección Monte Carlo (b+1)/(B+1).
    """
    if n_permutations < 1:
        raise ValueError("Se requiere al menos una permutación.")
    meta = table.groupby("record_id", sort=True).agg(
        expected_verse=("expected_verse", "first"), design_family=("design_family", "first"))
    for column in ("expected_verse", "design_family"):
        if (table.groupby("record_id")[column].nunique() != 1).any():
            raise ValueError("Metadatos inconsistentes dentro de un texto.")
    ids = {identifier: i for i, identifier in enumerate(meta.index)}
    positions = table.record_id.map(ids).to_numpy()
    observed_targets = meta.expected_verse.to_numpy()
    peaks = table.argmax_verse.to_numpy()
    masks = {"all": np.ones(len(table), dtype=bool)}
    masks.update({metric: table.metric.to_numpy() == metric for metric in METRICS})
    blocks = ([np.flatnonzero(meta.design_family.to_numpy() == family)
               for family in sorted(meta.design_family.unique())] if stratify_family
              else [np.arange(len(meta))])
    rng = np.random.default_rng(seed)
    samples = np.empty((n_permutations, len(masks)))
    for i in range(n_permutations):
        targets = observed_targets.copy()
        for block in blocks:
            targets[block] = rng.permutation(observed_targets[block])
        hits = targets[positions] == peaks
        for j, mask in enumerate(masks.values()):
            samples[i, j] = hits[mask].mean()
    rows = []
    for j, (scope, mask) in enumerate(masks.items()):
        observed = float(table.loc[mask, "hit"].mean())
        samples_column = samples[:, j]
        rows.append({"scope": scope, "n_score_rows": int(mask.sum()),
                     "n_texts": int(table.loc[mask, "record_id"].nunique()),
                     "hit_rate": observed, "null_mean": float(samples_column.mean()),
                     "null_p95": float(np.quantile(samples_column, .95)),
                     "p_raw": (1 + int((samples_column >= observed).sum())) / (n_permutations + 1),
                     "n_permutations": n_permutations, "seed": seed,
                     "scheme": "within_family" if stratify_family else "all_texts"})
    result = pd.DataFrame(rows)
    result["p_bh"] = fdr_bh(result.p_raw)
    result["correction_family"] = "four_scopes_within_mode_and_permutation_scheme"
    return result


def document_bootstrap(values, n_resamples=10000, seed=0):
    """Media e intervalo percentil del 95 % remuestreando documentos completos."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Se requieren al menos dos documentos finitos.")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"n_documents": len(x), "mean_document_weighted": float(x.mean()),
            "ci95_low": float(lower), "ci95_high": float(upper),
            "n_resamples": n_resamples, "seed": seed}
