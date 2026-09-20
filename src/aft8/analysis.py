"""Análisis reproducible del piloto de Mistral sobre artefactos verificados."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import importlib.metadata

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA, FastICA

from .results import ResultArchive, ARCHIVE_SHA256
from .statistics import (FAMILY_TARGETS, METRICS, VERSE_SET, conditional_permutation,
                         document_bootstrap, fdr_bh, nominal_summary, paired_comparison, score_peaks)

MAPPING = {"delta_p": "deltaP_max", "delta_e": "deltaE_sum",
           "gxa": "gxa_redistribution_max", "gxa_pad": "gxa_redist_pad_max"}


def read_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("Identificadores duplicados en el corpus.")
    return records


def verse_table(archive, run, corpus):
    lines, tokens = archive.table(run, "line"), archive.table(run, "token")
    meta = []
    for record in corpus:
        expected = record.get("expected_verse")
        if "expected_verse" not in record:
            match = re.fullmatch(r"v\.(\d+)", record.get("expected_transition_span", ""))
            if match is None:
                raise ValueError("Objetivo original no reconocido.")
            expected = int(match[1])
        meta.append({"record_id": record["record_id"], "design_family": record["design_family"],
                     "expected_verse": expected, "variant": record.get("variant", "original")})
    meta = pd.DataFrame(meta)
    if set(lines.record_id) != set(meta.record_id):
        raise ValueError("El corpus no corresponde a los resultados.")
    verses = lines.merge(meta, on="record_id", validate="many_to_one")
    for column, source in MAPPING.items():
        verses[column] = verses[source]
    entropy = tokens[~tokens.is_special].groupby(["record_id", "line_id"]).entropy_H.mean().rename("entropy")
    verses = verses.merge(entropy, on=["record_id", "line_id"], validate="one_to_one")
    return verses, tokens


def hidden_baselines(archive, verses):
    """Proyecciones ajustadas sobre este corpus, sin interpretación de generalización."""
    arrays = {identifier: archive.vectors("original", identifier) for identifier in sorted(verses.record_id.unique())}
    hidden = np.vstack([value["hidden_32"] for value in arrays.values()])
    projections = {
        "pca_hidden": PCA(n_components=1, random_state=0).fit_transform(hidden)[:, 0],
        "ica_hidden": FastICA(n_components=1, random_state=0, max_iter=1000).fit_transform(hidden)[:, 0],
        "latent_compressed": np.linalg.norm(PCA(n_components=8, random_state=0).fit_transform(hidden), axis=1),
    }
    rows, offset = [], 0
    for identifier, array in arrays.items():
        n = len(array["line_id"])
        differences = {name: np.r_[np.nan, np.abs(np.diff(value[offset:offset + n]))]
                       for name, value in projections.items()}
        for verse in range(1, 7):
            mask = (array["line_id"] == verse) & ~array["is_special"]
            row = {"record_id": identifier, "line_id": verse,
                   "raw_attention": float(np.sum(array["attn_cruda"][mask]))}
            row.update({name: float(np.nanmax(value[mask])) for name, value in differences.items()})
            rows.append(row)
        offset += n
    return verses.merge(pd.DataFrame(rows), on=["record_id", "line_id"], validate="one_to_one")


def hypothesis_scores(verses):
    tables = []
    for metric in METRICS:
        scores = score_peaks(verses, metric)
        scores = scores[scores.design_family.map(lambda family: metric in FAMILY_TARGETS[family])].copy()
        scores["metric"] = metric
        tables.append(scores)
    return pd.concat(tables, ignore_index=True)


def random_reference(table, n_resamples=20000, seed=0):
    """Predicciones uniformes por texto; conserva los pesos de las filas de evaluación."""
    ids = sorted(table.record_id.unique())
    positions = table.record_id.map({identifier: i for i, identifier in enumerate(ids)}).to_numpy()
    rng = np.random.default_rng(seed)
    predictions = rng.choice(VERSE_SET, size=(n_resamples, len(ids)))
    hits = predictions[:, positions] == table.expected_verse.to_numpy()
    masks = {"all": np.ones(len(table), dtype=bool)}
    masks.update({metric: table.metric.to_numpy() == metric for metric in METRICS})
    return pd.DataFrame([{"scope": scope, "n_score_rows": int(mask.sum()),
                          "n_texts": int(table.loc[mask, "record_id"].nunique()),
                          "null_mean": float(hits[:, mask].mean()),
                          "null_p95": float(np.quantile(hits[:, mask].mean(axis=1), .95)),
                          "seed": seed, "n_resamples": n_resamples}
                         for scope, mask in masks.items()])


def gxa_description(verses, tokens, corpus):
    explicit = {record["record_id"] for record in corpus if record["design_family"] in ("P3", "P5")}
    rows = []
    for identifier, frame in verses.groupby("record_id", sort=True):
        frame = frame.sort_values("line_id")
        expected = int(frame.expected_verse.iloc[0])
        values = frame.gxa.to_numpy()
        location = int(np.flatnonzero(frame.line_id.to_numpy() == expected)[0])
        scores = rankdata(values, method="average") / len(values)
        rows.append({"record_id": identifier, "explicit_gxa_hypothesis": identifier in explicit,
                     "expected_verse": expected, "peak_verse_all_six": int(frame.line_id.iloc[np.argmax(values)]),
                     "target_percentile_all_six": float(scores[location]),
                     "uniform_target_rank_reference": (len(values) + 1) / (2 * len(values)),
                     "target_minus_mean_other_redistribution": float(values[location] - np.delete(values, location).mean())})
    per_text = pd.DataFrame(rows)
    summaries = []
    for scope, ids in (("all_30", set(per_text.record_id)), ("explicit_12", explicit)):
        for support in ("gxa_redistribution", "gxa_redist_pad"):
            series = tokens.loc[tokens.record_id.isin(ids) & ~tokens.is_special, support].dropna()
            summaries.append({"scope": scope, "support": support, "n_texts": len(ids),
                              "n_tokens": len(series), "median": float(series.median()),
                              "p05": float(series.quantile(.05)), "p95": float(series.quantile(.95))})
    return per_text, pd.DataFrame(summaries)


def factorial_diagnostics(tokens):
    columns = ["deltaP_full_T10", "deltaP_top100_T10", "deltaP_full_T20", "deltaP_top100_T20"]
    frame = tokens[~tokens.is_special][["record_id"] + columns].dropna().copy()
    cells = pd.DataFrame([{"cell": column, "n_tokens": len(frame),
                           "mean_token_weighted": float(frame[column].mean()),
                           "median_token_weighted": float(frame[column].median())} for column in columns])
    frame["support_T1"] = frame[columns[0]] - frame[columns[1]]
    frame["support_T2"] = frame[columns[2]] - frame[columns[3]]
    frame["temperature_full"] = frame[columns[0]] - frame[columns[2]]
    frame["temperature_top100"] = frame[columns[1]] - frame[columns[3]]
    frame["interaction"] = frame.support_T1 - frame.support_T2
    effects = ["support_T1", "support_T2", "temperature_full", "temperature_top100", "interaction"]
    documents = frame.groupby("record_id")[effects].mean()
    summary = pd.DataFrame([{"effect": name, **document_bootstrap(documents[name].to_numpy()),
                             "mean_token_weighted": float(frame[name].mean())} for name in effects])
    return cells, documents.reset_index(), summary


def control_profiles(original, controls):
    rows = []
    for variant in ("D1", "D2"):
        selected = controls[controls.variant == variant]
        for first in (1, 2):
            for metric in METRICS:
                a = original[original.line_id >= first].groupby("line_id")[metric].mean()
                b = selected[selected.line_id >= first].groupby("line_id")[metric].mean().reindex(a.index)
                rows.append({"variant": variant, "metric": metric, "first_verse": first,
                             "n_profile_positions": len(a), "correlation_of_means": float(a.corr(b))})
    return pd.DataFrame(rows)


def run_analysis(archive_path, corpus_path, controls_path, output, expected_sha256=ARCHIVE_SHA256):
    output.mkdir(parents=True, exist_ok=True)
    corpus, controls = read_records(corpus_path), read_records(controls_path)
    with ResultArchive(archive_path, expected_sha256) as archive:
        verses, tokens = verse_table(archive, "original", corpus)
        control_verses, _ = verse_table(archive, "control", controls)
        verses = hidden_baselines(archive, verses)
        verses["gxa_uncorrected"] = verses.gxa
        columns = list(MAPPING) + ["entropy", "raw_attention", "pca_hidden", "ica_hidden", "latent_compressed"]
        residual = verses.copy()
        residual[columns] = verses[columns] - verses.groupby("line_id")[columns].transform("mean")
        verses.to_parquet(output / "verses_raw.parquet", index=False)
        residual.to_parquet(output / "verses_residual.parquet", index=False)
        summaries = []
        all_paired, all_permutation, global_rows = [], [], []
        for mode, data in (("raw", verses), ("residual", residual)):
            table = hypothesis_scores(data)
            table.to_csv(output / f"scores_{mode}.csv", index=False)
            summary = nominal_summary(table)
            summary["mode"] = mode
            summaries.append(summary)
            global_rows.append({"mode": mode, "n_texts": table.record_id.nunique(),
                                "n_score_rows": len(table), "hits": int(table.hit.sum()),
                                "hit_rate": float(table.hit.mean()), "inference": "descriptive_only"})
            comparisons = []
            for baseline in ("entropy", "raw_attention", "pca_hidden", "ica_hidden", "latent_compressed", "gxa_uncorrected"):
                scores = score_peaks(data, baseline)
                for metric in METRICS:
                    if baseline == "gxa_uncorrected" and metric != "gxa":
                        continue
                    comparisons.append({"mode": mode, "metric": metric, "baseline": baseline,
                                        **paired_comparison(table[table.metric == metric], scores)})
            paired = pd.DataFrame(comparisons)
            paired["p_bh"] = fdr_bh(paired.p_raw)
            paired["correction_family"] = "sixteen_paired_comparisons_within_mode"
            all_paired.append(paired)
            for stratify in (False, True):
                perm = conditional_permutation(table, stratify_family=stratify)
                perm["mode"] = mode
                all_permutation.append(perm)
            random_reference(table).to_csv(output / f"random_reference_{mode}.csv", index=False)
        pd.concat(summaries).to_csv(output / "nominal_binomial.csv", index=False)
        pd.DataFrame(global_rows).to_csv(output / "global_descriptive.csv", index=False)
        pd.concat(all_paired).to_csv(output / "paired_baselines.csv", index=False)
        pd.concat(all_permutation).to_csv(output / "conditional_permutations.csv", index=False)
        descriptions, distributions = gxa_description(verses, tokens, corpus)
        descriptions.to_csv(output / "gxa_per_text.csv", index=False)
        distributions.to_csv(output / "gxa_distributions.csv", index=False)
        support = score_peaks(verses, "gxa").merge(score_peaks(verses, "gxa_pad"), on="record_id", suffixes=("_common", "_padded"))
        support.to_csv(output / "gxa_support_sensitivity.csv", index=False)
        cells, documents, factorial = factorial_diagnostics(tokens)
        cells.to_csv(output / "factorial_cells.csv", index=False)
        documents.to_csv(output / "factorial_documents.csv", index=False)
        factorial.to_csv(output / "factorial_document_bootstrap.csv", index=False)
        control_profiles(verses, control_verses).to_csv(output / "control_profile_sensitivity.csv", index=False)
        hypothesis_scores(control_verses[control_verses.variant == "D1"]).to_csv(output / "control_D1_raw_scores.csv", index=False)
        provenance = {"analysis_version": "0.2.0", "status": "exploratory_reanalysis",
                      "archive_sha256": archive.sha256,
                      "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                      "controls_sha256": hashlib.sha256(controls_path.read_bytes()).hexdigest(),
                      "seed": 0, "permutations": 20000, "bootstrap_replicates": 10000,
                      "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "matplotlib")},
                      "code_sha256": {path.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(Path(__file__).parent.rglob("*.py"))},
                      "open_issues": ["GxA support selection", "smoke A001/A002 versus A001/A013", "span annotations A004/A016"],
                      "limitations": ["Conditional permutation exchangeability is not established by this analysis.",
                                      "Residuals and hidden projections are fitted on the evaluated corpus.",
                                      "Nominal binomials assume a uniform hit probability of 0.2.",
                                      "BH corrections apply within the stated families, not across the entire exploratory analysis.",
                                      "Span-level diagnostics are produced separately; candidate annotation approval remains pending."]}
        (output / "analysis_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from .analysis_report import write_report
    write_report(output)
    print(f"Análisis guardado en {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    args = parser.parse_args()
    run_analysis(args.archive, args.corpus, args.controls, args.output, args.archive_sha256)


if __name__ == "__main__":
    main()
