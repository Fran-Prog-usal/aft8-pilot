"""Diagnósticos instrumentales y análisis exploratorio de saliencia por spans."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .analysis import read_records, verse_table
from .results import ResultArchive, ARCHIVE_SHA256
from .spans import evaluate_span, normalized_word, matching_spans
from .statistics import METRICS, score_peaks


def instrument_tables(verses, tokens):
    """Distingue cocientes empíricos de rangos y fracciones del techo teórico."""
    valid = tokens[~tokens.is_special]
    distribution = []
    for support in ("full", "top100"):
        for temperature in ("T05", "T10", "T20"):
            column = f"deltaP_{support}_{temperature}"
            values = valid[column].dropna().to_numpy()
            p05, p95 = np.quantile(values, [.05, .95])
            distribution.append({"support": support, "temperature": temperature,
                                 "n_tokens": len(values), "median_nats": float(np.median(values)),
                                 "fraction_within_one_percent_of_ceiling": float(np.mean(values >= .99 * np.log(2))),
                                 "fraction_within_one_percent_of_floor": float(np.mean(values <= .01 * np.log(2))),
                                 "p95_minus_p05_nats": float(p95 - p05),
                                 "p95_minus_p05_fraction_ln2": float((p95 - p05) / np.log(2))})
    residual = verses.copy()
    residual[list(METRICS)] = verses[list(METRICS)] - verses.groupby("line_id")[list(METRICS)].transform("mean")
    resolution, normalization = [], []
    for mode, frame in (("raw", verses), ("residual", residual)):
        for metric, column in (("delta_p", "delta_P"), ("delta_e", "delta_E"), ("gxa", "gxa_redistribution")):
            lo, hi = np.quantile(valid[column].dropna(), [.01, .99])
            values = frame[metric].to_numpy()
            width = np.ptp(values)
            resolution.append({"mode": mode, "metric": metric,
                               "verse_statistic": {"delta_p": "max", "delta_e": "sum_signed", "gxa": "max"}[metric],
                               "verse_range": float(width), "token_p99_minus_p01": float(hi - lo),
                               "verse_range_over_token_percentile_range": float(width / (hi - lo)) if hi > lo else np.nan,
                               "reference": "empirical_range_ratio_not_discrimination_probability"})
            reference = score_peaks(frame, metric).set_index("record_id").argmax_verse
            token_values = valid[column].dropna().to_numpy()
            for name, center, scale in (("zscore", np.mean(token_values), np.std(token_values, ddof=1)),
                                         ("median_mad", np.median(token_values), 1.4826 * np.median(np.abs(token_values - np.median(token_values))))):
                if scale <= 0:
                    normalization.append({"mode": mode, "metric": metric, "transform": name, "status": "zero_scale"})
                    continue
                transformed = frame.copy()
                transformed[metric] = (values - center) / scale
                peaks = score_peaks(transformed, metric).set_index("record_id").argmax_verse
                normalization.append({"mode": mode, "metric": metric, "transform": name,
                                      "status": "calculated", "scale": float(scale),
                                      "center": float(center), "n_fit_tokens": len(token_values),
                                      "application": "affine_transform_after_verse_aggregation",
                                      "changed_argmax": int((peaks != reference).sum()),
                                      "n_texts": len(peaks)})
    return pd.DataFrame(distribution), pd.DataFrame(resolution), pd.DataFrame(normalization)


def state_description(tokens, output, archive):
    """HDBSCAN explícito de scikit-learn; PCA solo para visualizar, sin prueba de IAmotions."""
    frame = tokens[~tokens.is_special].dropna(subset=["delta_P", "delta_E", "gxa_redistribution"]).copy()
    values = frame[["delta_P", "delta_E", "gxa_redistribution"]].to_numpy()
    scaled = StandardScaler().fit_transform(values)
    labels = HDBSCAN(min_cluster_size=10, min_samples=10, copy=True).fit_predict(scaled)
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(scaled)
    frame["cluster"] = labels
    frame["pca_1"], frame["pca_2"] = coordinates[:, 0], coordinates[:, 1]
    frame[["record_id", "token_idx", "line_id", "cluster", "pca_1", "pca_2"]].to_parquet(output / "states.parquet", index=False)
    selected = labels >= 0
    n_clusters = len(set(labels[selected]))
    silhouette = (float(silhouette_score(scaled[selected], labels[selected]))
                  if 1 < n_clusters < selected.sum() else None)
    summary = {"n_states": len(frame), "n_clusters": n_clusters,
               "noise_fraction": float((~selected).mean()), "silhouette_excluding_noise": silhouette,
               "implementation": "sklearn.cluster.HDBSCAN", "min_cluster_size": 10, "min_samples": 10,
               "projection": "PCA_full_SVD", "inferential_p": None,
               "interpretation": "Descriptive token geometry; no emotion labels or proof of semantic clusters."}
    try:
        previous = pd.read_parquet(io.BytesIO(archive.read("original", "clusters.parquet")))
    except KeyError:
        summary["archived_reference_available"] = False
        for key in ("adjusted_rand_against_archived", "archived_noise_fraction", "archived_silhouette_on_current_standardization", "archived_labels_reproduced"):
            summary[key] = None
    else:
        summary["archived_reference_available"] = True
        pairs = frame[["record_id", "token_idx", "cluster"]].merge(
            previous[["record_id", "token_idx", "cluster"]], on=["record_id", "token_idx"],
            suffixes=("_current", "_archived"), validate="one_to_one")
        if len(pairs) != len(frame):
            raise ValueError("Los estados archivados no corresponden al conjunto actual.")
        summary["adjusted_rand_against_archived"] = float(adjusted_rand_score(pairs.cluster_current, pairs.cluster_archived))
        old_labels = pairs.cluster_archived.to_numpy()
        old_mask = old_labels >= 0
        summary["archived_noise_fraction"] = float((~old_mask).mean())
        old_clusters = len(set(old_labels[old_mask]))
        summary["archived_silhouette_on_current_standardization"] = float(silhouette_score(scaled[old_mask], old_labels[old_mask])) if 1 < old_clusters < old_mask.sum() else None
        summary["archived_labels_reproduced"] = bool(summary["adjusted_rand_against_archived"] == 1.0)
        pairs.to_parquet(output / "cluster_label_comparison.parquet", index=False)
    (output / "clustering_descriptive.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.scatter(coordinates[~selected, 0], coordinates[~selected, 1], s=5, color="lightgray", label="Ruido")
    ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=7, c=labels[selected], cmap="tab10")
    ax.set(xlabel="Componente principal 1", ylabel="Componente principal 2", title="Estados de tokens: proyección PCA descriptiva")
    ax.legend(frameon=False)
    fig.savefig(output / "figures/states_pca.png", dpi=160)
    plt.close(fig)
    return summary


def plot_maps(identifier, tokens, matrix, destination):
    """Muestra la matriz completa por token, con límites de verso y escala común."""
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    shown = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="viridis", origin="upper", interpolation="nearest")
    centers, labels = [], []
    for verse, frame in tokens[~tokens.is_special].groupby("line_id", sort=True):
        start, end = int(frame.token_idx.min()), int(frame.token_idx.max())
        centers.append((start + end) / 2)
        labels.append(f"v.{verse}")
        ax.axvline(start - .5, color="white", linewidth=.35)
        ax.axhline(start - .5, color="white", linewidth=.35)
    ax.set_xticks(centers, labels)
    ax.set_yticks(centers, labels)
    ax.set(xlabel="Token de contexto que recibe saliencia", ylabel="Token de consulta", title=f"{identifier} · saliencia normalizada")
    fig.colorbar(shown, ax=ax, label="Masa por token")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def run_diagnostics(archive_path, corpus_path, config_dir, output, expected_sha256=ARCHIVE_SHA256):
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures/maps").mkdir(parents=True, exist_ok=True)
    corpus = read_records(corpus_path)
    candidate = json.loads((config_dir / "gxa_annotations_candidate.json").read_text(encoding="utf-8"))
    archived = json.loads((config_dir / "gxa_transfers_archived.json").read_text(encoding="utf-8"))
    controls = json.loads((config_dir / "span_controls.json").read_text(encoding="utf-8"))
    stopwords = {normalized_word(word) for word in controls["stopwords"]}
    annotations = [{**record, "annotation_version": candidate["version"]} for record in candidate["records"]]
    annotations += [{**record, "annotation_version": "archived-pairs-current-controls", "review_status": "archived_pair"} for record in archived["records"]]
    annotations += [{"record_id": "A004", "source": source, "target": "pared", "annotation_version": "source-sensitivity",
                     "review_status": "pending"} for source in ("pecho", "mano")]
    annotations += [{"record_id": "A020", "source": None, "target": "devolverlo",
                     "annotation_version": "surface-sensitivity", "review_status": "pending"}]
    rows, trajectories, matching_audit = [], [], []
    with ResultArchive(archive_path, expected_sha256) as archive:
        verses, tokens = verse_table(archive, "original", corpus)
        distribution, resolution, normalization = instrument_tables(verses, tokens)
        distribution.to_csv(output / "predictive_distribution.csv", index=False)
        resolution.to_csv(output / "resolution_ratios.csv", index=False)
        normalization.to_csv(output / "normalization_sensitivity.csv", index=False)
        clustering = state_description(tokens, output, archive)
        for record in corpus:
            identifier, text = record["record_id"], record["text_content"]
            frame = tokens[tokens.record_id == identifier].sort_values("token_idx")
            matrix = archive.vectors("original", identifier)["salience"]
            plot_maps(identifier, frame, matrix, output / f"figures/maps/{identifier}.png")
            for annotation in [item for item in annotations if item["record_id"] == identifier]:
                result, series = evaluate_span(text, frame, matrix, annotation, stopwords)
                metadata = {"record_id": identifier, "annotation_version": annotation["annotation_version"],
                            "review_status": annotation.get("review_status"),
                            "explicit_gxa_hypothesis": record["design_family"] in ("P3", "P5"),
                            "hypothesized_verse": int(record["expected_transition_span"].split(".")[1])}
                rows.append({**metadata, **result})
                trajectories.extend({**metadata, "source": annotation.get("source"), "target": annotation.get("target"), **item} for item in series)
                if annotation.get("target"):
                    legacy_index = normalized_word(text).find(normalized_word(annotation["target"]))
                    matches = matching_spans(text, annotation["target"])
                    matching_audit.append({**metadata, "target": annotation["target"], "legacy_substring_start": legacy_index,
                                           "whole_phrase_start": matches[0][0] if matches else None,
                                           "same_start": bool(matches and matches[0][0] == legacy_index)})
        archive_hash = archive.sha256
    spans = pd.DataFrame(rows)
    spans.to_csv(output / "span_results.csv", index=False)
    pd.DataFrame(trajectories).to_csv(output / "span_trajectories.csv", index=False)
    curves = pd.DataFrame(trajectories)
    for identifier, data in curves[(curves.annotation_version == candidate["version"]) & curves.source.notna()].groupby("record_id"):
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot(data.query_token_idx, data.source_mass, label=f"Origen: {data.source.iloc[0]}", color="#286682")
        ax.plot(data.query_token_idx, data.target_mass, label=f"Destino: {data.target.iloc[0]}", color="#cf8c39")
        boundary = spans[(spans.record_id == identifier) & (spans.annotation_version == candidate["version"])].target_first_complete_query.iloc[0]
        ax.axvline(boundary, color="gray", linestyle="--", label="Destino completo")
        ax.set(xlabel="Índice del token de consulta", ylabel="Masa de saliencia normalizada", title=f"{identifier} · trayectoria exploratoria origen–destino")
        ax.legend(frameon=False)
        fig.savefig(output / f"figures/transfer_{identifier}.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(matching_audit).to_csv(output / "span_matching_audit.csv", index=False)
    summaries = []
    for version, data in spans.groupby("annotation_version", sort=True):
        for scope, subset in (("all", data), ("explicit_gxa", data[data.explicit_gxa_hypothesis])):
            transfers = subset[subset.status == "transfer_calculated"]
            summaries.append({"annotation_version": version, "scope": scope, "n_annotation_rows": len(subset),
                              "n_transfers_with_target_controls": len(transfers),
                              "target_exceeds_controls": int(transfers.target_exceeds_controls.fillna(False).astype(bool).sum()),
                              "source_decreases_more_than_controls": int(transfers.source_decreases_more_than_controls.fillna(False).astype(bool).sum()),
                              "interpretation": "descriptive_counts_not_validation"})
    pd.DataFrame(summaries).to_csv(output / "span_summary.csv", index=False)
    manifest = {"analysis_version": "diagnostics-1", "status": "exploratory",
                "archive_sha256": archive_hash, "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "config_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(config_dir.glob("*.json"))},
                "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))},
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
                "open_issues": ["GxA support alignment", "smoke mismatch", "A004/A016 annotation approval"],
                "numerical_stability_gpu": "not_tested", "semantic_specificity_control": "not_available",
                "clustering": clustering}
    (output / "diagnostics_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# Diagnósticos de saliencia e instrumentación", "",
              "Análisis exploratorio. Las alternativas de anotación no equivalen a decisiones metodológicas aprobadas.", "",
              "## Spans y transferencias", "",
              "La búsqueda utiliza frases completas, respeta acentos y selecciona la primera aparición. Se registran offsets, tokens, multiplicidad y ventanas. El objetivo todavía ausente tiene masa cero por construcción causal; su aparición no demuestra transferencia.", "",
              "`candidate-1` usa las anotaciones propuestas; `archived-pairs-current-controls` conserva los pares archivados pero aplica los controles actuales por palabras de contenido: no es una reproducción del antiguo control por tokens. `source-sensitivity` conserva las alternativas de A004 sin seleccionar la que produzca el resultado más favorable.", "",
              "Las ventanas antes/después se definen por la aparición completa del destino, no necesariamente por el verso hipotetizado. El verso esperado se conserva para revisar esa diferencia. Las comparaciones son descriptivas y no permiten inferir discriminación o ausencia de efecto.", "",
              "| Versión | Ámbito | Transferencias con control | Destino supera control | Caída de origen mayor que control |",
              "|---|---|---:|---:|---:|"]
    for item in summaries:
        report.append(f"| {item['annotation_version']} | {item['scope']} | {item['n_transfers_with_target_controls']} | {item['target_exceeds_controls']} | {item['source_decreases_more_than_controls']} |")
    report += ["", "Las tablas detallan los casos no calculables y los denominadores. A011 se corrige: «sí» no debe localizarse dentro de «sin». En A020, «devolver» aparece como parte de «devolverlo»; la lectura literal queda sin correspondencia y la forma completa se conserva como alternativa exploratoria pendiente, sin sustituirla silenciosamente.", "",
               "Se conservan 30 mapas individuales, con escala común de 0 a 1; las celdas no definidas quedan vacías.", "",
               "## Instrumentación", "",
               "El cociente rango del estadístico por verso / (p99−p1) de los tokens no es una fracción del techo teórico ni una probabilidad de discriminación. La suma de ΔE puede superar ese rango token a token. Las fracciones de ln(2) solo se usan para ΔP y aparecen en columnas distintas.", "",
               "La invariancia de máximos bajo transformaciones afines crecientes es una propiedad matemática, no una validación de la métrica. La saturación se presenta descriptivamente sin convertir umbrales convencionales en pruebas de calidad. La estabilidad GPU y la sensibilidad semántica específica requieren comprobaciones adicionales.", "",
               "## Espacio de estados", "",
               f"HDBSCAN de scikit-learn: {clustering['n_clusters']} grupos, {clustering['n_states']} estados y fracción de ruido {clustering['noise_fraction']:.4f}. Silhouette sin ruido: {clustering['silhouette_excluding_noise']}. La proyección es PCA explícita; no hay sustitución silenciosa de UMAP.", "",
               (f"Comparación con etiquetas archivadas: ARI={clustering['adjusted_rand_against_archived']:.6f}. Silhouette archivado sobre la estandarización actual: {clustering['archived_silhouette_on_current_standardization']:.6f}. Ambas versiones se conservan separadas; no se atribuye la diferencia a una biblioteca sin evidencia." if clustering["archived_reference_available"] else "El paquete no contiene etiquetas de clustering previas: no se realiza comparación archivada."), "",
               "![Estados PCA](figures/states_pca.png)", "",
               "La geometría de tokens correlacionados no demuestra categorías emocionales. No se reutiliza el p de permutación de columnas del análisis anterior, cuya intercambiabilidad temporal no está justificada aquí.", "",
               "## Mapas individuales", ""]
    report += [f"- [{record['record_id']}](figures/maps/{record['record_id']}.png)" for record in corpus]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"{len(spans)} filas de anotación; 30 mapas; resultados en {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256)
    args = parser.parse_args()
    run_diagnostics(args.archive, args.corpus, args.config, args.output, args.archive_sha256)


if __name__ == "__main__":
    main()
