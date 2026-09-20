"""Informe y figuras derivados de las tablas del análisis exploratorio."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_report(directory: Path):
    paired = pd.read_csv(directory / "paired_baselines.csv")
    entropy = paired[(paired["mode"] == "residual") & (paired.baseline == "entropy")]
    global_scores = pd.read_csv(directory / "global_descriptive.csv")
    perms = pd.read_csv(directory / "conditional_permutations.csv")
    profiles = pd.read_csv(directory / "control_profile_sensitivity.csv")
    factorial = pd.read_csv(directory / "factorial_document_bootstrap.csv")
    nominal = pd.read_csv(directory / "nominal_binomial.csv")
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    labels = {"delta_p": "ΔP", "delta_e": "ΔE", "gxa": "GxA"}

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    positions = np.arange(len(entropy))
    ax.bar(positions - .18, entropy.metric_hits / entropy.n_texts, .36, label="Métrica", color="#286682")
    ax.bar(positions + .18, entropy.baseline_hits / entropy.n_texts, .36, label="Entropía", color="#cf8c39")
    ax.set_xticks(positions, [f"{labels[row.metric]} (n={row.n_texts})" for row in entropy.itertuples()])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporción de aciertos")
    ax.set_title("Comparación emparejada tras restar el perfil posicional")
    ax.legend(frameon=False)
    fig.savefig(figures / "paired_entropy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for metric in labels:
        frame = profiles[(profiles.variant == "D2") & (profiles.metric == metric)]
        ax.plot(frame.first_verse, frame.correlation_of_means, "o-", label=labels[metric])
    ax.set_xticks([1, 2], ["Versos 1–6", "Versos 2–6"])
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set_ylabel("Correlación entre perfiles medios")
    ax.set_title("Sensibilidad del perfil original frente al control D2")
    ax.legend(frameon=False)
    fig.savefig(figures / "profile_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    names = ["Soporte a T=1", "Soporte a T=2", "Temperatura: completo", "Temperatura: top-100", "Interacción"]
    values = factorial.mean_document_weighted.to_numpy()
    ax.errorbar(values, np.arange(len(values)), xerr=np.vstack([
        values - factorial.ci95_low, factorial.ci95_high - values]), fmt="o", color="#286682", capsize=4)
    ax.set_yticks(np.arange(len(values)), names)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=.7)
    ax.set_xlabel("Diferencia de ΔP (nats)")
    ax.set_title("Media por documento e intervalo bootstrap del 95 %")
    fig.savefig(figures / "factorial_documents.png", dpi=180)
    plt.close(fig)

    text = ["# Reanálisis estadístico del piloto de Mistral", "",
            "Análisis exploratorio sobre resultados verificados. Las figuras y cifras se generan desde las tablas de esta carpeta.", "",
            "## Acierto y comparaciones emparejadas", ""]
    for row in global_scores.itertuples():
        text.append(f"- {row.mode}: {row.hits}/{row.n_score_rows} aciertos ({row.hit_rate:.2%}), procedentes de {row.n_texts} textos. El resumen global es descriptivo; sus filas no son observaciones independientes.")
    text += ["", "| Métrica | Aciertos | Entropía | p bilateral | p BH |", "|---|---:|---:|---:|---:|"]
    for row in entropy.itertuples():
        text.append(f"| {labels[row.metric]} | {row.metric_hits}/{row.n_texts} | {row.baseline_hits}/{row.n_texts} | {row.p_raw:.5f} | {row.p_bh:.5f} |")
    text += ["", "McNemar exacto usa los pares discordantes. BH se aplica a las 16 comparaciones de cada modo. Los resultados se interpretan como exploratorios: las proyecciones y el perfil posicional se ajustan sobre el propio corpus, creando dependencia adicional que este contraste no modela.", "",
             "![Comparación emparejada](figures/paired_entropy.png)", "",
             "## Referencias nulas y multiplicidad", "",
             f"El menor p ajustado de los ocho contrastes nominales por familia y métrica, en el modo residual, es {nominal[nominal['mode'] == 'residual'].p_bh.min():.5f}. La referencia uniforme p=0,2 es un supuesto nominal; no describe necesariamente el emparejamiento de objetivos y máximos en este corpus.", ""]
    for row in perms[(perms["mode"] == "residual") & (perms.scope == "all")].itertuples():
        text.append(f"- Permutación {row.scheme}: media nula {row.null_mean:.4f}, p Monte Carlo bruto {row.p_raw:.4f}, p BH {row.p_bh:.4f}.")
    text += ["", "Ambos esquemas conservan las métricas de cada texto juntas. La intercambiabilidad de los objetivos debe justificarse; la estratificación por familia es una sensibilidad, no una garantía de validez. Se muestran ambos resultados y no se selecciona el menor p. Las correcciones BH se limitan a las familias declaradas, no a toda la exploración.", "",
             "## Diagnósticos", "",
             "La correlación de medias por verso no demuestra igualdad por texto ni ausencia de sensibilidad al contenido. La figura muestra cuánto cambia al excluir el primer verso.", "",
             "![Sensibilidad de perfiles](figures/profile_sensitivity.png)", "",
             "Los intervalos factoriales remuestrean 30 documentos completos, después de promediar diferencias emparejadas dentro de cada documento. Estiman una media que da el mismo peso a cada texto, distinta de la media ponderada por tokens, que también se conserva en la tabla. Su interpretación poblacional requiere documentos representativos e independientes; el corpus es diseñado y pequeño.", "",
             "![Diagnóstico factorial](figures/factorial_documents.png)", "",
             "## GxA y alcance pendiente", "",
             "Se publican resúmenes separados de los 30 textos y los 12 con hipótesis GxA explícita, y ambas variantes de soporte. El percentil de rango medio entre seis versos tiene referencia 7/12 si el objetivo se elige uniformemente; no 0,5. Esa referencia tampoco describe automáticamente los objetivos reales, que se concentran al final.", "",
             "El resumen por verso no sustituye al análisis de spans y transferencias. El comando `aft8-diagnostics` genera esos cálculos y mapas individuales con versiones de anotación separadas; consultar [su informe](../diagnostics/report.md) después de ejecutarlo. La aprobación de las anotaciones permanece pendiente. No se emite un veredicto de validación instrumental basado en un umbral de acierto.", "",
             "Siguen abiertos la alineación de soportes GxA y el smoke A001/A002 frente a A001/A013. Estos resultados no constituyen por sí solos una validación de AFT8 ni demuestran ausencia de efecto.", ""]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
