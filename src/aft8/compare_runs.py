"""Compara dos extracciones sin exigir identidad binaria de contenedores NPZ."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_runs(first, second, rtol=1e-5, atol=1e-7):
    left = {path.name: path for path in (first / "gxa_vectors").glob("*.npz")}
    right = {path.name: path for path in (second / "gxa_vectors").glob("*.npz")}
    if not left or left.keys() != right.keys():
        raise ValueError("Las ejecuciones no contienen los mismos textos.")
    failures, maxima = [], {}
    for name in sorted(left):
        with np.load(left[name], allow_pickle=False) as a, np.load(right[name], allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                failures.append(name + ": claves distintas")
                continue
            for key in a.files:
                x, y = a[key], b[key]
                if x.shape != y.shape:
                    failures.append(name + ": forma distinta en " + key)
                    continue
                if x.dtype.kind in "fc" and y.dtype.kind in "fc":
                    valid = np.isfinite(x) & np.isfinite(y)
                    difference = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else 0.
                    maxima[key] = max(maxima.get(key, 0.), difference)
                    if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                        failures.append(name + ": diferencia numérica en " + key)
                elif not np.array_equal(x, y):
                    failures.append(name + ": valores distintos en " + key)
    return {"n_texts": len(left), "rtol": rtol, "atol": atol, "failures": failures,
            "maximum_absolute_difference": maxima, "scope": "Arrays guardados; no acredita por sí solo todas las tablas o el entorno."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['n_texts']} textos; {len(result['failures'])} discrepancias.")
    raise SystemExit(bool(result["failures"]))


if __name__ == "__main__":
    main()
