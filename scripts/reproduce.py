"""Verifica los artefactos y reproduce el análisis local, sin ejecutar la GPU."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_assets(directory):
    manifest = json.loads((ROOT / "config/release_assets.json").read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        path = directory / asset["name"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha256"]:
            raise ValueError(f"Hash incorrecto: {path.name}")
        print(f"Verificado: {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    assets, output = args.assets.resolve(), args.output.resolve()
    verify_assets(assets)
    if args.verify_only:
        return
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / "pilot30.jsonl"
    controls = output / "controls.jsonl"
    archive = assets / "results-mistral.tar.gz"

    def run(module, *arguments):
        subprocess.run([sys.executable, "-m", module, *map(str, arguments)], check=True, cwd=ROOT)

    run("aft8.corpus", assets / "AFT8_Pilot30_Results_Template_v1_0_Student_Edition.xlsx", corpus)
    run("aft8.controls", corpus, ROOT / "config/control_design.json", controls)
    for cohort, dataset in [("original", corpus), ("control", controls)]:
        run(
            "aft8.validation",
            archive,
            dataset,
            "--run",
            cohort,
            "--output",
            output / f"validation_{cohort}.json",
        )
    run("aft8.analysis", archive, corpus, controls, "--output", output / "analysis")
    run(
        "aft8.diagnostics",
        archive,
        corpus,
        "--config",
        ROOT / "config",
        "--output",
        output / "diagnostics",
    )
    print(f"Reproducción completada: {output}")


if __name__ == "__main__":
    main()
