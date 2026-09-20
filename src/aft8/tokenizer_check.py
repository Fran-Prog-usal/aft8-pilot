"""Comprueba decode, IDs y offsets con el tokenizador de la revisión fijada."""

import argparse
import hashlib
import json
from pathlib import Path

from .model import MODEL_ID, REVISION, tokenize_exact
from .results import ResultArchive


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, use_fast=True, cache_dir=str(args.cache)
    )
    checks = []
    with ResultArchive(args.archive) as archive:
        for run, path in (("original", args.corpus), ("control", args.controls)):
            table = archive.table(run, "token")
            for record in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
                encoding = tokenize_exact(tokenizer, record["text_content"])
                saved = table[table.record_id == record["record_id"]].sort_values("token_idx")
                checks.append(
                    {
                        "record_id": record["record_id"],
                        "decode_exact": encoding["decode_exact"],
                        "ids_match": encoding["input_ids"] == saved.token_id.tolist(),
                        "offsets_match": encoding["offsets"]
                        == list(zip(saved.char_start, saved.char_end)),
                    }
                )
    report = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checks": checks,
        "files_sha256": {
            p.relative_to(args.cache).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.cache.rglob("*"))
            if p.is_file() and p.suffix in (".json", ".model")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = all(
        all(row[key] for key in ("decode_exact", "ids_match", "offsets_match")) for row in checks
    )
    print(
        f"{len(checks)} textos: decode, IDs y offsets {'correctos' if passed else 'con discrepancias'}."
    )
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
