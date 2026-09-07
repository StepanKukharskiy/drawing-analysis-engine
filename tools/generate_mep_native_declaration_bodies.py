#!/usr/bin/env python3
"""Freeze independent native declaration bodies without changing legacy M7C."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
from src.drawing_engine.disciplines.mep.mep_native_declaration_bodies import extract_mep_native_declaration_bodies, validate_mep_native_declaration_bodies


def generate(*, source, registry, output_dir):
    m1 = json.loads(Path(registry).read_text(encoding="utf-8"))
    payload = extract_mep_native_declaration_bodies(pdf_path=source, sheet_registry=m1)
    errors = validate_mep_native_declaration_bodies(payload)
    if errors:
        raise ValueError("\n".join(errors))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "native-declaration-bodies.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest = {"mode": "automatic_independent_native_declaration_bodies", "source_pdf_sha256": payload["document"]["source_pdf_sha256"],
                "m1_payload_sha256": payload["m1_payload_sha256"], "artifact": {"path": output.name, "sha256": _file_sha256(output)},
                "summary": payload["summary"], "validation_passed": True, "calculated_records_used": False,
                "document_declaration_completeness_established": False, "quantity_eligible": False}
    (output_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(source=args.source, registry=args.registry, output_dir=args.output_dir), indent=2))
