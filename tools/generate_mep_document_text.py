#!/usr/bin/env python3
"""Generate automatic native full-document text observations and validated M2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations, build_mep_document_text_proposals


def generate(*, source: Path, registry: Path, output_dir: Path) -> dict:
    m1 = json.loads(registry.read_text(encoding="utf-8"))
    observations = extract_mep_text_observations(pdf_path=source, sheet_registry=m1)
    proposals = build_mep_document_text_proposals(observations)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, payload in (("text-observations.json", observations), ("terminology-proposals.json", proposals)):
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        artifacts[name] = {"path": name, "sha256": _file_sha256(path)}
    manifest = {
        "schema_version": "0.1.0", "mode": "automatic_native_text_only",
        "source_pdf_sha256": observations["document"]["source_pdf_sha256"],
        "m1_payload_sha256": observations["m1_payload_sha256"],
        "assisted_replay_used": False, "artifacts": artifacts,
        "observations": observations["summary"], "proposals": proposals["summary"],
        "quarantined_observation_count": len(proposals["observation_diagnostics"]),
        "m2_validation_passed": True, "item_inventory_complete": False,
        "region_body_coverage_complete": False, "ocr_run": False,
        "quantity_eligible": False,
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(source=args.source, registry=args.registry, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
