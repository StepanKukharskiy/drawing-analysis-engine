#!/usr/bin/env python3
"""Generate the frozen independent M7C declaration result from M1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_declared_data import (
    build_mep_declared_data,
    extract_mep_document_region_observations,
    validate_mep_declared_data,
    validate_mep_document_region_observations,
)


FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
DEFAULT_M1 = FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"
DEFAULT_SOURCE = ROOT / "M&P mark-up against shop systems piping.pdf"
DEFAULT_REGIONS = FIXTURE_ROOT / "m_and_p_coordination.declaration-region-observations.json"
DEFAULT_OUTPUT = FIXTURE_ROOT / "m_and_p_coordination.declared-data.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate(
    *,
    m1_path: Path,
    source_pdf_path: Path | None,
    regions_path: Path,
    output_path: Path,
) -> tuple[Path, Path]:
    registry = _load(m1_path)
    if source_pdf_path is None:
        region_observations = _load(regions_path)
    else:
        region_observations = extract_mep_document_region_observations(
            pdf_path=source_pdf_path, sheet_registry=registry
        )
        regions_path.parent.mkdir(parents=True, exist_ok=True)
        regions_path.write_text(
            json.dumps(region_observations, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    region_errors = validate_mep_document_region_observations(region_observations)
    if region_errors:
        raise ValueError("\n".join(region_errors))
    payload = build_mep_declared_data(
        sheet_registry=registry, region_observations=region_observations
    )
    errors = validate_mep_declared_data(payload)
    if errors:
        raise ValueError("\n".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return regions_path, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, default=DEFAULT_M1)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--reuse-frozen-regions", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in generate(
        m1_path=args.m1.resolve(),
        source_pdf_path=None if args.reuse_frozen_regions else args.source_pdf.resolve(),
        regions_path=args.regions.resolve(),
        output_path=args.output.resolve(),
    ):
        print(path)


if __name__ == "__main__":
    main()
