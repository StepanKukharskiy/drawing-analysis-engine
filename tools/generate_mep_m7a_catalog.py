#!/usr/bin/env python3
"""Generate the frozen M7A JSON catalog and flat CSV review table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_item_catalog import (  # noqa: E402
    build_mep_item_catalog,
    flatten_mep_item_catalog,
    validate_mep_item_catalog,
)


FIXTURE_ROOT = ROOT / "fixtures" / "mep" / "m_and_p_coordination"
CHECKPOINT_ROOT = FIXTURE_ROOT / "real_m2_m5_checkpoint"
M4_COVERAGE_ROOT = FIXTURE_ROOT / "real_m4_discrete_coverage"
DEFAULT_REGISTRY = FIXTURE_ROOT / "m_and_p_coordination.sheet-registry.json"
DEFAULT_BINDINGS = M4_COVERAGE_ROOT / "full_package.attribute-bindings.json"
DEFAULT_M5A = CHECKPOINT_ROOT / "pages_1a_1b.bounded-local-3d.json"
DEFAULT_COVERAGE = FIXTURE_ROOT / "m_and_p_coordination.occurrence-coverage-audit.json"
DEFAULT_M7B = FIXTURE_ROOT / "m_and_p_coordination.bounded-discrete-counts.json"
DEFAULT_JSON = FIXTURE_ROOT / "m_and_p_coordination.mep-item-catalog.json"
DEFAULT_CSV = FIXTURE_ROOT / "m_and_p_coordination.mep-item-catalog.review.csv"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate(
    *,
    registry_path: Path,
    bindings_path: Path,
    m5a_path: Path | None,
    coverage_path: Path | None,
    m7b_path: Path | None,
    json_path: Path,
    csv_path: Path,
) -> tuple[Path, Path]:
    payload = build_mep_item_catalog(
        sheet_registry=_load(registry_path),
        attribute_bindings=_load(bindings_path),
        bounded_local_3d=_load(m5a_path) if m5a_path is not None else None,
        occurrence_coverage_audit=(
            _load(coverage_path) if coverage_path is not None else None
        ),
        bounded_discrete_counts=_load(m7b_path) if m7b_path is not None else None,
    )
    errors = validate_mep_item_catalog(payload)
    if errors:
        raise ValueError("\n".join(errors))
    rows = flatten_mep_item_catalog(payload)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: "null" if value is None else value
                for key, value in row.items()
            }
            for row in rows
        )
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--m5a", type=Path, default=DEFAULT_M5A)
    parser.add_argument("--without-m5a", action="store_true")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--m7b", type=Path, default=DEFAULT_M7B)
    parser.add_argument("--without-m7b", action="store_true")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    paths = generate(
        registry_path=args.registry.resolve(),
        bindings_path=args.bindings.resolve(),
        m5a_path=None if args.without_m5a else args.m5a.resolve(),
        coverage_path=None if args.without_m7b else args.coverage.resolve(),
        m7b_path=None if args.without_m7b else args.m7b.resolve(),
        json_path=args.json.resolve(),
        csv_path=args.csv.resolve(),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
