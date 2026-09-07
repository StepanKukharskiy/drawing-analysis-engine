#!/usr/bin/env python3
"""Freeze the Candidate 08 concrete + M&P MEP shared takeoff checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.concrete.concrete_takeoff_adapter import build_concrete_takeoff_adapter
from src.drawing_engine.disciplines.mep.mep_takeoff_adapter import build_mep_takeoff_adapter
from src.drawing_engine.disciplines.rebar.rebar_takeoff_adapter import build_rebar_takeoff_adapter
from src.drawing_engine.project.takeoff_intelligence import (
    build_takeoff_intelligence_catalog,
    validate_takeoff_adapter,
    validate_takeoff_intelligence_catalog,
)


DEFAULT_ENGINEERING = (
    ROOT / "output" / "object_agnostic" / "candidate-08-staircase-page.engineering-graph.json"
)
DEFAULT_MEP = (
    ROOT
    / "fixtures"
    / "mep"
    / "m_and_p_coordination"
    / "m_and_p_coordination.mep-item-catalog.json"
)
DEFAULT_REBAR_POSITIVE = (
    ROOT / "output" / "object_agnostic" / "3179 ЛС (2)-2.engineering-graph.json"
)
DEFAULT_OUTPUT = ROOT / "fixtures" / "takeoff_intelligence" / "step1c_shared"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return path


def generate(
    *,
    engineering_path: Path = DEFAULT_ENGINEERING,
    mep_catalog_path: Path = DEFAULT_MEP,
    rebar_positive_path: Path = DEFAULT_REBAR_POSITIVE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Path]:
    candidate_graph = _load(engineering_path)
    concrete = build_concrete_takeoff_adapter(candidate_graph)
    candidate_rebar = build_rebar_takeoff_adapter(candidate_graph)
    positive_rebar = build_rebar_takeoff_adapter(_load(rebar_positive_path))
    mep = build_mep_takeoff_adapter(_load(mep_catalog_path))
    for payload in (concrete, candidate_rebar, positive_rebar, mep):
        errors = validate_takeoff_adapter(payload)
        if errors:
            raise ValueError("\n".join(errors))
    catalog = build_takeoff_intelligence_catalog(
        adapters=[concrete, candidate_rebar, positive_rebar, mep]
    )
    errors = validate_takeoff_intelligence_catalog(catalog)
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "candidate-08.concrete-takeoff-adapter.json": _write(
            output_dir / "candidate-08.concrete-takeoff-adapter.json", concrete
        ),
        "candidate-08.rebar-takeoff-adapter.json": _write(
            output_dir / "candidate-08.rebar-takeoff-adapter.json",
            candidate_rebar,
        ),
        "3179.rebar-takeoff-adapter.json": _write(
            output_dir / "3179.rebar-takeoff-adapter.json", positive_rebar
        ),
        "m-and-p.mep-takeoff-adapter.json": _write(
            output_dir / "m-and-p.mep-takeoff-adapter.json", mep
        ),
        "shared-takeoff-catalog.json": _write(
            output_dir / "shared-takeoff-catalog.json", catalog
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering", type=Path, default=DEFAULT_ENGINEERING)
    parser.add_argument("--mep-catalog", type=Path, default=DEFAULT_MEP)
    parser.add_argument("--rebar-positive", type=Path, default=DEFAULT_REBAR_POSITIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in generate(
        engineering_path=args.engineering.resolve(),
        mep_catalog_path=args.mep_catalog.resolve(),
        rebar_positive_path=args.rebar_positive.resolve(),
        output_dir=args.output_dir.resolve(),
    ).values():
        print(path)


if __name__ == "__main__":
    main()
