#!/usr/bin/env python3
"""Summarize uniformly generated drawing-understanding corpus outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _declared(page: dict[str, Any], kind: str) -> float | None:
    rows = page.get("declared_by_designer", {}).get(kind, [])
    return rows[0].get("value") if len(rows) == 1 else None


def summarize_result(source: Path, graph: dict[str, Any], estimate: dict[str, Any] | None) -> dict[str, Any]:
    pages = []
    page_timings = graph.get("timing", {}).get("page_understanding_seconds", [])
    estimate_pages = {item["page"]: item for item in (estimate or {}).get("pages", [])}
    for page in graph.get("pages", []):
        program = page.get("rebar_program", {})
        details = program.get("native_vector_detail_linking", {})
        families = details.get("physical_families", [])
        preview = page.get("solid_preview") or {}
        preview_paths = preview.get("rebar_paths", [])
        family_path_count = program.get("family_constrained_3d", {}).get("path_count", 0)
        rebar_3d_path_count = len(preview_paths) if preview_paths else family_path_count
        quantity = (page.get("quantities") or [{}])[0]
        estimate_page = estimate_pages.get(page.get("page"), {})
        pages.append(
            {
                "page": page.get("page"),
                "drawing_understanding_seconds": page_timings[len(pages)] if len(pages) < len(page_timings) else None,
                "view_count": len(page.get("view_hypotheses", [])),
                "view_frame_summary": page.get("view_frame_graph", {}).get("summary", {}),
                "dimension_ownership_summary": page.get("dimension_ownership", {}).get("summary", {}),
                "object_instance_count": page.get("object_instance_graph", {}).get("summary", {}).get("object_instance_count", 0),
                "solid_status": page.get("specialised_solver", {}).get("status"),
                "calculated_concrete_m3": quantity.get("net_concrete_m3"),
                "declared_concrete_m3": _declared(estimate_page, "concrete") if estimate_page else None,
                "detail_count": len(details.get("details", [])),
                "accepted_detail_placement_count": sum(item.get("state") == "accepted" for item in details.get("placement_associations", [])),
                "physical_family_count": len(families),
                "resolved_family_count": sum(item.get("constraint_status") == "resolved" for item in families),
                "rebar_3d_path_count": rebar_3d_path_count,
                "rebar_3d_path_basis": (
                    "solid_preview.drawing_constrained_rebar_paths"
                    if preview_paths
                    else "family_constrained_3d.path_count"
                ),
                "calculated_rebar_mass_kg": estimate_page.get("calculated_from_drawing", {}).get("reinforcement_mass_kg") if estimate_page else None,
                "declared_rebar_mass_kg": _declared(estimate_page, "reinforcement") if estimate_page else None,
            }
        )
    return {
        "source_pdf": str(source.resolve()),
        "pipeline": graph.get("pipeline"),
        "drawing_understanding_seconds": graph.get("timing", {}).get("drawing_understanding_seconds"),
        "canonical_graph_summary": graph.get("canonical_knowledge_graph", {}).get("summary", {}),
        "canonical_graph_validation": graph.get("canonical_knowledge_graph", {}).get("validation", {}),
        "estimate_available": estimate is not None,
        "pages": pages,
    }


def build_report(inputs: list[Path], graph_dir: Path, estimate_dir: Path) -> dict[str, Any]:
    results = []
    missing = []
    for source in inputs:
        graph_path = graph_dir / f"{source.stem}.engineering-graph.json"
        graph = _load(graph_path)
        if graph is None:
            missing.append(str(graph_path.resolve()))
            continue
        estimate = _load(estimate_dir / f"{source.stem}.estimate-comparison.json")
        results.append(summarize_result(source, graph, estimate))
    fingerprints = {
        item.get("pipeline", {}).get("sha256")
        for item in results
        if item.get("pipeline", {}).get("sha256")
    }
    return {
        "schema_version": "0.1.0",
        "status": "complete" if len(results) == len(inputs) and len(fingerprints) == 1 else "incomplete_or_mixed",
        "uniform_pipeline": len(results) == len(inputs) and len(fingerprints) == 1,
        "pipeline_fingerprints": sorted(fingerprints),
        "requested_input_count": len(inputs),
        "result_count": len(results),
        "missing_engineering_graphs": missing,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "output" / "object_agnostic")
    parser.add_argument("--estimate-dir", type=Path, default=ROOT / "output" / "estimates")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "corpus-regression.json")
    args = parser.parse_args()
    report = build_report(args.inputs, args.graph_dir, args.estimate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
