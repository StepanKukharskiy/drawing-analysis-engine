#!/usr/bin/env python3
"""Baseline structural-drawing PDF to Engineering Graph extractor.

This is deliberately a perception/evidence pipeline, not a quantity scraper.
It preserves page-space observations and PDF vectors, proposes semantic
candidates, and records what still requires geometric or expert resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import fitz


SCHEMA_VERSION = "0.1.0"

SECTION_RE = re.compile(
    r"^(?:\d+\s*[-–]\s*\d+|[A-ZА-Я]\s*[-–]\s*[A-ZА-Я])(?:\s*\(.+\))?$",
    re.I,
)
SPACING_RE = re.compile(r"(?P<spacing>\d{1,5})\s*[xх×]\s*(?P<count>\d{1,4})\s*=\s*(?P<extent>\d{1,6})", re.I)
NUMBER_RE = re.compile(r"^\s*(?P<value>\d{1,6}(?:[.,]\d+)?)\s*$")
REBAR_RE = re.compile(
    r"(?:[Ø⌀∅Фф]\s*)?(?P<diameter>\d{1,2})\s*"
    r"(?P<grade>A\s*\d{3}[СC]?|А\s*\d{3}[СC]?)"
    r"(?:\s+L\s*=\s*(?P<length>\d{2,6}))?",
    re.I,
)
REBAR_DESIGNATION_RE = re.compile(r"^(?:СК)?(?P<diameter>\d{1,2})\s*/\s*(?P<code>\d{2,4})$", re.I)
MATERIAL_RE = re.compile(
    r"(?P<kind>бетон|пенополистирол|сталь|арматур[аы])"
    r"(?:[^\n]*?\b(?P<class>[BВ]\s*\d{2,3}|A\s*\d{3}[СC]?|А\s*\d{3}[СC]?))?",
    re.I,
)
VOLUME_RE = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?:м|m)\s*[³3]", re.I)

VIEW_KEYWORDS = (
    "план",
    "вид ",
    "опалубка",
    "армирование",
    "разрез",
    "сечение",
)
TABLE_KEYWORDS = (
    "спецификация",
    "ведомость деталей",
    "ведомость расхода",
)

ENTITY_RULES = (
    ("reinforced_concrete_column", re.compile(r"колонн[аы]?\s*([КK]\s*\d+)", re.I)),
    ("reinforced_concrete_beam", re.compile(r"балк[аы]?\s*([БB][^\s,;]{1,6})", re.I)),
    ("stair_flight", re.compile(r"лестничн\w*\s+марш\w*\s*([^\n,;]{1,14})", re.I)),
    ("stair_stringer", re.compile(r"косоур\w*\s*([КK]\s*\d+)", re.I)),
    ("plinth_panel", re.compile(r"панел\w*\s+цокольн\w*\s*([ПP]Ц\s*-?\s*\d+)", re.I)),
    ("dock_leveller_panel", re.compile(r"панел\w*\s+доклевеллер\w*\s*([ДD]П\s*-?\s*\d+)", re.I)),
)


def _num(value: str) -> float:
    return float(value.replace(",", "."))


def _round(value: float) -> float:
    return round(float(value or 0), 3)


def _bbox(rect: Any) -> list[float]:
    return [_round(rect[0]), _round(rect[1]), _round(rect[2]), _round(rect[3])]


def _normalized_bbox(rect: Any, page_rect: fitz.Rect) -> list[float]:
    return [
        round((rect[0] - page_rect.x0) / page_rect.width, 6),
        round((rect[1] - page_rect.y0) / page_rect.height, 6),
        round((rect[2] - page_rect.x0) / page_rect.width, 6),
        round((rect[3] - page_rect.y0) / page_rect.height, 6),
    ]


def _serializable_geometry(value: Any) -> Any:
    if isinstance(value, fitz.Point):
        return [_round(value.x), _round(value.y)]
    if isinstance(value, fitz.Rect):
        return _bbox(value)
    if isinstance(value, fitz.Quad):
        return [_serializable_geometry(p) for p in (value.ul, value.ur, value.ll, value.lr)]
    if isinstance(value, (list, tuple)):
        return [_serializable_geometry(v) for v in value]
    if isinstance(value, float):
        return _round(value)
    return value


def _drawing_style(drawing: dict[str, Any]) -> dict[str, Any]:
    """Return the complete PyMuPDF paint / clipping style in stable names.

    ``Page.get_drawings(extended=True)`` returns several record kinds. Regular
    paths expose stroke and fill styling, clip records expose winding and close
    state, and transparency groups expose blend metadata. Keeping these values
    together prevents visually identical-looking candidates from losing the
    distinctions needed for dimension, hidden-line, hatch, and layer semantics.
    """

    source_to_output = {
        "color": "stroke",
        "fill": "fill",
        "width": "width_page",
        "lineCap": "line_cap",
        "lineJoin": "line_join",
        "closePath": "close_path",
        "dashes": "dashes",
        "stroke_opacity": "stroke_opacity",
        "fill_opacity": "fill_opacity",
        "even_odd": "even_odd",
        "blendmode": "blend_mode",
        "opacity": "opacity",
        "isolated": "isolated",
        "knockout": "knockout",
    }
    return {
        output_key: _serializable_geometry(drawing[source_key])
        for source_key, output_key in source_to_output.items()
        if source_key in drawing
    }


def _ocg_xrefs_by_name(page: fitz.Page) -> dict[str, list[int]]:
    """Index optional-content-group xrefs without assuming unique names."""

    result: dict[str, list[int]] = {}
    document = page.parent
    if document is None:
        return result
    for xref, metadata in document.get_ocgs().items():
        name = metadata.get("name")
        if name:
            result.setdefault(str(name), []).append(int(xref))
    return result


def extract_text_lines(page: fitz.Page) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            rect = fitz.Rect(line["bbox"])
            display = rect * page.rotation_matrix
            result.append(
                {
                    "text": text,
                    "bbox_page": _bbox(rect),
                    "bbox_pdf_unrotated": _bbox(rect),
                    "bbox_display": _bbox(display),
                    # Preserve normalized coordinates in both spaces. PyMuPDF
                    # text/vector coordinates use the unrotated crop box.
                    "bbox_normalized": _normalized_bbox(rect, page.cropbox),
                    "bbox_display_normalized": _normalized_bbox(display, page.rect),
                    "direction": [_round(v) for v in line.get("dir", (1, 0))],
                    "font_sizes": sorted({_round(s.get("size", 0)) for s in spans}),
                }
            )
    return result


def serialize_paths(page: fitz.Page, detail: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The extended scene retains clip paths and transparency groups. The
    # ordinary form only returns painted paths, which silently loses clipping,
    # blend hierarchy, and scene nesting information.
    drawings = page.get_drawings(extended=True)
    ocg_xrefs = _ocg_xrefs_by_name(page)
    item_types: Counter[str] = Counter()
    drawing_types: Counter[str] = Counter()
    record_ids: Counter[str] = Counter()
    primitives: list[dict[str, Any]] = []
    for index, drawing in enumerate(drawings):
        drawing_type = str(drawing.get("type", "unknown"))
        drawing_types[drawing_type] += 1
        for item in drawing.get("items", []):
            item_types[str(item[0])] += 1
        if detail == "none":
            continue
        bbox = drawing.get("rect", drawing.get("scissor"))
        if bbox is None:
            # Current PyMuPDF records always provide one of these, but keeping
            # a null box is safer than discarding future scene record kinds.
            bbox_page = None
            bbox_display = None
        else:
            bbox_page = _bbox(bbox)
            bbox_display = _bbox(fitz.Rect(bbox) * page.rotation_matrix)
        layer_name = drawing.get("layer") or None
        style = _drawing_style(drawing)
        primitive_type = {
            "clip": "pdf_vector_clip_path",
            "group": "pdf_vector_transparency_group",
        }.get(drawing_type, "pdf_vector_path")
        id_prefix = {"clip": "clip", "group": "group"}.get(drawing_type, "path")
        record_ids[id_prefix] += 1
        primitive: dict[str, Any] = {
            # Separate counters keep legacy painted-path IDs stable even when
            # extended scene records are interleaved before them.
            "id": f"{id_prefix}_{page.number + 1}_{record_ids[id_prefix]}",
            "type": primitive_type,
            "page": page.number + 1,
            "bbox_page": bbox_page,
            "bbox_pdf_unrotated": bbox_page,
            "bbox_display": bbox_display,
            "drawing_type": drawing_type,
            "scene_index": index,
            "scene_level": drawing.get("level"),
            "sequence_number": drawing.get("seqno"),
            "layer_name": layer_name,
            "ocg_xrefs": ocg_xrefs.get(layer_name, []) if layer_name else [],
            "style": style,
            # Preserve the original top-level fields consumed by existing
            # experiments while exposing the full style object above.
            "stroke": _serializable_geometry(drawing.get("color")),
            "fill": _serializable_geometry(drawing.get("fill")),
            "width_page": _round(drawing.get("width", 0)),
            "close_path": bool(drawing.get("closePath", False)),
            "item_types": [str(item[0]) for item in drawing.get("items", [])],
            "source": "pdf_vector",
        }
        if "scissor" in drawing:
            primitive["clip_scissor_page"] = _bbox(drawing["scissor"])
        if detail == "full":
            primitive["items"] = [_serializable_geometry(item) for item in drawing.get("items", [])]
        primitives.append(primitive)
    painted_path_count = sum(
        count for drawing_type, count in drawing_types.items() if drawing_type not in {"clip", "group"}
    )
    stats = {
        # Keep path_count compatible with the former non-extended extraction.
        "path_count": painted_path_count,
        "scene_record_count": len(drawings),
        "clip_path_count": drawing_types["clip"],
        "transparency_group_count": drawing_types["group"],
        "drawing_types": dict(drawing_types),
        "ocg_layer_names": sorted(name for name in ocg_xrefs),
        "path_item_count": sum(item_types.values()),
        "item_types": dict(item_types),
    }
    return primitives, stats


def confidence(perception: float, interpretation: float, calculation: float | None = None) -> dict[str, float]:
    value = {
        "perception": round(perception, 3),
        "interpretation": round(interpretation, 3),
    }
    if calculation is not None:
        value["calculation"] = round(calculation, 3)
    return value


def add_evidence(
    graph: dict[str, Any], page: int, line: dict[str, Any], method: str = "pdf_text"
) -> str:
    evidence_id = f"evidence_{len(graph['evidence']) + 1}"
    graph["evidence"].append(
        {
            "id": evidence_id,
            "page": page,
            "region_page": line["bbox_page"],
            "region_pdf_unrotated": line["bbox_pdf_unrotated"],
            "region_display": line["bbox_display"],
            "region_normalized": line["bbox_normalized"],
            "region_display_normalized": line["bbox_display_normalized"],
            "text": line["text"],
            "method": method,
        }
    )
    return evidence_id


def classify_page(text: str, path_count: int, image_count: int) -> dict[str, Any]:
    text_chars = len(text.strip())
    if text_chars == 0 and path_count > 0:
        mode = "vector_geometry_with_outlined_or_missing_text"
    elif text_chars < 180 and path_count > 1000:
        mode = "vector_geometry_with_sparse_text"
    elif path_count > 0 and text_chars > 0:
        mode = "hybrid_vector_text"
    elif image_count > 0:
        mode = "raster_or_image_dominant"
    else:
        mode = "unknown"
    return {
        "mode": mode,
        "native_text_chars": text_chars,
        "native_text_sufficient": text_chars >= 180,
        "requires_ocr_or_vlm_text_fallback": text_chars < 180,
    }


def detect_entities(graph: dict[str, Any], all_text: str, evidence_by_line: list[tuple[int, dict[str, Any], str]]) -> None:
    for entity_type, pattern in ENTITY_RULES:
        for match in pattern.finditer(all_text):
            label = re.sub(r"\s+", "", match.group(1)).upper()
            duplicate = any(e.get("mark") == label and e["entity_type"] == entity_type for e in graph["entities"])
            if duplicate:
                continue
            supporting = next((eid for _, line, eid in evidence_by_line if label.lower() in re.sub(r"\s+", "", line["text"]).lower()), None)
            graph["entities"].append(
                {
                    "id": f"entity_{len(graph['entities']) + 1}",
                    "entity_type": entity_type,
                    "mark": label,
                    "knowledge_status": "direct" if supporting else "inferred",
                    "evidence_ids": [supporting] if supporting else [],
                    "confidence": confidence(0.96 if supporting else 0.8, 0.92 if supporting else 0.68),
                }
            )


def analyze_lines(graph: dict[str, Any], page_number: int, lines: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], str]]:
    evidence_by_line: list[tuple[int, dict[str, Any], str]] = []
    for line in lines:
        text = re.sub(r"\s+", " ", line["text"]).strip()
        lowered = text.lower()
        interesting = False

        is_view = bool(SECTION_RE.match(text)) or any(keyword in lowered for keyword in VIEW_KEYWORDS)
        if is_view and len(text) <= 100:
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            graph["views"].append(
                {
                    "id": f"view_{len(graph['views']) + 1}",
                    "page": page_number,
                    "label": text,
                    "view_type": "section" if SECTION_RE.match(text) else "view_candidate",
                    "label_bbox_page": line["bbox_page"],
                    "label_bbox_display": line["bbox_display"],
                    "region_bbox_page": None,
                    "region_geometry": None,
                    "knowledge_status": "direct",
                    "region_status": "unknown",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.98, 0.76),
                }
            )
            interesting = True

        if any(keyword in lowered for keyword in TABLE_KEYWORDS):
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            graph["annotations"].append(
                {
                    "id": f"annotation_{len(graph['annotations']) + 1}",
                    "annotation_type": "table_title",
                    "page": page_number,
                    "text": text,
                    "table_region_bbox_page": None,
                    "knowledge_status": "direct",
                    "region_status": "unknown",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.99, 0.9),
                }
            )
            interesting = True

        spacing_match = SPACING_RE.search(text)
        if spacing_match:
            spacing = int(spacing_match.group("spacing"))
            count = int(spacing_match.group("count"))
            extent = int(spacing_match.group("extent"))
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            graph["dimensions"].append(
                {
                    "id": f"dimension_{len(graph['dimensions']) + 1}",
                    "dimension_type": "spacing_pattern",
                    "page": page_number,
                    "spacing_mm": spacing,
                    "interval_count": count,
                    "extent_mm": extent,
                    "arithmetic_check": {
                        "expected_extent_mm": spacing * count,
                        "status": "pass" if spacing * count == extent else "fail",
                    },
                    "applies_to": [],
                    "association_status": "unknown",
                    "knowledge_status": "direct",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.99, 0.72, 1.0),
                }
            )
            interesting = True
        elif (number_match := NUMBER_RE.match(text)) and len(text) <= 12:
            value = _num(number_match.group("value"))
            if 5 <= value <= 100000:
                evidence_id = add_evidence(graph, page_number, line)
                evidence_by_line.append((page_number, line, evidence_id))
                graph["annotations"].append(
                    {
                        "id": f"annotation_{len(graph['annotations']) + 1}",
                        "annotation_type": "unclassified_numeric_text",
                        "page": page_number,
                        "value": value,
                        "unit": None,
                        "semantic_role": "unknown",
                        "knowledge_status": "direct",
                        "evidence_ids": [evidence_id],
                        "confidence": confidence(0.98, 0.12),
                    }
                )
                interesting = True

        rebar_match = REBAR_RE.search(text)
        if rebar_match and ("A" in text.upper() or "А" in text.upper()):
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            graph["annotations"].append(
                {
                    "id": f"annotation_{len(graph['annotations']) + 1}",
                    "annotation_type": "rebar_property",
                    "page": page_number,
                    "diameter_mm": int(rebar_match.group("diameter")),
                    "steel_grade": re.sub(r"\s+", "", rebar_match.group("grade")).replace("А", "A"),
                    "length_mm": int(rebar_match.group("length")) if rebar_match.group("length") else None,
                    "identifies_entity": None,
                    "association_status": "unknown",
                    "knowledge_status": "direct",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.97, 0.82),
                }
            )
            interesting = True
        elif designation_match := REBAR_DESIGNATION_RE.match(text):
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            graph["annotations"].append(
                {
                    "id": f"annotation_{len(graph['annotations']) + 1}",
                    "annotation_type": "rebar_designation_candidate",
                    "page": page_number,
                    "diameter_mm_candidate": int(designation_match.group("diameter")),
                    "designation_code": designation_match.group("code"),
                    "interpretation_status": "inferred",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.97, 0.58),
                }
            )
            interesting = True

        material_match = MATERIAL_RE.search(text)
        if material_match:
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            volume_match = VOLUME_RE.search(text)
            graph["materials"].append(
                {
                    "id": f"material_{len(graph['materials']) + 1}",
                    "page": page_number,
                    "material_type": material_match.group("kind").lower(),
                    "class": material_match.group("class"),
                    "declared_volume_m3": _num(volume_match.group("value")) if volume_match else None,
                    "knowledge_status": "direct",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.98, 0.86),
                }
            )
            interesting = True

        if (volume_match := VOLUME_RE.search(text)) and not material_match:
            evidence_id = add_evidence(graph, page_number, line)
            evidence_by_line.append((page_number, line, evidence_id))
            graph["annotations"].append(
                {
                    "id": f"annotation_{len(graph['annotations']) + 1}",
                    "annotation_type": "declared_volume",
                    "page": page_number,
                    "value": _num(volume_match.group("value")),
                    "unit": "m3",
                    "applies_to": None,
                    "association_status": "unknown",
                    "knowledge_status": "direct",
                    "evidence_ids": [evidence_id],
                    "confidence": confidence(0.98, 0.5),
                }
            )
            interesting = True

        if not interesting:
            # Entity detection uses only source lines that are retained as evidence.
            if any(pattern.search(text) for _, pattern in ENTITY_RULES):
                evidence_id = add_evidence(graph, page_number, line)
                evidence_by_line.append((page_number, line, evidence_id))

    # Deduplicate evidence created when one line matches multiple detectors.
    unique: dict[tuple[int, tuple[float, ...], str], tuple[int, dict[str, Any], str]] = {}
    for item in evidence_by_line:
        key = (item[0], tuple(item[1]["bbox_page"]), item[1]["text"])
        unique.setdefault(key, item)
    return list(unique.values())


def extract_document(path: Path, geometry_detail: str = "summary") -> dict[str, Any]:
    doc = fitz.open(path)
    graph: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document": {
            "id": path.stem,
            "source_file": str(path.resolve()),
            "page_count": doc.page_count,
            "discipline": "structural",
            "units": "mm",
            "units_status": "inferred_from_structural_drawing_convention",
        },
        "sheets": [],
        "views": [],
        "primitives": [],
        "entities": [],
        "geometry": [],
        "dimensions": [],
        "annotations": [],
        "relationships": [],
        "materials": [],
        "calculations": [],
        "checks": [],
        "evidence": [],
        "unknowns": [],
        "pipeline": {
            "extractor": "rebar_pdf_pipeline",
            "extractor_version": SCHEMA_VERSION,
            "geometry_detail": geometry_detail,
        },
    }

    all_lines: list[dict[str, Any]] = []
    all_evidence_lines: list[tuple[int, dict[str, Any], str]] = []
    for page in doc:
        lines = extract_text_lines(page)
        all_lines.extend(lines)
        text = "\n".join(line["text"] for line in lines)
        primitives, vector_stats = serialize_paths(page, geometry_detail)
        graph["primitives"].extend(primitives)
        image_count = len(page.get_images(full=True))
        mode = classify_page(text, vector_stats["path_count"], image_count)
        graph["sheets"].append(
            {
                "id": f"sheet_{page.number + 1}",
                "page": page.number + 1,
                "coordinate_system": "pdf_unrotated_cropbox",
                "display_coordinate_system": "rotated_page_top_left",
                "page_size_points": [_round(page.cropbox.width), _round(page.cropbox.height)],
                "display_size_points": [_round(page.rect.width), _round(page.rect.height)],
                "rotation": page.rotation,
                "source_mode": mode,
                "text_line_count": len(lines),
                "embedded_image_count": image_count,
                "vector_stats": vector_stats,
            }
        )
        all_evidence_lines.extend(analyze_lines(graph, page.number + 1, lines))

    all_text = "\n".join(line["text"] for line in all_lines)
    detect_entities(graph, all_text, all_evidence_lines)

    graph["checks"].append(
        {
            "id": "check_spacing_arithmetic",
            "check_type": "spacing_expression_arithmetic",
            "status": "pass"
            if all(
                d.get("arithmetic_check", {}).get("status") != "fail"
                for d in graph["dimensions"]
            )
            else "fail",
            "checked_items": sum(d["dimension_type"] == "spacing_pattern" for d in graph["dimensions"]),
        }
    )

    unresolved = [
        (
            "view_regions",
            "Detected view labels are not yet expanded into primitive-supported polygonal regions. "
            "View regions may overlap where dimensions, leaders, or wide sections spill across layout columns.",
        ),
        ("dimension_associations", "Dimension values are not yet attached to extension lines and measured geometry."),
        ("leader_relations", "Leader/callout endpoints are not yet linked to the objects they identify."),
        ("cross_view_correspondence", "Projections in different views are not yet merged into physical entities."),
        ("metric_3d_geometry", "Concrete solids and rebar centerlines require constraint solving after cross-view matching."),
        ("hidden_schedule_validation", "Declared schedule values must be masked during inference and used only for evaluation."),
    ]
    for code, description in unresolved:
        graph["unknowns"].append(
            {
                "code": code,
                "description": description,
                "status": "unknown",
            }
        )
    if any(sheet["source_mode"]["requires_ocr_or_vlm_text_fallback"] for sheet in graph["sheets"]):
        graph["unknowns"].append(
            {
                "code": "missing_native_text",
                "description": "At least one page requires tiled OCR or a vision-language fallback for text represented as outlines.",
                "status": "unknown",
            }
        )

    return graph


def discover_inputs(inputs: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf" and path.exists():
            found.append(path)
        else:
            found.extend(sorted(Path().glob(raw)))
    unique = {path.resolve(): path for path in found if path.suffix.lower() == ".pdf"}
    return [unique[key] for key in sorted(unique, key=lambda p: str(p).lower())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="PDF files, glob patterns, or directories")
    parser.add_argument("--output-dir", default="output/analysis", help="Directory for Engineering Graph JSON")
    parser.add_argument(
        "--geometry-detail",
        choices=("none", "summary", "full"),
        default="full",
        help="none: stats only; summary: path metadata; full: include path segments",
    )
    args = parser.parse_args(argv)

    paths = discover_inputs(args.inputs)
    if not paths:
        parser.error("no PDFs found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        graph = extract_document(path, args.geometry_detail)
        output_path = output_dir / f"{path.stem}.engineering-graph.json"
        output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        page_modes = ",".join(sheet["source_mode"]["mode"] for sheet in graph["sheets"])
        print(
            f"{path.name}: {page_modes}; views={len(graph['views'])}; "
            f"dimensions={len(graph['dimensions'])}; annotations={len(graph['annotations'])}; "
            f"output={output_path}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
