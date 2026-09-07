"""Independent, bounded native schedule-body recovery for M7C.

Native table detection supplies candidates only. A body closes only through a
unique title, native-supported perimeter and cell edges, a tiled grid, retained
native words inside that boundary, and an actual body below the headers. Words
crossing internal cell edges keep all alternatives and block affected fields. Unruled text,
split tables, unsupported borders and competing candidates remain unresolved.
No drawing-derived item, count, length or identity is read by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any, Mapping

import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _id, _normalise_text, _role_candidates, _sha256
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry
from src.drawing_engine.core.vector_topology import _item_parts

LAYER = "mep_native_declaration_bodies"
SCHEMA_VERSION = "0.1.0"
METHOD_VERSION = "1.0.0"
EDGE_TOLERANCE = 0.75
_TITLE = re.compile(r"\b(?:SCHEDULES?|TABLE)\s*$", re.I)
_CONTINUED = re.compile(r"\b(?:CONTINUED|CONTINUATION|CONT'D)\b", re.I)
_MOUNTING = re.compile(r"\b(?:(?:WALL|FLOOR|SURFACE|CEILING|ROOF)[ -]MOUNTED|UNDERMOUNT)\b", re.I)
_EMPTY = re.compile(r"^(?:-+|N/?A|TBD|TBC)?$", re.I)


def _box(value):
    return isinstance(value, list) and len(value) == 4 and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in value
    ) and value[2] > value[0] and value[3] > value[1]


def _contains(outer, inner, tolerance=EDGE_TOLERANCE):
    return (outer[0] - tolerance <= inner[0] and outer[1] - tolerance <= inner[1]
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def _same_interval(left, right):
    return abs(left[0] - right[0]) <= EDGE_TOLERANCE and abs(left[2] - right[2]) <= EDGE_TOLERANCE


def _perimeter(box, edges):
    """Replay four covered sides from native segments; never fill a large gap."""
    sides = ((0, box[1], box[0], box[2]), (0, box[3], box[0], box[2]),
             (1, box[0], box[1], box[3]), (1, box[2], box[1], box[3]))
    result = []
    for axis, fixed, low, high in sides:
        intervals = []
        for edge in edges:
            start, end = edge["start_pdf"], edge["end_pdf"]
            if abs(start[1-axis] - fixed) > EDGE_TOLERANCE or abs(end[1-axis] - fixed) > EDGE_TOLERANCE:
                continue
            begin, finish = sorted((start[axis], end[axis]))
            if finish >= low - EDGE_TOLERANCE and begin <= high + EDGE_TOLERANCE:
                intervals.append((begin, finish, edge["id"]))
        cursor, refs = low, []
        for begin, finish, ref in sorted(intervals):
            if begin > cursor + EDGE_TOLERANCE:
                break
            if finish > cursor:
                cursor = finish
                refs.append(ref)
            if cursor >= high - EDGE_TOLERANCE:
                break
        result.append({"native_edge_refs": refs, "closed": cursor >= high - EDGE_TOLERANCE})
    return result


def _cell_words(box, words):
    return [word for word in words if _contains(box, word["bbox_pdf"])]


def _text(words):
    lines = defaultdict(list)
    for word in words:
        lines[(word["block_number"], word["line_number"])].append(word)
    ordered = sorted(lines.values(), key=lambda row: (min(w["bbox_pdf"][1] for w in row), min(w["bbox_pdf"][0] for w in row)))
    return "\n".join(" ".join(w["text"] for w in sorted(row, key=lambda w: w["word_number"])) for row in ordered)


def _region(candidate, page):
    box = candidate["bbox_pdf"]
    cells = []
    for cell_box in candidate["cell_bboxes_pdf"]:
        words = _cell_words(cell_box, page["native_words"])
        cells.append({
            "id": _id("mep_declaration_cell", candidate["id"], cell_box),
            "bbox_pdf": cell_box, "bbox_display": list(fitz.Rect(cell_box) * fitz.Matrix(page["pdf_to_display_matrix"])),
            "text": _text(words), "native_word_refs": [word["id"] for word in words],
            "edge_support": _perimeter(cell_box, page["native_edges"]),
        })
    cells.sort(key=lambda c: (c["bbox_pdf"][1], c["bbox_pdf"][0]))
    reasons = []
    perimeter = _perimeter(box, page["native_edges"])
    if not all(side["closed"] for side in perimeter):
        reasons.append("native_outer_boundary_not_closed")
    if any(not all(side["closed"] for side in cell["edge_support"]) for cell in cells):
        reasons.append("native_cell_boundaries_not_closed")
    area = sum(fitz.Rect(cell["bbox_pdf"]).get_area() for cell in cells)
    if abs(area - fitz.Rect(box).get_area()) > max(1, fitz.Rect(box).get_area() * 1e-6):
        reasons.append("cells_do_not_tile_complete_region")
    if any((fitz.Rect(a["bbox_pdf"]) & fitz.Rect(b["bbox_pdf"])).get_area() > 1
           for i, a in enumerate(cells) for b in cells[i+1:]):
        reasons.append("overlapping_cell_interiors")
    titles = [cell for cell in cells if _TITLE.search(_normalise_text(cell["text"]))]
    title = titles[0] if len(titles) == 1 else None
    if title is None or not _same_interval(title["bbox_pdf"], box) or abs(title["bbox_pdf"][1] - box[1]) > EDGE_TOLERANCE:
        reasons.append("unique_enclosed_title_not_established")
    words = [word for word in page["native_words"] if fitz.Rect(word["bbox_pdf"]).intersects(fitz.Rect(box))]
    memberships = Counter(ref for cell in cells for ref in cell["native_word_refs"])
    if any(not _contains(box, word["bbox_pdf"]) for word in words):
        reasons.append("native_text_crosses_outer_body_boundary")
    ambiguous_words = [{
        "native_word_ref": word["id"],
        "candidate_cell_refs": [cell["id"] for cell in cells
                                if fitz.Rect(word["bbox_pdf"]).intersects(fitz.Rect(cell["bbox_pdf"]))],
        "reason": "native_word_crosses_cell_boundary",
    } for word in words if memberships[word["id"]] != 1]
    if title and any(title["id"] in item["candidate_cell_refs"] for item in ambiguous_words):
        # A neighboring header may cross the title boundary while the title's
        # own words remain fully enclosed. Only the latter establishes title
        # ambiguity; all crossed cells still close their field applicability.
        if any(_TITLE.search(word["text"]) for word in words
               if word["id"] in {item["native_word_ref"] for item in ambiguous_words}):
            reasons.append("unique_title_text_ownership_unresolved")
    if any(_CONTINUED.search(word["text"]) for word in words):
        reasons.append("explicit_continuation_requires_wider_body")
    header_bottom = None
    header_cells, body_cells, notes = [], [], []
    if title:
        first_headers = [cell for cell in cells if abs(cell["bbox_pdf"][1] - title["bbox_pdf"][3]) <= EDGE_TOLERANCE]
        if len(first_headers) >= 2:
            header_bottom = max(cell["bbox_pdf"][3] for cell in first_headers)
            header_cells = [cell for cell in cells if cell is not title and cell["bbox_pdf"][3] <= header_bottom + EDGE_TOLERANCE]
            later = [cell for cell in cells if cell["bbox_pdf"][1] >= header_bottom - EDGE_TOLERANCE]
            notes = [cell for cell in later if _same_interval(cell["bbox_pdf"], box)]
            body_cells = [cell for cell in later if cell not in notes]
        if not body_cells or not any(cell["text"].strip() for cell in body_cells):
            reasons.append("data_body_not_established_below_headers")
    if not title:
        reasons.append("header_and_body_scope_unresolved")
    return {
        "id": candidate["id"], "page_ref": page["page_ref"], "page_number": page["page_number"],
        "bbox_pdf": box, "bbox_display": list(fitz.Rect(box) * fitz.Matrix(page["pdf_to_display_matrix"])),
        "title": title["text"] if title else None, "title_cell_ref": title["id"] if title else None,
        "cells": cells, "outer_boundary_support": perimeter,
        "native_word_refs": [word["id"] for word in words],
        "cell_ownership_ambiguities": ambiguous_words,
        "header_cell_refs": [cell["id"] for cell in header_cells],
        "body_cell_refs": [cell["id"] for cell in body_cells], "note_cell_refs": [cell["id"] for cell in notes],
        "state": "accepted" if not reasons else "abstained", "reasons": sorted(set(reasons)),
        "alternatives": [], "bounded_native_body_complete": not reasons,
        "document_declaration_completeness_established": False, "quantity_eligible": False,
    }


def _field_name(header):
    name = _normalise_text(header).upper()
    if name in {"ID", "TAG", "MARK", "PDI #"}:
        return "item_designation"
    if re.fullmatch(r"(?:QTY\.?|QUANTITY)(?:\s*\((?:EA|EACH|NO\.?)\))?", name):
        return "quantity"
    if "MANUFACTURER" in name and "MODEL" in name:
        return "manufacturer_model_statement"
    if name in {"MANUFACTURER", "MFR", "MFR."}:
        return "manufacturer"
    if name in {"MODEL", "MODEL NUMBER", "MODEL NO."}:
        return "model_number"
    if re.search(r"\bBASI[CS] OF DESIGN\b", name):
        return "basis_of_design_statement"
    if "MATERIAL DESCRIPTION" in name or "SPECIFICATION" in name:
        return "specification_statement"
    if "MOUNT" in name:
        return "mounting_statement"
    if "SIZE" in name or "DIMENSION" in name or name in {"HEIGHT", "DIAMETER"}:
        return "dimension_statement"
    return "declared_table_cell"


def _fields(region):
    if region["state"] != "accepted":
        return []
    cells = {cell["id"]: cell for cell in region["cells"]}
    headers = [cells[ref] for ref in region["header_cell_refs"]]
    ambiguous_cells = {ref for item in region["cell_ownership_ambiguities"] for ref in item["candidate_cell_refs"]}
    result = []
    for ref in region["body_cell_refs"]:
        cell = cells[ref]
        box = cell["bbox_pdf"]
        path = sorted([head for head in headers if head["text"].strip()
                       and head["bbox_pdf"][0] - EDGE_TOLERANCE <= box[0]
                       and box[2] <= head["bbox_pdf"][2] + EDGE_TOLERANCE],
                      key=lambda head: head["bbox_pdf"][1])
        unique_header = bool(path) and _same_interval(path[-1]["bbox_pdf"], box)
        raw_value = _normalise_text(cell["text"])
        field_name = _field_name(" ".join(head["text"] for head in path)) if unique_header else "unresolved_header"
        unresolved = [] if unique_header else ["header_applicability_unresolved"]
        value: object = raw_value
        unit = None
        if _EMPTY.fullmatch(raw_value):
            value = None
            unresolved.append("empty_or_unspecified_declared_value")
        elif field_name == "quantity":
            if re.fullmatch(r"\d+(?:\.\d+)?", raw_value) and math.isfinite(float(raw_value)):
                value, unit = float(raw_value), "ea"
            else:
                value = None
                unresolved.append("declared_quantity_not_explicit_numeric_count")
        if ref in ambiguous_cells or any(head["id"] in ambiguous_cells for head in path):
            value = None
            unresolved.append("cell_or_header_native_text_ownership_unresolved")
        result.append({
            "id": _id("mep_native_declared_field", region["id"], ref, field_name),
            "region_ref": region["id"], "page_ref": region["page_ref"], "page_number": region["page_number"],
            "cell_ref": ref, "bbox_display": cell["bbox_display"],
            "row_scope_ref": _id("mep_declaration_row", region["id"], box[1], box[3]),
            "header_path": [head["text"] for head in path], "header_cell_refs": [head["id"] for head in path],
            "field_name": field_name, "value": value, "unit": unit, "raw_text": cell["text"],
            "state": "derived" if not unresolved else "unresolved", "unresolved_facts": unresolved,
            "mounting_statements": sorted(set(match.group(0) for match in _MOUNTING.finditer(raw_value)))
                if not unresolved and field_name in {"basis_of_design_statement", "mounting_statement", "declared_table_cell"} else [],
            "native_word_refs": cell["native_word_refs"], "value_channel": "declared",
            "epistemic_state": "derived", "calculated_value_used": False,
            "physical_item_ref": None, "calculated_quantity": None, "approved_quantity": None,
            "quantity_eligible": False,
        })
    return result


def _assemble(source_pages):
    regions, fields, pages = [], [], []
    for page in source_pages:
        found = [_region(candidate, page) for candidate in page["table_candidates"]]
        eligible = [region for region in found if region["state"] == "accepted"]
        for region in eligible:
            title_refs = set(next(cell for cell in region["cells"] if cell["id"] == region["title_cell_ref"])["native_word_refs"])
            alternatives = [other["id"] for other in eligible if other is not region and title_refs.intersection(
                next(cell for cell in other["cells"] if cell["id"] == other["title_cell_ref"])["native_word_refs"])]
            if alternatives:
                region.update(state="abstained", reasons=["nonunique_complete_body_for_title"],
                              alternatives=alternatives, bounded_native_body_complete=False)
        covered_words = {ref for region in found if region["state"] == "accepted" for ref in region["native_word_refs"]}
        unresolved_headings = [heading for heading in page["heading_observations"] if not set(heading["native_word_refs"]) <= covered_words]
        pages.append({
            "page_ref": page["page_ref"], "page_number": page["page_number"],
            "region_refs": [region["id"] for region in found],
            "accepted_body_count": sum(region["state"] == "accepted" for region in found),
            "unresolved_headings": [dict(heading, reason="native_bounded_body_not_established") for heading in unresolved_headings],
            "native_table_scan_state": page["native_table_scan_state"],
            "document_declaration_completeness_established": False, "document_field_absence_established": False,
            "ocr_run": False, "quantity_eligible": False,
        })
        regions.extend(found)
        fields.extend(field for region in found for field in _fields(region))
    summary = {
        "page_count": len(pages), "table_candidate_count": len(regions),
        "accepted_bounded_body_count": sum(region["state"] == "accepted" for region in regions),
        "declared_cell_count": len(fields),
        "resolved_declared_quantity_count": sum(field["field_name"] == "quantity" and field["state"] == "derived" for field in fields),
        "field_counts": dict(sorted(Counter(field["field_name"] for field in fields).items())),
        "unresolved_heading_count": sum(len(page["unresolved_headings"]) for page in pages),
    }
    return regions, fields, pages, summary


def extract_mep_native_declaration_bodies(*, pdf_path: Path | str, sheet_registry: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_mep_sheet_registry(sheet_registry)
    if errors:
        raise ValueError("invalid M1 registry: " + "; ".join(errors))
    source_hash = _file_sha256(Path(pdf_path))
    if source_hash != sheet_registry["document"].get("source_pdf_sha256"):
        raise ValueError("source PDF does not match frozen M1 registry")
    scopes = {page["page_number"]: page for page in sheet_registry["pages"]}
    source_pages = []
    with fitz.open(pdf_path) as pdf:
        if sorted(scopes) != list(range(1, len(pdf) + 1)):
            raise ValueError("M1/source page coverage mismatch")
        for number, page in enumerate(pdf, 1):
            page_ref = scopes[number]["page_ref"]
            words = []
            for word in page.get_text("words"):
                native_ref = f"page[{number}].text_block[{word[5]}].line[{word[6]}].word[{word[7]}]"
                words.append({
                    "id": _id("mep_declaration_native_word", source_hash, native_ref),
                    "source_native_ref": native_ref, "text": word[4], "bbox_pdf": list(word[:4]),
                    "bbox_display": list(fitz.Rect(word[:4]) * page.rotation_matrix),
                    "block_number": word[5], "line_number": word[6], "word_number": word[7],
                })
            lines = defaultdict(list)
            for word in words:
                lines[(word["block_number"], word["line_number"])].append(word)
            headings = [{"text": _text(row), "native_word_refs": [word["id"] for word in row],
                         "role_candidates": _role_candidates(_text(row))}
                        for row in lines.values() if _role_candidates(_text(row))]
            candidates, edges = [], []
            scan = any(_TITLE.search(_normalise_text(heading["text"])) for heading in headings)
            if scan:
                drawings = page.get_drawings()
                for drawing_index, drawing in enumerate(drawings):
                    if drawing.get("type") not in {"s", "fs"}:
                        continue
                    for item_index, item in enumerate(drawing["items"]):
                        for part_index, part in enumerate(_item_parts(item)):
                            if part["kind"] != "line":
                                continue
                            native_ref = f"drawing[{drawing_index}].item[{item_index}].segment[{part_index}]"
                            edges.append({"id": _id("mep_declaration_native_edge", page_ref, native_ref),
                                          "source_native_ref": native_ref,
                                          "start_pdf": list(part["start"]), "end_pdf": list(part["end"])})
                tables = page.find_tables(paths=drawings, clip=page.rect * page.derotation_matrix,
                                          strategy="lines_strict", snap_tolerance=0.5,
                                          join_tolerance=0.5, intersection_tolerance=0.5)
                for table in tables.tables:
                    boxes = sorted({tuple(box) for row in table.rows for box in row.cells if box})
                    candidates.append({"id": _id("mep_native_declaration_region", page_ref, table.bbox, boxes),
                                       "bbox_pdf": list(table.bbox), "cell_bboxes_pdf": [list(box) for box in boxes]})
            source_pages.append({
                "page_ref": page_ref, "page_number": number, "pdf_to_display_matrix": list(page.rotation_matrix),
                "native_words": words, "native_edges": edges, "heading_observations": headings,
                "table_candidates": candidates,
                "native_table_scan_state": "native_ruled_candidates_extracted" if scan else "no_native_schedule_title_gate",
            })
    regions, fields, pages, summary = _assemble(source_pages)
    return {
        "schema_version": SCHEMA_VERSION, "layer": LAYER,
        "document": deepcopy(sheet_registry["document"]), "m1_payload_sha256": _sha256(sheet_registry),
        "method": {"name": "native_ruled_declaration_body_certificate", "version": METHOD_VERSION,
                   "pymupdf_version": fitz.VersionBind, "edge_tolerance_points": EDGE_TOLERANCE},
        "source_pages": source_pages, "regions": regions, "declared_fields": fields, "pages": pages, "summary": summary,
        "exchange_contract": {"declared_channel_only": True, "bounded_native_grid_body_only": True,
                              "calculated_records_used": False, "physical_identity_established": False,
                              "document_declaration_completeness_established": False, "reconciliation_emitted": False,
                              "quantity_eligible": False},
    }


def validate_mep_native_declaration_bodies(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("layer") != LAYER:
        return ["native declaration body schema/layer mismatch"]
    pages = payload.get("source_pages", [])
    if [page.get("page_number") for page in pages] != list(range(1, payload["document"]["page_count"] + 1)):
        errors.append("source page coverage mismatch")
    for page in pages:
        word_refs = [word.get("id") for word in page["native_words"]]
        if len(word_refs) != len(set(word_refs)) or any(not _box(word.get("bbox_pdf")) for word in page["native_words"]):
            errors.append("invalid native text provenance")
        if any(not _box(candidate.get("bbox_pdf")) or any(not _box(box) for box in candidate["cell_bboxes_pdf"])
               for candidate in page["table_candidates"]):
            errors.append("invalid table candidate geometry")
    if errors:
        return errors
    replay = _assemble(pages)
    for key, expected in zip(("regions", "declared_fields", "pages", "summary"), replay):
        if payload.get(key) != expected:
            errors.append(f"{key} does not replay from native evidence")
    contract = payload.get("exchange_contract", {})
    for key in ("declared_channel_only", "bounded_native_grid_body_only"):
        if contract.get(key) is not True:
            errors.append(f"{key} must remain true")
    for key in ("calculated_records_used", "physical_identity_established", "document_declaration_completeness_established",
                "reconciliation_emitted", "quantity_eligible"):
        if contract.get(key) is not False:
            errors.append(f"{key} must remain false")
    return errors
