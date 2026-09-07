"""Object-agnostic drawing observations, claims, and engineering hypotheses.

This stage deliberately stops before choosing a column, beam, stair, wall, or
slab template.  It records generic views, contours, dimensions, token roles,
and their evidence.  A specialised 3D solver may consume the result, but a
failed specialised solver never invalidates the object-agnostic record.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable, Iterable

import fitz

from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader
from src.drawing_engine.core.canonical_knowledge_graph import build_canonical_knowledge_graph
from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attachment_payload, propose_dimensions
from src.drawing_engine.core.dimension_ownership import (
    adjudicate_dimension_proposals,
    reclose_title_scope_dimensions,
    resolve_dimension_ownership,
)
from src.drawing_engine.core.exchange_records import build_exchange_records
from src.drawing_engine.disciplines.rebar.estimation_profile import estimate_rebar_quantities, load_estimation_profile
from src.drawing_engine.core.metric_equation_graph import build_metric_equation_graph
from src.drawing_engine.core.vector_topology import composite_topologies, extract_page_topology, topology_for_drawing
from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import solve_generic_profile_extrusion
from src.drawing_engine.disciplines.concrete.generic_prismatic_solver import solve_generic_prismatic
from src.drawing_engine.core.native_vector_detail_linking import extract_native_vector_details
from src.drawing_engine.disciplines.detail.native_detail_takeoff import solve_native_detail_takeoff
from src.drawing_engine.disciplines.rebar.physical_bar_family_solver import (
    build_family_constrained_scene,
    resolve_physical_bar_families,
    summarize_group_quantities,
)
from src.drawing_engine.core.object_instance_assembly import (
    assemble_object_instances,
    assemble_relation_certified_object_instances,
    scope_rebar_path_graph,
)
from src.drawing_engine.disciplines.concrete.object_instance_extrusion_solver import solve_object_instance_extrusions
from src.drawing_engine.disciplines.rebar.rebar_program import build_rebar_program
from src.drawing_engine.disciplines.rebar.rebar_path_graph import enrich_projection_identities
from src.drawing_engine.core.raster_fallback import (
    assess_native_page_quality,
    augment_observation_graph_with_raster,
    reconstruct_raster_observations,
)
from src.drawing_engine.disciplines.concrete.semantic_3d_solver import solve_semantic_3d
from src.drawing_engine.disciplines.rebar.section_rebar_extraction import extract_section_rebar, _resolve_unique_occurrence_targets
from src.drawing_engine.disciplines.concrete.scoped_profile_assembly import assemble_profiles_by_view_scope
from src.drawing_engine.disciplines.concrete.scope_local_profile_reclosure import reclose_profiles_in_title_scopes
from src.drawing_engine.disciplines.concrete.profile_physical_scope_binding import bind_profiles_to_physical_scopes
from src.drawing_engine.disciplines.concrete.physical_component_hypothesis import generate_physical_component_hypotheses
from src.drawing_engine.disciplines.concrete.physical_component_reclosure import reclose_physical_component_hypotheses
from src.drawing_engine.disciplines.concrete.single_prism_placement_evidence import derive_single_prism_placement_evidence
from src.drawing_engine.disciplines.concrete.physical_component_step4_replay import replay_single_prism_step4
from src.drawing_engine.disciplines.concrete.clear_span_prism_reconstruction import reconstruct_clear_span_prism
from src.drawing_engine.disciplines.concrete.banded_plan_sweep_evidence import derive_banded_plan_sweep_evidence
from src.drawing_engine.disciplines.concrete.open_structural_boundary_assembly import assemble_open_structural_boundaries
from src.drawing_engine.disciplines.concrete.flight_interface_closure import certify_flight_interface_closure
from src.drawing_engine.disciplines.concrete.flight_profile_pair_certificate import certify_flight_profile_pair
from src.drawing_engine.disciplines.concrete.landing_component_reconstruction import reconstruct_landing_component
from src.drawing_engine.disciplines.concrete.terminal_landing_support import (
    materialize_terminal_landing_context,
    reconstruct_terminal_landing_support,
)
from src.drawing_engine.disciplines.concrete.same_object_constructive_assembly import assemble_same_object_construction
from src.drawing_engine.disciplines.concrete.physical_object_quantity_promotion import (
    promote_constructive_union_physical_object,
)
from src.drawing_engine.core.view_frame_inference import infer_view_frames
from src.drawing_engine.core.view_segmentation import segment_views_by_titles
from src.drawing_engine.core.semantic_region_grouping import (
    PrimitiveGraph,
    augment_graph_with_text_roles,
    build_primitive_graph,
    propose_semantic_regions,
)


SECTION_SEARCH_RE = re.compile(r"(?<!\d)(?P<label>(?P<a>[1-9]\d*|[A-ZА-Я])\s*[-–—]\s*(?P=a))(?!\d)", re.I)
SPACING_RE = re.compile(r"\d{1,5}\s*[xх×]\s*\d{1,4}\s*=\s*\d{1,6}", re.I)
STEP_SPACING_RE = re.compile(r"(?:шаг|spacing|step|@)\s*=?\s*\d{1,5}", re.I)
REBAR_CALLOUT_RE = re.compile(
    r"^(?:[A-ZА-Я]{1,3})?(?P<diameter>\d{1,2})/(?P<code>\d{1,4})$",
    re.I,
)
VIEW_KEYWORDS = {
    "reinforcement_view_candidate": ("армир", "reinforcement"),
    "formwork_view_candidate": ("опалуб", "formwork"),
    "section_view_candidate": ("разрез", "section"),
    "plan_view_candidate": ("план", "plan"),
}


def _rect_distance(left: fitz.Rect, right: fitz.Rect) -> float:
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return math.hypot(dx, dy)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, fitz.Rect):
        return list(value)
    if isinstance(value, fitz.Point):
        return [value.x, value.y]
    return value


def _transform_preview_mesh(
    mesh: dict[str, Any], transform: dict[str, Any]
) -> dict[str, Any] | None:
    """Place a preview mesh with an already certified physical transform."""

    vertices = mesh.get("vertices_xyz_mm") or []
    triangles = mesh.get("triangles") or []
    origin = list(map(float, transform.get("origin_xyz_mm", []) or []))
    axes = transform.get("local_axes_xyz") or {}
    basis = [list(map(float, axes.get(axis, []) or [])) for axis in "xyz"]
    if not vertices or not triangles or len(origin) != 3 or any(len(axis) != 3 for axis in basis):
        return None
    placed = []
    for point in vertices:
        local = list(map(float, point))
        if len(local) != 3:
            return None
        placed.append(
            [
                origin[row] + sum(local[column] * basis[column][row] for column in range(3))
                for row in range(3)
            ]
        )
    return {
        "vertices_xyz_mm": placed,
        "triangles": deepcopy(triangles),
        "source_local_mesh_validation": deepcopy(mesh.get("validation")),
        "physical_transform": deepcopy(transform),
    }


def _line_segments(page: fitz.Page) -> list[tuple[fitz.Point, fitz.Point]]:
    segments = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                segments.append((item[1], item[2]))
    return segments


def _table_context(box: fitz.Rect, segments: list[tuple[fitz.Point, fitz.Point]]) -> bool:
    horizontal = 0
    vertical = 0
    neighborhood = box + (-35, -18, 35, 18)
    for start, end in segments:
        segment_box = fitz.Rect(start, end).normalize()
        if not neighborhood.intersects(segment_box + (-0.5, -0.5, 0.5, 0.5)):
            continue
        dx, dy = abs(end.x - start.x), abs(end.y - start.y)
        if dy <= 0.7 and dx >= max(12.0, box.width * 1.5):
            horizontal += 1
        elif dx <= 0.7 and dy >= max(8.0, box.height * 1.5):
            vertical += 1
    return horizontal >= 2 and vertical >= 2


def _dimension_match(box: fitz.Rect, text: str, dimensions: Iterable[DimensionAttachment]) -> DimensionAttachment | None:
    matches = []
    normalized = text.strip().replace(",", ".")
    for dimension in dimensions:
        if normalized != dimension.text.strip().replace(",", "."):
            continue
        candidate = fitz.Rect(dimension.text_bbox)
        if _rect_distance(box, candidate) <= 1.5:
            matches.append(dimension)
    if not matches:
        return None
    return max(matches, key=lambda item: (item.status == "accepted", item.score))


def _pair_compound_callout_roles(roles: list[dict[str, Any]]) -> None:
    compound_marks = [
        item
        for item in roles
        if item.get("basis") in {
            "compound_rebar_callout_syntax_outside_table_grid",
            "outlined_text_ocr_compound_rebar_callout_with_leader",
        }
    ]
    for upper in compound_marks:
        upper_box = fitz.Rect(upper["bbox_display"])
        candidates = []
        for lower in roles:
            if lower is upper or lower.get("resolved_role") == "schedule_or_table_value":
                continue
            match = REBAR_CALLOUT_RE.fullmatch(str(lower.get("text", "")))
            if match is None:
                continue
            lower_box = fitz.Rect(lower["bbox_display"])
            vertical_gap = lower_box.y0 - upper_box.y1
            if not (-0.25 * upper_box.height <= vertical_gap <= 0.80 * upper_box.height):
                continue
            if abs((lower_box.x0 + lower_box.x1) / 2 - (upper_box.x0 + upper_box.x1) / 2) > max(upper_box.width, lower_box.width) * 0.55:
                continue
            candidates.append((abs(vertical_gap), lower, match))
        if not candidates:
            continue
        _, lower, match = min(candidates, key=lambda item: (item[0], item[1]["id"]))
        if lower.get("basis") == "paired_second_line_in_compound_rebar_callout":
            continue
        count_spacing = {
            "count": int(match.group("diameter")),
            "spacing_mm": int(match.group("code")),
            "text_role_id": lower["id"],
        }
        lower.update(
            {
                "resolved_role": "rebar_count_spacing",
                "confidence": 0.91,
                "epistemic_state": "direct" if lower.get("text_method") != "geometry_gated_ocr" else "observed",
                "basis": "paired_second_line_in_compound_rebar_callout",
                "semantic_value": count_spacing,
            }
        )
        upper["paired_count_spacing"] = count_spacing


def resolve_text_roles(page: fitz.Page, dimensions: tuple[DimensionAttachment, ...]) -> list[dict[str, Any]]:
    """Assign one primary role per native text token, retaining alternatives."""

    words = page.get_text("words")
    text_lines = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(span.get("text", "") for span in spans).strip()
            rect = fitz.Rect(spans[0]["bbox"])
            for span in spans[1:]:
                rect |= fitz.Rect(span["bbox"])
            text_lines.append((text, rect))

    spacing_boxes = [rect for text, rect in text_lines if SPACING_RE.search(text)]
    segments = _line_segments(page)
    thin_segments = extract_thin_segments(page)
    target_boxes = [
        fitz.Rect(drawing["rect"])
        for drawing in page.get_drawings()
        if float(drawing.get("width") or 0) >= 1.0 or drawing.get("fill") is not None
    ]
    resolved = []
    for index, word in enumerate(words, start=1):
        text = str(word[4]).strip()
        if not text:
            continue
        box = fitz.Rect(word[:4])
        alternatives: list[dict[str, Any]] = []
        dimension = _dimension_match(box, text, dimensions)
        if dimension is not None:
            alternatives.append(
                {
                    "role": "dimension" if dimension.status == "accepted" else "dimension_candidate",
                    "confidence": 0.99 if dimension.status == "accepted" else 0.82,
                    "basis": f"{dimension.status}_dimension_attachment",
                    "evidence_refs": [dimension.attachment_id],
                }
            )
        if any(candidate.intersects(box) for candidate in spacing_boxes):
            alternatives.append(
                {"role": "spacing_expression_token", "confidence": 0.99, "basis": "spacing_arithmetic_context", "evidence_refs": []}
            )
        section_match = SECTION_SEARCH_RE.search(text)
        if section_match:
            alternatives.append(
                {"role": "section_label", "confidence": 0.98, "basis": "repeated_section_identifier", "evidence_refs": []}
            )
        folded = text.casefold()
        for role, keywords in VIEW_KEYWORDS.items():
            if any(keyword in folded for keyword in keywords):
                alternatives.append({"role": role, "confidence": 0.94, "basis": "view_label_lexeme", "evidence_refs": []})
        is_number = text.replace(",", ".").replace(".", "", 1).isdigit()
        table_context = _table_context(box, segments)
        rebar_callout = REBAR_CALLOUT_RE.fullmatch(text)
        plausible_diameter = (
            rebar_callout is not None
            and 4 <= int(rebar_callout.group("diameter")) <= 40
        )
        if (is_number or plausible_diameter) and table_context:
            alternatives.append({"role": "schedule_or_table_value", "confidence": 0.93, "basis": "orthogonal_grid_context", "evidence_refs": []})
        if plausible_diameter and not table_context:
            alternatives.append(
                {
                    "role": "identifier_candidate",
                    "confidence": 0.88,
                    "basis": "compound_rebar_callout_syntax_outside_table_grid",
                    "evidence_refs": [],
                    "semantic_value": {
                        "diameter_mm": int(rebar_callout.group("diameter")),
                        "length_code": int(rebar_callout.group("code")),
                    },
                }
            )
        if is_number and not alternatives:
            trace = trace_leader(box, thin_segments, search_radius=120, max_hops=8)
            terminals = [fitz.Point(*terminal) for terminal in trace["terminals"]]
            hits_target = any(point in target + (-3, -3, 3, 3) for point in terminals for target in target_boxes)
            if hits_target:
                alternatives.append(
                    {
                        "role": "identifier_candidate",
                        "confidence": 0.64,
                        "basis": "leader_terminal_on_heavy_or_filled_path",
                        "evidence_refs": [segment["drawing_ref"] for segment in trace["segments"]],
                    }
                )
        if not alternatives:
            alternatives.append(
                {
                    "role": "unclassified_number" if is_number else "unclassified_text",
                    "confidence": 0.35,
                    "basis": "no_exclusive_semantic_context",
                    "evidence_refs": [],
                }
            )

        priority = {
            "spacing_expression_token": 7,
            "dimension": 6,
            "dimension_candidate": 6,
            "section_label": 5,
            "schedule_or_table_value": 4,
            "reinforcement_view_candidate": 3,
            "formwork_view_candidate": 3,
            "section_view_candidate": 3,
            "plan_view_candidate": 3,
            "identifier_candidate": 2,
            "unclassified_number": 1,
            "unclassified_text": 0,
        }
        alternatives.sort(key=lambda item: (priority.get(item["role"], 0), item["confidence"]), reverse=True)
        primary = alternatives[0]
        resolved.append(
            {
                "id": f"text_role.{index:05d}",
                "text": text,
                "bbox_display": list(box),
                "resolved_role": primary["role"],
                "confidence": primary["confidence"],
                "epistemic_state": "direct" if primary["role"] in {"dimension", "spacing_expression_token", "section_label"} else "inferred",
                "basis": primary["basis"],
                "alternatives": alternatives[1:],
                "primitive_refs": [f"word[{index - 1}]", *primary["evidence_refs"]],
                **({"semantic_value": primary["semantic_value"]} if primary.get("semantic_value") else {}),
            }
        )
    _pair_compound_callout_roles(resolved)
    # Section identifiers are frequently split into several PDF words (for
    # example ``5``, ``-``, ``5``).  Add one line token rather than assigning
    # contradictory roles to its constituent words.
    for line_index, (text, box) in enumerate(text_lines, start=1):
        match = SECTION_SEARCH_RE.search(text)
        if not match:
            continue
        resolved.append(
            {
                "id": f"text_line_role.{line_index:05d}",
                "text": match.group("label"),
                "bbox_display": list(box),
                "resolved_role": "section_label",
                "confidence": 0.98,
                "epistemic_state": "direct",
                "basis": "repeated_section_identifier_across_native_spans",
                "alternatives": [],
                "primitive_refs": [f"text_line[{line_index - 1}]"],
            }
        )
    if _native_text_is_sparse(page):
        resolved.extend(_outlined_text_roles(page, dimensions, len(resolved)))
        _pair_compound_callout_roles(resolved)
    return resolved


def _outlined_text_roles(
    page: fitz.Page,
    dimensions: tuple[DimensionAttachment, ...],
    id_offset: int,
    clips: Iterable[fitz.Rect] | None = None,
    scale: int = 3,
    source_prefix: str = "page",
) -> list[dict[str, Any]]:
    """Recover semantic labels and mark callouts from outlined text.

    OCR is deliberately restricted to text-poor pages.  Numeric tokens become
    marks only when a native leader reaches heavy/filled drawing geometry;
    section labels and view keywords remain explicit OCR observations.
    """

    import numpy as np
    import pytesseract
    from PIL import Image

    passes = (
        ("rus+eng", "--psm 11 -l rus+eng"),
        ("numeric", "--psm 11 -l eng -c tessedit_char_whitelist=0123456789-/"),
    )
    raw = []
    clip_rows = [page.rect] if clips is None else [fitz.Rect(item) & page.rect for item in clips]
    for clip_index, clip in enumerate(clip_rows):
        if clip.is_empty or clip.width < 8 or clip.height < 8:
            continue
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            clip=clip,
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        image = Image.fromarray(
            np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
        )
        for pass_name, config in passes:
            try:
                data = pytesseract.image_to_data(
                    image,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                    timeout=60,
                )
            except (OSError, RuntimeError, pytesseract.TesseractError):
                continue
            line_tokens: defaultdict[tuple[int, int, int], list[tuple[str, float, fitz.Rect]]] = defaultdict(list)
            for index, value in enumerate(data["text"]):
                text = str(value).strip()
                confidence = float(data["conf"][index])
                if not text or confidence < 15:
                    continue
                box = fitz.Rect(
                    clip.x0 + float(data["left"][index]) / scale,
                    clip.y0 + float(data["top"][index]) / scale,
                    clip.x0 + float(data["left"][index] + data["width"][index]) / scale,
                    clip.y0 + float(data["top"][index] + data["height"][index]) / scale,
                )
                raw.append((f"{source_prefix}.{clip_index}.{pass_name}", text, confidence / 100.0, box))
                line_key = (
                    int(data.get("block_num", [0] * len(data["text"]))[index]),
                    int(data.get("par_num", [0] * len(data["text"]))[index]),
                    int(data.get("line_num", [0] * len(data["text"]))[index]),
                )
                line_tokens[line_key].append((text, confidence / 100.0, box))
            if pass_name == "rus+eng":
                for tokens in line_tokens.values():
                    if len(tokens) < 2:
                        continue
                    tokens.sort(key=lambda item: item[2].x0)
                    line_box = fitz.Rect(tokens[0][2])
                    for _, _, token_box in tokens[1:]:
                        line_box |= token_box
                    raw.append(
                        (
                            f"{source_prefix}.{clip_index}.line",
                            " ".join(item[0] for item in tokens),
                            sum(item[1] for item in tokens) / len(tokens),
                            line_box,
                        )
                    )

    segments = _line_segments(page)
    thin_segments = extract_thin_segments(page)
    target_boxes = [
        fitz.Rect(drawing["rect"])
        for drawing in page.get_drawings()
        if float(drawing.get("width") or 0) >= 1.0 or drawing.get("fill") is not None
    ]
    output = []
    seen: list[tuple[str, fitz.Rect]] = []
    for pass_name, text, confidence, box in sorted(raw, key=lambda row: (-row[2], row[3].y0, row[3].x0)):
        normalized = text.replace("–", "-").replace("—", "-").strip(" .,:;")
        folded = normalized.casefold()
        role = None
        basis = None
        evidence_refs: list[str] = []
        section_match = SECTION_SEARCH_RE.fullmatch(normalized)
        if STEP_SPACING_RE.search(normalized):
            role = "spacing_expression_token"
            basis = "outlined_text_ocr_mark_plus_spacing_annotation"
        elif section_match:
            role = "section_label"
            basis = "outlined_text_ocr_repeated_section_identifier"
        else:
            for candidate_role, keywords in VIEW_KEYWORDS.items():
                if any(keyword in folded for keyword in keywords):
                    role = candidate_role
                    basis = "outlined_text_ocr_view_label_lexeme"
                    break
        is_mark = normalized.isdigit() and 1 <= len(normalized) <= 2
        callout_match = REBAR_CALLOUT_RE.fullmatch(normalized)
        is_callout = (
            callout_match is not None
            and 4 <= int(callout_match.group("diameter")) <= 40
        )
        if role is None and is_callout:
            if _table_context(box, segments):
                continue
            trace = trace_leader(
                box,
                thin_segments,
                search_radius=150,
                max_hops=10,
                continue_through_intersections=True,
            )
            terminals = [fitz.Point(*terminal) for terminal in trace["terminals"]]
            if any(point in target + (-4, -4, 4, 4) for point in terminals for target in target_boxes):
                role = "identifier_candidate"
                basis = "outlined_text_ocr_compound_rebar_callout_with_leader"
                evidence_refs = [segment["drawing_ref"] for segment in trace["segments"]]
        if role is None and callout_match is not None and not _table_context(box, segments):
            role = "unclassified_text"
            basis = "outlined_text_ocr_compound_token_candidate"
        if role is None and is_mark:
            if any(fitz.Rect(item.text_bbox) + (-2, -2, 2, 2) & box for item in dimensions):
                continue
            if _table_context(box, segments):
                continue
            trace = trace_leader(
                box,
                thin_segments,
                search_radius=150,
                max_hops=10,
                continue_through_intersections=True,
            )
            terminals = [fitz.Point(*terminal) for terminal in trace["terminals"]]
            if any(point in target + (-4, -4, 4, 4) for point in terminals for target in target_boxes):
                role = "identifier_candidate"
                basis = "outlined_text_ocr_leader_terminal_on_heavy_or_filled_path"
                evidence_refs = [segment["drawing_ref"] for segment in trace["segments"]]
            elif trace["segments"]:
                # Preserve a possible cutting-plane endpoint without calling
                # it a bar mark. Acceptance still requires an identical peer,
                # aligned native end stems, a unique parent, and metric gates.
                role = "cutting_plane_endpoint_candidate"
                basis = "outlined_text_ocr_single_token_with_native_annotation_trace"
                evidence_refs = [segment["drawing_ref"] for segment in trace["segments"]]
        if role is None:
            continue
        if any(
            previous_text == normalized
            and box.intersects(previous_box)
            and abs(box.get_area() - previous_box.get_area()) <= 0.35 * max(box.get_area(), previous_box.get_area(), 1)
            for previous_text, previous_box in seen
        ):
            continue
        seen.append((normalized, box))
        output.append(
            {
                "id": f"ocr_text_role.{id_offset + len(output) + 1:05d}",
                "text": normalized,
                "bbox_display": list(box),
                "resolved_role": role,
                "confidence": round(confidence, 3),
                "epistemic_state": "observed",
                "basis": basis,
                "alternatives": [],
                "primitive_refs": [f"ocr[{pass_name}].token[{len(output)}]", *evidence_refs],
                "text_method": "geometry_gated_ocr",
                "ocr_pass": pass_name,
                **(
                    {
                        "semantic_value": {
                            "diameter_mm": int(callout_match.group("diameter")),
                            "length_code": int(callout_match.group("code")),
                        }
                    }
                    if role == "identifier_candidate" and callout_match is not None
                    else {}
                ),
            }
        )
    return sorted(output, key=lambda item: (item["bbox_display"][1], item["bbox_display"][0], item["id"]))


def _native_text_is_sparse(page: fitz.Page) -> bool:
    words = page.get_text("words")
    area = sum(fitz.Rect(word[:4]).get_area() for word in words)
    return len(words) <= 32 and area <= 0.02 * page.rect.get_area()


def _ocr_view_clips(page: fitz.Page, views: list[dict[str, Any]]) -> list[fitz.Rect]:
    page_area = page.rect.get_area()
    semantic_candidates = []
    for view in views:
        box = fitz.Rect(view["bbox_display"])
        if box.width < 36 or box.height < 36 or not 0.002 * page_area <= box.get_area() <= 0.45 * page_area:
            continue
        role = view.get("role_hypothesis")
        if role in {
            "reinforcement_view_candidate",
            "section_view_candidate",
            "plan_view_candidate",
            "formwork_view_candidate",
        }:
            semantic_candidates.append(box + (-8, -8, 8, 8))
            continue
        features = view.get("features", {}) or {}
        aspect = max(box.width / max(box.height, 0.1), box.height / max(box.width, 0.1))
        if (
            role == "drawing_view_candidate"
            and aspect >= 2.2
            and int(features.get("primitive_count", 0)) >= 20
            and float(features.get("orthogonal_axis_ratio", 0.0)) >= 0.55
            and int(features.get("dimension_proposals", 0)) >= 1
        ):
            # Compact transverse views can be geometry-complete before their
            # outlined title is read. Include the nearby annotation band so
            # OCR can recover that title without using page coordinates.
            semantic_candidates.append(
                box
                + (
                    -max(12.0, 0.35 * box.width),
                    -max(12.0, 0.32 * box.height),
                    max(12.0, 0.35 * box.width),
                    max(8.0, 0.08 * box.height),
                )
            )
    if not semantic_candidates:
        return []
    semantic_candidates.sort(key=lambda box: box.get_area(), reverse=True)
    return semantic_candidates[:8]


def _native_glyph_signature(drawing: dict[str, Any]) -> tuple[Any, ...] | None:
    """Translation-tolerant signature for one outlined native text glyph."""

    rect = fitz.Rect(drawing["rect"])
    if not 1.0 <= rect.width <= 20.0 or not 3.0 <= rect.height <= 25.0:
        return None

    def point(value: Any) -> tuple[float, float] | None:
        if not isinstance(value, fitz.Point):
            return None
        # Repeated CAD glyphs can differ by one device-grid increment after
        # PDF transforms. Whole-point quantisation preserves the distinctive
        # path topology while absorbing that sub-point export jitter.
        return (round(value.x - rect.x0), round(value.y - rect.y0))

    items = []
    for item in drawing.get("items", []):
        coordinates = tuple(candidate for value in item[1:] if (candidate := point(value)) is not None)
        if not coordinates:
            return None
        items.append((str(item[0]), coordinates))
    if not items:
        return None
    return (
        round(rect.width),
        round(rect.height),
        round(float(drawing.get("width") or 0.0) * 4) / 4,
        tuple(items),
    )


def _native_section_marker_roles(
    page: fitz.Page,
    text_roles: list[dict[str, Any]],
    id_offset: int,
) -> list[dict[str, Any]]:
    """Match endpoint digits to the same native glyph in section titles.

    Outlined CAD text often defeats broad OCR. A marker is still only an
    observation here: relation acceptance later requires two aligned matches,
    native end stems, title ownership, unique parent, metric agreement, and
    profile/section reprojection.
    """

    drawings = page.get_drawings()
    indexed = [
        (index, drawing, _native_glyph_signature(drawing))
        for index, drawing in enumerate(drawings)
    ]
    output = []
    seen: set[tuple[str, int]] = set()
    for role in text_roles:
        if role.get("resolved_role") != "section_label":
            continue
        match = SECTION_SEARCH_RE.fullmatch(str(role.get("text") or "").strip())
        if match is None:
            continue
        label = match.group("a").upper()
        title_box = fitz.Rect(role["bbox_display"])
        local = [
            (index, drawing, signature)
            for index, drawing, signature in indexed
            if signature is not None
            and title_box.contains(fitz.Rect(drawing["rect"]).tl)
            and title_box.contains(fitz.Rect(drawing["rect"]).br)
        ]
        signature_counts = Counter(signature for _, _, signature in local)
        title_signature = next(
            (
                signature
                for signature, count in sorted(
                    signature_counts.items(),
                    key=lambda item: (-item[1], str(item[0])),
                )
                if count >= 2
            ),
            None,
        )
        if title_signature is None:
            continue
        title_refs = [f"drawing[{index}]" for index, _, signature in local if signature == title_signature]
        for drawing_index, drawing, signature in indexed:
            if signature != title_signature or (label, drawing_index) in seen:
                continue
            box = fitz.Rect(drawing["rect"])
            if any(box.intersects(fitz.Rect(item["bbox_display"])) for item in text_roles if item.get("resolved_role") == "section_label"):
                continue
            seen.add((label, drawing_index))
            output.append(
                {
                    "id": f"native_marker_role.{id_offset + len(output) + 1:05d}",
                    "text": label,
                    "bbox_display": list(box),
                    "resolved_role": "cutting_plane_endpoint_candidate",
                    "confidence": 0.99,
                    "epistemic_state": "direct",
                    "basis": "native_vector_glyph_identity_with_repeated_section_title_digit",
                    "alternatives": [],
                    "primitive_refs": [f"drawing[{drawing_index}]", *title_refs, str(role["id"])],
                    "text_method": "native_vector_glyph_correspondence",
                }
            )
    return sorted(output, key=lambda item: (item["bbox_display"][1], item["bbox_display"][0], item["id"]))


def _merge_text_role_observations(
    existing: list[dict[str, Any]], additions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = list(existing)
    for addition in additions:
        box = fitz.Rect(addition["bbox_display"])
        normalized = "".join(str(addition["text"]).split()).casefold()
        duplicate = False
        for current in merged:
            current_box = fitz.Rect(current["bbox_display"])
            current_normalized = "".join(str(current["text"]).split()).casefold()
            if normalized != current_normalized:
                continue
            intersection = box & current_box
            if intersection.get_area() >= 0.45 * min(box.get_area(), current_box.get_area()):
                duplicate = True
                break
        if not duplicate:
            merged.append(addition)
    _pair_compound_callout_roles(merged)
    return merged


def extract_contours(
    page: fitz.Page,
    native_topology: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract generic native-vector closed-loop and open-profile candidates."""

    native_topology = native_topology or extract_page_topology(page)
    contours = []
    page_area = page.rect.get_area()
    drawings = page.get_drawings()
    for drawing_index, drawing in enumerate(drawings):
        rect = fitz.Rect(drawing["rect"])
        if rect.width < 2 or rect.height < 2 or rect.get_area() > 0.65 * page_area:
            continue
        topology = topology_for_drawing(native_topology, f"drawing[{drawing_index}]")
        closed = topology["closed"]
        if not closed and topology["segment_count"] < 2:
            continue
        contour_id = f"contour.{len(contours) + 1:05d}"
        segments = [
            {
                "id": segment["id"],
                "kind": segment["kind"],
                "start_display": segment["start_display"],
                "end_display": segment["end_display"],
                "control_points_display": segment["control_points_display"],
                "sample_points_display": segment.get("sample_points_display", []),
                "start_vertex_id": segment["start_vertex_id"],
                "end_vertex_id": segment["end_vertex_id"],
                "axis": segment["axis"],
                "length_points": segment["length_points"],
                "primitive_ref": segment["primitive_ref"],
            }
            for segment in topology["segments"]
        ]
        contours.append(
            {
                "id": contour_id,
                "kind": "closed_loop" if closed else "open_profile",
                "bbox_display": list(rect),
                "closed": closed,
                "primitive_refs": [f"drawing[{drawing_index}]"],
                "segments_display": segments,
                "vertex_ids": topology["vertex_ids"],
                "topology": {
                    "segment_count": topology["segment_count"],
                    "vertex_count": topology["vertex_count"],
                    "connected_component_count": topology["connected_component_count"],
                    "endpoint_vertex_count": topology["endpoint_vertex_count"],
                    "branch_vertex_count": topology["branch_vertex_count"],
                    "cycle_rank": topology["cycle_rank"],
                },
                "closure_validation": {
                    "status": "pass" if closed else "not_applicable",
                    "basis": "page-global native endpoint topology",
                    "endpoint_vertex_count": topology["endpoint_vertex_count"],
                    "branch_vertex_count": topology["branch_vertex_count"],
                },
                "epistemic_state": "observed",
                "basis": "native_vector_path_topology_with_page_global_vertices",
                "style": {
                    "width": drawing.get("width"),
                    "stroke": drawing.get("color"),
                    "fill": drawing.get("fill"),
                    "dash": drawing.get("dashes"),
                },
            }
        )
    for component in composite_topologies(native_topology):
        rect = fitz.Rect(component["bbox_display"])
        if rect.get_area() > 0.65 * page_area:
            continue
        contour_id = f"contour.{len(contours) + 1:05d}"
        contours.append(
            {
                "id": contour_id,
                "kind": "composite_closed_loop" if component["closed"] else "composite_open_profile",
                "bbox_display": component["bbox_display"],
                "closed": component["closed"],
                "primitive_refs": component["drawing_refs"],
                "segments_display": [
                    {
                        "id": segment["id"],
                        "kind": segment["kind"],
                        "start_display": segment["start_display"],
                        "end_display": segment["end_display"],
                        "control_points_display": segment["control_points_display"],
                        "sample_points_display": segment.get("sample_points_display", []),
                        "start_vertex_id": segment["start_vertex_id"],
                        "end_vertex_id": segment["end_vertex_id"],
                        "axis": segment["axis"],
                        "length_points": segment["length_points"],
                        "primitive_ref": segment["primitive_ref"],
                    }
                    for segment in component["segments"]
                ],
                "vertex_ids": component["vertex_ids"],
                "topology": {
                    key: component[key]
                    for key in (
                        "segment_count",
                        "vertex_count",
                        "connected_component_count",
                        "endpoint_vertex_count",
                        "branch_vertex_count",
                        "cycle_rank",
                    )
                },
                "closure_validation": {
                    "status": "pass" if component["closed"] else "not_applicable",
                    "basis": "compatible-style native paths joined through page-global vertices",
                    "endpoint_vertex_count": component["endpoint_vertex_count"],
                    "branch_vertex_count": component["branch_vertex_count"],
                },
                "epistemic_state": "derived",
                "basis": "native_vector_connectivity_and_compatible_style",
                "style": {"composite": True},
            }
        )
    return contours


def _compact_contour(contour: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in contour.items() if key not in {"style", "primitive_refs"}}
    segments = output.get("segments_display", []) or []
    if str(contour.get("kind", "")).startswith("composite_"):
        output["segment_refs"] = [segment["id"] for segment in segments]
        output.pop("segments_display", None)
        return output
    output["segments_display"] = [
        {
            key: value
            for key, value in segment.items()
            if value not in (None, [], {})
        }
        for segment in segments
    ]
    return output


def _union_find_components(page: fitz.Page) -> list[dict[str, Any]]:
    drawings = []
    for index, drawing in enumerate(page.get_drawings()):
        rect = fitz.Rect(drawing["rect"])
        if rect.width < 0.2 and rect.height < 0.2:
            continue
        if rect.width > 0.82 * page.rect.width and rect.height > 0.82 * page.rect.height:
            continue
        drawings.append((index, rect, drawing))
    parent = list(range(len(drawings)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    gap = max(1.5, 0.0015 * math.hypot(page.rect.width, page.rect.height))
    ordered = sorted(range(len(drawings)), key=lambda index: drawings[index][1].x0)
    active: list[int] = []
    for current in ordered:
        rect = drawings[current][1]
        active = [index for index in active if drawings[index][1].x1 + gap >= rect.x0]
        expanded = rect + (-gap, -gap, gap, gap)
        for other in active:
            candidate = drawings[other][1]
            if expanded.y1 < candidate.y0 or expanded.y0 > candidate.y1:
                continue
            union(current, other)
        active.append(current)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(drawings)):
        grouped[find(index)].append(index)
    components = []
    for members in grouped.values():
        if len(members) < 4:
            continue
        rect = fitz.Rect(drawings[members[0]][1])
        for member in members[1:]:
            rect |= drawings[member][1]
        if rect.width < 20 or rect.height < 20 or rect.get_area() > 0.70 * page.rect.get_area():
            continue
        if max(rect.width / max(rect.height, 0.1), rect.height / max(rect.width, 0.1)) > 35:
            continue
        horizontal = vertical = 0
        for member in members:
            item_rect = drawings[member][1]
            horizontal += item_rect.width >= max(5.0, 4 * item_rect.height)
            vertical += item_rect.height >= max(5.0, 4 * item_rect.width)
        components.append(
            {
                "bbox": rect,
                "drawing_indices": [drawings[member][0] for member in members],
                "primitive_count": len(members),
                "orthogonal_axis_ratio": (horizontal + vertical) / len(members),
            }
        )
    return sorted(components, key=lambda item: (item["primitive_count"], item["bbox"].get_area()), reverse=True)[:32]


def discover_view_hypotheses(
    page: fitz.Page,
    text_roles: list[dict[str, Any]],
    dimensions: tuple[DimensionAttachment, ...],
    contours: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return geometry-first view islands without choosing an object class."""

    label_roles = {
        "section_label",
        "reinforcement_view_candidate",
        "formwork_view_candidate",
        "section_view_candidate",
        "plan_view_candidate",
    }
    labels = [item for item in text_roles if item["resolved_role"] in label_roles]
    views = []
    for component in _union_find_components(page):
        rect = component["bbox"]
        nearby = [item for item in labels if _rect_distance(rect, fitz.Rect(item["bbox_display"])) <= 0.025 * math.hypot(page.rect.width, page.rect.height)]
        roles = Counter(item["resolved_role"] for item in nearby)
        if roles:
            semantic_roles = roles.copy()
            semantic_roles.pop("section_label", None)
            role = (
                semantic_roles.most_common(1)[0][0]
                if semantic_roles
                else "section_view_candidate"
            )
        elif component["orthogonal_axis_ratio"] >= 0.82 and component["primitive_count"] >= 40:
            role = "table_or_grid_candidate"
        else:
            role = "drawing_view_candidate"
        expanded = rect + (-4, -4, 4, 4)
        local_dimensions = [item for item in dimensions if expanded.intersects(fitz.Rect(item.text_bbox))]
        local_contours = [item for item in contours if expanded.contains(fitz.Rect(item["bbox_display"]).tl) and expanded.contains(fitz.Rect(item["bbox_display"]).br)]
        views.append(
            {
                "id": f"view_hypothesis.{len(views) + 1:03d}",
                "role_hypothesis": role,
                "bbox_display": list(rect),
                "confidence": min(0.94, 0.48 + 0.04 * math.log1p(component["primitive_count"]) + (0.12 if nearby else 0)),
                "epistemic_state": "inferred",
                "basis": "connected_native_geometry_with_optional_semantic_seed",
                "primitive_refs": [f"drawing[{index}]" for index in component["drawing_indices"]],
                "label_claim_refs": [item["id"] for item in nearby],
                "dimension_refs": [item.attachment_id for item in local_dimensions],
                "contour_refs": [item["id"] for item in local_contours],
                "features": {
                    "primitive_count": component["primitive_count"],
                    "orthogonal_axis_ratio": round(component["orthogonal_axis_ratio"], 4),
                    "dimension_proposals": len(local_dimensions),
                    "accepted_dimensions": sum(item.status == "accepted" for item in local_dimensions),
                    "contours": len(local_contours),
                },
            }
        )
    return views


def _serialize_observation_graph(graph: PrimitiveGraph) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "layer": "immutable_observation_graph",
        "nodes": [
            {
                "id": node.node_id,
                "kind": node.kind,
                "bbox_display": list(node.bbox),
                "primitive_refs": list(node.primitive_refs),
                "text": node.text,
                "layer_name": node.layer,
                "style": _jsonable(node.style),
                "orientation": node.orientation,
            }
            for node in graph.nodes
        ],
        "edges": [asdict(edge) for edge in graph.edges],
        "provenance": {"mode": "observed", "method": "native_pdf_scene_graph"},
    }


def _reinforcement_quantity_takeoff(solution: dict[str, Any]) -> dict[str, Any] | None:
    scene = solution.get("reinforcement_3d_input", {}).get("centerline_scene")
    if not scene:
        return None
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in scene.get("paths", []):
        grouped[path["mark"]].append(path)
    source_groups = {
        item.get("mark"): item
        for item in solution.get("reinforcement_3d_input", {}).get("resolved_groups", [])
    }
    groups = []
    for mark, paths in grouped.items():
        lengths = []
        for path in paths:
            points = path["points_xyz_mm"]
            length = sum(math.dist(left, right) for left, right in zip(points, points[1:]))
            lengths.append(length)
        metric = source_groups.get(mark, {}).get("dimension_anchored_solution", {})
        fabrication_resolved = (
            metric.get("status") == "pass"
            and metric.get("fabrication_eligibility", {}).get("installed_equals_cutting")
        )
        fabrication_each = (
            metric.get("solved_centerline", {}).get("installed_length_each_mm")
            if fabrication_resolved
            else None
        )
        groups.append(
            {
                "source_solver_label": mark,
                "role": paths[0]["role"],
                "count": len(paths),
                "diameter_mm": paths[0].get("diameter_mm"),
                "placed_centerline_length_each_mm": round(sum(lengths) / len(lengths), 3),
                "placed_centerline_length_total_mm": round(sum(lengths), 3),
                "fabrication_length_each_mm": fabrication_each,
                "fabrication_length_total_mm": None if fabrication_each is None else round(float(fabrication_each) * len(paths), 3),
                "fabrication_length_state": "drawing_derived" if fabrication_resolved else "unknown",
                "mass_state": "unknown",
                "basis": "resolved_3d_centerline_paths",
            }
        )
    resolved_fabrication_total = sum(
        float(item["fabrication_length_total_mm"])
        for item in groups
        if item["fabrication_length_total_mm"] is not None
    )
    all_fabrication_resolved = bool(groups) and all(item["fabrication_length_total_mm"] is not None for item in groups)
    return {
        "status": "fabrication_resolved" if all_fabrication_resolved else "partial_fabrication_lengths" if resolved_fabrication_total else "partial_placed_centerlines",
        "groups": groups,
        "total_placed_centerline_m": round(sum(item["placed_centerline_length_total_mm"] for item in groups) / 1000, 6),
        "resolved_fabrication_length_m": round(resolved_fabrication_total / 1000, 6),
        "fabrication_length_m": round(resolved_fabrication_total / 1000, 6) if all_fabrication_resolved else None,
        "mass_kg": None,
        "unknowns": ["fabrication hooks and bend corrections for unresolved groups", "material density or grade", "unresolved reinforcement groups"],
    }


def _constrain_section_marks_to_drawing_families(
    section_observations: dict[str, Any] | None,
    rebar_program: dict[str, Any],
) -> None:
    """Reject dimension-like section tokens absent from drawing bar families."""

    if not section_observations:
        return
    native_details = rebar_program.get("native_vector_detail_linking", {}).get("details", [])
    if not native_details:
        return
    allowed = {
        str(mark)
        for detail in native_details
        for mark in detail.get("marks", [])
    } | {
        str(mark["value"])
        for group in rebar_program.get("groups", [])
        for mark in [group.get("identity", {}).get("mark", {})]
        if mark.get("value") is not None
    }
    for section in section_observations.get("sections", []):
        rejected = section.setdefault("rejected_numeric_tokens", [])
        accepted_identities = []
        for identity in section.get("leader_identities", []):
            if identity["mark_text"] in allowed:
                accepted_identities.append(identity)
                continue
            rejected.append(
                {
                    "text": identity["mark_text"],
                    "bbox_display": identity["text_bbox_display"],
                    "state": "rejected",
                    "reason": "no matching drawing fabrication detail or independently resolved cross-view bar group",
                }
            )
        section["leader_identities"] = accepted_identities
        section["leader_identity_count"] = len(accepted_identities)
        # Re-run the local occurrence-to-contour matching after tokens without
        # an independently observed drawing-family identity have been removed.
        # Such tokens must not keep an otherwise unique bar-mark assignment
        # artificially ambiguous.
        _resolve_unique_occurrence_targets(accepted_identities, section.get("candidates", []))
        allowed_by_ref: dict[str, set[str]] = {}
        for identity in accepted_identities:
            for ref in identity.get("candidate_refs", []):
                allowed_by_ref.setdefault(ref, set()).add(identity["mark_text"])
        for candidate in section.get("candidates", []):
            candidate["leader_marks"] = sorted(
                allowed_by_ref.get(candidate["primitive_ref"], set()),
                key=lambda value: (len(value), value),
            )
            candidate["state"] = "resolved" if candidate["leader_marks"] else "candidate"
            if not candidate["leader_marks"]:
                candidate["reason"] = "rebar-like native geometry remains without a drawing-family identity"


def _progress_dimension_ownership(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only presentation-safe fields from completed dimension facts."""

    return {
        "attachments": [
            {
                "id": item.get("id"),
                "value_mm": item.get("value_mm"),
                "status": item.get("status"),
                "epistemic_state": item.get("epistemic_state"),
                "measured_endpoints": [
                    {"point_display": endpoint.get("point_display")}
                    for endpoint in item.get("measured_endpoints", [])
                ],
            }
            for item in payload.get("attachments", [])
        ]
    }


def _progress_view_frame_graph(payload: dict[str, Any]) -> dict[str, Any]:
    """Return accepted-state display geometry without solver internals."""

    return {
        "frames": [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "view_id",
                    "view_role_hypothesis",
                    "section_label",
                    "state",
                    "confidence",
                    "bbox_display",
                    "title",
                )
            }
            for item in payload.get("frames", [])
        ],
        "relations": [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "state",
                    "section_label",
                    "parent_view_id",
                    "section_view_id",
                    "trace",
                )
            }
            for item in payload.get("relations", [])
        ],
        "relation_candidates": [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "state",
                    "section_label",
                    "parent_view_id",
                    "section_view_id",
                    "reason",
                )
            }
            for item in payload.get("relation_candidates", [])
        ],
    }


def _progress_geometry_regions(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "role_hypothesis": item.get("role_hypothesis"),
            "bbox_display": item.get("bbox_display"),
            "state": item.get("state", "observed"),
        }
        for item in views
        if item.get("bbox_display")
    ]


def _progress_table_observations(text_roles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "text": item.get("text"),
            "bbox_display": item.get("bbox_display"),
            "state": item.get("epistemic_state", "observed"),
        }
        for item in text_roles
        if item.get("resolved_role") == "schedule_or_table_value" and item.get("bbox_display")
    ]


def _progress_native_rebar_details(program: dict[str, Any]) -> dict[str, Any]:
    native = program.get("native_vector_detail_linking", {}) or {}
    return {
        "details": [
            {
                "id": item.get("id"),
                "mark_display": item.get("mark_display"),
                "bbox_display": item.get("bbox_display"),
                "state": item.get("state", "observed"),
            }
            for item in native.get("details", [])
        ],
        "placement_associations": [
            {
                "id": item.get("id"),
                "detail_id": item.get("detail_id"),
                "marks": item.get("marks", []),
                "target_mark_bbox_display": item.get("target_mark_bbox_display"),
                "state": item.get("state", "review_candidate"),
                "leader_segments": [
                    {"start": segment.get("start"), "end": segment.get("end")}
                    for segment in (item.get("leader_trace") or {}).get("segments", [])
                ],
                "path_attachments": [
                    {
                        "fragment_points_display": attachment.get("fragment_points_display"),
                        "state": attachment.get("state", item.get("state", "review_candidate")),
                    }
                    for attachment in item.get("path_attachments", [])
                ],
            }
            for item in native.get("placement_associations", [])
        ],
    }


def understand_page(
    page: fitz.Page,
    estimation_profile: dict[str, Any] | None = None,
    progress_callback: Callable[[str, float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    dimension_proposals = propose_dimensions(page)
    text_roles = resolve_text_roles(page, dimension_proposals)
    if progress_callback is not None:
        progress_callback(
            "native_geometry_scan",
            0.06,
            {
                "page": page.number + 1,
                "dimension_proposal_count": len(dimension_proposals),
                "native_text_role_count": len(text_roles),
            },
        )
    native_topology = extract_page_topology(page)
    if progress_callback is not None:
        progress_callback(
            "native_geometry_scan",
            0.12,
            {
                "page": page.number + 1,
                "native_segment_count": len(native_topology.get("segments", [])),
            },
        )
    contours = extract_contours(page, native_topology)
    graph = augment_graph_with_text_roles(build_primitive_graph(page, dimension_proposals), text_roles)
    preliminary_views = discover_view_hypotheses(page, text_roles, dimension_proposals, contours)
    if progress_callback is not None:
        progress_callback(
            "native_geometry",
            0.18,
            {
                "page": page.number + 1,
                "native_segment_count": len(native_topology.get("segments", [])),
                "view_candidate_count": len(preliminary_views),
                "geometry_regions": _progress_geometry_regions(preliminary_views),
                "table_observations": _progress_table_observations(text_roles),
            },
        )
    crop_roles: list[dict[str, Any]] = []
    native_marker_roles: list[dict[str, Any]] = []
    if _native_text_is_sparse(page):
        crop_roles = _outlined_text_roles(
            page,
            dimension_proposals,
            len(text_roles),
            clips=_ocr_view_clips(page, preliminary_views),
            scale=5,
            source_prefix="view_crop",
        )
        text_roles = _merge_text_role_observations(text_roles, crop_roles)
        native_marker_roles = _native_section_marker_roles(
            page,
            text_roles,
            len(text_roles),
        )
        text_roles = _merge_text_role_observations(text_roles, native_marker_roles)
        if crop_roles:
            graph = augment_graph_with_text_roles(build_primitive_graph(page, dimension_proposals), text_roles)
            preliminary_views = discover_view_hypotheses(
                page,
                text_roles,
                dimension_proposals,
                contours,
            )
    observation_graph = _serialize_observation_graph(graph)
    perception_quality = assess_native_page_quality(page)
    raster_fallback = reconstruct_raster_observations(page, perception_quality)
    contours.extend(raster_fallback["generic_structures"]["contour_hypotheses"])
    preliminary_views.extend(raster_fallback["generic_structures"]["view_hypotheses"])
    observation_graph = augment_observation_graph_with_raster(observation_graph, raster_fallback)
    raw_dimension_ownership = resolve_dimension_ownership(
        page,
        dimension_proposals,
        preliminary_views,
        contours,
        native_topology=native_topology,
    )
    dimensions, dimension_ownership, dimension_adjudication = adjudicate_dimension_proposals(
        dimension_proposals,
        raw_dimension_ownership,
    )
    accepted_dimension_refs = {
        item.attachment_id for item in dimensions if item.status == "accepted"
    }
    # Only adjudicated dimensions become semantic facts or solver inputs.
    text_roles = resolve_text_roles(page, dimensions)
    text_roles = _merge_text_role_observations(text_roles, crop_roles)
    text_roles = _merge_text_role_observations(text_roles, native_marker_roles)
    graph = augment_graph_with_text_roles(build_primitive_graph(page, dimensions), text_roles)
    observation_graph = _serialize_observation_graph(graph)
    observation_graph = augment_observation_graph_with_raster(observation_graph, raster_fallback)
    # Rebuild final view scopes from adjudicated dimensions.  Preliminary
    # scopes are an ownership prerequisite, not an accepted downstream view
    # model; retaining proposal-only dimensions here changed legacy geometry
    # closure even when the final dimension set was identical.
    views = discover_view_hypotheses(page, text_roles, dimensions, contours)
    views.extend(raster_fallback["generic_structures"]["view_hypotheses"])
    for view in views:
        view.setdefault("features", {})["accepted_dimensions"] = sum(
            str(ref) in accepted_dimension_refs for ref in view.get("dimension_refs", [])
        )
        view["features"]["dimension_adjudication_completed"] = True
    view_segmentation = segment_views_by_titles(
        page.rect,
        views,
        text_roles,
        geometry_primitives=(
            {
                "id": f"drawing[{index}]",
                "bbox_display": list(drawing["rect"]),
            }
            for index, drawing in enumerate(page.get_drawings())
        ),
    )
    if progress_callback is not None:
        progress_callback(
            "dimensions",
            0.38,
            {
                "page": page.number + 1,
                "dimension_ownership": _progress_dimension_ownership(dimension_ownership),
                "geometry_regions": _progress_geometry_regions(views),
                "table_observations": _progress_table_observations(text_roles),
            },
        )
    scoped_profile_assembly = assemble_profiles_by_view_scope(
        native_topology,
        dimensions,
        dimension_ownership,
        view_segmentation,
        page_number=page.number + 1,
    )
    raster_dimension_ownership = raster_fallback["dimension_ownership"]
    metric_equation_graph = build_metric_equation_graph(page, views, dimensions)
    semantic_regions = propose_semantic_regions(page, dimensions, graph)
    section_rebar_observations = extract_section_rebar(
        page,
        semantic_regions,
        dimensions,
        text_roles,
    )
    object_instance_graph = assemble_object_instances(page.rect, views)
    view_frame_graph = infer_view_frames(
        {
            "page": page.number + 1,
            "view_hypotheses": views,
            "object_instance_graph": object_instance_graph,
            "section_rebar_observations": section_rebar_observations,
            "contour_hypotheses": contours,
            "dimension_ownership": dimension_ownership,
            "dimension_adjudication": dimension_adjudication,
            "title_anchored_view_segmentation": view_segmentation,
            "enforce_section_binding_prerequisites": True,
            "enforce_profile_section_reprojection": True,
            "enforce_compact_transverse_projection": True,
            "metric_equation_graph": metric_equation_graph,
            "native_segments": native_topology["segments"],
        },
        dimensions=dimensions,
        text_roles=text_roles,
        observation_graph=observation_graph,
        page_number=page.number + 1,
    )
    certified_object_instance_graph = assemble_relation_certified_object_instances(
        view_frame_graph,
        views,
    )
    if certified_object_instance_graph["instances"]:
        object_instance_graph = certified_object_instance_graph
        view_frame_graph = infer_view_frames(
            {
                "page": page.number + 1,
                "view_hypotheses": views,
                "object_instance_graph": object_instance_graph,
                "section_rebar_observations": section_rebar_observations,
                "contour_hypotheses": contours,
                "dimension_ownership": dimension_ownership,
                "dimension_adjudication": dimension_adjudication,
                "title_anchored_view_segmentation": view_segmentation,
                "enforce_section_binding_prerequisites": True,
                "enforce_profile_section_reprojection": True,
                "enforce_compact_transverse_projection": True,
                "metric_equation_graph": metric_equation_graph,
                "native_segments": native_topology["segments"],
            },
            dimensions=dimensions,
            text_roles=text_roles,
            observation_graph=observation_graph,
            page_number=page.number + 1,
        )
    if progress_callback is not None:
        progress_callback(
            "view_relations",
            0.56,
            {
                "page": page.number + 1,
                "dimension_ownership": _progress_dimension_ownership(dimension_ownership),
                "view_frame_graph": _progress_view_frame_graph(view_frame_graph),
            },
        )
    section_scope_refs = sorted(
        {
            str(section_ref)
            for scope in view_frame_graph.get("object_scopes", []) or []
            if scope.get("state") == "resolved_relative"
            for section_ref in scope.get("section_view_ids", []) or []
        }
        | {
            str(segment.get("id"))
            for segment in view_segmentation.get("segments", []) or []
            if segment.get("state") == "resolved"
            and str(segment.get("title") or "").strip().lower().startswith("section")
        }
    )
    title_scope_local_dimension_reclosure = reclose_title_scope_dimensions(
        page,
        dimensions,
        dimension_ownership,
        dimension_adjudication,
        view_segmentation,
        section_scope_refs,
        contours,
        native_topology,
    )
    scope_local_profile_reclosure = reclose_profiles_in_title_scopes(
        native_topology,
        dimensions,
        view_segmentation,
        title_scope_local_dimension_reclosure,
        page_number=page.number + 1,
    )
    profile_physical_scope_binding = bind_profiles_to_physical_scopes(
        scoped_profile_assembly,
        view_frame_graph,
        page_number=page.number + 1,
    )
    physical_component_hypothesis = generate_physical_component_hypotheses(
        scoped_profile_assembly,
        profile_physical_scope_binding,
        view_frame_graph,
        page_number=page.number + 1,
    )
    physical_component_placement_evidence = derive_single_prism_placement_evidence(
        physical_component_hypothesis,
        scoped_profile_assembly,
        view_frame_graph,
        dimensions,
        page_number=page.number + 1,
    )
    physical_component_reclosure = reclose_physical_component_hypotheses(
        physical_component_hypothesis,
        view_frame_graph,
        page_number=page.number + 1,
        placement_evidence=physical_component_placement_evidence,
    )
    physical_component_step4_replay = replay_single_prism_step4(
        physical_component_hypothesis,
        physical_component_placement_evidence,
        physical_component_reclosure,
        view_frame_graph,
        page_number=page.number + 1,
    )
    clear_span_prism_reconstruction = reconstruct_clear_span_prism(
        page,
        dimension_ownership,
        dimension_adjudication,
        view_segmentation,
        view_frame_graph,
        page_number=page.number + 1,
    )
    banded_plan_sweep_evidence = derive_banded_plan_sweep_evidence(
        dimension_ownership,
        dimension_adjudication,
        view_frame_graph,
        scoped_profile_assembly,
        page_number=page.number + 1,
        scope_local_profile_reclosure=scope_local_profile_reclosure,
    )
    # Step 1A can accept additional projected or terminal-less dimensions as
    # graph facts. Until relation-driven physical scopes close in Step 2, the
    # legacy solid solvers retain only their previously detector-qualified
    # input subset so dimension portability cannot silently change quantities.
    legacy_solver_dimensions = tuple(
        dimension
        for dimension in dimensions
        if dimension.status == "accepted" and dimension.proposal_status == "accepted"
    )
    solver_attempts = []
    try:
        specialised = solve_object_instance_extrusions(page, legacy_solver_dimensions, object_instance_graph, views)
        solver_attempts.append({"solver": "object_instance_profile_extrusions", "status": "resolved", "reason": None})
    except (RuntimeError, ValueError) as error:
        solver_attempts.append({"solver": "object_instance_profile_extrusions", "status": "gated_unavailable", "reason": str(error)})
        try:
            specialised = solve_semantic_3d(
                page,
                legacy_solver_dimensions,
                text_roles,
                regions=semantic_regions,
                section_rebar=section_rebar_observations,
            )
            solver_attempts.append({"solver": "cross_view_semantic", "status": "resolved", "reason": None})
        except (RuntimeError, ValueError) as semantic_error:
            solver_attempts.append({"solver": "cross_view_semantic", "status": "gated_unavailable", "reason": str(semantic_error)})
            try:
                specialised = solve_generic_prismatic(page, legacy_solver_dimensions)
                solver_attempts.append({"solver": "generic_prismatic", "status": "resolved", "reason": None})
            except (RuntimeError, ValueError) as generic_error:
                solver_attempts.append({"solver": "generic_prismatic", "status": "gated_unavailable", "reason": str(generic_error)})
                try:
                    specialised = solve_generic_profile_extrusion(page, legacy_solver_dimensions)
                    solver_attempts.append({"solver": "generic_profile_extrusion", "status": "resolved", "reason": None})
                except (RuntimeError, ValueError) as profile_error:
                    solver_attempts.append({"solver": "generic_profile_extrusion", "status": "gated_unavailable", "reason": str(profile_error)})
                    specialised = {"status": "gated_unavailable", "reason": str(profile_error)}

    if progress_callback is not None:
        progress_callback(
            "quantity_closure",
            0.72,
            {
                "page": page.number + 1,
                "dimension_ownership": _progress_dimension_ownership(dimension_ownership),
                "view_frame_graph": _progress_view_frame_graph(view_frame_graph),
                "quantities": (
                    [specialised["concrete_quantity_takeoff"]]
                    if specialised.get("constraint_validation", {}).get("status") == "pass"
                    else []
                ),
            },
        )

    evidence_claims: dict[str, Any] = {}
    claims = []
    for item in text_roles:
        if item["resolved_role"] in {"unclassified_text", "unclassified_number"}:
            continue
        claim_id = f"claim.{len(claims) + 1:05d}"
        claims.append(
            {
                "id": claim_id,
                "kind": "text_role",
                "subject": item["id"],
                "value": item["resolved_role"],
                "state": item["epistemic_state"],
                "basis": item["basis"],
                "confidence": item["confidence"],
                "evidence_ref": f"evidence.{claim_id}",
            }
        )
        evidence_claims[f"evidence.{claim_id}"] = {"page": page.number + 1, "primitive_refs": item["primitive_refs"], "bbox_display": item["bbox_display"]}
    for dimension in dimensions:
        if dimension.status != "accepted":
            continue
        ocr_observation = dimension.text_method == "geometry_gated_ocr"
        claim_id = f"claim.{len(claims) + 1:05d}"
        claims.append(
            {
                "id": claim_id,
                "kind": "metric_dimension",
                "subject": dimension.attachment_id,
                "value": dimension.value_mm,
                "unit": "mm",
                "state": "observed" if ocr_observation else "direct",
                "basis": (
                    "geometry_gated_ocr_dimension_with_adjudicated_native_chain"
                    if ocr_observation
                    else "dimension_annotation_with_adjudicated_native_chain"
                ),
                "confidence": dimension.score,
                "evidence_ref": f"evidence.{claim_id}",
            }
        )
        evidence_claims[f"evidence.{claim_id}"] = {
            "page": page.number + 1,
            "primitive_refs": [dimension.baseline.primitive_ref, *(line.primitive_ref for line in dimension.extension_lines), *dimension.terminal_refs],
            "bbox_display": list(dimension.text_bbox),
            "text_method": dimension.text_method,
            "text_confidence": dimension.text_confidence,
            "terminal_style": dimension.terminal_style,
        }

    solved = specialised.get("constraint_validation", {}).get("status") == "pass"
    reinforcement_quantities = _reinforcement_quantity_takeoff(specialised) if solved else None
    solid_preview = None
    if solved:
        solid_preview = {
            "mesh": specialised.get("procedural_evidence", {}).get("mesh"),
            "rebar_paths": specialised.get("reinforcement_3d_input", {}).get("centerline_scene", {}).get("paths", []),
        }
    reinforcement_groups = []
    if solved:
        for group_index, group in enumerate(specialised.get("reinforcement_3d_input", {}).get("resolved_groups", []), start=1):
            group_id = f"reinforcement_group.{group_index:03d}"
            primitive_refs = list(group.get("primitive_refs", [])) + list(group.get("section_boundary_refs", []))
            count_claim_id = f"claim.{len(claims) + 1:05d}"
            claims.append(
                {
                    "id": count_claim_id,
                    "kind": "reinforcement_count",
                    "subject": group_id,
                    "value": group["count"],
                    "state": "derived",
                    "basis": "cross_view_projection_topology",
                    "confidence": 0.91,
                    "evidence_ref": f"evidence.{count_claim_id}",
                }
            )
            evidence_claims[f"evidence.{count_claim_id}"] = {
                "page": page.number + 1,
                "primitive_refs": primitive_refs,
            }
            diameter_claim_id = f"claim.{len(claims) + 1:05d}"
            claims.append(
                {
                    "id": diameter_claim_id,
                    "kind": "reinforcement_diameter",
                    "subject": group_id,
                    "value": group.get("diameter_mm"),
                    "unit": "mm",
                    "state": "convention_dependent",
                    "basis": "graphical_projection_scale",
                    "convention_dependency": "unverified",
                    "confidence": 0.72,
                    "evidence_ref": f"evidence.{diameter_claim_id}",
                }
            )
            evidence_claims[f"evidence.{diameter_claim_id}"] = {
                "page": page.number + 1,
                "primitive_refs": primitive_refs,
            }
            reinforcement_groups.append(
                {
                    "id": group_id,
                    "role": group["role"],
                    "bar_mark": None,
                    "identity_state": "unresolved",
                    "source_solver_label": group.get("mark"),
                    "count_claim_ref": count_claim_id,
                    "diameter_claim_ref": diameter_claim_id,
                    "geometry_state": "derived",
                }
            )

    rebar_program = build_rebar_program(
        page,
        text_roles,
        specialised if solved else None,
        views,
        dimensions,
        contours,
        metric_equation_graph,
        section_rebar_observations,
    )
    scope_rebar_path_graph(rebar_program["physical_path_graph"], object_instance_graph, view_frame_graph)
    enrich_projection_identities(rebar_program["physical_path_graph"])
    rebar_program["native_vector_detail_linking"] = extract_native_vector_details(
        page,
        views,
        object_instance_graph,
        rebar_program["physical_path_graph"],
        dimensions,
        section_rebar_observations,
    )
    _constrain_section_marks_to_drawing_families(
        section_rebar_observations,
        rebar_program,
    )
    enrich_projection_identities(rebar_program["physical_path_graph"])
    resolve_physical_bar_families(
        rebar_program["native_vector_detail_linking"],
        rebar_program["physical_path_graph"],
        views,
        section_rebar_observations,
        rebar_program.get("groups"),
    )
    open_structural_boundary_assembly = assemble_open_structural_boundaries(
        native_topology,
        view_segmentation,
        title_scope_local_dimension_reclosure,
        view_frame_graph,
        dimensions,
        rebar_program["physical_path_graph"],
        page_number=page.number + 1,
    )
    flight_interface_closure = certify_flight_interface_closure(
        open_structural_boundary_assembly,
        title_scope_local_dimension_reclosure,
        banded_plan_sweep_evidence,
        page_number=page.number + 1,
    )
    flight_profile_pair_certification = certify_flight_profile_pair(
        scope_local_profile_reclosure,
        title_scope_local_dimension_reclosure,
        banded_plan_sweep_evidence,
        view_frame_graph,
        native_topology,
        dimensions,
        rebar_program["physical_path_graph"],
        page_number=page.number + 1,
        open_structural_boundary_assembly=open_structural_boundary_assembly,
        flight_interface_closure=flight_interface_closure,
    )
    landing_component_reconstruction = reconstruct_landing_component(
        dimension_ownership,
        dimension_adjudication,
        view_segmentation,
        title_scope_local_dimension_reclosure,
        open_structural_boundary_assembly,
        flight_interface_closure,
        flight_profile_pair_certification,
        banded_plan_sweep_evidence,
        clear_span_prism_reconstruction,
        page_number=page.number + 1,
    )
    terminal_landing_support_reconstruction = reconstruct_terminal_landing_support(
        open_structural_boundary_assembly,
        landing_component_reconstruction,
        banded_plan_sweep_evidence,
        clear_span_prism_reconstruction,
        page_number=page.number + 1,
    )
    same_object_constructive_assembly = assemble_same_object_construction(
        open_structural_boundary_assembly,
        flight_interface_closure,
        banded_plan_sweep_evidence,
        landing_component_reconstruction,
        page_number=page.number + 1,
        native_topology=native_topology,
        title_segmentation=view_segmentation,
    )
    physical_object_quantity_promotion = promote_constructive_union_physical_object(
        same_object_constructive_assembly,
        view_frame_graph,
        banded_plan_sweep_evidence,
        page_number=page.number + 1,
    )
    promoted_object_solved = physical_object_quantity_promotion.get("status") == "accepted"
    if promoted_object_solved and not solved:
        promoted_preview = physical_object_quantity_promotion["canonical_3d_preview"]
        representative_ref = promoted_preview.get("source_equivalence_member_ref")
        representative_materialization = next(
            (
                item
                for item in same_object_constructive_assembly.get(
                    "materialized_search_hypotheses", []
                )
                if item.get("placement_alternative", {}).get("id")
                == representative_ref
            ),
            None,
        )
        construction_region_overlays = [
            {
                "construction_region_ref": item.get("id"),
                "construction_role": item.get("construction_role"),
                "mesh": deepcopy(item.get("mesh")),
                "quantity_eligible": False,
            }
            for item in (representative_materialization or {}).get(
                "construction_regions", []
            )
            if item.get("mesh")
        ]
        solid_preview = {
            "mesh": {
                **deepcopy(promoted_preview["mesh"]),
                "validation": deepcopy(promoted_preview["mesh_validation"]),
            },
            "rebar_paths": [],
            "label": promoted_preview["label"],
            "absolute_orientation_resolved": False,
            "canonical_representation_only": True,
            "relative_placement_state": promoted_preview.get("relative_placement_state"),
            "physical_object_ref": promoted_preview["physical_object_ref"],
            "source_preview_ref": promoted_preview["id"],
            "construction_region_overlays": construction_region_overlays,
            "rendering_contract": {
                "surface_mode": "transparent_evidence_wireframe",
                "per_triangle_strokes": True,
                "construction_region_seams_visible_by_default": False,
                "construction_region_overlay_available": bool(
                    construction_region_overlays
                ),
                "construction_region_overlay_enabled": False,
                "presentation_camera": {
                    "projection_preset": "canonical_fold_revealing_axonometric",
                    "physical_transform_applied": False,
                    "absolute_orientation_claimed": False,
                },
            },
        }
    if solid_preview is not None and clear_span_prism_reconstruction.get("status") == "reclosed_pass":
        beam_preview = clear_span_prism_reconstruction.get("solid_preview") or {}
        beam_mesh = beam_preview.get("mesh") or {}
        beam_volume = clear_span_prism_reconstruction.get("clear_span_volume_candidate") or {}
        callouts = clear_span_prism_reconstruction.get("member_size_callouts") or []
        beam_placement = landing_component_reconstruction.get(
            "beam_relative_placement_certificate"
        ) or {}
        beam_transform = beam_placement.get("beam_transform") or {}
        placed_beam_mesh = (
            _transform_preview_mesh(beam_mesh, beam_transform)
            if beam_placement.get("relative_physical_placement_resolved")
            else None
        )
        if beam_mesh.get("vertices_xyz_mm") and beam_mesh.get("triangles"):
            solid_preview["separate_object_candidate_previews"] = [
                {
                    "record_type": "separate_object_candidate_preview",
                    "state": (
                        "resolved_relative_separate_object"
                        if placed_beam_mesh
                        else "canonical_relative_candidate"
                    ),
                    "classification": (
                        str(callouts[0].get("member_type"))
                        if callouts
                        else str(beam_volume.get("classification") or "structural_member")
                    ),
                    "mesh": deepcopy(placed_beam_mesh or beam_mesh),
                    "source_local_mesh": deepcopy(beam_mesh),
                    "volume_candidate_m3": beam_volume.get("value_m3"),
                    "quantity_eligible": False,
                    "included_in_primary_object": False,
                    "included_in_primary_quantity": False,
                    "relative_physical_placement_resolved": bool(placed_beam_mesh),
                    "presentation_frame": {
                        "kind": (
                            "shared_relative_scene"
                            if placed_beam_mesh
                            else "independent_candidate_viewport"
                        ),
                        "presentation_only_transform": None,
                        "physical_transform": deepcopy(beam_transform) if placed_beam_mesh else None,
                        "relative_physical_placement_resolved": bool(placed_beam_mesh),
                    },
                    "relative_placement_certificate_ref": beam_placement.get("id"),
                    "source_reconstruction_ref": (
                        clear_span_prism_reconstruction.get(
                            "unsigned_bounded_sweep_certificate", {}
                        ).get("id")
                    ),
                    "reason": (
                        "relative landing contact resolved; support overlap and additive quantity remain unresolved"
                        if placed_beam_mesh
                        else "support overlap and relative object placement remain unresolved"
                    ),
                }
            ]
    partial_profile_sweep_materialization = {
        "schema_version": "0.1.0",
        "layer": "partial_profile_sweep_materialization",
        "page": page.number + 1,
        "status": "no_materializable_hypotheses",
        "candidate_previews": [],
        "summary": {"hypothesis_count": 0, "materialized_candidate_count": 0},
        "contract": {"quantity_eligible": False},
    }
    if solid_preview is not None:
        partial_profile_sweep_materialization = materialize_terminal_landing_context(
            terminal_landing_support_reconstruction,
            solid_preview.get("mesh") or {},
            page_number=page.number + 1,
        )
        solid_preview["context_candidate_previews"] = deepcopy(
            partial_profile_sweep_materialization.get("candidate_previews", [])
        )
    rebar_program["drawing_detail_takeoff"] = solve_native_detail_takeoff(
        page,
        rebar_program["native_vector_detail_linking"],
        object_instance_graph,
        rebar_program["physical_path_graph"],
        specialised.get("concrete_3d_input") if solved else None,
        specialised.get("procedural_evidence", {}).get("mesh") if solved else None,
    )
    family_scene = build_family_constrained_scene(
        rebar_program["native_vector_detail_linking"],
        rebar_program["physical_path_graph"],
        (solid_preview or {}).get("mesh"),
        rebar_program.get("groups"),
        [specialised.get("procedural_evidence", {}).get("profile", {}).get("calculation_contour")]
        if solved and specialised.get("procedural_evidence", {}).get("profile", {}).get("calculation_contour")
        else None,
    )
    rebar_program["family_constrained_3d"] = family_scene
    drawing_takeoff = rebar_program["drawing_detail_takeoff"]
    group_takeoff = summarize_group_quantities(
        rebar_program.get("groups", []),
        rebar_program["native_vector_detail_linking"],
    )
    rebar_program["group_quantity_takeoff"] = group_takeoff
    active_estimation_profile = estimation_profile or load_estimation_profile()
    estimated_takeoff = estimate_rebar_quantities(
        rebar_program.get("groups", []),
        rebar_program["native_vector_detail_linking"],
        active_estimation_profile,
        object_instance_graph,
    )
    rebar_program["estimation_profile"] = estimated_takeoff["profile"]
    rebar_program["estimated_quantity_takeoff"] = estimated_takeoff
    if drawing_takeoff.get("status") == "resolved_drawing_takeoff":
        reinforcement_quantities = drawing_takeoff
        if solid_preview is not None:
            solid_preview["rebar_paths"] = drawing_takeoff.get("centerline_scene", {}).get("paths", [])
            solid_preview["rebar_scene_basis"] = "complete drawing-detail takeoff after family constraints"
    elif group_takeoff is not None:
        if reinforcement_quantities is not None:
            rebar_program["pre_family_solver_candidate_quantities"] = reinforcement_quantities
        reinforcement_quantities = group_takeoff
        if solid_preview is not None:
            solid_preview["pre_family_solver_candidate_paths"] = solid_preview.pop("rebar_paths", [])
            solid_preview["rebar_paths"] = family_scene.get("paths", [])
            solid_preview["candidate_rebar_paths"] = family_scene.get("candidate_paths", [])
            solid_preview["rebar_scene_basis"] = "only families whose identity, multiplicity, dimensionality, and reprojection closed"
    elif solid_preview is not None:
        rebar_program["pre_family_solver_candidate_quantities"] = reinforcement_quantities
        reinforcement_quantities = {
            "status": "partial",
            "state": "unknown",
            "total_placed_centerline_m": None,
            "resolved_fabrication_length_m": None,
            "mass_kg": None,
            "reason": "pre-family projected paths are not aggregable after physical identity and multiplicity gates",
            "schedule_values_used": False,
        }
        solid_preview["pre_family_solver_candidate_paths"] = solid_preview.pop("rebar_paths", [])
        solid_preview["rebar_paths"] = family_scene.get("paths", [])
        solid_preview["rebar_scene_basis"] = "only families whose identity, multiplicity, dimensionality, and reprojection closed"

    if progress_callback is not None:
        progress_callback(
            "reinforcement",
            0.91,
            {
                "page": page.number + 1,
                "dimension_ownership": _progress_dimension_ownership(dimension_ownership),
                "view_frame_graph": _progress_view_frame_graph(view_frame_graph),
                "quantities": (
                    [specialised["concrete_quantity_takeoff"]]
                    if solved
                    else physical_object_quantity_promotion.get(
                        "calculated_concrete_quantities", []
                    )
                ),
                "reinforcement_quantities": {
                    "status": (reinforcement_quantities or {}).get("status"),
                    "mass_kg": (reinforcement_quantities or {}).get("mass_kg"),
                },
                "estimated_reinforcement_quantities": {
                    "status": estimated_takeoff.get("status"),
                    "totals": {"mass_kg": estimated_takeoff.get("totals", {}).get("mass_kg")},
                },
                "rebar_program": {
                    "spacing_constraints": [
                        {
                            **{
                                key: item.get(key)
                                for key in (
                                    "id",
                                    "status",
                                    "spacing_mm",
                                    "interval_count",
                                    "epistemic_state",
                                    "bbox_display",
                                )
                            },
                            "chain_attachment": {
                                "measured_points_display": (item.get("chain_attachment") or {}).get(
                                    "measured_points_display"
                                )
                            },
                        }
                        for item in rebar_program.get("spacing_constraints", [])
                    ],
                    "native_vector_detail_linking": _progress_native_rebar_details(rebar_program),
                },
                "section_rebar_observations": {
                    "sections": [
                        {
                            "id": section.get("id") or f"section_rebar.{section_index + 1:03d}",
                            "candidates": [
                                {
                                    "primitive_ref": candidate.get("primitive_ref"),
                                    "state": candidate.get("state", "candidate"),
                                    "bbox_display": candidate.get("bbox_display"),
                                    "geometry_items": candidate.get("geometry_items", []),
                                }
                                for candidate in sorted(
                                    section.get("candidates", []),
                                    key=lambda item: item.get("state") == "resolved",
                                    reverse=True,
                                )[:120]
                            ],
                        }
                        for section_index, section in enumerate(section_rebar_observations.get("sections", []))
                    ]
                },
            },
        )

    relations = list(dimension_ownership.get("relations", []))
    relations.extend(metric_equation_graph.get("relations", []))
    for view in views:
        for dimension_ref in view["dimension_refs"]:
            relations.append({"type": "dimension_located_in_view", "from": dimension_ref, "to": view["id"]})
        for contour_ref in view["contour_refs"]:
            relations.append({"type": "contour_located_in_view", "from": contour_ref, "to": view["id"]})
    for instance in object_instance_graph["instances"]:
        for view_id in instance["view_ids"]:
            relations.append({"type": "projection_of_object_instance", "from": view_id, "to": instance["id"]})
    for group in reinforcement_groups:
        relations.append({"type": "reinforces", "from": group["id"], "to": "solid.001"})
    calculation_contours = []
    if solved:
        profile_contour = specialised.get("procedural_evidence", {}).get("profile", {}).get("calculation_contour")
        if profile_contour:
            calculation_contours.append(profile_contour)
    exchange_records = build_exchange_records(
        attachment_payload(dimension_proposals),
        view_segmentation,
        contours,
        view_frame_graph.get("shared_coordinate_system", {}),
        profile_assembly=scoped_profile_assembly,
        profile_scope_binding=profile_physical_scope_binding,
        physical_component_hypothesis=physical_component_hypothesis,
        physical_component_reclosure=physical_component_reclosure,
    )
    engineering = {
        "schema_version": "0.1.0",
        "layer": "compact_engineering_graph",
        "status": "partial" if not solved else "constraint_solved_subset",
        "perception_routing": {
            "quality": perception_quality,
            "raster_fallback_summary": raster_fallback["summary"],
            "raster_fallback_status": raster_fallback["status"],
            "raster_dimension_topology_status": raster_fallback["dimension_topology"]["status"],
            "raster_dimension_topology_summary": raster_fallback["dimension_topology"]["summary"],
            "raster_dimension_label_ocr_status": raster_fallback["dimension_label_ocr"]["status"],
            "raster_dimension_label_ocr_summary": raster_fallback["dimension_label_ocr"]["summary"],
            "raster_dimension_ownership_status": raster_dimension_ownership["status"],
            "raster_dimension_ownership_summary": raster_dimension_ownership["summary"],
            "model_provenance": raster_fallback["model_provenance"],
        },
        "view_hypotheses": views,
        "enforce_section_binding_prerequisites": True,
        "title_anchored_view_segmentation": view_segmentation,
        "title_scope_local_dimension_reclosure": title_scope_local_dimension_reclosure,
        "scoped_profile_assembly": scoped_profile_assembly,
        "scope_local_profile_reclosure": scope_local_profile_reclosure,
        "open_structural_boundary_assembly": open_structural_boundary_assembly,
        "flight_interface_closure": flight_interface_closure,
        "flight_profile_pair_certification": flight_profile_pair_certification,
        "landing_component_reconstruction": landing_component_reconstruction,
        "terminal_landing_support_reconstruction": terminal_landing_support_reconstruction,
        "partial_profile_sweep_materialization": partial_profile_sweep_materialization,
        "same_object_constructive_assembly": same_object_constructive_assembly,
        "physical_object_quantity_promotion": physical_object_quantity_promotion,
        "profile_physical_scope_binding": profile_physical_scope_binding,
        "physical_component_hypothesis": physical_component_hypothesis,
        "physical_component_placement_evidence": physical_component_placement_evidence,
        "physical_component_reclosure": physical_component_reclosure,
        "physical_component_step4_replay": physical_component_step4_replay,
        "clear_span_prism_reconstruction": clear_span_prism_reconstruction,
        "banded_plan_sweep_evidence": banded_plan_sweep_evidence,
        "exchange_records": exchange_records,
        "view_frame_graph": view_frame_graph,
        "object_instance_graph": object_instance_graph,
        "contour_hypotheses": [_compact_contour(item) for item in contours],
        "calculation_contours": calculation_contours,
        "dimension_ownership": dimension_ownership,
        "dimension_adjudication": dimension_adjudication,
        "raster_dimension_ownership": raster_dimension_ownership,
        "metric_equation_graph": metric_equation_graph,
        "section_rebar_observations": section_rebar_observations,
        "claims": claims,
        "relations": relations,
        "reinforcement_groups": reinforcement_groups,
        "rebar_program": rebar_program,
        "solid_hypotheses": (
            [{"id": "solid.001", "state": "derived", "basis": "closed_cross_view_constraint_system", "geometry": specialised["concrete_3d_input"]}]
            if solved
            else (
                [physical_object_quantity_promotion["accepted_physical_object"]]
                if promoted_object_solved
                else []
            )
        ),
        "quantities": (
            [specialised["concrete_quantity_takeoff"]]
            if solved
            else physical_object_quantity_promotion.get(
                "calculated_concrete_quantities", []
            )
        ),
        "reinforcement_quantities": reinforcement_quantities,
        "estimated_reinforcement_quantities": estimated_takeoff,
        "specialised_solver": {
            "status": "resolved" if solved or promoted_object_solved else "gated_unavailable",
            "reason": None if solved or promoted_object_solved else specialised.get("reason"),
            "selected": (
                specialised.get("pipeline_mode")
                if solved
                else (
                    "same_object_invariant_union"
                    if promoted_object_solved
                    else None
                )
            ),
        },
        "solver_attempts": solver_attempts,
        "solid_evidence": (
            specialised.get("procedural_evidence", {})
            if solved
            else (
                physical_object_quantity_promotion.get(
                    "scope_reclosure_certificate", {}
                )
                if promoted_object_solved
                else {}
            )
        ),
        "solid_preview": solid_preview,
        "unresolved": [
            (
                "absolute orientation unresolved; canonical relative preview emitted"
                if promoted_object_solved and not solved
                else (
                    "cross-view correspondence is not unique"
                    if not solved
                    else "remaining object views are not yet fused"
                )
            ),
            (
                "separate beam support-overlap boundary remains unresolved"
                if promoted_object_solved and not solved
                else (
                    "solid topology is not emitted without a closed constraint system"
                    if not solved
                    else "non-primary reinforcement topology"
                )
            ),
        ],
    }
    engineering["canonical_knowledge_graph"] = build_canonical_knowledge_graph(engineering)
    evidence = {
        "schema_version": "0.1.0",
        "layer": "evidence_store",
        "claims": evidence_claims,
        "contours": {
            item["id"]: {
                "page": page.number + 1,
                "primitive_refs": item["primitive_refs"],
                "bbox_display": item["bbox_display"],
                "topology": item.get("topology"),
                "segment_refs": [segment["id"] for segment in item.get("segments_display", [])],
            }
            for item in contours
        },
        "views": {item["id"]: {"page": page.number + 1, "primitive_refs": item["primitive_refs"], "bbox_display": item["bbox_display"]} for item in views},
        "view_frame_graph": view_frame_graph,
        "object_instances": {
            item["id"]: {"page": page.number + 1, "view_ids": item["view_ids"], "basis": item["basis"]}
            for item in object_instance_graph["instances"]
        },
        "calculation_contours": {
            item["id"]: {
                "page": page.number + 1,
                "primitive_refs": item["primitive_refs"],
                "dimension_refs": item["dimension_refs"],
                "bbox_display": item["bbox_display"],
                "closure_validation": item["closure_validation"],
            }
            for item in calculation_contours
        },
        "dimension_ownership": {
            item["id"]: {
                "page": page.number + 1,
                "dimension_ref": item["dimension_ref"],
                "primitive_refs": item["primitive_refs"],
                "measured_endpoints": item["measured_endpoints"],
                "topology_segment_refs": item.get("topology_segment_refs", []),
                "geometry_anchor_refs": item.get("geometry_anchor_refs", []),
                "status": item["status"],
            }
            for item in dimension_ownership.get("attachments", [])
        },
        "dimension_adjudication": dimension_adjudication,
        "title_anchored_view_segmentation": view_segmentation,
        "title_scope_local_dimension_reclosure": title_scope_local_dimension_reclosure,
        "scoped_profile_assembly": scoped_profile_assembly,
        "scope_local_profile_reclosure": scope_local_profile_reclosure,
        "open_structural_boundary_assembly": open_structural_boundary_assembly,
        "flight_interface_closure": flight_interface_closure,
        "flight_profile_pair_certification": flight_profile_pair_certification,
        "landing_component_reconstruction": landing_component_reconstruction,
        "terminal_landing_support_reconstruction": terminal_landing_support_reconstruction,
        "partial_profile_sweep_materialization": partial_profile_sweep_materialization,
        "same_object_constructive_assembly": same_object_constructive_assembly,
        "physical_object_quantity_promotion": physical_object_quantity_promotion,
        "profile_physical_scope_binding": profile_physical_scope_binding,
        "physical_component_hypothesis": physical_component_hypothesis,
        "physical_component_placement_evidence": physical_component_placement_evidence,
        "physical_component_reclosure": physical_component_reclosure,
        "physical_component_step4_replay": physical_component_step4_replay,
        "clear_span_prism_reconstruction": clear_span_prism_reconstruction,
        "banded_plan_sweep_evidence": banded_plan_sweep_evidence,
        "exchange_records": exchange_records,
        "raster_dimension_ownership": raster_dimension_ownership,
        "metric_equation_graph": metric_equation_graph,
        "section_rebar_observations": section_rebar_observations,
        "raster_fallback": raster_fallback,
    }
    return {
        "page": page.number + 1,
        "observation_graph": observation_graph,
        "engineering_graph": engineering,
        "evidence_store": evidence,
        "text_roles": text_roles,
        "contours": contours,
        "dimensions": dimensions,
        "dimension_proposals": dimension_proposals,
    }


def page_summary(record: dict[str, Any]) -> dict[str, Any]:
    engineering = record["engineering_graph"]
    frame_summary = engineering.get("view_frame_graph", {}).get("summary", {})
    segmentation_summary = engineering.get("title_anchored_view_segmentation", {}).get("summary", {})
    profile_summary = engineering.get("scoped_profile_assembly", {}).get("summary", {})
    profile_binding_summary = engineering.get("profile_physical_scope_binding", {}).get("summary", {})
    component_hypothesis_summary = engineering.get("physical_component_hypothesis", {}).get("summary", {})
    component_reclosure_summary = engineering.get("physical_component_reclosure", {}).get("summary", {})
    return {
        "page": record["page"],
        "observation_nodes": len(record["observation_graph"]["nodes"]),
        "observation_edges": len(record["observation_graph"]["edges"]),
        "perception_route": engineering.get("perception_routing", {}).get("quality", {}).get("route"),
        "raster_observations": engineering.get("perception_routing", {}).get("raster_fallback_summary", {}).get("observation_count", 0),
        "view_hypotheses": len(engineering["view_hypotheses"]),
        "title_anchored_view_scopes": segmentation_summary.get("resolved_segment_count", 0),
        "split_view_parents": segmentation_summary.get("split_parent_view_count", 0),
        "scoped_profile_candidates": profile_summary.get("profile_count", 0),
        "scoped_profile_abstentions": profile_summary.get("abstention_count", 0),
        "physically_scoped_profiles": profile_binding_summary.get("binding_count", 0),
        "profile_physical_scope_abstentions": profile_binding_summary.get("abstention_count", 0),
        "step5_reconstruction_inputs": profile_binding_summary.get("step5_reconstruction_input_count", 0),
        "physical_component_hypotheses": component_hypothesis_summary.get("hypothesis_count", 0),
        "physical_component_hypothesis_abstentions": component_hypothesis_summary.get("abstention_count", 0),
        "physical_component_reclosures_passed": component_reclosure_summary.get("reclosed_pass_count", 0),
        "physical_component_reclosures_failed": component_reclosure_summary.get("reclosed_fail_count", 0),
        "physical_component_reclosures_insufficient": component_reclosure_summary.get("insufficient_constraints_count", 0),
        "view_frames": frame_summary.get("frame_count", 0),
        "cutting_plane_relations": frame_summary.get("cutting_plane_relation_count", 0),
        "shared_coordinate_scopes": frame_summary.get("resolved_relative_scope_count", 0),
        "coordinate_mapped_views": frame_summary.get("axis_mapping_resolved_view_count", 0),
        "cross_view_metric_checks_passed": frame_summary.get("reprojection_pass_count", 0),
        "cross_view_metric_checks_failed": frame_summary.get("reprojection_fail_count", 0),
        "accepted_contour_correspondences": frame_summary.get("accepted_contour_correspondence_count", 0),
        "signed_contour_transforms": frame_summary.get("signed_contour_transform_count", 0),
        "path_reprojection_eligible_scopes": frame_summary.get("path_reprojection_eligible_scope_count", 0),
        "canonical_entities": engineering.get("canonical_knowledge_graph", {}).get("summary", {}).get("entity_count", 0),
        "canonical_relations": engineering.get("canonical_knowledge_graph", {}).get("summary", {}).get("relation_count", 0),
        "object_instances": engineering.get("object_instance_graph", {}).get("summary", {}).get("object_instance_count", 0),
        "shared_supporting_views": engineering.get("object_instance_graph", {}).get("summary", {}).get("shared_supporting_view_count", 0),
        "contours": len(engineering["contour_hypotheses"]),
        "calculation_contours": len(engineering.get("calculation_contours", [])),
        "section_rebar_candidates": engineering.get("section_rebar_observations", {}).get("candidate_count", 0),
        "claims": len(engineering["claims"]),
        "accepted_dimensions": sum(item.status == "accepted" for item in record["dimensions"]),
        "dimensions_with_geometry_owner": engineering.get("dimension_ownership", {}).get("summary", {}).get("accepted_count", 0),
        "globally_resolved_dimension_endpoints": engineering.get("dimension_ownership", {}).get("summary", {}).get("globally_resolved_endpoint_cluster_count", 0),
        "metric_equations_with_native_chain": engineering.get("metric_equation_graph", {}).get("summary", {}).get("accepted_chain_count", 0),
        "metric_equation_scopes": frame_summary.get("metric_equation_scope_count", 0),
        "rebar_program_groups": len(engineering.get("rebar_program", {}).get("groups", [])),
        "rebar_spacing_constraints": len(engineering.get("rebar_program", {}).get("spacing_constraints", [])),
        "rebar_spacing_associations": len(engineering.get("rebar_program", {}).get("spacing_associations", [])),
        "rebar_unassigned_spacing_constraints": len(engineering.get("rebar_program", {}).get("unassigned_spacing_constraints", [])),
        "rebar_identity_hypotheses": len(engineering.get("rebar_program", {}).get("identity_hypotheses", [])),
        "rebar_identity_associations": len(engineering.get("rebar_program", {}).get("identity_associations", [])),
        "native_vector_details": len(engineering.get("rebar_program", {}).get("native_vector_detail_linking", {}).get("details", [])),
        "native_detail_placement_links": len(engineering.get("rebar_program", {}).get("native_vector_detail_linking", {}).get("placement_associations", [])),
        "physical_bar_families": len(engineering.get("rebar_program", {}).get("native_vector_detail_linking", {}).get("physical_families", [])),
        "rebar_unassigned_identity_hypotheses": len(engineering.get("rebar_program", {}).get("unassigned_identity_hypotheses", [])),
        "rebar_metric_anchors": len(engineering.get("rebar_program", {}).get("metric_anchors", [])),
        "rebar_dimension_anchored_groups": sum(
            (group.get("placement", {}).get("metric_solution") or {}).get("status") == "pass"
            for group in engineering.get("rebar_program", {}).get("groups", [])
        ),
        "rebar_path_fragments": engineering.get("rebar_program", {}).get("physical_path_graph", {}).get("summary", {}).get("fragment_count", 0),
        "rebar_path_components": engineering.get("rebar_program", {}).get("physical_path_graph", {}).get("summary", {}).get("component_count", 0),
        "rebar_metric_projected_components": engineering.get("rebar_program", {}).get("physical_path_graph", {}).get("summary", {}).get("metric_projected_component_count", 0),
        "text_roles": dict(sorted(Counter(item["resolved_role"] for item in record["text_roles"]).items())),
        "solid_status": engineering["specialised_solver"]["status"],
        "solid_gate_reason": engineering["specialised_solver"]["reason"],
    }
