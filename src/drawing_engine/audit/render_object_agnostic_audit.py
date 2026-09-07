#!/usr/bin/env python3
"""Render object-agnostic view, contour, text-role, and constraint audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from collections import Counter
from math import isfinite
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.core.object_agnostic_understanding import page_summary, understand_page
from src.drawing_engine.disciplines.rebar.estimation_profile import DEFAULT_PROFILE_PATH, load_estimation_profile
from src.drawing_engine.audit.audit_presentation import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    audit_font_file,
    item_palette_color as _detail_palette_color,
    choose,
    duration_text,
    quantity_text,
    reason_text,
    review_action,
    term,
    unit_text,
    validate_language,
)
from src.drawing_engine.disciplines.concrete.schedule_comparison import extract_declared_schedules
from src.drawing_engine.core.automatic_adjudication import RULESET_VERSION, ruleset_sha256
from src.drawing_engine.core.canonical_knowledge_graph import build_canonical_knowledge_graph
from src.drawing_engine.project.review_feedback import canonical_graph_sha256
from src.drawing_engine.pipelines.generate_object_agnostic_bundle import pipeline_identity


COLORS = {
    "ink": (0.08, 0.12, 0.20),
    "muted": (0.35, 0.40, 0.48),
    "paper": (0.97, 0.98, 0.99),
    "white": (1.0, 1.0, 1.0),
    "view": (0.05, 0.57, 0.68),
    "contour": (0.69, 0.20, 0.56),
    "direct": (0.08, 0.58, 0.32),
    "candidate": (0.92, 0.57, 0.08),
    "declared": (0.18, 0.36, 0.72),
    "schedule": (0.43, 0.47, 0.54),
    "unknown": (0.43, 0.47, 0.54),
    "identifier": (0.42, 0.28, 0.73),
    "blocked": (0.78, 0.19, 0.18),
}

GROUP_COLORS = (
    (0.12, 0.38, 0.82),
    (0.68, 0.20, 0.62),
    (0.08, 0.48, 0.58),
    (0.35, 0.25, 0.70),
    (0.24, 0.43, 0.58),
)


def _detail_marks(detail: dict[str, Any]) -> tuple[str, ...]:
    raw = detail.get("marks") or [detail.get("mark") or detail.get("mark_display")]
    return tuple(sorted({str(value).strip().lstrip("MМ") for value in raw if str(value or "").strip()}))


def _detail_color_assignments(
    fabrication_details: list[dict[str, Any]],
    native_details: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Assign one color per detected detail, aliasing overlapping duplicate records."""
    native = sorted(native_details, key=lambda item: str(item.get("id") or ""))
    native_colors = {
        str(detail["id"]): _detail_palette_color(index)
        for index, detail in enumerate(native)
    }
    fabrication_colors: dict[str, tuple[float, float, float]] = {}
    next_index = len(native_colors)
    for detail in sorted(fabrication_details, key=lambda item: str(item.get("id") or "")):
        detail_id = str(detail.get("id") or detail.get("group_id") or f"fabrication.{next_index}")
        source_box = fitz.Rect(detail.get("source", {}).get("bbox_display", []))
        marks = _detail_marks(detail)
        best_native_id = None
        best_overlap = 0.0
        if not source_box.is_empty and source_box.get_area() > 0:
            for candidate in native:
                if marks and _detail_marks(candidate) != marks:
                    continue
                candidate_box = fitz.Rect(candidate.get("bbox_display", []))
                if candidate_box.is_empty or candidate_box.get_area() <= 0:
                    continue
                overlap = (source_box & candidate_box).get_area() / min(source_box.get_area(), candidate_box.get_area())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_native_id = str(candidate["id"])
        if best_native_id is not None and best_overlap >= 0.50:
            fabrication_colors[detail_id] = native_colors[best_native_id]
        else:
            fabrication_colors[detail_id] = _detail_palette_color(next_index)
            next_index += 1
    return {"native": native_colors, "fabrication": fabrication_colors}


def insert_text(page: fitz.Page, rect: fitz.Rect, text: str, size: float, color: tuple[float, float, float], *, bold: bool = False) -> None:
    if rect.width <= 0 or rect.height <= 0 or not all(isfinite(value) for value in (rect.x0, rect.y0, rect.x1, rect.y1)):
        return
    font = "AuditSansBold" if bold else "AuditSans"
    page.insert_textbox(rect, text, fontsize=size, fontname=font, fontfile=audit_font_file(bold), color=color, lineheight=1.22)


def _role_color(role: str) -> tuple[float, float, float]:
    if role == "dimension":
        return COLORS["direct"]
    if role == "dimension_candidate" or role == "spacing_expression_token":
        return COLORS["candidate"]
    if role == "schedule_or_table_value":
        return COLORS["schedule"]
    if role == "identifier_candidate":
        return COLORS["identifier"]
    return COLORS["view"]


def draw_overlay(
    output: fitz.Document,
    display: fitz.Document,
    source_page: fitz.Page,
    record: dict[str, Any],
    index: int,
    ocgs: dict[str, int],
) -> None:
    sidebar = 380
    page = output.new_page(width=source_page.rect.width + sidebar, height=source_page.rect.height)
    page.show_pdf_page(fitz.Rect(0, 0, source_page.rect.width, source_page.rect.height), display, index, oc=ocgs["SOURCE"])
    views = record["engineering_graph"]["view_hypotheses"]
    for view in views:
        rect = fitz.Rect(view["bbox_display"])
        page.draw_rect(rect, color=COLORS["view"], width=2.0, dashes="[7 4]", stroke_opacity=0.82, oc=ocgs["VIEW_HYPOTHESES"])
        tag = fitz.Rect(rect.x0, max(0, rect.y0 - 16), min(rect.x1, rect.x0 + 180), rect.y0)
        insert_text(page, tag, f"{view['id']} | {view['role_hypothesis']}", 6.5, COLORS["view"], bold=True)

    contours = sorted(record["contours"], key=lambda item: fitz.Rect(item["bbox_display"]).get_area(), reverse=True)
    minimum_area = source_page.rect.get_area() * 0.00008
    displayed_contours = [item for item in contours if fitz.Rect(item["bbox_display"]).get_area() >= minimum_area][:350]
    for contour in displayed_contours:
        rect = fitz.Rect(contour["bbox_display"])
        page.draw_rect(rect, color=COLORS["contour"], width=0.65, stroke_opacity=0.50, oc=ocgs["CONTOUR_HYPOTHESES"])

    relevant_roles = {
        "dimension",
        "dimension_candidate",
        "spacing_expression_token",
        "section_label",
        "schedule_or_table_value",
        "identifier_candidate",
        "reinforcement_view_candidate",
        "formwork_view_candidate",
        "section_view_candidate",
        "plan_view_candidate",
    }
    for role in record["text_roles"]:
        if role["resolved_role"] not in relevant_roles:
            continue
        rect = fitz.Rect(role["bbox_display"]) + (-1.2, -1.2, 1.2, 1.2)
        page.draw_rect(rect, color=_role_color(role["resolved_role"]), width=0.9, stroke_opacity=0.76, oc=ocgs["TEXT_ROLES"])

    for dimension in record["dimensions"]:
        if dimension.status != "accepted":
            continue
        page.draw_line(dimension.measured_points[0], dimension.measured_points[1], color=COLORS["direct"], width=1.6, stroke_opacity=0.80, oc=ocgs["DIRECT_DIMENSIONS"])
        if dimension.text_method == "geometry_gated_ocr":
            box = fitz.Rect(dimension.text_bbox)
            page.draw_rect(box, color=COLORS["direct"], width=1.5, dashes="[4 2]", stroke_opacity=0.90, oc=ocgs["DIRECT_DIMENSIONS"])
            tag = fitz.Rect(box.x0, max(0, box.y0 - 14), min(source_page.rect.x1, box.x0 + 110), box.y0)
            insert_text(page, tag, f"OCR {dimension.text} | {dimension.text_confidence:.2f}", 6.4, COLORS["direct"], bold=True)

    path_graph = record["engineering_graph"].get("rebar_program", {}).get("physical_path_graph", {})
    path_fragments = sorted(
        path_graph.get("fragments", []),
        key=lambda item: (
            bool(item.get("resolved_group_ids")),
            bool(item.get("repetition_group_ids")),
            item.get("candidate_score", 0.0),
        ),
        reverse=True,
    )[:1200]
    for fragment in path_fragments:
        points = [fitz.Point(*point) for point in fragment["geometry"]["points_display"]]
        if len(points) < 2:
            continue
        resolved = bool(fragment.get("resolved_group_ids"))
        color = COLORS["identifier"] if resolved else COLORS["view"]
        page.draw_polyline(
            points,
            color=color,
            width=2.4 if resolved else 0.75,
            stroke_opacity=0.88 if resolved else 0.38,
            oc=ocgs["REBAR_PATH_GRAPH"],
        )

    drawings = source_page.get_drawings()
    for group in record["engineering_graph"].get("rebar_program", {}).get("groups", []):
        metric = group.get("placement", {}).get("metric_solution") or {}
        if metric.get("status") != "pass":
            continue
        selected_segments = []
        for primitive_ref in metric["elevation_observations"]["primitive_refs"]:
            match = re.fullmatch(r"drawing\[(\d+)\]\.item\[(\d+)\]", primitive_ref)
            if match is None:
                continue
            drawing_index, item_index = map(int, match.groups())
            if drawing_index >= len(drawings) or item_index >= len(drawings[drawing_index]["items"]):
                continue
            item = drawings[drawing_index]["items"][item_index]
            if item[0] != "l":
                continue
            selected_segments.append((item[1], item[2]))
            page.draw_line(item[1], item[2], color=COLORS["identifier"], width=3.2, stroke_opacity=0.86, oc=ocgs["REBAR_METRIC_ANCHORS"])
            for endpoint in (item[1], item[2]):
                page.draw_circle(endpoint, 4.0, color=COLORS["direct"], fill=COLORS["white"], width=1.4, oc=ocgs["REBAR_METRIC_ANCHORS"])
        if selected_segments:
            solved = metric["solved_centerline"]
            first = selected_segments[0]
            x = min(first[0].x, first[1].x)
            y = min(first[0].y, first[1].y)
            tag = fitz.Rect(max(0, x - 45), max(0, y - 28), min(source_page.rect.x1, x + 235), max(0, y - 7))
            mark = group.get("identity", {}).get("mark", {}).get("value") or group["id"]
            insert_text(page, tag, f"MARK {mark} | Z {solved['z_start_mm']:g} -> {solved['z_end_mm']:g} | L {solved['installed_length_each_mm']:g} mm", 7.0, COLORS["identifier"], bold=True)

    selected_profiles = record["engineering_graph"].get("solid_evidence", {}).get("metric_profiles", [])
    selected_ids = set(record["engineering_graph"].get("solid_evidence", {}).get("selected_profile_ids", []))
    for profile in selected_profiles:
        if profile["id"] not in selected_ids:
            continue
        rect = fitz.Rect(profile["bbox_display"])
        page.draw_rect(rect, color=COLORS["direct"], width=3.0, stroke_opacity=0.95, oc=ocgs["SOLID_EVIDENCE"])
        points = [fitz.Point(*point) for point in profile.get("points_display", [])]
        if len(points) >= 3:
            page.draw_polyline(points, color=COLORS["direct"], width=2.1, closePath=True, stroke_opacity=0.95, oc=ocgs["SOLID_EVIDENCE"])
        tag = fitz.Rect(rect.x0, max(0, rect.y0 - 18), min(source_page.rect.x1, rect.x0 + 250), rect.y0)
        dimensions = " x ".join(f"{value:g}" for value in profile["dimensions_mm"])
        role = profile.get("role", "solid_profile").replace("_", " ").upper()
        insert_text(page, tag, f"{role} {dimensions} mm", 7.2, COLORS["direct"], bold=True)

    x0 = source_page.rect.width
    page.draw_rect(fitz.Rect(x0, 0, page.rect.x1, page.rect.y1), color=COLORS["paper"], fill=COLORS["paper"], width=0)
    margin = 24
    insert_text(page, fitz.Rect(x0 + margin, 24, page.rect.x1 - margin, 88), "OBJECT-AGNOSTIC\nDRAWING AUDIT", 19, COLORS["ink"], bold=True)
    insert_text(page, fitz.Rect(x0 + margin, 100, page.rect.x1 - margin, 176), "No object template is required at this stage. Boxes are hypotheses derived from source geometry, not reviewed regions.", 9, COLORS["muted"])
    summary = page_summary(record)
    cards = [
        ("OBSERVATION NODES", summary["observation_nodes"]),
        ("VIEW HYPOTHESES", summary["view_hypotheses"]),
        ("CONTOURS", summary["contours"]),
        ("DIRECT DIMENSIONS", summary["accepted_dimensions"]),
        ("ENGINEERING CLAIMS", summary["claims"]),
        ("REBAR PATH FRAGMENTS", summary["rebar_path_fragments"]),
    ]
    y = 194
    for label, value in cards:
        page.draw_rect(fitz.Rect(x0 + margin, y, page.rect.x1 - margin, y + 54), color=COLORS["candidate"], fill=COLORS["white"], width=1)
        insert_text(page, fitz.Rect(x0 + margin + 12, y + 8, page.rect.x1 - margin - 12, y + 26), label, 7.5, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(x0 + margin + 12, y + 28, page.rect.x1 - margin - 12, y + 49), str(value), 13, COLORS["ink"], bold=True)
        y += 64
    insert_text(page, fitz.Rect(x0 + margin, y + 6, page.rect.x1 - margin, y + 110), "ROLE COLORS\nGreen  direct/geometry-gated OCR dimension\nOrange  candidate/spacing\nCyan  view and generic rebar-path observation\nGray  schedule/table value\nPurple  identifier or resolved path", 8.2, COLORS["ink"])
    solid = record["engineering_graph"]["specialised_solver"]
    color = COLORS["direct"] if solid["status"] == "resolved" else COLORS["blocked"]
    gate_y = max(y + 125, page.rect.height - 220)
    page.draw_rect(fitz.Rect(x0 + margin, gate_y, page.rect.x1 - margin, page.rect.height - 24), color=color, fill=COLORS["white"], width=1.4)
    title = "SOLID CONSTRAINTS CLOSED" if solid["status"] == "resolved" else "SOLID REMAINS A HYPOTHESIS"
    quantities = record["engineering_graph"].get("quantities", [])
    method = solid.get("selected") or "none"
    reason = (
        f"{method}: validated watertight solid. Net concrete {quantities[0]['net_concrete_m3']:.4f} m3."
        if solid["status"] == "resolved" and quantities
        else solid["reason"]
    )
    insert_text(page, fitz.Rect(x0 + margin + 12, gate_y + 12, page.rect.x1 - margin - 12, gate_y + 42), title, 10, color, bold=True)
    insert_text(page, fitz.Rect(x0 + margin + 12, gate_y + 48, page.rect.x1 - margin - 12, page.rect.height - 36), str(reason), 8.5, COLORS["ink"])


def draw_summary(output: fitz.Document, source: Path, records: list[dict[str, Any]]) -> None:
    page = output.new_page(width=1600, height=1000)
    page.draw_rect(page.rect, color=COLORS["paper"], fill=COLORS["paper"], width=0)
    page.draw_rect(fitz.Rect(0, 0, 1600, 118), color=COLORS["ink"], fill=COLORS["ink"], width=0)
    insert_text(page, fitz.Rect(48, 25, 1520, 70), "OBJECT-AGNOSTIC ENGINEERING GRAPH", 24, COLORS["white"], bold=True)
    insert_text(page, fitz.Rect(48, 75, 1520, 104), source.name, 10, COLORS["white"])
    summaries = [page_summary(record) for record in records]
    totals = {
        "nodes": sum(item["observation_nodes"] for item in summaries),
        "views": sum(item["view_hypotheses"] for item in summaries),
        "contours": sum(item["contours"] for item in summaries),
        "claims": sum(item["claims"] for item in summaries),
    }
    for index, (label, value) in enumerate((
        ("OBSERVATION NODES", totals["nodes"]),
        ("GENERIC VIEWS", totals["views"]),
        ("CONTOURS", totals["contours"]),
        ("COMPACT CLAIMS", totals["claims"]),
    )):
        x = 48 + 385 * index
        page.draw_rect(fitz.Rect(x, 155, x + 350, 290), color=COLORS["candidate"], fill=COLORS["white"], width=1.3)
        insert_text(page, fitz.Rect(x + 18, 178, x + 330, 210), label, 9, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(x + 18, 228, x + 330, 275), str(value), 23, COLORS["ink"], bold=True)
    insert_text(page, fitz.Rect(48, 340, 760, 380), "THREE-LAYER OUTPUT", 15, COLORS["ink"], bold=True)
    insert_text(page, fitz.Rect(48, 400, 760, 650), "1. Observation graph\nImmutable native PDF nodes and membership edges.\n\n2. Compact engineering graph\nViews, contours, claims, relations, solid hypotheses and quantities.\n\n3. Evidence store\nEvery compact claim points back to source primitives and page coordinates.", 12, COLORS["ink"])
    roles = Counter(item["resolved_role"] for record in records for item in record["text_roles"])
    role_lines = [f"{role}: {count}" for role, count in roles.most_common() if role not in {"unclassified_text", "unclassified_number"}]
    insert_text(page, fitz.Rect(820, 340, 1520, 380), "RESOLVED TEXT ROLES", 15, COLORS["ink"], bold=True)
    insert_text(page, fitz.Rect(820, 400, 1520, 690), "\n".join(role_lines) or "No native text roles; geometry remains available.", 11, COLORS["ink"])
    solved = all(record["engineering_graph"]["specialised_solver"]["status"] == "resolved" for record in records)
    boundary = "A validated solid and concrete quantity are available for every page." if solved else "The observation and engineering graphs are available even though no unique solid has been proven. Concrete volume remains undefined until the cross-view constraint system closes."
    insert_text(page, fitz.Rect(48, 760, 1520, 910), f"FAIL-CLOSED SOLID BOUNDARY\n{boundary}\n\nSchedule/table values are classified as context, never used as geometric predictions.", 13, COLORS["ink"], bold=True)


def _primary_dimension_text(geometry: dict[str, Any], language: str) -> str:
    millimetres = unit_text(language, "mm")
    shape_type = geometry.get("shape_type")
    if shape_type == "rectangular_prism":
        return f"{' x '.join(f'{value:g}' for value in geometry['dimensions_mm'])} {millimetres}"
    if shape_type == "extruded_profile":
        dimensions = geometry["dimensions_mm"]
        return choose(
            language,
            f"profile {dimensions['profile_bbox_width_mm']:g} x {dimensions['profile_bbox_height_mm']:g} {millimetres}; depth {dimensions['extrusion_depth_mm']:g} {millimetres}",
            f"профиль {dimensions['profile_bbox_width_mm']:g} x {dimensions['profile_bbox_height_mm']:g} {millimetres}; глубина {dimensions['extrusion_depth_mm']:g} {millimetres}",
        )
    if shape_type == "multi_object_extrusion_collection":
        return "\n".join(
            choose(
                language,
                f"{item['object_instance_id']}: area {quantity_text(language, item['profile_area_mm2'] / 1e6, 'm2', 4)}; depth {item['extrusion_depth_mm']:g} {millimetres}",
                f"объект {index}: площадь {quantity_text(language, item['profile_area_mm2'] / 1e6, 'm2', 4)}; глубина {item['extrusion_depth_mm']:g} {millimetres}",
            )
            for index, item in enumerate(geometry.get("objects", []), start=1)
        )
    if shape_type == "constructive_union":
        return choose(
            language,
            f"one constructive union ({geometry.get('construction_region_count', 0)} nonadditive regions); absolute orientation unresolved",
            f"единое конструктивное объединение ({geometry.get('construction_region_count', 0)} неаддитивных областей); абсолютная ориентация не определена",
        )
    shaft = geometry["shaft"]
    return f"{shaft['width_x_mm']:g} x {shaft['depth_y_mm']:g} x {shaft['height_z_mm']:g} {millimetres}"


def _rebar_3d_path_count(engineering: dict[str, Any]) -> int:
    preview_paths = (engineering.get("solid_preview") or {}).get("rebar_paths")
    if isinstance(preview_paths, list):
        return len(preview_paths)
    return int((engineering.get("rebar_program") or {}).get("family_constrained_3d", {}).get("path_count", 0) or 0)


def _solid_replay(engineering: dict[str, Any]) -> dict[str, Any]:
    replay = engineering.get("solid_replay")
    return replay if isinstance(replay, dict) else {}


def _solid_replay_validation_text(replay: dict[str, Any], language: str = DEFAULT_LANGUAGE) -> str:
    """Return audit copy for accepted or abstaining Step 4 replay records."""

    audit = replay.get("audit") or {}
    records = audit.get("validation_records") or {}
    if replay.get("status") != "reclosed_pass":
        return choose(
            language,
            f"Solid replay: {replay.get('status', 'insufficient_constraints')}\n{replay.get('reason') or audit.get('reason') or 'required certified inputs are incomplete'}\n\nNo assembly mesh was published.",
            f"Повторная проверка тела: {replay.get('status', 'insufficient_constraints')}\n{replay.get('reason') or audit.get('reason') or 'неполны обязательные подтвержденные входные данные'}\n\nСборочная сетка не опубликована.",
        )
    components = records.get("components", []) or []
    interfaces = records.get("interfaces", []) or []
    overlaps = records.get("overlaps", []) or []
    volume = records.get("volume", {}) or {}
    reprojections = records.get("reprojections", []) or []
    component_lines = "\n".join(
        f"{item.get('id')}: watertight={item.get('validation', {}).get('watertight')} | boundary={item.get('validation', {}).get('boundary_edge_count')}"
        for item in components
    )
    interface_lines = "\n".join(
        f"{item.get('id')}: {item.get('area_mm2', 0):g} mm2"
        for item in interfaces
    )
    overlap_lines = "\n".join(
        f"{' / '.join(item.get('component_refs', []))}: {item.get('overlap_volume_mm3', 0):g} mm3"
        for item in overlaps
    )
    projection_lines = "\n".join(
        f"{view.get('id')}: {len(view.get('component_projections', []))} components | pass"
        for view in reprojections
    )
    return choose(
        language,
        f"COMPONENT VALIDATION\n{component_lines}\n\nINTERFACES\n{interface_lines}\n\nOVERLAP\n{overlap_lines}\n\nVOLUME\nanalytic={volume.get('analytic_component_sum_mm3', 0):g} mm3\nmesh={volume.get('mesh_component_sum_mm3', 0):g} mm3\n\nREPROJECTION\n{projection_lines}\n\nPreview only. Replay wrote no quantity.",
        f"ПРОВЕРКА КОМПОНЕНТОВ\n{component_lines}\n\nИНТЕРФЕЙСЫ\n{interface_lines}\n\nПЕРЕСЕЧЕНИЯ\n{overlap_lines}\n\nОБЪЕМ\nаналитический={volume.get('analytic_component_sum_mm3', 0):g} мм3\nсетка={volume.get('mesh_component_sum_mm3', 0):g} мм3\n\nРЕПРОЕКЦИЯ\n{projection_lines}\n\nТолько предпросмотр. Количество не записано.",
    )


def draw_3d_model(output: fitz.Document, source: Path, record: dict[str, Any], language: str, understanding_seconds: float) -> None:
    engineering = record["engineering_graph"]
    replay = _solid_replay(engineering)
    preview = engineering.get("solid_preview") or replay.get("solid_preview") or {}
    mesh = preview.get("mesh") or {}
    vertices = mesh.get("vertices_xyz_mm") or []
    faces = mesh.get("triangles") or []
    if not vertices or not faces:
        return
    page = output.new_page(width=1600, height=1000)
    page.draw_rect(page.rect, color=COLORS["paper"], fill=COLORS["paper"], width=0)
    page.draw_rect(fitz.Rect(0, 0, 1600, 120), color=COLORS["ink"], fill=COLORS["ink"], width=0)
    insert_text(page, fitz.Rect(48, 28, 1520, 70), choose(language, "AXONOMETRIC MODEL FROM DRAWING", "АКСОНОМЕТРИЧЕСКАЯ МОДЕЛЬ ПО ЧЕРТЕЖУ"), 24, COLORS["white"], bold=True)
    has_estimated_paths = bool(preview.get("candidate_rebar_paths"))
    subtitle = choose(
        language,
        f"Source drawing | page {record['page']} | concrete, drawing-constrained rebar and colored profile-estimate candidates" if has_estimated_paths else f"Source drawing | page {record['page']} | concrete model and only the rebar paths confirmed by drawing evidence",
        f"Исходный чертеж | страница {record['page']} | бетон, подтвержденная арматура и цветные кандидаты оценки" if has_estimated_paths else f"Исходный чертеж | страница {record['page']} | модель бетона и только подтвержденные по чертежу стержни",
    )
    if preview.get("label"):
        subtitle = f"{subtitle} | {preview['label']}"
    insert_text(page, fitz.Rect(48, 77, 1520, 108), f"{subtitle} | {duration_text(language, understanding_seconds)}", 10, COLORS["white"])

    rendering_contract = preview.get("rendering_contract") or {}
    presentation_camera = rendering_contract.get("presentation_camera") or {}
    projection_preset = presentation_camera.get("projection_preset")

    def project(point: list[float]) -> tuple[float, float, float]:
        x, y, z = point
        if projection_preset == "canonical_fold_revealing_axonometric":
            return (
                -x * 0.85 + y * 0.35,
                z - x * 0.20 - y * 0.25,
                -x + y - z * 0.15,
            )
        if projection_preset == "canonical_rising_path_axonometric":
            return (
                (-x - y) * 0.8660254,
                z + (-x + y) * 0.5,
                -x + y - z * 0.15,
            )
        return ((x - y) * 0.8660254, z + (x + y) * 0.5, x + y - z * 0.15)

    candidate_previews = preview.get("separate_object_candidate_previews") or []
    context_previews = preview.get("context_candidate_previews") or []
    scene_candidates = [*candidate_previews, *context_previews]
    placed_candidate_geometry = []
    for candidate in scene_candidates:
        candidate_mesh = candidate.get("mesh") or {}
        candidate_vertices = candidate_mesh.get("vertices_xyz_mm") or []
        candidate_faces = candidate_mesh.get("triangles") or []
        if (
            candidate.get("relative_physical_placement_resolved")
            and candidate_vertices
            and candidate_faces
        ):
            placed_candidate_geometry.append(
                (candidate, [project(vertex) for vertex in candidate_vertices], candidate_faces)
            )
    projected = [project(vertex) for vertex in vertices]
    scene_projected = [*projected, *(point for _, points, _ in placed_candidate_geometry for point in points)]
    x_values = [point[0] for point in scene_projected]
    y_values = [point[1] for point in scene_projected]
    target = fitz.Rect(70, 160, 1120, 900)
    scale = min(target.width / max(max(x_values) - min(x_values), 1), target.height / max(max(y_values) - min(y_values), 1)) * 0.88
    center_x = (min(x_values) + max(x_values)) / 2
    center_y = (min(y_values) + max(y_values)) / 2

    def screen(point: tuple[float, float, float]) -> fitz.Point:
        return fitz.Point(target.x0 + target.width / 2 + (point[0] - center_x) * scale, target.y0 + target.height / 2 - (point[1] - center_y) * scale)

    for face in sorted(
        faces,
        key=lambda item: sum(projected[index][2] for index in item) / len(item),
        reverse=True,
    ):
        points = [screen(projected[index]) for index in face]
        page.draw_polyline(
            points,
            color=COLORS["view"],
            fill=COLORS["view"],
            width=0.45,
            closePath=True,
            stroke_opacity=0.45,
            fill_opacity=0.13,
        )

    for candidate, candidate_projected, candidate_faces in placed_candidate_geometry:
        color = COLORS.get(str(candidate.get("display_color_role") or "candidate"), COLORS["candidate"])
        for face in sorted(
            candidate_faces,
            key=lambda item: sum(candidate_projected[index][2] for index in item) / len(item),
            reverse=True,
        ):
            page.draw_polyline(
                [screen(candidate_projected[index]) for index in face],
                color=color,
                fill=color,
                width=0.65,
                closePath=True,
                stroke_opacity=0.72,
                fill_opacity=0.10,
            )

    if rendering_contract.get("construction_region_overlay_enabled"):
        overlay_colors = (COLORS["candidate"], COLORS["identifier"], COLORS["blocked"])
        for region_index, region in enumerate(preview.get("construction_region_overlays") or []):
            region_mesh = region.get("mesh") or {}
            region_vertices = region_mesh.get("vertices_xyz_mm") or []
            region_faces = region_mesh.get("triangles") or []
            if not region_vertices or not region_faces:
                continue
            region_projected = [project(vertex) for vertex in region_vertices]
            color = overlay_colors[region_index % len(overlay_colors)]
            for face in region_faces:
                page.draw_polyline(
                    [screen(region_projected[index]) for index in face],
                    color=color,
                    fill=None,
                    width=0.35,
                    closePath=True,
                    stroke_opacity=0.34,
                )

    if candidate_previews:
        candidate = candidate_previews[0]
        candidate_mesh = candidate.get("mesh") or {}
        candidate_vertices = candidate_mesh.get("vertices_xyz_mm") or []
        candidate_faces = candidate_mesh.get("triangles") or []
        if candidate_vertices and candidate_faces:
            card = fitz.Rect(70, 150, 500, 382 if context_previews else 332)
            page.draw_rect(card, color=COLORS["candidate"], fill=COLORS["white"], width=1.2)
            classification = str(candidate.get("classification") or "object").upper()
            placement_resolved = bool(candidate.get("relative_physical_placement_resolved"))
            insert_text(
                page,
                fitz.Rect(card.x0 + 16, card.y0 + 12, card.x1 - 16, card.y0 + 34),
                choose(
                    language,
                    f"SEPARATE {classification} - PLACED" if placement_resolved else f"SEPARATE {classification} CANDIDATE",
                    f"ОТДЕЛЬНЫЙ {classification} - РАЗМЕЩЕН" if placement_resolved else f"ОТДЕЛЬНЫЙ КАНДИДАТ: {classification}",
                ),
                9.5,
                COLORS["candidate"],
                bold=True,
            )
            candidate_projected = [project(vertex) for vertex in candidate_vertices]
            cx_values = [point[0] for point in candidate_projected]
            cy_values = [point[1] for point in candidate_projected]
            candidate_target = fitz.Rect(card.x0 + 18, card.y0 + 42, card.x1 - 18, card.y1 - (102 if context_previews else 48))
            candidate_scale = min(
                candidate_target.width / max(max(cx_values) - min(cx_values), 1),
                candidate_target.height / max(max(cy_values) - min(cy_values), 1),
            ) * 0.88
            candidate_center_x = (min(cx_values) + max(cx_values)) / 2
            candidate_center_y = (min(cy_values) + max(cy_values)) / 2

            def candidate_screen(point: tuple[float, float, float]) -> fitz.Point:
                return fitz.Point(
                    candidate_target.x0 + candidate_target.width / 2 + (point[0] - candidate_center_x) * candidate_scale,
                    candidate_target.y0 + candidate_target.height / 2 - (point[1] - candidate_center_y) * candidate_scale,
                )

            for face in sorted(
                candidate_faces,
                key=lambda item: sum(candidate_projected[index][2] for index in item) / len(item),
                reverse=True,
            ):
                page.draw_polyline(
                    [candidate_screen(candidate_projected[index]) for index in face],
                    color=COLORS["candidate"],
                    fill=COLORS["candidate"],
                    width=0.45,
                    closePath=True,
                    stroke_opacity=0.55,
                    fill_opacity=0.12,
                )
            volume = candidate.get("volume_candidate_m3")
            volume_text = f"{float(volume):.3f} m³ candidate; not aggregated" if volume is not None else "Not aggregated"
            insert_text(
                page,
                fitz.Rect(card.x0 + 16, card.y1 - (94 if context_previews else 42), card.x1 - 16, card.y1 - (62 if context_previews else 10)),
                choose(
                    language,
                    f"{volume_text} | landing contact resolved" if placement_resolved else f"{volume_text} | relative placement unresolved",
                    f"{volume_text} | контакт с площадкой подтвержден" if placement_resolved else f"{volume_text} | взаимное положение не определено",
                ),
                7.5,
                COLORS["ink"],
            )
            if context_previews:
                terminal_support = next(
                    (
                        item
                        for item in context_previews
                        if item.get("classification") == "terminal_support_beam_candidate"
                    ),
                    {},
                )
                support_volume = terminal_support.get("volume_candidate_m3")
                support_volume_text = (
                    f"{float(support_volume):.3f} m³ candidate"
                    if support_volume is not None
                    else "volume unresolved"
                )
                insert_text(
                    page,
                    fitz.Rect(card.x0 + 16, card.y1 - 58, card.x1 - 16, card.y1 - 10),
                    choose(
                        language,
                        f"UPPER LANDING: section-only 150 mm slab, analysis-capped\nTOP SUPPORT: 200 x 300 x 2150 mm, {support_volume_text}; not aggregated",
                        f"ВЕРХНЯЯ ПЛОЩАДКА: плита 150 мм только по сечению, аналитическая граница\nВЕРХНЯЯ ОПОРА: 200 x 300 x 2150 мм, {support_volume_text}; не суммируется",
                    ),
                    7.2,
                    COLORS["ink"],
                )

    rebar_paths = preview.get("rebar_paths") or []
    candidate_rebar_paths = preview.get("candidate_rebar_paths") or []
    diameter_colors = {14.0: (0.86, 0.18, 0.16), 8.0: (0.12, 0.38, 0.82)}
    for path in rebar_paths:
        points = [screen(project(point)) for point in path["points_xyz_mm"]]
        if len(points) >= 2:
            color = diameter_colors.get(float(path.get("diameter_mm") or 0), COLORS["identifier"])
            page.draw_polyline(points, color=color, width=1.25, stroke_opacity=0.96)
    for path in candidate_rebar_paths:
        points = [screen(project(point)) for point in path["points_xyz_mm"]]
        if len(points) >= 2:
            try:
                color = _detail_palette_color(int(str(path.get("mark") or "0")))
            except ValueError:
                color = COLORS["candidate"]
            page.draw_polyline(points, color=color, width=0.9, stroke_opacity=0.56)

    if replay.get("status") == "reclosed_pass":
        x0 = 1160
        page.draw_rect(fitz.Rect(x0, 160, 1550, 900), color=COLORS["direct"], fill=COLORS["white"], width=1.4)
        insert_text(
            page,
            fitz.Rect(x0 + 24, 188, 1526, 225),
            choose(language, "CERTIFIED SOLID REPLAY", "ПОДТВЕРЖДЕННАЯ ПРОВЕРКА ТЕЛА"),
            15,
            COLORS["direct"],
            bold=True,
        )
        insert_text(
            page,
            fitz.Rect(x0 + 24, 250, 1526, 860),
            _solid_replay_validation_text(replay, language),
            9.5,
            COLORS["ink"],
        )
        return

    quantity = engineering["quantities"][0]
    geometry = engineering["solid_hypotheses"][0]["geometry"]
    dimension_text = _primary_dimension_text(geometry, language)
    validation = mesh.get("validation", {})
    placement_note = (
        choose(language, "\n\nThe objects are spaced apart only for presentation. Their relative assembly placement is not established by the drawing.", "\n\nОбъекты разнесены только для наглядности. Их взаимное положение в сборке чертежом не подтверждено.")
        if geometry.get("shape_type") == "multi_object_extrusion_collection"
        else ""
    )
    x0 = 1160
    page.draw_rect(fitz.Rect(x0, 160, 1550, 900), color=COLORS["direct"], fill=COLORS["white"], width=1.4)
    insert_text(page, fitz.Rect(x0 + 24, 188, 1526, 225), choose(language, "CONCRETE MODEL AVAILABLE", "МОДЕЛЬ БЕТОНА ПОСТРОЕНА"), 15, COLORS["direct"], bold=True)
    takeoff = engineering.get("rebar_program", {}).get("drawing_detail_takeoff", {})
    native_details = engineering.get("rebar_program", {}).get("native_vector_detail_linking", {}).get("details", [])
    symbolic_fabrication = [
        item.get("fabrication_geometry_solution", {})
        for item in native_details
        if item.get("fabrication_geometry_solution", {}).get("status") == "convention_dependent"
        and item.get("fabrication_geometry_solution", {}).get("topology") == "open_rectangular_loop_with_two_diagonal_hooks"
    ]
    fabrication_progress = choose(
        language,
        f"Hook topology/dimensions solved: {len(symbolic_fabrication)} | centerline radius still required",
        f"Крюки и их размеры замкнуты: {len(symbolic_fabrication)} | еще нужен радиус оси гиба",
    ) if symbolic_fabrication else choose(
        language,
        "Hook topology and bend dimensions remain unresolved",
        "Топология крюков и размеры гибов не замкнуты",
    )
    takeoff_totals = takeoff.get("totals", {})
    takeoff_resolved = takeoff.get("status") == "resolved_drawing_takeoff"
    estimated_takeoff = engineering.get("estimated_reinforcement_quantities", {})
    if not takeoff_resolved and estimated_takeoff.get("families"):
        profile = estimated_takeoff["profile"]
        fabrication_progress = choose(
            language,
            f"Estimate closes bends with profile Rcl={profile['stirrup_tie_centerline_radius_factor']:g}d; strict radius remains unknown",
            f"Оценка замыкает гибы по профилю Rос={profile['stirrup_tie_centerline_radius_factor']:g}d; строгий радиус неизвестен",
        )
    reinforcement = engineering.get("reinforcement_quantities") or {}
    installed = reinforcement.get("total_placed_centerline_m")
    partial_fabrication = reinforcement.get("resolved_fabrication_length_m")
    displayed_takeoff = takeoff if takeoff_resolved else estimated_takeoff
    diameter_lines = "\n".join(
        choose(
            language,
            f"Ø{row['diameter_mm']:g}: {row['physical_bar_count']} bars | {quantity_text(language, row['total_length_m'], 'm', 3)} | {_takeoff_mass_text(language, row)}",
            f"Ø{row['diameter_mm']:g}: {row['physical_bar_count']} шт. | {quantity_text(language, row['total_length_m'], 'm', 3)} | {_takeoff_mass_text(language, row)}",
        )
        for row in displayed_takeoff.get("by_diameter", [])
    )
    if takeoff_resolved:
        rebar_heading = choose(language, "Resolved rebar paths", "Подтвержденные пути арматуры")
        rebar_totals = choose(
            language,
            f"Total: {quantity_text(language, takeoff_totals['fabrication_length_m'], 'm', 3)} | {_takeoff_mass_text(language, takeoff_totals)}",
            f"Итого: {quantity_text(language, takeoff_totals['fabrication_length_m'], 'm', 3)} | {_takeoff_mass_text(language, takeoff_totals)}",
        )
        rebar_note = choose(
            language,
            "Every counted bar is shown. An engineer must still approve the takeoff before quotation.",
            "Показан каждый учтенный стержень. Перед КП требуется утверждение ведомости инженером.",
        )
        path_count_text = choose(
            language,
            f"{len(rebar_paths)}\nProjection envelopes (not counted): {len(candidate_rebar_paths)}",
            f"{len(rebar_paths)}\nПроекционных контуров (не учтены): {len(candidate_rebar_paths)}",
        )
        model_assumption_note = choose(
            language,
            "Thin colored envelopes remain excluded from the strict takeoff.",
            "Тонкие цветные контуры не входят в строгий расчет.",
        )
    else:
        estimated_totals = estimated_takeoff.get("totals", {})
        profile = estimated_takeoff.get("profile", {})
        coverage = estimated_takeoff.get("coverage", {})
        family_scene = engineering.get("rebar_program", {}).get(
            "family_constrained_3d", {}
        )
        no_closed_families = (
            family_scene.get("reason") == "no physical rebar families closed"
        )
        rebar_heading = choose(
            language,
            "Rebar quantity unavailable" if no_closed_families else "Profile estimate (strict result remains partial)",
            "Количество арматуры недоступно" if no_closed_families else "Оценка по профилю (строгий результат частичный)",
        )
        rebar_totals = (
            choose(
                language,
                "No physical rebar families closed; no reinforcement quantity emitted.",
                "Физические семейства арматуры не замкнуты; количество арматуры не выдано.",
            )
            if no_closed_families
            else choose(
                language,
                f"Estimate: {quantity_text(language, estimated_totals.get('fabrication_length_m'), 'm', 3)} | {quantity_text(language, estimated_totals.get('mass_kg'), 'kg', 1)}\nFamilies: {coverage.get('estimated_family_count', 0)}/{coverage.get('eligible_family_count', 0)} | Rcl={profile.get('stirrup_tie_centerline_radius_factor', 0):g}d",
                f"Оценка: {quantity_text(language, estimated_totals.get('fabrication_length_m'), 'm', 3)} | {quantity_text(language, estimated_totals.get('mass_kg'), 'kg', 1)}\nСемейства: {coverage.get('estimated_family_count', 0)}/{coverage.get('eligible_family_count', 0)} | Rос={profile.get('stirrup_tie_centerline_radius_factor', 0):g}d",
            )
        )
        shown = len(rebar_paths) + len(candidate_rebar_paths)
        estimated_bars = estimated_totals.get("physical_bar_count", 0)
        rebar_note = choose(
            language,
            f"The 3D page places {shown} of {estimated_bars} estimated bars. Families without unique placement stay absent. Every assumed value is listed in the graph and profile.",
            f"На 3D-странице размещено {shown} из {estimated_bars} оценочных стержней. Семейства без однозначного положения не показаны. Все допущения записаны в графе и профиле.",
        )
        path_count_text = choose(
            language,
            f"Drawing-constrained paths: {len(rebar_paths)}\nColored placement candidates: {len(candidate_rebar_paths)}",
            f"Подтвержденные пути: {len(rebar_paths)}\nЦветные кандидаты размещения: {len(candidate_rebar_paths)}",
        )
        model_assumption_note = choose(
            language,
            (
                "No rebar path is shown because identity and multiplicity remain unresolved."
                if no_closed_families
                else "Colored envelopes are estimated only under the named profile and do not change the strict takeoff."
            ),
            (
                "Пути арматуры не показаны, поскольку идентичность и кратность не определены."
                if no_closed_families
                else "Цветные контуры входят только в оценку по указанному профилю и не меняют строгий расчет."
            ),
        )
    details = choose(
        language,
        f"Primary dimensions\n{dimension_text}\n\nNet concrete\n{quantity_text(language, quantity['net_concrete_m3'], 'm3', 4)}\n\nModel check\nConcrete volume fully bounded: {'yes' if validation.get('watertight') else 'no'}\n\n{rebar_heading}\n{path_count_text}\n{fabrication_progress}\n{diameter_lines}\n{rebar_totals}\n\nConcrete is rendered as a transparent evidence wireframe. The camera is presentation-only and makes the certified low-to-high route legible; it does not resolve absolute orientation. {model_assumption_note} {rebar_note}{placement_note}",
        f"Основные размеры\n{dimension_text}\n\nОбъем бетона\n{quantity_text(language, quantity['net_concrete_m3'], 'm3', 4)}\n\nПроверка модели\nОбъем бетона полностью ограничен: {'да' if validation.get('watertight') else 'нет'}\n\n{rebar_heading}\n{path_count_text}\n{fabrication_progress}\n{diameter_lines}\n{rebar_totals}\n\nБетон показан как прозрачная доказательная каркасная модель. Камера используется только для представления и делает сертифицированный путь снизу вверх читаемым; абсолютная ориентация остается неразрешенной. {model_assumption_note} {rebar_note}{placement_note}",
    )
    insert_text(page, fitz.Rect(x0 + 24, 250, 1526, 860), details, 12, COLORS["ink"])


def draw_rebar_program(output: fitz.Document, source: Path, record: dict[str, Any]) -> None:
    program = record["engineering_graph"].get("rebar_program") or {}
    page = output.new_page(width=1600, height=1000)
    page.draw_rect(page.rect, color=COLORS["paper"], fill=COLORS["paper"], width=0)
    page.draw_rect(fitz.Rect(0, 0, 1600, 120), color=COLORS["ink"], fill=COLORS["ink"], width=0)
    insert_text(page, fitz.Rect(48, 28, 1520, 70), "SYMBOLIC REBAR PROGRAM", 24, COLORS["white"], bold=True)
    insert_text(
        page,
        fitz.Rect(48, 77, 1520, 106),
        f"{source.name} | page {record['page']} | geometry, distribution, and fabrication remain separate",
        10,
        COLORS["white"],
    )

    groups = program.get("groups", [])
    path_summary = program.get("physical_path_graph", {}).get("summary", {})
    spacing = program.get("spacing_constraints", program.get("unassigned_spacing_constraints", []))
    unassigned_spacing = program.get("unassigned_spacing_constraints", [])
    identities = program.get("identity_hypotheses", program.get("unassigned_identity_hypotheses", []))
    identity_associations = program.get("identity_associations", [])
    cards = [
        ("PROGRAM STATUS", program.get("status", "unresolved")),
        ("RESOLVED GROUPS", str(len(groups))),
        ("PATH GRAPH", f"{path_summary.get('fragment_count', 0)} fragments / {path_summary.get('component_count', 0)} components"),
        ("FABRICATION", program.get("fabrication_status", "unavailable")),
    ]
    for index, (label, value) in enumerate(cards):
        x = 48 + index * 385
        page.draw_rect(fitz.Rect(x, 150, x + 350, 242), color=COLORS["candidate"], fill=COLORS["white"], width=1.2)
        insert_text(page, fitz.Rect(x + 14, 164, x + 330, 184), label, 8, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(x + 14, 194, x + 330, 228), value, 13, COLORS["ink"], bold=True)

    insert_text(page, fitz.Rect(48, 278, 780, 310), "PARAMETRIC GROUPS", 14, COLORS["ink"], bold=True)
    if not groups:
        page.draw_rect(fitz.Rect(48, 326, 780, 466), color=COLORS["blocked"], fill=COLORS["white"], width=1.2)
        insert_text(
            page,
            fitz.Rect(66, 350, 760, 440),
            f"No physical bar group is sufficiently constrained yet. The generic graph still preserves {path_summary.get('fragment_count', 0)} candidate fragments, {path_summary.get('component_count', 0)} connected components, and {path_summary.get('metric_projected_component_count', 0)} metric projected-length observations. These are not summed as physical steel until cross-view identity is unique.",
            11,
            COLORS["ink"],
        )
    else:
        details_by_id = {item["id"]: item for item in program.get("fabrication_details", [])}
        y = 326
        for group in groups[:4]:
            topology = group["topology"]["family"]["value"]
            diameter = group["bar_spec"]["diameter_mm"]["value"]
            mark = group["identity"]["mark"]["value"] or "UNKNOWN"
            mark_evidence_count = len(group["identity"]["mark"].get("evidence_refs", []))
            distribution = group["distribution"]
            zones = distribution.get("zones", [])
            if zones:
                program_text = " | ".join(
                    f"{zone['spacing_mm']:g} x {zone['interval_count']}"
                    for zone in zones[:8]
                )
                if len(zones) > 8:
                    program_text += f" | +{len(zones) - 8} zones"
            else:
                program_text = distribution.get("type", "unresolved")
            metric = group.get("placement", {}).get("metric_solution") or {}
            if metric.get("status") == "pass":
                solved = metric["solved_centerline"]
                section_check = metric["section_validation"]
                reprojection = metric["reprojection"]
                group_lines = (
                    f"Topology: {topology} | dia: {diameter if diameter is not None else 'UNKNOWN'} mm | count: {group['quantity']['value']}\n"
                    f"Anchors Z: {solved['z_start_mm']:g} -> {solved['z_end_mm']:g} | sections: {section_check['passing_section_count']} PASS | reproj: {reprojection['max_endpoint_residual_mm']:g} mm\n"
                    f"Installed: {solved['installed_length_each_mm']:g} mm | Cut: {group['fabrication']['cutting_length_each_mm']:g} mm each | total: {group['fabrication']['cutting_length_total_mm'] / 1000:g} m"
                )
            else:
                group_lines = (
                    f"Topology: {topology} | dia: {diameter if diameter is not None else 'UNKNOWN'} mm | count: {group['quantity']['value']}\n"
                    f"Distribution: {program_text}\n"
                    + (
                        f"Cut: {group['fabrication']['cutting_length_each_mm']:g} mm each"
                        if group["fabrication"].get("cutting_length_each_mm") is not None
                        else (
                            f"Cut: {group['fabrication']['equation']} = UNKNOWN"
                            if group["fabrication"].get("equation")
                            else "Cut: UNKNOWN | mark-linked fabrication detail missing"
                        )
                    )
                )
            page.draw_rect(fitz.Rect(48, y, 780, y + 112), color=COLORS["direct"], fill=COLORS["white"], width=1.1)
            insert_text(
                page,
                fitz.Rect(64, y + 12, 550, y + 38),
                f"{group['id']} | mark {mark} ({mark_evidence_count} linked labels) | {group['role']['value']}",
                9.5,
                COLORS["direct"],
                bold=True,
            )
            insert_text(
                page,
                fitz.Rect(64, y + 42, 550, y + 102),
                group_lines,
                8.7,
                COLORS["ink"],
            )
            detail = details_by_id.get(group["fabrication"].get("detail_id"))
            if metric.get("status") == "pass":
                thumbnail = fitz.Rect(570, y + 8, 766, y + 104)
                page.draw_rect(thumbnail, color=COLORS["direct"], fill=COLORS["paper"], width=0.8)
                host_y0, host_y1 = thumbnail.y0 + 15, thumbnail.y1 - 16
                host_x = thumbnail.x0 + 90
                page.draw_line((host_x, host_y0), (host_x, host_y1), color=COLORS["schedule"], width=2.0)
                solved = metric["solved_centerline"]
                host_height = metric["metric_frame"]["host_height_mm"]
                bar_y0 = host_y1 - solved["z_end_mm"] / host_height * (host_y1 - host_y0)
                bar_y1 = host_y1 - solved["z_start_mm"] / host_height * (host_y1 - host_y0)
                page.draw_line((host_x, bar_y0), (host_x, bar_y1), color=COLORS["identifier"], width=4.0)
                for endpoint in ((host_x, bar_y0), (host_x, bar_y1)):
                    page.draw_circle(endpoint, 2.8, color=COLORS["direct"], fill=COLORS["white"], width=1.0)
                page.insert_text((host_x + 9, bar_y0 + 2), f"Z {solved['z_end_mm']:g}", fontsize=5.8, color=COLORS["ink"])
                page.insert_text((host_x + 9, bar_y1 + 2), f"Z {solved['z_start_mm']:g}", fontsize=5.8, color=COLORS["ink"])
                page.insert_text((thumbnail.x0 + 8, thumbnail.y0 + 12), f"L = {solved['installed_length_each_mm']:g} mm", fontsize=6.2, color=COLORS["identifier"])
                insert_text(page, fitz.Rect(thumbnail.x0 + 4, thumbnail.y1 - 13, thumbnail.x1 - 4, thumbnail.y1 - 2), "dimension anchors + cross-view reprojection", 5.2, COLORS["muted"])
            elif detail is not None:
                thumbnail = fitz.Rect(570, y + 8, 766, y + 104)
                page.draw_rect(thumbnail, color=COLORS["candidate"], fill=COLORS["paper"], width=0.8)
                geometry = detail["geometry"]
                for line in geometry.get("lines", []):
                    start = line["start_normalized"]
                    end = line["end_normalized"]
                    p0 = fitz.Point(thumbnail.x0 + 8 + start[0] * (thumbnail.width - 16), thumbnail.y0 + 8 + start[1] * (thumbnail.height - 16))
                    p1 = fitz.Point(thumbnail.x0 + 8 + end[0] * (thumbnail.width - 16), thumbnail.y0 + 8 + end[1] * (thumbnail.height - 16))
                    page.draw_line(p0, p1, color=COLORS["direct"], width=1.0)
                for bend in geometry.get("bends", []):
                    point = bend["vertex_normalized"]
                    center = fitz.Point(thumbnail.x0 + 8 + point[0] * (thumbnail.width - 16), thumbnail.y0 + 8 + point[1] * (thumbnail.height - 16))
                    page.draw_circle(center, 1.8, color=COLORS["blocked"], fill=COLORS["blocked"], width=0.5)
                for dimension in geometry.get("dimensions", []):
                    box = dimension["bbox_normalized"]
                    center = fitz.Point(thumbnail.x0 + 8 + ((box[0] + box[2]) / 2) * (thumbnail.width - 16), thumbnail.y0 + 8 + ((box[1] + box[3]) / 2) * (thumbnail.height - 16))
                    page.insert_text(center, str(dimension["value_mm"]), fontsize=5.5, color=COLORS["candidate"])
                insert_text(page, fitz.Rect(thumbnail.x0 + 4, thumbnail.y1 - 13, thumbnail.x1 - 4, thumbnail.y1 - 2), "image-derived lines / bends / dimensions", 5.2, COLORS["muted"])
            y += 126

    insert_text(page, fitz.Rect(830, 278, 1535, 310), "DIRECT REPETITION CONSTRAINTS", 14, COLORS["ink"], bold=True)
    if not spacing:
        insert_text(page, fitz.Rect(830, 330, 1515, 390), "No native spacing expression was resolved on this page.", 10, COLORS["muted"])
    else:
        y = 326
        for constraint in spacing[:9]:
            conflict = constraint["arithmetic"]["status"] == "conflict"
            target = constraint.get("association", {}).get("rebar_group_id") or "UNKNOWN"
            color = COLORS["blocked"] if conflict else COLORS["direct"] if target != "UNKNOWN" else COLORS["candidate"]
            display_expression = constraint["source_text"].replace("х", "x").replace("Х", "x").replace("×", "x")
            page.draw_rect(fitz.Rect(830, y, 1535, y + 42), color=color, fill=COLORS["white"], width=1.0)
            insert_text(
                page,
                fitz.Rect(844, y + 8, 1518, y + 34),
                f"{display_expression} | arithmetic {constraint['arithmetic']['status'].upper()} | group {target}",
                8.5,
                color,
                bold=True,
            )
            y += 50
        if len(spacing) > 9:
            insert_text(page, fitz.Rect(844, y + 4, 1518, y + 28), f"+ {len(spacing) - 9} additional constraints in the engineering graph", 8.5, COLORS["muted"], bold=True)

    validation = program.get("validation", {})
    conflicts = len(validation.get("spacing_arithmetic_conflicts", [])) + len(validation.get("group_distribution_conflicts", []))
    footer_color = COLORS["blocked"] if conflicts else COLORS["direct"]
    page.draw_rect(fitz.Rect(48, 850, 1535, 952), color=footer_color, fill=COLORS["white"], width=1.3)
    insert_text(page, fitz.Rect(66, 868, 1515, 896), "FAIL-CLOSED REBAR BOUNDARY", 10, footer_color, bold=True)
    insert_text(
        page,
        fitz.Rect(66, 904, 1515, 940),
        f"Concrete solid required: NO | Schedule lengths used: NO | Path graph: {path_summary.get('component_count', 0)} components / {path_summary.get('metric_projected_component_count', 0)} metric projections / {path_summary.get('installed_centerline_resolved_count', 0)} resolved physical instances | Fabrication: {program.get('fabrication_status', 'unavailable').upper()} | Review conflicts: {conflicts}.",
        9,
        COLORS["ink"],
    )


def _view_label(role: str, language: str) -> str:
    labels = {
        "reinforcement_view_candidate": ("REINFORCEMENT ELEVATION", "АРМИРОВАНИЕ / ФАСАД"),
        "formwork_view_candidate": ("FORMWORK / ELEVATION", "ОПАЛУБКА / ФАСАД"),
        "section_view_candidate": ("SECTION", "СЕЧЕНИЕ"),
        "plan_view_candidate": ("PLAN", "ПЛАН"),
        "drawing_view_candidate": ("DRAWING VIEW", "ВИД ЧЕРТЕЖА"),
    }
    english, russian = labels.get(role, (role.replace("_candidate", "").replace("_", " ").upper(), role))
    return choose(language, english, russian)


def _contour_points(page: fitz.Page, contour: dict[str, Any]) -> list[fitz.Point]:
    match = re.fullmatch(r"drawing\[(\d+)\]", (contour.get("primitive_refs") or [""])[0])
    if match is None:
        return []
    drawings = page.get_drawings()
    drawing_index = int(match.group(1))
    if drawing_index >= len(drawings):
        return []
    points: list[fitz.Point] = []
    for item in drawings[drawing_index].get("items", []):
        if item[0] == "l":
            segment = [item[1], item[2]]
        elif item[0] == "re":
            rect = fitz.Rect(item[1])
            segment = [rect.tl, rect.tr, rect.br, rect.bl, rect.tl]
        elif item[0] == "qu":
            quad = item[1]
            segment = [quad.ul, quad.ur, quad.lr, quad.ll, quad.ul]
        elif item[0] == "c":
            p0, p1, p2, p3 = item[1:5]
            segment = []
            for step in range(9):
                t = step / 8
                u = 1 - t
                segment.append(
                    fitz.Point(
                        u**3 * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t**3 * p3.x,
                        u**3 * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t**3 * p3.y,
                    )
                )
        else:
            continue
        for point in segment:
            if not points or point.distance_to(points[-1]) > 0.2:
                points.append(fitz.Point(point))
    if len(points) >= 3 and points[0].distance_to(points[-1]) > 2.0:
        return []
    return points[:-1] if len(points) >= 2 and points[0].distance_to(points[-1]) <= 2.0 else points


def _draw_hatched_polygon(page: fitz.Page, points: list[fitz.Point], color: tuple[float, float, float], oc: int) -> None:
    if len(points) < 3:
        return
    page.draw_polyline(points, color=color, fill=color, width=1.8, closePath=True, stroke_opacity=0.92, fill_opacity=0.10, oc=oc)
    box = fitz.Rect(points[0], points[0])
    for point in points[1:]:
        box.include_point(point)
    step = max(10.0, min(box.width, box.height) / 9)
    intercept = box.y0 - box.x1
    while intercept <= box.y1 - box.x0:
        intersections: list[fitz.Point] = []
        for start, end in zip(points, [*points[1:], points[0]]):
            left = start.y - start.x - intercept
            right = end.y - end.x - intercept
            if left == 0:
                intersections.append(start)
            if left * right < 0:
                fraction = left / (left - right)
                intersections.append(fitz.Point(start.x + fraction * (end.x - start.x), start.y + fraction * (end.y - start.y)))
        intersections.sort(key=lambda point: (point.x, point.y))
        for start, end in zip(intersections[0::2], intersections[1::2]):
            page.draw_line(start, end, color=color, width=0.55, stroke_opacity=0.35, oc=oc)
        intercept += step


def _selected_object_contours(source_page: fitz.Page, record: dict[str, Any]) -> list[tuple[dict[str, Any], list[fitz.Point], str]]:
    engineering = record["engineering_graph"]
    selected_profiles = {
        item["id"]: item
        for item in engineering.get("solid_evidence", {}).get("metric_profiles", [])
        if item["id"] in set(engineering.get("solid_evidence", {}).get("selected_profile_ids", []))
    }
    selected: list[tuple[dict[str, Any], list[fitz.Point], str]] = []
    used_contours: set[str] = set()
    for contour in engineering.get("calculation_contours", []):
        points = [fitz.Point(*point) for point in contour.get("polygon_points_display", [])]
        if len(points) >= 4 and contour.get("closure_validation", {}).get("status") == "pass":
            selected.append((contour, points, "CALCULATION CONTOUR"))
    for profile in selected_profiles.values():
        points = [fitz.Point(*point) for point in profile.get("points_display", [])]
        if len(points) >= 3:
            selected.append((profile, points, "CALCULATION CONTOUR"))
    contour_by_id = {item["id"]: item for item in record["contours"]}
    for view in engineering.get("view_hypotheses", []):
        if view["role_hypothesis"] == "table_or_grid_candidate" or _table_like_view(record, view):
            continue
        if view.get("features", {}).get("accepted_dimensions", 0) < 1:
            continue
        view_box = fitz.Rect(view["bbox_display"])
        candidates = []
        for contour_id in view.get("contour_refs", []):
            contour = contour_by_id.get(contour_id)
            if contour is None or not contour.get("closed") or contour_id in used_contours:
                continue
            area = fitz.Rect(contour["bbox_display"]).get_area()
            if 0.005 * view_box.get_area() <= area <= 0.80 * view_box.get_area():
                candidates.append((area, contour))
        for _, contour in sorted(candidates, reverse=True, key=lambda item: item[0])[:2]:
            points = _contour_points(source_page, contour)
            if len(points) >= 3:
                selected.append((contour, points, "OBJECT CONTOUR" if selected_profiles else "OBJECT CONTOUR CANDIDATE"))
                used_contours.add(contour["id"])
    return selected[:16]


def _page_declarations(declarations: dict[str, Any], page_number: int) -> dict[str, Any]:
    return next(
        (item for item in declarations.get("pages", []) if item["page"] == page_number),
        {"concrete": [], "reinforcement": [], "native_text_available": False},
    )


def _declared_cell_boxes(declaration: dict[str, Any]) -> list[fitz.Rect]:
    """Return exact declared-total cells and never a broad search region."""

    boxes = []
    for key in ("total_cell_bbox_display", "cell_bbox_display"):
        if declaration.get(key):
            boxes.append(fitz.Rect(declaration[key]))
    for evidence in declaration.get("evidence_cells", []):
        if evidence.get("bbox_display"):
            boxes.append(fitz.Rect(evidence["bbox_display"]))
    boxes.extend(fitz.Rect(box) for box in declaration.get("cell_bboxes_display", []))
    component_cells_are_exact = str(declaration.get("basis", "")).endswith("_component_cell_ocr_sum")
    for component in declaration.get("components", []):
        if component.get("total_cell_bbox_display"):
            boxes.append(fitz.Rect(component["total_cell_bbox_display"]))
        elif component.get("cell_bbox_display"):
            boxes.append(fitz.Rect(component["cell_bbox_display"]))
        elif component_cells_are_exact and component.get("bbox_display"):
            boxes.append(fitz.Rect(component["bbox_display"]))
        elif component.get("label") in {"overall", "total", "declared_total", "class_subtotal"} and component.get("bbox_display"):
            boxes.append(fitz.Rect(component["bbox_display"]))
    unique = {}
    for box in boxes:
        if box.is_empty:
            continue
        unique.setdefault(tuple(round(value, 2) for value in box), box)
    return list(unique.values())


def _comparison_status(declared: float | None, calculated: float | None) -> str:
    if calculated is None:
        return "calculation_unavailable"
    if declared is None:
        return "declaration_unavailable"
    tolerance = max(1e-6, abs(declared) * 1e-6)
    return "match" if abs(calculated - declared) <= tolerance else "discrepancy"


def _comparison_color(status: str) -> tuple[float, float, float]:
    if status == "match":
        return COLORS["direct"]
    if status == "discrepancy":
        return COLORS["blocked"]
    return COLORS["unknown"]


def _quantity_comparisons(declared: dict[str, Any], engineering: dict[str, Any]) -> dict[str, dict[str, Any]]:
    concrete_declared = declared["concrete"][0]["value"] if len(declared.get("concrete", [])) == 1 else None
    steel_declared = declared["reinforcement"][0]["value"] if len(declared.get("reinforcement", [])) == 1 else None
    quantities = engineering.get("quantities", [])
    concrete_calculated = quantities[0].get("net_concrete_m3") if quantities else None
    takeoff = engineering.get("rebar_program", {}).get("drawing_detail_takeoff", {})
    steel_calculated = takeoff.get("totals", {}).get("mass_kg") if takeoff.get("status") == "resolved_drawing_takeoff" else None

    def comparison(declared_value: float | None, calculated_value: float | None) -> dict[str, Any]:
        return {
            "declared": declared_value,
            "calculated": calculated_value,
            "delta": None if declared_value is None or calculated_value is None else calculated_value - declared_value,
            "status": _comparison_status(declared_value, calculated_value),
        }

    return {
        "concrete": comparison(concrete_declared, concrete_calculated),
        "steel": comparison(steel_declared, steel_calculated),
    }


def _draw_declared_schedule_evidence(
    page: fitz.Page,
    declared: dict[str, Any],
    ocg: int,
    discrepancy_ocg: int,
    language: str,
    comparisons: dict[str, dict[str, Any]],
) -> None:
    labels = {
        "concrete": choose(language, "DECLARED CONCRETE", "БЕТОН ПО ВЕДОМОСТИ"),
        "reinforcement": choose(language, "DECLARED STEEL", "СТАЛЬ ПО ВЕДОМОСТИ"),
    }
    for kind in ("concrete", "reinforcement"):
        comparison_kind = "concrete" if kind == "concrete" else "steel"
        for declaration in declared.get(kind, []):
            exact_boxes = _declared_cell_boxes(declaration)
            for box_index, box in enumerate(exact_boxes):
                page.draw_rect(
                    box + (-2, -2, 2, 2),
                    color=COLORS["declared"],
                    fill=COLORS["declared"],
                    width=2.0,
                    stroke_opacity=0.95,
                    fill_opacity=0.10,
                    oc=ocg,
                )
                if comparisons[comparison_kind]["status"] == "discrepancy":
                    page.draw_rect(
                        box + (-6, -6, 6, 6),
                        color=COLORS["blocked"],
                        width=2.4,
                        stroke_opacity=0.96,
                        oc=discrepancy_ocg,
                    )
                if box_index == 0:
                    tag = fitz.Rect(box.x0, max(0, box.y0 - 16), min(page.rect.x1, box.x0 + 180), box.y0)
                    insert_text(page, tag, labels[kind], 6.5, COLORS["declared"], bold=True)
            if not exact_boxes and declaration.get("bbox_display"):
                broad = fitz.Rect(declaration["bbox_display"])
                page.draw_rect(
                    broad,
                    color=COLORS["unknown"],
                    width=1.4,
                    stroke_opacity=0.72,
                    oc=ocg,
                )
                tag = fitz.Rect(broad.x0, max(0, broad.y0 - 16), min(page.rect.x1, broad.x0 + 220), broad.y0)
                insert_text(
                    page,
                    tag,
                    choose(language, "DECLARED TOTAL CELL NOT LOCATED", "ЯЧЕЙКА ИТОГА НЕ НАЙДЕНА"),
                    6.5,
                    COLORS["unknown"],
                    bold=True,
                )
    for candidate in declared.get("unresolved_table_candidates", []):
        if not candidate.get("bbox_display"):
            continue
        box = fitz.Rect(candidate["bbox_display"])
        page.draw_rect(box, color=COLORS["candidate"], width=1.5, stroke_opacity=0.60, oc=ocg)
        tag = fitz.Rect(box.x0, max(0, box.y0 - 16), min(page.rect.x1, box.x0 + 180), box.y0)
        insert_text(page, tag, choose(language, "TOTAL NOT READ", "ИТОГ НЕ ПРОЧИТАН"), 6.5, COLORS["candidate"], bold=True)


def _table_like_view(record: dict[str, Any], view: dict[str, Any]) -> bool:
    box = fitz.Rect(view["bbox_display"])
    schedule_tokens = 0
    for role in record.get("text_roles", []):
        if role.get("resolved_role") != "schedule_or_table_value":
            continue
        token_box = fitz.Rect(role["bbox_display"])
        center = fitz.Point((token_box.x0 + token_box.x1) / 2, (token_box.y0 + token_box.y1) / 2)
        schedule_tokens += center in box
    return schedule_tokens >= 3 and view.get("features", {}).get("orthogonal_axis_ratio", 0.0) >= 0.55


def _quantity_text(value: float | None, unit: str, digits: int = 3, language: str = DEFAULT_LANGUAGE) -> str:
    return quantity_text(language, value, unit, digits)


def _takeoff_mass_text(language: str, row: dict[str, Any], digits: int = 1) -> str:
    strict = row.get("mass_kg")
    approximate = row.get("convention_dependent_mass_kg")
    if strict is not None:
        return quantity_text(language, strict, "kg", digits)
    if approximate is not None:
        return f"≈ {quantity_text(language, approximate, 'kg', digits)}"
    return quantity_text(language, None, "kg", digits)


def draw_engineer_overlay(
    output: fitz.Document,
    display: fitz.Document,
    source_page: fitz.Page,
    record: dict[str, Any],
    declarations: dict[str, Any],
    index: int,
    ocgs: dict[str, int],
    language: str,
    understanding_seconds: float,
) -> None:
    sidebar = max(340.0, source_page.rect.width * 0.23)
    page = output.new_page(width=source_page.rect.width + sidebar, height=source_page.rect.height)
    page.show_pdf_page(fitz.Rect(0, 0, source_page.rect.width, source_page.rect.height), display, index, oc=ocgs["SOURCE"])
    engineering = record["engineering_graph"]
    program = engineering.get("rebar_program", {})
    graph = program.get("physical_path_graph", {})
    object_graph = engineering.get("object_instance_graph", {})
    views_by_id = {item["id"]: item for item in engineering.get("view_hypotheses", [])}
    native_details = program.get("native_vector_detail_linking", {})
    declared = _page_declarations(declarations, record["page"])
    comparisons = _quantity_comparisons(declared, engineering)
    _draw_declared_schedule_evidence(
        page,
        declared,
        ocgs["DECLARED_SCHEDULE"],
        ocgs["DISCREPANCIES"],
        language,
        comparisons,
    )
    linked_marks_by_instance: dict[str, set[str]] = {}
    for association in native_details.get("placement_associations", []):
        if association.get("state") != "accepted":
            continue
        for binding in association.get("object_instance_bindings", []):
            linked_marks_by_instance.setdefault(binding["object_instance_id"], set()).add(str(binding["mark"]))
    designations = {
        item["object_instance_id"]: item.get("designation")
        for item in native_details.get("variant_context", {}).get("instances", [])
    }

    for object_index, instance in enumerate(object_graph.get("instances", []), start=1):
        color = GROUP_COLORS[(object_index - 1) % len(GROUP_COLORS)]
        member_boxes = [fitz.Rect(views_by_id[view_id]["bbox_display"]) for view_id in instance["view_ids"] if view_id in views_by_id]
        for member_index, rect in enumerate(member_boxes):
            page.draw_rect(rect, color=color, width=3.1, stroke_opacity=0.94, oc=ocgs["OBJECT_INSTANCES"])
            tag = fitz.Rect(rect.x0, max(0, rect.y0 - 28), min(source_page.rect.x1, rect.x0 + 145), max(0, rect.y0 - 17))
            role = choose(language, "PRIMARY", "ОСНОВНОЙ ВИД") if member_index == 0 else choose(language, "SECTION", "СЕЧЕНИЕ")
            insert_text(page, tag, f"{choose(language, 'OBJECT', 'ОБЪЕКТ')} {object_index} | {role}", 7.0, color, bold=True)
            marks = sorted(linked_marks_by_instance.get(instance["id"], set()), key=lambda value: int(value))
            if marks:
                designation = designations.get(instance["id"]) or instance["id"]
                detail_tag = fitz.Rect(rect.x0, max(0, rect.y0 - 40), min(source_page.rect.x1, rect.x0 + 220), max(0, rect.y0 - 29))
                insert_text(page, detail_tag, f"{designation} | {choose(language, 'LINKED', 'СВЯЗАНО')} M{' M'.join(marks)}", 6.5, color, bold=True)
    for shared in object_graph.get("shared_supporting_views", []):
        view = views_by_id.get(shared["view_id"])
        if view is None:
            continue
        rect = fitz.Rect(view["bbox_display"])
        page.draw_rect(rect, color=COLORS["candidate"], width=2.2, stroke_opacity=0.62, oc=ocgs["OBJECT_INSTANCES"])
        insert_text(page, fitz.Rect(rect.x0, max(0, rect.y0 - 28), min(source_page.rect.x1, rect.x0 + 230), max(0, rect.y0 - 17)), choose(language, "DIMENSIONED VIEW - ITEM LINK NEEDS REVIEW", "РАЗМЕРНЫЙ ВИД - УТОЧНИТЬ ИЗДЕЛИЕ"), 6.7, COLORS["candidate"], bold=True)

    view_number = 0
    for view in engineering.get("view_hypotheses", []):
        if view["role_hypothesis"] == "table_or_grid_candidate" or _table_like_view(record, view):
            continue
        view_number += 1
        rect = fitz.Rect(view["bbox_display"])
        page.draw_rect(rect, color=COLORS["view"], width=1.5, stroke_opacity=0.58, oc=ocgs["DETECTED_VIEWS"])
        tag = fitz.Rect(rect.x0, max(0, rect.y0 - 17), min(rect.x1, rect.x0 + 175), rect.y0)
        insert_text(page, tag, f"V{view_number}  {_view_label(view['role_hypothesis'], language)}", 6.7, COLORS["view"], bold=True)

    for relation in engineering.get("view_frame_graph", {}).get("relations", []):
        trace = relation.get("trace", {})
        span = trace.get("virtual_span_display")
        if not span:
            box = fitz.Rect(trace.get("bbox_display", []))
            if box.is_empty:
                continue
            span = [[box.x0, (box.y0 + box.y1) / 2], [box.x1, (box.y0 + box.y1) / 2]] if trace.get("orientation") == "horizontal" else [[(box.x0 + box.x1) / 2, box.y0], [(box.x0 + box.x1) / 2, box.y1]]
        start, end = fitz.Point(*span[0]), fitz.Point(*span[1])
        page.draw_line(start, end, color=COLORS["identifier"], width=2.4, stroke_opacity=0.92, oc=ocgs["CUTTING_PLANES"])
        section_view = views_by_id.get(relation.get("section_view_id"))
        if section_view is not None:
            section_box = fitz.Rect(section_view["bbox_display"])
            page.draw_rect(section_box, color=COLORS["identifier"], width=1.2, stroke_opacity=0.44, oc=ocgs["CUTTING_PLANES"])
        tag_box = fitz.Rect(min(start.x, end.x), max(0, min(start.y, end.y) - 16), min(source_page.rect.x1, min(start.x, end.x) + 150), max(start.y, end.y))
        insert_text(page, tag_box, f"{relation.get('section_label', '?')}  {choose(language, 'SECTION LINKED', 'СЕЧЕНИЕ СВЯЗАНО')}", 6.4, COLORS["identifier"], bold=True)

    drawings = source_page.get_drawings()
    contours_by_id = {
        str(item["id"]): item for item in engineering.get("contour_hypotheses", [])
    }
    matched_by_child: Counter[str] = Counter()
    correspondence_records = (
        engineering.get("view_frame_graph", {})
        .get("shared_coordinate_system", {})
        .get("contour_correspondence", {})
        .get("records", [])
    )
    for correspondence in correspondence_records:
        if correspondence.get("state") != "accepted" or not correspondence.get("selected"):
            continue
        selected = correspondence["selected"]
        for primitive_ref in [
            *selected.get("parent_geometry_refs", []),
            *selected.get("child_geometry_refs", []),
        ]:
            match = re.fullmatch(r"drawing\[(\d+)\]\.item\[(\d+)\]\.segment\[(\d+)\]", str(primitive_ref))
            if match is None:
                continue
            drawing_index, item_index, _ = map(int, match.groups())
            if drawing_index >= len(drawings) or item_index >= len(drawings[drawing_index].get("items", [])):
                continue
            item = drawings[drawing_index]["items"][item_index]
            if item[0] == "l":
                page.draw_line(item[1], item[2], color=COLORS["direct"], width=3.0, stroke_opacity=0.88, oc=ocgs["CROSS_VIEW_MATCHES"])
        child_contour_id = selected.get("child_contour_id")
        child_contour = contours_by_id.get(str(child_contour_id)) if child_contour_id else None
        if child_contour is not None:
            for segment in child_contour.get("segments_display", []) or []:
                page.draw_line(
                    fitz.Point(*segment["start_display"]),
                    fitz.Point(*segment["end_display"]),
                    color=COLORS["direct"],
                    width=2.7,
                    stroke_opacity=0.82,
                    oc=ocgs["CROSS_VIEW_MATCHES"],
                )
        matched_by_child[str(correspondence.get("child_view_id"))] += 1
    for child_view_id, count in matched_by_child.items():
        child_view = views_by_id.get(child_view_id)
        if child_view is None:
            continue
        rect = fitz.Rect(child_view["bbox_display"])
        insert_text(
            page,
            fitz.Rect(rect.x0, max(0, rect.y0 - 29), min(rect.x1, rect.x0 + 210), max(0, rect.y0 - 18)),
            choose(
                language,
                f"SECTION PATH MATCHED ({count})",
                f"КОНТУР СЕЧЕНИЯ СОВПАЛ ({count})",
            ),
            6.4,
            COLORS["direct"],
            bold=True,
        )

    for contour, points, label in _selected_object_contours(source_page, record):
        _draw_hatched_polygon(page, points, COLORS["contour"], ocgs["OBJECT_CONTOURS"])
        rect = fitz.Rect(contour["bbox_display"])
        tag = fitz.Rect(rect.x0, max(0, rect.y0 - 15), min(source_page.rect.x1, rect.x0 + 150), rect.y0)
        localized_label = {
            "CALCULATION CONTOUR": choose(language, "CONTOUR USED FOR QUANTITY", "КОНТУР ДЛЯ РАСЧЕТА ОБЪЕМА"),
            "OBJECT CONTOUR": choose(language, "OBJECT CONTOUR", "КОНТУР ОБЪЕКТА"),
            "OBJECT CONTOUR CANDIDATE": choose(language, "POSSIBLE OBJECT OUTLINE - REVIEW", "ВОЗМОЖНЫЙ КОНТУР - ПРОВЕРИТЬ"),
        }.get(label, label)
        insert_text(page, tag, localized_label, 6.4, COLORS["contour"], bold=True)

    fabrication_details = program.get("fabrication_details", [])
    native_detail_list = native_details.get("details", [])
    detail_colors = _detail_color_assignments(fabrication_details, native_detail_list)
    group_by_id = {item["id"]: item for item in program.get("groups", [])}
    color_by_group = {group_id: GROUP_COLORS[index % len(GROUP_COLORS)] for index, group_id in enumerate(sorted(group_by_id))}
    for detail in fabrication_details:
        detail_id = str(detail.get("id") or detail.get("group_id") or "")
        if detail.get("group_id") and detail_id in detail_colors["fabrication"]:
            color_by_group[detail["group_id"]] = detail_colors["fabrication"][detail_id]
    fragments = sorted(
        graph.get("fragments", []),
        key=lambda item: (bool(item.get("resolved_group_ids")), bool(item.get("repetition_group_ids")), item.get("candidate_score", 0.0)),
        reverse=True,
    )
    fragment_state_counts = {"resolved": 0, "candidate": 0, "rejected": 0}
    fragment_shapes: dict[tuple[Any, ...], fitz.Shape] = {}
    for fragment in fragments:
        group_ids = fragment.get("resolved_group_ids", [])
        resolved = bool(group_ids)
        candidate = bool(fragment.get("mark_hypotheses")) or (
            bool(fragment.get("repetition_group_ids")) and fragment.get("candidate_score", 0) >= 0.54
        )
        state = "resolved" if resolved else "candidate" if candidate else "rejected"
        fragment_state_counts[state] += 1
        points = [fitz.Point(*point) for point in fragment["geometry"]["points_display"]]
        if len(points) < 2:
            continue
        color = color_by_group.get(group_ids[0], COLORS["candidate"]) if resolved else COLORS["candidate"] if candidate else COLORS["blocked"]
        width = 2.6 if resolved else 0.9 if candidate else 0.55
        opacity = 0.92 if resolved else 0.50 if candidate else 0.28
        ocg = ocgs["REBAR_DETECTED"] if state != "rejected" else ocgs["REBAR_REJECTED"]
        style = (color, width, opacity, ocg)
        fragment_shapes.setdefault(style, page.new_shape()).draw_polyline(points)
    for (color, width, opacity, ocg), shape in fragment_shapes.items():
        shape.finish(color=color, width=width, stroke_opacity=opacity, closePath=False, oc=ocg)
        shape.commit()

    for section in engineering.get("section_rebar_observations", {}).get("sections", []):
        for candidate in section.get("candidates", []):
            state = candidate.get("state", "candidate")
            color = COLORS["direct"] if state == "resolved" else COLORS["candidate"]
            drawn = False
            for item in candidate.get("geometry_items", []):
                if item["kind"] == "line":
                    page.draw_line(
                        fitz.Point(*item["start_display"]),
                        fitz.Point(*item["end_display"]),
                        color=color,
                        width=2.0 if state == "resolved" else 1.15,
                        stroke_opacity=0.90 if state == "resolved" else 0.62,
                        oc=ocgs["REBAR_DETECTED"],
                    )
                    drawn = True
            if not drawn:
                page.draw_rect(
                    fitz.Rect(candidate["bbox_display"]),
                    color=color,
                    width=1.1,
                    stroke_opacity=0.70,
                    oc=ocgs["REBAR_DETECTED"],
                )
        for token in section.get("rejected_numeric_tokens", []):
            page.draw_rect(
                fitz.Rect(token["bbox_display"]) + (-1, -1, 1, 1),
                color=COLORS["blocked"],
                width=0.75,
                stroke_opacity=0.68,
                oc=ocgs["REBAR_REJECTED"],
            )

    hypotheses = {item["id"]: item for item in program.get("identity_hypotheses", [])}
    for detail in fabrication_details:
        source_box = fitz.Rect(detail.get("source", {}).get("bbox_display", []))
        if source_box.is_empty:
            continue
        detail_id = str(detail.get("id") or detail.get("group_id") or "")
        color = detail_colors["fabrication"].get(detail_id, color_by_group.get(detail.get("group_id"), COLORS["identifier"]))
        page.draw_rect(source_box, color=color, fill=color, width=2.0, stroke_opacity=0.95, fill_opacity=0.08, oc=ocgs["DETAIL_LINKS"])
        mark = detail.get("mark") or "?"
        insert_text(page, fitz.Rect(source_box.x0, max(0, source_box.y0 - 16), min(source_page.rect.x1, source_box.x0 + 130), source_box.y0), f"{choose(language, 'DETAIL', 'ДЕТАЛЬ')} M{mark}", 6.5, color, bold=True)
        for association in program.get("identity_associations", []):
            if association.get("group_id") != detail.get("group_id") or association.get("status") != "accepted":
                continue
            hypothesis = hypotheses.get(association.get("hypothesis_id"), {})
            target_box = fitz.Rect(hypothesis.get("bbox_display", []))
            if not target_box.is_empty:
                page.draw_rect(target_box, color=color, fill=color, width=1.5, stroke_opacity=0.92, fill_opacity=0.10, oc=ocgs["DETAIL_LINKS"])
            leader_trace = association.get("leader_trace", {})
            for segment in leader_trace.get("segments", []):
                page.draw_line(
                    fitz.Point(*segment["start"]),
                    fitz.Point(*segment["end"]),
                    color=color,
                    width=1.5,
                    stroke_opacity=0.90,
                    oc=ocgs["DETAIL_LINKS"],
                )
            for terminal in leader_trace.get("terminals", []):
                page.draw_circle(fitz.Point(*terminal), 3.2, color=color, fill=color, width=0.8, fill_opacity=0.72, oc=ocgs["DETAIL_LINKS"])

    native_by_id = {item["id"]: item for item in native_detail_list}
    for detail in native_by_id.values():
        box = fitz.Rect(detail["bbox_display"])
        color = detail_colors["native"].get(detail["id"], COLORS["identifier"])
        page.draw_rect(box, color=color, fill=color, width=1.8, stroke_opacity=0.92, fill_opacity=0.07, oc=ocgs["DETAIL_LINKS"])
        insert_text(page, fitz.Rect(box.x0, max(0, box.y0 - 15), min(source_page.rect.x1, box.x0 + 150), box.y0), f"{choose(language, 'DETAIL', 'ДЕТАЛЬ')} M{detail['mark_display']}", 6.3, color, bold=True)
    for association in native_details.get("placement_associations", []):
        detail = native_by_id.get(association["detail_id"])
        if detail is None:
            continue
        target_box = fitz.Rect(association["target_mark_bbox_display"])
        color = detail_colors["native"].get(detail["id"], COLORS["identifier"])
        accepted = association.get("state") == "accepted"
        page.draw_rect(
            target_box,
            color=color,
            fill=color,
            width=1.7 if accepted else 1.0,
            stroke_opacity=0.94 if accepted else 0.58,
            fill_opacity=0.12 if accepted else 0.05,
            oc=ocgs["DETAIL_LINKS"],
        )
        for segment in association.get("leader_trace", {}).get("segments", []):
            page.draw_line(
                fitz.Point(*segment["start"]),
                fitz.Point(*segment["end"]),
                color=color,
                width=1.5 if accepted else 0.9,
                stroke_opacity=0.90 if accepted else 0.48,
                oc=ocgs["DETAIL_LINKS"],
            )
        for attachment in association.get("path_attachments", []):
            points = [fitz.Point(*point) for point in attachment.get("fragment_points_display", [])]
            if len(points) >= 2:
                page.draw_polyline(points, color=color, width=3.2 if accepted else 1.5, stroke_opacity=0.94 if accepted else 0.48, oc=ocgs["DETAIL_LINKS"])
            terminal = fitz.Point(*attachment["terminal_display"])
            page.draw_circle(terminal, 3.5 if accepted else 2.4, color=color, fill=color, width=0.8, fill_opacity=0.72 if accepted else 0.40, oc=ocgs["DETAIL_LINKS"])

    for dimension in record["dimensions"]:
        if dimension.status == "accepted":
            page.draw_line(dimension.measured_points[0], dimension.measured_points[1], color=COLORS["direct"], width=1.35, stroke_opacity=0.72, oc=ocgs["DIMENSIONS"])
    for ownership in engineering.get("dimension_ownership", {}).get("attachments", []):
        color = COLORS["direct"] if ownership.get("status") == "accepted" else COLORS["candidate"]
        for endpoint in ownership.get("measured_endpoints", []):
            if endpoint.get("state") == "unknown":
                continue
            point = fitz.Point(*endpoint["point_display"])
            page.draw_circle(
                point,
                2.4 if ownership.get("status") == "accepted" else 1.7,
                color=color,
                fill=color if ownership.get("status") == "accepted" else None,
                width=0.9,
                stroke_opacity=0.80,
                fill_opacity=0.45,
                oc=ocgs["DIMENSIONS"],
            )
    metric_equations = engineering.get("metric_equation_graph", {})
    for equation in metric_equations.get("constraints", []):
        chain = equation.get("chain_attachment") or {}
        if not chain:
            continue
        accepted = equation.get("status") == "accepted"
        color = COLORS["identifier"] if accepted else COLORS["candidate"]
        baseline = chain.get("baseline_display", [])
        if len(baseline) == 2:
            page.draw_line(
                fitz.Point(*baseline[0]),
                fitz.Point(*baseline[1]),
                color=color,
                width=2.1 if accepted else 1.2,
                stroke_opacity=0.82 if accepted else 0.48,
                oc=ocgs["DIMENSIONS"],
            )
        for extension in chain.get("extension_lines_display", []):
            if len(extension) == 2:
                page.draw_line(
                    fitz.Point(*extension[0]),
                    fitz.Point(*extension[1]),
                    color=color,
                    width=1.1,
                    stroke_opacity=0.72,
                    oc=ocgs["DIMENSIONS"],
                )
        page.draw_rect(
            fitz.Rect(equation["bbox_display"]) + (-1.5, -1.5, 1.5, 1.5),
            color=color,
            width=1.0,
            stroke_opacity=0.82 if accepted else 0.48,
            oc=ocgs["DIMENSIONS"],
        )

    x0 = source_page.rect.width
    page.draw_rect(fitz.Rect(x0, 0, page.rect.x1, page.rect.y1), color=COLORS["paper"], fill=COLORS["paper"], width=0)
    margin = 22
    insert_text(page, fitz.Rect(x0 + margin, 24, page.rect.x1 - margin, 58), choose(language, "DRAWING UNDERSTANDING", "ПОНИМАНИЕ ЧЕРТЕЖА"), 16, COLORS["ink"], bold=True)
    subtitle = choose(
        language,
        "Colored evidence comes from the drawing. Blue schedule cells are read only after the drawing calculation is locked.",
        "Цветные данные получены из чертежа. Синие ячейки ведомости читаются только после фиксации расчета.",
    )
    insert_text(page, fitz.Rect(x0 + margin, 62, page.rect.x1 - margin, 124), f"{subtitle}\n{duration_text(language, understanding_seconds)}", 8.2, COLORS["muted"], bold=True)
    legend = choose(
        language,
        f"THICK OUTLINE  views provisionally assigned to one item\nCYAN  detected drawing view\nMAGENTA HATCH  contour used for quantity or review\nGREEN THICK EDGE  section path matches the parent cut\nPURPLE DIMENSION CHAIN  equation checked and bound to a view\nONE DETAIL - ONE COLOR  detail, native leader, callout and rebar path\nORANGE LINE  rebar link needs review\nBLUE CELL  declared schedule evidence\nRED (layer off)  detection rejected and not used\nRebar traces: {fragment_state_counts['resolved']} linked / {fragment_state_counts['candidate']} review / {fragment_state_counts['rejected']} rejected",
        f"ТОЛСТЫЙ КОНТУР  виды предварительно отнесены к одному изделию\nГОЛУБОЙ  найденный вид чертежа\nПУРПУРНАЯ ШТРИХОВКА  контур для расчета или проверки\nТОЛСТАЯ ЗЕЛЕНАЯ ГРАНЬ  контур сечения совпал с разрезом\nФИОЛЕТОВАЯ РАЗМЕРНАЯ ЦЕПОЧКА  формула проверена и связана с видом\nОДНА ДЕТАЛЬ - ОДИН ЦВЕТ  деталь, штатный вынос, марка и линия арматуры\nОРАНЖЕВАЯ ЛИНИЯ  связь арматуры требует проверки\nСИНЯЯ ЯЧЕЙКА  итог из ведомости\nКРАСНЫЙ (слой выключен)  распознавание отклонено и не используется\nЛинии арматуры: {fragment_state_counts['resolved']} связано / {fragment_state_counts['candidate']} проверить / {fragment_state_counts['rejected']} отклонено",
    )
    insert_text(page, fitz.Rect(x0 + margin, 126, page.rect.x1 - margin, 264), legend, 7.3, COLORS["ink"], bold=True)

    concrete_comparison = comparisons["concrete"]
    steel_comparison = comparisons["steel"]
    concrete_declared = concrete_comparison["declared"]
    concrete_calculated = concrete_comparison["calculated"]
    steel_declared = steel_comparison["declared"]
    reinforcement = engineering.get("reinforcement_quantities") or {}
    installed = reinforcement.get("total_placed_centerline_m")
    fabrication = reinforcement.get("fabrication_length_m")
    partial_fabrication = reinforcement.get("resolved_fabrication_length_m")
    drawing_takeoff = program.get("drawing_detail_takeoff", {})
    estimated_takeoff = program.get("estimated_quantity_takeoff", {})
    takeoff_resolved = drawing_takeoff.get("status") == "resolved_drawing_takeoff"
    delta = concrete_comparison["delta"]
    y = 270
    quantity_height = 480 if estimated_takeoff.get("families") and not takeoff_resolved else (440 if takeoff_resolved else 350)
    page.draw_rect(fitz.Rect(x0 + margin, y, page.rect.x1 - margin, y + quantity_height), color=COLORS["view"], fill=COLORS["white"], width=1.2)
    insert_text(page, fitz.Rect(x0 + margin + 12, y + 12, page.rect.x1 - margin - 12, y + 36), choose(language, "QUANTITY CHECK", "ПРОВЕРКА ОБЪЕМОВ"), 10, COLORS["ink"], bold=True)
    concrete_rows = [
        (choose(language, "Concrete declared", "Бетон по ведомости"), _quantity_text(concrete_declared, "m3", language=language)),
        (choose(language, "Concrete from drawing", "Бетон по чертежу"), _quantity_text(concrete_calculated, "m3", language=language)),
        (choose(language, "Drawing - schedule", "Чертеж - ведомость"), quantity_text(language, delta, "m3", 4, signed=True) if delta is not None else choose(language, "NOT COMPARABLE", "НЕЛЬЗЯ СРАВНИТЬ")),
    ]
    row_y = y + 46
    for label, value in concrete_rows:
        insert_text(page, fitz.Rect(x0 + margin + 12, row_y, x0 + sidebar * 0.56, row_y + 24), label, 7.7, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(x0 + sidebar * 0.56, row_y, page.rect.x1 - margin - 10, row_y + 24), value, 8.1, COLORS["ink"], bold=True)
        row_y += 25
    if takeoff_resolved:
        row_y += 3
        insert_text(page, fitz.Rect(x0 + margin + 12, row_y, page.rect.x1 - margin - 12, row_y + 20), choose(language, "REBAR FROM DRAWING", "АРМАТУРА ПО ЧЕРТЕЖУ"), 8.3, COLORS["direct"], bold=True)
        row_y += 22
        left = x0 + margin + 12
        right = page.rect.x1 - margin - 12
        widths = (0.14, 0.25, 0.17, 0.23, 0.21)
        positions = [left]
        for width in widths:
            positions.append(positions[-1] + (right - left) * width)
        headers = (
            choose(language, "DIA, mm", "ДИАМ., мм"),
            choose(language, "MARKS", "МАРКИ"),
            choose(language, "BARS", "ШТ."),
            choose(language, "LENGTH", "ДЛИНА"),
            choose(language, "MASS", "МАССА"),
        )
        page.draw_rect(fitz.Rect(left, row_y, right, row_y + 22), color=COLORS["view"], fill=COLORS["paper"], width=0.8)
        for column, header in enumerate(headers):
            insert_text(page, fitz.Rect(positions[column] + 3, row_y + 4, positions[column + 1] - 2, row_y + 20), header, 6.2, COLORS["muted"], bold=True)
        row_y += 22
        for row in drawing_takeoff.get("by_diameter", []):
            values = (
                f"{row['diameter_mm']:g}",
                ",".join(row["marks"]),
                str(row["physical_bar_count"]),
                quantity_text(language, row["total_length_m"], "m", 3),
                _takeoff_mass_text(language, row),
            )
            page.draw_rect(fitz.Rect(left, row_y, right, row_y + 23), color=COLORS["schedule"], fill=COLORS["white"], width=0.45)
            for column, value in enumerate(values):
                insert_text(page, fitz.Rect(positions[column] + 3, row_y + 5, positions[column + 1] - 2, row_y + 21), value, 6.6, COLORS["ink"], bold=column in {0, 4})
            row_y += 23
        totals = drawing_takeoff["totals"]
        values = (
            choose(language, "ALL", "ИТОГО"),
            "3-10",
            str(totals["physical_bar_count"]),
            quantity_text(language, totals["fabrication_length_m"], "m", 3),
            _takeoff_mass_text(language, totals),
        )
        page.draw_rect(fitz.Rect(left, row_y, right, row_y + 25), color=COLORS["direct"], fill=COLORS["paper"], width=0.9)
        for column, value in enumerate(values):
            insert_text(page, fitz.Rect(positions[column] + 3, row_y + 5, positions[column + 1] - 2, row_y + 23), value, 6.7, COLORS["ink"], bold=True)
        row_y += 29
        declared_text = _quantity_text(steel_declared, "kg", 1, language)
        schedule_note = choose(
            language,
            f"Schedule mass: {declared_text} | nominal steel density 7850 kg/m³",
            f"Масса по ведомости: {declared_text} | расчетная плотность стали 7850 кг/м³",
        )
        insert_text(page, fitz.Rect(left, row_y, right, row_y + 20), schedule_note, 6.7, COLORS["muted"], bold=True)
        insert_text(page, fitz.Rect(left, row_y + 20, right, row_y + 40), choose(language, "Diameters derived from detail bend-radius families; schedule excluded from calculation.", "Диаметры получены из семейств радиусов гиба деталей; ведомость исключена из расчета."), 6.4, COLORS["blocked"], bold=True)
    else:
        insert_text(
            page,
            fitz.Rect(x0 + margin + 12, row_y, x0 + sidebar * 0.56, row_y + 24),
            choose(language, "REBAR FROM DRAWING", "АРМАТУРА ПО ЧЕРТЕЖУ"),
            7.7,
            COLORS["direct"],
            bold=True,
        )
        insert_text(
            page,
            fitz.Rect(x0 + sidebar * 0.56, row_y, page.rect.x1 - margin - 10, row_y + 24),
            choose(language, "UNKNOWN", "НЕ ОПРЕДЕЛЕНО"),
            8.1,
            COLORS["blocked"],
            bold=True,
        )
        row_y += 28
        estimated_totals = estimated_takeoff.get("totals", {})
        rows = [
            (choose(language, "Rebar declared", "Арматура по ведомости"), _quantity_text(steel_declared, "kg", 1, language)),
            (choose(language, "Strict fabrication", "Строгая длина"), _quantity_text(fabrication, "m", language=language) if fabrication is not None else (f"{choose(language, 'PARTIAL', 'ЧАСТИЧНО')} {quantity_text(language, partial_fabrication, 'm', 3)}" if partial_fabrication else choose(language, "UNKNOWN", "НЕ ОПРЕДЕЛЕНО"))),
            (choose(language, "Profile estimate length", "Оценочная длина"), _quantity_text(estimated_totals.get("fabrication_length_m"), "m", language=language)),
            (choose(language, "Profile estimate mass", "Оценочная масса"), _quantity_text(estimated_totals.get("mass_kg"), "kg", 1, language)),
        ]
        for label, value in rows:
            insert_text(page, fitz.Rect(x0 + margin + 12, row_y, x0 + sidebar * 0.56, row_y + 24), label, 7.7, COLORS["muted"], bold=True)
            insert_text(page, fitz.Rect(x0 + sidebar * 0.56, row_y, page.rect.x1 - margin - 10, row_y + 24), value, 8.1, COLORS["ink"], bold=True)
            row_y += 25
        if estimated_takeoff.get("by_diameter"):
            left = x0 + margin + 12
            right = page.rect.x1 - margin - 12
            widths = (0.14, 0.25, 0.14, 0.24, 0.23)
            positions = [left]
            for width in widths:
                positions.append(positions[-1] + (right - left) * width)
            headers = (
                choose(language, "DIA", "ДИАМ."),
                choose(language, "MARKS", "МАРКИ"),
                choose(language, "BARS", "ШТ."),
                choose(language, "LENGTH", "ДЛИНА"),
                choose(language, "MASS", "МАССА"),
            )
            page.draw_rect(fitz.Rect(left, row_y, right, row_y + 20), color=COLORS["candidate"], fill=COLORS["paper"], width=0.8)
            for column, header in enumerate(headers):
                insert_text(page, fitz.Rect(positions[column] + 3, row_y + 3, positions[column + 1] - 2, row_y + 18), header, 6.1, COLORS["muted"], bold=True)
            row_y += 20
            for row in estimated_takeoff["by_diameter"]:
                values = (
                    f"{row['diameter_mm']:g}",
                    ",".join(row["marks"]),
                    str(row["physical_bar_count"]),
                    quantity_text(language, row["total_length_m"], "m", 3),
                    quantity_text(language, row["mass_kg"], "kg", 1),
                )
                page.draw_rect(fitz.Rect(left, row_y, right, row_y + 21), color=COLORS["candidate"], fill=COLORS["white"], width=0.4)
                for column, value in enumerate(values):
                    insert_text(page, fitz.Rect(positions[column] + 3, row_y + 4, positions[column + 1] - 2, row_y + 19), value, 6.3, COLORS["ink"], bold=column in {0, 4})
                row_y += 21
            profile = estimated_takeoff.get("profile", {})
            coverage = estimated_takeoff.get("coverage", {})
            note = choose(
                language,
                f"ESTIMATE ONLY: {profile.get('id')} | Rcl={profile.get('stirrup_tie_centerline_radius_factor', 0):g}d | density {profile.get('steel_density_kg_m3', 0):g} kg/m³ | families {coverage.get('estimated_family_count', 0)}/{coverage.get('eligible_family_count', 0)}. Not used for approval discrepancy.",
                f"ТОЛЬКО ОЦЕНКА: {profile.get('id')} | Rос={profile.get('stirrup_tie_centerline_radius_factor', 0):g}d | плотность {profile.get('steel_density_kg_m3', 0):g} кг/м³ | семейства {coverage.get('estimated_family_count', 0)}/{coverage.get('eligible_family_count', 0)}. Не используется для утверждаемого расхождения.",
            )
            insert_text(page, fitz.Rect(left, row_y + 3, right, row_y + 40), note, 6.2, COLORS["candidate"], bold=True)

    comparison_top = y + quantity_height - 100
    for comparison_index, (kind, comparison, unit, digits) in enumerate(
        (
            ("concrete", concrete_comparison, "m3", 4),
            ("steel", steel_comparison, "kg", 1),
        )
    ):
        top = comparison_top + comparison_index * 50
        color = _comparison_color(comparison["status"])
        page.draw_rect(
            fitz.Rect(x0 + margin + 10, top, page.rect.x1 - margin - 10, top + 44),
            color=color,
            fill=COLORS["white"],
            width=1.2,
        )
        heading = choose(language, "CONCRETE", "БЕТОН") if kind == "concrete" else choose(language, "STEEL", "СТАЛЬ")
        delta_text = (
            quantity_text(language, comparison["delta"], unit, digits, signed=True)
            if comparison["delta"] is not None
            else choose(language, "NOT COMPARABLE", "НЕЛЬЗЯ СРАВНИТЬ")
        )
        insert_text(
            page,
            fitz.Rect(x0 + margin + 20, top + 5, page.rect.x1 - margin - 20, top + 18),
            f"{heading}: {term(language, comparison['status']).upper()} | {delta_text}",
            6.7,
            color,
            bold=True,
        )
        insert_text(
            page,
            fitz.Rect(x0 + margin + 20, top + 20, page.rect.x1 - margin - 20, top + 41),
            review_action(language, kind, comparison["status"]),
            6.1,
            COLORS["ink"],
            bold=True,
        )

    solid = engineering["specialised_solver"]
    solid_resolved = solid["status"] == "resolved"
    status_color = COLORS["direct"] if solid_resolved else COLORS["blocked"]
    status_y = y + quantity_height + 22
    page.draw_rect(fitz.Rect(x0 + margin, status_y, page.rect.x1 - margin, min(page.rect.y1 - 22, status_y + 155)), color=status_color, fill=COLORS["white"], width=1.2)
    if solid_resolved and takeoff_resolved:
        title = choose(language, "CONCRETE MODEL + REBAR TAKEOFF AVAILABLE", "МОДЕЛЬ БЕТОНА И АРМАТУРА ДОСТУПНЫ")
    elif solid_resolved:
        title = choose(language, "CONCRETE MODEL AVAILABLE - REBAR INCOMPLETE", "МОДЕЛЬ БЕТОНА ДОСТУПНА - АРМАТУРА НЕПОЛНАЯ")
    else:
        title = choose(language, "3D MODEL NOT AVAILABLE", "3D-МОДЕЛЬ НЕДОСТУПНА")
    insert_text(page, fitz.Rect(x0 + margin + 12, status_y + 12, page.rect.x1 - margin - 12, status_y + 36), title, 9.5, status_color, bold=True)
    path_summary = graph.get("summary", {})
    rebar_3d_path_count = _rebar_3d_path_count(engineering)
    frame_summary = engineering.get("view_frame_graph", {}).get("summary", {})
    ownership_summary = engineering.get("dimension_ownership", {}).get("summary", {})
    equation_summary = engineering.get("metric_equation_graph", {}).get("summary", {})
    metric_scope_resolved = bool(frame_summary.get("metric_equation_scope_count", 0))
    if not solid_resolved and frame_summary.get("metric_equation_scope_count", 0):
        next_action = choose(
            language,
            "Bind the grouped views to material-layer contours and close the remaining overall dimensions.",
            "Связать сгруппированные виды с контурами слоев материалов и замкнуть оставшиеся общие размеры.",
        )
    elif not solid_resolved:
        next_action = choose(
            language,
            "Confirm which highlighted views show the same item and which edges the dimensions measure.",
            "Подтвердить, какие выделенные виды показывают одно изделие и какие грани задают размеры.",
        )
    elif not takeoff_resolved:
        next_action = review_action(language, "steel", "calculation_unavailable")
    else:
        next_action = choose(
            language,
            "Engineer approval is required before quotation.",
            "Перед подготовкой КП требуется проверка инженера.",
        )
    insert_text(
        page,
        fitz.Rect(x0 + margin + 12, status_y + 44, page.rect.x1 - margin - 12, min(page.rect.y1 - 30, status_y + 142)),
        choose(
            language,
            f"Objects found: {object_graph.get('summary', {}).get('object_instance_count', 0)} | drawing views: {view_number}\nViews aligned in shared coordinates: {frame_summary.get('axis_mapping_resolved_view_count', 0)} of {view_number}\nCross-view metric checks: {frame_summary.get('reprojection_pass_count', 0)} pass / {frame_summary.get('reprojection_fail_count', 0)} fail\nSection paths matched to parent cuts: {frame_summary.get('accepted_contour_correspondence_count', 0)}\nSigned view transforms: {frame_summary.get('signed_contour_transform_count', 0)}\nSections linked to parent views: {frame_summary.get('cutting_plane_relation_count', 0)}\nDimensions attached to object geometry: {ownership_summary.get('accepted_count', 0)} of {ownership_summary.get('dimension_count', 0)}\nMetric equations: {equation_summary.get('accepted_chain_count', 0)} of {equation_summary.get('equation_count', 0)} | grouped views: {frame_summary.get('object_scope_resolved_count', 0)}\nRebar shown in 3D: {rebar_3d_path_count} bars\nView links requiring review: {path_summary.get('review_cross_view_identity_count', 0)}\nNext: {next_action}",
            f"Найдено изделий: {object_graph.get('summary', {}).get('object_instance_count', 0)} | видов чертежа: {view_number}\nСовмещено видов в общей системе: {frame_summary.get('axis_mapping_resolved_view_count', 0)} из {view_number}\nПроверок размеров между видами: {frame_summary.get('reprojection_pass_count', 0)} пройдено / {frame_summary.get('reprojection_fail_count', 0)} не пройдено\nКонтуров сечений совпало с разрезами: {frame_summary.get('accepted_contour_correspondence_count', 0)}\nОднозначно ориентировано между видами: {frame_summary.get('signed_contour_transform_count', 0)}\nСечений, связанных с исходными видами: {frame_summary.get('cutting_plane_relation_count', 0)}\nРазмеров, привязанных к геометрии: {ownership_summary.get('accepted_count', 0)} из {ownership_summary.get('dimension_count', 0)}\nРазмерных формул: {equation_summary.get('accepted_chain_count', 0)} из {equation_summary.get('equation_count', 0)} | видов в группе: {frame_summary.get('object_scope_resolved_count', 0)}\nАрматура, показанная в 3D: {rebar_3d_path_count} шт.\nСвязей видов требует проверки: {path_summary.get('review_cross_view_identity_count', 0)}\nДалее: {next_action}",
        ),
        7.0,
        COLORS["ink"],
    )


def draw_3d_or_status(output: fitz.Document, source: Path, record: dict[str, Any], language: str, understanding_seconds: float) -> None:
    engineering = record["engineering_graph"]
    replay = _solid_replay(engineering)
    preview = engineering.get("solid_preview") or replay.get("solid_preview") or {}
    mesh = preview.get("mesh") or {}
    if mesh.get("vertices_xyz_mm") and mesh.get("triangles"):
        draw_3d_model(output, source, record, language, understanding_seconds)
        return
    page = output.new_page(width=1600, height=1000)
    page.draw_rect(page.rect, color=COLORS["paper"], fill=COLORS["paper"], width=0)
    page.draw_rect(fitz.Rect(0, 0, 1600, 120), color=COLORS["ink"], fill=COLORS["ink"], width=0)
    insert_text(page, fitz.Rect(48, 28, 1520, 70), choose(language, "AXONOMETRIC MODEL - NOT GENERATED", "АКСОНОМЕТРИЧЕСКАЯ МОДЕЛЬ НЕ ПОСТРОЕНА"), 24, COLORS["white"], bold=True)
    subtitle = choose(language, f"Source drawing | page {record['page']} | the drawing does not yet determine one unique model", f"Исходный чертеж | страница {record['page']} | данных пока недостаточно для единственной модели")
    insert_text(page, fitz.Rect(48, 77, 1520, 106), f"{subtitle} | {duration_text(language, understanding_seconds)}", 10, COLORS["white"])
    insert_text(page, fitz.Rect(70, 172, 760, 215), choose(language, "WHAT WAS FOUND", "ЧТО НАЙДЕНО"), 15, COLORS["ink"], bold=True)
    graph = engineering.get("rebar_program", {}).get("physical_path_graph", {})
    summary = graph.get("summary", {})
    dim_counts = summary.get("projection_dimensionality_counts", {})
    identified_rebar = sum(value for key, value in dim_counts.items() if key != "ambiguous_projection")
    frame_summary = engineering.get("view_frame_graph", {}).get("summary", {})
    ownership_summary = engineering.get("dimension_ownership", {}).get("summary", {})
    equation_summary = engineering.get("metric_equation_graph", {}).get("summary", {})
    metric_scope_resolved = bool(frame_summary.get("metric_equation_scope_count", 0))
    found = choose(
        language,
        f"Objects found: {engineering.get('object_instance_graph', {}).get('summary', {}).get('object_instance_count', 0)}\nDrawing views found: {len(engineering.get('view_hypotheses', []))}\nViews grouped by metric equations: {frame_summary.get('object_scope_resolved_count', 0)}\nMetric equations on native chains: {equation_summary.get('accepted_chain_count', 0)} of {equation_summary.get('equation_count', 0)}\nSections linked to parent views: {frame_summary.get('cutting_plane_relation_count', 0)}\nObject contours found: {len(record.get('contours', []))}\nDimensions attached to object geometry: {ownership_summary.get('accepted_count', 0)} of {ownership_summary.get('dimension_count', 0)}\nRebar projections classified: {identified_rebar}\nConnections between views requiring review: {summary.get('review_cross_view_identity_count', 0)}",
        f"Найдено изделий: {engineering.get('object_instance_graph', {}).get('summary', {}).get('object_instance_count', 0)}\nНайдено видов чертежа: {len(engineering.get('view_hypotheses', []))}\nВидов сгруппировано размерными формулами: {frame_summary.get('object_scope_resolved_count', 0)}\nФормул на размерных цепочках: {equation_summary.get('accepted_chain_count', 0)} из {equation_summary.get('equation_count', 0)}\nСечений, связанных с исходными видами: {frame_summary.get('cutting_plane_relation_count', 0)}\nНайдено контуров изделий: {len(record.get('contours', []))}\nРазмеров, привязанных к геометрии: {ownership_summary.get('accepted_count', 0)} из {ownership_summary.get('dimension_count', 0)}\nРаспознано проекций арматуры: {identified_rebar}\nСвязей между видами, требующих проверки: {summary.get('review_cross_view_identity_count', 0)}",
    )
    insert_text(page, fitz.Rect(70, 230, 760, 500), found, 12, COLORS["ink"])
    insert_text(page, fitz.Rect(830, 172, 1530, 215), choose(language, "WHY NO CONCRETE VOLUME / 3D", "ПОЧЕМУ НЕТ ОБЪЕМА БЕТОНА / 3D"), 15, COLORS["blocked"], bold=True)
    unresolved = engineering.get("unresolved", [])
    if replay:
        explanation = _solid_replay_validation_text(replay, language)
    elif metric_scope_resolved:
        explanation = choose(
            language,
            "The metric equations group several views consistently, but they do not yet identify which nested contour belongs to each material layer or prove the axis/depth correspondence needed for a watertight solid.",
            "Размерные формулы согласованно объединяют несколько видов, но пока не определено, какой вложенный контур относится к каждому слою материала, и не доказано соответствие осей и глубины, необходимое для замкнутого тела.",
        )
    else:
        explanation = choose(
            language,
            "A unique 3D model needs at least two views of the same object, dimensions assigned to its edges, and a proven transform between those views. One or more of these links is still ambiguous.",
            "Для единственной 3D-модели нужны минимум два вида одного изделия, размеры, привязанные к его граням, и подтвержденное соответствие координат между видами. Одна или несколько этих связей пока неоднозначны.",
        )
    if unresolved and not metric_scope_resolved and not replay:
        explanation += "\n\n" + "\n".join(f"• {reason_text(language, item)}" for item in unresolved[:4])
    insert_text(page, fitz.Rect(830, 230, 1530, 540), explanation, 11, COLORS["ink"])
    page.draw_rect(fitz.Rect(70, 690, 1530, 910), color=COLORS["candidate"], fill=COLORS["white"], width=1.4)
    insert_text(page, fitz.Rect(94, 718, 1500, 750), choose(language, "WHAT THE ENGINEER MUST CONFIRM", "ЧТО НУЖНО ПОДТВЕРДИТЬ"), 12, COLORS["candidate"], bold=True)
    if replay:
        review_copy = choose(
            language,
            "Resolve the named replay constraint in its owning upstream stage, preserve the frozen graph binding and evidence IDs, then rerun replay. Do not publish a mesh or quantity from this abstaining record.",
            "Устранить указанное ограничение в отвечающем за него предыдущем этапе, сохранить привязку к замороженному графу и идентификаторы доказательств, затем повторить проверку. Не публиковать сетку или количество из этой записи с воздержанием.",
        )
    elif metric_scope_resolved:
        review_copy = choose(
            language,
            "The grouped views and their dimension chains are shown for review. Confirm the material represented by each nested contour and the remaining axis/depth correspondence; reinforcement marks still need unique paths between projections. Quantities remain unavailable until those constraints close.",
            "Сгруппированные виды и их размерные цепочки показаны для проверки. Подтвердить материал каждого вложенного контура и оставшееся соответствие осей и глубины; для марок арматуры еще нужны однозначные пути между проекциями. Количества остаются недоступными, пока эти ограничения не замкнуты.",
        )
    else:
        review_copy = choose(
            language,
            "Detected outlines are shown for review. Confirm which highlighted views show the same item, which edges the highlighted dimensions measure, and how reinforcement marks continue between views. Concrete volume and unconfirmed rebar remain unavailable until these links are confirmed.",
            "Найденные контуры показаны для проверки. Подтвердить, какие выделенные виды показывают одно изделие, какие грани задают выделенные размеры и как марки арматуры продолжаются между видами. Объем бетона и неподтвержденная арматура остаются недоступными до подтверждения этих связей.",
        )
    insert_text(
        page,
        fitz.Rect(94, 770, 1500, 880),
        review_copy,
        11,
        COLORS["ink"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_artifact_path(bundle_path: Path, declared_path: str) -> Path:
    declared = Path(declared_path)
    if declared.is_file():
        return declared
    relocated = bundle_path.parent / declared.name
    if relocated.is_file():
        return relocated
    raise ValueError(f"precomputed bundle artifact is missing: {declared_path}")


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload, raw


def _replay_text_roles(engineering: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_claims = evidence.get("claims", {})
    roles = []
    for claim in engineering.get("claims", []):
        if claim.get("kind") != "text_role":
            continue
        source = evidence_claims.get(claim.get("evidence_ref"), {})
        roles.append(
            {
                "id": claim.get("subject"),
                "text": claim.get("subject"),
                "resolved_role": claim.get("value"),
                "epistemic_state": claim.get("state"),
                "basis": claim.get("basis"),
                "confidence": claim.get("confidence"),
                "primitive_refs": source.get("primitive_refs", []),
                "bbox_display": source.get("bbox_display", []),
            }
        )
    return roles


def _replay_dimensions(engineering: dict[str, Any]) -> list[SimpleNamespace]:
    dimensions = []
    for attachment in engineering.get("dimension_ownership", {}).get("attachments", []):
        if attachment.get("status") != "accepted":
            continue
        points = [
            fitz.Point(*endpoint["point_display"])
            for endpoint in attachment.get("measured_endpoints", [])
            if endpoint.get("state") != "unknown" and len(endpoint.get("point_display", [])) == 2
        ]
        if len(points) != 2:
            continue
        dimensions.append(
            SimpleNamespace(
                status="accepted",
                attachment_id=attachment.get("dimension_ref"),
                value_mm=attachment.get("value_mm"),
                measured_points=tuple(points),
            )
        )
    return dimensions


def load_precomputed_audit_records(
    source: Path,
    bundle_path: Path,
    estimation_profile: dict[str, Any],
    artifact_loader=None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load split graph records after validating their immutable bindings."""

    bundle, _ = artifact_loader("bundle") if artifact_loader else _load_json_object(bundle_path, "precomputed bundle")
    source_sha256 = _sha256(source)
    if bundle.get("document_key") != f"pdf-sha256:{source_sha256}":
        raise ValueError("precomputed bundle source PDF SHA-256 does not match the audit source")
    pipeline = bundle.get("pipeline")
    if (
        not isinstance(pipeline, dict)
        or not isinstance(pipeline.get("version"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(pipeline.get("sha256", "")))
    ):
        raise ValueError("precomputed bundle has no valid pipeline fingerprint")
    bundle_profile = bundle.get("estimation_profile", {})
    if bundle_profile.get("id") != estimation_profile.get("id"):
        raise ValueError("precomputed bundle estimation profile does not match the requested profile")
    profile_sha256 = bundle_profile.get("sha256")
    if profile_sha256:
        actual_profile_hash = (hashlib.sha256(artifact_loader("estimation_profile")[1]).hexdigest() if artifact_loader
                               else _sha256(Path(estimation_profile["source_path"])))
        if profile_sha256 != actual_profile_hash:
            raise ValueError("precomputed bundle estimation profile SHA-256 is stale")
    processing_options = bundle.get("processing_options")
    if processing_options is not None and (not isinstance(processing_options, dict)
            or processing_options.get("page_rotation_removed") is not True
            or set(processing_options) - {"page_rotation_removed", "source_page_numbers"}):
        raise ValueError("precomputed bundle processing options are not supported by the audit renderer")

    artifacts: dict[str, tuple[Path, dict[str, Any], bytes]] = {}
    for key in ("observation_graph", "engineering_graph", "evidence_store"):
        declared_path = bundle.get("files", {}).get(key)
        if not declared_path:
            raise ValueError(f"precomputed bundle does not declare {key}")
        path = bundle_path if artifact_loader else _bundle_artifact_path(bundle_path, declared_path)
        payload, raw = artifact_loader(key) if artifact_loader else _load_json_object(path, key)
        if payload.get("pipeline") != pipeline:
            raise ValueError(f"{key} pipeline fingerprint does not match its bundle")
        artifacts[key] = (path, payload, raw)

    observation_pages = artifacts["observation_graph"][1].get("pages", [])
    engineering_pages = artifacts["engineering_graph"][1].get("pages", [])
    evidence_pages = artifacts["evidence_store"][1].get("pages", [])
    page_numbers = [[item.get("page") for item in pages] for pages in (observation_pages, engineering_pages, evidence_pages)]
    if not page_numbers[0] or page_numbers[0] != page_numbers[1] or page_numbers[0] != page_numbers[2]:
        raise ValueError("precomputed split graphs do not contain the same ordered pages")
    with fitz.open(source) as document:
        selected_pages = (processing_options or {}).get("source_page_numbers", list(range(1, document.page_count + 1)))
        if (not isinstance(selected_pages, list) or not selected_pages
                or any(type(n) is not int or not 1 <= n <= document.page_count for n in selected_pages)
                or selected_pages != sorted(set(selected_pages)) or page_numbers[0] != selected_pages):
            raise ValueError("precomputed split graph pages do not match the source PDF")
        if "source_page_numbers" in (processing_options or {}) and artifacts["engineering_graph"][1].get("processing_options") != processing_options:
            raise ValueError("engineering graph page selection differs from its bundle")

    records = []
    for observation, engineering, evidence in zip(observation_pages, engineering_pages, evidence_pages):
        records.append(
            {
                "page": engineering["page"],
                "observation_graph": observation,
                "engineering_graph": engineering,
                "evidence_store": evidence,
                "text_roles": _replay_text_roles(engineering, evidence),
                "contours": engineering.get("contour_hypotheses", []),
                "dimensions": _replay_dimensions(engineering),
                "dimension_proposals": [],
            }
        )
    engineering_path, _, engineering_raw = artifacts["engineering_graph"]
    engineering_payload = artifacts["engineering_graph"][1]
    canonical_graph = engineering_payload.get("canonical_knowledge_graph", {})
    declared_binding = bundle.get("result_binding") or {}
    actual_canonical_hash = canonical_graph_sha256(canonical_graph)
    binding_issues = []
    if declared_binding.get("source_pdf_sha256") != source_sha256:
        binding_issues.append("source_pdf_hash_missing_or_mismatched")
    if declared_binding.get("canonical_engineering_graph_sha256") != actual_canonical_hash:
        binding_issues.append("canonical_engineering_graph_hash_missing_or_mismatched")
    if declared_binding.get("pipeline") != pipeline:
        binding_issues.append("pipeline_binding_missing_or_mismatched")
    result_binding = {
        **declared_binding,
        "source_pdf_sha256": source_sha256,
        "canonical_engineering_graph_sha256": actual_canonical_hash,
        "pipeline": pipeline,
        "ruleset": declared_binding.get("ruleset") or {"version": "unknown", "sha256": None},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_versions": declared_binding.get("model_versions", []),
    }
    validation = {
        "status": "historical_superseded" if binding_issues else "pass",
        "binding_issues": binding_issues,
        "source_pdf_sha256": source_sha256,
        "pipeline": pipeline,
        "estimation_profile_id": bundle_profile.get("id"),
        "estimation_profile_sha256": profile_sha256,
        "processing_options": processing_options or {"page_rotation_removed": True},
        "engineering_graph_path": str(engineering_path.resolve()),
        "engineering_graph_sha256": hashlib.sha256(engineering_raw).hexdigest(),
        "result_binding": result_binding,
    }
    return records, bundle.get("timing", {}), validation


def load_precomputed_declarations(
    source: Path,
    comparison_path: Path,
    bundle_validation: dict[str, Any] | None,
    artifact_loader=None,
) -> dict[str, Any]:
    """Reuse declarations only when their frozen drawing graph is byte-identical."""

    if bundle_validation is None:
        raise ValueError("a precomputed comparison requires a validated precomputed bundle")
    comparison, _ = artifact_loader("comparison") if artifact_loader else _load_json_object(comparison_path, "estimate comparison")
    declared_source = Path(str(comparison.get("source_pdf", "")))
    if artifact_loader:
        if comparison.get("result_binding", {}).get("source_pdf_sha256") != bundle_validation["source_pdf_sha256"]:
            raise ValueError("estimate comparison source hash does not match the audit source")
    elif not declared_source.is_absolute() or declared_source.resolve() != source.resolve():
        raise ValueError("estimate comparison source path does not match the audit source")
    frozen = comparison.get("frozen_calculation", {})
    if frozen.get("sha256") != bundle_validation["engineering_graph_sha256"]:
        raise ValueError("estimate comparison does not bind to the precomputed engineering graph bytes")
    if frozen.get("pipeline") != bundle_validation["pipeline"]:
        raise ValueError("estimate comparison pipeline fingerprint does not match the bundle")
    declarations = [item.get("declared_by_designer", {}) for item in comparison.get("pages", [])]
    with fitz.open(source) as document:
        expected_pages = bundle_validation.get("processing_options", {}).get(
            "source_page_numbers", list(range(1, len(document) + 1)))
    if [item.get("page") for item in declarations] != expected_pages:
        raise ValueError("estimate comparison declarations do not cover ordered source pages")
    return {
        "schema_version": comparison.get("schema_version"),
        "timing": {
            "declared_schedule_extraction_seconds": comparison.get("timing", {}).get(
                "declared_schedule_extraction_seconds"
            )
        },
        "pages": declarations,
    }


def build_audit(
    source: Path,
    output_pdf: Path,
    language: str = DEFAULT_LANGUAGE,
    estimation_profile_path: Path = DEFAULT_PROFILE_PATH,
    precomputed_bundle_path: Path | None = None,
    comparison_path: Path | None = None,
    frozen_project=None,
    write_manifest: bool = True,
) -> Path:
    language = validate_language(language)
    understanding_started = perf_counter()
    raw = fitz.open(source)
    display = fitz.open()
    display.insert_pdf(raw)
    estimation_profile = frozen_project.load("estimation_profile") if frozen_project else load_estimation_profile(estimation_profile_path)
    if frozen_project:
        precomputed_bundle_path = frozen_project.database
        comparison_path = frozen_project.database
    for page in display:
        page.remove_rotation()
    if precomputed_bundle_path is None:
        records = []
        page_seconds = []
        for page in display:
            page_started = perf_counter()
            records.append(understand_page(page, estimation_profile=estimation_profile))
            page_seconds.append(round(perf_counter() - page_started, 6))
        understanding_seconds = round(perf_counter() - understanding_started, 6)
        analysis_source = "live_pdf_analysis"
        bundle_validation = None
    else:
        records, bundle_timing, bundle_validation = load_precomputed_audit_records(
            source,
            precomputed_bundle_path,
            estimation_profile,
            artifact_loader=frozen_project.artifact if frozen_project else None,
        )
        page_seconds = bundle_timing.get("page_understanding_seconds", [])
        understanding_seconds = bundle_timing.get("drawing_understanding_seconds")
        if not isinstance(understanding_seconds, (int, float)):
            raise ValueError("precomputed bundle is missing drawing-understanding timing")
        analysis_source = "frozen_sqlite_snapshot" if frozen_project else "validated_precomputed_bundle"
    if bundle_validation is not None:
        result_binding = {
            **bundle_validation["result_binding"],
            "engineering_graph_artifact_sha256": bundle_validation["engineering_graph_sha256"],
        }
    else:
        source_sha256 = _sha256(source)
        canonical_graph = build_canonical_knowledge_graph(
            [{"page": item["page"], **item["engineering_graph"]} for item in records],
            document_key=f"pdf-sha256:{source_sha256}",
        )
        result_binding = {
            "source_pdf_sha256": source_sha256,
            "canonical_engineering_graph_sha256": canonical_graph_sha256(canonical_graph),
            "pipeline": pipeline_identity(),
            "ruleset": {"version": RULESET_VERSION, "sha256": ruleset_sha256()},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_versions": [
                item["engineering_graph"].get("perception_routing", {}).get("model_provenance", {})
                for item in records
            ],
        }
    # Drawing-derived records are complete before the PDF is reopened for
    # independent declaration extraction.  No schedule value feeds a solver.
    declarations = (
        load_precomputed_declarations(source, comparison_path, bundle_validation,
                                      artifact_loader=frozen_project.artifact if frozen_project else None)
        if comparison_path is not None
        else extract_declared_schedules(source)
    )
    output = fitz.open()
    ocgs = {
        "SOURCE": output.add_ocg("SOURCE", on=1),
        "OBJECT_INSTANCES": output.add_ocg("OBJECT_INSTANCES", on=1),
        "DETECTED_VIEWS": output.add_ocg("DETECTED_VIEWS", on=1),
        "CUTTING_PLANES": output.add_ocg("CUTTING_PLANES", on=1),
        "CROSS_VIEW_MATCHES": output.add_ocg("CROSS_VIEW_MATCHES", on=1),
        "OBJECT_CONTOURS": output.add_ocg("OBJECT_CONTOURS", on=1),
        "REBAR_DETECTED": output.add_ocg("REBAR_DETECTED", on=1),
        "REBAR_REJECTED": output.add_ocg("REBAR_REJECTED", on=0),
        "DETAIL_LINKS": output.add_ocg("DETAIL_LINKS", on=1),
        "DIMENSIONS": output.add_ocg("DIMENSIONS", on=1),
        "DECLARED_SCHEDULE": output.add_ocg("DECLARED_SCHEDULE", on=1),
        "DISCREPANCIES": output.add_ocg("DISCREPANCIES", on=1),
    }
    for record in records:
        index = record["page"] - 1
        source_page = display[index]
        reference = output.new_page(width=source_page.rect.width, height=source_page.rect.height)
        reference.show_pdf_page(reference.rect, display, index)
        draw_engineer_overlay(output, display, source_page, record, declarations, index, ocgs, language, understanding_seconds)
        draw_3d_or_status(output, source, record, language, understanding_seconds)
    output.set_metadata({
        "title": choose(language, f"{source.name} - engineer drawing understanding audit", f"{source.name} - аудит понимания чертежа"),
        "author": choose(language, "Procedural drawing understanding pipeline", "Процедурный конвейер понимания чертежей"),
        "subject": choose(language, "Source, understood drawing evidence, quantity check, and axonometric model status", "Исходный лист, найденные данные, проверка объемов и состояние аксонометрической модели"),
        "creator": "PyMuPDF",
    })
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_pdf, garbage=4, deflate=True, clean=True)
    output.close()
    manifest = {
        "schema_version": "0.1.0",
        "source_pdf": str(source.resolve()),
        "output_pdf": str(output_pdf.resolve()),
        "presentation_language": language,
        "estimation_profile": {
            "id": estimation_profile["id"],
            "source_path": str(frozen_project.database) + "#estimation_profile" if frozen_project else estimation_profile["source_path"],
        },
        "timing": {
            "drawing_understanding_seconds": understanding_seconds,
            "page_understanding_seconds": page_seconds,
        },
        "drawing_understanding_source": analysis_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_binding": result_binding,
        "result_currency": (
            "historical_superseded"
            if bundle_validation is not None and bundle_validation.get("status") != "pass"
            else "current"
        ),
        "binding_issues": (bundle_validation or {}).get("binding_issues", []),
        "precomputed_bundle_validation": bundle_validation,
        "page_summaries": [page_summary(record) for record in records],
        "layers": list(ocgs),
        "page_structure": [
            item
            for index, record in enumerate(records)
            for item in (f"source_{record['page']}", f"engineer_overlay_{record['page']}", f"axonometric_or_status_{record['page']}")
        ],
        "declared_schedule_extracted_after_drawing_calculation": True,
    }
    if write_manifest:
        output_pdf.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    display.close()
    raw.close()
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "pdf")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=DEFAULT_LANGUAGE)
    parser.add_argument("--estimation-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--bundle", type=Path, help="validated frozen object-agnostic bundle used instead of re-analysing the PDF")
    parser.add_argument("--comparison", type=Path, help="validated frozen declaration comparison used instead of repeating schedule extraction")
    args = parser.parse_args()
    if (args.bundle is not None or args.comparison is not None) and len(args.inputs) != 1:
        parser.error("--bundle and --comparison require exactly one input PDF")
    if args.comparison is not None and args.bundle is None:
        parser.error("--comparison requires --bundle so its frozen graph hash can be validated")
    for source in args.inputs:
        output = args.output_dir / f"{source.stem}_object_agnostic_audit.pdf"
        print(
            build_audit(
                source,
                output,
                language=args.language,
                estimation_profile_path=args.estimation_profile,
                precomputed_bundle_path=args.bundle,
                comparison_path=args.comparison,
            )
        )


if __name__ == "__main__":
    main()
