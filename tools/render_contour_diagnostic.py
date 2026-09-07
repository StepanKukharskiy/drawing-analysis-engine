#!/usr/bin/env python3
"""Render provenance-backed contour roles and a fail-closed CSG proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

import fitz
from PIL import ImageDraw, ImageFont


COLORS = {
    "outer_closed": (0, 150, 80, 255),
    "outer_open": (230, 140, 0, 255),
    "section_composite": (0, 130, 210, 255),
    "internal_network": (190, 60, 160, 210),
    "view": (90, 90, 90, 180),
}


def classify_contour(contour: Mapping[str, Any], view_box: list[float]) -> str | None:
    box = [float(value) for value in contour.get("bbox_display") or ()]
    if len(box) != 4:
        return None
    vw, vh = max(1.0, view_box[2] - view_box[0]), max(1.0, view_box[3] - view_box[1])
    width, height = box[2] - box[0], box[3] - box[1]
    topology = contour.get("topology") or {}
    closed = bool(contour.get("closed"))
    endpoints = int(topology.get("endpoint_vertex_count") or 0)
    branches = int(topology.get("branch_vertex_count") or 0)
    cycles = int(topology.get("cycle_rank") or 0)
    if closed and width >= 0.55 * vw and height >= 0.45 * vh:
        return "outer_closed"
    if not closed and endpoints == 2 and branches == 0 and width >= 0.55 * vw and height >= 0.35 * vh:
        return "outer_open"
    compact_view = vw / vh <= 0.55
    if compact_view and height >= 0.65 * vh and width >= 0.20 * vw and branches <= 4:
        return "section_composite"
    if branches > 0 or cycles > 2:
        if width >= 0.15 * vw or height >= 0.15 * vh:
            return "internal_network"
    return None


def _draw_primitive(draw: ImageDraw.ImageDraw, item: Mapping[str, Any], scale: float, color: tuple[int, ...], width: int) -> None:
    def point(value: Any) -> tuple[float, float]:
        return float(value.x) * scale, float(value.y) * scale

    for part in item.get("items") or ():
        kind = part[0]
        if kind == "l":
            draw.line([point(part[1]), point(part[2])], fill=color, width=width)
        elif kind == "re":
            rect = part[1]
            draw.rectangle([rect.x0 * scale, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale], outline=color, width=width)
        elif kind == "qu":
            quad = part[1]
            points = [point(value) for value in (quad.ul, quad.ur, quad.lr, quad.ll, quad.ul)]
            draw.line(points, fill=color, width=width, joint="curve")
        elif kind == "c":
            p0, p1, p2, p3 = part[1:5]
            points = []
            for index in range(17):
                t = index / 16
                u = 1 - t
                x = u**3 * p0.x + 3 * u**2 * t * p1.x + 3 * u * t**2 * p2.x + t**3 * p3.x
                y = u**3 * p0.y + 3 * u**2 * t * p1.y + 3 * u * t**2 * p2.y + t**3 * p3.y
                points.append((x * scale, y * scale))
            draw.line(points, fill=color, width=width, joint="curve")


def render_diagnostic(pdf_path: Path, graph_path: Path, output_path: Path, record_path: Path, *, dpi: int = 144, language: str = "ru") -> dict[str, Any]:
    graph = json.loads(graph_path.read_text())
    page_record = graph["pages"][0]
    contours = {str(item["id"]): item for item in page_record["contour_hypotheses"]}
    with fitz.open(pdf_path) as document:
        page = document[0]
        drawings = page.get_drawings()
        scale = dpi / 72.0
        image = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).pil_image()
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=max(14, round(dpi / 8)))
    small = ImageFont.load_default(size=max(11, round(dpi / 11)))
    roles = []
    for view in page_record["view_hypotheses"]:
        view_box = [float(value) for value in view["bbox_display"]]
        draw.rectangle([value * scale for value in view_box], outline=COLORS["view"], width=2)
        selected = []
        for contour_id in view.get("contour_refs") or ():
            contour = contours.get(str(contour_id))
            if contour is None:
                continue
            role = classify_contour(contour, view_box)
            if role is None:
                continue
            selected.append((contour, role))
            for ref in contour.get("primitive_refs") or ():
                match = re.fullmatch(r"drawing\[(\d+)]", str(ref))
                if match and int(match.group(1)) < len(drawings):
                    _draw_primitive(draw, drawings[int(match.group(1))], scale, COLORS[role], max(3, round(dpi / 45)))
            box = [float(value) for value in contour["bbox_display"]]
            draw.rectangle([value * scale for value in box], outline=COLORS[role], width=max(2, round(dpi / 60)))
            draw.text((box[0] * scale + 3, box[1] * scale + 3), str(contour["id"]), fill=COLORS[role], font=small, stroke_width=2, stroke_fill=(255, 255, 255, 230))
            roles.append(
                {
                    "view_id": view["id"],
                    "contour_id": contour["id"],
                    "role": role,
                    "bbox_display": box,
                    "closed": bool(contour.get("closed")),
                    "topology": contour.get("topology"),
                    "primitive_refs": contour.get("primitive_refs") or [],
                }
            )
        has_outer = any(role in {"outer_closed", "outer_open"} for _, role in selected)
        status = (
            ("есть кандидат внешнего контура" if language == "ru" else "outer candidate present")
            if has_outer
            else ("внешний контур не замкнут" if language == "ru" else "outer contour unresolved")
        )
        draw.text((view_box[0] * scale, max(0, view_box[1] * scale - 22)), f"{view['id']}: {status}", fill=(30, 30, 30, 255), font=small, stroke_width=2, stroke_fill=(255, 255, 255, 240))

    legend_x, legend_y = 1120 * scale, 900 * scale
    draw.rounded_rectangle([legend_x, legend_y, legend_x + 500, legend_y + 245], radius=16, fill=(255, 255, 255, 235), outline=(70, 70, 70, 255), width=2)
    heading = "ДИАГНОСТИКА КОНТУРОВ" if language == "ru" else "CONTOUR DIAGNOSTIC"
    draw.text((legend_x + 18, legend_y + 12), heading, fill=(20, 20, 20, 255), font=font)
    labels = (
        [
            ("outer_closed", "замкнутый кандидат внешнего профиля"),
            ("outer_open", "незамкнутый кандидат внешнего профиля"),
            ("section_composite", "смешанная геометрия сечения"),
            ("internal_network", "внутренняя сеть / арматура"),
            ("view", "граница найденного вида"),
        ]
        if language == "ru"
        else [
            ("outer_closed", "closed outer-profile candidate"),
            ("outer_open", "open outer-profile candidate"),
            ("section_composite", "section host/rebar composite"),
            ("internal_network", "branched internal/rebar network"),
            ("view", "detected view boundary"),
        ]
    )
    for index, (role, label) in enumerate(labels):
        y = legend_y + 58 + index * 34
        draw.line([(legend_x + 20, y), (legend_x + 65, y)], fill=COLORS[role], width=7)
        draw.text((legend_x + 80, y - 9), label, fill=(30, 30, 30, 255), font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    outer = [item for item in roles if item["role"] in {"outer_closed", "outer_open"}]
    overlaps = []
    for index, left in enumerate(outer):
        for right in outer[index + 1 :]:
            if left["view_id"] != right["view_id"]:
                continue
            a, b = left["bbox_display"], right["bbox_display"]
            intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0.0, min(a[3], b[3]) - max(a[1], b[1])
            )
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - intersection
            if union and intersection / union >= 0.5:
                overlaps.append(
                    {
                        "view_id": left["view_id"],
                        "contour_ids": [left["contour_id"], right["contour_id"]],
                        "bbox_iou": round(intersection / union, 6),
                        "interpretation": "overlapping_alternative_or_layer_envelopes_not_independent_solids",
                    }
                )
    record = {
        "schema_version": "0.1.0",
        "source_pdf": str(pdf_path.resolve()),
        "source_graph": str(graph_path.resolve()),
        "contour_roles": roles,
        "csg_union_hypothesis": {
            "status": "candidate" if outer else "unavailable",
            "outer_contour_candidate_ids": [item["contour_id"] for item in outer],
            "overlapping_candidate_sets": overlaps,
            "operation": "deduplicate_face_silhouettes_then_extrude_proven_material_regions_then_boolean_union",
            "unified_outer_surface_state": "unresolved",
            "required_gates": [
                "one contour identity per physical face or material layer",
                "one physical coordinate frame",
                "material ownership per silhouette",
                "section-proven extrusion interval per component",
                "cross-view reprojection",
                "boolean-union watertightness",
                "insulation and void subtraction",
            ],
        },
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--language", choices=["ru", "en"], default="ru")
    args = parser.parse_args()
    record = render_diagnostic(args.pdf, args.graph, args.output, args.record, dpi=args.dpi, language=args.language)
    print(args.output)
    print(args.record)
    print(json.dumps(record["csg_union_hypothesis"], ensure_ascii=False))


if __name__ == "__main__":
    main()
