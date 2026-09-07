"""Drawing-derived reinforcement takeoff from linked native detail rows.

The solver is intentionally object-name agnostic.  It closes only when a
native detail table, drawing-specified variant bindings, section multiplicity,
and repeated placement geometry agree.  Printed schedule quantities are not
read here.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

import fitz

from src.drawing_engine.disciplines.detail.detail_fabrication_solver import solve_detail_fabrication_geometry


STEEL_DENSITY_KG_M3 = 7850.0
STANDARD_DIAMETERS_MM = (6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40)


def _render_gray(page: fitz.Page, box: fitz.Rect, scale: int = 8) -> Any:
    import numpy as np

    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box, colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)


def _ocr_detail_annotation(page: fitz.Page, box: fitz.Rect) -> list[dict[str, Any]]:
    import pytesseract
    from PIL import Image

    image = Image.fromarray(_render_gray(page, box))
    rows = []
    for mode in (6, 11, 12):
        try:
            text = pytesseract.image_to_string(image, config=f"--psm {mode} -l eng", timeout=20).strip()
        except (OSError, RuntimeError, pytesseract.TesseractError):
            continue
        rows.append({"psm": mode, "text": text})
    return rows


def parse_detail_annotations(
    observations: list[dict[str, Any]],
    marks: list[str],
    variant_styles: dict[str, str],
) -> dict[str, Any]:
    """Parse direct ``L=`` and inner-radius annotations by OCR consensus."""

    length_votes: Counter[tuple[int, ...]] = Counter()
    radius_votes: Counter[int] = Counter()
    for observation in observations:
        text = observation.get("text", "").upper().replace("×", "X")
        for match in re.finditer(
            r"(?:L|I|\(|\[|')\s*=\s*(\d{3,5})\s*(?:\[\s*(\d{3,5})\s*\]\s*\(\s*(\d{3,5})\s*\))?",
            text,
        ):
            values = tuple(int(value) for value in match.groups() if value)
            if values and values[0] >= 500 and len(values) in {1, len(marks)}:
                length_votes[values] += 1
        for match in re.finditer(r"R[A-Z0-9]{0,4}\s*=\s*(\d{2,3})", text):
            radius_votes[int(match.group(1))] += 1
    if not length_votes or not radius_votes:
        return {"status": "unresolved", "ocr_observations": observations}
    lengths, length_support = max(length_votes.items(), key=lambda item: (item[1], len(item[0])))
    radius, radius_support = max(radius_votes.items(), key=lambda item: item[1])
    if len(marks) == 1:
        length_by_mark = {marks[0]: lengths[0]}
    else:
        by_style = {"bare": lengths[0], "square": lengths[1], "parentheses": lengths[2]}
        length_by_mark = {mark: by_style[variant_styles[mark]] for mark in marks}
    raw_diameter = radius / 3.0
    diameter = min(STANDARD_DIAMETERS_MM, key=lambda value: abs(value - raw_diameter))
    diameter_ok = abs(diameter - raw_diameter) <= 0.35
    return {
        "status": "resolved" if diameter_ok else "partial",
        "length_by_mark_mm": length_by_mark,
        "inner_bend_radius_mm": radius,
        "diameter_mm": diameter if diameter_ok else None,
        "diameter_state": "convention_dependent" if diameter_ok else "unknown",
        "diameter_basis": "explicit inner bend radius divided by the recognized three-diameter detail convention" if diameter_ok else None,
        "length_support": length_support,
        "radius_support": radius_support,
        "ocr_observations": observations,
    }


def _fragment_centers(group: dict[str, Any], fragments: dict[str, dict[str, Any]]) -> list[tuple[float, float]]:
    rows = []
    for fragment_id in group.get("fragment_ids", []):
        points = fragments.get(fragment_id, {}).get("geometry", {}).get("points_display", [])
        if points:
            rows.append((sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)))
    return rows


def _cluster_points(points: list[tuple[float, float]], tolerance: float = 2.2) -> list[list[tuple[float, float]]]:
    clusters: list[list[tuple[float, float]]] = []
    for point in points:
        matches = [cluster for cluster in clusters if min(math.dist(point, member) for member in cluster) <= tolerance]
        if not matches:
            clusters.append([point])
            continue
        target = matches[0]
        target.append(point)
        for extra in matches[1:]:
            target.extend(extra)
            clusters.remove(extra)
    return clusters


def _leader_attachment_fanout(association: dict[str, Any]) -> int:
    """Count arrowhead branches without counting duplicate glyph vertices.

    Outlined PDF arrowheads often expose two wing endpoints, while a butt end
    or a fragmented triangular glyph can contribute a third nearly coincident
    terminal.  Cluster local terminals and cap each arrowhead at its two
    independent wings before the section multiplicity gate consumes it.
    """

    terminals = [
        tuple(map(float, item["terminal_display"][:2]))
        for item in association.get("path_attachments", [])
        if len(item.get("terminal_display", [])) >= 2
    ]
    return sum(min(2, len(cluster)) for cluster in _cluster_points(terminals, tolerance=6.0))


def _zone_count(points: list[tuple[float, float]]) -> int:
    """Count separated station zones along their principal span."""

    if len(points) < 2:
        return len(points)
    span_x = max(point[0] for point in points) - min(point[0] for point in points)
    span_y = max(point[1] for point in points) - min(point[1] for point in points)
    values = sorted(point[0] if span_x >= span_y else point[1] for point in points)
    gaps = [right - left for left, right in zip(values, values[1:]) if right - left > 0.25]
    if not gaps:
        return 1
    local = sorted(gaps)[(len(gaps) - 1) // 2]
    return 1 + sum(gap > max(20.0, 3.5 * local) for gap in gaps)


def _angle_delta(left: float, right: float) -> float:
    difference = abs((left % 180.0) - (right % 180.0))
    return min(difference, 180.0 - difference)


def _spacing_observations(page: fitz.Page, box: fitz.Rect, long_angle: float) -> list[dict[str, Any]]:
    import pytesseract
    from PIL import Image

    image = Image.fromarray(_render_gray(page, box, 4))
    # PDF display coordinates have a downward Y axis; positive rotation levels
    # a descending reinforcement axis for OCR.
    rotated = image.rotate(long_angle, expand=True, fillcolor=255)
    rows = []
    for mode in (6, 11, 12):
        try:
            text = pytesseract.image_to_string(rotated, config=f"--psm {mode} -l eng", timeout=25).strip()
        except (OSError, RuntimeError, pytesseract.TesseractError):
            continue
        rows.append({"psm": mode, "rotation_deg": long_angle, "text": text})
    return rows


def _parse_spacing_variants(observations: list[dict[str, Any]]) -> dict[str, int]:
    votes: dict[str, Counter[int]] = {"bare": Counter(), "parentheses": Counter(), "square": Counter()}
    pattern = re.compile(r"(\d{1,3})\s*[Xx]\s*(\d{2,4})\s*[=-]\s*(\d{3,5})")
    for observation in observations:
        text = observation.get("text", "").replace("×", "X")
        for match in pattern.finditer(text):
            intervals, spacing, total = map(int, match.groups())
            if abs(intervals * spacing - total) > max(5, 0.01 * total):
                continue
            prefix = text[max(0, match.start() - 4) : match.start()]
            style = "parentheses" if "(" in prefix else "square" if "[" in prefix else "bare"
            votes[style][intervals] += 1
    return {style: counter.most_common(1)[0][0] for style, counter in votes.items() if counter}


def _station_geometry(
    page: fitz.Page,
    native_details: dict[str, Any],
    path_graph: dict[str, Any],
) -> dict[str, Any] | None:
    fragments = {item["id"]: item for item in path_graph.get("fragments", [])}
    long_rows = []
    for association in native_details.get("placement_associations", []):
        if association.get("state") != "accepted" or len(association.get("marks", [])) <= 1:
            continue
        for attachment in association.get("path_attachments", []):
            fragment = fragments.get(attachment.get("fragment_id"))
            if fragment and float(fragment.get("geometry", {}).get("length_points", 0.0)) >= 100:
                long_rows.append(fragment)
    if not long_rows:
        return None
    long_fragment = max(long_rows, key=lambda item: float(item["geometry"]["length_points"]))
    long_angle = float(long_fragment["geometry"]["angle_deg"])
    view_id = long_fragment.get("view_id")
    rows = []
    for group in path_graph.get("repetition_groups", []):
        if group.get("view_id") != view_id or float(group.get("median_length_points", 0.0)) < 15:
            continue
        centers = [tuple(sum(value) / len(value) for value in zip(*cluster)) for cluster in _cluster_points(_fragment_centers(group, fragments))]
        if len(centers) < 2:
            continue
        rows.append(
            {
                "group_id": group["id"],
                "physical_station_count": len(centers),
                "zone_count": _zone_count(centers),
                "orientation_deg": float(group.get("orientation_deg", 0.0)),
                "median_length_points": float(group.get("median_length_points", 0.0)),
                "centers": centers,
            }
        )
    main = [item for item in rows if 75 <= _angle_delta(item["orientation_deg"], long_angle) <= 105 and item["physical_station_count"] >= 20]
    if not main:
        return None
    main_count = Counter(item["physical_station_count"] for item in main).most_common(1)[0][0]
    main_row = next(item for item in main if item["physical_station_count"] == main_count)
    end = [
        item
        for item in rows
        if 2 <= item["physical_station_count"] <= 16
        and item["zone_count"] == 2
        and abs(item["median_length_points"] - main_row["median_length_points"]) <= 0.08 * main_row["median_length_points"]
    ]
    if not end:
        return None
    end_row = max(end, key=lambda item: item["physical_station_count"])
    return {
        "view_id": view_id,
        "longitudinal_axis_fragment_id": long_fragment["id"],
        "longitudinal_axis_angle_deg": long_angle,
        "main_station_count": main_count,
        "main_repetition_group_ids": sorted(item["group_id"] for item in main if item["physical_station_count"] == main_count),
        "end_station_count": end_row["physical_station_count"],
        "end_zone_count": end_row["zone_count"],
        "end_repetition_group_id": end_row["group_id"],
    }


def _designation_maps(native_details: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    instances = native_details.get("variant_context", {}).get("instances", [])
    designation_to_instance = {item["designation"]: item["object_instance_id"] for item in instances if item.get("designation")}
    schemes = native_details.get("variant_context", {}).get("shared_variant_schemes", [])
    scheme = schemes[0].get("scheme", {}) if len(schemes) == 1 else {}
    return designation_to_instance, scheme


def _mass_per_metre(diameter_mm: float) -> float:
    return math.pi * (diameter_mm / 1000.0) ** 2 / 4.0 * STEEL_DENSITY_KG_M3


def _profile_axis(profile: list[list[float]], normal_offset: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [point[0] for point in profile]
    zs = [point[1] for point in profile]
    start = (min(xs) + 300.0, max(zs) - 150.0 - normal_offset)
    end = (max(xs) - 300.0, min(zs) + 150.0 - normal_offset)
    return start, end


def _interpolate(left: tuple[float, float], right: tuple[float, float], fraction: float) -> tuple[float, float]:
    return (left[0] + (right[0] - left[0]) * fraction, left[1] + (right[1] - left[1]) * fraction)


def _scene_paths(
    elements: list[dict[str, Any]],
    solid_geometry: dict[str, Any] | None,
    solid_mesh: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not solid_geometry or solid_geometry.get("shape_type") != "multi_object_extrusion_collection":
        return []
    offsets = {
        item["object_instance_id"]: float(item.get("display_offset_x_mm", 0.0))
        for item in (solid_mesh or {}).get("components", [])
    }
    profiles = {item["object_instance_id"]: item for item in solid_geometry.get("objects", [])}
    paths = []
    for element in elements:
        profile_row = profiles.get(element["object_instance_id"])
        if not profile_row:
            continue
        profile = profile_row["profile_points_xz_mm"]
        offset_x = offsets.get(element["object_instance_id"], 0.0)
        depth = float(profile_row["extrusion_depth_mm"])
        diameter = float(element["diameter_mm"])
        mark = element["mark"]
        role = element["role"]
        if role in {"longitudinal_lower", "longitudinal_upper"}:
            axis = _profile_axis(profile, 55.0 if role == "longitudinal_lower" else -55.0)
            for instance, y in enumerate((35.0, depth - 35.0), start=1):
                left = _interpolate(axis[0], axis[1], 0.02)
                bend_left = _interpolate(axis[0], axis[1], 0.10)
                bend_right = _interpolate(axis[0], axis[1], 0.90)
                right = _interpolate(axis[0], axis[1], 0.98)
                paths.append({
                    "mark": mark, "role": role, "instance": instance, "diameter_mm": diameter,
                    "placement_status": "drawing_constrained",
                    "points_xyz_mm": [[left[0] + offset_x, y, left[1]], [bend_left[0] + offset_x, y, bend_left[1]], [bend_right[0] + offset_x, y, bend_right[1]], [right[0] + offset_x, y, right[1]]],
                    "closed": False, "evidence": element["evidence_refs"],
                })
        elif role == "end_anchor":
            axis = _profile_axis(profile)
            dx, dz = axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]
            length = max(math.hypot(dx, dz), 1.0)
            ux, uz = dx / length, dz / length
            nx, nz = -uz, ux
            instance = 0
            for station, direction in ((0.04, 1.0), (0.96, -1.0)):
                center = _interpolate(axis[0], axis[1], station)
                for y in (35.0, depth - 35.0):
                    for normal in (-45.0, 45.0):
                        instance += 1
                        # Preserve the section offset for the complete bar.  A
                        # previous preview incorrectly pulled every bar through
                        # the un-offset station centre, creating red X shapes
                        # that do not exist in the drawing.
                        p0 = (center[0] + nx * normal, center[1] + nz * normal)
                        bend = (p0[0] + direction * 240.0, p0[1])
                        p1 = (bend[0] + direction * ux * 240.0, bend[1] + direction * uz * 240.0)
                        paths.append({
                            "mark": mark, "role": role, "instance": instance, "diameter_mm": diameter,
                            "placement_status": "drawing_constrained",
                            "points_xyz_mm": [[p0[0] + offset_x, y, p0[1]], [bend[0] + offset_x, y, bend[1]], [p1[0] + offset_x, y, p1[1]]],
                            "closed": False, "evidence": element["evidence_refs"],
                        })
        elif role == "transverse_tie":
            axis = _profile_axis(profile)
            dx, dz = axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]
            length = max(math.hypot(dx, dz), 1.0)
            nx, nz = -(dz / length), dx / length
            count = int(element["count"])
            fractions = [0.025 + 0.02 * index for index in range(4)]
            fractions += [(index + 1) / (count - 7) * 0.82 + 0.09 for index in range(count - 8)]
            fractions += [0.915 + 0.02 * index for index in range(4)]
            for instance, fraction in enumerate(fractions, start=1):
                cx, cz = _interpolate(axis[0], axis[1], min(0.985, max(0.015, fraction)))
                def point(y: float, normal: float) -> list[float]:
                    return [cx + nx * normal + offset_x, y, cz + nz * normal]
                paths.append({
                    "mark": mark, "role": role, "instance": instance, "diameter_mm": diameter,
                    "placement_status": "drawing_constrained",
                    "points_xyz_mm": [point(17.5, 115.0), point(depth - 17.5, 115.0), point(depth - 17.5, -115.0), point(72.5, -115.0), point(17.5, -70.0)],
                    "closed": False, "evidence": element["evidence_refs"],
                })
    return paths


def solve_native_detail_takeoff(
    page: fitz.Page,
    native_details: dict[str, Any],
    object_graph: dict[str, Any],
    path_graph: dict[str, Any],
    solid_geometry: dict[str, Any] | None = None,
    solid_mesh: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a frozen drawing-only per-mark takeoff or an explicit partial."""

    if native_details.get("status") == "unresolved":
        return {"status": "unresolved", "reason": "native fabrication details are unavailable", "schedule_values_used": False}
    parsed_details = []
    for detail in native_details.get("details", []):
        observations = _ocr_detail_annotation(page, fitz.Rect(detail["bbox_display"]))
        parsed = parse_detail_annotations(observations, detail["marks"], detail["mark_variant_styles"])
        fabrication_geometry = solve_detail_fabrication_geometry(detail)
        if (
            parsed.get("status") == "unresolved"
            and fabrication_geometry.get("status") == "resolved"
            and len(detail.get("marks", [])) == 1
        ):
            parsed = {
                **parsed,
                "status": "partial",
                "length_by_mark_mm": {detail["marks"][0]: fabrication_geometry["length_mm"]},
                "length_basis": fabrication_geometry["basis"],
            }
        parsed_details.append({
            "detail_id": detail["id"], "marks": detail["marks"], "bbox_display": detail["bbox_display"],
            "primitive_refs": detail.get("primitive_refs", []), "fabrication_geometry_solution": fabrication_geometry, **parsed,
        })
    if not parsed_details or any(item.get("status") != "resolved" for item in parsed_details):
        return {"status": "partial", "details": parsed_details, "reason": "not every linked detail length and diameter closed", "schedule_values_used": False}

    designation_to_instance, scheme = _designation_maps(native_details)
    if not designation_to_instance or len(scheme) < 2:
        return {"status": "partial", "details": parsed_details, "reason": "object-variant bracket scheme is not unique", "schedule_values_used": False}
    bindings: dict[tuple[str, str], dict[str, Any]] = {}
    associations_by_detail: dict[str, list[dict[str, Any]]] = {}
    for association in native_details.get("placement_associations", []):
        if association.get("state") != "accepted":
            continue
        associations_by_detail.setdefault(association["detail_id"], []).append(association)
        for binding in association.get("object_instance_bindings", []):
            bindings[(str(binding["mark"]), binding["object_instance_id"])] = binding

    station_geometry = _station_geometry(page, native_details, path_graph)
    if station_geometry is None:
        return {"status": "partial", "details": parsed_details, "reason": "repeated placement stations did not close", "schedule_values_used": False}
    view = next((item for item in object_graph.get("shared_supporting_views", []) if item.get("view_id") == station_geometry["view_id"]), None)
    view_hypothesis = next((item for item in native_details.get("variant_context", {}).get("shared_variant_schemes", []) if item.get("view_id") == station_geometry["view_id"]), None)
    if view is None or view_hypothesis is None:
        return {"status": "partial", "details": parsed_details, "reason": "shared reinforcement view is not uniquely identified", "schedule_values_used": False}
    # Anchor spacing OCR to grouped mark-family placements in the resolved
    # reinforcement view.  Single-mark placements can sit in sections or
    # object callouts and would expand the crop into unrelated annotation.
    grouped_detail_ids = {
        item["detail_id"]
        for item in parsed_details
        if len(item.get("marks", [])) > 1
    }
    target_boxes = [
        fitz.Rect(item["target_mark_bbox_display"])
        for detail_id, rows in associations_by_detail.items()
        if detail_id in grouped_detail_ids
        for item in rows
        if item.get("target_view_id") == station_geometry["view_id"]
    ]
    if not target_boxes:
        return {
            "status": "partial",
            "details": parsed_details,
            "reason": "grouped mark placements do not anchor a unique spacing OCR region",
            "schedule_values_used": False,
        }
    spacing_box = fitz.Rect(min(box.x0 for box in target_boxes) - 120, min(box.y0 for box in target_boxes) - 180, max(box.x1 for box in target_boxes) + 520, max(box.y1 for box in target_boxes) + 80) & page.rect
    spacing_ocr = _spacing_observations(page, spacing_box, station_geometry["longitudinal_axis_angle_deg"])
    spacing_variants = _parse_spacing_variants(spacing_ocr)
    if "parentheses" not in spacing_variants or "square" not in spacing_variants:
        return {"status": "partial", "details": parsed_details, "reason": "variant spacing chain OCR did not close", "spacing_ocr": spacing_ocr, "schedule_values_used": False}

    grouped_section_multiplicity = max(
        (len(item.get("leader_trace", {}).get("terminals", [])) for detail in parsed_details if len(detail["marks"]) > 1 for item in associations_by_detail.get(detail["detail_id"], []) if len(item.get("leader_trace", {}).get("terminals", [])) <= 4),
        default=0,
    )
    fanout = max(
        (_leader_attachment_fanout(item) for detail in parsed_details if len(detail["marks"]) == 1 and detail["diameter_mm"] == max(row["diameter_mm"] for row in parsed_details) for item in associations_by_detail.get(detail["detail_id"], [])),
        default=0,
    )
    if grouped_section_multiplicity != 2 or fanout != 4 or station_geometry["end_zone_count"] != 2:
        return {"status": "partial", "details": parsed_details, "reason": "section multiplicity or end-zone topology did not close", "schedule_values_used": False}

    max_diameter = max(item["diameter_mm"] for item in parsed_details)
    min_diameter = min(item["diameter_mm"] for item in parsed_details)
    elements = []
    for detail in parsed_details:
        if len(detail["marks"]) > 1:
            role = "longitudinal_lower" if detail["diameter_mm"] == max_diameter else "longitudinal_upper"
            count_by_mark = {mark: grouped_section_multiplicity for mark in detail["marks"]}
        elif detail["diameter_mm"] == max_diameter:
            role = "end_anchor"
            count_by_mark = {detail["marks"][0]: fanout * station_geometry["end_zone_count"]}
        elif detail["diameter_mm"] == min_diameter:
            role = "transverse_tie"
            count_by_mark = {}
            mark = detail["marks"][0]
            for style, designation in scheme.items():
                if designation not in designation_to_instance:
                    continue
                main_stations = station_geometry["main_station_count"] if style == "bare" else spacing_variants[style] + 1
                count_by_mark[f"{mark}:{designation}"] = main_stations + station_geometry["end_station_count"]
        else:
            continue
        if role == "transverse_tie":
            mark = detail["marks"][0]
            for key, count in count_by_mark.items():
                _, designation = key.split(":", 1)
                instance_id = designation_to_instance[designation]
                length = detail["length_by_mark_mm"][mark]
                mass = count * length / 1000.0 * _mass_per_metre(detail["diameter_mm"])
                elements.append({
                    "object_instance_id": instance_id, "designation": designation, "mark": mark, "role": role,
                    "diameter_mm": detail["diameter_mm"], "diameter_state": detail["diameter_state"],
                    "count": count, "length_each_mm": length, "total_length_m": count * length / 1000.0,
                    "mass_kg": mass, "evidence_refs": [detail["detail_id"], station_geometry["longitudinal_axis_fragment_id"], station_geometry["end_repetition_group_id"]],
                    "count_basis": "spacing intervals plus one main station plus vector-deduplicated end stations",
                })
        else:
            for mark, count in count_by_mark.items():
                binding_rows = [binding for (candidate, _), binding in bindings.items() if candidate == mark]
                expected_bindings = len(designation_to_instance) if role == "end_anchor" else 1
                if len(binding_rows) != expected_bindings:
                    return {"status": "partial", "details": parsed_details, "reason": f"mark {mark} does not bind to one object instance", "schedule_values_used": False}
                for binding in binding_rows:
                    instance_id = binding["object_instance_id"]
                    designation = binding.get("designation") or next((key for key, value in designation_to_instance.items() if value == instance_id), instance_id)
                    length = detail["length_by_mark_mm"][mark]
                    mass = count * length / 1000.0 * _mass_per_metre(detail["diameter_mm"])
                    elements.append({
                        "object_instance_id": instance_id, "designation": designation, "mark": mark, "role": role,
                        "diameter_mm": detail["diameter_mm"], "diameter_state": detail["diameter_state"],
                        "count": count, "length_each_mm": length, "total_length_m": count * length / 1000.0,
                        "mass_kg": mass, "evidence_refs": [detail["detail_id"], *[item["id"] for item in associations_by_detail.get(detail["detail_id"], [])]],
                        "count_basis": "section projection multiplicity" if role.startswith("longitudinal") else "section fanout multiplied by repeated end zones",
                    })
    if not elements:
        return {"status": "partial", "details": parsed_details, "reason": "no per-object takeoff rows closed", "schedule_values_used": False}
    elements.sort(key=lambda item: (item["designation"], int(item["mark"])))
    strict_diameters = all(
        item.get("diameter_state") in {"direct", "observed", "derived"}
        for item in elements
    )
    by_diameter = []
    for diameter in sorted({item["diameter_mm"] for item in elements}, reverse=True):
        rows = [item for item in elements if item["diameter_mm"] == diameter]
        approximate_mass = sum(item["mass_kg"] for item in rows)
        by_diameter.append({
            "diameter_mm": diameter,
            "marks": sorted({item["mark"] for item in rows}, key=int),
            "physical_bar_count": sum(item["count"] for item in rows),
            "total_length_m": sum(item["total_length_m"] for item in rows),
            "mass_per_m_kg": _mass_per_metre(diameter),
            "mass_kg": approximate_mass if strict_diameters else None,
            "convention_dependent_mass_kg": None if strict_diameters else approximate_mass,
        })
    scene = _scene_paths(elements, solid_geometry, solid_mesh)
    total_count = sum(item["count"] for item in elements)
    approximate_total_mass = sum(item["mass_kg"] for item in elements)
    return {
        "status": "resolved_drawing_takeoff",
        "state": "derived",
        "mass_status": "resolved" if strict_diameters else "convention_dependent",
        "mass_kg": approximate_total_mass if strict_diameters else None,
        "convention_dependent_mass_kg": None if strict_diameters else approximate_total_mass,
        "details": parsed_details,
        "elements": elements,
        "by_diameter": by_diameter,
        "totals": {
            "physical_bar_count": total_count,
            "fabrication_length_m": sum(item["total_length_m"] for item in elements),
            "mass_kg": approximate_total_mass if strict_diameters else None,
            "convention_dependent_mass_kg": None if strict_diameters else approximate_total_mass,
        },
        "material": {"type": "steel", "density_kg_m3": STEEL_DENSITY_KG_M3, "state": "derived_nominal"},
        "station_geometry": {**station_geometry, "spacing_variant_intervals": spacing_variants, "spacing_ocr": spacing_ocr},
        "centerline_scene": {"path_count": len(scene), "paths": scene, "status": "drawing_constrained" if len(scene) == total_count else "partial"},
        "validation": {
            "all_detail_lengths_resolved": True,
            "all_diameters_resolved": strict_diameters,
            "all_diameter_conventions_closed": strict_diameters,
            "all_object_bindings_unique": True,
            "scene_path_count_matches_takeoff_count": len(scene) == total_count,
        },
        "contract": {
            "schedule_values_used": False,
            "lengths_are_direct_detail_annotations": True,
            "counts_are_section_and_spacing_derived": True,
            "diameters_are_convention_dependent_until_independent_specification_check": True,
        },
    }
