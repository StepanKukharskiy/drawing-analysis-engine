"""Mark-linked fabrication-detail extraction and deterministic length solving.

This module deliberately treats an embedded detail image as drawing evidence,
not as permission to read neighbouring schedule values.  It links a known mark
to a nearby image primitive, recovers paired-stroke centreline candidates and
numeric labels inside that image, and then solves only the terms that are
actually constrained.  Missing hook or bend information remains explicit.
"""

from __future__ import annotations

from functools import lru_cache
import io
import math
import re
from typing import Any

import fitz


def _image_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    blocks = []
    for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
        if block.get("type") != 1 or not block.get("image"):
            continue
        box = fitz.Rect(block["bbox"])
        if box.width < 12 or box.height < 12:
            continue
        blocks.append(
            {
                "block_index": block_index,
                "image_number": int(block.get("number", block_index)),
                "bbox": box,
                "width_px": int(block.get("width", 0)),
                "height_px": int(block.get("height", 0)),
                "extension": block.get("ext"),
                "image": bytes(block["image"]),
            }
        )
    return blocks


def _mark_words(page: fitz.Page, mark: str) -> list[dict[str, Any]]:
    words = []
    for word_index, word in enumerate(page.get_text("words")):
        if str(word[4]).strip() != str(mark):
            continue
        words.append(
            {
                "bbox": fitz.Rect(word[:4]),
                "primitive_ref": f"native_word[{word_index}]",
            }
        )
    return words


def _match_detail_image(page: fitz.Page, mark: str) -> dict[str, Any] | None:
    """Match by local row adjacency; absolute page coordinates are never used."""

    ranked = []
    for word in _mark_words(page, mark):
        word_box = word["bbox"]
        word_center_y = (word_box.y0 + word_box.y1) / 2
        for image in _image_blocks(page):
            image_box = image["bbox"]
            gap = image_box.x0 - word_box.x1
            vertical_residual = max(image_box.y0 - word_center_y, word_center_y - image_box.y1, 0.0)
            height_ratio = image_box.height / max(word_box.height, 1.0)
            # A fabrication sketch and its mark occupy the same local row.  A
            # strict rightward gap prevents section labels elsewhere on the
            # page from claiming unrelated images.
            if not (0.0 <= gap <= 90.0 and vertical_residual <= 8.0 and 2.0 <= height_ratio <= 18.0):
                continue
            score = 1.0 - min(0.45, gap / 200.0) - min(0.25, vertical_residual / 32.0)
            ranked.append((score, gap, word, image))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    best = ranked[0]
    if len(ranked) > 1 and best[0] - ranked[1][0] < 0.08:
        return None
    return {
        "score": round(best[0], 3),
        "method": "same_row_mark_to_embedded_image_adjacency",
        "mark_bbox_display": list(best[2]["bbox"]),
        "mark_primitive_ref": best[2]["primitive_ref"],
        **best[3],
    }


def _line_angle(line: tuple[float, float, float, float]) -> float:
    angle = math.degrees(math.atan2(line[3] - line[1], line[2] - line[0])) % 180.0
    return angle


def _angle_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180.0
    return min(difference, 180.0 - difference)


def _paired_centerlines(image: Any) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    minimum = min(width, height)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    detected = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=max(25, int(minimum * 0.06)),
        minLineLength=max(30, int(minimum * 0.10)),
        maxLineGap=max(8, int(minimum * 0.025)),
    )
    if detected is None:
        return []
    raw = [tuple(float(value) for value in row[0]) for row in detected]
    candidates = []
    for left_index, left in enumerate(raw):
        left_angle = _line_angle(left)
        theta = math.radians(left_angle)
        unit = (math.cos(theta), math.sin(theta))
        normal = (-unit[1], unit[0])
        left_points = ((left[0], left[1]), (left[2], left[3]))
        left_projection = sorted(point[0] * unit[0] + point[1] * unit[1] for point in left_points)
        left_normal = sum(point[0] * normal[0] + point[1] * normal[1] for point in left_points) / 2
        left_length = left_projection[1] - left_projection[0]
        for right in raw[left_index + 1 :]:
            if _angle_distance(left_angle, _line_angle(right)) > 3.0:
                continue
            right_points = ((right[0], right[1]), (right[2], right[3]))
            right_projection = sorted(point[0] * unit[0] + point[1] * unit[1] for point in right_points)
            right_normal = sum(point[0] * normal[0] + point[1] * normal[1] for point in right_points) / 2
            separation = abs(right_normal - left_normal)
            if not (max(4.0, minimum * 0.008) <= separation <= minimum * 0.07):
                continue
            overlap_start = max(left_projection[0], right_projection[0])
            overlap_end = min(left_projection[1], right_projection[1])
            overlap = overlap_end - overlap_start
            right_length = right_projection[1] - right_projection[0]
            if overlap <= 0 or overlap / max(1.0, min(left_length, right_length)) < 0.62:
                continue
            centre_normal = (left_normal + right_normal) / 2
            start = (unit[0] * overlap_start + normal[0] * centre_normal, unit[1] * overlap_start + normal[1] * centre_normal)
            end = (unit[0] * overlap_end + normal[0] * centre_normal, unit[1] * overlap_end + normal[1] * centre_normal)
            candidates.append(
                {
                    "start_px": [round(start[0], 2), round(start[1], 2)],
                    "end_px": [round(end[0], 2), round(end[1], 2)],
                    "length_px": round(overlap, 2),
                    "angle_deg": round(left_angle, 2),
                    "paired_stroke_separation_px": round(separation, 2),
                }
            )

    # Hough emits many versions of the same stroke.  Keep the longest member
    # of each local angle/midpoint cluster so the record remains compact.
    deduplicated: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["length_px"], reverse=True):
        midpoint = [
            (candidate["start_px"][0] + candidate["end_px"][0]) / 2,
            (candidate["start_px"][1] + candidate["end_px"][1]) / 2,
        ]
        duplicate = False
        for kept in deduplicated:
            kept_midpoint = [
                (kept["start_px"][0] + kept["end_px"][0]) / 2,
                (kept["start_px"][1] + kept["end_px"][1]) / 2,
            ]
            if _angle_distance(candidate["angle_deg"], kept["angle_deg"]) <= 4.0 and math.dist(midpoint, kept_midpoint) <= minimum * 0.075:
                duplicate = True
                break
        if duplicate:
            continue
        candidate["start_normalized"] = [round(candidate["start_px"][0] / width, 6), round(candidate["start_px"][1] / height, 6)]
        candidate["end_normalized"] = [round(candidate["end_px"][0] / width, 6), round(candidate["end_px"][1] / height, 6)]
        candidate["orientation"] = (
            "horizontal"
            if min(candidate["angle_deg"], 180 - candidate["angle_deg"]) <= 12
            else "vertical"
            if abs(candidate["angle_deg"] - 90) <= 12
            else "diagonal"
        )
        candidate["state"] = "image_derived"
        deduplicated.append(candidate)
        if len(deduplicated) >= 16:
            break
    return deduplicated


def _line_intersection(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, float] | None:
    x1, y1 = left["start_px"]
    x2, y2 = left["end_px"]
    x3, y3 = right["start_px"]
    x4, y4 = right["end_px"]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    determinant_left = x1 * y2 - y1 * x2
    determinant_right = x3 * y4 - y3 * x4
    return (
        (determinant_left * (x3 - x4) - (x1 - x2) * determinant_right) / denominator,
        (determinant_left * (y3 - y4) - (y1 - y2) * determinant_right) / denominator,
    )


def _bend_candidates(lines: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    bends = []
    tolerance = min(width, height) * 0.10
    for left_index, left in enumerate(lines):
        for right_index, right in enumerate(lines[left_index + 1 :], start=left_index + 1):
            angle = _angle_distance(left["angle_deg"], right["angle_deg"])
            if not (20 <= angle <= 120):
                continue
            point = _line_intersection(left, right)
            if point is None:
                continue
            left_endpoint_distance = min(math.dist(point, left["start_px"]), math.dist(point, left["end_px"]))
            right_endpoint_distance = min(math.dist(point, right["start_px"]), math.dist(point, right["end_px"]))
            if max(left_endpoint_distance, right_endpoint_distance) > tolerance:
                continue
            bends.append(
                {
                    "id": f"bend_candidate.{len(bends) + 1:03d}",
                    "line_indexes": [left_index, right_index],
                    "vertex_px": [round(point[0], 2), round(point[1], 2)],
                    "vertex_normalized": [round(point[0] / width, 6), round(point[1] / height, 6)],
                    "deflection_angle_deg": round(angle, 2),
                    "arc": {"radius_mm": None, "tangent_points": None, "state": "unresolved"},
                    "state": "topology_candidate",
                }
            )
            if len(bends) >= 12:
                return bends
    return bends


def _ocr_dimensions(image: Any) -> list[dict[str, Any]]:
    import cv2
    import pytesseract

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    candidates = []
    # Sparse-text modes find the label boxes without asking OCR to understand
    # the entire engineering sketch as prose.
    for psm in (11, 12):
        try:
            data = pytesseract.image_to_data(
                gray,
                config=f"--psm {psm} -l eng -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
                timeout=7,
            )
        except (OSError, RuntimeError, pytesseract.TesseractError):
            continue
        for token, confidence, x, y, box_width, box_height in zip(
            data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
        ):
            normalized = re.sub(r"\D", "", str(token))
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                continue
            if not re.fullmatch(r"\d{2,5}", normalized) or confidence_value < 50:
                continue
            value = int(normalized)
            if not 10 <= value <= 10000:
                continue
            box = [float(x), float(y), float(x + box_width), float(y + box_height)]
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if any(math.dist(center, row["center"]) < min(width, height) * 0.06 for row in candidates):
                continue
            candidates.append({"box": box, "center": center, "detected_value": value, "detected_confidence": confidence_value / 100.0})

    observations = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["box"]
        pad = max(10, int(min(x1 - x0, y1 - y0) * 0.45))
        crop = gray[max(0, int(y0) - pad) : min(height, int(y1) + pad), max(0, int(x0) - pad) : min(width, int(x1) + pad)]
        orientation = "vertical" if y1 - y0 > 1.15 * (x1 - x0) else "horizontal"
        if orientation == "vertical":
            crop = __import__("numpy").rot90(crop, 1).copy()
        votes: dict[int, int] = {}
        for psm in (6, 7, 10, 11, 12):
            try:
                text = pytesseract.image_to_string(
                    crop,
                    config=f"--psm {psm} -l eng -c tessedit_char_whitelist=0123456789",
                    timeout=5,
                )
            except (OSError, RuntimeError, pytesseract.TesseractError):
                continue
            normalized = re.sub(r"\D", "", text)
            if re.fullmatch(r"\d{2,5}", normalized) and 10 <= int(normalized) <= 10000:
                votes[int(normalized)] = votes.get(int(normalized), 0) + 1
        if not votes:
            continue
        value, vote_count = max(votes.items(), key=lambda row: (row[1], row[0] == candidate["detected_value"]))
        if vote_count < 2:
            continue
        observations.append(
            {
                "value_mm": value,
                "text": str(value),
                "confidence": round(max(candidate["detected_confidence"], min(0.98, 0.55 + 0.08 * vote_count)), 3),
                "bbox_px": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "bbox_normalized": [round(x0 / width, 6), round(y0 / height, 6), round(x1 / width, 6), round(y1 / height, 6)],
                "orientation": orientation,
                "reading_rotation_deg": 90 if orientation == "vertical" else 0,
                "text_method": "embedded_image_ocr_local_consensus",
                "ocr_votes": {str(key): count for key, count in sorted(votes.items())},
                "state": "observed",
            }
        )
    return observations


def _attach_dimensions(dimensions: list[dict[str, Any]], lines: list[dict[str, Any]], width: int, height: int) -> None:
    if not lines:
        for observation in dimensions:
            observation["attachment"] = {"role": None, "state": "unknown", "score": 0.0}
        return
    points = [point for line in lines for point in (line["start_px"], line["end_px"])]
    shape = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
    center_x = (shape[0] + shape[2]) / 2
    center_y = (shape[1] + shape[3]) / 2
    for observation in dimensions:
        box = observation["bbox_px"]
        token_x = (box[0] + box[2]) / 2
        token_y = (box[1] + box[3]) / 2
        role = None
        score = 0.0
        if observation["orientation"] == "horizontal" and token_y < center_y and shape[0] - width * 0.2 <= token_x <= shape[2] + width * 0.2:
            role, score = "overall_width", 0.86
        elif observation["orientation"] == "vertical" and token_x > center_x and shape[1] - height * 0.2 <= token_y <= shape[3] + height * 0.2:
            role, score = "overall_height", 0.86
        observation["attachment"] = {
            "role": role,
            "state": "derived" if role else "unknown",
            "score": score,
            "basis": "orientation_and_containment_relative_to_extracted_detail_geometry" if role else None,
        }


@lru_cache(maxsize=64)
def _extract_cached(image_bytes: bytes) -> dict[str, Any]:
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return {"status": "unresolved", "lines": [], "bends": [], "dimensions": [], "width_px": 0, "height_px": 0}
    height, width = decoded.shape[:2]
    lines = _paired_centerlines(decoded)
    bends = _bend_candidates(lines, width, height)
    try:
        dimensions = _ocr_dimensions(decoded)
    except (ImportError, OSError):
        dimensions = []
    _attach_dimensions(dimensions, lines, width, height)
    orientations = {line["orientation"] for line in lines}
    topology = (
        "closed_loop_with_hook_candidates"
        if {"horizontal", "vertical", "diagonal"}.issubset(orientations)
        else "polyline_candidate"
        if lines
        else "unresolved"
    )
    return {
        "status": "partial_geometry" if lines else "unresolved",
        "width_px": width,
        "height_px": height,
        "topology_class": topology,
        "lines": lines,
        "bends": bends,
        "arcs": [
            {
                "bend_id": bend["id"],
                "radius_mm": None,
                "sweep_deg": bend["deflection_angle_deg"],
                "state": "unresolved_radius",
            }
            for bend in bends
        ],
        "dimensions": dimensions,
    }


def solve_cutting_length(detail: dict[str, Any] | None, quantity: int) -> dict[str, Any]:
    """Evaluate the constrained part of a fabrication equation, then fail closed."""

    if not detail:
        return {
            "status": "unresolved",
            "cutting_length_each_mm": None,
            "cutting_length_total_mm": None,
            "equation": None,
            "known_terms_total_mm": None,
            "missing_constraints": ["mark-linked fabrication detail", "fabrication dimensions", "bend and hook rules"],
        }
    attached: dict[str, list[float]] = {}
    for item in detail.get("geometry", {}).get("dimensions", []):
        role = item.get("attachment", {}).get("role")
        if role:
            attached.setdefault(role, []).append(float(item["value_mm"]))
    width_values = attached.get("overall_width", [])
    height_values = attached.get("overall_height", [])
    width = sum(width_values) / len(width_values) if width_values else None
    height = sum(height_values) / len(height_values) if height_values else None
    known = None if width is None or height is None else round(2 * width + 2 * height, 3)
    missing = []
    if width is None:
        missing.append("overall width attached to detail centreline")
    if height is None:
        missing.append("overall height attached to detail centreline")
    if detail.get("geometry", {}).get("bends"):
        missing.append("bend radii and centreline correction convention")
    if detail.get("geometry", {}).get("topology_class") == "closed_loop_with_hook_candidates":
        missing.append("complete hook count and hook extensions")
    if missing:
        equation = None
        if known is not None:
            equation = f"2*{width:g} + 2*{height:g} + hook_terms + bend_corrections"
        return {
            "status": "partial_equation" if known is not None else "partial_geometry",
            "cutting_length_each_mm": None,
            "cutting_length_total_mm": None,
            "equation": equation,
            "known_terms_total_mm": known,
            "missing_constraints": missing,
        }
    # This branch is intentionally narrow.  Future detail families may supply
    # complete segment/arc terms, at which point the same record can be summed.
    each = round(float(known), 3)
    return {
        "status": "resolved",
        "cutting_length_each_mm": each,
        "cutting_length_total_mm": round(each * quantity, 3),
        "equation": f"2*{width:g} + 2*{height:g}",
        "known_terms_total_mm": each,
        "missing_constraints": [],
    }


def attach_fabrication_details(
    page: fitz.Page,
    groups: list[dict[str, Any]],
    source_groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    details = []
    for group_index, group in enumerate(groups):
        mark = group.get("identity", {}).get("mark", {}).get("value")
        match = _match_detail_image(page, str(mark)) if mark is not None else None
        detail = None
        if match is not None:
            geometry = _extract_cached(match.pop("image"))
            detail = {
                "id": f"fabrication_detail.{len(details) + 1:03d}",
                "group_id": group["id"],
                "mark": str(mark),
                "association": {
                    "method": match.pop("method"),
                    "score": match.pop("score"),
                    "mark_bbox_display": match.pop("mark_bbox_display"),
                    "mark_primitive_ref": match.pop("mark_primitive_ref"),
                },
                "source": {
                    "kind": "embedded_raster_image",
                    "bbox_display": list(match.pop("bbox")),
                    "primitive_ref": f"image_block[{match['block_index']}]",
                    **match,
                },
                "geometry": geometry,
                "state": geometry["status"],
            }
            details.append(detail)
        metric = (
            (source_groups or [])[group_index].get("dimension_anchored_solution")
            if source_groups is not None and group_index < len(source_groups)
            else None
        )
        if (
            detail is None
            and group.get("topology", {}).get("family", {}).get("value") == "straight"
            and metric
            and metric.get("status") == "pass"
            and metric.get("fabrication_eligibility", {}).get("installed_equals_cutting")
        ):
            each = float(metric["solved_centerline"]["installed_length_each_mm"])
            start = float(metric["solved_centerline"]["z_start_mm"])
            end = float(metric["solved_centerline"]["z_end_mm"])
            solution = {
                "status": "resolved",
                "cutting_length_each_mm": each,
                "cutting_length_total_mm": round(each * int(group.get("quantity", {}).get("value", 0)), 3),
                "equation": f"{end:g} - {start:g} = {each:g}",
                "known_terms_total_mm": each,
                "missing_constraints": [],
                "length_equivalence": "installed_centerline_equals_cutting_length_for_validated_straight_unbent_topology",
            }
        else:
            solution = solve_cutting_length(detail, int(group.get("quantity", {}).get("value", 0)))
        group["fabrication"] = {
            **solution,
            "detail_id": None if detail is None else detail["id"],
            "basis": (
                "mark-linked_detail_geometry_and_attached_dimensions"
                if detail
                else "dimension_anchored_straight_centerline_with_cross_view_reprojection"
                if solution["status"] == "resolved"
                else None
            ),
            "schedule_length_used": False,
        }
    return details
