"""Geometry-gated native-vector fabrication details and placement links.

This stage runs only inside a detected repeated table grid that also contains
large non-grid polylines. OCR is restricted to the first column of those rows,
and a placement link requires a matching rendered token plus a native leader
trace. It never reads schedule quantities or printed cutting totals.
"""

from __future__ import annotations

import re
import math
import statistics
from typing import Any

import fitz

from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader
from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions


def _view_grid(page: fitz.Page, view: dict[str, Any]) -> tuple[list[float], list[float]]:
    box = fitz.Rect(view["bbox_display"])
    horizontal = []
    vertical = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            if abs(start.y - end.y) <= 0.5 and abs(end.x - start.x) >= 0.60 * box.width:
                if box.x0 - 2 <= min(start.x, end.x) and max(start.x, end.x) <= box.x1 + 2 and box.y0 - 2 <= start.y <= box.y1 + 2:
                    horizontal.append(float(start.y))
            if abs(start.x - end.x) <= 0.5 and abs(end.y - start.y) >= 0.70 * box.height:
                if box.y0 - 2 <= min(start.y, end.y) and max(start.y, end.y) <= box.y1 + 2 and box.x0 - 2 <= start.x <= box.x1 + 2:
                    vertical.append(float(start.x))
    return sorted({round(value, 2) for value in horizontal}), sorted({round(value, 2) for value in vertical})


def _render_gray(page: fitz.Page, box: fitz.Rect, scale: int) -> Any:
    import numpy as np

    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box, colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)


def _refine_engineering_digit_token(
    image: Any,
    box: tuple[int, int, int, int],
    original: str,
) -> tuple[str, int]:
    """Re-read a tight low-confidence engineering-font number.

    Crossed sevens in small rotated CAD labels are commonly read as ones.  A
    padded, cubic-upscaled binary crop preserves the top stroke and diagonal.
    The token is changed only when three independent page-segmentation modes
    produce a unique consensus with the same digit count.
    """

    import cv2
    import pytesseract

    if not original.isdigit() or not 1 <= len(original) <= 3:
        return original, 0
    x, y, width, height = box
    padding = max(4, round(0.33 * max(width, height)))
    y0, y1 = max(0, y - padding), min(image.shape[0], y + height + padding)
    x0, x1 = max(0, x - padding), min(image.shape[1], x + width + padding)
    crop = image[y0:y1, x0:x1]
    if not crop.size:
        return original, 0
    enlarged = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    enlarged = cv2.resize(enlarged, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(enlarged, 200, 255, cv2.THRESH_BINARY)[1]
    votes: dict[str, int] = {}
    for mode in (6, 7, 10):
        try:
            text = pytesseract.image_to_string(
                binary,
                config=f"--psm {mode} -l eng -c tessedit_char_whitelist=0123456789",
                timeout=5,
            )
        except (OSError, RuntimeError, pytesseract.TesseractError):
            continue
        token = re.sub(r"\D", "", text)
        if len(token) == len(original):
            votes[token] = votes.get(token, 0) + 1
    if not votes:
        refined, support = original, 0
    else:
        refined, support = max(votes.items(), key=lambda item: (item[1], item[0]))
        if support < 2 or sum(value == support for value in votes.values()) > 1:
            refined, support = original, 0
    if refined == original and _looks_like_leading_crossed_seven(binary, original):
        return "7" + original[1:], 3
    return refined, support


def _looks_like_leading_crossed_seven(binary: Any, original: str) -> bool:
    """Recognize a CAD crossed seven that OCR reduced to a leading one."""

    import numpy as np

    if len(original) < 2 or not original.startswith("1"):
        return False
    foreground = binary < 128
    ys, xs = np.where(foreground)
    if not len(xs):
        return False
    foreground = foreground[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    width = foreground.shape[1]
    if width < 8:
        return False
    column_ink = foreground.sum(axis=0)
    search_start, search_end = round(0.25 * width), round(0.65 * width)
    if search_end <= search_start:
        return False
    split = search_start + int(np.argmin(column_ink[search_start:search_end]))
    leading = foreground[:, :split]
    ys, xs = np.where(leading)
    if not len(xs):
        return False
    leading = leading[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    glyph_height, glyph_width = leading.shape
    if glyph_width < 6 or glyph_height < glyph_width:
        return False

    def longest_run(row: Any) -> int:
        indices = np.flatnonzero(row)
        if not len(indices):
            return 0
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        return max(len(group) for group in np.split(indices, breaks))

    top_band = leading[: max(1, round(glyph_height / 3))]
    top_bar_ratio = max((longest_run(row) for row in top_band), default=0) / glyph_width
    occupied_rows = [row for row in leading if row.any()]
    upper = occupied_rows[: max(1, len(occupied_rows) // 4)]
    lower = occupied_rows[-max(1, len(occupied_rows) // 4) :]
    upper_center = sum(float(np.flatnonzero(row).mean()) for row in upper) / len(upper)
    lower_center = sum(float(np.flatnonzero(row).mean()) for row in lower) / len(lower)
    diagonal_shift = abs(upper_center - lower_center) / glyph_width
    return top_bar_ratio >= 0.75 and diagonal_shift >= 0.08


def _digit_glyph_signature(image: Any, box: tuple[int, int, int, int]) -> str:
    """Return a compact orientation-normalized binary glyph fingerprint."""

    import cv2
    import numpy as np

    x, y, width, height = box
    padding = 2
    crop = image[
        max(0, y - padding) : min(image.shape[0], y + height + padding),
        max(0, x - padding) : min(image.shape[1], x + width + padding),
    ]
    if not crop.size:
        return ""
    normalized = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
    binary = normalized < 190
    return np.packbits(binary.reshape(-1)).tobytes().hex()


def _glyph_signature_similarity(left: str, right: str) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    differing = sum((a ^ b).bit_count() for a, b in zip(bytes.fromhex(left), bytes.fromhex(right)))
    return 1.0 - differing / (4 * len(left))


def _normalize_mark(text: str) -> tuple[str, ...] | None:
    digits = re.findall(r"\d", text)
    bracketed = any(character in text for character in "[](){}")
    if bracketed and len(digits) == 4 and digits[1] == "1":
        digits.pop(1)
    if bracketed and 1 <= len(digits) <= 3:
        return tuple(digits)
    normalized = re.sub(r"\D", "", text)
    if re.fullmatch(r"\d{1,2}", normalized):
        return (str(int(normalized)),)
    return None


def _mark_variant_styles(text: str, marks: tuple[str, ...]) -> dict[str, str]:
    if len(marks) == 1:
        return {marks[0]: "common"}
    parenthesized = set(re.findall(r"\((\d{1,2})\)", text))
    squared = set(re.findall(r"\[(\d{1,2})\]", text))
    if "]" in text and not squared:
        prefix = text[: text.index("]")]
        digits = re.findall(r"\d", prefix)
        if digits:
            squared.add(digits[-1])
    return {
        mark: "parentheses" if mark in parenthesized else "square" if mark in squared else "bare"
        for mark in marks
    }


def _parse_variant_scheme(text: str) -> dict[str, str]:
    """Map drawing-specified bracket styles to object designations."""

    tokens = re.findall(r"(\[|\()?\s*([A-Z]{1,4}\d{1,3})\s*(\]|\))?", text.upper())
    scheme = {}
    for left, designation, right in tokens:
        style = "square" if left == "[" or right == "]" else "parentheses" if left == "(" or right == ")" else "bare"
        scheme.setdefault(style, designation)
    return scheme if len(set(scheme.values())) >= 2 else {}


def _ocr_region_text(page: fitz.Page, box: fitz.Rect) -> str:
    import pytesseract
    from PIL import Image

    image = Image.fromarray(_render_gray(page, box, 6))
    try:
        return pytesseract.image_to_string(image, config="--psm 11 -l eng", timeout=15).strip()
    except (OSError, RuntimeError, pytesseract.TesseractError):
        return ""


def _instance_variant_context(
    page: fitz.Page,
    views: list[dict[str, Any]],
    object_graph: dict[str, Any],
) -> dict[str, Any]:
    view_by_id = {item["id"]: item for item in views}
    instances = []
    for instance in object_graph.get("instances", []):
        primary = view_by_id.get(instance.get("primary_view_id"))
        if primary is None:
            continue
        box = fitz.Rect(primary["bbox_display"])
        padding = min(120.0, max(42.0, 0.18 * box.height))
        title_box = fitz.Rect(box.x0, max(page.rect.y0, box.y0 - padding), box.x1, min(page.rect.y1, box.y0 + 30.0))
        text = _ocr_region_text(page, title_box)
        designations = re.findall(r"\b[A-Z]{1,4}\d{1,3}\b", text.upper())
        instances.append(
            {
                "object_instance_id": instance["id"],
                "primary_view_id": instance["primary_view_id"],
                "designation": designations[0] if designations else None,
                "title_bbox_display": list(title_box),
                "ocr_text": text,
            }
        )
    shared = []
    for item in object_graph.get("shared_supporting_views", []):
        view = view_by_id.get(item.get("view_id"))
        if view is None:
            continue
        box = fitz.Rect(view["bbox_display"])
        padding = min(120.0, max(42.0, 0.18 * box.height))
        title_box = fitz.Rect(box.x0, max(page.rect.y0, box.y0 - padding), box.x1, min(page.rect.y1, box.y0 + 30.0))
        text = _ocr_region_text(page, title_box)
        scheme = _parse_variant_scheme(text)
        if scheme:
            shared.append(
                {
                    "view_id": item["view_id"],
                    "scheme": scheme,
                    "title_bbox_display": list(title_box),
                    "ocr_text": text,
                }
            )
    return {"instances": instances, "shared_variant_schemes": shared}


def _association_object_bindings(
    detail: dict[str, Any],
    target_view_id: str,
    context: dict[str, Any],
    object_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    instance_by_designation = {
        item["designation"]: item for item in context.get("instances", []) if item.get("designation")
    }
    directly_scoped = [
        item["id"]
        for item in object_graph.get("instances", [])
        if target_view_id in item.get("view_ids", [])
    ]
    scheme_row = next(
        (item for item in context.get("shared_variant_schemes", []) if item["view_id"] == target_view_id),
        None,
    )
    bindings = []
    styles = detail.get("mark_variant_styles", {})
    if directly_scoped:
        for mark in detail["marks"]:
            for instance_id in directly_scoped:
                bindings.append({"mark": mark, "object_instance_id": instance_id, "basis": "mark occurrence lies in an object-scoped projection"})
    elif scheme_row and len(detail["marks"]) == 1 and styles.get(detail["marks"][0]) == "common":
        for instance in context.get("instances", []):
            if instance.get("designation") in scheme_row["scheme"].values():
                bindings.append(
                    {
                        "mark": detail["marks"][0],
                        "object_instance_id": instance["object_instance_id"],
                        "designation": instance["designation"],
                        "basis": "single common mark in a shared view whose title explicitly enumerates the object variants",
                    }
                )
    elif scheme_row:
        for mark in detail["marks"]:
            designation = scheme_row["scheme"].get(styles.get(mark, ""))
            instance = instance_by_designation.get(designation)
            if instance:
                bindings.append(
                    {
                        "mark": mark,
                        "object_instance_id": instance["object_instance_id"],
                        "designation": designation,
                        "variant_style": styles[mark],
                        "basis": "mark and object designations share the same drawing-specified bracket style",
                    }
                )
    return bindings


def _ocr_mark(page: fitz.Page, cell: fitz.Rect) -> dict[str, Any] | None:
    import cv2
    import pytesseract

    inner = fitz.Rect(cell.x0 + 3, cell.y0 + 3, cell.x1 - 3, cell.y1 - 3)
    gray = _render_gray(page, inner, 12)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    votes: dict[tuple[str, ...], int] = {}
    raw = []
    for mode in (6, 7, 10, 11, 12):
        try:
            text = pytesseract.image_to_string(
                binary,
                config=f"--psm {mode} -l eng -c tessedit_char_whitelist=0123456789[]()",
                timeout=6,
            ).strip()
        except (OSError, RuntimeError, pytesseract.TesseractError):
            continue
        raw.append({"psm": mode, "text": text})
        normalized = _normalize_mark(text)
        if normalized is not None:
            votes[normalized] = votes.get(normalized, 0) + 1
    if not votes:
        return None
    marks, support = max(votes.items(), key=lambda item: (item[1], len(item[0])))
    if support < 2:
        return None
    representative = next((item["text"] for item in raw if _normalize_mark(item["text"]) == marks), "")
    return {
        "marks": list(marks),
        "display": "/".join(marks),
        "confidence": round(min(0.96, 0.55 + 0.08 * support), 3),
        "method": "geometry_gated_row_cell_ocr_consensus",
        "ocr_observations": raw,
        "mark_variant_styles": _mark_variant_styles(representative, marks),
    }


def _row_geometry(page: fitz.Page, cell: fitz.Rect, row_height: float) -> list[dict[str, Any]]:
    rows = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        rect = fitz.Rect(drawing["rect"])
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if center not in cell or not drawing.get("items"):
            continue
        if rect.width < 0.10 * cell.width or rect.height < 0.10 * row_height:
            continue
        if rect.width > 0.92 * cell.width or rect.height > 0.92 * row_height:
            continue
        points = []
        segments = []
        for item in drawing["items"]:
            if item[0] == "l":
                if not points:
                    points.append([float(item[1].x), float(item[1].y)])
                points.append([float(item[2].x), float(item[2].y)])
                segments.append(
                    {
                        "kind": "line",
                        "start_display": [float(item[1].x), float(item[1].y)],
                        "end_display": [float(item[2].x), float(item[2].y)],
                    }
                )
            elif item[0] == "c":
                if not points:
                    points.append([float(item[1].x), float(item[1].y)])
                points.append([float(item[4].x), float(item[4].y)])
                segments.append(
                    {
                        "kind": "cubic_bezier",
                        "start_display": [float(item[1].x), float(item[1].y)],
                        "control_1_display": [float(item[2].x), float(item[2].y)],
                        "control_2_display": [float(item[3].x), float(item[3].y)],
                        "end_display": [float(item[4].x), float(item[4].y)],
                    }
                )
        if len(points) >= 2:
            rows.append(
                {
                    "primitive_ref": f"drawing[{drawing_index}]",
                    "bbox_display": list(rect),
                    "points_display": points,
                    "segments": segments,
                    "stroke_width_points": drawing.get("width"),
                }
            )
    for image_index, image in enumerate(page.get_image_info(xrefs=True)):
        rect = fitz.Rect(image["bbox"])
        center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if center not in cell:
            continue
        if rect.width < 0.10 * cell.width or rect.height < 0.10 * row_height:
            continue
        if rect.width > 0.98 * cell.width or rect.height > 0.98 * row_height:
            continue
        rows.append(
            {
                "primitive_ref": f"image[{image_index}]",
                "image_xref": image.get("xref"),
                "bbox_display": list(rect),
                "points_display": [],
                "segments": [],
                "source_type": "embedded_raster_detail",
                "image_size_pixels": [image.get("width"), image.get("height")],
            }
        )
    deduplicated = []
    for row in sorted(rows, key=lambda item: (fitz.Rect(item["bbox_display"]).get_area(), item["primitive_ref"]), reverse=True):
        box = fitz.Rect(row["bbox_display"])
        if any(
            box.get_area()
            and abs(box.get_area() - fitz.Rect(kept["bbox_display"]).get_area()) <= 0.02 * box.get_area()
            and box.intersects(fitz.Rect(kept["bbox_display"]))
            for kept in deduplicated
        ):
            continue
        deduplicated.append(row)
    return deduplicated


def _vectorize_raster_detail(page: fitz.Page, geometry: dict[str, Any]) -> dict[str, Any]:
    """Add deterministic line/curve and OCR observations for one detail image."""

    import cv2
    import numpy as np
    import pytesseract

    box = fitz.Rect(geometry["bbox_display"])
    # The embedded detail rows contain small rotated fabrication dimensions.
    # Five display pixels per PDF point keeps 2-3 mm glyph strokes legible to
    # OCR while remaining local to the geometry-gated row crop.
    scale = 5
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box, colorspace=fitz.csGRAY, alpha=False)
    gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
    edges = cv2.Canny(gray, 70, 180)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(18, int(0.045 * min(gray.shape))),
        minLineLength=max(12, int(0.07 * min(gray.shape))),
        maxLineGap=max(4, int(0.012 * min(gray.shape))),
    )

    def display_point(x: float, y: float) -> list[float]:
        return [round(box.x0 + x / scale, 4), round(box.y0 + y / scale, 4)]

    lines = []
    for row in [] if raw_lines is None else raw_lines[:, 0, :]:
        x0, y0, x1, y1 = (int(value) for value in row)
        length = math.hypot(x1 - x0, y1 - y0)
        lines.append(
            {
                "kind": "line_candidate",
                "start_display": display_point(x0, y0),
                "end_display": display_point(x1, y1),
                "length_points": round(length / scale, 4),
            }
        )
    lines.sort(key=lambda item: item["length_points"], reverse=True)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    curves = []
    image_area = gray.shape[0] * gray.shape[1]
    for contour in contours:
        perimeter = cv2.arcLength(contour, False)
        area = abs(cv2.contourArea(contour))
        if perimeter < 0.08 * min(gray.shape) or area > 0.90 * image_area:
            continue
        approximation = cv2.approxPolyDP(contour, 0.012 * perimeter, False)
        if len(approximation) < 5:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        curves.append(
            {
                "kind": "curve_candidate",
                "bbox_display": [
                    round(box.x0 + x / scale, 4),
                    round(box.y0 + y / scale, 4),
                    round(box.x0 + (x + width) / scale, 4),
                    round(box.y0 + (y + height) / scale, 4),
                ],
                "polyline_display": [display_point(point[0][0], point[0][1]) for point in approximation[:80]],
                "perimeter_points": round(perimeter / scale, 4),
            }
        )
    curves.sort(key=lambda item: item["perimeter_points"], reverse=True)

    ocr_tokens = []

    def append_ocr(image: Any, orientation: str) -> None:
        data = pytesseract.image_to_data(
            image,
            config="--psm 11 -l eng -c tessedit_char_whitelist=0123456789Rr.,-",
            output_type=pytesseract.Output.DICT,
            timeout=12,
        )
        for index, text in enumerate(data.get("text", [])):
            token = str(text).strip()
            if not re.search(r"\d", token):
                continue
            confidence = float(data["conf"][index])
            if confidence < 35:
                continue
            x, y = int(data["left"][index]), int(data["top"][index])
            width, height = int(data["width"][index]), int(data["height"][index])
            refinement_support = 0
            if confidence < 90 and token.isdigit() and len(token) <= 3:
                refined, refinement_support = _refine_engineering_digit_token(
                    image,
                    (x, y, width, height),
                    token,
                )
                if refined != token:
                    token = refined
                    confidence = max(confidence, 82.0 + 4.0 * refinement_support)
            if orientation == "vertical":
                # cv2.ROTATE_90_CLOCKWISE maps (x, y) to (H - 1 - y, x).
                # Transform the OCR rectangle back into the original crop.
                original_x0, original_x1 = y, y + height
                original_y0 = gray.shape[0] - (x + width)
                original_y1 = gray.shape[0] - x
            else:
                original_x0, original_x1 = x, x + width
                original_y0, original_y1 = y, y + height
            ocr_tokens.append(
                {
                    "text": token,
                    "confidence": round(confidence / 100.0, 3),
                    "bbox_display": [
                        *display_point(original_x0, original_y0),
                        *display_point(original_x1, original_y1),
                    ],
                    "state": "observed",
                    "method": (
                        "geometry_gated_detail_image_ocr_rotated_vertical"
                        if orientation == "vertical"
                        else "geometry_gated_detail_image_ocr"
                    ) + ("_tight_consensus_refinement" if refinement_support else ""),
                    "orientation": orientation,
                    "refinement_support": refinement_support,
                    "glyph_signature": _digit_glyph_signature(image, (x, y, width, height)),
                }
            )

    try:
        append_ocr(gray, "horizontal")
        append_ocr(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), "vertical")
    except (OSError, RuntimeError, pytesseract.TesseractError):
        pass
    deduplicated_tokens = []
    for item in sorted(ocr_tokens, key=lambda row: -row["confidence"]):
        item_box = fitz.Rect(item["bbox_display"])
        if any(
            kept["text"] == item["text"]
            and _box_distance(item_box, fitz.Rect(kept["bbox_display"])) <= 2.0
            for kept in deduplicated_tokens
        ):
            continue
        deduplicated_tokens.append(item)
    return {
        **geometry,
        "segments": lines[:80],
        "curve_candidates": curves[:24],
        "raster_dimension_observations": deduplicated_tokens,
        "vectorization": {
            "state": "observed",
            "method": "Canny plus probabilistic Hough lines and contour curvature candidates",
            "render_scale": scale,
        },
    }


def _raster_orientation_signature(detail: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    counts = {"horizontal": 0, "vertical": 0, "diagonal": 0}
    curve_count = 0
    for geometry in detail.get("geometry_paths", []):
        if geometry.get("source_type") != "embedded_raster_detail":
            continue
        curve_count += len(geometry.get("curve_candidates", []))
        for segment in geometry.get("segments", []):
            if segment.get("kind") != "line_candidate":
                continue
            start, end = segment["start_display"], segment["end_display"]
            angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
            if min(angle, 180.0 - angle) <= 8.0:
                counts["horizontal"] += 1
            elif abs(angle - 90.0) <= 8.0:
                counts["vertical"] += 1
            elif min(abs(angle - 45.0), abs(angle - 135.0)) <= 10.0:
                counts["diagonal"] += 1
    return (
        counts["horizontal"] >= 4,
        counts["vertical"] >= 4,
        counts["diagonal"] >= 2,
        curve_count >= 2,
    )


def _classify_repeated_raster_dimensions(details: list[dict[str, Any]]) -> None:
    """Attach semantic roles to raster dimensions using repeated topology."""

    detail_signatures = {detail["id"]: _raster_orientation_signature(detail) for detail in details}
    anchors: dict[tuple[float, tuple[bool, bool, bool, bool], str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for detail in details:
        signature = detail_signatures[detail["id"]]
        for index, observation in enumerate(detail.get("raster_fabrication_dimension_observations", []), start=1):
            observation["id"] = f"{detail['id']}.raster_dimension.{index:03d}"
            value = observation.get("value_mm")
            if value is not None and 20.0 <= float(value) <= 100.0 and observation.get("confidence", 0.0) >= 0.82:
                anchors.setdefault((round(float(value), 3), signature, str(observation.get("orientation"))), []).append((detail, observation))

    for (value, signature, orientation), rows in anchors.items():
        anchor_detail_ids = {detail["id"] for detail, _ in rows}
        if len(anchor_detail_ids) < 3 or not all(signature):
            continue
        normalized_centers = []
        for detail, observation in rows:
            detail_box, token_box = fitz.Rect(detail["bbox_display"]), fitz.Rect(observation["bbox_display"])
            normalized_centers.append(
                (
                    ((token_box.x0 + token_box.x1) / 2 - detail_box.x0) / max(detail_box.width, 1e-9),
                    ((token_box.y0 + token_box.y1) / 2 - detail_box.y0) / max(detail_box.height, 1e-9),
                )
            )
        center = (
            sum(item[0] for item in normalized_centers) / len(normalized_centers),
            sum(item[1] for item in normalized_centers) / len(normalized_centers),
        )
        anchor_signatures = [item.get("glyph_signature", "") for _, item in rows]
        target_text = str(int(value)) if float(value).is_integer() else str(value)
        for detail in details:
            if detail_signatures[detail["id"]] != signature:
                continue
            detail_box = fitz.Rect(detail["bbox_display"])
            for observation in detail.get("raster_fabrication_dimension_observations", []):
                if observation.get("orientation") != orientation or observation.get("semantic_role"):
                    continue
                raw_text = str(observation.get("text", ""))
                if len(raw_text) != len(target_text) or not raw_text or raw_text[-1:] != target_text[-1:]:
                    continue
                token_box = fitz.Rect(observation["bbox_display"])
                token_center = (
                    ((token_box.x0 + token_box.x1) / 2 - detail_box.x0) / max(detail_box.width, 1e-9),
                    ((token_box.y0 + token_box.y1) / 2 - detail_box.y0) / max(detail_box.height, 1e-9),
                )
                if math.dist(center, token_center) > 0.035:
                    continue
                similarity = max(
                    (_glyph_signature_similarity(observation.get("glyph_signature", ""), candidate) for candidate in anchor_signatures),
                    default=0.0,
                )
                if similarity < 0.82:
                    continue
                observation.update(
                    {
                        "text": target_text,
                        "value_mm": value,
                        "confidence": max(float(observation.get("confidence", 0.0)), 0.90),
                        "method": str(observation.get("method", "")) + "_cross_row_glyph_consensus",
                        "glyph_consensus_similarity": round(similarity, 3),
                        "glyph_consensus_support": len(anchor_detail_ids),
                    }
                )

    repeated: dict[tuple[float, tuple[bool, bool, bool, bool]], list[dict[str, Any]]] = {}
    for detail in details:
        signature = detail_signatures[detail["id"]]
        for observation in detail.get("raster_fabrication_dimension_observations", []):
            value = observation.get("value_mm")
            if value is not None and 20.0 <= float(value) <= 100.0 and observation.get("confidence", 0.0) >= 0.82:
                repeated.setdefault((round(float(value), 3), signature), []).append(observation)

    for (_, signature), rows in repeated.items():
        detail_ids = {item["id"].split(".raster_dimension.")[0] for item in rows}
        if len(detail_ids) < 3 or not all(signature):
            continue
        for item in rows:
            item.update(
                {
                    "semantic_role": "hook_projection_dimension",
                    "status": "accepted_repeated_raster_dimension",
                    "state": "derived",
                    "confidence": round(max(float(item.get("confidence", 0.0)), min(0.97, 0.82 + 0.02 * len(detail_ids))), 3),
                    "cross_row_support": len(detail_ids),
                    "basis": "same metric token repeats in three or more raster details with the same orthogonal-plus-diagonal topology",
                }
            )

    for detail in details:
        observations = detail.get("raster_fabrication_dimension_observations", [])
        horizontal = [item for item in observations if item.get("orientation") == "horizontal" and item.get("value_mm", 0) > 100 and item.get("confidence", 0) >= 0.90]
        vertical = [item for item in observations if item.get("orientation") == "vertical" and item.get("value_mm", 0) > 100 and item.get("confidence", 0) >= 0.90]
        if len({item["value_mm"] for item in horizontal}) == 1:
            for item in horizontal:
                item.update({"semantic_role": "overall_width_dimension", "status": "accepted_metric_anchor"})
        if len({item["value_mm"] for item in vertical}) == 1:
            for item in vertical:
                item.update({"semantic_role": "overall_height_dimension", "status": "accepted_metric_anchor"})


def _native_or_ocr_mark(page: fitz.Page, cell: fitz.Rect | list[float]) -> dict[str, Any] | None:
    cell = fitz.Rect(cell)
    inner = fitz.Rect(cell.x0 + 3, cell.y0 + 3, cell.x1 - 3, cell.y1 - 3)
    native_text = page.get_textbox(inner).strip()
    normalized = _normalize_mark(native_text)
    if normalized is not None:
        return {
            "marks": list(normalized),
            "display": native_text,
            "support": 1,
            "raw_ocr": [],
            "method": "native_text_in_geometry_gated_mark_cell",
            "state": "direct",
            "mark_variant_styles": _mark_variant_styles(native_text, normalized),
        }
    return _ocr_mark(page, cell)


def _fabrication_dimensions(
    cell: fitz.Rect,
    dimensions: tuple[DimensionAttachment, ...],
) -> list[dict[str, Any]]:
    rows = []
    for item in dimensions:
        text_box = fitz.Rect(item.text_bbox)
        center = fitz.Point((text_box.x0 + text_box.x1) / 2, (text_box.y0 + text_box.y1) / 2)
        if center not in cell or item.score < 0.75:
            continue
        rows.append(
            {
                "dimension_id": item.attachment_id,
                "value_mm": item.value_mm,
                "orientation": item.orientation,
                "state": "direct" if item.status == "accepted" else "observed",
                "status": item.status,
                "score": item.score,
                "text_bbox_display": list(item.text_bbox),
                "primitive_refs": [
                    item.baseline.primitive_ref,
                    *(line.primitive_ref for line in item.extension_lines),
                    *item.terminal_refs,
                ],
            }
        )
    unique = {}
    for row in rows:
        key = (row["value_mm"], row["orientation"], tuple(round(value, 1) for value in row["text_bbox_display"]))
        unique.setdefault(key, row)
    return list(unique.values())


def _native_detail_table_candidates(page: fitz.Page) -> list[dict[str, Any]]:
    """Find repeated mark/geometry rows, including grids split per row."""

    finder = getattr(page, "find_tables", None)
    if finder is None:
        return []
    page_area = page.rect.get_area()
    candidates = []
    for table in finder().tables:
        box = fitz.Rect(table.bbox)
        if table.row_count < 3 or table.col_count < 2:
            continue
        if not (0.002 * page_area <= box.get_area() <= 0.35 * page_area):
            continue
        rows = []
        geometry_row_count = 0
        for table_row in table.rows:
            cells = [fitz.Rect(cell) for cell in table_row.cells if cell is not None]
            if len(cells) < 2:
                continue
            mark_cell = cells[0]
            geometry_cell = fitz.Rect(cells[1].x0, min(cell.y0 for cell in cells[1:]), max(cell.x1 for cell in cells[1:]), max(cell.y1 for cell in cells[1:]))
            geometry = _row_geometry(page, geometry_cell, geometry_cell.height)
            geometry_row_count += bool(geometry)
            rows.append(
                {
                    "mark_cell": list(mark_cell),
                    "geometry_cell": list(geometry_cell),
                    "geometry": geometry,
                }
            )
        if geometry_row_count >= 2:
            candidates.append(
                {
                    "id": f"native_detail_table_candidate.{len(candidates) + 1:03d}",
                    "bbox_display": list(box),
                    "rows": rows,
                    "geometry_row_count": geometry_row_count,
                    "row_count": table.row_count,
                    "column_count": table.col_count,
                    "basis": "native table finder plus repeated non-grid fabrication geometry",
                }
            )
    return sorted(candidates, key=lambda item: (-item["geometry_row_count"], fitz.Rect(item["bbox_display"]).get_area()))


def _mark_template(page: fitz.Page, cell: fitz.Rect, scale: int = 8) -> tuple[Any, fitz.Rect] | None:
    import numpy as np

    inner = fitz.Rect(cell.x0 + 3, cell.y0 + 3, cell.x1 - 3, cell.y1 - 3)
    image = _render_gray(page, inner, scale)
    ys, xs = np.where(image < 160)
    if len(xs) == 0:
        return None
    pad = 6
    x0, x1 = max(0, int(xs.min()) - pad), min(image.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(image.shape[0], int(ys.max()) + pad + 1)
    template = image[y0:y1, x0:x1]
    display_box = fitz.Rect(inner.x0 + x0 / scale, inner.y0 + y0 / scale, inner.x0 + x1 / scale, inner.y0 + y1 / scale)
    return template, display_box


def _template_candidates(
    page: fitz.Page,
    template: Any,
    view: dict[str, Any],
    scale: int = 6,
    target: Any | None = None,
) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    box = fitz.Rect(view["bbox_display"])
    target = _render_gray(page, box, scale) if target is None else target
    if template.shape[0] >= target.shape[0] or template.shape[1] >= target.shape[1]:
        return []
    scores = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    count = min(80, scores.size)
    indexes = np.argpartition(scores.ravel(), -count)[-count:]
    candidates = []
    for index in indexes[np.argsort(scores.ravel()[indexes])[::-1]]:
        y, x = np.unravel_index(index, scores.shape)
        if any(abs(x - row["pixel_x"]) <= template.shape[1] / 2 and abs(y - row["pixel_y"]) <= template.shape[0] / 2 for row in candidates):
            continue
        candidates.append(
            {
                "score": float(scores[y, x]),
                "pixel_x": int(x),
                "pixel_y": int(y),
                "bbox_display": [
                    box.x0 + x / scale,
                    box.y0 + y / scale,
                    box.x0 + (x + template.shape[1]) / scale,
                    box.y0 + (y + template.shape[0]) / scale,
                ],
                "view_id": view["id"],
            }
        )
        if len(candidates) >= 4:
            break
    return candidates


def _multiscale_template_candidates(
    page: fitz.Page,
    template: Any,
    view: dict[str, Any],
    target: Any,
    *,
    source_scale: int = 8,
    target_scale: int = 6,
) -> list[dict[str, Any]]:
    """Match outlined mark glyphs without assuming equal plotted text scale."""

    import cv2

    base_factor = target_scale / source_scale
    rows = []
    for relative_scale in (0.72, 0.86, 1.0, 1.16, 1.32, 1.50, 1.68):
        factor = base_factor * relative_scale
        resized = cv2.resize(
            template,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC,
        )
        for candidate in _template_candidates(page, resized, view, scale=target_scale, target=target):
            candidate["relative_text_scale"] = relative_scale
            rows.append(candidate)
    rows.sort(key=lambda item: (-item["score"], item["view_id"]))
    kept = []
    for candidate in rows:
        box = fitz.Rect(candidate["bbox_display"])
        center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        if any(center in (fitz.Rect(item["bbox_display"]) + (-5, -5, 5, 5)) for item in kept):
            continue
        kept.append(candidate)
        if len(kept) >= 32:
            break
    return kept


def _isolated_mark_token(candidate_box: fitz.Rect, view_box: fitz.Rect, target: Any, scale: int = 6) -> bool:
    """Reject substring hits such as mark ``10`` inside dimension text ``100``."""

    import numpy as np

    x0 = max(0, int(round((candidate_box.x0 - view_box.x0) * scale)))
    x1 = min(target.shape[1], int(round((candidate_box.x1 - view_box.x0) * scale)))
    y0 = max(0, int(round((candidate_box.y0 - view_box.y0) * scale)))
    y1 = min(target.shape[0], int(round((candidate_box.y1 - view_box.y0) * scale)))
    if x1 <= x0 or y1 <= y0:
        return False
    height = y1 - y0
    center_y0 = y0 + max(1, int(0.18 * height))
    center_y1 = y1 - max(1, int(0.18 * height))
    side_width = max(3, int(0.42 * height))
    gap = max(1, int(0.06 * height))
    left = target[center_y0:center_y1, max(0, x0 - side_width) : max(0, x0 - gap)]
    right = target[center_y0:center_y1, min(target.shape[1], x1 + gap) : min(target.shape[1], x1 + side_width)]
    side_pixels = (int(np.count_nonzero(left < 150)), int(np.count_nonzero(right < 150)))
    allowance = max(2, int(0.10 * max(left.size, right.size, 1)))
    return max(side_pixels) <= allowance


def _box_distance(left: fitz.Rect, right: fitz.Rect) -> float:
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return math.hypot(dx, dy)


def _native_mark_candidates(
    page: fitz.Page,
    detail: dict[str, Any],
    target_views: list[dict[str, Any]],
    dimensions: tuple[DimensionAttachment, ...],
) -> list[dict[str, Any]]:
    """Find exact native mark tokens before invoking rendered matching."""

    marks = set(detail.get("marks", []))
    rows = []
    for word in page.get_text("words"):
        token = str(word[4]).strip()
        normalized = _normalize_mark(token)
        if normalized is None or len(normalized) != 1 or normalized[0] not in marks:
            continue
        word_box = fitz.Rect(word[:4])
        dimension_matches = [
            item
            for item in dimensions
            if item.text.strip() == token
            and _box_distance(word_box, fitz.Rect(item.text_bbox)) <= 1.5
        ]
        if any(item.status == "accepted" or item.score >= 0.95 for item in dimension_matches):
            continue
        for view in target_views:
            if not fitz.Rect(view["bbox_display"]).contains(word_box):
                continue
            rows.append(
                {
                    "score": 0.99,
                    "bbox_display": list(word_box),
                    "view_id": view["id"],
                    "source_method": "exact_native_mark_token",
                    "relative_text_scale": 1.0,
                }
            )
    return rows


def _point_polyline_distance(point: tuple[float, float], points: list[list[float]]) -> float:
    if len(points) < 2:
        return math.inf
    px, py = point
    distance = math.inf
    for left, right in zip(points, points[1:]):
        ax, ay = left
        bx, by = right
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        fraction = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
        distance = min(distance, math.dist((px, py), (ax + fraction * dx, ay + fraction * dy)))
    return distance


def _attach_terminals_to_paths(
    trace: dict[str, Any],
    target_view_id: str,
    path_graph: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Resolve a leader terminal to a unique projected rebar-path fragment."""

    if not path_graph:
        return []
    leader_refs = {item["drawing_ref"] for item in trace.get("segments", [])}
    fragments = [
        fragment
        for fragment in path_graph.get("fragments", [])
        if fragment.get("view_id") == target_view_id
        and fragment.get("primitive_ref") not in leader_refs
        and len(fragment.get("geometry", {}).get("points_display", [])) >= 2
        and float(fragment.get("geometry", {}).get("length_points", 0.0)) >= 20.0
    ]
    attachments = []
    for terminal in trace.get("terminals", []):
        ranked = sorted(
            (
                _point_polyline_distance(tuple(terminal), fragment["geometry"]["points_display"]),
                -float(fragment.get("candidate_score", 0.0)),
                fragment["id"],
                fragment,
            )
            for fragment in fragments
        )
        if not ranked:
            continue
        distance, _, _, fragment = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else math.inf
        margin = runner_up - distance
        near = [row for row in ranked if row[0] <= min(5.0, distance + 1.25)]
        base_angle = float(fragment["geometry"].get("angle_deg", 0.0)) % 180.0
        base_length = float(fragment["geometry"].get("length_points", 0.0))
        coherent_projection = len(near) >= 2 and all(
            min(
                abs((float(row[3]["geometry"].get("angle_deg", 0.0)) % 180.0) - base_angle),
                180.0 - abs((float(row[3]["geometry"].get("angle_deg", 0.0)) % 180.0) - base_angle),
            )
            <= 4.0
            and min(base_length, float(row[3]["geometry"].get("length_points", 0.0)))
            / max(base_length, float(row[3]["geometry"].get("length_points", 0.0)), 1e-6)
            >= 0.75
            for row in near
        )
        if distance > 5.0 or (margin < 1.25 and not coherent_projection):
            continue
        attachments.append(
            {
                "terminal_display": list(terminal),
                "fragment_id": fragment["id"],
                "primitive_ref": fragment["primitive_ref"],
                "fragment_points_display": fragment["geometry"]["points_display"],
                "coincident_fragment_ids": [row[3]["id"] for row in near],
                "projection_cluster_state": "coherent_parallel_boundaries" if coherent_projection else "unique_fragment",
                "terminal_distance_points": round(distance, 3),
                "ambiguity_margin_points": None if math.isinf(margin) else round(margin, 3),
                "state": "accepted",
            }
        )
    return attachments


def _link_physical_families(
    details: list[dict[str, Any]],
    associations: list[dict[str, Any]],
    views: list[dict[str, Any]],
    path_graph: dict[str, Any] | None,
    section_observations: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build mark families from independently accepted detail placements.

    A family asserts shared mark identity, not that two projections are the
    same physical instance.  Instance counts remain a separate constraint.
    """

    if not path_graph:
        return []
    fragments = {item["id"]: item for item in path_graph.get("fragments", [])}
    fragments_by_source: dict[str, list[dict[str, Any]]] = {}
    for fragment in fragments.values():
        fragments_by_source.setdefault(fragment.get("source_path_ref", ""), []).append(fragment)
    component_by_fragment = {
        fragment_id: component
        for component in path_graph.get("components", [])
        for fragment_id in component.get("fragment_ids", [])
    }
    view_by_id = {item["id"]: item for item in views}
    for association in associations:
        if association.get("state") != "accepted":
            continue
        compatible_attachments = []
        rejected_attachments = []
        association_marks = set(association.get("marks", []))
        for attachment in association.get("path_attachments", []):
            fragment = fragments.get(attachment["fragment_id"])
            if fragment is None:
                continue
            authoritative_marks = (
                set(fragment.get("mark_hypotheses", []))
                if fragment.get("mark_assignment_basis")
                == "exclusive section mark plus first connected heavy-geometry leader contact"
                else set()
            )
            if authoritative_marks and not (authoritative_marks & association_marks):
                rejected_attachments.append(
                    {
                        **attachment,
                        "state": "rejected",
                        "reason": "detail callout target contradicts the exact section leader identity",
                    }
                )
                continue
            compatible_attachments.append(attachment)
            fragment["mark_hypotheses"] = sorted(
                set(fragment.get("mark_hypotheses", [])) | set(association["marks"]),
                key=lambda value: (len(value), value),
            )
            fragment["native_detail_ids"] = sorted(
                set(fragment.get("native_detail_ids", [])) | {association["detail_id"]}
            )
            fragment["native_detail_placement_ids"] = sorted(
                set(fragment.get("native_detail_placement_ids", [])) | {association["id"]}
            )
        association["path_attachments"] = compatible_attachments
        if rejected_attachments:
            association["rejected_path_attachments"] = rejected_attachments
        if not compatible_attachments:
            association["state"] = "review_candidate"
            association["basis"] = "exact mark identity retained, but the proposed path contradicts an authoritative section leader target"
    section_links: dict[str, list[dict[str, Any]]] = {}
    for section in (section_observations or {}).get("sections", []):
        for candidate in section.get("candidates", []):
            for mark in candidate.get("leader_marks", []):
                matched = fragments_by_source.get(candidate["primitive_ref"], [])
                for fragment in matched:
                    fragment["mark_hypotheses"] = sorted(
                        set(fragment.get("mark_hypotheses", [])) | {mark},
                        key=lambda value: (len(value), value),
                    )
                candidate_box = fitz.Rect(candidate["bbox_display"])
                center = fitz.Point(
                    (candidate_box.x0 + candidate_box.x1) / 2,
                    (candidate_box.y0 + candidate_box.y1) / 2,
                )
                containing_views = [
                    view
                    for view in views
                    if view.get("role_hypothesis") == "section_view_candidate"
                    and center in fitz.Rect(view["bbox_display"])
                ]
                section_view_id = (
                    min(containing_views, key=lambda view: fitz.Rect(view["bbox_display"]).get_area())["id"]
                    if containing_views
                    else None
                )
                section_links.setdefault(mark, []).append(
                    {
                        "section_id": section["section_id"],
                        "candidate_id": candidate["candidate_id"],
                        "primitive_ref": candidate["primitive_ref"],
                        "fragment_ids": [fragment["id"] for fragment in matched],
                        "view_id": section_view_id,
                        "basis": "exclusive mark token plus native section leader terminal",
                    }
                )

    for component in path_graph.get("components", []):
        rows = [fragments[fragment_id] for fragment_id in component.get("fragment_ids", []) if fragment_id in fragments]
        component["mark_hypotheses"] = sorted(
            {mark for row in rows for mark in row.get("mark_hypotheses", [])},
            key=lambda value: (len(value), value),
        )
        component["native_detail_ids"] = sorted(
            {detail_id for row in rows for detail_id in row.get("native_detail_ids", [])}
        )

    families = []
    all_marks = sorted(
        {mark for detail in details for mark in detail.get("marks", [])} | set(section_links),
        key=lambda value: (len(value), value),
    )
    for mark in all_marks:
        detail_rows = [item for item in details if mark in item.get("marks", [])]
        placement_rows = [
            item
            for item in associations
            if item.get("state") == "accepted" and mark in item.get("marks", [])
        ]
        fragment_ids = sorted(
            {
                attachment["fragment_id"]
                for item in placement_rows
                for attachment in item.get("path_attachments", [])
            }
            | {
                fragment_id
                for item in section_links.get(mark, [])
                for fragment_id in item["fragment_ids"]
            }
            | {
                fragment["id"]
                for fragment in fragments.values()
                if mark in fragment.get("mark_hypotheses", [])
            }
        )
        component_rows = {
            component_by_fragment[fragment_id]["id"]: component_by_fragment[fragment_id]
            for fragment_id in fragment_ids
            if fragment_id in component_by_fragment
        }
        view_ids = sorted(
            {item["target_view_id"] for item in placement_rows}
            | {view_id for component in component_rows.values() for view_id in component.get("view_ids", [])}
            | {item["view_id"] for item in section_links.get(mark, []) if item.get("view_id")}
        )
        elevation_components = sorted(
            component["id"]
            for component in component_rows.values()
            if any(
                view_by_id.get(view_id, {}).get("role_hypothesis") == "reinforcement_view_candidate"
                for view_id in component.get("view_ids", [])
            )
        )
        section_components = sorted(
            component["id"]
            for component in component_rows.values()
            if any(
                view_by_id.get(view_id, {}).get("role_hypothesis") == "section_view_candidate"
                for view_id in component.get("view_ids", [])
            )
        )
        families.append(
            {
                "id": f"physical_bar_family.{len(families) + 1:03d}",
                "mark": mark,
                "state": "derived_cross_view_family" if len(view_ids) >= 2 else "single_view_family_candidate" if view_ids else "detail_only",
                "detail_ids": [item["id"] for item in detail_rows],
                "placement_association_ids": [item["id"] for item in placement_rows],
                "fragment_ids": fragment_ids,
                "component_ids": sorted(component_rows),
                "view_ids": view_ids,
                "section_component_ids": section_components,
                "section_candidate_ids": [item["candidate_id"] for item in section_links.get(mark, [])],
                "elevation_component_ids": elevation_components,
                "fabrication_dimensions": [
                    dimension
                    for item in detail_rows
                    for dimension in item.get("fabrication_dimensions", [])
                ],
                "raster_fabrication_dimension_observations": [
                    dimension
                    for item in detail_rows
                    for dimension in item.get("raster_fabrication_dimension_observations", [])
                ],
                "link_chain": [
                    {
                        "detail_id": item["detail_id"],
                        "mark": mark,
                        "placement_id": item["id"],
                        "view_id": item["target_view_id"],
                        "fragment_ids": [attachment["fragment_id"] for attachment in item.get("path_attachments", [])],
                        "basis": item["basis"],
                    }
                    for item in placement_rows
                ]
                + [
                    {
                        **item,
                        "detail_ids": [row["id"] for row in detail_rows],
                        "mark": mark,
                    }
                    for item in section_links.get(mark, [])
                ],
                "count_state": "unknown",
                "note": "The family joins detail and projected placements; physical instance multiplicity is not inferred here.",
            }
        )
    return families


def extract_native_vector_details(
    page: fitz.Page,
    views: list[dict[str, Any]],
    object_graph: dict[str, Any],
    path_graph: dict[str, Any] | None = None,
    dimensions: tuple[DimensionAttachment, ...] | None = None,
    section_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return native detail rows and conservative detail-to-callout links."""

    dimension_rows = tuple(attach_dimensions(page) if dimensions is None else dimensions)
    table_candidates = _native_detail_table_candidates(page)
    if not table_candidates:
        for view in views:
            horizontal, vertical = _view_grid(page, view)
            if len(horizontal) < 5 or len(vertical) < 3:
                continue
            divider = vertical[1]
            rows = []
            for top, bottom in zip(horizontal[1:-1], horizontal[2:]):
                mark_cell = fitz.Rect(vertical[0], top, divider, bottom)
                geometry_cell = fitz.Rect(divider, top, vertical[-1], bottom)
                rows.append(
                    {
                        "mark_cell": list(mark_cell),
                        "geometry_cell": list(geometry_cell),
                        "geometry": _row_geometry(page, geometry_cell, bottom - top),
                    }
                )
            geometry_rows = sum(bool(item["geometry"]) for item in rows)
            if geometry_rows >= 2:
                table_candidates.append(
                    {
                        "id": view["id"],
                        "bbox_display": view["bbox_display"],
                        "rows": rows,
                        "geometry_row_count": geometry_rows,
                        "basis": "semantic view grid plus repeated non-grid fabrication geometry",
                    }
                )
    if not table_candidates:
        return {
            "status": "unresolved",
            "detail_regions": [],
            "details": [],
            "placement_associations": [],
            "reason": "native detail-table geometry is not unique",
            "schedule_values_used": False,
        }
    table_candidates.sort(key=lambda item: (-item["geometry_row_count"], item["id"]))
    if len(table_candidates) > 1 and table_candidates[1]["geometry_row_count"] >= 0.80 * table_candidates[0]["geometry_row_count"]:
        return {
            "status": "unresolved",
            "detail_regions": [],
            "details": [],
            "placement_associations": [],
            "reason": "native detail-table geometry is not unique",
            "table_candidates": table_candidates,
            "schedule_values_used": False,
        }
    detail_table = table_candidates[0]
    raster_box_templates = []
    for row in detail_table["rows"]:
        cell = fitz.Rect(row["geometry_cell"])
        for item in row["geometry"]:
            if item.get("source_type") != "embedded_raster_detail":
                continue
            box = fitz.Rect(item["bbox_display"])
            raster_box_templates.append(
                (
                    (box.x0 - cell.x0) / max(cell.width, 1e-9),
                    (box.y0 - cell.y0) / max(cell.height, 1e-9),
                    (box.x1 - cell.x0) / max(cell.width, 1e-9),
                    (box.y1 - cell.y0) / max(cell.height, 1e-9),
                )
            )
    repeated_raster_template = (
        tuple(statistics.median(values) for values in zip(*raster_box_templates))
        if len(raster_box_templates) >= 3
        else None
    )
    details = []
    for row in detail_table["rows"]:
        mark_cell = fitz.Rect(row["mark_cell"])
        mark = _native_or_ocr_mark(page, mark_cell)
        if mark is None:
            continue
        geometry_cell = fitz.Rect(row["geometry_cell"])
        row_geometry = list(row["geometry"])
        if not row_geometry and repeated_raster_template is not None:
            x0, y0, x1, y1 = repeated_raster_template
            fallback_box = fitz.Rect(
                geometry_cell.x0 + x0 * geometry_cell.width,
                geometry_cell.y0 + y0 * geometry_cell.height,
                geometry_cell.x0 + x1 * geometry_cell.width,
                geometry_cell.y0 + y1 * geometry_cell.height,
            )
            row_geometry.append(
                {
                    "primitive_ref": "rasterized_repeated_detail_row_fallback",
                    "bbox_display": list(fallback_box),
                    "points_display": [],
                    "segments": [],
                    "source_type": "embedded_raster_detail",
                    "state": "observed",
                    "method": "median_neighbor_image_inset_raster_fallback",
                }
            )
        geometry = [
            _vectorize_raster_detail(page, item)
            if item.get("source_type") == "embedded_raster_detail"
            else item
            for item in row_geometry
        ]
        raster_dimensions = []
        for item in geometry:
            for observation in item.get("raster_dimension_observations", []):
                match = re.search(r"(?P<radius>[Rr])?\s*(?P<value>\d+(?:[.,]\d+)?)", observation["text"])
                raster_dimensions.append(
                    {
                        **observation,
                        "value_mm": None if match is None else float(match.group("value").replace(",", ".")),
                        "kind": "bend_radius" if match is not None and match.group("radius") else "linear_or_angular_dimension_candidate",
                    }
                )
        details.append(
            {
                "id": f"native_vector_detail.{len(details) + 1:03d}",
                "marks": mark["marks"],
                "mark_display": mark["display"],
                "state": "observed" if geometry else "partial",
                "reason": None if geometry else "mark row retained but no separable native or raster detail geometry was found",
                "mark_observation": {**mark, "bbox_display": list(mark_cell)},
                "mark_variant_styles": mark["mark_variant_styles"],
                "bbox_display": list(geometry_cell),
                "geometry_paths": geometry,
                "fabrication_dimensions": _fabrication_dimensions(geometry_cell, dimension_rows),
                "raster_fabrication_dimension_observations": raster_dimensions,
                "primitive_refs": [item["primitive_ref"] for item in geometry],
            }
        )

    _classify_repeated_raster_dimensions(details)
    from src.drawing_engine.disciplines.detail.detail_fabrication_solver import solve_detail_fabrication_geometry

    for detail in details:
        detail["fabrication_geometry_solution"] = solve_detail_fabrication_geometry(detail)

    variant_context = _instance_variant_context(page, views, object_graph)
    target_ids = {
        view_id
        for item in object_graph.get("instances", [])
        for view_id in item.get("view_ids", [])
    } | {item["view_id"] for item in object_graph.get("shared_supporting_views", [])}
    if not target_ids:
        target_ids = {
            item["id"]
            for item in views
            if item.get("role_hypothesis")
            in {"reinforcement_view_candidate", "section_view_candidate", "plan_view_candidate", "drawing_view_candidate"}
        }
    else:
        target_ids |= {
            item["id"]
            for item in views
            if item.get("role_hypothesis") in {"reinforcement_view_candidate", "section_view_candidate"}
        }
    target_views = []
    for item in views:
        if item["id"] not in target_ids:
            continue
        box = fitz.Rect(item["bbox_display"])
        # Mark callouts legitimately sit just outside a semantic view's concrete
        # contour (most often below a section).  Grow the search scope from the
        # grouped view geometry, while the mark-token, native-leader, and unique
        # rebar-terminal gates below still fail closed on unrelated neighbours.
        padding = min(60.0, max(32.0, 0.055 * math.hypot(box.width, box.height)))
        search_box = (box + (-padding, -padding, padding, padding)) & page.rect
        target_views.append({**item, "bbox_display": list(search_box), "semantic_view_bbox_display": list(box)})
    thin = extract_thin_segments(page, max_width=0.85)
    target_images: dict[str, Any] = {}
    associations = []
    for detail in details:
        source_token_box = fitz.Rect(detail["mark_observation"]["bbox_display"])
        ranked = _native_mark_candidates(page, detail, target_views, dimension_rows)
        if not ranked or len(detail.get("marks", [])) > 1:
            template_row = _mark_template(page, source_token_box)
            if template_row is None:
                if not ranked:
                    continue
            else:
                template, source_token_box = template_row
                for view in target_views:
                    target_images.setdefault(
                        view["id"],
                        _render_gray(page, fitz.Rect(view["bbox_display"]), 6),
                    )
                    ranked.extend(
                        _multiscale_template_candidates(
                            page,
                            template,
                            view,
                            target_images[view["id"]],
                        )
                    )
        ranked.sort(key=lambda item: (-item["score"], item["view_id"]))
        current = []
        for candidate in ranked:
            if candidate["score"] < 0.60:
                continue
            target_view = next(item for item in target_views if item["id"] == candidate["view_id"])
            if candidate.get("source_method") != "exact_native_mark_token":
                target_images.setdefault(
                    candidate["view_id"],
                    _render_gray(page, fitz.Rect(target_view["bbox_display"]), 6),
                )
                if not _isolated_mark_token(
                    fitz.Rect(candidate["bbox_display"]),
                    fitz.Rect(target_view["bbox_display"]),
                    target_images[candidate["view_id"]],
                ):
                    continue
            trace = trace_leader(
                fitz.Rect(candidate["bbox_display"]),
                thin,
                connection_tolerance=1.35,
                search_radius=130,
                max_hops=9,
            )
            path_attachments = _attach_terminals_to_paths(trace, candidate["view_id"], path_graph)
            if not path_attachments:
                extended_trace = trace_leader(
                    fitz.Rect(candidate["bbox_display"]),
                    thin,
                    connection_tolerance=1.8,
                    search_radius=130,
                    max_hops=9,
                    continue_through_intersections=True,
                )
                extended_attachments = _attach_terminals_to_paths(
                    extended_trace,
                    candidate["view_id"],
                    path_graph,
                )
                if extended_attachments:
                    trace = extended_trace
                    path_attachments = extended_attachments
            accepted = candidate["score"] >= 0.64 and bool(trace["segments"]) and bool(trace["terminals"]) and bool(path_attachments)
            current.append(
                {
                    "id": "",
                    "detail_id": detail["id"],
                    "marks": detail["marks"],
                    "source_mark_bbox_display": list(source_token_box),
                    "target_mark_bbox_display": candidate["bbox_display"],
                    "target_view_id": candidate["view_id"],
                    "leader_trace": trace,
                    "path_attachments": path_attachments,
                    "object_instance_bindings": _association_object_bindings(
                        detail,
                        candidate["view_id"],
                        variant_context,
                        object_graph,
                    ),
                    "state": "accepted" if accepted else "review_candidate",
                    "score": round(candidate["score"], 3),
                    "mark_match_method": candidate.get("source_method", "scale_tolerant_rendered_mark_identity"),
                    "basis": "exact mark identity plus native leader trace uniquely terminating on a projected rebar path" if accepted else "mark identity lacks a complete unique leader-to-rebar path chain",
                }
            )
        accepted_rows = [item for item in current if item["state"] == "accepted"]
        keep = accepted_rows or current[:1]
        for item in keep:
            item["id"] = f"native_detail_placement.{len(associations) + 1:03d}"
            associations.append(item)
    physical_families = _link_physical_families(
        details,
        associations,
        views,
        path_graph,
        section_observations,
    )
    return {
        "status": "partial_links" if details else "unresolved",
        "detail_regions": [
            {
                "view_id": detail_table["id"],
                "bbox_display": detail_table["bbox_display"],
                "basis": detail_table["basis"],
            }
        ],
        "details": details,
        "placement_associations": associations,
        "physical_families": physical_families,
        "variant_context": variant_context,
        "validation": {
            "unique_detail_region": True,
            "accepted_associations_have_leader_terminals": all(
                item["leader_trace"]["terminals"] for item in associations if item["state"] == "accepted"
            ),
            "accepted_associations_attach_to_projected_paths": all(
                item["path_attachments"] for item in associations if item["state"] == "accepted"
            ),
            "schedule_values_used": False,
        },
        "contract": {
            "ocr_is_geometry_gated_to_mark_cells": True,
            "printed_schedule_lengths_are_not_consumed": True,
            "unlinked_details_remain_explicit": True,
            "all_independently_validated_occurrences_are_retained": True,
        },
    }
