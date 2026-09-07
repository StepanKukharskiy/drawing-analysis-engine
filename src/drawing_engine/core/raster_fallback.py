"""Quality-gated raster observations in canonical PDF display coordinates.

Native PDF vectors and text remain primary.  This module runs only when a page
is image-dominant or native text is absent, and emits observations rather than
engineering facts.  Every pixel result records its transform, confidence, and
algorithm provenance so it can enter the immutable observation graph safely.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import lru_cache
import math
from time import perf_counter
from typing import Any, Mapping

import cv2
import fitz
import numpy as np
from PIL import Image
import pytesseract

from src.drawing_engine.core.dimension_attachment import ocr_isolated_dimension_label
from src.drawing_engine.core.dimension_ownership import resolve_raster_dimension_ownership


SCHEMA_VERSION = "0.6.0"
RENDER_SCALE = 2.0
DIMENSION_OCR_SCALE = 8.0


@lru_cache(maxsize=1)
def _tesseract_version() -> str:
    return str(pytesseract.get_tesseract_version()).splitlines()[0]


def assess_native_page_quality(page: fitz.Page) -> dict[str, Any]:
    drawings = [item for item in page.get_drawings() if item.get("type") != "clip"]
    native_path_count = sum(len(item.get("items", [])) for item in drawings)
    native_text = page.get_text("text").strip()
    image_rects: list[fitz.Rect] = []
    for image in page.get_images(full=True):
        try:
            image_rects.extend(page.get_image_rects(image[0]))
        except (RuntimeError, ValueError):
            continue
    page_area = max(float(page.rect.get_area()), 1.0)
    image_area_ratio = min(1.0, sum(float((rect & page.rect).get_area()) for rect in image_rects) / page_area)
    if image_area_ratio >= 0.35 and native_path_count < 12:
        route = "raster_reconstruct"
        reason = "page is image-dominant and native vector geometry is insufficient"
    elif not native_text and native_path_count >= 8:
        route = "hybrid_text_ocr"
        reason = "native vector geometry is usable but native text is absent or outlined"
    else:
        route = "native"
        reason = "native PDF text/vector content is sufficient"
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "perception_quality_route",
        "route": route,
        "reason": reason,
        "metrics": {
            "native_text_chars": len(native_text),
            "native_path_item_count": native_path_count,
            "embedded_image_count": len(image_rects),
            "embedded_image_area_ratio": round(image_area_ratio, 6),
        },
        "thresholds": {
            "minimum_native_text_chars": 1,
            "minimum_native_path_items": 8,
            "image_dominant_area_ratio": 0.35,
        },
        "contract": {
            "native_primary": True,
            "raster_output_is_observation_only": True,
            "schedule_values_used": False,
        },
    }


def _render_rgb(page: fitz.Page, scale: float = RENDER_SCALE) -> np.ndarray:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]


def _ocr_observations(rgb: np.ndarray, page: fitz.Page, scale: float) -> list[dict[str, Any]]:
    observations = []
    source_height, source_width = rgb.shape[:2]
    orientations = (
        ("horizontal", rgb),
        ("vertical_clockwise", cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)),
        ("vertical_counterclockwise", cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    )
    for orientation, oriented in orientations:
        data = pytesseract.image_to_data(
            Image.fromarray(oriented),
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
        for index, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                continue
            if not text or confidence < 45.0:
                continue
            left, top = float(data["left"][index]), float(data["top"][index])
            width, height = float(data["width"][index]), float(data["height"][index])
            if orientation == "vertical_clockwise":
                x0, x1 = top, top + height
                y0, y1 = source_height - (left + width), source_height - left
            elif orientation == "vertical_counterclockwise":
                x0, x1 = source_width - (top + height), source_width - top
                y0, y1 = left, left + width
            else:
                x0, y0, x1, y1 = left, top, left + width, top + height
            box = [x0 / scale, y0 / scale, x1 / scale, y1 / scale]
            observations.append(
                {
                    "kind": "text",
                    "bbox_display": [round(value, 4) for value in box],
                    "text": text,
                    "confidence": round(confidence / 100.0, 4),
                    "geometry_display": None,
                    "source_crop_display": [0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                    "method": "tesseract_geometry_routed_ocr",
                    "orientation": orientation,
                }
            )
    deduplicated = []
    for item in sorted(observations, key=lambda row: -row["confidence"]):
        box = fitz.Rect(item["bbox_display"])
        if any(
            item["text"].casefold() == kept["text"].casefold()
            and box.intersects(fitz.Rect(kept["bbox_display"]))
            and float((box & fitz.Rect(kept["bbox_display"])).get_area())
            >= 0.35 * min(float(box.get_area()), float(fitz.Rect(kept["bbox_display"]).get_area()))
            for kept in deduplicated
        ):
            continue
        deduplicated.append(item)
    deduplicated.sort(key=lambda item: (item["bbox_display"][1], item["bbox_display"][0], item["text"]))
    for index, item in enumerate(deduplicated, start=1):
        item["id"] = f"raster.text.{index:05d}"
    return deduplicated


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Return a dependency-free morphological centreline of black strokes."""

    remaining = binary.copy()
    skeleton = np.zeros_like(remaining)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(remaining):
        eroded = cv2.erode(remaining, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = eroded
    return skeleton


def _merged_hough_segments(
    skeleton: np.ndarray,
    *,
    scale: float,
    display_width: float,
    display_height: float,
) -> list[dict[str, Any]]:
    """Detect centreline segments and merge only close collinear fragments."""

    minimum = max(14, round(0.025 * min(skeleton.shape)))
    raw = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(18, minimum // 2),
        minLineLength=minimum,
        maxLineGap=max(4, round(2.5 * scale)),
    )
    if raw is None:
        return []

    angle_step = math.radians(1.5)
    rho_step = max(1.5, 0.9 * scale)
    groups: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for values in raw[:, 0, :]:
        x0, y0, x1, y1 = (float(value) for value in values)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < minimum:
            continue
        theta = math.atan2(dy, dx) % math.pi
        ux, uy = math.cos(theta), math.sin(theta)
        nx, ny = -uy, ux
        rho = 0.5 * ((x0 + x1) * nx + (y0 + y1) * ny)
        start, end = sorted((x0 * ux + y0 * uy, x1 * ux + y1 * uy))
        key = (round(theta / angle_step), round(rho / rho_step))
        groups[key].append(
            {
                "theta": theta,
                "rho": rho,
                "start": start,
                "end": end,
                "length": length,
            }
        )

    merged: list[dict[str, Any]] = []
    maximum_gap = max(5.0, 3.0 * scale)
    for rows in groups.values():
        rows.sort(key=lambda item: (item["start"], item["end"]))
        runs: list[list[dict[str, float]]] = []
        for row in rows:
            if not runs or row["start"] > max(item["end"] for item in runs[-1]) + maximum_gap:
                runs.append([row])
            else:
                runs[-1].append(row)
        for run in runs:
            support = len(run)
            theta = sum(item["theta"] * item["length"] for item in run) / sum(item["length"] for item in run)
            rho = sum(item["rho"] * item["length"] for item in run) / sum(item["length"] for item in run)
            start = min(item["start"] for item in run)
            end = max(item["end"] for item in run)
            ux, uy = math.cos(theta), math.sin(theta)
            nx, ny = -uy, ux
            points = [(start * ux + rho * nx, start * uy + rho * ny), (end * ux + rho * nx, end * uy + rho * ny)]
            points = [
                (
                    max(0.0, min(display_width, x / scale)),
                    max(0.0, min(display_height, y / scale)),
                )
                for x, y in points
            ]
            if math.dist(points[0], points[1]) >= 5.0:
                merged.append({"points": points, "support": support})
    merged.sort(
        key=lambda item: (
            round(min(item["points"][0][1], item["points"][1][1]), 4),
            round(min(item["points"][0][0], item["points"][1][0]), 4),
            round(max(item["points"][0][1], item["points"][1][1]), 4),
            round(max(item["points"][0][0], item["points"][1][0]), 4),
        )
    )
    return merged


def _short_oblique_segments(
    skeleton: np.ndarray,
    *,
    scale: float,
    display_width: float,
    display_height: float,
) -> list[dict[str, Any]]:
    """Recover short diagonal dimension terminals omitted by the main pass."""

    raw = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(6, round(3.0 * scale)),
        minLineLength=max(5, round(2.0 * scale)),
        maxLineGap=max(2, round(1.0 * scale)),
    )
    if raw is None:
        return []
    candidates = []
    for values in raw[:, 0, :]:
        x0, y0, x1, y1 = (float(value) / scale for value in values)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        length = math.hypot(dx, dy)
        if not 2.0 <= length <= 12.0 or min(dx, dy) < 0.8:
            continue
        angle = math.atan2(y1 - y0, x1 - x0) % math.pi
        candidates.append(
            {
                "points": [
                    (max(0.0, min(display_width, x0)), max(0.0, min(display_height, y0))),
                    (max(0.0, min(display_width, x1)), max(0.0, min(display_height, y1))),
                ],
                "support": 1,
                "angle": angle,
                "length": length,
                "method": "opencv_short_oblique_terminal_hough",
            }
        )
    selected = []
    for candidate in sorted(candidates, key=lambda item: (-item["length"], item["points"])):
        midpoint = tuple(sum(point[axis] for point in candidate["points"]) / 2 for axis in (0, 1))
        duplicate = False
        for current in selected:
            current_midpoint = tuple(sum(point[axis] for point in current["points"]) / 2 for axis in (0, 1))
            angle_delta = abs(candidate["angle"] - current["angle"])
            angle_delta = min(angle_delta, math.pi - angle_delta)
            if math.dist(midpoint, current_midpoint) <= 1.5 and angle_delta <= math.radians(10):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    selected.sort(key=lambda item: (min(point[1] for point in item["points"]), min(point[0] for point in item["points"])))
    return selected


def _line_observations(
    rgb: np.ndarray,
    *,
    display_width: float,
    display_height: float,
    scale: float,
    text_observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    geometry_binary = binary.copy()
    for item in text_observations:
        x0, y0, x1, y1 = item["bbox_display"]
        orientation = str(item.get("orientation") or "horizontal")
        run = (y1 - y0) if orientation.startswith("vertical") else (x1 - x0)
        cross = (x1 - x0) if orientation.startswith("vertical") else (y1 - y0)
        # Tesseract occasionally reads a single heavy rebar stroke as two
        # letters.  Such an implausibly long token must remain available to
        # geometry reconstruction even though the OCR observation is kept.
        if run / max(cross, 1e-6) > max(4.0, 2.5 * len(str(item.get("text") or ""))):
            continue
        pad = max(1, round(scale))
        cv2.rectangle(
            geometry_binary,
            (max(0, round(x0 * scale) - pad), max(0, round(y0 * scale) - pad)),
            (min(gray.shape[1] - 1, round(x1 * scale) + pad), min(gray.shape[0] - 1, round(y1 * scale) + pad)),
            0,
            thickness=-1,
        )
    skeleton = _skeletonize(geometry_binary)
    segments = _merged_hough_segments(
        skeleton,
        scale=scale,
        display_width=display_width,
        display_height=display_height,
    )
    for segment in segments:
        segment["method"] = "opencv_otsu_skeleton_hough_collinear_merge"
    segments.extend(
        _short_oblique_segments(
            skeleton,
            scale=scale,
            display_width=display_width,
            display_height=display_height,
        )
    )
    rows = []
    for segment in segments:
        (x0, y0), (x1, y1) = segment["points"]
        length = math.dist((x0, y0), (x1, y1))
        support = int(segment["support"])
        confidence = min(
            0.97,
            0.50
            + 0.30 * min(1.0, length / max(display_width, display_height))
            + 0.04 * min(4, support - 1),
        )
        rows.append(
            {
                "id": f"raster.line.{len(rows) + 1:05d}",
                "kind": "line_segment",
                "bbox_display": [round(min(x0, x1), 4), round(min(y0, y1), 4), round(max(x0, x1), 4), round(max(y0, y1), 4)],
                "text": None,
                "confidence": round(confidence, 4),
                "geometry_display": {
                    "kind": "line",
                    "points_display": [[round(x0, 4), round(y0, 4)], [round(x1, 4), round(y1, 4)]],
                    "length_points": round(length, 4),
                    "raw_segment_support": support,
                },
                "source_crop_display": [0.0, 0.0, float(display_width), float(display_height)],
                "method": segment["method"],
            }
        )
    return rows, geometry_binary


def _contour_observations(
    geometry_binary: np.ndarray,
    *,
    display_width: float,
    display_height: float,
    scale: float,
) -> list[dict[str, Any]]:
    contours, hierarchy = cv2.findContours(geometry_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    image_area = float(geometry_binary.shape[0] * geometry_binary.shape[1])
    minimum_perimeter = 0.035 * min(geometry_binary.shape)
    candidates = []
    for index, contour in enumerate(contours):
        perimeter = float(cv2.arcLength(contour, True))
        area = abs(float(cv2.contourArea(contour)))
        if perimeter < minimum_perimeter or area < 0.00002 * image_area or area > 0.96 * image_area:
            continue
        approximation = cv2.approxPolyDP(contour, max(1.0, 0.004 * perimeter), True)
        if not 3 <= len(approximation) <= 160:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if min(width, height) < max(3, round(1.5 * scale)):
            continue
        if width < 0.015 * geometry_binary.shape[1] and height < 0.015 * geometry_binary.shape[0]:
            continue
        points = [
            [round(float(point[0][0]) / scale, 4), round(float(point[0][1]) / scale, 4)]
            for point in approximation
        ]
        candidates.append(
            {
                "points": points,
                "bbox": [x / scale, y / scale, (x + width) / scale, (y + height) / scale],
                "perimeter": perimeter / scale,
                "area": area / (scale * scale),
                "parent_index": int(hierarchy[0][index][3]),
            }
        )
    candidates.sort(key=lambda item: (-item["area"], item["bbox"]))
    rows = []
    for candidate in candidates:
        length_ratio = min(1.0, candidate["perimeter"] / max(display_width, display_height))
        rows.append(
            {
                "id": f"raster.contour.{len(rows) + 1:05d}",
                "kind": "closed_contour_candidate",
                "bbox_display": [round(value, 4) for value in candidate["bbox"]],
                "text": None,
                "confidence": round(min(0.88, 0.46 + 0.30 * length_ratio), 4),
                "geometry_display": {
                    "kind": "closed_polyline",
                    "points_display": candidate["points"],
                    "perimeter_points": round(candidate["perimeter"], 4),
                    "area_square_points": round(candidate["area"], 4),
                    "closure_validation": "pixel_contour_closed",
                    "material_role": "unknown",
                },
                "source_crop_display": [0.0, 0.0, float(display_width), float(display_height)],
                "method": "opencv_binary_contour_candidate",
            }
        )
    return rows


def _endpoint_topology(lines: list[dict[str, Any]], *, snap_tolerance: float) -> dict[str, Any]:
    """Snap nearby endpoints while keeping line crossings explicitly unresolved."""

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    clusters: list[dict[str, Any]] = []
    endpoint_cluster: dict[tuple[str, int], int] = {}
    cell = max(snap_tolerance, 1e-6)
    for line in lines:
        for endpoint_index, point in enumerate(line["geometry_display"]["points_display"]):
            x, y = float(point[0]), float(point[1])
            key = (math.floor(x / cell), math.floor(y / cell))
            candidates = [
                cluster_index
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for cluster_index in buckets.get((key[0] + dx, key[1] + dy), [])
            ]
            cluster_index = next(
                (
                    candidate
                    for candidate in candidates
                    if math.dist((x, y), tuple(clusters[candidate]["point_sum"][axis] / clusters[candidate]["count"] for axis in (0, 1)))
                    <= snap_tolerance
                ),
                None,
            )
            if cluster_index is None:
                cluster_index = len(clusters)
                clusters.append({"point_sum": [x, y], "count": 1, "endpoints": []})
                buckets[key].append(cluster_index)
            else:
                clusters[cluster_index]["point_sum"][0] += x
                clusters[cluster_index]["point_sum"][1] += y
                clusters[cluster_index]["count"] += 1
            clusters[cluster_index]["endpoints"].append([line["id"], endpoint_index])
            endpoint_cluster[(line["id"], endpoint_index)] = cluster_index

    vertices = []
    for index, cluster in enumerate(clusters, start=1):
        vertices.append(
            {
                "id": f"raster.vertex.{index:05d}",
                "point_display": [round(value / cluster["count"], 4) for value in cluster["point_sum"]],
                "endpoint_refs": cluster["endpoints"],
                "degree": len(cluster["endpoints"]),
            }
        )
    edges = [
        {
            "id": f"raster.edge.{index:05d}",
            "line_observation_id": line["id"],
            "vertex_ids": [
                vertices[endpoint_cluster[(line["id"], 0)]]["id"],
                vertices[endpoint_cluster[(line["id"], 1)]]["id"],
            ],
        }
        for index, line in enumerate(lines, start=1)
    ]
    incident: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        for vertex_id in edge["vertex_ids"]:
            incident[vertex_id].append(edge["id"])
    unseen = {item["id"] for item in vertices}
    components = []
    edge_by_vertex = incident
    edge_map = {item["id"]: item for item in edges}
    while unseen:
        start = min(unseen)
        queue = deque([start])
        component_vertices: set[str] = set()
        component_edges: set[str] = set()
        while queue:
            vertex_id = queue.popleft()
            if vertex_id in component_vertices:
                continue
            component_vertices.add(vertex_id)
            unseen.discard(vertex_id)
            for edge_id in edge_by_vertex.get(vertex_id, []):
                component_edges.add(edge_id)
                queue.extend(edge_map[edge_id]["vertex_ids"])
        components.append(
            {
                "id": f"raster.component.{len(components) + 1:05d}",
                "vertex_ids": sorted(component_vertices),
                "edge_ids": sorted(component_edges),
                "closed_cycle_candidate": bool(component_vertices)
                and all(len(incident[vertex_id]) == 2 for vertex_id in component_vertices),
            }
        )
    return {
        "vertices": vertices,
        "edges": edges,
        "components": components,
        "crossing_connectivity": "unknown_until_semantic_or_native_topology_support",
        "snap_tolerance_display_points": snap_tolerance,
    }


def _rect_union(rects: list[fitz.Rect]) -> fitz.Rect:
    prepared = []
    for source in rects:
        rect = fitz.Rect(source)
        if rect.width < 0.01:
            rect.x0 -= 0.05
            rect.x1 += 0.05
        if rect.height < 0.01:
            rect.y0 -= 0.05
            rect.y1 += 0.05
        prepared.append(rect)
    result = fitz.Rect(prepared[0])
    for rect in prepared[1:]:
        result |= rect
    return result


def _rect_iou(left: fitz.Rect, right: fitz.Rect) -> float:
    intersection = float((left & right).get_area())
    return intersection / max(float((left | right).get_area()), 1e-9)


def raster_topology_to_generic_structures(
    observations: list[dict[str, Any]],
    vector_topology: Mapping[str, Any],
    *,
    display_width: float,
    display_height: float,
) -> dict[str, Any]:
    """Adapt raster candidates to the native pipeline's generic schemas.

    The adapter preserves endpoint components and never turns an unresolved
    line crossing into connectivity. Its contours and views remain
    observation-backed hypotheses; downstream metric and reprojection gates
    are still required before any engineering fact can be accepted.
    """

    page_rect = fitz.Rect(0.0, 0.0, display_width, display_height)
    page_area = max(float(page_rect.get_area()), 1.0)
    line_by_id = {
        str(item["id"]): item
        for item in observations
        if item.get("kind") == "line_segment"
    }
    edge_by_id = {str(item["id"]): item for item in vector_topology.get("edges", []) or []}

    contours: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for component in vector_topology.get("components", []) or []:
        edges = [edge_by_id[edge_id] for edge_id in component.get("edge_ids", []) if edge_id in edge_by_id]
        lines = [line_by_id[edge["line_observation_id"]] for edge in edges if edge.get("line_observation_id") in line_by_id]
        if not lines:
            continue
        rect = _rect_union([fitz.Rect(item["bbox_display"]) for item in lines])
        component_rows.append(
            {
                "id": str(component["id"]),
                "bbox": rect,
                "line_ids": [str(item["id"]) for item in lines],
            }
        )
        if len(edges) < 2 or rect.width < 2 or rect.height < 2 or rect.get_area() > 0.65 * page_area:
            continue
        vertex_ids = sorted({str(vertex_id) for edge in edges for vertex_id in edge.get("vertex_ids", [])})
        degrees = Counter(str(vertex_id) for edge in edges for vertex_id in edge.get("vertex_ids", []))
        closed = bool(component.get("closed_cycle_candidate"))
        segments = []
        for edge in edges:
            line = line_by_id.get(str(edge.get("line_observation_id")))
            if line is None:
                continue
            points = line["geometry_display"]["points_display"]
            start_vertex_id, end_vertex_id = edge["vertex_ids"]
            dx = abs(float(points[1][0]) - float(points[0][0]))
            dy = abs(float(points[1][1]) - float(points[0][1]))
            axis = "horizontal" if dy <= 0.7 and dx >= 1.0 else "vertical" if dx <= 0.7 and dy >= 1.0 else "oblique"
            segments.append(
                {
                    "id": str(edge["id"]),
                    "kind": "line",
                    "start_display": points[0],
                    "end_display": points[1],
                    "control_points_display": [],
                    "sample_points_display": points,
                    "start_vertex_id": start_vertex_id,
                    "end_vertex_id": end_vertex_id,
                    "axis": axis,
                    "length_points": line["geometry_display"]["length_points"],
                    "primitive_ref": line["id"],
                }
            )
        contours.append(
            {
                "id": f"contour.raster.{len(contours) + 1:05d}",
                "kind": "closed_loop" if closed else "open_profile",
                "bbox_display": [round(value, 4) for value in rect],
                "closed": closed,
                "primitive_refs": [str(item["id"]) for item in lines],
                "segments_display": segments,
                "vertex_ids": vertex_ids,
                "topology": {
                    "segment_count": len(segments),
                    "vertex_count": len(vertex_ids),
                    "connected_component_count": 1,
                    "endpoint_vertex_count": sum(degree == 1 for degree in degrees.values()),
                    "branch_vertex_count": sum(degree > 2 for degree in degrees.values()),
                    "cycle_rank": max(0, len(segments) - len(vertex_ids) + 1),
                },
                "closure_validation": {
                    "status": "candidate" if closed else "not_applicable",
                    "basis": "raster endpoint topology; line crossings remain unresolved",
                    "endpoint_vertex_count": sum(degree == 1 for degree in degrees.values()),
                    "branch_vertex_count": sum(degree > 2 for degree in degrees.values()),
                },
                "epistemic_state": "observed",
                "basis": "raster_endpoint_component_with_page_coordinate_provenance",
                "source_modality": "raster",
                "topology_component_ref": component["id"],
                "style": {"raster_candidate": True},
            }
        )

    closed_boxes = [fitz.Rect(item["bbox_display"]) for item in contours if item["closed"]]
    for observation in observations:
        if observation.get("kind") != "closed_contour_candidate":
            continue
        rect = fitz.Rect(observation["bbox_display"])
        points = observation.get("geometry_display", {}).get("points_display", []) or []
        if len(points) < 3 or rect.width < 2 or rect.height < 2 or rect.get_area() > 0.65 * page_area:
            continue
        if any(_rect_iou(rect, existing) >= 0.85 for existing in closed_boxes):
            continue
        contour_id = f"contour.raster.{len(contours) + 1:05d}"
        vertex_ids = [f"{contour_id}.vertex.{index:03d}" for index in range(1, len(points) + 1)]
        segments = []
        for index, (start, end) in enumerate(zip(points, points[1:] + points[:1]), start=1):
            dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
            axis = "horizontal" if dy <= 0.7 and dx >= 1.0 else "vertical" if dx <= 0.7 and dy >= 1.0 else "oblique"
            segments.append(
                {
                    "id": f"{contour_id}.segment.{index:03d}",
                    "kind": "line",
                    "start_display": start,
                    "end_display": end,
                    "control_points_display": [],
                    "sample_points_display": [start, end],
                    "start_vertex_id": vertex_ids[index - 1],
                    "end_vertex_id": vertex_ids[index % len(vertex_ids)],
                    "axis": axis,
                    "length_points": round(math.dist(start, end), 4),
                    "primitive_ref": observation["id"],
                }
            )
        contours.append(
            {
                "id": contour_id,
                "kind": "closed_loop",
                "bbox_display": observation["bbox_display"],
                "closed": True,
                "primitive_refs": [observation["id"]],
                "segments_display": segments,
                "vertex_ids": vertex_ids,
                "topology": {
                    "segment_count": len(segments),
                    "vertex_count": len(vertex_ids),
                    "connected_component_count": 1,
                    "endpoint_vertex_count": 0,
                    "branch_vertex_count": 0,
                    "cycle_rank": 1,
                },
                "closure_validation": {
                    "status": "candidate",
                    "basis": "closed pixel contour observation without physical-boundary classification",
                    "endpoint_vertex_count": 0,
                    "branch_vertex_count": 0,
                },
                "epistemic_state": "observed",
                "basis": "raster_pixel_contour_with_page_coordinate_provenance",
                "source_modality": "raster",
                "style": {"raster_candidate": True},
            }
        )
        closed_boxes.append(rect)

    parent = list(range(len(component_rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    gap = max(1.5, 0.0015 * math.hypot(display_width, display_height))
    ordered = sorted(range(len(component_rows)), key=lambda index: component_rows[index]["bbox"].x0)
    active: list[int] = []
    for current in ordered:
        rect = component_rows[current]["bbox"]
        active = [index for index in active if component_rows[index]["bbox"].x1 + gap >= rect.x0]
        expanded = rect + (-gap, -gap, gap, gap)
        for other in active:
            if expanded.intersects(component_rows[other]["bbox"] + (-0.05, -0.05, 0.05, 0.05)):
                union(current, other)
        active.append(current)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(component_rows)):
        grouped[find(index)].append(index)
    views = []
    for members in grouped.values():
        line_ids = sorted({line_id for member in members for line_id in component_rows[member]["line_ids"]})
        if len(line_ids) < 4:
            continue
        rect = _rect_union([component_rows[member]["bbox"] for member in members])
        if rect.width < 20 or rect.height < 20 or rect.get_area() > 0.70 * page_area:
            continue
        if max(rect.width / max(rect.height, 0.1), rect.height / max(rect.width, 0.1)) > 35:
            continue
        horizontal = vertical = 0
        for line_id in line_ids:
            points = line_by_id[line_id]["geometry_display"]["points_display"]
            dx, dy = abs(points[1][0] - points[0][0]), abs(points[1][1] - points[0][1])
            horizontal += dy <= 0.7 and dx >= 5.0
            vertical += dx <= 0.7 and dy >= 5.0
        expanded = rect + (-4, -4, 4, 4)
        local_contours = [
            item
            for item in contours
            if expanded.contains(fitz.Rect(item["bbox_display"]).tl)
            and expanded.contains(fitz.Rect(item["bbox_display"]).br)
        ]
        views.append(
            {
                "id": f"view_hypothesis.raster.{len(views) + 1:03d}",
                "role_hypothesis": "drawing_view_candidate",
                "bbox_display": [round(value, 4) for value in rect],
                "confidence": round(min(0.82, 0.42 + 0.04 * math.log1p(len(line_ids))), 4),
                "epistemic_state": "inferred",
                "basis": "raster_spatial_island_over_endpoint_components_without_crossing_merge",
                "primitive_refs": line_ids,
                "label_claim_refs": [],
                "dimension_refs": [],
                "contour_refs": [item["id"] for item in local_contours],
                "source_modality": "raster",
                "topology_component_refs": sorted(component_rows[member]["id"] for member in members),
                "features": {
                    "primitive_count": len(line_ids),
                    "orthogonal_axis_ratio": round((horizontal + vertical) / max(len(line_ids), 1), 4),
                    "accepted_dimensions": 0,
                    "contours": len(local_contours),
                    "topology_component_count": len(members),
                    "crossing_connectivity": vector_topology.get("crossing_connectivity"),
                },
            }
        )
    views.sort(key=lambda item: (item["bbox_display"][1], item["bbox_display"][0], item["id"]))
    for index, view in enumerate(views, start=1):
        view["id"] = f"view_hypothesis.raster.{index:03d}"

    return {
        "contour_hypotheses": contours,
        "view_hypotheses": views,
        "contract": {
            "same_generic_schema_as_native": True,
            "raster_hypotheses_are_not_accepted_entities": True,
            "line_crossings_do_not_create_connectivity": True,
            "physical_and_material_roles_unclassified": True,
        },
    }


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, start)
    position = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
    )
    return math.dist(point, (start[0] + position * dx, start[1] + position * dy))


def detect_raster_dimension_topology(
    observations: list[dict[str, Any]],
    vector_topology: Mapping[str, Any],
    *,
    display_width: float,
    display_height: float,
) -> dict[str, Any]:
    """Find complete geometric dimension chains without reading their labels."""

    edge_by_line = {
        str(edge["line_observation_id"]): str(edge["id"])
        for edge in vector_topology.get("edges", []) or []
    }
    lines = []
    for observation in observations:
        if observation.get("kind") != "line_segment":
            continue
        raw_points = observation.get("geometry_display", {}).get("points_display", []) or []
        if len(raw_points) != 2:
            continue
        start, end = tuple(map(float, raw_points[0])), tuple(map(float, raw_points[1]))
        dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
        length = math.dist(start, end)
        axis = "horizontal" if dy <= 0.7 and dx >= 4.0 else "vertical" if dx <= 0.7 and dy >= 4.0 else "oblique"
        lines.append(
            {
                "id": str(observation["id"]),
                "topology_edge_ref": edge_by_line.get(str(observation["id"])),
                "start": start,
                "end": end,
                "length": length,
                "axis": axis,
                "confidence": float(observation.get("confidence", 0.0)),
            }
        )

    terminals = [
        line
        for line in lines
        if line["axis"] == "oblique"
        and abs(line["end"][0] - line["start"][0]) >= 0.8
        and abs(line["end"][1] - line["start"][1]) >= 0.8
        and 2.0 <= line["length"] <= 12.0
    ]

    def terminal_rows(point: tuple[float, float]) -> list[dict[str, Any]]:
        return [
            line
            for line in terminals
            if _point_segment_distance(point, line["start"], line["end"]) <= 1.8
        ]

    raw_chains = []
    for baseline in lines:
        orientation = baseline["axis"]
        if orientation not in {"horizontal", "vertical"} or baseline["length"] < 18.0:
            continue
        horizontal = orientation == "horizontal"
        along_axis, cross_axis = (0, 1) if horizontal else (1, 0)
        baseline_cross = (baseline["start"][cross_axis] + baseline["end"][cross_axis]) / 2
        baseline_min = min(baseline["start"][along_axis], baseline["end"][along_axis])
        baseline_max = max(baseline["start"][along_axis], baseline["end"][along_axis])
        intersections = []
        for extension in lines:
            expected_axis = "vertical" if horizontal else "horizontal"
            if extension["axis"] != expected_axis or not 8.0 <= extension["length"] <= max(100.0, 4.0 * baseline["length"]):
                continue
            cross_values = (extension["start"][cross_axis], extension["end"][cross_axis])
            if min(cross_values) - 1.5 > baseline_cross or max(cross_values) + 1.5 < baseline_cross:
                continue
            along = (extension["start"][along_axis] + extension["end"][along_axis]) / 2
            if not baseline_min - 8.0 <= along <= baseline_max + 8.0:
                continue
            point = (along, baseline_cross) if horizontal else (baseline_cross, along)
            ticks = terminal_rows(point)
            if ticks:
                intersections.append({"along": along, "point": point, "extensions": [extension], "terminals": ticks})
        intersections.sort(key=lambda item: item["along"])
        clustered = []
        for row in intersections:
            if clustered and abs(row["along"] - clustered[-1]["along"]) <= 2.0:
                cluster = clustered[-1]
                cluster["extensions"].extend(row["extensions"])
                cluster["terminals"].extend(row["terminals"])
                cluster["along"] = sum(item["start"][along_axis] + item["end"][along_axis] for item in cluster["extensions"]) / (2 * len(cluster["extensions"]))
                cluster["point"] = (cluster["along"], baseline_cross) if horizontal else (baseline_cross, cluster["along"])
            else:
                clustered.append(row)
        if len(clustered) < 2:
            continue
        first, last = clustered[0], clustered[-1]
        separation = abs(last["along"] - first["along"])
        if separation < max(10.0, 0.45 * baseline["length"]):
            continue

        selected_extensions = []
        measured_points = []
        for intersection in (first, last):
            ranked = []
            for extension in intersection["extensions"]:
                far = max((extension["start"], extension["end"]), key=lambda point: abs(point[cross_axis] - baseline_cross))
                ranked.append((abs(far[cross_axis] - baseline_cross), extension["confidence"], extension["id"], extension, far))
            _, _, _, extension, far = max(ranked)
            selected_extensions.append(extension)
            measured_points.append(far)
        offsets = [point[cross_axis] - baseline_cross for point in measured_points]
        alignment_residual = abs(measured_points[0][cross_axis] - measured_points[1][cross_axis])
        alignment_tolerance = max(3.0, 0.03 * separation)
        if offsets[0] * offsets[1] <= 0 or alignment_residual > alignment_tolerance:
            continue

        midpoint = (first["along"] + last["along"]) / 2
        half_along = max(28.0, min(68.0, separation * 0.25))
        if horizontal:
            crop_rects = [
                fitz.Rect(midpoint - half_along, baseline_cross - 34, midpoint + half_along, baseline_cross + 3),
                fitz.Rect(midpoint - half_along, baseline_cross - 3, midpoint + half_along, baseline_cross + 34),
            ]
        else:
            crop_rects = [
                fitz.Rect(baseline_cross - 34, midpoint - half_along, baseline_cross + 3, midpoint + half_along),
                fitz.Rect(baseline_cross - 3, midpoint - half_along, baseline_cross + 34, midpoint + half_along),
            ]
        crop_rects = [rect & fitz.Rect(0.0, 0.0, display_width, display_height) for rect in crop_rects]
        endpoint_terminal_rows = [first["terminals"], last["terminals"]]
        terminal_ids = sorted({item["id"] for rows in endpoint_terminal_rows for item in rows})
        topology_refs = sorted(
            {
                ref
                for item in [baseline, *selected_extensions, *(row for rows in endpoint_terminal_rows for row in rows)]
                for ref in [item.get("topology_edge_ref")]
                if ref is not None
            }
        )
        score = min(
            0.88,
            0.58
            + 0.06 * min(2, len(endpoint_terminal_rows[0]))
            + 0.06 * min(2, len(endpoint_terminal_rows[1]))
            + 0.08 * max(0.0, 1.0 - alignment_residual / max(alignment_tolerance, 1e-9))
            + (0.04 if len(topology_refs) >= 5 else 0.0),
        )
        raw_chains.append(
            {
                "orientation": orientation,
                "baseline_ref": baseline["id"],
                "extension_line_refs": [item["id"] for item in selected_extensions],
                "terminal_refs": terminal_ids,
                "dimension_points_display": [list(first["point"]), list(last["point"])],
                "measured_points_display": [list(point) for point in measured_points],
                "topology_edge_refs": topology_refs,
                "endpoint_alignment_residual_points": round(alignment_residual, 4),
                "confidence": round(score, 4),
                "label_crop_bboxes_display": [
                    [round(value, 4) for value in rect]
                    for rect in crop_rects
                    if rect.width >= 3 and rect.height >= 3
                ],
            }
        )

    deduplicated = []
    for candidate in sorted(raw_chains, key=lambda item: (-item["confidence"], item["baseline_ref"])):
        points = candidate["dimension_points_display"]
        if any(
            candidate["orientation"] == current["orientation"]
            and math.dist(points[0], current["dimension_points_display"][0]) <= 2.5
            and math.dist(points[1], current["dimension_points_display"][1]) <= 2.5
            for current in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    deduplicated.sort(key=lambda item: (item["dimension_points_display"][0][1], item["dimension_points_display"][0][0], item["orientation"]))
    label_crops = []
    for index, chain in enumerate(deduplicated, start=1):
        chain_id = f"raster.dimension_chain.{index:04d}"
        chain["id"] = chain_id
        chain["status"] = "complete_geometry_candidate"
        chain["epistemic_state"] = "observed"
        chain["basis"] = "raster baseline with two perpendicular extensions, terminal geometry at both intersections, and aligned far endpoints"
        chain["value_mm"] = None
        for crop_index, bbox in enumerate(chain.pop("label_crop_bboxes_display"), start=1):
            crop = {
                "id": f"{chain_id}.label_crop.{crop_index}",
                "dimension_chain_ref": chain_id,
                "bbox_display": bbox,
                "orientation": chain["orientation"],
                "status": "ocr_not_run",
            }
            label_crops.append(crop)
            chain.setdefault("label_crop_refs", []).append(crop["id"])

    return {
        "schema_version": "0.1.0",
        "layer": "raster_dimension_topology",
        "status": "complete_geometry_candidates_emitted" if deduplicated else "no_complete_geometry_chain",
        "chains": deduplicated,
        "label_crop_candidates": label_crops,
        "summary": {
            "complete_geometry_candidate_count": len(deduplicated),
            "label_crop_candidate_count": len(label_crops),
        },
        "contract": {
            "normalized_raster_topology_only": True,
            "complete_baseline_extensions_and_terminals_required": True,
            "ocr_invoked_by_dimension_topology": False,
            "numeric_value_assigned": False,
            "engineering_dimension_accepted": False,
            "schedule_values_used": False,
        },
    }


def _ocr_pixel_transform(
    crop: fitz.Rect,
    width: int,
    height: int,
    rotation_degrees: int,
) -> list[float]:
    x_scale = crop.width / max(width, 1)
    y_scale = crop.height / max(height, 1)
    if rotation_degrees == 90:
        return [0.0, y_scale, -x_scale, 0.0, crop.x0 + width * x_scale, crop.y0]
    if rotation_degrees == -90:
        return [0.0, -y_scale, x_scale, 0.0, crop.x0, crop.y0 + height * y_scale]
    return [x_scale, 0.0, 0.0, y_scale, crop.x0, crop.y0]


def _transform_bbox(bbox: list[int] | None, transform: list[float], fallback: fitz.Rect) -> list[float]:
    if bbox is None:
        return [round(value, 4) for value in fallback]
    a, b, c, d, e, f = transform
    points = [
        (a * x + c * y + e, b * x + d * y + f)
        for x, y in (
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
        )
    ]
    return [
        round(min(point[0] for point in points), 4),
        round(min(point[1] for point in points), 4),
        round(max(point[0] for point in points), 4),
        round(max(point[1] for point in points), 4),
    ]


def ocr_raster_dimension_label_crops(
    page: fitz.Page,
    dimension_topology: Mapping[str, Any],
) -> dict[str, Any]:
    """OCR only geometry-authorized raster dimension-label crops."""

    attempts = []
    chain_by_id = {
        str(chain["id"]): chain
        for chain in dimension_topology.get("chains", []) or []
    }
    candidates_by_chain_value: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for crop in dimension_topology.get("label_crop_candidates", []) or []:
        crop_rect = fitz.Rect(crop["bbox_display"]) & page.rect
        if crop_rect.width < 3 or crop_rect.height < 3:
            continue
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(DIMENSION_OCR_SCALE, DIMENSION_OCR_SCALE),
            clip=crop_rect,
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        rotations = [(0, image)]
        if crop.get("orientation") == "vertical":
            rotations = [
                (90, image.rotate(90, expand=True)),
                (-90, image.rotate(-90, expand=True)),
            ]
        for rotation_degrees, oriented in rotations:
            transform = _ocr_pixel_transform(crop_rect, image.width, image.height, rotation_degrees)
            result = ocr_isolated_dimension_label(oriented)
            for raw_attempt in result["attempts"]:
                attempt = {
                    **deepcopy(raw_attempt),
                    "id": f"{crop['id']}.attempt.{len(attempts) + 1:04d}",
                    "dimension_chain_ref": crop["dimension_chain_ref"],
                    "crop_ref": crop["id"],
                    "source_crop_display": [round(value, 4) for value in crop_rect],
                    "orientation": crop.get("orientation"),
                    "rotation_degrees": rotation_degrees,
                    "render_size_pixels": [image.width, image.height],
                    "ocr_size_pixels": [oriented.width, oriented.height],
                    "ocr_pixel_to_display": [round(value, 8) for value in transform],
                    "bbox_display": _transform_bbox(raw_attempt.get("bbox_pixels"), transform, crop_rect),
                }
                attempts.append(attempt)
                if attempt.get("status") == "numeric_candidate":
                    candidates_by_chain_value[
                        (str(crop["dimension_chain_ref"]), str(attempt["normalized_text"]))
                    ].append(attempt)

    observations = []
    for (chain_ref, normalized_text), rows in sorted(candidates_by_chain_value.items()):
        strongest = max(rows, key=lambda item: (float(item["confidence"]), -int(item.get("psm") or 99), item["id"]))
        chain = chain_by_id.get(chain_ref, {})
        primitive_refs = sorted(
            {
                str(reference)
                for reference in (
                    chain.get("baseline_ref"),
                    *(chain.get("extension_line_refs", []) or []),
                    *(chain.get("terminal_refs", []) or []),
                )
                if reference is not None
            }
        )
        observation_id = f"raster.dimension_label.{len(observations) + 1:05d}"
        observations.append(
            {
                "id": observation_id,
                "kind": "text",
                "bbox_display": strongest["bbox_display"],
                "text": normalized_text,
                "confidence": strongest["confidence"],
                "geometry_display": None,
                "source_crop_display": strongest["source_crop_display"],
                "method": "tesseract_complete_dimension_chain_crop",
                "orientation": strongest["orientation"],
                "coordinate_transform": {
                    "ocr_pixel_to_display": strongest["ocr_pixel_to_display"],
                    "rotation_degrees": strongest["rotation_degrees"],
                    "render_scale_pixels_per_display_point": DIMENSION_OCR_SCALE,
                    "display_space": "rotation-normalized PyMuPDF page coordinates",
                },
                "dimension_chain_ref": chain_ref,
                "primitive_refs": primitive_refs,
                "topology_edge_refs": sorted(str(item) for item in chain.get("topology_edge_refs", []) or []),
                "crop_refs": sorted({str(item["crop_ref"]) for item in rows}),
                "ocr_attempt_refs": sorted(str(item["id"]) for item in rows),
                "numeric_value_candidate": float(normalized_text),
                "unit_interpretation": "drawing_dimension_unit_unresolved",
                "epistemic_state": "observed",
                "status": "numeric_observation_candidate",
            }
        )

    observations_by_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        observations_by_chain[str(observation["dimension_chain_ref"])].append(observation)
    chain_results = []
    for chain in dimension_topology.get("chains", []) or []:
        rows = observations_by_chain.get(str(chain["id"]), [])
        if len(rows) == 1:
            status = "unique_numeric_observation"
            selected = rows[0]["id"]
            reason = None
        elif len(rows) > 1:
            status = "ambiguous_conflicting_numeric_observations"
            selected = None
            reason = "geometry-authorized crops yielded multiple distinct numeric readings"
        else:
            status = "no_numeric_observation"
            selected = None
            reason = "no geometry-authorized crop yielded a sufficiently confident integer"
        chain_results.append(
            {
                "dimension_chain_ref": chain["id"],
                "status": status,
                "observation_refs": [item["id"] for item in rows],
                "selected_observation_ref": selected,
                "reason": reason,
            }
        )

    rejected_attempts = [item for item in attempts if item.get("status") != "numeric_candidate"]
    return {
        "schema_version": "0.1.0",
        "layer": "raster_dimension_label_ocr",
        "status": "observations_emitted" if observations else "no_numeric_observations",
        "observations": observations,
        "attempts": attempts,
        "chain_results": chain_results,
        "summary": {
            "eligible_chain_count": len(dimension_topology.get("chains", []) or []),
            "eligible_crop_count": len(dimension_topology.get("label_crop_candidates", []) or []),
            "ocr_attempt_count": len(attempts),
            "rejected_attempt_count": len(rejected_attempts),
            "numeric_observation_count": len(observations),
            "unique_chain_reading_count": sum(item["status"] == "unique_numeric_observation" for item in chain_results),
            "conflicting_chain_reading_count": sum(item["status"] == "ambiguous_conflicting_numeric_observations" for item in chain_results),
        },
        "contract": {
            "only_complete_dimension_chain_crops_inspected": True,
            "whole_page_numeric_text_does_not_bind_to_chains": True,
            "rejected_ocr_alternatives_preserved": True,
            "conflicting_numeric_observations_abstain": True,
            "numeric_observation_is_not_an_accepted_dimension": True,
            "metric_scale_assigned": False,
            "schedule_values_used": False,
        },
    }


def vectorize_raster_rgb(
    rgb: np.ndarray,
    *,
    display_width: float,
    display_height: float,
    scale: float,
    text_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a raster page into provenance-preserving vector candidates."""

    text_rows = list(text_observations or [])
    lines, geometry_binary = _line_observations(
        rgb,
        display_width=display_width,
        display_height=display_height,
        scale=scale,
        text_observations=text_rows,
    )
    contours = _contour_observations(
        geometry_binary,
        display_width=display_width,
        display_height=display_height,
        scale=scale,
    )
    return {
        "observations": [*lines, *contours],
        "vector_topology": _endpoint_topology(lines, snap_tolerance=max(1.0, 1.5 / scale)),
    }


def reconstruct_raster_observations(
    page: fitz.Page,
    quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    quality_record = deepcopy(dict(quality or assess_native_page_quality(page)))
    route = quality_record["route"]
    if route == "native":
        observations: list[dict[str, Any]] = []
        vector_topology: dict[str, Any] = {
            "vertices": [],
            "edges": [],
            "components": [],
            "crossing_connectivity": "not_invoked",
            "snap_tolerance_display_points": None,
        }
    else:
        rgb = _render_rgb(page)
        observations = _ocr_observations(rgb, page, RENDER_SCALE)
        vector_topology = {
            "vertices": [],
            "edges": [],
            "components": [],
            "crossing_connectivity": "not_invoked_for_text_only_route",
            "snap_tolerance_display_points": None,
        }
        if route == "raster_reconstruct":
            vectorized = vectorize_raster_rgb(
                rgb,
                display_width=float(page.rect.width),
                display_height=float(page.rect.height),
                scale=RENDER_SCALE,
                text_observations=observations,
            )
            observations.extend(vectorized["observations"])
            vector_topology = vectorized["vector_topology"]
    generic_structures = raster_topology_to_generic_structures(
        observations,
        vector_topology,
        display_width=float(page.rect.width),
        display_height=float(page.rect.height),
    )
    dimension_topology = detect_raster_dimension_topology(
        observations,
        vector_topology,
        display_width=float(page.rect.width),
        display_height=float(page.rect.height),
    )
    dimension_label_ocr = ocr_raster_dimension_label_crops(page, dimension_topology)
    observations.extend(dimension_label_ocr["observations"])
    dimension_ownership = resolve_raster_dimension_ownership(
        dimension_topology,
        dimension_label_ocr,
        vector_topology,
        generic_structures["view_hypotheses"],
        generic_structures["contour_hypotheses"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "raster_observation_fallback",
        "status": "not_invoked" if route == "native" else "observations_emitted" if observations else "invoked_no_observations",
        "route": route,
        "quality": quality_record,
        "observations": observations,
        "vector_topology": vector_topology,
        "generic_structures": generic_structures,
        "dimension_topology": dimension_topology,
        "dimension_label_ocr": dimension_label_ocr,
        "dimension_ownership": dimension_ownership,
        "coordinate_transform": {
            "pixel_to_display": [1.0 / RENDER_SCALE, 0.0, 0.0, 1.0 / RENDER_SCALE, 0.0, 0.0],
            "render_scale_pixels_per_display_point": RENDER_SCALE,
            "display_space": "rotation-normalized PyMuPDF page coordinates",
        },
        "model_provenance": {
            "ocr_engine": "tesseract",
            "ocr_version": _tesseract_version() if route != "native" else None,
            "dimension_label_ocr_method": "tesseract_complete_dimension_chain_crop",
            "dimension_label_render_scale": DIMENSION_OCR_SCALE,
            "line_engine": "opencv_otsu_skeleton_hough_collinear_merge",
            "terminal_engine": "opencv_short_oblique_terminal_hough",
            "contour_engine": "opencv_binary_contour_candidate",
            "dimension_topology_engine": "deterministic_terminal_backed_orthogonal_chain",
            "opencv_version": cv2.__version__,
            "parameters_frozen": True,
        },
        "summary": {
            "processing_seconds": round(perf_counter() - started, 6),
            "observation_count": len(observations),
            "text_observation_count": sum(item["kind"] == "text" for item in observations),
            "line_observation_count": sum(item["kind"] == "line_segment" for item in observations),
            "contour_observation_count": sum(item["kind"] == "closed_contour_candidate" for item in observations),
            "topology_vertex_count": len(vector_topology["vertices"]),
            "topology_component_count": len(vector_topology["components"]),
            "generic_contour_hypothesis_count": len(generic_structures["contour_hypotheses"]),
            "generic_view_hypothesis_count": len(generic_structures["view_hypotheses"]),
            "dimension_chain_candidate_count": dimension_topology["summary"]["complete_geometry_candidate_count"],
            "dimension_label_observation_count": dimension_label_ocr["summary"]["numeric_observation_count"],
            "dimension_label_conflict_count": dimension_label_ocr["summary"]["conflicting_chain_reading_count"],
            "raster_dimension_candidate_count": dimension_ownership["summary"]["dimension_candidate_count"],
            "accepted_raster_dimension_ownership_candidate_count": dimension_ownership["summary"]["accepted_candidate_count"],
        },
        "contract": {
            "observations_are_not_accepted_entities": True,
            "page_coordinates_preserved": True,
            "confidence_explicit": True,
            "raster_crossings_do_not_imply_connectivity": True,
            "contours_have_unknown_material_role": True,
            "generic_structures_remain_hypotheses": True,
            "raster_dimension_chains_are_geometry_candidates_only": True,
            "raster_dimension_labels_are_observations_only": True,
            "raster_dimension_ownership_is_solver_gated": True,
            "schedule_values_used": False,
        },
    }


def augment_observation_graph_with_raster(
    observation_graph: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(observation_graph))
    existing_ids = {str(item.get("id")) for item in result.get("nodes", [])}
    for observation in fallback.get("observations", []) or []:
        node_id = str(observation["id"])
        if node_id in existing_ids:
            raise ValueError(f"raster observation ID collides with native graph: {node_id}")
        result.setdefault("nodes", []).append(
            {
                "id": node_id,
                "kind": observation["kind"],
                "bbox_display": observation["bbox_display"],
                "primitive_refs": observation.get("primitive_refs", [node_id]),
                "text": observation.get("text"),
                "layer_name": None,
                "style": {},
                "orientation": "mixed",
                "observation_method": observation["method"],
                "confidence": observation["confidence"],
                "geometry_display": observation.get("geometry_display"),
                "source_crop_display": observation["source_crop_display"],
                "coordinate_transform": observation.get("coordinate_transform", fallback["coordinate_transform"]),
                "model_provenance": observation.get("model_provenance", fallback["model_provenance"]),
                "dimension_chain_ref": observation.get("dimension_chain_ref"),
                "topology_edge_refs": observation.get("topology_edge_refs", []),
                "crop_refs": observation.get("crop_refs", []),
                "ocr_attempt_refs": observation.get("ocr_attempt_refs", []),
                "numeric_value_candidate": observation.get("numeric_value_candidate"),
                "unit_interpretation": observation.get("unit_interpretation"),
                "epistemic_state": observation.get("epistemic_state"),
                "status": observation.get("status"),
            }
        )
    result["raster_fallback"] = deepcopy(dict(fallback))
    result.setdefault("provenance", {})["methods"] = [
        "native_pdf_scene_graph",
        *([] if fallback.get("status") == "not_invoked" else ["quality_gated_raster_fallback"]),
    ]
    return result
