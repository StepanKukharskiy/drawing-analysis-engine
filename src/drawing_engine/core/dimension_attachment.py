"""Vector-first dimension text to measured-endpoint attachment proposals.

The detector is drawing-neutral. It searches native numeric text, parallel
dimension baselines, perpendicular extension lines, and diagonal terminal
ticks. It emits relation records with exact PDF primitive references rather
than treating a number alone as a dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
import re
from typing import Any

import fitz


@dataclass(frozen=True)
class LineSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    width: float | None
    primitive_ref: str

    @property
    def length(self) -> float:
        return hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])


@dataclass(frozen=True)
class DimensionAttachment:
    attachment_id: str
    value_mm: float
    text: str
    text_method: str
    text_confidence: float
    text_bbox: tuple[float, float, float, float]
    orientation: str
    baseline: LineSegment
    extension_lines: tuple[LineSegment, LineSegment]
    dimension_points: tuple[tuple[float, float], tuple[float, float]]
    measured_points: tuple[tuple[float, float], tuple[float, float]]
    terminal_refs: tuple[str, ...]
    scale_points_per_mm: float
    scale_support: int
    endpoint_alignment_residual: float
    score: float
    status: str
    evidence: tuple[str, ...]
    proposal_status: str | None = None
    terminal_style: str = "terminal_less"
    terminal_refs_by_endpoint: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _segments(page: fitz.Page) -> tuple[LineSegment, ...]:
    segments: list[LineSegment] = []
    for drawing_index, drawing in enumerate(page.get_drawings()):
        width = drawing.get("width")
        for item_index, item in enumerate(drawing["items"]):
            if item[0] != "l":
                continue
            start, end = item[1], item[2]
            segment = LineSegment(
                start=(float(start.x), float(start.y)),
                end=(float(end.x), float(end.y)),
                width=None if width is None else float(width),
                primitive_ref=f"drawing[{drawing_index}].item[{item_index}]",
            )
            if segment.length >= 1.0:
                segments.append(segment)
    return tuple(segments)


def _axis(segment: LineSegment) -> str | None:
    dx = abs(segment.end[0] - segment.start[0])
    dy = abs(segment.end[1] - segment.start[1])
    if dy <= 0.7 and dx >= 4:
        return "horizontal"
    if dx <= 0.7 and dy >= 4:
        return "vertical"
    return None


def _native_numeric_words(page: fitz.Page) -> list[tuple[str, float, fitz.Rect, str, str, float]]:
    text_lines: list[tuple[fitz.Rect, tuple[float, float]]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            direction = tuple(float(value) for value in line.get("dir", (1.0, 0.0)))
            text_lines.append((fitz.Rect(line["bbox"]), direction))

    results = []
    for word in page.get_text("words"):
        text = str(word[4]).strip().replace(",", ".")
        if not text.replace(".", "", 1).isdigit():
            continue
        value = float(text)
        if not (10 <= value <= 100000):
            continue
        box = fitz.Rect(word[:4])
        center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        containing = [item for item in text_lines if center in item[0] + (-0.5, -0.5, 0.5, 0.5)]
        direction = min(containing, key=lambda item: item[0].get_area())[1] if containing else (1.0, 0.0)
        orientation = "vertical" if abs(direction[1]) > abs(direction[0]) else "horizontal"
        results.append((text, value, box, orientation, "native_pdf_text", 1.0))
    return results


def ocr_isolated_dimension_label(image: Any) -> dict[str, Any]:
    """Read one geometry-isolated label and retain every OCR alternative."""

    attempts = []
    try:
        import pytesseract
    except (ImportError, OSError):
        return {
            "selected": None,
            "attempts": [
                {
                    "psm": None,
                    "status": "engine_unavailable",
                    "reason": "pytesseract is unavailable",
                }
            ],
        }

    for psm in (7, 11):
        try:
            data = pytesseract.image_to_data(
                image.convert("L"),
                config=f"--psm {psm} -l eng -c tessedit_char_whitelist=0123456789.,",
                output_type=pytesseract.Output.DICT,
                timeout=5,
            )
        except RuntimeError:
            attempts.append(
                {
                    "psm": psm,
                    "status": "engine_error",
                    "reason": "tesseract timed out or rejected the isolated crop",
                }
            )
            continue
        token_indices = [index for index, text in enumerate(data["text"]) if str(text).strip()]
        tokens = [str(data["text"][index]) for index in token_indices]
        confidences = [
            float(data["conf"][index])
            for index in token_indices
            if float(data["conf"][index]) >= 0
        ]
        normalized = re.sub(r"\s+", "", "".join(tokens))
        bbox_pixels = None
        if token_indices:
            left = min(int(data["left"][index]) for index in token_indices)
            top = min(int(data["top"][index]) for index in token_indices)
            right = max(int(data["left"][index]) + int(data["width"][index]) for index in token_indices)
            bottom = max(int(data["top"][index]) + int(data["height"][index]) for index in token_indices)
            bbox_pixels = [left, top, right, bottom]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        attempt = {
            "psm": psm,
            "raw_text": "".join(tokens),
            "normalized_text": normalized,
            "confidence": round(max(0.0, min(1.0, confidence / 100)), 4),
            "bbox_pixels": bbox_pixels,
        }
        # Geometry-routed fallback is deliberately integer-only. Decimal-like
        # OCR commonly means two adjacent chained labels were concatenated.
        if not re.fullmatch(r"\d{2,6}", normalized):
            attempt.update(
                {
                    "status": "rejected",
                    "reason": "isolated crop did not yield one 2-6 digit integer",
                }
            )
            attempts.append(attempt)
            continue
        value = float(normalized)
        if not 10 <= value <= 100000:
            attempt.update(
                {
                    "status": "rejected",
                    "reason": "numeric result lies outside the supported engineering-label range",
                    "value": value,
                }
            )
            attempts.append(attempt)
            continue
        if confidence < 55:
            attempt.update(
                {
                    "status": "rejected",
                    "reason": "OCR confidence is below 0.55",
                    "value": value,
                }
            )
            attempts.append(attempt)
            continue
        attempt.update(
            {
                "status": "numeric_candidate",
                "reason": None,
                "value": value,
                "confidence": round(min(0.98, confidence / 100), 4),
            }
        )
        attempts.append(attempt)
    valid = [item for item in attempts if item.get("status") == "numeric_candidate"]
    # Preserve the native flow's established PSM preference: it historically
    # returned the first valid result (PSM 7 before PSM 11).
    selected = valid[0] if valid else None
    return {"selected": selected, "attempts": attempts}


def _ocr_value(image: Any) -> tuple[str, float] | None:
    """Compatibility wrapper for the native dimension attachment flow."""

    selected = ocr_isolated_dimension_label(image)["selected"]
    if selected is None:
        return None
    return str(selected["normalized_text"]), float(selected["confidence"])


def _ocr_numeric_words(
    page: fitz.Page,
    segments: tuple[LineSegment, ...],
) -> list[tuple[str, float, fitz.Rect, str, str, float]]:
    """OCR only labels next to complete native-vector dimension chains.

    Whole-sheet OCR is intentionally avoided: hatches and reinforcement are
    visually digit-like.  The native baseline, extension lines, and terminal
    ticks define a small label search region before raster recognition runs.
    """

    try:
        from PIL import Image
    except ImportError:
        return []

    chains: dict[tuple[Any, ...], tuple[LineSegment, str, tuple[Any, ...], tuple[Any, ...]]] = {}
    for baseline in segments:
        orientation = _axis(baseline)
        if orientation is None or baseline.length < 18.0:
            continue
        intersections = _dimension_intersections(baseline, orientation, segments)
        if len(intersections) < 2:
            continue
        first, last = intersections[0], intersections[-1]
        separation = abs(last[0] - first[0])
        if separation < max(10.0, baseline.length * 0.45):
            continue
        if not (_terminal_refs(first[2], segments) and _terminal_refs(last[2], segments)):
            continue
        key = (
            orientation,
            *(round(value, 1) for point in (first[2], last[2]) for value in point),
        )
        chains.setdefault(key, (baseline, orientation, first, last))

    observations = []
    for baseline, orientation, first, last in chains.values():
        separation = abs(last[0] - first[0])
        midpoint = (first[0] + last[0]) / 2
        half_along = max(28.0, min(68.0, separation * 0.25))
        candidates: list[tuple[str, float, fitz.Rect]] = []
        if orientation == "horizontal":
            cross = (baseline.start[1] + baseline.end[1]) / 2
            regions = [
                fitz.Rect(midpoint - half_along, cross - 34, midpoint + half_along, cross + 3),
                fitz.Rect(midpoint - half_along, cross - 3, midpoint + half_along, cross + 34),
            ]
        else:
            cross = (baseline.start[0] + baseline.end[0]) / 2
            regions = [
                fitz.Rect(cross - 34, midpoint - half_along, cross + 3, midpoint + half_along),
                fitz.Rect(cross - 3, midpoint - half_along, cross + 34, midpoint + half_along),
            ]
        for region in regions:
            region &= page.rect
            if region.width < 3 or region.height < 3:
                continue
            pixmap = page.get_pixmap(matrix=fitz.Matrix(8, 8), clip=region, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            rotations = (image,) if orientation == "horizontal" else (image.rotate(90, expand=True), image.rotate(-90, expand=True))
            for rotated in rotations:
                result = _ocr_value(rotated)
                if result is not None:
                    candidates.append((result[0], result[1], region))
        if not candidates:
            continue
        # Prefer the strongest preprocessing / page-segmentation consensus.
        # Ties favor the longer engineering dimension token.
        text, confidence, region = max(candidates, key=lambda item: (item[1], len(item[0])))
        observations.append((text, float(text), region, orientation, "geometry_gated_ocr", confidence))
    return observations


def _numeric_words(page: fitz.Page, segments: tuple[LineSegment, ...] | None = None) -> list[tuple[str, float, fitz.Rect, str, str, float]]:
    native = _native_numeric_words(page)
    if native:
        return native
    return _ocr_numeric_words(page, segments if segments is not None else _segments(page))


def _line_point(segment: LineSegment, orientation: str, minimum: bool) -> tuple[float, float]:
    points = (segment.start, segment.end)
    axis = 0 if orientation == "horizontal" else 1
    return min(points, key=lambda point: point[axis]) if minimum else max(points, key=lambda point: point[axis])


def _intersection_points(
    baseline: LineSegment,
    orientation: str,
    segments: tuple[LineSegment, ...],
    allowance: float = 2.0,
) -> list[tuple[float, LineSegment, tuple[float, float], tuple[float, float]]]:
    """Return perpendicular extension intersections ordered along baseline."""

    horizontal = orientation == "horizontal"
    baseline_coordinate = (baseline.start[1] + baseline.end[1]) / 2 if horizontal else (baseline.start[0] + baseline.end[0]) / 2
    baseline_min = min(baseline.start[0 if horizontal else 1], baseline.end[0 if horizontal else 1])
    baseline_max = max(baseline.start[0 if horizontal else 1], baseline.end[0 if horizontal else 1])
    along_min = baseline_min - allowance
    along_max = baseline_max + allowance
    candidates = []
    for segment in segments:
        if _axis(segment) != ("vertical" if horizontal else "horizontal"):
            continue
        if not (8.0 <= segment.length <= max(100.0, baseline.length * 4.0)):
            continue
        coordinates = (segment.start[1] if horizontal else segment.start[0], segment.end[1] if horizontal else segment.end[0])
        if min(coordinates) - 1.5 > baseline_coordinate or max(coordinates) + 1.5 < baseline_coordinate:
            continue
        along = (segment.start[0] + segment.end[0]) / 2 if horizontal else (segment.start[1] + segment.end[1]) / 2
        if not (along_min <= along <= along_max):
            continue
        intersection = (along, baseline_coordinate) if horizontal else (baseline_coordinate, along)
        far = max((segment.start, segment.end), key=lambda point: _distance(point, intersection))
        candidates.append((along, segment, intersection, far))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _terminal_refs(
    point: tuple[float, float],
    segments: tuple[LineSegment, ...],
) -> tuple[str, ...]:
    def point_segment_distance(segment: LineSegment) -> float:
        ax, ay = segment.start
        bx, by = segment.end
        px, py = point
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        if denominator == 0:
            return _distance(point, segment.start)
        position = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
        return hypot(px - (ax + position * dx), py - (ay + position * dy))

    refs = []
    for segment in segments:
        dx = abs(segment.end[0] - segment.start[0])
        dy = abs(segment.end[1] - segment.start[1])
        if dx < 0.8 or dy < 0.8 or not (2.0 <= segment.length <= 10.0):
            continue
        # CAD terminal ticks commonly cross the dimension line at their
        # midpoint; requiring a tick endpoint to touch loses otherwise exact
        # native geometry.
        if point_segment_distance(segment) <= 1.6:
            refs.append(segment.primitive_ref)
    return tuple(sorted(set(refs)))


def _terminal_kind(
    point: tuple[float, float],
    orientation: str,
    refs: tuple[str, ...],
    segments: tuple[LineSegment, ...],
) -> str:
    """Classify native short-oblique terminals without granting acceptance."""

    if not refs:
        return "none"
    along_axis = 0 if orientation == "horizontal" else 1
    signs = set()
    for segment in segments:
        if segment.primitive_ref not in refs:
            continue
        projections = [endpoint[along_axis] - point[along_axis] for endpoint in (segment.start, segment.end)]
        if min(projections) < -0.7:
            signs.add(-1)
        if max(projections) > 0.7:
            signs.add(1)
    # A tick crosses the dimension point. Arrow legs remain on one side of it.
    if signs == {-1, 1}:
        return "tick"
    if len(refs) >= 2 and len(signs) == 1:
        return "arrow"
    return "explicit_unclassified"


def _terminal_style(
    left_refs: tuple[str, ...],
    right_refs: tuple[str, ...],
    left_point: tuple[float, float],
    right_point: tuple[float, float],
    orientation: str,
    segments: tuple[LineSegment, ...],
) -> str:
    kinds = (
        _terminal_kind(left_point, orientation, left_refs, segments),
        _terminal_kind(right_point, orientation, right_refs, segments),
    )
    if kinds == ("none", "none"):
        return "terminal_less"
    if "none" in kinds:
        return "incomplete"
    return kinds[0] if kinds[0] == kinds[1] else "mixed_explicit"


def _dimension_intersections(
    baseline: LineSegment,
    orientation: str,
    segments: tuple[LineSegment, ...],
) -> list[tuple[float, LineSegment, tuple[float, float], tuple[float, float]]]:
    standard = _intersection_points(baseline, orientation, segments)
    if len(standard) >= 2 and _terminal_refs(standard[0][2], segments) and _terminal_refs(standard[-1][2], segments):
        return standard

    # Some CAD styles stop the baseline at inward arrow tips.  Search a small
    # exterior gap, but admit only extension intersections carrying explicit
    # terminal geometry so nearby object lines cannot widen the dimension.
    expanded = _intersection_points(baseline, orientation, segments, allowance=8.0)
    terminal_backed = [item for item in expanded if _terminal_refs(item[2], segments)]
    return terminal_backed if len(terminal_backed) >= 2 else standard


def _consistent_endpoint_pair(
    intersections: list[tuple[float, LineSegment, tuple[float, float], tuple[float, float]]],
    orientation: str,
    baseline: LineSegment,
) -> tuple[
    tuple[float, LineSegment, tuple[float, float], tuple[float, float]],
    tuple[float, LineSegment, tuple[float, float], tuple[float, float]],
]:
    """Choose endpoint extensions that project to the same side of a chain."""

    first_along, last_along = intersections[0][0], intersections[-1][0]
    first_rows = [item for item in intersections if abs(item[0] - first_along) <= 0.7]
    last_rows = [item for item in intersections if abs(item[0] - last_along) <= 0.7]
    cross_axis = 1 if orientation == "horizontal" else 0
    baseline_cross = (baseline.start[cross_axis] + baseline.end[cross_axis]) / 2

    def direction(item: tuple[float, LineSegment, tuple[float, float], tuple[float, float]]) -> int:
        delta = item[3][cross_axis] - baseline_cross
        return 1 if delta > 0.7 else -1 if delta < -0.7 else 0

    candidates = []
    for first in first_rows:
        for last in last_rows:
            first_direction, last_direction = direction(first), direction(last)
            same_side = first_direction != 0 and first_direction == last_direction
            residual = abs(first[3][cross_axis] - last[3][cross_axis])
            extension_length = _distance(first[2], first[3]) + _distance(last[2], last[3])
            candidates.append(
                (
                    0 if same_side else 1,
                    residual,
                    extension_length,
                    first[1].primitive_ref,
                    last[1].primitive_ref,
                    first,
                    last,
                )
            )
    selected = min(candidates)
    return selected[-2], selected[-1]


def attach_dimensions(page: fitz.Page) -> tuple[DimensionAttachment, ...]:
    """Return high-confidence and near-miss dimension relation proposals."""

    segments = _segments(page)
    # Short chain intervals are common at supports, gaps, and cover bands.  The
    # geometry gate below already requires two perpendicular intersections at
    # least 10 pt apart, so excluding baselines shorter than 18 pt discarded
    # legitimate sub-intervals while nearby longer intervals inherited their
    # labels.  Admit the short baseline here and retain the stricter completed
    # chain, scale-consensus, and ownership gates downstream.
    baselines = [segment for segment in segments if _axis(segment) is not None and segment.length >= 10.0]
    proposals: list[DimensionAttachment] = []
    serial = 1
    for text, value, box, orientation, text_method, text_confidence in _numeric_words(page, segments):
        center = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        for baseline in baselines:
            if _axis(baseline) != orientation:
                continue
            if orientation == "horizontal":
                baseline_cross = (baseline.start[1] + baseline.end[1]) / 2
                if not (-5.0 <= baseline_cross - box.y1 <= 18.0):
                    continue
                if not (min(baseline.start[0], baseline.end[0]) - 4 <= center[0] <= max(baseline.start[0], baseline.end[0]) + 4):
                    continue
            else:
                baseline_cross = (baseline.start[0] + baseline.end[0]) / 2
                if not (-5.0 <= baseline_cross - box.x1 <= 18.0):
                    continue
                if not (min(baseline.start[1], baseline.end[1]) - 4 <= center[1] <= max(baseline.start[1], baseline.end[1]) + 4):
                    continue

            intersections = _dimension_intersections(baseline, orientation, segments)
            if len(intersections) < 2:
                continue
            left, right = _consistent_endpoint_pair(intersections, orientation, baseline)
            separation = abs(right[0] - left[0])
            if separation < max(10.0, baseline.length * 0.45):
                continue
            left_ticks = _terminal_refs(left[2], segments)
            right_ticks = _terminal_refs(right[2], segments)
            terminal_refs = tuple(sorted(set(left_ticks + right_ticks)))
            terminal_style = _terminal_style(
                left_ticks,
                right_ticks,
                left[2],
                right[2],
                orientation,
                segments,
            )
            terminal_score = 0.10 if left_ticks and right_ticks else 0.05 if terminal_refs else 0.0
            text_score = 0.40 if text_method == "native_pdf_text" else 0.35 if text_confidence >= 0.80 else 0.30
            score = text_score + 0.25 + 0.15 + terminal_score
            along_axis = 0 if orientation == "horizontal" else 1
            cross_axis = 1 - along_axis
            measured_separation = abs(left[3][along_axis] - right[3][along_axis])
            alignment_residual = abs(left[3][cross_axis] - right[3][cross_axis])
            scale = measured_separation / value
            proposals.append(
                DimensionAttachment(
                    attachment_id=f"dimension_attachment.{serial:04d}",
                    value_mm=value,
                    text=text,
                    text_method=text_method,
                    text_confidence=text_confidence,
                    text_bbox=tuple(box),
                    orientation=orientation,
                    baseline=baseline,
                    extension_lines=(left[1], right[1]),
                    dimension_points=(left[2], right[2]),
                    measured_points=(left[3], right[3]),
                    terminal_refs=terminal_refs,
                    scale_points_per_mm=scale,
                    scale_support=0,
                    endpoint_alignment_residual=alignment_residual,
                    score=score,
                    status="ambiguous",
                    evidence=(
                        "native numeric text" if text_method == "native_pdf_text" else f"geometry-gated OCR numeric text confidence {text_confidence:.2f}",
                        "parallel native baseline",
                        "two perpendicular extension lines",
                        (
                            f"{terminal_style} native terminals at both endpoints"
                            if left_ticks and right_ticks
                            else "terminal-less chain candidate"
                            if terminal_style == "terminal_less"
                            else "terminal evidence incomplete"
                        ),
                    ),
                    terminal_style=terminal_style,
                    terminal_refs_by_endpoint=(left_ticks, right_ticks),
                )
            )
            serial += 1

    # A label can fall inside the deliberately generous search window of a
    # neighbouring chain interval (or a parallel object edge).  A printed
    # dimension belongs to the nearest baseline in the cross direction and,
    # among coincident chain baselines, to the interval containing its centre.
    # Apply this geometric disambiguation before scale consensus so a long
    # neighbouring interval cannot inherit a short support/gap label.
    by_label: dict[tuple[Any, ...], list[DimensionAttachment]] = {}
    for proposal in proposals:
        label_key = (
            proposal.text,
            tuple(round(value, 2) for value in proposal.text_bbox),
            proposal.orientation,
        )
        by_label.setdefault(label_key, []).append(proposal)
    nearest: list[DimensionAttachment] = []
    for rows in by_label.values():
        orientation = rows[0].orientation
        along_axis = 0 if orientation == "horizontal" else 1
        cross_axis = 1 - along_axis
        box = rows[0].text_bbox
        along_center = (box[along_axis] + box[along_axis + 2]) / 2
        cross_low, cross_high = box[cross_axis], box[cross_axis + 2]

        def gaps(item: DimensionAttachment) -> tuple[float, float]:
            baseline_cross = (item.baseline.start[cross_axis] + item.baseline.end[cross_axis]) / 2
            cross_gap = max(cross_low - baseline_cross, 0.0, baseline_cross - cross_high)
            interval = sorted(point[along_axis] for point in item.dimension_points)
            along_gap = max(interval[0] - along_center, 0.0, along_center - interval[1])
            return cross_gap, along_gap

        minimum_cross = min(gaps(item)[0] for item in rows)
        cross_nearest = [item for item in rows if gaps(item)[0] <= minimum_cross + 0.75]
        minimum_along = min(gaps(item)[1] for item in cross_nearest)
        nearest.extend(item for item in cross_nearest if gaps(item)[1] <= minimum_along + 0.75)

    # Multiple coincident PDF paths can describe one visual dimension. Retain
    # the highest-scoring relation for each text box and measured endpoint pair.
    deduplicated: dict[tuple[Any, ...], DimensionAttachment] = {}
    for proposal in nearest:
        key = (
            tuple(round(value, 1) for value in proposal.text_bbox),
            tuple(round(value, 1) for point in proposal.measured_points for value in point),
        )
        current = deduplicated.get(key)
        if current is None or proposal.score > current.score:
            deduplicated[key] = proposal
    candidates = tuple(deduplicated.values())
    validated = []
    for proposal in candidates:
        # Technical sheets commonly contain several view scales. A legitimate
        # scale therefore needs local consensus, not one global page scale.
        support_geometry = {
            (
                other.orientation,
                *(round(value, 1) for point in other.measured_points for value in point),
            )
            for other in candidates
            if abs(other.scale_points_per_mm / proposal.scale_points_per_mm - 1.0) <= 0.04
        }
        support = len(support_geometry)
        aligned = proposal.endpoint_alignment_residual <= max(
            3.0,
            abs(
                proposal.measured_points[1][0 if proposal.orientation == "horizontal" else 1]
                - proposal.measured_points[0][0 if proposal.orientation == "horizontal" else 1]
            )
            * 0.03,
        )
        both_terminals = all(proposal.terminal_refs_by_endpoint)
        required_support = 2 if proposal.text_method == "geometry_gated_ocr" and proposal.text_confidence >= 0.80 else 3
        scale_score = 0.10 if support >= required_support else 0.0
        score = proposal.score + scale_score
        accepted = both_terminals and aligned and support >= required_support and score >= 0.90
        evidence = proposal.evidence + (
            f"scale-mode support {support}",
            f"measured endpoint alignment residual {proposal.endpoint_alignment_residual:.2f} pt",
        )
        validated.append(
            replace(
                proposal,
                scale_support=support,
                score=score,
                status="accepted" if accepted else "ambiguous",
                proposal_status="accepted" if accepted else "ambiguous",
                evidence=evidence,
            )
        )
    return tuple(sorted(validated, key=lambda item: (item.text_bbox[1], item.text_bbox[0], -item.score)))


def propose_dimensions(page: fitz.Page) -> tuple[DimensionAttachment, ...]:
    """Return dimension proposals without granting detector-level acceptance.

    ``attach_dimensions`` retains its public detector result for compatibility,
    while the estimator pipeline uses this staged form.  Final acceptance is
    deliberately deferred until geometry ownership has been resolved.
    """

    return tuple(
        replace(
            item,
            status="proposed",
            proposal_status=item.proposal_status or item.status,
        )
        for item in attach_dimensions(page)
    )


def attachment_payload(attachments: tuple[DimensionAttachment, ...]) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "method": "native or geometry-gated OCR numeric text + native baseline + two perpendicular extensions + optional native terminals",
        "accepted_count": sum(item.status == "accepted" for item in attachments),
        "ambiguous_count": sum(item.status == "ambiguous" for item in attachments),
        "attachments": [
            {
                "id": item.attachment_id,
                "value_mm": item.value_mm,
                "text": item.text,
                "text_method": item.text_method,
                "text_confidence": item.text_confidence,
                "text_bbox_display": list(item.text_bbox),
                "orientation": item.orientation,
                "baseline": [list(item.baseline.start), list(item.baseline.end)],
                "extension_lines": [[list(line.start), list(line.end)] for line in item.extension_lines],
                "dimension_points_display": [list(point) for point in item.dimension_points],
                "measured_points_display": [list(point) for point in item.measured_points],
                "terminal_style": item.terminal_style,
                "terminal_refs_by_endpoint": [list(refs) for refs in item.terminal_refs_by_endpoint],
                "primitive_refs": [
                    item.baseline.primitive_ref,
                    *(line.primitive_ref for line in item.extension_lines),
                    *item.terminal_refs,
                ],
                "scale_points_per_mm": item.scale_points_per_mm,
                "scale_support": item.scale_support,
                "endpoint_alignment_residual_points": item.endpoint_alignment_residual,
                "score": item.score,
                "status": item.status,
                "proposal_status": item.proposal_status,
                "evidence": list(item.evidence),
            }
            for item in attachments
        ],
    }
