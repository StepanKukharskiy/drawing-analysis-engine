"""Independent declared-schedule extraction and frozen-result comparison."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import fitz

from src.drawing_engine.project.review_feedback import canonical_graph_sha256


VOLUME_RE = re.compile(r"(?P<value>\d+(?:[.,]\d+)?)\s*(?:м\s*[³3]|3\s*м)", re.I)
STEEL_TITLE = "Ведомость расхода стали, кг"
SUBTOTAL_RE = re.compile(r"^[иit]того$", re.I)
NUMBER_RE = re.compile(r"(?<![\w/])[-+]?\d+(?:[.,]\d+)?(?![\w/])")


@lru_cache(maxsize=1)
def _ocr_language() -> str:
    """Prefer the drawing language while remaining usable on lean installs."""

    import pytesseract

    try:
        installed = set(pytesseract.get_languages(config=""))
    except (OSError, RuntimeError, pytesseract.TesseractError):
        return "eng"
    selected = [language for language in ("rus", "eng") if language in installed]
    return "+".join(selected) or "eng"


def _fold(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def _numbers(text: str) -> list[float]:
    values = []
    for match in NUMBER_RE.finditer(text):
        value = _number(match.group())
        if value is not None:
            values.append(value)
    return values


def _number(text: str) -> float | None:
    normalized = text.strip().replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _union(rectangles: list[fitz.Rect]) -> list[float]:
    rect = fitz.Rect(rectangles[0])
    for candidate in rectangles[1:]:
        rect |= candidate
    return list(rect)


def _grid_regions(page: fitz.Page) -> list[dict[str, Any]]:
    """Find large native-vector table grids without semantic title dispatch."""

    candidates = []
    drawings = page.get_drawings()
    for drawing_index, drawing in enumerate(drawings):
        items = drawing.get("items", [])
        if len(items) != 2 or any(item[0] != "l" for item in items):
            continue
        segments = [(item[1], item[2]) for item in items]
        horizontal = next((row for row in segments if abs(row[0].y - row[1].y) <= 0.4), None)
        vertical = next((row for row in segments if abs(row[0].x - row[1].x) <= 0.4), None)
        if horizontal is None or vertical is None:
            continue
        rect = fitz.Rect(drawing["rect"])
        if rect.width < 0.15 * page.rect.width or rect.height < 0.08 * page.rect.height:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for other in drawings:
            for item in other.get("items", []):
                if item[0] != "l":
                    continue
                start, end = item[1], item[2]
                if abs(start.y - end.y) <= 0.4 and abs(end.x - start.x) >= 0.05 * rect.width:
                    if rect.x0 - 1 <= min(start.x, end.x) and max(start.x, end.x) <= rect.x1 + 1 and rect.y0 - 1 <= start.y <= rect.y1 + 1:
                        ys.append(float(start.y))
                if abs(start.x - end.x) <= 0.4 and abs(end.y - start.y) >= 20.0:
                    if rect.y0 - 1 <= min(start.y, end.y) and max(start.y, end.y) <= rect.y1 + 1 and rect.x0 - 1 <= start.x <= rect.x1 + 1:
                        xs.append(float(start.x))
        xs = sorted({round(value, 2) for value in xs})
        ys = sorted({round(value, 2) for value in ys})
        if len(xs) >= 4 and len(ys) >= 4:
            candidates.append({"bbox_display": list(rect), "xs": xs, "ys": ys, "primitive_ref": f"drawing[{drawing_index}]"})
    unique = {}
    for item in candidates:
        key = tuple(round(value, 1) for value in item["bbox_display"])
        unique.setdefault(key, item)
    return list(unique.values())


def _numeric_cell_observations(
    page: fitz.Page,
    box: fitz.Rect,
    *,
    scales: tuple[int, ...] = (8, 10, 12),
    modes: tuple[int, ...] = (8, 10, 13),
) -> list[dict[str, Any]]:
    import pytesseract
    from PIL import Image

    rows = []
    for scale in scales:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box, colorspace=fitz.csGRAY, alpha=False)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
        for mode in modes:
            try:
                text = pytesseract.image_to_string(
                    image,
                    config=f"--psm {mode} -l {_ocr_language()} -c tessedit_char_whitelist=0123456789.,-",
                    timeout=10,
                ).strip()
            except (OSError, RuntimeError, pytesseract.TesseractError):
                continue
            if text:
                rows.append({"scale": scale, "psm": mode, "text": text})
    return rows


def _numeric_candidates(observations: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return (value, support) candidates, preserving possible lost decimals."""

    scores: dict[float, float] = {}
    for observation in observations:
        token = observation["text"].strip().replace(",", ".")
        token = re.sub(r"[^0-9.\-]", "", token)
        if not token or token in {"-", "."}:
            continue
        if "." in token:
            value = _number(token)
            if value is not None:
                scores[round(value, 4)] = scores.get(round(value, 4), 0.0) + 3.0
            continue
        if not token.lstrip("-").isdigit():
            continue
        integer = int(token)
        for value, weight in ((float(integer), 0.5), (integer / 10.0, 1.0), (integer / 100.0, 0.7)):
            scores[round(value, 4)] = scores.get(round(value, 4), 0.0) + weight
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _best_value(candidates: list[tuple[float, float]], lower: float, upper: float) -> float | None:
    rows = [item for item in candidates if lower <= item[0] <= upper]
    return rows[0][0] if rows else None


def _ocr_grid_cell(
    page: fitz.Page,
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    scales: tuple[int, ...] = (8, 10, 12),
    modes: tuple[int, ...] = (8, 10, 13),
) -> dict[str, Any]:
    # Keep a sliver of the native cell border.  The outlined CAD font used by
    # several sheets places the first and last digit very close to that border;
    # an inset crop can turn 3 into 5 or drop the leading digit entirely.
    box = (fitz.Rect(left - 0.8, top - 0.8, right + 0.8, bottom + 0.8)) & page.rect
    observations = _numeric_cell_observations(page, box, scales=scales, modes=modes)
    return {"bbox_display": list(box), "observations": observations, "candidates": _numeric_candidates(observations)}


def _ocr_words_region(page: fitz.Page, box: fitz.Rect, *, scale: int = 4) -> list[dict[str, Any]]:
    """Read semantic OCR tokens once and retain their display coordinates."""

    import pytesseract
    from PIL import Image

    clipped = box & page.rect
    if clipped.is_empty or clipped.width <= 0 or clipped.height <= 0:
        return []
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clipped, colorspace=fitz.csGRAY, alpha=False)
    if pixmap.width <= 0 or pixmap.height <= 0:
        return []
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        data = pytesseract.image_to_data(
            image,
            config=f"--psm 11 -l {_ocr_language()}",
            output_type=pytesseract.Output.DICT,
            timeout=60,
        )
    except (OSError, RuntimeError, ValueError, pytesseract.TesseractError):
        return []
    words = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = -1.0
        if confidence < 15.0:
            continue
        left = clipped.x0 + float(data["left"][index]) / scale
        top = clipped.y0 + float(data["top"][index]) / scale
        right = left + float(data["width"][index]) / scale
        bottom = top + float(data["height"][index]) / scale
        words.append(
            {
                "text": text,
                "folded": _fold(text),
                "confidence": round(confidence / 100.0, 4),
                "bbox_display": [left, top, right, bottom],
            }
        )
    return words


def _table_line_segments(page: fitz.Page) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return horizontal (x0, x1, y) and vertical (x, y0, y1) native lines."""

    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            pairs = []
            if item[0] == "l":
                pairs.append((item[1], item[2]))
            elif item[0] == "re":
                rect = fitz.Rect(item[1])
                pairs.extend(
                    [
                        (fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x1, rect.y0)),
                        (fitz.Point(rect.x0, rect.y1), fitz.Point(rect.x1, rect.y1)),
                        (fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x0, rect.y1)),
                        (fitz.Point(rect.x1, rect.y0), fitz.Point(rect.x1, rect.y1)),
                    ]
                )
            for start, end in pairs:
                if abs(start.y - end.y) <= 0.5 and abs(start.x - end.x) >= 3.0:
                    horizontal.append((min(start.x, end.x), max(start.x, end.x), (start.y + end.y) / 2.0))
                if abs(start.x - end.x) <= 0.5 and abs(start.y - end.y) >= 3.0:
                    vertical.append(((start.x + end.x) / 2.0, min(start.y, end.y), max(start.y, end.y)))
    return horizontal, vertical


def _enclosing_cell(
    page: fitz.Page,
    evidence_box: fitz.Rect,
    segments: tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]] | None = None,
) -> fitz.Rect:
    """Tighten token evidence to its native vector cell when four borders exist."""

    horizontal, vertical = segments or _table_line_segments(page)
    centre = (evidence_box.tl + evidence_box.br) / 2.0
    # OCR boxes often include a cell border as part of an outlined glyph. Use
    # the token centre to select borders so that this raster artefact cannot
    # expand exact-cell evidence into the following row or column.
    min_horizontal = max(8.0, 1.25 * evidence_box.width)
    min_vertical = max(8.0, 1.25 * evidence_box.height)
    top = [
        y
        for x0, x1, y in horizontal
        if x1 - x0 >= min_horizontal and x0 - 2 <= centre.x <= x1 + 2 and y <= centre.y
    ]
    bottom = [
        y
        for x0, x1, y in horizontal
        if x1 - x0 >= min_horizontal and x0 - 2 <= centre.x <= x1 + 2 and y >= centre.y
    ]
    left = [
        x
        for x, y0, y1 in vertical
        if y1 - y0 >= min_vertical and y0 - 2 <= centre.y <= y1 + 2 and x <= centre.x
    ]
    right = [
        x
        for x, y0, y1 in vertical
        if y1 - y0 >= min_vertical and y0 - 2 <= centre.y <= y1 + 2 and x >= centre.x
    ]
    if not (top and bottom and left and right):
        return fitz.Rect(evidence_box)
    cell = fitz.Rect(max(left), max(top), min(right), min(bottom))
    if (
        cell.width < 3.0
        or cell.height < 3.0
        or cell.width > 0.45 * page.rect.width
        or cell.height > 0.25 * page.rect.height
    ):
        return fitz.Rect(evidence_box)
    return cell


def _grid_matrix(page: fitz.Page, region: dict[str, Any]) -> dict[str, Any]:
    xs, ys = region["xs"], region["ys"]
    words = _ocr_words_region(page, fitz.Rect(region["bbox_display"]), scale=5)
    rows = []
    for row_index, (top, bottom) in enumerate(zip(ys, ys[1:])):
        row = []
        for column_index, (left, right) in enumerate(zip(xs, xs[1:])):
            tokens = []
            for word in words:
                box = fitz.Rect(word["bbox_display"])
                centre = (box.tl + box.br) / 2.0
                if left <= centre.x <= right and top <= centre.y <= bottom:
                    tokens.append(word)
            tokens.sort(key=lambda item: (item["bbox_display"][1], item["bbox_display"][0]))
            row.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "text": " ".join(item["text"] for item in tokens),
                    "bbox_display": [left, top, right, bottom],
                    "tokens": tokens,
                }
            )
        rows.append(row)
    return {
        "bbox_display": region["bbox_display"],
        "rows": rows,
        "primitive_ref": region["primitive_ref"],
        "observation_basis": "native_vector_grid_plus_cell_ocr",
    }


def _native_table_matrices(page: fitz.Page) -> list[dict[str, Any]]:
    """Use PyMuPDF only as a native cell topology provider, never as truth."""

    try:
        tables = page.find_tables(strategy="lines").tables
    except (AttributeError, RuntimeError, ValueError):
        return []
    matrices = []
    page_area = page.rect.width * page.rect.height
    for table_index, table in enumerate(tables):
        rect = fitz.Rect(table.bbox)
        if table.row_count < 3 or table.col_count < 3 or rect.width < 0.12 * page.rect.width:
            continue
        if rect.width * rect.height > 0.7 * page_area:
            continue
        extracted = table.extract()
        rows = []
        for row_index, table_row in enumerate(table.rows):
            row = []
            for column_index, cell_box in enumerate(table_row.cells):
                text = ""
                if row_index < len(extracted) and column_index < len(extracted[row_index]):
                    text = extracted[row_index][column_index] or ""
                row.append(
                    {
                        "row": row_index,
                        "column": column_index,
                        "text": " ".join(str(text).split()),
                        "bbox_display": list(cell_box) if cell_box else None,
                        "tokens": [],
                    }
                )
            rows.append(row)
        matrices.append(
            {
                "bbox_display": list(rect),
                "rows": rows,
                "primitive_ref": f"native_table[{table_index}]",
                "observation_basis": "native_vector_table_cells_and_text",
            }
        )
    return matrices


def _matrix_text(matrix: dict[str, Any]) -> str:
    return _fold(" ".join(cell["text"] for row in matrix["rows"] for cell in row))


def _steel_context(text: str) -> bool:
    folded = _fold(text)
    return bool(
        re.search(
            r"стал|арматур|steel|rebar|[aа]\s*[245]\d{2}|гост\s*(?:34028|5781)",
            folded,
            re.I,
        )
    )


def _cell_value(cell: dict[str, Any], *, lower: float, upper: float) -> float | None:
    candidates = [value for value in _numbers(cell.get("text", "")) if lower <= value <= upper]
    return candidates[-1] if candidates else None


def _concrete_from_matrix(page: fitz.Page, matrix: dict[str, Any]) -> dict[str, Any] | None:
    for row in matrix["rows"]:
        row_text = _fold(" ".join(cell["text"] for cell in row))
        if not re.search(r"бетон|beton|concr", row_text, re.I):
            continue
        for cell in row:
            match = VOLUME_RE.search(cell["text"])
            if not match or not cell["bbox_display"]:
                continue
            value = _number(match.group("value"))
            if value is None:
                continue
            return {
                "value": value,
                "cells": [{"label": "declared_total", "value_m3": value, "bbox_display": cell["bbox_display"]}],
                "bbox_display": cell["bbox_display"],
                "primitive_ref": matrix["primitive_ref"],
                "basis": f"{matrix['observation_basis']}_material_total_cell",
            }
    return None


def _closed_arithmetic_rows(matrix: dict[str, Any]) -> tuple[tuple[int, int, int], list[dict[str, Any]]] | None:
    column_count = max((len(row) for row in matrix["rows"]), default=0)
    best: tuple[int, float, tuple[int, int, int], list[dict[str, Any]]] | None = None
    for quantity_column, unit_column, total_column in combinations(range(column_count), 3):
        closed = []
        residual_sum = 0.0
        for row_index, row in enumerate(matrix["rows"]):
            if total_column >= len(row):
                continue
            quantities = [value for value in _numbers(row[quantity_column]["text"]) if 1 <= value <= 10000 and abs(value - round(value)) <= 0.001]
            unit_masses = [value for value in _numbers(row[unit_column]["text"]) if 0.001 <= value <= 10000]
            totals = [value for value in _numbers(row[total_column]["text"]) if 0.001 <= value <= 100000]
            matches = []
            for quantity in quantities:
                for unit_mass in unit_masses:
                    for total in totals:
                        residual = abs(quantity * unit_mass - total)
                        if residual <= max(0.03, 0.012 * total):
                            matches.append((residual, quantity, unit_mass, total))
            if not matches:
                continue
            residual, quantity, unit_mass, total = min(matches)
            residual_sum += residual
            closed.append(
                {
                    "row": row_index,
                    "quantity": quantity,
                    "unit_mass_kg": unit_mass,
                    "value_kg": total,
                    "arithmetic_closed": True,
                    "quantity_bbox_display": row[quantity_column]["bbox_display"],
                    "unit_mass_bbox_display": row[unit_column]["bbox_display"],
                    "bbox_display": row[total_column]["bbox_display"],
                }
            )
        candidate = (len(closed), -residual_sum, (quantity_column, unit_column, total_column), closed)
        if len(closed) >= 2 and (best is None or candidate[:2] > best[:2]):
            best = candidate
    return (best[2], best[3]) if best is not None else None


def _steel_from_matrix(page: fitz.Page, matrix: dict[str, Any]) -> dict[str, Any] | None:
    if not _steel_context(_matrix_text(matrix)):
        return None
    for row_index, row in enumerate(matrix["rows"]):
        for cell in row:
            folded = _fold(cell["text"])
            if not re.search(r"всего|итого|grand\s*total|overall\s*total", folded, re.I):
                continue
            column = cell["column"]
            candidates = []
            for later_row in matrix["rows"][row_index + 1 :]:
                if column >= len(later_row):
                    continue
                value_cell = later_row[column]
                value = _cell_value(value_cell, lower=1.0, upper=100000.0)
                if value is not None and value_cell["bbox_display"]:
                    candidates.append((later_row[0]["row"] - row_index, value, value_cell))
            if candidates:
                _, value, value_cell = min(candidates, key=lambda item: item[0])
                return {
                    "value": value,
                    "cells": [{"label": "overall", "value_kg": value, "bbox_display": value_cell["bbox_display"]}],
                    "bbox_display": value_cell["bbox_display"],
                    "primitive_ref": matrix["primitive_ref"],
                    "basis": f"{matrix['observation_basis']}_overall_total_cell",
                }
    closed = _closed_arithmetic_rows(matrix)
    if closed is None:
        return None
    columns, rows = closed
    value = round(sum(row["value_kg"] for row in rows), 6)
    total_boxes = [fitz.Rect(row["bbox_display"]) for row in rows if row["bbox_display"]]
    if not total_boxes:
        return None
    return {
        "value": value,
        "cells": rows,
        "bbox_display": _union(total_boxes),
        "cell_bboxes_display": [list(box) for box in total_boxes],
        "column_roles": {"quantity": columns[0], "unit_mass": columns[1], "total_mass": columns[2]},
        "primitive_ref": matrix["primitive_ref"],
        "basis": f"{matrix['observation_basis']}_arithmetic_closed_row_total_sum",
    }


def _ocr_rows(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: ((item["bbox_display"][1] + item["bbox_display"][3]) / 2, item["bbox_display"][0])):
        box = fitz.Rect(word["bbox_display"])
        centre_y = (box.y0 + box.y1) / 2.0
        match = None
        for row in rows:
            row_centre = sum((item["bbox_display"][1] + item["bbox_display"][3]) / 2 for item in row) / len(row)
            row_height = max(item["bbox_display"][3] - item["bbox_display"][1] for item in row)
            if abs(centre_y - row_centre) <= max(5.0, 0.65 * max(row_height, box.height)):
                match = row
                break
        if match is None:
            match = []
            rows.append(match)
        match.append(word)
        match.sort(key=lambda item: item["bbox_display"][0])
    return rows


def _reocr_cell_value(
    page: fitz.Page,
    box: fitz.Rect,
    *,
    lower: float,
    upper: float,
    prefer_integer: bool = False,
) -> tuple[float | None, dict[str, Any]]:
    cell = _ocr_grid_cell(page, box.x0, box.y0, box.x1, box.y1)
    if prefer_integer:
        support: dict[float, float] = {}
        maximum_scale = max((observation["scale"] for observation in cell["observations"]), default=None)
        for observation in cell["observations"]:
            if maximum_scale is not None and observation["scale"] != maximum_scale:
                continue
            token = re.sub(r"[^0-9-]", "", observation["text"])
            if token and token not in {"-"}:
                value = _number(token)
                if value is not None and lower <= value <= upper:
                    # Outlined CAD digits separate more reliably at the highest
                    # render scale; use its PSM consensus before lower scales.
                    support[value] = support.get(value, 0.0) + 1.0
        if support:
            return max(support.items(), key=lambda item: (item[1], item[0]))[0], cell
    return _best_value(cell["candidates"], lower, upper), cell


def _outlined_steel_arithmetic(
    page: fitz.Page,
    words: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Close repeated class-subtotal rows in an outlined steel table."""

    steel_words = [word for word in words if re.search(r"стал|steel", word["folded"], re.I)]
    for region in _grid_regions(page):
        rect = fitz.Rect(region["bbox_display"])
        if not any(
            rect.x0 <= (fitz.Rect(word["bbox_display"]).x0 + fitz.Rect(word["bbox_display"]).x1) / 2 <= rect.x1
            and rect.y0 - 45 <= fitz.Rect(word["bbox_display"]).y0 <= rect.y1
            for word in steel_words
        ):
            continue
        xs, ys = region["xs"], region["ys"]
        row_indices = []
        for row_index, (top, bottom) in enumerate(zip(ys, ys[1:])):
            row_words = []
            for word in words:
                box = fitz.Rect(word["bbox_display"])
                centre = (box.tl + box.br) / 2.0
                if rect.x0 <= centre.x <= rect.x1 and top <= centre.y <= bottom:
                    row_words.append(word)
            first_column_words = [
                word
                for word in row_words
                if xs[0] <= (word["bbox_display"][0] + word["bbox_display"][2]) / 2 <= xs[1]
            ]
            numeric_words = [word for word in row_words if _numbers(word["text"])]
            if first_column_words and len(numeric_words) >= 2:
                row_indices.append(row_index)
        if len(row_indices) < 2:
            continue
        numeric_rows = []
        for row_index in row_indices:
            top, bottom = ys[row_index], ys[row_index + 1]
            numeric_rows.append(
                [
                    _ocr_grid_cell(page, left, top, right, bottom, scales=(8,))
                    for left, right in zip(xs[1:], xs[2:])
                ]
            )
        best = None
        column_count = len(xs) - 2
        for left_column in range(column_count - 2):
            middle_column = left_column + 1
            for total_column in range(middle_column + 1, min(column_count, middle_column + 3)):
                parsed_rows = []
                closure_count = 0
                support_sum = 0.0
                for row_offset, cells in enumerate(numeric_rows):
                    left_values = [(value, support) for value, support in cells[left_column]["candidates"] if 0.01 <= value <= 10000]
                    middle_values = [(value, support) for value, support in cells[middle_column]["candidates"] if 0.01 <= value <= 10000]
                    total_values = [(value, support) for value, support in cells[total_column]["candidates"] if 0.01 <= value <= 100000]
                    matches = []
                    for left_value, left_support in left_values:
                        for middle_value, middle_support in middle_values:
                            for total_value, total_support in total_values:
                                residual = abs(left_value + middle_value - total_value)
                                if residual <= max(0.05, 0.008 * total_value):
                                    matches.append(
                                        (
                                            left_support + middle_support + total_support,
                                            -residual,
                                            total_value,
                                        )
                                    )
                    arithmetic_closed = bool(matches)
                    if arithmetic_closed:
                        support, _, total_value = max(matches)
                        closure_count += 1
                    else:
                        direct = [(support, value) for value, support in total_values if 5 <= value <= 500]
                        if not direct:
                            continue
                        support, total_value = max(direct)
                    support_sum += support
                    parsed_rows.append(
                        {
                            "row": row_indices[row_offset],
                            "value_kg": total_value,
                            "arithmetic_closed": arithmetic_closed,
                            "bbox_display": cells[total_column]["bbox_display"],
                        }
                    )
                # Two independent closures establish the column role; a third
                # row may then use its directly printed subtotal when one OCR
                # operand is damaged by a cell border.
                score = (len(parsed_rows), closure_count, -left_column, -total_column, support_sum)
                if closure_count >= 2 and (best is None or score > best[0]):
                    best = (score, parsed_rows, total_column)
        if best is None:
            continue
        _, closed, total_column = best
        total_boxes = [fitz.Rect(row["bbox_display"]) for row in closed]
        return {
            "value": round(sum(row["value_kg"] for row in closed), 6),
            "cells": closed,
            "bbox_display": _union(total_boxes),
            "cell_bboxes_display": [list(box) for box in total_boxes],
            "primitive_ref": region["primitive_ref"],
            "basis": "outlined_schedule_arithmetic_closed_class_subtotal_sum",
            "total_column": total_column + 1,
        }
    return None


def _ocr_page_schedule_candidates(page: fitz.Page) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Fallback for outlined tables whose vector grid is fragmented."""

    words = _ocr_words_region(page, page.rect, scale=4)
    if not words:
        return None, None
    rows = _ocr_rows(words)
    segments = _table_line_segments(page)
    concrete = None
    material_rows = [
        index
        for index, row in enumerate(rows)
        if re.search(r"материал|material", _fold(" ".join(word["text"] for word in row)), re.I)
    ]
    for row_index, row in enumerate(rows):
        row_text = _fold(" ".join(word["text"] for word in row))
        near_material = any(0 < row_index - material_row <= 3 for material_row in material_rows)
        concrete_semantics = bool(re.search(r"бетон|beton|concr|\bbct\b|\b[вb]\s*\d{2,3}\b", row_text, re.I))
        if not concrete_semantics or ("bct" in row_text and not near_material):
            continue
        numeric_words = []
        for word in row:
            raw = word["text"].strip()
            value = _number(raw)
            if value is None or not (0.01 <= value <= 100.0):
                continue
            if not re.search(r"[.,]", raw):
                continue
            numeric_words.append((value, word))
        if not numeric_words:
            continue
        resolved_cells = []
        seen_boxes = set()
        for _, word in numeric_words:
            evidence = fitz.Rect(word["bbox_display"])
            cell_box = _enclosing_cell(page, evidence, segments)
            key = tuple(round(coordinate, 2) for coordinate in cell_box)
            if key in seen_boxes:
                continue
            seen_boxes.add(key)
            value, cell = _reocr_cell_value(page, cell_box, lower=0.01, upper=100.0)
            if value is not None:
                resolved_cells.append((value, cell_box, cell))
        if not resolved_cells:
            continue
        value = round(sum(item[0] for item in resolved_cells), 6)
        cell_boxes = [item[1] for item in resolved_cells]
        concrete = {
            "value": value,
            "cells": [
                {"label": "declared_total" if len(resolved_cells) == 1 else f"declared_component_{index + 1}", "value_m3": item[0], **item[2]}
                for index, item in enumerate(resolved_cells)
            ],
            "bbox_display": _union(cell_boxes),
            "cell_bboxes_display": [list(box) for box in cell_boxes],
            "primitive_ref": "full_page_semantic_ocr",
            "basis": (
                "outlined_schedule_material_row_cell_ocr"
                if len(resolved_cells) == 1
                else "outlined_schedule_material_row_component_cell_ocr_sum"
            ),
        }
        break

    steel = None
    page_text = _fold(" ".join(word["text"] for word in words))
    if _steel_context(page_text):
        header_words = [
            word
            for word in words
            if re.search(r"общий|overall|grand", word["folded"], re.I)
        ]
        for header in header_words:
            header_box = _enclosing_cell(page, fitz.Rect(header["bbox_display"]), segments)
            candidates = []
            for word in words:
                word_box = fitz.Rect(word["bbox_display"])
                centre = (word_box.tl + word_box.br) / 2.0
                if centre.y <= header_box.y1 or centre.y - header_box.y1 > 150:
                    continue
                if not (header_box.x0 - 2 <= centre.x <= header_box.x1 + 2):
                    continue
                if not re.fullmatch(r"\d+(?:[.,]\d+)?", word["text"].strip()):
                    continue
                value_cell = _enclosing_cell(page, word_box, segments)
                candidates.append((value_cell.y0 - header_box.y1, value_cell, word))
            if not candidates:
                continue
            _, cell_box, _ = min(candidates, key=lambda item: item[0])
            value, cell = _reocr_cell_value(page, cell_box, lower=1.0, upper=100000.0, prefer_integer=True)
            if value is None:
                continue
            steel = {
                "value": value,
                "cells": [{"label": "overall", "value_kg": value, **cell}],
                "bbox_display": list(cell_box),
                "primitive_ref": "full_page_semantic_ocr",
                "basis": "outlined_schedule_overall_total_cell_ocr",
            }
            break
        if steel is None:
            steel = _outlined_steel_arithmetic(page, words)
    return concrete, steel


def _ocr_schedule_declarations(
    page: fitz.Page,
    *,
    need_concrete: bool = True,
    need_reinforcement: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrices = [_grid_matrix(page, region) for region in _grid_regions(page)]
    matrices.extend(_native_table_matrices(page))
    concrete_candidates = (
        [candidate for matrix in matrices if (candidate := _concrete_from_matrix(page, matrix))]
        if need_concrete
        else []
    )
    steel_candidates = (
        [candidate for matrix in matrices if (candidate := _steel_from_matrix(page, matrix))]
        if need_reinforcement
        else []
    )
    if (need_concrete and not concrete_candidates) or (need_reinforcement and not steel_candidates):
        page_concrete, page_steel = _ocr_page_schedule_candidates(page)
        if need_concrete and page_concrete:
            concrete_candidates.append(page_concrete)
        if need_reinforcement and page_steel:
            steel_candidates.append(page_steel)

    concrete = []
    if concrete_candidates:
        selected = min(
            concrete_candidates,
            key=lambda item: fitz.Rect(item["bbox_display"]).width * fitz.Rect(item["bbox_display"]).height,
        )
        concrete.append(
            {
                "id": "declared.concrete.ocr.001",
                "kind": "concrete_volume",
                "value": selected["value"],
                "unit": "m3",
                "state": "declared_by_designer",
                "basis": selected["basis"],
                "page": page.number + 1,
                "bbox_display": selected["bbox_display"],
                "components": selected["cells"],
                "primitive_refs": [selected["primitive_ref"]],
            }
        )
    reinforcement = []
    if steel_candidates:
        selected = max(
            steel_candidates,
            key=lambda item: ("overall_total" in item["basis"], len(item["cells"])),
        )
        reinforcement.append(
            {
                "id": "declared.reinforcement.ocr.001",
                "kind": "reinforcement_mass",
                "value": selected["value"],
                "unit": "kg",
                "state": "declared_by_designer",
                "basis": selected["basis"],
                "page": page.number + 1,
                "bbox_display": selected["bbox_display"],
                "cell_bboxes_display": selected.get("cell_bboxes_display", [selected["bbox_display"]]),
                "aggregation_state": "derived" if "row_total_sum" in selected["basis"] else "direct",
                "components": selected["cells"],
                "primitive_refs": [selected["primitive_ref"]],
            }
        )
    return concrete, reinforcement


def _concrete_declarations(page: fitz.Page) -> list[dict[str, Any]]:
    declarations = []
    words = page.get_text("words")
    segments = _table_line_segments(page)
    for block_index, block in enumerate(page.get_text("blocks")):
        text = " ".join(str(block[4]).split())
        folded = text.casefold()
        concrete_at = folded.find("бетон")
        if concrete_at < 0:
            continue
        candidates = []
        for match in VOLUME_RE.finditer(text):
            # Prefer a unit-bearing quantity after the material name. Some CAD
            # exports reverse the cell reading order, so a nearby value before
            # the material is an explicit fallback.
            offset = match.start() - concrete_at
            priority = (0 if 0 <= offset <= 120 else 1, abs(offset))
            if abs(offset) <= 160:
                candidates.append((priority, match))
        if not candidates:
            continue
        _, match = min(candidates, key=lambda item: item[0])
        value = _number(match.group("value"))
        if value is None:
            continue
        block_box = fitz.Rect(block[:4])
        value_words = [
            fitz.Rect(word[:4])
            for word in words
            if block_box.intersects(fitz.Rect(word[:4]))
            and (word_value := _number(str(word[4]).rstrip("мМ³"))) is not None
            and abs(word_value - value) <= 0.0001
        ]
        value_box = value_words[-1] if value_words else block_box
        cell_box = _enclosing_cell(page, value_box, segments)
        declarations.append(
            {
                "id": f"declared.concrete.{len(declarations) + 1:03d}",
                "kind": "concrete_volume",
                "value": value,
                "unit": "m3",
                "state": "declared_by_designer",
                "basis": "material_schedule_text",
                "page": page.number + 1,
                "bbox_display": list(cell_box),
                "evidence_bbox_display": list(block_box),
                "components": [{"label": "declared_total", "value_m3": value, "bbox_display": list(cell_box)}],
                "source_text": text,
                "primitive_refs": [f"text_block[{block_index}]"],
            }
        )
    # Repeated extraction can occur when a CAD export duplicates text layers.
    unique = {}
    for item in declarations:
        key = (item["page"], round(item["value"], 6), tuple(round(value, 1) for value in item["bbox_display"]))
        unique.setdefault(key, item)
    return list(unique.values())


def _steel_declarations(page: fitz.Page) -> list[dict[str, Any]]:
    declarations = []
    anchors = page.search_for(STEEL_TITLE)
    words = page.get_text("words")
    segments = _table_line_segments(page)
    for anchor_index, anchor in enumerate(anchors, start=1):
        window = fitz.Rect(
            max(0, anchor.x0 - 110),
            anchor.y0,
            min(page.rect.x1, anchor.x1 + 300),
            min(page.rect.y1, anchor.y1 + 180),
        )
        local = [(fitz.Rect(word[:4]), str(word[4])) for word in words if window.intersects(fitz.Rect(word[:4]))]
        numeric = [(rect, text, _number(text)) for rect, text in local if _number(text) is not None]
        overall_headers = [(rect, text) for rect, text in local if text.casefold() == "всего"]
        value: float | None = None
        evidence = [anchor]
        value_boxes: list[fitz.Rect] = []
        method = "overall_total_cell"
        components: list[dict[str, Any]] = []
        if overall_headers:
            candidates = []
            for header, _ in overall_headers:
                for rect, text, number in numeric:
                    if rect.y0 <= header.y1 or rect.y0 - header.y1 > 100:
                        continue
                    if abs((rect.x0 + rect.x1 - header.x0 - header.x1) / 2) > 30:
                        continue
                    candidates.append((rect.y0 - header.y1, rect, text, number, header))
            if candidates:
                _, rect, text, value, header = min(candidates, key=lambda item: item[0])
                evidence.extend((header, rect))
                value_box = _enclosing_cell(page, rect, segments)
                value_boxes.append(value_box)
                components.append(
                    {"label": "overall", "value_kg": value, "source_text": text, "bbox_display": list(value_box)}
                )
        if value is None:
            method = "sum_of_declared_class_subtotals"
            subtotals = []
            for label_rect, label_text in local:
                if not SUBTOTAL_RE.match(label_text):
                    continue
                row = [
                    (rect, text, number)
                    for rect, text, number in numeric
                    if rect.x0 >= label_rect.x1
                    and rect.x0 - label_rect.x1 <= 90
                    and abs((rect.y0 + rect.y1 - label_rect.y0 - label_rect.y1) / 2) <= 7
                ]
                if not row:
                    continue
                rect, text, number = min(row, key=lambda item: item[0].x0)
                subtotals.append((label_rect, rect, text, number))
            if subtotals:
                value = sum(item[3] for item in subtotals)
                for label_rect, rect, text, number in subtotals:
                    evidence.extend((label_rect, rect))
                    value_box = _enclosing_cell(page, rect, segments)
                    value_boxes.append(value_box)
                    components.append(
                        {
                            "label": "class_subtotal",
                            "value_kg": number,
                            "source_text": text,
                            "bbox_display": list(value_box),
                        }
                    )
        if value is None:
            continue
        declarations.append(
            {
                "id": f"declared.reinforcement.{anchor_index:03d}",
                "kind": "reinforcement_mass",
                "value": round(value, 6),
                "unit": "kg",
                "state": "declared_by_designer",
                "basis": method,
                "page": page.number + 1,
                "bbox_display": _union(value_boxes or evidence),
                "cell_bboxes_display": [list(box) for box in value_boxes],
                "evidence_bbox_display": _union(evidence),
                "components": components,
                "primitive_refs": [f"schedule_anchor[{anchor_index - 1}]"],
            }
        )
    return declarations


def extract_declared_schedules(source_pdf: Path, *, page_numbers: list[int] | None = None) -> dict[str, Any]:
    """Open the PDF only for independent declaration extraction."""

    started = perf_counter()
    raw = fitz.open(source_pdf)
    selected_pages = list(range(1, len(raw) + 1)) if page_numbers is None else list(page_numbers)
    if (not selected_pages or any(type(n) is not int or not 1 <= n <= len(raw) for n in selected_pages)
            or selected_pages != sorted(set(selected_pages))):
        raw.close()
        raise ValueError("selected pages must be unique, ordered, one-based source page numbers")
    display = fitz.open()
    display.insert_pdf(raw)
    pages = []
    for number in selected_pages:
        page = display[number - 1]
        page.remove_rotation()
        concrete = _concrete_declarations(page)
        reinforcement = _steel_declarations(page)
        ocr_concrete: list[dict[str, Any]] = []
        ocr_reinforcement: list[dict[str, Any]] = []
        if not concrete or not reinforcement:
            ocr_concrete, ocr_reinforcement = _ocr_schedule_declarations(
                page,
                need_concrete=not concrete,
                need_reinforcement=not reinforcement,
            )
        pages.append(
            {
                "page": page.number + 1,
                "concrete": concrete or ocr_concrete,
                "reinforcement": reinforcement or ocr_reinforcement,
                "native_text_available": bool(page.get_text().strip()),
                "ocr_fallback_used": bool((not concrete and ocr_concrete) or (not reinforcement and ocr_reinforcement)),
            }
        )
    display.close()
    raw.close()
    return {
        "schema_version": "0.1.0",
        "source_pdf": str(source_pdf.resolve()),
        "timing": {"declared_schedule_extraction_seconds": round(perf_counter() - started, 6)},
        "pages": pages,
    }


def calculated_from_graph(engineering_graph: dict[str, Any]) -> dict[str, Any]:
    pages = []
    for page in engineering_graph.get("pages", []):
        quantities = page.get("quantities", [])
        concrete = quantities[0].get("net_concrete_m3") if quantities else None
        reinforcement = page.get("reinforcement_quantities") or {}
        centerline = reinforcement.get("total_placed_centerline_m")
        totals = reinforcement.get("totals", {})
        fabrication = reinforcement.get("fabrication_length_m", totals.get("fabrication_length_m"))
        mass = reinforcement.get("mass_kg", totals.get("mass_kg"))
        convention_dependent_mass = reinforcement.get(
            "convention_dependent_mass_kg",
            totals.get("convention_dependent_mass_kg"),
        )
        reinforcement_state = (
            "derived"
            if mass is not None
            else "convention_dependent"
            if convention_dependent_mass is not None
            else "partial"
            if centerline is not None or fabrication is not None
            else "unknown"
        )
        pages.append(
            {
                "page": page["page"],
                "element_id": f"page.{page['page']}.primary",
                "concrete_volume_m3": concrete,
                "concrete_state": "derived" if concrete is not None else "unknown",
                "reinforcement_centerline_m": centerline,
                "reinforcement_fabrication_length_m": fabrication,
                "reinforcement_mass_kg": mass,
                "reinforcement_convention_dependent_mass_kg": convention_dependent_mass,
                "reinforcement_state": reinforcement_state,
                "reason": None if concrete is not None else page.get("specialised_solver", {}).get("reason"),
            }
        )
    return {"pages": pages}


def _comparison(calculated: float | None, declared: float | None, unit: str) -> dict[str, Any]:
    if calculated is None or declared is None:
        return {
            "calculated": calculated,
            "declared": declared,
            "unit": unit,
            "delta": None,
            "delta_percent_of_declared": None,
            "status": "calculation_unavailable" if calculated is None else "declaration_unavailable",
            "severity": "unknown",
        }
    delta = calculated - declared
    percent = abs(delta) / abs(declared) * 100 if declared else math.inf
    severity = "pass" if abs(delta) <= 0.001 or percent <= 0.5 else "review" if percent <= 2.0 else "high"
    return {
        "calculated": calculated,
        "declared": declared,
        "unit": unit,
        "delta": round(delta, 6),
        "delta_percent_of_declared": round(percent, 3),
        "status": "match" if severity == "pass" else "discrepancy",
        "severity": severity,
    }


def build_estimate_comparison(source_pdf: Path, engineering_graph_path: Path) -> dict[str, Any]:
    """Freeze calculated bytes before opening the PDF for declarations."""

    started = perf_counter()
    frozen_bytes = engineering_graph_path.read_bytes()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    engineering_graph = json.loads(frozen_bytes)
    source_hash = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
    graph_binding = engineering_graph.get("result_binding") or {}
    binding_issues = []
    bound_source_hash = graph_binding.get("source_pdf_sha256")
    if bound_source_hash is not None and bound_source_hash != source_hash:
        raise ValueError("engineering graph source PDF hash does not match comparison source")
    if bound_source_hash is None:
        binding_issues.append("engineering_graph_source_hash_missing")
    canonical_graph = engineering_graph.get("canonical_knowledge_graph", {}) or {}
    result_binding = {
        **graph_binding,
        "source_pdf_sha256": source_hash,
        "canonical_engineering_graph_sha256": graph_binding.get(
            "canonical_engineering_graph_sha256"
        ) or canonical_graph_sha256(canonical_graph),
        "pipeline": graph_binding.get("pipeline") or engineering_graph.get("pipeline"),
        "ruleset": graph_binding.get("ruleset") or {"version": "unknown", "sha256": None},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_versions": graph_binding.get("model_versions", []),
        "engineering_graph_artifact_sha256": frozen_hash,
    }
    calculated = calculated_from_graph(engineering_graph)
    selected_pages = engineering_graph.get("processing_options", {}).get("source_page_numbers")
    if selected_pages is not None and selected_pages != [p["page"] for p in calculated["pages"]]:
        raise ValueError("selected page scope differs from frozen engineering graph")
    declarations = (extract_declared_schedules(source_pdf, page_numbers=selected_pages)
                    if selected_pages is not None else extract_declared_schedules(source_pdf))
    page_records = []
    for calculated_page in calculated["pages"]:
        page_number = calculated_page["page"]
        declared_page = next((item for item in declarations["pages"] if item["page"] == page_number), {"concrete": [], "reinforcement": [], "native_text_available": False})
        concrete_declared = declared_page["concrete"][0]["value"] if len(declared_page["concrete"]) == 1 else None
        reinforcement_declared = declared_page["reinforcement"][0]["value"] if len(declared_page["reinforcement"]) == 1 else None
        page_records.append(
            {
                "page": page_number,
                "element_id": calculated_page["element_id"],
                "calculated_from_drawing": calculated_page,
                "declared_by_designer": declared_page,
                "comparison": {
                    "concrete": _comparison(calculated_page["concrete_volume_m3"], concrete_declared, "m3"),
                    "reinforcement_mass": _comparison(calculated_page["reinforcement_mass_kg"], reinforcement_declared, "kg"),
                },
                "approved_for_quote": None,
            }
        )
    return {
        "schema_version": "0.1.0",
        "pipeline_mode": "drawing_first_schedule_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_binding": result_binding,
        "result_currency": "historical_superseded" if binding_issues else "current",
        "binding_issues": binding_issues,
        "source_pdf": str(source_pdf.resolve()),
        "timing": {
            "drawing_understanding_seconds": engineering_graph.get("timing", {}).get("drawing_understanding_seconds"),
            "declared_schedule_extraction_seconds": declarations.get("timing", {}).get("declared_schedule_extraction_seconds"),
            "estimate_comparison_seconds": round(perf_counter() - started, 6),
        },
        "frozen_calculation": {
            "engineering_graph_path": str(engineering_graph_path.resolve()),
            "sha256": frozen_hash,
            "byte_count": len(frozen_bytes),
            "frozen_before_declaration_extraction": True,
            "pipeline": engineering_graph.get("pipeline"),
        },
        "processing_sequence": [
            "read_and_hash_calculated_engineering_graph",
            "extract_calculated_quantities",
            "open_pdf_and_extract_declared_schedules",
            "compare_without_feedback",
        ],
        "contract": {
            "schedule_values_used_as_predictions": False,
            "calculated_declared_approved_are_separate": True,
            "unknown_is_not_zero": True,
        },
        "pages": page_records,
    }


def write_estimate_comparison(source_pdf: Path, engineering_graph_path: Path, output_path: Path) -> Path:
    payload = build_estimate_comparison(source_pdf, engineering_graph_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path
