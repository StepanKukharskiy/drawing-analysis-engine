"""Evidence-backed MEP sheet/package registry and floor-grid registration.

The registry is an isolated M1 observation layer.  It classifies pages from
their own title/grid evidence, preserves the existing raster quality decision,
and accepts an axis-aligned adjoining-sheet transform only after redundant
named-grid agreement.  A registration never establishes a routed-system
identity, physical continuation, clash, or quantity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import fitz
import pytesseract
from PIL import Image

SCHEMA_VERSION = "0.1.0"
LAYER = "mep_sheet_coordinate_registry"
OCR_RENDER_SCALE = 1.5

_GRID_LABEL = re.compile(r"(?:[A-Z]{1,2}|[1-9]\d?)(?:\.\d+)?", re.I)
_MECHANICAL_TITLE = re.compile(
    r"LEVEL\s+(\d+)\s+(?:AREA\s+([A-Z0-9.]+)\s+)?"
    r"MECHANICAL(?:\s+ROOM)?\s+PIPING\s+PLAN",
    re.I,
)
_LUBE_TITLE = re.compile(
    r"(?:1ST|FIRST)\s+FLOOR\s+ABOVE\s+GROUND\s+PIPING\s*[-–—]?\s*"
    r"AREA\s+([A-Z0-9.]+)\s*[-–—]?\s*LUBE\s+SYSTEM",
    re.I,
)
_LUBE_LEVEL_ONE = re.compile(
    r"(?:\b1ST\b|\bIST\b|\b[1I]\s*[|.]?\s*ST\b|\bFIRST\b)\s+FLOOR",
    re.I,
)
_HVAC_LEVEL_TITLE = re.compile(
    r"\b((?:PENTHOUSE|ROOF|BASEMENT|[A-Z0-9]+(?:ST|ND|RD|TH)?\s+FLOOR)\s+LEVEL)"
    r"\s*[-–—]\s*HVAC(?:\s*[-–—]\s*(?:NEW|EXISTING))?\b",
    re.I,
)
_MECHANICAL_SECTIONS_TITLE = re.compile(r"\bMECHANICAL\s+SECTIONS\b", re.I)
_MECHANICAL_SHEET = re.compile(r"\bL\d{2}-[A-Z]+-[A-Z]\.[A-Z0-9.]+\b", re.I)
_SHOP_SHEET = re.compile(r"\bBR\d+\b", re.I)
_HVAC_SHEET = re.compile(r"\bM-\d{3}\b", re.I)
_SCALE = re.compile(
    r"SCALE\s*:?[\s|]*(\d+)\s*/\s*(\d+)\s*[\"”]?\s*=\s*"
    r"(\d+)\s*['’]\s*-?\s*(\d+)",
    re.I,
)
_ANY_ARCHITECTURAL_SCALE = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*[\"”]?\s*=\s*(\d+)\s*['’]\s*-?\s*(\d+)",
    re.I,
)
_REVISION_AFTER = re.compile(
    r"\b([A-Z0-9])\s+ISSUED\s+FOR\s+REVIEW\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.I,
)
_REVISION_BEFORE = re.compile(
    r"\b([A-Z0-9])\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+ISSUED\s+FOR\s+REVIEW\b",
    re.I,
)


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_box(box: Iterable[object]) -> list[float]:
    return [round(float(value), 4) for value in box]


def _assess_page_quality_bounded(page: fitz.Page) -> dict[str, Any]:
    """Replay the shared quality thresholds without expensive image hashing.

    ``Page.get_image_rects()`` hashes and decodes every embedded image.  On the
    M1 fixture that turns a route decision into a many-minute operation even
    though image placement boxes are already present in the display list.  The
    bounding-box log also gives a safe path-count lower bound.  Only a page
    below the shared 12-path gate materializes drawings for the exact count.
    """

    bboxlog = page.get_bboxlog()
    path_operations = [
        item for item in bboxlog if str(item[0]) in {"fill-path", "stroke-path"}
    ]
    if len(path_operations) >= 12:
        native_path_count = 12
        count_state = "lower_bound_sufficient_for_route_gate"
    else:
        drawings = [item for item in page.get_drawings() if item.get("type") != "clip"]
        native_path_count = sum(len(item.get("items", [])) for item in drawings)
        count_state = "exact_below_route_gate"
    native_text = page.get_text("text").strip()
    image_rects = [
        fitz.Rect(item[1]) & page.rect
        for item in bboxlog
        if str(item[0]) == "fill-image" and len(item) >= 2
    ]
    page_area = max(float(page.rect.get_area()), 1.0)
    image_area_ratio = min(
        1.0,
        sum(float(rect.get_area()) for rect in image_rects) / page_area,
    )
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
        "schema_version": "0.6.0",
        "layer": "perception_quality_route",
        "route": route,
        "reason": reason,
        "method": "bounded_display_list_replay_of_structural_thresholds_v1",
        "metrics": {
            "native_text_chars": len(native_text),
            "native_path_item_count": native_path_count,
            "native_path_count_state": count_state,
            "native_path_paint_operation_count": len(path_operations),
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
            "shared_structural_thresholds_replayed_without_change": True,
        },
    }


def _normalise_text(value: object) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def _normalise_token(value: object) -> str:
    return re.sub(r"^[^A-Z0-9]+|[^A-Z0-9.]+$", "", str(value or "").upper())


def _rect_union(boxes: Iterable[Iterable[object]]) -> list[float] | None:
    rectangles = [fitz.Rect(box) for box in boxes]
    rectangles = [box for box in rectangles if not box.is_empty]
    if not rectangles:
        return None
    result = fitz.Rect(rectangles[0])
    for box in rectangles[1:]:
        result |= box
    return _round_box(result)


def _native_tokens(page: fitz.Page, page_ref: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for index, word in enumerate(page.get_text("words"), start=1):
        x0, y0, x1, y1, text, block, line, word_index = word[:8]
        value = _normalise_text(text)
        if not value:
            continue
        box = _round_box((x0, y0, x1, y1))
        tokens.append(
            {
                "id": _stable_id(
                    "mep_text_observation",
                    page_ref,
                    "native_pdf_text",
                    int(block),
                    int(line),
                    int(word_index),
                    value,
                    box,
                ),
                "text": value,
                "bbox_display": box,
                "confidence": 1.0,
                "method": "native_pdf_text",
                "reading_order": index,
            }
        )
    return tokens


def _ocr_tokens(page: fitz.Page, page_ref: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scale = OCR_RENDER_SCALE
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False,
        annots=False,
    )
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    data = pytesseract.image_to_data(
        image,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    tokens: list[dict[str, Any]] = []
    for index, raw in enumerate(data.get("text", []), start=1):
        value = _normalise_text(raw)
        try:
            confidence = float(data["conf"][index - 1]) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
        if not value or confidence < 0.45:
            continue
        left = float(data["left"][index - 1]) / scale
        top = float(data["top"][index - 1]) / scale
        width = float(data["width"][index - 1]) / scale
        height = float(data["height"][index - 1]) / scale
        box = _round_box((left, top, left + width, top + height))
        tokens.append(
            {
                "id": _stable_id(
                    "mep_text_observation", page_ref, "tesseract_page_ocr", value, box
                ),
                "text": value,
                "bbox_display": box,
                "confidence": round(confidence, 4),
                "method": "tesseract_page_ocr",
                "reading_order": index,
            }
        )
    drawing_number_tokens = _ocr_drawing_number_field(page, page_ref, tokens)
    level_tokens = _ocr_plan_title_level_field(page, page_ref, tokens)
    tokens.extend(drawing_number_tokens)
    tokens.extend(level_tokens)
    provenance = {
        "method": "tesseract_page_ocr",
        "model_version": str(pytesseract.get_tesseract_version()).splitlines()[0],
        "render_scale": scale,
        "pixel_to_display_matrix": [1.0 / scale, 0.0, 0.0, 1.0 / scale, 0.0, 0.0],
        "annotation_appearances_rendered": False,
        "minimum_confidence": 0.45,
        "drawing_number_field_crop_invoked": bool(drawing_number_tokens),
        "plan_title_level_crop_invoked": bool(level_tokens),
    }
    return tokens, provenance


def _ocr_plan_title_level_field(
    page: fitz.Page,
    page_ref: str,
    page_tokens: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recover the ordinal level from a geometrically discovered plan-title line."""

    wanted = {"FLOOR", "ABOVE", "GROUND", "PIPING", "AREA", "LUBE", "SYSTEM"}
    candidates = [
        item
        for item in page_tokens
        if _normalise_token(item.get("text")).rstrip(".-") in wanted
    ]
    groups: list[list[Mapping[str, Any]]] = []
    for item in sorted(candidates, key=lambda row: fitz.Rect(row["bbox_display"]).y0):
        box = fitz.Rect(item["bbox_display"])
        center_y = (box.y0 + box.y1) / 2
        matching = [
            group
            for group in groups
            if abs(
                center_y
                - sum(
                    (fitz.Rect(row["bbox_display"]).y0 + fitz.Rect(row["bbox_display"]).y1) / 2
                    for row in group
                ) / len(group)
            )
            <= 1.5 * max(box.height, 8.0)
        ]
        if matching:
            matching[0].append(item)
        else:
            groups.append([item])
    groups = [
        group
        for group in groups
        if len({_normalise_token(item.get("text")).rstrip(".-") for item in group} & wanted) >= 5
    ]
    if not groups:
        return []
    group = max(groups, key=lambda rows: _rect_union(row["bbox_display"] for row in rows)[1])
    group_box = fitz.Rect(_rect_union(row["bbox_display"] for row in group))
    height = max(group_box.height, 8.0)
    clip = fitz.Rect(
        max(page.rect.x0, group_box.x0 - 10.5 * height),
        max(page.rect.y0, group_box.y0 - 1.1 * height),
        min(page.rect.x1, group_box.x1 + height),
        min(page.rect.y1, group_box.y1 + 1.1 * height),
    )
    scale = 4.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=clip,
        colorspace=fitz.csGRAY,
        alpha=False,
        annots=False,
    )
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    raw_text = _normalise_text(pytesseract.image_to_string(image, config="--psm 13"))
    if not re.search(r"(?:\b1ST\b|\bIST\b|\b[|I]ST\b)", raw_text, re.I):
        return []
    box = _round_box(clip)
    return [
        {
            "id": _stable_id(
                "mep_text_observation", page_ref, "tesseract_plan_title_level_ocr", "01", box
            ),
            "text": "01",
            "bbox_display": box,
            "confidence": 0.8,
            "method": "tesseract_plan_title_level_ocr",
            "field_hint": "level",
            "normalization": {
                "method": "ordinal_plan_title_ocr_normalization_v1",
                "raw_text": raw_text,
                "normalised_level": "01",
            },
            "reading_order": len(page_tokens) + 1,
        }
    ]


def _ocr_drawing_number_field(
    page: fitz.Page,
    page_ref: str,
    page_tokens: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """OCR a local drawing-number cell only after its printed label is found."""

    drawing = [item for item in page_tokens if "DRAWING" in _normalise_token(item.get("text"))]
    number = [
        item
        for item in page_tokens
        if _normalise_token(item.get("text")).rstrip(".") in {"NO", "NUMBER"}
    ]
    anchors: list[fitz.Rect] = []
    for left in drawing:
        left_box = fitz.Rect(left["bbox_display"])
        for right in number:
            right_box = fitz.Rect(right["bbox_display"])
            if abs((left_box.y0 + left_box.y1 - right_box.y0 - right_box.y1) / 2) <= 3 * max(left_box.height, right_box.height):
                anchors.append(left_box | right_box)
    if not anchors:
        return []
    anchor = max(anchors, key=lambda box: box.y0)
    height = max(anchor.height, 8.0)
    clip = fitz.Rect(
        max(page.rect.x0, anchor.x0 - 0.5 * height),
        max(page.rect.y0, anchor.y1 + 0.25 * height),
        page.rect.x1,
        min(page.rect.y1, anchor.y1 + 13 * height),
    )
    scale = 4.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=clip,
        colorspace=fitz.csGRAY,
        alpha=False,
        annots=False,
    )
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    data = pytesseract.image_to_data(
        image,
        config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        output_type=pytesseract.Output.DICT,
    )
    output = []
    confusion = {"T": "7", "E": "6", "S": "5"}
    for index, raw in enumerate(data.get("text", []), start=1):
        raw_value = _normalise_token(raw)
        match = _SHOP_SHEET.search(raw_value)
        normalization = None
        if match is None:
            confused = re.fullmatch(r"BR([TES])", raw_value)
            if confused is not None:
                value = f"BR{confusion[confused.group(1)]}"
                match = _SHOP_SHEET.search(value)
                normalization = {
                    "method": "outlined_digit_ocr_confusion_resolution_v1",
                    "raw_text": raw_value,
                    "character_map": {confused.group(1): confusion[confused.group(1)]},
                }
        try:
            confidence = float(data["conf"][index - 1]) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
        if match is None or confidence < 0.2:
            continue
        left = clip.x0 + float(data["left"][index - 1]) / scale
        top = clip.y0 + float(data["top"][index - 1]) / scale
        width = float(data["width"][index - 1]) / scale
        height = float(data["height"][index - 1]) / scale
        value = match.group(0).upper()
        box = _round_box((left, top, left + width, top + height))
        output.append(
            {
                "id": _stable_id(
                    "mep_text_observation", page_ref, "tesseract_drawing_number_field_ocr", value, box
                ),
                "text": value,
                "bbox_display": box,
                "confidence": round(confidence, 4),
                "method": "tesseract_drawing_number_field_ocr",
                "field_hint": "sheet_number",
                "normalization": normalization,
                "reading_order": len(page_tokens) + index,
            }
        )
    return sorted(output, key=lambda item: (-item["confidence"], item["bbox_display"][1]))[:1]


def _joined(tokens: Iterable[Mapping[str, Any]]) -> str:
    return _normalise_text(" ".join(str(item.get("text") or "") for item in tokens))


def _token_refs_for_value(
    tokens: Iterable[Mapping[str, Any]], value: object
) -> list[str]:
    parts = [_normalise_token(item) for item in str(value or "").split()]
    parts = [item for item in parts if item]
    if not parts:
        return []
    records = list(tokens)
    normalised = [_normalise_token(item.get("text")) for item in records]
    for start in range(len(records) - len(parts) + 1):
        if normalised[start : start + len(parts)] == parts:
            return [str(item["id"]) for item in records[start : start + len(parts)]]
    wanted = set(parts)
    return [
        str(item["id"])
        for item, token in zip(records, normalised)
        if token in wanted
    ][: max(1, len(parts) * 2)]


def _field(
    name: str,
    value: object | None,
    tokens: list[Mapping[str, Any]],
    *,
    normalised_value: object | None = None,
) -> dict[str, Any]:
    if value is None:
        return {
            "field": name,
            "state": "unknown",
            "value": None,
            "normalised_value": None,
            "evidence_refs": [],
            "bbox_display": None,
            "method": None,
        }
    refs = _token_refs_for_value(tokens, value)
    by_id = {str(item["id"]): item for item in tokens}
    evidence = [by_id[ref] for ref in refs if ref in by_id]
    methods = sorted({str(item["method"]) for item in evidence})
    normalized_evidence = [item for item in evidence if item.get("normalization")]
    return {
        "field": name,
        "state": "derived" if normalized_evidence else "observed",
        "value": str(value),
        "normalised_value": str(normalised_value if normalised_value is not None else value),
        "evidence_refs": refs,
        "bbox_display": _rect_union(item["bbox_display"] for item in evidence),
        "method": (
            str(normalized_evidence[0]["normalization"]["method"])
            if normalized_evidence
            else methods[0] if len(methods) == 1 else "mixed_text_observations"
        ),
    }


def _scale_value(match: re.Match[str]) -> tuple[str, float] | None:
    numerator, denominator, feet, inches = (int(item) for item in match.groups())
    if numerator <= 0 or denominator <= 0:
        return None
    drawing_inches = 12 * feet + inches
    if drawing_inches <= 0:
        return None
    paper_inches = numerator / denominator
    exact = match.group(0)
    return exact, drawing_inches / paper_inches


def _parse_scale(text: str) -> tuple[str, float | None] | None:
    labelled = _SCALE.search(text)
    if labelled is not None:
        return _scale_value(labelled)
    candidates = [_scale_value(match) for match in _ANY_ARCHITECTURAL_SCALE.finditer(text)]
    candidates = [item for item in candidates if item is not None]
    ratios = {round(float(item[1]), 8) for item in candidates}
    if len(ratios) == 1:
        return candidates[0]
    if re.search(r"\bAS\s+INDICATED\b", text, re.I):
        return "As indicated", None
    return None


def _parse_page_fields(tokens: list[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    text = _joined(tokens)
    upper = text.upper()
    mechanical = _MECHANICAL_TITLE.search(text)
    lube = _LUBE_TITLE.search(text)
    ventilation = re.search(r"\b(?:MECHANICAL|HVAC)\s+(?:FLOOR\s+)?PLAN\s+AREA\s+([A-Z0-9.]+)\b", text, re.I)
    hvac_level = _HVAC_LEVEL_TITLE.search(text)
    mechanical_sections = _MECHANICAL_SECTIONS_TITLE.search(text)

    if mechanical:
        level = mechanical.group(1).zfill(2)
        area = mechanical.group(2)
        role = "mechanical_piping_plan"
        discipline = "mechanical"
        title = mechanical.group(0)
    elif lube or all(term in upper for term in ("PIPING", "AREA", "LUBE", "SYSTEM")):
        hinted_level = next(
            (
                str(item.get("text"))
                for item in tokens
                if item.get("field_hint") == "level"
            ),
            None,
        )
        level = hinted_level or ("01" if _LUBE_LEVEL_ONE.search(text) else None)
        area = lube.group(1) if lube else None
        if area is None:
            area_match = re.search(r"\bAREA\s+([A-Z0-9.]+)\b", text, re.I)
            area = area_match.group(1) if area_match else None
        role = "lube_system_shop_plan"
        discipline = "mechanical"
        title = lube.group(0) if lube else "LUBE SYSTEM PIPING PLAN"
    elif ventilation:
        level, area = None, ventilation.group(1)
        role, discipline, title = "mechanical_plan", "mechanical", ventilation.group(0)
    elif hvac_level:
        level, area = hvac_level.group(1), None
        role, discipline, title = "mechanical_plan", "mechanical", hvac_level.group(0)
    elif mechanical_sections:
        level, area = None, None
        role, discipline, title = "mechanical_section_sheet", "mechanical", mechanical_sections.group(0)
    else:
        return "unknown", {}

    hinted_sheet = next(
        (
            _SHOP_SHEET.search(str(item.get("text") or ""))
            for item in tokens
            if item.get("field_hint") == "sheet_number"
            and _SHOP_SHEET.search(str(item.get("text") or ""))
        ),
        None,
    )
    sheet_match = hinted_sheet or _MECHANICAL_SHEET.search(text) or _SHOP_SHEET.search(text)
    if sheet_match is None and (hvac_level or mechanical_sections):
        candidates = []
        for token in tokens:
            match = _HVAC_SHEET.fullmatch(str(token.get("text") or ""))
            box = token.get("bbox_display") or []
            if match and len(box) == 4:
                candidates.append((float(box[3]) - float(box[1]), match))
        candidates.sort(key=lambda row: row[0], reverse=True)
        if (candidates and (len(candidates) == 1
                or candidates[0][0] >= 2.0 * candidates[1][0])):
            sheet_match = candidates[0][1]
    if sheet_match is None and ventilation:
        # A reference bubble is not the sheet number. Require a locally
        # adjacent complete title phrase and a unique prominent sheet token.
        title_words = ventilation.group(0).upper().split()
        title_boxes = []
        for i in range(len(tokens)-len(title_words)+1):
            group = tokens[i:i+len(title_words)]
            if [str(t['text']).upper() for t in group] == title_words:
                title_boxes.append(fitz.Rect(_rect_union(t['bbox_display'] for t in group)))
        candidates = []
        for token in tokens:
            match = re.fullmatch(r"M\d+\.\d+", str(token['text']))
            box = fitz.Rect(token['bbox_display'])
            if match and any((box + (-3*box.height, -3*box.height, 3*box.height, 3*box.height)).intersects(b)
                             for b in title_boxes):
                candidates.append(match)
        if len({m.group(0) for m in candidates}) == 1:
            sheet_match = candidates[0]
    sheet_number = sheet_match.group(0).upper() if sheet_match else None
    scale = _parse_scale(text)
    revision_match = _REVISION_AFTER.search(text) or _REVISION_BEFORE.search(text)
    revision = None
    if revision_match:
        revision = {
            "code": revision_match.group(1).upper(),
            "date": revision_match.group(2),
            "description": "ISSUED FOR REVIEW",
        }

    fields = {
        "title": _field("title", title, tokens, normalised_value=title.upper()),
        "sheet_number": _field("sheet_number", sheet_number, tokens),
        "level": _field("level", level, tokens),
        "area": _field("area", area, tokens),
        "discipline": {
            "field": "discipline",
            "state": "derived",
            "value": discipline,
            "normalised_value": discipline,
            "evidence_refs": _token_refs_for_value(tokens, title),
            "bbox_display": _field("discipline", title, tokens)["bbox_display"],
            "method": "title_semantics_v1",
        },
        "scale": _field("scale", scale[0] if scale else None, tokens),
        "revision": {
            "field": "revision",
            "state": "observed" if revision else "unknown",
            "value": revision,
            "normalised_value": revision,
            "evidence_refs": _token_refs_for_value(tokens, "ISSUED FOR REVIEW") if revision else [],
            "bbox_display": _field("revision", "ISSUED FOR REVIEW", tokens)["bbox_display"] if revision else None,
            "method": tokens[0]["method"] if revision and tokens else None,
        },
    }
    fields["scale"]["drawing_inches_per_paper_inch"] = (
        round(scale[1], 8) if scale and scale[1] is not None else None
    )
    return role, fields


def _deduplicate_label_tokens(
    tokens: Iterable[Mapping[str, Any]], *, coordinate_tolerance: float = 1.0
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in sorted(tokens, key=lambda row: (-float(row.get("confidence", 0)), str(row["id"]))):
        label = _normalise_token(item.get("text"))
        if not _GRID_LABEL.fullmatch(label):
            continue
        box = fitz.Rect(item["bbox_display"])
        center = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        duplicate = False
        for kept in output:
            if kept["label"] != label:
                continue
            if math.dist(center, kept["center_display"]) <= coordinate_tolerance:
                kept["evidence_refs"].append(str(item["id"]))
                duplicate = True
                break
        if not duplicate:
            output.append(
                {
                    "label": label,
                    "center_display": center,
                    "evidence_refs": [str(item["id"])],
                    "confidence": float(item.get("confidence", 0)),
                }
            )
    return output


def propose_grid_axes(
    tokens: Iterable[Mapping[str, Any]],
    *,
    page_ref: str,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    """Propose named axes only when opposing label observations agree."""

    labels = _deduplicate_label_tokens(tokens)
    by_label: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in labels:
        by_label[item["label"]].append(item)
    cross_tolerance = max(2.0, 0.0025 * max(page_width, page_height))
    proposals: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        candidates: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                lx, ly = left["center_display"]
                rx, ry = right["center_display"]
                if abs(lx - rx) <= cross_tolerance and abs(ly - ry) >= 0.25 * page_height:
                    candidates["vertical"].append((left, right))
                if abs(ly - ry) <= cross_tolerance and abs(lx - rx) >= 0.25 * page_width:
                    candidates["horizontal"].append((left, right))
        for orientation, pairs in sorted(candidates.items()):
            if not pairs:
                continue
            coordinate_values = [
                0.5 * (pair[0]["center_display"][0] + pair[1]["center_display"][0])
                if orientation == "vertical"
                else 0.5 * (pair[0]["center_display"][1] + pair[1]["center_display"][1])
                for pair in pairs
            ]
            coordinate = sum(coordinate_values) / len(coordinate_values)
            residual = max(abs(value - coordinate) for value in coordinate_values)
            if residual > cross_tolerance:
                continue
            evidence_refs = sorted(
                {
                    ref
                    for pair in pairs
                    for endpoint in pair
                    for ref in endpoint["evidence_refs"]
                }
            )
            span_pairs = [
                sorted(
                    (
                        pair[0]["center_display"][1],
                        pair[1]["center_display"][1],
                    )
                    if orientation == "vertical"
                    else (
                        pair[0]["center_display"][0],
                        pair[1]["center_display"][0],
                    )
                )
                for pair in pairs
            ]
            label_span = [
                sum(pair[index] for pair in span_pairs) / len(span_pairs)
                for index in (0, 1)
            ]
            proposals.append(
                {
                    "record_type": "mep_grid_axis_proposal",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id(
                        "mep_grid_axis_proposal", page_ref, label, orientation, round(coordinate, 4)
                    ),
                    "page_ref": page_ref,
                    "label": label,
                    "orientation_display": orientation,
                    "coordinate_display": round(coordinate, 4),
                    "opposing_label_pair_count": len(pairs),
                    "opposing_label_span_display": [round(value, 4) for value in label_span],
                    "max_coordinate_residual_display": round(residual, 4),
                    "state": "observed",
                    "method": "opposing_equal_label_alignment_v1",
                    "evidence_refs": evidence_refs,
                    "physical_axis_identity_established": False,
                    "quantity_eligible": False,
                }
            )
    _adjudicate_opposing_axis_groups(proposals, cross_tolerance)
    orientation_counts = {
        orientation: sum(
            item["orientation_display"] == orientation and item["state"] == "observed"
            for item in proposals
        )
        for orientation in ("vertical", "horizontal")
    }
    sequence_proposals = (
        _propose_collinear_grid_sequences(
            labels,
            page_ref=page_ref,
            page_width=page_width,
            page_height=page_height,
        )
        if min(orientation_counts.values()) < 2
        else []
    )
    occupied = {
        (item["label"], item["orientation_display"])
        for item in proposals
    }
    proposals.extend(
        item
        for item in sequence_proposals
        if (item["label"], item["orientation_display"]) not in occupied
    )
    return sorted(
        proposals,
        key=lambda item: (
            item["orientation_display"],
            item["coordinate_display"],
            item["label"],
        ),
    )


def _adjudicate_opposing_axis_groups(
    proposals: list[dict[str, Any]], tolerance: float
) -> None:
    """Require two named axes to agree on the same opposing label bands."""

    for orientation in ("vertical", "horizontal"):
        rows = [item for item in proposals if item["orientation_display"] == orientation]
        for item in rows:
            span = item["opposing_label_span_display"]
            peers = [
                other
                for other in rows
                if other["label"] != item["label"]
                and max(
                    abs(float(span[index]) - float(other["opposing_label_span_display"][index]))
                    for index in (0, 1)
                )
                <= tolerance
            ]
            if peers:
                item["state"] = "observed"
                item["axis_group_evidence_refs"] = sorted(
                    {str(item["id"]), *(str(peer["id"]) for peer in peers)}
                )
            else:
                item["state"] = "candidate"
                item["axis_group_evidence_refs"] = []


def _propose_collinear_grid_sequences(
    labels: list[dict[str, Any]],
    *,
    page_ref: str,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    """Recover one-sided bubble rows while retaining their sequence context."""

    proposals: list[dict[str, Any]] = []
    for family, orientation, fixed_index, varying_index, minimum_span in (
        ("letters", "vertical", 1, 0, 0.25 * page_width),
        ("numbers", "horizontal", 0, 1, 0.25 * page_height),
    ):
        family_rows = [
            item
            for item in labels
            if (
                bool(re.fullmatch(r"[A-Z]{1,2}(?:\.\d+)?", item["label"]))
                if family == "letters"
                else bool(re.fullmatch(r"[1-9]\d?(?:\.\d+)?", item["label"]))
            )
        ]
        tolerance = max(4.0, 0.004 * max(page_width, page_height))
        groups: list[list[dict[str, Any]]] = []
        for item in sorted(family_rows, key=lambda row: row["center_display"][fixed_index]):
            matching = [
                group
                for group in groups
                if abs(
                    item["center_display"][fixed_index]
                    - sum(row["center_display"][fixed_index] for row in group) / len(group)
                )
                <= tolerance
            ]
            if matching:
                matching[0].append(item)
            else:
                groups.append([item])
        for group in groups:
            unique: dict[str, dict[str, Any]] = {}
            for item in sorted(group, key=lambda row: -row["confidence"]):
                unique.setdefault(item["label"], item)
            rows = list(unique.values())
            coordinates = [item["center_display"][varying_index] for item in rows]
            if len(rows) < 4 or max(coordinates) - min(coordinates) < minimum_span:
                continue
            context_refs = sorted(
                {ref for item in rows for ref in item["evidence_refs"]}
            )
            for item in rows:
                coordinate = item["center_display"][varying_index]
                proposals.append(
                    {
                        "record_type": "mep_grid_axis_proposal",
                        "record_version": SCHEMA_VERSION,
                        "id": _stable_id(
                            "mep_grid_axis_proposal",
                            page_ref,
                            item["label"],
                            orientation,
                            round(coordinate, 4),
                            "sequence",
                        ),
                        "page_ref": page_ref,
                        "label": item["label"],
                        "orientation_display": orientation,
                        "coordinate_display": round(coordinate, 4),
                        "opposing_label_pair_count": 0,
                        "sequence_distinct_label_count": len(rows),
                        "max_coordinate_residual_display": round(
                            max(
                                abs(
                                    row["center_display"][fixed_index]
                                    - sum(r["center_display"][fixed_index] for r in rows) / len(rows)
                                )
                                for row in rows
                            ),
                            4,
                        ),
                        "state": "candidate",
                        "method": "collinear_distinct_grid_label_sequence_v1",
                        "evidence_refs": sorted(set(item["evidence_refs"]) | set(context_refs)),
                        "physical_axis_identity_established": False,
                        "quantity_eligible": False,
                    }
                )
    return proposals


def _north_proposals(tokens: list[Mapping[str, Any]], page_ref: str) -> list[dict[str, Any]]:
    text = _joined(tokens).upper()
    proposals = []
    for label in ("PLAN NORTH", "TRUE NORTH"):
        if label not in text:
            continue
        refs = _token_refs_for_value(tokens, label)
        proposals.append(
            {
                "record_type": "mep_north_proposal",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_north_proposal", page_ref, label, refs),
                "page_ref": page_ref,
                "label": label,
                "state": "observed",
                "direction_display": None,
                "reason": "north label observed; arrow direction not uniquely traced",
                "evidence_refs": refs,
                "physical_orientation_established": False,
                "quantity_eligible": False,
            }
        )
    return proposals


def _divider_record(
    tokens: list[Mapping[str, Any]], quality: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    metrics = quality.get("metrics", {})
    if (
        0 < len(tokens) <= 8
        and int(metrics.get("native_path_item_count", 0)) == 0
        and int(metrics.get("embedded_image_count", 0)) == 0
    ):
        label = _joined(tokens)
        return "divider", {
            "title": _field("title", label, tokens, normalised_value=label.upper()),
            "sheet_number": _field("sheet_number", None, tokens),
            "level": _field("level", None, tokens),
            "area": _field("area", None, tokens),
            "discipline": _field("discipline", None, tokens),
            "scale": _field("scale", None, tokens),
            "revision": _field("revision", None, tokens),
        }
    return "unknown", {}


def _retained_text_observations(
    tokens: list[dict[str, Any]],
    fields: Mapping[str, Any],
    axes: Iterable[Mapping[str, Any]],
    north: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    refs = {
        str(ref)
        for record in [*fields.values(), *axes, *north]
        for ref in record.get("evidence_refs", [])
    }
    return [
        {
            "record_type": "mep_text_observation",
            "record_version": SCHEMA_VERSION,
            **{key: value for key, value in item.items() if key != "reading_order"},
        }
        for item in tokens
        if str(item["id"]) in refs
    ]


def build_sheet_page_record(
    *,
    page_ref: str,
    page_number: int,
    page_width: float,
    page_height: float,
    native_tokens: list[dict[str, Any]],
    quality: Mapping[str, Any],
    ocr_tokens: list[dict[str, Any]] | None = None,
    ocr_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one M1 page record from reusable observations."""

    role, fields = _parse_page_fields(native_tokens)
    divider_role, divider_fields = _divider_record(native_tokens, quality)
    if role == "unknown" and divider_role == "divider":
        role, fields = divider_role, divider_fields

    registry_route = str(quality.get("route") or "unknown")
    route_reason = str(quality.get("reason") or "")
    tokens = native_tokens
    provenance = None
    if role == "unknown" and ocr_tokens:
        ocr_role, ocr_fields = _parse_page_fields(ocr_tokens)
        if ocr_role != "unknown":
            role, fields = ocr_role, ocr_fields
            tokens = ocr_tokens
            registry_route = "hybrid_text_ocr"
            route_reason = "required MEP registry title fields were absent from native text"
            provenance = dict(ocr_provenance or {})

    if not fields:
        fields = {
            name: _field(name, None, tokens)
            for name in ("title", "sheet_number", "level", "area", "discipline", "scale", "revision")
        }
    axes = (
        []
        if role == "divider"
        else propose_grid_axes(
            tokens,
            page_ref=page_ref,
            page_width=page_width,
            page_height=page_height,
        )
    )
    north = [] if role == "divider" else _north_proposals(tokens, page_ref)
    return {
        "record_type": "mep_sheet_page_record",
        "record_version": SCHEMA_VERSION,
        "id": _stable_id("mep_sheet_page_record", page_ref),
        "page_ref": page_ref,
        "page_number": int(page_number),
        "page_size_display": [round(page_width, 4), round(page_height, 4)],
        "role": role,
        "role_state": "derived" if role != "unknown" else "unknown",
        "fields": fields,
        "quality_route": {
            "base_structural_quality_route": dict(quality),
            "mep_registry_route": registry_route,
            "reason": route_reason,
            "ocr_provenance": provenance,
        },
        "grid_axes": axes,
        "north_proposals": north,
        "text_observations": _retained_text_observations(tokens, fields, axes, north),
        "physical_continuation_established": False,
        "route_identity_established": False,
        "quantity_eligible": False,
    }


def _linear_fit(
    pairs: list[tuple[float, float, str]],
    *,
    expected_slope: float | None = None,
    residual_tolerance: float = 2.0,
) -> dict[str, Any] | None:
    if len({label for _, _, label in pairs}) < 2:
        return None
    if expected_slope is not None:
        offsets = [
            (target - expected_slope * source, label)
            for source, target, label in pairs
        ]
        candidate_clusters = []
        for center, _ in offsets:
            rows = [
                (source, target, label)
                for source, target, label in pairs
                if abs((target - expected_slope * source) - center) <= residual_tolerance
            ]
            labels = {label for _, _, label in rows}
            if len(labels) >= 2:
                candidate_clusters.append(rows)
        if not candidate_clusters:
            return None
        maximum = max(len({label for _, _, label in rows}) for rows in candidate_clusters)
        best = {
            tuple(sorted(label for _, _, label in rows)): rows
            for rows in candidate_clusters
            if len({label for _, _, label in rows}) == maximum
        }
        if len(best) != 1:
            return None
        pairs = next(iter(best.values()))
    mean_source = sum(source for source, _, _ in pairs) / len(pairs)
    mean_target = sum(target for _, target, _ in pairs) / len(pairs)
    denominator = sum((source - mean_source) ** 2 for source, _, _ in pairs)
    if denominator <= 1e-9:
        return None
    slope = (
        expected_slope
        if expected_slope is not None
        else sum(
            (source - mean_source) * (target - mean_target)
            for source, target, _ in pairs
        ) / denominator
    )
    intercept = mean_target - slope * mean_source
    residuals = [abs(slope * source + intercept - target) for source, target, _ in pairs]
    return {
        "slope": slope,
        "intercept": intercept,
        "maximum_residual": max(residuals),
        "labels": sorted({label for _, _, label in pairs}),
        "outlier_labels": sorted(
            {label for _, label in offsets} - {label for _, _, label in pairs}
        ) if expected_slope is not None else [],
    }


def _axis_map(page: Mapping[str, Any], orientation: str) -> dict[str, Mapping[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for axis in page.get("grid_axes", []):
        if (
            axis.get("orientation_display") == orientation
            and axis.get("state") == "observed"
            and axis.get("method") == "opposing_equal_label_alignment_v1"
        ):
            grouped[str(axis.get("label"))].append(axis)
    unique = {label: rows[0] for label, rows in grouped.items() if len(rows) == 1}
    families = {
        "alphabetic": [label for label in unique if label[:1].isalpha()],
        "numeric": [label for label in unique if label[:1].isdigit()],
    }
    dominant = max(families, key=lambda name: len(families[name]))
    other = "numeric" if dominant == "alphabetic" else "alphabetic"
    if len(families[dominant]) >= 2 * max(1, len(families[other])):
        unique = {label: unique[label] for label in families[dominant]}
    return unique


def propose_adjoining_sheet_transform(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a bounded similarity proposal when two grid directions overlap."""

    if source.get("role") in {"divider", "unknown"} or target.get("role") in {"divider", "unknown"}:
        return None
    source_vertical = _axis_map(source, "vertical")
    target_vertical = _axis_map(target, "vertical")
    source_horizontal = _axis_map(source, "horizontal")
    target_horizontal = _axis_map(target, "horizontal")
    vertical_labels = sorted(set(source_vertical) & set(target_vertical))
    horizontal_labels = sorted(set(source_horizontal) & set(target_horizontal))
    if not vertical_labels or not horizontal_labels:
        return None
    x_pairs = [
        (
            float(source_vertical[label]["coordinate_display"]),
            float(target_vertical[label]["coordinate_display"]),
            label,
        )
        for label in vertical_labels
    ]
    y_pairs = [
        (
            float(source_horizontal[label]["coordinate_display"]),
            float(target_horizontal[label]["coordinate_display"]),
            label,
        )
        for label in horizontal_labels
    ]
    source_scale = source.get("fields", {}).get("scale", {}).get("drawing_inches_per_paper_inch")
    target_scale = target.get("fields", {}).get("scale", {}).get("drawing_inches_per_paper_inch")
    expected_slope = (
        float(source_scale) / float(target_scale)
        if source_scale and target_scale
        else None
    )
    tolerance = max(2.0, 0.0025 * max(*source["page_size_display"], *target["page_size_display"]))
    x_fit = _linear_fit(
        x_pairs,
        expected_slope=expected_slope,
        residual_tolerance=tolerance,
    )
    y_fit = _linear_fit(
        y_pairs,
        expected_slope=expected_slope,
        residual_tolerance=tolerance,
    )
    if expected_slope is not None:
        x_fit = x_fit or _linear_fit(x_pairs)
        y_fit = y_fit or _linear_fit(y_pairs)
    state = "accepted"
    reasons: list[str] = []
    if x_fit is None or y_fit is None:
        state = "abstained"
        reasons.append("no_unique_redundant_grid_offset_consensus")
    if x_fit is not None and y_fit is not None:
        if x_fit["slope"] <= 0 or y_fit["slope"] <= 0:
            state = "abstained"
            reasons.append("axis_sign_or_rotation_unresolved")
        if abs(x_fit["slope"] - y_fit["slope"]) > 0.01 * max(abs(x_fit["slope"]), abs(y_fit["slope"]), 1.0):
            state = "rejected"
            reasons.append("non_uniform_grid_scale")
        if max(x_fit["maximum_residual"], y_fit["maximum_residual"]) > tolerance:
            state = "rejected"
            reasons.append("grid_reprojection_residual_exceeds_tolerance")
        if expected_slope is not None and abs(0.5 * (x_fit["slope"] + y_fit["slope"]) - expected_slope) > 0.02 * max(expected_slope, 1.0):
            state = "rejected"
            reasons.append("title_scale_conflicts_with_grid_scale")

    axis_evidence = sorted(
        {
            str(axis["id"])
            for label in [*vertical_labels, *horizontal_labels]
            for axis in (
                source_vertical.get(label),
                target_vertical.get(label),
                source_horizontal.get(label),
                target_horizontal.get(label),
            )
            if axis is not None
        }
    )
    matrix = None
    if state == "accepted" and x_fit is not None and y_fit is not None:
        scale = 0.5 * (x_fit["slope"] + y_fit["slope"])
        matrix = [
            round(scale, 10),
            0.0,
            0.0,
            round(scale, 10),
            round(x_fit["intercept"], 6),
            round(y_fit["intercept"], 6),
        ]
    source_ref = str(source["page_ref"])
    target_ref = str(target["page_ref"])
    return {
        "record_type": "mep_adjoining_sheet_transform",
        "record_version": SCHEMA_VERSION,
        "id": _stable_id("mep_adjoining_sheet_transform", source_ref, target_ref, axis_evidence),
        "source_page_ref": source_ref,
        "target_page_ref": target_ref,
        "state": state,
        "method": "redundant_named_grid_similarity_v1",
        "matrix_source_display_to_target_display": matrix,
        "shared_vertical_labels": vertical_labels,
        "shared_horizontal_labels": horizontal_labels,
        "grid_fit": {"x": x_fit, "y": y_fit},
        "explicit_scale_expected_slope": expected_slope,
        "maximum_residual_tolerance_display": round(tolerance, 4),
        "reasons": reasons or ["redundant_grid_and_scale_agreement"],
        "evidence_refs": axis_evidence,
        "physical_continuation_established": False,
        "route_identity_established": False,
        "quantity_eligible": False,
    }


def _packages(pages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for page in sorted(pages, key=lambda item: int(item["page_number"])):
        if page.get("role") == "divider":
            if current is not None:
                packages.append(current)
            label_field = page.get("fields", {}).get("title", {})
            label = str(label_field.get("value") or "")
            current = {
                "record_type": "mep_package_record",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_package_record", page["page_ref"], label),
                "label": label,
                "divider_page_ref": str(page["page_ref"]),
                "member_page_refs": [],
                "state": "derived",
                "method": "pdf_sequence_after_sparse_divider_v1",
                "evidence_refs": list(label_field.get("evidence_refs", [])),
                "physical_system_identity_established": False,
                "quantity_eligible": False,
            }
        elif current is not None:
            current["member_page_refs"].append(str(page["page_ref"]))
    if current is not None:
        packages.append(current)
    return packages


def build_mep_sheet_registry(
    *,
    document: Mapping[str, Any],
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    transforms = []
    eligible = [page for page in pages if page.get("role") not in {"divider", "unknown"}]
    for source_index, source in enumerate(eligible):
        for target in eligible[source_index + 1 :]:
            proposal = propose_adjoining_sheet_transform(source, target)
            if proposal is not None:
                transforms.append(proposal)
    packages = _packages(pages)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": dict(document),
        "pages": sorted(pages, key=lambda item: int(item["page_number"])),
        "packages": packages,
        "adjoining_sheet_transforms": sorted(
            transforms,
            key=lambda item: (item["source_page_ref"], item["target_page_ref"]),
        ),
        "summary": {
            "page_count": len(pages),
            "package_count": len(packages),
            "role_counts": {
                role: sum(page.get("role") == role for page in pages)
                for role in sorted({str(page.get("role")) for page in pages})
            },
            "quality_route_counts": {
                route: sum(
                    page.get("quality_route", {}).get("mep_registry_route") == route
                    for page in pages
                )
                for route in sorted(
                    {
                        str(page.get("quality_route", {}).get("mep_registry_route"))
                        for page in pages
                    }
                )
            },
            "grid_axis_proposal_count": sum(len(page.get("grid_axes", [])) for page in pages),
            "accepted_adjoining_sheet_transform_count": sum(
                item.get("state") == "accepted" for item in transforms
            ),
            "abstained_or_rejected_transform_count": sum(
                item.get("state") != "accepted" for item in transforms
            ),
        },
        "exchange_contract": {
            "page_and_title_observations_are_evidence_backed": True,
            "structural_view_or_dimension_thresholds_changed": False,
            "filename_or_sheet_dispatch_used": False,
            "page_order_or_area_label_establishes_continuation": False,
            "accepted_transform_requires_redundant_grid_or_datum_agreement": True,
            "route_identity_or_physical_continuation_established": False,
            "quantity_eligible": False,
        },
    }


def extract_mep_sheet_registry(
    pdf_path: Path | str, *, ocr_unknown_pages: bool = True, page_numbers: list[int] | None = None
) -> dict[str, Any]:
    source_path = Path(pdf_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = _file_sha256(source_path)
    document_key = f"pdf-sha256:{source_sha256}"
    pages: list[dict[str, Any]] = []
    with fitz.open(source_path) as pdf:
        selected = set(range(1, len(pdf) + 1)) if page_numbers is None else set(page_numbers)
        if not selected or any(type(n) is not int or not 1 <= n <= len(pdf) for n in selected):
            raise ValueError("selected page is outside the source PDF")
        for page in pdf:
            page_number = int(page.number) + 1
            page_ref = _stable_id("mep_page_observation", document_key, page_number, page.xref)
            processed = page_number in selected
            quality = (_assess_page_quality_bounded(page) if processed else
                       {"route": "not_processed", "reason": "outside requested page scope"})
            native = _native_tokens(page, page_ref) if processed else []
            native_role, _ = _parse_page_fields(native)
            divider_role, _ = _divider_record(native, quality)
            needs_ocr = processed and ocr_unknown_pages and (
                quality.get("route") != "native" or (
                    native_role == "unknown" and divider_role != "divider"
                )
            )
            ocr, ocr_provenance = ([], None)
            if needs_ocr:
                ocr, ocr_provenance = _ocr_tokens(page, page_ref)
            pages.append(
                build_sheet_page_record(
                    page_ref=page_ref,
                    page_number=page_number,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    native_tokens=native,
                    quality=quality,
                    ocr_tokens=ocr,
                    ocr_provenance=ocr_provenance,
                )
            )
    result = build_mep_sheet_registry(
        document={
            "document_key": document_key,
            "source_pdf_sha256": source_sha256,
            "source_bytes": source_path.stat().st_size,
            "page_count": len(pages),
        },
        pages=pages,
    )
    if page_numbers is not None:
        result["processing_scope"] = {"source_page_numbers": sorted(selected),
            "unselected_pages": "metadata_only_not_processed"}
    return result


def validate_mep_sheet_registry(
    payload: Mapping[str, Any], source_pdf: Path | str | None = None
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    pages = list(payload.get("pages", []))
    if int(payload.get("document", {}).get("page_count", -1)) != len(pages):
        errors.append("document page_count does not match pages")
    page_refs = [str(page.get("page_ref")) for page in pages]
    if len(set(page_refs)) != len(page_refs):
        errors.append("page_ref values are not unique")
    forbidden = {
        "accepted_route_ref",
        "confirmed_clash",
        "installed_length",
        "quantity",
        "physical_continuation_ref",
    }
    for page in pages:
        if page.get("record_type") != "mep_sheet_page_record":
            errors.append(f"{page.get('id')}: wrong page record_type")
        if page.get("quantity_eligible") is not False:
            errors.append(f"{page.get('id')}: page became quantity eligible")
        if forbidden.intersection(page):
            errors.append(f"{page.get('id')}: forbidden engineering authority field")
        observation_refs = {
            str(item.get("id")) for item in page.get("text_observations", [])
        }
        for field in page.get("fields", {}).values():
            for ref in field.get("evidence_refs", []):
                if str(ref) not in observation_refs:
                    errors.append(f"{page.get('id')}: missing field evidence {ref}")
        for axis in page.get("grid_axes", []):
            if axis.get("quantity_eligible") is not False:
                errors.append(f"{axis.get('id')}: axis became quantity eligible")
            if len(axis.get("evidence_refs", [])) < 2:
                errors.append(f"{axis.get('id')}: axis lacks opposing label evidence")
            if (
                axis.get("method") == "opposing_equal_label_alignment_v1"
                and axis.get("state") == "observed"
                and len(set(axis.get("axis_group_evidence_refs", []))) < 2
            ):
                errors.append(f"{axis.get('id')}: observed axis lacks named-axis group evidence")

    known_pages = set(page_refs)
    for package in payload.get("packages", []):
        members = [package.get("divider_page_ref"), *package.get("member_page_refs", [])]
        if any(str(ref) not in known_pages for ref in members):
            errors.append(f"{package.get('id')}: package references unknown page")
    known_axes = {
        str(axis.get("id"))
        for page in pages
        for axis in page.get("grid_axes", [])
    }
    for transform in payload.get("adjoining_sheet_transforms", []):
        if transform.get("quantity_eligible") is not False:
            errors.append(f"{transform.get('id')}: transform became quantity eligible")
        if transform.get("physical_continuation_established") is not False:
            errors.append(f"{transform.get('id')}: transform established continuation")
        if any(str(ref) not in known_axes for ref in transform.get("evidence_refs", [])):
            errors.append(f"{transform.get('id')}: transform references unknown axis")
        if transform.get("state") == "accepted":
            if transform.get("matrix_source_display_to_target_display") is None:
                errors.append(f"{transform.get('id')}: accepted transform lacks matrix")
            if len(set(transform.get("shared_vertical_labels", []))) < 2:
                errors.append(f"{transform.get('id')}: accepted transform lacks vertical redundancy")
            if len(set(transform.get("shared_horizontal_labels", []))) < 2:
                errors.append(f"{transform.get('id')}: accepted transform lacks horizontal redundancy")
            fits = transform.get("grid_fit", {})
            x_fit, y_fit = fits.get("x"), fits.get("y")
            tolerance = float(transform.get("maximum_residual_tolerance_display", -1))
            if not isinstance(x_fit, Mapping) or not isinstance(y_fit, Mapping):
                errors.append(f"{transform.get('id')}: accepted transform lacks grid fits")
            else:
                if len(set(x_fit.get("labels", []))) < 2 or len(set(y_fit.get("labels", []))) < 2:
                    errors.append(f"{transform.get('id')}: accepted transform fit lacks inlier redundancy")
                if max(float(x_fit.get("maximum_residual", math.inf)), float(y_fit.get("maximum_residual", math.inf))) > tolerance:
                    errors.append(f"{transform.get('id')}: accepted transform exceeds residual tolerance")
                if abs(float(x_fit.get("slope", math.inf)) - float(y_fit.get("slope", -math.inf))) > 0.01 * max(abs(float(x_fit.get("slope", 0))), abs(float(y_fit.get("slope", 0))), 1.0):
                    errors.append(f"{transform.get('id')}: accepted transform is not uniform scale")
                expected = transform.get("explicit_scale_expected_slope")
                if expected is not None and abs(0.5 * (float(x_fit["slope"]) + float(y_fit["slope"])) - float(expected)) > 0.02 * max(float(expected), 1.0):
                    errors.append(f"{transform.get('id')}: accepted transform conflicts with explicit scale")
    if source_pdf is not None:
        path = Path(source_pdf).resolve()
        if _file_sha256(path) != payload.get("document", {}).get("source_pdf_sha256"):
            errors.append("source PDF sha256 mismatch")
        if path.stat().st_size != payload.get("document", {}).get("source_bytes"):
            errors.append("source PDF byte count mismatch")
    return errors
