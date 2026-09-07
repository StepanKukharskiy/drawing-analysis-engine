"""Quality-gated item text OCR, with immutable pixels/words and no applicability.

Raster coverage measures attempted image area, not item detection recall.
Native text and overlapping OCR reads remain alternatives, never merged facts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from time import perf_counter

import fitz
from PIL import Image
import pytesseract

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import (
    _normalise_text, _stable_id, build_mep_terminology_proposals,
    interpret_mep_text, validate_mep_terminology_proposals,
)
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations

LAYER = "mep_item_ocr_observations"
SCHEMA_VERSION = "0.1.0"


def page_ocr_selection(page_record, *, minimum_native_chars_per_square_inch=.25):
    quality = page_record["quality_route"]
    base = quality["base_structural_quality_route"]
    metrics = base["metrics"]
    width, height = page_record["page_size_display"]
    density = metrics["native_text_chars"] / max(width * height / 72 ** 2, 1)
    graphic_support = metrics["native_path_item_count"] >= 8 or metrics["embedded_image_area_ratio"] >= .35
    selected = graphic_support and (base["route"] != "native"
        or quality["mep_registry_route"] != "native"
        or density < minimum_native_chars_per_square_inch)
    return {
        "selected": selected, "native_chars_per_square_inch": round(density, 8),
        "minimum_native_chars_per_square_inch": minimum_native_chars_per_square_inch,
        "graphic_support_available": graphic_support,
        "reason": "text_poor_graphic_page" if selected else
                  "no_graphic_support_for_item_ocr" if not graphic_support else "native_text_density_sufficient_for_this_gate",
        "m1_quality_route": deepcopy(quality), "item_absence_established": False,
    }


def _tile_plan(rect, scale, tile_pixels):
    # Disjoint cores partition the page; overlap only aids boundary recognition.
    step = tile_pixels / scale
    overlap = 64 / scale
    for y in range(math.ceil(rect.height / step)):
        for x in range(math.ceil(rect.width / step)):
            core = fitz.Rect(x * step, y * step, min((x + 1) * step, rect.width), min((y + 1) * step, rect.height))
            yield core, (core + (-overlap, -overlap, overlap, overlap)) & rect


def _run_ocr(image, timeout):
    return pytesseract.image_to_data(image, lang="eng", config="--psm 11",
        output_type=pytesseract.Output.DICT, timeout=timeout)


def _line_observations(crop, document, minimum_confidence):
    groups = defaultdict(list)
    for index, word in enumerate(crop.get("raw_tsv_rows", [])):
        if str(word.get("text", "")).strip():
            groups[tuple(word.get(key) for key in ("block_num", "par_num", "line_num"))].append((index, word))
    observations = []
    for words in groups.values():
        text = " ".join(str(word["text"]) for _, word in words)
        indices = [index for index, _ in words]
        identifier = _stable_id("mep_item_ocr_line", document["source_pdf_sha256"], crop["id"], indices, text)
        reasons, boxes, confidences = [], [], []
        for _, word in words:
            try:
                confidence = float(word["conf"]) / 100
                left, top, width, height = (float(word[key]) for key in ("left", "top", "width", "height"))
                if (not all(math.isfinite(v) for v in [confidence, left, top, width, height])
                        or not 0 <= confidence <= 1 or width <= 0 or height <= 0
                        or left < 0 or top < 0 or left + width > crop["pixel_size"][0]
                        or top + height > crop["pixel_size"][1]):
                    raise ValueError("invalid OCR word coordinates/confidence")
                boxes.append(fitz.Rect(left, top, left + width, top + height))
                confidences.append(confidence)
            except (KeyError, ValueError, TypeError):
                reasons.append("invalid_raw_word_geometry_or_confidence")
        bbox_pixel = None
        if len(boxes) == len(words):
            bbox_pixel = fitz.Rect(boxes[0])
            for box in boxes[1:]:
                bbox_pixel |= box
        confidence = min(confidences) if len(confidences) == len(words) else None
        if confidence is None or confidence < minimum_confidence:
            reasons.append("ocr_confidence_below_observation_handoff_gate")
        bbox_display = bbox_pixel * fitz.Matrix(crop["pixel_to_display_matrix"]) if bbox_pixel else None
        row = {
            "record_type": "mep_item_ocr_line", "record_version": SCHEMA_VERSION,
            "id": identifier, "page_ref": crop["page_ref"], "page_number": crop["page_number"],
            "source_pdf_sha256": document["source_pdf_sha256"],
            "raw_text": text, "text": text.strip(), "raw_word_indices": indices,
            "source_crop_ref": crop["id"], "bbox_pixel": list(bbox_pixel) if bbox_pixel else None,
            "bbox_display": list(bbox_display) if bbox_display else None,
            "bbox_pdf": list(bbox_display * fitz.Matrix(crop["display_to_pdf_matrix"])) if bbox_display else None,
            "confidence": confidence, "confidence_basis": "minimum_raw_word_confidence",
            "method": "tesseract_tiled_item_text_ocr", "engine_version": crop["engine_version"],
            "evidence_channels": ["raster_ocr_text"], "region_role": "unknown",
            "interpretation_scope_ref": identifier, "applicability_state": "unknown",
            "immutable_source_observation": True, "epistemic_state": "observed", "quantity_eligible": False,
        }
        try:
            row["text_interpretation"] = interpret_mep_text(row)
            one = build_mep_terminology_proposals(document=document, observations=[row])
            reasons.extend(validate_mep_terminology_proposals(one))
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            row["text_interpretation"] = None
            reasons.append(f"{type(error).__name__}: {error}")
        row["handoff_errors"] = sorted(set(reasons))
        row["handoff_state"] = "quarantined" if reasons else "proposal_only"
        observations.append(row)
    return observations


def _overlap_alternatives(row, candidates):
    if row["bbox_display"] is None:
        return []
    box = fitz.Rect(row["bbox_display"])
    alternatives = []
    for candidate in candidates:
        if candidate["id"] == row["id"] or candidate["page_ref"] != row["page_ref"] or not candidate.get("bbox_display"):
            continue
        other = fitz.Rect(candidate["bbox_display"])
        area = min(box.get_area(), other.get_area())
        overlap = (box & other).get_area() / area if area > 0 else 0
        if overlap >= .35:
            alternatives.append({
                "observation_ref": candidate["id"], "text": candidate["text"],
                "bbox_display": deepcopy(candidate["bbox_display"]),
                "overlap_fraction_of_smaller_box": round(overlap, 6),
                "normalized_text_agrees": _normalise_text(row["text"]) == _normalise_text(candidate["text"]),
                "state": "unresolved_overlap_alternative", "automatic_merge_established": False,
            })
    return sorted(alternatives, key=lambda value: value["observation_ref"])


def extract_mep_item_ocr_observations(*, pdf_path, sheet_registry, native_text_observations=None,
        render_scale=3.0, tile_pixels=3072, minimum_confidence=.45, max_crops=None,
        minimum_native_chars_per_square_inch=.25, timeout_seconds=45):
    errors = validate_mep_sheet_registry(sheet_registry)
    if errors:
        raise ValueError("invalid M1 registry: " + "; ".join(errors))
    if (not math.isfinite(render_scale) or render_scale <= 0 or not isinstance(tile_pixels, int) or tile_pixels < 256
            or not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1
            or not math.isfinite(minimum_native_chars_per_square_inch) or minimum_native_chars_per_square_inch < 0
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or max_crops is not None and (not isinstance(max_crops, int) or max_crops < 0)):
        raise ValueError("invalid OCR quality/budget configuration")
    source = Path(pdf_path).resolve()
    document = sheet_registry["document"]
    if _file_sha256(source) != document["source_pdf_sha256"]:
        raise ValueError("source PDF does not match M1")
    native = (extract_mep_text_observations(pdf_path=source, sheet_registry=sheet_registry)
              if native_text_observations is None else native_text_observations)
    if (native.get("m1_payload_sha256") != _sha256(sheet_registry)
            or native["document"]["source_pdf_sha256"] != document["source_pdf_sha256"]):
        raise ValueError("native text does not match M1/source")
    engine_version = str(pytesseract.get_tesseract_version()).splitlines()[0]
    crops, observations, pages = [], [], []
    attempted = 0
    with fitz.open(source) as pdf:
        scopes = sorted(sheet_registry["pages"], key=lambda row: row["page_number"])
        if [scope["page_number"] for scope in scopes] != list(range(1, len(pdf) + 1)):
            raise ValueError("M1 page registry is incomplete")
        for scope in scopes:
            page = pdf[scope["page_number"] - 1]
            if any(abs(a - b) > .0001 for a, b in zip(page.rect.br, scope["page_size_display"])):
                raise ValueError("M1 display frame differs from source")
            selection = page_ocr_selection(scope, minimum_native_chars_per_square_inch=minimum_native_chars_per_square_inch)
            page_crops, page_rows = [], []
            display_list = None
            if selection["selected"]:
                for core, clip in _tile_plan(page.rect, render_scale, tile_pixels):
                    crop = {
                        "id": _stable_id("mep_item_ocr_crop", document["source_pdf_sha256"], scope["page_ref"],
                                         list(clip), render_scale, engine_version, "eng", "--psm 11"),
                        "page_ref": scope["page_ref"], "page_number": scope["page_number"],
                        "core_display": list(core), "crop_display": list(clip),
                        "render_scale": render_scale, "engine_version": engine_version,
                        "engine": "tesseract", "language": "eng", "config": "--psm 11",
                        "orientation_degrees": 0, "display_to_pdf_matrix": list(page.derotation_matrix),
                        "annotation_appearances_rendered": False, "state": "not_run_budget",
                        "raw_tsv_rows": [], "observation_refs": [], "quantity_eligible": False,
                    }
                    if max_crops is None or attempted < max_crops:
                        attempted += 1
                        started = perf_counter()
                        try:
                            if display_list is None:
                                display_list = page.get_displaylist(annots=False)
                            pixels = display_list.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), clip=clip,
                                                           colorspace=fitz.csRGB, alpha=False)
                            crop.update({"pixel_size": [pixels.width, pixels.height],
                                "pixel_to_display_matrix": [1 / render_scale, 0, 0, 1 / render_scale,
                                                            pixels.x / render_scale, pixels.y / render_scale],
                                "pixel_sha256": hashlib.sha256(pixels.samples).hexdigest()})
                            data = _run_ocr(Image.frombytes("RGB", (pixels.width, pixels.height), pixels.samples), timeout_seconds)
                            lengths = {len(value) for value in data.values()}
                            if len(lengths) != 1:
                                raise ValueError("OCR TSV columns have inconsistent lengths")
                            crop["raw_tsv_rows"] = [{key: values[index] for key, values in data.items()}
                                                   for index in range(len(data.get("text", [])))]
                            crop["state"] = "completed"
                            rows = _line_observations(crop, document, minimum_confidence)
                            crop["observation_refs"] = [row["id"] for row in rows]
                            page_rows.extend(rows)
                        except (RuntimeError, ValueError, OSError) as error:
                            crop["state"] = "failed"
                            crop["error"] = f"{type(error).__name__}: {error}"
                        crop["elapsed_seconds"] = round(perf_counter() - started, 6)
                    page_crops.append(crop)
            for row in page_rows:
                row["native_overlap_alternatives"] = _overlap_alternatives(row, native["observations"])
                row["ocr_overlap_alternatives"] = _overlap_alternatives(row, page_rows)
            completed = [crop for crop in page_crops if crop["state"] == "completed"]
            processed_fraction = sum(fitz.Rect(crop["core_display"]).get_area() for crop in completed) / page.rect.get_area()
            pages.append({
                "page_ref": scope["page_ref"], "page_number": scope["page_number"],
                "page_size_display": list(page.rect.br), "selection": selection,
                "state": "not_selected_by_quality_gate" if not selection["selected"] else
                         "raster_scan_completed" if len(completed) == len(page_crops) else "partial_or_failed_raster_scan",
                "crop_refs": [crop["id"] for crop in page_crops], "observation_refs": [row["id"] for row in page_rows],
                "completed_crop_count": len(completed), "planned_crop_count": len(page_crops),
                "processed_raster_area_fraction": round(processed_fraction, 8) if selection["selected"] else None,
                "orientation_coverage": "horizontal_only", "item_inventory_complete": False,
                "item_recall_measured": False, "item_absence_established": False, "quantity_eligible": False,
            })
            crops.extend(page_crops)
            observations.extend(page_rows)
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": LAYER, "document": deepcopy(document),
        "m1_payload_sha256": _sha256(sheet_registry), "native_text_payload_sha256": _sha256(native),
        "configuration": {"render_scale": render_scale, "tile_pixels": tile_pixels, "minimum_confidence": minimum_confidence,
                          "max_crops": max_crops, "timeout_seconds": timeout_seconds,
                          "minimum_native_chars_per_square_inch": minimum_native_chars_per_square_inch},
        "pages": pages, "crops": crops, "observations": observations,
        "summary": {"page_count": len(pages), "selected_page_count": sum(page["selection"]["selected"] for page in pages),
                    "crop_state_counts": dict(sorted(Counter(crop["state"] for crop in crops).items())),
                    "observation_count": len(observations), "handoff_state_counts": dict(sorted(Counter(row["handoff_state"] for row in observations).items()))},
        "exchange_contract": {"observations_only": True, "native_text_replaced": False, "automatic_merge_established": False,
                              "item_applicability_established": False, "item_inventory_complete": False,
                              "item_absence_established": False, "quantity_eligible": False},
    }
    errors = validate_mep_item_ocr_observations(payload)
    if errors:
        raise ValueError("invalid item OCR observation contract: " + "; ".join(errors))
    return payload


def validate_mep_item_ocr_observations(payload):
    errors = []
    if payload.get("layer") != LAYER or payload.get("schema_version") != SCHEMA_VERSION:
        return ["invalid OCR layer/version"]
    pages, crops, rows = payload["pages"], payload["crops"], payload["observations"]
    if [page["page_number"] for page in pages] != list(range(1, payload["document"]["page_count"] + 1)):
        errors.append("page coverage incomplete")
    known_pages = {page["page_ref"]: page for page in pages}
    if len(known_pages) != len(pages):
        errors.append("duplicate page identities")
    for key in ("native_text_replaced", "automatic_merge_established", "item_applicability_established",
                "item_inventory_complete", "item_absence_established", "quantity_eligible"):
        if payload["exchange_contract"].get(key) is not False:
            errors.append("OCR contract acquired engineering authority")
    for records in (crops, rows):
        if len({row["id"] for row in records}) != len(records):
            errors.append("duplicate record IDs")
    for crop in crops:
        if crop["page_ref"] not in known_pages or crop["quantity_eligible"] is not False:
            errors.append("invalid crop ownership/authority")
        expected_id = _stable_id("mep_item_ocr_crop", payload["document"]["source_pdf_sha256"], crop["page_ref"],
                                crop["crop_display"], crop["render_scale"], crop["engine_version"], "eng", "--psm 11")
        if (crop["id"] != expected_id or crop["annotation_appearances_rendered"] is not False
                or crop["language"] != "eng" or crop["config"] != "--psm 11"
                or crop["orientation_degrees"] != 0):
            errors.append("crop source/method identity does not replay")
        if crop["state"] == "completed":
            scale = crop["render_scale"]
            clip = crop["crop_display"]
            # MuPDF applies tolerant float-to-pixel rounding; Python floor can
            # shift an exact tile boundary by one pixel (e.g. 3007.9999999999).
            pixel_rect = (fitz.Rect(clip) * fitz.Matrix(scale, scale)).irect
            expected_matrix = [1 / scale, 0, 0, 1 / scale, pixel_rect.x0 / scale, pixel_rect.y0 / scale]
            if any(abs(a - b) > .00001 for a, b in zip(crop["pixel_to_display_matrix"], expected_matrix)):
                errors.append("crop pixel transform does not replay")
            if crop["pixel_size"] != [pixel_rect.width, pixel_rect.height]:
                errors.append("crop pixel extent does not replay")
    expected = {row["id"]: row for crop in crops if crop["state"] == "completed"
                for row in _line_observations(crop, payload["document"], payload["configuration"]["minimum_confidence"])}
    if sorted(expected) != sorted(row["id"] for row in rows):
        errors.append("raw OCR line coverage does not replay")
    for row in rows:
        base = {key: value for key, value in row.items() if key not in ("native_overlap_alternatives", "ocr_overlap_alternatives")}
        if base != expected.get(row["id"]):
            errors.append(f"{row['id']}: raw text/confidence/transform/role does not replay")
        for alternative in row["native_overlap_alternatives"] + row["ocr_overlap_alternatives"]:
            if alternative.get("automatic_merge_established") is not False:
                errors.append("overlap alternative acquired merge authority")
    for page in pages:
        for key in ("item_inventory_complete", "item_recall_measured", "item_absence_established", "quantity_eligible"):
            if page.get(key) is not False:
                errors.append("page coverage acquired item authority")
        if sorted(page["observation_refs"]) != sorted(row["id"] for row in rows if row["page_ref"] == page["page_ref"]):
            errors.append("page observation ownership does not replay")
        if sorted(page["crop_refs"]) != sorted(crop["id"] for crop in crops if crop["page_ref"] == page["page_ref"]):
            errors.append("page crop ownership does not replay")
        owned = [crop for crop in crops if crop["page_ref"] == page["page_ref"]]
        config = payload["configuration"]
        expected_plan = list(_tile_plan(fitz.Rect(0, 0, *page["page_size_display"]), config["render_scale"], config["tile_pixels"])) if page["selection"]["selected"] else []
        if [(crop["core_display"], crop["crop_display"]) for crop in owned] != [(list(core), list(clip)) for core, clip in expected_plan]:
            errors.append("page disjoint raster coverage plan does not replay")
        completed = [crop for crop in owned if crop["state"] == "completed"]
        area = math.prod(page["page_size_display"])
        expected_fraction = round(sum(fitz.Rect(crop["core_display"]).get_area() for crop in completed) / area, 8) if page["selection"]["selected"] else None
        if (page["planned_crop_count"] != len(owned) or page["completed_crop_count"] != len(completed)
                or page["processed_raster_area_fraction"] != expected_fraction):
            errors.append("page raster coverage does not replay")
    return errors


def build_mep_item_ocr_proposals(payload):
    errors = validate_mep_item_ocr_observations(payload)
    if errors:
        raise ValueError("invalid OCR evidence: " + "; ".join(errors))
    valid = [row for row in payload["observations"] if row["handoff_state"] == "proposal_only"]
    proposals = build_mep_terminology_proposals(document=payload["document"], observations=valid)
    proposals["source_observations"] = deepcopy(payload["observations"])
    proposals["summary"]["source_observation_count"] = len(payload["observations"])
    proposals["observation_diagnostics"] = [{
        "observation_ref": row["id"], "page_ref": row["page_ref"], "state": "quarantined",
        "stage": "item_ocr_quality_and_role_handoff", "errors": row["handoff_errors"],
        "downstream_acceptance_closed": True, "quantity_eligible": False,
    } for row in payload["observations"] if row["handoff_state"] == "quarantined"]
    proposals["item_ocr_observation_ref"] = {"layer": LAYER, "payload_sha256": _sha256(payload)}
    errors = validate_mep_terminology_proposals(proposals)
    if errors:
        raise ValueError("invalid OCR M2 handoff: " + "; ".join(errors))
    return proposals
