"""Native drawing-reference links, independent of item or physical applicability.

The deliberately bounded recognizer handles split-circle alphanumeric riser
references. It never treats matching unstructured text as a reference, nor a
reference as proof that a detail applies to an equipment body or pipe segment.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import fitz

from src.drawing_engine.core.vector_topology import _cubic_points


def _id(kind, *values):
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"mep_document_{kind}.{digest[:20]}"


def _rect(points):
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def _center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def build_duct_plan_section_station_correspondence(*, relation,
                                                    size_certificate,
                                                    plan_station_coordinates_m,
                                                    landmark_pairs,
                                                    tolerance_m):
    """Bind plan stations to one linked section through native landmarks.

    The accepted document relation identifies the views. Independent landmark
    pairs resolve the signed one-dimensional station transform. Final
    acceptance additionally consumes the actual Step 1 size-applicability
    certificate; a section relation by itself never chooses a route.
    """
    evidence_refs = sorted({str(ref)
        for row in [relation, size_certificate, *landmark_pairs]
        for ref in row.get("evidence_refs", [])})
    reasons = []
    if (relation.get("state") != "accepted"
            or relation.get("mutually_unique") is not True
            or relation.get("search_complete") is not True
            or relation.get("plan_and_section_metric") is not True):
        reasons.append("unique_plan_to_section_relation_unresolved")
    if relation.get("sized_route_intersection_established") is not True:
        reasons.append("plan_section_cut_does_not_certify_sized_route")
    size_gates = size_certificate.get("certificates", {})
    if (size_certificate.get("record_type") !=
            "mep_duct_size_applicability_certificate"
            or size_certificate.get("state") != "accepted"
            or size_gates.get("unique_native_duct_interval") is not True
            or not size_certificate.get("route_composite_ref")):
        reasons.append("uniquely_sized_native_duct_interval_unresolved")
    document_ref = relation.get("source_pdf_sha256")
    if (not document_ref
            or size_certificate.get("source_pdf_sha256") != document_ref
            or any(row.get("source_pdf_sha256") != document_ref
                   for row in landmark_pairs)):
        reasons.append("station_evidence_document_scope_mismatch")
    try:
        tolerance = float(tolerance_m)
    except (TypeError, ValueError):
        tolerance = float("nan")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite station tolerance required")
    valid_pairs = []
    identities = []
    if landmark_pairs and any(row.get("state") != "accepted"
            or row.get("physical_identity_established") is not True
            for row in landmark_pairs):
        reasons.append("physical_native_landmark_targets_unresolved")
    for row in landmark_pairs:
        plan = row.get("plan", {})
        section = row.get("section", {})
        identity = str(row.get("identity"))
        try:
            plan_station = float(plan["coordinate_m"])
            section_station = float(section["coordinate_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if (row.get("state") != "accepted"
                or row.get("physical_identity_established") is not True
                or not math.isfinite(plan_station)
                or not math.isfinite(section_station)
                or not plan.get("native_refs") or not section.get("native_refs")
                or identity in {"", "None"}):
            continue
        identities.append(identity)
        valid_pairs.append((identity, plan_station, section_station, row))
    if len(valid_pairs) < 2 or len(set(identities)) != len(identities):
        reasons.append("two_unique_native_landmark_pairs_required")
    if valid_pairs and max(row[1] for row in valid_pairs) - min(
            row[1] for row in valid_pairs) <= tolerance:
        reasons.append("plan_landmark_station_span_degenerate")

    accepted_transforms = []
    for sign in (-1, 1):
        offsets = [section - sign * plan
                   for _, plan, section, _ in valid_pairs]
        offset = sum(offsets) / len(offsets) if offsets else 0.0
        residual = max((abs(value - offset) for value in offsets), default=float("inf"))
        if residual <= tolerance:
            accepted_transforms.append({"sign": sign, "offset_m": offset,
                                        "maximum_residual_m": residual})
    if len(accepted_transforms) != 1:
        reasons.append("unique_native_landmark_station_transform_unresolved")

    raw_stations = plan_station_coordinates_m
    try:
        plan_stations = [float(value) for value in raw_stations]
    except (TypeError, ValueError):
        plan_stations = []
    if (len(plan_stations) < 2 or any(not math.isfinite(value) for value in plan_stations)
            or any(right <= left for left, right in zip(plan_stations, plan_stations[1:]))):
        reasons.append("ordered_route_plan_stations_unresolved")
    if valid_pairs and plan_stations:
        low = min(row[1] for row in valid_pairs) - tolerance
        high = max(row[1] for row in valid_pairs) + tolerance
        if plan_stations[0] < low or plan_stations[-1] > high:
            reasons.append("route_stations_outside_native_landmark_span")

    state = "accepted" if not reasons else "abstained"
    transform = accepted_transforms[0] if state == "accepted" else None
    section_stations = ([round(transform["sign"] * value
                               + transform["offset_m"], 8)
                         for value in plan_stations] if transform else [])
    return {"record_type": "mep_duct_plan_section_station_correspondence",
        "id": _id("duct_station_correspondence", relation.get("id"),
                  size_certificate.get("id"), [row[0] for row in valid_pairs]),
        "state": state, "reasons": sorted(set(reasons)),
        "plan_to_section_relation_ref": relation.get("id"),
        "size_applicability_certificate_ref": size_certificate.get("id"),
        "sized_route_interval_ref": size_certificate.get("route_composite_ref"),
        "candidate_landmark_pair_refs": [row.get("id") for row in landmark_pairs],
        "landmark_pair_refs": [row[3].get("id") for row in valid_pairs],
        "signed_station_transform": ({**transform,
            "maximum_residual_m": round(transform["maximum_residual_m"], 8)}
            if transform else None),
        "plan_station_coordinates_m": plan_stations if state == "accepted" else [],
        "section_station_coordinates_m": section_stations,
        "evidence_refs": evidence_refs,
        "relative_z_established": False,
        "absolute_elevation_resolved": False,
        "quantity_eligible": False}


def _paths(drawings, page, texts):
    """Retain page-global primitive IDs; curved samples are display-only."""
    lines, circles = [], []
    code_texts = [t for t in texts if re.fullmatch(r"[A-Z]{1,3}|[1-9][0-9]*", t["text"])]
    for di, drawing in enumerate(drawings):
        items = drawing["items"]
        bbox = drawing["rect"]
        circle_candidate = (3 <= len(items) <= 100 and bbox.width > 6
                            and abs(bbox.width - bbox.height) < .04 * bbox.width)
        if circle_candidate:
            contained = [t for t in code_texts if (bbox * page.rotation_matrix).contains(fitz.Rect(t["bbox_display"]))]
            circle_candidate = (any(t["text"].isalpha() for t in contained)
                                and any(t["text"].isdigit() for t in contained))
        perimeter, dividers = [], []
        for ii, item in enumerate(items):
            if item[0] not in {"l", "c"}:
                continue
            points = _cubic_points(item) if item[0] == "c" else [tuple(item[1]), tuple(item[2])]
            points = [list(fitz.Point(p) * page.rotation_matrix) for p in points]
            row = {"source_primitive_ref": f"drawing[{di}].item[{ii}].segment[0]",
                   "points_display": points, "bbox_display": _rect(points), "kind": item[0]}
            if item[0] == "l":
                lines.append(row)
            if circle_candidate:
                center = list(bbox.tl + (bbox.br - bbox.tl) / 2)
                center = list(fitz.Point(center) * page.rotation_matrix)
                radius = bbox.width / 2
                # Samples must follow a circle; a diameter is kept separately.
                radial_samples = points if item[0] == "c" else [*points, _center(_rect(points))]
                errors = [abs(math.dist(p, center) - radius) / radius for p in radial_samples]
                if max(errors) < .035:
                    perimeter.append(row)
                else:
                    dividers.append(row)
        if not circle_candidate or len(perimeter) < 3:
            continue
        # PDF plotting may split one circle into consecutive drawing records.
        # Recover only a unique exact native endpoint cycle on the same circle.
        if any(math.dist(a["points_display"][-1], b["points_display"][0]) > .12
               for a, b in zip(perimeter, perimeter[1:] + perimeter[:1])):
            for other_di, other in enumerate(drawings):
                if other_di == di or not (bbox + (-.12, -.12, .12, .12)).contains(other["rect"]):
                    continue
                for ii, item in enumerate(other["items"]):
                    if item[0] not in {"l", "c"}:
                        continue
                    points = _cubic_points(item) if item[0] == "c" else [tuple(item[1]), tuple(item[2])]
                    points = [list(fitz.Point(p) * page.rotation_matrix) for p in points]
                    samples = points if item[0] == "c" else [*points, _center(_rect(points))]
                    if max(abs(math.dist(p, center) - radius) / radius for p in samples) < .035:
                        perimeter.append({"source_primitive_ref": f"drawing[{other_di}].item[{ii}].segment[0]",
                                          "points_display": points, "bbox_display": _rect(points), "kind": item[0]})
            remaining = list(perimeter)
            ordered = [remaining.pop(0)]
            while remaining:
                point = ordered[-1]["points_display"][-1]
                choices = [(index, reverse) for index, row in enumerate(remaining)
                           for reverse in (False, True)
                           if math.dist(point, row["points_display"][-1 if reverse else 0]) <= .12]
                if len(choices) != 1:
                    break
                index, reverse = choices[0]
                row = dict(remaining.pop(index))
                if reverse:
                    row["points_display"] = row["points_display"][::-1]
                ordered.append(row)
            if remaining or math.dist(ordered[-1]["points_display"][-1], ordered[0]["points_display"][0]) > .12:
                continue
            perimeter = ordered
        sampled = [p for row in perimeter for p in row["points_display"]]
        if sum(math.dist(a, b) for a, b in zip(sampled, sampled[1:])) < 5.8 * radius:
            continue
        circles.append({"bbox_display": list(bbox * page.rotation_matrix),
                        "source_paths": perimeter, "drawing_ref": f"drawing[{di}]"})
    return lines, circles


def extract_document_reference_observations(source, scope):
    """Scan an explicitly frozen package scope within one original PDF.

    Page numbers are acquisition scope, never a solver selector. They retain
    their original full-document indices and cannot cross source hashes.
    """
    source = Path(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != scope["source_pdf_sha256"]:
        raise ValueError("package scope source hash mismatch")
    numbers = scope["source_page_numbers"]
    if not numbers or numbers != sorted(set(numbers)):
        raise ValueError("package page numbers must be unique and sorted")
    output = {"schema_version": "0.1.0", "layer": "mep_document_reference_observations", "document": {
        "source_pdf_sha256": source_hash, "source_url": scope.get("source_url"),
        "document_scope_id": scope["document_scope_id"], "source_path": str(source)},
        "scope": scope, "pages": [], "text_observations": [], "markers": [], "views": []}
    with fitz.open(source) as pdf:
        if numbers[0] < 1 or numbers[-1] > len(pdf):
            raise ValueError("package scope page outside source PDF")
        output["document"]["full_source_page_count"] = len(pdf)
        for number in numbers:
            page = pdf[number - 1]
            page_ref = _id("page", source_hash, number)
            base = {"page_ref": page_ref, "source_page_number": number,
                    "source_pdf_sha256": source_hash}
            texts, blocks = [], []
            for block in page.get_text("dict")["blocks"]:
                block_texts = []
                for li, line in enumerate(block.get("lines", [])):
                    text = " ".join(span["text"] for span in line["spans"]).strip()
                    native = f"page[{number}].text_block[{block['number']}].line[{li}]"
                    row = {**base, "id": _id("text", source_hash, native, text), "text": text,
                           "source_native_ref": native,
                           "bbox_display": list(fitz.Rect(line["bbox"]) * page.rotation_matrix)}
                    texts.append(row)
                    block_texts.append(row)
                blocks.append(block_texts)
            drawings = page.get_drawings()
            lines, circles = _paths(drawings, page, texts)
            markers = []
            for circle in circles:
                box = circle["bbox_display"]
                cx, cy = _center(box)
                diameter = box[2] - box[0]
                tokens = [t for t in texts if fitz.Rect(box).contains(fitz.Rect(t["bbox_display"]))]
                upper = [t for t in tokens if re.fullmatch(r"[A-Z]{1,3}", t["text"])
                         and _center(t["bbox_display"])[1] < cy]
                lower = [t for t in tokens if re.fullmatch(r"[1-9][0-9]*", t["text"])
                         and _center(t["bbox_display"])[1] > cy]
                dividers = [row for row in lines if
                            abs(row["bbox_display"][1] - cy) < .02 * diameter
                            and row["bbox_display"][3] - row["bbox_display"][1] < .01 * diameter
                            and abs(row["bbox_display"][0] - box[0]) < .02 * diameter
                            and abs(row["bbox_display"][2] - box[2]) < .02 * diameter]
                if len(upper) != 1 or len(lower) != 1 or len(tokens) != 2 or not dividers:
                    continue
                code = upper[0]["text"] + "/" + lower[0]["text"]
                paths = circle["source_paths"] + dividers
                row = {**base, "id": _id("marker", source_hash, number, circle["drawing_ref"]),
                       "code": code, "bbox_display": box, "state": "observed",
                       "method": "native_closed_circle_divider_and_two_cell_text",
                       "text_refs": [upper[0]["id"], lower[0]["id"]],
                       "source_paths": paths, "evidence_refs": [t["id"] for t in (upper[0], lower[0])]}
                markers.append(row)
            views = []
            for block in blocks:
                captions = [t for t in block if re.search(r"\bRISER DIAGRAM\b", t["text"])
                            and not t["text"].endswith("DIAGRAMS")]
                scale = [t for t in block if re.search(r"\d.*=.+\d", t["text"])]
                index = [t for t in block if re.fullmatch(r"[1-9][0-9]*", t["text"])]
                if len(captions) != 1 or len(scale) != 1 or len(index) != 1:
                    continue
                caption = captions[0]
                box = caption["bbox_display"]
                height = box[3] - box[1]
                underlines = [line for line in lines if
                              abs(line["bbox_display"][1] - box[3]) < .15 * height
                              and line["bbox_display"][3] - line["bbox_display"][1] < .03 * height
                              and box[0] - height < line["bbox_display"][0] <= box[0]
                              and line["bbox_display"][2] >= box[2]]
                if len(underlines) != 1:
                    continue
                underline = underlines[0]
                views.append({**base, "id": _id("view", source_hash, caption["id"]),
                    "title": caption["text"], "view_number": index[0]["text"],
                    "title_bbox_display": box, "state": "observed",
                    "evidence_refs": [t["id"] for t in (caption, scale[0], index[0])],
                    "source_paths": [underline], "view_extent_state": "unknown",
                    "caption_baseline_display": underline["bbox_display"]})
            for marker in markers:
                x, y = _center(marker["bbox_display"])
                compatible = [view for view in views if
                              view["caption_baseline_display"][0] < x < view["caption_baseline_display"][2]
                              and y < view["caption_baseline_display"][1]]
                # Numbered caption baselines partition vertically stacked
                # diagram context. This assigns a reference anchor to a caption,
                # never a geometric item extent or physical applicability.
                if compatible:
                    first_baseline = min(v["caption_baseline_display"][1] for v in compatible)
                    compatible = [v for v in compatible if abs(v["caption_baseline_display"][1] - first_baseline) < .12]
                marker["destination_view_candidates"] = [v["id"] for v in compatible]
                marker["view_context_method"] = "first_numbered_native_caption_baseline_below_anchor"
                marker["role"] = "riser_view_anchor" if len(compatible) == 1 else "reference_marker"
                if len(compatible) > 1:
                    marker["role"] = "ambiguous_view_context"
            output["text_observations"].extend(texts)
            output["markers"].extend(markers)
            output["views"].extend(views)
            output["pages"].append({**base, "page_size_display": [page.rect.width, page.rect.height],
                "native_text_scan_complete": True, "native_drawing_scan_complete": True,
                "drawing_path_count": len(drawings), "native_text_line_count": len(texts),
                "ocr_run": False, "supported_marker_search_complete": True,
                "all_document_reference_classes_complete": False})
    return output


def build_document_reference_links(observations):
    """Deterministic replay over frozen native observations; no scope inference."""
    payload = json.loads(json.dumps(observations))
    payload["layer"] = "mep_document_reference_links"
    document = payload["document"]
    if any(row["source_pdf_sha256"] != document["source_pdf_sha256"]
           for kind in ("pages", "text_observations", "markers", "views") for row in payload[kind]):
        raise ValueError("cross-document reference observations prohibited")
    if [p["source_page_number"] for p in payload["pages"]] != payload["scope"]["source_page_numbers"]:
        raise ValueError("incomplete package search cannot establish unique destination")
    if any(not p.get("native_text_scan_complete") or not p.get("native_drawing_scan_complete") for p in payload["pages"]):
        raise ValueError("incomplete native page scan")
    texts = {t["id"]: t for t in payload["text_observations"]}
    pages = {p["page_ref"]: p for p in payload["pages"]}
    for marker in payload["markers"]:
        if marker["page_ref"] not in pages or len(marker["text_refs"]) != 2 or not marker["source_paths"]:
            raise ValueError("unbacked marker")
        tokens = [texts[ref] for ref in marker["text_refs"]]
        if marker["code"] != "/".join(t["text"] for t in tokens) or any(t["page_ref"] != marker["page_ref"] for t in tokens):
            raise ValueError("marker code does not replay native text")
    views = {v["id"]: v for v in payload["views"]}
    destinations = [m for m in payload["markers"] if m["role"] == "riser_view_anchor"]
    links = []
    for marker in payload["markers"]:
        if marker["role"] == "riser_view_anchor":
            continue
        candidates = [m for m in destinations if m["code"] == marker["code"]
                      and m["page_ref"] != marker["page_ref"]]
        accepted = len(candidates) == 1 and marker["role"] == "reference_marker"
        target = candidates[0] if accepted else None
        target_view = views[target["destination_view_candidates"][0]] if target else None
        links.append({"id": _id("reference_link", document["document_scope_id"], marker["id"]),
            "source_marker_ref": marker["id"], "code": marker["code"],
            "source_page_ref": marker["page_ref"], "source_page_number": marker["source_page_number"],
            "state": "accepted" if accepted else "abstained",
            "relation_type": "explicit_native_riser_reference",
            "destination_marker_ref": target["id"] if target else None,
            "destination_view_ref": target_view["id"] if target_view else None,
            "destination_page_ref": target["page_ref"] if target else None,
            "destination_page_number": target["source_page_number"] if target else None,
            "candidate_destination_refs": [m["id"] for m in candidates],
            "reason": "unique_structured_code_to_native_riser_anchor" if accepted else
                      "missing_destination_anchor" if not candidates else "non_unique_destination_or_view_context",
            "evidence_refs": [marker["id"], *([target["id"], target_view["id"]] if target else [])],
            "source_paths": marker["source_paths"],
            "destination_source_paths": (target["source_paths"] + target_view["source_paths"]) if target else [],
            "applicability_state": "unknown", "item_refs": [],
            "physical_identity_established": False, "physical_continuation_established": False,
            "quantity_eligible": False})
    payload["reference_links"] = links
    payload["applicability_links"] = [{"id": _id("applicability", link["id"]),
        "reference_link_ref": link["id"], "state": "unknown", "item_refs": [],
        "reason": "reference_identification_does_not_establish_item_applicability"} for link in links]
    payload["summary"] = {"scoped_page_count": len(payload["pages"]), "marker_count": len(payload["markers"]),
        "native_riser_view_count": len(views), "reference_link_states": dict(Counter(l["state"] for l in links)),
        "accepted_applicability_count": 0, "quantity_eligible": False}
    payload["authority"] = {"reference_identification_only": True, "view_extent_established": False,
        "item_applicability_established": False, "physical_identity_established": False,
        "physical_continuation_established": False, "engineer_approved": False, "quantity_eligible": False}
    return payload


def validate_document_reference_links(payload, *, source=None):
    expected = build_document_reference_links(payload)
    for key in ("reference_links", "applicability_links", "summary", "authority"):
        if payload.get(key) != expected[key]:
            raise ValueError(f"document reference replay mismatch: {key}")
    if source is not None:
        replay = extract_document_reference_observations(source, payload["scope"])
        for key in ("pages", "markers", "views", "text_observations"):
            if replay[key] != payload[key]:
                raise ValueError(f"native source replay mismatch: {key}")
    return True
