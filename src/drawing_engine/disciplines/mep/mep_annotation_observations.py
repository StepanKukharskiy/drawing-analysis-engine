"""Immutable PDF-annotation observations for the MEP coordination boundary.

This module intentionally stops before target grounding.  A review cloud and
its authored callout are source observations, not geometry instructions,
accepted clashes, clearance requirements, or quantity inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import fitz


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_pdf_annotation_observations"
_OBJECT_REFERENCE = re.compile(r"^(\d+)\s+(\d+)\s+R$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _numbers(value: Iterable[object] | None) -> list[float]:
    return [float(item) for item in (value or [])]


def _points(value: Iterable[object] | None) -> list[list[float]]:
    points = []
    for item in value or []:
        point = fitz.Point(item)
        points.append([float(point.x), float(point.y)])
    return points


def _matrix(value: fitz.Matrix) -> list[float]:
    return [float(item) for item in value]


def _xref_key(document: fitz.Document, xref: int, key: str) -> tuple[str, str | None]:
    kind, value = document.xref_get_key(xref, key)
    return str(kind), None if kind == "null" else str(value)


def _pdf_name(document: fitz.Document, xref: int, key: str) -> str | None:
    kind, value = _xref_key(document, xref, key)
    if kind != "name" or value is None:
        return None
    return value[1:] if value.startswith("/") else value


def _pdf_reference(document: fitz.Document, xref: int, key: str) -> dict[str, int] | None:
    kind, value = _xref_key(document, xref, key)
    if kind != "xref" or value is None:
        return None
    match = _OBJECT_REFERENCE.fullmatch(value)
    if match is None:
        return None
    return {"xref": int(match.group(1)), "generation": int(match.group(2))}


def _pdf_string(document: fitz.Document, xref: int, key: str) -> str | None:
    kind, value = _xref_key(document, xref, key)
    return value if kind in {"string", "array"} else None


def _annotation_kind(record: Mapping[str, Any]) -> str | None:
    subtype = record.get("annotation_subtype")
    intent = record.get("intent")
    subject = str(record.get("subject") or "").casefold()
    vertices = record.get("geometry", {}).get("vertices_display", [])
    if subtype == "Polygon" and (intent == "PolygonCloud" or "cloud" in subject):
        return "cloud"
    if subtype == "FreeText" and intent == "FreeTextCallout" and vertices:
        return "callout"
    if subtype == "FreeText" and (record.get("text") or "text box" in subject):
        return "text_box"
    return None


def _point_in_rect(point: Iterable[object], rect: Iterable[object], *, tolerance: float = 1e-6) -> bool:
    x, y = (float(item) for item in point)
    x0, y0, x1, y1 = (float(item) for item in rect)
    return x0 - tolerance <= x <= x1 + tolerance and y0 - tolerance <= y <= y1 + tolerance


def pair_annotation_observations(
    annotations: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair cloud/callout observations through explicit or unique evidence.

    Returns accepted observation pairs and explicit abstentions.  Geometric
    pairing is limited to the first callout leader point lying inside exactly
    one otherwise-unpaired cloud rectangle, with a mutual one-to-one match.
    """

    records = [dict(item) for item in annotations]
    by_xref = {
        int(item["source_object"]["xref"]): item
        for item in records
        if isinstance(item.get("source_object", {}).get("xref"), int)
    }
    paired_ids: set[str] = set()
    pairs: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []

    explicit_candidates: set[tuple[str, str, str]] = set()
    for item in records:
        reference = item.get("in_reply_to")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("xref"), int):
            continue
        target = by_xref.get(int(reference["xref"]))
        if target is None or target.get("page_ref") != item.get("page_ref"):
            continue
        item_kind = _annotation_kind(item)
        target_kind = _annotation_kind(target)
        kinds = {item_kind, target_kind}
        if "cloud" not in kinds or not kinds.intersection({"callout", "text_box"}):
            continue
        cloud = item if item_kind == "cloud" else target
        callout = target if item_kind == "cloud" else item
        relation = (
            "pdf_group_relationship"
            if item.get("reply_type") == "Group"
            else "pdf_reply_relationship"
        )
        explicit_candidates.add((str(cloud["id"]), str(callout["id"]), relation))

    by_id = {str(item["id"]): item for item in records}
    explicit_degree: Counter[str] = Counter()
    for cloud_ref, callout_ref, _relation in explicit_candidates:
        explicit_degree[cloud_ref] += 1
        explicit_degree[callout_ref] += 1
    for cloud_ref, callout_ref, relation in sorted(explicit_candidates):
        if explicit_degree[cloud_ref] != 1 or explicit_degree[callout_ref] != 1:
            abstentions.append(
                {
                    "record_type": "annotation_pair_abstention",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("annotation_pair_abstention", cloud_ref, callout_ref),
                    "page_ref": by_id[cloud_ref]["page_ref"],
                    "reason": "ambiguous_pdf_relationship",
                    "candidate_annotation_refs": sorted([cloud_ref, callout_ref]),
                }
            )
            continue
        pair = _build_pair(
            by_id[cloud_ref],
            by_id[callout_ref],
            relation=relation,
            evidence={
                "source": "pdf_annotation_irt",
                "cloud_source_object": by_id[cloud_ref]["source_object"],
                "callout_source_object": by_id[callout_ref]["source_object"],
            },
        )
        pairs.append(pair)
        paired_ids.update((cloud_ref, callout_ref))

    clouds = [
        item for item in records if _annotation_kind(item) == "cloud" and item["id"] not in paired_ids
    ]
    callouts = [
        item for item in records if _annotation_kind(item) == "callout" and item["id"] not in paired_ids
    ]
    candidates_by_callout: dict[str, list[str]] = {}
    callouts_by_cloud: defaultdict[str, list[str]] = defaultdict(list)
    for callout in callouts:
        leader_points = callout["geometry"].get("vertices_display", [])
        leader_target = leader_points[0] if leader_points else None
        candidates = [
            str(cloud["id"])
            for cloud in clouds
            if cloud.get("page_ref") == callout.get("page_ref")
            and leader_target is not None
            and _point_in_rect(leader_target, cloud["geometry"]["rect_display"])
        ]
        candidates_by_callout[str(callout["id"])] = sorted(candidates)
        for cloud_ref in candidates:
            callouts_by_cloud[cloud_ref].append(str(callout["id"]))

    for callout in sorted(callouts, key=lambda item: str(item["id"])):
        callout_ref = str(callout["id"])
        candidates = candidates_by_callout[callout_ref]
        if len(candidates) == 1 and len(callouts_by_cloud[candidates[0]]) == 1:
            cloud = by_id[candidates[0]]
            leader_target = callout["geometry"]["vertices_display"][0]
            pairs.append(
                _build_pair(
                    cloud,
                    callout,
                    relation="unique_geometric_leader_target",
                    evidence={
                        "source": "callout_leader_target_in_cloud_rect",
                        "leader_target_display": leader_target,
                        "candidate_cloud_refs": candidates,
                        "mutual_unique": True,
                    },
                )
            )
            paired_ids.update((str(cloud["id"]), callout_ref))
        elif candidates:
            abstentions.append(
                {
                    "record_type": "annotation_pair_abstention",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("annotation_pair_abstention", callout_ref, *candidates),
                    "page_ref": callout["page_ref"],
                    "reason": "ambiguous_geometric_pairing",
                    "candidate_annotation_refs": sorted([callout_ref, *candidates]),
                }
            )

    pairs.sort(key=lambda item: (int(item["page_number"]), str(item["id"])))
    abstentions.sort(key=lambda item: (str(item["page_ref"]), str(item["id"])))
    return pairs, abstentions


def _build_pair(
    cloud: Mapping[str, Any],
    callout: Mapping[str, Any],
    *,
    relation: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    cloud_ref = str(cloud["id"])
    callout_ref = str(callout["id"])
    return {
        "record_type": "annotation_pair_observation",
        "record_version": SCHEMA_VERSION,
        "id": _stable_id("annotation_pair_observation", cloud_ref, callout_ref, relation),
        "page_ref": str(cloud["page_ref"]),
        "page_number": int(cloud["page_number"]),
        "cloud_annotation_ref": cloud_ref,
        "callout_annotation_ref": callout_ref,
        "comment_form": _annotation_kind(callout),
        "relationship": relation,
        "epistemic_state": "direct" if relation.startswith("pdf_") else "derived",
        "evidence": dict(evidence),
        "markup_claim_only": True,
        "target_identity_ref": None,
        "confirmed_clash": False,
        "quantity_eligible": False,
    }


def extract_pdf_annotation_observations(pdf_path: Path | str, *, page_numbers: list[int] | None = None) -> dict[str, Any]:
    """Extract all PDF annotations without interpreting their engineering meaning."""

    source_path = Path(pdf_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = _file_sha256(source_path)
    document_key = f"pdf-sha256:{source_sha256}"
    pages: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    with fitz.open(source_path) as document:
        selected = set(range(1, len(document) + 1)) if page_numbers is None else set(page_numbers)
        if not selected or any(type(n) is not int or not 1 <= n <= len(document) for n in selected):
            raise ValueError("selected page is outside the source PDF")
        for page in document:
            page_number = int(page.number) + 1
            page_ref = _stable_id("mep_page_observation", document_key, page_number, page.xref)
            transform = page.transformation_matrix
            inverse = ~transform
            page_annotations: list[str] = []
            for annotation in (page.annots() or []) if page_number in selected else []:
                info = annotation.info or {}
                annotation_id = _stable_id(
                    "pdf_annotation_observation",
                    document_key,
                    page_number,
                    annotation.xref,
                )
                page_annotations.append(annotation_id)
                rect_display = annotation.rect
                vertices_display = _points(annotation.vertices)
                colors = annotation.colors or {}
                source_object_name = str(info.get("id") or "") or None
                annotations.append(
                    {
                        "record_type": "pdf_annotation_observation",
                        "record_version": SCHEMA_VERSION,
                        "id": annotation_id,
                        "document_key": document_key,
                        "page_ref": page_ref,
                        "page_number": page_number,
                        "epistemic_state": "observed",
                        "immutable_source_observation": True,
                        "annotation_subtype": str(annotation.type[1]),
                        "annotation_type_code": int(annotation.type[0]),
                        "intent": _pdf_name(document, annotation.xref, "IT"),
                        "extended_intent": _pdf_name(document, annotation.xref, "ITEx"),
                        "subject": str(info.get("subject") or ""),
                        "author": str(info.get("title") or ""),
                        "text": str(info.get("content") or ""),
                        "icon_name": str(info.get("name") or ""),
                        "created_at_pdf": str(info.get("creationDate") or ""),
                        "modified_at_pdf": str(info.get("modDate") or ""),
                        "flags": int(annotation.flags),
                        "opacity": float(annotation.opacity),
                        "annotation_rotation": int(annotation.rotation),
                        "geometry": {
                            "coordinate_space": "page_display_points_top_left",
                            "rect_display": _numbers(rect_display),
                            "rect_pdf": _numbers(rect_display * inverse),
                            "vertices_display": vertices_display,
                            "vertices_pdf": [
                                _numbers(fitz.Point(point) * inverse) for point in vertices_display
                            ],
                        },
                        "appearance": {
                            "stroke_color": _numbers(colors.get("stroke")),
                            "fill_color": _numbers(colors.get("fill")),
                            "pdf_color_array": _pdf_string(document, annotation.xref, "C"),
                            "default_appearance": _pdf_string(document, annotation.xref, "DA"),
                            "default_style": _pdf_string(document, annotation.xref, "DS"),
                            "border": {
                                str(key): list(value) if isinstance(value, tuple) else value
                                for key, value in (annotation.border or {}).items()
                            },
                        },
                        "source_object": {
                            "xref": int(annotation.xref),
                            "object_name": source_object_name,
                        },
                        "in_reply_to": _pdf_reference(document, annotation.xref, "IRT"),
                        "reply_type": _pdf_name(document, annotation.xref, "RT"),
                    }
                )
            pages.append(
                {
                    "record_type": "mep_page_observation",
                    "record_version": SCHEMA_VERSION,
                    "id": page_ref,
                    "document_key": document_key,
                    "page_number": page_number,
                    "source_page_object": {"xref": int(page.xref)},
                    "media_box_pdf": _numbers(page.mediabox),
                    "crop_box_pdf": _numbers(page.cropbox),
                    "page_rect_display": _numbers(page.rect),
                    "rotation_degrees": int(page.rotation),
                    "pdf_to_display_matrix": _matrix(transform),
                    "display_to_pdf_matrix": _matrix(inverse),
                    "source_content_stream_xrefs": [int(item) for item in page.get_contents()],
                    "annotation_refs": sorted(page_annotations),
                    "sheet_role": "unknown",
                    "native_raster_route": "unclassified",
                }
            )
            if page_numbers is not None:
                pages[-1]["annotation_scan_complete"] = page_number in selected

        pairs, abstentions = pair_annotation_observations(annotations)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "layer": LAYER,
            "document": {
                "document_key": document_key,
                "source_filename": source_path.name,
                "source_pdf_sha256": source_sha256,
                "source_bytes": source_path.stat().st_size,
                "page_count": int(document.page_count),
            },
            "pages": pages,
            "annotations": sorted(
                annotations,
                key=lambda item: (int(item["page_number"]), int(item["source_object"]["xref"])),
            ),
            "annotation_pairs": pairs,
            "pairing_abstentions": abstentions,
            "summary": {
                "page_count": len(pages),
                "annotation_count": len(annotations),
                "annotation_pair_count": len(pairs),
                "pairing_abstention_count": len(abstentions),
                "annotation_subtype_counts": dict(
                    sorted(Counter(item["annotation_subtype"] for item in annotations).items())
                ),
                "pair_relationship_counts": dict(
                    sorted(Counter(item["relationship"] for item in pairs).items())
                ),
            },
            "exchange_contract": {
                "page_record_type": "mep_page_observation",
                "annotation_record_type": "pdf_annotation_observation",
                "pair_record_type": "annotation_pair_observation",
                "record_version": SCHEMA_VERSION,
                "page_transforms_are_explicit": True,
                "source_pdf_objects_are_preserved": True,
                "annotation_observations_are_immutable": True,
                "annotation_pairs_are_markup_claims_only": True,
                "sheet_role_and_native_raster_route_are_deferred_to_m1": True,
                "target_grounding_is_deferred_to_m6": True,
                "route_identity_or_connectivity_established": False,
                "confirmed_clash_or_clearance_established": False,
                "quantity_eligible": False,
                "schedule_values_used": False,
            },
            "extractor": {
                "name": "pymupdf_pdf_annotation_objects",
                "version": SCHEMA_VERSION,
                "pymupdf_version": fitz.VersionBind,
            },
        }
    if page_numbers is not None:
        payload["processing_scope"] = {"source_page_numbers": sorted(selected),
            "unselected_pages": "metadata_only_not_processed"}
    errors = validate_pdf_annotation_observations(payload)
    if errors:
        raise ValueError("invalid extracted PDF annotation observations: " + "; ".join(errors))
    return payload


def validate_pdf_annotation_observations(
    payload: Mapping[str, Any],
    source_pdf: Path | str | None = None,
) -> list[str]:
    """Validate the frozen M0 exchange boundary, optionally against source bytes."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("layer") != LAYER:
        errors.append(f"layer must be {LAYER}")
    document = payload.get("document")
    if not isinstance(document, Mapping):
        errors.append("document must be an object")
        document = {}
    document_key = str(document.get("document_key") or "")
    source_hash = str(document.get("source_pdf_sha256") or "")
    if document_key != f"pdf-sha256:{source_hash}" or len(source_hash) != 64:
        errors.append("document_key must be derived from source_pdf_sha256")

    pages = payload.get("pages")
    annotations = payload.get("annotations")
    pairs = payload.get("annotation_pairs")
    abstentions = payload.get("pairing_abstentions")
    for name, value in (
        ("pages", pages),
        ("annotations", annotations),
        ("annotation_pairs", pairs),
        ("pairing_abstentions", abstentions),
    ):
        if not isinstance(value, list):
            errors.append(f"{name} must be a list")
    if errors and any(not isinstance(value, list) for value in (pages, annotations, pairs, abstentions)):
        return errors

    page_by_id: dict[str, Mapping[str, Any]] = {}
    page_numbers: set[int] = set()
    for index, page in enumerate(pages):
        prefix = f"pages[{index}]"
        page_id = str(page.get("id") or "")
        if not page_id or page_id in page_by_id:
            errors.append(f"{prefix}.id must be unique and non-empty")
        page_by_id[page_id] = page
        if page.get("record_type") != "mep_page_observation":
            errors.append(f"{prefix}.record_type is invalid")
        if page.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version is invalid")
        if page.get("document_key") != document_key:
            errors.append(f"{prefix}.document_key does not match")
        page_number = page.get("page_number")
        if not isinstance(page_number, int) or page_number < 1 or page_number in page_numbers:
            errors.append(f"{prefix}.page_number must be unique and positive")
        else:
            page_numbers.add(page_number)
        for field, length in (
            ("media_box_pdf", 4),
            ("crop_box_pdf", 4),
            ("page_rect_display", 4),
            ("pdf_to_display_matrix", 6),
            ("display_to_pdf_matrix", 6),
        ):
            if not isinstance(page.get(field), list) or len(page[field]) != length:
                errors.append(f"{prefix}.{field} must contain {length} numbers")
        if page.get("sheet_role") != "unknown" or page.get("native_raster_route") != "unclassified":
            errors.append(f"{prefix} must defer M1 classification")

    annotation_by_id: dict[str, Mapping[str, Any]] = {}
    annotation_page_refs: defaultdict[str, set[str]] = defaultdict(set)
    for index, annotation in enumerate(annotations):
        prefix = f"annotations[{index}]"
        annotation_id = str(annotation.get("id") or "")
        if not annotation_id or annotation_id in annotation_by_id:
            errors.append(f"{prefix}.id must be unique and non-empty")
        annotation_by_id[annotation_id] = annotation
        if annotation.get("record_type") != "pdf_annotation_observation":
            errors.append(f"{prefix}.record_type is invalid")
        if annotation.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version is invalid")
        if annotation.get("document_key") != document_key:
            errors.append(f"{prefix}.document_key does not match")
        if annotation.get("epistemic_state") != "observed":
            errors.append(f"{prefix}.epistemic_state must be observed")
        if annotation.get("immutable_source_observation") is not True:
            errors.append(f"{prefix} must be immutable")
        if not annotation.get("annotation_subtype"):
            errors.append(f"{prefix}.annotation_subtype is required")
        if not isinstance(annotation.get("author"), str) or not isinstance(annotation.get("text"), str):
            errors.append(f"{prefix}.author and text must be strings")
        source_object = annotation.get("source_object")
        if not isinstance(source_object, Mapping) or not isinstance(source_object.get("xref"), int):
            errors.append(f"{prefix}.source_object.xref is required")
        geometry = annotation.get("geometry")
        if not isinstance(geometry, Mapping):
            errors.append(f"{prefix}.geometry must be an object")
        else:
            if geometry.get("coordinate_space") != "page_display_points_top_left":
                errors.append(f"{prefix}.geometry.coordinate_space is invalid")
            for field in ("rect_display", "rect_pdf"):
                if not isinstance(geometry.get(field), list) or len(geometry[field]) != 4:
                    errors.append(f"{prefix}.geometry.{field} must contain four numbers")
        page_ref = str(annotation.get("page_ref") or "")
        if page_ref not in page_by_id:
            errors.append(f"{prefix}.page_ref is unknown")
        else:
            annotation_page_refs[page_ref].add(annotation_id)
            if annotation.get("page_number") != page_by_id[page_ref].get("page_number"):
                errors.append(f"{prefix}.page_number does not match page_ref")
        for forbidden in ("accepted_target_ref", "engineering_entity_ref", "quantity"):
            if forbidden in annotation:
                errors.append(f"{prefix}.{forbidden} is forbidden at M0")

    for page_id, page in page_by_id.items():
        if sorted(page.get("annotation_refs", [])) != sorted(annotation_page_refs[page_id]):
            errors.append(f"page {page_id} annotation_refs do not match observations")
    if document.get("page_count") != len(pages):
        errors.append("document.page_count does not match page observations")
    if page_numbers != set(range(1, len(pages) + 1)):
        errors.append("page observations must cover every page exactly once")

    paired_annotations: set[str] = set()
    pair_ids: set[str] = set()
    allowed_relationships = {
        "pdf_group_relationship",
        "pdf_reply_relationship",
        "unique_geometric_leader_target",
    }
    for index, pair in enumerate(pairs):
        prefix = f"annotation_pairs[{index}]"
        pair_id = str(pair.get("id") or "")
        if not pair_id or pair_id in pair_ids:
            errors.append(f"{prefix}.id must be unique and non-empty")
        pair_ids.add(pair_id)
        if pair.get("record_type") != "annotation_pair_observation":
            errors.append(f"{prefix}.record_type is invalid")
        if pair.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version is invalid")
        relationship = pair.get("relationship")
        if relationship not in allowed_relationships:
            errors.append(f"{prefix}.relationship is unsupported")
        cloud_ref = str(pair.get("cloud_annotation_ref") or "")
        callout_ref = str(pair.get("callout_annotation_ref") or "")
        cloud = annotation_by_id.get(cloud_ref)
        callout = annotation_by_id.get(callout_ref)
        if cloud is None or _annotation_kind(cloud) != "cloud":
            errors.append(f"{prefix}.cloud_annotation_ref is not a cloud")
        if callout is None or _annotation_kind(callout) not in {"callout", "text_box"}:
            errors.append(f"{prefix}.callout_annotation_ref is not a callout or text box")
        elif pair.get("comment_form") != _annotation_kind(callout):
            errors.append(f"{prefix}.comment_form does not match the source annotation")
        if cloud is not None and callout is not None:
            if cloud.get("page_ref") != callout.get("page_ref") or pair.get("page_ref") != cloud.get("page_ref"):
                errors.append(f"{prefix} annotations must share one page")
        for annotation_ref in (cloud_ref, callout_ref):
            if annotation_ref in paired_annotations:
                errors.append(f"{prefix} reuses an annotation from another pair")
            paired_annotations.add(annotation_ref)
        if pair.get("markup_claim_only") is not True:
            errors.append(f"{prefix} must remain a markup claim only")
        if pair.get("target_identity_ref") is not None:
            errors.append(f"{prefix}.target_identity_ref must remain unresolved")
        if pair.get("confirmed_clash") is not False or pair.get("quantity_eligible") is not False:
            errors.append(f"{prefix} cannot establish a clash or quantity")
        evidence = pair.get("evidence")
        if not isinstance(evidence, Mapping):
            errors.append(f"{prefix}.evidence must be an object")
        elif relationship == "unique_geometric_leader_target" and evidence.get("mutual_unique") is not True:
            errors.append(f"{prefix} geometric evidence must be mutually unique")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("summary must be an object")
    else:
        expected_summary = {
            "page_count": len(pages),
            "annotation_count": len(annotations),
            "annotation_pair_count": len(pairs),
            "pairing_abstention_count": len(abstentions),
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                errors.append(f"summary.{key} must be {expected}")

    contract = payload.get("exchange_contract")
    required_false = (
        "route_identity_or_connectivity_established",
        "confirmed_clash_or_clearance_established",
        "quantity_eligible",
        "schedule_values_used",
    )
    required_true = (
        "page_transforms_are_explicit",
        "source_pdf_objects_are_preserved",
        "annotation_observations_are_immutable",
        "annotation_pairs_are_markup_claims_only",
        "sheet_role_and_native_raster_route_are_deferred_to_m1",
        "target_grounding_is_deferred_to_m6",
    )
    if not isinstance(contract, Mapping):
        errors.append("exchange_contract must be an object")
    else:
        for key in required_true:
            if contract.get(key) is not True:
                errors.append(f"exchange_contract.{key} must be true")
        for key in required_false:
            if contract.get(key) is not False:
                errors.append(f"exchange_contract.{key} must be false")

    if source_pdf is not None:
        source_path = Path(source_pdf).resolve()
        if not source_path.is_file():
            errors.append("source PDF does not exist")
        else:
            if _file_sha256(source_path) != source_hash:
                errors.append("source PDF sha256 does not match bundle")
            if source_path.stat().st_size != document.get("source_bytes"):
                errors.append("source PDF byte size does not match bundle")
            with fitz.open(source_path) as source_document:
                source_annotation_count = sum(
                    len(list(page.annots() or [])) for page in source_document
                )
                if source_document.page_count != document.get("page_count"):
                    errors.append("source PDF page count does not match bundle")
                if source_annotation_count != len(annotations):
                    errors.append("source PDF annotation count does not match bundle")
    return errors
