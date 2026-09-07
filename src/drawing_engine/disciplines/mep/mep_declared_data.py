"""M7C independent document-region roles and declared MEP fields.

The adapter consumes M1-bound native document-region observations only.  It
never reads M4, M5, M7A, or M7B calculated records.  A heading can establish a
document-region role, but declaration fields are extracted only inside an
explicitly grouped or explicitly complete region.  Heading-only regions
preserve unresolved body coverage and cannot establish field absence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import fitz

from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_independent_declared_data"
REGION_OBSERVATION_LAYER = "mep_document_region_observations"
REGION_ROLES = ("schedule", "table", "note", "legend", "detail", "cut_sheet")

_ROLE_PATTERNS = {
    "schedule": re.compile(r"\b(?:EQUIPMENT|FIXTURE|VALVE|DAMPER|MEP)?\s*SCHEDULE\b", re.I),
    "table": re.compile(r"\bTABLE\b", re.I),
    "note": re.compile(r"\b(?:DRAWING|GENERAL|MEP)?\s*NOTES?\b", re.I),
    "legend": re.compile(r"\b(?:ABBREVIATIONS?|LEGEND|SYMBOLS?)\b", re.I),
    "detail": re.compile(r"\bDETAIL\b", re.I),
    "cut_sheet": re.compile(r"\b(?:CUT\s*SHEET|SUBMITTAL|PRODUCT\s+DATA|TECHNICAL\s+DATA)\b", re.I),
}
_FIELD_PATTERNS = {
    "manufacturer": re.compile(r"^\s*(?:MANUFACTURER|MFR)\s*[:#-]\s*(?P<value>.+?)\s*$", re.I),
    "model_number": re.compile(r"^\s*(?:MODEL(?:\s+(?:NO\.?|NUMBER))?)\s*[:#-]\s*(?P<value>.+?)\s*$", re.I),
    "specification_reference": re.compile(r"^\s*(?:SPECIFICATION|SPEC(?:IFICATION)?\s*(?:SECTION|REF(?:ERENCE)?)?)\s*[:#-]\s*(?P<value>.+?)\s*$", re.I),
    "mounting_type": re.compile(r"^\s*(?:MOUNTING(?:\s+TYPE)?|MOUNT)\s*[:#-]\s*(?P<value>.+?)\s*$", re.I),
    "quantity": re.compile(r"^\s*(?:QTY|QUANTITY)\s*[:#-]\s*(?P<value>\d+(?:\.\d+)?)\s*$", re.I),
}
_FORBIDDEN_KEYS = {
    "calculated_count",
    "calculated_quantity",
    "physical_count",
    "installed_length",
    "installed_length_m",
    "projected_length",
    "projected_length_m",
    "approved_quantity",
}


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id(kind: str, *parts: object) -> str:
    return f"{kind}.{_sha256(parts)[:20]}"


def _normalise_text(value: object) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split())


def _role_candidates(text: str) -> list[str]:
    return [role for role in REGION_ROLES if _ROLE_PATTERNS[role].search(text)]


def _bbox_union(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    boxes = [row.get("bbox_display") for row in rows]
    if not boxes or any(not isinstance(box, list) or len(box) != 4 for box in boxes):
        return None
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _region_groups(
    observations: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        group_ref = row.get("region_group_ref")
        if group_ref:
            by_group.setdefault(str(group_ref), []).append(row)
    groups = []
    used_groups = set()
    for anchor in observations:
        if not _role_candidates(_normalise_text(anchor.get("text"))):
            continue
        group_ref = anchor.get("region_group_ref")
        if group_ref:
            group_key = str(group_ref)
            if group_key in used_groups:
                continue
            used_groups.add(group_key)
            members = by_group[group_key]
        else:
            members = [anchor]
        groups.append((anchor, members))
    return groups


def extract_mep_document_region_observations(
    *, pdf_path: Path | str, sheet_registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract heading-bearing native text blocks without interpreting fields."""

    upstream_errors = validate_mep_sheet_registry(sheet_registry)
    if upstream_errors:
        raise ValueError("invalid M1 registry:\n" + "\n".join(upstream_errors))
    source_path = Path(pdf_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = _file_sha256(source_path)
    if source_sha256 != sheet_registry.get("document", {}).get("source_pdf_sha256"):
        raise ValueError("source PDF does not match the frozen M1 registry")
    pages_by_number = {
        int(row["page_number"]): row for row in sheet_registry.get("pages", [])
    }
    observations = []
    pages = []
    with fitz.open(source_path) as document:
        if document.page_count != len(pages_by_number):
            raise ValueError("source PDF page count does not match M1")
        for index, page in enumerate(document):
            page_number = index + 1
            page_scope = pages_by_number[page_number]
            page_ref = str(page_scope["page_ref"])
            refs = []
            for block in page.get_text("blocks"):
                text = str(block[4] or "").strip()
                candidates = _role_candidates(_normalise_text(text))
                if not candidates:
                    continue
                bbox = [round(float(value), 6) for value in block[:4]]
                identifier = _id(
                    "mep_document_region_observation",
                    sheet_registry.get("document", {}).get("document_key"),
                    page_ref,
                    bbox,
                    text,
                )
                observations.append({
                    "record_type": "mep_document_region_observation",
                    "record_version": SCHEMA_VERSION,
                    "id": identifier,
                    "page_ref": page_ref,
                    "page_number": page_number,
                    "text": text,
                    "bbox_display": bbox,
                    "role_candidates": candidates,
                    "region_group_ref": None,
                    "region_text_complete": False,
                    "method": "native_pdf_text_block_heading_scan",
                    "epistemic_state": "observed",
                    "quantity_eligible": False,
                })
                refs.append(identifier)
            pages.append({
                "page_ref": page_ref,
                "pdf_page_number": page_number,
                "observation_refs": sorted(refs),
                "observation_count": len(refs),
                "quantity_eligible": False,
            })
    observations.sort(key=lambda row: (row["page_number"], row["id"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": REGION_OBSERVATION_LAYER,
        "document": deepcopy(dict(sheet_registry.get("document", {}))),
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _sha256(sheet_registry),
        },
        "pages": pages,
        "observations": observations,
        "summary": {
            "page_count": len(pages),
            "observation_count": len(observations),
            "role_candidate_counts": dict(sorted(Counter(
                role for row in observations for role in row["role_candidates"]
            ).items())),
        },
        "exchange_contract": {
            "observation_layer_only": True,
            "native_text_primary": True,
            "region_role_not_yet_accepted": True,
            "declared_fields_not_yet_extracted": True,
            "calculated_records_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_document_region_observations(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def validate_mep_document_region_observations(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != REGION_OBSERVATION_LAYER:
        errors.append("layer mismatch")
    digest = payload.get("m1_contract_ref", {}).get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("m1_contract_ref.payload_sha256 must be a SHA-256 digest")
    pages = list(payload.get("pages", []))
    if [row.get("pdf_page_number") for row in pages] != list(range(1, len(pages) + 1)):
        errors.append("region observation pages must be complete and ordered")
    page_refs = {str(row.get("page_ref")) for row in pages}
    observations = list(payload.get("observations", []))
    observation_ids = {str(row.get("id")) for row in observations}
    if len(observation_ids) != len(observations):
        errors.append("region observation IDs must be unique")
    for row in observations:
        identifier = str(row.get("id"))
        if str(row.get("page_ref")) not in page_refs:
            errors.append(f"{identifier}: unknown page")
        candidates = list(row.get("role_candidates", []))
        if any(role not in REGION_ROLES for role in candidates):
            errors.append(f"{identifier}: invalid role candidates")
        replayed_candidates = _role_candidates(_normalise_text(row.get("text")))
        if replayed_candidates != candidates:
            errors.append(f"{identifier}: role candidates do not replay")
        if not candidates and not row.get("region_group_ref"):
            errors.append(f"{identifier}: non-heading observation lacks a region group")
        bbox = row.get("bbox_display")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in bbox)
        ):
            errors.append(f"{identifier}: invalid display bbox")
        if row.get("quantity_eligible") is not False:
            errors.append(f"{identifier}: observation cannot be quantity eligible")
    memberships = sorted(str(ref) for page in pages for ref in page.get("observation_refs", []))
    if memberships != sorted(observation_ids):
        errors.append("page-to-observation membership mismatch")
    if any(page.get("observation_count") != len(page.get("observation_refs", [])) for page in pages):
        errors.append("page observation count mismatch")
    expected_summary = {
        "page_count": len(pages),
        "observation_count": len(observations),
        "role_candidate_counts": dict(sorted(Counter(
            role for row in observations for role in row.get("role_candidates", [])
        ).items())),
    }
    if payload.get("summary") != expected_summary:
        errors.append("summary mismatch")
    contract = payload.get("exchange_contract", {})
    for key in (
        "observation_layer_only",
        "native_text_primary",
        "region_role_not_yet_accepted",
        "declared_fields_not_yet_extracted",
    ):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in ("calculated_records_used", "quantity_eligible"):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    if payload.get("quantity_eligible") is not False:
        errors.append("region observation payload cannot be quantity eligible")
    return errors


def _build_regions(
    sheet_registry: Mapping[str, Any],
    source_observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations_by_page: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_observations:
        observations_by_page.setdefault(str(row.get("page_ref")), []).append(row)
    output = []
    for page in sorted(
        sheet_registry.get("pages", []), key=lambda row: int(row.get("page_number", 0))
    ):
        page_ref = str(page.get("page_ref"))
        for anchor, members in _region_groups(observations_by_page.get(page_ref, [])):
            members = sorted(
                members,
                key=lambda row: (
                    float((row.get("bbox_display") or [0, 0])[1]),
                    float((row.get("bbox_display") or [0])[0]),
                    str(row.get("id")),
                ),
            )
            candidates = _role_candidates(_normalise_text(anchor.get("text")))
            explicit_group = bool(anchor.get("region_group_ref"))
            content_complete = explicit_group or anchor.get("region_text_complete") is True
            region_id = _id(
                "mep_declaration_region",
                page_ref,
                [str(row.get("id")) for row in members],
                candidates,
            )
            output.append({
                "record_type": "mep_document_region",
                "record_version": SCHEMA_VERSION,
                "id": region_id,
                "page_ref": page_ref,
                "pdf_page_number": int(page.get("page_number")),
                "drawing_sheet_number": page.get("fields", {}).get("sheet_number", {}).get("value"),
                "role": candidates[0] if len(candidates) == 1 else None,
                "role_candidates": candidates,
                "role_state": "accepted" if len(candidates) == 1 else "abstained",
                "bbox_display": _bbox_union(members),
                "text": "\n".join(str(row.get("text") or "") for row in members),
                "source_text_observation_refs": [str(row.get("id")) for row in members],
                "source_methods": sorted(set(str(row.get("method")) for row in members)),
                "content_scope": (
                    "explicit_region_group"
                    if explicit_group
                    else "explicit_complete_text_region"
                    if content_complete
                    else "heading_only_unresolved"
                ),
                "declaration_content_complete": content_complete,
                "epistemic_state": "derived",
                "reasons": (
                    []
                    if len(candidates) == 1 and content_complete
                    else ["multiple_document_region_roles"]
                    if len(candidates) != 1
                    else ["region_body_scope_not_established"]
                ),
                "quantity_eligible": False,
            })
    return sorted(output, key=lambda row: (row["pdf_page_number"], row["id"]))


def _declared_records(regions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for region in regions:
        if region.get("role_state") != "accepted" or region.get(
            "declaration_content_complete"
        ) is not True:
            continue
        for line_index, raw_line in enumerate(str(region.get("text") or "").splitlines()):
            line = _normalise_text(raw_line)
            for field_name, pattern in _FIELD_PATTERNS.items():
                match = pattern.fullmatch(line)
                if match is None:
                    continue
                raw_value = match.group("value").strip()
                if field_name == "quantity":
                    numeric = float(raw_value)
                    value: object = int(numeric) if numeric.is_integer() else numeric
                    unit = "ea"
                else:
                    value = raw_value
                    unit = None
                output.append({
                    "record_type": "mep_declared_field",
                    "record_version": SCHEMA_VERSION,
                    "id": _id(
                        "mep_declared_field",
                        region.get("id"),
                        line_index,
                        field_name,
                        value,
                    ),
                    "region_ref": str(region.get("id")),
                    "page_ref": str(region.get("page_ref")),
                    "pdf_page_number": int(region.get("pdf_page_number")),
                    "drawing_sheet_number": region.get("drawing_sheet_number"),
                    "region_role": region.get("role"),
                    "field_name": field_name,
                    "value": value,
                    "unit": unit,
                    "raw_text": raw_line,
                    "value_channel": "declared",
                    "epistemic_state": "direct",
                    "evidence_refs": [
                        str(region.get("id")),
                        *[str(ref) for ref in region.get("source_text_observation_refs", [])],
                    ],
                    "calculated_value_used": False,
                    "reviewed_value": None,
                    "quantity_eligible": False,
                })
    return sorted(output, key=lambda row: (row["pdf_page_number"], row["id"]))


def build_mep_declared_data(
    *,
    sheet_registry: Mapping[str, Any],
    region_observations: Mapping[str, Any],
) -> dict[str, Any]:
    """Build M7C solely from frozen M1-bound region observations."""

    upstream_errors = validate_mep_sheet_registry(sheet_registry)
    upstream_errors.extend(
        f"regions: {error}"
        for error in validate_mep_document_region_observations(region_observations)
    )
    if upstream_errors:
        raise ValueError("invalid M7C upstream:\n" + "\n".join(upstream_errors))
    if region_observations.get("document", {}).get("document_key") != sheet_registry.get(
        "document", {}
    ).get("document_key"):
        raise ValueError("M1 and region observation document keys do not match")
    if region_observations.get("m1_contract_ref", {}).get("payload_sha256") != _sha256(
        sheet_registry
    ):
        raise ValueError("region observations do not bind the supplied M1 registry")
    regions = _build_regions(
        sheet_registry, list(region_observations.get("observations", []))
    )
    declarations = _declared_records(regions)
    region_refs_by_page: dict[str, list[str]] = {}
    declaration_refs_by_page: dict[str, list[str]] = {}
    for region in regions:
        region_refs_by_page.setdefault(str(region["page_ref"]), []).append(str(region["id"]))
    for record in declarations:
        declaration_refs_by_page.setdefault(str(record["page_ref"]), []).append(str(record["id"]))
    pages = []
    for page in sorted(
        sheet_registry.get("pages", []), key=lambda row: int(row.get("page_number", 0))
    ):
        page_ref = str(page.get("page_ref"))
        region_refs = sorted(region_refs_by_page.get(page_ref, []))
        declaration_refs = sorted(declaration_refs_by_page.get(page_ref, []))
        pages.append({
            "record_type": "mep_declaration_page_coverage",
            "record_version": SCHEMA_VERSION,
            "page_ref": page_ref,
            "pdf_page_number": int(page.get("page_number")),
            "drawing_sheet_number": page.get("fields", {}).get("sheet_number", {}).get("value"),
            "sheet_role": page.get("role"),
            "region_refs": region_refs,
            "declared_field_refs": declaration_refs,
            "coverage_state": (
                "not_applicable_divider"
                if page.get("role") == "divider"
                else "declared_fields_observed"
                if declaration_refs
                else "recognized_regions_no_scoped_declarations"
                if region_refs
                else "no_recognized_declaration_region"
            ),
            "document_declaration_completeness_established": False,
            "quantity_eligible": False,
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(sheet_registry.get("document", {}))),
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _sha256(sheet_registry),
        },
        "region_observation_contract_ref": {
            "layer": region_observations.get("layer"),
            "schema_version": region_observations.get("schema_version"),
            "payload_sha256": _sha256(region_observations),
        },
        "region_role_contract": {
            "roles": list(REGION_ROLES),
            "unique_heading_role_required": True,
            "explicit_region_body_scope_required_for_field_extraction": True,
            "heading_only_does_not_establish_field_absence": True,
        },
        "pages": pages,
        "document_regions": regions,
        "declared_records": declarations,
        "summary": {
            "page_count": len(pages),
            "recognized_region_count": len(regions),
            "accepted_role_count": sum(row["role_state"] == "accepted" for row in regions),
            "complete_content_region_count": sum(row["declaration_content_complete"] for row in regions),
            "role_counts": dict(sorted(Counter(row["role"] for row in regions if row["role"]).items())),
            "declared_record_count": len(declarations),
            "declared_field_counts": dict(sorted(Counter(row["field_name"] for row in declarations).items())),
        },
        "negative_fixture_result": {
            "no_scoped_declared_fields_observed": len(declarations) == 0,
            "document_field_absence_established": False,
            "reason": (
                "recognized_headings_lack_explicit_complete_region_bodies"
                if not declarations
                else None
            ),
        },
        "exchange_contract": {
            "declared_channel_only": True,
            "calculated_records_used": False,
            "m4_or_m5_identity_used": False,
            "declared_quantity_is_not_calculated_quantity": True,
            "document_absence_not_inferred_from_zero_records": True,
            "installed_length_emitted": False,
            "reconciliation_emitted": False,
            "approved_for_quote": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_declared_data(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def validate_mep_declared_data(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    digest = payload.get("m1_contract_ref", {}).get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("m1_contract_ref.payload_sha256 must be a SHA-256 digest")
    region_digest = payload.get("region_observation_contract_ref", {}).get("payload_sha256")
    if not isinstance(region_digest, str) or len(region_digest) != 64:
        errors.append("region_observation_contract_ref.payload_sha256 must be a SHA-256 digest")

    def walk(value: object, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                location = f"{path}.{key}" if path else str(key)
                if key in _FORBIDDEN_KEYS:
                    errors.append(f"{location}: forbidden M7C field")
                walk(child, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload)
    pages = list(payload.get("pages", []))
    if [row.get("pdf_page_number") for row in pages] != list(range(1, len(pages) + 1)):
        errors.append("M7C pages must be complete and ordered")
    page_refs = {str(row.get("page_ref")) for row in pages}
    regions = list(payload.get("document_regions", []))
    region_ids = {str(row.get("id")) for row in regions}
    if len(region_ids) != len(regions):
        errors.append("document region IDs must be unique")
    for row in regions:
        identifier = str(row.get("id"))
        if str(row.get("page_ref")) not in page_refs:
            errors.append(f"{identifier}: unknown page")
        candidates = list(row.get("role_candidates", []))
        if any(role not in REGION_ROLES for role in candidates):
            errors.append(f"{identifier}: unsupported region role")
        if row.get("role_state") == "accepted":
            if len(candidates) != 1 or row.get("role") != candidates[0]:
                errors.append(f"{identifier}: accepted role is not unique")
        elif row.get("role_state") != "abstained":
            errors.append(f"{identifier}: invalid role state")
        if row.get("content_scope") == "heading_only_unresolved" and row.get(
            "declaration_content_complete"
        ) is not False:
            errors.append(f"{identifier}: heading-only content cannot be complete")
        if row.get("quantity_eligible") is not False:
            errors.append(f"{identifier}: region cannot be quantity eligible")
    declarations = list(payload.get("declared_records", []))
    declaration_ids = {str(row.get("id")) for row in declarations}
    if len(declaration_ids) != len(declarations):
        errors.append("declared record IDs must be unique")
    region_by_id = {str(row.get("id")): row for row in regions}
    for row in declarations:
        identifier = str(row.get("id"))
        region = region_by_id.get(str(row.get("region_ref")))
        if region is None:
            errors.append(f"{identifier}: unknown region")
        elif region.get("role_state") != "accepted" or region.get(
            "declaration_content_complete"
        ) is not True:
            errors.append(f"{identifier}: declaration came from an incomplete region")
        if row.get("field_name") not in _FIELD_PATTERNS:
            errors.append(f"{identifier}: unsupported declared field")
        if row.get("value_channel") != "declared":
            errors.append(f"{identifier}: value channel must be declared")
        if row.get("field_name") == "quantity":
            value = row.get("value")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
                or row.get("unit") != "ea"
            ):
                errors.append(f"{identifier}: declared quantity is invalid")
        elif not isinstance(row.get("value"), str) or not row.get("value"):
            errors.append(f"{identifier}: declared text value is invalid")
        if row.get("calculated_value_used") is not False:
            errors.append(f"{identifier}: calculated value entered M7C")
        if row.get("reviewed_value") is not None:
            errors.append(f"{identifier}: reviewed value must remain null")
        if row.get("quantity_eligible") is not False:
            errors.append(f"{identifier}: declared record cannot be quote eligible")
    page_region_membership = sorted(
        str(ref) for page in pages for ref in page.get("region_refs", [])
    )
    page_declaration_membership = sorted(
        str(ref) for page in pages for ref in page.get("declared_field_refs", [])
    )
    if page_region_membership != sorted(region_ids):
        errors.append("page-to-region membership mismatch")
    if page_declaration_membership != sorted(declaration_ids):
        errors.append("page-to-declaration membership mismatch")
    expected_summary = {
        "page_count": len(pages),
        "recognized_region_count": len(regions),
        "accepted_role_count": sum(row.get("role_state") == "accepted" for row in regions),
        "complete_content_region_count": sum(row.get("declaration_content_complete") is True for row in regions),
        "role_counts": dict(sorted(Counter(row.get("role") for row in regions if row.get("role")).items())),
        "declared_record_count": len(declarations),
        "declared_field_counts": dict(sorted(Counter(row.get("field_name") for row in declarations).items())),
    }
    if payload.get("summary") != expected_summary:
        errors.append("summary mismatch")
    negative = payload.get("negative_fixture_result", {})
    if negative.get("no_scoped_declared_fields_observed") is not (len(declarations) == 0):
        errors.append("negative fixture result mismatch")
    if negative.get("document_field_absence_established") is not False:
        errors.append("M7C cannot infer document field absence")
    contract = payload.get("exchange_contract", {})
    for key in (
        "declared_channel_only",
        "declared_quantity_is_not_calculated_quantity",
        "document_absence_not_inferred_from_zero_records",
    ):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in (
        "calculated_records_used",
        "m4_or_m5_identity_used",
        "installed_length_emitted",
        "reconciliation_emitted",
        "approved_for_quote",
        "quantity_eligible",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    if payload.get("quantity_eligible") is not False:
        errors.append("M7C payload must remain quantity ineligible")
    return errors
