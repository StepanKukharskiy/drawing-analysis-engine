"""Native full-document text observations and a fail-closed M2 handoff.

Only observed native block membership and frozen M1 title evidence supply
region hints here. Unknown regions remain unresolved; this is neither complete
declaration-body recovery nor item discovery. Invalid lines are quarantined
with their original evidence while unrelated lines can reach the M2 validator.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _role_candidates, _sha256
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import (
    _normalise_text,
    _stable_id,
    build_mep_terminology_proposals,
    interpret_mep_text,
    validate_mep_terminology_proposals,
)


LAYER = "mep_full_document_text_observations"
SCHEMA_VERSION = "0.1.0"


def extract_mep_text_observations(
    *, pdf_path: Path | str, sheet_registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve every nonempty native line, span reference and page coverage."""

    errors = validate_mep_sheet_registry(sheet_registry)
    if errors:
        raise ValueError("invalid M1 registry:\n" + "\n".join(errors))
    source_path = Path(pdf_path).resolve()
    source_hash = _file_sha256(source_path)
    if source_hash != sheet_registry.get("document", {}).get("source_pdf_sha256"):
        raise ValueError("source PDF does not match the frozen M1 registry")
    scopes = {row["page_number"]: row for row in sheet_registry["pages"]}
    selected = set(sheet_registry.get("processing_scope", {}).get("source_page_numbers", scopes))
    if not selected or not selected <= scopes.keys():
        raise ValueError("invalid native text page scope")
    observations, pages = [], []
    with fitz.open(source_path) as pdf:
        if sorted(scopes) != list(range(1, len(pdf) + 1)):
            raise ValueError("source PDF pages do not match M1")
        for page_number, page in enumerate(pdf, 1):
            scope = scopes[page_number]
            title_refs = set(scope.get("fields", {}).get("title", {}).get("evidence_refs", []))
            titles = [row for row in scope.get("text_observations", []) if row["id"] in title_refs]
            rows = []
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"] if page_number in selected else []
            for block in blocks:
                if block.get("type") != 0:
                    continue
                block_ref = f"page[{page_number}].text_block[{block['number']}]"
                block_text = "\n".join(
                    " ".join(span["text"] for span in line["spans"])
                    for line in block.get("lines", [])
                )
                # These are exclusion hints, not claims that a region body is
                # complete. Do not spread a heading across neighboring blocks.
                block_roles = [role for role in _role_candidates(block_text) if role != "detail"]
                for line_number, line in enumerate(block.get("lines", [])):
                    text = " ".join(span["text"] for span in line["spans"]).strip()
                    if not text:
                        continue
                    native_ref = f"{block_ref}.line[{line_number}]"
                    identifier = _stable_id("mep_native_text_line", source_hash, native_ref, text)
                    bbox_pdf = list(line["bbox"])
                    bbox_display = list(fitz.Rect(bbox_pdf) * page.rotation_matrix)
                    matched_titles = [
                        row["id"] for row in titles
                        if _normalise_text(row["text"]) in _normalise_text(text)
                        and fitz.Rect(row["bbox_display"]).intersects(fitz.Rect(bbox_display))
                    ]
                    region_role = next(
                        (role for role in ("schedule", "table", "cut_sheet", "legend", "note") if role in block_roles),
                        "title" if matched_titles else "unknown",
                    )
                    observation = {
                        "record_type": "mep_native_text_line", "record_version": SCHEMA_VERSION,
                        "id": identifier, "page_ref": scope["page_ref"], "page_number": page_number,
                        "text": text, "bbox_pdf": bbox_pdf, "bbox_display": bbox_display,
                        "source_native_ref": native_ref, "source_pdf_sha256": source_hash,
                        "native_spans": [
                            {
                                "source_native_ref": f"{native_ref}.span[{index}]",
                                "text": span["text"], "bbox_pdf": list(span["bbox"]),
                                "bbox_display": list(fitz.Rect(span["bbox"]) * page.rotation_matrix),
                            }
                            for index, span in enumerate(line["spans"])
                        ],
                        "interpretation_scope_ref": identifier,
                        "evidence_channels": ["native_pdf_text"],
                        "method": "native_pdf_text", "confidence": 1.0,
                        "region_role": region_role,
                        "region_role_evidence": {
                            "native_block_ref": block_ref, "block_role_candidates": block_roles,
                            "m1_title_evidence_refs": matched_titles,
                            "region_body_complete": False,
                        },
                        "epistemic_state": "observed", "quantity_eligible": False,
                    }
                    try:
                        observation["text_interpretation"] = interpret_mep_text(observation)
                    except (ValueError, ZeroDivisionError, OverflowError) as error:
                        observation["text_interpretation"] = None
                        observation["interpretation_error"] = f"{type(error).__name__}: {error}"
                    rows.append(observation)
            observations.extend(rows)
            pages.append({
                "page_ref": scope["page_ref"], "page_number": page_number,
                "observation_refs": [row["id"] for row in rows],
                "native_text_line_count": len(rows),
                "source_page_rotation_degrees": page.rotation,
                "pdf_to_display_matrix": list(page.rotation_matrix),
                "quality_route": deepcopy(scope["quality_route"]),
                "native_text_scan_complete": page_number in selected, "ocr_run": False,
                "region_body_coverage_complete": False, "item_inventory_complete": False,
                "quantity_eligible": False,
            })
    return {
        "schema_version": SCHEMA_VERSION, "layer": LAYER,
        "document": deepcopy(sheet_registry["document"]),
        "m1_payload_sha256": _sha256(sheet_registry),
        "extractor": {"name": "pymupdf_native_line_scan", "version": "1.0.0", "engine_version": fitz.VersionBind},
        "pages": pages, "observations": observations,
        "summary": {
            "page_count": len(pages), "observation_count": len(observations),
            "text_role_counts": dict(sorted(Counter(
                (row["text_interpretation"] or {}).get("text_role", "invalid") for row in observations
            ).items())),
        },
        "exchange_contract": {
            "native_observations_only": True, "all_source_pages_preserved": True,
            "region_body_coverage_complete": False, "item_inventory_complete": False,
            "declared_fields_used": False, "quantity_eligible": False,
        },
    }


def build_mep_document_text_proposals(observations: Mapping[str, Any]) -> dict[str, Any]:
    """Validate each line and preserve invalid evidence outside M2 proposals."""

    if observations.get("layer") != LAYER or observations.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid full-document text observation contract")
    retained = deepcopy(list(observations["observations"]))
    pages = observations["pages"]
    page_refs = {row["page_ref"] for row in pages}
    if (len(page_refs) != len(pages)
            or [row.get("page_number") for row in pages] != list(range(1, observations["document"]["page_count"] + 1))):
        raise ValueError("full-document page coverage does not close")
    page_owners = {ref: page["page_ref"] for page in pages for ref in page["observation_refs"]}
    refs = [row["id"] for row in retained]
    covered_refs = [ref for page in pages for ref in page["observation_refs"]]
    if len(set(refs)) != len(refs) or sorted(refs) != sorted(covered_refs):
        raise ValueError("native observation IDs or page coverage do not close")
    if any(row.get("native_text_line_count") != len(row["observation_refs"]) for row in pages):
        raise ValueError("page text coverage count does not replay")
    valid, diagnostics = [], []
    for row in retained:
        errors = []
        if row.get("page_ref") not in page_refs:
            errors.append("unregistered page reference")
        if page_owners.get(row["id"]) != row.get("page_ref"):
            errors.append("observation page ownership mismatch")
        if row.get("source_pdf_sha256") != observations["document"]["source_pdf_sha256"]:
            errors.append("source PDF hash mismatch")
        if row.get("quantity_eligible") is not False or row.get("epistemic_state") != "observed":
            errors.append("native observation acquired engineering authority")
        for key in ("bbox_pdf", "bbox_display"):
            box = row.get(key)
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
                    or box[2] < box[0] or box[3] < box[1]):
                errors.append(f"invalid {key} provenance")
        if not row.get("source_native_ref") or not row.get("native_spans"):
            errors.append("missing native line/span provenance")
        one = None
        try:
            if row.get("text_interpretation") != interpret_mep_text(row):
                errors.append("text role/unit interpretation does not replay")
            one = build_mep_terminology_proposals(document=observations["document"], observations=[row])
            errors.extend(validate_mep_terminology_proposals(one))
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            errors.append(f"{type(error).__name__}: {error}")
        if errors:
            diagnostics.append({
                "observation_ref": row["id"], "page_ref": row.get("page_ref"),
                "stage": "m2_text_interpretation_validation", "state": "quarantined",
                "errors": errors, "diagnostic_proposals": one["proposals"] if one else [],
                "downstream_acceptance_closed": True, "quantity_eligible": False,
            })
        else:
            valid.append(row)
    payload = build_mep_terminology_proposals(document=observations["document"], observations=valid)
    # Quarantine never deletes the source observation. Invalid proposals live
    # solely in diagnostics, never in the canonical proposal channel.
    payload["source_observations"] = retained
    payload["summary"]["source_observation_count"] = len(retained)
    payload["observation_diagnostics"] = diagnostics
    payload["document_text_observation_ref"] = {"layer": LAYER, "payload_sha256": _sha256(observations)}
    errors = validate_mep_terminology_proposals(payload)
    if errors:
        raise ValueError("invalid M2 exchange:\n" + "\n".join(errors))
    return payload
