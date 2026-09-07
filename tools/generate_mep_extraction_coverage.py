#!/usr/bin/env python3
"""Combine extraction coverage, without joining declarations to drawing facts.

This is a diagnostic job manifest, not an item catalog or recall measurement.
Missing stages remain not_processed; an empty extracted channel proves no absence.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_item_ocr_observations import page_ocr_selection, validate_mep_item_ocr_observations
from src.drawing_engine.disciplines.mep.mep_native_declaration_bodies import validate_mep_native_declaration_bodies
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import validate_native_target_discovery
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry
from src.drawing_engine.disciplines.mep.mep_text_observations import build_mep_document_text_proposals


def build_coverage(*, registry, native_text, native_targets=None, item_ocr=None, declaration_bodies=None):
    errors = validate_mep_sheet_registry(registry)
    if errors:
        raise ValueError("invalid registry: " + "; ".join(errors))
    stages = {"native_text": native_text, "native_targets": native_targets,
              "item_ocr": item_ocr, "independent_declarations": declaration_bodies}
    registry_hash = _sha256(registry)
    expected_pages = [(row["page_ref"], row["page_number"]) for row in registry["pages"]]
    for name, payload in stages.items():
        if payload is None:
            continue
        if payload.get("document") != registry["document"]:
            raise ValueError(name + ": source document mismatch")
        upstream = payload.get("m1_payload_sha256", payload.get("input_hashes", {}).get("m1"))
        if upstream != registry_hash:
            raise ValueError(name + ": M1 payload mismatch")
        if [(row["page_ref"], row["page_number"]) for row in payload["pages"]] != expected_pages:
            raise ValueError(name + ": registered page membership mismatch")
    text_proposals = build_mep_document_text_proposals(native_text)
    for payload, validate in ((native_targets, validate_native_target_discovery),
                              (item_ocr, validate_mep_item_ocr_observations),
                              (declaration_bodies, validate_mep_native_declaration_bodies)):
        if payload is not None:
            errors = validate(payload)
            if errors:
                raise ValueError("invalid extraction stage: " + "; ".join(errors))
    if native_targets is not None and native_targets["input_hashes"]["native_text"] != _sha256(native_text):
        raise ValueError("native target text input mismatch")
    if item_ocr is not None and item_ocr["native_text_payload_sha256"] != _sha256(native_text):
        raise ValueError("OCR native text input mismatch")

    indexed = {name: {row["page_ref"]: row for row in payload["pages"]} if payload is not None else {}
               for name, payload in stages.items()}
    invalid_text = Counter(row["page_ref"] for row in text_proposals["observation_diagnostics"])
    ocr_states = {}
    for row in (item_ocr or {}).get("observations", []):
        ocr_states.setdefault(row["page_ref"], Counter())[row["handoff_state"]] += 1
    summaries = {name: {"state": "not_processed"} for name in stages}
    summaries["native_text"] = {"page_count": len(native_text["pages"]),
                                "observation_count": len(native_text["observations"]),
                                "invalid_observation_count": len(text_proposals["observation_diagnostics"])}
    if native_targets is not None:
        summaries["native_targets"] = {"page_count": len(native_targets["pages"]),
            "search_count": len(native_targets["searches"]),
            "primitive_candidate_count": len(native_targets["primitive_candidates"]),
            "budget_limited_search_count": sum(row["state"] == "budget_limited" for row in native_targets["searches"])}
    if item_ocr is not None:
        summaries["item_ocr"] = {"page_count": len(item_ocr["pages"]),
            "observation_count": len(item_ocr["observations"]),
            "selected_page_count": sum(row["selection"]["selected"] for row in item_ocr["pages"]),
            "handoff_state_counts": dict(Counter(row["handoff_state"] for row in item_ocr["observations"]))}
    if declaration_bodies is not None:
        # The independent body's validator replays this summary from its cells.
        summaries["independent_declarations"] = deepcopy(declaration_bodies["summary"])
    pages = []
    for scope in registry["pages"]:
        ref = scope["page_ref"]
        native = indexed["native_text"][ref]
        targets = indexed["native_targets"].get(ref)
        ocr = indexed["item_ocr"].get(ref)
        declared = indexed["independent_declarations"].get(ref)
        if ocr is not None:
            selection = page_ocr_selection(scope, minimum_native_chars_per_square_inch=item_ocr["configuration"]["minimum_native_chars_per_square_inch"])
            expected_state = ("not_selected_by_quality_gate" if not selection["selected"] else
                              "raster_scan_completed" if ocr["completed_crop_count"] == ocr["planned_crop_count"] else
                              "partial_or_failed_raster_scan")
            if (ocr["selection"] != selection or ocr["state"] != expected_state
                    or ocr["orientation_coverage"] != "horizontal_only"
                    or ocr["page_size_display"] != scope["page_size_display"]):
                raise ValueError("OCR selection/state/orientation does not replay")
        pages.append({
            "page_ref": ref, "page_number": scope["page_number"],
            "native_text": {"state": "processed_native_layer_only" if native.get("native_text_scan_complete") is True else "partial_native_scan", "line_count": native["native_text_line_count"],
                            "invalid_observation_count": invalid_text[ref], "region_applicability": "unresolved"},
            "native_target_search": {"state": targets["stage_states"]["annotation_seeded_target_search"],
                                     "search_count": len(targets["search_refs"]),
                                     "retained_primitive_count": len(targets["primitive_candidate_refs"]),
                                     "scope": targets["search_completeness_scope"]} if targets else {"state": "not_processed"},
            "item_text_ocr": {"state": ocr["state"], "raw_line_count": len(ocr["observation_refs"]),
                              "line_handoff_counts": dict(ocr_states.get(ref, {})),
                              "processed_raster_area_fraction": ocr["processed_raster_area_fraction"],
                              "orientation_coverage": ocr["orientation_coverage"]} if ocr else {"state": "not_processed"},
            "independent_declarations": {"state": declared["native_table_scan_state"],
                                         "bounded_body_count": declared["accepted_body_count"],
                                         "unresolved_heading_count": len(declared["unresolved_headings"]),
                                         "whole_document_completeness": False} if declared else {"state": "not_processed"},
            "remaining_stages": {stage: "not_processed" for stage in (
                "unlabelled_route_equipment_accessory_discovery", "leader_contact_and_port_evidence",
                "plan_detail_section_reference_resolution", "item_to_detail_applicability",
                "technical_note_and_specification_applicability", "rotated_and_mixed_content_ocr",
                "unruled_or_continued_declaration_bodies",
                "automatic_m4_binding", "automatic_3d_and_quantity_certificates")},
            "item_inventory_complete": False, "independent_reviewed_item_denominator": None,
            "item_recall": None, "certified_automatic_item_count": None,
            "unextracted_field_absence_established": False,
        })
    return {
        "schema_version": "0.1.0", "layer": "mep_extraction_coverage",
        "mode": "automatic_observation_diagnostics", "document": deepcopy(registry["document"]),
        "input_hashes": {"m1": registry_hash, **{name: _sha256(payload) if payload is not None else None for name, payload in stages.items()}},
        "pages": pages,
        "stage_summaries": summaries,
        "acceptance": {"full_package_extraction_complete": False, "full_marked_audit_ready": False,
                       "independently_reviewed_coverage_denominator_available": False,
                       "calculated_declared_channels_joined": False, "reviewed_selectors_used": False,
                       "quote_approval_established": False},
        "limitations": [
            "Native candidate strokes are alternatives, not detected physical items.",
            "OCR image-area coverage and text counts do not measure item recall.",
            "Bounded declaration bodies do not prove all document fields were extracted or absent.",
            "The assisted four-occurrence audit remains separate from this automatic observation run.",
            "Installed quantities and duplicate identity require existing downstream certificates.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--native-text", type=Path, required=True)
    parser.add_argument("--native-targets", type=Path)
    parser.add_argument("--item-ocr", type=Path)
    parser.add_argument("--declaration-bodies", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("registry", "native_text", "native_targets", "item_ocr", "declaration_bodies")}
    payloads, artifact_hashes = {}, {}
    for name, path in paths.items():
        raw = path.read_bytes() if path else None
        payloads[name] = json.loads(raw) if raw is not None else None
        artifact_hashes[name] = hashlib.sha256(raw).hexdigest() if raw is not None else None
    del raw
    if _file_sha256(args.source) != payloads["registry"]["document"]["source_pdf_sha256"]:
        raise ValueError("source PDF mismatch")
    coverage = build_coverage(**payloads)
    if any(path and _file_sha256(path) != artifact_hashes[name] for name, path in paths.items()):
        raise RuntimeError("extraction artifact changed during coverage assembly; rerun")
    coverage["artifacts"] = {name: {"path": str(path.resolve()), "sha256": artifact_hashes[name]} if path else None
                             for name, path in paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    lines = ["# MEP extraction coverage", "", "Automatic observation diagnostics; full package extraction and the full marked audit remain incomplete.", "",
             "Counts below are source observations, not physical quantities or recall.", "",
             "| PDF page | Native lines | Geometry search | Retained strokes | OCR lines / quarantined | Declaration bodies / unresolved headings |",
             "| --- | ---: | --- | ---: | --- | --- |"]
    for row in coverage["pages"]:
        target, ocr, declaration = row["native_target_search"], row["item_text_ocr"], row["independent_declarations"]
        ocr_display = (f"{ocr['raw_line_count']} / {ocr['line_handoff_counts'].get('quarantined', 0)}"
                       if ocr["state"] not in {"not_processed", "not_selected_by_quality_gate"}
                       else "not searched")
        lines.append(f"| {row['page_number']} | {row['native_text']['line_count']} | {target['state']} | "
                     f"{target.get('retained_primitive_count', 'not processed')} | {ocr_display} | "
                     f"{declaration.get('bounded_body_count', 'not processed')} / {declaration.get('unresolved_heading_count', 'not processed')} |")
    lines += ["", "Every page still requires complete unlabelled target discovery, leader/contact/port evidence, detail/section references, item applicability, automatic M4 binding and downstream certificates.", "",
              "OCR is quality-selected and horizontal-only. An unselected page has not been OCR searched. Declaration extraction is independent; zero recovered bodies does not prove absence.", "",
              "The existing partial PDF still shows the frozen assisted checkpoint. This report does not add accepted items to it.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
