#!/usr/bin/env python3
"""Generate the complete compact MEP source denominator for one sheet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_mep_observed_takeoff import _artifact_json
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _bounded_page_records
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    SourceDispositionPack, validate_source_denominator,
    write_source_disposition_pack,
)
from src.drawing_engine.project.project_packed_store import PackedProjectStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        stream.write(body)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _source_refs(value: Any) -> set[str]:
    """Collect explicit native source refs from one bounded evidence object."""
    output: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "source_primitive_ref" and isinstance(item, str):
                output.add(item)
            elif key == "source_primitive_refs" and isinstance(item, list):
                output.update(ref for ref in item if isinstance(ref, str))
            else:
                output.update(_source_refs(item))
    elif isinstance(value, list):
        for item in value:
            output.update(_source_refs(item))
    return output


def _role_refs(*, page_number: int, page_ref: str, route_graph: Mapping[str, Any],
               fitting_hypotheses: Mapping[str, Any], equipment: Mapping[str, Any],
               stroke_queries: Mapping[str, Any], ownership: Mapping[str, Any]) -> dict[str, set[str]]:
    route_page = next(row for row in route_graph["pages"] if row["page_ref"] == page_ref)
    legacy_route = {row["source_primitive_ref"] for row in route_page["fragments"]
                    if row.get("source_primitive_ref")}

    fitting = set()
    for row in fitting_hypotheses.get("fitting_hypotheses", []):
        if row.get("page_ref") != page_ref:
            continue
        # Input route fragments and drafting-gap continuations remain route
        # evidence.  Only visible fitting bodies/rings are claimed here.
        fitting.update(row.get("source_primitive_refs", []))
        fitting.update(_source_refs(row.get("body_candidates", [])))
        fitting.update(_source_refs(row.get("native_ring_candidates", [])))

    equipment_refs = set()
    for row in equipment.get("identities", []):
        if row.get("page_ref") != page_ref:
            continue
        equipment_refs.update(row.get("source_primitive_refs", []))
        for candidate in row.get("port_outcome", {}).get("port_geometry_candidates", []):
            # Contact strokes are route-side incidence evidence, not part of
            # the equipment body/port itself.
            equipment_refs.update(candidate.get("source_primitive_refs", []))
    for row in equipment.get("body_motifs", []):
        if row.get("page_ref") == page_ref:
            equipment_refs.update(row.get("source_primitive_refs", []))
            for component in row.get("components", []):
                equipment_refs.update(component.get("source_primitive_refs", []))
    fitting.update(equipment_refs)

    annotation = set()
    query_page = next(row for row in stroke_queries["pages"]
                      if row["page_number"] == page_number)
    for outcome in query_page.get("outcomes", []):
        for role in outcome.get("roles", []):
            if role.get("state") == "accepted":
                annotation.update(role.get("source_primitive_refs", []))

    furniture = set()
    for row in ownership.get("non_route_drawing_content", []):
        if row.get("page_ref") == page_ref:
            furniture.update(row.get("source_primitive_refs", []))
    return {
        # Existing M3 is retained for lineage and competitor analysis only.
        # It must not decide the denominator's primary route disposition.
        "route_evidence": set(),
        "legacy_m3_route_candidate": legacy_route,
        "equipment_fitting_evidence": fitting,
        "annotation_dimension": annotation,
        "drawing_furniture": furniture,
    }


def _publish(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing denominator artifact: {destination}")
    os.replace(staging, destination)


def generate(*, source: Path, database: Path, ownership_path: Path,
             output_dir: Path, project_id: str, document_id: str,
             page_number: int) -> dict[str, Any]:
    source_sha256 = _file_sha256(source)
    ownership = json.loads(ownership_path.read_text())
    ownership_sha256 = _file_sha256(ownership_path)
    with PackedProjectStore(database) as store:
        snapshot = store.snapshot(project_id=project_id, document_id=document_id)
        registry = _artifact_json(store, project_id=project_id, document_id=document_id,
                                  name="sheet-registry")
        route_graph = _artifact_json(store, project_id=project_id, document_id=document_id,
                                     name="route-observations")
        fitting_hypotheses = _artifact_json(
            store, project_id=project_id, document_id=document_id, name="fitting-hypotheses")
        equipment = _artifact_json(
            store, project_id=project_id, document_id=document_id, name="equipment-2d-identities")
        stroke_queries = _artifact_json(
            store, project_id=project_id, document_id=document_id, name="stroke-ownership-queries")
    if registry["document"].get("source_pdf_sha256") != source_sha256:
        raise ValueError("source PDF differs from frozen M1 registry")
    registered = next(row for row in registry["pages"] if row["page_number"] == page_number)
    page_ref = registered["page_ref"]
    ownership_page = next(row for row in ownership["pages"] if row["page_ref"] == page_ref)
    regions = [row for row in ownership.get("regions", []) if row.get("page_ref") == page_ref]
    role_refs = _role_refs(
        page_number=page_number, page_ref=page_ref, route_graph=route_graph,
        fitting_hypotheses=fitting_hypotheses, equipment=equipment,
        stroke_queries=stroke_queries, ownership=ownership)

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_name = f"page-{page_number:03d}.native-descriptors.pack"
    disposition_name = f"page-{page_number:03d}.source-dispositions.pack"
    final_descriptor = output_dir / descriptor_name
    final_payload = final_descriptor.with_suffix(final_descriptor.suffix + ".payload")
    final_disposition = output_dir / disposition_name
    final_manifest = output_dir / "manifest.json"
    for destination in (final_descriptor, final_payload, final_disposition, final_manifest):
        if destination.exists():
            raise ValueError(f"denominator artifact already exists: {destination}")

    with tempfile.TemporaryDirectory(prefix="mep-source-denominator-", dir=output_dir) as temp_name:
        temporary = Path(temp_name)
        descriptor_path = temporary / descriptor_name
        writer = None
        with fitz.open(source) as pdf:
            page = pdf[page_number - 1]
            content_xrefs = page.get_contents()
            content = b"".join(pdf.xref_stream(xref) for xref in content_xrefs)
            writer = NativeDescriptorPackWriter(
                descriptor_path, page_ref=page_ref,
                pdf_to_display_matrix=list(page.rotation_matrix),
                minimum_member_length=1e100, cell_size=64)
            try:
                scan = _bounded_page_records(
                    page, page_ref, 64, 4000, candidate_sink=writer.append,
                    include_unretained=False, stream_native_drawings=True,
                    sink_all_candidates=True)
                descriptor_manifest = writer.finish()
            except BaseException:
                writer.abort()
                raise
        descriptor_pack = NativeDescriptorPack(descriptor_path, descriptor_manifest)
        disposition_path = temporary / disposition_name
        disposition_manifest = write_source_disposition_pack(
            descriptor_pack=descriptor_pack, output_path=disposition_path,
            page=ownership_page, regions=regions, role_refs=role_refs)
        disposition_pack = SourceDispositionPack(disposition_path, disposition_manifest)
        errors = validate_source_denominator(
            descriptor_pack=descriptor_pack, disposition_pack=disposition_pack)
        unsupported_count = sum(scan.get("unsupported_native_item_kinds", {}).values())
        if unsupported_count:
            errors.append("unsupported native vector items remain outside segment descriptors")
        if disposition_manifest["unmatched_evidence_reference_count"]:
            errors.append("existing evidence references are absent from the source denominator")

        manifest = {
            "schema_version": "0.1.0",
            "layer": "mep_source_primitive_denominator",
            "development_status": (
                "accepted_source_inventory_classification_provisional"
                if not errors else "development_rejected_source_inventory_incomplete"),
            "page_number": page_number,
            "page_ref": page_ref,
            "sheet_number": registered.get("fields", {}).get("sheet_number", {}).get("normalised_value"),
            "source": {
                "pdf_path": str(source.resolve()),
                "pdf_sha256": source_sha256,
                "native_content_stream_count": len(content_xrefs),
                "native_content_stream_byte_count": len(content),
                "native_content_stream_sha256": hashlib.sha256(content).hexdigest(),
            },
            "upstream": {
                "project_snapshot_v3": snapshot["id"],
                "sheet_region_ownership_sha256": ownership_sha256,
                "legacy_m3_used_only_as_role_evidence": True,
                "legacy_m3_controls_source_existence": False,
            },
            "native_descriptor_pack": {
                **descriptor_manifest,
                "path": str(final_descriptor.resolve()),
                "payload_path": str(final_payload.resolve()),
                "native_drawing_record_count": scan["native_drawing_record_count"],
                "unsupported_native_item_kinds": scan["unsupported_native_item_kinds"],
            },
            "source_disposition_pack": {
                **disposition_manifest,
                "path": str(final_disposition.resolve()),
            },
            "role_evidence": {
                role: {"source_primitive_count": len(refs),
                       "source_primitive_refs_sha256": _canonical_sha256(sorted(refs))}
                for role, refs in sorted(role_refs.items())
            },
            "acceptance_gate": {
                "native_drawing_paths_enumerated": scan["native_drawing_record_count"] > 0,
                "native_segments_enumerated": descriptor_manifest["record_count"] > 0,
                "descriptor_hash_verified": descriptor_pack.verify_hashes(),
                "disposition_hash_verified": disposition_pack.verify_hash(),
                "every_segment_assigned_exactly_once": not errors,
                "unaccounted_accepted_view_segment_count": disposition_manifest["unaccounted_segment_count"],
                "unsupported_native_item_count": unsupported_count,
                "status": "accepted_source_denominator" if not errors else "development_rejected",
                "errors": errors,
            },
            "rejected_downstream": {
                "all_sheet_audit": "development_rejected",
                "semantic_length_total": "development_rejected",
                "source_first_complete_claim": "development_rejected",
                "reason": "sheet_region_ownership_and_source_denominator_not_closed_in_legacy_graph",
            },
            "authority": {
                "source_coverage_established": not errors,
                "role_classification_provisional_until_m3_rebuild": True,
                "semantic_route_identity_established": False,
                "topology_established": False,
                "installed_length_established": False,
                "purchase_length_established": False,
                "quantity_eligible": False,
            },
        }
        manifest["reproducibility"] = {
            "descriptor_manifest_sha256": _canonical_sha256(descriptor_manifest),
            "disposition_manifest_sha256": _canonical_sha256(disposition_manifest),
            "role_evidence_sha256": _canonical_sha256(manifest["role_evidence"]),
        }
        if errors:
            raise ValueError("source denominator validation failed: " + "; ".join(errors))
        _publish(descriptor_path, final_descriptor)
        _publish(descriptor_path.with_suffix(descriptor_path.suffix + ".payload"), final_payload)
        _publish(disposition_path, final_disposition)
        _write_json(final_manifest, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "M&P mark-up against shop systems piping.pdf")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/projects/mep/project-v3.sqlite")
    parser.add_argument("--ownership", type=Path, default=(
        ROOT / "output/mep-sheet-region-ownership-2026-09-03/sheet-region-ownership.json"))
    parser.add_argument("--output-dir", type=Path, default=(
        ROOT / "output/mep-source-denominator-page5-2026-09-03"))
    parser.add_argument("--project-id", default="mep-coordination")
    parser.add_argument("--document-id", default="coordination-set")
    parser.add_argument("--page", type=int, default=5)
    args = parser.parse_args()
    manifest = generate(
        source=args.source, database=args.database, ownership_path=args.ownership,
        output_dir=args.output_dir, project_id=args.project_id,
        document_id=args.document_id, page_number=args.page)
    print(json.dumps({
        "status": manifest["acceptance_gate"]["status"],
        "page_number": manifest["page_number"],
        "native_drawing_path_count": manifest["source_disposition_pack"]["native_drawing_path_count"],
        "native_segment_count": manifest["source_disposition_pack"]["record_count"],
        "disposition_counts": manifest["source_disposition_pack"]["disposition_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
