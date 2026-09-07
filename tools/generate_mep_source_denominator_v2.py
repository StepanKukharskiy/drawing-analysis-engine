#!/usr/bin/env python3
"""Rebuild page-5 source dispositions and freeze authored PDF path context.

The earlier descriptor bytes are a valid exhaustive source scan, but their
published disposition sidecar is stale because it promoted legacy M3 members.
This generator verifies and reuses only those immutable descriptor bytes,
writes a fresh non-circular disposition pack, and groups descriptors into
authored PDF paint paths before any full-sheet classifier runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_mep_observed_takeoff import _artifact_json
from tools.generate_mep_source_denominator import (
    _canonical_sha256, _file_sha256, _role_refs, _write_json,
)
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack
from src.drawing_engine.disciplines.mep.mep_native_path_pack import (
    NativePathPack, validate_native_path_pack, write_native_path_pack,
)
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    SourceDispositionPack, validate_source_denominator,
    write_source_disposition_pack,
)
from src.drawing_engine.project.project_packed_store import PackedProjectStore


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(
    *, source: Path, database: Path, ownership_path: Path,
    reused_denominator_manifest_path: Path, output_dir: Path,
    project_id: str, document_id: str, page_number: int,
) -> dict[str, Any]:
    old = json.loads(reused_denominator_manifest_path.read_text())
    ownership = json.loads(ownership_path.read_text())
    source_sha256 = _file_sha256(source)
    if old.get("page_number") != page_number:
        raise ValueError("reused descriptor manifest belongs to another page")
    if old.get("source", {}).get("pdf_sha256") != source_sha256:
        raise ValueError("reused native descriptors belong to another source PDF")
    descriptor_info = dict(old["native_descriptor_pack"])
    descriptors = NativeDescriptorPack(Path(descriptor_info["path"]), descriptor_info)
    if not descriptors.verify_hashes():
        raise ValueError("reused native descriptor hash verification failed")

    with PackedProjectStore(database) as store:
        snapshot = store.snapshot(project_id=project_id, document_id=document_id)
        registry = _artifact_json(store, project_id=project_id, document_id=document_id,
                                  name="sheet-registry")
        route_graph = _artifact_json(store, project_id=project_id, document_id=document_id,
                                     name="route-observations")
        fitting_hypotheses = _artifact_json(
            store, project_id=project_id, document_id=document_id,
            name="fitting-hypotheses")
        equipment = _artifact_json(
            store, project_id=project_id, document_id=document_id,
            name="equipment-2d-identities")
        stroke_queries = _artifact_json(
            store, project_id=project_id, document_id=document_id,
            name="stroke-ownership-queries")
    if registry["document"].get("source_pdf_sha256") != source_sha256:
        raise ValueError("source PDF differs from frozen M1 registry")
    registered = next(row for row in registry["pages"]
                      if row["page_number"] == page_number)
    page_ref = registered["page_ref"]
    if descriptor_info.get("page_ref") != page_ref:
        raise ValueError("reused descriptors differ from the registered page")
    ownership_page = next(row for row in ownership["pages"]
                          if row["page_ref"] == page_ref)
    regions = [row for row in ownership.get("regions", [])
               if row.get("page_ref") == page_ref]
    role_refs = _role_refs(
        page_number=page_number, page_ref=page_ref, route_graph=route_graph,
        fitting_hypotheses=fitting_hypotheses, equipment=equipment,
        stroke_queries=stroke_queries, ownership=ownership)

    if output_dir.exists():
        raise ValueError(f"refusing to overwrite immutable output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=output_dir.name + ".", suffix=".partial", dir=output_dir.parent))
    try:
        disposition_name = f"page-{page_number:03d}.source-dispositions.pack"
        path_name = f"page-{page_number:03d}.native-authored-paths.pack"
        disposition_path = staging / disposition_name
        path_path = staging / path_name
        disposition_manifest = write_source_disposition_pack(
            descriptor_pack=descriptors, output_path=disposition_path,
            page=ownership_page, regions=regions, role_refs=role_refs)
        dispositions = SourceDispositionPack(disposition_path, disposition_manifest)
        errors = validate_source_denominator(
            descriptor_pack=descriptors, disposition_pack=dispositions)
        path_manifest = write_native_path_pack(
            descriptor_pack=descriptors, disposition_pack=dispositions,
            output_path=path_path)
        paths = NativePathPack(path_path, path_manifest)
        errors.extend(validate_native_path_pack(
            descriptor_pack=descriptors, disposition_pack=dispositions,
            path_pack=paths))
        disposition_counts = disposition_manifest["disposition_counts"]
        if disposition_counts.get("route_evidence", 0):
            errors.append("legacy or unsupported evidence directly promoted a route")
        legacy_count = disposition_manifest["candidate_role_counts"].get(
            "legacy_m3_route_candidate", 0)
        if legacy_count != len(role_refs["legacy_m3_route_candidate"]):
            errors.append("legacy M3 candidate-role lineage is incomplete")

        final_disposition = output_dir / disposition_name
        final_paths = output_dir / path_name
        disposition_manifest["path"] = str(final_disposition.resolve())
        path_manifest["path"] = str(final_paths.resolve())
        manifest = {
            "schema_version": "0.2.0",
            "layer": "mep_source_primitive_denominator",
            "development_status": (
                "accepted_source_denominator_path_context_frozen"
                if not errors else "development_rejected_source_inventory_incomplete"),
            "page_number": page_number,
            "page_ref": page_ref,
            "sheet_number": registered.get("fields", {}).get(
                "sheet_number", {}).get("normalised_value"),
            "source": {
                "pdf_path": str(source.resolve()), "pdf_sha256": source_sha256,
                "native_content_stream_sha256": old["source"][
                    "native_content_stream_sha256"],
            },
            "upstream": {
                "project_snapshot_v3": snapshot["id"],
                "sheet_region_ownership_sha256": _file_sha256(ownership_path),
                "reused_descriptor_manifest_path": str(
                    reused_denominator_manifest_path.resolve()),
                "reused_descriptor_manifest_sha256": _file_sha256(
                    reused_denominator_manifest_path),
                "descriptor_bytes_reused_as_verified_source_inventory_only": True,
                "stale_disposition_bytes_reused": False,
                "legacy_m3_used_only_as_candidate_role_bit": True,
                "legacy_m3_controls_source_existence": False,
                "legacy_m3_controls_primary_disposition": False,
            },
            "native_descriptor_pack": descriptor_info,
            "source_disposition_pack": disposition_manifest,
            "native_authored_path_pack": path_manifest,
            "role_evidence": {
                role: {"source_primitive_count": len(refs),
                       "source_primitive_refs_sha256": _canonical_sha256(sorted(refs))}
                for role, refs in sorted(role_refs.items())
            },
            "acceptance_gate": {
                "all_native_segments_accounted": (
                    disposition_manifest["record_count"] == len(descriptors)),
                "native_segment_count": len(descriptors),
                "expected_native_segment_count": descriptor_info["record_count"],
                "native_segment_denominator_matches_frozen_scan": (
                    len(descriptors) == descriptor_info["record_count"]),
                "unaccounted_segment_count": disposition_manifest[
                    "unaccounted_segment_count"],
                "legacy_M3_primary_route_count": disposition_counts.get(
                    "route_evidence", 0),
                "legacy_M3_candidate_role_count": legacy_count,
                "authored_path_count": path_manifest["record_count"],
                "path_pack_partitions_every_segment": (
                    path_manifest["source_segment_count"] == len(descriptors)),
                "descriptor_hash_verified": descriptors.verify_hashes(),
                "disposition_hash_verified": dispositions.verify_hash(),
                "path_hash_verified": paths.verify_hash(),
                "status": "accepted_path_context_frozen" if not errors else "development_rejected",
                "errors": errors,
            },
            "invalidated_legacy_claims": {
                "legacy_denominator": "development_rejected_stale_denominator",
                "legacy_page5_coverage_total": "development_rejected_stale_denominator",
                "legacy_page5_audit": "development_rejected_stale_denominator",
                "legacy_724_backlog": "retired_not_a_sheet_coverage_backlog",
                "all_sheet_audit": "development_rejected_stale_denominator",
                "semantic_length_total": "development_rejected_stale_denominator",
                "source_first_complete_claim": "development_rejected_stale_denominator",
            },
            "authority": {
                "source_coverage_established": not errors,
                "authored_pdf_path_context_established": not errors,
                "role_classification_provisional_until_page_wide_classification": True,
                "semantic_route_identity_established": False,
                "topology_established": False,
                "installed_length_established": False,
                "purchase_length_established": False,
                "quantity_eligible": False,
            },
        }
        manifest["reproducibility"] = {
            "source_disposition_pack_sha256": disposition_manifest["sha256"],
            "native_authored_path_pack_sha256": path_manifest["sha256"],
            "manifest_inputs_sha256": _canonical_sha256({
                "source": manifest["source"], "upstream": manifest["upstream"],
                "role_evidence": manifest["role_evidence"],
            }),
        }
        if errors:
            raise ValueError("v2 denominator validation failed: " + "; ".join(errors))
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        # Preserve a failed staging directory for forensic inspection.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "M&P mark-up against shop systems piping.pdf")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/projects/mep/project-v3.sqlite")
    parser.add_argument("--ownership", type=Path, default=ROOT /
                        "output/mep-sheet-region-ownership-2026-09-03/sheet-region-ownership.json")
    parser.add_argument("--reuse-denominator-manifest", type=Path, default=ROOT /
                        "output/mep-source-denominator-page5-2026-09-03/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT /
                        "output/mep-source-denominator-page5-v2-2026-09-03")
    parser.add_argument("--project-id", default="mep-coordination")
    parser.add_argument("--document-id", default="coordination-set")
    parser.add_argument("--page", type=int, default=5)
    args = parser.parse_args()
    payload = generate(
        source=args.source, database=args.database,
        ownership_path=args.ownership,
        reused_denominator_manifest_path=args.reuse_denominator_manifest,
        output_dir=args.output_dir, project_id=args.project_id,
        document_id=args.document_id, page_number=args.page)
    print(json.dumps({
        "status": payload["acceptance_gate"]["status"],
        "native_segment_count": payload["acceptance_gate"]["native_segment_count"],
        "authored_path_count": payload["acceptance_gate"]["authored_path_count"],
        "disposition_counts": payload["source_disposition_pack"]["disposition_counts"],
        "legacy_M3_candidate_role_count": payload["acceptance_gate"][
            "legacy_M3_candidate_role_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
