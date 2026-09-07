#!/usr/bin/env python3
"""Generate the package-wide coverage-first observed MEP takeoff."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_observed_takeoff import build_observed_mep_takeoff, validate_observed_mep_takeoff
from src.drawing_engine.project.project_packed_store import PackedProjectStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_key(store: PackedProjectStore, *, project_id: str, document_id: str, name: str) -> int:
    snapshot = store.snapshot(project_id=project_id, document_id=document_id)
    row = store.connection.execute(
        "SELECT artifact_key FROM snapshot_artifacts WHERE snapshot_key=? AND name=?",
        (snapshot["snapshot_key"], name)).fetchone()
    if row is None:
        raise KeyError(name)
    return row[0]


def _exact_collection(store: PackedProjectStore, artifact_key: int, name: str) -> list[dict]:
    query = ("SELECT * FROM hot_nodes WHERE artifact_key=? AND pointer GLOB ? "
             "AND pointer NOT GLOB ? ORDER BY pointer")
    rows = store.connection.execute(query, (
        artifact_key, f"/{name}/[0-9]*", f"/{name}/*/*"))
    return [json.loads(store._chunk_bytes(row["artifact_key"], row["payload_start"], row["payload_length"]))
            for row in rows]


from src.drawing_engine.project.project_artifact_reader import _artifact_json


def _m3_counts_by_page(store: PackedProjectStore, artifact_key: int) -> dict[str, Counter]:
    output: dict[str, Counter] = defaultdict(Counter)
    rows = store.connection.execute(
        "SELECT p.value AS page_ref,t.value AS record_type,count(*) AS n "
        "FROM hot_nodes h "
        "LEFT JOIN dictionaries p ON p.kind='page' AND p.key=h.page_key "
        "JOIN dictionaries t ON t.kind='type' AND t.key=h.type_key "
        "WHERE h.artifact_key=? GROUP BY p.value,t.value", (artifact_key,))
    for row in rows:
        if row["page_ref"]:
            output[row["page_ref"]][row["record_type"]] = row["n"]
    return output


def _source_first_sheet_coverage(
    *, source: Path, registry: dict, projected_occurrences: list[dict],
    canonical_segments: list[dict], duplicate_candidates: list[dict],
    network_boundaries: list[dict], m3_counts: dict[str, Counter],
) -> list[dict]:
    occurrence_counts = Counter(row["page_ref"] for row in projected_occurrences)
    canonical_occurrence_refs = {ref for row in canonical_segments
                                 for ref in row.get("source_page_occurrence_refs", [])}
    occurrence_by_id = {row["id"]: row for row in projected_occurrences}
    canonical_page_counts = Counter(occurrence_by_id[ref]["page_ref"]
                                    for ref in canonical_occurrence_refs if ref in occurrence_by_id)
    duplicate_candidate_page_counts = Counter()
    for row in duplicate_candidates:
        duplicate_candidate_page_counts[row.get("source_page_ref")] += 1
        duplicate_candidate_page_counts[row.get("target_page_ref")] += 1
    boundary_counts = Counter(row.get("page_ref") for row in network_boundaries)

    pages = {row["page_number"]: row for row in registry.get("pages", [])}
    output = []
    with fitz.open(source) as pdf:
        if len(pdf) != len(pages):
            raise ValueError("source PDF page count differs from the frozen sheet registry")
        for page_number in sorted(pages):
            registry_page = pages[page_number]
            page_ref = registry_page["page_ref"]
            page = pdf[page_number - 1]
            # A content-stream hash establishes source identity only.  It is
            # not a source denominator and cannot certify coverage without an
            # exhaustive native descriptor/disposition pack.
            content_xrefs = page.get_contents()
            content_streams = [pdf.xref_stream(xref) for xref in content_xrefs]
            source_bytes = b"".join(content_streams)
            inventory = {
                "page_number": page_number,
                "page_ref": page_ref,
                "page_rect_display": [round(page.rect.x0, 6), round(page.rect.y0, 6),
                                      round(page.rect.x1, 6), round(page.rect.y1, 6)],
                "native_content_stream_count": len(content_streams),
                "native_content_stream_byte_count": len(source_bytes),
                "native_content_stream_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "native_drawing_path_count": None,
                "path_enumeration_reason": "native_source_denominator_not_enumerated_in_legacy_baseline",
                "quality_route": registry_page.get("quality_route"),
            }
            counts = m3_counts.get(page_ref, Counter())
            output.append({
                "id": "mep_source_first_sheet_coverage." + hashlib.sha256(
                    json.dumps(inventory, sort_keys=True).encode()).hexdigest()[:20],
                "record_type": "mep_source_first_sheet_coverage",
                "state": "observed",
                "source_first_review_state": "development_rejected_source_denominator_missing",
                "page_number": page_number,
                "page_ref": page_ref,
                "sheet_number": registry_page.get("fields", {}).get("sheet_number", {}).get("normalised_value"),
                "sheet_role": registry_page.get("role"),
                "source_inventory": inventory,
                "source_inventory_sha256": hashlib.sha256(
                    json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "m3_observation_counts": {
                    "fragment_count": counts.get("mep_route_fragment_observation", 0),
                    "endpoint_count": counts.get("mep_route_endpoint_observation", 0),
                    "branch_count": counts.get("mep_route_branch_observation", 0),
                    "crossing_count": counts.get("mep_route_crossing_observation", 0),
                },
                "takeoff_coverage": {
                    "projected_occurrence_count": occurrence_counts[page_ref],
                    "accepted_duplicate_projection_occurrence_count": canonical_page_counts[page_ref],
                    "unresolved_duplicate_candidate_count": duplicate_candidate_page_counts[page_ref],
                    "unresolved_network_boundary_count": boundary_counts[page_ref],
                },
                "authority": {
                    "source_inventory_independent_from_candidate_rank": True,
                    "content_stream_hash_proves_coverage": False,
                    "complete_native_source_denominator_enumerated": False,
                    "semantic_route_absence_established": False,
                    "physical_run_identity_established": False,
                    "quantity_eligible": False,
                },
            })
    return output


def generate(*, database: Path, project_id: str, document_id: str,
             source: Path, output: Path) -> dict:
    with PackedProjectStore(database) as store:
        registry = _artifact_json(store, project_id=project_id, document_id=document_id,
                                  name="sheet-registry")
        expected = registry.get("document", {}).get("source_pdf_sha256")
        actual = _file_sha256(source)
        if expected != actual:
            raise ValueError("source PDF differs from the frozen M1 registry")

        keys = {name: _artifact_key(store, project_id=project_id, document_id=document_id, name=name)
                for name in ("route-observations", "cross-sheet-runs", "bounded-local-3d",
                             "network-hierarchy", "hvac-inventory")}
        occurrences = _exact_collection(store, keys["cross-sheet-runs"], "projected_route_occurrences")
        canonical = _exact_collection(store, keys["cross-sheet-runs"], "canonical_projected_segments")
        duplicates = _exact_collection(store, keys["cross-sheet-runs"], "overlap_duplicate_candidates")
        vertical = _exact_collection(store, keys["cross-sheet-runs"], "unresolved_vertical_spans")
        bounded = _exact_collection(store, keys["bounded-local-3d"], "bounded_local_3d_segments")
        segments = _exact_collection(store, keys["network-hierarchy"], "segments")
        junctions = _exact_collection(store, keys["network-hierarchy"], "junctions")
        boundaries = _exact_collection(store, keys["network-hierarchy"], "unresolved_boundaries")
        hvac = _exact_collection(store, keys["hvac-inventory"], "items")
        m3_counts = _m3_counts_by_page(store, keys["route-observations"])

        coverage = _source_first_sheet_coverage(
            source=source, registry=registry, projected_occurrences=occurrences,
            canonical_segments=canonical, duplicate_candidates=duplicates,
            network_boundaries=boundaries, m3_counts=m3_counts)
        payload = build_observed_mep_takeoff(
            document=registry["document"], projected_occurrences=occurrences,
            canonical_segments=canonical, duplicate_candidates=duplicates,
            bounded_local_3d_segments=bounded, network_segments=segments,
            junctions=junctions, unresolved_boundaries=boundaries,
            hvac_items=hvac, sheet_coverage=coverage,
            unresolved_vertical_spans=vertical)
        payload["input_payload_sha256"] = {
            "source_pdf": expected,
            "project_snapshot": store.snapshot(project_id=project_id, document_id=document_id)["id"],
        }
    errors = validate_observed_mep_takeoff(payload)
    if errors:
        raise ValueError("; ".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data/projects/mep/project-v3.sqlite")
    parser.add_argument("--project-id", default="mep-coordination")
    parser.add_argument("--document-id", default="coordination-set")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "M&P mark-up against shop systems piping.pdf")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/mep-observed-takeoff-2026-09-02/observed-takeoff.json")
    args = parser.parse_args()
    payload = generate(database=args.database, project_id=args.project_id,
                       document_id=args.document_id, source=args.source, output=args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
