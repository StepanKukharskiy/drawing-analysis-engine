"""Coverage-first observed MEP takeoff over frozen M3/M5 evidence.

This adapter deliberately stops before installed-run and marketplace authority.
It preserves every projected occurrence, reuses only accepted M5B duplicate
certificates, and keeps observed, deduplicated, and physical counts separate.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    return kind + "." + _sha256([kind, *parts])[:20]


def _polyline_length(points: Sequence[Sequence[float]]) -> float | None:
    if len(points) < 2:
        return None
    return round(sum(math.dist(a, b) for a, b in zip(points, points[1:])), 8)


def _numeric_sum(values: Iterable[float | None]) -> float:
    return round(sum(value for value in values if value is not None), 8)


def _occurrence_projection(occurrence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "occurrence_ref": occurrence["id"],
        "page_ref": occurrence["page_ref"],
        "route_target_ref": occurrence.get("route_target_ref"),
        "route_composite_ref": occurrence.get("route_composite_ref"),
        "points_display": occurrence.get("points_display", []),
        "projected_2d_length_m": occurrence.get("projected_2d_length_m"),
        "source_fragment_refs": occurrence.get("source_fragment_refs", []),
        "source_primitive_refs": occurrence.get("source_primitive_refs", []),
    }


def _semantic_overlay(network_segment: Mapping[str, Any] | None) -> dict[str, Any]:
    if not network_segment:
        return {name: {"state": "unknown", "observed_values": []}
                for name in ("system", "size", "elevation")}
    overlay = network_segment.get("semantic_overlay", {})
    return {name: {
        "state": overlay.get(name, {}).get("state", "unknown"),
        "observed_values": overlay.get(name, {}).get("observed_values", []),
        "accepted_relation_refs": overlay.get(name, {}).get("accepted_relation_refs", []),
    } for name in ("system", "size", "elevation")}


def _item_class(relation_type: str) -> tuple[str, list[str]]:
    if relation_type == "projected_native_bend":
        return "bend", ["bend"]
    if relation_type in {"projected_native_branch", "projected_fitting_body_branch"}:
        return "tee", ["tee", "other_three_port_fitting"]
    if relation_type == "projected_collinear_boundary_join":
        return "coupling_or_continuation", ["coupling", "authored_continuation", "plain_join"]
    return "other_fitting", ["other_fitting"]


def _build_route_ledger(
    occurrences: Sequence[Mapping[str, Any]],
    canonical_segments: Sequence[Mapping[str, Any]],
    duplicate_candidates: Sequence[Mapping[str, Any]],
    bounded_local_3d_segments: Sequence[Mapping[str, Any]],
    network_segments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    occurrence_by_id = {row["id"]: row for row in occurrences}
    segment_by_occurrence = {}
    for segment in network_segments:
        for ref in segment.get("source_occurrence_refs", []):
            segment_by_occurrence.setdefault(ref, segment)

    accepted_occurrence_refs = set()
    bounded_by_canonical = {row.get("canonical_projected_segment_ref"): row
                            for row in bounded_local_3d_segments
                            if row.get("state") == "accepted"}
    bounded_by_occurrence = {}
    for row in bounded_local_3d_segments:
        if row.get("state") != "accepted":
            continue
        for ref in row.get("source_page_occurrence_refs", []):
            bounded_by_occurrence[ref] = row

    competitor_refs_by_target: dict[str, set[str]] = defaultdict(set)
    for candidate in duplicate_candidates:
        if candidate.get("state") == "accepted":
            continue
        for key in ("source_route_target_ref", "target_route_target_ref"):
            ref = candidate.get(key)
            if ref:
                competitor_refs_by_target[ref].add(candidate["id"])

    ledger: list[dict[str, Any]] = []
    for canonical in canonical_segments:
        refs = [ref for ref in canonical.get("source_page_occurrence_refs", []) if ref in occurrence_by_id]
        if not refs:
            continue
        accepted_occurrence_refs.update(refs)
        rows = [occurrence_by_id[ref] for ref in refs]
        bounded = bounded_by_canonical.get(canonical["id"])
        bounded_length = _polyline_length(bounded.get("centreline_points_xyz_m", [])) if bounded else None
        representative = canonical.get("representative_projected_2d_length_m")
        if representative is None:
            representative = rows[0].get("projected_2d_length_m")
        ledger.append({
            "id": _stable_id("mep_observed_route_segment", canonical["id"]),
            "record_type": "mep_observed_route_segment",
            "state": "derived",
            "canonical_projected_segment_ref": canonical["id"],
            "duplicate_certificate_refs": canonical.get("overlap_duplicate_relation_refs", []),
            "duplicate_competitor_refs": [],
            "occurrences": [_occurrence_projection(row) for row in rows],
            "counts": {
                "observed_occurrence_count": len(rows),
                "deduplicated_projected_count": 1,
                "physical_instance_count": None,
            },
            "length_channels": {
                "visible_projected_length_m": _numeric_sum(row.get("projected_2d_length_m") for row in rows),
                "canonicalized_projected_length_m": representative,
                "bounded_local_3d_length_m": bounded_length,
                "unresolved_vertical_length_m": None,
                "installed_length_m": None,
                "purchase_length_m": None,
            },
            "semantic_overlay": _semantic_overlay(segment_by_occurrence.get(refs[0])),
            "authority": {
                "accepted_duplicate_certificate": True,
                "bounded_local_3d_geometry_established": bounded is not None,
                "physical_run_identity_established": False,
                "installed_length_established": False,
                "purchase_length_established": False,
                "quantity_eligible": False,
            },
            "unresolved_reasons": ["physical_instance_identity_not_established",
                                   "installed_run_not_closed", "purchase_rule_not_applicable"],
        })

    for occurrence in occurrences:
        if occurrence["id"] in accepted_occurrence_refs:
            continue
        target_ref = occurrence.get("route_target_ref")
        competitors = sorted(competitor_refs_by_target.get(target_ref, ()))
        bounded = bounded_by_occurrence.get(occurrence["id"])
        bounded_length = _polyline_length(bounded.get("centreline_points_xyz_m", [])) if bounded else None
        projected = occurrence.get("projected_2d_length_m")
        ledger.append({
            "id": _stable_id("mep_observed_route_segment", occurrence["id"]),
            "record_type": "mep_observed_route_segment",
            "state": "observed" if projected is None else "derived",
            "canonical_projected_segment_ref": None,
            "duplicate_certificate_refs": [],
            "duplicate_competitor_refs": competitors,
            "occurrences": [_occurrence_projection(occurrence)],
            "counts": {
                "observed_occurrence_count": 1,
                "deduplicated_projected_count": None if competitors else 1,
                "physical_instance_count": None,
            },
            "length_channels": {
                "visible_projected_length_m": projected,
                "canonicalized_projected_length_m": None if competitors else projected,
                "bounded_local_3d_length_m": bounded_length,
                "unresolved_vertical_length_m": None,
                "installed_length_m": None,
                "purchase_length_m": None,
            },
            "semantic_overlay": _semantic_overlay(segment_by_occurrence.get(occurrence["id"])),
            "authority": {
                "accepted_duplicate_certificate": False,
                "bounded_local_3d_geometry_established": bounded is not None,
                "physical_run_identity_established": False,
                "installed_length_established": False,
                "purchase_length_established": False,
                "quantity_eligible": False,
            },
            "unresolved_reasons": (["duplicate_projection_identity_unresolved"] if competitors else []) +
                ["physical_instance_identity_not_established", "installed_run_not_closed",
                 "purchase_rule_not_applicable"],
        })

    ledger.sort(key=lambda row: row["id"])
    visible = _numeric_sum(row["length_channels"]["visible_projected_length_m"] for row in ledger)
    canonicalized = _numeric_sum(row["length_channels"]["canonicalized_projected_length_m"] for row in ledger)
    unresolved_duplicate = _numeric_sum(
        row["length_channels"]["visible_projected_length_m"] for row in ledger
        if row["counts"]["deduplicated_projected_count"] is None)
    bounded = _numeric_sum(row["length_channels"]["bounded_local_3d_length_m"] for row in ledger)
    unresolved_projected_count = sum(
        occurrence.get("projected_2d_length_m") is None
        for row in ledger for occurrence in row["occurrences"])
    return ledger, {
        "visible_projected_length_m": visible,
        "visible_projected_length_state": "partial" if unresolved_projected_count else "derived",
        "visible_projected_length_unresolved_occurrence_count": unresolved_projected_count,
        "certified_canonicalized_projected_length_m": canonicalized,
        "certified_canonicalized_projected_length_state": (
            "partial" if unresolved_projected_count or unresolved_duplicate else "derived"),
        "duplicate_identity_unresolved_visible_length_m": unresolved_duplicate,
        "bounded_local_3d_length_m": bounded,
        "unresolved_vertical_length_m": None,
        "installed_length_m": None,
        "purchase_length_m": None,
    }


def _build_item_ledger(
    junctions: Sequence[Mapping[str, Any]],
    hvac_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger = []
    for junction in junctions:
        generic_class, alternatives = _item_class(junction.get("relation_type", ""))
        ledger.append({
            "id": _stable_id("mep_observed_item", junction["id"]),
            "record_type": "mep_observed_item_occurrence",
            "state": "observed_interface" if junction.get("state") == "accepted" else "candidate",
            "page_ref": junction.get("page_ref"),
            "generic_class": generic_class,
            "classification_alternatives": alternatives,
            "source_record_ref": junction["id"],
            "source_primitive_refs": junction.get("evidence_refs", []),
            "location_display": junction.get("point_display"),
            "counts": {"observed_occurrence_count": 1,
                       "deduplicated_projected_count": None,
                       "physical_instance_count": None},
            "exact_sku": None,
            "connection_standard": None,
            "authority": {"physical_item_identity_established": False,
                          "physical_count_established": False,
                          "quantity_eligible": False},
            "unresolved_reasons": list(junction.get("reasons", [])) +
                ["cross_sheet_duplicate_item_identity_not_established",
                 "physical_item_identity_not_established"],
        })
    for item in hvac_items:
        item_type = item.get("item_type")
        item_class = item.get("item_class") or "unresolved_item"
        generic_class = "equipment" if item_type == "equipment" else item_class
        ledger.append({
            "id": _stable_id("mep_observed_item", item["id"]),
            "record_type": "mep_observed_item_occurrence",
            "state": "observed",
            "page_ref": item.get("page_ref"),
            "generic_class": generic_class,
            "observed_item_type": item_type,
            "observed_item_class": item_class,
            "classification_alternatives": [],
            "source_record_ref": item["id"],
            "source_primitive_refs": item.get("source_observation_refs", []) + item.get("source_fragment_refs", []),
            "location_display": None,
            "tag": item.get("candidate", {}).get("tag"),
            "counts": {"observed_occurrence_count": 1,
                       "deduplicated_projected_count": None,
                       "physical_instance_count": None},
            "exact_sku": None,
            "connection_standard": None,
            "authority": {"physical_item_identity_established": False,
                          "physical_count_established": False,
                          "quantity_eligible": False},
            "unresolved_reasons": list(item.get("unresolved_reasons", [])) +
                ["cross_sheet_duplicate_item_identity_not_established",
                 "physical_item_identity_not_established"],
        })
    ledger.sort(key=lambda row: row["id"])

    category_counts = {name: {"observed_occurrence_count": 0,
                              "deduplicated_projected_count": None,
                              "physical_instance_count": None}
                       for name in ("bend", "tee", "coupling", "coupling_or_continuation",
                                    "valve", "damper", "accessory", "equipment")}
    for row in ledger:
        observed_class = row.get("observed_item_class")
        item_type = row.get("observed_item_type")
        if row["generic_class"] == "bend" or observed_class == "elbow":
            category_counts["bend"]["observed_occurrence_count"] += 1
        if row["generic_class"] == "tee":
            category_counts["tee"]["observed_occurrence_count"] += 1
        if row["generic_class"] == "coupling":
            category_counts["coupling"]["observed_occurrence_count"] += 1
        if row["generic_class"] == "coupling_or_continuation":
            category_counts["coupling_or_continuation"]["observed_occurrence_count"] += 1
        if observed_class and "valve" in observed_class:
            category_counts["valve"]["observed_occurrence_count"] += 1
        if observed_class and "damper" in observed_class:
            category_counts["damper"]["observed_occurrence_count"] += 1
        if item_type == "accessory":
            category_counts["accessory"]["observed_occurrence_count"] += 1
        if item_type == "equipment":
            category_counts["equipment"]["observed_occurrence_count"] += 1
    return ledger, category_counts


def _build_unresolved_ends(
    unresolved_boundaries: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    occurrence_by_id = {row["id"]: row for row in occurrences}
    output = []
    for boundary in unresolved_boundaries:
        occurrence = occurrence_by_id.get(boundary.get("source_occurrence_ref"))
        points = [] if occurrence is None else occurrence.get("points_display", [])
        markers = []
        if points:
            for endpoint_index, point in ((0, points[0]), (1, points[-1])):
                markers.append({"endpoint_index": endpoint_index, "point_display": point,
                                "marker_state": "unresolved"})
        output.append({
            "id": _stable_id("mep_observed_unresolved_route_end", boundary["id"]),
            "record_type": "mep_observed_unresolved_route_end",
            "state": "unknown",
            "page_ref": boundary.get("page_ref"),
            "network_boundary_ref": boundary["id"],
            "network_segment_ref": boundary.get("segment_ref"),
            "source_occurrence_ref": boundary.get("source_occurrence_ref"),
            "endpoint_markers": markers,
            "reason": boundary.get("reason", "missing_continuation_or_terminal_identity"),
            "continuation_established": False,
            "physical_terminal_identity_established": False,
            "quantity_eligible": False,
        })
    return sorted(output, key=lambda row: row["id"])


def build_observed_mep_takeoff(
    *,
    document: Mapping[str, Any],
    projected_occurrences: Sequence[Mapping[str, Any]],
    canonical_segments: Sequence[Mapping[str, Any]],
    duplicate_candidates: Sequence[Mapping[str, Any]],
    bounded_local_3d_segments: Sequence[Mapping[str, Any]],
    network_segments: Sequence[Mapping[str, Any]],
    junctions: Sequence[Mapping[str, Any]],
    unresolved_boundaries: Sequence[Mapping[str, Any]],
    hvac_items: Sequence[Mapping[str, Any]],
    sheet_coverage: Sequence[Mapping[str, Any]],
    unresolved_vertical_spans: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile the observation-only route and item ledgers."""
    route_ledger, length_channels = _build_route_ledger(
        projected_occurrences, canonical_segments, duplicate_candidates,
        bounded_local_3d_segments, network_segments)
    item_ledger, item_counts = _build_item_ledger(junctions, hvac_items)
    unresolved_ends = _build_unresolved_ends(unresolved_boundaries, projected_occurrences)
    if unresolved_vertical_spans:
        length_channels["unresolved_vertical_length_m"] = None

    reviewed_pages = sum(row.get("source_first_review_state") == "complete" for row in sheet_coverage)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "mep_coverage_first_observed_takeoff",
        "document": dict(document),
        "authority": {
            "observation_only": True,
            "source_first_coverage_independent_from_terminal_search": True,
            "accepted_m5b_duplicate_certificates_reused": True,
            "physical_run_identity_established": False,
            "installed_length_established": False,
            "purchase_length_established": False,
            "connector_selection_established": False,
            "marketplace_assembly_established": False,
            "quantity_eligible": False,
        },
        "length_channels": length_channels,
        "route_segment_ledger": route_ledger,
        "item_occurrence_ledger": item_ledger,
        "unresolved_route_ends": unresolved_ends,
        "unresolved_vertical_spans": list(unresolved_vertical_spans),
        "sheet_coverage": list(sheet_coverage),
        "item_category_counts": item_counts,
        "summary": {
            "visible_projected_occurrence_count": len(projected_occurrences),
            "visible_projected_length_unresolved_occurrence_count":
                length_channels["visible_projected_length_unresolved_occurrence_count"],
            "route_ledger_row_count": len(route_ledger),
            "accepted_duplicate_group_count": len(canonical_segments),
            "unresolved_duplicate_candidate_count": sum(
                row.get("state") != "accepted" for row in duplicate_candidates),
            "bounded_local_3d_segment_count": sum(
                row.get("state") == "accepted" for row in bounded_local_3d_segments),
            "unresolved_vertical_span_count": len(unresolved_vertical_spans),
            "observed_item_occurrence_count": len(item_ledger),
            "unresolved_route_boundary_count": len(unresolved_ends),
            "unresolved_endpoint_marker_count": sum(
                len(row["endpoint_markers"]) for row in unresolved_ends),
            "source_first_reviewed_sheet_count": reviewed_pages,
            "registered_sheet_count": len(sheet_coverage),
            "installed_length_m": None,
            "purchase_length_m": None,
        },
    }


def validate_observed_mep_takeoff(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("layer") != "mep_coverage_first_observed_takeoff":
        errors.append("unexpected layer")
    authority = payload.get("authority", {})
    for field in ("physical_run_identity_established", "installed_length_established",
                  "purchase_length_established", "connector_selection_established",
                  "marketplace_assembly_established", "quantity_eligible"):
        if authority.get(field) is not False:
            errors.append("authority.%s must be false" % field)
    channels = payload.get("length_channels", {})
    if channels.get("installed_length_m") is not None or channels.get("purchase_length_m") is not None:
        errors.append("installed and purchase lengths must remain null")
    route_rows = payload.get("route_segment_ledger", [])
    if len({row.get("id") for row in route_rows}) != len(route_rows):
        errors.append("route ledger IDs must be unique")
    for row in route_rows:
        counts = row.get("counts", {})
        if counts.get("observed_occurrence_count") != len(row.get("occurrences", [])):
            errors.append("route observed count mismatch: " + str(row.get("id")))
        if counts.get("physical_instance_count") is not None:
            errors.append("route physical count must remain null: " + str(row.get("id")))
        lengths = row.get("length_channels", {})
        if lengths.get("installed_length_m") is not None or lengths.get("purchase_length_m") is not None:
            errors.append("route installed and purchase lengths must remain null: " + str(row.get("id")))
    for row in payload.get("item_occurrence_ledger", []):
        counts = row.get("counts", {})
        if counts.get("observed_occurrence_count") != 1:
            errors.append("item observed occurrence count must be one: " + str(row.get("id")))
        if counts.get("physical_instance_count") is not None:
            errors.append("item physical count must remain null: " + str(row.get("id")))
    summary = payload.get("summary", {})
    if summary.get("route_ledger_row_count") != len(route_rows):
        errors.append("summary route ledger count mismatch")
    if summary.get("unresolved_route_boundary_count") != len(payload.get("unresolved_route_ends", [])):
        errors.append("summary unresolved boundary count mismatch")
    return errors
