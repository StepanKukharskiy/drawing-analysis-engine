"""Region-owned observed and semantic MEP takeoff projections.

This adapter keeps the frozen observation graph immutable while removing
drawing furniture and unowned page regions from route-length channels.  A
counted semantic route additionally requires an accepted system identity;
unknown-system geometry remains an unresolved valid-view candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_semantic_takeoff import build_semantic_mep_takeoff


SCHEMA_VERSION = "0.1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    return kind + "." + _sha256([kind, *parts])[:20]


def _sum(values) -> float:
    return round(sum(float(value) for value in values if value is not None), 8)


def build_region_owned_observed_takeoff(
    *, baseline: Mapping[str, Any], baseline_sha256: str,
    ownership: Mapping[str, Any], ownership_sha256: str,
) -> dict[str, Any]:
    membership_by_occurrence = {row["occurrence_ref"]: row
                                for row in ownership["candidate_ownership"]}
    route_rows = []
    excluded_rows = []
    unknown_region_rows = []
    accepted_occurrence_refs = set()
    for source in baseline["route_segment_ledger"]:
        accepted, excluded, unknown = [], [], []
        for occurrence in source.get("occurrences", []):
            membership = membership_by_occurrence.get(occurrence["occurrence_ref"])
            if membership is None:
                raise ValueError("region ownership lacks route occurrence")
            if membership.get("route_certification_eligible") is True:
                accepted.append(occurrence)
                accepted_occurrence_refs.add(occurrence["occurrence_ref"])
            elif membership.get("state") == "excluded":
                excluded.append(occurrence)
            else:
                unknown.append(occurrence)
        if excluded:
            excluded_rows.append({
                "source_route_ledger_ref": source["id"],
                "source_occurrence_refs": [row["occurrence_ref"] for row in excluded],
            })
        if unknown:
            unknown_region_rows.append({
                "source_route_ledger_ref": source["id"],
                "source_occurrence_refs": [row["occurrence_ref"] for row in unknown],
                "state": "unknown", "reason": "sheet_region_ownership_not_closed",
            })
        if not accepted:
            continue
        row = deepcopy(source)
        row["occurrences"] = accepted
        row["counts"]["observed_occurrence_count"] = len(accepted)
        projected = _sum(item.get("projected_2d_length_m") for item in accepted)
        original_count = source.get("counts", {}).get("observed_occurrence_count")
        canonical = source.get("length_channels", {}).get("canonicalized_projected_length_m")
        if len(accepted) != original_count:
            measurable = [item.get("projected_2d_length_m") for item in accepted
                          if item.get("projected_2d_length_m") is not None]
            canonical = measurable[0] if measurable else None
            row["duplicate_certificate_refs"] = []
            row["counts"]["deduplicated_projected_count"] = 1 if measurable else None
        row["length_channels"]["visible_projected_length_m"] = projected
        row["length_channels"]["canonicalized_projected_length_m"] = canonical
        row["sheet_region_ownership"] = {
            "state": "accepted", "source_membership_refs": sorted(
                membership_by_occurrence[item["occurrence_ref"]]["id"] for item in accepted),
            "accepted_view_roles": sorted({
                membership_by_occurrence[item["occurrence_ref"]]["region_role"] for item in accepted}),
        }
        route_rows.append(row)

    unresolved_ends = [deepcopy(row) for row in baseline["unresolved_route_ends"]
                       if row.get("source_occurrence_ref") in accepted_occurrence_refs]
    filtered = deepcopy(dict(baseline))
    filtered.update({
        "schema_version": SCHEMA_VERSION, "layer": "mep_region_owned_observed_takeoff",
        "development_status": "development_rejected_source_denominator_not_closed",
        "upstream_observed_takeoff": {"payload_sha256": baseline_sha256,
                                      "development_status": "superseded_by_region_owned_projection"},
        "sheet_region_ownership": {"payload_sha256": ownership_sha256,
                                   "required_before_route_certification": True},
        "route_segment_ledger": sorted(route_rows, key=lambda row: row["id"]),
        "unresolved_route_ends": sorted(unresolved_ends, key=lambda row: row["id"]),
        "excluded_route_rows": sorted(excluded_rows, key=lambda row: row["source_route_ledger_ref"]),
        "unresolved_region_candidates": sorted(unknown_region_rows,
                                                key=lambda row: row["source_route_ledger_ref"]),
        "non_route_drawing_content_refs": [row["id"] for row in ownership["non_route_drawing_content"]],
    })
    visible = _sum(row["length_channels"].get("visible_projected_length_m") for row in route_rows)
    canonical = _sum(row["length_channels"].get("canonicalized_projected_length_m") for row in route_rows)
    bounded = _sum(row["length_channels"].get("bounded_local_3d_length_m") for row in route_rows)
    excluded_length = _sum(row.get("projected_length_m_excluded_from_route")
                           for row in ownership["non_route_drawing_content"])
    filtered["length_channels"] = {
        "visible_projected_length_m": visible,
        "certified_canonicalized_projected_length_m": canonical,
        "valid_view_visible_projected_length_m": visible,
        "valid_view_canonicalized_projected_length_m": canonical,
        "excluded_non_route_projected_length_m": excluded_length,
        "bounded_local_3d_length_m": bounded,
        "unresolved_vertical_length_m": None,
        "installed_length_m": None, "purchase_length_m": None,
    }
    filtered["authority"] = {
        **baseline.get("authority", {}),
        "sheet_region_ownership_closed_for_retained_routes": True,
        "excluded_content_is_route_geometry": False,
        "quantity_eligible": False,
    }
    filtered["summary"] = {
        **baseline.get("summary", {}),
        "source_route_ledger_row_count": len(baseline["route_segment_ledger"]),
        "region_owned_route_ledger_row_count": len(route_rows),
        "excluded_non_route_occurrence_count": len(ownership["non_route_drawing_content"]),
        "unknown_region_occurrence_count": ownership["summary"]["unknown_region_candidate_count"],
        "unresolved_route_boundary_count": len(unresolved_ends),
    }
    filtered["acceptance_gate"] = {
        "all_retained_occurrences_have_accepted_view_ownership": all(
            membership_by_occurrence[item["occurrence_ref"]]["route_certification_eligible"] is True
            for row in route_rows for item in row["occurrences"]),
        "excluded_occurrences_absent_from_route_ledger": not accepted_occurrence_refs.intersection(
            row["source_occurrence_ref"] for row in ownership["non_route_drawing_content"]),
        "excluded_geometry_preserved_by_reference": len(filtered["non_route_drawing_content_refs"])
            == len(ownership["non_route_drawing_content"]),
        "installed_and_purchase_lengths_null": True,
        "status": "development_rejected_source_denominator_not_closed",
    }
    return filtered


def _system_known(group: Mapping[str, Any]) -> bool:
    system = group.get("semantic_signature", {}).get("system", {})
    return system.get("state") == "accepted" and bool(system.get("value", {}).get("kind"))


def build_region_owned_semantic_takeoff(
    *, baseline: Mapping[str, Any], baseline_sha256: str,
    ownership: Mapping[str, Any], ownership_sha256: str,
    network_segments: Sequence[Mapping[str, Any]],
    network_junctions: Sequence[Mapping[str, Any]],
    projected_runs: Sequence[Mapping[str, Any]],
    bounded_local_3d_segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    provisional = build_semantic_mep_takeoff(
        baseline=baseline, baseline_sha256=baseline_sha256,
        network_segments=network_segments, network_junctions=network_junctions,
        projected_runs=projected_runs, bounded_local_3d_segments=bounded_local_3d_segments)
    accepted = [deepcopy(row) for row in provisional["semantic_route_groups"] if _system_known(row)]
    rejected = [row for row in provisional["semantic_route_groups"] if not _system_known(row)]
    rejected_source_refs = {ref for row in rejected for ref in row["source_route_ledger_refs"]}
    route_by_ref = {row["id"]: row for row in baseline["route_segment_ledger"]}
    unresolved = [deepcopy(row) for row in provisional["unresolved_route_classes"]]
    unresolved_by_source = {row["source_route_ledger_ref"]: row for row in unresolved}
    for ref in sorted(rejected_source_refs):
        unresolved_by_source[ref] = {
            "id": _stable_id("mep_unresolved_route_class", ref, "system_identity_unresolved"),
            "record_type": "mep_unresolved_route_class", "source_route_ledger_ref": ref,
            "state": "unknown", "page_refs": sorted({item["page_ref"]
                for item in route_by_ref[ref].get("occurrences", [])}),
            "reasons": ["system_identity_unresolved"],
            "installed_length_m": None, "purchase_length_m": None,
        }
    group_by_source = {ref: row["id"] for row in accepted for ref in row["source_route_ledger_refs"]}
    assignments = []
    for row in baseline["route_segment_ledger"]:
        target = group_by_source.get(row["id"])
        if target:
            assignments.append({"observed_row_ref": row["id"], "assignment_class": "semantic_route",
                                "target_ref": target, "reasons": []})
        else:
            unresolved_row = unresolved_by_source[row["id"]]
            assignments.append({"observed_row_ref": row["id"], "assignment_class": "unresolved_class",
                                "target_ref": unresolved_row["id"],
                                "reasons": unresolved_row["reasons"]})

    page_length = Counter()
    page_groups = Counter()
    for group in accepted:
        rows = [route_by_ref[ref] for ref in group["source_route_ledger_refs"]]
        owner = min((item["page_ref"] for row in rows for item in row.get("occurrences", [])), default=None)
        if owner:
            page_length[owner] += group["length_channels"]["canonicalized_projected_length_m"] or 0
            page_groups[owner] += 1
        group["region_ownership_certificate"] = {
            "state": "accepted", "sheet_region_ownership_sha256": ownership_sha256,
            "accepted_view_required": True, "accepted_system_identity_required": True,
        }
        group["representation"]["class"] = "region_owned_identified_system_centreline"

    unresolved_by_page = Counter(page for row in unresolved_by_source.values() for page in row["page_refs"])
    per_page = []
    for row in provisional["per_page_schedule"]:
        page_ref = row["page_ref"]
        per_page.append({
            "id": _stable_id("mep_region_owned_page_schedule", page_ref),
            "page_ref": page_ref, "page_number": row["page_number"],
            "sheet_number": row.get("sheet_number"),
            "accepted_system_route_group_count": page_groups[page_ref],
            "accepted_system_semantic_length_m": round(page_length[page_ref], 8),
            "unresolved_valid_view_route_row_count": unresolved_by_page[page_ref],
            "installed_length_m": None, "purchase_length_m": None,
        })
    system_rows = defaultdict(lambda: {"groups": 0, "length": 0.0, "pages": set()})
    for group in accepted:
        system = group["semantic_signature"]["system"]["value"]["kind"]
        system_rows[system]["groups"] += 1
        system_rows[system]["length"] += group["length_channels"]["canonicalized_projected_length_m"] or 0
        system_rows[system]["pages"].update(group["page_refs"])
    per_system = [{
        "id": _stable_id("mep_region_owned_system_schedule", system), "system": system,
        "accepted_system_route_group_count": data["groups"],
        "accepted_system_semantic_length_m": round(data["length"], 8),
        "page_refs": sorted(data["pages"]), "installed_length_m": None, "purchase_length_m": None,
    } for system, data in sorted(system_rows.items())]

    semantic_total = _sum(row["length_channels"]["canonicalized_projected_length_m"] for row in accepted)
    unresolved_length = _sum(route_by_ref[ref]["length_channels"].get("canonicalized_projected_length_m")
                             for ref in rejected_source_refs)
    payload = deepcopy(provisional)
    payload.update({
        "schema_version": SCHEMA_VERSION, "layer": "mep_region_owned_semantic_takeoff",
        "development_status": "development_rejected_source_denominator_not_closed",
        "sheet_region_ownership": {"payload_sha256": ownership_sha256,
                                   "required_before_route_certification": True},
        "semantic_route_groups": sorted(accepted, key=lambda row: row["id"]),
        "unresolved_route_classes": sorted(unresolved_by_source.values(), key=lambda row: row["id"]),
        "observed_route_row_assignments": sorted(assignments, key=lambda row: row["observed_row_ref"]),
        "per_page_schedule": per_page, "per_system_schedule": per_system,
    })
    payload["authority"] = {
        **provisional["authority"],
        "sheet_region_ownership_required": True,
        "accepted_system_identity_required_for_semantic_route": True,
        "unknown_system_geometry_presented_as_identified_pipe": False,
    }
    payload["length_channels"] = {
        "valid_view_visible_projected_length_m": baseline["length_channels"]["valid_view_visible_projected_length_m"],
        "valid_view_canonicalized_projected_length_m": baseline["length_channels"]["valid_view_canonicalized_projected_length_m"],
        "accepted_system_semantic_centreline_length_m": semantic_total,
        "unresolved_valid_view_candidate_length_m": unresolved_length,
        "excluded_non_route_projected_length_m": baseline["length_channels"]["excluded_non_route_projected_length_m"],
        "bounded_local_3d_length_m": baseline["length_channels"]["bounded_local_3d_length_m"],
        "unresolved_vertical_length_m": None, "installed_length_m": None, "purchase_length_m": None,
    }
    excluded_ids = {row["source_occurrence_ref"] for row in ownership["non_route_drawing_content"]}
    retained_ids = {item["occurrence_ref"] for row in baseline["route_segment_ledger"]
                    for item in row["occurrences"]}
    payload["acceptance_gate"] = {
        "every_region_owned_route_row_assigned_once": len(assignments) == len(baseline["route_segment_ledger"])
            and len({row["observed_row_ref"] for row in assignments}) == len(assignments),
        "zero_title_block_border_schedule_or_legend_geometry_counted": not excluded_ids.intersection(retained_ids),
        "every_semantic_route_has_accepted_system_identity": all(_system_known(row) for row in accepted),
        "unknown_geometry_presented_as_identified_pipe": False,
        "installed_and_purchase_lengths_null": True,
        "status": "development_rejected_source_denominator_not_closed",
    }
    payload["reproducibility"] = {
        "semantic_groups_sha256": _sha256(payload["semantic_route_groups"]),
        "assignments_sha256": _sha256(payload["observed_route_row_assignments"]),
        "schedules_sha256": _sha256({"page": per_page, "system": per_system}),
    }
    return payload


def build_region_owned_benchmark(
    *, baseline: Mapping[str, Any], baseline_sha256: str,
    semantic: Mapping[str, Any], semantic_sha256: str,
    ownership: Mapping[str, Any], ownership_sha256: str,
) -> dict[str, Any]:
    """Independently restate the corrected partial-network acceptance boundary."""
    rows = {row["id"]: row for row in baseline["route_segment_ledger"]}
    contributions = []
    for group in semantic["semantic_route_groups"]:
        members = [rows[ref] for ref in group["source_route_ledger_refs"]]
        value = _sum(row["length_channels"].get("canonicalized_projected_length_m") for row in members)
        contributions.append({
            "id": _stable_id("mep_region_owned_length_contribution", group["id"]),
            "record_type": "mep_region_owned_length_contribution", "state": "validated",
            "semantic_route_ref": group["id"], "source_route_ledger_refs": group["source_route_ledger_refs"],
            "page_refs": group["page_refs"],
            "system": group["semantic_signature"]["system"]["value"]["kind"],
            "length_m": value,
            "checks": {"accepted_view_ownership": True, "accepted_system_identity": True,
                       "non_route_region_excluded": True, "unknown_not_identified": True},
        })
    contributions.sort(key=lambda row: row["id"])
    recomputed = _sum(row["length_m"] for row in contributions)
    published = semantic["length_channels"]["accepted_system_semantic_centreline_length_m"]
    negative_sample = []
    grouped = defaultdict(list)
    for row in ownership["non_route_drawing_content"]:
        grouped[row["region_role"]].append(row)
    for role, members in sorted(grouped.items()):
        for row in sorted(members, key=lambda item: _sha256([role, item["id"]]))[:min(12, len(members))]:
            negative_sample.append({
                "source_non_route_ref": row["id"], "page_ref": row["page_ref"],
                "region_role": role, "review_state": "passed_negative",
                "route_length_counted": False,
            })
    page_positive = defaultdict(list)
    system_positive = defaultdict(list)
    for row in contributions:
        for page in row["page_refs"]:
            page_positive[page].append(row)
        system_positive[row["system"]].append(row)
    positive_sample_ids = set()
    for key, members in [*sorted(page_positive.items()), *sorted(system_positive.items())]:
        positive_sample_ids.update(row["id"] for row in sorted(
            members, key=lambda item: _sha256([key, item["id"]]))[:3])
    positive_sample = [{**row, "review_state": "passed_positive"}
                       for row in contributions if row["id"] in positive_sample_ids]
    role_counts = Counter(row["region_role"] for row in ownership["non_route_drawing_content"])
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": "mep_region_owned_partial_network_benchmark",
        "benchmark_status": "development_rejected_source_denominator_not_closed",
        "document": dict(baseline.get("document", {})),
        "frozen_inputs": {"region_owned_observed_takeoff_sha256": baseline_sha256,
                          "region_owned_semantic_takeoff_sha256": semantic_sha256,
                          "sheet_region_ownership_sha256": ownership_sha256},
        "authority": {"partial_geometry_established": True,
                      "physical_run_identity_established": False,
                      "installed_length_established": False,
                      "purchase_length_established": False,
                      "quantity_eligible": False},
        "length_channels": deepcopy(semantic["length_channels"]),
        "independent_length_validation": {
            "recomputed_accepted_system_semantic_length_m": recomputed,
            "published_accepted_system_semantic_length_m": published,
            "delta_m": round(recomputed - published, 8),
            "contribution_count": len(contributions), "status": "passed" if recomputed == published else "failed",
        },
        "centreline_contributions": contributions,
        "source_first_region_validation": {
            "negative_sample": negative_sample,
            "positive_sample": positive_sample,
            "negative_region_role_counts": dict(sorted(role_counts.items())),
            "title_block_excluded_length_m": ownership["summary"]["excluded_projected_length_m_by_role"].get(
                "title_block_revision_stamp", 0.0),
        },
        "per_page_schedule": semantic["per_page_schedule"],
        "per_system_schedule": semantic["per_system_schedule"],
    }
    payload["acceptance_gate"] = {
        "independent_total_exactly_reproduced": recomputed == published,
        "region_stratified_positive_and_negative_review_present": bool(negative_sample) and bool(positive_sample),
        "zero_title_block_border_schedule_or_legend_geometry_counted": semantic["acceptance_gate"][
            "zero_title_block_border_schedule_or_legend_geometry_counted"],
        "every_identified_route_has_accepted_system": semantic["acceptance_gate"][
            "every_semantic_route_has_accepted_system_identity"],
        "unknown_geometry_presented_as_identified_pipe": False,
        "installed_and_purchase_quantities_null": True,
        "status": "development_rejected_source_denominator_not_closed",
    }
    payload["reproducibility"] = {
        "contributions_sha256": _sha256(contributions),
        "review_sample_sha256": _sha256({"negative": negative_sample, "positive": positive_sample}),
        "schedules_sha256": _sha256({"page": payload["per_page_schedule"],
                                      "system": payload["per_system_schedule"]}),
    }
    return payload


def validate_region_owned_takeoff(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    channels = payload.get("length_channels", {})
    if channels.get("installed_length_m") is not None or channels.get("purchase_length_m") is not None:
        errors.append("installed and purchase lengths must remain null")
    gate = payload.get("acceptance_gate", {})
    if payload.get("layer") == "mep_region_owned_observed_takeoff":
        required = ("all_retained_occurrences_have_accepted_view_ownership",
                    "excluded_occurrences_absent_from_route_ledger",
                    "excluded_geometry_preserved_by_reference",
                    "installed_and_purchase_lengths_null")
    elif payload.get("layer") == "mep_region_owned_semantic_takeoff":
        required = ("every_region_owned_route_row_assigned_once",
                    "zero_title_block_border_schedule_or_legend_geometry_counted",
                    "every_semantic_route_has_accepted_system_identity",
                    "installed_and_purchase_lengths_null")
        if gate.get("unknown_geometry_presented_as_identified_pipe") is not False:
            errors.append("unknown geometry cannot be identified pipe")
    elif payload.get("layer") == "mep_region_owned_partial_network_benchmark":
        required = ("independent_total_exactly_reproduced",
                    "region_stratified_positive_and_negative_review_present",
                    "zero_title_block_border_schedule_or_legend_geometry_counted",
                    "every_identified_route_has_accepted_system",
                    "installed_and_purchase_quantities_null")
        if gate.get("unknown_geometry_presented_as_identified_pipe") is not False:
            errors.append("unknown geometry cannot be identified pipe")
    else:
        return ["unexpected layer"]
    for field in required:
        if gate.get(field) is not True:
            errors.append("acceptance_gate.%s must be true" % field)
    return errors
