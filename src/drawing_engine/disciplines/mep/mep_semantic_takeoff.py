"""Semantic consolidation over a frozen observed MEP takeoff baseline.

The compiler converts only M5C-certified outlined routes to semantic
centrelines.  It reuses accepted pass-through junctions and accepted duplicate
projection certificates, while retaining every other observation as an
explicit unresolved assignment.  Installed and purchase authority are outside
this layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
ROUTE_FIELDS = ("system", "size", "elevation")
DISCRETE_CLASSES = ("elbow", "tee", "coupling", "valve", "damper", "equipment", "unresolved_symbol")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    return kind + "." + _sha256([kind, *parts])[:20]


def _numeric_sum(values) -> float:
    return round(sum(value for value in values if value is not None), 8)


def _distinct_values(field: str, overlays: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for overlay in overlays:
        if overlay.get("state") != "accepted":
            continue
        for value in overlay.get("observed_values", []):
            if field == "system":
                normalised = {"kind": value.get("kind"), "category": value.get("category")}
            elif field == "size":
                normalised = {"kind": value.get("kind"), "designation": value.get("designation"),
                              "value": value.get("value"), "unit": value.get("unit")}
            else:
                normalised = {"kind": value.get("kind"), "basis": value.get("basis"),
                              "value": value.get("value"), "unit": value.get("unit")}
            values[_canonical_json(normalised)] = normalised
    return [values[key] for key in sorted(values)]


def _consolidated_field(field: str, overlays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = _distinct_values(field, overlays)
    relation_refs = sorted({ref for overlay in overlays
                            for ref in overlay.get("accepted_relation_refs", [])})
    if not values:
        return {"state": "unknown", "value": None, "alternatives": [],
                "accepted_relation_refs": []}
    if len(values) == 1:
        return {"state": "accepted", "value": values[0], "alternatives": [],
                "accepted_relation_refs": relation_refs}
    return {"state": "conflicted", "value": None, "alternatives": values,
            "accepted_relation_refs": relation_refs}


def _signature(semantics: Mapping[str, Any]) -> str:
    compact = {}
    for field in (*ROUTE_FIELDS, "shape", "material"):
        value = semantics[field]
        compact[field] = {"state": value["state"], "value": value.get("value"),
                          "alternatives": value.get("alternatives", [])}
    return _canonical_json(compact)


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _item_class(row: Mapping[str, Any]) -> tuple[str, str]:
    observed = str(row.get("observed_item_class") or "").lower()
    generic = row.get("generic_class")
    if generic == "equipment" or row.get("observed_item_type") == "equipment":
        return "equipment", "explicit_equipment_observation"
    if observed == "elbow" or generic == "bend":
        return "elbow", "observed_bend_or_elbow_geometry"
    if generic == "tee":
        return "tee", "observed_three_port_branch_geometry"
    if generic == "coupling":
        return "coupling", "explicit_coupling_identity"
    if "valve" in observed:
        return "valve", "observed_valve_class"
    if "damper" in observed:
        return "damper", "observed_damper_class"
    return "unresolved_symbol", "generic_identity_not_uniquely_authored"


def _page_rollups(baseline: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = defaultdict(lambda: {"visible": [], "canonical": [], "bounded": [],
                                  "unmeasurable": 0, "route_rows": set()})
    for row in baseline["route_segment_ledger"]:
        occurrences = row.get("occurrences", [])
        for occurrence in occurrences:
            page_ref = occurrence["page_ref"]
            result[page_ref]["route_rows"].add(row["id"])
            value = occurrence.get("projected_2d_length_m")
            if value is None:
                result[page_ref]["unmeasurable"] += 1
            else:
                result[page_ref]["visible"].append(value)
        if occurrences:
            owner = sorted(occurrences, key=lambda value: (value["page_ref"], value["occurrence_ref"]))[0]["page_ref"]
            result[owner]["canonical"].append(row["length_channels"].get("canonicalized_projected_length_m"))
            result[owner]["bounded"].append(row["length_channels"].get("bounded_local_3d_length_m"))
    return result


def build_semantic_mep_takeoff(
    *, baseline: Mapping[str, Any], baseline_sha256: str,
    network_segments: Sequence[Mapping[str, Any]],
    network_junctions: Sequence[Mapping[str, Any]],
    projected_runs: Sequence[Mapping[str, Any]],
    bounded_local_3d_segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build semantic schedules without crossing the physical-run boundary."""
    route_rows = list(baseline["route_segment_ledger"])
    row_by_id = {row["id"]: row for row in route_rows}
    occurrence_to_row = {occurrence["occurrence_ref"]: row["id"]
                         for row in route_rows for occurrence in row.get("occurrences", [])}
    segment_by_id = {row["id"]: row for row in network_segments}
    segment_to_rows: dict[str, set[str]] = defaultdict(set)
    row_to_segments: dict[str, set[str]] = defaultdict(set)
    for segment in network_segments:
        for occurrence_ref in segment.get("source_occurrence_refs", []):
            row_ref = occurrence_to_row.get(occurrence_ref)
            if row_ref:
                segment_to_rows[segment["id"]].add(row_ref)
                row_to_segments[row_ref].add(segment["id"])

    bounded_by_canonical = {row.get("canonical_projected_segment_ref"): row
                            for row in bounded_local_3d_segments if row.get("state") == "accepted"}
    row_semantics: dict[str, dict[str, Any]] = {}
    for row_ref, segment_refs in row_to_segments.items():
        segments = [segment_by_id[ref] for ref in sorted(segment_refs)]
        semantics = {field: _consolidated_field(
            field, [segment.get("semantic_overlay", {}).get(field, {}) for segment in segments])
            for field in ROUTE_FIELDS}
        bounded = bounded_by_canonical.get(row_by_id[row_ref].get("canonical_projected_segment_ref"))
        shape = None if bounded is None else bounded.get("physical_envelope_dimension", {}).get("shape")
        semantics["shape"] = {"state": "derived" if shape else "unknown", "value": shape,
                              "alternatives": [], "evidence_refs": [] if bounded is None else [bounded["id"]]}
        semantics["material"] = {"state": "unknown", "value": None, "alternatives": [],
                                 "evidence_refs": []}
        row_semantics[row_ref] = semantics

    certified_rows = sorted(row_to_segments)
    unions = _UnionFind(certified_rows)
    accepted_pass_through = [row for row in network_junctions
                             if row.get("state") == "accepted" and row.get("run_pass_through") is True]
    accepted_join_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for junction in accepted_pass_through:
        members = sorted({row_ref for segment_ref in junction.get("segment_refs", [])
                          for row_ref in segment_to_rows.get(segment_ref, ())})
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if _signature(row_semantics[left]) == _signature(row_semantics[right]):
                    unions.union(left, right)
                    accepted_join_by_pair[(min(left, right), max(left, right))].add(junction["id"])

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for row_ref in certified_rows:
        members_by_root[unions.find(row_ref)].append(row_ref)
    run_refs_by_segment = defaultdict(set)
    for run in projected_runs:
        for segment_ref in run.get("segment_refs", []):
            run_refs_by_segment[segment_ref].add(run["id"])

    route_groups = []
    row_target: dict[str, str] = {}
    for members in sorted(members_by_root.values(), key=lambda value: tuple(sorted(value))):
        members = sorted(members)
        segments = sorted({ref for row_ref in members for ref in row_to_segments[row_ref]})
        joins = sorted({ref for index, left in enumerate(members) for right in members[index + 1:]
                        for ref in accepted_join_by_pair.get((left, right), ())})
        rows = [row_by_id[ref] for ref in members]
        group_id = _stable_id("mep_semantic_route", members)
        pages = sorted({occurrence["page_ref"] for row in rows for occurrence in row.get("occurrences", [])})
        semantics = row_semantics[members[0]]
        route_groups.append({
            "id": group_id, "record_type": "mep_semantic_route_group", "state": "derived",
            "source_route_ledger_refs": members,
            "source_occurrence_refs": sorted({occurrence["occurrence_ref"] for row in rows
                                               for occurrence in row.get("occurrences", [])}),
            "network_segment_refs": segments,
            "projected_run_refs": sorted({ref for segment in segments for ref in run_refs_by_segment[segment]}),
            "accepted_pass_through_junction_refs": joins,
            "page_refs": pages,
            "representation": {
                "class": "certified_outlined_semantic_centreline",
                "sidewalls_collapsed_once": True,
                "m5c_geometric_basis": "replayed_m3_5_outline",
            },
            "semantic_signature": semantics,
            "length_channels": {
                "visible_projected_length_m": _numeric_sum(
                    occurrence.get("projected_2d_length_m") for row in rows
                    for occurrence in row.get("occurrences", [])),
                "canonicalized_projected_length_m": _numeric_sum(
                    row["length_channels"].get("canonicalized_projected_length_m") for row in rows),
                "bounded_local_3d_length_m": _numeric_sum(
                    row["length_channels"].get("bounded_local_3d_length_m") for row in rows),
                "unresolved_vertical_length_m": None,
                "installed_length_m": None,
                "purchase_length_m": None,
            },
            "merge_certificate": {
                "state": "accepted" if len(members) > 1 else "not_applicable",
                "basis": "accepted_m5c_pass_through_with_exact_semantic_signature",
                "junction_refs": joins,
                "merged_interval_count": len(members),
            },
            "authority": {"projected_centreline_established": True,
                          "physical_run_identity_established": False,
                          "installed_length_established": False,
                          "purchase_length_established": False,
                          "quantity_eligible": False},
        })
        for row_ref in members:
            row_target[row_ref] = group_id

    route_assignments = []
    unresolved_route_rows = []
    for row in route_rows:
        target = row_target.get(row["id"])
        if target:
            assignment = "semantic_route"
            reasons = []
        else:
            assignment = "unresolved_class"
            target = _stable_id("mep_unresolved_route_class", row["id"])
            reasons = ["no_m5c_certified_outlined_centreline"]
            unresolved_route_rows.append({
                "id": target, "record_type": "mep_unresolved_route_class",
                "source_route_ledger_ref": row["id"], "state": "unknown",
                "page_refs": sorted({item["page_ref"] for item in row.get("occurrences", [])}),
                "reasons": reasons, "installed_length_m": None, "purchase_length_m": None,
            })
        route_assignments.append({"observed_row_ref": row["id"], "assignment_class": assignment,
                                  "target_ref": target, "reasons": reasons})

    junction_by_id = {row["id"]: row for row in network_junctions}
    discrete_items = []
    item_system_key: dict[str, str] = {}
    for row in baseline["item_occurrence_ledger"]:
        item_class, basis = _item_class(row)
        junction = junction_by_id.get(row.get("source_record_ref"))
        overlays = [] if junction is None else [segment_by_id[ref].get("semantic_overlay", {}).get("system", {})
                                                for ref in junction.get("segment_refs", [])
                                                if ref in segment_by_id]
        system = _consolidated_field("system", overlays)
        item_system_key[row["id"]] = _canonical_json({"system": system})
        discrete_items.append({
            "id": _stable_id("mep_semantic_discrete_item", row["id"]),
            "record_type": "mep_semantic_discrete_item", "state": row.get("state"),
            "source_observed_item_ref": row["id"], "page_ref": row.get("page_ref"),
            "semantic_class": item_class, "classification_basis": basis,
            "classification_alternatives": row.get("classification_alternatives", []),
            "system": system,
            "material": None, "connection_standard": None, "exact_sku": None,
            "counts": {"observed_occurrence_count": 1,
                       "deduplicated_projected_count": None,
                       "physical_instance_count": None},
            "deduplication": {"state": "unresolved",
                              "reason": "explicit_projection_registration_identity_not_established",
                              "certificate_refs": []},
            "authority": {"physical_item_identity_established": False,
                          "review_changes_evidence_state": False,
                          "quantity_eligible": False},
        })

    item_counts = {name: {"observed_occurrence_count": 0,
                          "deduplicated_projected_count": None,
                          "physical_instance_count": None}
                   for name in DISCRETE_CLASSES}
    for item in discrete_items:
        item_counts[item["semantic_class"]]["observed_occurrence_count"] += 1

    coverage_by_page = {row["page_ref"]: row for row in baseline["sheet_coverage"]}
    page_rollup = _page_rollups(baseline)
    boundary_count = Counter(row.get("page_ref") for row in baseline["unresolved_route_ends"])
    item_count = Counter(row.get("page_ref") for row in discrete_items)
    per_page = []
    for page_ref, coverage in sorted(coverage_by_page.items(), key=lambda pair: pair[1]["page_number"]):
        rollup = page_rollup[page_ref]
        has_occurrences = bool(rollup["route_rows"])
        per_page.append({
            "id": _stable_id("mep_semantic_page_schedule", page_ref),
            "page_number": coverage["page_number"], "page_ref": page_ref,
            "sheet_number": coverage.get("sheet_number"),
            "visible_projected_length_m": _numeric_sum(rollup["visible"]) if rollup["visible"] else None if has_occurrences else 0.0,
            "canonicalized_projected_length_m": _numeric_sum(rollup["canonical"]) if any(v is not None for v in rollup["canonical"]) else None if has_occurrences else 0.0,
            "bounded_local_3d_length_m": _numeric_sum(rollup["bounded"]),
            "unmeasurable_occurrence_count": rollup["unmeasurable"],
            "unresolved_boundary_count": boundary_count[page_ref],
            "observed_item_count": item_count[page_ref], "physical_item_count": None,
            "installed_length_m": None, "purchase_length_m": None,
            "canonical_ownership_basis": "lexicographically_first_occurrence_page_per_observed_ledger_row",
        })

    def system_bucket_from_semantics(semantics):
        system = semantics["system"]
        value = system.get("value")
        return value.get("kind") if system.get("state") == "accepted" and value else "unknown_or_conflicted"

    route_group_by_id = {row["id"]: row for row in route_groups}
    system_rows = defaultdict(lambda: {"route_refs": [], "visible": [], "canonical": [], "bounded": [],
                                       "unmeasurable": 0, "boundaries": 0, "items": 0})
    for row in route_rows:
        target = row_target.get(row["id"])
        semantics = route_group_by_id[target]["semantic_signature"] if target else None
        bucket = system_bucket_from_semantics(semantics) if semantics else "unknown_or_conflicted"
        data = system_rows[bucket]
        data["route_refs"].append(row["id"])
        for occurrence in row.get("occurrences", []):
            value = occurrence.get("projected_2d_length_m")
            if value is None: data["unmeasurable"] += 1
            else: data["visible"].append(value)
        data["canonical"].append(row["length_channels"].get("canonicalized_projected_length_m"))
        data["bounded"].append(row["length_channels"].get("bounded_local_3d_length_m"))
    segment_system = {ref: system_bucket_from_semantics({"system": _consolidated_field(
        "system", [segment.get("semantic_overlay", {}).get("system", {})])})
        for ref, segment in segment_by_id.items()}
    for boundary in baseline["unresolved_route_ends"]:
        system_rows[segment_system.get(boundary.get("network_segment_ref"), "unknown_or_conflicted")]["boundaries"] += 1
    for item in discrete_items:
        system = item["system"]
        value = system.get("value")
        bucket = value.get("kind") if system.get("state") == "accepted" and value else "unknown_or_conflicted"
        system_rows[bucket]["items"] += 1
    per_system = []
    for bucket, data in sorted(system_rows.items()):
        per_system.append({
            "id": _stable_id("mep_semantic_system_schedule", bucket), "system": bucket,
            "source_route_ledger_refs": sorted(data["route_refs"]),
            "visible_projected_length_m": _numeric_sum(data["visible"]) if data["visible"] else None,
            "canonicalized_projected_length_m": _numeric_sum(data["canonical"]),
            "bounded_local_3d_length_m": _numeric_sum(data["bounded"]),
            "unmeasurable_occurrence_count": data["unmeasurable"],
            "unresolved_boundary_count": data["boundaries"],
            "observed_item_count": data["items"], "physical_item_count": None,
            "installed_length_m": None, "purchase_length_m": None,
        })

    top_rows = sorted(route_rows, key=lambda row: row["length_channels"].get("visible_projected_length_m") or -1,
                      reverse=True)[:50]
    contributors = []
    for rank, row in enumerate(top_rows, 1):
        certified = row["id"] in row_target
        duplicate_held = bool(row.get("duplicate_competitor_refs"))
        contributors.append({
            "id": _stable_id("mep_length_contributor_audit", row["id"]), "rank": rank,
            "source_route_ledger_ref": row["id"],
            "visible_projected_length_m": row["length_channels"].get("visible_projected_length_m"),
            "canonicalized_projected_length_m": row["length_channels"].get("canonicalized_projected_length_m"),
            "page_refs": sorted({item["page_ref"] for item in row.get("occurrences", [])}),
            "source_primitive_refs": sorted({ref for item in row.get("occurrences", [])
                                             for ref in item.get("source_primitive_refs", [])}),
            "audit_state": ("held_unresolved_duplicate" if certified and duplicate_held else
                            "passed" if certified else "held_unresolved"),
            "dimension_hatching_check": "passed_by_m5c_m3_5_role_replay" if certified else "not_certified",
            "sidewall_check": "collapsed_once_to_certified_centreline" if certified else "not_applicable",
            "duplicate_check": "accepted_certificate_reused" if row.get("duplicate_certificate_refs") else
                               "unresolved_competitor_excluded" if row.get("duplicate_competitor_refs") else "no_duplicate_claim",
            "semantic_schedule_treatment": ("certified_centreline_excluded_from_canonical_total"
                                            if certified and duplicate_held else
                                            "certified_semantic_centreline" if certified else "unresolved_class"),
        })

    branch_rows = [row for row in network_junctions if "branch" in row.get("relation_type", "")]
    topology = {
        "branches": {"observed_count": len(branch_rows),
                     "accepted_count": sum(row.get("state") == "accepted" for row in branch_rows),
                     "source_record_refs": sorted(row["id"] for row in branch_rows)},
        "crossings_by_page": [{"page_ref": row["page_ref"],
                               "observed_count": row.get("m3_observation_counts", {}).get("crossing_count", 0)}
                              for row in baseline["sheet_coverage"]],
        "open_ends": {"observed_boundary_count": len(baseline["unresolved_route_ends"]),
                      "source_record_refs": [row["id"] for row in baseline["unresolved_route_ends"]]},
        "unscaled_occurrences": {"observed_count": sum(
            occurrence.get("projected_2d_length_m") is None for row in route_rows
            for occurrence in row.get("occurrences", [])),
            "source_occurrence_refs": sorted(occurrence["occurrence_ref"] for row in route_rows
                                             for occurrence in row.get("occurrences", [])
                                             if occurrence.get("projected_2d_length_m") is None)},
    }

    review_overlays = []
    for row in per_page + per_system:
        review_overlays.append({
            "id": _stable_id("mep_semantic_takeoff_review", row["id"]),
            "record_type": "mep_line_level_engineer_review_overlay",
            "schedule_line_ref": row["id"], "review_state": "not_reviewed",
            "review_decision": None, "reviewer": None, "reviewed_at": None,
            "frozen_baseline_sha256": baseline_sha256,
            "evidence_state_unchanged": True, "promotes_to_direct_evidence": False,
            "grants_installed_or_purchase_authority": False,
        })

    payload = {
        "schema_version": SCHEMA_VERSION, "layer": "mep_semantic_takeoff_consolidation",
        "document": dict(baseline.get("document", {})),
        "baseline": {"layer": baseline.get("layer"), "payload_sha256": baseline_sha256,
                     "route_row_count": len(route_rows),
                     "item_row_count": len(baseline["item_occurrence_ledger"]),
                     "frozen": True},
        "authority": {"observed_takeoff_baseline_frozen": True,
                      "semantic_centreline_requires_m5c_outline_certificate": True,
                      "only_explicit_overlap_duplicates_removed": True,
                      "physical_run_identity_established": False,
                      "installed_length_established": False,
                      "purchase_length_established": False,
                      "connector_selection_established": False,
                      "marketplace_assembly_established": False,
                      "quantity_eligible": False},
        "length_channels": {
            "visible_projected_length_m": baseline["length_channels"]["visible_projected_length_m"],
            "canonicalized_projected_length_m": baseline["length_channels"]["certified_canonicalized_projected_length_m"],
            "certified_semantic_centreline_length_m": _numeric_sum(
                group["length_channels"]["canonicalized_projected_length_m"] for group in route_groups),
            "bounded_local_3d_length_m": baseline["length_channels"]["bounded_local_3d_length_m"],
            "unresolved_vertical_length_m": None, "installed_length_m": None,
            "purchase_length_m": None,
        },
        "semantic_route_groups": sorted(route_groups, key=lambda row: row["id"]),
        "unresolved_route_classes": sorted(unresolved_route_rows, key=lambda row: row["id"]),
        "observed_route_row_assignments": sorted(route_assignments, key=lambda row: row["observed_row_ref"]),
        "discrete_items": sorted(discrete_items, key=lambda row: row["id"]),
        "observed_item_row_assignments": sorted([
            {"observed_row_ref": row["source_observed_item_ref"],
             "assignment_class": ("unresolved_class" if row["semantic_class"] == "unresolved_symbol"
                                  else "discrete_item"),
             "target_ref": row["id"], "semantic_class": row["semantic_class"]}
            for row in discrete_items], key=lambda row: row["observed_row_ref"]),
        "discrete_category_counts": item_counts,
        "topology_observations": topology,
        "largest_length_contributor_audit": contributors,
        "per_page_schedule": per_page, "per_system_schedule": per_system,
        "engineer_review_overlays": sorted(review_overlays, key=lambda row: row["id"]),
        "development_fixture": {
            "separate_complete_physical_run_positive_fixture_required": True,
            "current_package_role": "observed_takeoff_and_real_negative_fixture",
            "required_positive_closures": ["terminal_identity", "complete_physical_topology",
                                            "connection_standard", "connector_selection",
                                            "installed_length", "marketplace_sku"],
        },
    }
    payload["acceptance_gate"] = {
        "every_observed_route_row_assigned_once": len(route_assignments) == len(route_rows)
            and len({row["observed_row_ref"] for row in route_assignments}) == len(route_rows),
        "every_observed_discrete_row_assigned_once": len(discrete_items) == len(baseline["item_occurrence_ledger"])
            and len({row["source_observed_item_ref"] for row in discrete_items}) == len(discrete_items),
        "discrete_classification_total": sum(row["observed_occurrence_count"] for row in item_counts.values()),
        "reproducible_schedule_totals": (
            round(_numeric_sum(row["visible_projected_length_m"] for row in per_page), 8)
            == round(baseline["length_channels"]["visible_projected_length_m"], 8)
            and round(_numeric_sum(row["canonicalized_projected_length_m"] for row in per_page), 8)
            == round(baseline["length_channels"]["certified_canonicalized_projected_length_m"], 8)),
        "installed_and_purchase_lengths_null": True,
        "status": "accepted_observed_semantic_baseline",
    }
    payload["reproducibility"] = {
        "schedule_totals_sha256": _sha256({"per_page": per_page, "per_system": per_system}),
        "assignment_sha256": _sha256({"routes": payload["observed_route_row_assignments"],
                                      "items": [(row["source_observed_item_ref"], row["semantic_class"])
                                                for row in payload["discrete_items"]]}),
    }
    return payload


def validate_semantic_mep_takeoff(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("layer") != "mep_semantic_takeoff_consolidation":
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
    gate = payload.get("acceptance_gate", {})
    for field in ("every_observed_route_row_assigned_once", "every_observed_discrete_row_assigned_once",
                  "reproducible_schedule_totals", "installed_and_purchase_lengths_null"):
        if gate.get(field) is not True:
            errors.append("acceptance_gate.%s must be true" % field)
    if gate.get("discrete_classification_total") != payload.get("baseline", {}).get("item_row_count"):
        errors.append("discrete classification total mismatch")
    for row in payload.get("semantic_route_groups", []):
        if row.get("representation", {}).get("class") != "certified_outlined_semantic_centreline":
            errors.append("semantic route lacks outline centreline certificate: " + str(row.get("id")))
        if row.get("length_channels", {}).get("installed_length_m") is not None:
            errors.append("semantic route installed length must remain null: " + str(row.get("id")))
    for row in payload.get("discrete_items", []):
        if row.get("semantic_class") not in DISCRETE_CLASSES:
            errors.append("unexpected discrete class: " + str(row.get("semantic_class")))
        if row.get("counts", {}).get("physical_instance_count") is not None:
            errors.append("physical item count must remain null: " + str(row.get("id")))
        if any(row.get(field) is not None for field in ("material", "connection_standard", "exact_sku")):
            errors.append("unsupported material, connection standard, and SKU must remain null: " + str(row.get("id")))
    if any(row.get("promotes_to_direct_evidence") is not False
           for row in payload.get("engineer_review_overlays", [])):
        errors.append("review overlays must not promote direct evidence")
    return errors
