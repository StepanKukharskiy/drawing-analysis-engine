"""Fail-closed endpoint coverage for geometry-ready MEP run candidates.

The matrix is an investigation adapter, not a second topology solver.  It
canonicalizes duplicate page occurrences, preserves every bounded source
query and competitor, and selects a replay candidate only when both physical
endpoints have one explicit, uniquely bound terminal identity.  An outlined
envelope closure or an inferred bend remains a candidate, never a terminal.
Connection-standard discovery is reported independently from geometry.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from src.drawing_engine.project.takeoff_intelligence import canonical_sha256, stable_id


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_endpoint_coverage_matrix"


def _accepted_terminal(candidate: Mapping[str, Any]) -> bool:
    return (
        candidate.get("state") == "accepted"
        and candidate.get("explicit_terminal_identity") is True
        and candidate.get("uniquely_bound_to_endpoint") is True
        and isinstance(candidate.get("identity_key"), str)
        and bool(candidate.get("identity_key"))
    )


def _endpoint_outcome(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = bool(rows) and all(row.get("native_search_complete") is True for row in rows)
    accepted = [candidate for row in rows for candidate in row.get("interface_candidates", [])
                if _accepted_terminal(candidate)]
    keys = sorted({str(candidate["identity_key"]) for candidate in accepted})
    competitor_refs = sorted({str(candidate.get("id")) for row in rows
                              for candidate in row.get("interface_candidates", [])
                              if candidate.get("id")})
    source_refs = sorted({str(ref) for row in rows for ref in row.get("source_query_refs", [])})
    if complete and len(keys) == 1 and all(any(
            _accepted_terminal(candidate) and candidate["identity_key"] == keys[0]
            for candidate in row.get("interface_candidates", [])) for row in rows):
        state, reasons = "closed", []
    elif not complete:
        state, reasons = "not_searched", ["bounded_endpoint_source_search_incomplete"]
    elif accepted:
        state, reasons = "ambiguous", ["accepted_terminal_bindings_not_mutually_unique_across_occurrences"]
    elif competitor_refs:
        state, reasons = "ambiguous", ["endpoint_candidates_lack_explicit_unique_terminal_identity"]
    else:
        state, reasons = "apparently_absent", ["no_authored_terminal_identity_in_complete_bounded_search"]
    return {
        "state": state,
        "reasons": reasons,
        "canonical_terminal_identity_key": keys[0] if state == "closed" else None,
        "occurrence_count": len(rows),
        "complete_native_occurrence_search_count": sum(
            row.get("native_search_complete") is True for row in rows
        ),
        "raster_search_state_counts": dict(Counter(
            str(row.get("raster_search_state")) for row in rows
        )),
        "candidate_refs": competitor_refs,
        "source_query_refs": source_refs,
        "quantity_eligible": False,
    }


def build_endpoint_coverage_matrix(
    *, document: Mapping[str, Any], candidate_runs: Sequence[Mapping[str, Any]],
    endpoint_searches: Sequence[Mapping[str, Any]],
    connection_standard_search: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a ranked endpoint matrix and nominate at most one replay input."""
    run_ids = [str(row.get("network_run_ref")) for row in candidate_runs]
    if len(run_ids) != len(set(run_ids)) or any(not ref for ref in run_ids):
        raise ValueError("candidate runs require distinct network references")
    searches_by_run: dict[str, list[Mapping[str, Any]]] = {ref: [] for ref in run_ids}
    search_ids = []
    for row in endpoint_searches:
        identifier = str(row.get("id"))
        search_ids.append(identifier)
        run_ref = str(row.get("network_run_ref"))
        if run_ref not in searches_by_run:
            raise ValueError("endpoint search refers outside geometry-ready candidates")
        if row.get("canonical_endpoint_index") not in (0, 1):
            raise ValueError("endpoint search lacks canonical endpoint index")
        if not isinstance(row.get("source_inventory_sha256"), str):
            raise ValueError("endpoint search lacks immutable source inventory hash")
        searches_by_run[run_ref].append(row)
    if len(search_ids) != len(set(search_ids)):
        raise ValueError("endpoint search IDs must be unique")

    matrix = []
    for candidate in candidate_runs:
        run_ref = str(candidate["network_run_ref"])
        rows = searches_by_run[run_ref]
        by_endpoint = {index: [row for row in rows
                               if row.get("canonical_endpoint_index") == index]
                       for index in (0, 1)}
        expected_occurrences = int(candidate.get("duplicate_occurrence_count", 0))
        if expected_occurrences < 1 or any(len(by_endpoint[index]) != expected_occurrences
                                           for index in (0, 1)):
            raise ValueError(f"{run_ref}: endpoint searches do not cover every duplicate occurrence")
        endpoints = {str(index): _endpoint_outcome(by_endpoint[index]) for index in (0, 1)}
        terminal_closed = all(endpoints[str(index)]["state"] == "closed" for index in (0, 1))
        topology_closed = terminal_closed and candidate.get("complete_projected_topology") is True
        status = "eligible_for_m5c_physical_replay" if topology_closed else "endpoint_identity_unresolved"
        reasons = sorted({reason for outcome in endpoints.values() for reason in outcome["reasons"]})
        if terminal_closed and not topology_closed:
            reasons.append("complete_projected_topology_not_certified")
        matrix.append({
            "record_type": "mep_endpoint_coverage_row",
            "id": stable_id("mep_endpoint_coverage", run_ref,
                            canonical_sha256([row["source_inventory_sha256"] for row in rows])),
            "original_geometry_priority_rank": candidate.get("priority_rank"),
            "network_run_ref": run_ref,
            "segment_refs": deepcopy(candidate.get("segment_refs", [])),
            "source_page_refs": deepcopy(candidate.get("source_page_refs", [])),
            "duplicate_occurrence_count": expected_occurrences,
            "canonical_endpoint_outcomes": endpoints,
            "terminal_coverage_closed": terminal_closed,
            "complete_projected_topology": bool(candidate.get("complete_projected_topology")),
            "status": status,
            "reasons": reasons,
            "geometry_channels": {
                "bounded_local_3d_available": candidate.get("bounded_local_3d_available") is True,
                "physical_centreline_promotion_eligible": topology_closed,
                "measured_centreline_length_eligible": topology_closed,
            },
            "commercial_channels": {
                "connection_standard": deepcopy(connection_standard_search.get("resolved_standard")),
                "material": None,
                "specification": None,
                "sku": None,
                "marketplace_assembly_eligible": topology_closed and bool(
                    connection_standard_search.get("resolved_standard")
                ),
            },
            "quantity_eligible": False,
        })

    matrix.sort(key=lambda row: (
        0 if row["status"] == "eligible_for_m5c_physical_replay" else 1,
        -sum(row["canonical_endpoint_outcomes"][str(index)]["state"] == "closed"
             for index in (0, 1)),
        int(row.get("original_geometry_priority_rank") or 10**9),
        row["network_run_ref"],
    ))
    for rank, row in enumerate(matrix, start=1):
        row["endpoint_evidence_rank"] = rank
    eligible = [row for row in matrix if row["status"] == "eligible_for_m5c_physical_replay"]
    selected = eligible[0]["network_run_ref"] if eligible else None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(document)),
        "input_payload_sha256": {
            "candidate_runs": canonical_sha256(candidate_runs),
            "endpoint_searches": canonical_sha256(endpoint_searches),
            "connection_standard_search": canonical_sha256(connection_standard_search),
        },
        "endpoint_searches": deepcopy(list(endpoint_searches)),
        "connection_standard_search": deepcopy(dict(connection_standard_search)),
        "matrix": matrix,
        "selected_physical_replay_run_ref": selected,
        "physical_replay": {
            "state": "ready" if selected else "not_attempted",
            "reason": None if selected else "no_candidate_has_two_explicit_uniquely_bound_terminals_and_complete_topology",
        },
        "summary": {
            "geometry_ready_candidate_count": len(matrix),
            "endpoint_occurrence_search_count": len(endpoint_searches),
            "terminal_coverage_closed_count": sum(row["terminal_coverage_closed"] for row in matrix),
            "physical_replay_eligible_count": len(eligible),
            "endpoint_identity_unresolved_count": sum(
                row["status"] == "endpoint_identity_unresolved" for row in matrix
            ),
        },
        "authority": {
            "endpoint_search_grants_no_identity": True,
            "outlined_closure_is_not_a_physical_cap": True,
            "connection_standard_independent_from_geometry": True,
            "missing_connection_standard_does_not_erase_closed_geometry": True,
            "quantity_eligible": False,
        },
    }
    validate_endpoint_coverage_matrix(payload)
    return payload


def validate_endpoint_coverage_matrix(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("layer") != LAYER:
        raise ValueError("endpoint coverage schema or layer mismatch")
    rows = list(payload.get("matrix", []))
    if payload.get("summary", {}).get("geometry_ready_candidate_count") != len(rows):
        raise ValueError("endpoint coverage summary count mismatch")
    selected = payload.get("selected_physical_replay_run_ref")
    eligible = [row for row in rows if row.get("status") == "eligible_for_m5c_physical_replay"]
    if selected != (eligible[0]["network_run_ref"] if eligible else None):
        raise ValueError("physical replay selection differs from endpoint evidence ranking")
    for row in rows:
        outcomes = row.get("canonical_endpoint_outcomes", {})
        if set(outcomes) != {"0", "1"}:
            raise ValueError("each run requires exactly two canonical endpoint outcomes")
        if row.get("terminal_coverage_closed") is not all(
                outcomes[str(index)].get("state") == "closed" for index in (0, 1)):
            raise ValueError("terminal coverage state differs from endpoint outcomes")
        commercial = row.get("commercial_channels", {})
        if commercial.get("connection_standard") is None and commercial.get("marketplace_assembly_eligible"):
            raise ValueError("marketplace eligibility requires an independently resolved standard")
