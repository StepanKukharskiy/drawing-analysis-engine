"""Compile MEP physical-run readiness and automatic evidence from M4/M5/M5C.

Readiness is diagnostic only.  It distinguishes closed, ambiguous,
not-searched, contradicted, and apparently-absent fields without granting
physical or quantity authority.  A physical-run evidence record is emitted
only when the supplied frozen layers themselves deterministically close every
physical gate; XYZ geometry is copied from accepted M5A records, never from a
manual replacement polyline.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_marketplace_assembly import projected_length_observations_from_m5c
from src.drawing_engine.project.takeoff_intelligence import canonical_sha256, stable_id


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_physical_run_readiness"
STATES = {"closed", "ambiguous", "not_searched", "contradicted", "apparently_absent"}
FIELDS = (
    "complete_projected_topology",
    "system_service",
    "nominal_size",
    "physical_size",
    "terminal_identities",
    "fitting_equipment_ports",
    "elevation_vertical_spans",
    "three_dimensional_segment_coverage",
    "material_specification",
    "connection_standard",
)
GEOMETRY_REQUIRED_FIELDS = {
    "complete_projected_topology", "system_service", "nominal_size",
    "physical_size", "terminal_identities", "fitting_equipment_ports",
    "elevation_vertical_spans", "three_dimensional_segment_coverage",
}
PHYSICAL_REQUIRED_FIELDS = GEOMETRY_REQUIRED_FIELDS
MARKETPLACE_REQUIRED_FIELDS = {"material_specification", "connection_standard"}
WEIGHTS = {
    "complete_projected_topology": 5,
    "system_service": 3,
    "nominal_size": 3,
    "physical_size": 3,
    "terminal_identities": 5,
    "fitting_equipment_ports": 5,
    "elevation_vertical_spans": 4,
    "three_dimensional_segment_coverage": 5,
    "material_specification": 1,
    "connection_standard": 2,
}
TERMINAL_RELATION_TYPES = {
    "equipment_endpoint": "equipment_port",
    "riser_drop": "riser_drop",
    "continuation_endpoint": "cross_sheet_continuation",
    "physical_terminal": "physical_terminal",
    "package_boundary": "package_boundary",
    "detail_section_interface": "detail_section_interface",
}


def _outcome(
    state: str, *, reasons: Iterable[str], evidence_refs: Iterable[object] = (),
    values: Sequence[Mapping[str, Any]] = (), search_basis: str,
) -> dict[str, Any]:
    if state not in STATES:
        raise ValueError(f"unsupported readiness state: {state}")
    return {
        "state": state,
        "reasons": sorted(set(map(str, reasons))),
        "evidence_refs": sorted({str(ref) for ref in evidence_refs if ref}),
        "values": [deepcopy(dict(value)) for value in values],
        "search_basis": search_basis,
        "authority": {"physical_run_established": False, "quantity_eligible": False},
    }


def _semantic_outcome(
    segments: Sequence[Mapping[str, Any]], field: str,
) -> dict[str, Any]:
    overlays = [segment.get("semantic_overlay", {}).get(field, {}) for segment in segments]
    evidence = {
        ref for overlay in overlays
        for ref in (*overlay.get("accepted_relation_refs", []),
                    *overlay.get("unresolved_relation_refs", []),
                    *overlay.get("conflicting_relation_refs", []))
    }
    if any(overlay.get("state") == "conflicted" or overlay.get("conflicting_relation_refs")
           for overlay in overlays):
        return _outcome("contradicted", reasons=[f"conflicting_{field}_bindings"],
                        evidence_refs=evidence, search_basis="m5c_semantic_overlay")
    accepted = [overlay for overlay in overlays if overlay.get("state") == "accepted"]
    values = [value for overlay in accepted for value in overlay.get("observed_values", [])]
    distinct = {canonical_sha256({key: value for key, value in row.items() if key != "raw_text"})
                for row in values}
    if len(distinct) > 1:
        return _outcome("contradicted", reasons=[f"incompatible_{field}_values_across_run"],
                        evidence_refs=evidence, values=values,
                        search_basis="m5c_semantic_overlay")
    if len(accepted) == len(segments) and len(distinct) == 1:
        unique = [next(row for row in values if canonical_sha256(
            {key: value for key, value in row.items() if key != "raw_text"}) == next(iter(distinct)))]
        return _outcome("closed", reasons=[], evidence_refs=evidence, values=unique,
                        search_basis="m5c_semantic_overlay")
    if values or any(overlay.get("unresolved_relation_refs") for overlay in overlays):
        return _outcome("ambiguous", reasons=[f"{field}_does_not_cover_every_run_segment"],
                        evidence_refs=evidence, values=values,
                        search_basis="m5c_semantic_overlay")
    return _outcome("not_searched", reasons=[f"{field}_binding_not_available_for_run"],
                    evidence_refs=evidence, search_basis="m5c_semantic_overlay")


def _m5a_by_network_segment(
    segments: Sequence[Mapping[str, Any]], bounded_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    output = {}
    for segment in segments:
        occurrences = set(map(str, segment.get("source_occurrence_refs", [])))
        targets = set(map(str, segment.get("route_target_refs", [])))
        matches = [row for row in bounded_rows
                   if row.get("state") == "accepted"
                   and row.get("authority", {}).get(
                       "bounded_local_segment_identity_established") is True
                   and (
                       occurrences.intersection(map(str, row.get("source_page_occurrence_refs", [])))
                       or targets.intersection(map(str, row.get("source_route_composite_refs", [])))
                   )]
        output[str(segment.get("id"))] = matches
    return output


def _physical_size_outcome(
    segments: Sequence[Mapping[str, Any]], m5a_by_segment: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = [row for segment in segments for row in m5a_by_segment.get(str(segment.get("id")), [])]
    evidence = {str(row.get("id")) for row in rows}
    if any(len(m5a_by_segment.get(str(segment.get("id")), [])) > 1 for segment in segments):
        return _outcome("ambiguous", reasons=["multiple_m5a_segments_match_one_network_segment"],
                        evidence_refs=evidence, search_basis="m5a_bounded_local_geometry")
    if len(rows) != len(evidence):
        return _outcome("ambiguous", reasons=["one_m5a_segment_matches_multiple_network_segments"],
                        evidence_refs=evidence, search_basis="m5a_bounded_local_geometry")
    values = [row.get("physical_envelope_dimension", {}) for row in rows]
    dimensions = {canonical_sha256({
        "kind": value.get("kind"), "shape": value.get("shape"),
        "representative_outer_width_m": value.get("representative_outer_width_m"),
    }) for value in values}
    if len(rows) == len(segments) and rows and len(dimensions) == 1:
        return _outcome("closed", reasons=[], evidence_refs=evidence, values=[values[0]],
                        search_basis="m5a_bounded_local_geometry")
    if rows:
        return _outcome("ambiguous", reasons=["physical_size_does_not_cover_every_run_segment"],
                        evidence_refs=evidence, values=values,
                        search_basis="m5a_bounded_local_geometry")
    return _outcome("not_searched", reasons=["no_m5a_physical_envelope_for_run"],
                    search_basis="m5a_bounded_local_geometry")


def _three_dimensional_outcomes(
    segments: Sequence[Mapping[str, Any]], m5a_by_segment: Mapping[str, Sequence[Mapping[str, Any]]],
    partial_rows: Sequence[Mapping[str, Any]], unresolved_spans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [row for segment in segments for row in m5a_by_segment.get(str(segment.get("id")), [])]
    evidence = {str(row.get("id")) for row in matches}
    complete = bool(segments) and all(
        len(m5a_by_segment.get(str(segment.get("id")), [])) == 1 for segment in segments
    )
    if complete and len({str(row.get("id")) for row in matches}) != len(segments):
        complete = False
    run_refs = {
        str(ref)
        for segment in segments
        for ref in (
            segment.get("id"),
            *segment.get("route_target_refs", []),
            *segment.get("source_occurrence_refs", []),
            *segment.get("source_fragment_refs", []),
        )
        if ref
    }
    relevant_spans = [
        row for row in unresolved_spans
        if str(row.get("canonical_projected_segment_ref")) in run_refs
        or run_refs.intersection(map(str, row.get("endpoint_or_vertex_refs", [])))
    ]
    if complete:
        frames = {str(row.get("m1_metric_frame", {}).get("id")) for row in matches}
        if len(frames) != 1:
            geometry = _outcome("contradicted", reasons=["run_segments_use_incompatible_metric_frames"],
                                evidence_refs=evidence, search_basis="m5a_xyz_centrelines")
        else:
            geometry = _outcome("closed", reasons=[], evidence_refs=evidence,
                                values=[{"m5a_segment_ref": row.get("id"),
                                         "network_segment_ref": segment.get("id")}
                                        for segment, row in zip(segments, matches)],
                                search_basis="m5a_xyz_centrelines")
        span_refs = {str(row.get("id")) for row in relevant_spans}
        elevation = _outcome(
            "ambiguous" if relevant_spans else "closed",
            reasons=["unresolved_vertical_spans_remain"] if relevant_spans else [],
            evidence_refs={*evidence, *span_refs},
            values=[row.get("elevation", {}) for row in matches],
            search_basis="m5a_elevation_and_vertical_spans",
        )
        return geometry, elevation
    target_refs = {str(ref) for segment in segments for ref in segment.get("route_target_refs", [])}
    partial = [row for row in partial_rows if str(row.get("route_target_ref")) in target_refs]
    if partial or matches:
        refs = {*evidence, *(str(row.get("id")) for row in partial)}
        geometry = _outcome("ambiguous", reasons=["only_partial_or_incomplete_3d_coverage"],
                            evidence_refs=refs, search_basis="m5a_m5b_geometry")
        elevation = _outcome("ambiguous", reasons=sorted({
            str(row.get("reason") or "elevation_does_not_cover_every_run_segment") for row in partial
        }), evidence_refs=refs, values=[{
            "elevation_reference_basis": row.get("elevation_reference_basis"),
            "elevation_reference_m": row.get("elevation_reference_m"),
            "centreline_elevation_m": row.get("centreline_elevation_m"),
        } for row in partial], search_basis="m5a_m5b_elevation")
        return geometry, elevation
    return (
        _outcome("not_searched", reasons=["no_m5a_or_m5b_3d_geometry_for_run"],
                 search_basis="m5a_m5b_geometry"),
        _outcome("not_searched", reasons=["no_elevation_or_vertical_span_resolution_for_run"],
                 search_basis="m5a_m5b_elevation"),
    )


def _topology_outcome(run: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidence = {*map(str, run.get("scoped_trace_refs", [])),
                *map(str, run.get("endpoint_interface_candidate_refs", [])),
                *(str(segment.get("id")) for segment in segments)}
    if run.get("endpoint_interface_search_complete") is False:
        return _outcome(
            "not_searched",
            reasons={
                "endpoint_interface_search_incomplete",
                *map(str, run.get("unresolved_trace_reasons", [])),
            },
            evidence_refs=evidence,
            search_basis="m5c_scoped_trace_and_endpoint_interface_completion",
        )
    if run.get("complete_trace_established") is True and not run.get("uncovered_segment_refs"):
        return _outcome("closed", reasons=[], evidence_refs=evidence,
                        search_basis="m5c_scoped_trace_completion")
    if run.get("unsearched_junction_refs") or run.get("uncovered_segment_refs"):
        return _outcome("not_searched", reasons=(
            run.get("unresolved_trace_reasons") or ["projected_topology_search_incomplete"]
        ), evidence_refs=evidence, search_basis="m5c_scoped_trace_completion")
    if run.get("scoped_trace_refs"):
        return _outcome("ambiguous", reasons=(
            run.get("unresolved_trace_reasons") or ["projected_topology_ambiguous"]
        ), evidence_refs=evidence, search_basis="m5c_scoped_trace_completion")
    return _outcome("not_searched", reasons=["projected_topology_not_searched"],
                    evidence_refs=evidence, search_basis="m5c_scoped_trace_completion")


def _terminal_and_port_outcomes(
    run: Mapping[str, Any], junctions: Sequence[Mapping[str, Any]],
    attachments: Sequence[Mapping[str, Any]], relations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str | None, dict[str, Any]]:
    run_segments = set(map(str, run.get("segment_refs", [])))
    run_attachments = [row for row in attachments
                       if run_segments.intersection(map(str, row.get("segment_refs", [])))]
    terminals = []
    standards = set()
    evidence = set()
    for attachment in run_attachments:
        relation = relations.get(str(attachment.get("source_relation_ref")), {})
        relation_type = str(attachment.get("relation_type"))
        candidate = relation.get("candidate", {})
        terminal_kind = candidate.get("terminal_kind") or TERMINAL_RELATION_TYPES.get(relation_type)
        endpoint_index = candidate.get("endpoint_index")
        segment_refs = list(map(str, attachment.get("segment_refs", [])))
        if terminal_kind and len(segment_refs) == 1 and endpoint_index in (0, 1):
            terminal = {
                "source_relation_ref": relation.get("id"), "segment_ref": segment_refs[0],
                "endpoint_index": endpoint_index, "terminal_kind": terminal_kind,
                "connection_standard": candidate.get("connection_standard"),
                "evidence_refs": sorted({str(relation.get("id")),
                                         *map(str, relation.get("binding_evidence_refs", [])),
                                         *map(str, relation.get("target_refs", []))}),
            }
            terminals.append(terminal)
            evidence.update(terminal["evidence_refs"])
            if terminal["connection_standard"]:
                standards.add(str(terminal["connection_standard"]))
    keys = {(row["segment_ref"], row["endpoint_index"]) for row in terminals}
    if len(terminals) == 2 and len(keys) == 2:
        terminal_outcome = _outcome("closed", reasons=[], evidence_refs=evidence,
                                    values=terminals, search_basis="m4_terminal_relations")
    elif terminals:
        terminal_outcome = _outcome("ambiguous", reasons=["two_unique_run_terminals_not_closed"],
                                    evidence_refs=evidence, values=terminals,
                                    search_basis="m4_terminal_relations")
    else:
        endpoint_rows = list(run.get("projected_endpoint_occurrences", []))
        candidate_refs = {str(ref) for row in endpoint_rows
                          for ref in row.get("interface_candidate_refs", [])}
        endpoint_evidence = {
            str(ref) for row in endpoint_rows for ref in (
                row.get("composite_ref"), *row.get("complete_candidate_search_refs", []),
                *row.get("interface_candidate_refs", []),
            ) if ref
        }
        if candidate_refs:
            reasons = ["terminal_identity_candidates_not_uniquely_bound"]
            if run.get("endpoint_interface_search_complete") is not True:
                reasons.append("endpoint_interface_search_incomplete")
            terminal_outcome = _outcome(
                "ambiguous", reasons=reasons, evidence_refs=endpoint_evidence,
                values=[{
                    "segment_ref": row.get("segment_ref"),
                    "composite_ref": row.get("composite_ref"),
                    "page_ref": row.get("page_ref"),
                    "composite_endpoint_index": row.get("composite_endpoint_index"),
                    "interface_candidate_refs": deepcopy(row.get("interface_candidate_refs", [])),
                    "search_state": row.get("search_state"),
                } for row in endpoint_rows],
                search_basis="m5c_projected_endpoint_interface_search",
            )
        else:
            terminal_outcome = _outcome("not_searched", reasons=["terminal_identity_relations_not_available"],
                                        evidence_refs={*run.get("unresolved_boundary_refs", []),
                                                       *endpoint_evidence},
                                        search_basis="m4_m5c_terminal_relations")
    internal = [row for row in junctions if set(map(str, row.get("segment_refs", []))).issubset(run_segments)]
    endpoint_candidates = [row for row in junctions if run_segments.intersection(
        map(str, row.get("segment_refs", [])))]
    physical = [row for row in internal if row.get("physical_continuation_established") is True]
    conflicts = [row for row in endpoint_candidates if row.get("semantic_conflicts")]
    unresolved_candidates = [row for row in endpoint_candidates
                             if row.get("physical_continuation_established") is not True]
    if conflicts:
        port_outcome = _outcome("contradicted", reasons=["junction_semantic_conflicts"],
                                evidence_refs=[row.get("id") for row in conflicts],
                                search_basis="m5c_junction_ports")
    elif unresolved_candidates:
        port_outcome = _outcome(
            "ambiguous",
            reasons={"projected_endpoint_interfaces_lack_physical_port_certificate",
                     *(reason for row in unresolved_candidates
                       for reason in row.get("reasons", []))},
            evidence_refs=[row.get("id") for row in unresolved_candidates],
            search_basis="m5c_endpoint_interface_candidates",
        )
    elif len(physical) == len(internal) and terminal_outcome["state"] == "closed":
        port_outcome = _outcome("closed", reasons=[],
                                evidence_refs=[row.get("id") for row in physical] + list(evidence),
                                search_basis="m4_m5c_physical_ports")
    elif internal:
        port_outcome = _outcome("ambiguous", reasons=["projected_junctions_lack_physical_port_certificate"],
                                evidence_refs=[row.get("id") for row in internal],
                                search_basis="m5c_junction_ports")
    else:
        port_outcome = (_outcome("closed", reasons=[], evidence_refs=evidence,
                                 search_basis="single_segment_terminal_ports")
                        if terminal_outcome["state"] == "closed" else
                        _outcome("not_searched", reasons=["fitting_equipment_port_search_not_closed"],
                                 search_basis="m4_m5c_physical_ports"))
    standard = next(iter(standards)) if len(standards) == 1 and all(
        row.get("connection_standard") for row in terminals
    ) else None
    if standard:
        standard_outcome = _outcome("closed", reasons=[], evidence_refs=evidence,
                                    values=[{"connection_standard": standard}],
                                    search_basis="m4_terminal_relations")
    elif len(standards) > 1:
        standard_outcome = _outcome("contradicted", reasons=["incompatible_connection_standards"],
                                    evidence_refs=evidence, search_basis="m4_terminal_relations")
    elif terminals:
        standard_outcome = _outcome("ambiguous", reasons=["connection_standard_missing_on_terminal_relation"],
                                    evidence_refs=evidence, search_basis="m4_terminal_relations")
    else:
        standard_outcome = _outcome("not_searched", reasons=["connection_standard_not_searched"],
                                    search_basis="m4_terminal_relations")
    return terminal_outcome, port_outcome, terminals, standard, standard_outcome


def _material_outcome(
    run_segments: set[str], relations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    rows = [row for row in relations if row.get("state") == "accepted"
            and row.get("relation_type") in {"route_material", "route_specification"}
            and run_segments.intersection(map(str, row.get("target_segment_refs", row.get("target_refs", []))))]
    materials = [row.get("candidate", {}) for row in rows if row.get("relation_type") == "route_material"]
    specifications = [row.get("candidate", {}) for row in rows if row.get("relation_type") == "route_specification"]
    if len(materials) == 1 and len(specifications) == 1:
        value = {"material": materials[0].get("kind") or materials[0].get("material"),
                 "specification": specifications[0].get("reference") or specifications[0].get("specification")}
        return _outcome("closed", reasons=[], evidence_refs=[row.get("id") for row in rows],
                        values=[value], search_basis="m4_material_specification_relations"), value
    if rows:
        return _outcome("ambiguous", reasons=["material_and_specification_not_uniquely_paired"],
                        evidence_refs=[row.get("id") for row in rows],
                        values=[*materials, *specifications],
                        search_basis="m4_material_specification_relations"), None
    return _outcome("not_searched", reasons=["material_specification_outside_current_m4_search"],
                    search_basis="m4_material_specification_relations"), None


def _apply_negative_search_certificate(
    run: Mapping[str, Any], field: str, outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Use a complete source search only as a negative readiness cap.

    The certificate cannot close a field and normally applies only to an
    unsearched outcome.  A connection-standard search may also refine the
    narrow "missing on otherwise accepted terminals" ambiguity to apparently
    absent.  This never creates a joining rule from an empty candidate list.
    """
    certificate = run.get("readiness_source_search", {}).get(field, {})
    evidence_refs = certificate.get("evidence_refs", [])
    negative_eligible = outcome.get("state") == "not_searched" or (
        field == "connection_standard"
        and outcome.get("state") == "ambiguous"
        and outcome.get("reasons") == ["connection_standard_missing_on_terminal_relation"]
    )
    if (not negative_eligible
            or certificate.get("state") != "complete"
            or not evidence_refs):
        return deepcopy(dict(outcome))
    return _outcome(
        "apparently_absent",
        reasons=[str(certificate.get("negative_reason") or f"{field}_apparently_absent_after_complete_search")],
        evidence_refs=evidence_refs,
        search_basis=str(certificate.get("method") or "m5c_complete_source_search_certificate"),
    )


def _build_physical_evidence(
    *, run: Mapping[str, Any], segments: Sequence[Mapping[str, Any]],
    m5a_by_segment: Mapping[str, Sequence[Mapping[str, Any]]],
    outcomes: Mapping[str, Mapping[str, Any]], terminals: Sequence[Mapping[str, Any]],
    connection_standard: str | None, material: Mapping[str, Any] | None,
    network_hash: str,
) -> dict[str, Any] | None:
    if any(outcomes[field]["state"] != "closed" for field in GEOMETRY_REQUIRED_FIELDS):
        return None
    # Current automatic positive is deliberately narrow and generic: a bounded
    # straight run. Multi-segment port geometry must first be physically typed;
    # projected M5C joins are not silently upgraded.
    if len(segments) != 1 or run.get("junction_refs"):
        return None
    segment = segments[0]
    m5a = m5a_by_segment[str(segment["id"])][0]
    port_refs = [stable_id("mep_physical_run_port", run["id"], segment["id"], index)
                 for index in (0, 1)]
    terminal_rows = []
    for terminal in sorted(terminals, key=lambda row: int(row["endpoint_index"])):
        terminal_rows.append({
            "id": stable_id("mep_physical_terminal", run["id"], terminal["endpoint_index"]),
            "port_ref": port_refs[int(terminal["endpoint_index"])],
            "terminal_kind": terminal["terminal_kind"],
            "connection_standard": connection_standard,
            "evidence_refs": deepcopy(terminal["evidence_refs"]),
        })
    system = deepcopy(outcomes["system_service"]["values"][0])
    system["service"] = system.get("kind")
    size = deepcopy(outcomes["nominal_size"]["values"][0])
    physical = deepcopy(outcomes["physical_size"]["values"][0])
    evidence_refs = sorted({
        *[ref for field in GEOMETRY_REQUIRED_FIELDS for ref in outcomes[field]["evidence_refs"]],
        str(m5a.get("id")), str(segment.get("id")),
    })
    return {
        "id": stable_id("mep_physical_run_evidence", network_hash, run["id"]),
        "state": "observed", "network_run_ref": run["id"],
        "network_payload_sha256": network_hash,
        "accounted_network_segment_refs": [segment["id"]],
        "accounted_projected_occurrence_refs": deepcopy(segment.get("source_occurrence_refs", [])),
        "system": system,
        "source": {"page_refs": deepcopy(segment.get("page_refs", [])),
                   "detail_refs": [], "section_refs": []},
        "material": deepcopy(dict(material or {"material": None, "specification": None})),
        "cross_section": {
            "shape": physical.get("shape"), "nominal_size": size,
            "physical_dimensions": physical,
        },
        "terminals": terminal_rows,
        "centreline_segments": [{
            "id": stable_id("mep_physical_centreline", run["id"], segment["id"]),
            "network_segment_ref": segment["id"], "port_refs": port_refs,
            "points_xyz_m": deepcopy(m5a["centreline_points_xyz_m"]),
            "evidence_refs": [m5a["id"], segment["id"]],
        }],
        "components": [], "connections": [],
        "coverage": {
            "all_projected_occurrences_accounted_once": True,
            "reprojection_passed": True,
            "duplicate_occurrences_eliminated": True,
            "vertical_spans_resolved": True,
        },
        "evidence_refs": evidence_refs,
    }


def _projected_observations(
    run_ref: str, segments: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]], bounded_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    target_refs = {str(ref) for segment in segments for ref in segment.get("route_target_refs", [])}
    output = [{
        "source_ref": row.get("id"),
        "route_target_ref": row.get("route_target_ref"),
        "projected_length_m": row.get("projected_2d_length_m"),
        "measurement_scope_kind": "partial_2_5d_segment",
        "complete_run_coverage": False,
        "quantity_eligible": False,
    } for row in partial_rows if str(row.get("route_target_ref")) in target_refs]
    output.extend({
        "source_ref": row.get("id"),
        "route_target_ref": None,
        "projected_length_m": row.get("value_m"),
        "measurement_scope_kind": row.get("measurement_scope_kind"),
        "measurement_scope_ref": row.get("measurement_scope_ref"),
        "complete_run_coverage": row.get("complete_run_coverage") is True,
        "quantity_eligible": False,
    } for row in bounded_rows if str(row.get("network_run_ref")) == run_ref)
    return sorted(output, key=lambda row: (
        0 if row.get("measurement_scope_kind") == "bounded_complete_projected_subtrace" else 1,
        str(row.get("route_target_ref")), str(row["source_ref"]),
    ))


def build_mep_physical_run_readiness(
    *, attribute_bindings: Mapping[str, Any], bounded_local_3d: Mapping[str, Any],
    cross_sheet_runs: Mapping[str, Any], network_hierarchy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile readiness for every M5C run and emit only automatic positives."""
    layers = {attribute_bindings.get("layer"), bounded_local_3d.get("layer"),
              cross_sheet_runs.get("layer"), network_hierarchy.get("layer")}
    expected_layers = {"mep_page_local_attribute_bindings", "mep_bounded_local_3d_segments",
                       "mep_cross_sheet_run_hypotheses", "mep_projected_network_hierarchy"}
    if layers != expected_layers:
        raise ValueError("M4, M5A, M5B, and M5C inputs are required")
    document_keys = {str(row.get("document", {}).get("document_key")) for row in (
        attribute_bindings, bounded_local_3d, cross_sheet_runs, network_hierarchy
    )}
    if len(document_keys) != 1:
        raise ValueError("M4/M5A/M5B/M5C document keys must match")
    hashes = {"m4": canonical_sha256(attribute_bindings),
              "m5a": canonical_sha256(bounded_local_3d),
              "m5b": canonical_sha256(cross_sheet_runs),
              "m5c": canonical_sha256(network_hierarchy)}
    if network_hierarchy.get("input_payload_sha256", {}).get("m4") != hashes["m4"]:
        raise ValueError("M5C does not bind the supplied M4 payload")
    if network_hierarchy.get("input_payload_sha256", {}).get("m5b") != hashes["m5b"]:
        raise ValueError("M5C does not bind the supplied M5B payload")
    if bounded_local_3d.get("m4_contract_ref", {}).get("payload_sha256") != hashes["m4"]:
        raise ValueError("M5A does not bind the supplied M4 payload")
    if bounded_local_3d.get("m5_contract_ref", {}).get("payload_sha256") != hashes["m5b"]:
        raise ValueError("M5A does not bind the supplied M5B payload")

    segments_by_id = {str(row["id"]): row for row in network_hierarchy.get("segments", [])}
    junctions_by_id = {str(row["id"]): row for row in network_hierarchy.get("junctions", [])}
    relations = {str(row["id"]): row for row in attribute_bindings.get("relations", [])}
    bounded_rows = list(bounded_local_3d.get("bounded_local_3d_segments", []))
    partial_rows = list(cross_sheet_runs.get("partial_2_5d_centreline_segments", []))
    bounded_projected_rows = projected_length_observations_from_m5c(network_hierarchy)
    unresolved_spans = list(cross_sheet_runs.get("unresolved_vertical_spans", []))
    records = []
    automatic_evidence = []
    for run in sorted(network_hierarchy.get("runs", []), key=lambda row: str(row.get("id"))):
        segments = [segments_by_id[str(ref)] for ref in run.get("segment_refs", [])
                    if str(ref) in segments_by_id]
        m5a_map = _m5a_by_network_segment(segments, bounded_rows)
        geometry, elevation = _three_dimensional_outcomes(
            segments, m5a_map, partial_rows, unresolved_spans
        )
        interface_refs = {*map(str, run.get("junction_refs", [])),
                          *map(str, run.get("endpoint_interface_candidate_refs", []))}
        terminal, ports, terminal_values, standard, standard_outcome = _terminal_and_port_outcomes(
            run, [junctions_by_id[ref] for ref in sorted(interface_refs)
                  if str(ref) in junctions_by_id],
            network_hierarchy.get("attachments", []), relations,
        )
        material_outcome, material = _material_outcome(
            set(map(str, run.get("segment_refs", []))), attribute_bindings.get("relations", [])
        )
        outcomes = {
            "complete_projected_topology": _topology_outcome(run, segments),
            "system_service": _semantic_outcome(segments, "system"),
            "nominal_size": _semantic_outcome(segments, "size"),
            "physical_size": _physical_size_outcome(segments, m5a_map),
            "terminal_identities": terminal,
            "fitting_equipment_ports": ports,
            "elevation_vertical_spans": elevation,
            "three_dimensional_segment_coverage": geometry,
            "material_specification": material_outcome,
            "connection_standard": standard_outcome,
        }
        outcomes = {
            field: _apply_negative_search_certificate(run, field, outcome)
            for field, outcome in outcomes.items()
        }
        state_score = {"closed": 2, "ambiguous": 1, "not_searched": 0,
                       "apparently_absent": -1, "contradicted": -2}
        score = sum(WEIGHTS[field] * state_score[outcomes[field]["state"]] for field in FIELDS)
        evidence = _build_physical_evidence(
            run=run, segments=segments, m5a_by_segment=m5a_map, outcomes=outcomes,
            terminals=terminal_values, connection_standard=standard, material=material,
            network_hash=hashes["m5c"],
        )
        if evidence:
            automatic_evidence.append(evidence)
        blockers = [field for field in GEOMETRY_REQUIRED_FIELDS if outcomes[field]["state"] != "closed"]
        marketplace_blockers = [field for field in MARKETPLACE_REQUIRED_FIELDS
                                if outcomes[field]["state"] != "closed"]
        reasons = {reason for field in blockers for reason in outcomes[field]["reasons"]}
        if not blockers and evidence is None:
            # Field closure can describe a future physically typed multi-part
            # run.  The first producer intentionally supports only one bounded
            # straight segment; do not publish a deceptively empty abstention.
            reasons.add("automatic_multi_segment_physical_assembly_not_supported")
        records.append({
            "record_type": "mep_physical_run_readiness", "record_version": SCHEMA_VERSION,
            "id": stable_id("mep_physical_run_readiness", hashes["m5c"], run["id"]),
            "network_run_ref": run["id"],
            "source_page_refs": sorted({str(page) for segment in segments
                                        for page in segment.get("page_refs", [])}),
            "segment_refs": deepcopy(run.get("segment_refs", [])),
            "projected_length_observations": _projected_observations(
                str(run.get("id")), segments, partial_rows, bounded_projected_rows
            ),
            "field_outcomes": outcomes,
            "readiness_score": score,
            "physical_evidence_ready": evidence is not None,
            "marketplace_blocking_fields": sorted(marketplace_blockers),
            "automatic_physical_run_evidence_ref": evidence.get("id") if evidence else None,
            "abstention": None if evidence else {
                "state": "abstained", "blocking_fields": sorted(blockers),
                "reasons": sorted(reasons),
            },
            "authority": {"physical_run_established": False,
                          "installed_length_emitted": False, "quantity_eligible": False},
        })
    records.sort(key=lambda row: (-row["readiness_score"], row["network_run_ref"]))
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": LAYER,
        "document": deepcopy(network_hierarchy.get("document", {})),
        "input_payload_sha256": hashes,
        "field_contract": {"states": sorted(STATES), "fields": list(FIELDS),
                           "diagnostic_only": True, "manual_geometry_prohibited": True},
        "runs": records,
        "automatic_physical_run_evidence": automatic_evidence,
        "summary": {
            "run_count": len(records),
            "automatic_physical_run_evidence_count": len(automatic_evidence),
            "physical_evidence_ready_count": sum(row["physical_evidence_ready"] for row in records),
            "runs_with_projected_length_observation_count": sum(
                bool(row["projected_length_observations"]) for row in records
            ),
            "runs_with_bounded_projected_length_observation_count": sum(any(
                observation.get("measurement_scope_kind") == "bounded_complete_projected_subtrace"
                for observation in row["projected_length_observations"]
            ) for row in records),
            "runs_with_unique_bounded_projected_length_observation_count": sum(sum(
                observation.get("measurement_scope_kind") == "bounded_complete_projected_subtrace"
                for observation in row["projected_length_observations"]
            ) == 1 for row in records),
            "field_state_counts": {field: dict(Counter(
                row["field_outcomes"][field]["state"] for row in records
            )) for field in FIELDS},
        },
        "authority": {"physical_run_established": False,
                      "installed_length_emitted": False, "quantity_eligible": False},
    }
    errors = validate_mep_physical_run_readiness(payload)
    if errors:
        raise ValueError("invalid physical-run readiness output:\n" + "\n".join(errors))
    return payload


def validate_mep_physical_run_readiness(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("layer") != LAYER:
        errors.append("readiness schema or layer mismatch")
    records = list(payload.get("runs", []))
    ids = [str(row.get("id")) for row in records]
    if len(ids) != len(set(ids)):
        errors.append("readiness IDs must be unique")
    evidence_rows = list(payload.get("automatic_physical_run_evidence", []))
    evidence_by_id = {str(row.get("id")): row for row in evidence_rows}
    evidence_ids = set(evidence_by_id)
    if len(evidence_ids) != len(evidence_rows):
        errors.append("automatic physical-run evidence IDs must be unique")
    for row in records:
        identifier = str(row.get("id"))
        outcomes = row.get("field_outcomes", {})
        if set(outcomes) != set(FIELDS):
            errors.append(f"{identifier}: readiness fields mismatch")
            continue
        for field, outcome in outcomes.items():
            if outcome.get("state") not in STATES:
                errors.append(f"{identifier}: {field} state is invalid")
            if outcome.get("authority") != {"physical_run_established": False,
                                             "quantity_eligible": False}:
                errors.append(f"{identifier}: {field} gained authority")
        ready = all(outcomes[field]["state"] == "closed" for field in PHYSICAL_REQUIRED_FIELDS)
        evidence_ref = row.get("automatic_physical_run_evidence_ref")
        if row.get("physical_evidence_ready") is not (ready and evidence_ref in evidence_ids):
            errors.append(f"{identifier}: physical evidence readiness mismatch")
        evidence = evidence_by_id.get(str(evidence_ref))
        if evidence is not None:
            expected = list(map(str, row.get("segment_refs", [])))
            actual = list(map(str, evidence.get("accounted_network_segment_refs", [])))
            centreline = list(map(str, (
                segment.get("network_segment_ref")
                for segment in evidence.get("centreline_segments", [])
            )))
            if evidence.get("network_run_ref") != row.get("network_run_ref"):
                errors.append(f"{identifier}: evidence run reference mismatch")
            if evidence.get("network_payload_sha256") != payload.get(
                    "input_payload_sha256", {}).get("m5c"):
                errors.append(f"{identifier}: evidence network hash mismatch")
            if (Counter(expected) != Counter(actual) or len(actual) != len(set(actual))
                    or Counter(expected) != Counter(centreline)
                    or len(centreline) != len(set(centreline))):
                errors.append(f"{identifier}: evidence segment coverage mismatch")
        if row.get("authority") != {"physical_run_established": False,
                                    "installed_length_emitted": False,
                                    "quantity_eligible": False}:
            errors.append(f"{identifier}: readiness record gained downstream authority")
    if payload.get("authority") != {"physical_run_established": False,
                                    "installed_length_emitted": False,
                                    "quantity_eligible": False}:
        errors.append("readiness payload gained downstream authority")
    summary = payload.get("summary", {})
    if summary.get("run_count") != len(records):
        errors.append("summary.run_count mismatch")
    if summary.get("automatic_physical_run_evidence_count") != len(evidence_ids):
        errors.append("summary automatic evidence count mismatch")
    if summary.get("physical_evidence_ready_count") != sum(
            bool(row.get("physical_evidence_ready")) for row in records):
        errors.append("summary physical-evidence-ready count mismatch")
    expected_state_counts = {field: dict(Counter(
        row.get("field_outcomes", {}).get(field, {}).get("state") for row in records
    )) for field in FIELDS}
    if summary.get("field_state_counts") != expected_state_counts:
        errors.append("summary field-state counts mismatch")
    return errors
