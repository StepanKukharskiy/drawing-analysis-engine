"""Recover partial terminal-landing and support context from a section view."""

from __future__ import annotations

from hashlib import sha256
from math import hypot
from typing import Any, Mapping

from src.drawing_engine.disciplines.concrete.partial_profile_sweep_materializer import materialize_partial_profile_sweeps


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _abstain(page_number: int, reason_code: str, evidence_refs: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "terminal_landing_support_reconstruction",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "terminal_landing_partial": None,
        "terminal_support_candidate": None,
        "evidence_refs": sorted(set(evidence_refs or [])),
        "contract": {
            "partial_section_extent_is_physical_length": False,
            "analysis_cap_is_physical_boundary": False,
            "support_identity_resolved": False,
            "included_in_primary_object": False,
            "quantity_eligible": False,
        },
    }


def _distance(left: list[float], right: list[float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _candidate_size_vocabulary(clear_span_prism_reconstruction: Mapping[str, Any]) -> tuple[list[float], list[str]]:
    callouts = clear_span_prism_reconstruction.get("member_size_callouts", []) or []
    values = sorted(
        {
            float(value)
            for item in callouts
            for value in (item.get("width_mm"), item.get("depth_mm"))
            if value is not None and float(value) > 0
        }
    )
    return values, [str(item.get("id")) for item in callouts if item.get("id")]


def _snap_to_vocabulary(value_mm: float, vocabulary_mm: list[float]) -> tuple[float, float | None]:
    if not vocabulary_mm:
        return value_mm, None
    target = min(vocabulary_mm, key=lambda candidate: abs(candidate - value_mm))
    tolerance = max(5.0, 0.04 * target)
    return (target, target) if abs(target - value_mm) <= tolerance else (value_mm, None)


def _terminal_profile_observation(
    open_structural_boundary_assembly: Mapping[str, Any],
    section_scope_ref: str,
    thickness_mm: float,
) -> dict[str, Any] | None:
    scope = next(
        (
            item
            for item in open_structural_boundary_assembly.get("scope_results", []) or []
            if str(item.get("scope_ref")) == section_scope_ref
        ),
        None,
    )
    if not scope:
        return None
    scale = float(scope.get("metric_scale_points_per_mm") or 0.0)
    if scale <= 0:
        return None
    chains = list(scope.get("branch_free_chains", []) or [])
    stepped = [
        item
        for item in chains
        if item.get("role") == "stepped_upper_surface" and item.get("boundary_eligible", True)
    ]
    waists = [
        item
        for item in chains
        if item.get("role") == "waist_underside" and item.get("boundary_eligible", True)
    ]
    if not stepped or not waists:
        return None
    top_chain = min(
        stepped,
        key=lambda item: min(float(point[1]) for point in item.get("ordered_points_display", []) or [[0.0, 1e30]]),
    )
    top_points = [list(map(float, point)) for point in top_chain.get("ordered_points_display", []) or []]
    if len(top_points) < 2:
        return None
    top_start = min(top_points, key=lambda point: point[1])
    waist_endpoints = [
        (item, list(map(float, point)))
        for item in waists
        for point in (
            (item.get("ordered_points_display") or [])[0],
            (item.get("ordered_points_display") or [])[-1],
        )
    ]
    waist_chain, waist_endpoint = min(waist_endpoints, key=lambda row: _distance(row[1], top_start))

    candidates = []
    endpoint_tolerance = max(2.0, 35.0 * scale)
    for chain in chains:
        if chain.get("role") != "support_or_landing_boundary" or not chain.get("boundary_eligible", True):
            continue
        points = [list(map(float, point)) for point in chain.get("ordered_points_display", []) or []]
        if len(points) != 5:
            continue
        if _distance(points[0], waist_endpoint) < _distance(points[-1], waist_endpoint):
            points.reverse()
        if _distance(points[-1], waist_endpoint) > endpoint_tolerance:
            continue
        p0, p1, p2, p3, p4 = points
        horizontal_1 = abs(p0[1] - p1[1]) <= max(1e-6, 2.0 * scale)
        vertical_1 = abs(p1[0] - p2[0]) <= max(1e-6, 2.0 * scale)
        horizontal_2 = abs(p2[1] - p3[1]) <= max(1e-6, 2.0 * scale)
        vertical_2 = abs(p3[0] - p4[0]) <= max(1e-6, 2.0 * scale)
        native_thickness_mm = abs(p0[1] - top_start[1]) / scale
        if (
            not all((horizontal_1, vertical_1, horizontal_2, vertical_2))
            or abs(native_thickness_mm - thickness_mm) > max(5.0, 0.05 * thickness_mm)
        ):
            continue
        candidates.append(
            {
                "top_chain": top_chain,
                "waist_chain": waist_chain,
                "support_chain": chain,
                "top_start": top_start,
                "points": points,
                "scale_points_per_mm": scale,
                "native_thickness_mm": native_thickness_mm,
                "visible_section_extent_mm": abs(top_start[0] - p0[0]) / scale,
                "support_width_mm": abs(p3[0] - p1[0]) / scale,
                "support_drop_below_slab_mm": abs(p2[1] - p1[1]) / scale,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def reconstruct_terminal_landing_support(
    open_structural_boundary_assembly: Mapping[str, Any],
    landing_component_reconstruction: Mapping[str, Any],
    banded_plan_sweep_evidence: Mapping[str, Any],
    clear_span_prism_reconstruction: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Publish finite preview context without converting an analysis cap to quantity."""

    thickness = landing_component_reconstruction.get("section_thickness_certificate") or {}
    gauge = landing_component_reconstruction.get("termination_convention_certificate") or {}
    thickness_mm = float(thickness.get("thickness_mm") or 0.0)
    section_scope_ref = str((landing_component_reconstruction.get("physical_object_scope") or {}).get("section_title_scope_ref") or "")
    if not section_scope_ref:
        section_scope_ref = next(
            (
                str(ref)
                for ref in landing_component_reconstruction.get("evidence_refs", []) or []
                if str(ref).startswith("title_view_segment.") and str(ref) != str((landing_component_reconstruction.get("plan_footprint_certificate") or {}).get("title_scope_ref"))
            ),
            "",
        )
    if thickness_mm <= 0 or not section_scope_ref or gauge.get("state") != "resolved_relative_unsigned":
        return _abstain(page_number, "terminal_section_metric_frame_unresolved")

    observation = _terminal_profile_observation(
        open_structural_boundary_assembly,
        section_scope_ref,
        thickness_mm,
    )
    if observation is None:
        return _abstain(page_number, "unique_terminal_landing_profile_unresolved", [section_scope_ref])

    bands = list((banded_plan_sweep_evidence.get("plan_band_certificate") or {}).get("bands", []) or [])
    widths = {
        round(float(item.get("sweep_width_mm") or 0.0), 6)
        for item in bands
        if float(item.get("sweep_width_mm") or 0.0) > 0
    }
    clear_span = clear_span_prism_reconstruction.get("clear_span_certificate") or {}
    if len(widths) != 1 or clear_span.get("state") != "resolved":
        return _abstain(page_number, "terminal_plan_sweep_or_support_span_unresolved", [section_scope_ref])
    flight_width_mm = next(iter(widths))
    clear_span_mm = float(clear_span.get("clear_span_mm") or 0.0)
    clear_span_ref = str(clear_span.get("clear_span_dimension_ref") or "")
    if min(flight_width_mm, clear_span_mm) <= 0 or not clear_span_ref:
        return _abstain(page_number, "terminal_plan_sweep_or_support_span_unresolved", [section_scope_ref])

    vocabulary, vocabulary_refs = _candidate_size_vocabulary(clear_span_prism_reconstruction)
    support_width_mm, width_match = _snap_to_vocabulary(observation["support_width_mm"], vocabulary)
    support_drop_mm, drop_match = _snap_to_vocabulary(observation["support_drop_below_slab_mm"], vocabulary)
    evidence_refs = sorted(
        {
            section_scope_ref,
            clear_span_ref,
            str(observation["top_chain"].get("id")),
            str(observation["waist_chain"].get("id")),
            str(observation["support_chain"].get("id")),
            *map(str, observation["top_chain"].get("split_edge_refs", []) or []),
            *map(str, observation["waist_chain"].get("split_edge_refs", []) or []),
            *map(str, observation["support_chain"].get("split_edge_refs", []) or []),
            *map(str, thickness.get("evidence_refs", []) or []),
            *map(str, gauge.get("evidence_refs", []) or []),
            *(str(item.get("id")) for item in bands if item.get("id")),
            *vocabulary_refs,
        }
        - {""}
    )
    landing_ref = _stable_id(page_number, "terminal_landing_partial", *evidence_refs)
    support_ref = _stable_id(page_number, "terminal_support_candidate", *evidence_refs)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "terminal_landing_support_reconstruction",
        "page": page_number,
        "status": "resolved_partial_context",
        "reason_code": "terminal_extent_analysis_capped_and_support_identity_unresolved",
        "terminal_landing_partial": {
            "id": landing_ref,
            "record_type": "terminal_landing_partial",
            "state": "resolved_partial_extent",
            "classification": "upper_landing_or_floor_slab",
            "thickness_mm": thickness_mm,
            "native_measured_thickness_mm": round(observation["native_thickness_mm"], 3),
            "sweep_width_mm": flight_width_mm,
            "visible_section_extent_mm": round(observation["visible_section_extent_mm"], 3),
            "longitudinal_extent_complete": False,
            "external_cap_role": "analysis_cap",
            "plan_projection_state": "not_depicted_in_schematic_stair_plan",
            "plan_absence_is_hard_contradiction": False,
            "reprojection_eligible": False,
            "physical_object_scope_state": "stair_or_floor_scope_unresolved",
            "quantity_eligible": False,
            "evidence_refs": evidence_refs,
        },
        "terminal_support_candidate": {
            "id": support_ref,
            "record_type": "terminal_support_candidate",
            "state": "resolved_geometry_unresolved_identity",
            "classification": "beam_candidate",
            "section_width_mm": support_width_mm,
            "native_measured_section_width_mm": round(observation["support_width_mm"], 3),
            "drop_below_slab_mm": support_drop_mm,
            "native_measured_drop_below_slab_mm": round(observation["support_drop_below_slab_mm"], 3),
            "clear_span_mm": clear_span_mm,
            "size_vocabulary_matches_mm": [value for value in (width_match, drop_match) if value is not None],
            "member_identity_resolved": False,
            "support_overlap_boundary_resolved": False,
            "plan_projection_state": "support_span_only",
            "reprojection_eligible": False,
            "volume_candidate_m3": support_width_mm * support_drop_mm * clear_span_mm / 1_000_000_000.0,
            "quantity_eligible": False,
            "evidence_refs": evidence_refs,
        },
        "evidence_refs": evidence_refs,
        "contract": {
            "partial_section_extent_is_physical_length": False,
            "analysis_cap_is_physical_boundary": False,
            "support_identity_resolved": False,
            "included_in_primary_object": False,
            "quantity_eligible": False,
        },
    }


def _partial_sweep_inputs(
    reconstruction: Mapping[str, Any],
    primary_mesh: Mapping[str, Any],
    *,
    page_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Translate stair evidence into drawing-neutral partial-sweep records."""

    vertices = [list(map(float, point)) for point in primary_mesh.get("vertices_xyz_mm", []) or []]
    if not vertices:
        return [], []
    landing = reconstruction.get("terminal_landing_partial") or {}
    support = reconstruction.get("terminal_support_candidate") or {}
    thickness = float(landing.get("thickness_mm") or 0.0)
    visible_extent = float(landing.get("visible_section_extent_mm") or 0.0)
    flight_width = float(landing.get("sweep_width_mm") or 0.0)
    support_width = float(support.get("section_width_mm") or 0.0)
    support_drop = float(support.get("drop_below_slab_mm") or 0.0)
    clear_span = float(support.get("clear_span_mm") or 0.0)
    if min(thickness, visible_extent, flight_width, support_width, support_drop, clear_span) <= 0:
        return [], []

    # This is an adapter-owned placement proposal.  The generic materializer
    # sees only the explicit rigid transform below and has no stair, vertical,
    # or canonical-axis assumptions.
    top_z = max(point[2] for point in vertices)
    top_vertices = [point for point in vertices if abs(point[2] - top_z) <= 1e-3]
    if len(top_vertices) < 2:
        return [], []
    y0, y1 = min(point[1] for point in top_vertices), max(point[1] for point in top_vertices)
    if abs((y1 - y0) - flight_width) > max(2.0, 0.02 * flight_width):
        return [], []
    flight_start_x = min(top_vertices, key=lambda point: abs(point[0]))[0]
    outward_sign = -1.0 if flight_start_x < 0 else 1.0
    axes = {
        "x": [outward_sign, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    }
    shared_refs = sorted(set(map(str, reconstruction.get("evidence_refs", []) or [])))
    landing_id = str(landing.get("id") or "")
    support_id = str(support.get("id") or "")
    landing_transform_id = _stable_id(page_number, "local_to_world_transform", landing_id, "canonical_terminal")
    support_transform_id = _stable_id(page_number, "local_to_world_transform", support_id, "canonical_terminal")
    landing_hypothesis_id = _stable_id(page_number, "partial_profile_sweep_hypothesis", landing_id)
    support_hypothesis_id = _stable_id(page_number, "partial_profile_sweep_hypothesis", support_id)

    hypotheses = [
        {
            "id": landing_hypothesis_id,
            "record_type": "partial_profile_sweep_hypothesis",
            "state": "resolved_search_hypothesis",
            "construction_role": "terminal_slab_context",
            "classification": "upper_landing_or_floor_slab",
            "profile_boundary": {
                "id": _stable_id(page_number, "partial_profile_boundary", landing_id),
                "record_type": "partial_profile_boundary",
                "state": "resolved",
                "closure_kind": "analysis_capped_closed",
                "ordered_points_uv_mm": [
                    [0.0, 0.0],
                    [visible_extent, 0.0],
                    [visible_extent, thickness],
                    [0.0, thickness],
                ],
                "classified_caps": [
                    {
                        "id": _stable_id(page_number, "partial_profile_cap", landing_id, "analysis"),
                        "boundary_edge_index": 1,
                        "classification": "analysis_cap",
                        "state": "resolved",
                        "evidence_refs": shared_refs,
                    },
                    {
                        "id": _stable_id(page_number, "partial_profile_cap", landing_id, "connection"),
                        "boundary_edge_index": 3,
                        "classification": "internal_seam_hypothesis",
                        "state": "resolved",
                        "evidence_refs": shared_refs,
                    },
                ],
                "quantity_eligible": False,
                "evidence_refs": shared_refs,
            },
            "sweep_bound": {
                "id": _stable_id(page_number, "partial_sweep_bound", landing_id),
                "record_type": "partial_sweep_bound",
                "state": "resolved",
                "closure_kind": "physical_bounded",
                "depth_mm": flight_width,
                "quantity_eligible": False,
                "evidence_refs": shared_refs,
            },
            "local_to_world_transform_alternatives": [
                {
                    "id": landing_transform_id,
                    "record_type": "local_to_world_transform",
                    "state": "resolved",
                    "origin_xyz_mm": [flight_start_x, y0, top_z - thickness],
                    "local_axes_xyz": axes,
                    "absolute_orientation_resolved": False,
                    "basis": "accepted_primary_preview_terminal_anchor",
                    "evidence_refs": shared_refs,
                }
            ],
            "quantity_eligible": False,
            "evidence_refs": shared_refs,
        },
        {
            "id": support_hypothesis_id,
            "record_type": "partial_profile_sweep_hypothesis",
            "state": "resolved_search_hypothesis",
            "construction_role": "terminal_support_context",
            "classification": "terminal_support_beam_candidate",
            "profile_boundary": {
                "id": _stable_id(page_number, "partial_profile_boundary", support_id),
                "record_type": "partial_profile_boundary",
                "state": "resolved",
                "closure_kind": "physical_closed",
                "ordered_points_uv_mm": [
                    [0.0, 0.0],
                    [support_width, 0.0],
                    [support_width, support_drop],
                    [0.0, support_drop],
                ],
                "classified_caps": [],
                "quantity_eligible": False,
                "evidence_refs": shared_refs,
            },
            "sweep_bound": {
                "id": _stable_id(page_number, "partial_sweep_bound", support_id),
                "record_type": "partial_sweep_bound",
                "state": "resolved",
                "closure_kind": "physical_bounded",
                "depth_mm": clear_span,
                "quantity_eligible": False,
                "evidence_refs": shared_refs,
            },
            "local_to_world_transform_alternatives": [
                {
                    "id": support_transform_id,
                    "record_type": "local_to_world_transform",
                    "state": "resolved",
                    "origin_xyz_mm": [flight_start_x, min(point[1] for point in vertices), top_z - thickness - support_drop],
                    "local_axes_xyz": axes,
                    "absolute_orientation_resolved": False,
                    "basis": "accepted_primary_preview_terminal_anchor",
                    "evidence_refs": shared_refs,
                }
            ],
            "quantity_eligible": False,
            "evidence_refs": shared_refs,
        },
    ]
    projection_evidence = [
        {
            "id": _stable_id(page_number, "projection_coverage_evidence", landing_id, "section"),
            "record_type": "projection_coverage_evidence",
            "state": "resolved",
            "hypothesis_ref": landing_hypothesis_id,
            "projection_role": "section",
            "coverage_state": "partial",
            "derived_from_generated_geometry": False,
            "evidence_refs": shared_refs,
        },
        {
            "id": _stable_id(page_number, "projection_coverage_evidence", landing_id, "plan"),
            "record_type": "projection_coverage_evidence",
            "state": "resolved",
            "hypothesis_ref": landing_hypothesis_id,
            "projection_role": "plan",
            "coverage_state": "not_depicted",
            "derived_from_generated_geometry": False,
            "evidence_refs": shared_refs,
        },
        {
            "id": _stable_id(page_number, "projection_coverage_evidence", support_id, "section"),
            "record_type": "projection_coverage_evidence",
            "state": "resolved",
            "hypothesis_ref": support_hypothesis_id,
            "projection_role": "section",
            "coverage_state": "covered",
            "derived_from_generated_geometry": False,
            "evidence_refs": shared_refs,
        },
        {
            "id": _stable_id(page_number, "projection_coverage_evidence", support_id, "plan"),
            "record_type": "projection_coverage_evidence",
            "state": "resolved",
            "hypothesis_ref": support_hypothesis_id,
            "projection_role": "plan",
            "coverage_state": "partial",
            "derived_from_generated_geometry": False,
            "evidence_refs": shared_refs,
        },
    ]
    return hypotheses, projection_evidence


def materialize_terminal_landing_context(
    reconstruction: Mapping[str, Any],
    primary_mesh: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Adapt terminal evidence to generic audit-only partial sweep records."""

    if reconstruction.get("status") != "resolved_partial_context":
        return {
            "schema_version": SCHEMA_VERSION,
            "layer": "partial_profile_sweep_materialization",
            "page": page_number,
            "status": "no_materializable_hypotheses",
            "candidate_previews": [],
            "summary": {"hypothesis_count": 0, "materialized_candidate_count": 0},
            "contract": {"quantity_eligible": False},
        }
    landing = reconstruction.get("terminal_landing_partial") or {}
    support = reconstruction.get("terminal_support_candidate") or {}
    hypotheses, projection_evidence = _partial_sweep_inputs(
        reconstruction,
        primary_mesh,
        page_number=page_number,
    )
    materialization = materialize_partial_profile_sweeps(
        hypotheses,
        projection_evidence,
        page_number=page_number,
    )
    for candidate in materialization["candidate_previews"]:
        candidate["record_type"] = "context_candidate_preview"
        candidate["relative_physical_placement_resolved"] = True
        candidate["absolute_orientation_resolved"] = False
        if candidate["classification"] == "upper_landing_or_floor_slab":
            candidate.update(
                {
                    "state": "resolved_relative_partial_extent",
                    "display_color_role": "identifier",
                    "external_cap_role": "analysis_cap",
                    "source_reconstruction_ref": landing.get("id"),
                    "reason": "section-only terminal slab is analysis-capped; stair-versus-floor scope remains unresolved",
                }
            )
        else:
            candidate.update(
                {
                    "state": "resolved_relative_unclassified_support",
                    "display_color_role": "candidate",
                    "volume_candidate_m3": support.get("volume_candidate_m3"),
                    "member_identity_resolved": False,
                    "source_reconstruction_ref": support.get("id"),
                    "reason": "section geometry and plan span agree; member identity and support overlap remain unresolved",
                }
            )
    materialization["source_adapter"] = {
        "record_type": "drawing_evidence_to_partial_sweep_adapter",
        "state": "resolved_for_fixture",
        "source_layer": "terminal_landing_support_reconstruction",
        "drawing_class_assumptions_confined_to_adapter": True,
        "generic_materializer_has_stair_semantics": False,
        "quantity_eligible": False,
    }
    return materialization
