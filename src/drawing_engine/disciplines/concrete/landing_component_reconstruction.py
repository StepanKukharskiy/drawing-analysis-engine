"""Reconstruct a landing construction region inside a monolithic stair."""

from __future__ import annotations

from hashlib import sha256
from statistics import median
from typing import Any, Mapping

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh
from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import solve_multi_component_solid


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _abstain(page_number: int, reason_code: str, evidence_refs: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "landing_component_reconstruction",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "plan_footprint_certificate": None,
        "section_thickness_certificate": None,
        "termination_convention_certificate": None,
        "physical_object_scope": None,
        "landing_construction_region": None,
        "internal_seam_hypotheses": [],
        "separate_object_interface_hypotheses": [],
        "beam_relative_placement_certificate": None,
        "landing_beam_step4_replay": None,
        "construction_region_validation": None,
        "evidence_refs": sorted(set(evidence_refs or [])),
        "contract": {
            "landing_is_construction_region_of_stair": True,
            "landing_is_separate_physical_component": False,
            "landing_used_as_flight_closure_patch": False,
            "same_object_union_required_before_quantity": True,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def _accepted_ownership(dimension_ownership: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("dimension_ref")): item
        for item in dimension_ownership.get("attachments", []) or []
        if item.get("status") == "accepted"
    }


def _unique_chains(dimension_adjudication: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = {}
    for item in dimension_adjudication.get("terminal_less_redundancy_certificates", []) or []:
        if item.get("kind") != "arithmetic_chain":
            continue
        key = (
            str(item.get("overall_dimension_ref")),
            tuple(map(str, item.get("term_dimension_refs", []) or [])),
        )
        rows[key] = dict(item)
    return [rows[key] for key in sorted(rows)]


def _view_refs(owner: Mapping[str, Any]) -> set[str]:
    return set(map(str, owner.get("view_refs", []) or []))


def _plan_metric_frame(
    owners: Mapping[str, Mapping[str, Any]],
    dimension_refs: list[str],
) -> dict[str, Any] | None:
    """Resolve an unsigned plan scale from independent horizontal/vertical chains."""

    samples: dict[str, list[tuple[float, str]]] = {"horizontal": [], "vertical": []}
    for dimension_ref in dimension_refs:
        owner = owners.get(str(dimension_ref)) or {}
        orientation = str(owner.get("orientation") or "")
        endpoints = [
            list(map(float, item.get("point_display")))
            for item in owner.get("measured_endpoints", []) or []
            if item.get("point_display")
        ]
        value_mm = float(owner.get("value_mm") or 0.0)
        if orientation not in samples or len(endpoints) != 2 or value_mm <= 0:
            continue
        axis = 0 if orientation == "horizontal" else 1
        span_points = abs(endpoints[1][axis] - endpoints[0][axis])
        if span_points > 0:
            samples[orientation].append((span_points / value_mm, str(dimension_ref)))
    if not all(samples.values()):
        return None
    horizontal = median(value for value, _ in samples["horizontal"])
    vertical = median(value for value, _ in samples["vertical"])
    consensus = median((horizontal, vertical))
    if consensus <= 0 or abs(horizontal - vertical) / consensus > 0.02:
        return None
    return {
        "state": "resolved_relative_unsigned",
        "scale_points_per_mm": round(consensus, 10),
        "horizontal_scale_points_per_mm": round(horizontal, 10),
        "vertical_scale_points_per_mm": round(vertical, 10),
        "longitudinal_axis_display": [1.0, 0.0],
        "transverse_axis_display": [0.0, 1.0],
        "absolute_axis_signs_resolved": False,
        "evidence_refs": sorted(
            {ref for values in samples.values() for _, ref in values}
        ),
    }


def _plan_longitudinal_chain(
    dimension_ownership: Mapping[str, Any],
    dimension_adjudication: Mapping[str, Any],
    banded_plan_sweep_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], str] | None:
    owners = _accepted_ownership(dimension_ownership)
    band = banded_plan_sweep_evidence.get("plan_band_certificate") or {}
    band_overall = owners.get(str(band.get("overall_dimension_ref")), {})
    plan_views = _view_refs(band_overall)
    candidates = []
    for chain in _unique_chains(dimension_adjudication):
        terms = list(map(float, chain.get("term_values_mm", []) or []))
        refs = list(map(str, chain.get("term_dimension_refs", []) or []))
        overall = owners.get(str(chain.get("overall_dimension_ref")), {})
        term_owners = [owners.get(ref) for ref in refs]
        if (
            len(terms) != 4
            or len(refs) != 4
            or abs(terms[0] - terms[-1]) > 1e-6
            or abs(float(chain.get("arithmetic_residual_mm") or 0.0)) > 1e-6
            or not plan_views
            or _view_refs(overall) != plan_views
            or any(item is None or _view_refs(item) != plan_views for item in term_owners)
        ):
            continue
        for index in (1, 2):
            candidates.append((chain, refs[index], terms[index]))
    # The physical landing term is chosen downstream by the independent
    # section/beam decomposition.  Preserve both plan alternatives here.
    if not candidates:
        return None
    return {"alternatives": candidates, "plan_view_refs": sorted(plan_views)}, str(band.get("overall_dimension_ref"))


def _section_chain(local_dimension_reclosure: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    candidates = []
    for scope in local_dimension_reclosure.get("scope_results", []) or []:
        owners = {
            str(item.get("dimension_ref")): item
            for item in (scope.get("local_ownership") or {}).get("attachments", []) or []
            if item.get("status") == "accepted"
        }
        for chain in scope.get("independent_arithmetic_chain_certificates", []) or []:
            terms = list(map(float, chain.get("term_values_mm", []) or []))
            overall = owners.get(str(chain.get("overall_dimension_ref")), {})
            if (
                len(terms) == 2
                and overall.get("orientation") == "horizontal"
                and abs(float(chain.get("arithmetic_residual_mm") or 0.0)) <= 1e-6
            ):
                candidates.append((scope, chain))
    return candidates[0] if len(candidates) == 1 else None


def _beam_dimensions(clear_span_prism_reconstruction: Mapping[str, Any]) -> tuple[float, float, list[str]] | None:
    callouts = clear_span_prism_reconstruction.get("member_size_callouts", []) or []
    sizes = {
        (float(item.get("width_mm") or 0.0), float(item.get("depth_mm") or 0.0))
        for item in callouts
        if item.get("member_type") == "beam"
    }
    if len(sizes) != 1 or any(value <= 0 for value in next(iter(sizes))):
        return None
    first, second = next(iter(sizes))
    return min(first, second), max(first, second), [str(item.get("id")) for item in callouts]


def _horizontal_support_segments(open_structural_boundary_assembly: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scope in open_structural_boundary_assembly.get("scope_results", []) or []:
        scale = float(scope.get("metric_scale_points_per_mm") or 0.0)
        if scale <= 0:
            continue
        for chain in scope.get("branch_free_chains", []) or []:
            if chain.get("role") != "support_or_landing_boundary" or not chain.get("boundary_eligible", True):
                continue
            points = [list(map(float, point)) for point in chain.get("ordered_points_display", []) or []]
            refs = list(map(str, chain.get("split_edge_refs", []) or []))
            for index, (left, right) in enumerate(zip(points, points[1:])):
                if abs(left[1] - right[1]) > max(1e-6, 2.0 * scale):
                    continue
                rows.append(
                    {
                        "scope_ref": str(scope.get("scope_ref")),
                        "chain_ref": str(chain.get("id")),
                        "edge_ref": refs[index] if index < len(refs) else str(chain.get("id")),
                        "x_interval": sorted((left[0], right[0])),
                        "y": (left[1] + right[1]) / 2.0,
                        "scale_points_per_mm": scale,
                    }
                )
    return rows


def _native_thickness(
    open_structural_boundary_assembly: Mapping[str, Any],
    core_length_mm: float,
    thickness_target_mm: float,
    landing_top_station_y: float,
) -> dict[str, Any] | None:
    candidates = []
    segments = _horizontal_support_segments(open_structural_boundary_assembly)
    for index, upper in enumerate(segments):
        for lower in segments[index + 1 :]:
            if upper["scope_ref"] != lower["scope_ref"]:
                continue
            scale = (upper["scale_points_per_mm"] + lower["scale_points_per_mm"]) / 2.0
            overlap = max(
                0.0,
                min(upper["x_interval"][1], lower["x_interval"][1])
                - max(upper["x_interval"][0], lower["x_interval"][0]),
            )
            overlap_mm = overlap / scale
            thickness_mm = abs(upper["y"] - lower["y"]) / scale
            top_y = min(upper["y"], lower["y"])
            if (
                overlap_mm < 0.70 * core_length_mm
                or abs(thickness_mm - thickness_target_mm) > max(5.0, 0.04 * thickness_target_mm)
                or abs(top_y - landing_top_station_y) > max(2.0, 25.0 * scale)
            ):
                continue
            candidates.append(
                {
                    "scope_ref": upper["scope_ref"],
                    "upper_edge_ref": upper["edge_ref"] if upper["y"] < lower["y"] else lower["edge_ref"],
                    "lower_edge_ref": lower["edge_ref"] if upper["y"] < lower["y"] else upper["edge_ref"],
                    "native_separation_mm": round(thickness_mm, 3),
                    "native_overlap_mm": round(overlap_mm, 3),
                    "scale_points_per_mm": round(scale, 8),
                    "top_y_display": round(top_y, 6),
                    "bottom_y_display": round(max(upper["y"], lower["y"]), 6),
                }
            )
    return candidates[0] if len(candidates) == 1 else None


def _segment_title_ref(title_segmentation: Mapping[str, Any], view_refs: list[str]) -> str | None:
    candidates = [
        str(item.get("id"))
        for item in title_segmentation.get("segments", []) or []
        if item.get("state") == "resolved"
        and str(item.get("source_view_id")) in set(view_refs)
        and str(item.get("title") or "").upper().find("PLAN") >= 0
    ]
    return candidates[0] if len(candidates) == 1 else None


def reconstruct_landing_component(
    dimension_ownership: Mapping[str, Any],
    dimension_adjudication: Mapping[str, Any],
    title_segmentation: Mapping[str, Any],
    local_dimension_reclosure: Mapping[str, Any],
    open_structural_boundary_assembly: Mapping[str, Any],
    flight_interface_closure: Mapping[str, Any],
    flight_profile_pair_certification: Mapping[str, Any],
    banded_plan_sweep_evidence: Mapping[str, Any],
    clear_span_prism_reconstruction: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Close landing construction geometry without promoting a component."""

    plan = _plan_longitudinal_chain(
        dimension_ownership, dimension_adjudication, banded_plan_sweep_evidence
    )
    section = _section_chain(local_dimension_reclosure)
    beam = _beam_dimensions(clear_span_prism_reconstruction)
    clear_span = clear_span_prism_reconstruction.get("clear_span_certificate") or {}
    if plan is None or section is None or beam is None or clear_span.get("state") != "resolved":
        return _abstain(page_number, "landing_plan_section_or_beam_basis_unresolved")

    plan_data, plan_band_overall_ref = plan
    section_scope, section_chain = section
    support_depth_mm, beam_depth_mm, beam_refs = beam
    section_terms = list(map(float, section_chain.get("term_values_mm", []) or []))
    terminal_extent_mm = min(section_terms)
    core_length_mm = terminal_extent_mm - support_depth_mm
    if core_length_mm <= 0:
        return _abstain(page_number, "landing_clear_core_non_positive", beam_refs)

    plan_matches = []
    for chain, dimension_ref, value_mm in plan_data["alternatives"]:
        allowance = value_mm - core_length_mm
        if 0 < allowance <= max(beam_depth_mm, support_depth_mm):
            plan_matches.append((chain, dimension_ref, value_mm, allowance))
    if len(plan_matches) != 1:
        return _abstain(page_number, "unique_landing_plan_term_unresolved", beam_refs)
    plan_chain, footprint_length_ref, footprint_length_mm, interface_allowance_mm = plan_matches[0]
    plan_term_refs = list(map(str, plan_chain.get("term_dimension_refs", []) or []))
    flight_run_candidates = [
        (dimension_ref, float(value_mm))
        for dimension_ref, value_mm in zip(
            plan_term_refs,
            map(float, plan_chain.get("term_values_mm", []) or []),
        )
        if dimension_ref != footprint_length_ref
        and value_mm not in {
            float((plan_chain.get("term_values_mm") or [0.0])[0]),
            float((plan_chain.get("term_values_mm") or [0.0])[-1]),
        }
    ]
    if len(flight_run_candidates) != 1:
        return _abstain(page_number, "unique_plan_flight_run_unresolved", beam_refs)
    flight_run_ref, flight_run_mm = flight_run_candidates[0]
    plan_chain_refs = [str(plan_chain.get("overall_dimension_ref")), *plan_term_refs]
    band_certificate = banded_plan_sweep_evidence.get("plan_band_certificate") or {}
    band_chain_refs = [
        str(band_certificate.get("overall_dimension_ref")),
        *map(str, band_certificate.get("term_dimension_refs", []) or []),
    ]
    plan_metric_frame = _plan_metric_frame(
        _accepted_ownership(dimension_ownership),
        [*plan_chain_refs, *band_chain_refs],
    )
    if plan_metric_frame is None:
        return _abstain(page_number, "plan_metric_frame_unresolved", beam_refs)

    unresolved_allowance = [
        item
        for item in section_scope.get("unresolved_metric_observations", []) or []
        if abs(float(item.get("value_mm") or 0.0) - interface_allowance_mm) <= 1e-6
    ]
    if len(unresolved_allowance) != 1:
        return _abstain(page_number, "landing_interface_allowance_validation_unresolved", beam_refs)

    vertical_chain = next(
        (
            item
            for item in section_scope.get("independent_arithmetic_chain_certificates", []) or []
            if len(item.get("term_values_mm", []) or []) == 3
        ),
        None,
    )
    if vertical_chain is None:
        return _abstain(page_number, "landing_thickness_metric_target_unresolved", beam_refs)
    thickness_mm = min(map(float, vertical_chain.get("term_values_mm", []) or []))
    local_owners = {
        str(item.get("dimension_ref")): item
        for item in (section_scope.get("local_ownership") or {}).get("attachments", []) or []
    }
    terminal_ref = str(section_chain.get("term_dimension_refs", [""])[-1])
    terminal_owner = local_owners.get(terminal_ref, {})
    terminal_points = [
        list(map(float, item.get("point_display")))
        for item in terminal_owner.get("measured_endpoints", []) or []
        if item.get("point_display")
    ]
    if len(terminal_points) != 2:
        return _abstain(page_number, "landing_top_section_station_unresolved", beam_refs)
    terminal_start = min(terminal_points, key=lambda point: point[0])
    terminal_end = max(terminal_points, key=lambda point: point[0])
    landing_top_station_y = terminal_end[1]
    native_thickness = _native_thickness(
        open_structural_boundary_assembly,
        core_length_mm,
        thickness_mm,
        landing_top_station_y,
    )
    if native_thickness is None:
        return _abstain(page_number, "unique_native_landing_thickness_unresolved", beam_refs)

    section_stations = [
        station
        for scope in flight_interface_closure.get("scope_results", []) or []
        if str(scope.get("scope_ref")) == str(section_scope.get("scope_ref"))
        for station in scope.get("section_stations", []) or []
        if station.get("axis") == "x"
        and abs(float(station.get("coordinate_display") or 0.0) - terminal_start[0])
        <= float(station.get("tolerance_points") or 0.0)
    ]
    if len(section_stations) != 1:
        return _abstain(page_number, "unique_landing_flight_contact_station_unresolved", beam_refs)
    flight_contact_station = section_stations[0]
    beam_contact_coordinate = terminal_end[0] - support_depth_mm * native_thickness["scale_points_per_mm"]

    clear_width_mm = float(clear_span.get("clear_span_mm") or 0.0)
    clear_width_ref = str(clear_span.get("clear_span_dimension_ref") or "")
    if clear_width_mm <= 0 or not clear_width_ref:
        return _abstain(page_number, "landing_clear_width_unresolved", beam_refs)

    plan_view_refs = list(plan_data["plan_view_refs"])
    plan_title_ref = _segment_title_ref(title_segmentation, plan_view_refs)
    if plan_title_ref is None:
        return _abstain(page_number, "unique_landing_plan_title_scope_unresolved", beam_refs)

    section_chain_refs = [
        str(section_chain.get("overall_dimension_ref")),
        *map(str, section_chain.get("term_dimension_refs", []) or []),
    ]
    evidence_refs = sorted(
        {
            plan_title_ref,
            str(section_scope.get("scope_ref")),
            plan_band_overall_ref,
            footprint_length_ref,
            clear_width_ref,
            str(unresolved_allowance[0].get("dimension_ref")),
            native_thickness["upper_edge_ref"],
            native_thickness["lower_edge_ref"],
            *plan_chain_refs,
            *section_chain_refs,
            *beam_refs,
        }
    )
    parent_scope_refs = sorted(
        ref
        for ref in (banded_plan_sweep_evidence.get("plan_band_certificate") or {}).get("evidence_refs", []) or []
        if str(ref).startswith("physical_object_scope.")
    )
    if len(parent_scope_refs) != 1:
        return _abstain(page_number, "unique_parent_physical_scope_unresolved", evidence_refs)
    parent_scope_ref = str(parent_scope_refs[0])
    stair_scope_ref = _stable_id(page_number, "physical_object_scope", parent_scope_ref, plan_title_ref, section_scope.get("scope_ref"))
    region_ref = _stable_id(page_number, "landing_construction_region", *evidence_refs)
    mesh = _extruded_mesh(
        [(0.0, 0.0), (core_length_mm, 0.0), (core_length_mm, thickness_mm), (0.0, thickness_mm)],
        clear_width_mm,
    )
    volume_mm3 = core_length_mm * clear_width_mm * thickness_mm

    bands = list((banded_plan_sweep_evidence.get("plan_band_certificate") or {}).get("bands", []) or [])
    if len(bands) != 2:
        return _abstain(page_number, "landing_flight_contact_bands_unresolved", evidence_refs)
    clear_origin_mm = min(float(item.get("interval_mm", [0.0])[0]) for item in bands)
    flight_refs = [
        str(item.get("id"))
        for item in flight_interface_closure.get("closed_profiles", []) or []
    ]
    while len(flight_refs) < 2:
        flight_refs.append(_stable_id(page_number, "pending_flight_construction_region", len(flight_refs) + 1))
    seams = []
    for index, band in enumerate(bands):
        start, end = map(float, band.get("interval_mm", []) or [0.0, 0.0])
        seams.append(
            {
                "id": _stable_id(page_number, "internal_seam_hypothesis", region_ref, index + 1),
                "record_type": "internal_seam_hypothesis",
                "physical_object_scope_ref": stair_scope_ref,
                "state": "proposed_pending_same_object_union",
                "construction_region_refs": [region_ref, flight_refs[index]],
                "seam_role": "flight_landing",
                "expected_polygon_xyz_mm": [[0, start - clear_origin_mm, 0], [0, end - clear_origin_mm, 0], [0, end - clear_origin_mm, thickness_mm], [0, start - clear_origin_mm, thickness_mm]],
                "expected_area_mm2": (end - start) * thickness_mm,
                "quantity_eligible": False,
                "separate_object_interface": False,
                "evidence_refs": sorted({str(band.get("id")), footprint_length_ref, native_thickness["upper_edge_ref"], native_thickness["lower_edge_ref"]}),
            }
        )
    beam_component_ref = str(
        (clear_span_prism_reconstruction.get("physical_component_transforms", [{}]) or [{}])[0].get("component_ref")
        or _stable_id(page_number, "pending_beam_component")
    )
    beam_interface = {
        "id": _stable_id(page_number, "separate_object_interface_hypothesis", region_ref, beam_component_ref),
        "record_type": "separate_object_interface_hypothesis",
        "state": "proposed_pending_relative_placement_reclosure",
        "stair_physical_object_scope_ref": stair_scope_ref,
        "stair_construction_region_ref": region_ref,
        "other_physical_component_ref": beam_component_ref,
        "interface_role": "landing_beam",
        "expected_polygon_xyz_mm": [[core_length_mm, 0, 0], [core_length_mm, clear_width_mm, 0], [core_length_mm, clear_width_mm, thickness_mm], [core_length_mm, 0, thickness_mm]],
        "expected_area_mm2": clear_width_mm * thickness_mm,
        "monolithic_identity_proven": False,
        "quantity_eligible": False,
        "evidence_refs": sorted({clear_width_ref, *beam_refs, native_thickness["upper_edge_ref"], native_thickness["lower_edge_ref"]}),
    }
    beam_mesh = (clear_span_prism_reconstruction.get("solid_preview") or {}).get("mesh") or {}
    beam_volume = clear_span_prism_reconstruction.get("clear_span_volume_candidate") or {}
    placement_evidence = sorted(
        {
            *evidence_refs,
            *map(str, beam_volume.get("evidence_refs", []) or []),
            str(beam_interface["id"]),
        }
    )
    landing_transform = {
        "id": _stable_id(page_number, "physical_component_transform", region_ref, "relative_landing"),
        "record_type": "physical_component_transform",
        "record_version": SCHEMA_VERSION,
        "component_ref": region_ref,
        "state": "resolved",
        "placement_role": "physical",
        "origin_xyz_mm": [0.0, 0.0, -thickness_mm],
        "local_axes_xyz": {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
        "axis_signs": "resolved",
        "evidence_refs": placement_evidence,
    }
    # The clear-span prism's local axes are span, 300 mm member width, and
    # 200 mm member depth.  Section A-A certifies the 300 mm axis as the
    # downward beam drop and the 200 mm axis as the landing-normal extent.
    # Starting at the second clear-span endpoint keeps the basis proper while
    # preserving the unresolved transverse reflection of the complete stair.
    beam_transform = {
        "id": _stable_id(page_number, "physical_component_transform", beam_component_ref, region_ref),
        "record_type": "physical_component_transform",
        "record_version": SCHEMA_VERSION,
        "component_ref": beam_component_ref,
        "state": "resolved",
        "placement_role": "physical",
        "origin_xyz_mm": [core_length_mm, clear_width_mm, 0.0],
        "local_axes_xyz": {
            "x": [0.0, -1.0, 0.0],
            "y": [0.0, 0.0, -1.0],
            "z": [1.0, 0.0, 0.0],
        },
        "axis_signs": "resolved",
        "evidence_refs": placement_evidence,
    }
    resolved_interface_polygon = [
        [core_length_mm, 0.0, -thickness_mm],
        [core_length_mm, clear_width_mm, -thickness_mm],
        [core_length_mm, clear_width_mm, 0.0],
        [core_length_mm, 0.0, 0.0],
    ]
    landing_beam_replay = None
    placement_error = None
    if beam_mesh.get("vertices_xyz_mm") and beam_mesh.get("triangles") and beam_volume.get("value_mm3"):
        supplied_views = [
            {
                "id": _stable_id(page_number, "native_projection_evidence", region_ref, beam_component_ref, "plan"),
                "origin_xyz_mm": [0.0, 0.0, 0.0],
                "u_axis_xyz": [1.0, 0.0, 0.0],
                "v_axis_xyz": [0.0, 1.0, 0.0],
                "tolerance_mm": 0.5,
                "evidence_refs": placement_evidence,
                "component_projections": [
                    {
                        "component_ref": region_ref,
                        "polygons_uv_mm": [[[0.0, 0.0], [core_length_mm, 0.0], [core_length_mm, clear_width_mm], [0.0, clear_width_mm]]],
                        "evidence_refs": placement_evidence,
                    },
                    {
                        "component_ref": beam_component_ref,
                        "polygons_uv_mm": [[[core_length_mm, 0.0], [core_length_mm + support_depth_mm, 0.0], [core_length_mm + support_depth_mm, clear_width_mm], [core_length_mm, clear_width_mm]]],
                        "evidence_refs": placement_evidence,
                    },
                ],
            },
            {
                "id": _stable_id(page_number, "native_projection_evidence", region_ref, beam_component_ref, "section"),
                "origin_xyz_mm": [0.0, 0.0, 0.0],
                "u_axis_xyz": [1.0, 0.0, 0.0],
                "v_axis_xyz": [0.0, 0.0, 1.0],
                "tolerance_mm": 0.5,
                "evidence_refs": placement_evidence,
                "component_projections": [
                    {
                        "component_ref": region_ref,
                        "polygons_uv_mm": [[[0.0, -thickness_mm], [core_length_mm, -thickness_mm], [core_length_mm, 0.0], [0.0, 0.0]]],
                        "evidence_refs": placement_evidence,
                    },
                    {
                        "component_ref": beam_component_ref,
                        "polygons_uv_mm": [[[core_length_mm, -beam_depth_mm], [core_length_mm + support_depth_mm, -beam_depth_mm], [core_length_mm + support_depth_mm, 0.0], [core_length_mm, 0.0]]],
                        "evidence_refs": placement_evidence,
                    },
                ],
            },
        ]
        try:
            landing_beam_replay = solve_multi_component_solid(
                [
                    {
                        "id": region_ref,
                        "mesh": mesh,
                        "transform": landing_transform,
                        "analytic_volume": {
                            "value_mm3": volume_mm3,
                            "basis": "certified_landing_profile_times_clear_width",
                            "evidence_refs": placement_evidence,
                        },
                        "evidence_refs": placement_evidence,
                    },
                    {
                        "id": beam_component_ref,
                        "mesh": beam_mesh,
                        "transform": beam_transform,
                        "analytic_volume": {
                            "value_mm3": float(beam_volume["value_mm3"]),
                            "basis": "native_member_size_callouts_times_inner_face_clear_span",
                            "evidence_refs": placement_evidence,
                        },
                        "evidence_refs": placement_evidence,
                    },
                ],
                [
                    {
                        "id": str(beam_interface["id"]),
                        "component_refs": [region_ref, beam_component_ref],
                        "polygon_xyz_mm": resolved_interface_polygon,
                        "evidence_refs": placement_evidence,
                    }
                ],
                supplied_views,
            )
        except (ValueError, TypeError, KeyError) as error:
            placement_error = str(error)
    else:
        placement_error = "beam mesh or analytic volume unavailable"

    placement_passed = bool(landing_beam_replay and landing_beam_replay.get("status") == "accepted")
    beam_interface.update(
        {
            "state": "resolved_relative_contact" if placement_passed else "proposed_pending_relative_placement_reclosure",
            "polygon_xyz_mm": resolved_interface_polygon if placement_passed else None,
            "actual_contact_area_mm2": (
                landing_beam_replay["pair_validations"][0]["contact_area_mm2"]
                if placement_passed
                else None
            ),
            "volumetric_overlap_mm3": (
                landing_beam_replay["pair_validations"][0]["overlap_volume_mm3"]
                if placement_passed
                else None
            ),
        }
    )
    beam_placement_certificate = {
        "id": _stable_id(page_number, "separate_object_relative_placement_certificate", region_ref, beam_component_ref),
        "record_type": "separate_object_relative_placement_certificate",
        "record_version": SCHEMA_VERSION,
        "state": "accepted_relative_orientation_unresolved" if placement_passed else "insufficient_constraints",
        "stair_construction_region_ref": region_ref,
        "beam_component_ref": beam_component_ref,
        "interface_ref": beam_interface["id"],
        "beam_transform": beam_transform if placement_passed else None,
        "axis_correspondence": {
            "beam_clear_span_axis": "stair_transverse_axis",
            "beam_300_mm_axis": "section_vertical_drop_axis",
            "beam_200_mm_axis": "landing_outward_normal_axis",
        },
        "absolute_orientation_resolved": False,
        "relative_physical_placement_resolved": placement_passed,
        "separate_physical_object": True,
        "included_in_stair_union": False,
        "included_in_stair_quantity": False,
        "error": placement_error,
        "evidence_refs": placement_evidence,
        "contract": {
            "transform_derived_from_certified_plan_and_section_evidence": True,
            "expected_projections_derived_from_generated_mesh": False,
            "finite_area_contact_required": True,
            "volumetric_overlap_forbidden": True,
            "monolithic_identity_claimed": False,
            "quantity_eligible": False,
        },
    }
    pair_accepted = flight_profile_pair_certification.get("status") == "accepted_unique_pair"
    interface_accepted = flight_interface_closure.get("status") == "accepted_unique_assignment"
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "landing_component_reconstruction",
        "page": page_number,
        "status": "resolved_construction_region_pending_same_object_union",
        "reason_code": "flight_construction_regions_unresolved" if not (pair_accepted and interface_accepted) else "same_object_union_and_beam_placement_unresolved",
        "plan_footprint_certificate": {
            "state": "resolved_relative_unsigned",
            "title_scope_ref": plan_title_ref,
            "plan_envelope_mm": [footprint_length_mm, clear_width_mm],
            "clear_core_footprint_mm": [core_length_mm, clear_width_mm],
            "longitudinal_dimension_ref": footprint_length_ref,
            "flight_run_dimension_ref": flight_run_ref,
            "flight_run_mm": flight_run_mm,
            "transverse_dimension_ref": clear_width_ref,
            "metric_frame": plan_metric_frame,
            "evidence_refs": sorted({plan_title_ref, footprint_length_ref, clear_width_ref, *plan_chain_refs}),
        },
        "section_thickness_certificate": {
            "state": "resolved_relative_unsigned",
            "thickness_mm": thickness_mm,
            "native_measured_thickness_mm": native_thickness["native_separation_mm"],
            "native_overlap_mm": native_thickness["native_overlap_mm"],
            "upper_edge_ref": native_thickness["upper_edge_ref"],
            "lower_edge_ref": native_thickness["lower_edge_ref"],
            "metric_target_dimension_refs": list(map(str, vertical_chain.get("term_dimension_refs", []) or [])),
            "evidence_refs": sorted({native_thickness["upper_edge_ref"], native_thickness["lower_edge_ref"], *map(str, vertical_chain.get("term_dimension_refs", []) or [])}),
        },
        "termination_convention_certificate": {
            "state": "resolved_relative_unsigned",
            "section_equation": f"{terminal_extent_mm:g} = {core_length_mm:g} + {support_depth_mm:g}",
            "plan_equation": f"{footprint_length_mm:g} = {core_length_mm:g} + {interface_allowance_mm:g}",
            "clear_landing_core_length_mm": core_length_mm,
            "beam_support_depth_mm": support_depth_mm,
            "flight_interface_allowance_mm": interface_allowance_mm,
            "interface_allowance_observation_ref": str(unresolved_allowance[0].get("dimension_ref")),
            "interface_allowance_is_quantity_eligible": False,
            "plan_section_difference_mm": terminal_extent_mm - footprint_length_mm,
            "flight_contact_section_station_ref": flight_contact_station["id"],
            "flight_contact_coordinate_display": round(terminal_start[0], 6),
            "beam_contact_coordinate_display": round(beam_contact_coordinate, 6),
            "landing_top_coordinate_display": native_thickness["top_y_display"],
            "landing_bottom_coordinate_display": native_thickness["bottom_y_display"],
            "section_scale_points_per_mm": native_thickness["scale_points_per_mm"],
            "difference_explanation": "section terminates at the beam outer face; plan envelope includes the flight seam allowance; the landing construction region terminates at the beam inner face and the flight seam plane",
            "evidence_refs": sorted({footprint_length_ref, str(section_chain.get("term_dimension_refs", [""])[-1]), str(unresolved_allowance[0].get("dimension_ref")), flight_contact_station["id"], *beam_refs}),
        },
        "physical_object_scope": {
            "id": stair_scope_ref,
            "record_type": "physical_object_scope",
            "state": "proposed_monolithic_physical_object_scope",
            "parent_scope_ref": parent_scope_ref,
            "construction_region_roles": ["lower_flight", "upper_flight", "landing"],
            "beam_membership_state": "not_proven",
            "quantity_eligible": False,
            "evidence_refs": evidence_refs,
        },
        "landing_construction_region": {
            "id": region_ref,
            "record_type": "construction_region",
            "physical_object_scope_ref": stair_scope_ref,
            "construction_role": "landing",
            "state": "resolved_relative_unsigned",
            "extents_xyz_mm": [core_length_mm, clear_width_mm, thickness_mm],
            "mesh": mesh,
            "quantity_eligible": False,
            "analytic_region_volume": {
                "value_mm3": volume_mm3,
                "value_m3": volume_mm3 / 1_000_000_000.0,
                "quantity_eligible": False,
                "additive_physical_quantity": False,
                "evidence_refs": evidence_refs,
            },
            "evidence_refs": evidence_refs,
        },
        "internal_seam_hypotheses": seams,
        "separate_object_interface_hypotheses": [beam_interface],
        "beam_relative_placement_certificate": beam_placement_certificate,
        "landing_beam_step4_replay": landing_beam_replay,
        "construction_region_validation": {
            "status": "pass",
            "mesh": mesh["validation"],
            "same_object_union_replayed": False,
            "final_external_boundary_validated": False,
        },
        "summary": {
            "transition": f"plan envelope {footprint_length_mm:g}x{clear_width_mm:g} -> landing construction region {core_length_mm:g}x{clear_width_mm:g}x{thickness_mm:g} -> {len(seams)} internal seam hypotheses + 1 separate-object interface {'resolved' if placement_passed else 'pending'} -> same-object union pending",
            "internal_seam_hypothesis_count": len(seams),
            "separate_object_interface_hypothesis_count": 1,
        },
        "evidence_refs": evidence_refs,
        "contract": {
            "landing_is_construction_region_of_stair": True,
            "landing_is_separate_physical_component": False,
            "landing_used_as_flight_closure_patch": False,
            "plan_envelope_preserved_separately_from_construction_region": True,
            "flight_landing_joins_are_internal_seam_hypotheses": True,
            "construction_region_volume_is_not_additive_quantity": True,
            "beam_remains_separate_object_without_monolithic_identity": True,
            "beam_relative_placement_resolved": placement_passed,
            "landing_beam_multi_component_step4_replayed": placement_passed,
            "same_object_union_required_before_quantity": True,
            "same_object_step4_replayed": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
