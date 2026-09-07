"""Assemble construction-region evidence for one physical solid.

This Step 5 layer does not add region volumes.  It may invoke the same-object
Step 4 union only after every region, internal seam, final external boundary,
analytic union volume, and supplied reprojection is independently resolved.
"""

from __future__ import annotations

from hashlib import sha256
import itertools
import math
from typing import Any, Mapping

from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon, box
from shapely.ops import polygonize

from src.drawing_engine.disciplines.concrete.bounded_profile_sweep_materializer import materialize_bounded_profile_sweep_union
from src.drawing_engine.disciplines.concrete.constructive_union_enumerator import enumerate_constructive_union_hypotheses
from src.drawing_engine.disciplines.concrete.profile_to_sweep_transform_enumerator import enumerate_profile_to_sweep_transforms
from src.drawing_engine.disciplines.concrete.plan_direction_evidence import derive_plan_direction_evidence


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _line_intervals(geometry: Any) -> list[tuple[float, float]]:
    if isinstance(geometry, LineString):
        rows = [geometry]
    elif isinstance(geometry, MultiLineString):
        rows = list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        rows = [item for item in geometry.geoms if isinstance(item, LineString)]
    else:
        rows = []
    return sorted(
        (min(point[0] for point in line.coords), max(point[0] for point in line.coords))
        for line in rows
        if line.length > 1e-6
    )


def _native_folded_plan_topology_certificate(
    native_topology: Mapping[str, Any],
    title_segmentation: Mapping[str, Any],
    landing_reconstruction: Mapping[str, Any],
    bands: list[Mapping[str, Any]],
    *,
    page_number: int,
) -> dict[str, Any] | None:
    """Certify the dogleg fold from one native concave plan outline."""

    footprint = landing_reconstruction.get("plan_footprint_certificate") or {}
    metric_frame = footprint.get("metric_frame") or {}
    scale = float(metric_frame.get("scale_points_per_mm") or 0.0)
    title_ref = str(footprint.get("title_scope_ref") or "")
    clear_width = float((footprint.get("clear_core_footprint_mm") or [0.0, 0.0])[1])
    if scale <= 0 or not title_ref or clear_width <= 0 or len(bands) != 2:
        return None
    title_scope = next(
        (
            item
            for item in title_segmentation.get("segments", []) or []
            if str(item.get("id")) == title_ref and item.get("state") == "resolved"
        ),
        None,
    )
    if title_scope is None:
        return None
    primitive_refs = set(map(str, title_scope.get("primitive_refs", []) or []))
    by_drawing: dict[str, list[Mapping[str, Any]]] = {}
    for segment in native_topology.get("segments", []) or []:
        if (
            str(segment.get("drawing_ref")) not in primitive_refs
            and str(segment.get("primitive_ref")) not in primitive_refs
        ) or segment.get("kind") != "line":
            continue
        by_drawing.setdefault(str(segment.get("drawing_ref")), []).append(segment)

    candidates = []
    for drawing_ref, segments in by_drawing.items():
        lines = [
            LineString([segment["start_display"], segment["end_display"]])
            for segment in segments
        ]
        polygons = list(polygonize(lines))
        if len(polygons) != 1:
            continue
        polygon = polygons[0]
        coordinates = list(polygon.exterior.coords)[:-1]
        if len(coordinates) < 8 or len(coordinates) == 4:
            continue
        minimum_x, minimum_y, maximum_x, maximum_y = polygon.bounds
        if abs((maximum_y - minimum_y) / scale - clear_width) > max(10.0, 0.02 * clear_width):
            continue
        transverse_origin_mm = min(
            float(band.get("interval_mm", [0.0])[0]) for band in bands
        )
        y_samples = [
            minimum_y
            + (sum(map(float, band.get("interval_mm", []) or [])) / 2.0 - transverse_origin_mm)
            * scale
            for band in bands
        ]
        gap_start = float(bands[0].get("interval_mm", [0.0, 0.0])[1])
        gap_end = float(bands[1].get("interval_mm", [0.0, 0.0])[0])
        y_samples.append(
            minimum_y + ((gap_start + gap_end) / 2.0 - transverse_origin_mm) * scale
        )
        intervals = [
            _line_intervals(
                polygon.intersection(
                    LineString([(minimum_x - 1.0, y), (maximum_x + 1.0, y)])
                )
            )
            for y in y_samples
        ]
        if any(len(rows) != 1 for rows in intervals):
            continue
        first, second, gap = (rows[0] for rows in intervals)
        tolerance = max(0.75, 2.0 * scale)
        if max(abs(first[index] - second[index]) for index in (0, 1)) > tolerance:
            continue
        shared_endpoints = [
            index
            for index in (0, 1)
            if abs(gap[index] - first[index]) <= tolerance
            and abs(gap[1 - index] - first[1 - index]) > tolerance
        ]
        if len(shared_endpoints) != 1 or gap[1] - gap[0] >= first[1] - first[0] - tolerance:
            continue
        shared_index = shared_endpoints[0]
        landing_outer_x = gap[shared_index]
        seam_x = gap[1 - shared_index]
        landing_sign = 1.0 if landing_outer_x > seam_x else -1.0
        normalized = [
            [
                round((float(x) - seam_x) * landing_sign / scale, 6),
                round((float(y) - minimum_y) / scale, 6),
            ]
            for x, y in coordinates
        ]
        candidates.append(
            {
                "drawing_ref": drawing_ref,
                "segment_refs": sorted(str(item.get("id")) for item in segments),
                "polygon_uv_mm": normalized,
                "flight_interval_display": [first[0], first[1]],
                "landing_interval_display": [gap[0], gap[1]],
                "landing_seam_coordinate_display": seam_x,
                "landing_outer_coordinate_display": landing_outer_x,
                "landing_positive_display_sign": landing_sign,
                "plan_to_relative_xy": {
                    "origin_display": [seam_x, minimum_y],
                    "u_axis_display": [landing_sign, 0.0],
                    "v_axis_display": [0.0, 1.0],
                    "scale_points_per_mm": scale,
                },
                "native_stroke_width_points": float(
                    (native_topology.get("drawing_styles", {}).get(drawing_ref) or {}).get("width")
                    or 0.0
                ),
            }
        )
    if len(candidates) != 1:
        return None
    selected = candidates[0]
    native_vertex_tolerance_points = float(
        native_topology.get("vertex_tolerance_points") or 0.0
    )
    native_endpoint_resolution_mm = (
        2.0
        * max(selected["native_stroke_width_points"], native_vertex_tolerance_points)
        / scale
    )
    evidence_refs = sorted(
        {
            title_ref,
            selected["drawing_ref"],
            *selected["segment_refs"],
            *map(str, metric_frame.get("evidence_refs", []) or []),
            *(str(item.get("id")) for item in bands),
        }
    )
    return {
        "id": _stable_id(page_number, "folded_plan_topology_certificate", *evidence_refs),
        "record_type": "folded_plan_topology_certificate",
        "state": "resolved_relative_unsigned",
        "title_scope_ref": title_ref,
        "native_outline_drawing_ref": selected["drawing_ref"],
        "native_outline_segment_refs": selected["segment_refs"],
        "polygon_uv_mm": selected["polygon_uv_mm"],
        "plan_to_relative_xy": selected["plan_to_relative_xy"],
        "flight_bands_share_longitudinal_interval": True,
        "landing_connects_same_band_endpoint": True,
        "required_profile_longitudinal_relation": "same_side_of_landing_seam",
        "landing_seam_coordinate_display": selected["landing_seam_coordinate_display"],
        "landing_outer_coordinate_display": selected["landing_outer_coordinate_display"],
        "landing_positive_display_sign": selected["landing_positive_display_sign"],
        "native_stroke_width_points": selected["native_stroke_width_points"],
        "native_vertex_tolerance_points": native_vertex_tolerance_points,
        "native_endpoint_resolution_mm": native_endpoint_resolution_mm,
        "precision_basis": "two native stroke/vertex endpoint envelopes in the certified metric frame",
        "flight_body_length_mm": abs(
            selected["landing_seam_coordinate_display"]
            - selected["flight_interval_display"][0 if selected["landing_positive_display_sign"] > 0 else 1]
        )
        / scale,
        "landing_length_mm": abs(
            selected["landing_outer_coordinate_display"]
            - selected["landing_seam_coordinate_display"]
        )
        / scale,
        "absolute_orientation_resolved": False,
        "alternative_independent": True,
        "evidence_refs": evidence_refs,
        "quantity_eligible": False,
    }


def _classified_assignment_boundaries(
    page_number: int,
    scope_ref: str,
    region_by_proposal: Mapping[str, str],
    global_assignment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seams = []
    external = []
    for assignment in global_assignment.get("assignments", []) or []:
        proposal_ref = str(assignment.get("proposal_ref"))
        region_ref = region_by_proposal.get(proposal_ref)
        options = {
            str(option.get("id")): option
            for option in assignment.get("interface_options", []) or []
        }
        for classified in assignment.get("classified_interfaces", []) or []:
            option_ref = str(classified.get("interface_option_ref"))
            option = options.get(option_ref, {})
            derived = option.get("derived_physical_interface") or {}
            classification = str(classified.get("classification"))
            evidence_refs = sorted(
                {
                    str(ref)
                    for ref in [
                        *list(option.get("evidence_refs", []) or []),
                        option_ref,
                        str(classified.get("section_station_ref") or ""),
                    ]
                    if ref
                }
            )
            base = {
                "physical_object_scope_ref": scope_ref,
                "construction_region_ref": region_ref,
                "source_assignment_ref": str(assignment.get("id")),
                "source_interface_option_ref": option_ref,
                "source_derived_interface_ref": str(derived.get("id") or ""),
                "section_station_ref": str(classified.get("section_station_ref") or ""),
                "projected_plan_station_refs": list(map(str, classified.get("projected_plan_station_refs", []) or [])),
                "native_drawing_edge": bool(derived.get("native_drawing_edge", False)),
                "quantity_eligible": False,
                "evidence_refs": evidence_refs,
            }
            if classification == "landing_interface":
                seams.append(
                    {
                        "id": _stable_id(page_number, "internal_seam_hypothesis", global_assignment.get("id"), option_ref),
                        "record_type": "internal_seam_hypothesis",
                        "state": "proposed_pending_global_union",
                        "seam_role": "flight_landing",
                        "source_record_reclassification": "derived_physical_end_cap_to_internal_seam_hypothesis",
                        **base,
                    }
                )
            elif classification == "external_support_end":
                external.append(
                    {
                        "id": _stable_id(page_number, "external_boundary_hypothesis", global_assignment.get("id"), option_ref),
                        "record_type": "external_boundary_hypothesis",
                        "state": "proposed_pending_global_union",
                        "boundary_role": "support_end",
                        **base,
                    }
                )
    return seams, external


def _native_boundary_mm(
    boundary: Mapping[str, Any],
    section_gauge: Mapping[str, Any],
) -> list[list[float]]:
    scale = float(section_gauge.get("section_scale_points_per_mm") or 0.0)
    contact_x = float(section_gauge.get("flight_contact_coordinate_display") or 0.0)
    landing_top = float(section_gauge.get("landing_top_coordinate_display") or 0.0)
    if scale <= 0:
        raise ValueError("section gauge scale is unresolved")
    return [
        [(float(point[0]) - contact_x) / scale, (landing_top - float(point[1])) / scale]
        for point in boundary.get("ordered_boundary_display", []) or []
    ]


def _certified_section_contact_anchors(
    boundary: Mapping[str, Any],
    landing_reconstruction: Mapping[str, Any],
    folded_plan_topology: Mapping[str, Any] | None,
    *,
    page_number: int,
) -> list[dict[str, Any]]:
    """Publish a landing-bearing anchor only from independent section/plan evidence."""

    if not folded_plan_topology:
        return []
    gauge = landing_reconstruction.get("termination_convention_certificate") or {}
    footprint = landing_reconstruction.get("plan_footprint_certificate") or {}
    thickness_record = landing_reconstruction.get("section_thickness_certificate") or {}
    clear_length = float((footprint.get("clear_core_footprint_mm") or [0.0])[0])
    thickness = float(thickness_record.get("thickness_mm") or 0.0)
    profile = Polygon(_native_boundary_mm(boundary, gauge))
    landing_section = Polygon(
        [(0.0, -thickness), (clear_length, -thickness), (clear_length, 0.0), (0.0, 0.0)]
    )
    overlap_area = float(profile.intersection(landing_section).area)
    if not profile.is_valid or min(clear_length, thickness) <= 0:
        return []
    station_ref = str(gauge.get("flight_contact_section_station_ref") or "")
    evidence_refs = sorted(
        {
            str(boundary.get("id")),
            station_ref,
            str(folded_plan_topology.get("id")),
            *map(str, footprint.get("evidence_refs", []) or []),
            *map(str, thickness_record.get("evidence_refs", []) or []),
            *map(str, folded_plan_topology.get("evidence_refs", []) or []),
        }
        - {""}
    )
    authorization = None
    if overlap_area > 1e-6:
        authorization_id = _stable_id(
            page_number, "overlap_geometry_authorization_certificate", *evidence_refs
        )
        authorization = {
            "id": authorization_id,
            "record_type": "overlap_geometry_authorization_certificate",
            "state": "accepted",
            "authorized_seam_kind": "overlap_union",
            "native_section_overlap_area_mm2": overlap_area,
            "independent_of_candidate_mesh_intersection": True,
            "basis": "native_section_profile_intersection_with_certified_landing_section_and_folded_plan_topology",
            "evidence_refs": evidence_refs,
            "quantity_eligible": False,
        }
    return [
        {
            "kind": "certified_section_contact_station",
            "station_local_mm": 0.0,
            "source_ref": station_ref,
            "evidence_refs": evidence_refs,
            "landing_overlap_authorization": authorization,
        }
    ]


def _station_reclosed_boundary_mm(
    boundary: Mapping[str, Any],
    section_gauge: Mapping[str, Any],
    transform_hypothesis: Mapping[str, Any],
    landing_reconstruction: Mapping[str, Any],
    *,
    page_number: int,
) -> tuple[list[list[float]], dict[str, Any] | None]:
    """Extend an open waist chain to an independently certified interface station."""

    points = _native_boundary_mm(boundary, section_gauge)
    if transform_hypothesis.get("interface_anchor_kind") != "certified_section_contact_station":
        return points, None
    footprint = landing_reconstruction.get("plan_footprint_certificate") or {}
    thickness = float(
        (landing_reconstruction.get("section_thickness_certificate") or {}).get("thickness_mm")
        or 0.0
    )
    clear_length = float((footprint.get("clear_core_footprint_mm") or [0.0])[0])
    profile = Polygon(points)
    landing_section = Polygon(
        [(0.0, -thickness), (clear_length, -thickness), (clear_length, 0.0), (0.0, 0.0)]
    )
    if profile.intersection(landing_section).area > 1e-6:
        return points, None

    cap_matches = [
        item
        for item in boundary.get("assignment_specific_caps", []) or []
        if item.get("classification") == "landing_interface"
    ]
    if len(cap_matches) != 1:
        raise ValueError("station_reclosure_landing_interface_cap_unresolved")
    cap = cap_matches[0]
    cap_display_points = (
        cap.get("cap_endpoints_display") or cap.get("ordered_points_display") or []
    )
    if len(cap_display_points) < 2:
        raise ValueError("station_reclosure_landing_interface_endpoints_unresolved")
    cap_points = [
        [
            (float(point[0]) - float(section_gauge["flight_contact_coordinate_display"]))
            / float(section_gauge["section_scale_points_per_mm"]),
            (float(section_gauge["landing_top_coordinate_display"]) - float(point[1]))
            / float(section_gauge["section_scale_points_per_mm"]),
        ]
        for point in (cap_display_points[0], cap_display_points[-1])
    ]
    cap_indices = []
    for cap_point in cap_points:
        matches = [
            index for index, point in enumerate(points) if math.dist(point, cap_point) <= 1e-3
        ]
        if len(matches) != 1:
            raise ValueError("station_reclosure_cap_vertex_unresolved")
        cap_indices.append(matches[0])
    if len(set(cap_indices)) != 2:
        raise ValueError("station_reclosure_cap_vertices_not_distinct")
    top_index = min(cap_indices, key=lambda index: abs(points[index][1]))
    waist_index = max(cap_indices, key=lambda index: abs(points[index][1]))
    count = len(points)
    waist_neighbors = [points[(waist_index - 1) % count], points[(waist_index + 1) % count]]
    waist_neighbor = next(
        (
            point
            for point in waist_neighbors
            if math.dist(point, points[top_index]) > 1e-3
            and abs(point[0] - points[waist_index][0]) > 1e-3
        ),
        None,
    )
    top_station_candidates = [
        (index, point)
        for index, point in enumerate(points)
        if abs(point[0]) <= 1e-3 and abs(point[1]) <= 1e-3
    ]
    if waist_neighbor is None or len(top_station_candidates) != 1:
        raise ValueError("station_reclosure_native_chain_or_top_endpoint_unresolved")
    top_station_index, top_station = top_station_candidates[0]
    waist = points[waist_index]
    slope = (waist[1] - waist_neighbor[1]) / (waist[0] - waist_neighbor[0])
    extended_waist = [0.0, waist[1] + slope * (0.0 - waist[0])]
    if not math.isfinite(extended_waist[1]) or extended_waist[1] > -thickness + 1e-3:
        raise ValueError("station_reclosure_does_not_cover_certified_landing_thickness")

    remove = set(cap_indices)
    rebuilt = []
    for index, point in enumerate(points):
        if index in remove:
            continue
        rebuilt.append(list(point))
        if index == top_station_index:
            rebuilt.append(extended_waist)
    rebuilt_polygon = Polygon(rebuilt)
    if not rebuilt_polygon.is_valid or rebuilt_polygon.area <= 0:
        raise ValueError("station_reclosure_profile_invalid")
    evidence_refs = sorted(
        {
            str(boundary.get("id")),
            str(transform_hypothesis.get("source_interface_anchor_ref") or ""),
            str(cap.get("interface_option_ref") or ""),
            *map(str, boundary.get("evidence_refs", []) or []),
            *map(str, transform_hypothesis.get("evidence_refs", []) or []),
            *map(str, footprint.get("evidence_refs", []) or []),
        }
        - {""}
    )
    certificate = {
        "id": _stable_id(page_number, "station_reclosed_boundary_certificate", *evidence_refs),
        "record_type": "station_reclosed_boundary_certificate",
        "state": "resolved_search_hypothesis",
        "source_boundary_ref": str(boundary.get("id")),
        "certified_interface_station_local_mm": 0.0,
        "native_waist_endpoint_uv_mm": waist,
        "native_waist_continuation_point_uv_mm": waist_neighbor,
        "derived_waist_endpoint_uv_mm": extended_waist,
        "derived_cap_endpoints_uv_mm": [top_station, extended_waist],
        "landing_thickness_covered_mm": thickness,
        "derivation": "native_waist_line_extrapolated_to_independently_certified_interface_station",
        "quantity_eligible": False,
        "evidence_refs": evidence_refs,
    }
    return rebuilt, certificate


def _plan_bounded_boundary_mm(
    points: list[list[float]],
    boundary: Mapping[str, Any],
    folded_plan_topology: Mapping[str, Any] | None,
    *,
    page_number: int,
) -> tuple[list[list[float]], dict[str, Any] | None]:
    """Trim support/landing overrun only at native plan-certified end planes."""

    if not folded_plan_topology:
        return points, None
    body_length = float(folded_plan_topology.get("flight_body_length_mm") or 0.0)
    landing_length = float(folded_plan_topology.get("landing_length_mm") or 0.0)
    if min(body_length, landing_length) <= 0:
        raise ValueError("folded_plan_longitudinal_bounds_unresolved")
    polygon = Polygon(points)
    minimum_z, maximum_z = polygon.bounds[1], polygon.bounds[3]
    clipped = polygon.intersection(
        box(-body_length, minimum_z - 1.0, landing_length, maximum_z + 1.0)
    )
    if not isinstance(clipped, Polygon) or not clipped.is_valid or clipped.area <= 0:
        raise ValueError("plan_bounded_profile_clipping_invalid")
    removed_area = float(polygon.area - clipped.area)
    if removed_area <= 1e-6:
        return points, None
    evidence_refs = sorted(
        {
            str(boundary.get("id")),
            str(folded_plan_topology.get("id")),
            *map(str, boundary.get("evidence_refs", []) or []),
            *map(str, folded_plan_topology.get("evidence_refs", []) or []),
        }
    )
    certificate = {
        "id": _stable_id(page_number, "plan_bounded_profile_certificate", *evidence_refs),
        "record_type": "plan_bounded_profile_certificate",
        "state": "resolved_search_hypothesis",
        "source_boundary_ref": str(boundary.get("id")),
        "flight_body_bound_mm": body_length,
        "landing_bound_mm": landing_length,
        "removed_section_area_mm2": removed_area,
        "basis": "native_section_profile_intersected_with_native_folded_plan_longitudinal_bounds",
        "evidence_refs": evidence_refs,
        "quantity_eligible": False,
    }
    return [list(map(float, point)) for point in list(clipped.exterior.coords)[:-1]], certificate


def _transformed_interface_cap_line(
    boundary: Mapping[str, Any],
    section_gauge: Mapping[str, Any],
    transform_hypothesis: Mapping[str, Any],
) -> LineString:
    caps = [
        item
        for item in boundary.get("assignment_specific_caps", []) or []
        if item.get("classification") == "landing_interface"
    ]
    if len(caps) != 1:
        raise ValueError("temporary boundary requires one landing-interface cap")
    points = caps[0].get("cap_endpoints_display") or caps[0].get("ordered_points_display") or []
    if len(points) < 2:
        raise ValueError("landing-interface cap endpoints are unresolved")
    scale = float(section_gauge["section_scale_points_per_mm"])
    contact_x = float(section_gauge["flight_contact_coordinate_display"])
    landing_top = float(section_gauge["landing_top_coordinate_display"])
    transform = transform_hypothesis.get("local_to_world_transform") or {}
    origin = list(map(float, transform.get("origin_xyz_mm", []) or []))
    centroid = list(map(float, transform_hypothesis.get("profile_centroid_uv_mm", []) or []))
    x_sign = float(((transform.get("local_axes_xyz") or {}).get("x") or [1.0])[0])
    transformed = []
    for point in (points[0], points[-1]):
        x = (float(point[0]) - contact_x) / scale
        z = (landing_top - float(point[1])) / scale
        transformed.append([origin[0] + x_sign * (x - centroid[0]), z])
    return LineString(transformed)


def _interface_cap_semantics(
    boundary: Mapping[str, Any],
    transform_hypothesis: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the assignment's cap meaning through geometric materialization."""

    caps = [
        item
        for item in boundary.get("assignment_specific_caps", []) or []
        if item.get("classification") == "landing_interface"
    ]
    if len(caps) != 1:
        raise ValueError("temporary boundary requires one classified interface cap")
    cap = caps[0]
    authorization = transform_hypothesis.get("landing_overlap_authorization")
    overlap_authorized = bool(
        isinstance(authorization, Mapping)
        and authorization.get("state") == "accepted"
        and authorization.get("authorized_seam_kind") == "overlap_union"
        and authorization.get("independent_of_candidate_mesh_intersection") is True
    )
    station_reclosed = (
        transform_hypothesis.get("interface_anchor_kind")
        == "certified_section_contact_station"
    )
    return {
        "source_cap_classification": (
            "landing_bearing_profile_boundary"
            if overlap_authorized
            else "station_reclosed_interface_boundary"
            if station_reclosed
            else "interface_boundary"
        ),
        "classified_cap_preserved_as_search_evidence": True,
        "source_interface_option_ref": str(cap.get("interface_option_ref") or ""),
        "source_section_station_ref": str(cap.get("section_station_ref") or ""),
        "source_projected_plan_station_refs": list(
            map(str, cap.get("projected_plan_station_refs", []) or [])
        ),
        "required_contact_semantics": (
            "overlap_union" if overlap_authorized else "coincident_face"
        ),
        "overlap_authorization_certificate": authorization if overlap_authorized else None,
    }


def _landing_seam_polygon_hypotheses(
    boundary: Mapping[str, Any],
    band: Mapping[str, Any],
    section_gauge: Mapping[str, Any],
    clear_origin_mm: float,
    page_number: int,
    transform_hypothesis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scale = float(section_gauge["section_scale_points_per_mm"])
    contact_x = float(section_gauge["flight_contact_coordinate_display"])
    landing_top = float(section_gauge["landing_top_coordinate_display"])
    start_y, end_y = map(float, band.get("interval_mm", []) or [0.0, 0.0])
    transform = transform_hypothesis.get("local_to_world_transform") or {}
    axes = transform.get("local_axes_xyz") or {}
    origin = list(map(float, transform.get("origin_xyz_mm", []) or []))
    centroid = list(map(float, transform_hypothesis.get("profile_centroid_uv_mm", []) or []))
    if len(origin) != 3 or len(centroid) != 2:
        raise ValueError("profile-to-sweep transform is incomplete")
    x_sign = float((axes.get("x") or [1.0])[0])
    y_sign = float((axes.get("y") or [0.0, 1.0])[1])
    y0, y1 = sorted([origin[1], origin[1] + y_sign * (end_y - start_y)])
    records = []
    for cap in boundary.get("assignment_specific_caps", []) or []:
        if cap.get("classification") != "landing_interface":
            continue
        points = cap.get("cap_endpoints_display") or cap.get("ordered_points_display", []) or []
        if len(points) < 2:
            continue
        start, end = points[0], points[-1]
        left_local = [(float(start[0]) - contact_x) / scale, (landing_top - float(start[1])) / scale]
        right_local = [(float(end[0]) - contact_x) / scale, (landing_top - float(end[1])) / scale]
        left = [origin[0] + x_sign * (left_local[0] - centroid[0]), origin[2] + left_local[1] - centroid[1]]
        right = [origin[0] + x_sign * (right_local[0] - centroid[0]), origin[2] + right_local[1] - centroid[1]]
        identity = _stable_id(
            page_number,
            "landing_seam_polygon_hypothesis",
            boundary.get("id"),
            band.get("id"),
            cap.get("interface_option_ref"),
            transform_hypothesis.get("id"),
        )
        records.append(
            {
                "id": identity,
                "record_type": "landing_seam_polygon_hypothesis",
                "state": "resolved_search_hypothesis",
                "source_boundary_ref": boundary.get("id"),
                "source_interface_option_ref": cap.get("interface_option_ref"),
                "source_band_ref": band.get("id"),
                "source_transform_hypothesis_ref": transform_hypothesis.get("id"),
                "polygon_xyz_mm": [
                    [left[0], y0, left[1]],
                    [left[0], y1, left[1]],
                    [right[0], y1, right[1]],
                    [right[0], y0, right[1]],
                ],
                "evidence_refs": sorted(
                    {
                        str(boundary.get("id")),
                        str(band.get("id")),
                        str(transform_hypothesis.get("id")),
                        *map(str, cap.get("evidence_refs", []) or []),
                    }
                ),
                "quantity_eligible": False,
            }
        )
    return records


def _independent_projection_records(
    boundaries: list[Mapping[str, Any]],
    bands: list[Mapping[str, Any]],
    transform_hypotheses: list[Mapping[str, Any]],
    landing_reconstruction: Mapping[str, Any],
    folded_plan_topology: Mapping[str, Any] | None,
    *,
    page_number: int,
) -> list[dict[str, Any]]:
    """Build expected views exclusively from native/certified drawing evidence."""

    if not folded_plan_topology:
        raise ValueError("native_folded_plan_topology_unresolved")
    gauge = landing_reconstruction.get("termination_convention_certificate") or {}
    footprint = landing_reconstruction.get("plan_footprint_certificate") or {}
    thickness_record = landing_reconstruction.get("section_thickness_certificate") or {}
    clear_length, clear_width = map(
        float, footprint.get("clear_core_footprint_mm", []) or []
    )
    thickness = float(thickness_record.get("thickness_mm") or 0.0)
    if min(clear_length, clear_width, thickness) <= 0:
        raise ValueError("independent_projection_metric_basis_unresolved")
    station_reclosed = [
        _station_reclosed_boundary_mm(
            boundary,
            gauge,
            transform,
            landing_reconstruction,
            page_number=page_number,
        )
        for boundary, transform in zip(boundaries, transform_hypotheses)
    ]
    plan_bounded = [
        _plan_bounded_boundary_mm(
            points,
            boundary,
            folded_plan_topology,
            page_number=page_number,
        )
        for boundary, (points, _certificate) in zip(boundaries, station_reclosed)
    ]
    section_polygons = [points for points, _certificate in plan_bounded]
    section_polygons.append(
        [[0.0, -thickness], [clear_length, -thickness], [clear_length, 0.0], [0.0, 0.0]]
    )
    evidence_refs = sorted(
        {
            str(folded_plan_topology.get("id")),
            *(str(boundary.get("id")) for boundary in boundaries),
            *(str(band.get("id")) for band in bands),
            *(
                str(certificate.get("id"))
                for records in (station_reclosed, plan_bounded)
                for _points, certificate in records
                if certificate
            ),
            *map(str, folded_plan_topology.get("evidence_refs", []) or []),
            *map(str, footprint.get("evidence_refs", []) or []),
            *map(str, thickness_record.get("evidence_refs", []) or []),
            *map(str, gauge.get("evidence_refs", []) or []),
        }
    )
    projection_key = _stable_id(
        page_number,
        "independent_native_projection_basis",
        *(str(boundary.get("id")) for boundary in boundaries),
        *(str(band.get("id")) for band in bands),
        *(
            str(certificate.get("id"))
            for records in (station_reclosed, plan_bounded)
            for _points, certificate in records
            if certificate
        ),
    )
    metric_frame = footprint.get("metric_frame") or {}
    drawing_resolution_mm = float(
        folded_plan_topology.get("native_vertex_tolerance_points") or 0.0
    ) / float(metric_frame.get("scale_points_per_mm") or 1.0)
    tolerance_mm = max(
        2.0,
        drawing_resolution_mm,
        float(folded_plan_topology.get("native_endpoint_resolution_mm") or 0.0),
    )
    return [
        {
            "id": _stable_id(page_number, "native_projection_evidence", projection_key, "plan"),
            "record_type": "native_projection_evidence",
            "state": "resolved",
            "source_geometry_kind": "native_vector_and_certified_dimensions",
            "source_geometry_detail": "native_plan_outline_with_certified_metric_frame",
            "derived_from_generated_geometry": False,
            "alternative_independent_of_transform": True,
            "projection_role": "plan",
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "u_axis_xyz": [1.0, 0.0, 0.0],
            "v_axis_xyz": [0.0, 1.0, 0.0],
            "polygons_uv_mm": [folded_plan_topology["polygon_uv_mm"]],
            "tolerance_mm": tolerance_mm,
            "evidence_refs": evidence_refs,
            "quantity_eligible": False,
        },
        {
            "id": _stable_id(page_number, "native_projection_evidence", projection_key, "section"),
            "record_type": "native_projection_evidence",
            "state": "resolved",
            "source_geometry_kind": "native_vector_and_certified_dimensions",
            "source_geometry_detail": "native_section_boundaries_with_certified_landing_section",
            "derived_from_generated_geometry": False,
            "alternative_independent_of_transform": True,
            "projection_role": "section",
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "u_axis_xyz": [1.0, 0.0, 0.0],
            "v_axis_xyz": [0.0, 0.0, 1.0],
            "polygons_uv_mm": section_polygons,
            "tolerance_mm": tolerance_mm,
            "evidence_refs": evidence_refs,
            "quantity_eligible": False,
        },
    ]


def _materialize_assignment_band_hypothesis(
    *,
    page_number: int,
    alternative_id: str,
    search_scope: Mapping[str, Any],
    boundaries: list[Mapping[str, Any]],
    bands: list[Mapping[str, Any]],
    transform_hypotheses: list[Mapping[str, Any]],
    landing_reconstruction: Mapping[str, Any],
    supplied_projections: list[Mapping[str, Any]],
    folded_plan_topology: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    section_gauge = landing_reconstruction.get("termination_convention_certificate") or {}
    landing_footprint = landing_reconstruction.get("plan_footprint_certificate") or {}
    landing_thickness = landing_reconstruction.get("section_thickness_certificate") or {}
    clear_length, clear_width = map(
        float, landing_footprint.get("clear_core_footprint_mm", []) or []
    )
    thickness = float(landing_thickness.get("thickness_mm") or 0.0)
    if (
        len(boundaries) != len(bands)
        or len(boundaries) != len(transform_hypotheses)
        or not boundaries
        or min(clear_length, clear_width, thickness) <= 0
    ):
        raise ValueError("bounded sweep or landing metric evidence is incomplete")
    clear_origin = min(float(band["interval_mm"][0]) for band in bands)
    sweep_hypotheses = []
    world_profiles: list[tuple[str, Polygon, LineString, float, float, float]] = []
    seam_polygon_hypotheses = []
    for ordinal, (boundary, band, transform_hypothesis) in enumerate(
        zip(boundaries, bands, transform_hypotheses), start=1
    ):
        profile_points, station_reclosure = _station_reclosed_boundary_mm(
            boundary,
            section_gauge,
            transform_hypothesis,
            landing_reconstruction,
            page_number=page_number,
        )
        profile_points, plan_bound = _plan_bounded_boundary_mm(
            profile_points,
            boundary,
            folded_plan_topology,
            page_number=page_number,
        )
        profile = Polygon(profile_points)
        if not profile.is_valid or profile.area <= 0:
            raise ValueError("temporary native section boundary is invalid")
        certified_profile_centroid = list(
            map(float, transform_hypothesis.get("profile_centroid_uv_mm", []) or [])
        )
        if len(certified_profile_centroid) != 2:
            raise ValueError("profile transform lacks its independently derived centroid")
        profile_origin_x, profile_origin_z = certified_profile_centroid
        transform = transform_hypothesis.get("local_to_world_transform") or {}
        axes = transform.get("local_axes_xyz") or {}
        z_sign = float((axes.get("z") or [0.0, 0.0, 1.0])[2])
        local_profile_points = [
            [point[0] - profile_origin_x, z_sign * (point[1] - profile_origin_z)]
            for point in profile_points
        ]
        start_y, end_y = map(float, band.get("interval_mm", []) or [0.0, 0.0])
        depth = end_y - start_y
        if not math.isclose(depth, float(band.get("sweep_width_mm") or 0.0), abs_tol=1e-6):
            raise ValueError("plan band interval and sweep width disagree")
        region_id = _stable_id(page_number, "construction_region", alternative_id, boundary.get("id"))
        transform_id = str(transform.get("id") or "")
        origin = list(map(float, transform.get("origin_xyz_mm", []) or []))
        if len(origin) != 3 or not transform_id:
            raise ValueError("profile-to-sweep transform hypothesis is unresolved")
        evidence_refs = sorted(
            {
                str(boundary.get("id")),
                str(band.get("id")),
                *map(str, boundary.get("evidence_refs", []) or []),
                *map(str, band.get("evidence_refs", []) or []),
                *map(str, section_gauge.get("evidence_refs", []) or []),
                *map(str, transform_hypothesis.get("evidence_refs", []) or []),
                *(
                    [str(station_reclosure.get("id"))]
                    if station_reclosure
                    else []
                ),
                *([str(plan_bound.get("id"))] if plan_bound else []),
            }
        )
        sweep_hypotheses.append(
            {
                "id": _stable_id(page_number, "bounded_profile_sweep_hypothesis", alternative_id, ordinal),
                "record_type": "bounded_profile_sweep_hypothesis",
                "physical_object_scope_ref": search_scope["id"],
                "construction_region_id": region_id,
                "construction_role": "flight_construction_region",
                "state": "resolved",
                "profile_boundary": {
                    "id": boundary["id"],
                    "record_type": "bounded_profile_boundary",
                    "state": "resolved",
                    "ordered_points_uv_mm": local_profile_points,
                    "classified_caps": [],
                    "source_record_type": "temporary_closed_boundary_hypothesis",
                    "station_reclosed_boundary_certificate": station_reclosure,
                    "plan_bounded_profile_certificate": plan_bound,
                    "evidence_refs": evidence_refs,
                },
                "sweep_bound": {
                    "id": str(band.get("id")),
                    "record_type": "bounded_sweep",
                    "state": "resolved",
                    "depth_mm": depth,
                    "interval_mm": [start_y, end_y],
                    "evidence_refs": list(map(str, band.get("evidence_refs", []) or [])),
                },
                "local_to_world_transform": {**transform, "evidence_refs": evidence_refs},
                "evidence_refs": evidence_refs,
                "quantity_eligible": False,
            }
        )
        x_sign = float((axes.get("x") or [1.0])[0])
        y_sign = float((axes.get("y") or [0.0, 1.0])[1])
        world_profile = Polygon(
            [
                [
                    origin[0] + x_sign * (point[0] - profile_origin_x),
                    origin[2] + point[1] - profile_origin_z,
                ]
                for point in profile_points
            ]
        )
        y0, y1 = sorted([origin[1], origin[1] + y_sign * depth])
        world_profiles.append(
            (
                region_id,
                world_profile,
                _transformed_interface_cap_line(boundary, section_gauge, transform_hypothesis),
                y0,
                y1,
                depth,
            )
        )
        seam_polygon_hypotheses.extend(
            _landing_seam_polygon_hypotheses(
                boundary,
                band,
                section_gauge,
                clear_origin,
                page_number,
                transform_hypothesis,
            )
        )

    landing_region_id = _stable_id(page_number, "construction_region", alternative_id, "core")
    landing_evidence = sorted(
        {
            *map(str, landing_footprint.get("evidence_refs", []) or []),
            *map(str, landing_thickness.get("evidence_refs", []) or []),
            *map(str, section_gauge.get("evidence_refs", []) or []),
        }
    )
    sweep_hypotheses.append(
        {
            "id": _stable_id(page_number, "bounded_profile_sweep_hypothesis", alternative_id, "core"),
            "record_type": "bounded_profile_sweep_hypothesis",
            "physical_object_scope_ref": search_scope["id"],
            "construction_region_id": landing_region_id,
            "construction_role": "core_construction_region",
            "state": "resolved",
            "profile_boundary": {
                "id": _stable_id(page_number, "bounded_profile_boundary", alternative_id, "core"),
                "record_type": "bounded_profile_boundary",
                "state": "resolved",
                "ordered_points_uv_mm": [[0.0, 0.0], [clear_length, 0.0], [clear_length, thickness], [0.0, thickness]],
                "classified_caps": [],
                "evidence_refs": landing_evidence,
            },
            "sweep_bound": {
                "id": _stable_id(page_number, "bounded_sweep", alternative_id, "core"),
                "record_type": "bounded_sweep",
                "state": "resolved",
                "depth_mm": clear_width,
                "evidence_refs": landing_evidence,
            },
            "local_to_world_transform": {
                "id": _stable_id(page_number, "profile_to_band_transform", alternative_id, "core"),
                "record_type": "local_to_world_transform",
                "state": "resolved",
                "origin_xyz_mm": [0.0, 0.0, -thickness],
                "local_axes_xyz": {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
                "transform_role": "certified_core_plan_and_section_placement",
                "accepted_physical_placement": False,
                "evidence_refs": landing_evidence,
                "quantity_eligible": False,
            },
            "evidence_refs": landing_evidence,
            "quantity_eligible": False,
        }
    )

    landing_section = Polygon([(0.0, -thickness), (clear_length, -thickness), (clear_length, 0.0), (0.0, 0.0)])
    seams = []
    overlap_total = 0.0
    for ordinal, (region_id, profile, interface_cap_line, y0, y1, depth) in enumerate(
        world_profiles, start=1
    ):
        seam_semantics = _interface_cap_semantics(
            boundaries[ordinal - 1], transform_hypotheses[ordinal - 1]
        )
        intersection = profile.intersection(landing_section)
        overlap_area = float(intersection.area)
        seam_evidence = sorted(
            {
                str(boundaries[ordinal - 1]["id"]),
                str(bands[ordinal - 1]["id"]),
                str(transform_hypotheses[ordinal - 1]["id"]),
                *landing_evidence,
            }
        )
        if overlap_area <= 1e-6:
            if (
                transform_hypotheses[ordinal - 1].get("interface_anchor_kind")
                == "certified_section_contact_station"
            ):
                linework = profile.boundary.intersection(landing_section.boundary)
            else:
                linework = (
                    interface_cap_line
                    .intersection(profile.boundary)
                    .intersection(landing_section.boundary)
                )
            if isinstance(linework, LineString):
                lines = [linework]
            elif isinstance(linework, MultiLineString):
                lines = list(linework.geoms)
            elif isinstance(linework, GeometryCollection):
                lines = [item for item in linework.geoms if isinstance(item, LineString)]
            else:
                lines = []
            seam_ordinal = 0
            for line in lines:
                coordinates = list(line.coords)
                for start, end in zip(coordinates, coordinates[1:]):
                    if math.dist(start, end) <= 1e-6:
                        continue
                    seam_ordinal += 1
                    seams.append(
                        {
                            "id": _stable_id(
                                page_number,
                                "internal_seam_hypothesis",
                                alternative_id,
                                ordinal,
                                seam_ordinal,
                            ),
                            "record_type": "internal_seam_hypothesis",
                            "state": "resolved",
                            "seam_kind": "coincident_face",
                            "construction_region_refs": [region_id, landing_region_id],
                            "polygon_xyz_mm": [
                                [float(start[0]), y0, float(start[1])],
                                [float(start[0]), y1, float(start[1])],
                                [float(end[0]), y1, float(end[1])],
                                [float(end[0]), y0, float(end[1])],
                            ],
                            "polygon_source_kind": "certified_boundary_intersection",
                            "calculation_basis": "transformed_native_profile_boundary_intersection_with_certified_core_boundary",
                            **seam_semantics,
                            "evidence_refs": seam_evidence,
                            "quantity_eligible": False,
                        }
                    )
            continue
        overlap_volume = overlap_area * depth
        overlap_total += overlap_volume
        seams.append(
            {
                "id": _stable_id(page_number, "internal_seam_hypothesis", alternative_id, ordinal),
                "record_type": "internal_seam_hypothesis",
                "state": "resolved",
                "seam_kind": "overlap_union",
                "construction_region_refs": [region_id, landing_region_id],
                "overlap_volume_mm3": overlap_volume,
                "calculation_basis": "native_section_profile_intersection_times_certified_plan_band",
                **seam_semantics,
                "evidence_refs": seam_evidence,
                "quantity_eligible": False,
            }
        )

    flight_volume_terms = [
        float(profile.area) * depth
        for _, profile, _, _, _, depth in world_profiles
    ]
    landing_volume = clear_length * clear_width * thickness
    union_volume = sum(flight_volume_terms) + landing_volume - overlap_total
    analytic = {
        "id": _stable_id(page_number, "analytic_union_certificate", alternative_id),
        "record_type": "analytic_union_certificate",
        "state": "resolved",
        "value_mm3": union_volume,
        "basis": "certified_profile_area_times_sweeps_plus_core_minus_analytic_section_overlaps",
        "independent_of_generated_mesh": True,
        "calculation_terms": [
            *[
                {"kind": "profile_sweep", "value_mm3": value}
                for value in flight_volume_terms
            ],
            {"kind": "core_sweep", "value_mm3": landing_volume},
            {"kind": "overlap_deduction", "value_mm3": overlap_total},
        ],
        "evidence_refs": sorted({*(str(item["id"]) for item in boundaries), *(str(item["id"]) for item in bands), *landing_evidence}),
        "quantity_eligible": False,
    }
    materialized = materialize_bounded_profile_sweep_union(
        search_scope,
        sweep_hypotheses,
        seams,
        supplied_projections,
        analytic,
        alternative_id=alternative_id,
    )
    materialized["profile_sweep_hypotheses"] = sweep_hypotheses
    materialized["landing_seam_polygon_hypotheses"] = seam_polygon_hypotheses
    return materialized, seam_polygon_hypotheses


def assemble_same_object_construction(
    open_structural_boundary_assembly: Mapping[str, Any],
    flight_interface_closure: Mapping[str, Any],
    banded_plan_sweep_evidence: Mapping[str, Any],
    landing_component_reconstruction: Mapping[str, Any],
    *,
    page_number: int,
    native_topology: Mapping[str, Any] | None = None,
    title_segmentation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate drawing evidence, then delegate completion to the enumerator."""

    landing = landing_component_reconstruction.get("landing_construction_region") or {}
    scope = landing_component_reconstruction.get("physical_object_scope") or {}
    bands = list((banded_plan_sweep_evidence.get("plan_band_certificate") or {}).get("bands", []) or [])
    open_scopes = list(open_structural_boundary_assembly.get("scope_results", []) or [])
    closure_scopes = list(flight_interface_closure.get("scope_results", []) or [])
    if (
        landing.get("record_type") != "construction_region"
        or scope.get("record_type") != "physical_object_scope"
        or not bands
        or not open_scopes
        or not closure_scopes
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "layer": "same_object_constructive_assembly",
            "page": page_number,
            "status": "insufficient_constraints",
            "reason_code": "same_object_scope_regions_or_plan_bands_unresolved",
            "physical_object_scope": scope or None,
            "construction_regions": [],
            "internal_seam_hypotheses": [],
            "external_boundary_hypotheses": [],
            "separate_object_interface_hypotheses": [],
            "assembly_alternatives": [],
            "step4_same_object_union": None,
            "quantity_eligible": False,
        }

    scope_ref = str(scope.get("id"))
    proposals = [
        {**proposal, "source_scope_ref": str(open_scope.get("scope_ref"))}
        for open_scope in open_scopes
        for proposal in open_scope.get("flight_boundary_proposals", []) or []
    ]
    swept_regions = []
    region_by_proposal = {}
    for proposal in sorted(proposals, key=lambda item: str(item.get("id"))):
        region_ref = _stable_id(
            page_number,
            "construction_region_hypothesis",
            proposal.get("id"),
            scope_ref,
        )
        region_by_proposal[str(proposal.get("id"))] = region_ref
        swept_regions.append(
            {
                "id": region_ref,
                "record_type": "construction_region",
                "physical_object_scope_ref": scope_ref,
                "construction_role": "bounded_profile_sweep",
                "state": "incomplete_open_boundary",
                "section_boundary_proposal_ref": str(proposal.get("id")),
                "source_scope_ref": str(proposal.get("source_scope_ref")),
                "placement_candidate_refs": [str(item.get("id")) for item in bands],
                "independent_landing_closure_required": False,
                "final_union_required": True,
                "quantity_eligible": False,
                "evidence_refs": sorted(
                    set(map(str, proposal.get("evidence_refs", []) or []))
                ),
            }
        )

    global_assignments = [
        assignment
        for closure_scope in closure_scopes
        for assignment in closure_scope.get("global_assignment_candidates", []) or []
    ]
    placement_alternatives = []
    materialized_search_hypotheses = []
    materialization_failures = []
    search_regions = []
    resolved_search_seams = []
    native_projection_records = []
    analytic_union_certificates = []
    profile_to_sweep_transform_by_id = {}
    assignment_landing_seam_polygon_by_id = {}
    all_seams = []
    all_external = []
    temporary_boundaries = list(
        flight_interface_closure.get("temporary_closed_boundary_hypotheses", []) or []
    )
    temporary_by_assignment = {
        (str(item.get("source_global_assignment_ref")), str(item.get("source_proposal_ref"))): item
        for item in temporary_boundaries
    }
    search_scope = {
        **scope,
        "state": "resolved_search_hypothesis",
        "source_scope_state": scope.get("state"),
        "resolved_for_search_only": True,
        "accepted_physical_object_scope": False,
        "quantity_eligible": False,
    }
    folded_plan_topology = _native_folded_plan_topology_certificate(
        native_topology or {},
        title_segmentation or {},
        landing_component_reconstruction,
        bands,
        page_number=page_number,
    )
    plan_direction_evidence = derive_plan_direction_evidence(
        native_topology or {}, title_segmentation or {}, folded_plan_topology, bands
    )
    for assignment in global_assignments:
        seams, external = _classified_assignment_boundaries(
            page_number, scope_ref, region_by_proposal, assignment
        )
        all_seams.extend(seams)
        all_external.extend(external)
        proposal_refs = sorted(
            {
                str(item.get("proposal_ref"))
                for item in assignment.get("assignments", []) or []
                if str(item.get("proposal_ref")) in region_by_proposal
            }
        )
        band_mappings = (
            itertools.permutations(bands, len(proposal_refs))
            if proposal_refs and len(proposal_refs) <= len(bands)
            else [()]
        )
        for mapping_ordinal, mapped_bands in enumerate(band_mappings, start=1):
            mapped_boundaries = [
                temporary_by_assignment.get(
                    (str(assignment.get("id")), proposal_ref)
                )
                for proposal_ref in proposal_refs
            ]
            if any(item is None for item in mapped_boundaries):
                alternative_id = _stable_id(
                    page_number,
                    "constructive_placement_alternative",
                    assignment.get("id"),
                    mapping_ordinal,
                    *(str(item.get("id")) for item in mapped_bands),
                )
                placement_alternatives.append(
                    {
                        "id": alternative_id,
                        "record_type": "constructive_placement_alternative",
                        "physical_object_scope_ref": scope_ref,
                        "source_assignment_ref": str(assignment.get("id")),
                        "construction_region_refs": [
                            *(region_by_proposal[ref] for ref in proposal_refs),
                            str(landing.get("id")),
                        ],
                        "internal_seam_refs": [item["id"] for item in seams],
                        "supplied_projection_refs": [],
                        "separate_object_interface_refs": [],
                        "evidence_refs": [str(assignment.get("id"))],
                    }
                )
                materialization_failures.append(
                    {
                        "alternative_ref": alternative_id,
                        "reason": "temporary_assignment_boundary_unresolved",
                    }
                )
                continue
            section_gauge = landing_component_reconstruction.get("termination_convention_certificate") or {}
            clear_sweep_origin = min(float(item["interval_mm"][0]) for item in mapped_bands)
            for reflection_sign in (-1, 1):
                transform_choices = [
                    enumerate_profile_to_sweep_transforms(
                        boundary,
                        band,
                        section_gauge,
                        clear_sweep_origin_mm=clear_sweep_origin,
                        shared_reflection_signs=(reflection_sign,),
                        additional_interface_anchors=_certified_section_contact_anchors(
                            boundary,
                            landing_component_reconstruction,
                            folded_plan_topology,
                            page_number=page_number,
                        ),
                    )
                    for boundary, band in zip(mapped_boundaries, mapped_bands)
                ]
                for selected_transforms in itertools.product(*transform_choices):
                    alternative_id = _stable_id(
                        page_number,
                        "constructive_placement_alternative",
                        assignment.get("id"),
                        mapping_ordinal,
                        reflection_sign,
                        *(str(item.get("id")) for item in selected_transforms),
                    )
                    region_placements = [
                        {
                            "construction_region_ref": region_by_proposal[proposal_ref],
                            "placement_evidence_ref": str(transform.get("id")),
                            "sweep_width_mm": float(band.get("sweep_width_mm") or 0.0),
                            "state": "resolved_search_hypothesis",
                        }
                        for proposal_ref, band, transform in zip(
                            proposal_refs, mapped_bands, selected_transforms
                        )
                    ]
                    region_placements.append(
                        {
                            "construction_region_ref": str(landing.get("id")),
                            "placement_evidence_ref": str(
                                (landing_component_reconstruction.get("plan_footprint_certificate") or {}).get("title_scope_ref")
                                or ""
                            ),
                            "state": "resolved_search_hypothesis",
                        }
                    )
                    placement_alternatives.append(
                        {
                            "id": alternative_id,
                            "record_type": "constructive_placement_alternative",
                            "physical_object_scope_ref": scope_ref,
                            "source_assignment_ref": str(assignment.get("id")),
                            "construction_region_placements": region_placements,
                            "internal_seam_refs": [item["id"] for item in seams],
                            "supplied_projection_refs": [],
                            "separate_object_interface_refs": [],
                            "evidence_refs": sorted(
                                {
                                    str(assignment.get("id")),
                                    *(str(item.get("id")) for item in mapped_bands),
                                    *(str(item.get("id")) for item in selected_transforms),
                                }
                            ),
                        }
                    )
                    for transform in selected_transforms:
                        profile_to_sweep_transform_by_id[str(transform["id"])] = transform
                    try:
                        supplied_projections = _independent_projection_records(
                            list(mapped_boundaries),
                            list(mapped_bands),
                            list(selected_transforms),
                            landing_component_reconstruction,
                            folded_plan_topology,
                            page_number=page_number,
                        )
                        for projection in supplied_projections:
                            if projection["projection_role"] == "plan":
                                projection["required_direction_evidence"] = plan_direction_evidence
                                projection["evidence_refs"] = sorted(set(
                                    projection["evidence_refs"] + plan_direction_evidence["evidence_refs"]
                                ))
                        materialized, seam_polygons = _materialize_assignment_band_hypothesis(
                            page_number=page_number,
                            alternative_id=alternative_id,
                            search_scope=search_scope,
                            boundaries=list(mapped_boundaries),
                            bands=list(mapped_bands),
                            transform_hypotheses=list(selected_transforms),
                            landing_reconstruction=landing_component_reconstruction,
                            supplied_projections=supplied_projections,
                            folded_plan_topology=folded_plan_topology,
                        )
                    except (ValueError, TypeError, KeyError) as error:
                        materialization_failures.append(
                            {"alternative_ref": alternative_id, "reason": str(error)}
                        )
                        continue
                    materialized["placement_alternative"]["source_assignment_ref"] = str(
                        assignment.get("id")
                    )
                    materialized_search_hypotheses.append(materialized)
                    search_regions.extend(materialized["construction_regions"])
                    resolved_search_seams.extend(materialized["internal_seams"])
                    native_projection_records.extend(materialized["supplied_projections"])
                    analytic_union_certificates.append(materialized["analytic_union_certificate"])
                    for polygon in seam_polygons:
                        assignment_landing_seam_polygon_by_id[str(polygon["id"])] = polygon

    region_records = [*swept_regions, landing]
    separate_interfaces = list(landing_component_reconstruction.get("separate_object_interface_hypotheses", []) or [])
    native_projection_records = sorted(
        {
            str(item.get("id")): item for item in native_projection_records
        }.values(),
        key=lambda item: str(item.get("id")),
    )
    materialized_alternatives = [
        item["placement_alternative"] for item in materialized_search_hypotheses
    ]
    enumeration = enumerate_constructive_union_hypotheses(
        [search_scope],
        search_regions if materialized_alternatives else region_records,
        materialized_alternatives if materialized_alternatives else placement_alternatives,
        resolved_search_seams if materialized_alternatives else all_seams,
        native_projection_records,
    )
    sweep_by_region = {
        str(sweep["construction_region_id"]): sweep
        for hypothesis in materialized_search_hypotheses
        for sweep in hypothesis["profile_sweep_hypotheses"]
    }
    boundary_by_id = {str(boundary["id"]): boundary for boundary in temporary_boundaries}
    directed_bindings = []
    for replay in enumeration["kernel_replays"]:
        kernel = replay.get("kernel_result") or replay.get("residual_diagnostics") or {}
        for view in kernel.get("supplied_view_reprojections", []):
            for path in view.get("directed_surface_path_reprojections", []):
                for run in path.get("band_runs", []):
                    owners = run.get("construction_region_refs", [])
                    sources = [sweep_by_region[ref]["profile_boundary"]["id"] for ref in owners if ref in sweep_by_region]
                    directed_bindings.append({
                        "constructive_union_hypothesis_ref": replay["constructive_union_hypothesis_ref"],
                        "native_direction_path_ref": path["path_ref"],
                        "band_ref": run["band_ref"],
                        "construction_region_refs": owners,
                        "temporary_boundary_refs": sources,
                        "section_profile_proposal_refs": sorted({boundary_by_id[ref]["source_proposal_ref"] for ref in sources if ref in boundary_by_id}),
                        "measured_ascent_mm": run["total_rise_mm"],
                        "direction_status": run["status"],
                        "assignment_status": replay["status"],
                        "one_to_one_region_binding": len(owners) == 1,
                        "quantity_eligible": False,
                    })
    evidence_refs = sorted(
        {
            str(ref)
            for item in [scope, *region_records, *all_seams, *all_external, *separate_interfaces]
            for ref in item.get("evidence_refs", []) or []
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "same_object_constructive_assembly",
        "page": page_number,
        "status": enumeration["status"],
        "reason_code": enumeration["reason_code"],
        "physical_object_scope": scope,
        "search_physical_object_scope": search_scope,
        "construction_regions": region_records,
        "temporary_closed_boundary_hypotheses": temporary_boundaries,
        "folded_plan_topology_certificate": folded_plan_topology,
        "plan_direction_evidence": plan_direction_evidence,
        "directed_profile_band_bindings": directed_bindings,
        "materialized_search_hypotheses": materialized_search_hypotheses,
        "materialization_failures": materialization_failures,
        "profile_to_sweep_transform_alternatives": sorted(
            profile_to_sweep_transform_by_id.values(), key=lambda item: str(item["id"])
        ),
        "profile_to_band_transform_alternatives": sorted(
            profile_to_sweep_transform_by_id.values(), key=lambda item: str(item["id"])
        ),
        "assignment_specific_landing_seam_polygons": sorted(
            assignment_landing_seam_polygon_by_id.values(), key=lambda item: str(item["id"])
        ),
        "native_projection_evidence": native_projection_records,
        "analytic_union_certificates": analytic_union_certificates,
        "internal_seam_hypotheses": all_seams,
        "external_boundary_hypotheses": all_external,
        "separate_object_interface_hypotheses": separate_interfaces,
        "placement_alternatives": placement_alternatives,
        "assembly_alternatives": enumeration["evaluations"],
        "complete_union_hypotheses": enumeration["complete_union_hypotheses"],
        "kernel_replays": enumeration["kernel_replays"],
        "invariant_union_equivalence_certificate": enumeration[
            "invariant_union_equivalence_certificate"
        ],
        "placement_unresolved_union_equivalence_class": (
            enumeration["invariant_union_equivalence_certificate"]
            if enumeration["status"] == "accepted_invariant_equivalence_class"
            else None
        ),
        "invariant_concrete_volume_candidate": enumeration[
            "invariant_union_volume_candidate"
        ],
        "step4_same_object_union": enumeration["survivors"][0]["kernel_result"] if len(enumeration["survivors"]) == 1 else None,
        "quantity_eligible": enumeration["quantity_eligible"],
        "summary": {
            "local_assignment_count": enumeration["summary"]["assignment_count"],
            "placement_alternative_count": enumeration["summary"]["placement_alternative_count"],
            "construction_region_count": len(region_records),
            "temporary_closed_boundary_hypothesis_count": len(temporary_boundaries),
            "temporary_region_materialization_count": len(search_regions),
            "materialized_union_hypothesis_count": enumeration["summary"]["materialized_union_hypothesis_count"],
            "pre_kernel_rejection_count": enumeration["summary"]["pre_kernel_rejection_count"],
            "hard_pre_kernel_rejection_count": enumeration["summary"]["hard_pre_kernel_rejection_count"],
            "certified_elimination_count": enumeration["summary"]["certified_elimination_count"],
            "unresolved_live_alternative_count": enumeration["summary"]["unresolved_live_alternative_count"],
            "kernel_replay_count": enumeration["summary"]["kernel_replay_count"],
            "step4_same_object_survivor_count": enumeration["summary"]["survivor_count"],
            "invariant_equivalence_member_count": len(
                (
                    enumeration.get("invariant_union_equivalence_certificate")
                    or {}
                ).get("member_hypothesis_refs", [])
                or []
            ),
            "transition": enumeration["summary"]["transition"],
        },
        "evidence_refs": evidence_refs,
        "contract": {
            "physical_object_scope_distinct_from_construction_regions": True,
            "construction_regions_are_not_additive_components": True,
            "flight_landing_joins_are_internal_seams": True,
            "internal_coincident_faces_removed_by_union": True,
            "final_union_volume_only": True,
            "analytic_union_mesh_agreement_required": True,
            "plan_and_section_reprojection_required": True,
            "beam_is_separate_without_monolithic_identity": bool(separate_interfaces),
            "same_object_step4_union_invoked": enumeration["summary"]["kernel_replay_count"] > 0,
            "temporary_boundaries_are_search_inputs_only": True,
            "closed_profiles_remain_unique_acceptance_only": True,
            "expected_projections_come_from_native_or_certified_drawing_evidence": True,
            "generated_mesh_projection_cannot_supply_expected_projection": True,
            "analytic_union_is_independent_of_generated_mesh": True,
            "unevaluated_hypotheses_are_not_ambiguity": True,
            "unreplayed_live_alternatives_block_unique_acceptance": True,
            "construction_region_seam_graph_connected_before_step4": True,
            "disconnected_complete_seam_graph_is_a_hard_contradiction": True,
            "assigned_interface_caps_require_coincident_faces": True,
            "overlap_union_requires_independent_authorization": True,
            "equal_volume_alone_is_not_invariant_equivalence": True,
            "absolute_section_viewing_direction_may_remain_unresolved": True,
            "directed_native_plan_evidence_required_for_band_assignment": True,
            "quantity_eligible": enumeration["quantity_eligible"],
            "schedule_values_used": False,
        },
    }
