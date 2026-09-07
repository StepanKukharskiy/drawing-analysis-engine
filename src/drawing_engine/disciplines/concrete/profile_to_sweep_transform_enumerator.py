"""Enumerate evidence-bounded placements of a profile on a finite sweep.

The profile remains a search hypothesis.  A placement maps its classified
internal-interface cap to the target interface plane, maps the support cap to
the opposite longitudinal side, and maps the local sweep endpoints onto the
two certified interval endpoints.  Longitudinal sign and shared-coordinate
reflection remain explicit alternatives for downstream seam/reprojection
falsification.
"""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

from shapely.geometry import Polygon


SCHEMA_VERSION = "0.1.0"


def _stable_id(kind: str, *parts: Any) -> str:
    encoded = "\0".join((kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _cap(boundary: Mapping[str, Any], classification: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in boundary.get("assignment_specific_caps", []) or []
        if item.get("classification") == classification
    ]
    if len(matches) != 1:
        raise ValueError(f"profile requires exactly one {classification} cap")
    return matches[0]


def _cap_endpoints(cap: Mapping[str, Any]) -> list[list[float]]:
    points = cap.get("cap_endpoints_display") or cap.get("ordered_points_display") or []
    if len(points) < 2:
        raise ValueError("classified cap lacks two endpoints")
    return [list(map(float, points[0])), list(map(float, points[-1]))]


def _metric_point(point: Sequence[float], gauge: Mapping[str, Any]) -> list[float]:
    scale = float(gauge.get("section_scale_points_per_mm") or 0.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("section metric scale is unresolved")
    return [
        (float(point[0]) - float(gauge.get("flight_contact_coordinate_display") or 0.0)) / scale,
        (float(gauge.get("landing_top_coordinate_display") or 0.0) - float(point[1])) / scale,
    ]


def enumerate_profile_to_sweep_transforms(
    boundary: Mapping[str, Any],
    sweep: Mapping[str, Any],
    section_gauge: Mapping[str, Any],
    *,
    clear_sweep_origin_mm: float,
    longitudinal_signs: Sequence[int] = (-1, 1),
    shared_reflection_signs: Sequence[int] = (-1, 1),
    additional_interface_anchors: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return every certified cap-to-bounded-interval transform alternative."""

    if boundary.get("record_type") != "temporary_closed_boundary_hypothesis":
        raise ValueError("transform enumeration requires a temporary closed boundary")
    profile_points = [
        _metric_point(point, section_gauge)
        for point in boundary.get("ordered_boundary_display", []) or []
    ]
    profile = Polygon(profile_points)
    if len(profile_points) < 3 or not profile.is_valid or profile.area <= 0:
        raise ValueError("temporary profile is not a valid metric polygon")
    interval = list(map(float, sweep.get("interval_mm", []) or []))
    if len(interval) != 2 or interval[1] <= interval[0]:
        raise ValueError("bounded sweep interval is unresolved")
    depth = interval[1] - interval[0]
    if not math.isclose(depth, float(sweep.get("sweep_width_mm") or 0.0), abs_tol=1e-6):
        raise ValueError("bounded sweep interval and width disagree")

    interface_cap = _cap(boundary, "landing_interface")
    support_cap = _cap(boundary, "external_support_end")
    interface_points = [_metric_point(point, section_gauge) for point in _cap_endpoints(interface_cap)]
    support_points = [_metric_point(point, section_gauge) for point in _cap_endpoints(support_cap)]
    interface_station = sum(point[0] for point in interface_points) / 2.0
    support_station = sum(point[0] for point in support_points) / 2.0
    centroid_x, centroid_z = float(profile.centroid.x), float(profile.centroid.y)
    evidence_refs = sorted(
        {
            str(boundary.get("id")),
            str(sweep.get("id")),
            *map(str, boundary.get("evidence_refs", []) or []),
            *map(str, sweep.get("evidence_refs", []) or []),
            *map(str, section_gauge.get("evidence_refs", []) or []),
            *map(str, interface_cap.get("evidence_refs", []) or []),
            *map(str, support_cap.get("evidence_refs", []) or []),
        }
    )
    anchors = [
        {
            "kind": "classified_interface_cap_station",
            "station_local_mm": interface_station,
            "source_ref": str(interface_cap.get("interface_option_ref") or ""),
            "evidence_refs": list(interface_cap.get("evidence_refs", []) or []),
            "landing_overlap_authorization": None,
        }
    ]
    for anchor in additional_interface_anchors:
        station = float(anchor.get("station_local_mm") or 0.0)
        refs = list(map(str, anchor.get("evidence_refs", []) or []))
        if not math.isfinite(station) or not refs:
            raise ValueError("additional interface anchors require a finite station and evidence")
        anchors.append(
            {
                "kind": str(anchor.get("kind") or "certified_interface_station"),
                "station_local_mm": station,
                "source_ref": str(anchor.get("source_ref") or ""),
                "evidence_refs": refs,
                "landing_overlap_authorization": anchor.get(
                    "landing_overlap_authorization"
                ),
            }
        )

    alternatives = []
    for anchor in anchors:
        anchor_station = float(anchor["station_local_mm"])
        for longitudinal_sign in longitudinal_signs:
            if longitudinal_sign not in {-1, 1}:
                raise ValueError("longitudinal signs must be -1 or 1")
            for reflection_sign in shared_reflection_signs:
                if reflection_sign not in {-1, 1}:
                    raise ValueError("reflection signs must be -1 or 1")
                sweep_endpoint = interval[0] if reflection_sign == 1 else interval[1]
                origin_y = sweep_endpoint - clear_sweep_origin_mm
                identity = _stable_id(
                    "profile_to_sweep_transform_hypothesis",
                    boundary.get("id"),
                    sweep.get("id"),
                    anchor["kind"],
                    anchor_station,
                    longitudinal_sign,
                    reflection_sign,
                )
                anchor_evidence = sorted(
                    {*evidence_refs, *map(str, anchor["evidence_refs"])}
                )
                alternatives.append(
                    {
                    "id": identity,
                    "record_type": "profile_to_sweep_transform_hypothesis",
                    "state": "resolved_search_hypothesis",
                    "source_boundary_ref": str(boundary.get("id")),
                    "source_sweep_ref": str(sweep.get("id")),
                    "interface_cap_ref": str(interface_cap.get("interface_option_ref") or ""),
                    "external_support_cap_ref": str(support_cap.get("interface_option_ref") or ""),
                    "interface_anchor_kind": anchor["kind"],
                    "source_interface_anchor_ref": anchor["source_ref"],
                    "longitudinal_sign": longitudinal_sign,
                    "shared_coordinate_reflection_sign": reflection_sign,
                    "classified_interface_cap_station_local_mm": interface_station,
                    "interface_station_local_mm": anchor_station,
                    "external_support_station_local_mm": support_station,
                    "mapped_interface_plane_x_mm": 0.0,
                    "mapped_external_support_x_mm": longitudinal_sign
                    * (support_station - anchor_station),
                    "sweep_start_maps_to_band_endpoint_mm": sweep_endpoint,
                    "sweep_end_maps_to_band_endpoint_mm": interval[1] if reflection_sign == 1 else interval[0],
                    "profile_centroid_uv_mm": [centroid_x, centroid_z],
                    "local_to_world_transform": {
                        "id": _stable_id("local_to_world_transform", identity),
                        "record_type": "local_to_world_transform",
                        "state": "resolved",
                        "origin_xyz_mm": [
                            longitudinal_sign * (centroid_x - anchor_station),
                            origin_y,
                            centroid_z,
                        ],
                        "local_axes_xyz": {
                            "x": [float(longitudinal_sign), 0.0, 0.0],
                            "y": [0.0, float(reflection_sign), 0.0],
                            "z": [0.0, 0.0, float(longitudinal_sign * reflection_sign)],
                        },
                        "transform_role": "certified_profile_cap_to_bounded_sweep",
                        "accepted_physical_placement": False,
                        "evidence_refs": anchor_evidence,
                        "quantity_eligible": False,
                    },
                    "mapping_certificate": {
                        "interface_cap_maps_to_certified_interface_plane": True,
                        "interface_anchor_is_classified_cap": anchor["kind"]
                        == "classified_interface_cap_station",
                        "interface_anchor_is_independently_certified_station": anchor[
                            "kind"
                        ]
                        != "classified_interface_cap_station",
                        "external_support_cap_maps_to_opposite_longitudinal_side": (
                            not math.isclose(
                                support_station, anchor_station, abs_tol=1e-6
                            )
                        ),
                        "origin_derived_from_cap_to_interface_correspondence": True,
                        "bounded_sweep_endpoints_mapped_bijectively": True,
                        "proper_rigid_basis_preserved_by_local_profile_reparameterization": True,
                        "interface_cap_station_spread_mm": abs(interface_points[1][0] - interface_points[0][0]),
                        "external_support_cap_station_spread_mm": abs(support_points[1][0] - support_points[0][0]),
                        "search_hypothesis_only": True,
                    },
                    "landing_overlap_authorization": anchor[
                        "landing_overlap_authorization"
                    ],
                    "evidence_refs": anchor_evidence,
                    "quantity_eligible": False,
                    }
                )
    return alternatives
