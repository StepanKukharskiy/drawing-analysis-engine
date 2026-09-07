"""Materialize evidence-bounded profile sweeps as construction regions."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import Polygon

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh


SCHEMA_VERSION = "0.1.0"
LENGTH_TOLERANCE_MM = 1e-6
VOLUME_TOLERANCE_MM3 = 1.0


def _stable_id(kind: str, *parts: Any) -> str:
    encoded = "\0".join((kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _evidence(record: Mapping[str, Any], field: str) -> list[str]:
    refs = sorted({str(item) for item in record.get("evidence_refs", []) or [] if str(item)})
    if not refs:
        raise ValueError(f"{field} requires evidence_refs")
    return refs


def _vector(value: Any, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a finite three-vector")
    return vector


def _transform_point(point: Sequence[float], transform: Mapping[str, Any]) -> list[float]:
    origin = _vector(transform.get("origin_xyz_mm"), "transform.origin_xyz_mm")
    axes = transform.get("local_axes_xyz") or {}
    x_axis = _vector(axes.get("x"), "transform.local_axes_xyz.x")
    y_axis = _vector(axes.get("y"), "transform.local_axes_xyz.y")
    z_axis = _vector(axes.get("z"), "transform.local_axes_xyz.z")
    basis = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-9):
        raise ValueError("local-to-world axes must be orthonormal")
    if not math.isclose(abs(float(np.linalg.det(basis))), 1.0, abs_tol=1e-9):
        raise ValueError("local-to-world axes must form a rigid basis")
    return (origin + basis @ np.asarray(point, dtype=np.float64)).tolist()


def _polygon_key(points: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(sorted(tuple(round(float(value), 6) for value in point) for point in points))


def materialize_bounded_profile_sweep_union(
    physical_object_scope: Mapping[str, Any],
    profile_sweep_hypotheses: Sequence[Mapping[str, Any]],
    internal_seam_hypotheses: Sequence[Mapping[str, Any]],
    supplied_projection_evidence: Sequence[Mapping[str, Any]],
    analytic_union_certificate: Mapping[str, Any],
    *,
    existing_construction_regions: Sequence[Mapping[str, Any]] = (),
    existing_region_projections: Sequence[Mapping[str, Any]] = (),
    alternative_id: str,
) -> dict[str, Any]:
    """Materialize one complete bounded-sweep union hypothesis.

    Profiles, transforms, cap classifications, views, seams, and the analytic
    union are independent inputs.  The result is directly consumable by the
    constructive union enumerator; no physical component identity is created.
    """

    scope_ref = str(physical_object_scope.get("id") or "")
    if physical_object_scope.get("record_type") != "physical_object_scope":
        raise ValueError("materializer requires a physical_object_scope")
    if physical_object_scope.get("state") not in {
        "resolved",
        "resolved_relative",
        "resolved_search_hypothesis",
    }:
        raise ValueError("physical object scope is unresolved")
    scope_evidence = _evidence(physical_object_scope, scope_ref)
    if not profile_sweep_hypotheses:
        raise ValueError("at least one bounded profile sweep is required")
    if len(supplied_projection_evidence) < 2:
        raise ValueError("at least two supplied projection views are required")

    regions = [deepcopy(dict(item)) for item in existing_construction_regions]
    if existing_region_projections:
        raise ValueError(
            "generated or region-derived projections cannot supply expected drawing projections"
        )
    views = []
    allowed_projection_sources = {
        "native_vector_geometry",
        "certified_dimension_geometry",
        "native_vector_and_certified_dimensions",
        "synthetic_known_truth",
    }
    for view in supplied_projection_evidence:
        view_id = str(view.get("id") or "")
        if view.get("record_type") != "native_projection_evidence":
            raise ValueError(f"{view_id} is not native projection evidence")
        if view.get("state") != "resolved":
            raise ValueError(f"{view_id} projection evidence is unresolved")
        if view.get("source_geometry_kind") not in allowed_projection_sources:
            raise ValueError(f"{view_id} projection source is not independent drawing evidence")
        if view.get("derived_from_generated_geometry") is not False:
            raise ValueError(f"{view_id} projection independence is not certified")
        polygons = view.get("polygons_uv_mm") or []
        if not polygons or any(
            len(points) < 3 or not Polygon(points).is_valid or Polygon(points).area <= LENGTH_TOLERANCE_MM
            for points in polygons
        ):
            raise ValueError(f"{view_id} lacks valid native projection polygons")
        _evidence(view, view_id)
        views.append(deepcopy(dict(view)))
    cap_faces: dict[str, dict[str, Any]] = {}
    for hypothesis in profile_sweep_hypotheses:
        hypothesis_id = str(hypothesis.get("id") or "")
        if hypothesis.get("record_type") != "bounded_profile_sweep_hypothesis":
            raise ValueError(f"{hypothesis_id} has the wrong record type")
        if hypothesis.get("state") != "resolved":
            raise ValueError(f"{hypothesis_id} is unresolved")
        if str(hypothesis.get("physical_object_scope_ref")) != scope_ref:
            raise ValueError(f"{hypothesis_id} is outside the physical object scope")
        hypothesis_evidence = _evidence(hypothesis, hypothesis_id)
        profile = hypothesis.get("profile_boundary") or {}
        transform = hypothesis.get("local_to_world_transform") or {}
        sweep = hypothesis.get("sweep_bound") or {}
        if profile.get("record_type") != "bounded_profile_boundary" or profile.get("state") != "resolved":
            raise ValueError(f"{hypothesis_id} profile boundary is unresolved")
        if transform.get("record_type") != "local_to_world_transform" or transform.get("state") != "resolved":
            raise ValueError(f"{hypothesis_id} local-to-world transform is unresolved")
        if sweep.get("record_type") != "bounded_sweep" or sweep.get("state") != "resolved":
            raise ValueError(f"{hypothesis_id} sweep bound is unresolved")
        profile_evidence = _evidence(profile, f"{hypothesis_id}.profile_boundary")
        transform_evidence = _evidence(transform, f"{hypothesis_id}.local_to_world_transform")
        sweep_evidence = _evidence(sweep, f"{hypothesis_id}.sweep_bound")
        boundary = [list(map(float, point)) for point in profile.get("ordered_points_uv_mm", []) or []]
        polygon = Polygon(boundary)
        if len(boundary) < 3 or not polygon.is_valid or polygon.area <= LENGTH_TOLERANCE_MM:
            raise ValueError(f"{hypothesis_id} temporary section boundary is not a valid polygon")
        depth_mm = float(sweep.get("depth_mm") or 0.0)
        if not math.isfinite(depth_mm) or depth_mm <= 0:
            raise ValueError(f"{hypothesis_id} sweep depth is invalid")

        region_id = str(hypothesis.get("construction_region_id") or _stable_id("construction_region", hypothesis_id))
        mesh = _extruded_mesh(boundary, depth_mm)
        region_transform = {
            "id": _stable_id("physical_component_transform", region_id, transform.get("id")),
            "record_type": "physical_component_transform",
            "record_version": SCHEMA_VERSION,
            "component_ref": region_id,
            "state": "resolved",
            "placement_role": "physical",
            "origin_xyz_mm": list(map(float, transform.get("origin_xyz_mm", []))),
            "local_axes_xyz": deepcopy(transform.get("local_axes_xyz")),
            "axis_signs": "resolved",
            "evidence_refs": transform_evidence,
        }
        analytic_volume = float(polygon.area) * depth_mm
        region_evidence = sorted(
            set([*scope_evidence, *hypothesis_evidence, *profile_evidence, *transform_evidence, *sweep_evidence])
        )
        regions.append(
            {
                "id": region_id,
                "record_type": "construction_region",
                "physical_object_scope_ref": scope_ref,
                "construction_role": str(hypothesis.get("construction_role") or "bounded_profile_sweep"),
                "state": "resolved",
                "mesh": mesh,
                "transform": region_transform,
                "analytic_volume": {
                    "value_mm3": analytic_volume,
                    "basis": "closed_metric_profile_area_times_certified_bounded_sweep",
                    "evidence_refs": region_evidence,
                },
                "temporary_section_boundary": {
                    "record_type": "temporary_section_boundary",
                    "ordered_points_uv_mm": boundary,
                    "classified_caps": deepcopy(profile.get("classified_caps", []) or []),
                    "state": "resolved",
                    "quantity_eligible": False,
                },
                "evidence_refs": region_evidence,
                "quantity_eligible": False,
            }
        )

        for cap in profile.get("classified_caps", []) or []:
            cap_id = str(cap.get("id") or "")
            edge_index = int(cap.get("boundary_edge_index", -1))
            if not cap_id or cap_id in cap_faces or not 0 <= edge_index < len(boundary):
                raise ValueError(f"{hypothesis_id} has an invalid classified cap")
            if cap.get("classification") not in {"internal_seam", "external_support_end"}:
                raise ValueError(f"{cap_id} has an unsupported cap classification")
            following = (edge_index + 1) % len(boundary)
            start, end = boundary[edge_index], boundary[following]
            polygon_world = [
                _transform_point([start[0], 0.0, start[1]], transform),
                _transform_point([start[0], depth_mm, start[1]], transform),
                _transform_point([end[0], depth_mm, end[1]], transform),
                _transform_point([end[0], 0.0, end[1]], transform),
            ]
            cap_faces[cap_id] = {
                "id": cap_id,
                "record_type": "construction_region_cap_face",
                "construction_region_ref": region_id,
                "classification": cap["classification"],
                "polygon_xyz_mm": polygon_world,
                "state": "resolved",
                "evidence_refs": sorted(set([*region_evidence, *_evidence(cap, cap_id)])),
                "quantity_eligible": False,
            }

    region_ids = {str(item.get("id")) for item in regions}
    resolved_seams = []
    for seam in internal_seam_hypotheses:
        seam_id = str(seam.get("id") or "")
        if seam.get("record_type") != "internal_seam_hypothesis" or seam.get("state") != "resolved":
            raise ValueError(f"{seam_id} seam hypothesis is unresolved")
        seam_kind = str(seam.get("seam_kind") or "coincident_face")
        semantic_fields = {
            key: deepcopy(seam.get(key))
            for key in (
                "source_cap_classification",
                "source_interface_option_ref",
                "source_section_station_ref",
                "source_projected_plan_station_refs",
                "required_contact_semantics",
                "overlap_authorization_certificate",
            )
            if key in seam
        }
        if seam_kind == "overlap_union":
            refs = list(map(str, seam.get("construction_region_refs", []) or []))
            overlap_volume = float(seam.get("overlap_volume_mm3") or 0.0)
            if (
                len(set(refs)) != 2
                or any(ref not in region_ids for ref in refs)
                or not math.isfinite(overlap_volume)
                or overlap_volume <= 0
            ):
                raise ValueError(f"{seam_id} has an invalid certified overlap")
            resolved_seams.append(
                {
                    "id": seam_id,
                    "record_type": "internal_seam",
                    "physical_object_scope_ref": scope_ref,
                    "construction_region_refs": refs,
                    "seam_kind": "overlap_union",
                    "state": "resolved",
                    "analytic_overlap_volume_mm3": overlap_volume,
                    **semantic_fields,
                    "evidence_refs": _evidence(seam, seam_id),
                    "quantity_eligible": False,
                }
            )
            continue
        if seam_kind != "coincident_face":
            raise ValueError(f"{seam_id} has an unsupported seam kind")
        certified_polygon = seam.get("polygon_xyz_mm")
        if certified_polygon is not None:
            refs = list(map(str, seam.get("construction_region_refs", []) or []))
            points = [list(map(float, point)) for point in certified_polygon or []]
            source_kind = str(seam.get("polygon_source_kind") or "")
            if (
                len(set(refs)) != 2
                or any(ref not in region_ids for ref in refs)
                or len(points) < 3
                or source_kind not in {
                    "certified_boundary_intersection",
                    "certified_interface_geometry",
                    "synthetic_known_truth",
                }
            ):
                raise ValueError(f"{seam_id} has an invalid certified coincident polygon")
            resolved_seams.append(
                {
                    "id": seam_id,
                    "record_type": "internal_seam",
                    "physical_object_scope_ref": scope_ref,
                    "construction_region_refs": refs,
                    "seam_kind": "coincident_face",
                    "state": "resolved",
                    "polygon_xyz_mm": points,
                    "polygon_source_kind": source_kind,
                    **semantic_fields,
                    "evidence_refs": _evidence(seam, seam_id),
                    "quantity_eligible": False,
                }
            )
            continue
        face_refs = list(map(str, seam.get("cap_face_refs", []) or []))
        faces = [cap_faces.get(ref) for ref in face_refs]
        if len(faces) != 2 or any(face is None for face in faces):
            raise ValueError(f"{seam_id} must reference two resolved cap faces")
        if any(face["classification"] != "internal_seam" for face in faces):
            raise ValueError(f"{seam_id} references a non-seam cap")
        if _polygon_key(faces[0]["polygon_xyz_mm"]) != _polygon_key(faces[1]["polygon_xyz_mm"]):
            raise ValueError(f"{seam_id} cap polygons are not congruent and coincident")
        refs = [str(face["construction_region_ref"]) for face in faces]
        if len(set(refs)) != 2 or any(ref not in region_ids for ref in refs):
            raise ValueError(f"{seam_id} does not join two construction regions")
        resolved_seams.append(
            {
                "id": seam_id,
                "record_type": "internal_seam",
                "physical_object_scope_ref": scope_ref,
                "construction_region_refs": refs,
                "seam_kind": "coincident_face",
                "state": "resolved",
                "polygon_xyz_mm": faces[0]["polygon_xyz_mm"],
                **semantic_fields,
                "evidence_refs": sorted(
                    set([*_evidence(seam, seam_id), *(ref for face in faces for ref in face["evidence_refs"])])
                ),
                "quantity_eligible": False,
            }
        )

    analytic_refs = _evidence(analytic_union_certificate, "analytic_union_certificate")
    if analytic_union_certificate.get("record_type") != "analytic_union_certificate":
        raise ValueError("analytic union certificate has the wrong record type")
    if analytic_union_certificate.get("state") != "resolved":
        raise ValueError("analytic union certificate is unresolved")
    if analytic_union_certificate.get("independent_of_generated_mesh") is not True:
        raise ValueError("analytic union certificate independence is unresolved")
    if not analytic_union_certificate.get("calculation_terms"):
        raise ValueError("analytic union certificate requires independently certified calculation terms")
    region_sum = sum(float((item.get("analytic_volume") or {}).get("value_mm3") or 0.0) for item in regions)
    overlap_deduction = sum(
        float(item.get("overlap_volume_mm3") or 0.0)
        for item in internal_seam_hypotheses
        if item.get("seam_kind") == "overlap_union"
    )
    derived_union = region_sum - overlap_deduction
    certified_union = float(analytic_union_certificate.get("value_mm3") or 0.0)
    tolerance = max(VOLUME_TOLERANCE_MM3, 1e-8 * max(derived_union, certified_union))
    if certified_union <= 0 or abs(certified_union - derived_union) > tolerance:
        raise ValueError("analytic union certificate disagrees with materialized regions")

    placement = {
        "id": alternative_id,
        "record_type": "constructive_placement_alternative",
        "physical_object_scope_ref": scope_ref,
        "construction_region_refs": sorted(region_ids),
        "internal_seam_refs": sorted(item["id"] for item in resolved_seams),
        "supplied_projection_refs": sorted(item["id"] for item in views),
        "separate_object_interface_refs": [],
        "analytic_union_volume": {
            "value_mm3": certified_union,
            "basis": str(analytic_union_certificate.get("basis") or "independent_analytic_union_certificate"),
            "evidence_refs": analytic_refs,
        },
        "evidence_refs": sorted(
            set([*scope_evidence, *analytic_refs, *(ref for item in regions for ref in item.get("evidence_refs", []))])
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "bounded_profile_sweep_materializer",
        "status": "materialized",
        "physical_object_scope": deepcopy(dict(physical_object_scope)),
        "temporary_section_boundaries": [item["temporary_section_boundary"] for item in regions if item.get("temporary_section_boundary")],
        "construction_regions": regions,
        "cap_faces": sorted(cap_faces.values(), key=lambda item: item["id"]),
        "internal_seams": resolved_seams,
        "supplied_projections": views,
        "analytic_union_certificate": {
            **deepcopy(dict(analytic_union_certificate)),
            "derived_region_sum_mm3": region_sum,
            "derived_overlap_deduction_mm3": overlap_deduction,
            "derived_union_mm3": derived_union,
            "residual_mm3": derived_union - certified_union,
        },
        "placement_alternative": placement,
        "contract": {
            "closed_temporary_section_boundary_required": True,
            "classified_caps_required": True,
            "independent_local_to_world_transform_required": True,
            "analytic_region_volume_derived": True,
            "resolved_internal_seam_polygon_required": True,
            "independent_plan_and_section_projections_required": True,
            "generated_geometry_cannot_supply_expected_projections": True,
            "analytic_union_certificate_required": True,
            "mesh_derived_analytic_union_forbidden": True,
            "physical_component_identity_created": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
