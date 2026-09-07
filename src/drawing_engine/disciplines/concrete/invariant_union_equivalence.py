"""Certify sign-invariant equivalence among accepted constructive unions."""

from __future__ import annotations

from hashlib import sha256
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh


SCHEMA_VERSION = "0.1.0"
MAX_FLOAT_SNAP_MM = 0.01


def _stable_id(member_refs: Sequence[str]) -> str:
    encoded = "\0".join(sorted(member_refs)).encode("utf-8")
    return f"invariant_union_equivalence_certificate.evidence_{sha256(encoded).hexdigest()[:16]}"


def _precision_tolerance(*vertex_sets: np.ndarray) -> float:
    scale = max(
        1.0,
        *(float(np.max(np.abs(vertices))) for vertices in vertex_sets if vertices.size),
    )
    return min(MAX_FLOAT_SNAP_MM, max(1e-6, 8.0 * float(np.spacing(np.float32(scale)))))


def _normalized_vertices(vertices: np.ndarray, signs: Sequence[int]) -> np.ndarray:
    transformed = vertices * np.asarray(signs, dtype=np.float64)
    return transformed - np.min(transformed, axis=0)


def _point_set_residual_mm(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return math.inf
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return max(
        float(np.max(np.min(distances, axis=0))),
        float(np.max(np.min(distances, axis=1))),
    )


def _surface_residual_mm(
    left_vertices: np.ndarray,
    left_faces: np.ndarray,
    right_vertices: np.ndarray,
    right_faces: np.ndarray,
) -> float:
    """Compare geometric surfaces without requiring identical triangulation."""

    left_mesh = trimesh.Trimesh(vertices=left_vertices, faces=left_faces, process=False)
    right_mesh = trimesh.Trimesh(vertices=right_vertices, faces=right_faces, process=False)
    left_samples = np.vstack((left_vertices, left_mesh.triangles_center))
    right_samples = np.vstack((right_vertices, right_mesh.triangles_center))
    _, left_to_right, _ = trimesh.proximity.closest_point_naive(
        right_mesh, left_samples
    )
    _, right_to_left, _ = trimesh.proximity.closest_point_naive(
        left_mesh, right_samples
    )
    return max(float(np.max(left_to_right)), float(np.max(right_to_left)))


def _topology_signature(kernel: Mapping[str, Any]) -> tuple[Any, ...]:
    validation = ((kernel.get("external_boundary") or {}).get("validation") or {})
    return tuple(
        validation.get(key)
        for key in (
            "boundary_edge_count",
            "non_manifold_edge_count",
            "connected_solid_count",
            "watertight",
            "winding_consistent",
            "euler_characteristic",
        )
    )


def _seam_graph_signature(kernel: Mapping[str, Any], tolerance_mm: float) -> tuple[Any, ...]:
    roles = {
        str(region.get("id")): str(region.get("construction_role") or "unspecified")
        for region in kernel.get("construction_regions", []) or []
    }
    area_quantum = max(1e-6, tolerance_mm * tolerance_mm)
    edges = []
    for seam in kernel.get("internal_seams", []) or []:
        region_roles = tuple(
            sorted(roles.get(str(ref), "unresolved") for ref in seam.get("construction_region_refs", []) or [])
        )
        area = float(seam.get("area_mm2") or 0.0)
        edges.append(
            (
                region_roles,
                str(seam.get("seam_kind") or ""),
                str(seam.get("required_contact_semantics") or ""),
                int(round(area / area_quantum)),
            )
        )
    return tuple(sorted(edges))


def _projection_geometry(view: Mapping[str, Any]):
    polygons = []
    for points in view.get("polygons_uv_mm", []) or []:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    return unary_union(polygons) if polygons else None


def _component_topology(geometry: Any) -> tuple[int, int]:
    if geometry is None or geometry.is_empty:
        return 0, 0
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    return len(polygons), sum(len(polygon.interiors) for polygon in polygons)


def _axis_index(vector: Sequence[float]) -> int | None:
    values = np.abs(np.asarray(vector, dtype=np.float64))
    if len(values) != 3 or not math.isclose(float(np.max(values)), 1.0, abs_tol=1e-9):
        return None
    index = int(np.argmax(values))
    if sum(float(value) > 1e-9 for value in values) != 1:
        return None
    return index


def _projection_pair_check(
    left_view: Mapping[str, Any],
    right_view: Mapping[str, Any],
    mesh_signs: Sequence[int],
    left_kernel_view: Mapping[str, Any],
    right_kernel_view: Mapping[str, Any],
    tolerance_mm: float,
) -> dict[str, Any]:
    left = _projection_geometry(left_view)
    right = _projection_geometry(right_view)
    u_index = _axis_index(right_view.get("u_axis_xyz", []) or [])
    v_index = _axis_index(right_view.get("v_axis_xyz", []) or [])
    errors = []
    if left is None or right is None or u_index is None or v_index is None:
        errors.append("projection_geometry_or_axes_unresolved")
        return {"status": "fail", "errors": errors}
    reflected = affinity.scale(
        right,
        xfact=float(mesh_signs[u_index]),
        yfact=float(mesh_signs[v_index]),
        origin=(0.0, 0.0),
    )
    translation = [float(left.bounds[i] - reflected.bounds[i]) for i in (0, 1)]
    reflected = affinity.translate(
        reflected,
        xoff=float(left.bounds[0] - reflected.bounds[0]),
        yoff=float(left.bounds[1] - reflected.bounds[1]),
    )
    difference_area = float(left.symmetric_difference(reflected).area)
    area_tolerance = max(
        1e-6,
        tolerance_mm * max(float(left.length), float(reflected.length), 1.0),
    )
    if difference_area > area_tolerance:
        errors.append("native_projection_geometry_not_equivalent")
    if _component_topology(left) != _component_topology(reflected):
        errors.append("native_projection_topology_not_equivalent")
    if left_kernel_view.get("status") != "pass" or right_kernel_view.get("status") != "pass":
        errors.append("kernel_projection_replay_not_accepted")
    residual_keys = (
        "meaningful_expected_component_count",
        "meaningful_actual_component_count",
        "normalized_expected_component_count",
        "normalized_actual_component_count",
    )
    left_residual = left_kernel_view.get("precision_aware_residual") or {}
    right_residual = right_kernel_view.get("precision_aware_residual") or {}
    if any(left_residual.get(key) != right_residual.get(key) for key in residual_keys):
        errors.append("kernel_projection_component_topology_not_equivalent")
    direction_checks = []
    left_direction = left_view.get("required_direction_evidence")
    right_direction = right_view.get("required_direction_evidence")
    if left_direction is not None or right_direction is not None:
        left_paths = (left_direction or {}).get("paths") or []
        right_paths = (right_direction or {}).get("paths") or []
        if (not left_paths or len(left_paths) != len(right_paths)
            or (left_direction or {}).get("state") != "accepted"
            or (right_direction or {}).get("state") != "accepted"):
            errors.append("directed_projection_evidence_unresolved")
        left_replays = {r.get("path_ref"): r for r in left_kernel_view.get("directed_surface_path_reprojections", [])}
        right_replays = {r.get("path_ref"): r for r in right_kernel_view.get("directed_surface_path_reprojections", [])}
        for a, b in zip(left_paths, right_paths):
            a_samples, b_samples = a.get("samples") or [], b.get("samples") or []
            passed = bool(a_samples) and len(a_samples) == len(b_samples)
            maximum = 0.0
            for sa, sb in zip(a_samples, b_samples):
                transformed = np.asarray(sb["uv_mm"]) * [mesh_signs[u_index], mesh_signs[v_index]] + translation
                maximum = max(maximum, float(np.linalg.norm(np.asarray(sa["uv_mm"]) - transformed)))
                passed &= sa.get("band_ref") == sb.get("band_ref")
            passed &= maximum <= tolerance_mm
            passed &= a.get("convention_ref") == b.get("convention_ref") and a.get("elevation_order") == b.get("elevation_order")
            passed &= np.allclose(np.asarray(b.get("elevation_axis_xyz")) * mesh_signs,
                                  np.asarray(a.get("elevation_axis_xyz")), atol=1e-9)
            passed &= left_replays.get(a.get("id"), {}).get("status") == "pass"
            passed &= right_replays.get(b.get("id"), {}).get("status") == "pass"
            direction_checks.append({"left_path_ref": a.get("id"), "right_path_ref": b.get("id"),
                                     "maximum_station_residual_mm": maximum, "status": "pass" if passed else "fail"})
            if not passed:
                errors.append("directed_projection_binding_not_preserved")
    return {
        "projection_role": left_view.get("projection_role"),
        "reflection_signs_uv": [int(mesh_signs[u_index]), int(mesh_signs[v_index])],
        "symmetric_difference_area_mm2": difference_area,
        "area_tolerance_mm2": area_tolerance,
        "component_topology": list(_component_topology(left)),
        "directed_surface_evidence_checks": direction_checks,
        "status": "fail" if errors else "pass",
        "errors": errors,
    }


def _pair_certificate(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    projection_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    left_kernel = left["kernel_result"]
    right_kernel = right["kernel_result"]
    left_mesh = (left_kernel.get("external_boundary") or {}).get("mesh") or {}
    right_mesh = (right_kernel.get("external_boundary") or {}).get("mesh") or {}
    left_vertices = np.asarray(left_mesh.get("vertices_xyz_mm", []), dtype=np.float64)
    right_vertices = np.asarray(right_mesh.get("vertices_xyz_mm", []), dtype=np.float64)
    left_faces = np.asarray(left_mesh.get("triangles", []), dtype=np.int64)
    right_faces = np.asarray(right_mesh.get("triangles", []), dtype=np.int64)
    tolerance = _precision_tolerance(left_vertices, right_vertices)
    errors = []
    matching_signs = None
    mesh_matches: list[tuple[tuple[int, int, int], float]] = []
    max_vertex_residual = None
    if (
        left_vertices.ndim != 2
        or right_vertices.ndim != 2
        or left_vertices.shape[1:] != (3,)
        or right_vertices.shape[1:] != (3,)
        or left_faces.ndim != 2
        or right_faces.ndim != 2
    ):
        errors.append("external_boundary_mesh_unresolved")
    else:
        normalized_left = _normalized_vertices(left_vertices, (1, 1, 1))
        for signs in itertools.product((-1, 1), repeat=3):
            normalized_right = _normalized_vertices(right_vertices, signs)
            surface_residual = _surface_residual_mm(
                normalized_left,
                left_faces,
                normalized_right,
                right_faces,
            )
            if surface_residual > tolerance:
                continue
            mesh_matches.append(
                (tuple(map(int, signs)), surface_residual)
            )
        if not mesh_matches:
            errors.append("external_boundary_meshes_not_congruent_under_reflection")
    topology_equal = _topology_signature(left_kernel) == _topology_signature(right_kernel)
    if not topology_equal:
        errors.append("external_boundary_topology_not_identical")
    seam_graph_equal = _seam_graph_signature(left_kernel, tolerance) == _seam_graph_signature(
        right_kernel, tolerance
    )
    if not seam_graph_equal:
        errors.append("internal_seam_graph_not_identical")

    left_volume = left_kernel.get("volume_validation") or {}
    right_volume = right_kernel.get("volume_validation") or {}
    volume_tolerance = max(
        float(left_volume.get("tolerance_mm3") or 0.0),
        float(right_volume.get("tolerance_mm3") or 0.0),
        1e-6,
    )
    volume_fields = ("analytic_union_mm3", "mesh_union_mm3", "manifold_union_mm3")
    volume_equal = all(
        abs(float(left_volume.get(key) or 0.0) - float(right_volume.get(key) or 0.0))
        <= volume_tolerance
        for key in volume_fields
    )
    if not volume_equal:
        errors.append("analytic_or_mesh_union_volume_not_identical")

    projection_checks = []
    if mesh_matches:
        left_views = {
            str((projection_index.get(ref) or {}).get("projection_role") or ref): projection_index.get(ref)
            for ref in left.get("supplied_projection_refs", []) or []
            if projection_index.get(ref) is not None
        }
        right_views = {
            str((projection_index.get(ref) or {}).get("projection_role") or ref): projection_index.get(ref)
            for ref in right.get("supplied_projection_refs", []) or []
            if projection_index.get(ref) is not None
        }
        left_kernel_views = {
            str(
                item.get("projection_role")
                or (projection_index.get(str(item.get("id"))) or {}).get(
                    "projection_role"
                )
                or item.get("id")
            ): item
            for item in left_kernel.get("supplied_view_reprojections", []) or []
        }
        right_kernel_views = {
            str(
                item.get("projection_role")
                or (projection_index.get(str(item.get("id"))) or {}).get(
                    "projection_role"
                )
                or item.get("id")
            ): item
            for item in right_kernel.get("supplied_view_reprojections", []) or []
        }
        if set(left_views) != set(right_views) or len(left_views) < 2:
            errors.append("supplied_projection_roles_not_identical")
        else:
            failed_projection_checks = []
            for candidate_signs, vertex_residual in mesh_matches:
                candidate_checks = [
                    _projection_pair_check(
                        left_views[role],
                        right_views[role],
                        candidate_signs,
                        left_kernel_views.get(role, {}),
                        right_kernel_views.get(role, {}),
                        tolerance,
                    )
                    for role in sorted(left_views)
                ]
                if all(check["status"] == "pass" for check in candidate_checks):
                    matching_signs = candidate_signs
                    max_vertex_residual = vertex_residual
                    projection_checks = candidate_checks
                    break
                failed_projection_checks = candidate_checks
            if matching_signs is None:
                projection_checks = failed_projection_checks
                errors.append("native_projections_not_equivalent_under_mesh_reflection")
    return {
        "left_hypothesis_ref": left["constructive_union_hypothesis_ref"],
        "right_hypothesis_ref": right["constructive_union_hypothesis_ref"],
        "allowed_transform_kind": "axis_sign_reflection_and_translation",
        "reflection_signs_xyz": list(matching_signs) if matching_signs is not None else None,
        "mesh_tolerance_mm": tolerance,
        "maximum_vertex_residual_mm": max_vertex_residual,
        "external_boundary_mesh_congruence": "pass" if mesh_matches else "fail",
        "external_boundary_topology_identity": "pass" if topology_equal else "fail",
        "internal_seam_graph_identity": "pass" if seam_graph_equal else "fail",
        "analytic_and_mesh_volume_identity": "pass" if volume_equal else "fail",
        "projection_checks": projection_checks,
        "errors": sorted(set(errors)),
        "status": "fail" if errors else "pass",
    }


def certify_invariant_union_equivalence(
    survivors: Sequence[Mapping[str, Any]],
    supplied_projection_evidence: Sequence[Mapping[str, Any]],
    *,
    all_competing_alternatives_resolved: bool,
) -> dict[str, Any] | None:
    """Accept one equivalence class only after every pair closes invariantly."""

    if len(survivors) < 2:
        return None
    projection_index = {
        str(item.get("id")): item for item in supplied_projection_evidence
    }
    pair_checks = [
        _pair_certificate(left, right, projection_index)
        for left, right in itertools.combinations(survivors, 2)
    ]
    member_refs = sorted(
        str(item.get("constructive_union_hypothesis_ref")) for item in survivors
    )
    errors = []
    if not all_competing_alternatives_resolved:
        errors.append("unreplayed_live_alternatives_remain")
    if any(item["status"] != "pass" for item in pair_checks):
        errors.append("pairwise_invariant_equivalence_failed")
    volumes = [
        float((item["kernel_result"].get("volume_validation") or {}).get("analytic_union_mm3") or 0.0)
        for item in survivors
    ]
    if not volumes or min(volumes) <= 0:
        errors.append("positive_invariant_volume_unresolved")
    identity = _stable_id(member_refs)
    accepted = not errors
    volume_mm3 = volumes[0] if accepted else None
    return {
        "id": identity,
        "record_type": "invariant_union_equivalence_certificate",
        "record_version": SCHEMA_VERSION,
        "state": "accepted" if accepted else "unresolved",
        "physical_placement_state": (
            "resolved_in_directed_plan_gauge" if accepted and any(
                view.get("required_direction_evidence") for view in supplied_projection_evidence
            ) else "unresolved_reflection"
        ),
        "member_hypothesis_refs": member_refs,
        "preserved_transform_alternative_count": len(member_refs),
        "pairwise_checks": pair_checks,
        "invariance_checks": {
            "pairwise_congruent_final_meshes_under_allowed_reflection": "pass" if accepted else "fail",
            "identical_external_boundary_topology": "pass" if accepted else "fail",
            "identical_internal_seam_graph": "pass" if accepted else "fail",
            "identical_analytic_and_mesh_volume": "pass" if accepted else "fail",
            "identical_native_plan_and_section_projections": "pass" if accepted else "fail",
            "directed_native_surface_evidence_preserved": "pass" if accepted else "fail",
            "all_competing_alternatives_replayed_or_hard_rejected": (
                "pass" if all_competing_alternatives_resolved else "fail"
            ),
        },
        "invariant_union_volume_candidate": (
            {
                "value_mm3": volume_mm3,
                "value_m3": volume_mm3 / 1_000_000_000.0,
                "state": "derived",
                "basis": "pairwise_congruent_step4_union_equivalence_class",
                "signed_physical_placement_resolved": False,
                "quantity_eligible": True,
            }
            if accepted
            else None
        ),
        "errors": errors,
        "quantity_eligible": accepted,
        "evidence_refs": member_refs,
        "contract": {
            "equal_volume_alone_is_insufficient": True,
            "axis_permutation_is_not_an_allowed_equivalence": True,
            "native_expected_projections_are_not_generated_from_candidate_meshes": True,
            "absolute_viewing_direction_remains_unresolved": True,
            "reflections_must_preserve_directed_native_evidence": True,
        },
    }
