"""Fail-closed Step 4 kernels for physical solids.

The multi-object path consumes positioned component meshes and explicit shared
interfaces.  The same-object path unions construction regions, removes their
internal seams, and validates one external boundary.  Neither path infers
placement from drawing layout or reads declared quantities.  Both require
analytic-volume agreement and matching supplied orthographic projections in
at least two non-parallel views.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import itertools
import json
import math
from typing import Any, Mapping, Sequence

import manifold3d as md
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
import trimesh


SCHEMA_VERSION = "0.1.0"
LENGTH_TOLERANCE_MM = 1e-6
AREA_TOLERANCE_MM2 = 1e-5
MANIFOLD_RELATIVE_VOLUME_TOLERANCE = 4.0 * float(np.finfo(np.float32).eps)
MAX_NUMERICAL_SNAP_MM = 0.01


class SolidKernelValidationError(ValueError):
    """A fail-closed kernel result carrying every available residual."""

    def __init__(self, errors: Sequence[str], diagnostics: Mapping[str, Any]):
        self.errors = list(dict.fromkeys(map(str, errors)))
        self.diagnostics = deepcopy(dict(diagnostics))
        super().__init__("; ".join(self.errors))


def _evidence(value: Any, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} requires non-empty evidence_refs")
    refs = sorted({str(item) for item in value if str(item)})
    if not refs:
        raise ValueError(f"{field} requires non-empty evidence_refs")
    return refs


def _vector(value: Any, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must be a finite XYZ vector")
    return vector


def _physical_transform(component: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    component_id = str(component.get("id"))
    transform = component.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError(f"{component_id} has no physical transform")
    transform_id = str(transform.get("id"))
    if not transform_id or transform_id == "None":
        raise ValueError(f"{component_id} physical transform has no stable id")
    if transform.get("record_type") != "physical_component_transform":
        raise ValueError(f"{component_id} transform has an unsupported record type")
    if transform.get("record_version") != SCHEMA_VERSION:
        raise ValueError(f"{component_id} transform has an unsupported record version")
    if str(transform.get("component_ref")) != component_id:
        raise ValueError(f"{component_id} transform does not reference its component")
    if transform.get("state") != "resolved" or transform.get("placement_role") != "physical":
        raise ValueError(f"{component_id} transform is not resolved physical placement")
    if transform.get("axis_signs") != "resolved":
        raise ValueError(f"{component_id} transform axis signs are unresolved")
    origin = _vector(transform.get("origin_xyz_mm"), f"{component_id}.transform.origin_xyz_mm")
    axes = transform.get("local_axes_xyz")
    if not isinstance(axes, Mapping):
        raise ValueError(f"{component_id} transform has no local_axes_xyz")
    basis = np.column_stack(
        [_vector(axes.get(axis), f"{component_id}.transform.local_axes_xyz.{axis}") for axis in "xyz"]
    )
    if not np.allclose(basis.T @ basis, np.identity(3), atol=1e-9):
        raise ValueError(f"{component_id} transform axes are not orthonormal")
    if not math.isclose(float(np.linalg.det(basis)), 1.0, abs_tol=1e-9):
        raise ValueError(f"{component_id} transform is not a proper rigid transform")
    record = {
        "id": transform_id,
        "record_type": "physical_component_transform",
        "record_version": SCHEMA_VERSION,
        "component_ref": component_id,
        "state": "resolved",
        "placement_role": "physical",
        "origin_xyz_mm": origin.tolist(),
        "local_axes_xyz": {axis: basis[:, index].tolist() for index, axis in enumerate("xyz")},
        "axis_signs": "resolved",
        "evidence_refs": _evidence(transform.get("evidence_refs"), f"{component_id}.transform"),
    }
    return origin, basis, record


def _edge_validation(faces: np.ndarray) -> tuple[int, int, int]:
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for left, right in zip(face, (*face[1:], face[0])):
            edge = tuple(sorted((int(left), int(right))))
            counts[edge] = counts.get(edge, 0) + 1
    return len(counts), sum(value == 1 for value in counts.values()), sum(value != 2 for value in counts.values())


def _component_mesh(component: Mapping[str, Any]) -> dict[str, Any]:
    component_id = str(component.get("id"))
    if not component_id or component_id == "None":
        raise ValueError("every component requires an id")
    mesh_input = component.get("mesh")
    if not isinstance(mesh_input, Mapping):
        raise ValueError(f"{component_id} has no mesh")
    vertices = np.asarray(mesh_input.get("vertices_xyz_mm"), dtype=np.float64)
    faces = np.asarray(mesh_input.get("triangles"), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4 or not np.all(np.isfinite(vertices)):
        raise ValueError(f"{component_id} mesh vertices are invalid")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
        raise ValueError(f"{component_id} mesh triangles are invalid")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError(f"{component_id} mesh triangle index is out of range")

    origin, basis, transform = _physical_transform(component)
    global_vertices = vertices @ basis.T + origin
    mesh = trimesh.Trimesh(vertices=global_vertices, faces=faces.copy(), process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError(f"{component_id} mesh is not a consistently wound watertight solid")
    winding_reversed = False
    if mesh.volume < 0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
        winding_reversed = True
    edge_count, boundary_count, invalid_edge_count = _edge_validation(mesh.faces)
    if boundary_count or invalid_edge_count:
        raise ValueError(f"{component_id} mesh has boundary or non-manifold edges")
    pieces = mesh.split(only_watertight=False)
    if len(pieces) != 1:
        raise ValueError(f"{component_id} mesh contains multiple disconnected solids")

    analytic = component.get("analytic_volume")
    if not isinstance(analytic, Mapping):
        raise ValueError(f"{component_id} has no analytic volume record")
    analytic_volume = float(analytic.get("value_mm3", 0.0))
    if not math.isfinite(analytic_volume) or analytic_volume <= 0:
        raise ValueError(f"{component_id} analytic volume must be positive")
    analytic_basis = str(analytic.get("basis") or "")
    if not analytic_basis:
        raise ValueError(f"{component_id} analytic volume has no basis")
    analytic_evidence = _evidence(analytic.get("evidence_refs"), f"{component_id}.analytic_volume")
    mesh_volume = float(mesh.volume)
    volume_tolerance = max(
        1.0, MANIFOLD_RELATIVE_VOLUME_TOLERANCE * analytic_volume
    )
    volume_residual = mesh_volume - analytic_volume
    if abs(volume_residual) > volume_tolerance:
        raise ValueError(f"{component_id} analytic and mesh volumes disagree")

    manifold = md.Manifold(
        md.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float64),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )
    if manifold.status() != md.Error.NoError or manifold.is_empty():
        raise ValueError(f"{component_id} mesh does not form a valid manifold")
    manifold_volume = float(manifold.volume())
    if abs(manifold_volume - analytic_volume) > volume_tolerance:
        raise ValueError(f"{component_id} analytic and manifold volumes disagree")

    evidence_refs = sorted(
        set(_evidence(component.get("evidence_refs"), component_id))
        | set(transform["evidence_refs"])
        | set(analytic_evidence)
    )
    return {
        "id": component_id,
        "mesh": mesh,
        "manifold": manifold,
        "transform": transform,
        "analytic_volume": {
            "value_mm3": analytic_volume,
            "basis": analytic_basis,
            "evidence_refs": analytic_evidence,
        },
        "validation": {
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": int(len(mesh.faces)),
            "edge_count": edge_count,
            "boundary_edge_count": boundary_count,
            "non_manifold_edge_count": invalid_edge_count,
            "connected_solid_count": len(pieces),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "input_winding_reversed": winding_reversed,
            "euler_characteristic": int(mesh.euler_number),
            "mesh_volume_mm3": mesh_volume,
            "manifold_volume_mm3": manifold_volume,
            "analytic_volume_mm3": analytic_volume,
            "analytic_mesh_residual_mm3": volume_residual,
            "volume_tolerance_mm3": volume_tolerance,
            "bounds_xyz_mm": np.asarray(mesh.bounds).tolist(),
            "status": "pass",
        },
        "evidence_refs": evidence_refs,
    }


def _plane_frame(points: np.ndarray, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{field} must contain at least three finite XYZ points")
    origin = points[0]
    normal = None
    first_axis = None
    for index in range(1, len(points) - 1):
        left = points[index] - origin
        right = points[index + 1] - origin
        cross = np.cross(left, right)
        if np.linalg.norm(cross) > LENGTH_TOLERANCE_MM:
            first_axis = left / np.linalg.norm(left)
            normal = cross / np.linalg.norm(cross)
            break
    if normal is None or first_axis is None:
        raise ValueError(f"{field} is degenerate")
    if np.max(np.abs((points - origin) @ normal)) > LENGTH_TOLERANCE_MM:
        raise ValueError(f"{field} is not planar")
    second_axis = np.cross(normal, first_axis)
    return origin, first_axis, second_axis


def _project(points: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    relative = points - origin
    return np.column_stack((relative @ u_axis, relative @ v_axis))


def _polygon(points: np.ndarray, field: str) -> Polygon:
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= AREA_TOLERANCE_MM2:
        raise ValueError(f"{field} is not a valid finite-area polygon")
    return polygon


def _numerical_length_resolution_mm(*arrays: np.ndarray) -> float:
    """Bound float-backend noise without admitting drawing-scale gaps."""

    scale = max(
        1.0,
        *(float(np.max(np.abs(array))) for array in arrays if array.size),
    )
    float_resolution = 32.0 * float(np.finfo(np.float32).eps) * scale
    return min(MAX_NUMERICAL_SNAP_MM, max(LENGTH_TOLERANCE_MM, float_resolution))


def _canonical_seam_plane(
    points: np.ndarray,
    meshes: Sequence[trimesh.Trimesh],
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit one certified plane and project only its numerical point noise."""

    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{field} must contain at least three finite XYZ points")
    origin = np.mean(points, axis=0)
    _, singular_values, basis = np.linalg.svd(points - origin, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= LENGTH_TOLERANCE_MM:
        raise ValueError(f"{field} is degenerate")
    normal = basis[-1] / np.linalg.norm(basis[-1])
    residuals = (points - origin) @ normal
    tolerance = _numerical_length_resolution_mm(
        points,
        *(np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes),
    )
    if float(np.max(np.abs(residuals))) > tolerance:
        raise ValueError(f"{field} certified interface is not planar within numerical precision")
    canonical = points - residuals[:, None] * normal
    first_axis = None
    for point in canonical[1:]:
        direction = point - canonical[0]
        direction -= float(np.dot(direction, normal)) * normal
        if np.linalg.norm(direction) > LENGTH_TOLERANCE_MM:
            first_axis = direction / np.linalg.norm(direction)
            break
    if first_axis is None:
        raise ValueError(f"{field} is degenerate")
    second_axis = np.cross(normal, first_axis)
    return canonical, origin, first_axis, second_axis, normal, tolerance


def _refresh_component_geometry(component: dict[str, Any]) -> None:
    mesh = component["mesh"]
    manifold = md.Manifold(
        md.Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float64),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )
    if manifold.status() != md.Error.NoError or manifold.is_empty():
        raise ValueError(f"{component['id']} snapped mesh does not form a valid manifold")
    component["manifold"] = manifold


def _snap_regions_to_certified_seam(
    seam_id: str,
    points: np.ndarray,
    components: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    meshes = [item["mesh"] for item in components]
    canonical, plane_origin, u_axis, v_axis, normal, tolerance = _canonical_seam_plane(
        points, meshes, seam_id
    )
    footprint = _polygon(_project(canonical, plane_origin, u_axis, v_axis), seam_id)
    footprint_gate = footprint.buffer(tolerance)
    component_records = []
    for component in components:
        mesh = component["mesh"]
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        projected = _project(vertices, plane_origin, u_axis, v_axis)
        signed = (vertices - plane_origin) @ normal
        footprint_indices = [
            index
            for index, uv in enumerate(projected)
            if footprint_gate.covers(Point(float(uv[0]), float(uv[1])))
        ]
        minimum = min((abs(float(signed[index])) for index in footprint_indices), default=math.inf)
        cap_indices = [
            index
            for index in footprint_indices
            if abs(abs(float(signed[index])) - minimum) <= tolerance
        ]
        # A certified seam may cover only part of a larger native cap face.
        # Preserve planarity by extending the selection across connected
        # near-plane face vertices; millimetre-scale offsets remain excluded.
        adjacency: dict[int, set[int]] = {index: set() for index in range(len(vertices))}
        for face in np.asarray(mesh.faces, dtype=np.int64):
            for left in face:
                adjacency[int(left)].update(int(right) for right in face if right != left)
        expanded = set(cap_indices)
        frontier = list(cap_indices)
        while frontier:
            index = frontier.pop()
            for neighbor in adjacency[index]:
                if neighbor in expanded or abs(float(signed[neighbor])) > tolerance:
                    continue
                expanded.add(neighbor)
                frontier.append(neighbor)
        cap_indices = sorted(expanded)
        snapped = []
        residual_records = []
        reported_indices = sorted(set(footprint_indices) | set(cap_indices))
        for index in reported_indices:
            residual = float(signed[index])
            selected = index in cap_indices
            snap = selected and abs(residual) <= tolerance
            residual_records.append(
                {
                    "vertex_index": index,
                    "signed_residual_mm": residual,
                    "absolute_residual_mm": abs(residual),
                    "selected_cap_vertex": selected,
                    "snapped": snap,
                }
            )
            if snap:
                vertices[index] -= residual * normal
                snapped.append(index)
        if snapped:
            mesh.vertices = vertices
            _refresh_component_geometry(component)
        component_records.append(
            {
                "construction_region_ref": component["id"],
                "minimum_absolute_residual_mm": None if not math.isfinite(minimum) else minimum,
                "snap_tolerance_mm": tolerance,
                "snapped_vertex_indices": snapped,
                "vertex_residuals": residual_records,
            }
        )
    return canonical, {
        "canonical_plane": {
            "origin_xyz_mm": plane_origin.tolist(),
            "normal_xyz": normal.tolist(),
            "snap_tolerance_mm": tolerance,
            "maximum_permitted_numerical_snap_mm": MAX_NUMERICAL_SNAP_MM,
        },
        "certified_polygon_original_plane_residuals_mm": [
            float(value) for value in (points - plane_origin) @ normal
        ],
        "construction_region_residuals": component_records,
        "genuine_millimetre_scale_gaps_are_not_snapped": True,
    }


def _polygon_components(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon):
        return [item for item in geometry.geoms if not item.is_empty]
    if isinstance(geometry, GeometryCollection):
        return [item for child in geometry.geoms for item in _polygon_components(child)]
    return []


def _component_boundary_residual(component: Polygon, counterpart: Any) -> float:
    coordinates = list(component.exterior.coords)
    coordinates.extend(point for ring in component.interiors for point in ring.coords)
    return max(
        (
            float(Point(float(x), float(y)).distance(counterpart.boundary))
            for x, y in coordinates
        ),
        default=0.0,
    )


def _projection_residual(
    expected: Any,
    actual: Any,
    requested_tolerance_mm: float,
) -> dict[str, Any]:
    bounds = np.asarray([expected.bounds, actual.bounds], dtype=np.float64)
    resolution = max(
        requested_tolerance_mm,
        _numerical_length_resolution_mm(bounds),
    )
    components = []
    for kind, difference, counterpart in (
        ("missing", expected.difference(actual), actual),
        ("extra", actual.difference(expected), expected),
    ):
        for ordinal, component in enumerate(_polygon_components(difference), start=1):
            boundary_residual = _component_boundary_residual(component, counterpart)
            sliver_area_bound = resolution * float(component.length) + math.pi * resolution**2
            sub_resolution = (
                boundary_residual <= resolution + LENGTH_TOLERANCE_MM
                and float(component.area) <= sliver_area_bound
            )
            components.append(
                {
                    "kind": kind,
                    "ordinal": ordinal,
                    "area_mm2": float(component.area),
                    "perimeter_mm": float(component.length),
                    "maximum_boundary_residual_mm": boundary_residual,
                    "sub_resolution_sliver": sub_resolution,
                }
            )
    meaningful = [item for item in components if not item["sub_resolution_sliver"]]
    meaningful_area = sum(item["area_mm2"] for item in meaningful)
    meaningful_boundary = max(
        (item["maximum_boundary_residual_mm"] for item in meaningful),
        default=0.0,
    )
    area_tolerance = max(
        AREA_TOLERANCE_MM2,
        resolution * float(expected.length + actual.length),
    )
    raw_symmetric_difference = float(actual.symmetric_difference(expected).area)
    raw_hausdorff = float(actual.hausdorff_distance(expected))
    passed = meaningful_area <= area_tolerance and meaningful_boundary <= resolution + LENGTH_TOLERANCE_MM
    return {
        "raw_symmetric_difference_area_mm2": raw_symmetric_difference,
        "normalized_symmetric_difference_area_mm2": meaningful_area,
        "area_tolerance_mm2": area_tolerance,
        "raw_hausdorff_distance_mm": raw_hausdorff,
        "maximum_meaningful_boundary_residual_mm": meaningful_boundary,
        "requested_tolerance_mm": requested_tolerance_mm,
        "effective_numerical_resolution_mm": resolution,
        "component_residuals": components,
        "sub_resolution_sliver_count": sum(item["sub_resolution_sliver"] for item in components),
        "meaningful_component_count": len(meaningful),
        "status": "pass" if passed else "fail",
    }


def _surface_coverage(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    field: str,
) -> tuple[float, float, dict[str, Any]]:
    origin, u_axis, v_axis = _plane_frame(points, field)
    normal = np.cross(u_axis, v_axis)
    expected = _polygon(_project(points, origin, u_axis, v_axis), field)
    triangles = []
    for face in mesh.faces:
        triangle = np.asarray(mesh.vertices[face], dtype=np.float64)
        if np.max(np.abs((triangle - origin) @ normal)) <= LENGTH_TOLERANCE_MM:
            projected = Polygon(_project(triangle, origin, u_axis, v_axis))
            if projected.is_valid and projected.area > AREA_TOLERANCE_MM2:
                triangles.append(projected)
    surface = unary_union(triangles) if triangles else Polygon()
    covered = surface.intersection(expected) if not surface.is_empty else surface
    residual = _projection_residual(expected, covered, 0.0)
    missing_area = sum(
        item["area_mm2"]
        for item in residual["component_residuals"]
        if item["kind"] == "missing" and not item["sub_resolution_sliver"]
    )
    return float(expected.area), missing_area, residual


def _contact_area(left: trimesh.Trimesh, right: trimesh.Trimesh) -> float:
    area = 0.0
    for left_index, left_face in enumerate(left.faces):
        left_triangle = np.asarray(left.vertices[left_face], dtype=np.float64)
        normal = np.asarray(left.face_normals[left_index], dtype=np.float64)
        origin = left_triangle[0]
        u_axis = left_triangle[1] - origin
        length = np.linalg.norm(u_axis)
        if length <= LENGTH_TOLERANCE_MM:
            continue
        u_axis /= length
        v_axis = np.cross(normal, u_axis)
        left_polygon = Polygon(_project(left_triangle, origin, u_axis, v_axis))
        for right_index, right_face in enumerate(right.faces):
            if float(np.dot(normal, right.face_normals[right_index])) > -1.0 + 1e-8:
                continue
            right_triangle = np.asarray(right.vertices[right_face], dtype=np.float64)
            if np.max(np.abs((right_triangle - origin) @ normal)) > LENGTH_TOLERANCE_MM:
                continue
            right_polygon = Polygon(_project(right_triangle, origin, u_axis, v_axis))
            intersection = left_polygon.intersection(right_polygon)
            if not intersection.is_empty:
                area += float(intersection.area)
    return area


def _interface_records(
    interfaces: Sequence[Mapping[str, Any]],
    components: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    records = []
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen = set()
    for interface in sorted(interfaces, key=lambda item: str(item.get("id"))):
        interface_id = str(interface.get("id"))
        if not interface_id or interface_id == "None" or interface_id in seen:
            raise ValueError("shared interface ids must be present and unique")
        seen.add(interface_id)
        refs = sorted(map(str, interface.get("component_refs", [])))
        if len(refs) != 2 or refs[0] == refs[1] or any(item not in components for item in refs):
            raise ValueError(f"{interface_id} must reference two distinct known components")
        points = np.asarray(interface.get("polygon_xyz_mm"), dtype=np.float64)
        area, left_missing, _ = _surface_coverage(
            components[refs[0]]["mesh"], points, interface_id
        )
        _, right_missing, _ = _surface_coverage(
            components[refs[1]]["mesh"], points, interface_id
        )
        tolerance = max(AREA_TOLERANCE_MM2, 1e-8 * area)
        if left_missing > tolerance or right_missing > tolerance:
            raise ValueError(f"{interface_id} is not a shared surface of both components")
        record = {
            "id": interface_id,
            "component_refs": refs,
            "polygon_xyz_mm": points.tolist(),
            "area_mm2": area,
            "left_missing_area_mm2": left_missing,
            "right_missing_area_mm2": right_missing,
            "evidence_refs": _evidence(interface.get("evidence_refs"), interface_id),
            "status": "pass",
        }
        records.append(record)
        by_pair.setdefault(tuple(refs), []).append(record)
    return records, by_pair


def _pair_validation(
    components: Mapping[str, dict[str, Any]],
    interfaces_by_pair: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    component_ids = sorted(components)
    for position, left_id in enumerate(component_ids):
        for right_id in component_ids[position + 1 :]:
            left = components[left_id]
            right = components[right_id]
            intersection = left["manifold"] ^ right["manifold"]
            overlap = 0.0 if intersection.is_empty() else float(intersection.volume())
            overlap_tolerance = max(
                1.0,
                1e-8
                * min(
                    left["analytic_volume"]["value_mm3"],
                    right["analytic_volume"]["value_mm3"],
                ),
            )
            if overlap > overlap_tolerance:
                raise ValueError(f"{left_id} and {right_id} have unintended volumetric overlap")
            contact_area = _contact_area(left["mesh"], right["mesh"])
            supplied = interfaces_by_pair.get((left_id, right_id), [])
            supplied_area = sum(item["area_mm2"] for item in supplied)
            interface_tolerance = max(AREA_TOLERANCE_MM2, 1e-8 * max(contact_area, supplied_area, 1.0))
            if contact_area > interface_tolerance and not supplied:
                raise ValueError(f"{left_id} and {right_id} have an unrecorded shared interface")
            if supplied and abs(contact_area - supplied_area) > interface_tolerance:
                raise ValueError(f"{left_id} and {right_id} shared interface area disagrees with the meshes")
            records.append(
                {
                    "component_refs": [left_id, right_id],
                    "overlap_volume_mm3": overlap,
                    "overlap_tolerance_mm3": overlap_tolerance,
                    "contact_area_mm2": contact_area,
                    "interface_refs": [item["id"] for item in supplied],
                    "status": "pass",
                }
            )
    return records


def _polygons_2d(value: Any, field: str):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} requires at least one polygon")
    polygons = []
    for index, points in enumerate(value):
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1:] != (2,) or len(array) < 3 or not np.all(np.isfinite(array)):
            raise ValueError(f"{field}[{index}] is invalid")
        polygons.append(_polygon(array, f"{field}[{index}]"))
    merged = unary_union(polygons)
    if not merged.is_valid or merged.area <= AREA_TOLERANCE_MM2:
        raise ValueError(f"{field} does not form a valid finite-area projection")
    return merged


def _mesh_projection(mesh: trimesh.Trimesh, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray):
    polygons = []
    for face in mesh.faces:
        projected = Polygon(_project(np.asarray(mesh.vertices[face]), origin, u_axis, v_axis))
        if projected.is_valid and projected.area > AREA_TOLERANCE_MM2:
            polygons.append(projected)
    return unary_union(polygons) if polygons else Polygon()


def _view_records(
    supplied_views: Sequence[Mapping[str, Any]],
    components: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(supplied_views) < 2:
        raise ValueError("multi-component solid requires at least two supplied views")
    records = []
    normals = []
    seen = set()
    for view in sorted(supplied_views, key=lambda item: str(item.get("id"))):
        view_id = str(view.get("id"))
        if not view_id or view_id == "None" or view_id in seen:
            raise ValueError("supplied view ids must be present and unique")
        seen.add(view_id)
        origin = _vector(view.get("origin_xyz_mm"), f"{view_id}.origin_xyz_mm")
        u_axis = _vector(view.get("u_axis_xyz"), f"{view_id}.u_axis_xyz")
        v_axis = _vector(view.get("v_axis_xyz"), f"{view_id}.v_axis_xyz")
        if not math.isclose(float(np.linalg.norm(u_axis)), 1.0, abs_tol=1e-9):
            raise ValueError(f"{view_id} u axis is not unit length")
        if not math.isclose(float(np.linalg.norm(v_axis)), 1.0, abs_tol=1e-9):
            raise ValueError(f"{view_id} v axis is not unit length")
        if not math.isclose(float(np.dot(u_axis, v_axis)), 0.0, abs_tol=1e-9):
            raise ValueError(f"{view_id} projection axes are not orthogonal")
        normals.append(np.cross(u_axis, v_axis))
        projections = view.get("component_projections")
        if not isinstance(projections, (list, tuple)):
            raise ValueError(f"{view_id} has no component projections")
        by_component: dict[str, Mapping[str, Any]] = {}
        for projection in projections:
            component_ref = str(projection.get("component_ref"))
            if component_ref in by_component:
                raise ValueError(f"{view_id} has duplicate projections for {component_ref}")
            by_component[component_ref] = projection
        if set(by_component) != set(components):
            raise ValueError(f"{view_id} must supply exactly one projection for every component")
        projection_records = []
        for component_id in sorted(components):
            projection = by_component[component_id]
            expected = _polygons_2d(
                projection.get("polygons_uv_mm"),
                f"{view_id}.{component_id}.polygons_uv_mm",
            )
            actual = _mesh_projection(components[component_id]["mesh"], origin, u_axis, v_axis)
            tolerance = float(projection.get("tolerance_mm", view.get("tolerance_mm", 0.0)))
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError(f"{view_id}.{component_id} has an invalid reprojection tolerance")
            area_tolerance = max(
                AREA_TOLERANCE_MM2,
                tolerance * float(expected.length + actual.length),
            )
            symmetric_difference = float(actual.symmetric_difference(expected).area)
            hausdorff = float(actual.hausdorff_distance(expected))
            if hausdorff > tolerance + LENGTH_TOLERANCE_MM or symmetric_difference > area_tolerance:
                raise ValueError(f"{view_id}.{component_id} supplied-view reprojection failed")
            projection_records.append(
                {
                    "component_ref": component_id,
                    "expected_area_mm2": float(expected.area),
                    "actual_area_mm2": float(actual.area),
                    "symmetric_difference_area_mm2": symmetric_difference,
                    "area_tolerance_mm2": area_tolerance,
                    "hausdorff_distance_mm": hausdorff,
                    "tolerance_mm": tolerance,
                    "evidence_refs": _evidence(
                        projection.get("evidence_refs"),
                        f"{view_id}.{component_id}",
                    ),
                    "status": "pass",
                }
            )
        records.append(
            {
                "id": view_id,
                "projection_role": view.get("projection_role"),
                "origin_xyz_mm": origin.tolist(),
                "u_axis_xyz": u_axis.tolist(),
                "v_axis_xyz": v_axis.tolist(),
                "normal_axis_xyz": normals[-1].tolist(),
                "component_projections": projection_records,
                "evidence_refs": _evidence(view.get("evidence_refs"), view_id),
                "status": "pass",
            }
        )
    if not any(np.linalg.norm(np.cross(left, right)) > 1e-8 for index, left in enumerate(normals) for right in normals[index + 1 :]):
        raise ValueError("supplied views do not contain two non-parallel projection normals")
    return records


def _normalized_input(
    components: Sequence[Mapping[str, Any]],
    interfaces: Sequence[Mapping[str, Any]],
    supplied_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    views = []
    for view in sorted(deepcopy(list(supplied_views)), key=lambda item: str(item.get("id"))):
        if isinstance(view.get("component_projections"), list):
            view["component_projections"].sort(key=lambda item: str(item.get("component_ref")))
        views.append(view)
    return {
        "components": sorted(deepcopy(list(components)), key=lambda item: str(item.get("id"))),
        "interfaces": sorted(deepcopy(list(interfaces)), key=lambda item: str(item.get("id"))),
        "supplied_views": views,
    }


def solve_multi_component_solid(
    components: Sequence[Mapping[str, Any]],
    interfaces: Sequence[Mapping[str, Any]],
    supplied_views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one explicitly positioned solid assembly or abstain by error."""

    normalized = _normalized_input(components, interfaces, supplied_views)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    component_records: dict[str, dict[str, Any]] = {}
    for component in normalized["components"]:
        record = _component_mesh(component)
        if record["id"] in component_records:
            raise ValueError("component ids must be unique")
        component_records[record["id"]] = record
    if not component_records:
        raise ValueError("positioned solid requires at least one component")
    transform_ids = [item["transform"]["id"] for item in component_records.values()]
    if len(set(transform_ids)) != len(transform_ids):
        raise ValueError("physical component transform ids must be unique")

    interface_records, interfaces_by_pair = _interface_records(normalized["interfaces"], component_records)
    pair_records = _pair_validation(component_records, interfaces_by_pair)
    view_records = _view_records(normalized["supplied_views"], component_records)

    vertices = []
    triangles = []
    mesh_components = []
    for component_id in sorted(component_records):
        record = component_records[component_id]
        base = len(vertices)
        local_vertices = np.asarray(record["mesh"].vertices).tolist()
        local_faces = np.asarray(record["mesh"].faces).tolist()
        vertices.extend(local_vertices)
        triangles.extend([[base + int(index) for index in face] for face in local_faces])
        mesh_components.append(
            {
                "component_ref": component_id,
                "vertex_start": base,
                "vertex_count": len(local_vertices),
                "triangle_start": len(triangles) - len(local_faces),
                "triangle_count": len(local_faces),
                "transform": record["transform"],
            }
        )

    analytic_total = sum(item["analytic_volume"]["value_mm3"] for item in component_records.values())
    mesh_total = sum(item["validation"]["mesh_volume_mm3"] for item in component_records.values())
    evidence_refs = sorted(
        {
            *(
                ref
                for item in component_records.values()
                for ref in item["evidence_refs"]
            ),
            *(ref for item in interface_records for ref in item["evidence_refs"]),
            *(ref for view in view_records for ref in view["evidence_refs"]),
            *(
                ref
                for view in view_records
                for projection in view["component_projections"]
                for ref in projection["evidence_refs"]
            ),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "generic_multi_component_solid_kernel",
        "status": "accepted",
        "input_sha256": sha256(encoded).hexdigest(),
        "components": [
            {
                "id": component_id,
                "transform": component_records[component_id]["transform"],
                "analytic_volume": component_records[component_id]["analytic_volume"],
                "validation": component_records[component_id]["validation"],
                "evidence_refs": component_records[component_id]["evidence_refs"],
            }
            for component_id in sorted(component_records)
        ],
        "shared_interfaces": interface_records,
        "pair_validations": pair_records,
        "supplied_view_reprojections": view_records,
        "volume_validation": {
            "analytic_component_sum_mm3": analytic_total,
            "mesh_component_sum_mm3": mesh_total,
            "residual_mm3": mesh_total - analytic_total,
            "status": "pass",
        },
        "assembly_mesh": {
            "vertices_xyz_mm": vertices,
            "triangles": triangles,
            "components": mesh_components,
            "validation": {
                "component_count": len(component_records),
                "vertex_count": len(vertices),
                "triangle_count": len(triangles),
                "boundary_edge_count": 0,
                "separately_watertight": True,
                "unintended_overlap_count": 0,
            },
        },
        "audit": {
            "evidence_refs": evidence_refs,
            "component_transform_refs": [item["transform"]["id"] for item in component_records.values()],
            "shared_interface_refs": [item["id"] for item in interface_records],
            "supplied_view_refs": [item["id"] for item in view_records],
        },
        "contract": {
            "physical_transforms_required": True,
            "presentation_offsets_are_physical_placement": False,
            "components_validated_separately": True,
            "all_finite_area_contacts_are_recorded": True,
            "analytic_volume_agreement_required": True,
            "two_non_parallel_reprojections_required": True,
            "quantity_eligible": True,
            "schedule_values_used": False,
        },
    }


def directed_surface_evidence_errors(view):
    """Missing/invalid expectations are unresolved, not geometric negatives."""
    evidence = view.get("required_direction_evidence")
    if evidence is None:
        return []
    if not isinstance(evidence, Mapping) or evidence.get("state") != "accepted" or not evidence.get("paths"):
        return ["supplied_direction_evidence_unresolved"]
    errors, seen = [], set()
    for path in evidence["paths"]:
        try:
            path_id = str(path.get("id") or "")
            if not path_id or path_id in seen:
                raise ValueError("directed_path_identity_invalid")
            seen.add(path_id)
            _evidence(path.get("evidence_refs"), path_id)
            if path.get("independent_of_candidate_geometry") is not True:
                raise ValueError("direction_evidence_not_independent")
            if path.get("elevation_order") != "nondecreasing":
                raise ValueError("unsupported_surface_elevation_order")
            samples, runs = path.get("samples") or [], path.get("band_runs") or []
            points = np.asarray([s.get("uv_mm") for s in samples], dtype=float)
            if len(samples) < 2 or points.shape != (len(samples), 2) or not np.all(np.isfinite(points)) or not runs:
                raise ValueError("directed_surface_samples_invalid")
            axis = _vector(path.get("elevation_axis_xyz"), path_id)
            u_axis = _vector(view.get("u_axis_xyz"), "directed_view.u_axis_xyz")
            v_axis = _vector(view.get("v_axis_xyz"), "directed_view.v_axis_xyz")
            if not (math.isclose(float(np.linalg.norm(axis)), 1., abs_tol=1e-9)
                    and abs(float(axis @ u_axis)) < 1e-9 and abs(float(axis @ v_axis)) < 1e-9):
                raise ValueError("surface_elevation_axis_not_projection_normal")
            tolerance = float(path.get("tolerance_mm", -1))
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("invalid_surface_elevation_tolerance")
            seen_indices, seen_bands = set(), set()
            for run in runs:
                indices = run.get("sample_indices") or []
                if (len(indices) < 2 or any(not isinstance(i, int) or i < 0 or i >= len(samples) for i in indices)
                    or indices != sorted(set(indices)) or set(indices) & seen_indices
                    or not run.get("band_ref") or run["band_ref"] in seen_bands
                    or any(samples[i].get("band_ref") != run["band_ref"] for i in indices)):
                    raise ValueError("directed_surface_run_binding_invalid")
                seen_indices.update(indices)
                seen_bands.add(run["band_ref"])
            if seen_indices != {i for i, s in enumerate(samples) if s.get("band_ref")}:
                raise ValueError("directed_surface_run_coverage_incomplete")
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append(str(error))
    return errors


def _surface_elevations(mesh, points, origin, u_axis, v_axis, axis):
    xyz = np.asarray(mesh.vertices) - origin
    uv = np.column_stack((xyz @ u_axis, xyz @ v_axis))
    triangles = uv[np.asarray(mesh.faces)]
    heights = (xyz @ axis)[np.asarray(mesh.faces)]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1 = b - a, c - a
    determinants = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    nonvertical = np.abs(determinants) > 1e-10
    result = []
    for point in points:
        offset = point - a
        alpha = np.zeros(len(a))
        beta = np.zeros(len(a))
        alpha[nonvertical] = (offset[:, 0] * v1[:, 1] - offset[:, 1] * v1[:, 0])[nonvertical] / determinants[nonvertical]
        beta[nonvertical] = (v0[:, 0] * offset[:, 1] - v0[:, 1] * offset[:, 0])[nonvertical] / determinants[nonvertical]
        hit = nonvertical & (alpha >= -1e-8) & (beta >= -1e-8) & (alpha + beta <= 1 + 1e-8)
        values = heights[:, 0] + alpha * (heights[:, 1] - heights[:, 0]) + beta * (heights[:, 2] - heights[:, 0])
        result.append(float(np.max(values[hit])) if np.any(hit) else None)
    return result


def _directed_surface_path_records(view, mesh, origin, u_axis, v_axis, region_records=None):
    """Replay supplied directed stations against the exterior upper surface.

    Native observations supply station coordinates and order. Triangle
    barycentrics measure actual elevation; no generated projection can supply
    the expected direction. Every station and adjacent residual is retained.
    """
    evidence = view.get("required_direction_evidence")
    if evidence is None:
        return [], []
    evidence_errors = directed_surface_evidence_errors({**view, "u_axis_xyz": u_axis, "v_axis_xyz": v_axis})
    if evidence_errors:
        return [], evidence_errors
    records, errors = [], []
    for path in evidence["paths"]:
        path_errors = []
        path_id = str(path.get("id") or "")
        try:
            _evidence(path.get("evidence_refs"), path_id)
            if path.get("independent_of_candidate_geometry") is not True:
                raise ValueError("direction_evidence_not_independent")
            if path.get("elevation_order") != "nondecreasing":
                raise ValueError("unsupported_surface_elevation_order")
            axis = _vector(path.get("elevation_axis_xyz"), path_id)
            if not (math.isclose(float(np.linalg.norm(axis)), 1., abs_tol=1e-9)
                    and abs(float(axis @ u_axis)) < 1e-9 and abs(float(axis @ v_axis)) < 1e-9):
                raise ValueError("surface_elevation_axis_not_projection_normal")
            tolerance = float(path.get("tolerance_mm", -1))
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("invalid_surface_elevation_tolerance")
            samples = path.get("samples") or []
            if len(samples) < 2 or not path.get("band_runs"):
                raise ValueError("directed_surface_samples_missing")
            points = np.asarray([s.get("uv_mm") for s in samples], dtype=float)
            if points.shape != (len(samples), 2) or not np.all(np.isfinite(points)):
                raise ValueError("invalid_directed_surface_station")
            values = _surface_elevations(mesh, points, origin, u_axis, v_axis, axis)
            region_values = {
                ref: _surface_elevations(region["mesh"], points, origin, u_axis, v_axis, axis)
                for ref, region in (region_records or {}).items()
            }
            sample_records = []
            for index, (sample, point, value) in enumerate(zip(samples, points, values)):
                owners = sorted(ref for ref, heights in region_values.items()
                                if value is not None and heights[index] is not None and abs(heights[index] - value) <= max(tolerance, 1e-5))
                sample_records.append({"uv_mm": point.tolist(), "band_ref": sample.get("band_ref"),
                                       "surface_elevation_mm": value, "construction_region_refs": owners,
                                       "status": "pass" if value is not None else "uncovered"})
            if any(s["surface_elevation_mm"] is None for s in sample_records):
                path_errors.append("directed_surface_station_uncovered")
            residuals = []
            for index, (left, right) in enumerate(zip(sample_records, sample_records[1:])):
                delta = (right["surface_elevation_mm"] - left["surface_elevation_mm"]
                         if left["surface_elevation_mm"] is not None and right["surface_elevation_mm"] is not None else None)
                passed = delta is not None and delta >= -tolerance
                residuals.append({"from_sample": index, "to_sample": index + 1, "rise_mm": delta,
                                  "status": "pass" if passed else "fail"})
            if any(r["status"] != "pass" for r in residuals):
                path_errors.append("directed_surface_elevation_order_contradiction")
            runs = []
            for run in path["band_runs"]:
                indices = run.get("sample_indices") or []
                if len(indices) < 2 or any(not isinstance(i, int) or i < 0 or i >= len(samples) for i in indices):
                    raise ValueError("directed_surface_run_indices_invalid")
                if any(samples[i].get("band_ref") != run.get("band_ref") for i in indices):
                    raise ValueError("directed_surface_run_band_mismatch")
                start, end = [sample_records[i]["surface_elevation_mm"] for i in (indices[0], indices[-1])]
                rise = end - start if start is not None and end is not None else None
                passed = rise is not None and rise > tolerance
                runs.append({"band_ref": run["band_ref"], "sample_indices": indices,
                             "construction_region_refs": sorted({ref for i in indices for ref in sample_records[i]["construction_region_refs"]}),
                             "total_rise_mm": rise, "status": "pass" if passed else "fail"})
            if any(r["status"] != "pass" for r in runs):
                path_errors.append("directed_surface_band_ascent_contradiction")
            record = {"path_ref": path_id, "samples": sample_records, "adjacent_residuals": residuals,
                      "band_runs": runs, "tolerance_mm": tolerance, "evidence_refs": path["evidence_refs"]}
        except (ValueError, TypeError, KeyError) as error:
            path_errors.append(str(error))
            record = {"path_ref": path_id}
        record.update(status="fail" if path_errors else "pass", errors=path_errors)
        records.append(record)
        errors.extend(f"{path_id}:{error}" for error in path_errors)
    return records, errors


def _same_object_view_records(
    supplied_views: Sequence[Mapping[str, Any]],
    mesh: trimesh.Trimesh,
    region_records: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if len(supplied_views) < 2:
        raise ValueError("constructive solid requires at least two supplied views")
    records = []
    errors = []
    normals = []
    seen = set()
    for view in sorted(supplied_views, key=lambda item: str(item.get("id"))):
        view_id = str(view.get("id"))
        if not view_id or view_id == "None" or view_id in seen:
            raise ValueError("supplied view ids must be present and unique")
        seen.add(view_id)
        origin = _vector(view.get("origin_xyz_mm"), f"{view_id}.origin_xyz_mm")
        u_axis = _vector(view.get("u_axis_xyz"), f"{view_id}.u_axis_xyz")
        v_axis = _vector(view.get("v_axis_xyz"), f"{view_id}.v_axis_xyz")
        if not math.isclose(float(np.linalg.norm(u_axis)), 1.0, abs_tol=1e-9):
            raise ValueError(f"{view_id} u axis is not unit length")
        if not math.isclose(float(np.linalg.norm(v_axis)), 1.0, abs_tol=1e-9):
            raise ValueError(f"{view_id} v axis is not unit length")
        if not math.isclose(float(np.dot(u_axis, v_axis)), 0.0, abs_tol=1e-9):
            raise ValueError(f"{view_id} projection axes are not orthogonal")
        normal = np.cross(u_axis, v_axis)
        normals.append(normal)
        expected = _polygons_2d(view.get("polygons_uv_mm"), f"{view_id}.polygons_uv_mm")
        actual = _mesh_projection(mesh, origin, u_axis, v_axis)
        tolerance = float(view.get("tolerance_mm", 0.0))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{view_id} has an invalid reprojection tolerance")
        residual = _projection_residual(expected, actual, tolerance)
        if residual["status"] != "pass":
            errors.append(f"{view_id} supplied-view reprojection failed")
        directed_records, directed_errors = _directed_surface_path_records(view, mesh, origin, u_axis, v_axis, region_records)
        errors.extend(directed_errors)
        records.append(
            {
                "id": view_id,
                "projection_role": view.get("projection_role"),
                "origin_xyz_mm": origin.tolist(),
                "u_axis_xyz": u_axis.tolist(),
                "v_axis_xyz": v_axis.tolist(),
                "normal_axis_xyz": normal.tolist(),
                "expected_area_mm2": float(expected.area),
                "actual_area_mm2": float(actual.area),
                "symmetric_difference_area_mm2": residual[
                    "normalized_symmetric_difference_area_mm2"
                ],
                "area_tolerance_mm2": residual["area_tolerance_mm2"],
                "hausdorff_distance_mm": residual["raw_hausdorff_distance_mm"],
                "tolerance_mm": tolerance,
                "precision_aware_residual": residual,
                "directed_surface_path_reprojections": directed_records,
                "evidence_refs": _evidence(view.get("evidence_refs"), view_id),
                "status": "fail" if directed_errors else residual["status"],
            }
        )
    if not any(
        np.linalg.norm(np.cross(left, right)) > 1e-8
        for index, left in enumerate(normals)
        for right in normals[index + 1 :]
    ):
        raise ValueError("supplied views do not contain two non-parallel projection normals")
    return records, errors


def solve_same_object_constructive_solid(
    physical_object_scope: Mapping[str, Any],
    construction_regions: Sequence[Mapping[str, Any]],
    internal_seams: Sequence[Mapping[str, Any]],
    analytic_union_volume: Mapping[str, Any],
    supplied_views: Sequence[Mapping[str, Any]],
    *,
    separate_object_interfaces: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Union several construction regions into one validated physical object.

    Construction regions are calculation aids, not additive physical
    components.  Their coincident faces or declared overlap volumes disappear
    in the boolean union.  Separate-object interfaces remain on the existing
    multi-component path and cannot be used to authorize this union.
    """

    scope_id = str(physical_object_scope.get("id") or "")
    if not scope_id or scope_id == "None":
        raise ValueError("physical_object_scope requires a stable id")
    if physical_object_scope.get("record_type") != "physical_object_scope":
        raise ValueError("same-object union requires a physical_object_scope record")
    if physical_object_scope.get("state") not in {
        "resolved",
        "resolved_relative",
        "resolved_search_hypothesis",
    }:
        raise ValueError("physical_object_scope is not resolved")
    scope_evidence = _evidence(physical_object_scope.get("evidence_refs"), scope_id)
    if len(construction_regions) < 2:
        raise ValueError("same-object union requires at least two construction regions")

    region_records: dict[str, dict[str, Any]] = {}
    normalized_regions = []
    for region in construction_regions:
        region_id = str(region.get("id") or "")
        if not region_id or region_id == "None" or region_id in region_records:
            raise ValueError("construction region ids must be present and unique")
        if region.get("record_type") != "construction_region":
            raise ValueError(f"{region_id} is not a construction_region")
        if str(region.get("physical_object_scope_ref")) != scope_id:
            raise ValueError(f"{region_id} is outside the physical object scope")
        if region.get("state") != "resolved":
            raise ValueError(f"{region_id} construction geometry is unresolved")
        component = _component_mesh(region)
        region_records[region_id] = component
        normalized_regions.append(
            {
                "id": region_id,
                "record_type": "construction_region",
                "physical_object_scope_ref": scope_id,
                "construction_role": str(region.get("construction_role") or "unspecified"),
                "state": "resolved",
                "transform": component["transform"],
                "analytic_volume": component["analytic_volume"],
                "validation": component["validation"],
                "evidence_refs": component["evidence_refs"],
                "temporary_bounded_region_for_union": True,
                "independent_physical_component_closure": False,
                "quantity_eligible": False,
            }
        )

    region_ids = set(region_records)
    seam_records = []
    validation_errors: list[str] = []
    residual_diagnostics: dict[str, Any] = {
        "internal_seams": [],
        "supplied_view_reprojections": [],
    }
    seams_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_seams = set()
    for seam in internal_seams:
        seam_id = str(seam.get("id") or "")
        if not seam_id or seam_id == "None" or seam_id in seen_seams:
            raise ValueError("internal seam ids must be present and unique")
        seen_seams.add(seam_id)
        if seam.get("record_type") != "internal_seam":
            raise ValueError(f"{seam_id} is not an internal_seam")
        if str(seam.get("physical_object_scope_ref")) != scope_id:
            raise ValueError(f"{seam_id} is outside the physical object scope")
        refs = sorted(map(str, seam.get("construction_region_refs", []) or []))
        if len(refs) != 2 or refs[0] == refs[1] or any(ref not in region_ids for ref in refs):
            raise ValueError(f"{seam_id} must reference two construction regions")
        seam_kind = str(seam.get("seam_kind") or "")
        if seam_kind not in {"coincident_face", "overlap_union"}:
            raise ValueError(f"{seam_id} has an unsupported seam_kind")
        record = {
            "id": seam_id,
            "record_type": "internal_seam",
            "physical_object_scope_ref": scope_id,
            "construction_region_refs": refs,
            "seam_kind": seam_kind,
            "source_cap_classification": seam.get("source_cap_classification"),
            "required_contact_semantics": seam.get("required_contact_semantics"),
            "source_interface_option_ref": seam.get("source_interface_option_ref"),
            "state": "resolved",
            "evidence_refs": _evidence(seam.get("evidence_refs"), seam_id),
            "quantity_eligible": False,
        }
        if seam_kind == "coincident_face":
            points = np.asarray(seam.get("polygon_xyz_mm"), dtype=np.float64)
            points, snap_certificate = _snap_regions_to_certified_seam(
                seam_id,
                points,
                [region_records[refs[0]], region_records[refs[1]]],
            )
            area, left_missing, left_coverage_residual = _surface_coverage(
                region_records[refs[0]]["mesh"], points, seam_id
            )
            _, right_missing, right_coverage_residual = _surface_coverage(
                region_records[refs[1]]["mesh"], points, seam_id
            )
            tolerance = max(AREA_TOLERANCE_MM2, 1e-8 * area)
            seam_status = "pass"
            if left_missing > tolerance or right_missing > tolerance:
                seam_status = "fail"
                validation_errors.append(
                    f"{seam_id} is not a shared construction-region surface"
                )
            record.update(
                {
                    "polygon_xyz_mm": points.tolist(),
                    "area_mm2": area,
                    "left_missing_area_mm2": left_missing,
                    "right_missing_area_mm2": right_missing,
                    "area_tolerance_mm2": tolerance,
                    "seam_plane_snap_certificate": snap_certificate,
                    "left_surface_coverage_residual": left_coverage_residual,
                    "right_surface_coverage_residual": right_coverage_residual,
                    "status": seam_status,
                }
            )
            residual_diagnostics["internal_seams"].append(
                {
                    "internal_seam_ref": seam_id,
                    "construction_region_refs": refs,
                    "area_mm2": area,
                    "left_missing_area_mm2": left_missing,
                    "right_missing_area_mm2": right_missing,
                    "area_tolerance_mm2": tolerance,
                    "seam_plane_snap_certificate": snap_certificate,
                    "left_surface_coverage_residual": left_coverage_residual,
                    "right_surface_coverage_residual": right_coverage_residual,
                    "status": seam_status,
                }
            )
        seam_records.append(record)
        seams_by_pair.setdefault(tuple(refs), []).append(record)

    separate_interface_records = []
    seen_interface_ids = set()
    for interface in separate_object_interfaces:
        interface_id = str(interface.get("id") or "")
        if not interface_id or interface_id == "None" or interface_id in seen_interface_ids:
            raise ValueError("separate_object_interface ids must be present and unique")
        seen_interface_ids.add(interface_id)
        if str(interface.get("record_type")) != "separate_object_interface":
            raise ValueError("separate-object relations require separate_object_interface records")
        refs = set(map(str, interface.get("object_scope_refs", []) or []))
        if scope_id not in refs or len(refs) != 2:
            raise ValueError("separate_object_interface must join two distinct physical scopes")
        separate_interface_records.append(
            {
                **deepcopy(dict(interface)),
                "id": interface_id,
                "record_type": "separate_object_interface",
                "object_scope_refs": sorted(refs),
                "evidence_refs": _evidence(interface.get("evidence_refs"), interface_id),
                "quantity_eligible": False,
            }
        )

    pair_records = []
    for left_id, right_id in itertools.combinations(sorted(region_ids), 2):
        left, right = region_records[left_id], region_records[right_id]
        intersection = left["manifold"] ^ right["manifold"]
        overlap = 0.0 if intersection.is_empty() else float(intersection.volume())
        overlap_tolerance = max(
            1.0,
            1e-8
            * min(
                left["analytic_volume"]["value_mm3"],
                right["analytic_volume"]["value_mm3"],
            ),
        )
        contact_area = _contact_area(left["mesh"], right["mesh"])
        supplied = seams_by_pair.get((left_id, right_id), [])
        coincident = [item for item in supplied if item["seam_kind"] == "coincident_face"]
        overlap_seams = [item for item in supplied if item["seam_kind"] == "overlap_union"]
        numerical_seam_overlap_tolerance = sum(
            float(item.get("area_mm2") or 0.0)
            * float(
                (item.get("seam_plane_snap_certificate") or {})
                .get("canonical_plane", {})
                .get("snap_tolerance_mm", 0.0)
            )
            for item in coincident
            if item.get("status") == "pass"
        )
        effective_overlap_tolerance = max(
            overlap_tolerance,
            numerical_seam_overlap_tolerance,
        )
        pair_errors = []
        if overlap > effective_overlap_tolerance:
            if len(overlap_seams) != 1 or coincident:
                pair_errors.append(
                    f"{left_id} and {right_id} overlap without one overlap_union seam"
                )
        elif coincident and all(item.get("status") == "pass" for item in coincident):
            # The independently certified surface coverage is authoritative
            # after bounded plane snapping.  A float-backend overlap thinner
            # than that same bound is recorded but not recast as geometry.
            pass
        elif contact_area > AREA_TOLERANCE_MM2:
            supplied_area = sum(float(item["area_mm2"]) for item in coincident)
            tolerance = max(AREA_TOLERANCE_MM2, 1e-8 * max(contact_area, supplied_area))
            if overlap_seams or not coincident or abs(contact_area - supplied_area) > tolerance:
                pair_errors.append(
                    f"{left_id} and {right_id} have an unrecorded internal seam"
                )
        elif supplied:
            pair_errors.append(
                f"{left_id} and {right_id} declare an internal seam without contact or overlap"
            )
        validation_errors.extend(pair_errors)
        pair_records.append(
            {
                "construction_region_refs": [left_id, right_id],
                "overlap_volume_mm3": overlap,
                "overlap_tolerance_mm3": overlap_tolerance,
                "numerical_seam_overlap_tolerance_mm3": numerical_seam_overlap_tolerance,
                "effective_overlap_tolerance_mm3": effective_overlap_tolerance,
                "contact_area_mm2": contact_area,
                "internal_seam_refs": [item["id"] for item in supplied],
                "errors": pair_errors,
                "status": "fail" if pair_errors else "pass",
            }
        )

    union = None
    for region_id in sorted(region_records):
        union = region_records[region_id]["manifold"] if union is None else union + region_records[region_id]["manifold"]
    if union is None or union.is_empty() or union.status() != md.Error.NoError:
        raise ValueError("construction-region union failed")
    raw = union.to_mesh64()
    vertices = np.asarray(raw.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(raw.tri_verts, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        validation_errors.append(
            "constructive union is not a consistently wound watertight solid"
        )
    pieces = mesh.split(only_watertight=False)
    if len(pieces) != 1:
        validation_errors.append(
            "constructive union does not form one connected physical object"
        )
    edge_count, boundary_count, invalid_edge_count = _edge_validation(mesh.faces)
    if boundary_count or invalid_edge_count:
        validation_errors.append("constructive union has boundary or non-manifold edges")

    analytic_value = float(analytic_union_volume.get("value_mm3") or 0.0)
    if not math.isfinite(analytic_value) or analytic_value <= 0:
        raise ValueError("analytic_union_volume must be positive")
    analytic_basis = str(analytic_union_volume.get("basis") or "")
    if not analytic_basis:
        raise ValueError("analytic_union_volume requires a basis")
    analytic_evidence = _evidence(analytic_union_volume.get("evidence_refs"), "analytic_union_volume")
    mesh_volume = float(mesh.volume)
    manifold_volume = float(union.volume())
    volume_tolerance = max(
        1.0, MANIFOLD_RELATIVE_VOLUME_TOLERANCE * analytic_value
    )
    if abs(mesh_volume - analytic_value) > volume_tolerance:
        validation_errors.append("analytic union and union mesh volumes disagree")
    if abs(manifold_volume - analytic_value) > volume_tolerance:
        validation_errors.append("analytic union and manifold volumes disagree")

    view_records, view_errors = _same_object_view_records(supplied_views, mesh, region_records)
    validation_errors.extend(view_errors)
    residual_diagnostics["supplied_view_reprojections"] = deepcopy(view_records)
    residual_diagnostics["region_pair_validations"] = deepcopy(pair_records)
    residual_diagnostics["union_validation"] = {
        "connected_solid_count": len(pieces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edge_count": boundary_count,
        "non_manifold_edge_count": invalid_edge_count,
        "analytic_mesh_residual_mm3": mesh_volume - analytic_value,
        "analytic_manifold_residual_mm3": manifold_volume - analytic_value,
        "volume_tolerance_mm3": volume_tolerance,
    }
    if validation_errors:
        raise SolidKernelValidationError(validation_errors, residual_diagnostics)
    region_additive_sum = sum(item["analytic_volume"]["value_mm3"] for item in region_records.values())
    external_boundary_id = f"external_boundary.{sha256(scope_id.encode('utf-8')).hexdigest()[:16]}"
    evidence_refs = sorted(
        {
            *scope_evidence,
            *analytic_evidence,
            *(ref for item in region_records.values() for ref in item["evidence_refs"]),
            *(ref for item in seam_records for ref in item["evidence_refs"]),
            *(ref for item in separate_interface_records for ref in item["evidence_refs"]),
            *(ref for item in view_records for ref in item["evidence_refs"]),
        }
    )
    normalized = {
        "physical_object_scope": deepcopy(dict(physical_object_scope)),
        "construction_region_ids": sorted(region_ids),
        "internal_seam_ids": sorted(item["id"] for item in seam_records),
        "analytic_union_volume": deepcopy(dict(analytic_union_volume)),
        "supplied_view_ids": sorted(item["id"] for item in view_records),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "generic_same_object_constructive_solid_kernel",
        "status": "accepted",
        "input_sha256": sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "physical_object_scope": {
            "id": scope_id,
            "record_type": "physical_object_scope",
            "state": physical_object_scope["state"],
            "evidence_refs": scope_evidence,
        },
        "construction_regions": sorted(normalized_regions, key=lambda item: item["id"]),
        "internal_seams": sorted(seam_records, key=lambda item: item["id"]),
        "external_boundary": {
            "id": external_boundary_id,
            "record_type": "external_boundary",
            "physical_object_scope_ref": scope_id,
            "state": "resolved",
            "mesh": {
                "vertices_xyz_mm": np.asarray(mesh.vertices).tolist(),
                "triangles": np.asarray(mesh.faces).tolist(),
            },
            "validation": {
                "vertex_count": int(len(mesh.vertices)),
                "triangle_count": int(len(mesh.faces)),
                "edge_count": edge_count,
                "boundary_edge_count": boundary_count,
                "non_manifold_edge_count": invalid_edge_count,
                "connected_solid_count": len(pieces),
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "euler_characteristic": int(mesh.euler_number),
                "bounds_xyz_mm": np.asarray(mesh.bounds).tolist(),
                "status": "pass",
            },
            "evidence_refs": evidence_refs,
            "quantity_eligible": True,
        },
        "separate_object_interfaces": separate_interface_records,
        "region_pair_validations": pair_records,
        "supplied_view_reprojections": view_records,
        "volume_validation": {
            "construction_region_additive_sum_mm3": region_additive_sum,
            "analytic_union_mm3": analytic_value,
            "mesh_union_mm3": mesh_volume,
            "manifold_union_mm3": manifold_volume,
            "internal_overlap_deduction_mm3": region_additive_sum - analytic_value,
            "analytic_mesh_residual_mm3": mesh_volume - analytic_value,
            "tolerance_mm3": volume_tolerance,
            "basis": analytic_basis,
            "evidence_refs": analytic_evidence,
            "status": "pass",
        },
        "evidence_refs": evidence_refs,
        "contract": {
            "one_physical_object_scope_required": True,
            "construction_regions_are_not_additive_components": True,
            "independent_physical_profile_closure_required": False,
            "temporary_region_caps_require_internal_seam_certificate": True,
            "internal_seams_are_quantity_ineligible": True,
            "internal_coincident_faces_removed_by_union": True,
            "one_connected_watertight_union_required": True,
            "analytic_union_mesh_agreement_required": True,
            "two_non_parallel_reprojections_required": True,
            "separate_objects_use_multi_component_contact_path": True,
            "quantity_eligible": True,
            "schedule_values_used": False,
        },
    }
