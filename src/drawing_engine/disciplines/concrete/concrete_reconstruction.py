"""Universal parametric concrete reconstruction from semantic input records.

The engine has no drawing-name branches. It consumes dimensions, placed
corbels, and placed recesses produced by earlier semantic stages. A source
adapter may contain reviewed values while dimension/leader attachment remains
incomplete, but that provenance is carried into the resulting payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import manifold3d as md
import numpy as np
import trimesh


@dataclass(frozen=True)
class CorbelSpec:
    direction_x: int
    projection_mm: float
    depth_mm: float
    taper_bottom_z_mm: float
    rectangular_bottom_z_mm: float
    top_z_mm: float
    status: str


@dataclass(frozen=True)
class RecessSpec:
    center_x_mm: float
    center_z_mm: float
    width_mm: float
    height_mm: float
    depth_mm: float
    face_y: int
    status: str


@dataclass(frozen=True)
class ConcreteModelSpec:
    object_name: str
    width_x_mm: float
    depth_y_mm: float
    height_z_mm: float
    corbels: tuple[CorbelSpec, ...]
    recesses: tuple[RecessSpec, ...]
    declared_volume_m3: float | None
    source_status: str
    evidence: tuple[str, ...]
    unknowns: tuple[str, ...]


def spec_from_semantics(data: dict[str, Any]) -> ConcreteModelSpec:
    """Validate and normalize a drawing-independent semantic model record."""

    shaft = data["shaft"]
    corbels = tuple(
        CorbelSpec(
            direction_x=int(item["direction_x"]),
            projection_mm=float(item["projection_mm"]),
            depth_mm=float(item.get("depth_mm", shaft["depth_y_mm"])),
            taper_bottom_z_mm=float(item["taper_bottom_z_mm"]),
            rectangular_bottom_z_mm=float(item["rectangular_bottom_z_mm"]),
            top_z_mm=float(item["top_z_mm"]),
            status=item.get("status", "unknown"),
        )
        for item in data.get("corbels", [])
    )
    recesses = tuple(
        RecessSpec(
            center_x_mm=float(item["center_x_mm"]),
            center_z_mm=float(item["center_z_mm"]),
            width_mm=float(item["width_mm"]),
            height_mm=float(item["height_mm"]),
            depth_mm=float(item["depth_mm"]),
            face_y=int(item.get("face_y", 1)),
            status=item.get("status", "unknown"),
        )
        for item in data.get("recesses", [])
    )
    spec = ConcreteModelSpec(
        object_name=data["object_name"],
        width_x_mm=float(shaft["width_x_mm"]),
        depth_y_mm=float(shaft["depth_y_mm"]),
        height_z_mm=float(shaft["height_z_mm"]),
        corbels=corbels,
        recesses=recesses,
        declared_volume_m3=(
            None if data.get("declared_volume_m3") is None else float(data["declared_volume_m3"])
        ),
        source_status=data.get("source_status", "unknown"),
        evidence=tuple(data.get("evidence", [])),
        unknowns=tuple(data.get("unknowns", [])),
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: ConcreteModelSpec) -> None:
    if min(spec.width_x_mm, spec.depth_y_mm, spec.height_z_mm) <= 0:
        raise ValueError("shaft dimensions must be positive")
    for corbel in spec.corbels:
        if corbel.direction_x not in {-1, 1}:
            raise ValueError("corbel direction_x must be -1 or +1")
        if not (
            0 <= corbel.taper_bottom_z_mm
            < corbel.rectangular_bottom_z_mm
            < corbel.top_z_mm
            <= spec.height_z_mm
        ):
            raise ValueError("corbel Z levels must be ordered inside the shaft height")
        if min(corbel.projection_mm, corbel.depth_mm) <= 0:
            raise ValueError("corbel dimensions must be positive")
    for recess in spec.recesses:
        if recess.face_y not in {-1, 1}:
            raise ValueError("recess face_y must be -1 or +1")
        if min(recess.width_mm, recess.height_mm, recess.depth_mm) <= 0:
            raise ValueError("recess dimensions must be positive")


def _corbel_profile_xz(spec: ConcreteModelSpec, corbel: CorbelSpec) -> np.ndarray:
    half_width = spec.width_x_mm / 2
    inner = corbel.direction_x * half_width
    outer = corbel.direction_x * (half_width + corbel.projection_mm)
    if corbel.direction_x > 0:
        points = (
            (inner, corbel.taper_bottom_z_mm),
            (outer, corbel.rectangular_bottom_z_mm),
            (outer, corbel.top_z_mm),
            (inner, corbel.top_z_mm),
        )
    else:
        points = (
            (inner, corbel.taper_bottom_z_mm),
            (inner, corbel.top_z_mm),
            (outer, corbel.top_z_mm),
            (outer, corbel.rectangular_bottom_z_mm),
        )
    return np.asarray(points, dtype=np.float64)


def analytic_volume_mm3(spec: ConcreteModelSpec) -> float:
    shaft = spec.width_x_mm * spec.depth_y_mm * spec.height_z_mm
    corbels = sum(
        corbel.projection_mm
        * corbel.depth_mm
        * (
            (corbel.top_z_mm - corbel.rectangular_bottom_z_mm)
            + (corbel.rectangular_bottom_z_mm - corbel.taper_bottom_z_mm) / 2
        )
        for corbel in spec.corbels
    )
    recesses = sum(
        recess.width_mm * recess.height_mm * recess.depth_mm
        for recess in spec.recesses
    )
    return shaft + corbels - recesses


def build_manifold(spec: ConcreteModelSpec) -> md.Manifold:
    """Build a CSG solid in temporary coordinates (X, Z, Y)."""

    solid = md.Manifold.cube((spec.width_x_mm, spec.height_z_mm, spec.depth_y_mm)).translate(
        (-spec.width_x_mm / 2, 0.0, -spec.depth_y_mm / 2)
    )
    for corbel in spec.corbels:
        wing = md.Manifold.extrude(md.CrossSection([_corbel_profile_xz(spec, corbel)]), corbel.depth_mm)
        wing = wing.translate((0.0, 0.0, -corbel.depth_mm / 2))
        solid = solid + wing

    for recess in spec.recesses:
        cutter_y = (
            spec.depth_y_mm / 2 - recess.depth_mm
            if recess.face_y > 0
            else -spec.depth_y_mm / 2 - 1.0
        )
        cutter = md.Manifold.cube(
            (recess.width_mm, recess.height_mm, recess.depth_mm + 1.0)
        ).translate(
            (
                recess.center_x_mm - recess.width_mm / 2,
                recess.center_z_mm - recess.height_mm / 2,
                cutter_y,
            )
        )
        solid = solid - cutter
    if solid.status() != md.Error.NoError:
        raise RuntimeError(f"Manifold3D failed to build {spec.object_name}: {solid.status()}")
    return solid


def engineering_mesh(spec: ConcreteModelSpec) -> tuple[md.Manifold, trimesh.Trimesh]:
    """Return the robust solid and a Z-up millimetre triangle mesh."""

    solid = build_manifold(spec)
    raw = solid.to_mesh64()
    raw_vertices = np.asarray(raw.vert_properties, dtype=np.float64)[:, :3]
    vertices = raw_vertices[:, [0, 2, 1]]
    faces = np.asarray(raw.tri_verts, dtype=np.int64)[:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(f"{spec.object_name} mesh is not a valid closed solid")
    expected = analytic_volume_mm3(spec)
    if not np.isclose(mesh.volume, expected, atol=1e-3):
        raise RuntimeError(f"{spec.object_name} mesh volume {mesh.volume} != {expected}")
    return solid, mesh


def reconstruction_payload(spec: ConcreteModelSpec) -> dict[str, Any]:
    solid, mesh = engineering_mesh(spec)
    analytic = analytic_volume_mm3(spec)
    drawing_volume = analytic / 1_000_000_000
    return {
        "schema_version": "0.2.0",
        "object": spec.object_name,
        "scope": "concrete host and explicitly placed recesses only; reinforcement excluded",
        "source_status": spec.source_status,
        "units": "millimeter",
        "coordinate_system": {
            "origin": "shaft base centre",
            "+X": "configured corbel span direction",
            "+Y": "configured face normal",
            "+Z": "up",
        },
        "operation": "construct shaft; union placed corbel prisms; subtract placed recess boxes",
        "semantic_input": {
            "shaft": {
                "width_x_mm": spec.width_x_mm,
                "depth_y_mm": spec.depth_y_mm,
                "height_z_mm": spec.height_z_mm,
            },
            "corbels": [corbel.__dict__ for corbel in spec.corbels],
            "recesses": [recess.__dict__ for recess in spec.recesses],
        },
        "mesh": {
            "vertex_count": int(len(mesh.vertices)),
            "triangle_count": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "euler_characteristic": int(mesh.euler_number),
            "genus": int(solid.genus()),
            "bounds_xyz_mm": np.asarray(mesh.bounds).tolist(),
        },
        "volume_validation": {
            "analytic_mm3": analytic,
            "mesh_mm3": float(mesh.volume),
            "manifold_mm3": float(solid.volume()),
            "drawing_model_m3": drawing_volume,
            "declared_m3": spec.declared_volume_m3,
            "drawing_minus_declared_m3": (
                None
                if spec.declared_volume_m3 is None
                else round(drawing_volume - spec.declared_volume_m3, 4)
            ),
        },
        "evidence": list(spec.evidence),
        "unknowns": list(spec.unknowns),
    }
