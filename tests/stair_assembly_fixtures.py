"""Synthetic same-object construction-region fixtures for Step 4."""

from __future__ import annotations

from typing import Any

from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh


FLIGHT_WIDTH_MM = 975.0
FLIGHT_GAP_MM = 200.0
ASSEMBLY_WIDTH_MM = 2150.0
FLIGHT_RUN_MM = 2925.0
SLAB_THICKNESS_MM = 150.0
LANDING_DEPTH_MM = 975.0


def _region(
    identifier: str,
    role: str,
    origin_xyz_mm: tuple[float, float, float],
    size_xyz_mm: tuple[float, float, float],
) -> dict[str, Any]:
    width, depth, height = size_xyz_mm
    mesh = _extruded_mesh(
        [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
        depth,
    )
    return {
        "id": identifier,
        "record_type": "construction_region",
        "physical_object_scope_ref": "physical_object_scope.synthetic_stair",
        "construction_role": role,
        "state": "resolved",
        "mesh": {
            "vertices_xyz_mm": mesh["vertices_xyz_mm"],
            "triangles": mesh["triangles"],
        },
        "transform": {
            "id": f"physical_component_transform.{identifier}",
            "record_type": "physical_component_transform",
            "record_version": "0.1.0",
            "component_ref": identifier,
            "state": "resolved",
            "placement_role": "physical",
            "origin_xyz_mm": list(origin_xyz_mm),
            "local_axes_xyz": {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            },
            "axis_signs": "resolved",
            "evidence_refs": [f"synthetic.transform.{identifier}"],
        },
        "analytic_volume": {
            "value_mm3": width * depth * height,
            "basis": "synthetic_rectangular_construction_region",
            "evidence_refs": [f"synthetic.analytic.{identifier}"],
        },
        "evidence_refs": [f"synthetic.region.{identifier}"],
    }


def same_object_stair_fixture(
    *,
    overlapping_regions: bool = False,
    disconnected: bool = False,
    omit_internal_seam: bool = False,
    wrong_analytic_union: bool = False,
    wrong_plan_reprojection: bool = False,
) -> dict[str, Any]:
    """Return one stair object built from three non-additive regions."""

    landing_x = 100.0 if disconnected else -100.0 if overlapping_regions else 0.0
    landing_depth = LANDING_DEPTH_MM - landing_x
    regions = [
        _region(
            "construction_region.lower_flight",
            "lower_flight",
            (-FLIGHT_RUN_MM, 0.0, 0.0),
            (FLIGHT_RUN_MM, FLIGHT_WIDTH_MM, SLAB_THICKNESS_MM),
        ),
        _region(
            "construction_region.upper_flight",
            "upper_flight",
            (-FLIGHT_RUN_MM, FLIGHT_WIDTH_MM + FLIGHT_GAP_MM, 0.0),
            (FLIGHT_RUN_MM, FLIGHT_WIDTH_MM, SLAB_THICKNESS_MM),
        ),
        _region(
            "construction_region.landing",
            "landing",
            (landing_x, 0.0, 0.0),
            (landing_depth, ASSEMBLY_WIDTH_MM, SLAB_THICKNESS_MM),
        ),
    ]
    seams = []
    if not disconnected and not omit_internal_seam:
        for ordinal, (flight_ref, y0) in enumerate(
            (
                ("construction_region.lower_flight", 0.0),
                ("construction_region.upper_flight", FLIGHT_WIDTH_MM + FLIGHT_GAP_MM),
            ),
            start=1,
        ):
            seam = {
                "id": f"internal_seam.flight_landing.{ordinal}",
                "record_type": "internal_seam",
                "physical_object_scope_ref": "physical_object_scope.synthetic_stair",
                "construction_region_refs": [flight_ref, "construction_region.landing"],
                "seam_kind": "overlap_union" if overlapping_regions else "coincident_face",
                "state": "resolved",
                "evidence_refs": [f"synthetic.internal_seam.{ordinal}"],
            }
            if overlapping_regions:
                seam["overlap_authorization_certificate"] = {
                    "id": f"overlap_geometry_authorization_certificate.synthetic.{ordinal}",
                    "record_type": "overlap_geometry_authorization_certificate",
                    "state": "accepted",
                    "authorized_seam_kind": "overlap_union",
                    "independent_of_candidate_mesh_intersection": True,
                    "evidence_refs": [f"synthetic.known_overlap.{ordinal}"],
                }
            if not overlapping_regions:
                seam["polygon_xyz_mm"] = [
                    [0.0, y0, 0.0],
                    [0.0, y0 + FLIGHT_WIDTH_MM, 0.0],
                    [0.0, y0 + FLIGHT_WIDTH_MM, SLAB_THICKNESS_MM],
                    [0.0, y0, SLAB_THICKNESS_MM],
                ]
            seams.append(seam)

    additive = sum(region["analytic_volume"]["value_mm3"] for region in regions)
    overlap = (
        2.0 * 100.0 * FLIGHT_WIDTH_MM * SLAB_THICKNESS_MM
        if overlapping_regions
        else 0.0
    )
    analytic_union = additive - overlap
    if wrong_analytic_union:
        analytic_union = additive

    plan_polygons = [
        [[-FLIGHT_RUN_MM, 0.0], [0.0, 0.0], [0.0, FLIGHT_WIDTH_MM], [-FLIGHT_RUN_MM, FLIGHT_WIDTH_MM]],
        [
            [-FLIGHT_RUN_MM, FLIGHT_WIDTH_MM + FLIGHT_GAP_MM],
            [0.0, FLIGHT_WIDTH_MM + FLIGHT_GAP_MM],
            [0.0, ASSEMBLY_WIDTH_MM],
            [-FLIGHT_RUN_MM, ASSEMBLY_WIDTH_MM],
        ],
        [[landing_x, 0.0], [LANDING_DEPTH_MM, 0.0], [LANDING_DEPTH_MM, ASSEMBLY_WIDTH_MM], [landing_x, ASSEMBLY_WIDTH_MM]],
    ]
    if wrong_plan_reprojection:
        plan_polygons[-1][-1][0] -= 50.0
    section_start = min(-FLIGHT_RUN_MM, landing_x)
    supplied_views = [
        {
            "id": "view.synthetic.plan",
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "u_axis_xyz": [1.0, 0.0, 0.0],
            "v_axis_xyz": [0.0, 1.0, 0.0],
            "polygons_uv_mm": plan_polygons,
            "tolerance_mm": 1e-6,
            "evidence_refs": ["synthetic.view.plan"],
        },
        {
            "id": "view.synthetic.section",
            "origin_xyz_mm": [0.0, 0.0, 0.0],
            "u_axis_xyz": [1.0, 0.0, 0.0],
            "v_axis_xyz": [0.0, 0.0, 1.0],
            "polygons_uv_mm": [[
                [section_start, 0.0],
                [LANDING_DEPTH_MM, 0.0],
                [LANDING_DEPTH_MM, SLAB_THICKNESS_MM],
                [section_start, SLAB_THICKNESS_MM],
            ]],
            "tolerance_mm": 1e-6,
            "evidence_refs": ["synthetic.view.section"],
        },
    ]
    return {
        "physical_object_scope": {
            "id": "physical_object_scope.synthetic_stair",
            "record_type": "physical_object_scope",
            "state": "resolved",
            "evidence_refs": ["synthetic.scope.stair"],
        },
        "construction_regions": regions,
        "internal_seams": seams,
        "analytic_union_volume": {
            "value_mm3": analytic_union,
            "basis": "synthetic_exact_prism_union",
            "evidence_refs": ["synthetic.analytic.union"],
        },
        "supplied_views": supplied_views,
        "separate_object_interfaces": [
            {
                "id": "separate_object_interface.stair_beam",
                "record_type": "separate_object_interface",
                "object_scope_refs": [
                    "physical_object_scope.synthetic_stair",
                    "physical_object_scope.synthetic_beam",
                ],
                "state": "candidate",
                "evidence_refs": ["synthetic.separate.stair_beam"],
            }
        ],
    }
