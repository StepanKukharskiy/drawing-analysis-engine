"""Materialize evidence-backed partial profile sweeps for audit and search.

This layer deliberately stops short of physical-object or quantity closure.  A
temporary cap may make an incomplete drawing profile renderable, but it never
becomes a physical boundary or a source of calculated volume.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from shapely.geometry import Polygon

from src.drawing_engine.disciplines.concrete.bounded_profile_sweep_materializer import _evidence, _stable_id, _transform_point
from src.drawing_engine.disciplines.concrete.generic_profile_extrusion_solver import _extruded_mesh


SCHEMA_VERSION = "0.1.0"
_PROFILE_CLOSURES = {"physical_closed", "analysis_capped_closed"}
_SWEEP_CLOSURES = {"physical_bounded", "analysis_capped_bounded"}
_COVERAGE_STATES = {"covered", "partial", "not_depicted", "unresolved"}


def _validate_projection_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_hypothesis: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        record_id = str(record.get("id") or "")
        hypothesis_ref = str(record.get("hypothesis_ref") or "")
        if record.get("record_type") != "projection_coverage_evidence":
            raise ValueError(f"{record_id} has the wrong projection record type")
        if record.get("state") != "resolved" or not hypothesis_ref:
            raise ValueError(f"{record_id} projection coverage is unresolved")
        if record.get("coverage_state") not in _COVERAGE_STATES:
            raise ValueError(f"{record_id} has an unsupported projection coverage state")
        if record.get("derived_from_generated_geometry") is not False:
            raise ValueError(f"{record_id} projection evidence must be independent of generated geometry")
        _evidence(record, record_id)
        by_hypothesis.setdefault(hypothesis_ref, []).append(deepcopy(dict(record)))
    return by_hypothesis


def _classified_cap_faces(
    profile: Mapping[str, Any],
    boundary: Sequence[Sequence[float]],
    depth_mm: float,
    transform: Mapping[str, Any],
) -> list[dict[str, Any]]:
    faces = []
    seen: set[str] = set()
    for cap in profile.get("classified_caps", []) or []:
        cap_id = str(cap.get("id") or "")
        edge_index = int(cap.get("boundary_edge_index", -1))
        classification = str(cap.get("classification") or "")
        if not cap_id or cap_id in seen or not 0 <= edge_index < len(boundary):
            raise ValueError("partial profile has an invalid classified cap")
        if classification not in {
            "external_boundary",
            "external_support_end",
            "internal_seam_hypothesis",
            "analysis_cap",
        }:
            raise ValueError(f"{cap_id} has an unsupported cap classification")
        cap_evidence = _evidence(cap, cap_id)
        following = (edge_index + 1) % len(boundary)
        start, end = boundary[edge_index], boundary[following]
        faces.append(
            {
                "id": cap_id,
                "record_type": "partial_sweep_cap_face",
                "classification": classification,
                "state": "resolved_search_geometry",
                "polygon_xyz_mm": [
                    _transform_point([start[0], 0.0, start[1]], transform),
                    _transform_point([start[0], depth_mm, start[1]], transform),
                    _transform_point([end[0], depth_mm, end[1]], transform),
                    _transform_point([end[0], 0.0, end[1]], transform),
                ],
                "physical_boundary": classification != "analysis_cap",
                "quantity_eligible": False,
                "evidence_refs": cap_evidence,
            }
        )
        seen.add(cap_id)
    return faces


def materialize_partial_profile_sweeps(
    profile_sweep_hypotheses: Sequence[Mapping[str, Any]],
    supplied_projection_evidence: Sequence[Mapping[str, Any]],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Materialize every evidence-valid transform without selecting a winner.

    The returned meshes are deterministic search/audit geometry.  Even a
    watertight preview remains quantity-ineligible because this layer neither
    resolves physical-object identity nor promotes temporary caps.
    """

    projections_by_hypothesis = _validate_projection_records(supplied_projection_evidence)
    candidates: list[dict[str, Any]] = []
    hypothesis_ids: set[str] = set()
    transform_count = 0
    for hypothesis in profile_sweep_hypotheses:
        hypothesis_id = str(hypothesis.get("id") or "")
        if hypothesis.get("record_type") != "partial_profile_sweep_hypothesis":
            raise ValueError(f"{hypothesis_id} has the wrong hypothesis record type")
        if hypothesis.get("state") != "resolved_search_hypothesis" or not hypothesis_id:
            raise ValueError(f"{hypothesis_id} partial sweep hypothesis is unresolved")
        if hypothesis_id in hypothesis_ids:
            raise ValueError(f"duplicate partial sweep hypothesis {hypothesis_id}")
        if hypothesis.get("quantity_eligible") is not False:
            raise ValueError(f"{hypothesis_id} must be quantity-ineligible")
        hypothesis_evidence = _evidence(hypothesis, hypothesis_id)
        hypothesis_ids.add(hypothesis_id)

        profile = hypothesis.get("profile_boundary") or {}
        sweep = hypothesis.get("sweep_bound") or {}
        if profile.get("record_type") != "partial_profile_boundary" or profile.get("state") != "resolved":
            raise ValueError(f"{hypothesis_id} profile boundary is unresolved")
        if profile.get("closure_kind") not in _PROFILE_CLOSURES:
            raise ValueError(f"{hypothesis_id} has an unsupported profile closure")
        if sweep.get("record_type") != "partial_sweep_bound" or sweep.get("state") != "resolved":
            raise ValueError(f"{hypothesis_id} sweep bound is unresolved")
        if sweep.get("closure_kind") not in _SWEEP_CLOSURES:
            raise ValueError(f"{hypothesis_id} has an unsupported sweep closure")
        profile_evidence = _evidence(profile, f"{hypothesis_id}.profile_boundary")
        sweep_evidence = _evidence(sweep, f"{hypothesis_id}.sweep_bound")
        boundary = [list(map(float, point)) for point in profile.get("ordered_points_uv_mm", []) or []]
        polygon = Polygon(boundary)
        if len(boundary) < 3 or not polygon.is_valid or polygon.area <= 1e-6:
            raise ValueError(f"{hypothesis_id} temporary profile is not a valid polygon")
        depth_mm = float(sweep.get("depth_mm") or 0.0)
        if depth_mm <= 0:
            raise ValueError(f"{hypothesis_id} has an invalid sweep depth")
        projection_records = projections_by_hypothesis.get(hypothesis_id, [])
        if not projection_records:
            raise ValueError(f"{hypothesis_id} requires supplied projection coverage evidence")

        transforms = list(hypothesis.get("local_to_world_transform_alternatives", []) or [])
        if not transforms:
            raise ValueError(f"{hypothesis_id} has no local-to-world transform alternatives")
        local_mesh = _extruded_mesh([tuple(point) for point in boundary], depth_mm)
        for transform in transforms:
            transform_id = str(transform.get("id") or "")
            if transform.get("record_type") != "local_to_world_transform" or transform.get("state") != "resolved":
                raise ValueError(f"{transform_id} local-to-world transform is unresolved")
            transform_evidence = _evidence(transform, transform_id)
            mesh = deepcopy(local_mesh)
            mesh["vertices_xyz_mm"] = [
                [round(value, 4) for value in _transform_point(point, transform)]
                for point in local_mesh["vertices_xyz_mm"]
            ]
            cap_faces = _classified_cap_faces(profile, boundary, depth_mm, transform)
            profile_complete = profile.get("closure_kind") == "physical_closed"
            sweep_complete = sweep.get("closure_kind") == "physical_bounded"
            physical_boundary_complete = profile_complete and sweep_complete and all(
                item["physical_boundary"] for item in cap_faces
            )
            candidate_id = _stable_id(
                "partial_profile_sweep_candidate",
                page_number,
                hypothesis_id,
                transform_id,
            )
            evidence_refs = sorted(
                {
                    *hypothesis_evidence,
                    *profile_evidence,
                    *sweep_evidence,
                    *transform_evidence,
                    *(
                        ref
                        for record in projection_records
                        for ref in record.get("evidence_refs", []) or []
                    ),
                }
            )
            candidates.append(
                {
                    "id": candidate_id,
                    "record_type": "partial_profile_sweep_candidate",
                    "record_version": SCHEMA_VERSION,
                    "state": "materialized_search_geometry",
                    "source_hypothesis_ref": hypothesis_id,
                    "construction_role": str(hypothesis.get("construction_role") or "partial_profile_sweep"),
                    "classification": str(hypothesis.get("classification") or "unclassified_partial_sweep"),
                    "mesh": mesh,
                    "local_to_world_transform": deepcopy(dict(transform)),
                    "classified_cap_faces": cap_faces,
                    "projection_coverage_records": projection_records,
                    "geometry_complete": physical_boundary_complete,
                    "physical_boundary_complete": physical_boundary_complete,
                    "contains_analysis_caps": not physical_boundary_complete,
                    "analytic_preview_enclosure": {
                        "value_mm3": float(polygon.area) * depth_mm,
                        "basis": "temporary_profile_area_times_supplied_sweep_depth",
                        "is_physical_volume": False,
                        "quantity_eligible": False,
                    },
                    "included_in_primary_object": False,
                    "included_in_primary_quantity": False,
                    "quantity_eligible": False,
                    "evidence_refs": evidence_refs,
                }
            )
            transform_count += 1

    unknown_projection_refs = sorted(set(projections_by_hypothesis) - hypothesis_ids)
    if unknown_projection_refs:
        raise ValueError(f"projection evidence references unknown hypotheses: {unknown_projection_refs}")
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "partial_profile_sweep_materialization",
        "page": page_number,
        "status": "materialized_partial_hypotheses" if candidates else "no_materializable_hypotheses",
        "candidate_previews": candidates,
        "summary": {
            "hypothesis_count": len(profile_sweep_hypotheses),
            "transform_alternative_count": transform_count,
            "materialized_candidate_count": len(candidates),
            "physically_complete_candidate_count": sum(
                bool(item["physical_boundary_complete"]) for item in candidates
            ),
            "quantity_eligible_candidate_count": 0,
        },
        "contract": {
            "analysis_cap_is_physical_boundary": False,
            "generated_mesh_is_projection_evidence": False,
            "materialization_selects_unique_transform": False,
            "materialization_resolves_physical_identity": False,
            "quantity_eligible": False,
        },
    }
