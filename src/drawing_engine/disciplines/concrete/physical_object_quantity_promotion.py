"""Promote a validated constructive union to one calculated physical object."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), kind, *(str(part) for part in parts))).encode(
        "utf-8"
    )
    return f"{kind}.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _abstain(page_number: int, reason_code: str, evidence_refs: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "physical_object_quantity_promotion",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "scope_reclosure_certificate": None,
        "accepted_physical_object": None,
        "calculated_concrete_quantities": [],
        "canonical_3d_preview": None,
        "quantity_eligible": False,
        "evidence_refs": sorted({str(ref) for ref in evidence_refs if str(ref)}),
        "contract": {
            "one_physical_object_per_equivalence_class": True,
            "construction_regions_are_never_aggregated_separately": True,
            "equivalence_members_are_never_aggregated_separately": True,
            "absolute_orientation_required_for_invariant_volume": False,
            "schedule_values_used": False,
        },
    }


def _relation_scope(
    view_frame_graph: Mapping[str, Any],
    source_scope: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    expected_views = {
        str(source_scope.get("parent_view_id") or ""),
        str(source_scope.get("section_view_id") or ""),
    }
    if "" in expected_views or len(expected_views) != 2:
        return None, "source_scope_view_pair_unresolved"
    candidates = [
        scope
        for scope in (view_frame_graph.get("shared_coordinate_system") or {}).get(
            "scopes", []
        )
        or []
        if scope.get("state") == "resolved_relative"
        and scope.get("scope_kind") == "cut_relation_component"
        and set(map(str, scope.get("view_ids", []) or [])) == expected_views
    ]
    if len(candidates) != 1:
        return None, "unique_relation_driven_scope_unresolved"
    scope = candidates[0]
    if scope.get("conflicts"):
        return None, "relation_driven_scope_has_conflicts"
    reprojections = list(scope.get("reprojection_validations", []) or [])
    if not reprojections or any(item.get("status") != "pass" for item in reprojections):
        return None, "relation_driven_scope_reprojection_unresolved"
    constraints = list(scope.get("cut_plane_constraints", []) or [])
    if not constraints or any(not item.get("relation_id") for item in constraints):
        return None, "relation_driven_cut_constraint_unresolved"
    return scope, None


def _complete_competitor_disposition(assembly: Mapping[str, Any]) -> bool:
    summary = assembly.get("summary") or {}
    placements = int(summary.get("placement_alternative_count") or 0)
    materialized = int(summary.get("materialized_union_hypothesis_count") or 0)
    rejected = int(summary.get("pre_kernel_rejection_count") or 0)
    hard_rejected = int(summary.get("hard_pre_kernel_rejection_count") or 0)
    eliminated = int(summary.get("certified_elimination_count") or 0)
    replays = int(summary.get("kernel_replay_count") or 0)
    return bool(
        placements
        and int(summary.get("unresolved_live_alternative_count") or 0) == 0
        and materialized == replays
        and rejected == hard_rejected
        and placements == materialized + rejected + eliminated
    )


def _member_replays(
    assembly: Mapping[str, Any], certificate: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], str | None]:
    member_refs = list(map(str, certificate.get("member_hypothesis_refs", []) or []))
    if len(member_refs) < 2 or len(member_refs) != len(set(member_refs)):
        return [], "equivalence_member_refs_invalid"
    replay_by_ref = {
        str(item.get("constructive_union_hypothesis_ref")): item
        for item in assembly.get("kernel_replays", []) or []
        if item.get("status") == "accepted"
    }
    if set(member_refs) != set(replay_by_ref):
        return [], "equivalence_members_do_not_match_all_survivors"
    members = [replay_by_ref[ref] for ref in sorted(member_refs)]
    expected_views = {str(view["id"]): view for view in assembly.get("native_projection_evidence", [])}
    for member in members:
        replay_views = {str(view["id"]): view for view in member["kernel_result"].get("supplied_view_reprojections", [])}
        for ref in member.get("supplied_projection_refs", []):
            required = expected_views.get(ref, {}).get("required_direction_evidence")
            if required is None:
                continue
            paths = required.get("paths") or []
            actual = replay_views.get(ref, {}).get("directed_surface_path_reprojections") or []
            if (required.get("state") != "accepted" or not paths
                or {p.get("id") for p in paths} != {p.get("path_ref") for p in actual}
                or any(p.get("status") != "pass" for p in actual)):
                return [], "member_directed_projection_replay_unresolved"
    return members, None


def _invariant_volume(
    members: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    analytic_values = []
    for member in members:
        validation = (member.get("kernel_result") or {}).get("volume_validation") or {}
        if validation.get("status") != "pass":
            return None, "member_volume_validation_unresolved"
        value = float(validation.get("analytic_union_mm3") or 0.0)
        tolerance = float(validation.get("tolerance_mm3") or 0.0)
        mesh = float(validation.get("mesh_union_mm3") or 0.0)
        manifold = float(validation.get("manifold_union_mm3") or 0.0)
        if (
            not all(math.isfinite(item) and item > 0 for item in (value, mesh, manifold))
            or tolerance <= 0
            or abs(mesh - value) > tolerance
            or abs(manifold - value) > tolerance
        ):
            return None, "member_analytic_mesh_volume_disagreement"
        analytic_values.append((value, tolerance))
    reference = analytic_values[0][0]
    if any(abs(value - reference) > min(tolerance, analytic_values[0][1]) for value, tolerance in analytic_values[1:]):
        return None, "equivalence_member_volumes_unequal"
    candidate_value = float(candidate.get("value_mm3") or 0.0)
    if (
        candidate.get("quantity_eligible") is not True
        or not math.isfinite(candidate_value)
        or abs(candidate_value - reference) > analytic_values[0][1]
    ):
        return None, "invariant_volume_candidate_mismatch"
    return reference, None


def _equivalence_contract_errors(certificate: Mapping[str, Any]) -> list[str]:
    errors = []
    if certificate.get("state") != "accepted" or certificate.get("quantity_eligible") is not True:
        errors.append("equivalence_certificate_not_accepted")
    checks = list(certificate.get("pairwise_checks", []) or [])
    if not checks or any(item.get("status") != "pass" for item in checks):
        errors.append("pairwise_equivalence_incomplete")
    for check in checks:
        if check.get("allowed_transform_kind") != "axis_sign_reflection_and_translation":
            errors.append("forbidden_equivalence_transform")
            continue
        signs = check.get("reflection_signs_xyz") or []
        if len(signs) != 3 or any(sign not in {-1, 1} for sign in signs):
            errors.append("reflection_signs_unresolved")
    contract = certificate.get("contract") or {}
    if contract.get("axis_permutation_is_not_an_allowed_equivalence") is not True:
        errors.append("axis_permutation_guard_missing")
    invariance = certificate.get("invariance_checks") or {}
    if not invariance or any(value != "pass" for value in invariance.values()):
        errors.append("invariance_checks_incomplete")
    return sorted(set(errors))


def aggregate_calculated_concrete_quantities(
    quantities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate exactly-once physical-object quantities without deduplicating silently."""

    records = []
    seen_objects = set()
    for quantity in quantities:
        if quantity.get("record_type") != "calculated_concrete_quantity":
            raise ValueError("calculated concrete aggregation received a wrong record type")
        object_ref = str(quantity.get("physical_object_ref") or "")
        if not object_ref or object_ref in seen_objects:
            raise ValueError("calculated concrete aggregation would double-count one physical object")
        if (
            quantity.get("aggregation_scope") != "physical_object"
            or quantity.get("source_kind") != "accepted_physical_object_union"
            or quantity.get("construction_regions_aggregated_separately") is not False
            or quantity.get("equivalence_members_aggregated_separately") is not False
        ):
            raise ValueError("construction regions or equivalence members cannot be aggregated")
        value = float(quantity.get("net_concrete_m3") or 0.0)
        if quantity.get("quantity_eligible") is not True or not math.isfinite(value) or value <= 0:
            raise ValueError("calculated concrete quantity is not eligible")
        seen_objects.add(object_ref)
        records.append(deepcopy(dict(quantity)))
    return records


def promote_constructive_union_physical_object(
    same_object_assembly: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    source_scope_evidence: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Bind one invariant union class to one accepted relation-driven object."""

    proposed_scope = same_object_assembly.get("physical_object_scope") or {}
    source_scope = source_scope_evidence.get("search_physical_object_scope") or {}
    refs = sorted(
        {
            str(proposed_scope.get("id") or ""),
            str(source_scope.get("id") or ""),
        }
        - {""}
    )
    if same_object_assembly.get("status") != "accepted_invariant_equivalence_class":
        return _abstain(page_number, "constructive_union_equivalence_unresolved", refs)
    if (
        proposed_scope.get("record_type") != "physical_object_scope"
        or proposed_scope.get("state") != "proposed_monolithic_physical_object_scope"
        or source_scope.get("record_type") != "physical_object_scope"
        or str(proposed_scope.get("parent_scope_ref") or "") != str(source_scope.get("id") or "")
    ):
        return _abstain(page_number, "proposed_scope_parent_mismatch", refs)
    relation_scope, relation_error = _relation_scope(view_frame_graph, source_scope)
    if relation_error:
        return _abstain(page_number, relation_error, refs)
    refs.extend(
        [
            str(relation_scope.get("id")),
            *map(str, relation_scope.get("evidence_refs", []) or []),
        ]
    )
    if not _complete_competitor_disposition(same_object_assembly):
        return _abstain(page_number, "competing_alternatives_not_fully_replayed_or_eliminated", refs)
    equivalence = same_object_assembly.get("invariant_union_equivalence_certificate") or {}
    equivalence_errors = _equivalence_contract_errors(equivalence)
    if equivalence_errors:
        return _abstain(page_number, equivalence_errors[0], refs)
    members, member_error = _member_replays(same_object_assembly, equivalence)
    if member_error:
        return _abstain(page_number, member_error, refs)
    candidate = same_object_assembly.get("invariant_concrete_volume_candidate") or {}
    volume_mm3, volume_error = _invariant_volume(members, candidate)
    if volume_error:
        return _abstain(page_number, volume_error, refs)

    # Only after the whole equivalence class passes: prefer its least redundant
    # triangulation, then stable ID. This is a representation choice, never a
    # ranking that can accept or eliminate a physical interpretation.
    representative = min(members, key=lambda item: (
        len(item["kernel_result"]["external_boundary"]["mesh"]["triangles"]),
        len(item["kernel_result"]["external_boundary"]["mesh"]["vertices_xyz_mm"]),
        str(item.get("constructive_union_hypothesis_ref")),
    ))
    kernel = representative["kernel_result"]
    boundary = kernel.get("external_boundary") or {}
    validation = boundary.get("validation") or {}
    if (
        validation.get("status") != "pass"
        or validation.get("watertight") is not True
        or int(validation.get("connected_solid_count") or 0) != 1
    ):
        return _abstain(page_number, "canonical_external_boundary_unresolved", refs)

    proposed_scope_ref = str(proposed_scope["id"])
    relation_scope_ref = str(relation_scope["id"])
    equivalence_ref = str(equivalence["id"])
    object_id = _stable_id(
        page_number,
        "physical_object",
        proposed_scope_ref,
        relation_scope_ref,
        equivalence_ref,
    )
    accepted_scope_id = _stable_id(
        page_number, "accepted_physical_object_scope", proposed_scope_ref, relation_scope_ref
    )
    quantity_id = _stable_id(page_number, "calculated_concrete_quantity", object_id)
    preview_id = _stable_id(page_number, "canonical_3d_preview", object_id)
    member_refs = list(map(str, equivalence.get("member_hypothesis_refs", []) or []))
    construction_roles = list(map(str, proposed_scope.get("construction_region_roles", []) or []))
    scope_certificate = {
        "id": _stable_id(page_number, "physical_object_scope_reclosure_certificate", object_id),
        "record_type": "physical_object_scope_reclosure_certificate",
        "state": "accepted",
        "source_proposed_scope_ref": proposed_scope_ref,
        "source_search_scope_ref": str(source_scope["id"]),
        "relation_driven_scope_ref": relation_scope_ref,
        "relation_refs": sorted(
            str(item.get("relation_id"))
            for item in relation_scope.get("cut_plane_constraints", []) or []
        ),
        "view_ids": sorted(map(str, relation_scope.get("view_ids", []) or [])),
        "reprojection_status": "pass",
        "constructive_union_equivalence_ref": equivalence_ref,
        "all_competitors_replayed_or_hard_eliminated": True,
        "absolute_orientation_state": "unresolved",
        "invariant_quantity_state": "accepted",
        "evidence_refs": sorted(set(refs)),
    }
    accepted_scope = {
        "id": accepted_scope_id,
        "record_type": "physical_object_scope",
        "state": "accepted_relative_orientation_unresolved",
        "source_proposed_scope_ref": proposed_scope_ref,
        "relation_driven_scope_ref": relation_scope_ref,
        "scope_reclosure_certificate_ref": scope_certificate["id"],
        "accepted_physical_object_scope": True,
        "absolute_orientation_resolved": False,
        "quantity_eligible": True,
        "evidence_refs": sorted(set(refs)),
    }
    physical_object = {
        "id": object_id,
        "record_type": "physical_object",
        "state": "accepted",
        "physical_object_scope": accepted_scope,
        "physical_object_scope_ref": accepted_scope_id,
        "source_constructive_union_equivalence_ref": equivalence_ref,
        "construction_region_roles": construction_roles,
        "construction_regions_are_subordinate_nonadditive_evidence": True,
        "equivalence_member_refs": member_refs,
        "equivalence_members_are_transform_provenance_not_objects": True,
        "excluded_separate_object_interface_refs": sorted(
            str(item.get("id"))
            for item in same_object_assembly.get(
                "separate_object_interface_hypotheses", []
            )
            or []
        ),
        "external_boundary_ref": boundary.get("id"),
        "geometry": {
            "shape_type": "constructive_union",
            "construction_region_count": len(construction_roles),
            "external_boundary_ref": boundary.get("id"),
            "absolute_orientation_resolved": False,
        },
        "quantity_eligible": True,
        "evidence_refs": sorted(set([*refs, equivalence_ref])),
    }
    quantity = {
        "id": quantity_id,
        "record_type": "calculated_concrete_quantity",
        "status": "procedurally_calculated",
        "state": "calculated",
        "physical_object_ref": object_id,
        "physical_object_scope_ref": accepted_scope_id,
        "aggregation_scope": "physical_object",
        "source_kind": "accepted_physical_object_union",
        "units": "m3",
        "gross_concrete_mm3": volume_mm3,
        "gross_concrete_m3": volume_mm3 / 1_000_000_000.0,
        "net_concrete_mm3": volume_mm3,
        "net_concrete_m3": volume_mm3 / 1_000_000_000.0,
        "constructive_union_equivalence_ref": equivalence_ref,
        "equivalence_member_refs": member_refs,
        "construction_region_roles": construction_roles,
        "construction_regions_aggregated_separately": False,
        "equivalence_members_aggregated_separately": False,
        "separate_objects_included": False,
        "absolute_orientation_resolved": False,
        "quantity_eligible": True,
        "evidence_refs": sorted(set([*refs, equivalence_ref])),
    }
    quantities = aggregate_calculated_concrete_quantities([quantity])
    preview = {
        "id": preview_id,
        "record_type": "canonical_3d_preview",
        "state": "resolved_relative",
        "label": "absolute orientation unresolved",
        "physical_object_ref": object_id,
        "source_equivalence_member_ref": representative[
            "constructive_union_hypothesis_ref"
        ],
        "canonical_representation_only": True,
        "relative_placement_state": equivalence.get("physical_placement_state"),
        "absolute_orientation_resolved": False,
        "mesh": deepcopy(boundary.get("mesh")),
        "mesh_validation": deepcopy(validation),
        "quantity_eligible": False,
        "evidence_refs": sorted(set([*refs, equivalence_ref])),
    }
    normalized = {
        "scope_certificate": scope_certificate,
        "accepted_physical_object": physical_object,
        "quantity": quantity,
        "preview_source_ref": preview["source_equivalence_member_ref"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "physical_object_quantity_promotion",
        "page": page_number,
        "status": "accepted",
        "reason_code": None,
        "input_sha256": sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scope_reclosure_certificate": scope_certificate,
        "accepted_physical_object": physical_object,
        "calculated_concrete_quantities": quantities,
        "canonical_3d_preview": preview,
        "excluded_from_aggregation": {
            "construction_region_roles": construction_roles,
            "equivalence_member_refs": member_refs,
            "separate_object_interface_refs": physical_object[
                "excluded_separate_object_interface_refs"
            ],
        },
        "quantity_eligible": True,
        "evidence_refs": sorted(set([*refs, equivalence_ref])),
        "contract": {
            "one_physical_object_per_equivalence_class": True,
            "construction_regions_are_never_aggregated_separately": True,
            "equivalence_members_are_never_aggregated_separately": True,
            "separate_objects_require_independent_boundary_closure": True,
            "absolute_orientation_required_for_invariant_volume": False,
            "canonical_preview_is_not_absolute_placement": True,
            "schedule_values_used": False,
        },
    }
