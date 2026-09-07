"""Drawing-neutral enumeration and replay of same-object union hypotheses."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from src.drawing_engine.disciplines.concrete.invariant_union_equivalence import certify_invariant_union_equivalence
from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import directed_surface_evidence_errors, solve_same_object_constructive_solid


SCHEMA_VERSION = "0.1.0"


def _records_by_id(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    duplicates = set()
    for record in records:
        record_id = str(record.get("id") or "")
        if not record_id or record_id == "None" or record_id in indexed:
            duplicates.add(record_id)
            continue
        indexed[record_id] = record
    return indexed, duplicates


def _refs(alternative: Mapping[str, Any], key: str, placement_key: str | None = None) -> list[str]:
    if placement_key and alternative.get(placement_key) is not None:
        return [
            str(item.get("construction_region_ref") or "")
            for item in alternative.get(placement_key, []) or []
        ]
    return list(map(str, alternative.get(key, []) or []))


def _resolve_records(
    refs: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    kind: str,
    errors: list[str],
) -> list[Mapping[str, Any]]:
    resolved = []
    if not refs:
        errors.append(f"{kind}_refs_missing")
        return resolved
    if len(set(refs)) != len(refs) or any(not ref for ref in refs):
        errors.append(f"{kind}_refs_invalid")
    for ref in refs:
        record = records.get(ref)
        if record is None:
            errors.append(f"{kind}_ref_unresolved:{ref}")
        else:
            resolved.append(record)
    return resolved


def _materialize_regions(
    alternative: Mapping[str, Any],
    region_index: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    placements = alternative.get("construction_region_placements")
    refs = _refs(alternative, "construction_region_refs", "construction_region_placements")
    records = _resolve_records(refs, region_index, "construction_region", errors)
    placement_by_ref = {
        str(item.get("construction_region_ref") or ""): item
        for item in placements or []
    }
    materialized = []
    for record in records:
        region = deepcopy(dict(record))
        region_id = str(region.get("id"))
        placement = placement_by_ref.get(region_id, {})
        if placement.get("transform") is not None:
            region["transform"] = deepcopy(placement["transform"])
        if region.get("record_type") != "construction_region":
            errors.append(f"construction_region_wrong_type:{region_id}")
        if region.get("state") != "resolved":
            errors.append(f"construction_region_unmaterialized:{region_id}")
        if not isinstance(region.get("mesh"), Mapping):
            errors.append(f"construction_region_mesh_missing:{region_id}")
        if not isinstance(region.get("transform"), Mapping):
            errors.append(f"construction_region_transform_missing:{region_id}")
        if not isinstance(region.get("analytic_volume"), Mapping):
            errors.append(f"construction_region_analytic_volume_missing:{region_id}")
        materialized.append(region)
    return materialized


def _accepted_elimination(
    alternative_id: str,
    alternative: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    certificate = alternative.get("elimination_certificate")
    if certificate is None:
        return None, []
    errors = []
    if not isinstance(certificate, Mapping):
        return None, ["elimination_certificate_invalid"]
    if certificate.get("record_type") != "alternative_elimination_certificate":
        errors.append("elimination_certificate_wrong_type")
    if certificate.get("status") != "accepted":
        errors.append("elimination_certificate_not_accepted")
    if str(certificate.get("alternative_ref") or "") != alternative_id:
        errors.append("elimination_certificate_alternative_mismatch")
    if certificate.get("basis") not in {"hard_contradiction", "dominance"}:
        errors.append("elimination_certificate_basis_invalid")
    if not certificate.get("evidence_refs"):
        errors.append("elimination_certificate_evidence_missing")
    return (deepcopy(dict(certificate)) if not errors else None), errors


def _construction_region_seam_graph_connected(
    regions: Sequence[Mapping[str, Any]],
    seams: Sequence[Mapping[str, Any]],
) -> bool:
    """Require one connected same-object construction graph before geometry."""

    nodes = {str(item.get("id")) for item in regions}
    if len(nodes) < 2:
        return False
    adjacency = {node: set() for node in nodes}
    for seam in seams:
        refs = list(map(str, seam.get("construction_region_refs", []) or []))
        if len(refs) != 2 or refs[0] == refs[1] or any(ref not in nodes for ref in refs):
            continue
        adjacency[refs[0]].add(refs[1])
        adjacency[refs[1]].add(refs[0])
    reached = set()
    pending = [next(iter(nodes))]
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(adjacency[node] - reached)
    return reached == nodes


def _accepted_overlap_authorization(
    seam: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate evidence that authorizes an intentional volumetric join.

    A mesh intersection is deliberately not an authorization source.  The
    certificate must precede candidate geometry and point to drawing or
    known-truth evidence independent of that intersection.
    """

    certificate = seam.get("overlap_authorization_certificate")
    if not isinstance(certificate, Mapping):
        return None, ["independent_overlap_authorization_missing"]
    errors = []
    if certificate.get("record_type") != "overlap_geometry_authorization_certificate":
        errors.append("overlap_authorization_wrong_type")
    if certificate.get("state") != "accepted":
        errors.append("overlap_authorization_not_accepted")
    if certificate.get("authorized_seam_kind") != "overlap_union":
        errors.append("overlap_authorization_kind_mismatch")
    if certificate.get("independent_of_candidate_mesh_intersection") is not True:
        errors.append("overlap_authorization_not_independent")
    if not certificate.get("evidence_refs"):
        errors.append("overlap_authorization_evidence_missing")
    return (deepcopy(dict(certificate)) if not errors else None), errors


def _certify_internal_seam_semantics(
    seams: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Require materialized contacts to preserve their assigned semantics."""

    checks = []
    errors = []
    for seam in seams:
        seam_id = str(seam.get("id") or "")
        seam_kind = str(seam.get("seam_kind") or "")
        required = str(seam.get("required_contact_semantics") or "")
        authorization = None
        seam_errors = []
        if seam_kind == "overlap_union":
            authorization, authorization_errors = _accepted_overlap_authorization(seam)
            seam_errors.extend(authorization_errors)
        if required and seam_kind != required:
            seam_errors.append(
                f"assigned_{required}_materialized_as_{seam_kind or 'unknown'}"
            )
        seam_errors = sorted(set(seam_errors))
        if seam_errors:
            errors.append(f"internal_seam_semantics_contradiction:{seam_id}")
        checks.append(
            {
                "internal_seam_ref": seam_id,
                "source_cap_classification": seam.get("source_cap_classification"),
                "required_contact_semantics": required or None,
                "materialized_seam_kind": seam_kind,
                "overlap_authorization_certificate": authorization,
                "errors": seam_errors,
                "status": "fail" if seam_errors else "pass",
            }
        )
    return {
        "record_type": "internal_seam_semantics_certificate",
        "state": "accepted" if not errors else "contradiction",
        "checks": checks,
        "candidate_mesh_intersection_is_not_overlap_authorization": True,
        "quantity_eligible": False,
    }, errors


def enumerate_constructive_union_hypotheses(
    physical_object_scopes: Sequence[Mapping[str, Any]],
    construction_region_hypotheses: Sequence[Mapping[str, Any]],
    placement_alternatives: Sequence[Mapping[str, Any]],
    internal_seam_hypotheses: Sequence[Mapping[str, Any]],
    supplied_projection_evidence: Sequence[Mapping[str, Any]],
    *,
    separate_object_interfaces: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Materialize and replay every complete same-object union alternative.

    Alternative records carry only references plus an independently supported
    ``analytic_union_volume``.  Counts and status are derived from actual
    materialization and kernel outcomes; an unevaluated alternative can never
    become an ambiguity certificate.
    """

    scope_index, duplicate_scopes = _records_by_id(physical_object_scopes)
    region_index, duplicate_regions = _records_by_id(construction_region_hypotheses)
    seam_index, duplicate_seams = _records_by_id(internal_seam_hypotheses)
    view_index, duplicate_views = _records_by_id(supplied_projection_evidence)
    interface_index, duplicate_interfaces = _records_by_id(separate_object_interfaces)
    duplicate_errors = sorted(
        {
            *(f"duplicate_physical_object_scope_id:{item}" for item in duplicate_scopes),
            *(f"duplicate_construction_region_id:{item}" for item in duplicate_regions),
            *(f"duplicate_internal_seam_id:{item}" for item in duplicate_seams),
            *(f"duplicate_projection_id:{item}" for item in duplicate_views),
            *(f"duplicate_separate_object_interface_id:{item}" for item in duplicate_interfaces),
        }
    )

    evaluations = []
    complete_hypotheses = []
    kernel_replays = []
    survivors = []
    for ordinal, alternative in enumerate(placement_alternatives, start=1):
        alternative_id = str(alternative.get("id") or f"placement_alternative.{ordinal:04d}")
        errors = list(duplicate_errors)
        elimination, elimination_errors = _accepted_elimination(alternative_id, alternative)
        errors.extend(elimination_errors)
        if elimination is not None and not errors:
            evaluations.append(
                {
                    "id": alternative_id,
                    "record_type": "constructive_union_hypothesis_evaluation",
                    "source_assignment_ref": alternative.get("source_assignment_ref"),
                    "physical_object_scope_ref": str(alternative.get("physical_object_scope_ref") or ""),
                    "construction_region_refs": _refs(
                        alternative,
                        "construction_region_refs",
                        "construction_region_placements",
                    ),
                    "internal_seam_refs": list(map(str, alternative.get("internal_seam_refs", []) or [])),
                    "supplied_projection_refs": list(map(str, alternative.get("supplied_projection_refs", []) or [])),
                    "pre_kernel_rejection_reasons": [],
                    "materialization_status": "eliminated",
                    "elimination_certificate": elimination,
                    "step4_same_object_union_replayed": False,
                    "step4_status": "not_required",
                    "quantity_eligible": False,
                }
            )
            continue
        scope_ref = str(alternative.get("physical_object_scope_ref") or "")
        scope = scope_index.get(scope_ref)
        if scope is None:
            errors.append(f"physical_object_scope_ref_unresolved:{scope_ref}")
        elif scope.get("record_type") != "physical_object_scope":
            errors.append(f"physical_object_scope_wrong_type:{scope_ref}")
        elif scope.get("state") not in {
            "resolved",
            "resolved_relative",
            "resolved_search_hypothesis",
        }:
            errors.append(f"physical_object_scope_unresolved:{scope_ref}")

        regions = _materialize_regions(alternative, region_index, errors)
        seam_refs = list(map(str, alternative.get("internal_seam_refs", []) or []))
        seams = _resolve_records(seam_refs, seam_index, "internal_seam", errors)
        for seam in seams:
            seam_id = str(seam.get("id"))
            if seam.get("record_type") != "internal_seam" or seam.get("state") != "resolved":
                errors.append(f"internal_seam_unresolved:{seam_id}")
        seam_semantics_certificate, seam_semantics_errors = _certify_internal_seam_semantics(
            seams
        )
        errors.extend(seam_semantics_errors)
        if regions and not _construction_region_seam_graph_connected(regions, seams):
            errors.append("construction_region_seam_graph_disconnected")

        view_refs = list(map(str, alternative.get("supplied_projection_refs", []) or []))
        views = _resolve_records(view_refs, view_index, "supplied_projection", errors)
        if len(views) < 2:
            errors.append("two_supplied_projections_required")
        for view in views:
            for reason in directed_surface_evidence_errors(view):
                errors.append(f"supplied_direction_evidence_unresolved:{view.get('id')}:{reason}")

        interface_refs = list(map(str, alternative.get("separate_object_interface_refs", []) or []))
        interfaces = []
        if interface_refs:
            interfaces = _resolve_records(
                interface_refs, interface_index, "separate_object_interface", errors
            )

        analytic_union = alternative.get("analytic_union_volume")
        if not isinstance(analytic_union, Mapping):
            errors.append("analytic_union_volume_missing")
            analytic_union = {}
        elif not analytic_union.get("evidence_refs"):
            errors.append("analytic_union_volume_evidence_missing")

        errors = sorted(set(errors))
        hard_pre_kernel_contradiction = bool(errors) and all(
            error == "construction_region_seam_graph_disconnected"
            or error.startswith("internal_seam_semantics_contradiction:")
            for error in errors
        )
        evaluation = {
            "id": alternative_id,
            "record_type": "constructive_union_hypothesis_evaluation",
            "source_assignment_ref": alternative.get("source_assignment_ref"),
            "physical_object_scope_ref": scope_ref,
            "construction_region_refs": [str(item.get("id")) for item in regions],
            "internal_seam_refs": [str(item.get("id")) for item in seams],
            "internal_seam_semantics_certificate": seam_semantics_certificate,
            "supplied_projection_refs": [str(item.get("id")) for item in views],
            "pre_kernel_rejection_reasons": errors,
            "pre_kernel_rejection_classification": (
                "hard_contradiction" if hard_pre_kernel_contradiction else "unresolved"
            ) if errors else None,
            "materialization_status": "rejected" if errors else "complete",
            "step4_same_object_union_replayed": False,
            "step4_status": "not_invoked",
            "quantity_eligible": False,
        }
        if errors:
            evaluations.append(evaluation)
            continue

        complete_hypotheses.append(
            {
                "id": alternative_id,
                "physical_object_scope_ref": scope_ref,
                "construction_region_refs": evaluation["construction_region_refs"],
                "internal_seam_refs": evaluation["internal_seam_refs"],
                "supplied_projection_refs": evaluation["supplied_projection_refs"],
                "analytic_union_volume": deepcopy(dict(analytic_union)),
            }
        )
        evaluation["step4_same_object_union_replayed"] = True
        try:
            kernel = solve_same_object_constructive_solid(
                scope,
                regions,
                seams,
                analytic_union,
                views,
                separate_object_interfaces=interfaces,
            )
        except (ValueError, TypeError, KeyError) as error:
            evaluation["step4_status"] = "reclosed_fail"
            evaluation["step4_reason"] = str(error)
            evaluation["step4_reasons"] = list(
                getattr(error, "errors", [str(error)])
            )
            evaluation["step4_residual_diagnostics"] = deepcopy(
                getattr(error, "diagnostics", {})
            )
            kernel_replays.append(
                {
                    "constructive_union_hypothesis_ref": alternative_id,
                    "status": "reclosed_fail",
                    "reason": str(error),
                    "reasons": evaluation["step4_reasons"],
                    "residual_diagnostics": evaluation[
                        "step4_residual_diagnostics"
                    ],
                }
            )
        else:
            evaluation["step4_status"] = "accepted"
            evaluation["quantity_eligible"] = True
            replay = {
                "constructive_union_hypothesis_ref": alternative_id,
                "source_assignment_ref": alternative.get("source_assignment_ref"),
                "supplied_projection_refs": evaluation["supplied_projection_refs"],
                "status": "accepted",
                "kernel_result": kernel,
            }
            kernel_replays.append(replay)
            survivors.append(replay)
        evaluations.append(evaluation)

    materialized_count = len(complete_hypotheses)
    rejection_count = sum(item["materialization_status"] == "rejected" for item in evaluations)
    hard_rejection_count = sum(
        item.get("pre_kernel_rejection_classification") == "hard_contradiction"
        for item in evaluations
    )
    elimination_count = sum(item["materialization_status"] == "eliminated" for item in evaluations)
    replay_count = sum(item["step4_same_object_union_replayed"] for item in evaluations)
    survivor_count = len(survivors)
    unresolved_count = rejection_count - hard_rejection_count
    assignment_refs = {
        str(item.get("source_assignment_ref"))
        for item in placement_alternatives
        if item.get("source_assignment_ref")
    }
    assignment_count = len(assignment_refs) if assignment_refs else len(placement_alternatives)
    equivalence_certificate = certify_invariant_union_equivalence(
        survivors,
        supplied_projection_evidence,
        all_competing_alternatives_resolved=unresolved_count == 0,
    )
    equivalence_accepted = bool(
        equivalence_certificate
        and equivalence_certificate.get("state") == "accepted"
    )
    if not placement_alternatives:
        status, reason_code = "insufficient_constraints", "placement_alternatives_absent"
    elif survivor_count > 1 and equivalence_accepted:
        status, reason_code = "accepted_invariant_equivalence_class", None
    elif survivor_count > 1:
        status, reason_code = "ambiguous_survivors", "multiple_step4_union_survivors"
    elif survivor_count == 1 and unresolved_count:
        status, reason_code = "insufficient_constraints", "unreplayed_competing_alternatives"
    elif survivor_count == 1:
        status, reason_code = "accepted_unique_union", None
    elif replay_count == 0 and hard_rejection_count == len(placement_alternatives):
        status, reason_code = "reclosed_fail", "all_placement_alternatives_hard_rejected"
    elif replay_count == 0:
        all_reasons = {
            reason
            for item in evaluations
            for reason in item["pre_kernel_rejection_reasons"]
        }
        reason_code = (
            "regions_unmaterialized"
            if any(
                reason.startswith("construction_region_")
                or reason.startswith("physical_object_scope_unresolved")
                for reason in all_reasons
            )
            else "complete_union_hypotheses_unresolved"
        )
        status = "insufficient_constraints"
    elif unresolved_count:
        status, reason_code = "insufficient_constraints", "unreplayed_competing_alternatives"
    else:
        status, reason_code = "reclosed_fail", "all_complete_union_hypotheses_rejected"

    accepted_alternative_ref = (
        str(survivors[0]["constructive_union_hypothesis_ref"])
        if status == "accepted_unique_union" and len(survivors) == 1
        else None
    )
    for evaluation in evaluations:
        evaluation["quantity_eligible"] = (
            str(evaluation["id"]) == accepted_alternative_ref
        )
    for replay in kernel_replays:
        replay["quantity_eligible"] = (
            str(replay["constructive_union_hypothesis_ref"])
            == accepted_alternative_ref
        )
        replay["invariant_equivalence_class_member"] = bool(
            equivalence_accepted
            and str(replay["constructive_union_hypothesis_ref"])
            in set(equivalence_certificate["member_hypothesis_refs"])
        )

    normalized = {
        "physical_object_scopes": sorted(
            (deepcopy(dict(item)) for item in physical_object_scopes),
            key=lambda item: str(item.get("id")),
        ),
        "construction_region_hypotheses": sorted(
            (deepcopy(dict(item)) for item in construction_region_hypotheses),
            key=lambda item: str(item.get("id")),
        ),
        "placement_alternatives": [deepcopy(dict(item)) for item in placement_alternatives],
        "internal_seam_hypotheses": sorted(
            (deepcopy(dict(item)) for item in internal_seam_hypotheses),
            key=lambda item: str(item.get("id")),
        ),
        "supplied_projection_evidence": sorted(
            (deepcopy(dict(item)) for item in supplied_projection_evidence),
            key=lambda item: str(item.get("id")),
        ),
        "separate_object_interfaces": sorted(
            (deepcopy(dict(item)) for item in separate_object_interfaces),
            key=lambda item: str(item.get("id")),
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "constructive_union_enumerator",
        "status": status,
        "reason_code": reason_code,
        "input_sha256": sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "evaluations": evaluations,
        "complete_union_hypotheses": complete_hypotheses,
        "kernel_replays": kernel_replays,
        "survivors": survivors,
        "invariant_union_equivalence_certificate": equivalence_certificate,
        "invariant_union_volume_candidate": (
            equivalence_certificate.get("invariant_union_volume_candidate")
            if equivalence_accepted
            else None
        ),
        "summary": {
            "assignment_count": assignment_count,
            "placement_alternative_count": len(placement_alternatives),
            "materialized_union_hypothesis_count": materialized_count,
            "pre_kernel_rejection_count": rejection_count,
            "hard_pre_kernel_rejection_count": hard_rejection_count,
            "certified_elimination_count": elimination_count,
            "unresolved_live_alternative_count": unresolved_count,
            "kernel_replay_count": replay_count,
            "survivor_count": survivor_count,
            "transition": (
                f"{assignment_count} assignments -> {materialized_count} materialized union hypotheses -> "
                f"{rejection_count} pre-kernel rejections -> {replay_count} kernel replays -> "
                f"{survivor_count} survivors"
            ),
        },
        "quantity_eligible": status in {
            "accepted_unique_union",
            "accepted_invariant_equivalence_class",
        },
        "contract": {
            "arbitrary_region_count_supported": True,
            "arbitrary_placement_alternatives_supported": True,
            "counts_derive_from_evaluation": True,
            "every_complete_hypothesis_replayed_through_step4": True,
            "unevaluated_hypotheses_are_not_ambiguity": True,
            "unreplayed_live_alternatives_block_unique_acceptance": True,
            "hard_contradiction_or_dominance_certificate_required_for_elimination": True,
            "unique_survivor_or_certified_invariant_equivalence_class_required_for_quantity": True,
            "equal_survivor_volumes_alone_are_not_equivalence": True,
            "connected_construction_region_seam_graph_required_before_step4": True,
            "seam_graph_connectivity_is_necessary_not_sufficient": True,
            "assigned_interface_caps_require_matching_contact_semantics": True,
            "overlap_union_requires_independent_authorization": True,
            "candidate_mesh_intersection_is_not_overlap_authorization": True,
            "schedule_values_used": False,
        },
    }
