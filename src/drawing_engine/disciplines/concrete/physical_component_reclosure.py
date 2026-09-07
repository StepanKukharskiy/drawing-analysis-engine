"""Deterministic Step 5 Slice 2 cross-view identity and placement replay.

Every quantity-free physical-component hypothesis is replayed against its
accepted parent/section relation and shared relative coordinate scope.  A
certificate passes only when all identity, placement, transform, extent,
interface, and non-additivity gates close.  This layer does not assign a
``physical_component_ref``, publish a physical transform, invoke Step 4, or
emit a quantity; those actions remain Slice 3 responsibilities.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Iterable, Mapping

from src.drawing_engine.disciplines.concrete.unsigned_bounded_sweep_certificate import (
    validate_unsigned_bounded_sweep_certificate,
)


SCHEMA_VERSION = "0.1.0"
_GATE_ORDER = (
    "relation_replay",
    "unique_component_grouping",
    "independently_supported_depth_and_position",
    "relative_axis_signs_and_offsets",
    "compatible_metric_extents",
    "explicit_interfaces",
    "separate_profiles_do_not_imply_additive_count",
)
_PRIMARY_BLOCKER_ORDER = (
    "relative_axis_signs_and_offsets",
    "unique_component_grouping",
    "independently_supported_depth_and_position",
    "explicit_interfaces",
    "relation_replay",
    "compatible_metric_extents",
    "separate_profiles_do_not_imply_additive_count",
)


def _stable_id(kind: str, page_number: int, *parts: str) -> str:
    encoded = "\0".join((str(page_number), *parts)).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _gate(status: str, reason_code: str | None, reason: str, evidence_refs: Iterable[Any] = (), **fields: Any) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "evidence_refs": sorted({str(item) for item in evidence_refs if item is not None and str(item)}),
        **fields,
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _known_refs(*records: Any) -> set[str]:
    refs: set[str] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                if child_key == "id" and child is not None:
                    refs.add(str(child))
                if child_key.endswith("_ref") and child is not None:
                    refs.add(str(child))
                elif child_key.endswith("_refs") and isinstance(child, (list, tuple, set)):
                    refs.update(str(item) for item in child if item is not None and str(item))
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)

    for record in records:
        visit(record)
    return refs


def _relation_gate(
    hypothesis: Mapping[str, Any],
    object_scopes: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    physical_scope_ref = str(hypothesis.get("physical_scope_ref") or "")
    scope = object_scopes.get(physical_scope_ref)
    if scope is None:
        return _gate("fail", "physical_scope_missing", "hypothesis physical scope is absent from the replay graph", [physical_scope_ref]), None, None
    relation_refs = sorted(set(map(str, scope.get("relation_refs", []) or [])))
    candidates = [relations.get(ref) for ref in relation_refs]
    if any(item is None for item in candidates):
        return _gate("fail", "relation_reference_missing", "physical scope cites a relation absent from the replay graph", [physical_scope_ref, *relation_refs]), scope, None
    accepted = [
        item
        for item in candidates
        if item is not None
        and item.get("type") == "cut_at"
        and item.get("state") == "accepted"
        and item.get("integration_certificate", {}).get("status") == "passed"
    ]
    if len(accepted) != 1:
        return _gate(
            "insufficient_constraints",
            "unique_parent_section_relation_unresolved",
            "exactly one accepted parent/section relation must support the physical scope",
            [physical_scope_ref, *relation_refs],
            candidate_relation_refs=relation_refs,
        ), scope, None
    relation = accepted[0]
    parent_ref = str(relation.get("parent_view_id") or "")
    section_ref = str(relation.get("section_view_id") or "")
    if (
        parent_ref not in set(map(str, scope.get("parent_view_ids", []) or []))
        or section_ref not in set(map(str, scope.get("section_view_ids", []) or []))
    ):
        return _gate(
            "fail",
            "relation_scope_membership_contradiction",
            "accepted relation parent or section view contradicts physical-scope membership",
            [physical_scope_ref, relation.get("id"), parent_ref, section_ref],
        ), scope, relation
    return _gate(
        "pass",
        None,
        "unique accepted parent/section relation reclosed",
        [physical_scope_ref, relation.get("id"), parent_ref, section_ref],
        relation_ref=str(relation.get("id")),
        parent_view_ref=parent_ref,
        section_view_ref=section_ref,
    ), scope, relation


def _grouping_gate(
    hypothesis: Mapping[str, Any],
    alternative_sets: Mapping[str, Mapping[str, Any]],
    groupings_by_scope: Mapping[str, list[Mapping[str, Any]]],
    hypotheses_by_id: Mapping[str, Mapping[str, Any]],
    known_refs: set[str],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    hypothesis_ref = str(hypothesis.get("id") or "")
    physical_scope_ref = str(hypothesis.get("physical_scope_ref") or "")
    alternative_ref = str(hypothesis.get("alternative_set_ref") or "")
    alternative = alternative_sets.get(alternative_ref)
    if alternative is None or hypothesis_ref not in set(map(str, alternative.get("hypothesis_refs", []) or [])):
        return _gate("fail", "alternative_set_contradiction", "hypothesis does not belong to its claimed alternative set", [hypothesis_ref, alternative_ref]), None
    resolved = [item for item in groupings_by_scope.get(physical_scope_ref, []) if item.get("state") == "resolved"]
    if not resolved:
        return _gate(
            "insufficient_constraints",
            "component_grouping_evidence_missing",
            "no resolved component-grouping record selects physical hypotheses in this scope",
            [physical_scope_ref, hypothesis_ref, alternative_ref],
        ), None
    if len(resolved) != 1:
        return _gate(
            "insufficient_constraints",
            "component_grouping_not_unique",
            "more than one resolved component grouping remains",
            [physical_scope_ref, hypothesis_ref, *(item.get("id") for item in resolved)],
            candidate_grouping_refs=sorted(str(item.get("id")) for item in resolved),
        ), None
    grouping = resolved[0]
    selected = list(map(str, grouping.get("hypothesis_refs", []) or []))
    unknown = sorted(set(selected) - hypotheses_by_id.keys())
    wrong_scope = sorted(
        ref
        for ref in selected
        if ref in hypotheses_by_id
        and str(hypotheses_by_id[ref].get("physical_scope_ref") or "")
        != physical_scope_ref
    )
    if not selected or unknown or wrong_scope or len(selected) != len(set(selected)):
        return _gate(
            "fail",
            "component_grouping_reference_contradiction",
            "resolved grouping has empty, duplicate, unknown, or cross-scope hypothesis references",
            [str(grouping.get("id")), *selected, *unknown, *wrong_scope],
        ), grouping
    selected_alternatives = [
        str(hypotheses_by_id[ref].get("alternative_set_ref") or "")
        for ref in selected
    ]
    if len(selected_alternatives) != len(set(selected_alternatives)):
        return _gate(
            "fail",
            "multiple_alternatives_selected_as_additive_components",
            "one grouping cannot select multiple hypotheses from the same alternative set",
            [str(grouping.get("id")), *selected, *selected_alternatives],
        ), grouping
    scope_hypotheses = {
        ref
        for ref, item in hypotheses_by_id.items()
        if str(item.get("physical_scope_ref") or "") == physical_scope_ref
    }
    excluded_records = {
        str(item.get("hypothesis_ref")): item
        for item in grouping.get("excluded_hypothesis_records", []) or []
        if item.get("hypothesis_ref") is not None
    }
    expected_excluded = scope_hypotheses - set(selected)
    if grouping.get("completeness") != "complete" or set(excluded_records) != expected_excluded:
        return _gate(
            "insufficient_constraints",
            "component_grouping_incomplete",
            "unique grouping must explicitly classify every unselected hypothesis",
            [str(grouping.get("id")), *scope_hypotheses, *excluded_records],
            unclassified_hypothesis_refs=sorted(expected_excluded - excluded_records.keys()),
        ), grouping
    for excluded_ref, record in excluded_records.items():
        evidence_refs = set(map(str, record.get("evidence_refs", []) or []))
        if (
            record.get("state") != "resolved"
            or record.get("disposition")
            not in {
                "rejected_alternative",
                "supporting_projection_only",
                "duplicate_projection",
            }
            or not evidence_refs
            or not evidence_refs <= known_refs
        ):
            return _gate(
                "fail",
                "excluded_hypothesis_disposition_invalid",
                "every unselected hypothesis needs a resolved evidence-backed disposition",
                [str(grouping.get("id")), excluded_ref, *evidence_refs],
            ), grouping
    if hypothesis_ref not in selected:
        return _gate(
            "insufficient_constraints",
            "hypothesis_not_selected_by_unique_grouping",
            "the unique grouping selects another bounded alternative",
            [hypothesis_ref, str(grouping.get("id")), *selected],
            selected_hypothesis_refs=selected,
        ), grouping
    return _gate(
        "pass",
        None,
        "one resolved scope-level grouping selects this hypothesis",
        [hypothesis_ref, str(grouping.get("id")), *selected],
        grouping_ref=str(grouping.get("id")),
        selected_hypothesis_refs=selected,
    ), grouping


def _placement_gate(
    hypothesis: Mapping[str, Any],
    relation: Mapping[str, Any] | None,
    placements_by_hypothesis: Mapping[str, list[Mapping[str, Any]]],
    known_refs: set[str],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    hypothesis_ref = str(hypothesis.get("id") or "")
    resolved = [item for item in placements_by_hypothesis.get(hypothesis_ref, []) if item.get("state") == "resolved"]
    if not resolved:
        return _gate("insufficient_constraints", "depth_and_position_evidence_missing", "no resolved component placement record supplies independent depth and position", [hypothesis_ref]), None
    if len(resolved) != 1:
        return _gate("insufficient_constraints", "component_placement_not_unique", "more than one resolved component placement remains", [hypothesis_ref, *(item.get("id") for item in resolved)]), None
    placement = resolved[0]
    if relation is None:
        return _gate("insufficient_constraints", "relation_required_for_placement", "placement cannot be replayed before the parent/section relation closes", [hypothesis_ref, placement.get("id")]), placement
    expected = {
        "physical_scope_ref": str(hypothesis.get("physical_scope_ref") or ""),
        "relation_ref": str(relation.get("id") or ""),
        "parent_view_ref": str(relation.get("parent_view_id") or ""),
        "section_view_ref": str(relation.get("section_view_id") or ""),
    }
    contradictions = [key for key, value in expected.items() if str(placement.get(key) or "") != value]
    if contradictions or set(map(str, placement.get("profile_refs", []) or [])) != set(map(str, hypothesis.get("profile_refs", []) or [])):
        return _gate("fail", "placement_scope_contradiction", "placement relation, views, scope, or profile membership contradicts the hypothesis", [hypothesis_ref, placement.get("id"), *expected.values()], contradictory_fields=contradictions), placement
    depth = placement.get("depth", {}) or {}
    position = placement.get("position", {}) or {}
    depth_refs = set(map(str, depth.get("dimension_refs", []) or []))
    position_refs = set(map(str, position.get("dimension_refs", []) or []))
    coordinates = position.get("coordinates_mm", {}) or {}
    support_refs = depth_refs | position_refs
    placement_evidence = set(map(str, placement.get("evidence_refs", []) or []))
    if (
        depth.get("state") != "resolved"
        or not _finite_number(depth.get("value_mm"))
        or float(depth["value_mm"]) <= 0
        or not depth_refs
    ):
        return _gate("insufficient_constraints", "independent_depth_unresolved", "positive depth with dimension evidence is required", [hypothesis_ref, placement.get("id"), *depth_refs]), placement
    if (
        position.get("state") != "resolved"
        or len(coordinates) < 2
        or not position_refs
        or any(not _finite_number(value) for value in coordinates.values())
    ):
        return _gate("insufficient_constraints", "independent_position_unresolved", "at least two finite in-plane coordinates with dimension evidence are required", [hypothesis_ref, placement.get("id"), *position_refs]), placement
    if depth_refs & position_refs:
        return _gate("fail", "depth_position_evidence_not_independent", "the same dimension evidence cannot independently establish both depth and position", [hypothesis_ref, placement.get("id"), *(depth_refs & position_refs)]), placement
    if not support_refs <= placement_evidence or not support_refs <= known_refs:
        return _gate("fail", "placement_evidence_reference_missing", "depth or position cites evidence absent from the frozen upstream graph", [hypothesis_ref, placement.get("id"), *support_refs]), placement
    return _gate(
        "pass",
        None,
        "depth and in-plane position have disjoint drawing-dimension support",
        [hypothesis_ref, placement.get("id"), *support_refs],
        placement_ref=str(placement.get("id")),
        depth_dimension_refs=sorted(depth_refs),
        position_dimension_refs=sorted(position_refs),
    ), placement


def _axis_gate(
    hypothesis: Mapping[str, Any],
    scope: Mapping[str, Any] | None,
    relation: Mapping[str, Any] | None,
    placement: Mapping[str, Any] | None,
    coordinates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    hypothesis_ref = str(hypothesis.get("id") or "")
    coordinate_ref = str((scope or {}).get("shared_coordinate_scope_id") or "")
    coordinate = coordinates.get(coordinate_ref)
    if coordinate is None:
        return _gate("fail", "shared_coordinate_scope_missing", "physical scope shared-coordinate record is absent", [hypothesis_ref, coordinate_ref])
    mappings = {
        str(item.get("view_id")): item
        for item in coordinate.get("view_axis_mappings", []) or []
        if item.get("view_id") is not None
    }
    required_views = {
        str((relation or {}).get("parent_view_id") or ""),
        str((relation or {}).get("section_view_id") or ""),
    } - {""}
    if not required_views or any(mappings.get(ref, {}).get("state") != "resolved_relative" for ref in required_views):
        return _gate("insufficient_constraints", "relative_axis_mapping_unresolved", "parent and section views need resolved-relative axis mappings", [hypothesis_ref, coordinate_ref, *required_views])
    orientation_certificate = coordinate.get("signed_orientation_certificate") or {}
    unsigned_certificate = (placement or {}).get("unsigned_bounded_sweep_certificate") or {}
    unsigned_mode = bool(
        orientation_certificate.get("status") not in {"accepted", "contradiction"}
        and unsigned_certificate.get("status") == "accepted"
        and not validate_unsigned_bounded_sweep_certificate(unsigned_certificate)
        and str(unsigned_certificate.get("hypothesis_ref") or "") == hypothesis_ref
        and str(unsigned_certificate.get("physical_scope_ref") or "")
        == str(hypothesis.get("physical_scope_ref") or "")
        and str(unsigned_certificate.get("signed_orientation_certificate_ref") or "")
        == str(orientation_certificate.get("id") or "")
    )
    if orientation_certificate and orientation_certificate.get("status") != "accepted":
        constraint_certificates = orientation_certificate.get("constraint_certificates", []) or []
        marker_statuses = sorted(
            {
                str(item.get("oriented_cutting_plane", {}).get("status") or "unknown")
                for item in constraint_certificates
            }
        )
        landmark_statuses = sorted(
            {
                str(item.get("signed_shared_axis", {}).get("status") or "unknown")
                for item in constraint_certificates
            }
        )
        if not unsigned_mode:
            contradiction = orientation_certificate.get("status") == "contradiction"
            return _gate(
                "fail" if contradiction else "insufficient_constraints",
                "signed_orientation_certificate_contradiction" if contradiction else "signed_orientation_certificate_unresolved",
                "orientation-sensitive placement requires the scope-level signed certificate; no accepted sign-invariant bounded-sweep certificate applies",
                [hypothesis_ref, coordinate_ref, orientation_certificate.get("id")],
                oriented_cutting_plane_statuses=marker_statuses,
                signed_shared_axis_statuses=landmark_statuses,
            )
    correspondence_rows = [item for item in coordinate.get("contour_correspondences", []) or [] if item.get("state") == "accepted"]
    if len(correspondence_rows) != 1:
        return _gate("insufficient_constraints", "unique_contour_correspondence_unresolved", "exactly one accepted contour correspondence is required", [hypothesis_ref, coordinate_ref, *(item.get("id") for item in correspondence_rows)])
    correspondence = correspondence_rows[0]
    signed = correspondence.get("selected", {}).get("signed_transform", {}) or {}
    candidates = signed.get("candidates", []) or []
    if signed.get("state") != "resolved" and not unsigned_mode:
        reason_code = "mirrored_contour_transform_unresolved" if len(candidates) > 1 else "signed_contour_transform_unresolved"
        return _gate(
            "insufficient_constraints",
            reason_code,
            str(signed.get("reason") or "signed contour transform is unresolved"),
            [hypothesis_ref, coordinate_ref, correspondence.get("id"), *correspondence.get("selected", {}).get("primitive_refs", [])],
            signed_transform_candidates=candidates,
        )
    signed_mapping = signed.get("child_coordinate_to_parent", {}) or {}
    if not unsigned_mode and (
        not isinstance(signed_mapping.get("sign"), int)
        or isinstance(signed_mapping.get("sign"), bool)
        or signed_mapping.get("sign") not in {-1, 1}
        or not _finite_number(signed_mapping.get("offset_mm"))
    ):
        return _gate("fail", "signed_contour_transform_invalid", "resolved contour transform lacks a valid sign and offset", [hypothesis_ref, coordinate_ref, correspondence.get("id")])
    cut_rows = [item for item in coordinate.get("cut_plane_constraints", []) or [] if str(item.get("relation_id")) == str((relation or {}).get("id"))]
    if len(cut_rows) != 1:
        return _gate("insufficient_constraints", "unique_cut_plane_position_unresolved", "exactly one cut-plane position constraint is required", [hypothesis_ref, coordinate_ref, *(item.get("relation_id") for item in cut_rows)])
    cut = cut_rows[0]
    coordinate_candidates = cut.get("coordinate_candidates_mm", []) or []
    if (
        not coordinate_candidates
        or any(not _finite_number(value) for value in coordinate_candidates)
        or (not unsigned_mode and len(coordinate_candidates) != 1)
        or (
            cut.get("viewing_sign_state") not in {"resolved", "irrelevant"}
            and not unsigned_mode
        )
    ):
        return _gate("insufficient_constraints", "relative_cut_plane_sign_and_offset_unresolved", "cut-plane sign and offset remain ambiguous", [hypothesis_ref, coordinate_ref, cut.get("relation_id")], coordinate_candidates_mm=coordinate_candidates)
    if placement is None:
        return _gate("insufficient_constraints", "component_relative_transform_missing", "resolved component-relative transform evidence is missing", [hypothesis_ref, coordinate_ref])
    transform = placement.get("relative_transform", {}) or {}
    axis_signs = transform.get("axis_signs", {}) or {}
    offsets = transform.get("offsets_mm", {}) or {}
    object_axes = {str(item.get(key)) for item in mappings.values() for key in ("u", "v", "normal") if item.get(key)}
    canonical_axis_signs = transform.get("canonical_axis_signs", {}) or {}
    if (
        transform.get("state")
        not in ({"resolved_up_to_reflection"} if unsigned_mode else {"resolved"})
        or set(axis_signs) != object_axes
        or set(offsets) != object_axes
        or any(
            (
                value is not None
                if unsigned_mode
                else not isinstance(value, int)
                or isinstance(value, bool)
                or value not in {-1, 1}
            )
            for value in axis_signs.values()
        )
        or any(not _finite_number(value) for value in offsets.values())
        or (
            unsigned_mode
            and (
                set(canonical_axis_signs) != object_axes
                or any(value != 1 for value in canonical_axis_signs.values())
            )
        )
    ):
        return _gate("insufficient_constraints", "component_relative_transform_unresolved", "component transform must resolve every relative axis sign and offset", [hypothesis_ref, placement.get("id"), coordinate_ref])
    section_mapping = transform.get("section_to_parent", {}) or {}
    if unsigned_mode:
        certificate_candidates = unsigned_certificate.get("signed_transform_alternatives", []) or []
        placement_candidates = section_mapping.get("candidates", []) or []
        if section_mapping.get("state") != "unresolved_reflection" or certificate_candidates != placement_candidates:
            return _gate(
                "fail",
                "unsigned_bounded_sweep_transform_contradiction",
                "placement reflection alternatives contradict the accepted unsigned bounded-sweep certificate",
                [hypothesis_ref, placement.get("id"), unsigned_certificate.get("id")],
            )
        return _gate(
            "pass",
            None,
            "signed orientation remains unresolved; the accepted plan-bounded sweep is invariant under both mirrored transforms",
            [
                hypothesis_ref,
                placement.get("id"),
                coordinate_ref,
                correspondence.get("id"),
                cut.get("relation_id"),
                unsigned_certificate.get("id"),
            ],
            shared_coordinate_scope_ref=coordinate_ref,
            orientation_sensitivity="sign_invariant",
            signed_orientation_state="unresolved",
            unsigned_bounded_sweep_certificate_ref=unsigned_certificate.get("id"),
            canonical_axis_signs=dict(sorted(canonical_axis_signs.items())),
            offsets_mm={key: float(offsets[key]) for key in sorted(offsets)},
        )
    if (
        section_mapping.get("sign") != signed_mapping.get("sign")
        or not _finite_number(section_mapping.get("offset_mm"))
        or abs(float(section_mapping["offset_mm"]) - float(signed_mapping["offset_mm"])) > 1e-6
        or not _finite_number(transform.get("cut_plane_coordinate_mm"))
        or abs(float(transform["cut_plane_coordinate_mm"]) - float(coordinate_candidates[0])) > 1e-6
    ):
        return _gate("fail", "relative_transform_contradiction", "component transform contradicts the reclosed contour or cut-plane transform", [hypothesis_ref, placement.get("id"), coordinate_ref, correspondence.get("id")])
    return _gate(
        "pass",
        None,
        "relative axis signs and offsets reclose against contour and cut-plane evidence",
        [hypothesis_ref, placement.get("id"), coordinate_ref, correspondence.get("id"), cut.get("relation_id")],
        shared_coordinate_scope_ref=coordinate_ref,
        axis_signs=dict(sorted(axis_signs.items())),
        offsets_mm={key: float(offsets[key]) for key in sorted(offsets)},
    )


def _metric_gate(
    hypothesis: Mapping[str, Any],
    scope: Mapping[str, Any] | None,
    relation: Mapping[str, Any] | None,
    placement: Mapping[str, Any] | None,
    coordinates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    certificate = hypothesis.get("compatibility_certificate", {}) or {}
    if certificate.get("status") != "passed" or certificate.get("metric_extents", {}).get("status") != "passed":
        return _gate("fail", "hypothesis_metric_extent_contradiction", "Slice 1 metric compatibility certificate is not passed", [hypothesis.get("id")])
    metric_pair = (relation or {}).get("integration_certificate", {}).get("relation_scoped_metric_pair", {}) or {}
    if metric_pair.get("status") != "passed":
        return _gate("insufficient_constraints", "relation_metric_extents_unresolved", "parent and section relation lacks a passed metric pair", [hypothesis.get("id"), (relation or {}).get("id")])
    coordinate_ref = str((scope or {}).get("shared_coordinate_scope_id") or "")
    coordinate = coordinates.get(coordinate_ref, {})
    validations = []
    for index, item in enumerate(coordinate.get("reprojection_validations", []) or [], start=1):
        validation_ref = str(item.get("id") or f"{coordinate_ref}.reprojection.{index}")
        validations.append((validation_ref, item))
    cited = set(map(str, (scope or {}).get("reprojection_validation_refs", []) or []))
    by_ref = dict(validations)
    if not cited or not cited <= by_ref.keys():
        return _gate("fail", "metric_reprojection_reference_missing", "physical scope cites missing metric reprojection evidence", [hypothesis.get("id"), coordinate_ref, *cited])
    if any(by_ref[ref].get("status") == "fail" for ref in cited):
        return _gate("fail", "metric_extent_reprojection_contradiction", "a cited metric reprojection failed", [hypothesis.get("id"), coordinate_ref, *cited])
    if any(by_ref[ref].get("status") != "pass" for ref in cited):
        return _gate("insufficient_constraints", "metric_extent_reprojection_unresolved", "every cited metric reprojection must pass", [hypothesis.get("id"), coordinate_ref, *cited])
    placement_refs = set(map(str, (placement or {}).get("metric_extent_refs", []) or []))
    if placement is None or not cited <= placement_refs:
        return _gate("insufficient_constraints", "component_metric_extent_support_missing", "component placement does not cite every passed cross-view metric extent", [hypothesis.get("id"), coordinate_ref, *cited])
    return _gate("pass", None, "profile, relation, and shared-coordinate metric extents are compatible", [hypothesis.get("id"), (relation or {}).get("id"), coordinate_ref, *cited], metric_extent_refs=sorted(cited))


def _interface_gate(
    hypothesis: Mapping[str, Any],
    grouping: Mapping[str, Any] | None,
    known_refs: set[str],
) -> dict[str, Any]:
    hypothesis_ref = str(hypothesis.get("id") or "")
    if grouping is None or hypothesis_ref not in set(map(str, grouping.get("hypothesis_refs", []) or [])):
        return _gate("insufficient_constraints", "component_interfaces_wait_for_unique_grouping", "interfaces cannot close before one grouping selects this hypothesis", [hypothesis_ref])
    selected = set(map(str, grouping.get("hypothesis_refs", []) or []))
    interfaces = grouping.get("interfaces", {}) or {}
    if interfaces.get("state") != "resolved" or interfaces.get("completeness") != "complete":
        return _gate("insufficient_constraints", "explicit_interfaces_missing", "resolved complete interface evidence is required for the selected grouping", [hypothesis_ref, grouping.get("id")])
    records = interfaces.get("records", []) or []
    if len(selected) == 1:
        if records or interfaces.get("no_external_interfaces_required") is not True:
            return _gate("fail", "single_component_interface_contradiction", "a single selected component must explicitly declare that no external interface is required", [hypothesis_ref, grouping.get("id")])
        return _gate("pass", None, "unique single-component grouping explicitly requires no external interface", [hypothesis_ref, grouping.get("id")], interface_refs=[])
    adjacency: dict[str, set[str]] = defaultdict(set)
    interface_refs = []
    for record in records:
        refs = list(map(str, record.get("hypothesis_refs", []) or []))
        evidence_refs = set(map(str, record.get("evidence_refs", []) or []))
        if record.get("state") != "resolved" or len(refs) != 2 or len(set(refs)) != 2 or not set(refs) <= selected or not evidence_refs or not evidence_refs <= known_refs:
            return _gate("fail", "interface_record_contradiction", "interface record is unresolved, ungrounded, or references hypotheses outside the grouping", [hypothesis_ref, grouping.get("id"), record.get("id"), *refs, *evidence_refs])
        left, right = refs
        adjacency[left].add(right)
        adjacency[right].add(left)
        interface_refs.append(str(record.get("id")))
    reached = set()
    stack = [min(selected)]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        stack.extend(adjacency.get(current, set()) - reached)
    if reached != selected:
        return _gate("insufficient_constraints", "component_interface_graph_incomplete", "explicit interfaces do not connect every selected component hypothesis", [hypothesis_ref, grouping.get("id"), *interface_refs])
    return _gate("pass", None, "explicit interface graph connects every selected component hypothesis", [hypothesis_ref, grouping.get("id"), *interface_refs], interface_refs=sorted(interface_refs))


def _non_additive_gate(
    hypothesis: Mapping[str, Any],
    grouping: Mapping[str, Any] | None,
    known_refs: set[str],
) -> dict[str, Any]:
    hypothesis_ref = str(hypothesis.get("id") or "")
    if hypothesis.get("additive_count_interpretation") != "unresolved" or hypothesis.get("additive_component_identity_established") is not False:
        return _gate("fail", "slice1_additive_promotion_contradiction", "Slice 1 hypothesis was improperly promoted to an additive count", [hypothesis_ref])
    if len(hypothesis.get("profile_refs", []) or []) > 1 and hypothesis.get("compatibility_certificate", {}).get("grouping_basis") != "shared_boundary_edges":
        return _gate("fail", "separate_profile_additive_inference", "multiple separately drawn profiles were grouped without shared-edge evidence", [hypothesis_ref, *hypothesis.get("profile_refs", [])])
    if grouping is None or hypothesis_ref not in set(map(str, grouping.get("hypothesis_refs", []) or [])):
        return _gate("insufficient_constraints", "non_additive_identity_waits_for_grouping", "non-additive identity cannot close before a unique grouping selects this hypothesis", [hypothesis_ref])
    if grouping.get("separately_drawn_profiles_are_additive") is not False:
        return _gate("fail", "separate_profile_additive_inference", "grouping treats separately drawn profiles as additive occurrences", [hypothesis_ref, grouping.get("id")])
    identity_records = {
        str(item.get("hypothesis_ref")): item
        for item in grouping.get("component_identity_records", []) or []
        if item.get("hypothesis_ref") is not None
    }
    selected = set(map(str, grouping.get("hypothesis_refs", []) or []))
    if set(identity_records) != selected:
        return _gate("insufficient_constraints", "component_identity_records_incomplete", "every selected hypothesis needs one explicit non-additive identity record", [hypothesis_ref, grouping.get("id"), *selected])
    record = identity_records[hypothesis_ref]
    evidence_refs = set(map(str, record.get("identity_evidence_refs", []) or []))
    profile_refs = set(map(str, hypothesis.get("profile_refs", []) or []))
    if (
        record.get("state") != "resolved"
        or record.get("separately_drawn_profile_basis") is not False
        or not evidence_refs
        or not evidence_refs <= known_refs
        or not evidence_refs - {hypothesis_ref, *profile_refs}
    ):
        return _gate("fail", "non_additive_identity_evidence_invalid", "component identity lacks independent evidence beyond separate profile occurrence", [hypothesis_ref, grouping.get("id"), *evidence_refs])
    return _gate("pass", None, "component identity is supported independently of separate profile occurrence", [hypothesis_ref, grouping.get("id"), *evidence_refs], identity_evidence_refs=sorted(evidence_refs))


def reclose_physical_component_hypotheses(
    hypothesis_generation: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    *,
    page_number: int,
    placement_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay every Slice 1 hypothesis through all Slice 2 acceptance gates."""

    evidence = (
        placement_evidence
        if placement_evidence is not None
        else view_frame_graph.get("component_placement_evidence", {}) or {}
    )
    hypotheses = sorted(hypothesis_generation.get("hypotheses", []) or [], key=lambda item: str(item.get("id")))
    hypotheses_by_id = {str(item.get("id")): item for item in hypotheses}
    alternative_sets = {
        str(item.get("id")): item
        for item in hypothesis_generation.get("alternative_sets", []) or []
        if item.get("id") is not None
    }
    object_scopes = {
        str(item.get("id")): item
        for item in view_frame_graph.get("object_scopes", []) or []
        if item.get("id") is not None
    }
    relations = {
        str(item.get("id")): item
        for item in view_frame_graph.get("relations", []) or []
        if item.get("id") is not None
    }
    coordinates = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
        if item.get("id") is not None
    }
    groupings_by_scope: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evidence.get("component_groupings", []) or []:
        groupings_by_scope[str(item.get("physical_scope_ref") or "")].append(item)
    placements_by_hypothesis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evidence.get("component_placements", []) or []:
        placements_by_hypothesis[str(item.get("hypothesis_ref") or "")].append(item)
    frozen_refs = _known_refs(hypothesis_generation, view_frame_graph)

    certificates = []
    for hypothesis in hypotheses:
        hypothesis_ref = str(hypothesis.get("id") or "")
        relation_gate, scope, relation = _relation_gate(hypothesis, object_scopes, relations)
        grouping_gate, grouping = _grouping_gate(
            hypothesis,
            alternative_sets,
            groupings_by_scope,
            hypotheses_by_id,
            frozen_refs,
        )
        placement_gate, placement = _placement_gate(
            hypothesis, relation, placements_by_hypothesis, frozen_refs
        )
        gates = {
            "relation_replay": relation_gate,
            "unique_component_grouping": grouping_gate,
            "independently_supported_depth_and_position": placement_gate,
            "relative_axis_signs_and_offsets": _axis_gate(
                hypothesis, scope, relation, placement, coordinates
            ),
            "compatible_metric_extents": _metric_gate(
                hypothesis, scope, relation, placement, coordinates
            ),
            "explicit_interfaces": _interface_gate(
                hypothesis, grouping, frozen_refs
            ),
            "separate_profiles_do_not_imply_additive_count": _non_additive_gate(
                hypothesis, grouping, frozen_refs
            ),
        }
        statuses = [gates[name]["status"] for name in _GATE_ORDER]
        status = (
            "reclosed_fail"
            if "fail" in statuses
            else "reclosed_pass"
            if statuses == ["pass"] * len(statuses)
            else "insufficient_constraints"
        )
        primary_gate = next(
            (name for name in _PRIMARY_BLOCKER_ORDER if gates[name]["status"] == "fail"),
            None,
        ) or next(
            (
                name
                for name in _PRIMARY_BLOCKER_ORDER
                if gates[name]["status"] == "insufficient_constraints"
            ),
            None,
        )
        primary = gates[primary_gate] if primary_gate is not None else None
        evidence_refs = sorted(
            {
                hypothesis_ref,
                *(ref for gate in gates.values() for ref in gate.get("evidence_refs", []) or []),
            }
        )
        certificate_id = _stable_id(
            "physical_component_identity_placement_certificate",
            page_number,
            hypothesis_ref,
            status,
            *(f"{name}:{gates[name]['status']}:{gates[name].get('reason_code')}" for name in _GATE_ORDER),
            *evidence_refs,
        )
        certificates.append(
            {
                "record_type": "physical_component_identity_placement_certificate",
                "record_version": SCHEMA_VERSION,
                "id": certificate_id,
                "page": page_number,
                "state": "accepted" if status == "reclosed_pass" else "abstained" if status == "insufficient_constraints" else "rejected",
                "status": status,
                "epistemic_state": "derived" if status == "reclosed_pass" else "unknown",
                "hypothesis_ref": hypothesis_ref,
                "physical_scope_ref": hypothesis.get("physical_scope_ref"),
                "gates": gates,
                "primary_blocker_gate": primary_gate,
                "reason_code": None if primary is None else primary.get("reason_code"),
                "reason": "every cross-view identity and placement gate reclosed" if primary is None else primary.get("reason"),
                "slice3_input_eligible": status == "reclosed_pass",
                "physical_component_ref": None,
                "physical_transform_ref": None,
                "step4_kernel_invocation_eligible": False,
                "quantity_eligible": False,
                "evidence_refs": evidence_refs,
            }
        )

    statuses = [item["status"] for item in certificates]
    accepted = [item for item in certificates if item["status"] == "reclosed_pass"]
    status = (
        "reclosed_fail"
        if "reclosed_fail" in statuses
        else "reclosed_pass"
        if certificates and len(accepted) == len(certificates)
        else "insufficient_constraints"
        if certificates
        else "abstained"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "physical_component_cross_view_reclosure",
        "page": page_number,
        "status": status,
        "certificates": certificates,
        "accepted_hypothesis_refs": [item["hypothesis_ref"] for item in accepted],
        "summary": {
            "hypothesis_count": len(hypotheses),
            "reclosed_pass_count": statuses.count("reclosed_pass"),
            "reclosed_fail_count": statuses.count("reclosed_fail"),
            "insufficient_constraints_count": statuses.count("insufficient_constraints"),
            "slice3_input_count": len(accepted),
            "primary_blockers": dict(
                sorted(
                    {
                        code: sum(item.get("reason_code") == code for item in certificates)
                        for code in {item.get("reason_code") for item in certificates if item.get("reason_code")}
                    }.items()
                )
            ),
        },
        "contract": {
            "all_hypotheses_replayed": True,
            "unique_component_grouping_required": True,
            "independent_depth_and_position_required": True,
            "relative_axis_signs_and_offsets_required": True,
            "compatible_metric_extents_required": True,
            "explicit_interfaces_required": True,
            "separate_profile_never_implies_additive_count": True,
            "physical_component_refs_assigned": False,
            "physical_transforms_published": False,
            "step4_kernel_invocation_authorized": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def validate_physical_component_reclosure(payload: Mapping[str, Any]) -> list[str]:
    """Validate Slice 2 certificates without crossing the Slice 3 boundary."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("layer") != "physical_component_cross_view_reclosure":
        errors.append("layer must be physical_component_cross_view_reclosure")
    contract = payload.get("contract", {}) or {}
    for field in (
        "physical_component_refs_assigned",
        "physical_transforms_published",
        "step4_kernel_invocation_authorized",
        "quantity_eligible",
        "schedule_values_used",
    ):
        if contract.get(field) is not False:
            errors.append(f"contract.{field} must be false")
    ids = []
    hypotheses = []
    accepted = []
    for index, item in enumerate(payload.get("certificates", []) or []):
        prefix = f"certificates[{index}]"
        ids.append(str(item.get("id")))
        hypotheses.append(str(item.get("hypothesis_ref")))
        if item.get("record_type") != "physical_component_identity_placement_certificate":
            errors.append(f"{prefix}.record_type must be physical_component_identity_placement_certificate")
        if item.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version must be {SCHEMA_VERSION}")
        gates = item.get("gates", {}) or {}
        if set(gates) != set(_GATE_ORDER):
            errors.append(f"{prefix}.gates must contain the complete Slice 2 gate set")
        if any(gates.get(name, {}).get("status") not in {"pass", "insufficient_constraints", "fail"} for name in _GATE_ORDER):
            errors.append(f"{prefix}.gates contain an unsupported status")
        gate_statuses = [gates.get(name, {}).get("status") for name in _GATE_ORDER]
        expected_status = "reclosed_fail" if "fail" in gate_statuses else "reclosed_pass" if gate_statuses == ["pass"] * len(_GATE_ORDER) else "insufficient_constraints"
        if item.get("status") != expected_status:
            errors.append(f"{prefix}.status does not match gate outcomes")
        if item.get("slice3_input_eligible") is not (expected_status == "reclosed_pass"):
            errors.append(f"{prefix}.slice3_input_eligible does not match reclosure")
        if expected_status == "reclosed_pass":
            accepted.append(str(item.get("hypothesis_ref")))
        if item.get("physical_component_ref") is not None or item.get("physical_transform_ref") is not None:
            errors.append(f"{prefix} cannot assign Slice 3 component or transform refs")
        if item.get("step4_kernel_invocation_eligible") is not False:
            errors.append(f"{prefix} cannot authorize Step 4")
        if item.get("quantity_eligible") is not False:
            errors.append(f"{prefix} must remain quantity-ineligible")
    if len(ids) != len(set(ids)):
        errors.append("certificate ids must be globally unique")
    if len(hypotheses) != len(set(hypotheses)):
        errors.append("every hypothesis must have exactly one Slice 2 certificate")
    if sorted(map(str, payload.get("accepted_hypothesis_refs", []) or [])) != sorted(accepted):
        errors.append("accepted_hypothesis_refs must exactly match reclosed-pass certificates")
    return errors
