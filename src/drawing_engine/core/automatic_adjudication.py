"""Deterministic, drawing-neutral adjudication of canonical graph candidates.

This pass replaces manual candidate clicking where the existing evidence graph
already proves an outcome.  It never upgrades an ambiguous visual hypothesis
merely because it has the highest score: acceptance requires a closed semantic
certificate, rejection requires an explicit contradiction or a uniquely
displaced functional alternative, and everything else remains an abstention.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from src.drawing_engine.project.review_feedback import (
    apply_review_record,
    build_review_candidate_catalog,
    create_review_record,
    validate_review_record,
)


RULESET_VERSION = "0.2.0"
AUTOMATIC_ACTOR_ID = f"automatic-adjudicator-{RULESET_VERSION}"
BAD_STATES = {"ambiguous", "candidate", "partial", "rejected", "unresolved", "unknown"}
ACCEPTED_STATES = {"direct", "observed", "derived"}
FUNCTIONAL_RELATIONS = {
    "callout_targets",
    "dimension_of",
    "detail_defines",
    "same_bar_family",
    "section_of",
    "spacing_of",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def ruleset_sha256() -> str:
    contract = {
        "version": RULESET_VERSION,
        "accepted_states": sorted(ACCEPTED_STATES),
        "bad_states": sorted(BAD_STATES),
        "functional_relations": sorted(FUNCTIONAL_RELATIONS),
        "rules": [
            "canonical_relation_already_closed",
            "section_binding_prerequisites_closed",
            "unique_object_projection_membership",
            "unique_exact_detail_mark_identity",
            "unique_exact_mark_family_identity",
            "closed_mark_family_path_triangle",
            "resolved_mark_occurrence",
            "dimension_anchored_projected_path",
            "view_role_closed_by_metric_frame",
            "dimension_owned_closed_contour",
            "explicit_hard_rejection",
            "displaced_functional_relation",
        ],
    }
    return sha256(_json(contract).encode("utf-8")).hexdigest()


def _state(record: Mapping[str, Any]) -> str:
    return str(record.get("state") or "").casefold()


def _status(record: Mapping[str, Any]) -> str:
    return str(record.get("status") or "").casefold()


def _relation_is_closed(relation: Mapping[str, Any], entity_ids: set[str]) -> bool:
    if relation.get("type") == "cut_at":
        integration = relation.get("attributes", {}).get("integration_certificate", {}) or {}
        if not (
            integration.get("status") == "passed"
            and integration.get("title_segmentation_closed") is True
            and integration.get("dimension_adjudication_closed") is True
        ):
            return False
    return (
        _state(relation) in ACCEPTED_STATES
        and _status(relation) not in BAD_STATES
        and relation.get("from") in entity_ids
        and relation.get("to") in entity_ids
    )


def _object_projection_certificate(
    relation: Mapping[str, Any],
    entity_by_id: Mapping[str, Mapping[str, Any]],
    relation_groups: Mapping[tuple[str, str], list[Mapping[str, Any]]],
) -> bool:
    """Accept only unique view/object membership with redundant geometric basis."""

    relation_type = str(relation.get("type") or "")
    source = entity_by_id.get(str(relation.get("from") or ""), {})
    target = entity_by_id.get(str(relation.get("to") or ""), {})
    basis_value = relation.get("provenance", {}).get("basis") or []
    basis = {str(item) for item in basis_value} if isinstance(basis_value, list) else set()
    required = {"strong_cross_axis_overlap", "compatible_relative_extent"}
    if not required.issubset(basis):
        return False
    if relation_type == "section_of":
        if source.get("entity_type") != "view" or target.get("entity_type") != "object":
            return False
        alternatives = relation_groups.get((relation_type, str(relation.get("from"))), [])
        return len(alternatives) == 1
    if relation_type == "projects_to":
        if source.get("entity_type") != "object" or target.get("entity_type") != "view":
            return False
        role = relation.get("attributes", {}).get("projection_role")
        if not role:
            return False
        alternatives = [
            item
            for item in relation_groups.get((relation_type, str(relation.get("from"))), [])
            if item.get("attributes", {}).get("projection_role") == role
        ]
        return len(alternatives) == 1
    return False


def _single_mark(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace(" ", "")
    match = re.fullmatch(r"M?([A-ZА-Я]?\d+[A-ZА-Я]?)", text)
    return match.group(1) if match else None


def _exact_detail_mark_certificate(
    relation: Mapping[str, Any],
    entity_by_id: Mapping[str, Mapping[str, Any]],
    relation_groups: Mapping[tuple[str, str], list[Mapping[str, Any]]],
) -> bool:
    if relation.get("type") != "detail_defines":
        return False
    detail = entity_by_id.get(str(relation.get("from") or ""), {})
    family = entity_by_id.get(str(relation.get("to") or ""), {})
    if detail.get("entity_type") != "detail_shape" or family.get("entity_type") != "bar_family":
        return False
    if _state(detail) not in {"direct", "observed", "derived"} or _status(detail) not in {"observed", "accepted", "resolved"}:
        return False
    detail_mark = _single_mark((detail.get("attributes") or {}).get("mark_display"))
    family_mark = _single_mark((family.get("attributes") or {}).get("mark"))
    alternatives = relation_groups.get(("detail_defines", str(relation.get("from"))), [])
    return detail_mark is not None and detail_mark == family_mark and len(alternatives) == 1


def _exact_mark_family_certificate(
    relation: Mapping[str, Any],
    entity_by_id: Mapping[str, Mapping[str, Any]],
    relation_groups: Mapping[tuple[str, str], list[Mapping[str, Any]]],
) -> bool:
    if relation.get("type") != "same_bar_family":
        return False
    mark = entity_by_id.get(str(relation.get("from") or ""), {})
    family = entity_by_id.get(str(relation.get("to") or ""), {})
    if mark.get("entity_type") != "mark_occurrence" or family.get("entity_type") != "bar_family":
        return False
    if _state(mark) not in ACCEPTED_STATES or _status(mark) not in {"accepted", "resolved"}:
        return False
    mark_token = _single_mark((mark.get("attributes") or {}).get("token"))
    family_mark = _single_mark((family.get("attributes") or {}).get("mark"))
    alternatives = relation_groups.get(("same_bar_family", str(relation.get("from"))), [])
    return mark_token is not None and mark_token == family_mark and len(alternatives) == 1


def _mark_family_path_triangles(
    accepted_relation_ids: set[str],
    relation_by_id: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str]]:
    family_by_mark: dict[str, set[str]] = defaultdict(set)
    paths_by_mark: dict[str, set[str]] = defaultdict(set)
    for relation_id in accepted_relation_ids:
        relation = relation_by_id[relation_id]
        if relation.get("type") == "same_bar_family":
            family_by_mark[str(relation["from"])].add(str(relation["to"]))
        elif relation.get("type") == "callout_targets":
            paths_by_mark[str(relation["from"])].add(str(relation["to"]))
    return {
        (family_id, path_id)
        for mark_id, families in family_by_mark.items()
        for family_id in families
        for path_id in paths_by_mark.get(mark_id, set())
    }


def _view_role_certificate(entity: Mapping[str, Any], incident_closed_types: set[str]) -> str | None:
    attributes = entity.get("attributes") or {}
    role = str(attributes.get("role") or "")
    confidence = attributes.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or float(confidence) < 0.82:
        return None
    if role not in {"section_view_candidate", "reinforcement_view_candidate", "formwork_view_candidate"}:
        return None
    frame = attributes.get("frame") or {}
    scale_resolved = (frame.get("scale") or {}).get("state") == "resolved"
    if role == "section_view_candidate":
        has_label = bool(frame.get("section_label"))
        if has_label and (scale_resolved or bool({"cut_at", "section_of"} & incident_closed_types)):
            return "view_role_closed_by_label_metric_frame"
        return None
    if scale_resolved and any(
        item.get("status") == "accepted" for item in frame.get("metric_spans", []) or []
    ):
        return "view_role_closed_by_metric_frame"
    return None


def _entity_outcome(
    entity: Mapping[str, Any],
    *,
    accepted_relation_types: set[str],
    dimension_owned_contours: set[str],
) -> tuple[str, str, str]:
    entity_type = str(entity.get("entity_type") or "")
    state, status = _state(entity), _status(entity)
    attributes = entity.get("attributes") or {}
    if status == "rejected" or state == "rejected":
        return "rejected", "explicit_hard_rejection", "the canonical graph records an explicit rejection"
    if entity_type == "mark_occurrence" and state in ACCEPTED_STATES and status in {"accepted", "resolved"}:
        return "accepted", "resolved_mark_occurrence", "mark identity and its connector chain are already resolved"
    if entity_type == "projected_path" and (
        status == "resolved_from_dimension_anchors_and_cross_view_validation"
        or str(attributes.get("physical_path_state") or "").startswith("resolved_from_dimension_anchors")
    ):
        return "accepted", "dimension_anchored_projected_path", "metric anchors and cross-view validation close the projected path"
    if entity_type == "view":
        rule = _view_role_certificate(entity, accepted_relation_types)
        if rule:
            return "accepted", rule, "role has a semantic label or metric frame plus independent relation support"
        return "abstained", "view_role_frame_not_closed", "view role lacks a sufficiently closed metric/semantic frame"
    if entity_type == "contour":
        closure = attributes.get("closure_validation") or {}
        if (
            str(entity.get("id")) in dimension_owned_contours
            and attributes.get("closed") is True
            and closure.get("status") == "pass"
        ):
            return "accepted", "dimension_owned_closed_contour", "closed native contour has unique accepted dimension ownership"
        return "abstained", "contour_material_or_correspondence_unresolved", "contour exists, but material ownership or cross-view correspondence is not unique"
    if entity_type == "projected_path":
        return "abstained", "projected_path_without_physical_identity", "visible 2D path is not yet a unique physical bar"
    if entity_type == "mark_occurrence":
        return "abstained", "mark_role_or_target_unresolved", "numeric token or leader target is not uniquely resolved"
    return "abstained", "entity_constraint_not_closed", "entity does not have a supported automatic acceptance certificate"


def _decision(
    catalog_candidate: Mapping[str, Any],
    *,
    outcome: str,
    reason_code: str,
    document_key: str,
    correction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(catalog_candidate["candidate_id"])
    digest = sha256(f"{document_key}\0{candidate_id}\0{RULESET_VERSION}".encode("utf-8")).hexdigest()[:20]
    evidence_refs = [
        str(item)
        for item in catalog_candidate.get("evidence_refs", []) or []
        if "schedule" not in str(item).casefold() and "declared" not in str(item).casefold()
    ]
    if not evidence_refs:
        evidence_refs = [str(catalog_candidate.get("subject_id") or candidate_id)]
    result = {
        "decision_id": f"automatic.decision.{digest}",
        "candidate_id": candidate_id,
        "task_type": catalog_candidate["task_type"],
        "decision": outcome,
        "evidence_refs": evidence_refs,
        "reason_code": reason_code,
    }
    if correction is not None:
        result["correction"] = deepcopy(dict(correction))
    return result


def adjudicate_canonical_graph(canonical_graph: Mapping[str, Any]) -> dict[str, Any]:
    """Adjudicate every review candidate and materialize the safe decided subset."""

    catalog = build_review_candidate_catalog(canonical_graph)
    entity_by_id = {str(item["id"]): item for item in canonical_graph.get("entities", []) or []}
    relation_by_id = {str(item["id"]): item for item in canonical_graph.get("relations", []) or []}
    entity_ids = set(entity_by_id)
    relation_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for relation in relation_by_id.values():
        relation_groups[(str(relation.get("type")), str(relation.get("from")))].append(relation)

    accepted_relation_rules: dict[str, str] = {}
    for relation_id, relation in relation_by_id.items():
        if _relation_is_closed(relation, entity_ids):
            accepted_relation_rules[relation_id] = (
                "section_binding_prerequisites_closed"
                if relation.get("type") == "cut_at"
                else "canonical_relation_already_closed"
            )
        elif _object_projection_certificate(relation, entity_by_id, relation_groups):
            accepted_relation_rules[relation_id] = "unique_object_projection_membership"
        elif _exact_detail_mark_certificate(relation, entity_by_id, relation_groups):
            accepted_relation_rules[relation_id] = "unique_exact_detail_mark_identity"
        elif _exact_mark_family_certificate(relation, entity_by_id, relation_groups):
            accepted_relation_rules[relation_id] = "unique_exact_mark_family_identity"
    accepted_relation_ids = set(accepted_relation_rules)
    triangle_targets = _mark_family_path_triangles(accepted_relation_ids, relation_by_id)
    for relation_id, relation in relation_by_id.items():
        if (
            relation_id not in accepted_relation_ids
            and relation.get("type") == "projects_to"
            and (str(relation.get("from")), str(relation.get("to"))) in triangle_targets
        ):
            accepted_relation_ids.add(relation_id)
            accepted_relation_rules[relation_id] = "closed_mark_family_path_triangle"
    accepted_relation_types_by_entity: dict[str, set[str]] = defaultdict(set)
    dimension_owned_contours: set[str] = set()
    for relation_id in accepted_relation_ids:
        relation = relation_by_id[relation_id]
        relation_type = str(relation["type"])
        accepted_relation_types_by_entity[str(relation["from"])].add(relation_type)
        accepted_relation_types_by_entity[str(relation["to"])].add(relation_type)
        if relation_type == "dimension_of" and entity_by_id.get(str(relation["to"]), {}).get("entity_type") == "contour":
            dimension_owned_contours.add(str(relation["to"]))

    accepted_functional_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for relation_id in accepted_relation_ids:
        relation = relation_by_id[relation_id]
        relation_type = str(relation["type"])
        if relation_type in FUNCTIONAL_RELATIONS:
            accepted_functional_targets[(relation_type, str(relation["from"]))].add(str(relation["to"]))

    outcomes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for candidate in catalog["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        outcome, rule, explanation = "abstained", "constraint_not_closed", "no unique deterministic certificate"
        if candidate.get("subject_kind") == "relation":
            relation = relation_by_id.get(str(candidate.get("subject_id")))
            if relation is None:
                rule, explanation = "missing_relation_subject", "candidate relation is absent from the canonical graph"
            elif candidate_id in accepted_relation_ids:
                outcome = "accepted"
                rule = accepted_relation_rules[candidate_id]
                explanation = {
                    "canonical_relation_already_closed": "typed relation has valid endpoints and an accepted evidence state",
                    "section_binding_prerequisites_closed": "section binding uniquely closes title-segment membership and adjudicated dimension ownership for both views",
                    "unique_object_projection_membership": "object/view membership is unique and supported by redundant projection geometry",
                    "unique_exact_detail_mark_identity": "one observed detail and one family carry the same exact single mark",
                    "unique_exact_mark_family_identity": "one resolved mark occurrence has one family with the identical mark",
                    "closed_mark_family_path_triangle": "mark-to-family and mark-to-path relations independently close the same family-to-path edge",
                }[rule]
            elif _status(relation) == "rejected" or _state(relation) == "rejected":
                outcome, rule = "rejected", "explicit_hard_rejection"
                explanation = "the canonical graph records an explicit relation rejection"
            else:
                slot = (str(relation.get("type")), str(relation.get("from")))
                accepted_targets = accepted_functional_targets.get(slot, set())
                if relation.get("type") in FUNCTIONAL_RELATIONS and accepted_targets and str(relation.get("to")) not in accepted_targets:
                    outcome, rule = "rejected", "displaced_functional_relation"
                    explanation = "a different target uniquely closes this functional semantic relation"
                else:
                    rule = "relation_inference_not_uniquely_closed"
                    explanation = "relation lacks a unique topology, ownership, or reprojection certificate"
        elif candidate.get("subject_kind") == "entity":
            entity = entity_by_id.get(str(candidate.get("subject_id")))
            if entity is not None:
                outcome, rule, explanation = _entity_outcome(
                    entity,
                    accepted_relation_types=accepted_relation_types_by_entity.get(str(entity["id"]), set()),
                    dimension_owned_contours=dimension_owned_contours,
                )
            else:
                rule, explanation = "missing_entity_subject", "candidate entity is absent from the canonical graph"
        else:
            unresolved = candidate.get("unresolved") or {}
            if _status(unresolved) == "rejected" or _state(unresolved) == "rejected":
                outcome, rule = "rejected", "explicit_hard_rejection"
                explanation = str(unresolved.get("reason") or "explicitly rejected unresolved candidate")
            elif unresolved.get("kind") == "dimension_ownership_attachment":
                rule = "dimension_ownership_not_unique"
                explanation = str(unresolved.get("reason") or "dimension endpoints do not identify one owner")
            elif unresolved.get("kind") == "cut_at_relation_candidate":
                rule = "section_parent_or_trace_not_unique"
                explanation = str(unresolved.get("reason") or "section parent or cutting trace is ambiguous")

        row = {
            "candidate_id": candidate_id,
            "task_type": candidate["task_type"],
            "subject_kind": candidate.get("subject_kind"),
            "engine_decision": candidate.get("engine_decision"),
            "automatic_outcome": outcome,
            "rule": rule,
            "explanation": explanation,
        }
        outcomes.append(row)
        if outcome in {"accepted", "rejected", "corrected"}:
            decisions.append(
                _decision(
                    candidate,
                    outcome=outcome,
                    reason_code=rule,
                    document_key=catalog["document_key"],
                )
            )

    record = create_review_record(
        canonical_graph,
        reviewer_id=AUTOMATIC_ACTOR_ID,
        reviewer_role="system",
        layer="automatic_adjudication_delta",
        decisions=decisions,
    )
    validation = validate_review_record(canonical_graph, record, allow_empty=True)
    if validation["status"] != "pass":
        raise RuntimeError("automatic adjudication produced an invalid delta: " + "; ".join(validation["errors"]))
    overlay = apply_review_record(canonical_graph, record) if decisions else None

    outcome_counts = Counter(item["automatic_outcome"] for item in outcomes)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rule_counts = Counter()
    for item in outcomes:
        task_counts[item["task_type"]][item["automatic_outcome"]] += 1
        rule_counts[item["rule"]] += 1
    newly_accepted = sum(
        item["automatic_outcome"] == "accepted" and item["engine_decision"] != "accepted"
        for item in outcomes
    )
    report = {
        "schema_version": "0.1.0",
        "layer": "automatic_adjudication_report",
        "document_key": catalog["document_key"],
        "base_canonical_graph_sha256": catalog["base_canonical_graph_sha256"],
        "catalog_sha256": catalog["catalog_sha256"],
        "ruleset": {"version": RULESET_VERSION, "sha256": ruleset_sha256()},
        "summary": {
            "candidate_count": len(outcomes),
            "accepted_count": outcome_counts["accepted"],
            "rejected_count": outcome_counts["rejected"],
            "corrected_count": outcome_counts["corrected"],
            "abstained_count": outcome_counts["abstained"],
            "automatic_completion_rate": round(
                (len(outcomes) - outcome_counts["abstained"]) / len(outcomes), 6
            ) if outcomes else None,
            "newly_accepted_count": newly_accepted,
            "confirmed_existing_accept_count": outcome_counts["accepted"] - newly_accepted,
            "by_task": {task: dict(sorted(counts.items())) for task, counts in sorted(task_counts.items())},
            "by_rule": dict(sorted(rule_counts.items())),
        },
        "outcomes": outcomes,
        "validation": validation,
        "contract": {
            "human_required_for_decided_subset": False,
            "highest_score_alone_can_accept": False,
            "abstention_preserved_when_constraints_do_not_close": True,
            "base_graph_mutated": False,
            "schedule_values_used": False,
            "quantities_require_solver_reclosure": True,
        },
    }
    return {"delta": record, "report": report, "overlay": overlay}
