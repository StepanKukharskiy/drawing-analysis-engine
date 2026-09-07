"""Versioned, provenance-preserving engineer review deltas.

Review decisions never mutate the immutable observation or canonical drawing
graph.  They are validated against stable canonical IDs and materialized as a
separate effective overlay for deterministic downstream replay.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from src.drawing_engine.core.canonical_knowledge_graph import RELATION_TYPES


SCHEMA_VERSION = "0.1.0"
DECISIONS = {"accepted", "rejected", "corrected"}
DELTA_LAYERS = {"engineer_review_delta", "automatic_adjudication_delta"}
TASK_TYPES = {
    "primitive_role",
    "view_role",
    "leader_target",
    "dimension_ownership",
    "cross_view_identity",
    "contour_correspondence",
    "section_parent",
    "spacing_ownership",
    "other",
}
ATTRIBUTE_FIELDS = {
    "role",
    "role_hypothesis",
    "semantic_role",
    "projection_dimensionality",
    "mark",
    "material_role",
}

_ROOT_KEYS = {
    "schema_version",
    "layer",
    "document_key",
    "base_canonical_graph_sha256",
    "catalog_sha256",
    "reviewer",
    "created_at",
    "decisions",
    "contract",
}
_DECISION_KEYS = {
    "decision_id",
    "candidate_id",
    "task_type",
    "decision",
    "evidence_refs",
    "reason_code",
    "note",
    "correction",
}
_CORRECTION_KEYS = {
    "relation": {"kind", "relation_type", "from", "to"},
    "attribute": {"kind", "entity_id", "field", "value"},
    "geometry": {"kind", "entity_id", "geometry_display"},
}

_RELATION_TASK = {
    "callout_targets": "leader_target",
    "dimension_of": "dimension_ownership",
    "detail_defines": "cross_view_identity",
    "projects_to": "cross_view_identity",
    "same_bar_family": "cross_view_identity",
    "cut_at": "section_parent",
    "section_of": "section_parent",
    "spacing_of": "spacing_ownership",
}

_ENTITY_TASK = {
    "view": "view_role",
    "projected_path": "primitive_role",
    "mark_occurrence": "primitive_role",
    "contour": "contour_correspondence",
}

_LEADER_TARGET_CANDIDATE_LIMIT = 8
_SINGLE_LEADER_GROUP_KIND = "single_target_leader_group"
_SINGLE_LEADER_PAIR_KIND = "single_target_leader_pair"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any, length: int = 20) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()[:length]


def canonical_graph_sha256(canonical_graph: Mapping[str, Any]) -> str:
    """Hash the exact canonical graph content used as the review base."""

    return sha256(_json(canonical_graph).encode("utf-8")).hexdigest()


def _score(record: Mapping[str, Any]) -> float | None:
    for container in (record, record.get("attributes") or {}):
        for key in ("confidence", "score", "candidate_score", "association_score"):
            value = container.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0.0, min(1.0, float(value)))
    return None


def _engine_decision(record: Mapping[str, Any], *, subject_kind: str) -> str:
    status = str(record.get("status") or "").lower()
    state = str(record.get("state") or "").lower()
    if status == "rejected":
        return "rejected"
    if subject_kind == "relation" and (
        status in {"accepted", "resolved", "pass"}
        or state in {"direct", "observed", "derived"}
    ):
        return "accepted"
    if subject_kind == "entity":
        return "proposed"
    return "abstained"


def _predicted_value(entity: Mapping[str, Any]) -> Any:
    attributes = entity.get("attributes") or {}
    for key in (
        "role_hypothesis",
        "semantic_role",
        "projection_dimensionality",
        "mark",
        "token",
    ):
        if attributes.get(key) is not None:
            return deepcopy(attributes[key])
    return None


def _unresolved_task(item: Mapping[str, Any]) -> str:
    relation_type = item.get("relation_type")
    if relation_type in _RELATION_TASK:
        return _RELATION_TASK[str(relation_type)]
    kind = str(item.get("kind") or "")
    if "dimension_ownership" in kind:
        return "dimension_ownership"
    if "cut_at" in kind or "section" in kind:
        return "section_parent"
    if "contour" in kind:
        return "contour_correspondence"
    return "other"


def _point_segment_distance(point: list[float], start: list[float], end: list[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _path_distance_to_terminals(path: Mapping[str, Any], terminals: list[list[float]]) -> float:
    fragments = path.get("attributes", {}).get("source_fragments", []) or []
    distances: list[float] = []
    for fragment in fragments:
        points = fragment.get("geometry", {}).get("points_display") or []
        if len(points) < 2:
            continue
        for terminal in terminals:
            for start, end in zip(points, points[1:]):
                if len(start) >= 2 and len(end) >= 2:
                    distances.append(_point_segment_distance(terminal, start, end))
    if distances:
        return min(distances)
    bbox = path.get("attributes", {}).get("bbox_display") or []
    if len(bbox) != 4:
        return math.inf
    x0, y0, x1, y1 = (float(value) for value in bbox)
    for x, y in terminals:
        closest = [max(x0, min(x1, float(x))), max(y0, min(y1, float(y)))]
        distances.append(math.hypot(float(x) - closest[0], float(y) - closest[1]))
    return min(distances, default=math.inf)


def _ranked_nearby_paths(
    mark: Mapping[str, Any],
    page_paths: Iterable[Mapping[str, Any]],
    trace_primitive_refs: set[str],
    terminals: list[list[float]],
) -> list[tuple[float, str, Mapping[str, Any]]]:
    ranked: list[tuple[float, str, Mapping[str, Any]]] = []
    mark_views = set(mark.get("attributes", {}).get("view_ids", []) or [])
    for path in page_paths:
        path_primitive_refs = {
            str(reference)
            for fragment in path.get("attributes", {}).get("source_fragments", []) or []
            for reference in (fragment.get("primitive_ref"), fragment.get("source_path_ref"))
            if reference
        }
        if trace_primitive_refs & path_primitive_refs:
            continue
        path_views = set(path.get("attributes", {}).get("view_ids", []) or [])
        view_penalty = 0 if not mark_views or mark_views & path_views else 1
        distance = _path_distance_to_terminals(path, terminals)
        if math.isfinite(distance):
            ranked.append((distance + 1_000_000.0 * view_penalty, str(path["id"]), path))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def _vague_leader_review_candidates(canonical_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose missing leader targets without promoting proximity to engineering truth."""

    document_key = str(canonical_graph.get("document_key") or "")
    entities = list(canonical_graph.get("entities", []) or [])
    paths_by_page: dict[str, list[Mapping[str, Any]]] = {}
    for entity in entities:
        if entity.get("entity_type") != "projected_path":
            continue
        page_key = str(entity.get("provenance", {}).get("page_key") or "")
        paths_by_page.setdefault(page_key, []).append(entity)
    targets_by_mark: dict[str, set[str]] = {}
    for relation in canonical_graph.get("relations", []) or []:
        if relation.get("type") == "callout_targets":
            targets_by_mark.setdefault(str(relation.get("from")), set()).add(str(relation.get("to")))

    candidates: list[dict[str, Any]] = []
    for mark in entities:
        if mark.get("entity_type") != "mark_occurrence":
            continue
        trace = mark.get("attributes", {}).get("leader_trace") or {}
        trace_primitive_refs = {
            str(segment.get("drawing_ref"))
            for segment in trace.get("segments", []) or []
            if segment.get("drawing_ref")
        }
        terminals = [
            [float(point[0]), float(point[1])]
            for point in trace.get("terminals", []) or []
            if isinstance(point, list) and len(point) >= 2
        ]
        if not terminals:
            continue
        mark_id = str(mark["id"])
        existing_targets = sorted(targets_by_mark.get(mark_id, set()))
        if len(existing_targets) == 1:
            continue
        page_key = str(mark.get("provenance", {}).get("page_key") or "")
        evidence_refs = sorted(mark.get("provenance", {}).get("evidence_refs", []))
        group_id = f"review.leader_target_group.{_digest([document_key, mark_id])}"
        candidates.append(
            {
                "candidate_id": group_id,
                "task_type": "leader_target",
                "subject_kind": "unresolved",
                "subject_id": group_id,
                "engine_decision": "abstained",
                "score": None,
                "evidence_refs": evidence_refs,
                "page_key": page_key,
                "unresolved": {
                    "kind": "vague_leader_target_group",
                    "mark_id": mark_id,
                    "existing_target_ids": existing_targets,
                    "reason": "leader trace does not close to exactly one canonical projected path",
                    "evidence_refs": evidence_refs,
                    "page_key": page_key,
                },
            }
        )
        if existing_targets:
            continue
        ranked = _ranked_nearby_paths(
            mark,
            paths_by_page.get(page_key, []),
            trace_primitive_refs,
            terminals,
        )
        for penalized_distance, path_id, path in ranked[:_LEADER_TARGET_CANDIDATE_LIMIT]:
            distance = _path_distance_to_terminals(path, terminals)
            pair_id = f"review.leader_target_pair.{_digest([document_key, mark_id, path_id])}"
            pair_evidence = sorted(
                set(evidence_refs) | set(path.get("provenance", {}).get("evidence_refs", []))
            )
            candidates.append(
                {
                    "candidate_id": pair_id,
                    "task_type": "leader_target",
                    "subject_kind": "unresolved",
                    "subject_id": pair_id,
                    "relation_type": "callout_targets",
                    "from": mark_id,
                    "to": path_id,
                    "engine_decision": "abstained",
                    "score": None,
                    "evidence_refs": pair_evidence,
                    "page_key": page_key,
                    "unresolved": {
                        "kind": "vague_leader_target_pair",
                        "mark_id": mark_id,
                        "target_id": path_id,
                        "terminal_distance_points": round(distance, 6),
                        "reason": "proximity-ranked review proposal; deterministic identity remains unresolved",
                        "evidence_refs": pair_evidence,
                        "page_key": page_key,
                    },
                }
            )
    return candidates


def _single_target_leader_review_candidates(canonical_graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Add blind alternatives for traces that currently have one canonical target."""

    document_key = str(canonical_graph.get("document_key") or "")
    entities = list(canonical_graph.get("entities", []) or [])
    paths_by_page: dict[str, list[Mapping[str, Any]]] = {}
    for entity in entities:
        if entity.get("entity_type") == "projected_path":
            page_key = str(entity.get("provenance", {}).get("page_key") or "")
            paths_by_page.setdefault(page_key, []).append(entity)
    targets_by_mark: dict[str, set[str]] = {}
    for relation in canonical_graph.get("relations", []) or []:
        if relation.get("type") == "callout_targets":
            targets_by_mark.setdefault(str(relation.get("from")), set()).add(str(relation.get("to")))

    candidates: list[dict[str, Any]] = []
    for mark in entities:
        if mark.get("entity_type") != "mark_occurrence":
            continue
        trace = mark.get("attributes", {}).get("leader_trace") or {}
        terminals = [
            [float(point[0]), float(point[1])]
            for point in trace.get("terminals", []) or []
            if isinstance(point, list) and len(point) >= 2
        ]
        if not terminals:
            continue
        mark_id = str(mark["id"])
        existing_targets = sorted(targets_by_mark.get(mark_id, set()))
        if len(existing_targets) != 1:
            continue
        page_key = str(mark.get("provenance", {}).get("page_key") or "")
        evidence_refs = sorted(mark.get("provenance", {}).get("evidence_refs", []))
        group_id = f"review.single_leader_target_group.{_digest([document_key, mark_id])}"
        candidates.append(
            {
                "candidate_id": group_id,
                "task_type": "leader_target",
                "subject_kind": "unresolved",
                "subject_id": group_id,
                "engine_decision": "abstained",
                "score": None,
                "evidence_refs": evidence_refs,
                "page_key": page_key,
                "unresolved": {
                    "kind": _SINGLE_LEADER_GROUP_KIND,
                    "mark_id": mark_id,
                    "existing_target_ids": existing_targets,
                    "reason": "single canonical target held out for blind engineer verification",
                    "evidence_refs": evidence_refs,
                    "page_key": page_key,
                },
            }
        )
        trace_primitive_refs = {
            str(segment.get("drawing_ref"))
            for segment in trace.get("segments", []) or []
            if segment.get("drawing_ref")
        }
        ranked = _ranked_nearby_paths(
            mark,
            paths_by_page.get(page_key, []),
            trace_primitive_refs,
            terminals,
        )
        alternative_count = 0
        for _penalized_distance, path_id, path in ranked:
            if path_id in existing_targets:
                continue
            distance = _path_distance_to_terminals(path, terminals)
            pair_id = f"review.single_leader_target_pair.{_digest([document_key, mark_id, path_id])}"
            pair_evidence = sorted(
                set(evidence_refs) | set(path.get("provenance", {}).get("evidence_refs", []))
            )
            candidates.append(
                {
                    "candidate_id": pair_id,
                    "task_type": "leader_target",
                    "subject_kind": "unresolved",
                    "subject_id": pair_id,
                    "relation_type": "callout_targets",
                    "from": mark_id,
                    "to": path_id,
                    "engine_decision": "abstained",
                    "score": None,
                    "evidence_refs": pair_evidence,
                    "page_key": page_key,
                    "unresolved": {
                        "kind": _SINGLE_LEADER_PAIR_KIND,
                        "mark_id": mark_id,
                        "target_id": path_id,
                        "terminal_distance_points": round(distance, 6),
                        "reason": "proximity-ranked blind alternative; deterministic identity remains unchanged",
                        "evidence_refs": pair_evidence,
                        "page_key": page_key,
                    },
                }
            )
            alternative_count += 1
            if alternative_count >= _LEADER_TARGET_CANDIDATE_LIMIT - 1:
                break
    return candidates


def build_review_candidate_catalog(canonical_graph: Mapping[str, Any]) -> dict[str, Any]:
    """Expose stable, reviewable candidates without changing graph semantics."""

    document_key = str(canonical_graph.get("document_key") or "")
    graph_hash = canonical_graph_sha256(canonical_graph)
    candidates: list[dict[str, Any]] = []
    for entity in canonical_graph.get("entities", []) or []:
        task_type = _ENTITY_TASK.get(str(entity.get("entity_type")))
        if task_type is None:
            continue
        candidates.append(
            {
                "candidate_id": str(entity["id"]),
                "task_type": task_type,
                "subject_kind": "entity",
                "subject_id": str(entity["id"]),
                "engine_decision": _engine_decision(entity, subject_kind="entity"),
                "predicted_value": _predicted_value(entity),
                "score": _score(entity),
                "evidence_refs": sorted(entity.get("provenance", {}).get("evidence_refs", [])),
                "page_key": entity.get("provenance", {}).get("page_key"),
            }
        )
    for relation in canonical_graph.get("relations", []) or []:
        relation_type = str(relation.get("type") or "")
        task_type = _RELATION_TASK.get(relation_type)
        if task_type is None:
            continue
        candidates.append(
            {
                "candidate_id": str(relation["id"]),
                "task_type": task_type,
                "subject_kind": "relation",
                "subject_id": str(relation["id"]),
                "relation_type": relation_type,
                "from": relation.get("from"),
                "to": relation.get("to"),
                "engine_decision": _engine_decision(relation, subject_kind="relation"),
                "score": _score(relation),
                "evidence_refs": sorted(relation.get("provenance", {}).get("evidence_refs", [])),
                "page_key": relation.get("provenance", {}).get("page_key"),
            }
        )
    for item in canonical_graph.get("unresolved", []) or []:
        candidate_id = f"review.candidate.{_digest([document_key, item])}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "task_type": _unresolved_task(item),
                "subject_kind": "unresolved",
                "subject_id": candidate_id,
                "engine_decision": "abstained",
                "score": _score(item),
                "evidence_refs": sorted(item.get("evidence_refs", [])),
                "page_key": item.get("page_key"),
                "unresolved": deepcopy(item),
            }
        )
    candidates.extend(_vague_leader_review_candidates(canonical_graph))
    candidates.sort(
        key=lambda item: (
            item["task_type"],
            float(item.get("unresolved", {}).get("terminal_distance_points", math.inf)),
            str(item.get("to") or ""),
            item["candidate_id"],
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": "review_candidate_catalog",
        "document_key": document_key,
        "base_canonical_graph_sha256": graph_hash,
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "by_task": {
                task: sum(item["task_type"] == task for item in candidates)
                for task in sorted(TASK_TYPES)
            },
        },
        "contract": {
            "candidate_ids_reference_frozen_canonical_graph": True,
            "schedule_values_used": False,
            "catalog_does_not_change_graph_state": True,
        },
    }
    payload["catalog_sha256"] = sha256(_json(payload).encode("utf-8")).hexdigest()
    return payload


def build_single_target_leader_review_catalog(canonical_graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build an isolated catalog for blind verification of unique leader targets."""

    base = build_review_candidate_catalog(canonical_graph)
    candidates = deepcopy(base["candidates"])
    entities = {str(item["id"]): item for item in canonical_graph.get("entities", []) or []}
    single_marks = {
        str(item.get("unresolved", {}).get("mark_id"))
        for item in _single_target_leader_review_candidates(canonical_graph)
        if item.get("unresolved", {}).get("kind") == _SINGLE_LEADER_GROUP_KIND
    }
    for candidate in candidates:
        if candidate.get("relation_type") != "callout_targets" or str(candidate.get("from")) not in single_marks:
            continue
        mark = entities.get(str(candidate.get("from")), {})
        target = entities.get(str(candidate.get("to")), {})
        terminals = [
            [float(point[0]), float(point[1])]
            for point in mark.get("attributes", {}).get("leader_trace", {}).get("terminals", []) or []
            if isinstance(point, list) and len(point) >= 2
        ]
        distance = _path_distance_to_terminals(target, terminals)
        if math.isfinite(distance):
            candidate["terminal_distance_points"] = round(distance, 6)
    candidates.extend(_single_target_leader_review_candidates(canonical_graph))
    candidates.sort(
        key=lambda item: (
            item["task_type"],
            float(item.get("unresolved", {}).get("terminal_distance_points", math.inf)),
            str(item.get("to") or ""),
            item["candidate_id"],
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": "single_target_leader_review_candidate_catalog",
        "document_key": base["document_key"],
        "base_canonical_graph_sha256": base["base_canonical_graph_sha256"],
        "base_catalog_sha256": base["catalog_sha256"],
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "single_target_case_count": sum(
                item.get("unresolved", {}).get("kind") == _SINGLE_LEADER_GROUP_KIND
                for item in candidates
            ),
            "by_task": {
                task: sum(item["task_type"] == task for item in candidates)
                for task in sorted(TASK_TYPES)
            },
        },
        "contract": {
            "candidate_ids_reference_frozen_canonical_graph": True,
            "single_target_review_is_separate_from_vague_review": True,
            "schedule_values_used": False,
            "catalog_does_not_change_graph_state": True,
        },
    }
    payload["catalog_sha256"] = sha256(_json(payload).encode("utf-8")).hexdigest()
    return payload


def _review_catalog_for_record(
    canonical_graph: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    base = build_review_candidate_catalog(canonical_graph)
    if record.get("catalog_sha256") == base["catalog_sha256"]:
        return base
    single_target = build_single_target_leader_review_catalog(canonical_graph)
    if record.get("catalog_sha256") == single_target["catalog_sha256"]:
        return single_target
    return base


def create_review_record(
    canonical_graph: Mapping[str, Any],
    *,
    reviewer_id: str,
    reviewer_role: str = "engineer",
    layer: str = "engineer_review_delta",
    created_at: str | None = None,
    decisions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if layer not in DELTA_LAYERS:
        raise ValueError(f"unsupported review delta layer: {layer}")
    catalog = build_review_candidate_catalog(canonical_graph)
    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "document_key": catalog["document_key"],
        "base_canonical_graph_sha256": catalog["base_canonical_graph_sha256"],
        "catalog_sha256": catalog["catalog_sha256"],
        "reviewer": {"id": str(reviewer_id), "role": str(reviewer_role)},
        "created_at": timestamp,
        "decisions": [deepcopy(dict(item)) for item in decisions],
        "contract": {
            "observation_graph_mutated": False,
            "canonical_graph_mutated": False,
            "schedule_values_used": False,
            "review_is_a_separate_overlay": True,
        },
    }


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_review_record(
    canonical_graph: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate schema, graph binding, canonical targets, and trust boundaries."""

    errors: list[str] = []
    catalog = _review_catalog_for_record(canonical_graph, record)
    candidate_by_id = {item["candidate_id"]: item for item in catalog["candidates"]}
    entity_ids = {str(item["id"]) for item in canonical_graph.get("entities", []) or []}
    unknown_root_keys = set(record) - _ROOT_KEYS
    if unknown_root_keys:
        errors.append(f"unsupported review fields: {sorted(unknown_root_keys)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported review schema_version")
    if record.get("layer") not in DELTA_LAYERS:
        errors.append(f"review layer must be one of {sorted(DELTA_LAYERS)}")
    if record.get("document_key") != catalog["document_key"]:
        errors.append("review document_key does not match canonical graph")
    if record.get("base_canonical_graph_sha256") != catalog["base_canonical_graph_sha256"]:
        errors.append("review base graph hash does not match canonical graph")
    if record.get("catalog_sha256") != catalog["catalog_sha256"]:
        errors.append("review candidate catalog hash does not match canonical graph")
    reviewer = record.get("reviewer") or {}
    if isinstance(reviewer, Mapping) and set(reviewer) - {"id", "role"}:
        errors.append("reviewer contains unsupported fields")
    if not str(reviewer.get("id") or "").strip():
        errors.append("reviewer.id is required")
    if not str(reviewer.get("role") or "").strip():
        errors.append("reviewer.role is required")
    if not _valid_timestamp(record.get("created_at")):
        errors.append("created_at must be an ISO-8601 timestamp with timezone")
    contract = record.get("contract") or {}
    if contract.get("schedule_values_used") is not False:
        errors.append("review contract must state schedule_values_used=false")
    decisions = record.get("decisions")
    if not isinstance(decisions, list):
        errors.append("decisions must be a list")
        decisions = []
    if not decisions and not allow_empty:
        errors.append("at least one review decision is required")
    decision_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        prefix = f"decisions[{index}]"
        if not isinstance(decision, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        unknown_decision_keys = set(decision) - _DECISION_KEYS
        if unknown_decision_keys:
            errors.append(f"{prefix} contains unsupported fields: {sorted(unknown_decision_keys)}")
        decision_id = str(decision.get("decision_id") or "")
        if not decision_id:
            errors.append(f"{prefix}.decision_id is required")
        elif decision_id in decision_ids:
            errors.append(f"duplicate decision_id: {decision_id}")
        decision_ids.add(decision_id)
        candidate_id = str(decision.get("candidate_id") or "")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            errors.append(f"{prefix}.candidate_id is not in the frozen catalog")
        elif decision.get("task_type") != candidate["task_type"]:
            errors.append(f"{prefix}.task_type does not match the candidate catalog")
        if candidate_id in candidate_ids:
            errors.append(f"duplicate candidate decision: {candidate_id}")
        candidate_ids.add(candidate_id)
        outcome = decision.get("decision")
        if outcome not in DECISIONS:
            errors.append(f"{prefix}.decision is unsupported")
        evidence_refs = decision.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not all(isinstance(item, str) and item for item in evidence_refs):
            errors.append(f"{prefix}.evidence_refs must be a list of non-empty strings")
        elif any("schedule" in item.casefold() or "declared" in item.casefold() for item in evidence_refs):
            errors.append(f"{prefix}.evidence_refs cannot cite declared schedule evidence")
        if not str(decision.get("reason_code") or "").strip():
            errors.append(f"{prefix}.reason_code is required")
        correction = decision.get("correction")
        if outcome == "corrected" and not isinstance(correction, Mapping):
            errors.append(f"{prefix}.correction is required for a corrected decision")
            continue
        if outcome != "corrected" and correction is not None:
            errors.append(f"{prefix}.correction is allowed only for a corrected decision")
            continue
        if outcome != "corrected":
            continue
        kind = correction.get("kind")
        if kind in _CORRECTION_KEYS and set(correction) - _CORRECTION_KEYS[kind]:
            errors.append(f"{prefix}.correction contains unsupported fields")
        if kind == "relation":
            if correction.get("relation_type") not in RELATION_TYPES:
                errors.append(f"{prefix}.correction.relation_type is unsupported")
            if correction.get("from") not in entity_ids or correction.get("to") not in entity_ids:
                errors.append(f"{prefix}.correction relation endpoints must be canonical entities")
        elif kind == "attribute":
            if correction.get("entity_id") not in entity_ids:
                errors.append(f"{prefix}.correction.entity_id must be canonical")
            if correction.get("field") not in ATTRIBUTE_FIELDS:
                errors.append(f"{prefix}.correction.field is not reviewable")
            if correction.get("value") is None:
                errors.append(f"{prefix}.correction.value is required")
        elif kind == "geometry":
            if correction.get("entity_id") not in entity_ids:
                errors.append(f"{prefix}.correction.entity_id must be canonical")
            if not correction.get("geometry_display"):
                errors.append(f"{prefix}.correction.geometry_display is required")
        else:
            errors.append(f"{prefix}.correction.kind is unsupported")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "decision_count": len(decisions),
        "candidate_count": len(candidate_by_id),
        "base_canonical_graph_sha256": catalog["base_canonical_graph_sha256"],
        "schedule_values_used": False,
    }


def apply_review_record(canonical_graph: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize reviewed entities/relations as an overlay; keep the base untouched."""

    base_hash = canonical_graph_sha256(canonical_graph)
    validation = validate_review_record(canonical_graph, record)
    if validation["status"] != "pass":
        raise ValueError("invalid review record: " + "; ".join(validation["errors"]))
    catalog = _review_catalog_for_record(canonical_graph, record)
    candidate_by_id = {item["candidate_id"]: item for item in catalog["candidates"]}
    entities = {str(item["id"]): deepcopy(item) for item in canonical_graph.get("entities", []) or []}
    relations = {str(item["id"]): deepcopy(item) for item in canonical_graph.get("relations", []) or []}
    annotations: list[dict[str, Any]] = []
    created_relations = 0
    for decision in record["decisions"]:
        candidate = candidate_by_id[decision["candidate_id"]]
        outcome = decision["decision"]
        annotation = {
            "decision_id": decision["decision_id"],
            "candidate_id": decision["candidate_id"],
            "task_type": decision["task_type"],
            "decision": outcome,
            "reason_code": decision["reason_code"],
            "evidence_refs": list(decision["evidence_refs"]),
            "reviewer": deepcopy(record["reviewer"]),
        }
        if candidate["subject_kind"] == "relation" and candidate["subject_id"] in relations:
            if outcome in {"rejected", "corrected"}:
                relations.pop(candidate["subject_id"], None)
            else:
                relations[candidate["subject_id"]]["review_status"] = "accepted"
                relations[candidate["subject_id"]]["review_decision_id"] = decision["decision_id"]
        elif candidate["subject_kind"] == "entity" and candidate["subject_id"] in entities:
            entities[candidate["subject_id"]]["review_status"] = outcome
            entities[candidate["subject_id"]]["review_decision_id"] = decision["decision_id"]
        correction = decision.get("correction")
        if correction:
            annotation["correction"] = deepcopy(correction)
            if correction["kind"] == "relation":
                relation_id = f"reviewed.relation.{_digest([base_hash, decision['decision_id'], correction])}"
                relations[relation_id] = {
                    "id": relation_id,
                    "type": correction["relation_type"],
                    "from": correction["from"],
                    "to": correction["to"],
                    "state": "inferred",
                    "review_status": "corrected",
                    "review_decision_id": decision["decision_id"],
                    "provenance": {
                        "document_key": record["document_key"],
                        "source": "engineer_review_delta",
                        "evidence_refs": list(decision["evidence_refs"]),
                    },
                }
                created_relations += 1
            elif correction["kind"] == "attribute":
                entity = entities[correction["entity_id"]]
                entity.setdefault("reviewed_attributes", {})[correction["field"]] = deepcopy(correction["value"])
            elif correction["kind"] == "geometry":
                entities[correction["entity_id"]]["reviewed_geometry_display"] = deepcopy(correction["geometry_display"])
        annotations.append(annotation)
    if canonical_graph_sha256(canonical_graph) != base_hash:
        raise RuntimeError("base canonical graph was mutated while applying review")
    effective_relations = sorted(relations.values(), key=lambda item: item["id"])
    entity_ids = set(entities)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "reviewed_canonical_overlay",
        "document_key": record["document_key"],
        "base_canonical_graph_sha256": base_hash,
        "effective_entities": sorted(entities.values(), key=lambda item: item["id"]),
        "effective_relations": effective_relations,
        "review_annotations": annotations,
        "validation": {
            "status": "pass",
            "base_graph_unchanged": canonical_graph_sha256(canonical_graph) == base_hash,
            "relation_endpoints_exist": all(
                item["from"] in entity_ids and item["to"] in entity_ids
                for item in effective_relations
            ),
            "schedule_values_used": False,
        },
        "summary": {
            "decision_count": len(annotations),
            "accepted_count": sum(item["decision"] == "accepted" for item in annotations),
            "rejected_count": sum(item["decision"] == "rejected" for item in annotations),
            "corrected_count": sum(item["decision"] == "corrected" for item in annotations),
            "created_reviewed_relation_count": created_relations,
        },
        "contract": {
            "base_observations_remain_authoritative": True,
            "reviewed_does_not_mean_direct": True,
            "schedule_values_used": False,
        },
    }


def load_review_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review record must be a JSON object")
    return payload


def write_review_record(
    path: Path,
    canonical_graph: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> Path:
    validation = validate_review_record(canonical_graph, record, allow_empty=allow_empty)
    if validation["status"] != "pass":
        raise ValueError("invalid review record: " + "; ".join(validation["errors"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path
