"""Graph-bound review data for repeated same-mark placement pairs.

Equal mark text only generates review candidates.  It never establishes that
two callouts are additive placements, duplicate projections, fragments of one
projected bar, or separate physical objects.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping

from src.drawing_engine.project.review_feedback import canonical_graph_sha256


SCHEMA_VERSION = "0.1.0"
CATALOG_LAYER = "placement_pair_review_catalog"
DELTA_LAYER = "placement_pair_review_delta"
AUTOMATIC_LAYER = "automatic_placement_pair_adjudication"
OUTCOMES = {
    "duplicate_projection",
    "same_projected_bar_fragments",
    "additive_placements",
    "separate_objects",
    "not_a_valid_mark_pair",
    "unknown",
}

_DELTA_KEYS = {
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
_DECISION_KEYS = {"pair_id", "outcome", "note"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any, length: int = 20) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()[:length]


def _catalog_hash(catalog: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
    return sha256(_json(payload).encode("utf-8")).hexdigest()


def normalize_mark_token(value: Any) -> str:
    """Normalize harmless text variation without interpreting mark semantics."""

    return "".join(str(value or "").split()).upper()


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = [float(item) for item in value]
    return box if box[2] >= box[0] and box[3] >= box[1] else None


def _page_number(page_key: Any) -> int:
    try:
        return int(str(page_key).split(":")[-1])
    except ValueError:
        return 1


def _string_set(attributes: Mapping[str, Any], key: str) -> set[str]:
    return {str(value) for value in attributes.get(key, []) or [] if value}


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    return round(len(left & right) / len(union), 6) if union else None


def _center(box: list[float] | None) -> list[float] | None:
    if box is None:
        return None
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def _distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return None
    return round(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])), 6)


def _bbox_iou(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return round(intersection / union, 6) if union > 0 else (1.0 if left == right else 0.0)


def _bbox_distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return round(
        math.hypot(
            max(left[0] - right[2], right[0] - left[2], 0.0),
            max(left[1] - right[3], right[1] - left[3], 0.0),
        ),
        6,
    )


def _dimensionality(attributes: Mapping[str, Any]) -> tuple[Any, list[float] | None]:
    value = attributes.get("projection_dimensionality")
    if not isinstance(value, Mapping):
        return value, None
    normalized_center = value.get("normalized_center_in_view")
    if not isinstance(normalized_center, list) or len(normalized_center) < 2:
        normalized_center = None
    return value.get("value"), deepcopy(normalized_center)


def _fragment_refs(attributes: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for fragment in attributes.get("source_fragments", []) or []:
        for key in ("id", "primitive_ref", "source_path_ref"):
            if fragment.get(key):
                refs.add(str(fragment[key]))
    return refs


def _occurrence_record(
    mark: Mapping[str, Any],
    target: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> dict[str, Any]:
    mark_attributes = mark.get("attributes", {}) or {}
    target_attributes = target.get("attributes", {}) or {}
    page_key = str(mark.get("provenance", {}).get("page_key") or "")
    dimensionality, normalized_center = _dimensionality(target_attributes)
    evidence_refs = sorted(
        set(mark.get("provenance", {}).get("evidence_refs", []) or [])
        | set(target.get("provenance", {}).get("evidence_refs", []) or [])
        | set(relation.get("provenance", {}).get("evidence_refs", []) or [])
    )
    return {
        "occurrence_id": str(mark["id"]),
        "target_id": str(target["id"]),
        "relation_id": str(relation["id"]),
        "page_key": page_key,
        "page_number": _page_number(page_key),
        "mark_token": str(mark_attributes.get("token") or ""),
        "source_occurrence_type": mark_attributes.get("source_occurrence_type"),
        "mark_bbox_display": _bbox(mark_attributes.get("bbox_display")),
        "leader_trace": deepcopy(mark_attributes.get("leader_trace") or {}),
        "target_bbox_display": _bbox(target_attributes.get("bbox_display")),
        "projection_dimensionality": dimensionality,
        "normalized_center_in_view": normalized_center,
        "view_ids": sorted(_string_set(target_attributes, "view_ids")),
        "object_ids": sorted(_string_set(target_attributes, "object_ids")),
        "coordinate_scope_ids": sorted(_string_set(target_attributes, "coordinate_scope_ids")),
        "source_fragment_refs": sorted(_fragment_refs(target_attributes)),
        "evidence_refs": evidence_refs,
    }


def _pair_features(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_views, right_views = set(left["view_ids"]), set(right["view_ids"])
    left_objects, right_objects = set(left["object_ids"]), set(right["object_ids"])
    left_scopes, right_scopes = set(left["coordinate_scope_ids"]), set(right["coordinate_scope_ids"])
    left_fragments, right_fragments = set(left["source_fragment_refs"]), set(right["source_fragment_refs"])
    left_composites = set(left.get("accepted_composite_projected_bar_ids", []) or [])
    right_composites = set(right.get("accepted_composite_projected_bar_ids", []) or [])
    same_page = left["page_key"] == right["page_key"]
    return {
        "same_page": same_page,
        "same_target_id": left["target_id"] == right["target_id"],
        "left_view_count": len(left_views),
        "right_view_count": len(right_views),
        "left_object_count": len(left_objects),
        "right_object_count": len(right_objects),
        "view_id_jaccard": _jaccard(left_views, right_views),
        "object_id_jaccard": _jaccard(left_objects, right_objects),
        "coordinate_scope_jaccard": _jaccard(left_scopes, right_scopes),
        "source_fragment_jaccard": _jaccard(left_fragments, right_fragments),
        "accepted_composite_projected_bar_jaccard": _jaccard(left_composites, right_composites),
        "same_accepted_composite_projected_bar": bool(left_composites & right_composites),
        "distinct_accepted_composite_projected_bars": bool(
            left_composites and right_composites and not (left_composites & right_composites)
        ),
        "same_projection_dimensionality": (
            left["projection_dimensionality"] == right["projection_dimensionality"]
            if left["projection_dimensionality"] is not None
            and right["projection_dimensionality"] is not None
            else None
        ),
        "target_bbox_iou": (
            _bbox_iou(left["target_bbox_display"], right["target_bbox_display"])
            if same_page
            else None
        ),
        "target_center_distance_points": (
            _distance(_center(left["target_bbox_display"]), _center(right["target_bbox_display"]))
            if same_page
            else None
        ),
        "normalized_center_distance": _distance(
            left["normalized_center_in_view"], right["normalized_center_in_view"]
        ),
        "left_conflicts_with_spacing": bool(left.get("conflicting_spacing_constraint_ids")),
        "right_conflicts_with_spacing": bool(right.get("conflicting_spacing_constraint_ids")),
    }


def classify_placement_pair_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only relations certified by existing graph topology.

    Equal mark text is deliberately absent from the decision rules: it only
    created the pair.  Cross-view pairs remain unknown until a physical
    reprojection relation exists; span or proximity is not enough.
    """

    if features.get("left_conflicts_with_spacing") or features.get("right_conflicts_with_spacing"):
        return {
            "outcome": "not_a_valid_mark_pair",
            "state": "accepted",
            "certificate": "numeric_token_matches_nearby_structured_spacing_constraint",
        }
    if features.get("same_target_id") is True:
        return {
            "outcome": "duplicate_projection",
            "state": "accepted",
            "certificate": "same_unique_projected_path_target",
        }
    if features.get("same_accepted_composite_projected_bar") is True:
        return {
            "outcome": "same_projected_bar_fragments",
            "state": "accepted",
            "certificate": "same_unique_accepted_composite_projected_bar",
        }
    if (
        int(features.get("left_object_count") or 0) > 0
        and int(features.get("right_object_count") or 0) > 0
        and features.get("object_id_jaccard") == 0.0
    ):
        return {
            "outcome": "separate_objects",
            "state": "accepted",
            "certificate": "disjoint_canonical_object_scopes",
        }
    return {
        "outcome": "unknown",
        "state": "abstained",
        "certificate": None,
        "reason": (
            "no unique topology/object/reprojection certificate; distinct accepted projected composites "
            "do not by themselves prove additive physical placements"
            if features.get("distinct_accepted_composite_projected_bars")
            else "no unique topology/object/reprojection certificate; disjoint targets in one view may be "
            "fragments of one projected bar"
        ),
    }


def adjudicate_placement_pair_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Classify frozen placement pairs without mutating the canonical graph."""

    decisions = []
    for pair in catalog.get("pairs", []) or []:
        result = classify_placement_pair_features(pair.get("features", {}))
        decisions.append(
            {
                "pair_id": pair["pair_id"],
                "normalized_mark_token": pair["normalized_mark_token"],
                **result,
                "evidence_refs": deepcopy(pair.get("evidence_refs", [])),
            }
        )
    accepted = [item for item in decisions if item["state"] == "accepted"]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": AUTOMATIC_LAYER,
        "document_key": catalog.get("document_key"),
        "base_canonical_graph_sha256": catalog.get("base_canonical_graph_sha256"),
        "catalog_sha256": catalog.get("catalog_sha256"),
        "decisions": decisions,
        "summary": {
            "pair_count": len(decisions),
            "accepted_count": len(accepted),
            "abstained_count": len(decisions) - len(accepted),
            "outcomes": {
                outcome: sum(item["outcome"] == outcome for item in decisions)
                for outcome in sorted(OUTCOMES)
            },
        },
        "contract": {
            "same_mark_is_candidate_generation_only": True,
            "cross_view_requires_physical_reprojection": True,
            "disjoint_same_view_targets_prove_additive": False,
            "schedule_values_used": False,
            "canonical_graph_mutated": False,
        },
    }


def select_unresolved_placement_pair_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze only pairs that deterministic certificates did not close."""

    adjudication = adjudicate_placement_pair_catalog(catalog)
    decision_by_id = {
        str(item["pair_id"]): item
        for item in adjudication.get("decisions", []) or []
    }
    source_pairs = list(catalog.get("pairs", []) or [])
    selected = [
        deepcopy(pair)
        for pair in source_pairs
        if decision_by_id.get(str(pair["pair_id"]), {}).get("state") != "accepted"
    ]
    frozen = deepcopy(dict(catalog))
    frozen.pop("catalog_sha256", None)
    frozen["pairs"] = selected
    frozen["summary"] = {
        **deepcopy(dict(catalog.get("summary", {}) or {})),
        "pair_count": len(selected),
        "source_pair_count": len(source_pairs),
        "automatically_closed_pair_count": len(source_pairs) - len(selected),
        "automatically_closed_outcomes": {
            outcome: sum(
                decision.get("state") == "accepted" and decision.get("outcome") == outcome
                for decision in decision_by_id.values()
            )
            for outcome in sorted(OUTCOMES)
        },
    }
    frozen["filtering"] = {
        "method": "deterministic_adjudication_abstentions_only",
        "labels_used": False,
        "prior_reviews_used": False,
    }
    frozen["contract"] = {
        **deepcopy(dict(catalog.get("contract", {}) or {})),
        "automatically_certified_pairs_excluded_from_manual_review": True,
    }
    frozen["catalog_sha256"] = _catalog_hash(frozen)
    return frozen


def exclude_previously_reviewed_placement_pairs(
    catalog: Mapping[str, Any],
    pair_ids: Iterable[str],
) -> dict[str, Any]:
    """Avoid duplicate review work using pair presence, never prior outcomes."""

    excluded_ids = {str(pair_id) for pair_id in pair_ids}
    source_pairs = list(catalog.get("pairs", []) or [])
    selected = [
        deepcopy(pair)
        for pair in source_pairs
        if str(pair["pair_id"]) not in excluded_ids
    ]
    frozen = deepcopy(dict(catalog))
    frozen.pop("catalog_sha256", None)
    frozen["pairs"] = selected
    frozen["summary"] = {
        **deepcopy(dict(catalog.get("summary", {}) or {})),
        "pair_count": len(selected),
        "preexisting_review_pair_count": len(source_pairs) - len(selected),
    }
    frozen["prior_review_filtering"] = {
        "method": "stable_pair_id_presence_only",
        "prior_outcomes_used": False,
        "excluded_pair_ids": sorted(
            str(pair["pair_id"])
            for pair in source_pairs
            if str(pair["pair_id"]) in excluded_ids
        ),
    }
    frozen["contract"] = {
        **deepcopy(dict(catalog.get("contract", {}) or {})),
        "prior_review_outcomes_do_not_select_candidates": True,
    }
    frozen["catalog_sha256"] = _catalog_hash(frozen)
    return frozen


def select_distinct_composite_placement_pairs(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep unresolved pairs whose targets belong to distinct accepted composites."""

    source_pairs = list(catalog.get("pairs", []) or [])
    selected = [
        deepcopy(pair)
        for pair in source_pairs
        if pair.get("features", {}).get("distinct_accepted_composite_projected_bars") is True
    ]
    frozen = deepcopy(dict(catalog))
    frozen.pop("catalog_sha256", None)
    frozen["pairs"] = selected
    frozen["summary"] = {
        **deepcopy(dict(catalog.get("summary", {}) or {})),
        "pair_count": len(selected),
        "distinct_composite_source_pair_count": len(source_pairs),
    }
    frozen["composite_filtering"] = {
        "method": "distinct_accepted_composite_projected_bars_only",
        "labels_used": False,
        "prior_review_outcomes_used": False,
    }
    frozen["contract"] = {
        **deepcopy(dict(catalog.get("contract", {}) or {})),
        "distinct_composites_remain_relationship_candidates_only": True,
    }
    frozen["catalog_sha256"] = _catalog_hash(frozen)
    return frozen


def build_placement_pair_catalog(canonical_graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build all pairs of equal-token callouts that each have one unique target."""

    document_key = str(canonical_graph.get("document_key") or "")
    entities = {str(item["id"]): item for item in canonical_graph.get("entities", []) or []}
    composite_memberships: defaultdict[str, set[str]] = defaultdict(set)
    for entity in entities.values():
        if entity.get("entity_type") != "projected_bar_composite":
            continue
        attributes = entity.get("attributes", {}) or {}
        if attributes.get("representation_state") != "accepted":
            continue
        for path_id in attributes.get("member_projected_path_ids", []) or []:
            composite_memberships[str(path_id)].add(str(entity["id"]))
    target_relations: dict[str, list[Mapping[str, Any]]] = {}
    for relation in canonical_graph.get("relations", []) or []:
        if relation.get("type") == "callout_targets":
            target_relations.setdefault(str(relation.get("from")), []).append(relation)

    groups: dict[str, list[dict[str, Any]]] = {}
    spacing_by_page: dict[str, list[tuple[int, str, list[float] | None]]] = defaultdict(list)
    for entity in entities.values():
        if entity.get("entity_type") != "spacing_constraint":
            continue
        attributes = entity.get("attributes", {}) or {}
        value = attributes.get("spacing_mm")
        if value is None:
            continue
        page_key = str(entity.get("provenance", {}).get("page_key") or "")
        spacing_by_page[page_key].append((int(value), str(entity["id"]), _bbox(attributes.get("bbox_display"))))
    excluded_occurrence_count = 0
    for mark in entities.values():
        if mark.get("entity_type") != "mark_occurrence":
            continue
        normalized_token = normalize_mark_token(mark.get("attributes", {}).get("token"))
        relations = target_relations.get(str(mark["id"]), [])
        if not normalized_token or len(relations) != 1:
            excluded_occurrence_count += 1
            continue
        relation = relations[0]
        target = entities.get(str(relation.get("to")))
        if target is None or target.get("entity_type") != "projected_path":
            excluded_occurrence_count += 1
            continue
        occurrence = _occurrence_record(mark, target, relation)
        occurrence["accepted_composite_projected_bar_ids"] = sorted(
            composite_memberships.get(str(target["id"]), set())
        )
        occurrence["conflicting_spacing_constraint_ids"] = []
        if normalized_token.isdigit() and occurrence.get("mark_bbox_display"):
            token_value = int(normalized_token)
            conflicts = []
            for spacing_value, spacing_id, spacing_bbox in spacing_by_page.get(occurrence["page_key"], []):
                distance = _bbox_distance(occurrence["mark_bbox_display"], spacing_bbox)
                if token_value == spacing_value and distance is not None and distance <= 24.0:
                    conflicts.append(spacing_id)
            occurrence["conflicting_spacing_constraint_ids"] = sorted(conflicts)
        groups.setdefault(normalized_token, []).append(occurrence)

    pairs: list[dict[str, Any]] = []
    for normalized_token, occurrences in sorted(groups.items()):
        occurrences.sort(key=lambda item: item["occurrence_id"])
        if len(occurrences) < 2:
            continue
        for left_index, left in enumerate(occurrences):
            for right in occurrences[left_index + 1 :]:
                pair_id = f"review.placement_pair.{_digest([document_key, normalized_token, left['occurrence_id'], right['occurrence_id']])}"
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "normalized_mark_token": normalized_token,
                        "left": deepcopy(left),
                        "right": deepcopy(right),
                        "features": _pair_features(left, right),
                        "evidence_refs": sorted(set(left["evidence_refs"]) | set(right["evidence_refs"])),
                    }
                )
    pairs.sort(key=lambda item: (item["normalized_mark_token"], item["pair_id"]))
    catalog: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "layer": CATALOG_LAYER,
        "document_key": document_key,
        "base_canonical_graph_sha256": canonical_graph_sha256(canonical_graph),
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "repeated_mark_group_count": len({item["normalized_mark_token"] for item in pairs}),
            "excluded_occurrence_count": excluded_occurrence_count,
        },
        "contract": {
            "same_mark_is_candidate_generation_only": True,
            "pair_outcomes_are_not_inferred": True,
            "composite_membership_may_certify_split_fragments": True,
            "distinct_composites_do_not_prove_additive_placements": True,
            "schedule_values_used": False,
            "canonical_graph_mutated": False,
        },
    }
    catalog["catalog_sha256"] = _catalog_hash(catalog)
    return catalog


def _mark_sort_key(value: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(value), value)
    except ValueError:
        return (1, value, value)


def _feature_stratum(pair: Mapping[str, Any]) -> tuple[bool, bool, str, Any]:
    features = pair.get("features", {}) or {}
    view_overlap = features.get("view_id_jaccard")
    if view_overlap is None:
        view_bucket = "unknown_view"
    elif float(view_overlap) >= 0.999999:
        view_bucket = "same_view"
    elif float(view_overlap) <= 0.000001:
        view_bucket = "different_view"
    else:
        view_bucket = "partial_view_overlap"
    return (
        bool(features.get("distinct_accepted_composite_projected_bars")),
        bool(features.get("same_target_id")),
        view_bucket,
        features.get("same_projection_dimensionality"),
    )


def _distance_key(pair: Mapping[str, Any]) -> tuple[float, float, str]:
    features = pair.get("features", {}) or {}
    normalized = features.get("normalized_center_distance")
    display = features.get("target_center_distance_points")
    return (
        float(normalized) if isinstance(normalized, (int, float)) else -1.0,
        float(display) if isinstance(display, (int, float)) else -1.0,
        str(pair.get("pair_id") or ""),
    )


def _diverse_pairs_for_mark(pairs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order one mark's pairs by useful feature strata, never by a label."""

    representatives: dict[tuple[bool, bool, str, Any], Mapping[str, Any]] = {}
    for pair in pairs:
        stratum = _feature_stratum(pair)
        current = representatives.get(stratum)
        if current is None or _distance_key(pair) > _distance_key(current):
            representatives[stratum] = pair

    view_priority = {
        "same_view": 0,
        "different_view": 1,
        "partial_view_overlap": 2,
        "unknown_view": 3,
    }
    ordered = sorted(
        representatives.values(),
        key=lambda pair: (
            not bool(pair.get("features", {}).get("distinct_accepted_composite_projected_bars")),
            bool(pair.get("features", {}).get("same_target_id")),
            view_priority[_feature_stratum(pair)[2]],
            pair.get("features", {}).get("same_projection_dimensionality") is not True,
            -_distance_key(pair)[0],
            -_distance_key(pair)[1],
            str(pair.get("pair_id") or ""),
        ),
    )
    represented_ids = {str(pair["pair_id"]) for pair in ordered}
    remainder = sorted(
        (pair for pair in pairs if str(pair["pair_id"]) not in represented_ids),
        key=lambda pair: (
            not bool(pair.get("features", {}).get("distinct_accepted_composite_projected_bars")),
            bool(pair.get("features", {}).get("same_target_id")),
            -_distance_key(pair)[0],
            -_distance_key(pair)[1],
            str(pair.get("pair_id") or ""),
        ),
    )
    return [deepcopy(pair) for pair in ordered + remainder]


def select_diverse_placement_pair_catalog(
    catalog: Mapping[str, Any],
    *,
    max_pairs: int,
    max_pairs_per_mark: int = 2,
) -> dict[str, Any]:
    """Freeze a small deterministic, label-free subset of a large pair catalog."""

    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if max_pairs_per_mark <= 0:
        raise ValueError("max_pairs_per_mark must be positive")
    source_pairs = list(catalog.get("pairs", []) or [])
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for pair in source_pairs:
        groups.setdefault(str(pair.get("normalized_mark_token") or ""), []).append(pair)
    ordered_by_mark = {
        mark: _diverse_pairs_for_mark(pairs)
        for mark, pairs in groups.items()
    }
    marks = sorted(ordered_by_mark, key=_mark_sort_key)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    # First preserve token coverage. Later rounds add feature diversity to the
    # richest groups without allowing a large repeated group to dominate.
    for round_index in range(max_pairs_per_mark):
        round_marks = marks
        if round_index:
            round_marks = sorted(
                marks,
                key=lambda mark: (
                    -len({_feature_stratum(pair) for pair in groups[mark]}),
                    -len({pair[side]["occurrence_id"] for pair in groups[mark] for side in ("left", "right")}),
                    _mark_sort_key(mark),
                ),
            )
        for mark in round_marks:
            candidates = ordered_by_mark[mark]
            if round_index >= len(candidates):
                continue
            pair = candidates[round_index]
            pair_id = str(pair["pair_id"])
            if pair_id in selected_ids:
                continue
            selected.append(deepcopy(pair))
            selected_ids.add(pair_id)
            if len(selected) >= max_pairs:
                break
        if len(selected) >= max_pairs:
            break

    selected.sort(key=lambda item: (_mark_sort_key(str(item["normalized_mark_token"])), item["pair_id"]))
    frozen = deepcopy(dict(catalog))
    frozen.pop("catalog_sha256", None)
    frozen["pairs"] = selected
    frozen["summary"] = {
        **deepcopy(dict(catalog.get("summary", {}) or {})),
        "pair_count": len(selected),
        "source_pair_count": len(source_pairs),
        "repeated_mark_group_count": len({item["normalized_mark_token"] for item in selected}),
    }
    frozen["sampling"] = {
        "method": "deterministic_feature_strata_round_robin",
        "max_pairs": max_pairs,
        "max_pairs_per_mark": max_pairs_per_mark,
        "labels_used": False,
        "prior_reviews_used": False,
    }
    frozen["catalog_sha256"] = _catalog_hash(frozen)
    return frozen


def validate_placement_pair_review(
    canonical_graph: Mapping[str, Any],
    catalog: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on stale graphs, catalogs, malformed decisions, or unknown pairs."""

    errors: list[str] = []
    if set(record) - _DELTA_KEYS:
        errors.append(f"unexpected review keys: {sorted(set(record) - _DELTA_KEYS)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if record.get("layer") != DELTA_LAYER:
        errors.append(f"layer must be {DELTA_LAYER}")
    graph_hash = canonical_graph_sha256(canonical_graph)
    if catalog.get("base_canonical_graph_sha256") != graph_hash:
        errors.append("catalog base graph hash does not match canonical graph")
    if record.get("base_canonical_graph_sha256") != graph_hash:
        errors.append("review base graph hash does not match canonical graph")
    expected_catalog_hash = _catalog_hash(catalog)
    if catalog.get("catalog_sha256") != expected_catalog_hash:
        errors.append("catalog hash is invalid")
    if record.get("catalog_sha256") != expected_catalog_hash:
        errors.append("review catalog hash does not match frozen catalog")
    if record.get("document_key") != catalog.get("document_key"):
        errors.append("review document_key does not match catalog")

    reviewer = record.get("reviewer")
    if not isinstance(reviewer, Mapping) or not str(reviewer.get("id") or "").strip():
        errors.append("reviewer.id is required")
    contract = record.get("contract")
    required_contract = {
        "canonical_graph_mutated": False,
        "schedule_values_used": False,
        "review_is_a_separate_overlay": True,
    }
    if not isinstance(contract, Mapping) or any(contract.get(key) != value for key, value in required_contract.items()):
        errors.append("review contract must preserve graph separation and exclude schedule values")

    pair_ids = {str(item["pair_id"]) for item in catalog.get("pairs", []) or []}
    decisions = record.get("decisions")
    if not isinstance(decisions, list):
        errors.append("decisions must be a list")
        decisions = []
    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            errors.append(f"decisions[{index}] must be an object")
            continue
        if set(decision) - _DECISION_KEYS:
            errors.append(f"decisions[{index}] has unexpected keys")
        pair_id = str(decision.get("pair_id") or "")
        if pair_id not in pair_ids:
            errors.append(f"decisions[{index}].pair_id is not in the frozen catalog")
        if pair_id in seen:
            errors.append(f"decisions[{index}].pair_id is duplicated")
        seen.add(pair_id)
        if decision.get("outcome") not in OUTCOMES:
            errors.append(f"decisions[{index}].outcome is invalid")
        if "note" in decision and not isinstance(decision.get("note"), str):
            errors.append(f"decisions[{index}].note must be a string")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "decision_count": len(decisions),
        "base_graph_unchanged": True,
        "schedule_values_used": False,
    }
