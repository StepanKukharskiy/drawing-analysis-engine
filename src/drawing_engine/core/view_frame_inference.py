"""Drawing-neutral view frames and cutting-plane relation proposals.

The module does not assign structural object classes or rely on page layout
templates.  It turns existing view, dimension, object-scope, text, and native
line observations into an auditable coordinate-frame layer.  Every field is
resolved independently; a known metric scale never implies a known projection
direction or object-space axis mapping.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

from src.drawing_engine.core.shared_coordinate_system import solve_shared_coordinate_system
from src.drawing_engine.core.contour_correspondence import solve_contour_correspondence
from src.drawing_engine.disciplines.concrete.signed_orientation_certificate import certify_signed_orientation


SCHEMA_VERSION = "0.1.0"
SECTION_LABEL_RE = re.compile(
    r"(?<![0-9A-ZА-Я])(?P<axis>[1-9]\d*|[A-ZА-Я])\s*[-–—]\s*(?P=axis)(?![0-9A-ZА-Я])",
    re.IGNORECASE,
)


def _get(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _bbox(item: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    value = item.get("bbox_display") or item.get("text_bbox")
    if value is None or len(value) != 4:
        return None
    return tuple(float(number) for number in value)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _diagonal(box: tuple[float, float, float, float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _contains(box: tuple[float, float, float, float], point: tuple[float, float], padding: float = 0.0) -> bool:
    return box[0] - padding <= point[0] <= box[2] + padding and box[1] - padding <= point[1] <= box[3] + padding


def _intersection_fraction(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height / max(min(_area(left), _area(right)), 1e-9)


def _intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    padding: float = 0.1,
) -> bool:
    """Intersect boxes while retaining zero-width native line bboxes."""

    return not (
        left[2] < right[0] - padding
        or left[0] > right[2] + padding
        or left[3] < right[1] - padding
        or left[1] > right[3] + padding
    )


def _box_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def _normalise_section_label(text: Any) -> str | None:
    match = SECTION_LABEL_RE.search(str(text or "").upper())
    if not match:
        return None
    value = match.group("axis").upper()
    return f"{value}-{value}"


def _transform_box(
    box: tuple[float, float, float, float],
    matrix: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if matrix is None:
        return box
    a, b, c, d, e, f = matrix
    points = [
        (a * x + c * y + e, b * x + d * y + f)
        for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]))
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _normalise_dimension(item: Any) -> dict[str, Any] | None:
    dimension_id = _get(item, "attachment_id") or _get(item, "id")
    scale = _get(item, "scale_points_per_mm")
    if scale is None:
        style = _get(item, "style", {}) or {}
        scale = style.get("scale_points_per_mm")
    if not dimension_id or scale is None or float(scale) <= 0:
        return None
    status = _get(item, "status", "accepted")
    measured_points = _get(item, "measured_points") or ()
    text_bbox = _get(item, "text_bbox") or _get(item, "bbox_display")
    score = _get(item, "score")
    if score is None:
        score = (_get(item, "style", {}) or {}).get("score", 0.8)
    return {
        "id": str(dimension_id),
        "status": str(status),
        "value_mm": _get(item, "value_mm"),
        "scale_points_per_mm": float(scale),
        "orientation": _get(item, "orientation"),
        "measured_points": [list(map(float, point)) for point in measured_points],
        "text_bbox": None if text_bbox is None else list(map(float, text_bbox)),
        "score": float(score),
        "primitive_refs": list(_get(item, "primitive_refs", ()) or ()),
    }


def _dimension_index(
    dimensions: Iterable[Any],
    observation_graph: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in dimensions:
        record = _normalise_dimension(item)
        if record is not None:
            records[record["id"]] = record
    for node in (observation_graph or {}).get("nodes", []):
        if node.get("kind") != "dimension" or node.get("id") in records:
            continue
        record = _normalise_dimension(node)
        if record is not None:
            records[record["id"]] = record
    return records


def _claim_refs(engineering: Mapping[str, Any]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)
    for claim in engineering.get("claims", []):
        if claim.get("kind") == "metric_dimension" and claim.get("subject"):
            refs[str(claim["subject"])].append(str(claim.get("evidence_ref") or claim.get("id")))
    return refs


def _cluster_scalars(records: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for record in sorted(records, key=lambda item: item["scale_points_per_mm"]):
        matching = next(
            (
                cluster
                for cluster in clusters
                if abs(record["scale_points_per_mm"] - sum(item["scale_points_per_mm"] for item in cluster) / len(cluster))
                / max(record["scale_points_per_mm"], 1e-9)
                <= tolerance
            ),
            None,
        )
        (matching if matching is not None else clusters.append([]) or clusters[-1]).append(record)
    return clusters


def _scale_field(records: list[dict[str, Any]], claim_refs: Mapping[str, list[str]]) -> dict[str, Any]:
    accepted = [item for item in records if item["status"] == "accepted"]
    if not accepted:
        return {
            "state": "unresolved",
            "value_points_per_mm": None,
            "candidates": [],
            "confidence": 0.0,
            "evidence_refs": [],
            "reason": "no accepted dimension geometry with a local scale was supplied for this view",
        }
    clusters = _cluster_scalars(accepted, tolerance=0.03)
    candidates = []
    for cluster in clusters:
        weight = sum(max(item["score"], 0.01) for item in cluster)
        value = sum(item["scale_points_per_mm"] * max(item["score"], 0.01) for item in cluster) / weight
        evidence = sorted(
            {
                ref
                for item in cluster
                for ref in (item["id"], *item["primitive_refs"], *claim_refs.get(item["id"], []))
            }
        )
        candidates.append(
            {
                "value_points_per_mm": round(value, 9),
                "support": len(cluster),
                "confidence": round(sum(item["score"] for item in cluster) / len(cluster), 3),
                "dimension_refs": [item["id"] for item in cluster],
                "evidence_refs": evidence,
            }
        )
    candidates.sort(key=lambda item: (-item["support"], -item["confidence"], item["value_points_per_mm"]))
    winner = candidates[0]
    unique = len(candidates) == 1 or winner["support"] > candidates[1]["support"]
    if unique:
        return {
            "state": "resolved",
            **winner,
            "candidates": candidates[1:],
            "reason": "unique local metric-scale consensus",
        }
    return {
        "state": "unresolved",
        "value_points_per_mm": None,
        "candidates": candidates,
        "confidence": 0.0,
        "evidence_refs": sorted({ref for item in candidates for ref in item["evidence_refs"]}),
        "reason": "two or more incompatible local scale clusters have equal support",
    }


def _cluster_points(points: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for item in points:
        matching = next(
            (
                cluster
                for cluster in clusters
                if math.dist(
                    item["point"],
                    (
                        sum(member["point"][0] for member in cluster) / len(cluster),
                        sum(member["point"][1] for member in cluster) / len(cluster),
                    ),
                )
                <= tolerance
            ),
            None,
        )
        (matching if matching is not None else clusters.append([]) or clusters[-1]).append(item)
    return clusters


def _origin_field(
    records: list[dict[str, Any]],
    view_box: tuple[float, float, float, float],
    ownership_records: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    horizontal = [item for item in records if item["status"] == "accepted" and item["orientation"] == "horizontal"]
    vertical = [item for item in records if item["status"] == "accepted" and item["orientation"] == "vertical"]
    tolerance = max(1.5, 0.003 * _diagonal(view_box))
    intersections = []
    ownership_records = list(ownership_records or [])
    ownership_by_dimension = {
        str(item.get("dimension_ref")): item
        for item in ownership_records
        if item.get("status") == "accepted"
    }
    for left in horizontal:
        left_owner = ownership_by_dimension.get(left["id"])
        if left_owner is None:
            continue
        left_endpoints = {
            str(row.get("selected_geometry_anchor_ref")): row
            for row in left_owner.get("measured_endpoints", [])
            if str(row.get("selected_geometry_anchor_ref") or "").startswith("geometry_vertex.")
        }
        for right in vertical:
            right_owner = ownership_by_dimension.get(right["id"])
            if right_owner is None:
                continue
            for row in right_owner.get("measured_endpoints", []):
                anchor_ref = row.get("selected_geometry_anchor_ref")
                if not str(anchor_ref or "").startswith("geometry_vertex.") or str(anchor_ref) not in left_endpoints:
                    continue
                other = left_endpoints[str(anchor_ref)]
                intersections.append(
                    {
                        "point": (
                            (float(row["point_display"][0]) + float(other["point_display"][0])) / 2.0,
                            (float(row["point_display"][1]) + float(other["point_display"][1])) / 2.0,
                        ),
                        "dimension_refs": sorted({left["id"], right["id"]}),
                        "geometry_anchor_ref": str(anchor_ref),
                        "basis": "orthogonal owned dimensions share one page-global geometry vertex",
                    }
                )
    exact_anchor_count = len(intersections)
    for left in horizontal:
        for right in vertical:
            for a in left["measured_points"]:
                for b in right["measured_points"]:
                    if math.dist(a, b) <= tolerance:
                        intersections.append(
                            {
                                "point": ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0),
                                "dimension_refs": sorted({left["id"], right["id"]}),
                                "geometry_anchor_ref": None,
                                "basis": "orthogonal dimension axes share one display point",
                            }
                        )
    candidates = []
    for cluster in _cluster_points(intersections, tolerance):
        point = (
            sum(item["point"][0] for item in cluster) / len(cluster),
            sum(item["point"][1] for item in cluster) / len(cluster),
        )
        candidates.append(
            {
                "point_display": [round(point[0], 6), round(point[1], 6)],
                "support": len({ref for item in cluster for ref in item["dimension_refs"]}),
                "dimension_refs": sorted({ref for item in cluster for ref in item["dimension_refs"]}),
                "geometry_anchor_refs": sorted(
                    {
                        str(item["geometry_anchor_ref"])
                        for item in cluster
                        if item.get("geometry_anchor_ref") is not None
                    }
                ),
                "basis": (
                    "owned page-global geometry vertex"
                    if any(item.get("geometry_anchor_ref") for item in cluster)
                    else "display-space dimension-axis intersection"
                ),
            }
        )
    candidates.sort(key=lambda item: (-item["support"], item["point_display"]))
    # A more popular corner is still not a unique datum.  Resolve only when
    # the orthogonal dimension axes identify exactly one display-space point.
    exact_candidates = [item for item in candidates if item["geometry_anchor_refs"]]
    unique = len(exact_candidates) == 1 or (not exact_candidates and len(candidates) == 1)
    if candidates and unique:
        winner = exact_candidates[0] if exact_candidates else candidates[0]
        return {
            "state": "resolved",
            "object_origin_display": winner["point_display"],
            "display_anchor": winner["point_display"],
            "confidence": 0.86,
            "evidence_refs": winner["dimension_refs"],
            "candidates": candidates[1:],
            "geometry_anchor_refs": winner["geometry_anchor_refs"],
            "reason": (
                "unique owned geometry vertex shared by orthogonal dimensions"
                if winner["geometry_anchor_refs"]
                else "unique shared endpoint of orthogonal accepted dimension axes"
            ),
        }
    return {
        "state": "unresolved",
        "object_origin_display": None,
        "display_anchor": [view_box[0], view_box[1]],
        "confidence": 0.0,
        "evidence_refs": sorted({ref for item in candidates for ref in item["dimension_refs"]}),
        "candidates": candidates,
        "reason": (
            "multiple dimension-axis datum intersections remain geometrically possible"
            if candidates
            else "no shared endpoint of orthogonal accepted dimensions identifies an object datum"
        ),
        "display_anchor_basis": "view bounding-box anchor for local calculations only; not an object datum",
        "owned_anchor_observation_count": exact_anchor_count,
    }


def _object_memberships(engineering: Mapping[str, Any]) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = defaultdict(set)
    assembly = engineering.get("object_instance_graph", {})
    for view_id, object_ids in assembly.get("view_membership", {}).items():
        memberships[str(view_id)].update(map(str, object_ids))
    for instance in assembly.get("instances", []):
        for view_id in instance.get("view_ids", []):
            memberships[str(view_id)].add(str(instance["id"]))
    for relation in engineering.get("relations", []):
        if relation.get("type") == "projection_of_object_instance" and relation.get("from") and relation.get("to"):
            memberships[str(relation.get("from"))].add(str(relation.get("to")))
    return memberships


def _title_segment_views(
    engineering: Mapping[str, Any],
    views: list[Mapping[str, Any]],
    dimensions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Materialize only genuinely split title scopes as downstream views.

    Unsplit source views retain their established IDs.  When one connected
    native component was deterministically partitioned by Step 1B, each
    resolved segment receives its stable segment ID so downstream relations
    cannot collapse the distinct projections back into the source component.
    """

    segments = list(
        engineering.get("title_anchored_view_segmentation", {}).get("segments", [])
        or []
    )
    resolved_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for segment in segments:
        if segment.get("state") == "resolved" and segment.get("view_id") is not None:
            resolved_by_source[str(segment["view_id"])].append(segment)

    effective = []
    segment_by_view_id: dict[str, Mapping[str, Any]] = {}
    for source in views:
        source_id = str(source["id"])
        resolved = resolved_by_source.get(source_id, [])
        if len(resolved) <= 1:
            record = dict(source)
            if resolved:
                segment = resolved[0]
                record.update(
                    {
                        "title_segment_id": str(segment["id"]),
                        "title": segment.get("title"),
                        "normalised_title": segment.get("normalised_title"),
                        "scale_ratio": segment.get("scale_ratio"),
                    }
                )
                segment_by_view_id[source_id] = segment
            effective.append(record)
            continue

        for segment in sorted(resolved, key=lambda item: str(item["id"])):
            box = _bbox(segment)
            if box is None:
                continue
            dimension_refs = []
            for ref in source.get("dimension_refs", []) or []:
                dimension = dimensions.get(str(ref))
                if dimension is None:
                    continue
                text_box = dimension.get("text_bbox")
                if text_box and _contains(box, _center(tuple(text_box)), padding=1.0):
                    dimension_refs.append(str(ref))
            role = str(segment.get("role") or source.get("role_hypothesis") or "drawing_view_candidate")
            if role == "section_label":
                role = "section_view_candidate"
            record = {
                **dict(source),
                "id": str(segment["id"]),
                "source_view_id": source_id,
                "title_segment_id": str(segment["id"]),
                "role_hypothesis": role,
                "bbox_display": list(box),
                "primitive_refs": list(segment.get("primitive_refs", [])),
                "dimension_refs": sorted(dimension_refs),
                "title": segment.get("title"),
                "normalised_title": segment.get("normalised_title"),
                "scale_ratio": segment.get("scale_ratio"),
                "supporting_projection_only": role in {
                    "detail_view_candidate",
                    "reinforcement_view_candidate",
                },
                "features": dict(source.get("features", {}) or {}),
            }
            record.setdefault("features", {})["accepted_dimensions"] = sum(
                dimensions[ref].get("status") == "accepted"
                for ref in record["dimension_refs"]
                if ref in dimensions
            )
            effective.append(record)
            segment_by_view_id[str(segment["id"])] = segment
    return effective, segment_by_view_id


def _owner_field(view_id: str, memberships: Mapping[str, set[str]]) -> dict[str, Any]:
    candidates = sorted(memberships.get(view_id, set()))
    if len(candidates) == 1:
        return {
            "state": "resolved",
            "object_instance_id": candidates[0],
            "object_scope_id": candidates[0],
            "ownership_kind": "explicit_object_instance_membership",
            "epistemic_state": "inferred",
            "physical_object_identity_state": "resolved",
            "quantity_aggregation_eligible": True,
            "candidates": [],
            "confidence": 0.95,
            "evidence_refs": [view_id, candidates[0]],
            "reason": "unique object-instance membership already supported by the engineering graph",
        }
    return {
        "state": "unresolved",
        "object_instance_id": None,
        "object_scope_id": None,
        "ownership_kind": None,
        "epistemic_state": "unknown",
        "physical_object_identity_state": "unresolved",
        "quantity_aggregation_eligible": False,
        "candidates": candidates,
        "confidence": 0.0,
        "evidence_refs": [view_id, *candidates],
        "reason": "view has no object-instance membership" if not candidates else "view belongs to multiple object-instance hypotheses",
    }


def _cut_components(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    relations_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        section_id = str(relation["section_view_id"])
        parent_id = str(relation["parent_view_id"])
        adjacency[section_id].add(parent_id)
        adjacency[parent_id].add(section_id)
        relations_by_view[section_id].append(relation)
        relations_by_view[parent_id].append(relation)
    components = []
    visited: set[str] = set()
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        stack = [seed]
        view_ids = set()
        while stack:
            view_id = stack.pop()
            if view_id in visited:
                continue
            visited.add(view_id)
            view_ids.add(view_id)
            stack.extend(sorted(adjacency[view_id] - visited, reverse=True))
        component_relations = {
            relation["id"]: relation
            for view_id in view_ids
            for relation in relations_by_view[view_id]
        }
        components.append(
            {
                "view_ids": sorted(view_ids),
                "relations": [component_relations[key] for key in sorted(component_relations)],
            }
        )
    return components


def _resolve_object_ownership(
    engineering: Mapping[str, Any],
    memberships: Mapping[str, set[str]],
    accepted_cuts: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Close view ownership from explicit instances, cuts, or metric equations.

    A containment scope is not promoted to an object-instance assertion.  It
    receives its own identifier and resolves ownership only when at least two
    independently accepted section views connect one coherent component.
    """

    views = [str(item["id"]) for item in engineering.get("view_hypotheses", [])]
    ownership = {view_id: _owner_field(view_id, memberships) for view_id in views}
    scopes = []
    candidates = []
    derived_serial = 1
    for component in _cut_components(accepted_cuts):
        component_views = component["view_ids"]
        relations = component["relations"]
        explicit_ids = sorted(
            {
                object_id
                for view_id in component_views
                for object_id in memberships.get(view_id, set())
            }
        )
        section_ids = sorted({str(item["section_view_id"]) for item in relations})
        parent_ids = sorted({str(item["parent_view_id"]) for item in relations})
        evidence_refs = sorted(
            {
                ref
                for relation in relations
                for ref in (relation["id"], *relation.get("evidence_refs", []))
            }
        )
        if len(explicit_ids) == 1:
            scope_id = explicit_ids[0]
            scope = {
                "id": scope_id,
                "state": "resolved",
                "epistemic_state": "inferred",
                "ownership_kind": "unique_explicit_instance_propagated_through_cut_relations",
                "object_instance_id": scope_id,
                "physical_object_identity_state": "resolved",
                "quantity_aggregation_eligible": True,
                "view_ids": component_views,
                "section_view_ids": section_ids,
                "parent_view_ids": parent_ids,
                "relation_refs": [item["id"] for item in relations],
                "confidence": round(min(item["confidence"] for item in relations), 3),
                "evidence_refs": evidence_refs,
            }
        elif not explicit_ids and len(section_ids) >= 2 and len(relations) >= 2:
            scope_id = f"view_containment_scope.{derived_serial:03d}"
            derived_serial += 1
            scope = {
                "id": scope_id,
                "state": "resolved",
                "epistemic_state": "inferred",
                "ownership_kind": "unique_redundant_view_containment_component",
                "object_instance_id": None,
                "physical_object_identity_state": "unresolved",
                "quantity_aggregation_eligible": False,
                "view_ids": component_views,
                "section_view_ids": section_ids,
                "parent_view_ids": parent_ids,
                "relation_refs": [item["id"] for item in relations],
                "confidence": round(0.9 * min(item["confidence"] for item in relations), 3),
                "evidence_refs": evidence_refs,
            }
        else:
            candidates.append(
                {
                    "id": f"object_scope_candidate.{len(candidates) + 1:03d}",
                    "state": "unresolved",
                    "view_ids": component_views,
                    "object_instance_candidates": explicit_ids,
                    "section_view_ids": section_ids,
                    "parent_view_ids": parent_ids,
                    "relation_refs": [item["id"] for item in relations],
                    "evidence_refs": evidence_refs,
                    "reason": (
                        "accepted cut component contains conflicting explicit object memberships"
                        if len(explicit_ids) > 1
                        else "one section-to-parent relation is insufficient to create a new physical-object scope"
                    ),
                }
            )
            continue
        scopes.append(scope)
        for view_id in component_views:
            ownership[view_id] = {
                "state": "resolved",
                "object_instance_id": scope["object_instance_id"],
                "object_scope_id": scope_id,
                "ownership_kind": scope["ownership_kind"],
                "epistemic_state": "inferred",
                "physical_object_identity_state": scope["physical_object_identity_state"],
                "quantity_aggregation_eligible": scope["quantity_aggregation_eligible"],
                "candidates": [],
                "confidence": scope["confidence"],
                "evidence_refs": [scope_id, view_id, *evidence_refs],
                "reason": (
                    "unique explicit membership propagated through accepted view-containment relations"
                    if scope["object_instance_id"] is not None
                    else "unique redundant component of accepted section-host and parent-trace containment relations"
                ),
            }
    for metric_scope in engineering.get("metric_equation_graph", {}).get("scopes", []) or []:
        if metric_scope.get("status") != "accepted":
            continue
        component_views = sorted(
            view_id for view_id in map(str, metric_scope.get("view_ids", [])) if view_id in ownership
        )
        resolved_scope_ids = sorted(
            {
                str(ownership[view_id]["object_scope_id"])
                for view_id in component_views
                if ownership[view_id].get("state") == "resolved"
                and ownership[view_id].get("object_scope_id") is not None
            }
        )
        evidence_refs = sorted(
            {
                str(metric_scope["id"]),
                *map(str, metric_scope.get("evidence_refs", [])),
            }
        )
        if len(component_views) < 3 or resolved_scope_ids:
            candidates.append(
                {
                    "id": f"object_scope_candidate.{len(candidates) + 1:03d}",
                    "state": "unresolved",
                    "view_ids": component_views,
                    "object_instance_candidates": [],
                    "section_view_ids": [],
                    "parent_view_ids": [],
                    "relation_refs": list(metric_scope.get("relation_refs", [])),
                    "evidence_refs": evidence_refs,
                    "reason": (
                        "metric-equation component overlaps an independently resolved object scope"
                        if resolved_scope_ids
                        else "metric-equation component contains fewer than three independently bound views"
                    ),
                }
            )
            continue
        scope_id = str(metric_scope["id"])
        scope = {
            "id": scope_id,
            "state": "resolved",
            "epistemic_state": "inferred",
            "ownership_kind": "unique_redundant_metric_equation_component",
            "object_instance_id": None,
            "physical_object_identity_state": "unresolved",
            "quantity_aggregation_eligible": False,
            "view_ids": component_views,
            "section_view_ids": [],
            "parent_view_ids": [],
            "relation_refs": list(metric_scope.get("relation_refs", [])),
            "confidence": 0.84,
            "evidence_refs": evidence_refs,
        }
        scopes.append(scope)
        for view_id in component_views:
            ownership[view_id] = {
                "state": "resolved",
                "object_instance_id": None,
                "object_scope_id": scope_id,
                "ownership_kind": scope["ownership_kind"],
                "epistemic_state": "inferred",
                "physical_object_identity_state": "unresolved",
                "quantity_aggregation_eligible": False,
                "candidates": [],
                "confidence": scope["confidence"],
                "evidence_refs": [scope_id, view_id, *evidence_refs],
                "reason": "complete arithmetic dimension chains connect the view through two independent shared metric extents",
            }
    return ownership, scopes, candidates


def _section_occurrences(
    text_roles: Iterable[Mapping[str, Any]],
    text_to_display_transform: tuple[float, float, float, float, float, float] | None,
) -> list[dict[str, Any]]:
    occurrences = []
    for item in text_roles:
        role = str(item.get("resolved_role", ""))
        text = str(item.get("text") or "").strip().upper()
        label = _normalise_section_label(text)
        occurrence_kind = "full_section_label"
        if label is None:
            if role not in {
                "identifier_candidate",
                "unclassified_number",
                "unclassified_text",
                "cutting_plane_endpoint_candidate",
            }:
                continue
            token_match = re.fullmatch(r"[1-9]\d*|[A-ZА-Я]", text)
            if token_match is None:
                continue
            token = token_match.group(0)
            label = f"{token}-{token}"
            occurrence_kind = "single_endpoint_token_candidate"
        elif role != "section_label":
            continue
        box = _bbox(item)
        if label is None or box is None:
            continue
        box = _transform_box(box, text_to_display_transform)
        occurrences.append(
            {
                "id": str(item.get("id") or f"section_label.{len(occurrences) + 1:03d}"),
                "label": label,
                "occurrence_kind": occurrence_kind,
                "bbox_display": list(box),
                "confidence": float(item.get("confidence", 0.8)),
                "primitive_refs": list(item.get("primitive_refs", [])),
            }
        )
    return occurrences


def _section_bindings(
    engineering: Mapping[str, Any],
    views: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bindings = []
    for section in engineering.get("section_rebar_observations", {}).get("sections", []):
        label = _normalise_section_label(section.get("label"))
        host = _bbox({"bbox_display": section.get("host_bbox_display")})
        if label is None or host is None:
            continue
        candidates = []
        for view in views:
            box = _bbox(view)
            if box is None:
                continue
            overlap = _intersection_fraction(host, box)
            if overlap < 0.35:
                continue
            role_bonus = 0.1 if "section" in str(view.get("role_hypothesis", "")) else 0.0
            candidates.append(
                {
                    "view_id": str(view["id"]),
                    "score": round(min(0.99, 0.75 * overlap + role_bonus), 3),
                    "evidence_refs": [str(section.get("section_id")), str(view["id"])],
                }
            )
        candidates.sort(key=lambda item: (-item["score"], item["view_id"]))
        unique = len(candidates) == 1 or (
            len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] >= 0.15
        )
        bindings.append({
            "label": label,
            "section_id": str(section.get("section_id")),
            "view_id": candidates[0]["view_id"] if candidates and unique else None,
            "state": "resolved" if candidates and unique else "unresolved",
            "candidates": candidates,
            "reason": (
                "unique section host containment"
                if candidates and unique
                else "section host overlaps multiple view hypotheses without a unique winner"
                if candidates
                else "section host is not contained by a view hypothesis"
            ),
        })

    existing = {
        (item["label"], item.get("view_id"))
        for item in bindings
        if item.get("view_id") is not None
    }
    view_by_segment = {
        str(view.get("title_segment_id")): view
        for view in views
        if view.get("title_segment_id") is not None
    }
    view_by_source = {str(view["id"]): view for view in views}
    for segment in engineering.get("title_anchored_view_segmentation", {}).get("segments", []) or []:
        if segment.get("state") != "resolved":
            continue
        label = _normalise_section_label(
            segment.get("normalised_title") or segment.get("title")
        )
        if label is None:
            continue
        view = view_by_segment.get(str(segment.get("id"))) or view_by_source.get(
            str(segment.get("view_id"))
        )
        if view is None or (label, str(view["id"])) in existing:
            continue
        bindings.append(
            {
                "label": label,
                "section_id": str(segment["id"]),
                "view_id": str(view["id"]),
                "state": "resolved",
                "candidates": [
                    {
                        "view_id": str(view["id"]),
                        "score": float(segment.get("confidence", 0.9)),
                        "evidence_refs": [str(segment["id"]), str(view["id"])],
                    }
                ],
                "reason": "resolved title-anchored section scope",
            }
        )
        existing.add((label, str(view["id"])))
    bindings.sort(key=lambda item: (item["label"], item["section_id"]))
    return bindings


def _title_scale_points_per_mm(segment: Mapping[str, Any] | None) -> float | None:
    match = re.fullmatch(r"\s*1\s*:\s*(\d+(?:\.\d+)?)\s*", str((segment or {}).get("scale_ratio") or ""))
    if match is None or float(match.group(1)) <= 0:
        return None
    return 72.0 / (25.4 * float(match.group(1)))


def _relation_metric_pair_certificate(
    engineering: Mapping[str, Any],
    *,
    views: list[Mapping[str, Any]],
    section_view_id: str,
    parent_view_id: str,
    section_segment: Mapping[str, Any] | None,
    parent_segment: Mapping[str, Any] | None,
    trace_orientation: str | None,
) -> dict[str, Any]:
    """Accept a metric pair only inside an independently evidenced cut relation.

    This does not rewrite Step 1A adjudication. A terminal-less dimension that
    remains globally ambiguous may support this relation only when its two
    native endpoints are uniquely owned and its locally derived vector scale
    is valid. A printed title scale, when present, must agree; its absence does
    not erase native metric evidence. The independently observed span and
    local scale in the other view must also agree. These spans remain
    quantity-ineligible until physical reconstruction closes.
    """

    expected_parent = None
    expected_section = "horizontal" if trace_orientation in {"horizontal", "vertical"} else None
    view_by_id = {str(view["id"]): view for view in views}
    adjudication_by_dimension = {
        str(item.get("dimension_ref")): item
        for item in engineering.get("dimension_adjudication", {}).get("records", []) or []
        if item.get("dimension_ref") is not None
    }

    def candidates(
        view_id: str,
        segment: Mapping[str, Any] | None,
        orientation: str | None,
    ) -> list[dict[str, Any]]:
        view = view_by_id.get(view_id, {})
        box = _bbox(view)
        title_scale = _title_scale_points_per_mm(segment)
        rows = []
        for ownership in engineering.get("dimension_ownership", {}).get("attachments", []) or []:
            dimension_ref = str(ownership.get("dimension_ref"))
            if dimension_ref not in set(map(str, view.get("dimension_refs", []) or [])):
                continue
            if orientation is not None and ownership.get("orientation") != orientation:
                continue
            endpoints = list(ownership.get("measured_endpoints", []) or [])
            if len(endpoints) != 2 or any(item.get("state") != "resolved" for item in endpoints):
                continue
            if box is not None and any(
                not _contains(box, tuple(map(float, item.get("point_display", ()))), padding=1.5)
                for item in endpoints
                if len(item.get("point_display", ())) == 2
            ):
                continue
            adjudication = adjudication_by_dimension.get(dimension_ref, {})
            certificate = adjudication.get("certificate", {}) or {}
            local_scale = certificate.get("view_local_scale", {}) or {}
            scale = local_scale.get("scale_points_per_mm")
            globally_accepted = adjudication.get("status") == "accepted"
            relation_scoped = bool(
                scale
                and certificate.get("unique_preliminary_view_owner") is True
                and certificate.get("exact_endpoint_geometry_resolved") is True
                and int(local_scale.get("support_count", 0)) >= 1
                and (
                    title_scale is None
                    or abs(float(scale) / title_scale - 1.0) <= 0.03
                )
            )
            if not globally_accepted and not relation_scoped:
                continue
            value = ownership.get("value_mm")
            if value is None or float(value) <= 0:
                continue
            rows.append(
                {
                    "id": f"relation_metric_span.{dimension_ref}.{view_id}",
                    "dimension_ref": dimension_ref,
                    "view_id": view_id,
                    "status": "accepted",
                    "epistemic_state": "derived" if relation_scoped and not globally_accepted else "direct",
                    "acceptance_scope": "cutting_plane_relation" if relation_scoped and not globally_accepted else "global_dimension_adjudication",
                    "global_dimension_adjudication_status": adjudication.get("status"),
                    "quantity_aggregation_eligible": False,
                    "score": min(
                        0.94,
                        float(adjudication.get("certificate", {}).get("adjudicated_score", 0.9)),
                    ),
                    "value_mm": float(value),
                    "orientation": ownership.get("orientation"),
                    "scale_points_per_mm": float(scale) if scale else title_scale,
                    "measured_points": [list(item["point_display"]) for item in endpoints],
                    "primitive_refs": list(ownership.get("primitive_refs", [])),
                    "geometry_anchor_refs": list(ownership.get("geometry_anchor_refs", [])),
                    "semantic_role": "overall_shared_axis_span",
                    "evidence_refs": sorted(
                        {
                            dimension_ref,
                            str(ownership.get("id")),
                            str(adjudication.get("id")),
                            str((segment or {}).get("id")),
                            *map(str, ownership.get("primitive_refs", [])),
                        }
                        - {"None"}
                    ),
                }
            )
        return rows

    parent_rows = candidates(parent_view_id, parent_segment, expected_parent)
    section_rows = candidates(section_view_id, section_segment, expected_section)

    def scale_clusters(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(rows, key=lambda item: (item["scale_points_per_mm"], item["dimension_ref"])):
            cluster = next(
                (
                    item
                    for item in clusters
                    if abs(
                        float(row["scale_points_per_mm"])
                        / (sum(float(member["scale_points_per_mm"]) for member in item) / len(item))
                        - 1.0
                    ) <= 0.03
                ),
                None,
            )
            (cluster if cluster is not None else clusters.append([]) or clusters[-1]).append(row)
        return clusters

    parent_clusters = scale_clusters(parent_rows)
    section_clusters = scale_clusters(section_rows)
    selected_parent_cluster = None
    selected_section_cluster = section_clusters[0] if len(section_clusters) == 1 else None
    if selected_section_cluster is not None:
        section_scale = sum(float(item["scale_points_per_mm"]) for item in selected_section_cluster) / len(selected_section_cluster)
        compatible = [
            cluster
            for cluster in parent_clusters
            if abs(
                (sum(float(item["scale_points_per_mm"]) for item in cluster) / len(cluster))
                / section_scale
                - 1.0
            ) <= 0.03
        ]
        if len(compatible) == 1:
            selected_parent_cluster = compatible[0]
        elif not compatible and len(parent_clusters) == 1:
            selected_parent_cluster = parent_clusters[0]
    unique = selected_parent_cluster is not None and selected_section_cluster is not None
    if not unique:
        return {
            "status": "unresolved",
            "parent_view_id": parent_view_id,
            "section_view_id": section_view_id,
            "spans": [],
            "candidate_pair_count": len(parent_clusters) * len(section_clusters),
            "evidence_refs": sorted(
                {ref for item in [*parent_rows, *section_rows] for ref in item["evidence_refs"]}
            ),
            "reason": "no unique independently owned metric span agrees across both views",
        }
    parent = max(selected_parent_cluster, key=lambda item: (item["global_dimension_adjudication_status"] == "accepted", item["score"], item["dimension_ref"]))
    section = max(selected_section_cluster, key=lambda item: (item["global_dimension_adjudication_status"] == "accepted", item["score"], item["dimension_ref"]))
    parent_scale = sum(float(item["scale_points_per_mm"]) for item in selected_parent_cluster) / len(selected_parent_cluster)
    section_scale = sum(float(item["scale_points_per_mm"]) for item in selected_section_cluster) / len(selected_section_cluster)
    scale_ratio = parent_scale / section_scale
    parent = {
        **parent,
        "scale_points_per_mm": parent_scale,
        "scale_support_count": len(selected_parent_cluster),
        "supporting_span_ids": [item["id"] for item in selected_parent_cluster],
        "evidence_refs": sorted({ref for item in selected_parent_cluster for ref in item["evidence_refs"]}),
    }
    section = {
        **section,
        "scale_points_per_mm": section_scale,
        "scale_support_count": len(selected_section_cluster),
        "supporting_span_ids": [item["id"] for item in selected_section_cluster],
        "evidence_refs": sorted({ref for item in selected_section_cluster for ref in item["evidence_refs"]}),
    }
    return {
        "status": "passed",
        "parent_view_id": parent_view_id,
        "section_view_id": section_view_id,
        "spans": [parent, section],
        "candidate_pair_count": len(parent_clusters) * len(section_clusters),
        "transverse_extent_mm": round(float(section["value_mm"]), 6),
        "metric_scale_ratio": round(scale_ratio, 6),
        "metric_scale_compatibility": {
            "status": "passed",
            "parent_scale_points_per_mm": parent["scale_points_per_mm"],
            "section_scale_points_per_mm": section["scale_points_per_mm"],
            "same_plotted_scale": 0.97 <= scale_ratio <= 1.03,
            "reason": "both views have independently resolved native metric scales; equal plotted scale is not required",
        },
        "evidence_refs": sorted({*parent["evidence_refs"], *section["evidence_refs"]}),
        "reason": "unique independently scaled parent span and transverse section extent support the accepted cut",
    }


def _section_binding_prerequisite_certificate(
    engineering: Mapping[str, Any],
    *,
    label: str,
    section_view_id: str,
    parent_view_id: str,
    views: list[Mapping[str, Any]] | None = None,
    trace_orientation: str | None = None,
) -> dict[str, Any]:
    """Close dimension adjudication and title segmentation on one view pair."""

    views = list(views or engineering.get("view_hypotheses", []))
    view_by_id = {str(view["id"]): view for view in views}
    segments = list(
        engineering.get("title_anchored_view_segmentation", {}).get("segments", [])
        or []
    )

    def segment_for(view_id: str) -> Mapping[str, Any] | None:
        view = view_by_id.get(view_id, {})
        segment_id = view.get("title_segment_id")
        if segment_id is not None:
            return next(
                (item for item in segments if str(item.get("id")) == str(segment_id)),
                None,
            )
        matches = [
            item
            for item in segments
            if str(item.get("view_id")) == view_id and item.get("state") == "resolved"
        ]
        return matches[0] if len(matches) == 1 else None

    section_segment = segment_for(section_view_id)
    parent_segment = segment_for(parent_view_id)
    section_title_matches = bool(
        section_segment
        and section_segment.get("state") == "resolved"
        and _normalise_section_label(section_segment.get("normalised_title") or section_segment.get("title")) == label
    )
    parent_title_resolved = bool(parent_segment and parent_segment.get("state") == "resolved")

    accepted_by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in engineering.get("dimension_adjudication", {}).get("records", []) or []:
        if record.get("status") != "accepted":
            continue
        dimension_ref = str(record.get("dimension_ref"))
        for view in views:
            if dimension_ref in set(map(str, view.get("dimension_refs", []) or [])):
                accepted_by_view[str(view["id"])].append(record)
                continue
            if view.get("source_view_id") is None and str(view["id"]) in set(
                map(str, record.get("view_refs", []) or [])
            ):
                accepted_by_view[str(view["id"])].append(record)
    section_dimensions = accepted_by_view.get(section_view_id, [])
    parent_dimensions = accepted_by_view.get(parent_view_id, [])
    metric_pair = _relation_metric_pair_certificate(
        engineering,
        views=views,
        section_view_id=section_view_id,
        parent_view_id=parent_view_id,
        section_segment=section_segment,
        parent_segment=parent_segment,
        trace_orientation=trace_orientation,
    )
    title_closed = section_title_matches and parent_title_resolved
    dimension_closed = bool(section_dimensions and parent_dimensions) or metric_pair["status"] == "passed"
    unresolved = []
    if not section_title_matches:
        unresolved.append("title_segmentation.section_view_title_anchor")
    if not parent_title_resolved:
        unresolved.append("title_segmentation.parent_view_title_anchor")
    if not section_dimensions:
        unresolved.append("dimension_adjudication.section_view_owned_dimension")
    if not parent_dimensions:
        unresolved.append("dimension_adjudication.parent_view_owned_dimension")
    evidence = sorted(
        {
            *(
                [str(section_segment.get("id")), *map(str, section_segment.get("evidence_refs", []))]
                if section_segment
                else []
            ),
            *(
                [str(parent_segment.get("id")), *map(str, parent_segment.get("evidence_refs", []))]
                if parent_segment
                else []
            ),
            *(
                str(ref)
                for record in [*section_dimensions, *parent_dimensions]
                for ref in [record.get("id"), record.get("ownership_ref"), *record.get("evidence_refs", [])]
                if ref
            ),
            *metric_pair.get("evidence_refs", []),
        }
    )
    return {
        "status": "passed" if title_closed and dimension_closed else "unresolved",
        "title_segmentation_closed": title_closed,
        "dimension_adjudication_closed": dimension_closed,
        "section_view_id": section_view_id,
        "parent_view_id": parent_view_id,
        "section_segment_id": section_segment.get("id") if section_segment else None,
        "parent_segment_id": parent_segment.get("id") if parent_segment else None,
        "section_dimension_adjudication_refs": [str(item["id"]) for item in section_dimensions],
        "parent_dimension_adjudication_refs": [str(item["id"]) for item in parent_dimensions],
        "accepted_metric_evidence_in_both_views": dimension_closed,
        "relation_scoped_metric_pair": metric_pair,
        "evidence_refs": evidence,
        "unresolved_fields": unresolved,
    }


def _occurrence_view(
    occurrence: Mapping[str, Any],
    views: list[Mapping[str, Any]],
    excluded: set[str],
) -> list[str]:
    center = _center(tuple(occurrence["bbox_display"]))
    view_by_id = {str(view["id"]): view for view in views}
    contained = []
    nearby = []
    for view in views:
        if str(view["id"]) in excluded:
            continue
        box = _bbox(view)
        if box is not None and _contains(box, center, max(1.0, 0.005 * _diagonal(box))):
            contained.append((str(view["id"]), _area(box)))
        elif box is not None:
            distance = _box_distance(tuple(occurrence["bbox_display"]), box)
            if distance <= max(18.0, 0.02 * _diagonal(box)):
                nearby.append((round(distance, 6), str(view["id"]), _area(box)))
    if contained:
        titled = [
            item
            for item in contained
            if view_by_id.get(item[0], {}).get("title")
        ]
        if titled:
            contained = titled
    if not contained:
        if not nearby:
            return []
        nearby.sort()
        minimum_distance = nearby[0][0]
        nearest = [item for item in nearby if item[0] <= minimum_distance + 1.0]
        minimum_area = min(item[2] for item in nearest)
        return sorted(item[1] for item in nearest if item[2] <= minimum_area * 1.05)
    minimum = min(area for _, area in contained)
    return sorted(view_id for view_id, area in contained if area <= minimum * 1.05)


def _line_groups(
    observation_graph: Mapping[str, Any] | None,
    parent_box: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    lines = []
    for node in (observation_graph or {}).get("nodes", []):
        if node.get("kind") not in {"line_segment", "path"} or node.get("orientation") not in {"horizontal", "vertical"}:
            continue
        style = node.get("style", {}) or {}
        if node.get("kind") == "path" and (style.get("fill") is not None or style.get("close_path")):
            continue
        box = _bbox(node)
        if box is None or not _intersects(box, parent_box):
            continue
        length = max(box[2] - box[0], box[3] - box[1])
        if length < max(8.0, 0.025 * _diagonal(parent_box)):
            continue
        lines.append({"id": str(node["id"]), "orientation": node["orientation"], "bbox": box})
    tolerance = max(1.25, 0.003 * _diagonal(parent_box))
    gap_tolerance = max(10.0, 0.06 * _diagonal(parent_box))
    groups: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda item: (item["orientation"], item["bbox"])):
        coordinate = (
            (line["bbox"][1] + line["bbox"][3]) / 2.0
            if line["orientation"] == "horizontal"
            else (line["bbox"][0] + line["bbox"][2]) / 2.0
        )
        def compatible(group: list[dict[str, Any]]) -> bool:
            if group[0]["orientation"] != line["orientation"]:
                return False
            mean_coordinate = sum(
                (
                    (item["bbox"][1] + item["bbox"][3]) / 2.0
                    if item["orientation"] == "horizontal"
                    else (item["bbox"][0] + item["bbox"][2]) / 2.0
                )
                for item in group
            ) / len(group)
            if abs(coordinate - mean_coordinate) > tolerance:
                return False
            if line["orientation"] == "horizontal":
                along_min, along_max = line["bbox"][0], line["bbox"][2]
                group_min = min(item["bbox"][0] for item in group)
                group_max = max(item["bbox"][2] for item in group)
            else:
                along_min, along_max = line["bbox"][1], line["bbox"][3]
                group_min = min(item["bbox"][1] for item in group)
                group_max = max(item["bbox"][3] for item in group)
            gap = max(group_min - along_max, along_min - group_max, 0.0)
            return gap <= gap_tolerance

        matching = next(
            (
                group
                for group in groups
                if compatible(group)
            ),
            None,
        )
        (matching if matching is not None else groups.append([]) or groups[-1]).append(line)
    results = []
    for group in groups:
        union = (
            min(item["bbox"][0] for item in group),
            min(item["bbox"][1] for item in group),
            max(item["bbox"][2] for item in group),
            max(item["bbox"][3] for item in group),
        )
        results.append(
            {
                "orientation": group[0]["orientation"],
                "bbox_display": list(union),
                "primitive_refs": sorted(item["id"] for item in group),
            }
        )
    return results


def _endpoint_chain_trace_candidates(
    observation_graph: Mapping[str, Any] | None,
    parent_box: tuple[float, float, float, float],
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Infer an interrupted cut trace from paired labels and native end stems.

    Some CAD conventions draw only two inward arrow/stem chains and omit the
    line through the object.  The virtual span is emitted only when identical
    single tokens form one aligned pair and each token has a unique nearby
    perpendicular native stem.
    """

    endpoint_tokens = [
        item
        for item in occurrences
        if item["occurrence_kind"] == "single_endpoint_token_candidate"
    ]
    if len(endpoint_tokens) < 2:
        return []
    diagonal = _diagonal(parent_box)
    annotation_padding = max(24.0, 0.03 * diagonal)
    native_axes = []
    for node in (observation_graph or {}).get("nodes", []):
        if node.get("kind") not in {"path", "line_segment"} or node.get("orientation") not in {"horizontal", "vertical"}:
            continue
        box = _bbox(node)
        if box is None or (
            not _intersects(box, parent_box)
            and all(
                _box_distance(box, tuple(token["bbox_display"])) > annotation_padding
                for token in endpoint_tokens
            )
        ):
            continue
        style = node.get("style", {}) or {}
        if style.get("fill") is not None or style.get("close_path"):
            continue
        width = style.get("width")
        if width is not None and float(width) > 1.6:
            continue
        length = max(box[2] - box[0], box[3] - box[1])
        if not 4.0 <= length <= max(100.0, 0.12 * diagonal):
            continue
        native_axes.append(
            {
                "id": str(node["id"]),
                "kind": str(node["kind"]),
                "orientation": str(node["orientation"]),
                "bbox": box,
                "length": length,
            }
        )

    def unique_stem(token: dict[str, Any], orientation: str) -> dict[str, Any] | None:
        token_box = tuple(token["bbox_display"])
        limit = max(18.0, 0.015 * diagonal)
        candidates = [
            (round(_box_distance(token_box, item["bbox"]), 6), item)
            for item in native_axes
            if item["orientation"] == orientation and _box_distance(token_box, item["bbox"]) <= limit
        ]
        path_candidates = [item for item in candidates if item[1]["kind"] == "path"]
        if path_candidates:
            candidates = path_candidates
        candidates.sort(key=lambda item: (item[0], -item[1]["length"], item[1]["id"]))
        if not candidates:
            return None
        best_distance, best = candidates[0]
        if len(candidates) > 1:
            second_distance, second = candidates[1]
            best_cross = _center(best["bbox"])[0 if orientation == "vertical" else 1]
            second_cross = _center(second["bbox"])[0 if orientation == "vertical" else 1]
            same_chain = abs(best_cross - second_cross) <= 1.5
            if not same_chain and second_distance - best_distance < 2.0:
                return None
        return best

    results = []
    for left, right in combinations(endpoint_tokens, 2):
        left_center, right_center = _center(tuple(left["bbox_display"])), _center(tuple(right["bbox_display"]))
        dx, dy = abs(right_center[0] - left_center[0]), abs(right_center[1] - left_center[1])
        orientation = "horizontal" if dx >= dy else "vertical"
        along, cross = (dx, dy) if orientation == "horizontal" else (dy, dx)
        if along < max(30.0, 0.08 * max(parent_box[2] - parent_box[0], parent_box[3] - parent_box[1])):
            continue
        if cross > max(12.0, 0.04 * along):
            continue
        perpendicular = "vertical" if orientation == "horizontal" else "horizontal"
        first, second = sorted((left, right), key=lambda item: _center(tuple(item["bbox_display"]))[0 if orientation == "horizontal" else 1])
        first_stem, second_stem = unique_stem(first, perpendicular), unique_stem(second, perpendicular)
        if first_stem is None or second_stem is None or first_stem["id"] == second_stem["id"]:
            continue
        axis = 0 if orientation == "horizontal" else 1
        first_token_axis = _center(tuple(first["bbox_display"]))[axis]
        second_token_axis = _center(tuple(second["bbox_display"]))[axis]
        first_stem_axis = _center(first_stem["bbox"])[axis]
        second_stem_axis = _center(second_stem["bbox"])[axis]
        first_offset = first_stem_axis - first_token_axis
        second_offset = second_stem_axis - second_token_axis
        if (
            first_stem_axis >= second_stem_axis
            or first_offset * second_offset > 0
            or max(abs(first_offset), abs(second_offset)) > max(18.0, 0.05 * along)
        ):
            continue
        first_anchor = _center(first_stem["bbox"])
        second_anchor = _center(second_stem["bbox"])
        bbox = (
            min(first_anchor[0], second_anchor[0]),
            min(first_anchor[1], second_anchor[1]),
            max(first_anchor[0], second_anchor[0]),
            max(first_anchor[1], second_anchor[1]),
        )
        results.append(
            {
                "orientation": orientation,
                "bbox_display": list(bbox),
                "virtual_span_display": [list(first_anchor), list(second_anchor)],
                "primitive_refs": sorted({first_stem["id"], second_stem["id"]}),
                "label_occurrence_refs": sorted({first["id"], second["id"]}),
                "label_support": 2,
                "endpoint_pair_supported": True,
                "full_section_label_supported": False,
                "semantic_support": "paired_identical_endpoint_tokens_with_unique_native_stems",
                "geometry_state": "inferred_between_native_endpoint_chains",
                "confidence": 0.86,
            }
        )
    return results


def _trace_candidates(
    observation_graph: Mapping[str, Any] | None,
    parent_box: tuple[float, float, float, float],
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_diagonal = _diagonal(parent_box)
    groups = _line_groups(observation_graph, parent_box)

    def evaluate(group: dict[str, Any]) -> dict[str, Any] | None:
        box = tuple(group["bbox_display"])
        supported = []
        for occurrence in occurrences:
            point = _center(tuple(occurrence["bbox_display"]))
            if group["orientation"] == "horizontal":
                cross = abs(point[1] - (box[1] + box[3]) / 2.0)
                along = max(box[0] - point[0], point[0] - box[2], 0.0)
            else:
                cross = abs(point[0] - (box[0] + box[2]) / 2.0)
                along = max(box[1] - point[1], point[1] - box[3], 0.0)
            label_box = tuple(occurrence["bbox_display"])
            cross_limit = max(8.0, 0.5 * max(label_box[2] - label_box[0], label_box[3] - label_box[1]))
            along_limit = max(14.0, 0.025 * parent_diagonal)
            if cross <= cross_limit and along <= along_limit:
                supported.append(occurrence)
        if not supported:
            return None
        span = max(box[2] - box[0], box[3] - box[1]) / max(parent_diagonal, 1e-9)
        endpoint_tokens = [
            item for item in supported
            if item["occurrence_kind"] == "single_endpoint_token_candidate"
        ]
        along_axis = 0 if group["orientation"] == "horizontal" else 1
        token_coordinates = [_center(tuple(item["bbox_display"]))[along_axis] for item in endpoint_tokens]
        trace_span = max(box[2] - box[0], box[3] - box[1])
        endpoint_pair = (
            len(endpoint_tokens) >= 2
            and max(token_coordinates) - min(token_coordinates) >= max(20.0, 0.45 * trace_span)
        )
        full_label = any(item["occurrence_kind"] == "full_section_label" for item in supported)
        confidence = 0.58 + 0.12 * len(supported) + 0.12 * min(span / 0.5, 1.0) + (0.08 if endpoint_pair else 0.0)
        return {
            **group,
            "label_occurrence_refs": sorted(item["id"] for item in supported),
            "label_support": len(supported),
            "endpoint_pair_supported": endpoint_pair,
            "full_section_label_supported": full_label,
            "semantic_support": "paired_identical_endpoint_tokens" if endpoint_pair else "full_section_label" if full_label else "single_endpoint_token_candidate",
            "confidence": round(min(0.97, confidence), 3),
        }

    atomic = [candidate for group in groups if (candidate := evaluate(group)) is not None]
    composite = []
    collinear_tolerance = max(1.5, 0.004 * parent_diagonal)
    for left, right in combinations(atomic, 2):
        if left["orientation"] != right["orientation"]:
            continue
        left_box, right_box = tuple(left["bbox_display"]), tuple(right["bbox_display"])
        if left["orientation"] == "horizontal":
            cross_gap = abs((left_box[1] + left_box[3] - right_box[1] - right_box[3]) / 2.0)
        else:
            cross_gap = abs((left_box[0] + left_box[2] - right_box[0] - right_box[2]) / 2.0)
        if cross_gap > collinear_tolerance:
            continue
        merged = {
            "orientation": left["orientation"],
            "bbox_display": [
                min(left_box[0], right_box[0]),
                min(left_box[1], right_box[1]),
                max(left_box[2], right_box[2]),
                max(left_box[3], right_box[3]),
            ],
            "primitive_refs": sorted({*left["primitive_refs"], *right["primitive_refs"]}),
        }
        candidate = evaluate(merged)
        if candidate is not None and candidate["endpoint_pair_supported"]:
            composite.append(candidate)

    endpoint_chains = _endpoint_chain_trace_candidates(observation_graph, parent_box, occurrences)
    by_refs = {
        (tuple(item["primitive_refs"]), tuple(item["label_occurrence_refs"])): item
        for item in (*atomic, *composite, *endpoint_chains)
    }
    candidates = list(by_refs.values())
    candidates.sort(
        key=lambda item: (
            -item["endpoint_pair_supported"],
            -item["label_support"],
            -item["confidence"],
            item["primitive_refs"],
        )
    )
    return candidates


def _cutting_planes(
    engineering: Mapping[str, Any],
    views: list[Mapping[str, Any]],
    text_roles: Iterable[Mapping[str, Any]],
    observation_graph: Mapping[str, Any] | None,
    memberships: Mapping[str, set[str]],
    text_to_display_transform: tuple[float, float, float, float, float, float] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    occurrences = _section_occurrences(text_roles, text_to_display_transform)
    occurrences_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        occurrences_by_label[occurrence["label"]].append(occurrence)
    bindings = _section_bindings(engineering, views)
    view_by_id = {str(view["id"]): view for view in views}
    accepted = []
    candidates = []
    for binding in sorted(bindings, key=lambda item: (item["label"], item["section_id"])):
        label = binding["label"]
        section_view_id = binding.get("view_id")
        if section_view_id is None:
            candidates.append(
                {
                    "id": f"cutting_plane_candidate.{len(candidates) + 1:03d}",
                    "type": "cut_at",
                    "section_label": label,
                    "section_view_id": None,
                    "parent_view_id": None,
                    "state": "candidate",
                    "confidence": 0.0,
                    "evidence_refs": sorted(
                        {
                            ref
                            for item in binding.get("candidates", [])
                            for ref in item.get("evidence_refs", [])
                        }
                    ),
                    "reason": binding["reason"],
                    "unresolved_fields": ["section_view_id", "parent_view_id", "trace"],
                }
            )
            continue
        parent_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for occurrence in occurrences_by_label.get(label, []):
            for parent_view_id in _occurrence_view(occurrence, views, {str(section_view_id)}):
                parent_occurrences[parent_view_id].append(occurrence)
        section_objects = memberships.get(section_view_id, set())
        object_scoped_parents = {
            view_id: items
            for view_id, items in parent_occurrences.items()
            if section_objects & memberships.get(view_id, set())
        }
        if object_scoped_parents:
            parent_occurrences = defaultdict(list, object_scoped_parents)
        if not parent_occurrences:
            candidates.append(
                {
                    "id": f"cutting_plane_candidate.{len(candidates) + 1:03d}",
                    "type": "cut_at",
                    "section_label": label,
                    "section_view_id": section_view_id,
                    "parent_view_id": None,
                    "state": "candidate",
                    "confidence": 0.0,
                    "evidence_refs": [binding["section_id"], section_view_id],
                    "reason": "no repeated section-label occurrence is scoped to another view",
                    "unresolved_fields": ["parent_view_id", "trace"],
                }
            )
            continue
        traces_by_parent = {
            parent_view_id: _trace_candidates(
                observation_graph,
                _bbox(view_by_id[parent_view_id]),
                parent_labels,
            )
            for parent_view_id, parent_labels in parent_occurrences.items()
            if _bbox(view_by_id[parent_view_id]) is not None
        }
        paired_parents = {
            parent_view_id
            for parent_view_id, traces in traces_by_parent.items()
            if any(item["endpoint_pair_supported"] for item in traces)
        }
        if paired_parents:
            parent_occurrences = defaultdict(
                list,
                {
                    parent_view_id: parent_labels
                    for parent_view_id, parent_labels in parent_occurrences.items()
                    if parent_view_id in paired_parents
                },
            )
        for parent_view_id, parent_labels in sorted(parent_occurrences.items()):
            parent_box = _bbox(view_by_id[parent_view_id])
            section_box = _bbox(view_by_id[section_view_id])
            section_label_match = SECTION_LABEL_RE.fullmatch(str(label).strip())
            numeric_section_pair = bool(
                section_label_match
                and str(section_label_match.group("axis")).isdigit()
            )
            if (
                engineering.get("enforce_compact_transverse_projection")
                and numeric_section_pair
                and parent_box is not None
                and section_box is not None
                and _area(section_box) >= _area(parent_box)
            ):
                candidates.append(
                    {
                        "id": f"cutting_plane_candidate.{len(candidates) + 1:03d}",
                        "type": "cut_at",
                        "section_label": label,
                        "section_view_id": section_view_id,
                        "parent_view_id": parent_view_id,
                        "state": "candidate",
                        "confidence": 0.0,
                        "evidence_refs": [binding["section_id"], section_view_id, parent_view_id],
                        "reason": "reciprocal projection title is not the compact transverse section",
                        "unresolved_fields": ["compact_transverse_section"],
                    }
                )
                continue
            traces = traces_by_parent.get(parent_view_id, [])
            supported_traces = [
                item
                for item in traces
                if item["endpoint_pair_supported"] or item["full_section_label_supported"]
            ]
            unique_trace = bool(supported_traces) and (
                len(supported_traces) == 1
                or supported_traces[0]["endpoint_pair_supported"] > supported_traces[1]["endpoint_pair_supported"]
                or supported_traces[0]["label_support"] > supported_traces[1]["label_support"]
                or supported_traces[0]["confidence"] - supported_traces[1]["confidence"] >= 0.15
            )
            trace = supported_traces[0] if supported_traces else None
            unique_parent = len(parent_occurrences) == 1 or bool(trace and trace["endpoint_pair_supported"])
            object_candidates = sorted(memberships.get(section_view_id, set()) & memberships.get(parent_view_id, set()))
            evidence = sorted(
                {
                    binding["section_id"],
                    section_view_id,
                    parent_view_id,
                    *(item["id"] for item in parent_labels),
                    *(trace["primitive_refs"] if trace and unique_trace else ()),
                }
            )
            integration = _section_binding_prerequisite_certificate(
                engineering,
                label=label,
                section_view_id=section_view_id,
                parent_view_id=parent_view_id,
                views=views,
                trace_orientation=trace.get("orientation") if trace else None,
            )
            prerequisites_passed = integration["status"] == "passed"
            if unique_parent and unique_trace and prerequisites_passed:
                accepted.append(
                    {
                        "id": f"cutting_plane_relation.{len(accepted) + 1:03d}",
                        "type": "cut_at",
                        "section_label": label,
                        "section_id": binding["section_id"],
                        "section_view_id": section_view_id,
                        "parent_view_id": parent_view_id,
                        "parent_object": {
                            "state": "resolved" if len(object_candidates) == 1 else "unresolved",
                            "object_instance_id": object_candidates[0] if len(object_candidates) == 1 else None,
                            "candidates": object_candidates,
                        },
                        "trace": trace,
                        "state": "accepted",
                        "confidence": round(min(binding["candidates"][0]["score"], trace["confidence"]), 3),
                        "evidence_refs": sorted({*evidence, *integration["evidence_refs"]}),
                        "integration_certificate": integration,
                        "provenance": {
                            "method": "section-host containment, repeated label, unique collinear native trace, title segmentation, and dimension ownership adjudication",
                            "source_layers": ["compact_engineering_graph", "immutable_observation_graph"],
                        },
                        "unresolved_fields": [] if len(object_candidates) == 1 else ["parent_object.object_instance_id"],
                    }
                )
                continue
            candidates.append(
                {
                    "id": f"cutting_plane_candidate.{len(candidates) + 1:03d}",
                    "type": "cut_at",
                    "section_label": label,
                    "section_view_id": section_view_id,
                    "parent_view_id": parent_view_id,
                    "trace_candidates": traces,
                    "state": "candidate",
                    "confidence": trace["confidence"] if trace else traces[0]["confidence"] if traces else 0.0,
                    "evidence_refs": evidence,
                    "integration_certificate": integration,
                    "reason": (
                        "matching section labels occur in more than one possible parent view"
                        if not unique_parent
                        else "no complete section-label or paired-endpoint cutting trace is supported in the parent view"
                        if not supported_traces
                        else "multiple cutting traces remain equally supported"
                        if not unique_trace
                        else "section binding prerequisites are not closed: " + ", ".join(integration["unresolved_fields"])
                        if not prerequisites_passed
                        else "cutting-plane relation remains unresolved"
                    ),
                    "unresolved_fields": (
                        ["parent_view_id"]
                        if not unique_parent
                        else ["trace"]
                        if not unique_trace
                        else integration["unresolved_fields"]
                        if not prerequisites_passed
                        else []
                    ),
                }
            )
    conflicts: set[str] = set()
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in accepted:
        by_section[str(relation["section_id"])].append(relation)
        by_label_parent[(str(relation["section_label"]), str(relation["parent_view_id"]))].append(relation)
    for rows in by_section.values():
        if len(rows) > 1:
            object_ids = {
                item.get("parent_object", {}).get("object_instance_id") for item in rows
            }
            if None in object_ids or len(object_ids) != len(rows):
                conflicts.update(str(item["id"]) for item in rows)
    for rows in by_label_parent.values():
        if len(rows) > 1:
            conflicts.update(str(item["id"]) for item in rows)
    if conflicts:
        retained = []
        for relation in accepted:
            if str(relation["id"]) not in conflicts:
                retained.append(relation)
                continue
            candidates.append(
                {
                    **relation,
                    "id": f"cutting_plane_candidate.{len(candidates) + 1:03d}",
                    "state": "candidate",
                    "reason": "duplicate section binding or ambiguous parent remains after independent trace and metric checks",
                    "unresolved_fields": ["section_view_id", "parent_view_id"],
                }
            )
        accepted = retained
    return accepted, candidates, bindings


def _promote_reprojected_cut_scopes(
    frames: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    object_scopes: list[dict[str, Any]],
    object_scope_candidates: list[dict[str, Any]],
    shared_coordinates: Mapping[str, Any],
) -> None:
    """Promote one cut component only after its relative reprojection closes."""

    frame_by_view = {str(item["view_id"]): item for item in frames}
    for relation in relations:
        section_view_id = str(relation["section_view_id"])
        parent_view_id = str(relation["parent_view_id"])
        coordinate_scope = next(
            (
                item
                for item in shared_coordinates.get("scopes", []) or []
                if item.get("state") == "resolved_relative"
                and {section_view_id, parent_view_id} <= set(map(str, item.get("view_ids", [])))
                and any(
                    check.get("status") == "pass"
                    for check in item.get("reprojection_validations", []) or []
                )
            ),
            None,
        )
        metric_pair = relation.get("integration_certificate", {}).get(
            "relation_scoped_metric_pair", {}
        ) or {}
        if coordinate_scope is None or metric_pair.get("status") != "passed":
            continue
        contour_matches = [
            item
            for item in coordinate_scope.get("contour_correspondences", []) or []
            if item.get("state") == "accepted"
            and {str(item.get("parent_view_id")), str(item.get("child_view_id"))}
            == {parent_view_id, section_view_id}
        ]
        if len(contour_matches) != 1:
            continue
        existing_scopes = [
            item
            for item in object_scopes
            if {section_view_id, parent_view_id}
            <= set(map(str, item.get("view_ids", [])))
        ]
        if len(existing_scopes) == 1:
            existing_scope = existing_scopes[0]
            existing_scope["shared_coordinate_scope_id"] = str(coordinate_scope["id"])
            existing_scope["reprojection_validation_refs"] = [
                str(item.get("id") or f"{coordinate_scope['id']}.reprojection.{index}")
                for index, item in enumerate(
                    coordinate_scope.get("reprojection_validations", []), start=1
                )
                if item.get("status") == "pass"
            ]
            existing_scope["contour_correspondence_ref"] = str(contour_matches[0]["id"])
            existing_scope.setdefault(
                "extrusion_depth_mm", metric_pair.get("transverse_extent_mm")
            )
            existing_scope["evidence_refs"] = sorted(
                {
                    *map(str, existing_scope.get("evidence_refs", [])),
                    str(coordinate_scope["id"]),
                    str(contour_matches[0]["id"]),
                    *map(str, coordinate_scope.get("evidence_refs", [])),
                }
            )
            continue
        if existing_scopes:
            continue

        section_frame = frame_by_view[section_view_id]
        source_view_id = section_frame.get("source_view_id")
        shared_values = {float(item["value_mm"]) for item in metric_pair.get("spans", [])}
        supporting = []
        if source_view_id is not None:
            for frame in frames:
                if (
                    frame.get("source_view_id") != source_view_id
                    or str(frame["view_id"]) == section_view_id
                    or not frame.get("supporting_projection_only")
                ):
                    continue
                if any(
                    item.get("status") == "accepted"
                    and item.get("value_mm") is not None
                    and float(item["value_mm"]) in shared_values
                    for item in frame.get("metric_spans", []) or []
                ):
                    supporting.append(str(frame["view_id"]))

        scope_id = f"physical_object_scope.{len(object_scopes) + 1:03d}"
        scope = {
            "id": scope_id,
            "state": "resolved_relative",
            "epistemic_state": "derived",
            "ownership_kind": "unique_title_trace_metric_reprojection_component",
            "object_instance_id": None,
            "physical_object_identity_state": "resolved_relative",
            "quantity_aggregation_eligible": False,
            "solid_aggregation_eligible": False,
            "view_ids": [parent_view_id, section_view_id],
            "section_view_ids": [section_view_id],
            "parent_view_ids": [parent_view_id],
            "supporting_projection_view_ids": sorted(supporting),
            "relation_refs": [str(relation["id"])],
            "shared_coordinate_scope_id": str(coordinate_scope["id"]),
            "reprojection_validation_refs": [
                str(item.get("id") or f"{coordinate_scope['id']}.reprojection.{index}")
                for index, item in enumerate(
                    coordinate_scope.get("reprojection_validations", []), start=1
                )
                if item.get("status") == "pass"
            ],
            "contour_correspondence_ref": str(contour_matches[0]["id"]),
            "extrusion_depth_mm": metric_pair.get("transverse_extent_mm"),
            "confidence": round(min(float(relation.get("confidence", 0.8)), 0.9), 3),
            "evidence_refs": sorted(
                {
                    str(relation["id"]),
                    str(coordinate_scope["id"]),
                    *map(str, relation.get("evidence_refs", [])),
                    *map(str, coordinate_scope.get("evidence_refs", [])),
                }
            ),
            "unresolved_fields": [
                "physical_origin",
                "absolute_axis_signs",
                "solid_hypothesis",
            ],
        }
        object_scopes.append(scope)
        object_scope_candidates[:] = [
            item
            for item in object_scope_candidates
            if not (
                set(map(str, item.get("view_ids", [])))
                == {section_view_id, parent_view_id}
                and item.get("reason")
                == "one section-to-parent relation is insufficient to create a new physical-object scope"
            )
        ]
        for view_id in [parent_view_id, section_view_id]:
            owner = {
                "state": "resolved_relative",
                "object_instance_id": None,
                "object_scope_id": scope_id,
                "ownership_kind": scope["ownership_kind"],
                "epistemic_state": "derived",
                "physical_object_identity_state": "resolved_relative",
                "quantity_aggregation_eligible": False,
                "candidates": [],
                "confidence": scope["confidence"],
                "evidence_refs": [scope_id, view_id, *scope["evidence_refs"]],
                "reason": "unique accepted cut and metric reprojection close one relative physical scope",
            }
            frame_by_view[view_id]["parent_object"] = owner
            frame_by_view[view_id]["unresolved_fields"] = [
                item
                for item in frame_by_view[view_id].get("unresolved_fields", [])
                if item != "parent_object.object_instance_id"
            ]
        for view_id in supporting:
            frame_by_view[view_id]["supporting_projection_of_scope"] = {
                "state": "supporting_projection",
                "object_scope_id": scope_id,
                "quantity_aggregation_eligible": False,
                "evidence_refs": [scope_id, view_id],
            }
        relation["parent_object"] = dict(frame_by_view[parent_view_id]["parent_object"])
        relation["unresolved_fields"] = [
            item
            for item in relation.get("unresolved_fields", [])
            if item != "parent_object.object_instance_id"
        ]


def infer_view_frames(
    engineering: Mapping[str, Any],
    *,
    dimensions: Iterable[Any] = (),
    text_roles: Iterable[Mapping[str, Any]] = (),
    observation_graph: Mapping[str, Any] | None = None,
    page_number: int | None = None,
    text_to_display_transform: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Infer partial view frames without filling unsupported 3D fields.

    ``engineering`` is one compact engineering-graph page.  Optional native
    dimension records provide measured endpoints; the immutable observation
    graph can independently provide scale-bearing dimension nodes and native
    trace geometry.  ``text_roles`` is optional because label text is not yet
    retained by compact view references.  If native text boxes are in the
    unrotated PDF coordinate system, pass the page's six-coefficient affine
    rotation matrix through ``text_to_display_transform``.
    """

    transform = None if text_to_display_transform is None else tuple(float(value) for value in text_to_display_transform)
    if transform is not None and len(transform) != 6:
        raise ValueError("text_to_display_transform must contain six affine coefficients")
    dimension_by_id = _dimension_index(dimensions, observation_graph)
    views, _ = _title_segment_views(
        engineering,
        list(engineering.get("view_hypotheses", [])),
        dimension_by_id,
    )
    metric_claim_refs = _claim_refs(engineering)
    memberships = _object_memberships(engineering)
    accepted_cuts, cut_candidates, section_bindings = _cutting_planes(
        engineering,
        views,
        text_roles,
        observation_graph,
        memberships,
        transform,
    )
    ownership_by_view, object_scopes, object_scope_candidates = _resolve_object_ownership(
        engineering,
        memberships,
        accepted_cuts,
    )
    for relation in accepted_cuts:
        section_owner = ownership_by_view.get(str(relation["section_view_id"]), {})
        parent_owner = ownership_by_view.get(str(relation["parent_view_id"]), {})
        section_key = section_owner.get("object_instance_id") or section_owner.get("object_scope_id")
        parent_key = parent_owner.get("object_instance_id") or parent_owner.get("object_scope_id")
        if section_owner.get("state") == parent_owner.get("state") == "resolved" and section_key == parent_key and section_key is not None:
            relation["parent_object"] = dict(parent_owner)
            relation["unresolved_fields"] = [
                field
                for field in relation.get("unresolved_fields", [])
                if field != "parent_object.object_instance_id"
            ]
    cuts_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted_cuts:
        cuts_by_section[item["section_view_id"]].append(item)
    labels_by_view = {
        binding["view_id"]: binding["label"]
        for binding in section_bindings
        if binding.get("view_id") is not None
    }
    frames = []
    ownership_by_dimension: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for attachment in engineering.get("dimension_ownership", {}).get("attachments", []) or []:
        if attachment.get("dimension_ref"):
            ownership_by_dimension[str(attachment["dimension_ref"])].append(attachment)
    equations_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for equation in engineering.get("metric_equation_graph", {}).get("constraints", []) or []:
        chain = equation.get("chain_attachment") or {}
        view_id = equation.get("view_id")
        if equation.get("status") != "accepted" or view_id is None or not chain:
            continue
        equations_by_view[str(view_id)].append(
            {
                "id": str(equation["id"]),
                "status": "accepted",
                "score": float(chain.get("score", 1.0)),
                "scale_points_per_mm": float(chain["scale_points_per_mm"]),
                "value_mm": float(equation["extent_mm"]),
                "orientation": str(chain["orientation"]),
                "measured_points": list(chain.get("measured_points_display", [])),
                "primitive_refs": list(chain.get("primitive_refs", [])),
                "semantic_role": "distribution_extent",
                "equation": equation.get("arithmetic", {}).get("equation"),
            }
        )
    relation_metric_spans_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in accepted_cuts:
        pair = relation.get("integration_certificate", {}).get(
            "relation_scoped_metric_pair", {}
        ) or {}
        if pair.get("status") != "passed":
            continue
        for span in pair.get("spans", []) or []:
            if span.get("view_id") is not None:
                relation_metric_spans_by_view[str(span["view_id"])].append(dict(span))
    for view in views:
        view_id = str(view["id"])
        box = _bbox(view)
        if box is None:
            continue
        local_dimensions = [dimension_by_id[ref] for ref in view.get("dimension_refs", []) if ref in dimension_by_id]
        local_equations = equations_by_view.get(view_id, [])
        local_relation_spans = relation_metric_spans_by_view.get(view_id, [])
        local_ownership = [
            attachment
            for dimension in local_dimensions
            for attachment in ownership_by_dimension.get(dimension["id"], [])
        ]
        scale = _scale_field(
            [*local_dimensions, *local_equations, *local_relation_spans],
            metric_claim_refs,
        )
        origin = _origin_field(local_dimensions, box, local_ownership)
        owner = ownership_by_view.get(view_id, _owner_field(view_id, memberships))
        cuts = cuts_by_section.get(view_id, [])
        projection = {
            "state": "unresolved",
            "vector_object_xyz": None,
            "confidence": 0.0,
            "evidence_refs": [],
            "candidates": [
                {"space": "object", "axis": axis}
                for axis in ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
            ],
            "reason": "the drawing does not yet define a canonical object-space basis and viewing sign",
        }
        relative_constraints = []
        parent_view = {
            "state": "unresolved",
            "view_id": None,
            "view_ids": [],
            "confidence": 0.0,
            "evidence_refs": [],
            "reason": "no unique cutting-plane relation connects this view to a parent projection",
        }
        if cuts:
            parent_ids = sorted({cut["parent_view_id"] for cut in cuts})
            parent_evidence = sorted({ref for cut in cuts for ref in (cut["id"], *cut["evidence_refs"])})
            parent_view = {
                "state": "resolved" if len(parent_ids) == 1 else "resolved_set",
                "view_id": parent_ids[0] if len(parent_ids) == 1 else None,
                "view_ids": parent_ids,
                "confidence": min(cut["confidence"] for cut in cuts),
                "evidence_refs": parent_evidence,
                "reason": "accepted cutting-plane relation" if len(parent_ids) == 1 else "multiple independently evidenced parent-view cutting-plane mentions",
            }
            projection_candidates = []
            for cut in cuts:
                trace_orientation = cut["trace"]["orientation"]
                along = [1.0, 0.0] if trace_orientation == "horizontal" else [0.0, 1.0]
                normal = [0.0, 1.0] if trace_orientation == "horizontal" else [1.0, 0.0]
                relative_constraints.append(
                    {
                        "type": "section_axis_parallel_to_parent_cut_trace",
                        "section_axis": "u",
                        "parent_view_id": cut["parent_view_id"],
                        "parent_display_direction": along,
                        "state": "derived",
                        "evidence_refs": [cut["id"], *cut["trace"]["primitive_refs"]],
                    }
                )
                projection_candidates.extend(
                    [
                        {"space": "parent_display", "parent_view_id": cut["parent_view_id"], "vector": normal},
                        {"space": "parent_display", "parent_view_id": cut["parent_view_id"], "vector": [-normal[0], -normal[1]]},
                    ]
                )
            projection["candidates"] = projection_candidates
            projection["reason"] = "cut trace fixes the in-parent projection axis but arrow direction or viewing sign is unresolved"
            projection["evidence_refs"] = sorted({ref for cut in cuts for ref in (cut["id"], *cut["trace"]["primitive_refs"])})
        unresolved = []
        if scale["state"] != "resolved":
            unresolved.append("scale.value_points_per_mm")
        if origin["state"] != "resolved":
            unresolved.append("origin.object_origin_display")
        unresolved.extend(["axes.object_axis_mapping", "projection_direction.vector_object_xyz"])
        if owner["state"] != "resolved":
            unresolved.append("parent_object.object_instance_id")
        if labels_by_view.get(view_id) and parent_view["state"] == "unresolved":
            unresolved.append("parent_view.view_id")
        evidence_refs = sorted(
            {
                view_id,
                *view.get("primitive_refs", []),
                *scale.get("evidence_refs", []),
                *origin.get("evidence_refs", []),
                *owner.get("evidence_refs", []),
                *parent_view.get("evidence_refs", []),
            }
        )
        resolved_fields = 2 + (scale["state"] == "resolved") + (origin["state"] == "resolved") + (owner["state"] == "resolved") + parent_view["state"].startswith("resolved")
        frames.append(
            {
                "id": f"view_frame.{len(frames) + 1:03d}",
                "view_id": view_id,
                "view_role_hypothesis": view.get("role_hypothesis"),
                "section_label": labels_by_view.get(view_id),
                "state": "resolved" if not unresolved else "partial" if resolved_fields > 2 else "candidate",
                "confidence": round(min(float(view.get("confidence", 0.5)), 0.42 + 0.08 * resolved_fields), 3),
                "scale": scale,
                "metric_spans": [
                    {
                        "id": item["id"],
                        "status": item["status"],
                        "value_mm": item.get("value_mm"),
                        "orientation": item.get("orientation"),
                        "measured_points": item.get("measured_points", []),
                        "primitive_refs": item.get("primitive_refs", []),
                        "semantic_role": next(
                            (
                                attachment.get("semantic_role")
                                for attachment in ownership_by_dimension.get(item["id"], [])
                                if attachment.get("status") == "accepted"
                            ),
                            None,
                        ),
                        "owner_entity_refs": sorted(
                            {
                                str(owner_ref)
                                for attachment in ownership_by_dimension.get(item["id"], [])
                                if attachment.get("status") == "accepted"
                                for owner_ref in attachment.get("owner_entity_refs", [])
                            }
                        ),
                        "geometry_anchor_refs": sorted(
                            {
                                str(anchor_ref)
                                for attachment in ownership_by_dimension.get(item["id"], [])
                                if attachment.get("status") == "accepted"
                                for anchor_ref in attachment.get("geometry_anchor_refs", [])
                                if anchor_ref
                            }
                        ),
                    }
                    for item in local_dimensions
                ]
                + [
                    {
                        "id": item["id"],
                        "status": item["status"],
                        "value_mm": item["value_mm"],
                        "orientation": item["orientation"],
                        "measured_points": item["measured_points"],
                        "primitive_refs": item["primitive_refs"],
                        "semantic_role": item["semantic_role"],
                        "owner_entity_refs": [view_id],
                        "geometry_anchor_refs": [],
                        "equation": item["equation"],
                    }
                    for item in local_equations
                ]
                + [
                    {
                        "id": item["id"],
                        "status": item["status"],
                        "value_mm": item["value_mm"],
                        "orientation": item["orientation"],
                        "measured_points": item["measured_points"],
                        "primitive_refs": item["primitive_refs"],
                        "semantic_role": item["semantic_role"],
                        "owner_entity_refs": [view_id],
                        "geometry_anchor_refs": item.get("geometry_anchor_refs", []),
                        "epistemic_state": item["epistemic_state"],
                        "acceptance_scope": item["acceptance_scope"],
                        "global_dimension_adjudication_status": item[
                            "global_dimension_adjudication_status"
                        ],
                        "quantity_aggregation_eligible": False,
                    }
                    for item in local_relation_spans
                ],
                "origin": origin,
                "axes": {
                    "display_basis": {
                        "state": "direct",
                        "u": [1.0, 0.0],
                        "v": [0.0, 1.0],
                        "handedness": "PDF display coordinates; v increases downward",
                        "evidence_refs": [view_id],
                    },
                    "object_axis_mapping": {
                        "state": "unresolved",
                        "u": None,
                        "v": None,
                        "normal": None,
                        "candidates": ["XY", "XZ", "YZ"],
                        "reason": "orthogonality alone does not name object-space axes",
                    },
                    "relative_constraints": relative_constraints,
                },
                "projection_direction": projection,
                "parent_view": parent_view,
                "parent_object": owner,
                "bbox_display": list(box),
                "source_view_id": view.get("source_view_id"),
                "title_segment_id": view.get("title_segment_id"),
                "title": view.get("title"),
                "normalised_title": view.get("normalised_title"),
                "scale_ratio": view.get("scale_ratio"),
                "supporting_projection_only": bool(view.get("supporting_projection_only")),
                "evidence_refs": evidence_refs,
                "provenance": {
                    "method": "drawing-neutral metric, containment, topology, and unique-evidence constraints",
                    "source_layers": ["compact_engineering_graph", "immutable_observation_graph"],
                    "page": page_number,
                },
                "unresolved_fields": sorted(set(unresolved)),
            }
        )
    shared_coordinates = solve_shared_coordinate_system(
        frames,
        accepted_cuts,
        object_scopes,
        engineering.get("object_instance_graph", {}),
    )
    solve_contour_correspondence(
        frames,
        shared_coordinates,
        engineering.get("contour_hypotheses", []),
        views,
        engineering.get("dimension_ownership", {}),
        engineering.get("native_segments", []),
    )
    if engineering.get("enforce_profile_section_reprojection"):
        for relation in accepted_cuts:
            matching_scopes = [
                scope
                for scope in shared_coordinates.get("scopes", []) or []
                if {str(relation["parent_view_id"]), str(relation["section_view_id"])}
                <= set(map(str, scope.get("view_ids", [])))
            ]
            correspondences = [
                item
                for scope in matching_scopes
                for item in scope.get("contour_correspondences", []) or []
                if item.get("state") == "accepted"
                and {str(item.get("parent_view_id")), str(item.get("child_view_id"))}
                == {str(relation["parent_view_id"]), str(relation["section_view_id"])}
            ]
            scale_transfers = [
                item["selected"]["scale_transfer_certificate"]
                for item in correspondences
                if ((item.get("selected") or {}).get("scale_transfer_certificate") or {}).get("status") == "passed"
            ]
            if len(scale_transfers) == 1:
                metric_pair = relation.get("integration_certificate", {}).get(
                    "relation_scoped_metric_pair", {}
                ) or {}
                metric_pair["metric_scale_ratio"] = 1.0
                metric_pair["metric_scale_compatibility"] = {
                    "status": "passed",
                    "parent_scale_points_per_mm": scale_transfers[0]["parent_scale_points_per_mm"],
                    "section_scale_points_per_mm": scale_transfers[0]["source_view_scale_points_per_mm"],
                    "same_plotted_scale": True,
                    "basis": scale_transfers[0]["basis"],
                    "display_span_residual_points": scale_transfers[0]["display_span_residual_points"],
                    "reason": "unique native silhouette reprojection transfers the independently resolved section scale",
                }
            reprojections = [
                item
                for scope in matching_scopes
                for item in scope.get("reprojection_validations", []) or []
                if item.get("status") == "pass"
            ]
            relation["profile_section_reprojection_certificate"] = {
                "status": "passed" if len(correspondences) == 1 and reprojections else "unresolved",
                "contour_correspondence_refs": [str(item["id"]) for item in correspondences],
                "metric_scale_transfer": scale_transfers[0] if len(scale_transfers) == 1 else None,
                "reprojection_validation_refs": [
                    str(item.get("id") or f"reprojection.{index}")
                    for index, item in enumerate(reprojections, start=1)
                ],
                "reason": (
                    "one native profile/section correspondence and metric reprojection pass"
                    if len(correspondences) == 1 and reprojections
                    else "profile/section reprojection has not closed uniquely"
                ),
            }
        unresolved_projection_relations = [
            item
            for item in accepted_cuts
            if item["profile_section_reprojection_certificate"]["status"] != "passed"
        ]
        accepted_cuts = [
            item
            for item in accepted_cuts
            if item["profile_section_reprojection_certificate"]["status"] == "passed"
        ]
        for item in unresolved_projection_relations:
            cut_candidates.append(
                {
                    **item,
                    "id": f"cutting_plane_candidate.{len(cut_candidates) + 1:03d}",
                    "state": "candidate",
                    "reason": item["profile_section_reprojection_certificate"]["reason"],
                    "unresolved_fields": ["profile_section_reprojection_certificate"],
                }
            )
    _promote_reprojected_cut_scopes(
        frames,
        accepted_cuts,
        object_scopes,
        object_scope_candidates,
        shared_coordinates,
    )
    certify_signed_orientation(
        frames,
        shared_coordinates,
        engineering.get("dimension_ownership", {}),
        engineering.get("native_segments", []),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "view_frame_graph",
        "status": "resolved_subset" if frames else "unresolved",
        "frames": frames,
        "relations": accepted_cuts,
        "relation_candidates": cut_candidates,
        "section_view_bindings": section_bindings,
        "object_scopes": object_scopes,
        "object_scope_candidates": object_scope_candidates,
        "shared_coordinate_system": shared_coordinates,
        "validation": {
            "accepted_cut_relations_are_unique": all(item["state"] == "accepted" for item in accepted_cuts),
            "accepted_cut_relation_count": len(accepted_cuts),
            "ambiguous_cut_relations_remain_candidates": True,
            "accepted_cut_relations_require_integration_certificate": True,
            "schedule_values_used": False,
            "object_class_template_used": False,
            "filename_dispatch_used": False,
            "equal_labels_alone_create_object_ownership": False,
        },
        "summary": {
            "frame_count": len(frames),
            "metric_scale_resolved_count": sum(item["scale"]["state"] == "resolved" for item in frames),
            "object_origin_resolved_count": sum(item["origin"]["state"] == "resolved" for item in frames),
            "object_scope_resolved_count": sum(item["parent_object"]["state"] == "resolved" for item in frames),
            "resolved_object_scope_count": len(object_scopes),
            "resolved_relative_physical_scope_count": sum(
                item.get("state") == "resolved_relative"
                and item.get("physical_object_identity_state") == "resolved_relative"
                for item in object_scopes
            ),
            "explicit_object_instance_scope_count": sum(item["object_instance_id"] is not None for item in object_scopes),
            "derived_view_containment_scope_count": sum(
                item["object_instance_id"] is None
                and item.get("ownership_kind") != "unique_redundant_metric_equation_component"
                for item in object_scopes
            ),
            "metric_equation_scope_count": sum(
                item.get("ownership_kind") == "unique_redundant_metric_equation_component"
                for item in object_scopes
            ),
            "unresolved_object_scope_candidate_count": len(object_scope_candidates),
            "cutting_plane_relation_count": len(accepted_cuts),
            "cutting_plane_candidate_count": len(cut_candidates),
            **shared_coordinates["summary"],
        },
        "contract": {
            "fields_resolve_independently": True,
            "projection_direction_is_not_inferred_from_view_role_alone": True,
            "ambiguous_relations_fail_closed": True,
            "source_geometry_remains_authoritative": True,
            "derived_scope_is_not_an_object_instance_assertion": True,
            "relative_coordinate_gauge_is_not_a_physical_origin": True,
            "signed_orientation_requires_native_markers_and_asymmetric_landmarks": True,
            "component_placement_cannot_select_coordinate_sign": True,
        },
    }
