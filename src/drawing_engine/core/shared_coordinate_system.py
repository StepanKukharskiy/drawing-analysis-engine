"""Close drawing-neutral relative coordinate frames across orthographic views.

The solver names an arbitrary but stable object-space gauge and propagates it
only through accepted cutting-plane relations or independently assembled
orthogonal projection pairs.  A relative gauge is useful for reprojection and
identity matching, but it is not promoted to a physical datum without direct
dimension evidence.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
import math
from typing import Any


SCHEMA_VERSION = "0.1.0"
_ANCHOR_MAPPING = {"u": "X", "v": "Z", "normal": "Y"}


def _area(frame: Mapping[str, Any]) -> float:
    box = frame.get("bbox_display") or (0, 0, 0, 0)
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _derived(mapping: Mapping[str, str], mode: str, forward: bool) -> dict[str, str]:
    """Propagate an unsigned axis permutation through one orthographic edge."""

    if mode == "cut_horizontal":
        return (
            {"u": mapping["u"], "v": mapping["normal"], "normal": mapping["v"]}
            if forward
            else {"u": mapping["u"], "v": mapping["normal"], "normal": mapping["v"]}
        )
    if mode == "cut_vertical":
        return (
            {"u": mapping["v"], "v": mapping["normal"], "normal": mapping["u"]}
            if forward
            else {"u": mapping["normal"], "v": mapping["u"], "normal": mapping["v"]}
        )
    if mode == "pair_side":
        return {"u": mapping["normal"], "v": mapping["v"], "normal": mapping["u"]}
    if mode == "pair_stack":
        return {"u": mapping["u"], "v": mapping["normal"], "normal": mapping["v"]}
    raise ValueError(f"unsupported orthographic constraint mode: {mode}")


def _metric_spans(frame: Mapping[str, Any], orientation: str) -> list[Mapping[str, Any]]:
    return [
        item
        for item in frame.get("metric_spans", []) or []
        if item.get("status") == "accepted"
        and item.get("orientation") == orientation
        and item.get("value_mm") is not None
        and float(item["value_mm"]) > 0
    ]


def _reprojection_check(
    constraint: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parent = frames[constraint["parent_view_id"]]
    child = frames[constraint["child_view_id"]]
    mode = constraint["mode"]
    if mode == "cut_horizontal":
        parent_orientation, child_orientation, shared_axis = "horizontal", "horizontal", "u"
    elif mode == "cut_vertical":
        # The section's vertical display axis preserves the parent profile's
        # vertical axis.  Its horizontal 200 mm span is the orthogonal depth.
        parent_orientation, child_orientation, shared_axis = "vertical", "vertical", "v"
    elif mode == "pair_side":
        parent_orientation, child_orientation, shared_axis = "vertical", "vertical", "v"
    else:
        parent_orientation, child_orientation, shared_axis = "horizontal", "horizontal", "u"
    pairs = []
    for left in _metric_spans(parent, parent_orientation):
        for right in _metric_spans(child, child_orientation):
            residual = abs(float(left["value_mm"]) - float(right["value_mm"]))
            tolerance = max(2.0, 0.02 * max(float(left["value_mm"]), float(right["value_mm"])))
            pairs.append((residual, tolerance, left, right))
    if not pairs:
        return {
            "status": "unknown",
            "shared_axis": shared_axis,
            "residual_mm": None,
            "tolerance_mm": None,
            "evidence_refs": list(constraint.get("evidence_refs", [])),
            "reason": "no metric spans on the shared display axis",
        }
    residual, tolerance, left, right = min(
        pairs,
        key=lambda item: (item[0], max(float(item[2]["value_mm"]), float(item[3]["value_mm"]))),
    )
    explicitly_same_span = (
        left.get("semantic_role") == right.get("semantic_role") == "overall_shared_axis_span"
    )
    if residual > tolerance and not explicitly_same_span:
        return {
            "status": "unknown",
            "shared_axis": shared_axis,
            "residual_mm": round(residual, 6),
            "tolerance_mm": round(tolerance, 6),
            "evidence_refs": sorted(
                {
                    *constraint.get("evidence_refs", []),
                    *[str(item["id"]) for item in _metric_spans(parent, parent_orientation)],
                    *[str(item["id"]) for item in _metric_spans(child, child_orientation)],
                }
            ),
            "reason": "no dimension ownership evidence proves that the unequal spans measure the same overall axis",
        }
    return {
        "status": "pass" if residual <= tolerance else "fail",
        "shared_axis": shared_axis,
        "parent_value_mm": float(left["value_mm"]),
        "child_value_mm": float(right["value_mm"]),
        "residual_mm": round(residual, 6),
        "tolerance_mm": round(tolerance, 6),
        "evidence_refs": sorted(
            {
                *constraint.get("evidence_refs", []),
                str(left["id"]),
                str(right["id"]),
                *left.get("primitive_refs", []),
                *right.get("primitive_refs", []),
            }
        ),
        "reason": "independent dimensioned spans agree" if residual <= tolerance else "dimensioned shared-axis spans conflict",
    }


def _cut_plane_constraint(
    constraint: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    mappings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any] | None:
    if not constraint["mode"].startswith("cut_"):
        return None
    relation = constraint["source"]
    trace_box = relation.get("trace", {}).get("bbox_display")
    parent = frames[constraint["parent_view_id"]]
    scale = parent.get("scale", {})
    if not trace_box or scale.get("state") != "resolved" or not scale.get("value_points_per_mm"):
        return None
    orientation = relation["trace"]["orientation"]
    coordinate_index = 1 if orientation == "horizontal" else 0
    trace_coordinate = (float(trace_box[coordinate_index]) + float(trace_box[coordinate_index + 2])) / 2.0
    origin = parent.get("origin", {})
    anchor = origin.get("object_origin_display")
    anchor_state = "dimension_datum" if origin.get("state") == "resolved" and anchor else "view_gauge"
    if not anchor:
        box = parent["bbox_display"]
        anchor = [float(box[0]), float(box[3])]
    display_delta = trace_coordinate - float(anchor[coordinate_index])
    delta_mm = display_delta / float(scale["value_points_per_mm"])
    plane_axis = mappings[constraint["child_view_id"]]["normal"]
    magnitude = round(abs(delta_mm), 6)
    candidates = [magnitude] if magnitude == 0 else [-magnitude, magnitude]
    return {
        "relation_id": relation.get("id"),
        "section_view_id": constraint["child_view_id"],
        "parent_view_id": constraint["parent_view_id"],
        "object_axis": plane_axis,
        "coordinate_candidates_mm": candidates,
        "signed_coordinate_in_parent_gauge_mm": round(delta_mm, 6),
        "viewing_sign_state": "unresolved" if len(candidates) == 2 else "irrelevant",
        "parent_anchor_state": anchor_state,
        "parent_anchor_display": list(anchor),
        "trace_coordinate_display": round(trace_coordinate, 6),
        "evidence_refs": list(constraint.get("evidence_refs", [])),
    }


def _constraints(
    relations: Iterable[Mapping[str, Any]],
    object_instance_graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for relation in relations:
        if relation.get("type") != "cut_at" or relation.get("state") != "accepted":
            continue
        orientation = relation.get("trace", {}).get("orientation")
        if orientation not in {"horizontal", "vertical"}:
            continue
        rows.append(
            {
                "id": f"coordinate_constraint.cut.{len(rows) + 1:03d}",
                "kind": "cut_at",
                "parent_view_id": str(relation["parent_view_id"]),
                "child_view_id": str(relation["section_view_id"]),
                "mode": f"cut_{orientation}",
                "trace": {
                    "orientation": orientation,
                    "bbox_display": list(relation.get("trace", {}).get("bbox_display", [])),
                    "primitive_refs": list(relation.get("trace", {}).get("primitive_refs", [])),
                },
                "source_relation_id": str(relation.get("id")),
                "source": relation,
                "evidence_refs": [str(relation.get("id")), *relation.get("trace", {}).get("primitive_refs", [])],
            }
        )
    for instance in object_instance_graph.get("instances", []) or []:
        mode = instance.get("layout_mode")
        if (
            mode not in {"right", "left", "above", "below"}
            or not instance.get("primary_view_id")
            or not instance.get("section_view_id")
        ):
            continue
        rows.append(
            {
                "id": f"coordinate_constraint.pair.{len(rows) + 1:03d}",
                "kind": "orthogonal_projection_pair",
                "parent_view_id": str(instance["primary_view_id"]),
                "child_view_id": str(instance["section_view_id"]),
                "mode": "pair_side" if mode in {"right", "left"} else "pair_stack",
                "source": instance,
                "evidence_refs": [str(instance["id"]), *map(str, instance.get("basis", []))],
            }
        )
    return rows


def _groups(
    object_scopes: Iterable[Mapping[str, Any]],
    object_instance_graph: Mapping[str, Any],
    constraints: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups = []
    for instance in object_instance_graph.get("instances", []) or []:
        groups.append(
            {
                "source_id": str(instance["id"]),
                "kind": "physical_object_candidate",
                "object_instance_id": str(instance["id"]),
                "view_ids": sorted(map(str, instance.get("view_ids", []))),
                "preferred_anchor_view_id": (
                    str(instance["primary_view_id"])
                    if instance.get("primary_view_id") is not None
                    else None
                ),
            }
        )
    for scope in object_scopes:
        view_ids = sorted(map(str, scope.get("view_ids", [])))
        if len(view_ids) < 2 or any(set(view_ids) <= set(item["view_ids"]) for item in groups):
            continue
        groups.append(
            {
                "source_id": str(scope.get("object_scope_id") or scope["id"]),
                "kind": "view_containment_scope",
                "object_instance_id": scope.get("object_instance_id"),
                "view_ids": view_ids,
                "preferred_anchor_view_id": None,
            }
        )
    covered = {view_id for group in groups for view_id in group["view_ids"]}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for item in constraints:
        left, right = item["parent_view_id"], item["child_view_id"]
        if left in covered or right in covered:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = set()
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        queue, component = [seed], set()
        while queue:
            item = queue.pop()
            if item in component:
                continue
            component.add(item)
            queue.extend(adjacency[item] - component)
        visited.update(component)
        if len(component) >= 2:
            groups.append(
                {
                    "source_id": f"cut_component.{len(groups) + 1:03d}",
                    "kind": "cut_relation_component",
                    "object_instance_id": None,
                    "view_ids": sorted(component),
                    "preferred_anchor_view_id": None,
                }
            )
    return groups


def _anchor(group: Mapping[str, Any], frames: Mapping[str, Mapping[str, Any]], constraints: Iterable[Mapping[str, Any]]) -> str:
    preferred = group.get("preferred_anchor_view_id")
    if preferred in frames:
        return str(preferred)
    parent_support = defaultdict(int)
    for item in constraints:
        parent_support[item["parent_view_id"]] += 1
    return max(
        (view_id for view_id in group["view_ids"] if view_id in frames),
        key=lambda view_id: (
            parent_support[view_id],
            frames[view_id].get("scale", {}).get("state") == "resolved",
            frames[view_id].get("view_role_hypothesis") != "section_view_candidate",
            _area(frames[view_id]),
            view_id,
        ),
    )


def _propagate(
    anchor_view_id: str,
    view_ids: set[str],
    constraints: list[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    mappings = {anchor_view_id: dict(_ANCHOR_MAPPING)}
    conflicts = []
    queue = deque([anchor_view_id])
    by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in constraints:
        by_view[item["parent_view_id"]].append(item)
        by_view[item["child_view_id"]].append(item)
    while queue:
        current = queue.popleft()
        for item in by_view[current]:
            if item["parent_view_id"] == current:
                other, candidate = item["child_view_id"], _derived(mappings[current], item["mode"], True)
            else:
                other, candidate = item["parent_view_id"], _derived(mappings[current], item["mode"], False)
            if other not in view_ids:
                continue
            if other not in mappings:
                mappings[other] = candidate
                queue.append(other)
            elif mappings[other] != candidate:
                conflicts.append(
                    {
                        "constraint_id": item["id"],
                        "view_id": other,
                        "existing_mapping": mappings[other],
                        "candidate_mapping": candidate,
                        "evidence_refs": item.get("evidence_refs", []),
                    }
                )
    return mappings, conflicts


def solve_shared_coordinate_system(
    frames: list[dict[str, Any]],
    relations: Iterable[Mapping[str, Any]],
    object_scopes: Iterable[Mapping[str, Any]],
    object_instance_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve relative object axes and metric cross-view consistency."""

    object_instance_graph = object_instance_graph or {}
    frames_by_view = {str(item["view_id"]): item for item in frames}
    constraints = _constraints(relations, object_instance_graph)
    scopes = []
    membership: dict[str, list[str]] = defaultdict(list)
    for group in _groups(object_scopes, object_instance_graph, constraints):
        view_ids = {item for item in group["view_ids"] if item in frames_by_view}
        local_constraints = [
            item
            for item in constraints
            if item["parent_view_id"] in view_ids and item["child_view_id"] in view_ids
        ]
        if len(view_ids) < 2 or not local_constraints:
            continue
        anchor_view_id = _anchor(group, frames_by_view, local_constraints)
        mappings, conflicts = _propagate(anchor_view_id, view_ids, local_constraints)
        validations = [_reprojection_check(item, frames_by_view) for item in local_constraints]
        plane_constraints = [
            plane
            for item in local_constraints
            for plane in [_cut_plane_constraint(item, frames_by_view, mappings)]
            if plane is not None and item["child_view_id"] in mappings
        ]
        failed = any(item["status"] == "fail" for item in validations)
        all_mapped = view_ids <= mappings.keys()
        has_cut_topology = any(item["kind"] == "cut_at" for item in local_constraints)
        pass_count = sum(item["status"] == "pass" for item in validations)
        state = (
            "ambiguous"
            if conflicts or failed
            else "resolved_relative"
            if all_mapped and (has_cut_topology or pass_count > 0)
            else "partial"
        )
        scope_id = f"shared_coordinate_scope.{len(scopes) + 1:03d}"
        scope = {
            "id": scope_id,
            "source_scope_id": group["source_id"],
            "scope_kind": group["kind"],
            "object_instance_id": group["object_instance_id"],
            "state": state,
            "anchor_view_id": anchor_view_id,
            "gauge": {
                "state": "relative",
                "axis_definition": "anchor display u=X, display v=Z, view normal=Y",
                "absolute_axis_sign_state": "unresolved",
                "physical_origin_state": "unresolved",
            },
            "view_ids": sorted(view_ids),
            "view_axis_mappings": [
                {
                    "view_id": view_id,
                    **mappings[view_id],
                    "state": "resolved_relative" if state == "resolved_relative" else "candidate",
                }
                for view_id in sorted(mappings)
            ],
            "constraint_ids": [item["id"] for item in local_constraints],
            "reprojection_validations": validations,
            "cut_plane_constraints": plane_constraints,
            "conflicts": conflicts,
            "bar_identity_eligible": state == "resolved_relative" and pass_count > 0,
            "quantity_aggregation_eligible": (
                state == "resolved_relative"
                and pass_count > 0
                and group["object_instance_id"] is not None
                and all(frames_by_view[item].get("origin", {}).get("state") == "resolved" for item in mappings)
            ),
            "evidence_refs": sorted(
                {
                    group["source_id"],
                    *[ref for item in local_constraints for ref in item.get("evidence_refs", [])],
                    *[ref for item in validations for ref in item.get("evidence_refs", [])],
                }
            ),
        }
        scopes.append(scope)
        for view_id in mappings:
            frame = frames_by_view[view_id]
            mapping = mappings[view_id]
            if state != "resolved_relative":
                frame.setdefault("shared_coordinate_candidates", []).append(
                    {"scope_id": scope_id, **mapping, "reason": "orthogonal layout is not closed by a shared metric span"}
                )
                continue
            membership[view_id].append(scope_id)
            frame["shared_coordinate_scope_id"] = scope_id
            frame["axes"]["object_axis_mapping"] = {
                "state": "resolved_relative",
                **mapping,
                "absolute_axis_sign_state": "unresolved",
                "evidence_refs": scope["evidence_refs"],
                "reason": "propagated through accepted orthographic constraints from a stable object-space gauge",
            }
            frame["projection_direction"] = {
                "state": "axis_resolved_sign_unresolved",
                "axis": mapping["normal"],
                "vector_object_xyz": None,
                "confidence": 0.82 if state == "resolved_relative" else 0.55,
                "evidence_refs": scope["evidence_refs"],
                "candidates": [f"+{mapping['normal']}", f"-{mapping['normal']}"],
                "reason": "orthographic constraints resolve the normal axis but not the viewing sign",
            }
            frame["unresolved_fields"] = [
                item for item in frame.get("unresolved_fields", []) if item != "axes.object_axis_mapping"
            ]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "shared_relative_coordinate_system",
        "status": "resolved_subset" if any(item["state"] == "resolved_relative" for item in scopes) else "unresolved",
        "scopes": scopes,
        "constraints": [
            {key: value for key, value in item.items() if key != "source"}
            for item in constraints
        ],
        "view_membership": {key: sorted(value) for key, value in sorted(membership.items())},
        "summary": {
            "scope_count": len(scopes),
            "resolved_relative_scope_count": sum(item["state"] == "resolved_relative" for item in scopes),
            "axis_mapping_resolved_view_count": len(membership),
            "reprojection_pass_count": sum(
                check["status"] == "pass"
                for scope in scopes
                for check in scope["reprojection_validations"]
            ),
            "reprojection_fail_count": sum(
                check["status"] == "fail"
                for scope in scopes
                for check in scope["reprojection_validations"]
            ),
            "bar_identity_eligible_scope_count": sum(item["bar_identity_eligible"] for item in scopes),
            "quantity_aggregation_eligible_scope_count": sum(item["quantity_aggregation_eligible"] for item in scopes),
        },
        "validation": {
            "schedule_values_used": False,
            "filename_dispatch_used": False,
            "object_class_template_used": False,
            "absolute_axis_sign_is_not_invented": True,
            "view_gauge_is_not_a_physical_origin": True,
            "conflicting_axis_cycles_abstain": True,
        },
    }


def reclose_shared_coordinate_system(
    frames: list[dict[str, Any]],
    relations: Iterable[Mapping[str, Any]],
    object_instance_graph: Mapping[str, Any] | None = None,
    *,
    certified_constraint_ids: Iterable[str] = (),
    translation_errors: Iterable[str] = (),
    input_uniqueness_errors: Iterable[str] = (),
) -> dict[str, Any]:
    """Re-run coordinate gates over translated certified native constraints.

    This is the replay boundary used by :mod:`src.drawing_engine.disciplines.concrete.solver_replay`.  It never
    consumes quantities and mutates only private copies of the supplied view
    frames while re-running the existing coordinate solver.
    """

    native_frames = deepcopy(frames)
    native_relations = [deepcopy(dict(item)) for item in relations]
    native_objects = deepcopy(dict(object_instance_graph or {}))
    constraint_ids = sorted({str(item) for item in certified_constraint_ids})
    errors = sorted({str(item) for item in translation_errors if str(item)})
    frame_ids = [str(item.get("view_id")) for item in native_frames if item.get("view_id") is not None]
    frame_by_id = {str(item.get("view_id")): item for item in native_frames if item.get("view_id") is not None}

    uniqueness_errors = sorted({str(item) for item in input_uniqueness_errors if str(item)})
    if len(frame_ids) != len(set(frame_ids)):
        uniqueness_errors.append("native view IDs are not unique")
    cut_parent_by_section: dict[str, set[str]] = defaultdict(set)
    for relation in native_relations:
        if relation.get("type") == "cut_at":
            cut_parent_by_section[str(relation.get("section_view_id"))].add(str(relation.get("parent_view_id")))
    for section_id, parent_ids in sorted(cut_parent_by_section.items()):
        if len(parent_ids) > 1:
            uniqueness_errors.append(f"section view {section_id} has multiple certified cut parents")

    object_membership: dict[str, set[str]] = defaultdict(set)
    valid_instances = []
    incomplete_instances = []
    for instance in native_objects.get("instances", []) or []:
        instance_id = str(instance.get("id"))
        primary = instance.get("primary_view_id")
        section = instance.get("section_view_id")
        view_ids = list(map(str, instance.get("view_ids", []) or []))
        if not primary or not section or str(primary) == str(section):
            incomplete_instances.append(instance_id)
            continue
        if len(view_ids) != len(set(view_ids)) or {str(primary), str(section)} - set(view_ids):
            uniqueness_errors.append(f"object {instance_id} has non-unique projection membership")
            continue
        for view_id in view_ids:
            object_membership[view_id].add(instance_id)
        valid_instances.append(instance)
    for view_id, object_ids in sorted(object_membership.items()):
        if len(object_ids) > 1:
            uniqueness_errors.append(f"view {view_id} belongs to multiple certified objects")
    native_objects["instances"] = valid_instances

    referenced_pairs = [
        (str(item.get("parent_view_id")), str(item.get("section_view_id")))
        for item in native_relations
        if item.get("type") == "cut_at"
    ] + [
        (str(item.get("primary_view_id")), str(item.get("section_view_id")))
        for item in valid_instances
    ]
    referenced_views = {view_id for pair in referenced_pairs for view_id in pair}
    missing_views = sorted(referenced_views - frame_by_id.keys())
    metric_missing = sorted(
        view_id
        for view_id in referenced_views & frame_by_id.keys()
        if frame_by_id[view_id].get("scale", {}).get("state") != "resolved"
        or not isinstance(frame_by_id[view_id].get("scale", {}).get("value_points_per_mm"), (int, float))
        or isinstance(frame_by_id[view_id].get("scale", {}).get("value_points_per_mm"), bool)
        or not math.isfinite(float(frame_by_id[view_id]["scale"]["value_points_per_mm"]))
        or float(frame_by_id[view_id]["scale"]["value_points_per_mm"]) <= 0
        or not any(
            item.get("status") == "accepted"
            and isinstance(item.get("value_mm"), (int, float))
            and not isinstance(item.get("value_mm"), bool)
            and math.isfinite(float(item["value_mm"]))
            and float(item["value_mm"]) > 0
            for item in frame_by_id[view_id].get("metric_spans", []) or []
        )
    )

    has_topology = bool(referenced_pairs)
    if not constraint_ids:
        status = "insufficient_constraints"
        reason = "no certified coordinate topology is available"
        result = None
    elif uniqueness_errors:
        status = "reclosed_fail"
        reason = "certified native inputs fail the uniqueness gate"
        result = None
    elif not has_topology:
        status = "insufficient_constraints"
        reason = "no complete certified coordinate topology is available"
        result = None
    elif errors or missing_views or metric_missing or incomplete_instances:
        status = "insufficient_constraints"
        reason = "certified constraints cannot be translated into complete metric native inputs"
        result = None
    else:
        result = solve_shared_coordinate_system(native_frames, native_relations, [], native_objects)
        scopes = result.get("scopes", []) or []
        validations = [
            validation
            for scope in scopes
            for validation in scope.get("reprojection_validations", []) or []
        ]
        axis_fail = any(scope.get("conflicts") or scope.get("state") == "ambiguous" for scope in scopes)
        axis_pass = bool(scopes) and all(
            scope.get("state") == "resolved_relative"
            and len(scope.get("view_axis_mappings", [])) == len(scope.get("view_ids", []))
            for scope in scopes
        )
        reprojection_fail = any(item.get("status") == "fail" for item in validations)
        reprojection_pass = bool(validations) and all(item.get("status") == "pass" for item in validations)
        if axis_fail or reprojection_fail:
            status = "reclosed_fail"
            reason = "axis or reprojection gates failed on the translated native constraints"
        elif not axis_pass or not reprojection_pass:
            status = "insufficient_constraints"
            reason = "metric evidence does not close every axis and reprojection gate"
        else:
            status = "reclosed_pass"
            reason = "metric, uniqueness, axis, and reprojection gates reclosed"

    validations = [] if result is None else [
        validation
        for scope in result.get("scopes", []) or []
        for validation in scope.get("reprojection_validations", []) or []
    ]
    scopes = [] if result is None else result.get("scopes", []) or []
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "shared_coordinate_system",
        "status": status,
        "reason": reason,
        "certified_constraint_ids": constraint_ids,
        "gates": {
            "metric": {
                "status": "pass" if not metric_missing and has_topology else "insufficient",
                "missing_view_ids": metric_missing,
            },
            "uniqueness": {
                "status": "fail" if uniqueness_errors else "pass" if has_topology else "insufficient",
                "errors": uniqueness_errors,
            },
            "axis": {
                "status": (
                    "fail"
                    if any(scope.get("conflicts") or scope.get("state") == "ambiguous" for scope in scopes)
                    else "pass"
                    if scopes and all(scope.get("state") == "resolved_relative" for scope in scopes)
                    else "insufficient"
                ),
                "resolved_scope_count": sum(scope.get("state") == "resolved_relative" for scope in scopes),
            },
            "reprojection": {
                "status": (
                    "fail"
                    if any(item.get("status") == "fail" for item in validations)
                    else "pass"
                    if validations and all(item.get("status") == "pass" for item in validations)
                    else "insufficient"
                ),
                "pass_count": sum(item.get("status") == "pass" for item in validations),
                "fail_count": sum(item.get("status") == "fail" for item in validations),
                "unknown_count": sum(item.get("status") == "unknown" for item in validations),
            },
        },
        "translation": {
            "status": "pass" if not errors and not missing_views else "insufficient",
            "errors": errors,
            "missing_native_view_ids": missing_views,
            "incomplete_object_ids": incomplete_instances,
        },
        "coordinate_system": result,
        "contract": {
            "certified_constraints_only": True,
            "native_metric_gates_rerun": True,
            "native_uniqueness_gate_rerun": True,
            "axis_gate_rerun": True,
            "reprojection_gate_rerun": True,
            "quantities_read": False,
            "quantities_written": False,
        },
    }
