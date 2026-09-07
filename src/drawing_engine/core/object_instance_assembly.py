"""Procedural grouping of drawing views into physical-object hypotheses.

The grouping is deliberately based on repeated projection layout and metric
evidence.  It does not use filenames, object-class templates, or schedule
values.  A group is an auditable hypothesis until downstream geometry closes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import fitz


def _box(view: dict[str, Any]) -> fitz.Rect:
    return fitz.Rect(view["bbox_display"])


def _aspect(rect: fitz.Rect) -> float:
    return max(rect.width / max(rect.height, 0.1), rect.height / max(rect.width, 0.1))


def _overlap_fraction(left: fitz.Rect, right: fitz.Rect, axis: str) -> float:
    if axis == "x":
        overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
        return overlap / max(min(left.width, right.width), 0.1)
    overlap = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return overlap / max(min(left.height, right.height), 0.1)


def _pair_score(primary: dict[str, Any], section: dict[str, Any], mode: str) -> tuple[float, list[str]] | None:
    large = _box(primary)
    small = _box(section)
    area_ratio = small.get_area() / max(large.get_area(), 0.1)
    if not 0.015 <= area_ratio <= 0.22:
        return None

    if mode == "right":
        gap, span, alignment = small.x0 - large.x1, large.width, _overlap_fraction(large, small, "y")
        center_residual = abs((small.y0 + small.y1 - large.y0 - large.y1) / 2) / max(large.height, 0.1)
    elif mode == "left":
        gap, span, alignment = large.x0 - small.x1, large.width, _overlap_fraction(large, small, "y")
        center_residual = abs((small.y0 + small.y1 - large.y0 - large.y1) / 2) / max(large.height, 0.1)
    elif mode == "below":
        gap, span, alignment = small.y0 - large.y1, large.height, _overlap_fraction(large, small, "x")
        center_residual = abs((small.x0 + small.x1 - large.x0 - large.x1) / 2) / max(large.width, 0.1)
    else:
        gap, span, alignment = large.y0 - small.y1, large.height, _overlap_fraction(large, small, "x")
        center_residual = abs((small.x0 + small.x1 - large.x0 - large.x1) / 2) / max(large.width, 0.1)
    if gap < 0 or gap > 0.35 * span or alignment < 0.55 or center_residual > 0.42:
        return None

    proximity = 1.0 - gap / max(0.35 * span, 0.1)
    centered = 1.0 - center_residual / 0.42
    ratio_fit = 1.0 - min(abs(area_ratio - 0.09) / 0.13, 1.0)
    orthogonality = min(float(section.get("features", {}).get("orthogonal_axis_ratio", 0.0)) / 0.75, 1.0)
    score = 0.30 * proximity + 0.28 * alignment + 0.17 * centered + 0.12 * ratio_fit + 0.13 * orthogonality
    return min(score, 0.99), [
        f"{mode}_side_repeated_projection_layout",
        "strong_cross_axis_overlap",
        "smaller_dimensioned_orthogonal_projection",
        "compatible_relative_extent",
    ]


def _best_assignment(
    primary: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    mode: str,
) -> list[tuple[dict[str, Any], dict[str, Any], float, list[str]]]:
    options: dict[str, list[tuple[dict[str, Any], float, list[str]]]] = defaultdict(list)
    for section in sections:
        for candidate in primary:
            result = _pair_score(candidate, section, mode)
            if result is not None:
                options[section["id"]].append((candidate, result[0], result[1]))

    section_by_id = {item["id"]: item for item in sections}
    match_by_primary: dict[str, tuple[str, dict[str, Any], float, list[str]]] = {}

    def augment(section_id: str, visited: set[str]) -> bool:
        ranked = sorted(options.get(section_id, []), key=lambda item: (-item[1], item[0]["id"]))
        for candidate, score, factors in ranked:
            primary_id = candidate["id"]
            if primary_id in visited:
                continue
            visited.add(primary_id)
            previous = match_by_primary.get(primary_id)
            if previous is None or augment(previous[0], visited):
                match_by_primary[primary_id] = (section_id, candidate, score, factors)
                return True
        return False

    for section in sorted(sections, key=lambda item: item["id"]):
        augment(section["id"], set())
    return [
        (candidate, section_by_id[section_id], score, factors)
        for section_id, candidate, score, factors in match_by_primary.values()
    ]


def assemble_object_instances(page_rect: fitz.Rect, views: list[dict[str, Any]]) -> dict[str, Any]:
    """Group complementary metric projections without choosing an object class."""

    page_area = max(page_rect.get_area(), 1.0)
    primary = []
    sections = []
    for view in views:
        rect = _box(view)
        features = view.get("features", {})
        dimensions = int(features.get("accepted_dimensions", 0))
        area_fraction = rect.get_area() / page_area
        aspect = _aspect(rect)
        primitive_count = int(features.get("primitive_count", 0))
        orthogonality = float(features.get("orthogonal_axis_ratio", 0.0))
        if dimensions >= 2 and area_fraction >= 0.04 and aspect <= 3.0 and primitive_count >= 24:
            primary.append(view)
        if dimensions >= 1 and 0.001 <= area_fraction <= 0.04 and aspect >= 2.8 and orthogonality >= 0.55 and primitive_count >= 16:
            sections.append(view)

    assignments_by_mode = {
        mode: _best_assignment(primary, sections, mode)
        for mode in ("right", "left", "below", "above")
    }
    selected_mode, assignments = max(
        assignments_by_mode.items(),
        key=lambda item: (len(item[1]), sum(row[2] for row in item[1]), item[0]),
    )

    instances = []
    for index, (large, small, score, factors) in enumerate(
        sorted(assignments, key=lambda row: (_box(row[0]).y0, _box(row[0]).x0)),
        start=1,
    ):
        instances.append(
            {
                "id": f"object_instance.{index:03d}",
                "role": "physical_object_candidate",
                "state": "inferred",
                "confidence": round(score, 3),
                "primary_view_id": large["id"],
                "section_view_id": small["id"],
                "view_ids": [large["id"], small["id"]],
                "projection_roles": {
                    large["id"]: "primary_metric_projection",
                    small["id"]: "compact_orthogonal_projection",
                },
                "basis": factors,
                "layout_mode": selected_mode,
                "geometry_status": "multi_view_hypothesis_not_yet_closed",
            }
        )

    assigned = {view_id for item in instances for view_id in item["view_ids"]}
    shared = [
        {
            "view_id": item["id"],
            "state": "unresolved",
            "role": "shared_or_unpaired_dimensioned_view",
            "reason": "large metric view has no unique compact projection in the repeated layout motif",
        }
        for item in primary
        if item["id"] not in assigned
    ]
    view_membership = {
        view_id: [item["id"] for item in instances if view_id in item["view_ids"]]
        for view_id in sorted(assigned)
    }
    return {
        "schema_version": "0.1.0",
        "layer": "procedural_object_instance_graph",
        "status": "object_instances_hypothesized" if instances else "unresolved",
        "instances": instances,
        "shared_supporting_views": shared,
        "view_membership": view_membership,
        "layout_consensus": {
            "selected_mode": selected_mode if assignments else None,
            "support_count": len(assignments),
            "candidate_support_by_mode": {mode: len(rows) for mode, rows in assignments_by_mode.items()},
        },
        "unassigned_view_ids": sorted(item["id"] for item in views if item["id"] not in assigned and item["id"] not in {row["view_id"] for row in shared}),
        "validation": {
            "unique_view_membership": all(len(ids) == 1 for ids in view_membership.values()),
            "unique_primary_membership": len({item["primary_view_id"] for item in instances}) == len(instances),
            "unique_section_membership": len({item["section_view_id"] for item in instances}) == len(instances),
            "schedule_values_used": False,
            "object_class_template_used": False,
        },
        "contract": {
            "instance_is_hypothesis_until_geometry_closes": True,
            "unpaired_metric_views_are_not_forced_into_objects": True,
            "cross_view_rebar_matching_must_respect_instance_scope": True,
        },
        "summary": {
            "object_instance_count": len(instances),
            "shared_supporting_view_count": len(shared),
            "assigned_view_count": len(assigned),
        },
    }


def assemble_relation_certified_object_instances(
    view_frame_graph: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create physical scopes only from fully certified projection relations.

    Layout proximity is intentionally absent. Each instance requires a unique
    parent/section pair, title-scope ownership, a native marker trace, a shared
    metric extent, and unique profile/section reprojection evidence.
    """

    relations = [
        item
        for item in view_frame_graph.get("relations", []) or []
        if item.get("state") == "accepted"
        and item.get("integration_certificate", {}).get("status") == "passed"
        and item.get("profile_section_reprojection_certificate", {}).get("status") == "passed"
    ]
    parent_counts = defaultdict(int)
    section_counts = defaultdict(int)
    for relation in relations:
        parent_counts[str(relation.get("parent_view_id"))] += 1
        section_counts[str(relation.get("section_view_id"))] += 1
    scopes = list(view_frame_graph.get("object_scopes", []) or [])
    instances = []
    rejected = []
    for relation in relations:
        parent_id = str(relation["parent_view_id"])
        section_id = str(relation["section_view_id"])
        integration = relation["integration_certificate"]
        metric_pair = integration.get("relation_scoped_metric_pair", {}) or {}
        scope_matches = [
            item
            for item in scopes
            if item.get("state") == "resolved_relative"
            and {parent_id, section_id} <= set(map(str, item.get("view_ids", [])))
            and item.get("contour_correspondence_ref")
        ]
        reasons = []
        if parent_counts[parent_id] != 1:
            reasons.append("parent projection is not unique")
        if section_counts[section_id] != 1:
            reasons.append("section projection is not unique")
        if not integration.get("parent_segment_id") or not integration.get("section_segment_id"):
            reasons.append("title-scope ownership is incomplete")
        if not relation.get("trace", {}).get("primitive_refs"):
            reasons.append("native cutting marker trace is incomplete")
        if metric_pair.get("status") != "passed" or not metric_pair.get("transverse_extent_mm"):
            reasons.append("independent shared metric extent is incomplete")
        if len(scope_matches) != 1:
            reasons.append("profile/section reprojection scope is not unique")
        if reasons:
            rejected.append({"relation_id": relation.get("id"), "parent_view_id": parent_id, "section_view_id": section_id, "reasons": reasons})
            continue
        scope = scope_matches[0]
        instance_id = f"object_instance.certified.{len(instances) + 1:03d}"
        instances.append(
            {
                "id": instance_id,
                "role": "physical_object_scope",
                "state": "certified",
                "confidence": round(min(float(relation.get("confidence", 0.8)), float(scope.get("confidence", 0.8))), 3),
                "primary_view_id": parent_id,
                "section_view_id": section_id,
                "view_ids": [parent_id, section_id],
                "projection_roles": {parent_id: "longitudinal_metric_projection", section_id: "transverse_metric_section"},
                "section_label": relation.get("section_label"),
                "extrusion_depth_mm": float(metric_pair["transverse_extent_mm"]),
                "object_scope_id": scope["id"],
                "relation_id": relation["id"],
                "basis": sorted(
                    {
                        str(relation["id"]),
                        str(scope["id"]),
                        str(integration["parent_segment_id"]),
                        str(integration["section_segment_id"]),
                        *map(str, relation["trace"]["primitive_refs"]),
                        *(str(item["id"]) for item in metric_pair.get("spans", [])),
                        str(scope["contour_correspondence_ref"]),
                        *map(str, scope.get("reprojection_validation_refs", [])),
                    }
                ),
                "certificate": {
                    "status": "passed",
                    "title_scope_refs": [integration["parent_segment_id"], integration["section_segment_id"]],
                    "native_marker_refs": list(relation["trace"]["primitive_refs"]),
                    "metric_span_refs": [item["id"] for item in metric_pair.get("spans", [])],
                    "contour_correspondence_ref": scope["contour_correspondence_ref"],
                    "reprojection_validation_refs": list(scope.get("reprojection_validation_refs", [])),
                },
                "geometry_status": "projection_identity_certified_solid_pending",
                "quantity_aggregation_eligible": False,
            }
        )
    assigned = {view_id for item in instances for view_id in item["view_ids"]}
    membership = {view_id: [item["id"] for item in instances if view_id in item["view_ids"]] for view_id in sorted(assigned)}
    shared_supporting = [
        {
            "view_id": str(item["id"]),
            "state": "supporting_projection_candidate",
            "role": str(item.get("role_hypothesis") or "drawing_view_candidate"),
            "reason": "unassigned semantic projection may carry explicitly bracketed multi-object annotations",
        }
        for item in views
        if str(item["id"]) not in assigned
        and "section" not in str(item.get("role_hypothesis") or "")
        and str(item.get("role_hypothesis") or "") != "drawing_view_candidate"
        and int((item.get("features") or {}).get("primitive_count", 0)) >= 24
    ]
    return {
        "schema_version": "0.1.0",
        "layer": "relation_certified_object_instance_graph",
        "status": "object_instances_certified" if instances else "unresolved",
        "instances": instances,
        "view_membership": membership,
        "shared_supporting_views": shared_supporting,
        "rejected_relation_scopes": rejected,
        "unassigned_view_ids": sorted(str(item["id"]) for item in views if str(item["id"]) not in assigned),
        "validation": {
            "unique_view_membership": all(len(ids) == 1 for ids in membership.values()),
            "title_scope_ownership_required": True,
            "native_section_markers_required": True,
            "independent_metric_extent_required": True,
            "profile_section_reprojection_required": True,
            "spatial_proximity_used_as_certificate": False,
            "schedule_values_used": False,
            "object_class_template_used": False,
        },
        "contract": {
            "instance_precedes_solid_but_not_quantity": True,
            "quantity_requires_watertight_solid_reclosure": True,
            "cross_view_rebar_matching_must_respect_instance_scope": True,
        },
        "summary": {"object_instance_count": len(instances), "rejected_relation_scope_count": len(rejected), "assigned_view_count": len(assigned), "shared_supporting_view_count": len(shared_supporting)},
    }


def scope_rebar_path_graph(
    path_graph: dict[str, Any],
    assembly: dict[str, Any],
    view_frame_graph: dict[str, Any] | None = None,
) -> None:
    """Attach instance scope to path observations before identity matching."""

    membership = assembly.get("view_membership", {})
    coordinate_system = (view_frame_graph or {}).get("shared_coordinate_system", {})
    eligible_scopes = {
        item["id"]
        for item in coordinate_system.get("scopes", [])
        if item.get("bar_identity_eligible")
    }
    coordinate_membership = {
        view_id: [scope_id for scope_id in scope_ids if scope_id in eligible_scopes]
        for view_id, scope_ids in coordinate_system.get("view_membership", {}).items()
    }
    frame_by_view = {
        item["view_id"]: item
        for item in (view_frame_graph or {}).get("frames", [])
    }
    fragments = {item["id"]: item for item in path_graph.get("fragments", [])}
    for fragment in fragments.values():
        fragment["object_instance_ids"] = sorted(
            {instance_id for view_id in fragment.get("view_ids", []) for instance_id in membership.get(view_id, [])}
        )
        fragment["coordinate_scope_ids"] = sorted(
            {scope_id for view_id in fragment.get("view_ids", []) for scope_id in coordinate_membership.get(view_id, [])}
        )
    scoped = 0
    for component in path_graph.get("components", []):
        component["object_instance_ids"] = sorted(
            {
                instance_id
                for view_id in component.get("view_ids", [])
                for instance_id in membership.get(view_id, [])
            }
            | {
                instance_id
                for fragment_id in component.get("fragment_ids", [])
                for instance_id in fragments.get(fragment_id, {}).get("object_instance_ids", [])
            }
        )
        component["coordinate_scope_ids"] = sorted(
            {
                scope_id
                for view_id in component.get("view_ids", [])
                for scope_id in coordinate_membership.get(view_id, [])
            }
            | {
                scope_id
                for fragment_id in component.get("fragment_ids", [])
                for scope_id in fragments.get(fragment_id, {}).get("coordinate_scope_ids", [])
            }
        )
        if len(component.get("view_ids", [])) == 1:
            frame = frame_by_view.get(component["view_ids"][0], {})
            mapping = frame.get("axes", {}).get("object_axis_mapping", {})
            box = component.get("bbox_display", [])
            if mapping.get("state") == "resolved_relative" and len(box) == 4:
                display_axis = "u" if box[2] - box[0] >= box[3] - box[1] else "v"
                component["object_axis_projection"] = {
                    "state": "resolved_relative",
                    "axis": mapping.get(display_axis),
                    "station_axis": mapping.get("v" if display_axis == "u" else "u"),
                    "view_normal_axis": mapping.get("normal"),
                    "display_axis": display_axis,
                    "view_frame_id": frame.get("id"),
                    "absolute_axis_sign_state": mapping.get("absolute_axis_sign_state", "unresolved"),
                }
        scoped += bool(component["object_instance_ids"])
    path_graph["object_instance_scopes"] = [
        {"object_instance_id": item["id"], "view_ids": item["view_ids"]}
        for item in assembly.get("instances", [])
    ]
    path_graph["coordinate_scopes"] = [
        {
            "coordinate_scope_id": item["id"],
            "view_ids": item["view_ids"],
            "bar_identity_eligible": item.get("bar_identity_eligible", False),
            "path_reprojection_eligible": item.get("path_reprojection_eligible", False),
            "view_axis_mappings": item.get("view_axis_mappings", []),
            "contour_correspondences": item.get("contour_correspondences", []),
            "view_scales_points_per_mm": {
                str(frame["view_id"]): frame.get("scale", {}).get("value_points_per_mm")
                for frame in (view_frame_graph or {}).get("frames", [])
                if str(frame.get("view_id")) in set(map(str, item.get("view_ids", [])))
            },
        }
        for item in (view_frame_graph or {}).get("shared_coordinate_system", {}).get("scopes", [])
        if item.get("bar_identity_eligible")
    ]
    path_graph.setdefault("summary", {})["object_scoped_component_count"] = scoped
    path_graph.setdefault("summary", {})["coordinate_scoped_component_count"] = sum(
        bool(item.get("coordinate_scope_ids")) for item in path_graph.get("components", [])
    )
    path_graph.setdefault("validation", {})["cross_object_candidate_identity_count"] = 0
