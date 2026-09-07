"""Drawing-neutral intermediate representation for reinforcement programs.

Concrete geometry and reinforcement semantics have intentionally separate
lifecycles. This module records symbolic bar groups, repetition constraints,
identity hypotheses, and fabrication unknowns even when no concrete solid can
be closed. It never reads schedules or drawing titles.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import fitz

from src.drawing_engine.core.cross_view_identity import extract_thin_segments, trace_leader
from src.drawing_engine.core.dimension_attachment import DimensionAttachment, attach_dimensions
from src.drawing_engine.disciplines.rebar.rebar_fabrication import attach_fabrication_details
from src.drawing_engine.disciplines.rebar.rebar_path_graph import build_rebar_path_graph, enrich_projection_identities


SPACING_EXPRESSION_RE = re.compile(
    r"(?P<spacing>\d{1,5})\s*[xх×]\s*(?P<intervals>\d{1,4})\s*=\s*(?P<extent>\d{1,6})",
    re.I,
)
STEP_SPACING_RE = re.compile(r"(?:шаг|spacing|step|@)\s*=?\s*(?P<spacing>\d{1,5})", re.I)
MARK_BEFORE_STEP_RE = re.compile(
    r"(?P<mark>(?:[A-ZА-Я]{1,3})?\d{1,3}(?:/\d{1,4})?)\s*(?=(?:шаг|spacing|step|@))",
    re.I,
)
STANDALONE_MARK_RE = re.compile(r"^(?:[A-ZА-Я]{1,3})?\d{1,3}(?:/\d{1,4})?$", re.I)
DRAWING_REF_RE = re.compile(r"drawing\[(?P<index>\d+)\]")


def _point_rect_distance(point: list[float], rect: fitz.Rect) -> float:
    x, y = point
    return math.hypot(max(rect.x0 - x, x - rect.x1, 0.0), max(rect.y0 - y, y - rect.y1, 0.0))


def _rect_distance(left: fitz.Rect, right: fitz.Rect) -> float:
    return math.hypot(
        max(left.x0 - right.x1, right.x0 - left.x1, 0.0),
        max(left.y0 - right.y1, right.y0 - left.y1, 0.0),
    )


def _view_for_box(box: fitz.Rect, views: list[dict[str, Any]], padding: float = 0.0) -> str | None:
    center = fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    candidates = []
    for view in views:
        view_box = fitz.Rect(view["bbox_display"]) + (-padding, -padding, padding, padding)
        if center in view_box:
            candidates.append((view_box.get_area(), view["id"]))
    return min(candidates)[1] if candidates else None


def _drawing_boxes(page: fitz.Page, refs: Iterable[str]) -> list[fitz.Rect]:
    drawings = page.get_drawings()
    indexes = sorted(
        {
            int(match.group("index"))
            for ref in refs
            if (match := DRAWING_REF_RE.search(ref)) is not None
        }
    )
    return [fitz.Rect(drawings[index]["rect"]) for index in indexes if index < len(drawings)]


def _text_lines(page: fitz.Page) -> Iterable[tuple[str, fitz.Rect, list[str]]]:
    for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
        for line_index, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not text:
                continue
            box = fitz.Rect(spans[0]["bbox"])
            for span in spans[1:]:
                box |= fitz.Rect(span["bbox"])
            refs = [f"text_block[{block_index}].line[{line_index}].span[{index}]" for index in range(len(spans))]
            yield text, box, refs


def extract_spacing_constraints(
    page: fitz.Page,
    text_roles: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Extract direct repetition equations without assigning them to bars."""

    constraints = []
    sources = [
        {"text": text, "box": box, "refs": refs, "method": "native_text"}
        for text, box, refs in _text_lines(page)
    ]
    sources.extend(
        {
            "text": str(item.get("text") or ""),
            "box": fitz.Rect(item["bbox_display"]),
            "refs": [item["id"], *item.get("primitive_refs", [])],
            "method": item.get("text_method", "native_text_role"),
        }
        for item in text_roles
        if item.get("resolved_role") == "spacing_expression_token"
    )
    for source_index, source in enumerate(sources):
        text, box, refs = source["text"], source["box"], source["refs"]
        for match_index, match in enumerate(SPACING_EXPRESSION_RE.finditer(text), start=1):
            spacing = int(match.group("spacing"))
            intervals = int(match.group("intervals"))
            extent = int(match.group("extent"))
            expected = spacing * intervals
            constraints.append(
                {
                    "id": f"spacing_constraint.{len(constraints) + 1:03d}",
                    "type": "distribution_segment",
                    "spacing_mm": spacing,
                    "interval_count": intervals,
                    "physical_count_if_isolated": intervals + 1,
                    "extent_mm": extent,
                    "arithmetic": {
                        "equation": f"{spacing} * {intervals} = {extent}",
                        "expected_extent_mm": expected,
                        "residual_mm": expected - extent,
                        "status": "pass" if expected == extent else "conflict",
                    },
                    "association": {"rebar_group_id": None, "state": "unknown"},
                    "state": "direct",
                    "source_text": match.group(0),
                    "text_method": source["method"],
                    "bbox_display": list(box),
                    "primitive_refs": [*refs, f"spacing_match[{match_index - 1}]"],
                }
            )
        for match_index, match in enumerate(STEP_SPACING_RE.finditer(text), start=1):
            spacing = int(match.group("spacing"))
            prefix = text[: match.start()]
            mark_match = MARK_BEFORE_STEP_RE.search(prefix + text[match.start() :])
            mark_token = mark_match.group("mark") if mark_match else None
            if mark_token is None:
                nearby = []
                for candidate_index, candidate in enumerate(sources):
                    if candidate_index == source_index or not STANDALONE_MARK_RE.fullmatch(candidate["text"].strip()):
                        continue
                    candidate_box = candidate["box"]
                    horizontal_overlap = max(0.0, min(box.x1, candidate_box.x1) - max(box.x0, candidate_box.x0))
                    if horizontal_overlap <= 0 and abs(candidate_box.x0 - box.x0) > 12:
                        continue
                    distance = math.hypot(
                        max(box.x0 - candidate_box.x1, candidate_box.x0 - box.x1, 0.0),
                        max(box.y0 - candidate_box.y1, candidate_box.y0 - box.y1, 0.0),
                    )
                    if distance <= 18:
                        nearby.append((distance, candidate["text"].strip(), candidate["refs"]))
                nearby.sort(key=lambda item: item[0])
                if len(nearby) == 1 or (len(nearby) > 1 and nearby[1][0] - nearby[0][0] >= 4):
                    mark_token = nearby[0][1]
                    refs = [*refs, *nearby[0][2]]
            signature = (spacing, mark_token, tuple(round(value, 1) for value in box))
            if any(item.get("dedup_signature") == signature for item in constraints):
                continue
            constraints.append(
                {
                    "id": f"annotated_spacing_constraint.{sum(item['type'] == 'annotated_distribution_spacing' for item in constraints) + 1:03d}",
                    "type": "annotated_distribution_spacing",
                    "spacing_mm": spacing,
                    "interval_count": None,
                    "physical_count_if_isolated": None,
                    "extent_mm": None,
                    "mark_token": mark_token,
                    "arithmetic": {
                        "equation": None,
                        "expected_extent_mm": None,
                        "residual_mm": None,
                        "status": "not_applicable",
                    },
                    "association": {"projected_path_component_id": None, "state": "unknown"},
                    "state": "observed",
                    "source_text": match.group(0),
                    "text_method": source["method"],
                    "bbox_display": list(box),
                    "primitive_refs": [*refs, f"step_spacing_match[{match_index - 1}]"],
                    "dedup_signature": signature,
                }
            )
    for item in constraints:
        item.pop("dedup_signature", None)
    return constraints


def _canonical_spacing(value: float) -> float:
    """Snap only very-near 25 mm drafting increments; otherwise preserve."""

    candidate = round(value / 25.0) * 25.0
    if candidate > 0 and abs(value - candidate) <= min(12.0, 0.12 * candidate):
        return candidate
    return round(value, 3)


def compress_linear_distribution(positions_mm: Iterable[float]) -> dict[str, Any]:
    """Compress resolved instances into auditable piecewise-equal zones."""

    positions = sorted(float(value) for value in positions_mm)
    if not positions:
        return {
            "type": "piecewise_linear_array",
            "axis": "Z",
            "zones": [],
            "instance_positions_mm": [],
            "resolved_count": 0,
            "expansion_validation": {"status": "unresolved", "reason": "no resolved positions"},
        }
    if len(positions) == 1:
        return {
            "type": "individual",
            "axis": "Z",
            "zones": [],
            "instance_positions_mm": positions,
            "resolved_count": 1,
            "expansion_validation": {"status": "pass", "expanded_count": 1, "max_abs_position_residual_mm": 0.0},
        }

    differences = [right - left for left, right in zip(positions, positions[1:])]
    labels = [_canonical_spacing(value) for value in differences]
    zones = []
    start_interval = 0
    for interval_index in range(1, len(labels) + 1):
        if interval_index < len(labels) and abs(labels[interval_index] - labels[start_interval]) <= 1e-6:
            continue
        interval_count = interval_index - start_interval
        spacing = labels[start_interval]
        start = positions[start_interval]
        observed_end = positions[interval_index]
        expected_end = start + interval_count * spacing
        raw = differences[start_interval:interval_index]
        zones.append(
            {
                "id": f"distribution_zone.{len(zones) + 1:03d}",
                "axis": "Z",
                "start_mm": round(start, 3),
                "end_mm": round(expected_end, 3),
                "observed_end_mm": round(observed_end, 3),
                "spacing_mm": spacing,
                "interval_count": interval_count,
                "physical_count_if_isolated": interval_count + 1,
                "extent_mm": round(interval_count * spacing, 3),
                "end_residual_mm": round(expected_end - observed_end, 3),
                "max_interval_residual_mm": round(max(abs(value - spacing) for value in raw), 3),
                "state": "derived",
            }
        )
        start_interval = interval_index

    expanded = expand_linear_distribution({"zones": zones})
    residuals = [left - right for left, right in zip(expanded, positions)] if len(expanded) == len(positions) else []
    max_residual = max((abs(value) for value in residuals), default=math.inf)
    tolerance = 15.0
    validation_status = "pass" if len(expanded) == len(positions) and max_residual <= tolerance else "conflict"
    return {
        "type": "piecewise_linear_array",
        "axis": "Z",
        "zones": zones,
        "instance_positions_mm": [round(value, 3) for value in positions],
        "resolved_count": len(positions),
        "count_equation": f"1 + sum(interval_count) = {len(positions)}",
        "expansion_validation": {
            "status": validation_status,
            "expanded_count": len(expanded),
            "resolved_count": len(positions),
            "max_abs_position_residual_mm": None if math.isinf(max_residual) else round(max_residual, 3),
            "tolerance_mm": tolerance,
        },
    }


def expand_linear_distribution(distribution: dict[str, Any]) -> list[float]:
    """Deterministically instantiate piecewise zones, sharing boundaries."""

    positions: list[float] = []
    for zone in distribution.get("zones", []):
        values = [float(zone["start_mm"]) + index * float(zone["spacing_mm"]) for index in range(int(zone["interval_count"]) + 1)]
        if positions and values and abs(values[0] - positions[-1]) <= 15.0:
            values = values[1:]
        positions.extend(values)
    return positions


def _path_length(points: list[list[float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _program_group(group: dict[str, Any], scene_paths: list[dict[str, Any]], index: int) -> dict[str, Any]:
    role = str(group.get("role", "unknown"))
    is_transverse = "tie" in role or "transverse" in role
    group_paths = [item for item in scene_paths if item.get("mark") == group.get("mark")]
    if is_transverse:
        half = group.get("section_centerline_half_size_mm", [None, None])
        parameters = {
            "width_mm": None if half[0] is None else round(2 * float(half[0]), 3),
            "height_mm": None if half[1] is None else round(2 * float(half[1]), 3),
        }
        topology = "closed_polyline"
        distribution = compress_linear_distribution(group.get("z_mm", []))
        orientation = "XY"
    else:
        parameters = {}
        topology = "straight"
        distribution = {
            "type": "cross_section_instances",
            "axis": "Z",
            "positions_xy_mm": [[round(float(x), 3), round(float(y), 3)] for x, y in group.get("xy_mm", [])],
            "resolved_count": int(group.get("count", 0)),
            "expansion_validation": {
                "status": "pass" if len(group.get("xy_mm", [])) == int(group.get("count", 0)) else "conflict",
                "expanded_count": len(group.get("xy_mm", [])),
                "resolved_count": int(group.get("count", 0)),
            },
        }
        orientation = "Z"
    placed_lengths = [_path_length(item["points_xyz_mm"]) for item in group_paths]
    evidence = sorted(set(group.get("primitive_refs", [])) | set(group.get("section_boundary_refs", [])))
    metric_solution = group.get("dimension_anchored_solution")
    installed = (metric_solution or {}).get("solved_centerline", {})
    return {
        "id": f"rebar_group.{index:03d}",
        "identity": {
            "mark": {"value": None, "state": "unknown", "evidence_refs": []},
            "source_solver_label": group.get("mark"),
        },
        "role": {"value": role, "state": "derived"},
        "bar_spec": {
            "diameter_mm": {
                "value": group.get("diameter_mm"),
                "state": "convention_dependent" if group.get("diameter_mm") is not None else "unknown",
                "basis": "graphical_projection_scale" if group.get("diameter_mm") is not None else None,
            },
            "steel_grade": {"value": None, "state": "unknown"},
            "standard": {"value": None, "state": "unknown"},
        },
        "topology": {
            "family": {"value": topology, "state": "derived"},
            "parameters": parameters,
            "geometry_2d_state": "drawing_constrained" if parameters else "partial",
        },
        "placement": {
            "host": "object.concrete.host",
            "orientation": orientation,
            "state": "drawing_constrained",
            "metric_solution": metric_solution,
        },
        "distribution": distribution,
        "quantity": {
            "value": int(group.get("count", 0)),
            "state": "derived",
            "basis": "resolved projection topology",
        },
        "placed_geometry": {
            "expanded_path_count": len(group_paths),
            "centerline_length_each_min_mm": None if not placed_lengths else round(min(placed_lengths), 3),
            "centerline_length_each_mean_mm": None if not placed_lengths else round(sum(placed_lengths) / len(placed_lengths), 3),
            "centerline_length_each_max_mm": None if not placed_lengths else round(max(placed_lengths), 3),
            "centerline_length_total_mm": round(sum(placed_lengths), 3),
            "state": "derived",
            "note": (
                "Dimension-anchored installed centerline; fabrication equivalence is evaluated separately."
                if metric_solution and metric_solution.get("status") == "pass"
                else "Placed centerline geometry is not a fabrication cutting length."
            ),
        },
        "lengths": {
            "projected_drawing": {
                "value_mm": None,
                "status": "observation_only",
                "note": "A visible projection is never emitted as a metric quantity by itself.",
            },
            "installed_centerline": {
                "value_each_mm": installed.get("installed_length_each_mm"),
                "raw_value_each_mm": installed.get("raw_installed_length_each_mm"),
                "status": "geometry_resolved" if metric_solution and metric_solution.get("status") == "pass" else "unresolved",
                "basis": "dimension anchors plus cross-view reprojection" if metric_solution and metric_solution.get("status") == "pass" else None,
            },
            "fabrication_cutting": {"value_each_mm": None, "status": "unresolved"},
        },
        "fabrication": {
            "cutting_length_each_mm": None,
            "cutting_length_total_mm": None,
            "status": "unresolved",
            "missing_constraints": ["bend radii", "hook extensions", "fabrication dimension convention"],
        },
        "evidence": {"primitive_refs": evidence},
    }


def _link_path_graph_to_groups(
    path_graph: dict[str, Any],
    groups: list[dict[str, Any]],
    raw_groups: list[dict[str, Any]],
) -> None:
    """Attach independently observed fragments to solved groups by provenance."""

    fragment_by_ref = {
        ref: item
        for item in path_graph.get("fragments", [])
        for ref in (item["primitive_ref"], *item.get("duplicate_primitive_refs", []))
    }
    component_by_fragment = {
        fragment_id: component
        for component in path_graph.get("components", [])
        for fragment_id in component.get("fragment_ids", [])
    }
    links = []
    resolved_components = set()
    for group, raw_group in zip(groups, raw_groups):
        metric = raw_group.get("dimension_anchored_solution") or {}
        refs = set(raw_group.get("primitive_refs", [])) | set(raw_group.get("section_boundary_refs", []))
        refs.update(metric.get("elevation_observations", {}).get("primitive_refs", []))
        fragments = [fragment_by_ref[ref] for ref in sorted(refs) if ref in fragment_by_ref]
        component_ids = sorted(
            {
                component_by_fragment[item["id"]]["id"]
                for item in fragments
                if item["id"] in component_by_fragment
            }
        )
        if not fragments:
            continue
        for fragment in fragments:
            fragment.setdefault("resolved_group_ids", []).append(group["id"])
        group["placement"]["path_component_ids"] = component_ids
        link = {
            "id": f"path_group_link.{len(links) + 1:04d}",
            "group_id": group["id"],
            "fragment_ids": [item["id"] for item in fragments],
            "component_ids": component_ids,
            "method": "exact_source_primitive_provenance",
            "state": "derived",
        }
        links.append(link)
        if metric.get("status") != "pass":
            continue
        selected_refs = set(metric.get("elevation_observations", {}).get("primitive_refs", []))
        installed = metric.get("solved_centerline", {})
        selected_components = {}
        for ref in selected_refs:
            fragment = fragment_by_ref.get(ref)
            component = component_by_fragment.get(fragment["id"]) if fragment else None
            if component is None:
                continue
            selected_components[component["id"]] = component
        quantity = int(group.get("quantity", {}).get("value") or 0)
        multiplicity = quantity // len(selected_components) if selected_components and quantity % len(selected_components) == 0 else None
        for component in selected_components.values():
            component["physical_path_state"] = "resolved_from_dimension_anchors_and_cross_view_validation"
            component["installed_centerline"] = {
                "value_mm": installed.get("installed_length_each_mm"),
                "raw_value_mm": installed.get("raw_installed_length_each_mm"),
                "status": "geometry_resolved",
                "group_id": group["id"],
                "physical_instance_multiplicity": multiplicity,
                "total_value_mm": (
                    None
                    if multiplicity is None or installed.get("installed_length_each_mm") is None
                    else float(installed["installed_length_each_mm"]) * multiplicity
                ),
                "basis": "exact primitive link to independently validated metric group",
            }
            resolved_components.add(component["id"])
    path_graph["group_links"] = links
    resolved_instances = sum(
        int(component.get("installed_centerline", {}).get("physical_instance_multiplicity") or 0)
        for component in path_graph.get("components", [])
        if component["id"] in resolved_components
    )
    path_graph["summary"]["installed_centerline_resolved_component_count"] = len(resolved_components)
    path_graph["summary"]["installed_centerline_resolved_count"] = resolved_instances
    enrich_projection_identities(path_graph)


def _group_geometry(
    page: fitz.Page,
    raw_group: dict[str, Any],
    program_group: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    if program_group["topology"]["family"]["value"] == "straight":
        primary_boxes = [fitz.Rect(box) for box in raw_group.get("projection_bboxes_display", [])]
    else:
        primary_boxes = _drawing_boxes(page, raw_group.get("section_boundary_refs", []))
    all_boxes = [
        *primary_boxes,
        *_drawing_boxes(page, raw_group.get("primitive_refs", [])),
    ]
    view_ids = sorted(
        {
            view_id
            for box in all_boxes
            if (view_id := _view_for_box(box, views, padding=12.0)) is not None
        }
    )
    return {
        "group_id": program_group["id"],
        "primary_boxes": primary_boxes,
        "all_boxes": all_boxes,
        "view_ids": view_ids,
        "topology": program_group["topology"]["family"]["value"],
        "count": program_group["quantity"]["value"],
    }


def _associate_identities(
    page: fitz.Page,
    hypotheses: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    raw_groups: list[dict[str, Any]],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not groups or not hypotheses:
        return []
    geometries = {
        group["id"]: _group_geometry(page, raw, group, views)
        for group, raw in zip(groups, raw_groups)
    }
    thin = extract_thin_segments(page)
    traces: dict[str, dict[str, Any]] = {}
    seeds: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        trace = trace_leader(fitz.Rect(hypothesis["bbox_display"]), thin, search_radius=150, max_hops=9)
        traces[hypothesis["id"]] = trace
        if not trace["segments"] or not trace["terminals"]:
            continue
        distances = {
            group_id: min(
                (_point_rect_distance(terminal, box) for terminal in trace["terminals"] for box in geometry["primary_boxes"]),
                default=math.inf,
            )
            for group_id, geometry in geometries.items()
        }
        # A direct seed requires a section-like cross-view context where at
        # least two group topologies are simultaneously testable. This keeps a
        # lone nearby tie line in an elevation from stealing a longitudinal mark.
        if sum(distance <= 30.0 for distance in distances.values()) < 2:
            continue
        hypothesis_view = _view_for_box(fitz.Rect(hypothesis["bbox_display"]), views, padding=12.0)
        ranked = []
        for group in groups:
            geometry = geometries[group["id"]]
            distance = distances[group["id"]]
            distance_factor = 0.40 if distance <= 8.0 else 0.28 if distance <= 15.0 else 0.15 if distance <= 30.0 else 0.0
            topology = geometry["topology"]
            topology_factor = 0.0
            if topology == "straight" and len(trace["terminals"]) >= min(4, geometry["count"]):
                topology_factor = 0.20
            elif topology == "closed_polyline" and len(trace["terminals"]) <= 2:
                topology_factor = 0.20
            factors = {
                "leader_chain": 0.15,
                "leader_terminal_to_group_geometry": distance_factor,
                "same_semantic_view": 0.15 if hypothesis_view in geometry["view_ids"] else 0.0,
                "cross_view_topology": topology_factor,
                "exclusive_identifier_role": 0.10,
            }
            ranked.append((sum(factors.values()), group, factors, distance))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best = ranked[0]
        margin = best[0] - ranked[1][0] if len(ranked) > 1 else best[0]
        if best[0] >= 0.80 and margin >= 0.15:
            seeds.append(
                {
                    "id": f"identity_association.{len(seeds) + 1:03d}",
                    "hypothesis_id": hypothesis["id"],
                    "mark": hypothesis["token"],
                    "group_id": best[1]["id"],
                    "status": "accepted",
                    "method": "direct_leader_terminal_and_cross_view_topology",
                    "score": round(best[0], 3),
                    "ambiguity_margin": round(margin, 3),
                    "terminal_distance_points": round(best[3], 3),
                    "factors": best[2],
                    "leader_trace": trace,
                }
            )

    token_targets: dict[str, set[str]] = {}
    for seed in seeds:
        token_targets.setdefault(seed["mark"], set()).add(seed["group_id"])
    unique_targets = {token: next(iter(targets)) for token, targets in token_targets.items() if len(targets) == 1}
    associations = list(seeds)
    associated_ids = {item["hypothesis_id"] for item in associations}
    for hypothesis in hypotheses:
        if hypothesis["id"] in associated_ids or hypothesis["token"] not in unique_targets:
            continue
        trace = traces[hypothesis["id"]]
        view_id = _view_for_box(fitz.Rect(hypothesis["bbox_display"]), views, padding=12.0)
        if not trace["segments"] or not trace["terminals"] or view_id is None:
            continue
        factors = {
            "unique_direct_mark_seed": 0.45,
            "connected_leader_chain": 0.20,
            "leader_terminal": 0.15,
            "recognized_semantic_view": 0.10,
            "exclusive_identifier_role": 0.10,
        }
        associations.append(
            {
                "id": f"identity_association.{len(associations) + 1:03d}",
                "hypothesis_id": hypothesis["id"],
                "mark": hypothesis["token"],
                "group_id": unique_targets[hypothesis["token"]],
                "status": "accepted",
                "method": "cross_view_propagation_from_unique_direct_mark_seed",
                "score": round(sum(factors.values()), 3),
                "ambiguity_margin": None,
                "terminal_distance_points": None,
                "factors": factors,
                "leader_trace": trace,
            }
        )
    return associations


def _associate_spacing(
    constraints: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    geometries: dict[str, dict[str, Any]],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    associations = []
    for constraint in constraints:
        constraint_view = _view_for_box(fitz.Rect(constraint["bbox_display"]), views, padding=18.0)
        ranked = []
        for group in groups:
            distribution = group["distribution"]
            spacings = {float(zone["spacing_mm"]) for zone in distribution.get("zones", [])}
            spacing_match = any(abs(float(constraint["spacing_mm"]) - value) <= max(2.0, 0.02 * value) for value in spacings)
            factors = {
                "arithmetic_pass": 0.20 if constraint["arithmetic"]["status"] == "pass" else 0.0,
                "piecewise_distribution_topology": 0.20 if distribution.get("type") == "piecewise_linear_array" else 0.0,
                "spacing_matches_resolved_zone": 0.35 if spacing_match else 0.0,
                "same_semantic_view": 0.25 if constraint_view in geometries[group["id"]]["view_ids"] else 0.0,
            }
            ranked.append((sum(factors.values()), group, factors))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            continue
        best = ranked[0]
        margin = best[0] - ranked[1][0] if len(ranked) > 1 else best[0]
        if best[0] >= 0.80 and margin >= 0.20:
            associations.append(
                {
                    "id": f"spacing_association.{len(associations) + 1:03d}",
                    "constraint_id": constraint["id"],
                    "group_id": best[1]["id"],
                    "status": "accepted",
                    "method": "view_containment_and_distribution_topology",
                    "score": round(best[0], 3),
                    "ambiguity_margin": round(margin, 3),
                    "view_id": constraint_view,
                    "factors": best[2],
                }
            )
    return associations


def _associate_mark_spacing_to_paths(
    constraints: list[dict[str, Any]],
    path_graph: dict[str, Any],
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind ``mark + spacing`` only to a unique repeated path in one view."""

    fragments = {item["id"]: item for item in path_graph.get("fragments", [])}
    existing_by_token: dict[str, set[str]] = {}
    for hypothesis in path_graph.get("mark_hypotheses", []):
        if hypothesis.get("state") == "accepted" and hypothesis.get("component_id"):
            existing_by_token.setdefault(str(hypothesis.get("token")), set()).add(hypothesis["component_id"])
    associations = []
    for constraint in constraints:
        mark = constraint.get("mark_token")
        if constraint.get("type") != "annotated_distribution_spacing" or not mark:
            continue
        targets = existing_by_token.get(str(mark), set())
        method = "same_mark_existing_unique_path_target"
        if len(targets) == 1:
            component_id = next(iter(targets))
            score, margin = 1.0, None
        else:
            constraint_box = fitz.Rect(constraint["bbox_display"])
            view_id = _view_for_box(constraint_box, views, padding=18.0)
            ranked = []
            for component in path_graph.get("components", []):
                if view_id is None or view_id not in component.get("view_ids", []):
                    continue
                repeated = any(
                    fragments.get(fragment_id, {}).get("repetition_group_ids")
                    for fragment_id in component.get("fragment_ids", [])
                )
                if not repeated or not component.get("bbox_display"):
                    continue
                ranked.append((_rect_distance(constraint_box, fitz.Rect(component["bbox_display"])), component["id"]))
            ranked.sort(key=lambda item: (item[0], item[1]))
            if not ranked:
                continue
            score_distance, component_id = ranked[0]
            margin = ranked[1][0] - score_distance if len(ranked) > 1 else math.inf
            view = next((item for item in views if item["id"] == view_id), None)
            view_diagonal = math.hypot(
                fitz.Rect(view["bbox_display"]).width,
                fitz.Rect(view["bbox_display"]).height,
            ) if view else math.hypot(595.0, 842.0)
            if score_distance > max(45.0, 0.12 * view_diagonal) or margin < max(8.0, 0.03 * view_diagonal):
                continue
            score = round(max(0.0, 1.0 - score_distance / max(view_diagonal, 1.0)), 3)
            method = "unique_same_view_repetition_component_by_containment_and_margin"
        association_id = f"spacing_path_association.{len(associations) + 1:04d}"
        association = {
            "id": association_id,
            "constraint_id": constraint["id"],
            "mark_token": str(mark),
            "component_id": component_id,
            "state": "accepted",
            "method": method,
            "score": score,
            "ambiguity_margin_points": None if margin is None or math.isinf(margin) else round(margin, 3),
            "schedule_values_used": False,
        }
        associations.append(association)
        constraint["association"] = {
            "projected_path_component_id": component_id,
            "state": "derived",
            "association_id": association_id,
        }
        component = next(item for item in path_graph.get("components", []) if item["id"] == component_id)
        component["mark_hypotheses"] = sorted(
            set(component.get("mark_hypotheses", [])) | {str(mark)},
            key=lambda value: (len(value), value),
        )
        path_graph.setdefault("mark_hypotheses", []).append(
            {
                "id": f"path_mark_hypothesis.spacing.{len(associations):05d}",
                "token": str(mark),
                "text_role_id": constraint["id"],
                "fragment_id": component.get("fragment_ids", [None])[0],
                "component_id": component_id,
                "state": "accepted",
                "source_occurrence_type": "mark_spacing_annotation",
                "uniqueness_basis": method,
                "leader_trace": {"segments": [], "terminals": [], "terminal_method": None},
                "count_spacing_observation": {
                    "count": None,
                    "spacing_mm": constraint["spacing_mm"],
                    "text_role_id": constraint["id"],
                },
            }
        )
    return associations


def build_rebar_program(
    page: fitz.Page,
    text_roles: list[dict[str, Any]],
    solved_geometry: dict[str, Any] | None,
    view_hypotheses: list[dict[str, Any]] | None = None,
    dimensions: Iterable[DimensionAttachment] | None = None,
    contours: list[dict[str, Any]] | None = None,
    metric_equation_graph: dict[str, Any] | None = None,
    section_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an independent symbolic reinforcement layer for every page."""

    spacing = [dict(item) for item in metric_equation_graph.get("constraints", [])] if metric_equation_graph is not None else []
    extracted_spacing = extract_spacing_constraints(page, text_roles)
    if metric_equation_graph is None:
        spacing.extend(extracted_spacing)
    else:
        spacing.extend(item for item in extracted_spacing if item["type"] == "annotated_distribution_spacing")
    dimension_rows = tuple(attach_dimensions(page) if dimensions is None else dimensions)
    views = view_hypotheses or []
    path_graph = build_rebar_path_graph(
        page,
        dimension_rows,
        views,
        contours or [],
        text_roles,
        section_observations,
    )
    spacing_path_associations = _associate_mark_spacing_to_paths(spacing, path_graph, views)
    identity_hypotheses = [
        {
            "id": f"rebar_identity_hypothesis.{index:03d}",
            "token": item["text"],
            "role": "rebar_mark_candidate",
            "state": "inferred",
            "confidence": item["confidence"],
            "basis": item["basis"],
            "bbox_display": item["bbox_display"],
            "target_group_id": None,
            "primitive_refs": item["primitive_refs"],
        }
        for index, item in enumerate(
            (item for item in text_roles if item["resolved_role"] == "identifier_candidate"),
            start=1,
        )
    ]
    reinforcement = (solved_geometry or {}).get("reinforcement_3d_input", {})
    scene_paths = reinforcement.get("centerline_scene", {}).get("paths", [])
    raw_groups = reinforcement.get("resolved_groups", [])
    groups = [
        _program_group(group, scene_paths, index)
        for index, group in enumerate(raw_groups, start=1)
    ]
    _link_path_graph_to_groups(path_graph, groups, raw_groups)
    geometries = {
        group["id"]: _group_geometry(page, raw, group, views)
        for group, raw in zip(groups, raw_groups)
    }
    identity_associations = _associate_identities(page, identity_hypotheses, groups, raw_groups, views)
    spacing_associations = _associate_spacing(spacing, groups, geometries, views)

    identity_by_hypothesis = {item["hypothesis_id"]: item for item in identity_associations}
    for hypothesis in identity_hypotheses:
        association = identity_by_hypothesis.get(hypothesis["id"])
        if association is not None:
            hypothesis["target_group_id"] = association["group_id"]
            hypothesis["association_state"] = "derived"
            hypothesis["association_id"] = association["id"]
        else:
            hypothesis["association_state"] = "unknown"

    spacing_by_constraint = {item["constraint_id"]: item for item in spacing_associations}
    for constraint in spacing:
        association = spacing_by_constraint.get(constraint["id"])
        if association is not None:
            constraint["association"] = {
                "rebar_group_id": association["group_id"],
                "state": "derived",
                "association_id": association["id"],
            }

    identity_conflicts = []
    for group in groups:
        candidates = [item for item in identity_associations if item["group_id"] == group["id"]]
        marks = sorted({item["mark"] for item in candidates})
        if len(marks) == 1:
            group["identity"]["mark"] = {
                "value": marks[0],
                "state": "derived",
                "evidence_refs": sorted(item["hypothesis_id"] for item in candidates),
            }
        elif len(marks) > 1:
            identity_conflicts.append(
                {"group_id": group["id"], "candidate_marks": marks, "association_ids": [item["id"] for item in candidates]}
            )
        source_constraint_ids = sorted(
            item["constraint_id"] for item in spacing_associations if item["group_id"] == group["id"]
        )
        if source_constraint_ids:
            group["distribution"]["source_constraint_ids"] = source_constraint_ids

    fabrication_details = attach_fabrication_details(page, groups, raw_groups)
    for group in groups:
        fabrication = group["fabrication"]
        group["lengths"]["fabrication_cutting"] = {
            "value_each_mm": fabrication.get("cutting_length_each_mm"),
            "value_total_mm": fabrication.get("cutting_length_total_mm"),
            "status": "fabrication_resolved" if fabrication["status"] == "resolved" else fabrication["status"],
            "basis": fabrication.get("basis"),
        }
    fabrication_states = [group["fabrication"]["status"] for group in groups]
    if fabrication_states and all(state == "resolved" for state in fabrication_states):
        fabrication_status = "resolved"
    elif any(state != "unresolved" for state in fabrication_states):
        fabrication_status = "partial"
    else:
        fabrication_status = "unavailable"

    assigned_spacing_ids = set(spacing_by_constraint)
    assigned_identity_ids = set(identity_by_hypothesis)
    unassigned_spacing = [item for item in spacing if item["id"] not in assigned_spacing_ids]
    unassigned_identities = [item for item in identity_hypotheses if item["id"] not in assigned_identity_ids]
    distribution_conflicts = [
        group["id"]
        for group in groups
        if group["distribution"].get("expansion_validation", {}).get("status") == "conflict"
    ]
    arithmetic_conflicts = [item["id"] for item in spacing if item.get("arithmetic", {}).get("status") == "conflict"]
    if groups:
        status = "partial_groups_resolved"
    elif spacing or identity_hypotheses or path_graph["fragments"]:
        status = "observations_only"
    else:
        status = "unresolved"
    metric_anchors = []
    for group in groups:
        solution = group.get("placement", {}).get("metric_solution") or {}
        for anchor in solution.get("anchors", []):
            metric_anchors.append(
                {
                    **anchor,
                    "id": f"{group['id']}.{anchor['id']}",
                    "source_anchor_id": anchor["id"],
                    "group_id": group["id"],
                }
            )
    return {
        "schema_version": "0.1.0",
        "layer": "rebar_program",
        "status": status,
        "groups": groups,
        "spacing_constraints": spacing,
        "identity_hypotheses": identity_hypotheses,
        "spacing_associations": spacing_associations,
        "spacing_path_associations": spacing_path_associations,
        "identity_associations": identity_associations,
        "fabrication_details": fabrication_details,
        "metric_anchors": metric_anchors,
        "physical_path_graph": path_graph,
        "unassigned_spacing_constraints": unassigned_spacing,
        "unassigned_identity_hypotheses": unassigned_identities,
        "validation": {
            "group_distribution_conflicts": distribution_conflicts,
            "spacing_arithmetic_conflicts": arithmetic_conflicts,
            "identity_assignment_conflicts": identity_conflicts,
            "association_counts": {
                "spacing_accepted": len(spacing_associations),
                "identity_accepted": len(identity_associations),
            },
            "false_confident_claim_policy": "conflicts remain review items and are never silently normalized",
        },
        "quantity_status": "partial" if groups else "unavailable",
        "fabrication_status": fabrication_status,
        "unknowns": [
            "unassigned mark-to-graphical-group identities" if unassigned_identities else "remaining explicit bar marks",
            "unassigned spacing-constraint-to-group associations" if unassigned_spacing else "remaining distribution notation",
            "fabrication bend and hook rules" if fabrication_status != "resolved" else "remaining fabrication detail families",
            "remaining unresolved cutting lengths and mass",
        ],
        "contract": {
            "requires_concrete_solid": False,
            "schedule_values_used": False,
            "geometry_distribution_fabrication_separated": True,
            "deterministic_distribution_expander": True,
            "associations_require_unique_score_margin": True,
            "schedule_lengths_used": False,
            "raster_details_preserve_image_provenance": True,
            "dimension_anchored_paths_require_reprojection": True,
            "generic_path_graph_precedes_object_solver": True,
            "visible_projection_is_not_physical_length": True,
        },
    }
