"""Fail-closed physical bar-family constraints and 3D reprojection.

The solver is drawing neutral: it consumes semantic view roles, mark/leader
links, object scopes, and projected path topology.  It never reads a printed
schedule and never treats a visible 2D path as a cutting length by itself.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from src.drawing_engine.project.placement_pair_review import classify_placement_pair_features


_COMPLEMENTARY = {
    frozenset(("axis_projection_1d", "axis_end_projection_0d")),
    frozenset(("axis_projection_1d", "planar_path_projection_2d")),
}


def _role(component: dict[str, Any], views: dict[str, dict[str, Any]]) -> str | None:
    roles = {
        views.get(view_id, {}).get("role_hypothesis")
        for view_id in component.get("view_ids", [])
    }
    if "reinforcement_view_candidate" in roles:
        return "elevation"
    if "section_view_candidate" in roles or "plan_view_candidate" in roles:
        return "section"
    return None


def _scope_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_scope = set(left.get("object_instance_ids", []))
    right_scope = set(right.get("object_instance_ids", []))
    if left_scope and right_scope and not left_scope & right_scope:
        return False
    left_coordinates = set(left.get("coordinate_scope_ids", []))
    right_coordinates = set(right.get("coordinate_scope_ids", []))
    return not left_coordinates or not right_coordinates or bool(left_coordinates & right_coordinates)


def _pair_score(mark: str, elevation: dict[str, Any], section: dict[str, Any]) -> tuple[float, list[str]]:
    evidence = ["same evidence-backed mark", "complementary semantic view roles"]
    dimensions = frozenset(
        (
            elevation.get("projection_dimensionality", {}).get("value"),
            section.get("projection_dimensionality", {}).get("value"),
        )
    )
    elevation_axis = elevation.get("object_axis_projection", {})
    section_axis = section.get("object_axis_projection", {})
    orthogonal_axis_pair = (
        dimensions == frozenset(("axis_projection_1d",))
        and elevation_axis.get("state") == "resolved_relative"
        and section_axis.get("state") == "resolved_relative"
        and elevation_axis.get("axis") != section_axis.get("axis")
        and elevation_axis.get("station_axis") == section_axis.get("station_axis")
    )
    if dimensions not in _COMPLEMENTARY and not orthogonal_axis_pair:
        return -math.inf, evidence
    if mark not in elevation.get("mark_hypotheses", []) or mark not in section.get("mark_hypotheses", []):
        return -math.inf, evidence
    if not _scope_compatible(elevation, section):
        return -math.inf, evidence
    evidence.extend((
        (
            "orthogonal one-dimensional projections share one physical station axis"
            if orthogonal_axis_pair
            else "complementary projection dimensionality"
        ),
        "compatible object-instance scope",
    ))
    score = 4.0
    elevation_coordinates = set(elevation.get("coordinate_scope_ids", []))
    section_coordinates = set(section.get("coordinate_scope_ids", []))
    if elevation_coordinates and section_coordinates and elevation_coordinates & section_coordinates:
        score += 0.75
        evidence.append("shared cross-view coordinate scope")
    if elevation.get("mark_hypotheses") == [mark]:
        score += 1.0
        evidence.append("exclusive elevation mark")
    if section.get("mark_hypotheses") == [mark]:
        score += 1.0
        evidence.append("exclusive section mark")
    if elevation.get("projected_length", {}).get("value_mm") is not None:
        score += 0.5
        evidence.append("metric elevation projection")
    if section.get("projection_dimensionality", {}).get("value") == "axis_end_projection_0d":
        score += 0.5
        evidence.append("section observes an axis end")
    if orthogonal_axis_pair:
        score += 0.5
        evidence.append("view frames resolve different projected object axes")
    return score, evidence


def _resolved_group_certificates(
    groups: list[dict[str, Any]] | None,
    path_graph: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index drawing-derived groups whose identity and count already close.

    The symbolic rebar program can establish a stronger certificate than a
    later visual similarity match: exact primitive provenance, a leader-bound
    mark, a validated metric/repetition solution, and cross-view reprojection.
    Preserve and reuse that certificate instead of making the family solver
    rediscover the same relation from component boxes.
    """

    accepted_projection_groups = {
        str(item.get("group_id"))
        for item in path_graph.get("cross_view_projection_identities", [])
        if item.get("state") == "accepted" and item.get("group_id")
    }
    result: dict[str, dict[str, Any]] = {}
    for group in groups or []:
        mark_claim = group.get("identity", {}).get("mark", {})
        mark = mark_claim.get("value")
        quantity = group.get("quantity", {})
        distribution_validation = group.get("distribution", {}).get("expansion_validation", {})
        metric = group.get("placement", {}).get("metric_solution") or {}
        metric_validation = metric.get("validation", {})
        metric_cross_view = (
            metric.get("status") == "pass"
            and metric_validation.get("status") == "pass"
            and metric.get("section_validation", {}).get("status") == "pass"
            and metric.get("reprojection", {}).get("supporting_primitive_count", 0) > 0
        )
        repeated_cross_view = (
            str(group.get("id")) in accepted_projection_groups
            and distribution_validation.get("status") == "pass"
        )
        checks = {
            "leader_bound_unique_mark": mark is not None and mark_claim.get("state") in {"direct", "observed", "derived"},
            "derived_positive_count": quantity.get("state") == "derived" and isinstance(quantity.get("value"), int) and quantity["value"] > 0,
            "path_provenance_present": bool(group.get("placement", {}).get("path_component_ids")),
            "cross_view_metric_or_repetition_closure": metric_cross_view or repeated_cross_view,
        }
        if not all(checks.values()):
            continue
        key = str(mark)
        if key in result:
            # One printed mark mapping to two independent groups is not a
            # unique physical-family identity.
            result.pop(key, None)
            continue
        result[key] = {
            "status": "pass",
            "group_id": group["id"],
            "mark": key,
            "component_ids": sorted(group["placement"]["path_component_ids"]),
            "count": int(quantity["value"]),
            "checks": checks,
            "basis": (
                "leader-bound mark plus dimension-anchored cross-view reprojection"
                if metric_cross_view
                else "leader-bound mark plus exact group provenance and validated repeated distribution"
            ),
            "evidence_refs": sorted(mark_claim.get("evidence_refs", [])),
            "schedule_values_used": False,
        }
    return result


def _multiplicity(mark: str, sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Count only exclusive, independently connected axis-end observations."""

    exact = [
        item
        for item in sections
        if item.get("mark_hypotheses") == [mark]
        and item.get("projection_dimensionality", {}).get("value") == "axis_end_projection_0d"
        and item.get("physical_path_state") == "candidate_nonbranching_projection"
    ]
    by_view: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in exact:
        if len(item.get("view_ids", [])) == 1:
            by_view[item["view_ids"][0]].append(item)
    counts = [len(rows) for rows in by_view.values() if rows]
    if not counts:
        return {"state": "unknown", "value": None, "reason": "no exclusive axis-end observations"}
    consensus = Counter(counts).most_common()
    if len(consensus) > 1 and consensus[0][1] == consensus[1][1]:
        return {"state": "unknown", "value": None, "reason": "section multiplicities disagree without a unique consensus"}
    value, support = consensus[0]
    return {
        "state": "derived",
        "value": value,
        "basis": "count of exclusive, nonbranching axis-end projections in a section; repeated sections must agree",
        "supporting_view_count": support,
        "component_ids": sorted(item["id"] for rows in by_view.values() if len(rows) == value for item in rows),
    }


def _paired_outline_centers(candidates: list[dict[str, Any]]) -> list[list[float]]:
    """Collapse two nearby parallel contour boundaries into one bar axis."""

    rows = []
    for item in candidates:
        box = item.get("bbox_display") or []
        if len(box) != 4:
            continue
        width, height = box[2] - box[0], box[3] - box[1]
        if max(width, height) < 8.0 or max(width, height) / max(min(width, height), 0.5) < 6.0:
            continue
        rows.append({"item": item, "box": box, "vertical": height >= width, "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
    pairs = []
    used = set()
    for index, left in enumerate(rows):
        if index in used:
            continue
        matches = []
        for other_index, right in enumerate(rows[index + 1 :], start=index + 1):
            if other_index in used or left["vertical"] != right["vertical"]:
                continue
            left_box, right_box = left["box"], right["box"]
            if left["vertical"]:
                overlap = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
                span = min(left_box[3] - left_box[1], right_box[3] - right_box[1])
                separation = abs(left["center"][0] - right["center"][0])
            else:
                overlap = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
                span = min(left_box[2] - left_box[0], right_box[2] - right_box[0])
                separation = abs(left["center"][1] - right["center"][1])
            if span > 0 and overlap / span >= 0.80 and 0.5 <= separation <= 8.0:
                matches.append((separation, other_index, right))
        if not matches:
            continue
        _, other_index, right = min(matches)
        used.update((index, other_index))
        pairs.append([(left["center"][0] + right["center"][0]) / 2, (left["center"][1] + right["center"][1]) / 2])
    return pairs


def _observed_section_multiplicity(
    mark: str,
    observations: dict[str, Any] | None,
    views: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for section in (observations or {}).get("sections", []):
        exact = [item for item in section.get("candidates", []) if item.get("leader_marks") == [mark]]
        filled = [item for item in exact if item.get("kind") == "filled_end_projection" and item.get("state") == "resolved"]
        centers = [
            [(item["bbox_display"][0] + item["bbox_display"][2]) / 2, (item["bbox_display"][1] + item["bbox_display"][3]) / 2]
            for item in filled
        ]
        basis = "exclusive filled axis-end projections"
        if not centers:
            centers = _paired_outline_centers(exact)
            basis = "exclusive leader-linked paired contour boundaries"
        if centers:
            candidates.append((section, centers, basis))
    if not candidates:
        return None
    counts = Counter(len(centers) for _, centers, _ in candidates)
    # Different sections can expose only part of a family.  A majority is not
    # a completeness proof, so conflicting non-zero counts must abstain.
    if len(counts) > 1:
        return None
    value, support = counts.most_common(1)[0]
    section, centers, basis = next(item for item in candidates if len(item[1]) == value)
    normalized = []
    for center in centers:
        containing = [
            view
            for view in views.values()
            if view.get("role_hypothesis") in {"section_view_candidate", "plan_view_candidate"}
            and view["bbox_display"][0] <= center[0] <= view["bbox_display"][2]
            and view["bbox_display"][1] <= center[1] <= view["bbox_display"][3]
        ]
        if not containing:
            return None
        view = min(containing, key=lambda item: (item["bbox_display"][2] - item["bbox_display"][0]) * (item["bbox_display"][3] - item["bbox_display"][1]))
        box = view["bbox_display"]
        normalized.append([(center[0] - box[0]) / (box[2] - box[0]), (center[1] - box[1]) / (box[3] - box[1])])
    return {
        "state": "derived",
        "value": value,
        "basis": f"{basis}; repeated section counts require consensus",
        "supporting_view_count": support,
        "section_id": section["section_id"],
        "candidate_ids": [item["candidate_id"] for item in section.get("candidates", []) if item.get("leader_marks") == [mark]],
        "normalized_centers": normalized,
        "display_centers": centers,
        "host_bbox_display": section.get("host_bbox_display"),
    }


def _fold_detail_legs_into_physical_bars(
    multiplicity: dict[str, Any],
    family: dict[str, Any],
    details_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Map paired outline legs to instances only when detail topology proves it."""

    if (
        multiplicity.get("state") != "derived"
        or not str(multiplicity.get("basis") or "").startswith(
            "exclusive leader-linked paired contour boundaries"
        )
    ):
        return multiplicity
    details = [
        details_by_id[detail_id]
        for detail_id in family.get("detail_ids", []) or []
        if detail_id in details_by_id
    ]
    if not details:
        return multiplicity
    solutions = [detail.get("fabrication_geometry_solution", {}) for detail in details]
    two_leg_topology_closed = all(
        solution.get("topology") == "open_rectangular_loop_with_two_diagonal_hooks"
        and solution.get("closure_validation", {}).get("topology_closed") is True
        for solution in solutions
    )
    if not two_leg_topology_closed:
        return multiplicity
    observed_leg_count = int(multiplicity.get("value") or 0)
    if observed_leg_count <= 0 or observed_leg_count % 2:
        return {
            **multiplicity,
            "state": "unknown",
            "value": None,
            "projected_leg_observation_count": observed_leg_count,
            "projected_legs_per_physical_bar": 2,
            "reason": "closed two-leg fabrication topology conflicts with an odd projected-leg count",
        }
    return {
        **multiplicity,
        "value": observed_leg_count // 2,
        "projected_leg_observation_count": observed_leg_count,
        "projected_legs_per_physical_bar": 2,
        "detail_ids": sorted(str(detail["id"]) for detail in details),
        "basis": (
            f"{multiplicity['basis']}; closed fabrication detail topology "
            "maps two projected legs to one physical bar"
        ),
    }


def _station_fit(rows: list[tuple[str, float, float]]) -> dict[str, Any] | None:
    if len(rows) < 4:
        return None
    xs = [item[1] for item in rows]
    ys = [item[2] for item in rows]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    variance = sum((value - mean_x) ** 2 for value in xs)
    if variance <= 1e-9:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    intercept = mean_y - slope * mean_x
    residuals = [abs((slope * x + intercept) - y) for x, y in zip(xs, ys)]
    y_variance = sum((value - mean_y) ** 2 for value in ys)
    explained = sum(((slope * x + intercept) - mean_y) ** 2 for x in xs)
    r_squared = 0.0 if y_variance <= 1e-9 else min(1.0, explained / y_variance)
    return {
        "marks": [item[0] for item in rows],
        "slope": slope,
        "intercept": intercept,
        "maximum_normalized_residual": max(residuals),
        "r_squared": r_squared,
    }


def _resolve_mark_station_axes(
    native_details: dict[str, Any],
    components: dict[str, dict[str, Any]],
    views: dict[str, dict[str, Any]],
    path_graph: dict[str, Any],
) -> None:
    """Resolve a section axis when repeated exact marks prove one affine station order."""

    by_pair: defaultdict[tuple[str, str], list[tuple[str, dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    for family in native_details.get("physical_families", []):
        mark = str(family.get("mark"))
        rows = [components[item] for item in family.get("component_ids", []) if item in components]
        elevations = [
            item for item in rows
            if _role(item, views) == "elevation"
            and item.get("mark_hypotheses") == [mark]
            and item.get("object_axis_projection", {}).get("state") == "resolved_relative"
        ]
        sections = [
            item for item in rows
            if _role(item, views) == "section"
            and item.get("mark_hypotheses") == [mark]
            and item.get("projection_dimensionality", {}).get("value") == "axis_projection_1d"
        ]
        if len(elevations) != 1 or not sections or len(elevations[0].get("view_ids", [])) != 1:
            continue
        for section_view_id in sorted({view_id for item in sections for view_id in item.get("view_ids", [])}):
            section_rows = [item for item in sections if item.get("view_ids") == [section_view_id]]
            if section_rows:
                by_pair[(elevations[0]["view_ids"][0], section_view_id)].append((mark, elevations[0], section_rows))

    certificates = []
    for (elevation_view_id, section_view_id), families in sorted(by_pair.items()):
        elevation_view = views.get(elevation_view_id, {})
        section_view = views.get(section_view_id, {})
        elevation_box = elevation_view.get("bbox_display") or []
        section_box = section_view.get("bbox_display") or []
        if len(elevation_box) != 4 or len(section_box) != 4:
            continue
        branches = []
        for branch in ("minimum", "maximum"):
            observations = []
            for mark, elevation, sections in families:
                elevation_station = ((elevation["bbox_display"][0] + elevation["bbox_display"][2]) / 2 - elevation_box[0]) / max(elevation_box[2] - elevation_box[0], 1e-9)
                section_stations = sorted(
                    ((item["bbox_display"][0] + item["bbox_display"][2]) / 2 - section_box[0]) / max(section_box[2] - section_box[0], 1e-9)
                    for item in sections
                )
                observations.append((mark, section_stations[0 if branch == "minimum" else -1], elevation_station))
            fit = _station_fit(observations)
            if fit is not None:
                branches.append((fit["maximum_normalized_residual"], -fit["r_squared"], branch, fit))
        if not branches:
            continue
        _, _, branch, fit = min(branches)
        if fit["r_squared"] < 0.98 or fit["maximum_normalized_residual"] > 0.04 or abs(fit["slope"]) < 0.20:
            continue
        parent_axis = families[0][1].get("object_axis_projection", {})
        if not parent_axis.get("station_axis") or not parent_axis.get("view_normal_axis") or not parent_axis.get("axis"):
            continue
        certificate_id = f"mark_station_axis_certificate.{len(certificates) + 1:03d}"
        certificate = {
            "id": certificate_id,
            "state": "accepted",
            "elevation_view_id": elevation_view_id,
            "section_view_id": section_view_id,
            "section_display_u_axis": parent_axis["station_axis"],
            "section_display_v_axis": parent_axis["view_normal_axis"],
            "section_normal_axis": parent_axis["axis"],
            "selected_branch": branch,
            "marks": fit["marks"],
            "affine_fit": {
                "slope": round(fit["slope"], 6),
                "intercept": round(fit["intercept"], 6),
                "r_squared": round(fit["r_squared"], 6),
                "maximum_normalized_residual": round(fit["maximum_normalized_residual"], 6),
            },
            "basis": "four or more exclusive cross-view marks share one unique affine station order",
            "schedule_values_used": False,
        }
        certificates.append(certificate)
        for _, _, sections in families:
            for section in sections:
                section["object_axis_projection"] = {
                    "state": "resolved_relative",
                    "axis": parent_axis["view_normal_axis"],
                    "station_axis": parent_axis["station_axis"],
                    "view_normal_axis": parent_axis["axis"],
                    "display_axis": "v",
                    "certificate_id": certificate_id,
                    "absolute_axis_sign_state": "unresolved",
                }
    path_graph["mark_station_axis_certificates"] = certificates


def _mirror_equivalence(multiplicity: dict[str, Any]) -> dict[str, Any]:
    """Prove when unresolved axis sign only permutes identical instances."""

    host = multiplicity.get("host_bbox_display") or []
    centers = multiplicity.get("display_centers") or []
    if len(host) != 4 or host[2] <= host[0] or len(centers) < 2:
        return {"status": "unresolved", "reason": "no complete multi-instance section coordinate set"}
    coordinates = sorted((float(point[0]) - host[0]) / (host[2] - host[0]) for point in centers)
    mirrored = sorted(1.0 - value for value in coordinates)
    residual = max(abs(left - right) for left, right in zip(coordinates, mirrored))
    return {
        "status": "pass" if residual <= 0.02 else "unresolved",
        "set_invariant_under_axis_reflection": residual <= 0.02,
        "normalized_station_coordinates": [round(value, 6) for value in coordinates],
        "maximum_reflection_residual": round(residual, 6),
        "basis": "the unresolved axis sign only permutes same-mark section instances",
    }


def _bootstrap_projection_mark_families(
    native_details: dict[str, Any],
    path_graph: dict[str, Any],
    views: dict[str, dict[str, Any]],
) -> None:
    """Seed physical families from accepted placed-view marks.

    Fabrication details refine a physical family; they are not a prerequisite
    for recording a leader-bound mark on a projected path.  Only marks backed
    by an accepted leader trace or an exact section terminal participate.
    """

    families = native_details.setdefault("physical_families", [])
    existing_marks = {str(item.get("mark")) for item in families}
    fragments = {item["id"]: item for item in path_graph.get("fragments", [])}
    accepted_hypotheses = {
        item["id"]: item
        for item in path_graph.get("mark_hypotheses", [])
        if item.get("state") == "accepted"
    }
    evidence_by_component: defaultdict[str, set[str]] = defaultdict(set)
    hypotheses_by_component: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for hypothesis_id, hypothesis in accepted_hypotheses.items():
        component_id = hypothesis.get("component_id")
        if component_id:
            evidence_by_component[component_id].add(hypothesis_id)
            hypotheses_by_component[component_id].append(hypothesis)
    for component in path_graph.get("components", []):
        if any(
            fragments.get(fragment_id, {}).get("mark_assignment_basis")
            == "exclusive section mark plus first connected heavy-geometry leader contact"
            for fragment_id in component.get("fragment_ids", [])
        ):
            evidence_by_component[component["id"]].update(
                fragments[fragment_id].get("primitive_ref", fragment_id)
                for fragment_id in component.get("fragment_ids", [])
                if fragments.get(fragment_id, {}).get("mark_assignment_basis")
            )

    by_mark: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in path_graph.get("components", []):
        marks = component.get("mark_hypotheses", [])
        if len(marks) != 1 or component["id"] not in evidence_by_component:
            continue
        by_mark[str(marks[0])].append(component)

    for mark, rows in sorted(by_mark.items(), key=lambda item: (len(item[0]), item[0])):
        if mark in existing_marks:
            continue
        view_ids = sorted({view_id for row in rows for view_id in row.get("view_ids", [])})
        elevation_components = sorted(row["id"] for row in rows if _role(row, views) == "elevation")
        section_components = sorted(row["id"] for row in rows if _role(row, views) == "section")
        mark_hypotheses = [
            hypothesis
            for row in rows
            for hypothesis in hypotheses_by_component.get(row["id"], [])
            if str(hypothesis.get("token")) == mark
        ]
        semantic_values = [
            item["semantic_value"] for item in mark_hypotheses if item.get("semantic_value")
        ]
        unique_semantic_values = {
            (int(item["diameter_mm"]), int(item["length_code"])) for item in semantic_values
        }
        component_by_id = {row["id"]: row for row in rows}
        count_spacing_observations = []
        for item in mark_hypotheses:
            observation = item.get("count_spacing_observation")
            component = component_by_id.get(item.get("component_id"))
            if not observation or component is None:
                continue
            dimensionality = component.get("projection_dimensionality") or {}
            count_spacing_observations.append(
                {
                    **observation,
                    "hypothesis_id": item["id"],
                    "target_component_id": component["id"],
                    "view_ids": sorted(component.get("view_ids", [])),
                    "object_ids": sorted(component.get("object_instance_ids", [])),
                    "coordinate_scope_ids": sorted(component.get("coordinate_scope_ids", [])),
                    "source_fragment_ids": sorted(component.get("fragment_ids", [])),
                    "target_bbox_display": component.get("bbox_display"),
                    "normalized_center_in_view": dimensionality.get("normalized_center_in_view"),
                }
            )
        ambiguous_occurrence_count = sum(
            item.get("state") != "accepted" and str(item.get("token")) == mark
            for item in path_graph.get("mark_hypotheses", [])
        )
        families.append(
            {
                "id": f"physical_bar_family.{len(families) + 1:03d}",
                "mark": mark,
                "state": "derived_cross_view_family" if len(view_ids) >= 2 else "single_view_family_candidate",
                "family_origin": "evidence_backed_placed_projection_mark",
                "detail_ids": [],
                "placement_association_ids": [],
                "fragment_ids": sorted({fragment_id for row in rows for fragment_id in row.get("fragment_ids", [])}),
                "component_ids": sorted(row["id"] for row in rows),
                "view_ids": view_ids,
                "section_component_ids": section_components,
                "section_candidate_ids": [],
                "elevation_component_ids": elevation_components,
                "fabrication_dimensions": [],
                "raster_fabrication_dimension_observations": [],
                "identity_evidence_refs": sorted(
                    {ref for row in rows for ref in evidence_by_component[row["id"]]}
                ),
                "compound_callout_observation": (
                    {
                        "diameter_mm": next(iter(unique_semantic_values))[0],
                        "length_code": next(iter(unique_semantic_values))[1],
                        "state": "observed",
                        "basis": "native compound rebar callout outside a table grid",
                        "evidence_refs": sorted(item["id"] for item in mark_hypotheses),
                    }
                    if len(unique_semantic_values) == 1
                    else None
                ),
                "callout_count_spacing_observations": count_spacing_observations,
                "ambiguous_mark_hypothesis_count": ambiguous_occurrence_count,
                "link_chain": [],
                "schedule_values_used": False,
            }
        )
        existing_marks.add(mark)


def _callout_multiplicity(family: dict[str, Any]) -> dict[str, Any] | None:
    observations = {
        str(item.get("text_role_id")): item
        for item in family.get("callout_count_spacing_observations", [])
        if item.get("count") is not None
    }
    if not observations:
        return None
    rows = list(observations.values())
    if len(rows) != 1:
        parents = list(range(len(rows)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        pair_certificates = []
        unresolved = []
        for left_index, left in enumerate(rows):
            for right_index in range(left_index + 1, len(rows)):
                right = rows[right_index]
                left_box, right_box = left.get("target_bbox_display"), right.get("target_bbox_display")
                bbox_iou = None
                if left_box and right_box:
                    lx0, ly0, lx1, ly1 = map(float, left_box)
                    rx0, ry0, rx1, ry1 = map(float, right_box)
                    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
                        0.0, min(ly1, ry1) - max(ly0, ry0)
                    )
                    union_area = (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection
                    bbox_iou = round(intersection / union_area, 6) if union_area > 0 else 0.0
                left_center = left.get("normalized_center_in_view")
                right_center = right.get("normalized_center_in_view")
                center_distance = (
                    math.dist(left_center[:2], right_center[:2])
                    if isinstance(left_center, list)
                    and isinstance(right_center, list)
                    and len(left_center) >= 2
                    and len(right_center) >= 2
                    else None
                )
                left_views, right_views = set(left.get("view_ids", [])), set(right.get("view_ids", []))
                left_objects, right_objects = set(left.get("object_ids", [])), set(right.get("object_ids", []))
                left_fragments = set(left.get("source_fragment_ids", []))
                right_fragments = set(right.get("source_fragment_ids", []))
                result = classify_placement_pair_features(
                    {
                        "same_page": True,
                        "same_target_id": left.get("target_component_id") == right.get("target_component_id"),
                        "left_view_count": len(left_views),
                        "right_view_count": len(right_views),
                        "view_id_jaccard": len(left_views & right_views) / len(left_views | right_views) if left_views | right_views else None,
                        "left_object_count": len(left_objects),
                        "right_object_count": len(right_objects),
                        "object_id_jaccard": len(left_objects & right_objects) / len(left_objects | right_objects) if left_objects | right_objects else None,
                        "source_fragment_jaccard": len(left_fragments & right_fragments) / len(left_fragments | right_fragments) if left_fragments | right_fragments else None,
                        "target_bbox_iou": bbox_iou,
                        "normalized_center_distance": center_distance,
                    }
                )
                pair_certificates.append(
                    {
                        "left_hypothesis_id": left.get("hypothesis_id"),
                        "right_hypothesis_id": right.get("hypothesis_id"),
                        **result,
                    }
                )
                if result["outcome"] == "duplicate_projection":
                    if int(left["count"]) != int(right["count"]):
                        unresolved.append("duplicate target has contradictory count annotations")
                    else:
                        union(left_index, right_index)
                elif result["outcome"] != "additive_placements":
                    unresolved.append(result.get("reason") or result["outcome"])
        if unresolved:
            return {
                "state": "unknown",
                "value": None,
                "reason": "; ".join(sorted(set(unresolved))),
                "observations": rows,
                "placement_pair_certificates": pair_certificates,
            }
        duplicate_classes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(rows):
            duplicate_classes[find(index)].append(row)
        partial = int(family.get("ambiguous_mark_hypothesis_count", 0)) > 0
        return {
            "state": "derived_partial" if partial else "derived",
            "value": sum(int(class_rows[0]["count"]) for class_rows in duplicate_classes.values()),
            "basis": "sum of additive same-view placements after duplicate-target collapse",
            "observations": rows,
            "placement_pair_certificates": pair_certificates,
        }
    observation = rows[0]
    partial = int(family.get("ambiguous_mark_hypothesis_count", 0)) > 0
    return {
        "state": "derived_partial" if partial else "derived",
        "value": int(observation["count"]),
        "basis": (
            "resolved count field in one paired native callout; other same-mark placements remain ambiguous"
            if partial
            else "count field in one uniquely placed native compound rebar callout"
        ),
        "observations": [observation],
    }


def resolve_physical_bar_families(
    native_details: dict[str, Any],
    path_graph: dict[str, Any],
    views: list[dict[str, Any]],
    section_observations: dict[str, Any] | None = None,
    groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve unique section/elevation pairs before any family-aware 3D step."""

    components = {item["id"]: item for item in path_graph.get("components", [])}
    view_by_id = {item["id"]: item for item in views}
    _bootstrap_projection_mark_families(native_details, path_graph, view_by_id)
    _resolve_mark_station_axes(native_details, components, view_by_id, path_graph)
    group_certificates = _resolved_group_certificates(groups, path_graph)
    details_by_id = {
        str(detail["id"]): detail
        for detail in native_details.get("details", []) or []
        if detail.get("id")
    }
    resolved = 0
    for family in native_details.get("physical_families", []):
        mark = str(family["mark"])
        certificate = group_certificates.get(mark)
        if certificate:
            family["component_ids"] = sorted(
                set(family.get("component_ids", [])) | set(certificate["component_ids"])
            )
            family["source_group_id"] = certificate["group_id"]
            family["group_constraint_certificate"] = certificate
        rows = [components[item] for item in family.get("component_ids", []) if item in components]
        elevations = [item for item in rows if _role(item, view_by_id) == "elevation"]
        sections = [item for item in rows if _role(item, view_by_id) == "section"]
        ranked = []
        for elevation in elevations:
            for section in sections:
                score, evidence = _pair_score(mark, elevation, section)
                if math.isfinite(score):
                    ranked.append((score, elevation, section, evidence))
        ranked.sort(key=lambda item: (item[0], item[1]["id"], item[2]["id"]), reverse=True)
        family["projection_matches"] = [
            {
                "elevation_component_id": elevation["id"],
                "section_component_id": section["id"],
                "score": score,
                "state": "candidate",
                "basis": evidence,
            }
            for score, elevation, section, evidence in ranked[:12]
        ]
        multiplicity = (
            _observed_section_multiplicity(mark, section_observations, view_by_id)
            or _callout_multiplicity(family)
            or _multiplicity(mark, sections)
        )
        multiplicity = _fold_detail_legs_into_physical_bars(
            multiplicity,
            family,
            details_by_id,
        )
        family["mirror_equivalence"] = _mirror_equivalence(multiplicity)
        if certificate:
            multiplicity = {
                "state": "derived",
                "value": certificate["count"],
                "basis": certificate["basis"],
                "component_ids": certificate["component_ids"],
                "evidence_refs": certificate["evidence_refs"],
            }
        family["multiplicity"] = multiplicity
        family["count_state"] = multiplicity["state"]
        family["count"] = multiplicity["value"]
        if certificate and ranked:
            family["projection_matches"][0]["state"] = "accepted"
            family["accepted_projection_match"] = {
                **family["projection_matches"][0],
                "section_component_ids": multiplicity.get("component_ids", []),
                "acceptance_certificate": certificate["group_id"],
            }
            family["constraint_status"] = "resolved"
            family["constraint_reason"] = None
            resolved += 1
            continue
        if certificate:
            # Some dimension-anchored solvers validate section coordinates
            # directly from native primitives rather than materialising a
            # separate 0D path component.  The family identity/count is still
            # closed, while 3D lifting correctly remains gated on an explicit
            # projection pair.
            family["constraint_status"] = "resolved"
            family["constraint_reason"] = None
            family["scene_status"] = "projection_pair_not_materialised"
            resolved += 1
            continue
        if not ranked:
            family["constraint_status"] = "unresolved"
            family["constraint_reason"] = "no complementary section/elevation pair"
            continue
        best = ranked[0]
        elevation_scores: dict[str, float] = {}
        for score, elevation, _, _ in ranked:
            elevation_scores[elevation["id"]] = max(score, elevation_scores.get(elevation["id"], -math.inf))
        ordered_elevations = sorted(elevation_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        runner_up = ordered_elevations[1][1] if len(ordered_elevations) > 1 else -math.inf
        if ordered_elevations[0][1] - runner_up < 1.0:
            family["constraint_status"] = "ambiguous"
            family["constraint_reason"] = "elevation family has no unique cross-view score margin"
            continue
        if multiplicity["state"] != "derived":
            family["constraint_status"] = "partial"
            family["constraint_reason"] = "identity is unique but physical multiplicity is unresolved"
            continue
        family["projection_matches"][0]["state"] = "accepted"
        family["accepted_projection_match"] = {
            **family["projection_matches"][0],
            "section_component_ids": multiplicity.get("component_ids", []),
        }
        family["constraint_status"] = "resolved"
        family["constraint_reason"] = None
        resolved += 1
    family_count = len(native_details.get("physical_families", []))
    native_details["family_constraint_validation"] = {
        "status": "pass" if family_count and resolved == family_count else "partial",
        "resolved_family_count": resolved,
        "family_count": family_count,
        "ordering": "placed-view mark, optional detail refinement, object scope, cross-view projection, multiplicity, then 3D",
        "schedule_values_used": False,
        "resolved_group_certificate_count": len(group_certificates),
    }
    return native_details["family_constraint_validation"]


def summarize_group_quantities(
    groups: list[dict[str, Any]],
    native_details: dict[str, Any],
) -> dict[str, Any] | None:
    """Expose closed per-family facts without manufacturing a full takeoff."""

    family_by_group = {
        family.get("source_group_id"): family
        for family in native_details.get("physical_families", [])
        if family.get("source_group_id")
    }
    rows = []
    for group in groups:
        family = family_by_group.get(group.get("id"))
        if not family or family.get("constraint_status") != "resolved":
            continue
        fabrication = group.get("fabrication", {})
        diameter = group.get("bar_spec", {}).get("diameter_mm", {})
        grade = group.get("bar_spec", {}).get("steel_grade", {})
        rows.append(
            {
                "physical_family_id": family["id"],
                "group_id": group["id"],
                "mark": str(family["mark"]),
                "count": int(group["quantity"]["value"]),
                "count_state": group["quantity"]["state"],
                "fabrication_length_each_mm": fabrication.get("cutting_length_each_mm"),
                "fabrication_length_total_mm": fabrication.get("cutting_length_total_mm"),
                "fabrication_state": fabrication.get("status", "unresolved"),
                "diameter_mm": diameter.get("value"),
                "diameter_state": diameter.get("state", "unknown"),
                "steel_grade": grade.get("value"),
                "steel_grade_state": grade.get("state", "unknown"),
                "mass_kg": None,
                "mass_state": "unknown",
                "mass_reason": "diameter and steel grade must be explicitly confirmed before mass aggregation",
                "evidence_refs": [family["id"], group["id"]],
            }
        )
    if not rows:
        return None
    all_groups_closed = len(rows) == len(groups)
    all_lengths_closed = all(item["fabrication_length_total_mm"] is not None for item in rows)
    resolved_length_mm = sum(
        float(item["fabrication_length_total_mm"])
        for item in rows
        if item["fabrication_length_total_mm"] is not None
    )
    return {
        "status": "partial",
        "takeoff_kind": "partial_drawing_takeoff",
        "state": "derived_partial",
        "families": rows,
        "physical_bar_count": sum(item["count"] for item in rows) if all_groups_closed else None,
        "resolved_physical_bar_count": sum(item["count"] for item in rows),
        "total_placed_centerline_m": None,
        "fabrication_length_m": round(resolved_length_mm / 1000.0, 6) if all_lengths_closed else None,
        "resolved_fabrication_length_m": round(resolved_length_mm / 1000.0, 6),
        "mass_kg": None,
        "reason": "at least one family still lacks a fabrication length, explicit diameter, or steel grade",
        "validation": {
            "all_physical_families_closed": all_groups_closed,
            "all_fabrication_lengths_closed": all_lengths_closed,
            "mass_inputs_closed": False,
            "schedule_values_used": False,
        },
        "schedule_values_used": False,
    }


def _mesh_bounds(mesh: dict[str, Any] | None) -> tuple[list[float], list[float]] | None:
    vertices = (mesh or {}).get("vertices") or (mesh or {}).get("vertices_xyz_mm") or []
    if not vertices:
        for component in (mesh or {}).get("components", []):
            vertices.extend(component.get("vertices", []))
    if not vertices:
        return None
    return ([min(point[i] for point in vertices) for i in range(3)], [max(point[i] for point in vertices) for i in range(3)])


def build_family_constrained_scene(
    native_details: dict[str, Any],
    path_graph: dict[str, Any],
    mesh: dict[str, Any] | None,
    groups: list[dict[str, Any]] | None = None,
    calculation_contours: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Lift only closed 1D/elevation plus 0D/section family constraints."""

    bounds = _mesh_bounds(mesh)
    if bounds is None:
        return {"status": "unresolved", "path_count": 0, "paths": [], "reason": "a validated concrete mesh is required"}
    physical_families = native_details.get("physical_families", []) or []
    if not physical_families:
        return {
            "status": "unresolved",
            "path_count": 0,
            "paths": [],
            "candidate_path_count": 0,
            "candidate_paths": [],
            "rejected_families": [],
            "reason": "no physical rebar families closed",
            "concrete_mesh_status": "available",
            "schedule_values_used": False,
            "ordering_validation": "generated only after family identity and multiplicity constraints",
        }
    minimum, maximum = bounds
    components = {item["id"]: item for item in path_graph.get("components", [])}
    group_by_id = {item["id"]: item for item in groups or []}
    detail_by_id = {item["id"]: item for item in native_details.get("details", [])}
    paths = []
    candidate_paths = []
    rejected = []
    for family in physical_families:
        match = family.get("accepted_projection_match")
        group = group_by_id.get(family.get("source_group_id"))
        if family.get("constraint_status") == "resolved" and group:
            topology = group.get("topology", {}).get("family", {}).get("value")
            diameter = group.get("bar_spec", {}).get("diameter_mm", {})
            evidence = [family["id"], group["id"]]
            if topology == "straight":
                metric = group.get("placement", {}).get("metric_solution") or {}
                solved = metric.get("solved_centerline", {})
                positions = group.get("distribution", {}).get("positions_xy_mm", [])
                z0, z1 = solved.get("z_start_mm"), solved.get("z_end_mm")
                valid = (
                    metric.get("status") == "pass"
                    and len(positions) == int(family["count"])
                    and z0 is not None
                    and z1 is not None
                    and minimum[2] <= float(z0) < float(z1) <= maximum[2]
                    and all(minimum[0] <= float(x) <= maximum[0] and minimum[1] <= float(y) <= maximum[1] for x, y in positions)
                )
                if valid:
                    for instance, (x, y) in enumerate(positions, start=1):
                        paths.append(
                            {
                                "mark": family["mark"],
                                "role": "physical_bar_family",
                                "instance": instance,
                                "diameter_mm": diameter.get("value") if diameter.get("state") in {"direct", "observed", "derived"} else None,
                                "diameter_observation": diameter,
                                "placement_status": "drawing_constrained",
                                "fabrication_status": group.get("fabrication", {}).get("status"),
                                "points_xyz_mm": [[float(x), float(y), float(z0)], [float(x), float(y), float(z1)]],
                                "closed": False,
                                "evidence": evidence,
                                "reprojection_validation": {"status": "pass", "basis": family["group_constraint_certificate"]["basis"]},
                            }
                        )
                    continue
            elif topology == "closed_polyline":
                parameters = group.get("topology", {}).get("parameters", {})
                width, height = parameters.get("width_mm"), parameters.get("height_mm")
                stations = group.get("distribution", {}).get("instance_positions_mm", [])
                valid = (
                    width is not None
                    and height is not None
                    and len(stations) == int(family["count"])
                    and float(width) > 0
                    and float(height) > 0
                    and -float(width) / 2 >= minimum[0]
                    and float(width) / 2 <= maximum[0]
                    and -float(height) / 2 >= minimum[1]
                    and float(height) / 2 <= maximum[1]
                    and all(minimum[2] <= float(z) <= maximum[2] for z in stations)
                )
                if valid:
                    half_x, half_y = float(width) / 2, float(height) / 2
                    for instance, station in enumerate(stations, start=1):
                        z = float(station)
                        paths.append(
                            {
                                "mark": family["mark"],
                                "role": "physical_bar_family",
                                "instance": instance,
                                "diameter_mm": diameter.get("value") if diameter.get("state") in {"direct", "observed", "derived"} else None,
                                "diameter_observation": diameter,
                                "placement_status": "drawing_constrained",
                                "fabrication_status": group.get("fabrication", {}).get("status"),
                                "points_xyz_mm": [
                                    [-half_x, -half_y, z],
                                    [half_x, -half_y, z],
                                    [half_x, half_y, z],
                                    [-half_x, half_y, z],
                                    [-half_x, -half_y, z],
                                ],
                                "closed": True,
                                "evidence": evidence,
                                "reprojection_validation": {"status": "pass", "basis": family["group_constraint_certificate"]["basis"]},
                            }
                        )
                    continue
            rejected.append({"mark": family.get("mark"), "reason": "closed family parameters fail concrete-bound reprojection"})
            continue
        if family.get("constraint_status") != "resolved" or not match:
            rejected.append({"mark": family.get("mark"), "reason": family.get("constraint_reason") or family.get("scene_status")})
            continue
        elevation = components.get(match["elevation_component_id"], {})
        section = components.get(match["section_component_id"], {})
        elevation_axis = elevation.get("object_axis_projection", {})
        section_axis = section.get("object_axis_projection", {})
        orthogonal_axis_pair = (
            elevation.get("projection_dimensionality", {}).get("value") == "axis_projection_1d"
            and section.get("projection_dimensionality", {}).get("value") == "axis_projection_1d"
            and elevation_axis.get("state") == "resolved_relative"
            and section_axis.get("state") == "resolved_relative"
            and elevation_axis.get("axis") != section_axis.get("axis")
            and elevation_axis.get("station_axis") == section_axis.get("station_axis")
        )
        if orthogonal_axis_pair:
            mirror = family.get("mirror_equivalence", {})
            multiplicity = family.get("multiplicity", {})
            host = multiplicity.get("host_bbox_display") or []
            centers = multiplicity.get("display_centers") or []
            details = [detail_by_id[item] for item in family.get("detail_ids", []) if item in detail_by_id]
            dimensions = [
                item
                for detail in details
                for item in detail.get("raster_fabrication_dimension_observations", [])
                if item.get("confidence", 0.0) >= 0.90 and item.get("value_mm") is not None
            ]
            horizontal = sorted(
                {
                    float(item["value_mm"])
                    for item in dimensions
                    if item.get("semantic_role") == "overall_width_dimension"
                }
            ) or sorted({float(item["value_mm"]) for item in dimensions if item.get("orientation") == "horizontal" and item["value_mm"] >= 100})
            vertical = sorted(
                {
                    float(item["value_mm"])
                    for item in dimensions
                    if item.get("semantic_role") == "overall_height_dimension"
                }
            ) or sorted({float(item["value_mm"]) for item in dimensions if item.get("orientation") == "vertical" and item["value_mm"] >= 100})
            fabrication_solutions = [
                detail.get("fabrication_geometry_solution", {})
                for detail in details
                if detail.get("fabrication_geometry_solution")
            ]
            linear_hook_topology_closed = bool(fabrication_solutions) and all(
                item.get("topology") == "open_rectangular_loop_with_two_diagonal_hooks"
                and item.get("closure_validation", {}).get("linear_dimensions_assigned")
                and item.get("closure_validation", {}).get("topology_closed")
                for item in fabrication_solutions
            )
            contour = next(
                (
                    item
                    for item in calculation_contours or []
                    if item.get("role") == "dimensioned_elevation_profile"
                    and len(item.get("bbox_display", [])) == 4
                ),
                None,
            )
            valid_envelope = (
                (
                    mirror.get("status") == "pass"
                    or bool(section_axis.get("certificate_id"))
                )
                and len(host) == 4
                and host[2] > host[0]
                and host[3] > host[1]
                and len(centers) == int(family.get("count") or 0)
                and len(horizontal) == 1
                and len(vertical) == 1
                and contour is not None
            )
            if valid_envelope:
                contour_box = contour["bbox_display"]
                elevation_box = elevation.get("bbox_display") or []
                section_box = section.get("bbox_display") or []
                if len(elevation_box) == 4 and len(section_box) == 4 and contour_box[3] > contour_box[1]:
                    z_center = maximum[2] - (
                        ((elevation_box[1] + elevation_box[3]) / 2 - contour_box[1])
                        / (contour_box[3] - contour_box[1])
                        * (maximum[2] - minimum[2])
                    )
                    y_center = minimum[1] + (
                        ((section_box[1] + section_box[3]) / 2 - host[1])
                        / (host[3] - host[1])
                        * (maximum[1] - minimum[1])
                    )
                    half_y, half_z = horizontal[0] / 2, vertical[0] / 2
                    stations = [
                        minimum[0] + ((float(center[0]) - host[0]) / (host[2] - host[0])) * (maximum[0] - minimum[0])
                        for center in centers
                    ]
                    inside = (
                        minimum[1] <= y_center - half_y < y_center + half_y <= maximum[1]
                        and minimum[2] <= z_center - half_z < z_center + half_z <= maximum[2]
                        and all(minimum[0] <= station <= maximum[0] for station in stations)
                    )
                    if inside:
                        for instance, station in enumerate(stations, start=1):
                            candidate_paths.append(
                                {
                                    "mark": family["mark"],
                                    "role": "projection_constrained_fabrication_envelope",
                                    "instance": instance,
                                    "placement_status": (
                                        "candidate_centerline_radius_unresolved"
                                        if linear_hook_topology_closed
                                        else "candidate_hook_topology_unresolved"
                                    ),
                                    "points_xyz_mm": [
                                        [station, y_center - half_y, z_center - half_z],
                                        [station, y_center + half_y, z_center - half_z],
                                        [station, y_center + half_y, z_center + half_z],
                                        [station, y_center - half_y, z_center + half_z],
                                        [station, y_center - half_y, z_center - half_z],
                                    ],
                                    "closed": True,
                                    "evidence": [family["id"], elevation["id"], section["id"], *(item["id"] for item in details)],
                                    "reprojection_validation": {
                                        "status": "candidate",
                                        "basis": "orthogonal mark-linked projections plus two high-confidence fabrication envelope dimensions",
                                        "mirror_equivalence": mirror,
                                        "axis_certificate_id": section_axis.get("certificate_id"),
                                        "fabrication_solution": fabrication_solutions[0] if len(fabrication_solutions) == 1 else None,
                                        "unresolved": (
                                            "centerline bend radius or fabrication-standard certificate"
                                            if linear_hook_topology_closed
                                            else "hook, bend radius, and centerline deduction"
                                        ),
                                    },
                                }
                            )
            rejected.append(
                {
                    "mark": family["mark"],
                    "reason": (
                        (
                            "projection envelope exported separately; centerline bend radius remains unresolved"
                            if linear_hook_topology_closed
                            else "projection envelope exported separately; hook and bend topology remain unresolved"
                        )
                        if any(item["mark"] == family["mark"] for item in candidate_paths)
                        else "orthogonal projection pair lacks a complete orientation-certified fabrication envelope"
                    ),
                }
            )
            continue
        if elevation.get("projection_dimensionality", {}).get("value") != "axis_projection_1d" or section.get("projection_dimensionality", {}).get("value") != "axis_end_projection_0d":
            rejected.append({"mark": family["mark"], "reason": "the accepted pair is not a straight-axis lift"})
            continue
        length = elevation.get("projected_length", {}).get("value_mm")
        section_components = [
            components[item]
            for item in family.get("multiplicity", {}).get("component_ids", [])
            if item in components
        ]
        centers = [
            item.get("projection_dimensionality", {}).get("normalized_center_in_view")
            for item in section_components
        ]
        if not centers:
            centers = family.get("multiplicity", {}).get("normalized_centers", [])
            section_components = [section] * len(centers)
        if not centers or any(center is None for center in centers) or length is None or not 0 < float(length) <= 1.05 * (maximum[2] - minimum[2]):
            rejected.append({"mark": family["mark"], "reason": "metric length or normalized section position fails reprojection bounds"})
            continue
        count = int(family["count"])
        if count != len(centers):
            rejected.append({"mark": family["mark"], "reason": "section coordinates do not enumerate the derived multiplicity"})
            continue
        z_mid = (minimum[2] + maximum[2]) / 2.0
        z0, z1 = z_mid - float(length) / 2.0, z_mid + float(length) / 2.0
        if z0 < minimum[2] - 1e-6 or z1 > maximum[2] + 1e-6:
            rejected.append({"mark": family["mark"], "reason": "lifted centerline leaves the concrete bounds"})
            continue
        for instance, (center, section_component) in enumerate(zip(centers, section_components), start=1):
            x = minimum[0] + float(center[0]) * (maximum[0] - minimum[0])
            y = minimum[1] + float(center[1]) * (maximum[1] - minimum[1])
            paths.append(
                {
                    "mark": family["mark"],
                    "role": "physical_bar_family",
                    "instance": instance,
                    "diameter_mm": None,
                    "placement_status": "drawing_constrained",
                    "points_xyz_mm": [[x, y, z0], [x, y, z1]],
                    "closed": False,
                    "evidence": [family["id"], elevation["id"], section_component["id"]],
                    "reprojection_validation": {"status": "pass", "length_residual_mm": 0.0, "section_center_residual": 0.0},
                }
            )
    return {
        "status": "drawing_constrained" if paths else "partial",
        "path_count": len(paths),
        "paths": paths,
        "candidate_path_count": len(candidate_paths),
        "candidate_paths": candidate_paths,
        "rejected_families": rejected,
        "reason": None if paths else "no physical rebar families closed",
        "concrete_mesh_status": "available",
        "schedule_values_used": False,
        "ordering_validation": "generated only after family identity and multiplicity constraints",
    }
