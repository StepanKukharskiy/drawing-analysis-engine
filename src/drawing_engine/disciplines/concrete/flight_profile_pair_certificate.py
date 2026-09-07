"""Canonicalize section contours and certify at most one physical flight pair."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _drawing_ref(ref: Any) -> str:
    return str(ref).split(".item[", 1)[0]


def _primitive_ref(ref: Any) -> str:
    return str(ref).split(".segment[", 1)[0]


def _normal_style(style: Mapping[str, Any] | Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(style, Mapping):
        stroke = style.get("stroke")
        width = style.get("width")
        dash = style.get("dash")
    else:
        stroke, width, dash = list(style)
    return (
        tuple(round(float(value), 3) for value in (stroke or ())),
        round(float(width or 0.0), 3),
        str(dash or "").replace(" ", ""),
    )


def _canonical_cycle(values: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not values:
        return ()
    forward = [tuple(values[index:] + values[:index]) for index in range(len(values))]
    reversed_values = [(-dx, -dy) for dx, dy in reversed(values)]
    reverse = [
        tuple(reversed_values[index:] + reversed_values[:index])
        for index in range(len(reversed_values))
    ]
    return min((*forward, *reverse))


def _metric_path_signature(profile: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    points = [tuple(map(float, point)) for point in profile.get("ordered_boundary_display", []) or []]
    scale = float((profile.get("dimension_certificate") or {}).get("scale_points_per_mm") or 0.0)
    if len(points) < 3 or scale <= 0:
        return ()
    vectors = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx = round((end[0] - start[0]) / scale, 3)
        dy = round((end[1] - start[1]) / scale, 3)
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            vectors.append((dx, dy))
    return _canonical_cycle(vectors)


def _bridge_signature(profile: Mapping[str, Any]) -> tuple[str, ...]:
    signatures = []
    for bridge in profile.get("derived_bridges", []) or []:
        payload = {
            key: value
            for key, value in bridge.items()
            if key not in {"id", "evidence_refs"}
        }
        payload["evidence_refs"] = sorted(map(str, bridge.get("evidence_refs", []) or []))
        signatures.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return tuple(sorted(signatures))


def _correspondence_evidence(
    view_frame_graph: Mapping[str, Any],
    scope_ref: str,
) -> tuple[set[tuple[Any, ...]], dict[str, list[dict[str, Any]]], list[str]]:
    structural_styles: set[tuple[Any, ...]] = set()
    transforms_by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correspondence_refs = []
    for coordinate_scope in (
        view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
    ):
        for correspondence in coordinate_scope.get("contour_correspondences", []) or []:
            if str(correspondence.get("child_view_id")) != scope_ref:
                continue
            correspondence_refs.append(str(correspondence.get("id")))
            for candidate in correspondence.get("candidates", []) or []:
                if candidate.get("state") != "pass":
                    continue
                child = candidate.get("child_signature") or {}
                style = child.get("style_signature")
                if style:
                    structural_styles.add(_normal_style(style))
                transform = candidate.get("signed_transform") or {}
                alternatives = [
                    {
                        "sign": int(item["sign"]),
                        "offset_mm": round(float(item["offset_mm"]), 6),
                    }
                    for item in transform.get("candidates", []) or []
                    if item.get("sign") in {-1, 1} and item.get("offset_mm") is not None
                ]
                record = {
                    "correspondence_ref": correspondence.get("id"),
                    "state": transform.get("state"),
                    "alternatives": alternatives,
                }
                for ref in map(str, candidate.get("child_geometry_refs", []) or []):
                    transforms_by_edge[ref].append(record)
    return structural_styles, transforms_by_edge, sorted(set(correspondence_refs))


def _profile_styles(profile: Mapping[str, Any], topology: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    styles = topology.get("drawing_styles", {}) or {}
    return sorted(
        {
            _normal_style(styles[ref])
            for ref in map(_drawing_ref, profile.get("source_edge_refs", []) or [])
            if ref in styles
        }
    )


def _profile_transform_alternatives(
    profile: Mapping[str, Any],
    transforms_by_edge: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    for edge_ref in map(str, profile.get("source_edge_refs", []) or []):
        records.extend(transforms_by_edge.get(edge_ref, []))
    unique = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item for item in records
    }
    return [unique[key] for key in sorted(unique)]


def _dimension_annotation_refs(dimensions: Iterable[Any]) -> set[str]:
    refs = set()
    for dimension in dimensions:
        baseline = _field(dimension, "baseline")
        if baseline is not None:
            refs.add(str(_field(baseline, "primitive_ref", "")))
        refs.update(
            str(_field(line, "primitive_ref", ""))
            for line in _field(dimension, "extension_lines", ()) or ()
        )
        refs.update(map(str, _field(dimension, "terminal_refs", ()) or ()))
    return {_primitive_ref(ref) for ref in refs if ref}


def _rebar_refs(rebar_path_graph: Mapping[str, Any]) -> set[str]:
    evidence_backed_fragment_ids = {
        str(fragment_ref)
        for component in rebar_path_graph.get("components", []) or []
        if component.get("mark_hypotheses") or component.get("resolved_group_ids")
        for fragment_ref in component.get("fragment_ids", []) or []
    }
    fragments = rebar_path_graph.get("fragments", []) or []
    evidence_backed_fragment_ids.update(
        str(item.get("id"))
        for item in fragments
        if item.get("mark_hypotheses") or item.get("resolved_group_ids")
    )
    return {
        str(item.get("primitive_ref"))
        for item in fragments
        if item.get("primitive_ref") and str(item.get("id")) in evidence_backed_fragment_ids
    }


def _flight_metric_targets(
    local_dimension_reclosure: Mapping[str, Any],
    scope_ref: str,
) -> tuple[list[float], float | None]:
    scope = next(
        (
            item
            for item in local_dimension_reclosure.get("scope_results", []) or []
            if str(item.get("scope_ref")) == scope_ref
        ),
        {},
    )
    ownership = {
        str(item.get("dimension_ref")): item
        for item in (scope.get("local_ownership") or {}).get("attachments", []) or []
    }
    rise_targets = []
    for certificate in scope.get("independent_arithmetic_chain_certificates", []) or []:
        overall_ref = str(certificate.get("overall_dimension_ref"))
        overall = ownership.get(overall_ref, {})
        total = float(certificate.get("total_value_mm") or 0.0)
        if overall.get("orientation") != "vertical" or total <= 0:
            continue
        rise_targets.extend(
            float(value)
            for value in certificate.get("term_values_mm", []) or []
            if float(value) >= 0.25 * total
        )
    repeated = [
        float(item.get("value_mm"))
        for item in scope.get("repeated_projected_measurement_certificates", []) or []
        if item.get("orientation") == "horizontal" and float(item.get("value_mm") or 0) > 0
    ]
    return sorted(rise_targets)[:2], min(repeated) if repeated else None


def _profile_geometry(profile: Mapping[str, Any]) -> dict[str, Any]:
    points = [tuple(map(float, point)) for point in profile.get("ordered_boundary_display", []) or []]
    scale = float((profile.get("dimension_certificate") or {}).get("scale_points_per_mm") or 0.0)
    if not points or scale <= 0:
        return {
            "scale_points_per_mm": scale,
            "horizontal_span_mm": None,
            "vertical_span_mm": None,
            "horizontal_edge_count": 0,
            "vertical_edge_count": 0,
            "diagonal_edge_count": 0,
            "points_display": points,
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    horizontal = vertical = diagonal = 0
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx = abs(end[0] - start[0]) / scale
        dy = abs(end[1] - start[1]) / scale
        if dy <= 3.0 and dx > 3.0:
            horizontal += 1
        elif dx <= 3.0 and dy > 3.0:
            vertical += 1
        elif dx > 3.0 and dy > 3.0:
            diagonal += 1
    return {
        "scale_points_per_mm": scale,
        "horizontal_span_mm": round((max(xs) - min(xs)) / scale, 3),
        "vertical_span_mm": round((max(ys) - min(ys)) / scale, 3),
        "horizontal_edge_count": horizontal,
        "vertical_edge_count": vertical,
        "diagonal_edge_count": diagonal,
        "centroid_display": [
            round(sum(xs) / len(xs), 6),
            round(sum(ys) / len(ys), 6),
        ],
        "points_display": points,
    }


def _canonicalize_scope(
    profiles: list[dict[str, Any]],
    topology: Mapping[str, Any],
    transforms_by_edge: Mapping[str, list[dict[str, Any]]],
    rebar_refs: set[str],
    annotation_refs: set[str],
    page_number: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        signature = {
            "source_edge_refs": sorted(map(str, profile.get("source_edge_refs", []) or [])),
            "metric_path_signature": _metric_path_signature(profile),
            "bridge_provenance_signature": _bridge_signature(profile),
            "transform_alternatives": _profile_transform_alternatives(profile, transforms_by_edge),
        }
        key = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        groups[key].append(profile)
        signatures[key] = signature

    canonical = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda item: str(item.get("id")))
        representative = members[0]
        source_edges = set(map(str, signatures[key]["source_edge_refs"]))
        primitive_support = {_primitive_ref(ref) for ref in source_edges}
        canonical.append(
            {
                "id": _stable_id(page_number, "canonical_section_contour", key),
                "state": "derived",
                "member_profile_refs": [str(item.get("id")) for item in members],
                "representative_profile_ref": str(representative.get("id")),
                "source_edge_refs": sorted(source_edges),
                "native_edge_support_sha256": hashlib.sha256(
                    "\0".join(sorted(source_edges)).encode("utf-8")
                ).hexdigest(),
                "metric_path_signature": [list(item) for item in signatures[key]["metric_path_signature"]],
                "bridge_provenance_signature": list(signatures[key]["bridge_provenance_signature"]),
                "transform_alternatives": signatures[key]["transform_alternatives"],
                "native_style_signatures": [
                    [list(style[0]), style[1], style[2]]
                    for style in _profile_styles(representative, topology)
                ],
                "geometry": _profile_geometry(representative),
                "closure": dict(representative.get("closure") or {}),
                "rebar_fragment_overlap_refs": sorted(
                    ref for ref in source_edges if _primitive_ref(ref) in rebar_refs
                ),
                "dimension_annotation_overlap_refs": sorted(primitive_support & annotation_refs),
                "derived_bridge_count": len(representative.get("derived_bridges", []) or []),
                "merged_provenance_refs": sorted(
                    {
                        str(ref)
                        for member in members
                        for ref in member.get("evidence_refs", []) or []
                    }
                ),
                "quantity_eligible": False,
            }
        )
    return canonical


def _dominance_and_role_filter(
    canonical: list[dict[str, Any]],
    structural_styles: set[tuple[Any, ...]],
    rise_targets: list[float],
    tread_pitch_mm: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weaker = set()
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in canonical:
        base = {
            "support": item["source_edge_refs"],
            "path": item["metric_path_signature"],
            "transforms": item["transform_alternatives"],
        }
        by_base[json.dumps(base, sort_keys=True, separators=(",", ":"))].append(item)
    for group in by_base.values():
        minimum_bridges = min(item["derived_bridge_count"] for item in group)
        for item in group:
            if item["derived_bridge_count"] > minimum_bridges:
                weaker.add(item["id"])

    survivors = []
    decisions = []
    for item in canonical:
        styles = {
            _normal_style(style) for style in item.get("native_style_signatures", []) or []
        }
        geometry = item["geometry"]
        vertical_span = geometry.get("vertical_span_mm")
        horizontal_span = geometry.get("horizontal_span_mm")
        rise_match = (
            bool(rise_targets)
            and vertical_span is not None
            and any(
                abs(vertical_span - target) <= max(75.0, 0.15 * target)
                for target in rise_targets
            )
        )
        run_support = (
            tread_pitch_mm is not None
            and horizontal_span is not None
            and horizontal_span >= 3.0 * tread_pitch_mm
        )
        stepped_waist_topology = (
            geometry.get("horizontal_edge_count", 0) >= 3
            and geometry.get("vertical_edge_count", 0) >= 3
            and geometry.get("diagonal_edge_count", 0) >= 1
        )
        gates = {
            "branch_free_closed_boundary": bool(
                item.get("closure", {}).get("closed")
                and item.get("closure", {}).get("branch_free")
                and item.get("closure", {}).get("unique_completion")
            ),
            "no_unnecessary_bridge_or_weaker_evidence": item["id"] not in weaker,
            "no_dimension_annotation_overlap": not item["dimension_annotation_overlap_refs"],
            "no_reinforcement_path_overlap": not item["rebar_fragment_overlap_refs"],
            "structural_native_style_compatible": bool(structural_styles and styles)
            and styles <= structural_styles,
            "projected_rise_station_agreement": rise_match,
            "repeated_tread_run_support": run_support,
            "stepped_surface_and_waist_underside_topology": stepped_waist_topology,
        }
        hard_pass = all(gates.values())
        reason_codes = [key for key, passed in gates.items() if not passed]
        decisions.append(
            {
                "canonical_contour_ref": item["id"],
                "status": "flight_role_eligible" if hard_pass else "rejected",
                "gates": gates,
                "reason_codes": reason_codes,
                "quantity_eligible": False,
                "evidence_refs": [item["id"], *item["source_edge_refs"]],
            }
        )
        if hard_pass:
            survivors.append(item)
    return survivors, decisions


def _shared_interface(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_points = left.get("geometry", {}).get("points_display", []) or []
    right_points = right.get("geometry", {}).get("points_display", []) or []
    scales = [
        float(left.get("geometry", {}).get("scale_points_per_mm") or 0),
        float(right.get("geometry", {}).get("scale_points_per_mm") or 0),
    ]
    tolerance = max(1.5, 15.0 * max(scales))
    matches = []
    for first in left_points:
        for second in right_points:
            residual = math.dist(first, second)
            if residual <= tolerance:
                matches.append(
                    {
                        "point_display": [
                            round((first[0] + second[0]) / 2.0, 6),
                            round((first[1] + second[1]) / 2.0, 6),
                        ],
                        "residual_points": round(residual, 6),
                    }
                )
    unique = {
        tuple(item["point_display"]): item for item in matches
    }
    interfaces = [unique[key] for key in sorted(unique)]
    return {
        "status": "pass" if 1 <= len(interfaces) <= 2 else "fail",
        "interface_points": interfaces,
        "tolerance_points": round(tolerance, 6),
        "unique_interface_count": len(interfaces),
    }


def _pair_candidates(
    contours: list[dict[str, Any]],
    rise_targets: list[float],
    structural_styles: set[tuple[Any, ...]],
    plan_band_certificate: Mapping[str, Any],
    page_number: int,
    landing_interface_certificate: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pairs = []
    bands = list(plan_band_certificate.get("bands", []) or [])
    equal_sweep_width = float(plan_band_certificate.get("equal_sweep_width_mm") or 0.0)
    plan_pass = (
        len(bands) == 2
        and equal_sweep_width > 0
        and all(float(item.get("sweep_width_mm") or 0) > 0 for item in bands)
        and bool(plan_band_certificate.get("bands_are_disjoint"))
    )
    for first, second in itertools.combinations(contours, 2):
        ordered = sorted(
            (first, second),
            key=lambda item: float(item.get("geometry", {}).get("centroid_display", [0, 0])[1]),
        )
        upper, lower = ordered[0], ordered[1]
        scales = [float(item["geometry"]["scale_points_per_mm"]) for item in ordered]
        spans = sorted(float(item["geometry"]["vertical_span_mm"]) for item in ordered)
        target_spans = sorted(rise_targets)
        station_pass = len(target_spans) == 2 and all(
            abs(value - target) <= max(75.0, 0.15 * target)
            for value, target in zip(spans, target_spans)
        )
        styles = [
            {_normal_style(style) for style in item.get("native_style_signatures", []) or []}
            for item in ordered
        ]
        interface = _shared_interface(first, second)
        certified_interface = dict(landing_interface_certificate or {})
        interface_gate = (
            certified_interface.get("status") == "pass"
            if certified_interface
            else interface["status"] == "pass"
        )
        gates = {
            "two_equal_plan_sweep_bands": plan_pass
            and all(
                abs(float(item.get("sweep_width_mm")) - equal_sweep_width)
                <= max(2.0, 0.01 * equal_sweep_width)
                for item in bands
            ),
            "projected_station_agreement": station_pass,
            "compatible_scale": abs(scales[0] / max(scales[1], 1e-12) - 1.0) <= 0.04,
            "compatible_native_styles": bool(structural_styles)
            and styles[0] == styles[1]
            and styles[0] <= structural_styles,
            "non_overlapping_component_placement": plan_pass,
            "explicit_landing_interfaces": interface_gate,
            "plan_reprojection": plan_pass,
            "section_reprojection": all(item.get("transform_alternatives") for item in ordered),
            "reinforcement_paths_excluded": all(
                not item.get("rebar_fragment_overlap_refs") for item in ordered
            ),
        }
        passed = all(gates.values())
        evidence = sorted(
            {
                first["id"],
                second["id"],
                *map(str, plan_band_certificate.get("evidence_refs", []) or []),
                *[
                    str(record.get("correspondence_ref"))
                    for item in ordered
                    for record in item.get("transform_alternatives", []) or []
                ],
            }
        )
        pairs.append(
            {
                "id": _stable_id(page_number, "flight_profile_pair_candidate", first["id"], second["id"]),
                "status": "pass" if passed else "rejected",
                "lower_flight_contour_ref": lower["id"],
                "upper_flight_contour_ref": upper["id"],
                "plan_band_refs": [str(item.get("id")) for item in bands],
                "landing_interface": certified_interface or interface,
                "gates": gates,
                "reason_codes": [key for key, value in gates.items() if not value],
                "evidence_refs": evidence,
                "quantity_eligible": False,
            }
        )
    return pairs


def certify_flight_profile_pair(
    scope_local_profile_reclosure: Mapping[str, Any],
    local_dimension_reclosure: Mapping[str, Any],
    banded_plan_sweep_evidence: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    topology: Mapping[str, Any],
    dimensions: Iterable[Any],
    rebar_path_graph: Mapping[str, Any],
    *,
    page_number: int,
    open_structural_boundary_assembly: Mapping[str, Any] | None = None,
    flight_interface_closure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the exact 128 -> N -> M flight-pair gate, or abstain."""

    scope_results = []
    all_rebar_refs = _rebar_refs(rebar_path_graph)
    annotation_refs = _dimension_annotation_refs(dimensions)
    plan_certificate = banded_plan_sweep_evidence.get("plan_band_certificate") or {}
    open_scopes = {
        str(item.get("scope_ref")): item
        for item in (open_structural_boundary_assembly or {}).get("scope_results", []) or []
    }
    interface_scopes = {
        str(item.get("scope_ref")): item
        for item in (flight_interface_closure or {}).get("scope_results", []) or []
    }
    for local_scope in scope_local_profile_reclosure.get("scope_results", []) or []:
        scope_ref = str(local_scope.get("scope_ref"))
        assembly = local_scope.get("assembly") or {}
        open_scope = open_scopes.get(scope_ref, {})
        interface_scope = interface_scopes.get(scope_ref, {})
        legacy_profiles = list(assembly.get("profiles", []) or [])
        open_profiles = list(
            interface_scope.get("closed_profiles", [])
            if interface_scope
            else open_scope.get("closed_profiles", []) or []
        )
        profiles = [*legacy_profiles, *open_profiles]
        structural_styles, transforms_by_edge, correspondence_refs = _correspondence_evidence(
            view_frame_graph, scope_ref
        )
        scope_annotation_refs = annotation_refs - {
            _primitive_ref(ref)
            for ref in open_scope.get("dual_role_structural_override_edge_refs", []) or []
        }
        rise_targets, tread_pitch = _flight_metric_targets(
            local_dimension_reclosure, scope_ref
        )
        canonical = _canonicalize_scope(
            profiles,
            topology,
            transforms_by_edge,
            all_rebar_refs,
            scope_annotation_refs,
            page_number,
        )
        raw_support = {
            str(ref)
            for profile in profiles
            for ref in profile.get("source_edge_refs", []) or []
        }
        structural_reprojection_edges = set(transforms_by_edge)
        missing_structural_edges = sorted(structural_reprojection_edges - raw_support)
        structural_abstentions = [
            {
                "reason_code": item.get("reason_code"),
                "component_drawing_refs": list(item.get("component_drawing_refs", []) or []),
                "branch_vertex_refs": list(item.get("branch_vertex_refs", []) or []),
                "evidence_refs": list(item.get("evidence_refs", []) or []),
            }
            for item in assembly.get("abstentions", []) or []
            if structural_reprojection_edges
            & set(map(str, item.get("evidence_refs", []) or []))
        ]
        survivors, decisions = _dominance_and_role_filter(
            canonical,
            structural_styles,
            rise_targets,
            tread_pitch,
        )
        pairs = _pair_candidates(
            survivors,
            rise_targets,
            structural_styles,
            plan_certificate,
            page_number,
            (interface_scope.get("accepted_assignment") or {}).get("landing_interface_match"),
        )
        passing = [item for item in pairs if item["status"] == "pass"]
        certificate = None
        if len(passing) == 1:
            pair = passing[0]
            certificate = {
                "id": _stable_id(page_number, "flight_profile_pair_certificate", pair["id"]),
                "status": "accepted",
                "pair_candidate_ref": pair["id"],
                "lower_flight_contour_ref": pair["lower_flight_contour_ref"],
                "upper_flight_contour_ref": pair["upper_flight_contour_ref"],
                "landing_interface": pair["landing_interface"],
                "evidence_refs": pair["evidence_refs"],
                "landing_geometry_established": False,
                "mesh_construction_eligible": False,
                "quantity_eligible": False,
            }
        status = (
            "accepted_unique_pair"
            if certificate
            else "ambiguous_multiple_pairs"
            if len(passing) > 1
            else "insufficient_constraints"
        )
        contour_transition = (
            f"{len(profiles)} raw candidates -> {len(canonical)} canonical contours -> "
            f"{len(passing)} admissible pairs -> "
            + ("unique pass" if certificate else "explicit ambiguity" if passing else "no admissible pair")
        )
        boundary_transition = (
            f"{int(open_scope.get('native_structural_support_count', 0))} native structural supports -> "
            f"{int(open_scope.get('eligible_edge_count', 0))} eligible edges -> "
            f"{int(open_scope.get('branch_free_chain_count', 0))} branch-free chains -> "
            f"{int(open_scope.get('interface_bounded_flight_proposal_count', 0))} interface-bounded flight proposals -> "
            f"{len(open_profiles)} closed profiles -> {len(passing)} admissible pairs"
        )
        interface_transition = (
            f"{int(interface_scope.get('open_flight_proposal_count', 0))} open flight proposals -> "
            f"{int(interface_scope.get('valid_port_assignment_count', 0))} valid port assignments -> "
            f"{len(open_profiles)} closed profiles -> {len(passing)} admissible flight pairs"
        )
        scope_results.append(
            {
                "scope_ref": scope_ref,
                "status": status,
                "reason_code": (
                    None
                    if certificate
                    else "multiple_physically_distinct_pairs_survive"
                    if len(passing) > 1
                    else "no_flight_pair_survives_exact_evidence_gates"
                ),
                "transition": contour_transition,
                "structural_boundary_transition": boundary_transition,
                "interface_closure_transition": interface_transition,
                "legacy_raw_candidate_count": len(legacy_profiles),
                "open_boundary_closed_profile_count": len(open_profiles),
                "incomplete_flight_boundary_proposal_refs": [
                    str(item.get("id"))
                    for item in open_scope.get("flight_boundary_proposals", []) or []
                    if not (item.get("closure") or {}).get("closed")
                ],
                "raw_candidate_count": len(profiles),
                "canonical_contour_count": len(canonical),
                "collapsed_representation_count": len(profiles) - len(canonical),
                "flight_role_eligible_contour_count": len(survivors),
                "generated_pair_count": len(pairs),
                "admissible_pair_count": len(passing),
                "rise_targets_mm": rise_targets,
                "tread_pitch_mm": tread_pitch,
                "structural_style_signatures": [
                    [list(style[0]), style[1], style[2]] for style in sorted(structural_styles)
                ],
                "contour_correspondence_refs": correspondence_refs,
                "structural_reprojection_edge_count": len(structural_reprojection_edges),
                "structural_reprojection_edges_missing_from_raw_candidates": missing_structural_edges,
                "structural_boundary_upstream_abstentions": structural_abstentions,
                "canonical_contours": canonical,
                "contour_filter_decisions": decisions,
                "pair_candidates": pairs,
                "flight_profile_pair_certificate": certificate,
                "quantity_eligible": False,
            }
        )

    certificates = [
        item["flight_profile_pair_certificate"]
        for item in scope_results
        if item.get("flight_profile_pair_certificate")
    ]
    ambiguous_scope_count = sum(
        item.get("status") == "ambiguous_multiple_pairs" for item in scope_results
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "flight_profile_pair_certification",
        "page": page_number,
        "status": (
            "accepted_unique_pair"
            if len(certificates) == 1
            else "ambiguous_multiple_pairs"
            if ambiguous_scope_count
            else "insufficient_constraints"
        ),
        "scope_results": scope_results,
        "certificates": certificates,
        "summary": {
            "scope_count": len(scope_results),
            "raw_candidate_count": sum(item["raw_candidate_count"] for item in scope_results),
            "canonical_contour_count": sum(item["canonical_contour_count"] for item in scope_results),
            "collapsed_representation_count": sum(
                item["collapsed_representation_count"] for item in scope_results
            ),
            "flight_role_eligible_contour_count": sum(
                item["flight_role_eligible_contour_count"] for item in scope_results
            ),
            "generated_pair_count": sum(item["generated_pair_count"] for item in scope_results),
            "admissible_pair_count": sum(item["admissible_pair_count"] for item in scope_results),
            "accepted_pair_certificate_count": len(certificates),
            "structural_reprojection_edge_count": sum(
                item["structural_reprojection_edge_count"] for item in scope_results
            ),
            "structural_reprojection_edge_missing_count": sum(
                len(item["structural_reprojection_edges_missing_from_raw_candidates"])
                for item in scope_results
            ),
            "transition": scope_results[0]["transition"] if len(scope_results) == 1 else None,
            "structural_boundary_transition": (
                scope_results[0]["structural_boundary_transition"]
                if len(scope_results) == 1
                else None
            ),
            "interface_closure_transition": (
                scope_results[0]["interface_closure_transition"]
                if len(scope_results) == 1
                else None
            ),
            "contour_rejection_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for scope in scope_results
                        for decision in scope["contour_filter_decisions"]
                        for reason in decision["reason_codes"]
                    ).items()
                )
            ),
        },
        "contract": {
            "canonical_group_requires_identical_native_edge_support": True,
            "canonical_group_requires_identical_metric_path_signature": True,
            "canonical_group_requires_identical_bridge_provenance": True,
            "canonical_group_requires_identical_transform_alternatives": True,
            "bbox_or_area_deduplication_forbidden": True,
            "strictly_weaker_bridge_evidence_is_dominated": True,
            "reinforcement_paths_cannot_be_concrete_boundaries": True,
            "unique_semantic_pair_required": True,
            "unresolved_158_175_labels_are_not_pair_prerequisites": True,
            "open_boundary_closed_profiles_are_pair_inputs": True,
            "incomplete_boundary_proposals_remain_auditable": True,
            "interface_closed_profiles_are_pair_inputs": True,
            "landing_or_mesh_construction_performed": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
