"""M5 cross-sheet MEP continuation and quantity-ineligible run hypotheses.

M5 consumes accepted M1 registrations and independently accepted M4
page-local relations.  A continuation joins only when endpoint evidence,
system, size, elevation, transformed coincidence, and mutual uniqueness all
close.  Coincident geometry in a tiled-sheet overlap is canonicalised before
run assembly, but only through the same attribute and bidirectional-uniqueness
gates.

Projected page occurrences remain separate from canonical segment and run
hypotheses.  No record is eligible for installed length, fitting counts,
clash status, material quantities, or takeoff; those remain M7 concerns.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_route_observations import validate_mep_route_graph


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_cross_sheet_run_hypotheses"
METHOD_VERSION = "1.0.0"
_ATTRIBUTE_TYPES = ("route_system", "route_size", "route_elevation")


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in candidate.items()
        if key not in {"raw_text", "terminology_entry_ref", "observed_symbol_kind"}
    }


def _semantic_key(candidate: Mapping[str, Any]) -> str:
    return json.dumps(
        _semantic_candidate(candidate),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _continuation_key(candidate: Mapping[str, Any]) -> str:
    value = _semantic_candidate(candidate)
    raw_text = " ".join(str(candidate.get("raw_text") or "").upper().split())
    if raw_text:
        value["explicit_tag_text"] = raw_text
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _apply(matrix: Iterable[object], point: Iterable[object]) -> list[float]:
    a, b, c, d, e, f = (float(value) for value in matrix)
    x, y = (float(value) for value in point)
    return [a * x + c * y + e, b * x + d * y + f]


def _inverse(matrix: Iterable[object]) -> list[float]:
    a, b, c, d, e, f = (float(value) for value in matrix)
    determinant = a * d - b * c
    if abs(determinant) <= 1e-12:
        raise ValueError("accepted M1 registration matrix is singular")
    return [
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
        (c * f - d * e) / determinant,
        (b * e - a * f) / determinant,
    ]


def _compose(after: Iterable[object], before: Iterable[object]) -> list[float]:
    aa, ab, ac, ad, ae, af = (float(value) for value in after)
    ba, bb, bc, bd, be, bf = (float(value) for value in before)
    return [
        aa * ba + ac * bb,
        ab * ba + ad * bb,
        aa * bc + ac * bd,
        ab * bc + ad * bd,
        aa * be + ac * bf + ae,
        ab * be + ad * bf + af,
    ]


def _registration_edges(
    sheet_registry: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    direct = {}
    adjacency: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sheet_registry.get("adjoining_sheet_transforms", []):
        if row.get("state") != "accepted":
            continue
        matrix = row.get("matrix_source_display_to_target_display")
        if not isinstance(matrix, list) or len(matrix) != 6:
            continue
        source, target = str(row["source_page_ref"]), str(row["target_page_ref"])
        fits = row.get("grid_fit", {})
        fit_residuals = [
            float(fit.get("maximum_residual", 0.0))
            for fit in (fits.get("x"), fits.get("y"))
            if isinstance(fit, Mapping)
        ]
        forward = {
            "registration_ref": str(row["id"]),
            "source_page_ref": source,
            "target_page_ref": target,
            "matrix": [float(value) for value in matrix],
            "tolerance": float(row.get("maximum_residual_tolerance_display", 0.75)),
            "observed_residual": max(fit_residuals, default=0.0),
        }
        reverse = {
            **forward,
            "source_page_ref": target,
            "target_page_ref": source,
            "matrix": _inverse(matrix),
        }
        direct[(source, target)] = forward
        direct[(target, source)] = reverse
        adjacency[source].append(forward)
        adjacency[target].append(reverse)
    for rows in adjacency.values():
        rows.sort(key=lambda row: (row["target_page_ref"], row["registration_ref"]))
    return direct, dict(adjacency)


def _alternate_transform(
    adjacency: Mapping[str, list[Mapping[str, Any]]],
    source: str,
    target: str,
    excluded_registration_ref: str,
) -> list[float] | None:
    queue = deque([(source, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], frozenset({source}))])
    while queue:
        page_ref, matrix, visited = queue.popleft()
        for edge in adjacency.get(page_ref, []):
            if str(edge["registration_ref"]) == excluded_registration_ref:
                continue
            next_page = str(edge["target_page_ref"])
            if next_page in visited:
                continue
            composed = _compose(edge["matrix"], matrix)
            if next_page == target:
                return composed
            queue.append((next_page, composed, visited | {next_page}))
    return None


def _page_indexes(route_graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for page in route_graph.get("pages", []):
        output[str(page["page_ref"])] = {
            "page": page,
            "fragments": {str(row["id"]): row for row in page.get("fragments", [])},
            "endpoints": {str(row["id"]): row for row in page.get("endpoints", [])},
            "vertices": {str(row["id"]): row for row in page.get("vertices", [])},
            "branch_vertex_refs": {
                str(row["vertex_ref"]) for row in page.get("branches", [])
            },
        }
    return output


def _relation_indexes(
    bindings: Mapping[str, Any],
) -> tuple[dict[str, dict[str, list[Mapping[str, Any]]]], dict[str, list[Mapping[str, Any]]]]:
    by_fragment: defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    continuations: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for relation in bindings.get("relations", []):
        if relation.get("state") != "accepted":
            continue
        relation_type = str(relation.get("relation_type") or "")
        if relation_type in _ATTRIBUTE_TYPES:
            for fragment_ref in relation.get("target_fragment_refs", []):
                by_fragment[str(fragment_ref)][relation_type].append(relation)
        elif relation_type == "continuation_endpoint":
            for endpoint_ref in relation.get("target_refs", []):
                continuations[str(endpoint_ref)].append(relation)
    return (
        {
            fragment_ref: {kind: list(rows) for kind, rows in kinds.items()}
            for fragment_ref, kinds in by_fragment.items()
        },
        dict(continuations),
    )


def _attribute_value(
    fragment_ref: str,
    relation_type: str,
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    rows = list(relations_by_fragment.get(fragment_ref, {}).get(relation_type, []))
    values = {_semantic_key(row.get("candidate", {})) for row in rows}
    refs = sorted(str(row["id"]) for row in rows)
    if len(values) != 1:
        reason = (
            f"missing_{relation_type.removeprefix('route_')}"
            if not values
            else f"conflicting_{relation_type.removeprefix('route_')}"
        )
        return None, refs, reason
    row = rows[0]
    return _semantic_candidate(row.get("candidate", {})), refs, None


def _metres(value: object, unit: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    factors = {"m": 1.0, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
    factor = factors.get(str(unit or "").lower())
    return None if factor is None else number * factor


def _compatible_attributes(
    left_fragment_ref: str,
    right_fragment_ref: str,
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
) -> tuple[dict[str, bool], list[str], dict[str, Any], list[str]]:
    certificates = {}
    reasons = []
    values = {}
    refs = []
    for relation_type in _ATTRIBUTE_TYPES:
        name = relation_type.removeprefix("route_")
        left, left_refs, left_reason = _attribute_value(
            left_fragment_ref, relation_type, relations_by_fragment
        )
        right, right_refs, right_reason = _attribute_value(
            right_fragment_ref, relation_type, relations_by_fragment
        )
        refs.extend([*left_refs, *right_refs])
        values[name] = {"source": left, "target": right}
        if left_reason:
            reasons.append(f"source_{left_reason}")
        if right_reason:
            reasons.append(f"target_{right_reason}")
        compatible = left is not None and right is not None
        if compatible and relation_type == "route_elevation":
            left_m = _metres(left.get("value"), left.get("unit"))
            right_m = _metres(right.get("value"), right.get("unit"))
            compatible = (
                left_m is not None
                and right_m is not None
                and left.get("basis") == right.get("basis")
                and abs(left_m - right_m) <= 1e-4
            )
        elif compatible:
            compatible = _semantic_key(left) == _semantic_key(right)
        certificates[f"compatible_{name}"] = compatible
        if left is not None and right is not None and not compatible:
            reasons.append(f"incompatible_{name}")
    return certificates, reasons, values, sorted(set(refs))


def _endpoint_rows(
    continuations: Mapping[str, list[Mapping[str, Any]]],
    pages: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for page_ref, indexes in pages.items():
        for endpoint_ref, relations in continuations.items():
            endpoint = indexes["endpoints"].get(endpoint_ref)
            if endpoint is None:
                continue
            continuation_values = {
                _continuation_key(row.get("candidate", {})) for row in relations
            }
            output.append(
                {
                    "page_ref": page_ref,
                    "endpoint": endpoint,
                    "fragment": indexes["fragments"][str(endpoint["fragment_ref"])],
                    "relation_refs": sorted(str(row["id"]) for row in relations),
                    "continuation_value": (
                        json.loads(next(iter(continuation_values)))
                        if len(continuation_values) == 1
                        else None
                    ),
                    "continuation_is_unique": len(continuation_values) == 1,
                    "boundary_branch": str(endpoint.get("vertex_ref"))
                    in indexes["branch_vertex_refs"],
                }
            )
    return sorted(output, key=lambda row: (row["page_ref"], row["endpoint"]["id"]))


def _mark_mutual_unique(
    rows: list[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    certificate_key: str,
    reason: str,
) -> None:
    eligible = [row for row in rows if not row["reasons"]]
    left_counts = Counter(str(row[left_key]) for row in eligible)
    right_counts = Counter(str(row[right_key]) for row in eligible)
    for row in rows:
        unique = (
            not row["reasons"]
            and left_counts[str(row[left_key])] == 1
            and right_counts[str(row[right_key])] == 1
        )
        row["certificates"][certificate_key] = unique
        if not unique and not row["reasons"]:
            row["reasons"].append(reason)
        row["reasons"] = sorted(set(row["reasons"]))
        row["state"] = "accepted" if not row["reasons"] else "abstained"
        row["quantity_eligible"] = False


def _continuation_candidates(
    pages: Mapping[str, Mapping[str, Any]],
    registrations: Mapping[tuple[str, str], Mapping[str, Any]],
    adjacency: Mapping[str, list[Mapping[str, Any]]],
    continuations: Mapping[str, list[Mapping[str, Any]]],
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    tolerance: float,
) -> list[dict[str, Any]]:
    endpoints = _endpoint_rows(continuations, pages)
    output = []
    for index, left in enumerate(endpoints):
        for right in endpoints[index + 1 :]:
            if left["page_ref"] == right["page_ref"]:
                continue
            forward = registrations.get((left["page_ref"], right["page_ref"]))
            source, target = left, right
            if forward is None:
                forward = registrations.get((right["page_ref"], left["page_ref"]))
                source, target = right, left
            reasons = []
            certificates = {
                "accepted_m1_registration": forward is not None,
                "explicit_endpoint_bound_continuation_evidence": bool(
                    source["relation_refs"] and target["relation_refs"]
                ),
                "continuation_evidence_is_unambiguous": bool(
                    source["continuation_is_unique"]
                    and target["continuation_is_unique"]
                    and source["continuation_value"] == target["continuation_value"]
                ),
                "boundary_is_not_branched": not (
                    source["boundary_branch"] or target["boundary_branch"]
                ),
                "transform_cycle_residual_within_tolerance": False,
                "transformed_endpoint_coincidence": False,
            }
            if forward is None:
                reasons.append("missing_accepted_m1_registration")
                transformed = None
                distance = None
                registration_ref = None
                cycle_residual = None
            else:
                registration_ref = str(forward["registration_ref"])
                transformed = _apply(
                    forward["matrix"], source["endpoint"]["point_display"]
                )
                distance = math.dist(transformed, target["endpoint"]["point_display"])
                coincidence_tolerance = max(
                    float(tolerance), float(forward["observed_residual"])
                )
                certificates["transformed_endpoint_coincidence"] = (
                    distance <= coincidence_tolerance
                )
                if distance > coincidence_tolerance:
                    reasons.append("transformed_endpoints_do_not_coincide")
                alternate = _alternate_transform(
                    adjacency,
                    source["page_ref"],
                    target["page_ref"],
                    registration_ref,
                )
                cycle_residual = (
                    None
                    if alternate is None
                    else math.dist(
                        transformed,
                        _apply(alternate, source["endpoint"]["point_display"]),
                    )
                )
                certificates["transform_cycle_residual_within_tolerance"] = (
                    cycle_residual is None or cycle_residual <= coincidence_tolerance
                )
                if not certificates["transform_cycle_residual_within_tolerance"]:
                    reasons.append("transform_cycle_residual_exceeds_tolerance")
            if not certificates["continuation_evidence_is_unambiguous"]:
                reasons.append("ambiguous_continuation_evidence")
            if not certificates["boundary_is_not_branched"]:
                reasons.append("boundary_branch_ambiguity")
            attrs, attr_reasons, attribute_values, attribute_refs = _compatible_attributes(
                str(source["fragment"]["id"]),
                str(target["fragment"]["id"]),
                relations_by_fragment,
            )
            certificates.update(attrs)
            reasons.extend(attr_reasons)
            output.append(
                {
                    "record_type": "mep_cross_sheet_continuation_candidate",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id(
                        "mep_cross_sheet_continuation",
                        source["endpoint"]["id"],
                        target["endpoint"]["id"],
                        registration_ref,
                    ),
                    "source_page_ref": source["page_ref"],
                    "target_page_ref": target["page_ref"],
                    "source_endpoint_ref": str(source["endpoint"]["id"]),
                    "target_endpoint_ref": str(target["endpoint"]["id"]),
                    "source_fragment_ref": str(source["fragment"]["id"]),
                    "target_fragment_ref": str(target["fragment"]["id"]),
                    "registration_ref": registration_ref,
                    "continuation_relation_refs": sorted(
                        [*source["relation_refs"], *target["relation_refs"]]
                    ),
                    "attribute_relation_refs": attribute_refs,
                    "attribute_values": attribute_values,
                    "transformed_source_point_display": transformed,
                    "target_point_display": target["endpoint"]["point_display"],
                    "endpoint_residual_display_points": (
                        None if distance is None else round(distance, 6)
                    ),
                    "transform_cycle_residual_display_points": (
                        None if cycle_residual is None else round(cycle_residual, 6)
                    ),
                    "method": {
                        "name": "mutual_unique_registered_endpoint_continuation",
                        "version": METHOD_VERSION,
                    },
                    "certificates": certificates,
                    "reasons": sorted(set(reasons)),
                    "state": "abstained",
                    "quantity_eligible": False,
                }
            )
    _mark_mutual_unique(
        output,
        left_key="source_endpoint_ref",
        right_key="target_endpoint_ref",
        certificate_key="mutual_unique_pairing",
        reason="continuation_pair_is_not_mutual_unique",
    )
    return sorted(output, key=lambda row: row["id"])


def _style_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    compared = [
        key
        for key in ("width_display_points", "dash_pattern")
        if left.get(key) is not None and right.get(key) is not None
    ]
    return bool(compared) and all(left.get(key) == right.get(key) for key in compared)


def _composite_indexes(
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    composites = {
        str(row["id"]): row
        for row in bindings.get("outlined_route_composites", [])
        if row.get("state") == "accepted"
    }
    by_fragment = {
        str(fragment_ref): composite_ref
        for composite_ref, composite in composites.items()
        for fragment_ref in composite.get("member_fragment_refs", [])
    }
    return composites, by_fragment


def _path_residual(
    source_points: Iterable[Iterable[object]],
    target_points: Iterable[Iterable[object]],
    matrix: Iterable[object],
) -> float | None:
    source = [_apply(matrix, point) for point in source_points]
    target = [[float(value) for value in point] for point in target_points]
    if len(source) != len(target) or len(source) < 2:
        return None
    direct = max(math.dist(left, right) for left, right in zip(source, target))
    reverse = max(math.dist(left, right) for left, right in zip(source, reversed(target)))
    return min(direct, reverse)


def _duplicate_candidates(
    pages: Mapping[str, Mapping[str, Any]],
    registrations: Mapping[tuple[str, str], Mapping[str, Any]],
    adjacency: Mapping[str, list[Mapping[str, Any]]],
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    composites: Mapping[str, Mapping[str, Any]],
    composite_by_fragment: Mapping[str, str],
    tolerance: float,
) -> list[dict[str, Any]]:
    output = []
    seen_pairs = set()
    closure_support_refs = {
        str(ref)
        for composite in composites.values()
        for ref in composite.get("supporting_closure_fragment_refs", [])
    }
    for (source_page, target_page), registration in sorted(registrations.items()):
        unordered = tuple(sorted((source_page, target_page)))
        if unordered in seen_pairs or source_page >= target_page:
            continue
        seen_pairs.add(unordered)
        source_rows = pages[source_page]["fragments"].values()
        target_rows = pages[target_page]["fragments"].values()
        coincidence_tolerance = max(
            float(tolerance), float(registration["observed_residual"])
        )
        alternate = _alternate_transform(
            adjacency,
            source_page,
            target_page,
            str(registration["registration_ref"]),
        )
        for left in source_rows:
            if str(left["id"]) in closure_support_refs:
                continue
            left_points = left.get("geometry", {}).get("points_display", [])
            if len(left_points) != 2:
                continue
            transformed = [_apply(registration["matrix"], point) for point in left_points]
            for right in target_rows:
                if str(right["id"]) in closure_support_refs:
                    continue
                right_points = right.get("geometry", {}).get("points_display", [])
                if len(right_points) != 2:
                    continue
                direct = max(
                    math.dist(transformed[0], right_points[0]),
                    math.dist(transformed[1], right_points[1]),
                )
                reverse = max(
                    math.dist(transformed[0], right_points[1]),
                    math.dist(transformed[1], right_points[0]),
                )
                residual = min(direct, reverse)
                if residual > coincidence_tolerance:
                    continue
                reasons = []
                cycle_residual = None
                cycle_ok = True
                if alternate is not None:
                    alternate_points = [_apply(alternate, point) for point in left_points]
                    cycle_residual = max(
                        math.dist(a, b) for a, b in zip(transformed, alternate_points)
                    )
                    cycle_ok = cycle_residual <= coincidence_tolerance
                    if not cycle_ok:
                        reasons.append("transform_cycle_residual_exceeds_tolerance")
                style_ok = _style_compatible(left.get("style", {}), right.get("style", {}))
                if not style_ok:
                    reasons.append("non_color_style_is_incompatible")
                attrs, attr_reasons, attribute_values, attribute_refs = _compatible_attributes(
                    str(left["id"]), str(right["id"]), relations_by_fragment
                )
                reasons.extend(attr_reasons)
                source_target_ref = composite_by_fragment.get(
                    str(left["id"]), str(left["id"])
                )
                target_target_ref = composite_by_fragment.get(
                    str(right["id"]), str(right["id"])
                )
                source_composite = composites.get(source_target_ref)
                target_composite = composites.get(target_target_ref)
                if (source_composite is None) != (target_composite is None):
                    signature_residual = None
                    signature_ok = False
                elif source_composite is None:
                    signature_residual = residual
                    signature_ok = True
                else:
                    signature_residual = _path_residual(
                        source_composite.get("derived_geometry", {}).get(
                            "centreline_points_display", []
                        ),
                        target_composite.get("derived_geometry", {}).get(
                            "centreline_points_display", []
                        ),
                        registration["matrix"],
                    )
                    signature_ok = (
                        signature_residual is not None
                        and signature_residual <= coincidence_tolerance
                    )
                if not signature_ok:
                    reasons.append("transformed_semantic_path_signature_does_not_match")
                certificates = {
                    "accepted_m1_registration": True,
                    "transformed_geometry_coincidence": True,
                    "transform_cycle_residual_within_tolerance": cycle_ok,
                    "non_color_style_compatible": style_ok,
                    "matching_transformed_path_signatures": signature_ok,
                    **attrs,
                }
                output.append(
                    {
                        "record_type": "mep_overlap_duplicate_candidate",
                        "record_version": SCHEMA_VERSION,
                        "id": _stable_id(
                            "mep_overlap_duplicate",
                            left["id"],
                            right["id"],
                            registration["registration_ref"],
                        ),
                        "source_page_ref": source_page,
                        "target_page_ref": target_page,
                        "source_fragment_ref": str(left["id"]),
                        "target_fragment_ref": str(right["id"]),
                        "source_route_target_ref": source_target_ref,
                        "target_route_target_ref": target_target_ref,
                        "registration_ref": str(registration["registration_ref"]),
                        "attribute_relation_refs": attribute_refs,
                        "attribute_values": attribute_values,
                        "maximum_geometry_residual_display_points": round(residual, 6),
                        "semantic_path_signature_residual_display_points": (
                            None
                            if signature_residual is None
                            else round(signature_residual, 6)
                        ),
                        "transform_cycle_residual_display_points": (
                            None if cycle_residual is None else round(cycle_residual, 6)
                        ),
                        "stroke_color_match": left.get("style", {}).get("stroke")
                        == right.get("style", {}).get("stroke"),
                        "color_only_identity_eligible": False,
                        "method": {
                            "name": "registered_bidirectional_overlap_geometry",
                            "version": METHOD_VERSION,
                        },
                        "certificates": certificates,
                        "reasons": sorted(set(reasons)),
                        "state": "abstained",
                        "quantity_eligible": False,
                    }
                )
    _mark_mutual_unique(
        output,
        left_key="source_fragment_ref",
        right_key="target_fragment_ref",
        certificate_key="mutual_unique_duplicate_pairing",
        reason="duplicate_overlap_pair_is_not_mutual_unique",
    )
    eligible_pairs = {
        (str(row["source_route_target_ref"]), str(row["target_route_target_ref"]))
        for row in output
        if row["state"] == "accepted"
    }
    targets_by_source: defaultdict[str, set[str]] = defaultdict(set)
    sources_by_target: defaultdict[str, set[str]] = defaultdict(set)
    for source_ref, target_ref in eligible_pairs:
        targets_by_source[source_ref].add(target_ref)
        sources_by_target[target_ref].add(source_ref)
    for row in output:
        source_ref = str(row["source_route_target_ref"])
        target_ref = str(row["target_route_target_ref"])
        unique = (
            row["state"] == "accepted"
            and targets_by_source[source_ref] == {target_ref}
            and sources_by_target[target_ref] == {source_ref}
        )
        row["certificates"]["mutual_unique_semantic_target_pairing"] = unique
        if not unique and row["state"] == "accepted":
            row["reasons"].append("semantic_overlap_pair_is_not_mutual_unique")
            row["reasons"] = sorted(set(row["reasons"]))
            row["state"] = "abstained"
    return sorted(output, key=lambda row: row["id"])


def _projected_occurrences(
    pages: Mapping[str, Mapping[str, Any]],
    composites: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    consumed = {
        str(ref)
        for composite in composites.values()
        for ref in [
            *composite.get("member_fragment_refs", []),
            *composite.get("supporting_closure_fragment_refs", []),
        ]
    }
    for page_ref, indexes in sorted(pages.items()):
        for fragment in indexes["fragments"].values():
            if str(fragment["id"]) in consumed:
                continue
            metric = fragment.get("local_metric_observation", {})
            output.append(
                {
                    "record_type": "mep_projected_route_occurrence",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("mep_projected_route_occurrence", fragment["id"]),
                    "page_ref": page_ref,
                    "route_target_ref": str(fragment["id"]),
                    "fragment_ref": str(fragment["id"]),
                    "source_fragment_refs": [str(fragment["id"])],
                    "source_primitive_ref": fragment.get("source_primitive_ref"),
                    "points_display": deepcopy(
                        fragment.get("geometry", {}).get("points_display", [])
                    ),
                    "projected_2d_length_m": metric.get("projected_path_m"),
                    "projected_2d_length_state": metric.get("state", "unknown"),
                    "physical_segment_ref": None,
                    "state": "observed",
                    "quantity_eligible": False,
                }
            )
    fragments = {
        str(fragment["id"]): fragment
        for indexes in pages.values()
        for fragment in indexes["fragments"].values()
    }
    for composite_ref, composite in sorted(composites.items()):
        member_refs = sorted(str(ref) for ref in composite["member_fragment_refs"])
        metrics = [
            fragments[ref].get("local_metric_observation", {})
            for ref in member_refs
        ]
        projected = [
            float(metric["projected_path_m"])
            for metric in metrics
            if metric.get("projected_path_m") is not None
        ]
        output.append(
            {
                "record_type": "mep_projected_route_occurrence",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_projected_route_occurrence", composite_ref),
                "page_ref": str(composite["page_ref"]),
                "route_target_ref": composite_ref,
                "route_composite_ref": composite_ref,
                "fragment_ref": None,
                "source_fragment_refs": member_refs,
                "source_primitive_refs": sorted(
                    str(ref)
                    for ref in composite.get("member_source_primitive_refs", [])
                ),
                "points_display": deepcopy(
                    composite.get("derived_geometry", {}).get(
                        "centreline_points_display", []
                    )
                ),
                "projected_2d_length_m": (
                    None if len(projected) != len(member_refs) else round(sum(projected) / len(projected), 8)
                ),
                "projected_2d_length_state": (
                    "derived" if len(projected) == len(member_refs) else "unknown"
                ),
                "physical_segment_ref": None,
                "state": "derived",
                "quantity_eligible": False,
            }
        )
    return sorted(output, key=lambda row: row["id"])


def _canonical_segments(
    duplicates: list[Mapping[str, Any]],
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    occurrence_by_target = {str(row["route_target_ref"]): row for row in occurrences}
    output = []
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for duplicate in duplicates:
        if duplicate.get("state") == "accepted":
            grouped[
                (
                    str(duplicate["source_route_target_ref"]),
                    str(duplicate["target_route_target_ref"]),
                )
            ].append(duplicate)
    for target_pair, rows in sorted(grouped.items()):
        route_target_refs = sorted(target_pair)
        fragment_refs = sorted(
            {
                str(ref)
                for row in rows
                for ref in (row["source_fragment_ref"], row["target_fragment_ref"])
            }
        )
        canonical_id = _stable_id("mep_canonical_projected_segment", route_target_refs)
        lengths = [
            occurrence_by_target[ref]["projected_2d_length_m"]
            for ref in route_target_refs
            if occurrence_by_target[ref]["projected_2d_length_m"] is not None
        ]
        relation_refs = sorted(str(row["id"]) for row in rows)
        output.append(
            {
                "record_type": "mep_canonical_projected_segment",
                "record_version": SCHEMA_VERSION,
                "id": canonical_id,
                "source_page_occurrence_refs": [
                    occurrence_by_target[ref]["id"] for ref in route_target_refs
                ],
                "source_fragment_refs": fragment_refs,
                "source_route_target_refs": route_target_refs,
                "overlap_duplicate_relation_ref": relation_refs[0],
                "overlap_duplicate_relation_refs": relation_refs,
                "representative_projected_2d_length_m": (
                    None if not lengths else round(sum(lengths) / len(lengths), 8)
                ),
                "source_occurrence_count": len(route_target_refs),
                "physical_segment_identity_established": True,
                "state": "derived",
                "quantity_eligible": False,
            }
        )
        for ref in route_target_refs:
            occurrence_by_target[ref]["physical_segment_ref"] = canonical_id
    return sorted(output, key=lambda row: row["id"])


def _centreline_targets(
    pages: Mapping[str, Mapping[str, Any]],
    composites: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fragments = {
        str(fragment["id"]): fragment
        for indexes in pages.values()
        for fragment in indexes["fragments"].values()
    }
    consumed = {
        str(ref)
        for composite in composites.values()
        for ref in [
            *composite.get("member_fragment_refs", []),
            *composite.get("supporting_closure_fragment_refs", []),
        ]
    }
    targets = []
    for page_ref, indexes in sorted(pages.items()):
        for fragment in indexes["fragments"].values():
            if str(fragment["id"]) in consumed:
                continue
            targets.append(
                {
                    "id": str(fragment["id"]),
                    "page_ref": page_ref,
                    "fragment_refs": [str(fragment["id"])],
                    "points_display": deepcopy(
                        fragment.get("geometry", {}).get("points_display", [])
                    ),
                    "path_display_points": fragment.get("geometry", {}).get(
                        "path_display_points"
                    ),
                    "projected_path_m": fragment.get(
                        "local_metric_observation", {}
                    ).get("projected_path_m"),
                }
            )
    for composite_ref, composite in sorted(composites.items()):
        member_refs = sorted(str(ref) for ref in composite["member_fragment_refs"])
        projected = [
            fragments[ref].get("local_metric_observation", {}).get(
                "projected_path_m"
            )
            for ref in member_refs
        ]
        targets.append(
            {
                "id": composite_ref,
                "page_ref": str(composite["page_ref"]),
                "fragment_refs": member_refs,
                "points_display": deepcopy(
                    composite.get("derived_geometry", {}).get(
                        "centreline_points_display", []
                    )
                ),
                "path_display_points": composite.get("derived_geometry", {}).get(
                    "projected_path_display_points"
                ),
                "projected_path_m": (
                    None
                    if any(value is None for value in projected)
                    else round(sum(float(value) for value in projected) / len(projected), 8)
                ),
            }
        )
    return sorted(targets, key=lambda row: row["id"])


def _target_attribute_value(
    fragment_refs: Iterable[str],
    relation_type: str,
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    values = []
    refs = []
    reasons = []
    for fragment_ref in fragment_refs:
        value, relation_refs, reason = _attribute_value(
            str(fragment_ref), relation_type, relations_by_fragment
        )
        refs.extend(relation_refs)
        if reason:
            reasons.append(reason)
        else:
            values.append(value)
    semantic = {_semantic_key(value) for value in values}
    if reasons or len(semantic) != 1:
        return None, sorted(set(refs)), reasons[0] if reasons else f"conflicting_{relation_type}"
    return values[0], sorted(set(refs)), None


def _resolved_3d_segments(
    pages: Mapping[str, Mapping[str, Any]],
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    composites: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for target in _centreline_targets(pages, composites):
            system, system_refs, system_reason = _target_attribute_value(
                target["fragment_refs"], "route_system", relations_by_fragment
            )
            size, size_refs, size_reason = _target_attribute_value(
                target["fragment_refs"], "route_size", relations_by_fragment
            )
            elevation, elevation_refs, elevation_reason = _target_attribute_value(
                target["fragment_refs"], "route_elevation", relations_by_fragment
            )
            if system_reason or size_reason or elevation_reason:
                continue
            if elevation.get("basis") != "centreline":
                continue
            z = _metres(elevation.get("value"), elevation.get("unit"))
            if z is None or target.get("projected_path_m") is None:
                continue
            display_length = float(target.get("path_display_points") or 0.0)
            if display_length <= 0:
                continue
            metric_factor = float(target["projected_path_m"]) / display_length
            points_3d = [
                [
                    round(float(point[0]) * metric_factor, 8),
                    round(float(point[1]) * metric_factor, 8),
                    round(z, 8),
                ]
                for point in target["points_display"]
            ]
            output.append(
                {
                    "record_type": "mep_resolved_3d_centreline_segment",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("mep_resolved_3d_centreline", target["id"]),
                    "page_ref": target["page_ref"],
                    "route_target_ref": target["id"],
                    "source_fragment_refs": target["fragment_refs"],
                    "system": system,
                    "size": size,
                    "centreline_elevation_m": round(z, 8),
                    "points_3d_m": points_3d,
                    "projected_2d_length_m": target["projected_path_m"],
                    "resolved_3d_length_m": target["projected_path_m"],
                    "relation_refs": sorted(
                        [*system_refs, *size_refs, *elevation_refs]
                    ),
                    "coordinate_state": "page_local_xy_with_resolved_centreline_z",
                    "state": "derived",
                    "quantity_eligible": False,
                }
            )
    return sorted(output, key=lambda row: row["id"])


def _partial_2_5d_segments(
    pages: Mapping[str, Mapping[str, Any]],
    relations_by_fragment: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    composites: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for target in _centreline_targets(pages, composites):
            system, system_refs, system_reason = _target_attribute_value(
                target["fragment_refs"], "route_system", relations_by_fragment
            )
            size, size_refs, size_reason = _target_attribute_value(
                target["fragment_refs"], "route_size", relations_by_fragment
            )
            elevation, elevation_refs, elevation_reason = _target_attribute_value(
                target["fragment_refs"], "route_elevation", relations_by_fragment
            )
            if system_reason or size_reason or elevation_reason:
                continue
            if elevation.get("basis") == "centreline":
                continue
            elevation_m = _metres(elevation.get("value"), elevation.get("unit"))
            display_length = float(target.get("path_display_points") or 0.0)
            if elevation_m is None or target.get("projected_path_m") is None or display_length <= 0:
                continue
            metric_factor = float(target["projected_path_m"]) / display_length
            output.append(
                {
                    "record_type": "mep_partial_2_5d_centreline_segment",
                    "record_version": SCHEMA_VERSION,
                    "id": _stable_id("mep_partial_2_5d_centreline", target["id"]),
                    "page_ref": target["page_ref"],
                    "route_target_ref": target["id"],
                    "source_fragment_refs": target["fragment_refs"],
                    "system": system,
                    "size": size,
                    "points_xy_m": [
                        [
                            round(float(point[0]) * metric_factor, 8),
                            round(float(point[1]) * metric_factor, 8),
                        ]
                        for point in target["points_display"]
                    ],
                    "elevation_reference_basis": elevation.get("basis"),
                    "elevation_reference_m": round(elevation_m, 8),
                    "centreline_elevation_m": None,
                    "projected_2d_length_m": target["projected_path_m"],
                    "resolved_3d_length_m": None,
                    "relation_refs": sorted([*system_refs, *size_refs, *elevation_refs]),
                    "reason": "elevation_reference_does_not_resolve_centreline_offset",
                    "state": "unknown",
                    "quantity_eligible": False,
                }
            )
    return sorted(output, key=lambda row: row["id"])


def _unresolved_vertical_spans(bindings: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for relation in bindings.get("relations", []):
        if relation.get("state") != "accepted" or relation.get("relation_type") != "riser_drop":
            continue
        output.append(
            {
                "record_type": "mep_unresolved_vertical_span",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_unresolved_vertical_span", relation["id"]),
                "page_ref": relation.get("page_ref"),
                "riser_drop_relation_ref": relation["id"],
                "endpoint_or_vertex_refs": deepcopy(relation.get("target_refs", [])),
                "vertical_extent_m": None,
                "reason": "second_endpoint_elevation_is_unresolved",
                "state": "unknown",
                "quantity_eligible": False,
            }
        )
    return sorted(output, key=lambda row: row["id"])


def _run_hypotheses(
    continuations: list[Mapping[str, Any]],
    canonical_segments: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parent = {}

    def find(ref: str) -> str:
        parent.setdefault(ref, ref)
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    relation_refs: defaultdict[str, set[str]] = defaultdict(set)
    canonical_by_target = {
        ref: row["id"]
        for row in canonical_segments
        for ref in row.get("source_route_target_refs", row["source_fragment_refs"])
    }
    for row in canonical_segments:
        refs = list(row.get("source_route_target_refs", row["source_fragment_refs"]))
        union(refs[0], refs[1])
        relation_refs[refs[0]].update(row.get("overlap_duplicate_relation_refs", [row["overlap_duplicate_relation_ref"]]))
    accepted_continuation_refs = set()
    for row in continuations:
        if row.get("state") != "accepted":
            continue
        left, right = str(row["source_fragment_ref"]), str(row["target_fragment_ref"])
        union(left, right)
        relation_refs[left].add(str(row["id"]))
        accepted_continuation_refs.add(str(row["id"]))
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for ref in parent:
        groups[find(ref)].append(ref)
    output = []
    for root, refs in sorted(groups.items()):
        refs = sorted(refs)
        joined = sorted(
            {
                relation_ref
                for ref in refs
                for relation_ref in relation_refs.get(ref, set())
            }
        )
        # Multiple page occurrences of one duplicated interval establish one
        # canonical segment, not a run.  Until nonduplicated adjacent path
        # intervals are explicitly represented, only a certified endpoint
        # continuation can assemble a run.
        if not accepted_continuation_refs.intersection(joined):
            continue
        output.append(
            {
                "record_type": "mep_physical_run_hypothesis",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id("mep_physical_run_hypothesis", refs, joined),
                "source_fragment_occurrence_refs": refs,
                "canonical_projected_segment_refs": sorted(
                    {canonical_by_target[ref] for ref in refs if ref in canonical_by_target}
                ),
                "accepted_relation_refs": joined,
                "physical_run_identity_established": True,
                "installed_length_emitted": False,
                "state": "derived",
                "quantity_eligible": False,
            }
        )
    return output


def build_mep_cross_sheet_runs(
    *,
    sheet_registry: Mapping[str, Any],
    route_graph: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    endpoint_tolerance_display_points: float = 1.0,
    duplicate_tolerance_display_points: float = 1.0,
) -> dict[str, Any]:
    """Build M5 candidates and quantity-ineligible physical-run hypotheses."""

    if sheet_registry.get("layer") != "mep_sheet_coordinate_registry":
        raise ValueError("sheet_registry must be the M1 registry")
    route_errors = validate_mep_route_graph(route_graph)
    if route_errors:
        raise ValueError("invalid M3 route payload:\n" + "\n".join(route_errors))
    binding_errors = validate_mep_attribute_bindings(attribute_bindings)
    if binding_errors:
        raise ValueError("invalid M4 binding payload:\n" + "\n".join(binding_errors))
    document_keys = {
        str(payload.get("document", {}).get("document_key"))
        for payload in (sheet_registry, route_graph, attribute_bindings)
        if payload.get("document", {}).get("document_key")
    }
    if len(document_keys) > 1:
        raise ValueError("M1, M3, and M4 document keys do not match")

    pages = _page_indexes(route_graph)
    registrations, adjacency = _registration_edges(sheet_registry)
    relations_by_fragment, continuations = _relation_indexes(attribute_bindings)
    composites, composite_by_fragment = _composite_indexes(attribute_bindings)
    continuation_candidates = _continuation_candidates(
        pages,
        registrations,
        adjacency,
        continuations,
        relations_by_fragment,
        endpoint_tolerance_display_points,
    )
    duplicate_candidates = _duplicate_candidates(
        pages,
        registrations,
        adjacency,
        relations_by_fragment,
        composites,
        composite_by_fragment,
        duplicate_tolerance_display_points,
    )
    occurrences = _projected_occurrences(pages, composites)
    canonical_segments = _canonical_segments(duplicate_candidates, occurrences)
    resolved_3d = _resolved_3d_segments(pages, relations_by_fragment, composites)
    partial_2_5d = _partial_2_5d_segments(pages, relations_by_fragment, composites)
    vertical_spans = _unresolved_vertical_spans(attribute_bindings)
    runs = _run_hypotheses(continuation_candidates, canonical_segments)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(route_graph.get("document", {}))),
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _canonical_sha256(sheet_registry),
        },
        "m3_contract_ref": {
            "layer": route_graph.get("layer"),
            "schema_version": route_graph.get("schema_version"),
            "payload_sha256": _canonical_sha256(route_graph),
        },
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": _canonical_sha256(attribute_bindings),
        },
        "continuation_candidates": continuation_candidates,
        "overlap_duplicate_candidates": duplicate_candidates,
        "projected_route_occurrences": occurrences,
        "canonical_projected_segments": canonical_segments,
        "partial_2_5d_centreline_segments": partial_2_5d,
        "resolved_3d_centreline_segments": resolved_3d,
        "unresolved_vertical_spans": vertical_spans,
        "physical_run_hypotheses": runs,
        "summary": {
            "continuation_candidate_count": len(continuation_candidates),
            "accepted_continuation_count": sum(
                row["state"] == "accepted" for row in continuation_candidates
            ),
            "overlap_duplicate_candidate_count": len(duplicate_candidates),
            "accepted_overlap_duplicate_count": sum(
                row["state"] == "accepted" for row in duplicate_candidates
            ),
            "projected_route_occurrence_count": len(occurrences),
            "canonical_projected_segment_count": len(canonical_segments),
            "partial_2_5d_centreline_segment_count": len(partial_2_5d),
            "resolved_3d_centreline_segment_count": len(resolved_3d),
            "unresolved_vertical_span_count": len(vertical_spans),
            "physical_run_hypothesis_count": len(runs),
        },
        "exchange_contract": {
            "mutual_unique_continuation_required": True,
            "accepted_m1_registration_required": True,
            "compatible_system_size_elevation_required": True,
            "endpoint_bound_continuation_evidence_required": True,
            "transform_cycle_consistency_required": True,
            "overlap_canonicalized_before_run_assembly": True,
            "duplicate_interval_alone_does_not_establish_run": True,
            "source_occurrences_separate_from_canonical_segments": True,
            "projected_2d_and_resolved_3d_lengths_separate": True,
            "unresolved_vertical_spans_preserved": True,
            "installed_length_emitted": False,
            "fitting_count_emitted": False,
            "confirmed_clash_established": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_cross_sheet_runs(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def _walk_items(
    value: object, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk_items(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, (*path, f"[{index}]"))


def validate_mep_cross_sheet_runs(payload: Mapping[str, Any]) -> list[str]:
    """Validate M5 reference closure and its strict M7 quantity boundary."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for name in ("m1_contract_ref", "m3_contract_ref", "m4_contract_ref"):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    allowed_false = {
        "installed_length_emitted",
        "fitting_count_emitted",
        "confirmed_clash_established",
    }
    forbidden = {
        "installed_length",
        "fitting_count",
        "confirmed_clash",
        "clash_status",
        "takeoff",
        "quantity",
    }
    for path, key, value in _walk_items(payload):
        location = ".".join((*path, key))
        if key in forbidden:
            errors.append(f"{location}: forbidden M5/M7 output")
        if key in allowed_false and value is not False:
            errors.append(f"{location}: authority flag must remain false")
        if key == "quantity_eligible" and value is not False:
            errors.append(f"{location}: quantity_eligible must remain false")
    continuations = list(payload.get("continuation_candidates", []))
    duplicates = list(payload.get("overlap_duplicate_candidates", []))
    continuation_refs = {str(row.get("id")) for row in continuations}
    duplicate_refs = {str(row.get("id")) for row in duplicates}
    if len(continuation_refs) != len(continuations):
        errors.append("continuation candidate IDs must be unique")
    if len(duplicate_refs) != len(duplicates):
        errors.append("overlap duplicate candidate IDs must be unique")
    for row in [*continuations, *duplicates]:
        if row.get("state") == "accepted":
            if row.get("reasons") or not all(row.get("certificates", {}).values()):
                errors.append(f"{row.get('id')}: accepted candidate has an open gate")
        elif row.get("state") == "abstained":
            if not row.get("reasons"):
                errors.append(f"{row.get('id')}: abstention lacks reason")
        else:
            errors.append(f"{row.get('id')}: invalid candidate state")
    accepted_duplicates = {
        str(row["id"]) for row in duplicates if row.get("state") == "accepted"
    }
    occurrences = {
        str(row["id"]): row for row in payload.get("projected_route_occurrences", [])
    }
    segments = list(payload.get("canonical_projected_segments", []))
    segment_refs = {str(row.get("id")) for row in segments}
    for segment in segments:
        if str(segment.get("overlap_duplicate_relation_ref")) not in accepted_duplicates:
            errors.append(f"{segment.get('id')}: canonical segment lacks accepted duplicate")
        if any(str(ref) not in occurrences for ref in segment.get("source_page_occurrence_refs", [])):
            errors.append(f"{segment.get('id')}: unknown source-page occurrence")
        if segment.get("physical_segment_identity_established") is not True:
            errors.append(f"{segment.get('id')}: accepted overlap lacks segment identity")
    for run in payload.get("physical_run_hypotheses", []):
        if any(str(ref) not in segment_refs for ref in run.get("canonical_projected_segment_refs", [])):
            errors.append(f"{run.get('id')}: unknown canonical segment")
        if run.get("installed_length_emitted") is not False:
            errors.append(f"{run.get('id')}: run emitted installed length")
    contract = payload.get("exchange_contract", {})
    for key in (
        "mutual_unique_continuation_required",
        "accepted_m1_registration_required",
        "compatible_system_size_elevation_required",
        "endpoint_bound_continuation_evidence_required",
        "transform_cycle_consistency_required",
        "overlap_canonicalized_before_run_assembly",
        "duplicate_interval_alone_does_not_establish_run",
        "source_occurrences_separate_from_canonical_segments",
        "projected_2d_and_resolved_3d_lengths_separate",
        "unresolved_vertical_spans_preserved",
    ):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in (
        "installed_length_emitted",
        "fitting_count_emitted",
        "confirmed_clash_established",
        "schedule_values_used",
        "quantity_eligible",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    return errors
