"""Generic M3.5 certificates for outlined projected MEP routes.

Two native strokes may describe the edges of one drawn route envelope.  This
layer preserves those strokes and publishes a derived page-local corridor and
centreline only when geometry, topology, style, ownership, provenance, an
explicit envelope-closing feature, and mutual uniqueness all close.  Parallel
distance alone is intentionally insufficient.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_route_observations import validate_mep_route_graph


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_outlined_route_composites"
METHOD_VERSION = "1.0.0"


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{kind}.{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _length(points: Sequence[Sequence[object]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _interpolate(points: Sequence[Sequence[object]], fraction: float) -> list[float]:
    lengths = [math.dist(left, right) for left, right in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 1e-9:
        return [float(points[0][0]), float(points[0][1])]
    target = max(0.0, min(1.0, fraction)) * total
    walked = 0.0
    for index, segment_length in enumerate(lengths):
        if walked + segment_length >= target or index == len(lengths) - 1:
            ratio = 0.0 if segment_length <= 1e-9 else (target - walked) / segment_length
            left, right = points[index], points[index + 1]
            return [
                float(left[0]) + ratio * (float(right[0]) - float(left[0])),
                float(left[1]) + ratio * (float(right[1]) - float(left[1])),
            ]
        walked += segment_length
    return [float(points[-1][0]), float(points[-1][1])]


def _samples(points: Sequence[Sequence[object]], count: int = 17) -> list[list[float]]:
    return [_interpolate(points, index / (count - 1)) for index in range(count)]


def _angle(left: Sequence[object], right: Sequence[object]) -> float:
    return math.atan2(float(right[1]) - float(left[1]), float(right[0]) - float(left[0]))


def _angle_difference(left: float, right: float) -> float:
    difference = abs(left - right) % (2.0 * math.pi)
    return min(difference, 2.0 * math.pi - difference)


def _style_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_width = left.get("width_display_points")
    right_width = right.get("width_display_points")
    if left_width is None or right_width is None:
        return False
    tolerance = max(0.15, 0.2 * max(float(left_width), float(right_width)))
    return (
        abs(float(left_width) - float(right_width)) <= tolerance
        and left.get("dash_pattern") == right.get("dash_pattern")
    )


def _exact_representation_key(fragment: Mapping[str, Any]) -> tuple[Any, ...]:
    """Group only byte-equivalent projected stroke representations.

    Every native fragment remains in the certificate.  The group prevents
    repeated paints of the same rail from manufacturing several geometric
    pairings; different coordinates, styles, ownership or provenance remain
    independent competitors.
    """
    points = tuple(tuple(float(value) for value in point)
                   for point in fragment.get("geometry", {}).get(
                       "points_display", []))
    geometry = min(points, tuple(reversed(points))) if points else points
    style = fragment.get("style", {})
    provenance = fragment.get("provenance", {})
    ownership = fragment.get("ownership", {})
    return (
        fragment.get("geometry", {}).get("kind"), geometry,
        style.get("width_display_points"), str(style.get("dash_pattern")),
        str(style.get("stroke")), str(style.get("fill")),
        ownership.get("page_ref"), ownership.get("view_scope_ref"),
        ownership.get("package_ref"), provenance.get("method"),
    )


def _provenance_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_provenance = left.get("provenance", {})
    right_provenance = right.get("provenance", {})
    return (
        left.get("source_kind") == right.get("source_kind")
        and left_provenance.get("method") == right_provenance.get("method")
        and left_provenance.get("coordinate_space")
        == right_provenance.get("coordinate_space")
    )


def _ownership_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("page_ref", "view_scope_ref", "package_ref")
    return all(left.get("ownership", {}).get(key) == right.get("ownership", {}).get(key) for key in keys)


def _prepare_polyline(row: Mapping[str, Any]) -> dict[str, Any]:
    """Pure geometry preparation, local to one builder invocation/page.

    M3 already normalized the style. Retain that interpretation unchanged;
    only prepare the existing float width used by pair metrics. No input is
    mutated, and no candidate or certificate is cached.
    """
    points = row["geometry"]["points_display"]
    return {"samples": _samples(points), "length": _length(points),
            "width": float(row.get("style", {}).get("width_display_points") or 0.0)}


def _parallel_metrics(left: Mapping[str, Any], right: Mapping[str, Any], *,
                      left_prepared=None, right_prepared=None) -> dict[str, Any]:
    # Existing callers outside M3.5 retain the ordinary uncached call form.
    if left_prepared is None:
        left_prepared = _prepare_polyline(left)
    if right_prepared is None:
        right_prepared = _prepare_polyline(right)
    left_samples = left_prepared["samples"]
    forward = right_prepared["samples"]
    reverse = list(reversed(forward))
    forward_cost = sum(math.dist(a, b) for a, b in zip(left_samples, forward))
    reverse_cost = sum(math.dist(a, b) for a, b in zip(left_samples, reverse))
    right_samples = reverse if reverse_cost < forward_cost else forward
    reversed_right = reverse_cost < forward_cost
    separations = [math.dist(a, b) for a, b in zip(left_samples, right_samples)]
    mean = sum(separations) / len(separations)
    maximum_deviation = max(abs(value - mean) for value in separations)
    tangent_differences = [
        math.degrees(_angle_difference(_angle(a, b), _angle(c, d)))
        for a, b, c, d in zip(
            left_samples,
            left_samples[1:],
            right_samples,
            right_samples[1:],
        )
    ]
    left_length = left_prepared["length"]
    right_length = right_prepared["length"]
    width = max(left_prepared["width"], right_prepared["width"])
    return {
        "right_reversed": reversed_right,
        "left_samples": left_samples,
        "right_samples": right_samples,
        "left_length_display_points": left_length,
        "right_length_display_points": right_length,
        "length_ratio": min(left_length, right_length) / max(left_length, right_length, 1e-9),
        "mean_separation_display_points": mean,
        "maximum_separation_deviation_display_points": maximum_deviation,
        "maximum_tangent_difference_degrees": max(tangent_differences, default=180.0),
        "endpoint_separations_display_points": [separations[0], separations[-1]],
        "member_width_display_points": width,
    }


def _shortest_cap_path(
    page: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    metrics: Mapping[str, Any],
    excluded_fragment_refs: Iterable[str] = (),
) -> tuple[str | None, list[str]]:
    """Find a short, style/provenance-compatible path closing matched ends."""

    fragments = {str(row["id"]): row for row in page.get("fragments", [])}
    excluded = {str(left["id"]), str(right["id"]),
                *(str(ref) for ref in excluded_fragment_refs)}
    right_vertices = list(right.get("endpoint_vertex_refs", []))
    if metrics["right_reversed"]:
        right_vertices.reverse()
    end_pairs = list(zip(left.get("endpoint_vertex_refs", []), right_vertices))
    edges: defaultdict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for fragment_ref, fragment in fragments.items():
        if fragment_ref in excluded:
            continue
        if not _style_compatible(left.get("style", {}), fragment.get("style", {})):
            continue
        if not _provenance_compatible(left, fragment) or not _ownership_compatible(left, fragment):
            continue
        vertices = list(fragment.get("endpoint_vertex_refs", []))
        if len(vertices) != 2 or vertices[0] == vertices[1]:
            continue
        segment_length = float(fragment.get("geometry", {}).get("path_display_points") or 0.0)
        edges[str(vertices[0])].append((str(vertices[1]), fragment_ref, segment_length))
        edges[str(vertices[1])].append((str(vertices[0]), fragment_ref, segment_length))

    limit = max(
        2.5 * float(metrics["mean_separation_display_points"]),
        4.0 * float(metrics["member_width_display_points"]),
        1.0,
    )
    found: list[tuple[float, int, list[str]]] = []
    for end_index, (source, target) in enumerate(end_pairs):
        queue = deque([(str(source), 0.0, tuple())])
        best = {str(source): 0.0}
        while queue:
            vertex, distance, path = queue.popleft()
            if vertex == str(target) and path:
                found.append((distance, end_index, list(path)))
                break
            for next_vertex, fragment_ref, edge_length in edges.get(vertex, []):
                next_distance = distance + edge_length
                if next_distance > limit or next_distance >= best.get(next_vertex, math.inf):
                    continue
                best[next_vertex] = next_distance
                queue.append((next_vertex, next_distance, (*path, fragment_ref)))
    if not found:
        return None, []
    _distance, end_index, path = min(found, key=lambda row: (row[0], row[1], row[2]))
    return ("start_cap_path" if end_index == 0 else "end_cap_path"), path


def _corridor_geometry(metrics: Mapping[str, Any]) -> dict[str, Any]:
    left = metrics["left_samples"]
    right = metrics["right_samples"]
    centreline = [
        [round((a[0] + b[0]) / 2.0, 6), round((a[1] + b[1]) / 2.0, 6)]
        for a, b in zip(left, right)
    ]
    # Keep only direction-changing samples and the endpoints.
    simplified = [centreline[0]]
    for index in range(1, len(centreline) - 1):
        before = _angle(centreline[index - 1], centreline[index])
        after = _angle(centreline[index], centreline[index + 1])
        if math.degrees(_angle_difference(before, after)) > 0.25:
            simplified.append(centreline[index])
    simplified.append(centreline[-1])
    corridor = [
        [round(point[0], 6), round(point[1], 6)]
        for point in [*left, *reversed(right)]
    ]
    return {
        "centreline_points_display": simplified,
        "corridor_boundary_points_display": corridor,
        "corridor_width_display_points": round(float(metrics["mean_separation_display_points"]), 6),
        "projected_path_display_points": round(_length(simplified), 6),
    }


def build_mep_outlined_route_composites(
    *,
    route_graph: Mapping[str, Any],
    maximum_separation_to_length_ratio: float = 0.08,
    collapse_exact_overpaint_representations: bool = False,
) -> dict[str, Any]:
    """Certify mutually unique outlined route envelopes from frozen M3.

    The default arguments preserve the established generic replay. Rectangular
    duct integrations may explicitly admit wider envelopes and exact authored
    overpaint aliases; every aliased M3/native reference remains in the
    certificate and in the mutual-uniqueness check.
    """

    errors = validate_mep_route_graph(route_graph)
    if errors:
        raise ValueError("invalid M3 route payload:\n" + "\n".join(errors))
    ratio = float(maximum_separation_to_length_ratio)
    if not 0.0 < ratio <= 0.2:
        raise ValueError("outline separation ratio must be in (0, 0.2]")
    candidates: list[dict[str, Any]] = []
    supported_by_member: defaultdict[str, set[str]] = defaultdict(set)
    for page in route_graph.get("pages", []):
        source_rows = [
            row
            for row in page.get("fragments", [])
            if row.get("geometry", {}).get("kind") == "line"
            and len(row.get("geometry", {}).get("points_display", [])) >= 2
        ]
        if collapse_exact_overpaint_representations:
            representation_groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in source_rows:
                representation_groups[_exact_representation_key(row)].append(row)
            grouped_rows = [sorted(group, key=lambda row: str(row["id"]))
                            for group in representation_groups.values()]
        else:
            grouped_rows = [[row] for row in source_rows]
        grouped_rows.sort(key=lambda group: str(group[0]["id"]))
        rows = [group[0] for group in grouped_rows]
        aliases_by_ref = {str(group[0]["id"]): group for group in grouped_rows}
        # Optional representation grouping preserves every source ID while
        # preventing exact authored overpaints from manufacturing pair choices.
        # A passing first gate has mean separation <= min(lengths)/5, so at
        # least one paired sample is within that distance. Therefore the two
        # length/5-expanded bounds must overlap. Batch by those conservative
        # bounds while retaining every pair that could reach the certificate.
        prepared = [_prepare_polyline(row) for row in rows] if len(rows) > 1 else []
        pair_cells: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        possible_pairs = set()
        cell_size = 64.0
        filter_geometry = [geometry if geometry is not None else {
            'length': _length(row['geometry']['points_display'])}
            for row, geometry in zip(rows, prepared)]
        for index, (row, geometry) in enumerate(zip(rows, filter_geometry)):
            points = row['geometry']['points_display']
            radius = geometry['length'] / 5.0
            box = [min(point[0] for point in points) - radius,
                   min(point[1] for point in points) - radius,
                   max(point[0] for point in points) + radius,
                   max(point[1] for point in points) + radius]
            cells = [(x, y)
                     for x in range(math.floor(box[0] / cell_size),
                                    math.floor(box[2] / cell_size) + 1)
                     for y in range(math.floor(box[1] / cell_size),
                                    math.floor(box[3] / cell_size) + 1)]
            for cell in cells:
                for other in pair_cells[cell]:
                    left_length = filter_geometry[other]['length']
                    right_length = geometry['length']
                    if min(left_length, right_length) / max(
                            left_length, right_length, 1e-9) >= .9:
                        possible_pairs.add((other, index))
                pair_cells[cell].append(index)
        for index, right_index in sorted(possible_pairs):
                left = rows[index]
                right = rows[right_index]
                provenance_ok = _provenance_compatible(left, right)
                style_ok = _style_compatible(left.get("style", {}), right.get("style", {}))
                ownership_ok = _ownership_compatible(left, right)
                metrics = _parallel_metrics(left, right, left_prepared=prepared[index],
                                            right_prepared=prepared[right_index])
                mean = float(metrics["mean_separation_display_points"])
                minimum_length = min(
                    float(metrics["left_length_display_points"]),
                    float(metrics["right_length_display_points"]),
                )
                if (
                    minimum_length
                    < max(
                        10.0 * float(metrics["member_width_display_points"]),
                        5.0 * mean,
                    )
                    or float(metrics["length_ratio"]) < 0.9
                ):
                    continue
                persistent = (
                    mean > max(0.1, 0.15 * float(metrics["member_width_display_points"]))
                    and mean <= max(
                        12.0 * float(metrics["member_width_display_points"]),
                        ratio * min(
                            float(metrics["left_length_display_points"]),
                            float(metrics["right_length_display_points"]),
                        ),
                    )
                    and float(metrics["length_ratio"]) >= 0.98
                    and float(metrics["maximum_separation_deviation_display_points"])
                    <= max(0.35, 0.08 * mean)
                )
                synchronized = (
                    float(metrics["maximum_tangent_difference_degrees"]) <= 2.0
                    and all(abs(float(value) - mean) <= max(0.35, 0.08 * mean) for value in metrics["endpoint_separations_display_points"])
                )
                member_refs = sorted((str(left["id"]), str(right["id"])))
                representation_member_groups = [
                    sorted(str(row["id"]) for row in aliases_by_ref[ref])
                    for ref in member_refs]
                all_member_rows = [row for ref in member_refs
                                   for row in aliases_by_ref[ref]]
                alias_refs = {ref for group in representation_member_groups
                              for ref in group}
                closure_kind, closure_refs = _shortest_cap_path(
                    page, left, right, metrics,
                    excluded_fragment_refs=alias_refs)
                closure_ok = closure_kind is not None
                locally_supported = all(
                    (provenance_ok, style_ok, ownership_ok, persistent, synchronized, closure_ok)
                )
                candidate_id = _stable_id("mep_outlined_route_composite", page["page_ref"], member_refs)
                reasons = []
                gates = (
                    (provenance_ok, "incompatible_member_provenance"),
                    (style_ok, "incompatible_member_style"),
                    (ownership_ok, "incompatible_member_ownership"),
                    (persistent, "parallel_separation_is_not_persistent"),
                    (synchronized, "turns_or_endpoints_are_not_synchronized"),
                    (closure_ok, "no_envelope_closing_feature"),
                )
                reasons.extend(reason for passed, reason in gates if not passed)
                candidate = {
                    "record_type": "mep_outlined_route_composite_candidate",
                    "record_version": SCHEMA_VERSION,
                    "id": candidate_id,
                    "page_ref": str(page["page_ref"]),
                    "member_fragment_refs": member_refs,
                    "member_source_primitive_refs": sorted(
                        str(row["source_primitive_ref"])
                        for row in all_member_rows
                    ),
                    "supporting_closure_fragment_refs": sorted(closure_refs),
                    "closure_kind": closure_kind,
                    "geometry_metrics": {
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in metrics.items()
                        if key not in {"left_samples", "right_samples"}
                    },
                    "stroke_color_match": left.get("style", {}).get("stroke")
                    == right.get("style", {}).get("stroke"),
                    "color_only_identity_eligible": False,
                    "derived_geometry": _corridor_geometry(metrics),
                    "method": {
                        "name": "closed_parallel_route_envelope",
                        "version": METHOD_VERSION,
                        **({"parameters": {
                            "maximum_separation_to_length_ratio": ratio,
                            "collapse_exact_overpaint_representations": True,
                        }} if (ratio != 0.08
                               or collapse_exact_overpaint_representations)
                           else {}),
                    },
                    "certificates": {
                        "compatible_provenance": provenance_ok,
                        "compatible_non_color_style": style_ok,
                        "compatible_page_view_package_ownership": ownership_ok,
                        "persistent_parallel_separation": persistent,
                        "synchronized_turns_and_endpoints": synchronized,
                        "explicit_envelope_closing_feature": closure_ok,
                        "mutual_unique_pairing": False,
                    },
                    "state": "abstained",
                    "epistemic_state": "unknown",
                    "reasons": reasons,
                    "native_strokes_preserved": True,
                    "physical_route_identity_established": False,
                    "quantity_eligible": False,
                }
                if any(len(group) > 1 for group in representation_member_groups):
                    candidate["equivalent_member_fragment_groups"] = (
                        representation_member_groups)
                    candidate["exact_duplicate_representation_count"] = sum(
                        len(group) - 1 for group in representation_member_groups)
                candidates.append(candidate)
                if locally_supported:
                    for member_ref in {
                            ref for group in representation_member_groups
                            for ref in group}:
                        supported_by_member[member_ref].add(candidate_id)

    for candidate in candidates:
        if candidate["reasons"]:
            continue
        represented_refs = [
            ref for group in candidate.get("equivalent_member_fragment_groups",
                                           [candidate["member_fragment_refs"]])
            for ref in group]
        unique = all(
            supported_by_member[member_ref] == {candidate["id"]}
            for member_ref in represented_refs
        )
        candidate["certificates"]["mutual_unique_pairing"] = unique
        if unique:
            candidate["state"] = "accepted"
            candidate["epistemic_state"] = "derived"
        else:
            candidate["reasons"] = ["equally_supported_alternative_pairing"]

    candidates.sort(key=lambda row: (row["page_ref"], row["member_fragment_refs"], row["id"]))
    accepted = [deepcopy(row) for row in candidates if row["state"] == "accepted"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(route_graph.get("document", {}))),
        "m3_contract_ref": {
            "layer": route_graph.get("layer"),
            "schema_version": route_graph.get("schema_version"),
            "payload_sha256": _canonical_sha256(route_graph),
        },
        "candidates": candidates,
        "accepted_composites": accepted,
        "summary": {
            "candidate_count": len(candidates),
            "accepted_composite_count": len(accepted),
            "abstained_candidate_count": sum(row["state"] == "abstained" for row in candidates),
            "state_counts": dict(sorted(Counter(row["state"] for row in candidates).items())),
        },
        "exchange_contract": {
            "native_strokes_preserved": True,
            "parallelism_or_distance_alone_sufficient": False,
            "envelope_closure_required": True,
            "mutual_unique_pairing_required": True,
            "physical_route_identity_established": False,
            "installed_length_emitted": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    validation_errors = validate_mep_outlined_route_composites(payload)
    if validation_errors:
        raise ValueError("\n".join(validation_errors))
    return payload


def _walk(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, f"[{index}]"))


def validate_mep_outlined_route_composites(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    digest = payload.get("m3_contract_ref", {}).get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("m3_contract_ref.payload_sha256 must be a SHA-256 digest")
    candidates = list(payload.get("candidates", []))
    accepted = list(payload.get("accepted_composites", []))
    ids = [str(row.get("id")) for row in candidates]
    if len(ids) != len(set(ids)):
        errors.append("candidate IDs must be unique")
    by_id = {str(row.get("id")): row for row in candidates}
    if any(str(row.get("id")) not in by_id or by_id[str(row.get("id"))] != row for row in accepted):
        errors.append("accepted_composites must replay from candidates")
    for row in candidates:
        if row.get("state") == "accepted":
            if row.get("reasons") or not all(row.get("certificates", {}).values()):
                errors.append(f"{row.get('id')}: accepted composite has an open gate")
        elif row.get("state") == "abstained":
            if not row.get("reasons"):
                errors.append(f"{row.get('id')}: abstention lacks reason")
        else:
            errors.append(f"{row.get('id')}: invalid state")
        if row.get("native_strokes_preserved") is not True:
            errors.append(f"{row.get('id')}: native strokes are not preserved")
    for path, key, value in _walk(payload):
        location = ".".join((*path, key))
        if key in {"installed_length", "quantity", "takeoff", "physical_continuation"}:
            errors.append(f"{location}: forbidden composite output")
        if key == "quantity_eligible" and value is not False:
            errors.append(f"{location}: quantity_eligible must remain false")
        if key in {"physical_route_identity_established", "installed_length_emitted"} and value is not False:
            errors.append(f"{location}: authority flag must remain false")
    contract = payload.get("exchange_contract", {})
    for key in ("native_strokes_preserved", "envelope_closure_required", "mutual_unique_pairing_required"):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in ("parallelism_or_distance_alone_sufficient", "physical_route_identity_established", "installed_length_emitted", "quantity_eligible"):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    return errors
