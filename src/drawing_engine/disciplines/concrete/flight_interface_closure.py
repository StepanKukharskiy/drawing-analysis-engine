"""Close flight boundary ports only through certified physical interfaces."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import heapq
import itertools
import math
from typing import Any, Mapping


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, kind: str, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), kind, *(str(part) for part in parts))).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _primitive_ref(ref: Any) -> str:
    return str(ref).split(".split[", 1)[0]


def _cluster(values: list[tuple[float, str, list[float]]], tolerance: float) -> list[list[tuple[float, str, list[float]]]]:
    groups: list[list[tuple[float, str, list[float]]]] = []
    for row in sorted(values, key=lambda item: (item[0], item[1])):
        if groups and abs(row[0] - sum(item[0] for item in groups[-1]) / len(groups[-1])) <= tolerance:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _section_stations(local: Mapping[str, Any], scale: float, page_number: int) -> list[dict[str, Any]]:
    x_values, y_values = [], []
    for item in (local.get("local_ownership") or {}).get("attachments", []) or []:
        if item.get("status") != "accepted":
            continue
        ref = str(item.get("dimension_ref"))
        for endpoint in item.get("measured_endpoints", []) or []:
            point = endpoint.get("point_display")
            if not point:
                continue
            row = (float(point[0]), ref, list(map(float, point)))
            x_values.append(row)
            y_values.append((float(point[1]), ref, list(map(float, point))))
    tolerance = max(2.0, 20.0 * scale)
    stations = []
    for axis, values in (("x", x_values), ("y", y_values)):
        for group in _cluster(values, tolerance):
            dimension_refs = sorted({item[1] for item in group})
            coordinate = sum(item[0] for item in group) / len(group)
            # One dimension may independently certify a cut line when both of
            # its measured endpoints lie on that line. Otherwise two distinct
            # dimensions must repeat the station.
            per_dimension = defaultdict(int)
            for _, ref, _ in group:
                per_dimension[ref] += 1
            repeated_by_endpoints = any(count >= 2 for count in per_dimension.values())
            independently_supported = repeated_by_endpoints or len(dimension_refs) >= 2
            if not independently_supported:
                continue
            stations.append(
                {
                    "id": _stable_id(page_number, "certified_section_interface_station", axis, round(coordinate, 6), *dimension_refs),
                    "state": "derived",
                    "axis": axis,
                    "coordinate_display": round(coordinate, 6),
                    "tolerance_points": round(tolerance, 6),
                    "dimension_refs": dimension_refs,
                    "endpoint_observation_count": len(group),
                    "independent_dimension_count": len(dimension_refs),
                    "evidence_refs": dimension_refs,
                    "quantity_eligible": False,
                }
            )
    return sorted(stations, key=lambda item: (item["axis"], item["coordinate_display"], item["id"]))


def _support_graph(chains: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]]:
    graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    points: dict[str, list[float]] = {}
    for chain in chains:
        if chain.get("role") != "support_or_landing_boundary" or not chain.get("boundary_eligible", True):
            continue
        vertices = list(map(str, chain.get("ordered_vertex_refs", []) or []))
        geometry = [list(map(float, point)) for point in chain.get("ordered_points_display", []) or []]
        split_refs = list(map(str, chain.get("split_edge_refs", []) or []))
        if len(vertices) != len(geometry) or len(split_refs) + 1 != len(vertices):
            continue
        for vertex, point in zip(vertices, geometry):
            points[vertex] = point
        for index, edge_ref in enumerate(split_refs):
            left, right = vertices[index], vertices[index + 1]
            length = math.dist(geometry[index], geometry[index + 1])
            edge = {
                "edge_ref": edge_ref,
                "source_edge_ref": _primitive_ref(edge_ref),
                "chain_ref": chain["id"],
                "length_points": length,
            }
            graph[left].append({**edge, "target": right})
            graph[right].append({**edge, "target": left})
    return graph, points


def _shortest_paths(
    graph: Mapping[str, list[dict[str, Any]]],
    points: Mapping[str, list[float]],
    start: str,
) -> dict[str, dict[str, Any]]:
    if start not in points:
        return {start: {"distance_points": 0.0, "unique": True, "vertices": [start], "edges": []}}
    distance = {start: 0.0}
    count = {start: 1}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    queue = [(0.0, start)]
    while queue:
        current_distance, vertex = heapq.heappop(queue)
        if current_distance > distance[vertex] + 1e-9:
            continue
        for edge in graph.get(vertex, []):
            target = str(edge["target"])
            candidate = current_distance + float(edge["length_points"])
            if target not in distance or candidate < distance[target] - 1e-7:
                distance[target] = candidate
                count[target] = count[vertex]
                previous[target] = (vertex, edge)
                heapq.heappush(queue, (candidate, target))
            elif abs(candidate - distance[target]) <= 1e-7:
                count[target] = min(2, count.get(target, 0) + count[vertex])
    output = {}
    for target in distance:
        vertices, edges = [target], []
        current = target
        while current != start and current in previous:
            source, edge = previous[current]
            edges.append(edge)
            vertices.append(source)
            current = source
        vertices.reverse()
        edges.reverse()
        output[target] = {
            "distance_points": round(distance[target], 6),
            "unique": count.get(target, 0) == 1 and current == start,
            "vertices": vertices,
            "edges": edges,
            "points_display": [list(points[vertex]) for vertex in vertices if vertex in points],
        }
    output.setdefault(start, {"distance_points": 0.0, "unique": True, "vertices": [start], "edges": [], "points_display": []})
    return output


def _path_with_port(path: Mapping[str, Any], port: Mapping[str, Any], points: Mapping[str, list[float]]) -> list[list[float]]:
    rows = [list(map(float, point)) for point in path.get("points_display", []) or []]
    port_point = list(map(float, port["point_display"]))
    if not rows:
        return [port_point]
    if math.dist(rows[0], port_point) > 1e-6:
        rows.insert(0, port_point)
    return rows


def _interface_options(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    graph: Mapping[str, list[dict[str, Any]]],
    points: Mapping[str, list[float]],
    stations: list[dict[str, Any]],
    scale: float,
    profile_cap_bound: float,
    plan_certificate: Mapping[str, Any],
    page_number: int,
) -> list[dict[str, Any]]:
    left_vertex, right_vertex = str(left["vertex_ref"]), str(right["vertex_ref"])
    local_points = {**points, left_vertex: list(left["point_display"]), right_vertex: list(right["point_display"])}
    left_paths = _shortest_paths(graph, local_points, left_vertex)
    right_paths = _shortest_paths(graph, local_points, right_vertex)
    plan_refs = [str(plan_certificate.get("id")), *map(str, plan_certificate.get("evidence_refs", []) or [])]
    options = []
    direct = left_paths.get(right_vertex)
    common_port_stations = [
        station
        for station in stations
        if abs(
            float(left["point_display"][0 if station["axis"] == "x" else 1])
            - float(station["coordinate_display"])
        )
        <= float(station["tolerance_points"])
        and abs(
            float(right["point_display"][0 if station["axis"] == "x" else 1])
            - float(station["coordinate_display"])
        )
        <= float(station["tolerance_points"])
    ]
    if direct and direct.get("unique") and direct.get("edges") and common_port_stations:
        station = min(
            common_port_stations,
            key=lambda item: (
                abs(
                    float(left["point_display"][0 if item["axis"] == "x" else 1])
                    - float(item["coordinate_display"])
                )
                + abs(
                    float(right["point_display"][0 if item["axis"] == "x" else 1])
                    - float(item["coordinate_display"])
                ),
                str(item["id"]),
            ),
        )
        source_refs = sorted({str(edge["source_edge_ref"]) for edge in direct["edges"]})
        options.append(
            {
                "id": _stable_id(page_number, "flight_interface_option", left["id"], right["id"], "native", *source_refs),
                "state": "valid",
                "kind": "native_support_or_landing_chain",
                "left_port_ref": left["id"],
                "right_port_ref": right["id"],
                "native_path_edge_refs": source_refs,
                "native_path_chain_refs": sorted({str(edge["chain_ref"]) for edge in direct["edges"]}),
                "ordered_points_display": _path_with_port(direct, left, local_points),
                "derived_physical_interface": None,
                "station_ref": station["id"],
                "path_length_points": direct["distance_points"],
                "same_certified_interface_station": True,
                "evidence_refs": sorted(
                    {
                        left["id"],
                        right["id"],
                        station["id"],
                        *station["evidence_refs"],
                        *source_refs,
                        *plan_refs,
                    }
                ),
                "quantity_eligible": False,
            }
        )
        return options

    for station in stations:
        axis_index = 0 if station["axis"] == "x" else 1
        perpendicular_index = 1 - axis_index
        coordinate = float(station["coordinate_display"])
        tolerance = float(station["tolerance_points"])
        station_local_points = [
            point
            for point in local_points.values()
            if abs(float(point[axis_index]) - coordinate) <= tolerance
        ]
        if len(station_local_points) < 2:
            continue
        perpendicular_values = [float(point[perpendicular_index]) for point in station_local_points]
        station_local_cap_bound = math.hypot(
            max(perpendicular_values) - min(perpendicular_values),
            2.0 * tolerance,
        )
        cap_bound = min(station_local_cap_bound, profile_cap_bound)
        left_nodes = [
            vertex
            for vertex, path in left_paths.items()
            if path.get("unique")
            and vertex in local_points
            and abs(float(local_points[vertex][axis_index]) - coordinate) <= tolerance
        ]
        right_nodes = [
            vertex
            for vertex, path in right_paths.items()
            if path.get("unique")
            and vertex in local_points
            and abs(float(local_points[vertex][axis_index]) - coordinate) <= tolerance
        ]
        for left_node, right_node in itertools.product(left_nodes, right_nodes):
            if left_node == right_node:
                continue
            start, end = local_points[left_node], local_points[right_node]
            cap_length = math.dist(start, end)
            if (
                cap_length <= max(1.0, 5.0 * scale)
                or cap_length > cap_bound + tolerance
            ):
                continue
            left_path, right_path = left_paths[left_node], right_paths[right_node]
            left_edge_ids = {str(edge["edge_ref"]) for edge in left_path["edges"]}
            right_edge_ids = {str(edge["edge_ref"]) for edge in right_path["edges"]}
            if left_edge_ids & right_edge_ids:
                continue
            left_points_ordered = _path_with_port(left_path, left, local_points)
            right_points_ordered = _path_with_port(right_path, right, local_points)
            ordered = [*left_points_ordered, list(end), *list(reversed(right_points_ordered))[1:]]
            source_edges = sorted(
                {
                    str(edge["source_edge_ref"])
                    for edge in [*left_path["edges"], *right_path["edges"]]
                }
            )
            interface_id = _stable_id(page_number, "derived_physical_end_cap", left["id"], right["id"], station["id"], left_node, right_node)
            objective = float(left_path["distance_points"]) + cap_length + float(right_path["distance_points"])
            options.append(
                {
                    "id": _stable_id(page_number, "flight_interface_option", interface_id),
                    "state": "valid",
                    "kind": "derived_physical_end_cap",
                    "left_port_ref": left["id"],
                    "right_port_ref": right["id"],
                    "native_path_edge_refs": source_edges,
                    "native_path_chain_refs": sorted(
                        {
                            str(edge["chain_ref"])
                            for edge in [*left_path["edges"], *right_path["edges"]]
                        }
                    ),
                    "ordered_points_display": ordered,
                    "derived_physical_interface": {
                        "id": interface_id,
                        "record_type": "derived_physical_interface",
                        "state": "derived",
                        "interface_kind": "end_cap",
                        "start_display": list(start),
                        "end_display": list(end),
                        "length_points": round(cap_length, 6),
                        "length_mm": round(cap_length / scale, 3),
                        "section_local_cap_bound_points": round(cap_bound, 6),
                        "station_interface_envelope_points": round(station_local_cap_bound, 6),
                        "profile_endpoint_separation_envelope_points": round(profile_cap_bound, 6),
                        "cap_bound_basis": "section_profile_endpoint_and_station_envelopes",
                        "cut_station_ref": station["id"],
                        "native_drawing_edge": False,
                        "quantity_eligible": False,
                    },
                    "station_ref": station["id"],
                    "path_length_points": round(objective, 6),
                    "same_certified_interface_station": True,
                    "evidence_refs": sorted(
                        {
                            left["id"],
                            right["id"],
                            station["id"],
                            *station["evidence_refs"],
                            *source_edges,
                            *plan_refs,
                        }
                    ),
                    "quantity_eligible": False,
                }
            )
    unique = {item["id"]: item for item in options}
    return [unique[key] for key in sorted(unique)]


def _preferred_options(options: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    native = [item for item in options if item["kind"] == "native_support_or_landing_chain"]
    candidates = native or [item for item in options if item["kind"] == "derived_physical_end_cap"]
    if not candidates:
        return []
    best = min(float(item["path_length_points"]) for item in candidates)
    tolerance = max(1.0, 10.0 * scale)
    return [item for item in candidates if float(item["path_length_points"]) <= best + tolerance]


def _proposal_assignments(
    proposal: Mapping[str, Any],
    role_chains: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, list[dict[str, Any]]],
    points: Mapping[str, list[float]],
    stations: list[dict[str, Any]],
    scale: float,
    plan_certificate: Mapping[str, Any],
    page_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step = role_chains.get(str(proposal.get("stepped_surface_chain_ref")), {})
    waist = role_chains.get(str(proposal.get("waist_underside_chain_ref")), {})
    step_ports = step.get("endpoint_ports", []) or []
    waist_ports = waist.get("endpoint_ports", []) or []
    all_options, assignments = [], []
    if len(step_ports) != 2 or len(waist_ports) != 2:
        return all_options, assignments
    # Bound a derived cap in the section frame itself.  The symmetric endpoint
    # separation envelope is the largest nearest-native-boundary separation at
    # an open end; support chains may reach a certified station, but cannot
    # turn that local closure into an arbitrarily long section chord.
    profile_cap_bound = max(
        max(min(math.dist(left["point_display"], right["point_display"]) for right in waist_ports) for left in step_ports),
        max(min(math.dist(right["point_display"], left["point_display"]) for left in step_ports) for right in waist_ports),
    )
    for mapping in ((0, 1), (1, 0)):
        per_end = []
        for step_index, waist_index in enumerate(mapping):
            options = _interface_options(
                step_ports[step_index],
                waist_ports[waist_index],
                graph,
                points,
                stations,
                scale,
                profile_cap_bound,
                plan_certificate,
                page_number,
            )
            all_options.extend(options)
            per_end.append(_preferred_options(options, scale))
        for selected in itertools.product(*per_end) if all(per_end) else ():
            assignment_id = _stable_id(page_number, "flight_port_assignment", proposal["id"], *mapping, *(item["id"] for item in selected))
            assignments.append(
                {
                    "id": assignment_id,
                    "state": "valid",
                    "proposal_ref": proposal["id"],
                    "step_to_waist_port_mapping": list(mapping),
                    "interface_option_refs": [item["id"] for item in selected],
                    "interface_options": list(selected),
                    "native_interface_count": sum(
                        item["kind"] == "native_support_or_landing_chain"
                        for item in selected
                    ),
                    "assignment_objective_points": round(
                        sum(float(item["path_length_points"]) for item in selected),
                        6,
                    ),
                    "evidence_refs": sorted({proposal["id"], *(ref for item in selected for ref in item["evidence_refs"])}),
                    "quantity_eligible": False,
                }
            )
    if assignments:
        preferred_native_count = max(item["native_interface_count"] for item in assignments)
        native_preferred = [
            item for item in assignments
            if item["native_interface_count"] == preferred_native_count
        ]
        best_objective = min(item["assignment_objective_points"] for item in native_preferred)
        tolerance = max(1.0, 10.0 * scale)
        for item in assignments:
            admissible = (
                item["native_interface_count"] == preferred_native_count
                and item["assignment_objective_points"] <= best_objective + tolerance
            )
            item["preference_status"] = "admissible" if admissible else "dominated"
            item["preference_basis"] = (
                "maximum_native_interface_count_then_minimum_total_interface_path"
            )
    return sorted({item["id"]: item for item in all_options}.values(), key=lambda item: item["id"]), sorted(assignments, key=lambda item: item["id"])


def _interface_midpoint(option: Mapping[str, Any]) -> list[float]:
    points = option.get("ordered_points_display", []) or []
    if not points:
        return [0.0, 0.0]
    return [
        round((float(points[0][0]) + float(points[-1][0])) / 2.0, 6),
        round((float(points[0][1]) + float(points[-1][1])) / 2.0, 6),
    ]


def _classify_global_assignment(assignments: tuple[dict[str, Any], dict[str, Any]], plan_certificate: Mapping[str, Any]) -> dict[str, Any] | None:
    options = [item for assignment in assignments for item in assignment["interface_options"]]
    first, second = assignments
    distances = []
    for left_index, left in enumerate(first["interface_options"]):
        for right_index, right in enumerate(second["interface_options"]):
            distance = math.dist(_interface_midpoint(left), _interface_midpoint(right))
            distances.append((distance, left_index, right_index))
    distances.sort()
    if not distances or (len(distances) > 1 and abs(distances[1][0] - distances[0][0]) <= 1.0):
        return None
    _, first_landing, second_landing = distances[0]
    first_landing_option = first["interface_options"][first_landing]
    second_landing_option = second["interface_options"][second_landing]
    shared_native_edges = sorted(
        set(map(str, first_landing_option.get("native_path_edge_refs", []) or []))
        & set(map(str, second_landing_option.get("native_path_edge_refs", []) or []))
    )
    station_refs = {
        str(item.get("station_ref"))
        for item in (first_landing_option, second_landing_option)
        if item.get("station_ref")
    }
    common_station = (
        len(station_refs) == 1
        and all(item.get("station_ref") for item in (first_landing_option, second_landing_option))
    )
    if not shared_native_edges and not common_station:
        return None
    classified = []
    for assignment_index, assignment in enumerate(assignments):
        landing_index = first_landing if assignment_index == 0 else second_landing
        classified.append(
            {
                **assignment,
                "classified_interfaces": [
                    {
                        "interface_option_ref": option["id"],
                        "classification": "landing_interface" if index == landing_index else "external_support_end",
                        "section_station_ref": option.get("station_ref"),
                        "projected_plan_station_refs": [str(item.get("id")) for item in plan_certificate.get("bands", []) or []],
                    }
                    for index, option in enumerate(assignment["interface_options"])
                ],
            }
        )
    return {
        "assignments": classified,
        "landing_interface_midpoint_residual_points": round(distances[0][0], 6),
        "landing_interface_match": {
            "status": "pass",
            "basis": "shared_native_support_edges" if shared_native_edges else "common_certified_section_station",
            "shared_native_edge_refs": shared_native_edges,
            "common_section_station_ref": next(iter(station_refs)) if common_station else None,
            "interface_option_refs": [first_landing_option["id"], second_landing_option["id"]],
            "finite_area_contact_requires_landing_reconstruction": True,
            "quantity_eligible": False,
        },
        "plan_band_assignment_state": "unresolved_reflection",
    }


def _closed_profile(
    proposal: Mapping[str, Any],
    assignment: Mapping[str, Any],
    role_chains: Mapping[str, Mapping[str, Any]],
    page_number: int,
) -> dict[str, Any]:
    step = role_chains[str(proposal["stepped_surface_chain_ref"])]
    waist = role_chains[str(proposal["waist_underside_chain_ref"])]
    mapping = assignment["step_to_waist_port_mapping"]
    interfaces = assignment["interface_options"]
    boundary = [list(point) for point in step["ordered_points_display"]]
    boundary.extend(list(point) for point in interfaces[1]["ordered_points_display"][1:])
    waist_points = [list(point) for point in waist["ordered_points_display"]]
    if mapping[1] == 1:
        waist_points.reverse()
    boundary.extend(waist_points[1:])
    reverse_first = list(reversed(interfaces[0]["ordered_points_display"]))
    boundary.extend(list(point) for point in reverse_first[1:])
    if len(boundary) > 1 and math.dist(boundary[0], boundary[-1]) <= 1e-6:
        boundary.pop()
    source_edges = sorted(
        {
            *map(str, step.get("source_edge_refs", []) or []),
            *map(str, waist.get("source_edge_refs", []) or []),
            *(ref for item in interfaces for ref in item.get("native_path_edge_refs", []) or []),
        }
    )
    derived_interfaces = [item["derived_physical_interface"] for item in interfaces if item.get("derived_physical_interface")]
    identity = _stable_id(page_number, "interface_closed_flight_profile", proposal["id"], assignment["id"])
    evidence_refs = sorted(
        {
            proposal["id"],
            assignment["id"],
            *source_edges,
            *(ref for item in interfaces for ref in item.get("evidence_refs", []) or []),
        }
    )
    return {
        "id": identity,
        "record_type": "interface_closed_flight_profile",
        "state": "resolved",
        "scope_ref": proposal.get("scope_ref"),
        "source_proposal_ref": proposal["id"],
        "port_assignment_ref": assignment["id"],
        "source_edge_refs": source_edges,
        "primitive_refs": sorted({_primitive_ref(ref).split(".segment[", 1)[0] for ref in source_edges}),
        "ordered_boundary_display": boundary,
        "derived_bridges": [],
        "derived_physical_interfaces": derived_interfaces,
        "dimension_certificate": dict(proposal.get("dimension_certificate") or {}),
        "closure": {
            "closed": True,
            "branch_free": True,
            "unique_completion": True,
            "scale_bounded": True,
            "interface_assignment_unique": True,
            "derived_bridge_count": 0,
            "derived_physical_interface_count": len(derived_interfaces),
        },
        "evidence_refs": evidence_refs,
        "quantity_eligible": False,
    }


def _temporary_closed_boundary_hypothesis(
    proposal: Mapping[str, Any],
    assignment: Mapping[str, Any],
    global_assignment_ref: str,
    role_chains: Mapping[str, Mapping[str, Any]],
    page_number: int,
) -> dict[str, Any]:
    """Publish one assignment-scoped closure without accepting a profile."""

    profile = _closed_profile(proposal, assignment, role_chains, page_number)
    classifications = {
        str(item.get("interface_option_ref")): item
        for item in assignment.get("classified_interfaces", []) or []
    }
    caps = []
    for option in assignment.get("interface_options", []) or []:
        option_ref = str(option.get("id"))
        classification = classifications.get(option_ref, {})
        derived = option.get("derived_physical_interface") or {}
        cap_endpoints = []
        if derived.get("start_display") is not None and derived.get("end_display") is not None:
            cap_endpoints = [
                list(map(float, derived["start_display"])),
                list(map(float, derived["end_display"])),
            ]
        caps.append(
            {
                "interface_option_ref": option_ref,
                "classification": classification.get("classification"),
                "section_station_ref": classification.get("section_station_ref"),
                "projected_plan_station_refs": list(
                    map(str, classification.get("projected_plan_station_refs", []) or [])
                ),
                "ordered_points_display": [
                    list(map(float, point))
                    for point in option.get("ordered_points_display", []) or []
                ],
                "cap_endpoints_display": cap_endpoints,
                "derived_physical_interface_ref": derived.get("id"),
                "native_drawing_edge": bool(derived.get("native_drawing_edge", False)),
                "evidence_refs": list(map(str, option.get("evidence_refs", []) or [])),
                "quantity_eligible": False,
            }
        )
    identity = _stable_id(
        page_number,
        "temporary_closed_boundary_hypothesis",
        global_assignment_ref,
        assignment.get("id"),
    )
    return {
        **profile,
        "id": identity,
        "record_type": "temporary_closed_boundary_hypothesis",
        "state": "resolved_search_hypothesis",
        "source_global_assignment_ref": global_assignment_ref,
        "source_local_assignment_ref": assignment.get("id"),
        "assignment_specific_caps": caps,
        "accepted_physical_profile": False,
        "search_input_only": True,
        "evidence_refs": sorted(
            {
                *map(str, profile.get("evidence_refs", []) or []),
                global_assignment_ref,
                str(assignment.get("id")),
            }
        ),
        "quantity_eligible": False,
    }


def certify_flight_interface_closure(
    open_structural_boundary_assembly: Mapping[str, Any],
    local_dimension_reclosure: Mapping[str, Any],
    banded_plan_sweep_evidence: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Enumerate complete port assignments and accept exactly one global result."""

    local_by_scope = {
        str(item.get("scope_ref")): item
        for item in local_dimension_reclosure.get("scope_results", []) or []
    }
    plan_certificate = banded_plan_sweep_evidence.get("plan_band_certificate") or {}
    plan_pass = (
        plan_certificate.get("state") == "resolved_relative_unsigned"
        and len(plan_certificate.get("bands", []) or []) == 2
        and bool(plan_certificate.get("bands_are_disjoint"))
    )
    scope_results = []
    for scope in open_structural_boundary_assembly.get("scope_results", []) or []:
        scope_ref = str(scope.get("scope_ref"))
        local = local_by_scope.get(scope_ref, {})
        proposals = list(scope.get("flight_boundary_proposals", []) or [])
        role_chains = {str(item.get("id")): item for item in scope.get("branch_free_chains", []) or []}
        scale = float(scope.get("metric_scale_points_per_mm") or 0.0)
        stations = _section_stations(local, scale, page_number) if scale > 0 else []
        graph, points = _support_graph(list(role_chains.values()))
        options_by_proposal, assignments_by_proposal = {}, {}
        for proposal in proposals:
            proposal = {**proposal, "scope_ref": scope_ref}
            options, assignments = _proposal_assignments(
                proposal,
                role_chains,
                graph,
                points,
                stations,
                scale,
                plan_certificate,
                page_number,
            ) if plan_pass and scale > 0 else ([], [])
            options_by_proposal[proposal["id"]] = options
            assignments_by_proposal[proposal["id"]] = assignments

        global_candidates = []
        admissible_by_proposal = {
            proposal_ref: [
                item for item in assignments
                if item.get("preference_status") == "admissible"
            ]
            for proposal_ref, assignments in assignments_by_proposal.items()
        }
        if len(proposals) == 2 and all(admissible_by_proposal.get(item["id"]) for item in proposals):
            for selected in itertools.product(*(admissible_by_proposal[item["id"]] for item in proposals)):
                classification = _classify_global_assignment(selected, plan_certificate)
                if classification is None:
                    continue
                global_candidates.append(
                    {
                        "id": _stable_id(page_number, "global_flight_port_assignment", *(item["id"] for item in selected)),
                        "state": "valid",
                        **classification,
                        "evidence_refs": sorted({*(item["id"] for item in selected), str(plan_certificate.get("id"))}),
                        "quantity_eligible": False,
                    }
                )
        accepted = global_candidates[0] if len(global_candidates) == 1 else None
        proposals_by_id = {
            str(item["id"]): {**item, "scope_ref": scope_ref}
            for item in proposals
        }
        temporary_boundaries = []
        for global_assignment in global_candidates:
            for assignment in global_assignment["assignments"]:
                temporary_boundaries.append(
                    _temporary_closed_boundary_hypothesis(
                        proposals_by_id[str(assignment["proposal_ref"])],
                        assignment,
                        str(global_assignment["id"]),
                        role_chains,
                        page_number,
                    )
                )
        closed_profiles = []
        if accepted:
            for assignment in accepted["assignments"]:
                closed_profiles.append(
                    _closed_profile(
                        proposals_by_id[str(assignment["proposal_ref"])],
                        assignment,
                        role_chains,
                        page_number,
                    )
                )
        status = "accepted_unique_assignment" if accepted else "ambiguous_assignments" if global_candidates else "insufficient_constraints"
        transition = (
            f"{len(proposals)} open flight proposals -> {len(global_candidates)} valid port assignments -> "
            f"{len(closed_profiles)} closed profiles"
        )
        scope_results.append(
            {
                "scope_ref": scope_ref,
                "status": status,
                "reason_code": None if accepted else "multiple_valid_port_assignments" if global_candidates else "no_complete_station_certified_port_assignment",
                "transition": transition,
                "open_flight_proposal_count": len(proposals),
                "section_station_count": len(stations),
                "valid_port_assignment_count": len(global_candidates),
                "closed_profile_count": len(closed_profiles),
                "section_stations": stations,
                "interface_options_by_proposal": options_by_proposal,
                "proposal_assignments": assignments_by_proposal,
                "global_assignment_candidates": global_candidates,
                "accepted_assignment": accepted,
                "temporary_closed_boundary_hypotheses": temporary_boundaries,
                "closed_profiles": closed_profiles,
                "quantity_eligible": False,
            }
        )
    certificates = [item["accepted_assignment"] for item in scope_results if item.get("accepted_assignment")]
    profiles = [profile for item in scope_results for profile in item["closed_profiles"]]
    temporary_boundaries = [
        hypothesis
        for item in scope_results
        for hypothesis in item["temporary_closed_boundary_hypotheses"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "flight_interface_closure",
        "page": page_number,
        "status": "accepted_unique_assignment" if len(certificates) == 1 else "ambiguous_assignments" if any(item["status"] == "ambiguous_assignments" for item in scope_results) else "insufficient_constraints",
        "reason_code": (
            None
            if len(certificates) == 1
            else "multiple_valid_port_assignments"
            if any(item["status"] == "ambiguous_assignments" for item in scope_results)
            else "no_complete_station_certified_port_assignment"
        ),
        "scope_results": scope_results,
        "certificates": certificates,
        "temporary_closed_boundary_hypotheses": temporary_boundaries,
        "closed_profiles": profiles,
        "summary": {
            "scope_count": len(scope_results),
            "open_flight_proposal_count": sum(item["open_flight_proposal_count"] for item in scope_results),
            "valid_port_assignment_count": sum(item["valid_port_assignment_count"] for item in scope_results),
            "closed_profile_count": len(profiles),
            "temporary_closed_boundary_hypothesis_count": len(temporary_boundaries),
            "transition": scope_results[0]["transition"] if len(scope_results) == 1 else None,
        },
        "contract": {
            "section_and_projected_plan_stations_required": True,
            "every_matched_port_pair_requires_one_common_certified_station": True,
            "native_support_or_landing_chains_preferred": True,
            "derived_end_cap_requires_certified_endpoints_and_station": True,
            "derived_end_cap_is_not_a_native_drawing_edge": True,
            "all_valid_port_assignments_enumerated": True,
            "temporary_assignment_boundaries_published_before_uniqueness": True,
            "unique_global_assignment_required": True,
            "landing_geometry_constructed": False,
            "mesh_construction_performed": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
