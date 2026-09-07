"""Native compound direction symbols and band-local surface-order evidence.

This bounded recognizer requires an open circle, a branch-free native shaft,
an open arrowhead, and repeated cross-band step strokes. It does not assign
profiles by list order or use candidate meshes to supply expected evidence.
"""

from collections import defaultdict
from hashlib import sha256
import math

from shapely.geometry import LineString, Polygon

from src.drawing_engine.core.vector_topology import _style_key


CONVENTION = {
    "id": "drawing_convention.iso_7519_1991.4_9_b.circle_to_open_arrow.ascent.v1",
    "state": "convention_dependent",
    "source": "https://preview.sist.si/sist-preview/18010/abe9e763d58f4340aa2879a207d0416e/SIST-EN-ISO-7519-1998.pdf",
    "meaning": "open circle denotes bottom riser; open arrowhead denotes top riser",
    "applicability": "complete compound symbol on band-centred path crossing repeated step strokes",
}


def _id(kind, refs):
    return kind + "." + sha256("\0".join(sorted(refs)).encode()).hexdigest()[:16]


def _open_continuous(segment):
    style = segment.get("style") or {}
    return (
        style.get("fill") is None
        and style.get("stroke") is not None
        and str(style.get("dash") or "").replace(" ", "") in {"", "[]0"}
        and float(style.get("width") or 0) > 0
    )


def _circle_components(segments, tolerance):
    curves = {s["id"]: s for s in segments if s.get("kind") == "cubic"}
    by_vertex = defaultdict(set)
    for ref, curve in curves.items():
        for end in ("start", "end"):
            by_vertex[(curve[end + "_vertex_id"], _style_key(curve["style"]))].add(ref)
    remaining = set(curves)
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            ref = pending.pop()
            if ref in component:
                continue
            component.add(ref)
            curve = curves[ref]
            for end in ("start", "end"):
                pending.extend(by_vertex[(curve[end + "_vertex_id"], _style_key(curve["style"]))] - component)
        remaining -= component
        records = [curves[ref] for ref in sorted(component)]
        degree = defaultdict(int)
        for curve in records:
            for end in ("start", "end"):
                degree[curve[end + "_vertex_id"]] += 1
        if len(records) < 3 or any(count != 2 for count in degree.values()):
            continue
        points = [p for curve in records for p in curve.get("sample_points_display", [])]
        if not points:
            continue
        low = [min(p[i] for p in points) for i in (0, 1)]
        high = [max(p[i] for p in points) for i in (0, 1)]
        center = [(a + b) / 2 for a, b in zip(low, high)]
        radius = sum(b - a for a, b in zip(low, high)) / 4
        residual = max(abs(math.dist(p, center) - radius) for p in points)
        if radius <= 2 * tolerance or residual > min(tolerance, .05 * radius):
            continue
        yield records, center, radius, residual


def _shaft_paths(lines, center, tolerance):
    # Merge exact duplicate strokes only; retain every native support ref.
    edges = {}
    adjacency = defaultdict(set)
    coordinates = {}
    for line in lines:
        a, b = line["start_vertex_id"], line["end_vertex_id"]
        points = tuple(sorted((tuple(line["start_display"]), tuple(line["end_display"]))))
        key = (tuple(sorted((a, b))), points)
        edge = edges.setdefault(key, {"vertices": (a, b), "refs": []})
        edge["refs"].append(line["id"])
        for end, vertex in (("start", a), ("end", b)):
            adjacency[vertex].add(key)
            coordinates[vertex] = line[end + "_display"]
    for start in sorted(adjacency):
        if len(adjacency[start]) != 1 or math.dist(coordinates[start], center) > tolerance:
            continue
        vertex, previous = start, None
        visited, path, refs = set(), [coordinates[start]], []
        while vertex not in visited:
            visited.add(vertex)
            choices = adjacency[vertex] - ({previous} if previous else set())
            if len(choices) == 2 and previous is not None:
                # Both wings must terminate, point behind the tip, and be
                # symmetric about the incoming shaft. Crossings are not joins.
                tip, prior = coordinates[vertex], path[-2]
                incoming = [prior[i] - tip[i] for i in (0, 1)]
                length = math.hypot(*incoming)
                wings = []
                wing_refs = []
                for key in sorted(choices):
                    other = next(v for v in edges[key]["vertices"] if v != vertex)
                    if len(adjacency[other]) != 1:
                        break
                    vector = [coordinates[other][i] - tip[i] for i in (0, 1)]
                    along = sum(vector[i] * incoming[i] for i in (0, 1)) / length
                    across = (incoming[0] * vector[1] - incoming[1] * vector[0]) / length
                    if not (tolerance < along < .35 * length and .2 < abs(across) / along < 2):
                        break
                    wings.append((along, across))
                    wing_refs.extend(edges[key]["refs"])
                if len(wings) == 2 and wings[0][1] * wings[1][1] < 0 and max(
                    abs(wings[0][0] - wings[1][0]), abs(wings[0][1] + wings[1][1])
                ) <= tolerance:
                    yield path, sorted(refs), sorted(wing_refs), math.dist(path[0], center)
                break
            if len(choices) != 1:
                break
            key = next(iter(choices))
            vertex = next(v for v in edges[key]["vertices"] if v != vertex)
            refs.extend(edges[key]["refs"])
            path.append(coordinates[vertex])
            previous = key


def derive_plan_direction_evidence(topology, segmentation, folded_plan, bands):
    """Return native path evidence, or an explicit unresolved requirement.

    Coordinates use the already certified plan gauge. Recognition is invariant
    to path order, source path splitting, translation and native width changes.
    Absolute section viewing direction is deliberately not inferred.
    """
    result = {
        "record_type": "directed_surface_path_evidence",
        "state": "unknown", "reason_code": "unique_native_direction_path_unresolved",
        "paths": [], "candidates": [], "quantity_eligible": False,
        "convention": dict(CONVENTION), "evidence_refs": [],
    }
    if not folded_plan:
        return result
    scope_ref = folded_plan["title_scope_ref"]
    scopes = [s for s in segmentation.get("segments", []) if s.get("id") == scope_ref and s.get("state") == "resolved"]
    frame = folded_plan.get("plan_to_relative_xy") or {}
    scale = float(frame.get("scale_points_per_mm") or 0)
    if len(scopes) != 1 or scale <= 0 or not bands:
        return result
    origin = frame["origin_display"]
    u_axis, v_axis = frame["u_axis_display"], frame["v_axis_display"]

    def uv(point):
        return [sum((point[i] - origin[i]) * axis[i] for i in (0, 1)) / scale for axis in (u_axis, v_axis)]

    refs = set(scopes[0]["primitive_refs"])
    segments = [s for s in topology.get("segments", []) if
                (s.get("drawing_ref") in refs or s.get("primitive_ref") in refs) and _open_continuous(s)]
    tolerance = max(float(topology.get("vertex_tolerance_points") or 0), 1e-5)
    band_origin = min(b["interval_mm"][0] for b in bands)
    footprint = Polygon(folded_plan["polygon_uv_mm"])
    candidates = []
    for circle, center, radius, circle_residual in _circle_components(segments, tolerance):
        style = _style_key(circle[0]["style"])
        lines = [s for s in segments if s.get("kind") == "line" and _style_key(s["style"]) == style]
        for path, shaft_refs, wing_refs, center_residual in _shaft_paths(lines, center, tolerance):
            points = [uv(p) for p in path]
            if not footprint.buffer(tolerance / scale).covers(LineString(points)):
                continue
            samples, runs, used_bands, support_refs = [], [], [], set()
            for start, end in zip(points, points[1:]):
                matching = [b for b in bands if abs(start[1] - end[1]) <= tolerance / scale
                            and abs((start[1] + end[1]) / 2 - (sum(b["interval_mm"]) / 2 - band_origin)) <= 2 * tolerance / scale]
                if not matching:
                    # Landing-only turns are retained between the band samples.
                    if min(start[0], end[0]) >= -tolerance / scale:
                        samples.append({"uv_mm": list(start), "band_ref": None})
                    continue
                if len(matching) != 1:
                    break
                band = matching[0]
                low_y, high_y = [v - band_origin for v in band["interval_mm"]]
                stations = []
                for line in lines:
                    a, b = uv(line["start_display"]), uv(line["end_display"])
                    if (abs(a[0] - b[0]) <= tolerance / scale
                        and max(abs(min(a[1], b[1]) - low_y), abs(max(a[1], b[1]) - high_y)) <= 2 * tolerance / scale
                        and min(start[0], end[0]) - tolerance / scale <= a[0] <= min(0, max(start[0], end[0])) + tolerance / scale):
                        stations.append(a[0])
                        support_refs.add(line["id"])
                ordered = []
                for station in sorted(stations):
                    if not ordered or station - ordered[-1] > tolerance / scale:
                        ordered.append(station)
                if len(ordered) < 4 or band["id"] in used_bands:
                    break
                intervals = [b - a for a, b in zip(ordered, ordered[1:])]
                pitch = sorted(intervals)[len(intervals) // 2]
                if max(abs(length - pitch) for length in intervals) > 2 * tolerance / scale:
                    break
                midpoints = [(a + b) / 2 for a, b in zip(ordered, ordered[1:])]
                if end[0] < start[0]:
                    midpoints.reverse()
                indices = list(range(len(samples), len(samples) + len(midpoints)))
                samples.extend({"uv_mm": [x, start[1]], "band_ref": band["id"]} for x in midpoints)
                runs.append({"band_ref": band["id"], "sample_indices": indices,
                             "longitudinal_ascent_sign": 1 if end[0] > start[0] else -1,
                             "native_tread_pitch_mm": pitch})
                used_bands.append(band["id"])
            else:
                if set(used_bands) != {b["id"] for b in bands} or len(runs) < 2:
                    continue
                evidence_refs = sorted({scope_ref, folded_plan["id"], CONVENTION["id"],
                                        *shaft_refs, *wing_refs, *support_refs, *(c["id"] for c in circle),
                                        *(b["id"] for b in bands)})
                candidates.append({
                    "id": _id("native_directed_surface_path", evidence_refs),
                    "state": "derived", "interpretation_state": "convention_dependent",
                    "source_scope_ref": scope_ref, "convention_ref": CONVENTION["id"],
                    "ordered_points_display": path, "ordered_points_uv_mm": points,
                    "circle_edge_refs": [c["id"] for c in circle],
                    "shaft_edge_refs": shaft_refs, "arrowhead_edge_refs": wing_refs,
                    "circle_radius_points": radius, "circle_fit_residual_points": circle_residual,
                    "circle_shaft_residual_points": center_residual,
                    "elevation_axis_xyz": [0., 0., 1.], "elevation_order": "nondecreasing",
                    "require_positive_rise_per_run": True, "band_runs": runs, "samples": samples,
                    "tolerance_mm": tolerance / scale,
                    "independent_of_candidate_geometry": True,
                    "evidence_refs": evidence_refs, "quantity_eligible": False,
                })
    result["candidates"] = candidates
    result["evidence_refs"] = sorted({scope_ref, *(r for c in candidates for r in c["evidence_refs"])})
    result["id"] = _id("directed_surface_path_evidence", result["evidence_refs"])
    if len(candidates) == 1:
        result.update(state="accepted", reason_code=None, paths=candidates)
    return result
