"""Bounded projected elbows with two independently traced native boundaries.

No path is selected through a branch, no crossing is turned into a junction,
and no missing stroke is filled. Duplicate drawing strokes retain all source
references while identical geometric edges share a traversal edge.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
import math

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import _ports, _dot, _sub, TOLERANCE, curve_may_touch_boundary
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible, _samples
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.core.vector_topology import point_distance_to_segment


def _source_style_compatibility_exhaustive(style, sources):
    """Uncached reference for differential tests."""
    return ((row, _style_compatible(style, _normalise_style(row['source_native_segment']['style'])))
            for row in sources)


def _points_on_segments_exhaustive(points, edges):
    """Exhaustive reference batch, retaining the original coordinate objects."""
    return ([v for v in points if point_distance_to_segment(v, p, q) <= TOLERANCE]
            for p, q in edges)


def _freeze_style(value):
    if isinstance(value, dict):
        return ('dict', tuple((key, _freeze_style(item)) for key, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(map(_freeze_style, value)))
    return value


def _source_style_compatibility(style, sources):
    """Memoize pure normalization by complete raw value within this call only."""
    normalized = {}
    for row in sources:
        raw = row['source_native_segment']['style']
        key = _freeze_style(raw)
        if key not in normalized:
            normalized[key] = _normalise_style(raw)
        yield row, _style_compatible(style, normalized[key])


def _point_index_guard(points, edges, tolerance):
    """Bounded broad-phase roundoff guard; None requests the exact reference.

    Do not extend the experiment's numeric domain by silently pruning unusual
    input. Unsupported coordinates retain the original predicate and behavior.
    """
    if (any(len(edge) != 2 for edge in edges)
            or any(len(point) != 2 for point in chain(points, chain.from_iterable(edges)))):
        return None
    coordinates = [v for point in points for v in point]
    coordinates.extend(v for edge in edges for point in edge for v in point)
    if (not math.isfinite(tolerance) or not 0 <= tolerance <= 1e12
            or any(not math.isfinite(v) or abs(v) > 1e12 for v in coordinates)):
        return None
    return 64 * math.ulp(max(1., max(map(abs, coordinates), default=0.)))


def _indexed_point_hits(points, edges, tolerance=TOLERANCE, stats=None):
    """Conservative two-axis index; the original distance predicate is final.

    Return original point positions per edge, never new geometry or source IDs.
    Duplicate positions remain distinct. No graph budget is changed here.
    """
    guard = _point_index_guard(points, edges, tolerance)
    if guard is None:
        if stats is not None:
            stats.update(distance_tests=len(points)*len(edges), exhaustive_fallback=True)
        return [[i for i, v in enumerate(points)
                 if point_distance_to_segment(v, p, q) <= tolerance] for p, q in edges]
    padding = tolerance + guard
    axes = [sorted((point[axis], i) for i, point in enumerate(points)) for axis in (0, 1)]
    coordinates = [[value for value, _ in axis] for axis in axes]
    output, tested = [], 0
    for p, q in edges:
        bounds = [(min(p[a], q[a]) - padding, max(p[a], q[a]) + padding) for a in (0, 1)]
        ranges = [(bisect_left(coordinates[a], bounds[a][0]),
                   bisect_right(coordinates[a], bounds[a][1])) for a in (0, 1)]
        axis = 0 if ranges[0][1] - ranges[0][0] <= ranges[1][1] - ranges[1][0] else 1
        lo, hi = ranges[axis]
        hits = []
        for _, i in axes[axis][lo:hi]:
            if bounds[1-axis][0] <= points[i][1-axis] <= bounds[1-axis][1]:
                tested += 1
                if point_distance_to_segment(points[i], p, q) <= tolerance:
                    hits.append(i)
        output.append(sorted(hits))
    if stats is not None:
        stats['distance_tests'] = tested
    return output


def _points_on_segments(points, edges):
    """Production batch: index each invocation, preserving coordinate objects."""
    points, edges = list(points), list(edges)
    hits = _indexed_point_hits(points, edges)
    return ([points[i] for i in indices] for indices in hits)


def trace_bend_boundaries(ports, sources, max_edges=1500):
    """Replay a unique two-path boundary graph inside the two port planes."""
    edges, curves = {}, []
    for row, compatible in _source_style_compatibility(ports[0]['style'], sources):
        native = row['source_native_segment']
        if not compatible:
            continue
        if native['kind'] != 'line':
            curves.append(row)
            continue
        p, q = row['points_display'][0], row['points_display'][-1]
        if math.dist(p, q) <= TOLERANCE:
            continue
        # Portal caps are observed cross sections. They cannot be used to
        # travel from one sidewall to the other.
        if any(all(abs(_dot(_sub(v, port['center']), port['outward'])) <= TOLERANCE
                   for v in (p, q)) for port in ports):
            continue
        if any(all(_dot(_sub(v, port['center']), port['outward']) < -TOLERANCE
                   for v in (p, q)) for port in ports):
            continue
        key = tuple(sorted((tuple(p), tuple(q))))
        edges.setdefault(key, []).append(row['source_primitive_ref'])
    if len(edges) > max_edges:
        return [], ['native_boundary_graph_budget_exceeded']
    # Cluster only representation-rounding differences. Retain original rows;
    # coordinates here are traversal keys, never replacements for observations.
    originals = {p for edge in edges for p in edge} | {tuple(p) for port in ports for p in port['sides']}
    canonical = {}
    for point in sorted(originals):
        nearby = {canonical[p] for p in canonical if math.dist(p, point) <= TOLERANCE}
        if len(nearby) > 1:
            return [], ['ambiguous_native_coordinate_rounding']
        canonical[point] = next(iter(nearby)) if nearby else point
    points = set(canonical.values())
    adjacency, edge_refs = defaultdict(set), defaultdict(set)
    for ((p, q), refs), hits in zip(edges.items(), _points_on_segments(points, edges)):
        on = sorted((math.dist(p, v), v) for v in hits)
        for (_, a), (_, b) in zip(on, on[1:]):
            if a == b:
                continue
            middle = [(x + y) / 2 for x, y in zip(a, b)]
            if any(_dot(_sub(middle, port['center']), port['outward']) < -TOLERANCE for port in ports):
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_refs[tuple(sorted((a, b)))].update(refs)
    goals = {canonical[tuple(p)] for p in ports[1]['sides']}
    paths = []
    for start in ports[0]['sides']:
        current = canonical[tuple(start)]
        visited, refs = [current], set()
        while current not in goals:
            options = adjacency[current] - set(visited)
            if len(options) != 1:
                return [], ['native_boundary_branch_or_missing_trace']
            other = next(iter(options))
            refs.update(edge_refs[tuple(sorted((current, other)))])
            visited.append(other)
            current = other
            if len(visited) > max_edges:
                return [], ['native_boundary_graph_budget_exceeded']
        paths.append({'points_display': [list(p) for p in visited], 'source_primitive_refs': sorted(refs)})
    if (paths[0]['points_display'][-1] == paths[1]['points_display'][-1]
            or set(map(tuple, paths[0]['points_display'])).intersection(map(tuple, paths[1]['points_display']))):
        return [], ['sidewall_paths_are_not_independent']
    samples = [_samples(path['points_display'], 17) for path in paths]
    widths = [math.dist(a, b) for a, b in zip(*samples)]
    mean_width = sum(port['width'] for port in ports) / 2
    if any(abs(w - mean_width) > max(.15, .08 * mean_width) for w in widths):
        return [], ['bend_sidewall_separation_is_not_persistent']
    # Unsupported curves can only add uncertainty, never disappear from a
    # complete source search because a linear trace happened to be found.
    for row in curves:
        if curve_may_touch_boundary(row, paths):
            return [], ['unsupported_curve_at_native_boundary']
    return paths, []


def discover_native_bends(*, index, composites, graph, page_ref):
    fragments = {r['id']: r for p in graph['pages'] for r in p['fragments']}
    ports = list(_ports([r for r in composites['accepted_composites'] if r['page_ref'] == page_ref], fragments))
    records = []
    for a, b in combinations(ports, 2):
        w = max(a['width'], b['width'])
        if (a['composite_ref'] == b['composite_ref'] or math.dist(a['center'], b['center']) > 4 * w
                or abs(_dot(a['outward'], b['outward'])) > .001
                or abs(a['width'] - b['width']) > max(.15, .05 * w)
                or _dot(_sub(b['center'], a['center']), a['outward']) <= 0
                or _dot(_sub(a['center'], b['center']), b['outward']) <= 0
                or not _style_compatible(a['style'], b['style'])):
            continue
        centers = [a['center'], b['center']]
        box = [min(p[0] for p in centers) - 2*w, min(p[1] for p in centers) - 2*w,
               max(p[0] for p in centers) + 2*w, max(p[1] for p in centers) + 2*w]
        sources, complete, regions = index.query(box)
        paths, reasons = trace_bend_boundaries([a, b], sources)
        if not complete:
            reasons.append('incomplete_native_boundary_search')
        refs = sorted({ref for path in paths for ref in path['source_primitive_refs']})
        records.append({'id': _stable_id('mep_native_bend_connection', a['id'], b['id']),
            'record_type': 'mep_native_boundary_connection', 'record_version': '0.1.0',
            'page_ref': page_ref, 'composite_refs': [a['composite_ref'], b['composite_ref']],
            'port_refs': [a['id'], b['id']], 'ports': [a, b], 'boundary_paths': paths,
            'source_primitive_refs': refs,
            'source_rows': deepcopy(sources),
            'search': {'region_refs': regions, 'bbox_display': box, 'complete': complete,
                       'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in sources))},
            'relation_type': 'projected_native_bend', 'state': 'accepted' if not reasons else 'abstained',
            'reasons': reasons, 'physical_continuation_established': False, 'quantity_eligible': False})
    incidence = defaultdict(list)
    for row in records:
        for ref in row['port_refs']:
            incidence[ref].append(row)
    for row in records:
        if any(len(incidence[ref]) != 1 for ref in row['port_refs']):
            row['state'] = 'abstained'
            row['reasons'].append('non_unique_projected_port_destination')
    return records
