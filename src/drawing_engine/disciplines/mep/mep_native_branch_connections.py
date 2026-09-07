"""Conservative native multi-port junction discovery, separate from M4.

Ports only nominate a bounded search. A projected branch requires a single
native perimeter connecting all portal sides, with no branch in that perimeter,
missing trace, hidden through-stroke, or second destination. Observed source
rows remain unchanged; portal planes only delimit the local replay scope.
"""

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import math

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import _ports, _dot, _sub, TOLERANCE, boundary_coverage, curve_may_touch_boundary
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style, _segment_intersection
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.core.vector_topology import point_distance_to_segment


def trace_branch_boundaries(ports, sources, max_edges=1500, diagnostics=None, search_bbox=None):
    """Require one open-sided native perimeter, not intersecting pipe strokes."""
    edges, curves = {}, []
    for row in sources:
        native = row['source_native_segment']
        if not _style_compatible(ports[0]['style'], _normalise_style(native['style'])):
            continue
        if native['kind'] != 'line':
            curves.append(row)
            continue
        p, q = row['points_display'][0], row['points_display'][-1]
        if math.dist(p, q) <= TOLERANCE:
            continue
        if any(all(_dot(_sub(v, port['center']), port['outward']) < -TOLERANCE
                   for v in (p, q)) for port in ports):
            continue
        edges.setdefault(tuple(sorted((tuple(p), tuple(q)))), []).append(row['source_primitive_ref'])
    if len(edges) > max_edges:
        return [], ['native_boundary_graph_budget_exceeded']
    originals = {p for edge in edges for p in edge} | {tuple(p) for port in ports for p in port['sides']}
    canonical = {}
    for point in sorted(originals):
        nearby = {canonical[p] for p in canonical if math.dist(p, point) <= TOLERANCE}
        if len(nearby) > 1:
            return [], ['ambiguous_native_coordinate_rounding']
        canonical[point] = next(iter(nearby)) if nearby else point
    points = set(canonical.values())
    adjacency, edge_refs = defaultdict(set), defaultdict(set)
    for (p, q), refs in edges.items():
        on = sorted((math.dist(p, v), v) for v in points
                    if point_distance_to_segment(v, p, q) <= TOLERANCE)
        for (_, a), (_, b) in zip(on, on[1:]):
            if a == b:
                continue
            middle = [(x + y) / 2 for x, y in zip(a, b)]
            if any(all(abs(_dot(_sub(v, port['center']), port['outward'])) <= TOLERANCE for v in (a, b))
                   and point_distance_to_segment(middle, *port['sides']) <= TOLERANCE for port in ports):
                continue  # Remove only the portal aperture, retaining shoulders.
            if any(_dot(_sub(middle, port['center']), port['outward']) < -TOLERANCE for port in ports):
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_refs[tuple(sorted((a, b)))].update(refs)
    owner = {}
    for port in ports:
        for side in port['sides']:
            point = canonical[tuple(side)]
            if point in owner:
                return [], ['shared_portal_side_endpoint']
            owner[point] = port['id']
    paths, visited_sides, used_points = [], set(), set()
    port_neighbors = defaultdict(set)
    for start in sorted(owner):
        if start in visited_sides:
            continue
        current, visited, refs = start, [start], set()
        while True:
            options = adjacency[current] - set(visited)
            if len(options) != 1:
                if diagnostics is not None:
                    diagnostics.update(failure_point_display=list(current),
                        competing_next_points_display=[list(p) for p in sorted(options)],
                        trace_points_display=[list(p) for p in visited],
                        incident_source_primitive_refs=sorted({ref for p in adjacency[current]
                            for ref in edge_refs[tuple(sorted((current, p)))]}))
                return [], ['native_boundary_branch_or_missing_trace']
            other = next(iter(options))
            refs.update(edge_refs[tuple(sorted((current, other)))])
            visited.append(other)
            current = other
            if current in owner:
                break
            if len(visited) > max_edges:
                return [], ['native_boundary_graph_budget_exceeded']
        if len(adjacency[current]) != 1:
            return [], ['native_through_stroke_at_portal_side']
        if owner[start] == owner[current] or used_points.intersection(visited):
            return [], ['sidewall_paths_are_not_independent']
        visited_sides.update((start, current))
        used_points.update(visited)
        port_neighbors[owner[start]].add(owner[current])
        port_neighbors[owner[current]].add(owner[start])
        paths.append({'points_display': [list(p) for p in visited],
                      'source_primitive_refs': sorted(refs),
                      'port_refs': [owner[start], owner[current]]})
    reached, pending = set(), [ports[0]['id']]
    for ref in pending:
        if ref not in reached:
            reached.add(ref)
            pending.extend(port_neighbors[ref] - reached)
    if len(paths) != len(ports) or reached != {p['id'] for p in ports}:
        return [], ['disconnected_crossing_perimeters']
    segments = [(a, b) for path in paths for a, b in zip(path['points_display'], path['points_display'][1:])]
    if any(_segment_intersection(a, b, c, d) is not None for (a, b), (c, d) in combinations(segments, 2)):
        return [], ['crossing_native_perimeter_paths']
    if search_bbox is not None and any(not (search_bbox[0] - TOLERANCE <= p[0] <= search_bbox[2] + TOLERANCE
            and search_bbox[1] - TOLERANCE <= p[1] <= search_bbox[3] + TOLERANCE) for p in used_points):
        return [], ['native_boundary_trace_leaves_complete_search_scope']
    for row in curves:
        if curve_may_touch_boundary(row, paths):
            return [], ['unsupported_curve_at_native_boundary']
    return paths, []


def branch_port_candidates(ports):
    """Nominate aligned T port groups; these geometry tests grant no join."""
    candidates = {}
    for a, b in combinations(ports, 2):
        delta = _sub(b['center'], a['center'])
        distance = _dot(delta, a['outward'])
        width = max(a['width'], b['width'])
        if (a['composite_ref'] == b['composite_ref'] or _dot(a['outward'], b['outward']) > -1 + 1e-8
                or not 0 < distance <= 12 * width
                or math.dist(delta, [distance * v for v in a['outward']]) > TOLERANCE):
            continue
        for c in ports:
            if c['composite_ref'] in {a['composite_ref'], b['composite_ref']} or abs(_dot(a['outward'], c['outward'])) > 1e-8:
                continue
            t = _dot(_sub(c['center'], a['center']), a['outward'])
            center = [a['center'][i] + t * a['outward'][i] for i in (0, 1)]
            if not all(0 < _dot(_sub(center, p['center']), p['outward']) <= 12 * max(width, c['width'])
                       and _style_compatible(a['style'], p['style']) for p in (a, b, c)):
                continue
            group = sorted((a, b, c), key=lambda p: p['id'])
            candidates[tuple(p['id'] for p in group)] = group
    return [candidates[key] for key in sorted(candidates)]


def port_attachment_observations(ports, sources):
    """Report exact sidewall attachments and literal gaps without filling them."""
    observations = []
    for port in ports:
        for side in port['sides']:
            attached, gaps = set(), []
            for row in sources:
                native = row['source_native_segment']
                if native['kind'] != 'line' or not _style_compatible(port['style'], _normalise_style(native['style'])):
                    continue
                p, q = row['points_display'][0], row['points_display'][-1]
                positions = [_dot(_sub(point, side), port['outward']) for point in (p, q)]
                if point_distance_to_segment(side, p, q) <= TOLERANCE and max(positions) > TOLERANCE:
                    attached.add(row['source_primitive_ref'])
                if all(math.dist(point, [side[i] + t * port['outward'][i] for i in (0, 1)]) <= TOLERANCE
                       for point, t in zip((p, q), positions)) and min(positions) > TOLERANCE:
                    gaps.append((min(positions), row['source_primitive_ref']))
            gap = min((value for value, _ in gaps), default=None)
            observations.append({'port_ref': port['id'], 'side_point_display': list(side),
                'exact_forward_native_source_refs': sorted(attached),
                'first_collinear_gap_display_points': gap if not attached else None,
                'first_collinear_source_refs': sorted(ref for distance, ref in gaps
                    if gap is not None and abs(distance - gap) <= TOLERANCE) if not attached else [],
                'gap_repaired': False, 'quantity_eligible': False})
    return observations


def discover_native_branches(*, index, composites, graph, page_ref):
    fragments = {r['id']: r for p in graph['pages'] for r in p['fragments']}
    ports = list(_ports([r for r in composites['accepted_composites'] if r['page_ref'] == page_ref], fragments))
    records = []
    for group in branch_port_candidates(ports):
        width = max(p['width'] for p in group)
        centers = [p['center'] for p in group]
        box = [min(p[0] for p in centers) - 2 * width, min(p[1] for p in centers) - 2 * width,
               max(p[0] for p in centers) + 2 * width, max(p[1] for p in centers) + 2 * width]
        sources, complete, regions = index.query(box)
        diagnostics = {}
        paths, reasons = trace_branch_boundaries(group, sources, diagnostics=diagnostics, search_bbox=box)
        through_refs = set()
        for a, b in combinations(group, 2):
            if _dot(a['outward'], b['outward']) > -1 + 1e-8:
                continue
            coverage = [boundary_coverage(side, min(b['sides'], key=lambda other: math.dist(side, other)),
                                           sources, a['style']) for side in a['sides']]
            if all(refs is not None for refs in coverage):
                through_refs.update(ref for refs in coverage for ref in refs)
        if through_refs and reasons:
            reasons.append('opposing_sidewall_continuation_requires_fitting_body_applicability')
        if not complete:
            reasons.append('incomplete_native_boundary_search')
        refs = sorted({ref for path in paths for ref in path['source_primitive_refs']})
        records.append({'id': _stable_id('mep_native_branch_connection', *(p['id'] for p in group)),
            'record_type': 'mep_native_boundary_connection', 'record_version': '0.1.0',
            'page_ref': page_ref, 'composite_refs': [p['composite_ref'] for p in group],
            'port_refs': [p['id'] for p in group], 'ports': group, 'boundary_paths': paths,
            'source_primitive_refs': refs, 'source_rows': deepcopy(sources),
            'trace_diagnostics': diagnostics,
            'port_attachment_observations': port_attachment_observations(group, sources),
            'through_stroke_source_primitive_refs': sorted(through_refs),
            'search': {'region_refs': regions, 'bbox_display': box, 'complete': complete,
                       'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in sources))},
            'relation_type': 'projected_native_branch', 'state': 'accepted' if not reasons else 'abstained',
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
