"""Source-bound completion of a projected interval between certified interfaces.

Completion concerns exactly the named native corridor. The interfaces remain
unresolved physical continuations; a crop edge is never an interface witness.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import math

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import TOLERANCE, _dot, _sub, _ports, boundary_coverage
from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style, _segment_intersection


def trace_candidates(*, graph, composites, boundary_connections):
    """All original straight composites with independently certified end joins."""
    bindings = {'m3_5_contract_ref': {'payload_sha256': _sha256(composites)},
                'outlined_route_composites': composites['accepted_composites']}
    connections = replay_boundary_connections(boundary_connections, graph, bindings)
    incidents = defaultdict(list)
    for connection in connections:
        for port in connection['ports']:
            incidents[port['id']].append(connection)
    fragments = {r['id']: r for page in graph['pages'] for r in page['fragments']}
    ports = defaultdict(list)
    for port in _ports(composites['accepted_composites'], fragments):
        ports[port['composite_ref']].append(port)
    candidates = []
    for composite in sorted(composites['accepted_composites'], key=lambda c: c['id']):
        ends = sorted(ports[composite['id']], key=lambda p: p['end'])
        accepted = [[c for c in incidents[p['id']] if c['state'] == 'accepted'] for p in ends]
        if len(ends) != 2 or any(len(rows) != 1 for rows in accepted):
            continue
        corners = [point for port in ends for point in port['sides']]
        margin = 2 * max(p['width'] for p in ends)
        box = [min(p[axis] for p in corners) - margin for axis in (0, 1)] + [
            max(p[axis] for p in corners) + margin for axis in (0, 1)]
        candidates.append({'id': _stable_id('mep_projected_trace_scope', composite['id'], [r[0]['id'] for r in accepted]),
            'scope_kind': 'one_native_composite_between_certified_junction_port_interfaces',
            'page_ref': composite['page_ref'], 'composite_ref': composite['id'],
            'input_composite': deepcopy(composite), 'ports': deepcopy(ends),
            'interfaces': [{'port_ref': port['id'], 'point_display': deepcopy(port['center']),
                'side_points_display': deepcopy(port['sides']), 'classification': 'certified_scope_interface',
                'native_junction_ref': rows[0]['id'], 'native_junction_kind': rows[0]['relation_type'],
                'incident_candidate_refs': sorted(c['id'] for c in incidents[port['id']]),
                'unresolved_competing_candidate_refs': sorted(c['id'] for c in incidents[port['id']] if c['state'] != 'accepted'),
                'out_of_scope_continuation_resolved': False} for port, rows in zip(ends, accepted)],
            'expected_query_bbox_display': box, 'query_margin_display_points': margin})
    return candidates


def _clip(a, b, length, half):
    """Clip a native straight segment in the corridor's local coordinates."""
    lo, hi = 0., 1.
    for axis, limits in enumerate(((0., length), (-half, half))):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            if not limits[0] - TOLERANCE <= a[axis] <= limits[1] + TOLERANCE:
                return None
            continue
        enter, leave = sorted((value - a[axis]) / delta for value in limits)
        lo, hi = max(lo, enter), min(hi, leave)
        if lo > hi + 1e-9:
            return None
    return [[a[i] + value * (b[i] - a[i]) for i in (0, 1)] for value in (lo, hi)]


class TraceSearchBudgetExceeded(ValueError):
    pass


def straight_supports(sources, pair_budget=2000000, diagnostics=None):
    """Exact continuous collinear unions; native segmentation is not branching."""
    supports = {}
    lines = sorted((s for s in sources if s['source_native_segment']['kind'] == 'line'), key=lambda s: s['source_primitive_ref'])
    checks = 0
    for seed in lines:
        if seed['source_primitive_ref'] in supports:
            continue
        a, b = seed['points_display'][0], seed['points_display'][-1]
        length = math.dist(a, b)
        if length <= TOLERANCE:
            continue
        direction = [(b[i] - a[i]) / length for i in (0, 1)]
        normal = [-direction[1], direction[0]]
        style = _normalise_style(seed['source_native_segment']['style'])
        intervals = []
        for source in lines:
            checks += 1
            if checks > pair_budget:
                raise TraceSearchBudgetExceeded('native_straight_support_work_budget_exhausted')
            if not _style_compatible(style, _normalise_style(source['source_native_segment']['style'])):
                continue
            points = (source['points_display'][0], source['points_display'][-1])
            if any(abs(_dot(_sub(p, a), normal)) > TOLERANCE for p in points):
                continue
            endpoints = sorted((_dot(_sub(p, a), direction), p) for p in points)
            intervals.append((endpoints[0][0], endpoints[1][0], source['source_primitive_ref'], endpoints[0][1], endpoints[1][1]))
        groups = []
        for lo, hi, ref, start, end in sorted(intervals):
            if not groups or lo > groups[-1]['hi'] + TOLERANCE:
                groups.append({'lo': lo, 'hi': hi, 'points_display': [start, end], 'source_primitive_refs': [ref]})
            else:
                group = groups[-1]
                group['source_primitive_refs'].append(ref)
                if hi > group['hi']:
                    group['hi'], group['points_display'][1] = hi, end
        for group in groups:
            if group['hi'] - group['lo'] <= TOLERANCE:
                continue
            result = {'id': _stable_id('mep_native_straight_support', sorted(group['source_primitive_refs'])),
                'points_display': group['points_display'], 'source_primitive_refs': sorted(group['source_primitive_refs'])}
            for ref in group['source_primitive_refs']:
                supports[ref] = result
    if diagnostics is not None:
        diagnostics['pair_check_count'] = checks
    return supports


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(point, a, b):
    length = math.dist(a, b)
    return length > TOLERANCE and abs(_cross(a, b, point)) <= TOLERANCE * length and (
        min(a[0], b[0]) - TOLERANCE <= point[0] <= max(a[0], b[0]) + TOLERANCE
        and min(a[1], b[1]) - TOLERANCE <= point[1] <= max(a[1], b[1]) + TOLERANCE)


def _inside(point, hull):
    inside = False
    for a, b in zip(hull, hull[1:] + hull[:1]):
        if _on_segment(point, a, b):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]) and point[0] < a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            inside = not inside
    return inside


def bounded_inline_detail_components(source_classifications, sources, local, length, half,
                                     protected_route_source_refs):
    """Certify finite native detail ink inside continuously covered sidewalls.

    A dense native component can depict a coupling, hanger or other inline
    drafting detail without creating a centreline branch.  The certificate is
    deliberately geometric: every source segment must be present, the component
    must be bounded away from both scope exits, and it must stay inside the two
    independently covered route walls.  Long/open arms, sparse strokes and any
    source ink already owned by a route remain competitors.

    This explains an obstruction to a *projected* corridor search.  It does not
    classify a physical fitting/support or establish physical continuity.
    """
    by_ref = {row['source_primitive_ref']: row for row in sources}
    unresolved = {row['source_primitive_ref'] for row in source_classifications
                  if row['classification'] == 'unresolved_incident_endpoint_or_interior_feature'
                  and by_ref[row['source_primitive_ref']]['source_native_segment']['kind'] == 'line'}
    endpoint_members = defaultdict(set)
    for ref in unresolved:
        for point in by_ref[ref]['points_display']:
            endpoint_members[(round(point[0] / TOLERANCE), round(point[1] / TOLERANCE))].add(ref)
    adjacency = defaultdict(set)
    for refs in endpoint_members.values():
        for ref in refs:
            adjacency[ref].update(refs)
    components, remaining = [], set(unresolved)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue, refs = [seed], []
        for ref in queue:
            refs.append(ref)
            for other in sorted(adjacency[ref].intersection(remaining)):
                remaining.remove(other)
                queue.append(other)
        points = [point for ref in refs for point in by_ref[ref]['points_display']]
        projected = [local(point) for point in points]
        along = [point[0] for point in projected]
        across = [point[1] for point in projected]
        # Eight observed pieces excludes an arbitrary short/open stroke.  A
        # component may lie on both walls (a native cap/rail) or be a compact
        # two-dimensional body wholly between them.
        spans_walls = min(across) <= -half + TOLERANCE and max(across) >= half - TOLERANCE
        body_2d = max(along) - min(along) > TOLERANCE and max(across) - min(across) > TOLERANCE
        reasons = []
        if len(refs) < 8:
            reasons.append('insufficient_native_component_members')
        if min(along) <= TOLERANCE or max(along) >= length - TOLERANCE:
            reasons.append('component_reaches_named_scope_exit')
        if max(along) - min(along) > 2 * half + TOLERANCE:
            reasons.append('component_has_long_axial_extent')
        if ((min(across) < -half - TOLERANCE or max(across) > half + TOLERANCE)
                and (not spans_walls or max(across) - min(across) > 4 * half + TOLERANCE)):
            reasons.append('component_exits_route_envelope_without_bounded_transverse_passage')
        if not spans_walls and not body_2d:
            reasons.append('component_is_sparse_inside_envelope')
        dual = sorted(set(refs).intersection(protected_route_source_refs or ()))
        if dual:
            reasons.append('component_ink_also_supports_certified_route')
        components.append({'id': _stable_id('mep_bounded_inline_detail_component', sorted(refs)),
            'source_primitive_refs': sorted(refs), 'state': 'accepted' if not reasons else 'abstained',
            'reasons': reasons, 'member_count': len(refs),
            'local_bbox': [min(along), min(across), max(along), max(across)],
            'spans_both_route_walls': spans_walls, 'bounded_two_dimensional_body': body_2d,
            'dual_role_source_refs': dual, 'physical_role_established': False,
            'physical_continuity_established': False, 'quantity_eligible': False})
    return components


def native_outer_cycle(group):
    """Walk native segment faces; observed intersections grant no pipe join."""
    segments = [s['points_display'] for s in group]
    points = [p for segment in segments for p in segment]
    for i, (a, b) in enumerate(segments):
        for c, d in segments[:i]:
            crossing = _segment_intersection(a, b, c, d)
            if crossing:
                points.append(crossing[0])
    canonical = []
    for point in sorted(points):
        matches = [p for p in canonical if math.dist(point, p) <= TOLERANCE]
        if len(matches) > 1:
            return []
        if not matches:
            canonical.append(tuple(point))
    adjacency = defaultdict(set)
    for a, b in segments:
        on = sorted((math.dist(a, p), p) for p in canonical if _on_segment(p, a, b))
        for (_, p), (_, q) in zip(on, on[1:]):
            if p != q:
                adjacency[p].add(q)
                adjacency[q].add(p)
    neighbors = {p: sorted(values, key=lambda q: math.atan2(q[1] - p[1], q[0] - p[0])) for p, values in adjacency.items()}
    visited, cycles = set(), []
    for start in sorted((p, q) for p in adjacency for q in adjacency[p]):
        if start in visited:
            continue
        previous, current = start
        path = []
        while (previous, current) not in visited:
            visited.add((previous, current))
            path.append(previous)
            choices = neighbors[current]
            previous, current = current, choices[(choices.index(previous) - 1) % len(choices)]
        if (previous, current) == start and len(path) >= 3 and len(set(path)) == len(path):
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(path, path[1:] + path[:1])) / 2
            cycles.append((abs(area), path))
    if not cycles:
        return []
    path = max(cycles)[1]
    if not all(_inside(p, path) for segment in segments for p in segment):
        return []
    return [list(p) for p in path]


def closed_line_motifs(unresolved, all_supports, sources, local, half, box, style, through_refs):
    """Native closed finite glyph/body with no additional open exterior arm.

    A native face walk nominates a boundary: every edge must independently have
    source line coverage. Neither a bounding box nor a boolean union closes it.
    """
    candidates = []
    for support in all_supports:
        a, b = map(local, support['points_display'])
        on_wall = abs(a[1] - b[1]) <= TOLERANCE and abs(abs(a[1]) - half) <= TOLERANCE
        if not on_wall and math.dist(a, b) <= 8 * half and all(
                box[0] + TOLERANCE < p[0] < box[2] - TOLERANCE and box[1] + TOLERANCE < p[1] < box[3] - TOLERANCE
                for p in support['points_display']):
            candidates.append(support)
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(candidates):
        for j in range(i):
            b = candidates[j]
            if any(_on_segment(p, *b['points_display']) for p in a['points_display']) or any(
                    _on_segment(p, *a['points_display']) for p in b['points_display']) or _segment_intersection(*a['points_display'], *b['points_display']):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, candidate in enumerate(candidates):
        groups[find(i)].append(candidate)
    motifs = []
    for group in groups.values():
        hull = native_outer_cycle(group)
        if len(hull) < 3:
            continue
        if not any(all(_inside(p, hull) for p in s['points_display']) for s in unresolved):
            continue
        measured = [local(p) for p in hull]
        if any(max(p[i] for p in measured) - min(p[i] for p in measured) > 8 * half for i in (0, 1)):
            continue
        if any(not (box[0] + TOLERANCE < p[0] < box[2] - TOLERANCE and box[1] + TOLERANCE < p[1] < box[3] - TOLERANCE) for p in hull):
            continue
        paths = [boundary_coverage(a, b, sources, style) for a, b in zip(hull, hull[1:] + hull[:1])]
        if any(not path for path in paths):
            continue
        exits = []
        through = []
        for support in all_supports:
            a, b = map(local, support['points_display'])
            on_wall = abs(a[1] - b[1]) <= TOLERANCE and abs(abs(a[1]) - half) <= TOLERANCE
            inside = [_inside(p, hull) for p in support['points_display']]
            if not on_wall and any(inside) and not all(inside):
                exits.append(support['id'])
            elif not on_wall and not any(inside):
                p, q = support['points_display']
                crosses = any((_cross(p, q, c) * _cross(p, q, d) < 0 and _cross(c, d, p) * _cross(c, d, q) < 0)
                              for c, d in zip(hull, hull[1:] + hull[:1]))
                if crosses:
                    (through if support['id'] in through_refs else exits).append(support['id'])
        if exits:
            continue
        refs = sorted({ref for path in paths for ref in path})
        motifs.append({'id': _stable_id('mep_closed_trace_line_motif', hull, refs),
            'boundary_points_display': hull, 'boundary_source_primitive_refs': refs,
            'native_boundary_paths': [{'points_display': [a, b], 'source_primitive_refs': path}
                for a, b, path in zip(hull, hull[1:] + hull[:1], paths)],
            'additional_open_port_support_refs': [], 'entire_body_inside_complete_query': True,
            'independent_transverse_support_refs': sorted(through),
            'interpretation': 'finite_closed_line_detail_with_no_additional_projected_port',
            'epistemic_state': 'inferred', 'physical_fitting_class': 'unknown', 'quantity_eligible': False})
    return motifs


def native_dimension_role_candidates(scope, query):
    """Native ticked dimension motifs, without granting exclusive ownership.

    Existing dimension terminals nominate candidates. Exact native contacts,
    repeated extension feet and an unbranched native text connector supply the
    witness. The shared scope owner still checks protected/competing geometry.
    No dimension value is applied to the route or to quantities.
    """
    from src.drawing_engine.core.dimension_attachment import LineSegment, _axis, _terminal_refs, _terminal_kind
    from src.drawing_engine.disciplines.mep.mep_terminology_proposals import interpret_mep_text
    from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import FrozenNativeQueries

    rows, search = query.get('source_rows', []), query.get('search', {})
    texts = query.get('source_observations', [])
    FrozenNativeQueries(query)
    refs = [r['source_primitive_ref'] for r in rows]
    if (query.get('page_ref') != scope.get('page_ref') or search.get('complete') is not True
            or search.get('budget_exhausted') or len(refs) != len(set(refs))
            or len({text['id'] for text in texts}) != len(texts)
            or search.get('source_rows_sha256') != _sha256(rows)
            or search.get('all_source_refs_sha256') != _sha256(sorted(refs))
            or (search.get('source_observations_sha256') is not None
                and search['source_observations_sha256'] != _sha256(texts))):
        return []
    architectural_texts = {text['id'] for text in texts
        if text.get('method') == 'native_pdf_text' and interpret_mep_text(text).get('text_role') == 'architectural_distance'}
    if not architectural_texts:
        return []
    box = search['bbox_display']

    def inside(point, bounds=box):
        return all(bounds[i] - TOLERANCE <= point[i] <= bounds[i + 2] + TOLERANCE for i in (0, 1))

    def signature(row):
        style = row['source_native_segment']['style']
        return style.get('width'), style.get('dash')

    lines = [r for r in rows if r['source_native_segment']['kind'] == 'line'
        and r['source_native_segment']['style'].get('stroke') is not None
        and r['source_native_segment']['style'].get('fill') is None
        and r['source_native_segment']['style'].get('width') is not None]
    native_lines = tuple(LineSegment(tuple(r['points_display'][0]), tuple(r['points_display'][-1]),
        r['source_native_segment']['style']['width'], r['source_primitive_ref']) for r in lines)
    by_ref = {r['source_primitive_ref']: r for r in lines}
    groups = defaultdict(list)
    for row in lines:
        groups[signature(row)].append(row)
    supports = []
    for style, members in groups.items():
        try:
            merged = straight_supports(members)
        except TraceSearchBudgetExceeded:
            return []  # No role claim: the original trace blockers remain.
        for support in {r['id']: r for r in merged.values()}.values():
            segment = LineSegment(*map(tuple, support['points_display']), style[0], support['id'])
            supports.append({**support, 'style_signature': style, 'axis': _axis(segment)})
    candidates = []
    for baseline in supports:
        orientation = baseline['axis']
        if orientation is None:
            continue
        along = 0 if orientation == 'horizontal' else 1
        cross = 1 - along
        points = sorted(baseline['points_display'], key=lambda p: p[along])
        if abs(points[0][cross] - points[1][cross]) > TOLERANCE or not all(inside(p) for p in points):
            continue
        ticks = []
        feet = []
        for point in points:
            terminals = [ref for ref in _terminal_refs(tuple(point), native_lines)
                         if _on_segment(point, *by_ref[ref]['points_display'])]
            terminal_groups = defaultdict(list)
            for ref in terminals:
                terminal_groups[signature(by_ref[ref])].append(ref)
            tick_groups = []
            for terminal_style, terminal_refs in terminal_groups.items():
                geometry = {tuple(sorted(map(tuple, by_ref[ref]['points_display']))) for ref in terminal_refs}
                if (len(geometry) == 2 and terminal_style[0] > baseline['style_signature'][0]
                        and _terminal_kind(tuple(point), orientation, tuple(terminal_refs), native_lines) == 'tick'):
                    tick_groups.append(terminal_refs)
            if len(tick_groups) != 1:
                break
            terminals = tick_groups[0]
            extensions = [s for s in supports if s['style_signature'] == baseline['style_signature']
                and s['axis'] in {'horizontal', 'vertical'} and s['axis'] != orientation
                and _on_segment(point, *s['points_display'])
                and min(p[cross] for p in s['points_display']) < point[cross] - TOLERANCE
                and max(p[cross] for p in s['points_display']) > point[cross] + TOLERANCE]
            if len(extensions) != 1:
                break
            ticks.append(sorted(terminals))
            feet.append(extensions[0])
        if len(ticks) != 2:
            continue
        profiles = [sorted(p[cross] - point[cross] for p in foot['points_display']) for foot, point in zip(feet, points)]
        if any(abs(a - b) > TOLERANCE for a, b in zip(*profiles)):
            continue
        geometric_refs = sorted({*baseline['source_primitive_refs'], *(ref for foot in feet for ref in foot['source_primitive_refs']),
                                 *(ref for terminal in ticks for ref in terminal)})
        if not all(all(box[i] + TOLERANCE < p[i] < box[i+2] - TOLERANCE for i in (0, 1))
                   for ref in geometric_refs for p in by_ref[ref]['points_display']):
            continue
        connections = []
        for text in texts:
            if (text.get('page_ref') != scope['page_ref'] or text.get('method') != 'native_pdf_text'
                    or not text.get('source_native_ref') or not text.get('source_pdf_sha256')):
                continue
            bounds = text['bbox_display']
            if not all(inside(p) for p in (bounds[:2], bounds[2:])):
                continue
            if bounds[along + 2] - bounds[along] <= bounds[cross + 2] - bounds[cross]:
                continue
            height = bounds[cross + 2] - bounds[cross]
            if max(abs(v) for v in profiles[0]) > 2 * height:
                continue
            eligible = [r for r in lines if signature(r) == baseline['style_signature']
                        and r['source_primitive_ref'] not in geometric_refs]
            for start in eligible:
                for end, point in enumerate(start['points_display']):
                    if not inside(point, bounds):
                        continue
                    path, current, entered = [], start, end
                    for _ in range(32):
                        ref = current['source_primitive_ref']
                        if ref in path or not all(inside(p) for p in current['points_display']):
                            break
                        path.append(ref)
                        if any(_on_segment(p, *current['points_display'])
                               and all(math.dist(p, q) > TOLERANCE for q in current['points_display'])
                               for other in eligible if other['source_primitive_ref'] not in path
                               for p in other['points_display']):
                            break  # A midpoint branch cannot disappear into a text connector.
                        endpoint = current['points_display'][1 - entered]
                        if _on_segment(endpoint, *points):
                            connections.append((text, path))
                            break
                        adjacent = [(r, i) for r in eligible if r['source_primitive_ref'] not in path
                                    for i, p in enumerate(r['points_display']) if math.dist(endpoint, p) <= TOLERANCE]
                        if len(adjacent) != 1:
                            break
                        current, entered = adjacent[0]
        connections = list({(text['id'], tuple(path)): (text, path) for text, path in connections}.values())
        if not any(text['id'] in architectural_texts for text, _ in connections):
            continue
        all_refs = sorted({*geometric_refs, *(ref for _, path in connections for ref in path)})
        candidates.append({'id': _stable_id('mep_native_dimension_role_candidate', scope['id'], baseline['id'],
                                              sorted(text['id'] for text, _ in connections)),
            'page_ref': scope['page_ref'], 'scope_ref': scope['id'], 'role': 'dimension_annotation',
            'state': 'candidate', 'epistemic_state': 'inferred', 'exclusive_role': False,
            'role_geometry_established': len(connections) == 1,
            'reasons': [] if len(connections) == 1 else ['competing_native_dimension_text_paths'],
            'source_primitive_refs': all_refs,
            'source_observation_refs': sorted({text['id'] for text, _ in connections}),
            'native_query_ref': query.get('id'), 'native_query_sha256': _sha256(query),
            'baseline': baseline, 'dimension_points_display': points,
            'extension_feet': feet, 'terminal_source_refs_by_endpoint': ticks,
            'text_connectors': [{'source_observation_ref': text['id'], 'source_primitive_refs': path,
                                 'text_bbox_display': text['bbox_display']} for text, path in connections],
            'entire_motif_inside_complete_query': True,
            'physical_continuation_established': False, 'quantity_eligible': False})
    return candidates


def transverse_cubic_crossing(source, local, length, half):
    """Prove one transverse passage from native Bezier controls, not samples.

    The derivative's Bernstein coefficients prove monotonicity across the
    corridor. De Casteljau subdivision bounds the entire intervening curve
    strictly between the two scope interfaces. No endpoint is inside the
    corridor. Other incident primitives are still classified independently;
    this establishes neither a stroke role nor a route connection.
    """
    native = source['source_native_segment']
    controls = native.get('control_points_display', [])
    if (native['kind'] != 'cubic' or len(controls) != 2
            or native['style'].get('fill') is not None or native['style'].get('stroke') is None):
        return None
    matrix = source.get('pdf_to_display_matrix', [1, 0, 0, 1, 0, 0])
    def transform(p):
        a, b, c, d, e, f = matrix
        return [a*p[0]+c*p[1]+e, b*p[0]+d*p[1]+f]
    points = [local(transform(p)) for p in [native['start_display'], *controls, native['end_display']]]
    if points[0][1] > points[-1][1]:
        points.reverse()
    if not points[0][1] < -half-TOLERANCE or not points[-1][1] > half+TOLERANCE:
        return None
    if any(b[1] < a[1] - TOLERANCE for a, b in zip(points, points[1:])):
        return None

    def split(poly, t):
        levels = [poly]
        while len(levels[-1]) > 1:
            levels.append([[(1-t)*a[i]+t*b[i] for i in (0, 1)]
                           for a, b in zip(levels[-1], levels[-1][1:])])
        return [level[0] for level in levels], [level[-1] for level in reversed(levels)]

    def root_bounds(value):
        lo, hi = 0., 1.
        for _ in range(48):
            mid = (lo+hi)/2
            if split(points, mid)[0][-1][1] < value:
                lo = mid
            else:
                hi = mid
        return lo, hi

    lo = root_bounds(-half-TOLERANCE)[0]
    hi = root_bounds(half+TOLERANCE)[1]
    clipped = split(split(points, lo)[1], (hi-lo)/(1-lo))[0]
    if not all(TOLERANCE < p[0] < length-TOLERANCE for p in clipped):
        return None
    return {'method': 'monotone_native_cubic_transverse_v1',
        'source_primitive_ref': source['source_primitive_ref'],
        'native_geometry_sha256': _sha256(native), 'native_controls_local': points,
        'crossing_parameter_interval': [lo, hi], 'crossing_control_hull_local': clipped,
        'native_endpoint_attachment_inside_corridor': False,
        'other_incident_primitives_examined_separately': True,
        'route_identity_established': False, 'physical_continuation_established': False}


def bound_annotation_stroke_roles(scope, query, attribute_bindings, protected_route_source_refs):
    """Reuse accepted M4 dot-leader ownership without granting applicability.

    M4 already replays the complete native dot and leader path against an exact
    route target.  A trace may use that same certificate to explain the leader
    ink inside its corridor.  Route boundary IDs are explicitly excluded, and
    the role cannot carry any attribute or physical-continuity authority.
    """
    accepted_refs = {ref for relation in attribute_bindings.get('relations', [])
                     if relation.get('state') == 'accepted'
                     and relation.get('target_kind') == 'route_composite'
                     and relation.get('target_refs') == [scope['composite_ref']]
                     for ref in relation.get('binding_evidence_refs', [])}
    query_refs = {row['source_primitive_ref'] for row in query.get('source_rows', [])}
    protected = set(protected_route_source_refs or ())
    roles = []
    for evidence in attribute_bindings.get('binding_evidence', []):
        certificate = evidence.get('automatic_search_certificate', {})
        if (evidence.get('id') not in accepted_refs
                or evidence.get('state') != 'observed'
                or evidence.get('method') != {'name': 'complete_native_dot_leader_contact', 'version': '1.0.0'}
                or evidence.get('page_ref') != scope['page_ref']
                or evidence.get('target_kind') != 'route_composite'
                or evidence.get('target_refs') != [scope['composite_ref']]
                or certificate.get('reviewed_selectors_used') is not False
                or not certificate.get('leader_observation_ref')
                or not certificate.get('source_searches_sha256')):
            continue
        geometric = set(evidence.get('geometric_evidence_refs', []))
        role_refs = sorted(geometric.intersection(query_refs) - protected)
        if not role_refs:
            continue
        roles.append({'id': _stable_id('mep_bound_annotation_stroke_role', evidence['id'], role_refs),
            'role': 'native_text_dot_leader', 'state': 'accepted', 'epistemic_state': 'derived',
            'binding_evidence_ref': evidence['id'],
            'source_observation_ref': certificate['leader_observation_ref'],
            'source_primitive_refs': role_refs,
            'excluded_route_boundary_source_refs': sorted(geometric.intersection(protected)),
            'dual_role_source_refs': [], 'reasons': [],
            'attribute_applicability_established': False,
            'physical_continuation_established': False, 'quantity_eligible': False})
    return roles


def certify_trace(scope, query, *, protected_route_source_refs=None, source_pdf_sha256=None,
                  attribute_bindings=None):
    ports = scope['ports']
    origin = ports[0]['center']
    delta = _sub(ports[1]['center'], origin)
    length = math.hypot(*delta)
    along = [v / length for v in delta]
    across = [-along[1], along[0]]
    half = ports[0]['width'] / 2

    def local(point):
        return [_dot(_sub(point, origin), axis) for axis in (along, across)]

    row = {'id': _stable_id('mep_projected_trace_completion', scope['id']),
        'record_type': 'mep_projected_trace_completion', 'record_version': '0.1.0',
        'page_ref': scope['page_ref'], 'scope_ref': scope['id'], 'scope': deepcopy(scope),
        'composite_refs': [scope['composite_ref']], 'native_junction_refs': sorted(i['native_junction_ref'] for i in scope['interfaces']),
        'endpoint_classifications': deepcopy(scope['interfaces']),
        'state': 'abstained', 'epistemic_state': 'unknown', 'complete_trace_established': False,
        'projected_scope_complete': False, 'physical_continuation_established': False,
        'physical_run_complete': False, 'quantity_eligible': False, 'engineer_approved': False,
        'source_query_ref': query.get('id'), 'source_query_sha256': _sha256(query),
        'source_primitive_refs': [], 'source_classifications': [], 'reasons': []}
    search, sources = query['search'], query['source_rows']
    if query.get('scope_ref') != scope['id'] or query.get('page_ref') != scope['page_ref']:
        row['reasons'].append('query_does_not_belong_to_trace_scope')
    if search.get('bbox_display') != scope['expected_query_bbox_display']:
        row['reasons'].append('complete_corridor_and_competitor_margin_not_queried')
    if not search.get('complete') or search.get('budget_exhausted'):
        row['reasons'].append('native_source_search_incomplete_or_exhausted')
    query_refs = [r['source_primitive_ref'] for r in sources]
    if len(query_refs) != len(set(query_refs)) or _sha256(sorted(query_refs)) != search.get('all_source_refs_sha256'):
        row['reasons'].append('native_query_source_inventory_changed')
    if _sha256(sources) != search.get('source_rows_sha256'):
        row['reasons'].append('native_query_source_geometry_changed')
    if row['reasons']:
        return row
    if (len(scope['interfaces']) != 2 or {i['port_ref'] for i in scope['interfaces']} != {p['id'] for p in ports}
            or any(i.get('classification') != 'certified_scope_interface' or not i.get('native_junction_ref') for i in scope['interfaces'])):
        row['reasons'].append('unexplained_endpoint_or_missing_certified_scope_interface')
    if any(i.get('unresolved_competing_candidate_refs') for i in scope['interfaces']):
        row['reasons'].append('unresolved_competing_native_interface_destination')
    work = {}
    pair_budget = search.get('classification_pair_budget', 2000000)
    if not isinstance(pair_budget, int) or isinstance(pair_budget, bool) or pair_budget <= 0:
        row['reasons'].append('invalid_native_classification_budget')
        return row
    try:
        supports = straight_supports(sources, pair_budget=pair_budget, diagnostics=work)
    except TraceSearchBudgetExceeded as error:
        row['reasons'].append(str(error))
        return row
    walls = []
    for side in ports[0]['sides']:
        opposite = min(ports[1]['sides'], key=lambda p: abs(local(p)[1] - local(side)[1]))
        refs = boundary_coverage(side, opposite, sources, ports[0]['style'])
        walls.append({'points_display': [side, opposite], 'source_primitive_refs': refs or [], 'complete': refs is not None})
    row['native_sidewall_coverage'] = walls
    if not all(wall['complete'] for wall in walls):
        row['reasons'].append('missing_native_sidewall_coverage')
    role_records = []
    if query.get('stroke_role_queries'):
        from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import scoped_native_stroke_roles
        if protected_route_source_refs is None or not source_pdf_sha256:
            raise ValueError('trace stroke roles require the frozen page route and document context')
        corridor_rows = {r['source_primitive_ref']: r for r in sources}
        own_refs = set(scope['input_composite']['member_source_primitive_refs'])
        for context in query['stroke_role_queries']:
            context_rows = {r['source_primitive_ref']: r for r in context['source_rows']}
            if context['page_ref'] != scope['page_ref'] or not own_refs.issubset(context_rows):
                raise ValueError('trace stroke role context omits the current route or changes page')
            if any(corridor_rows[ref] != source for ref, source in context_rows.items() if ref in corridor_rows):
                raise ValueError('trace stroke role context changes original native corridor evidence')
            for text in context.get('source_observations', []):
                if text.get('page_ref') != scope['page_ref'] or text.get('source_pdf_sha256') != source_pdf_sha256:
                    raise ValueError('trace annotation context differs from original document/page')
            if context.get('grid_context', {}).get('source_pdf_sha256', source_pdf_sha256) != source_pdf_sha256:
                raise ValueError('trace grid context differs from original document')
            protected = sorted(set(protected_route_source_refs).intersection(context_rows))
            role_scope = {'id': _stable_id('mep_trace_stroke_role_scope', scope['id'], context['search']['bbox_display']),
                'page_ref': scope['page_ref'], 'route_boundary_source_refs': protected}
            result = scoped_native_stroke_roles(role_scope, context)
            role_records.append(result)
        row['scoped_native_stroke_roles'] = role_records
    if attribute_bindings is not None:
        roles = bound_annotation_stroke_roles(
            scope, query, attribute_bindings, protected_route_source_refs)
        if roles:
            role_records.append({'scope_ref': scope['id'], 'page_ref': scope['page_ref'],
                'role_source': 'accepted_m4_complete_native_dot_leader_contact',
                'roles': roles, 'all_source_rows_retained': True,
                'attribute_applicability_established': False,
                'physical_continuation_established': False, 'quantity_eligible': False})
            row['bound_annotation_stroke_roles'] = roles
    for source in sources:
        native, points = source['source_native_segment'], source['points_display']
        observed = {'source_primitive_ref': source['source_primitive_ref'], 'native_kind': native['kind']}
        support = supports.get(source['source_primitive_ref'])
        local_points = [local(point) for point in (support['points_display'] if support else points)]
        if support:
            observed['native_straight_support_ref'] = support['id']
        if native['kind'] != 'line':
            # Conservative control-hull/search bounds, not endpoints only.
            box = source.get('search_bbox_display', source.get('bbox_display'))
            corners = [local([x, y]) for x in (box[0], box[2]) for y in (box[1], box[3])]
            intersects = (min(p[0] for p in corners) < length - TOLERANCE and max(p[0] for p in corners) > TOLERANCE
                and min(p[1] for p in corners) < half + TOLERANCE and max(p[1] for p in corners) > -half - TOLERANCE)
            observed['classification'] = 'unsupported_native_class_inside_trace' if intersects else 'outside_bounded_trace'
            if (intersects and search.get('native_curve_policy') == 'monotone_cubic_transverse_v1'
                    and source['source_primitive_ref'] not in (protected_route_source_refs or ())):
                crossing = transverse_cubic_crossing(source, local, length, half)
                if crossing:
                    observed.update(classification='transverse_curve_without_native_endpoint_attachment',
                                    native_curve_crossing=crossing)
        else:
            a, b = local_points[0], local_points[-1]
            clipped = _clip(a, b, length, half)
            if clipped is None:
                observed['classification'] = 'outside_bounded_trace'
            elif all(abs(p[0]) <= TOLERANCE or abs(p[0] - length) <= TOLERANCE for p in clipped) and abs(clipped[0][0] - clipped[1][0]) <= TOLERANCE:
                observed['classification'] = 'certified_interface_plane_observation'
            elif abs(a[1] - b[1]) <= TOLERANCE and abs(abs(a[1]) - half) <= TOLERANCE:
                observed['classification'] = 'coincident_native_sidewall_observation'
            elif (abs(a[1]) > half + TOLERANCE and abs(b[1]) > half + TOLERANCE and a[1] * b[1] < 0
                    and abs(clipped[0][1] - clipped[1][1]) >= 2 * half - 2 * TOLERANCE):
                observed['classification'] = 'transverse_crossing_without_native_endpoint_attachment'
            else:
                observed['classification'] = 'unresolved_incident_endpoint_or_interior_feature'
        row['source_classifications'].append(observed)
    inline_components = bounded_inline_detail_components(row['source_classifications'], sources, local,
        length, half, protected_route_source_refs)
    accepted_inline = {ref: component for component in inline_components if component['state'] == 'accepted'
                       for ref in component['source_primitive_refs']}
    for observed in row['source_classifications']:
        component = accepted_inline.get(observed['source_primitive_ref'])
        if observed['classification'] == 'unresolved_incident_endpoint_or_interior_feature' and component:
            observed.update(classification='bounded_inline_native_detail_with_continuous_sidewalls',
                            bounded_inline_detail_ref=component['id'])
    row['bounded_inline_detail_components'] = inline_components
    unresolved = [supports[r['source_primitive_ref']] for r in row['source_classifications']
        if r['classification'] == 'unresolved_incident_endpoint_or_interior_feature' and r['source_primitive_ref'] in supports]
    all_supports = list({r['id']: r for r in supports.values()}.values())
    if len(all_supports) > 1500:
        row['reasons'].append('native_body_graph_support_budget_exhausted')
        return row
    through_refs = {r.get('native_straight_support_ref') for r in row['source_classifications']
                    if r['classification'] == 'transverse_crossing_without_native_endpoint_attachment'}
    motifs = closed_line_motifs(unresolved, all_supports, sources, local, half,
                               scope['expected_query_bbox_display'], ports[0]['style'], through_refs)
    motifs = [m for m in motifs if not any(m['id'] != n['id'] and all(_inside(p, n['boundary_points_display'])
                for p in m['boundary_points_display']) for n in motifs)]
    row['closed_native_line_motifs'] = motifs
    for observed in row['source_classifications']:
        if observed['classification'] != 'unresolved_incident_endpoint_or_interior_feature':
            continue
        support = supports.get(observed['source_primitive_ref'])
        matches = [m for m in motifs if support and all(_inside(p, m['boundary_points_display']) for p in support['points_display'])]
        if len(matches) == 1:
            observed.update(classification='closed_native_line_motif_detail', native_line_motif_ref=matches[0]['id'])
    if role_records:
        by_source = defaultdict(list)
        for record in role_records:
            for role in record['roles']:
                for ref in role['source_primitive_refs']:
                    by_source[ref].append(role)
        for observed in row['source_classifications']:
            if observed['classification'] not in {'unsupported_native_class_inside_trace', 'unresolved_incident_endpoint_or_interior_feature'}:
                continue
            claims = by_source[observed['source_primitive_ref']]
            if claims and all(r['state'] == 'accepted' and not r.get('dual_role_source_refs') and not r.get('reasons') for r in claims):
                roles = {r['role'] for r in claims}
                if len(roles) == 1:
                    observed.update(classification='scope_certified_native_stroke_role',
                        native_stroke_role=next(iter(roles)), native_stroke_role_refs=sorted({r['id'] for r in claims}))
    counts = Counter(r['classification'] for r in row['source_classifications'])
    for name in ('unsupported_native_class_inside_trace', 'unresolved_incident_endpoint_or_interior_feature'):
        if counts[name]:
            row['reasons'].append(name)
    row['coverage'] = {'complete_source_query': True, 'query_margin_display_points': scope['query_margin_display_points'],
        'classification_pair_budget': pair_budget, 'classification_pair_checks': work['pair_check_count'],
        'body_graph_support_budget': 1500,
        'source_primitive_count': len(sources), 'source_classification_counts': dict(counts),
        'supported_interior_primitive_classes': ['native_straight_sidewall', 'independent_transverse_line_crossing', 'closed_native_line_motif'],
        'unsupported_classes_examined': sorted({r['native_kind'] for r in row['source_classifications'] if r['native_kind'] != 'line'}),
        'all_source_rows_retained_in_query': True, 'certified_endpoint_count': len(scope['interfaces']),
        'complete_page_inventory_established': False}
    row['source_primitive_refs'] = sorted(query_refs)
    row['native_straight_supports'] = all_supports
    if not row['reasons']:
        inferred_roles = any(role.get('epistemic_state') == 'inferred' for result in role_records for role in result['roles'])
        row.update(state='accepted', epistemic_state='inferred' if motifs or inferred_roles else 'derived', complete_trace_established=True, projected_scope_complete=True)
    return row


def certify_junction_interior(connection, *, version=2):
    """Inventory the closed native region between an already replayed pair of ports."""
    if version not in (1,2):
        raise ValueError('unsupported junction interior certificate version')
    paths = connection['boundary_paths']
    result = {'id': _stable_id('mep_projected_junction_coverage', connection['id']),
        'source_connection_ref': connection['id'], 'page_ref': connection['page_ref'],
        'state': 'abstained', 'projected_scope_complete': False, 'reasons': [],
        'source_classifications': [], 'source_rows_sha256': _sha256(connection['source_rows']),
        'physical_continuation_established': False, 'quantity_eligible': False}
    if (connection['state'] != 'accepted' or len(connection['ports']) != 2 or len(paths) != 2
            or not connection['search']['complete']):
        result['reasons'] = ['unsupported_or_uncovered_native_junction']
        return result
    hull = paths[0]['points_display'] + list(reversed(paths[1]['points_display']))
    def partial_sidewall(a,b):
        # Straight rectangular interfaces only. A long native member may end
        # midway along a tiny bridge boundary while extending outside the
        # scope. Its intersection is boundary ink, not a hidden interior arm.
        if (version<2 or connection.get('relation_type')!='projected_collinear_boundary_join'
                or any(len(p['points_display'])!=2 for p in paths)):
            return False
        for path in paths:
            c,d=path['points_display'];length=math.dist(c,d)
            if length<=TOLERANCE or any(abs(_cross(c,d,p))>TOLERANCE*length for p in (a,b)):
                continue
            direction=[(d[i]-c[i])/length for i in (0,1)]
            low,high=sorted(_dot(_sub(p,c),direction) for p in (a,b))
            if min(high,length)-max(low,0)>TOLERANCE:
                return True
        return False
    boundary_refs = {ref for path in paths for ref in path['source_primitive_refs']}
    edges = list(zip(hull, hull[1:]+hull[:1]))
    result['scope_boundary_display'] = deepcopy(hull)
    for source in connection['source_rows']:
        ref, native = source['source_primitive_ref'], source['source_native_segment']
        points = source['points_display']
        if ref in boundary_refs:
            classification = 'certified_native_junction_sidewall'
        elif native['kind'] == 'line':
            a, b = points[0], points[-1]
            if any(_on_segment(a,c,d) and _on_segment(b,c,d) for c,d in edges):
                classification = 'coincident_native_junction_boundary_observation'
            elif partial_sidewall(a,b):
                classification = 'partial_coincident_native_junction_sidewall'
            elif (math.dist(connection['ports'][0]['center'], connection['ports'][1]['center']) > TOLERANCE
                  and any(all(_dot(_sub(v,port['center']),port['outward']) <= TOLERANCE for v in (a,b))
                      and any(_dot(_sub(v,port['center']),port['outward']) < -TOLERANCE for v in (a,b))
                      for port in connection['ports'])):
                classification = 'outside_junction_scope_at_named_port'
            elif any(all(abs(_dot(_sub(v, port['center']), port['outward'])) <= TOLERANCE
                       for v in (a,b)) for port in connection['ports']) and (version<2 or
                       all(_inside(v,hull) or any(_on_segment(v,c,d) for c,d in edges) for v in (a,b))):
                classification = 'named_native_port_plane'
            elif (_inside(a, hull) or _inside(b, hull) or (version>=2 and
                  any(_on_segment(v,c,d) for v in (a,b) for c,d in edges))):
                classification = 'unresolved_incident_junction_geometry'
            elif any(_segment_intersection(a,b,c,d) for c,d in edges):
                classification = 'transverse_crossing_without_native_endpoint_attachment'
            else:
                classification = 'outside_junction_scope'
        else:
            box = source.get('search_bbox_display', source['bbox_display'])
            corners = [[box[0],box[1]], [box[2],box[1]], [box[2],box[3]], [box[0],box[3]]]
            intersects = (any(_inside(p,hull) for p in corners) or any(_inside(p,corners) for p in hull)
                or any(_segment_intersection(a,b,c,d) for a,b in edges for c,d in zip(corners,corners[1:]+corners[:1])))
            classification = 'unresolved_native_curve_at_junction' if intersects else 'outside_junction_scope'
        result['source_classifications'].append({'source_primitive_ref': ref, 'classification': classification})
    result['reasons'] = sorted({r['classification'] for r in result['source_classifications']
                                if r['classification'].startswith('unresolved_')})
    if not result['reasons']:
        result.update(state='accepted', projected_scope_complete=True)
    return result


def connected_trace_scopes(scoped_traces, connections, composites, graph):
    """Compose certified corridors and covered interfaces; keep parent gaps exact.

    Named exits are actual native port planes. They explicitly do not resolve
    the continuation outside the listed scope. Branches are boundaries, not a
    choice of which arm to follow. No new member is nominated or fabricated.
    """
    from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _samples
    by_target = {ref: t for t in scoped_traces for ref in t['composite_refs']}
    by_composite = {r['id']: r for r in composites['accepted_composites']}
    fragments = {r['id']: r for p in graph['pages'] for r in p['fragments']}
    joins = [j for j in connections if j['state'] == 'accepted' and len(j['composite_refs']) == 2]
    coverage = [certify_junction_interior(j) for j in joins]
    covered_joins = {r['source_connection_ref'] for r in coverage if r['projected_scope_complete']}
    adjacent = defaultdict(list)
    for j in joins:
        a,b = j['composite_refs']
        adjacent[a].append((b,j)); adjacent[b].append((a,j))

    def groups(members, allowed_joins):
        remaining, output = set(members), []
        while remaining:
            queue = [min(remaining)]; remaining.remove(queue[0])
            for ref in queue:
                for other,j in adjacent[ref]:
                    if other in remaining and j['id'] in allowed_joins:
                        remaining.remove(other); queue.append(other)
            output.append(sorted(queue))
        return output

    parents = []
    for group in groups(adjacent, {j['id'] for j in joins}):
        local = [j for j in joins if set(j['composite_refs']).issubset(group)]
        missing = [{'composite_ref': ref, 'scope_ref': by_target.get(ref,{}).get('scope_ref'),
            'reason_codes': by_target[ref]['reasons'] if ref in by_target else ['member_corridor_not_searched']}
            for ref in group if ref not in by_target or not by_target[ref]['projected_scope_complete']]
        uncovered = sorted(j['id'] for j in local if j['id'] not in covered_joins)
        parents.append({'id': _stable_id('mep_connected_projected_scope', group), 'composite_refs': group,
            'junction_refs': sorted(j['id'] for j in local), 'uncovered_members': missing,
            'uncovered_junction_refs': uncovered, 'projected_scope_complete': not missing and not uncovered,
            'physical_continuation_established': False, 'quantity_eligible': False})
    completed = []
    eligible = {ref for ref,t in by_target.items() if t['projected_scope_complete']}
    for group in groups(eligible, covered_joins):
        local = [j for j in joins if j['id'] in covered_joins and set(j['composite_refs']).intersection(group)]
        if not local:
            continue
        parts = [{'kind': 'certified_member_corridor', 'source_ref': ref,
                  'points_display': deepcopy(by_composite[ref]['derived_geometry']['centreline_points_display'])} for ref in group]
        exits = []
        for j in local:
            points = [p['center'] for p in j['ports']]
            if j['relation_type'] == 'projected_native_bend':
                sampled = [_samples(p['points_display'],17) for p in j['boundary_paths']]
                points = [[(a+b)/2 for a,b in zip(p,q)] for p,q in zip(*sampled)]
            parts.append({'kind': 'covered_native_junction', 'source_ref': j['id'], 'points_display': points})
            for port in j['ports']:
                if port['composite_ref'] not in group:
                    exits.append({'port_ref': port['id'], 'point_display': port['center'],
                        'outside_member_ref': port['composite_ref'], 'source_connection_ref': j['id'],
                        'classification': 'named_scope_exit_at_native_port', 'external_continuation_resolved': False})
        for ref in group:
            for interface in by_target[ref]['endpoint_classifications']:
                if interface['native_junction_ref'] not in covered_joins:
                    exits.append({**deepcopy(interface), 'classification': 'named_scope_exit_at_native_port',
                                  'external_continuation_resolved': False})
        scales = {fragments[f]['local_metric_observation'].get('drawing_inches_per_paper_inch')
                  for ref in group for f in by_composite[ref]['member_fragment_refs']}
        length = sum(math.dist(a,b) for part in parts for a,b in zip(part['points_display'],part['points_display'][1:]))
        scale = next(iter(scales)) if len(scales) == 1 else None
        completed.append({'id': _stable_id('mep_complete_connected_subtrace', group, sorted(j['id'] for j in local)),
            'page_ref': by_composite[group[0]]['page_ref'], 'composite_refs': group,
            'covered_junction_refs': sorted(j['id'] for j in local), 'centreline_parts': parts,
            'scope_exits': exits, 'member_corridor_count': len(group), 'covered_interface_count': len(local),
            'projected_length_display_points': length,
            'projected_length_m': length*scale*.0254/72 if scale is not None else None,
            'length_basis': 'sum of certified corridor centrelines and sampled paired native junction boundaries',
            'state': 'derived', 'projected_scope_complete': True,
            'external_continuation_resolved': False, 'constant_elevation_scope_established': False,
            'physical_continuation_established': False, 'installed_length_m': None, 'quantity_eligible': False})
    return {'junction_coverage': coverage, 'parent_scopes': parents, 'complete_subtraces': completed,
            'whole_network_completion_established': False, 'quantity_eligible': False}


def build_projected_trace_completion(*, graph, composites, boundary_connections, source_queries,
                                     attribute_bindings=None):
    scopes = trace_candidates(graph=graph, composites=composites, boundary_connections=boundary_connections)
    queries = {q['scope_ref']: q for q in source_queries}
    if len(queries) != len(source_queries) or set(queries) != {s['id'] for s in scopes}:
        raise ValueError('trace source queries do not cover the exact automatic scope candidate set')
    protected = defaultdict(set)
    for composite in composites['accepted_composites']:
        protected[composite['page_ref']].update(composite['member_source_primitive_refs'])
    if attribute_bindings is not None:
        if (attribute_bindings.get('m3_contract_ref', {}).get('payload_sha256') != _sha256(graph)
                or attribute_bindings.get('m3_5_contract_ref', {}).get('payload_sha256') != _sha256(composites)):
            raise ValueError('trace annotation roles require matching frozen M3/M3.5/M4 inputs')
    records = [certify_trace(scope, queries[scope['id']],
               protected_route_source_refs=protected[scope['page_ref']],
               source_pdf_sha256=graph['document']['source_pdf_sha256'],
               attribute_bindings=attribute_bindings) for scope in scopes]
    result = {'schema_version': '0.1.0', 'layer': 'mep_projected_trace_completion', 'document': deepcopy(graph['document']),
        'm3_payload_sha256': _sha256(graph), 'm35_payload_sha256': _sha256(composites),
        'boundary_connections_sha256': _sha256(boundary_connections), 'source_queries_sha256': _sha256(source_queries),
        'scoped_traces': records, 'complete_scoped_trace_count': sum(r['projected_scope_complete'] for r in records),
        'physical_continuation_established': False, 'quantity_eligible': False}
    if attribute_bindings is not None:
        result['attribute_bindings_sha256'] = _sha256(attribute_bindings)
    if source_queries and all(q['search'].get('connected_trace_policy') == 'covered_native_interfaces_v1' for q in source_queries):
        result['connected_trace_completion'] = connected_trace_scopes(records, boundary_connections['connections'], composites, graph)
    return result


def replay_projected_trace_completion(payload, **inputs):
    expected = build_projected_trace_completion(**inputs)
    if _sha256(payload) != _sha256(expected):
        raise ValueError('projected trace completion differs from full source replay')
    return expected
