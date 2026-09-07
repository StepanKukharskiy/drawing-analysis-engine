"""Drafting-pattern fitting hypotheses, separate from exact boundary topology.

A native closed body, independent closed ring and three paired mouth interfaces
support an inferred projected junction. Literal gaps remain immutable observations;
paired cap-ink overlap can support a separate drafting interpretation, never a
physical continuation or a change to the native snapping tolerance.
"""

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import math
from statistics import median

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import TOLERANCE, _dot, _sub, _ports, boundary_coverage
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


def _point(frame, x, y):
    return [frame[0][i] * x + frame[1][i] * y for i in (0, 1)]


def _local(point, frame):
    return [_dot(point, axis) for axis in frame]


def _compatible(row, style):
    native = row['source_native_segment']
    return native['style'].get('stroke') is not None and _style_compatible(style, _normalise_style(native['style']))


def plotting_precision(sources, style, frame, box, width):
    """A local coordinate-grid observation, not a replacement snap tolerance."""
    coordinates = [set(), set()]
    for row in sources:
        if not _compatible(row, style):
            continue
        for point in (row['points_display'][0], row['points_display'][-1]):
            if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]:
                for axis, value in enumerate(_local(point, frame)):
                    coordinates[axis].add(round(value, 3))
    differences = [b - a for values in coordinates for a, b in zip(sorted(values), sorted(values)[1:])
                   if b - a > 4 * TOLERANCE]
    bins = defaultdict(list)
    for difference in differences:
        if difference <= width / 4:
            bins[round(difference / TOLERANCE)].append(difference)
    if not bins or len(differences) < 12:
        return {'state': 'unknown', 'quantum_display_points': None, 'sample_count': len(differences)}
    values = max(bins.values(), key=lambda values: (len(values), -median(values)))
    quantum = median(values)
    agreeing = sum(abs(d - round(d / quantum) * quantum) <= 2 * TOLERANCE for d in differences)
    return {'state': 'observed_grid' if len(values) >= 4 and agreeing / len(differences) >= .85 else 'unknown',
        'quantum_display_points': quantum, 'sample_count': len(differences),
        'quantized_difference_count': agreeing, 'confidence_is_not_identity': True}


def native_rings(sources, style):
    """Find closed four-quadrant native cubic cycles; retain duplicate strokes."""
    groups = defaultdict(list)
    for row in sources:
        if row['source_native_segment']['kind'] != 'cubic' or not _compatible(row, style):
            continue
        points = row['points_display']
        a, b = points[0], points[-1]
        if min(abs(a[i] - b[i]) for i in (0, 1)) <= TOLERANCE:
            continue
        for center in ([a[0], b[1]], [b[0], a[1]]):
            radii = [math.dist(p, center) for p in points]
            radius = (radii[0] + radii[-1]) / 2
            if radius <= TOLERANCE or max(abs(r - radius) for r in radii) > max(TOLERANCE, .06 * radius):
                continue
            key = (*[round(v / TOLERANCE) for v in center], round(radius / TOLERANCE))
            middle = points[len(points) // 2]
            quadrant = math.floor((math.atan2(middle[1] - center[1], middle[0] - center[0]) % (2 * math.pi)) / (math.pi / 2))
            groups[key].append((row, center, radius, quadrant))
    rings = []
    for rows in groups.values():
        if {entry[3] for entry in rows} != {0, 1, 2, 3}:
            continue
        # Every sampled arc must begin and end at observed cycle vertices.
        ends = [tuple(round(v, 3) for v in point) for row, *_ in rows
                for point in (row['points_display'][0], row['points_display'][-1])]
        if len(set(ends)) != 4:
            continue
        refs = sorted(row['source_primitive_ref'] for row, *_ in rows)
        center = [median(entry[1][i] for entry in rows) for i in (0, 1)]
        radius = median(entry[2] for entry in rows)
        rings.append({'id': _stable_id('mep_native_fitting_ring', refs), 'center_display': center,
            'radius_display_points': radius, 'source_primitive_refs': refs,
            'closed_native_cycle': True, 'state': 'derived'})
    return rings


def native_bodies(ring, sources, style, frame, allowance):
    """Closed rectangles nominated by synchronized native sidewall endpoints."""
    center = _local(ring['center_display'], frame)
    radius = ring['radius_display_points']
    sides = [[], []]
    for row in sources:
        if row['source_native_segment']['kind'] != 'line' or not _compatible(row, style):
            continue
        a, b = [_local(p, frame) for p in (row['points_display'][0], row['points_display'][-1])]
        lo, hi = sorted((a[0], b[0]))
        if abs(a[1] - b[1]) > TOLERANCE or not 2 * radius - allowance <= hi - lo <= 6 * radius:
            continue
        if not (lo <= center[0] - radius + allowance and center[0] + radius - allowance <= hi):
            continue
        for sign, bucket in ((-1, 0), (1, 1)):
            if abs(a[1] - center[1] - sign * radius) <= allowance:
                sides[bucket].append((lo, hi, a[1], row['source_primitive_ref']))
    bodies = {}
    for a in sides[0]:
        for b in sides[1]:
            if max(abs(a[i] - b[i]) for i in (0, 1)) > TOLERANCE:
                continue
            local = [[a[0], a[2]], [a[1], a[2]], [b[1], b[2]], [b[0], b[2]]]
            points = [_point(frame, *p) for p in local]
            coverage = [boundary_coverage(p, q, sources, style) for p, q in zip(points, points[1:] + points[:1])]
            if any(refs is None for refs in coverage):
                continue
            key = tuple(round(v, 3) for p in local for v in p)
            refs = sorted({ref for rows in coverage for ref in rows})
            bodies[key] = {'id': _stable_id('mep_native_fitting_body', points, refs),
                'corners_display': points, 'bounds_local': [a[0], a[2], a[1], b[2]],
                'native_boundary_paths': [{'points_display': [p, q], 'source_primitive_refs': rows}
                    for p, q, rows in zip(points, points[1:] + points[:1], coverage)],
                'source_primitive_refs': refs, 'closed_native_body': True, 'state': 'derived'}
    values = list(bodies.values())
    # A unique containing rectangle is a geometric choice, not a score.
    outer = [a for a in values if all(a['bounds_local'][0] <= b['bounds_local'][0] + TOLERANCE
        and a['bounds_local'][1] <= b['bounds_local'][1] + TOLERANCE
        and a['bounds_local'][2] >= b['bounds_local'][2] - TOLERANCE
        and a['bounds_local'][3] >= b['bounds_local'][3] - TOLERANCE for b in values)]
    return outer if len(outer) == 1 else values


def native_collar(port, sources):
    """A transverse native cap with two observed, paired exterior shoulders."""
    cap = boundary_coverage(*port['sides'], sources, port['style'])
    result = {'state': 'abstained', 'source_primitive_refs': [], 'shoulders': []}
    if not cap:
        return result
    result.update(state='observed_cap', source_primitive_refs=cap)
    for side in port['sides']:
        normal = [(side[i] - port['center'][i]) / (port['width'] / 2) for i in (0, 1)]
        candidates = []
        for row in sources:
            if row['source_native_segment']['kind'] != 'line' or not _compatible(row, port['style']):
                continue
            points = [row['points_display'][0], row['points_display'][-1]]
            if any(abs(_dot(_sub(p, side), port['outward'])) > TOLERANCE for p in points):
                continue
            for point in points:
                extent = _dot(_sub(point, side), normal)
                if TOLERANCE < extent <= port['width'] / 2:
                    candidates.append((extent, point))
        shoulder = None
        for extent, point in sorted(candidates, reverse=True):
            refs = boundary_coverage(side, point, sources, port['style'])
            if refs:
                shoulder = {'extent_display_points': extent, 'outer_endpoint_display': point,
                    'source_primitive_refs': refs}
                break
        if shoulder is None:
            return result
        result['shoulders'].append(shoulder)
    if abs(result['shoulders'][0]['extent_display_points'] - result['shoulders'][1]['extent_display_points']) > 2 * TOLERANCE:
        return result
    result.update(state='supported', source_primitive_refs=sorted(set(cap + [ref for s in result['shoulders'] for ref in s['source_primitive_refs']])))
    return result


def port_interface(port, body, sources, frame, precision):
    center, direction = _local(port['center'], frame), _local(port['outward'], frame)
    axis = max((0, 1), key=lambda i: abs(direction[i]))
    if abs(abs(direction[axis]) - 1) > 1e-8:
        return {'state': 'abstained', 'reason': 'nonorthogonal_body_interface'}
    bounds = body['bounds_local']
    face = bounds[axis] if direction[axis] > 0 else bounds[axis + 2]
    distance = (face - center[axis]) / direction[axis]
    mouth = [[side[i] + distance * port['outward'][i] for i in (0, 1)] for side in port['sides']]
    result = {'port_ref': port['id'], 'composite_ref': port['composite_ref'], 'mouth_sides_display': mouth,
        'distance_display_points': distance, 'state': 'abstained', 'epistemic_state': 'unknown',
        'source_primitive_refs': [], 'drafting_gap_hypothesis': None}
    if distance < -TOLERANCE or any(not (bounds[0] - TOLERANCE <= p[0] <= bounds[2] + TOLERANCE
            and bounds[1] - TOLERANCE <= p[1] <= bounds[3] + TOLERANCE) for p in map(lambda p: _local(p, frame), mouth)):
        result['reason'] = 'port_does_not_meet_body_face'
        return result
    paths = [boundary_coverage(a, b, sources, port['style']) for a, b in zip(port['sides'], mouth)]
    ending = [[row['source_primitive_ref'] for row in sources if _compatible(row, port['style'])
               and row['source_native_segment']['kind'] == 'line'
               and (abs(distance) <= TOLERANCE or all(abs(_dot(_sub(point, target),
                    [-port['outward'][1], port['outward'][0]])) <= TOLERANCE for point in row['points_display']))
               and any(math.dist(point, target) <= TOLERANCE for point in (row['points_display'][0], row['points_display'][-1]))]
              for target in mouth]
    collar = native_collar(port, sources)
    result['native_collar'] = collar
    if collar['state'] == 'abstained':
        result['reason'] = 'native_port_cap_missing'
        return result
    if all(path is not None for path in paths) and all(ending):
        result.update(state='supported', epistemic_state='derived',
            source_primitive_refs=sorted({ref for refs in [*paths, *ending, collar['source_primitive_refs']] for ref in refs}))
        return result
    hypothesis = drafting_gap_for_port(port, sources, precision)
    if hypothesis is None or not all(ending):
        result['reason'] = 'paired_native_neck_or_body_attachment_missing'
        return result
    starts = hypothesis['continuation_endpoints_display']
    paths = [boundary_coverage(a, b, sources, port['style']) for a, b in zip(starts, mouth)]
    if any(path is None for path in paths):
        result['reason'] = 'native_neck_after_drafting_gap_does_not_reach_body'
        return result
    result.update(state='supported', epistemic_state='inferred', drafting_gap_hypothesis=hypothesis,
        source_primitive_refs=sorted({ref for refs in [*paths, *ending, collar['source_primitive_refs'],
            hypothesis['cap_source_primitive_refs'], hypothesis['continuation_source_primitive_refs']] for ref in refs}))
    return result


def drafting_gap_for_port(port, sources, precision):
    """Bounded paired interruption; body applicability must close separately."""
    quantum = precision.get('quantum_display_points') if precision['state'] == 'observed_grid' else None
    if quantum is None:
        return None
    gaps, gap_refs = [], []
    for side in port['sides']:
        candidates = []
        for row in sources:
            if row['source_native_segment']['kind'] != 'line' or not _compatible(row, port['style']):
                continue
            points = [row['points_display'][0], row['points_display'][-1]]
            along = [_dot(_sub(point, side), port['outward']) for point in points]
            if max(along) - min(along) <= TOLERANCE or not TOLERANCE < min(along) <= quantum + 2 * TOLERANCE:
                continue
            if all(math.dist(point, [side[i] + t * port['outward'][i] for i in (0, 1)]) <= TOLERANCE
                   for point, t in zip(points, along)):
                candidates.append((min(along), row['source_primitive_ref']))
        gap = min((a for a, _ in candidates), default=None)
        gaps.append(gap)
        gap_refs.append(sorted(ref for value, ref in candidates if gap is not None and abs(value - gap) <= TOLERANCE))
    if None in gaps or abs(gaps[0] - gaps[1]) > TOLERANCE:
        return None
    starts = [[side[i] + gap * port['outward'][i] for i in (0, 1)] for side, gap in zip(port['sides'], gaps)]
    if any(boundary_coverage(a, b, sources, port['style']) is not None for a, b in zip(port['sides'], starts)):
        return None  # An overlapping source already closes this side; no paired literal gap exists.
    ends = [[point[i] + quantum * port['outward'][i] for i in (0, 1)] for point in starts]
    paths = [boundary_coverage(a, b, sources, port['style']) for a, b in zip(starts, ends)]
    caps = [boundary_coverage(*points, sources, port['style']) for points in (port['sides'], starts)]
    by_ref = {row['source_primitive_ref']: row for row in sources}
    widths = [min((by_ref[ref]['source_native_segment']['style'].get('width') or 0 for ref in refs), default=0)
              for refs in caps if refs is not None]
    if any(path is None for path in paths) or any(cap is None for cap in caps) or len(widths) != 2 or max(gaps) >= sum(widths) / 2:
        return None
    hypothesis = {'id': _stable_id('mep_drafting_gap_hypothesis', port['id'], starts),
        'record_type': 'mep_drafting_gap_hypothesis', 'epistemic_state': 'inferred', 'port_ref': port['id'],
        'body_applicability_established': False,
        'original_endpoints_display': deepcopy(port['sides']), 'continuation_endpoints_display': starts,
        'observed_gaps_display_points': gaps, 'plotting_precision': deepcopy(precision),
        'cap_source_primitive_refs': sorted({ref for refs in caps for ref in refs}),
        'continuation_source_primitive_refs': sorted({ref for refs in gap_refs for ref in refs}),
        'cap_stroke_widths_display_points': widths,
        'cap_ink_overlap_display_points': sum(widths) / 2 - max(gaps),
        'ink_overlap_basis': 'transverse_cap_stroke_normal_extent',
        'line_end_cap_style': 'unobserved_not_assumed_or_required_for_transverse_normal_extent',
        'uncertainty_interval_display_points': [0, max(gaps)], 'native_strokes_modified': False,
        'physical_continuation_established': False, 'quantity_eligible': False}
    return hypothesis


def fitting_hypotheses_for_query(query):
    ports, sources = query['ports'], query['source_rows']
    base = {'id': _stable_id('mep_fitting_hypothesis', query['id']), 'record_type': 'mep_fitting_body_hypothesis',
        'record_version': '0.1.0', 'page_ref': query['page_ref'], 'source_query_ref': query['id'],
        'source_query_sha256': _sha256(query), 'search': deepcopy(query['search']),
        'ports': deepcopy(ports), 'composite_refs': [p['composite_ref'] for p in ports],
        'state': 'abstained', 'epistemic_state': 'unknown', 'accepted_projected_connection': False,
        'exact_boundary_certificate': False, 'physical_continuation_established': False,
        'port_inventory_complete': False, 'quantity_eligible': False, 'engineer_approved': False,
        'body_candidates': [], 'reasons': [], 'source_primitive_refs': []}
    if not query['search']['complete'] or _sha256(sorted(r['source_primitive_ref'] for r in sources)) != query['search']['all_source_refs_sha256']:
        base['reasons'] = ['incomplete_or_changed_native_source_query']
        return base
    opposite = [(a, b) for a, b in combinations(ports, 2) if _dot(a['outward'], b['outward']) < -1 + 1e-8]
    if len(ports) != 3 or len(opposite) != 1:
        base['reasons'] = ['three_orthogonal_port_pattern_missing']
        return base
    a, b = opposite[0]
    c = next(p for p in ports if p not in (a, b))
    if (abs(_dot(a['outward'], c['outward'])) > 1e-8
            or abs(_dot(_sub(a['center'], b['center']), [-a['outward'][1], a['outward'][0]])) > TOLERANCE
            or any(p['page_ref'] != query['page_ref'] or p['width'] <= 0
                   or not _style_compatible(p['style'], a['style']) for p in ports)):
        base['reasons'] = ['incompatible_port_frame_style_or_page']
        return base
    frame = [a['outward'], [-a['outward'][1], a['outward'][0]]]
    hub = [a['center'][i] + _dot(_sub(c['center'], a['center']), frame[0]) * frame[0][i] for i in (0, 1)]
    precision = plotting_precision(sources, a['style'], frame, query['search']['bbox_display'], min(p['width'] for p in ports))
    allowance = (precision['quantum_display_points'] if precision['state'] == 'observed_grid' else 0) + 2 * TOLERANCE
    rings = [ring for ring in native_rings(sources, a['style'])
             if max(abs(v) for v in _local(_sub(ring['center_display'], hub), frame)) <= allowance
             and min(abs(2 * ring['radius_display_points'] - p['width']) for p in ports) <= allowance]
    base.update(plotting_precision=precision, centreline_junction_display=hub,
                native_ring_candidates=rings, fitting_class='three_port_tee_hypothesis',
                classification_alternatives=['tee', 'three_port_valve', 'riser_tee'])
    base['drafting_gap_hypotheses'] = [gap for p in ports if (gap := drafting_gap_for_port(p, sources, precision))]
    for ring in rings:
        for body in native_bodies(ring, sources, a['style'], frame, allowance):
            interfaces = [port_interface(p, body, sources, frame, precision) for p in ports]
            base['body_candidates'].append({'body': body, 'ring': ring, 'port_interfaces': interfaces,
                'supported_port_count': sum(p['state'] == 'supported' for p in interfaces),
                'opposed_paired_collars': all(next(i for i in interfaces if i['port_ref'] == p['id'])
                    .get('native_collar', {}).get('state') == 'supported' for p in (a, b))})
    supported = [candidate for candidate in base['body_candidates']
                 if candidate['supported_port_count'] == 3 and candidate['opposed_paired_collars']]
    if len(supported) != 1:
        base['reasons'] = ['no_complete_body_and_three_port_interfaces' if not supported else 'competing_fitting_body_hypotheses']
        return base
    chosen = supported[0]
    base.update(state='accepted', epistemic_state='inferred', accepted_projected_connection=True,
        selected_body_ref=chosen['body']['id'], selected_ring_ref=chosen['ring']['id'],
        source_primitive_refs=sorted(set(chosen['body']['source_primitive_refs'] + chosen['ring']['source_primitive_refs']
            + [ref for interface in chosen['port_interfaces'] for ref in interface['source_primitive_refs']])),
        retained_through_stroke_refs=deepcopy(query.get('through_stroke_source_primitive_refs', [])),
        authority_boundary='inferred_projected_body_and_port_connectivity_only')
    return base


def build_fitting_hypotheses(*, source_queries, composites, graph):
    original_composites = {r['id']: r for r in composites['accepted_composites']}
    fragments = {r['id']: r for page in graph['pages'] for r in page['fragments']}
    native_ports = {p['id']: p for p in _ports(list(original_composites.values()), fragments)}
    records = []
    query_ids = set()
    for query in source_queries:
        if query['id'] in query_ids:
            raise ValueError('duplicate fitting source query')
        query_ids.add(query['id'])
        if any(native_ports.get(port['id']) != port for port in query['ports']):
            raise ValueError('fitting hypothesis ports differ from frozen M3/M3.5')
        row = fitting_hypotheses_for_query(query)
        row['input_composites'] = [deepcopy(original_composites[ref]) for ref in row['composite_refs']]
        row['input_fragments'] = [deepcopy(fragments[ref]) for comp in row['input_composites'] for ref in comp['member_fragment_refs']]
        row['sustained_route_evidence'] = [{'composite_ref': comp['id'],
            'centreline_points_display': deepcopy(comp['derived_geometry']['centreline_points_display']),
            'observed_extent_in_outline_widths': comp['derived_geometry']['projected_path_display_points']
                / comp['geometry_metrics']['mean_separation_display_points'],
            'member_fragment_refs': deepcopy(comp['member_fragment_refs'])} for comp in row['input_composites']]
        row['competing_port_search'] = {'scope': 'all original accepted composite endpoints inside complete native query',
            'complete': query['search']['complete'], 'other_body_interface_candidates': []}
        if row['accepted_projected_connection']:
            if any(r['observed_extent_in_outline_widths'] < 8 for r in row['sustained_route_evidence']):
                row.update(state='abstained', accepted_projected_connection=False)
                row['reasons'].append('short_outline_arm_does_not_exclude_actuator_or_handle')
            chosen = next(c for c in row['body_candidates'] if c['body']['id'] == row['selected_body_ref'])
            a = next(p for p in row['ports'] if any(_dot(p['outward'], q['outward']) < -1 + 1e-8 for q in row['ports']))
            frame = [a['outward'], [-a['outward'][1], a['outward'][0]]]
            box = query['search']['bbox_display']
            for port in native_ports.values():
                if (port['page_ref'] != row['page_ref'] or port['id'] in {p['id'] for p in row['ports']}
                        or not (box[0] <= port['center'][0] <= box[2] and box[1] <= port['center'][1] <= box[3])):
                    continue
                interface = port_interface(port, chosen['body'], query['source_rows'], frame, row['plotting_precision'])
                row['competing_port_search']['other_body_interface_candidates'].append(interface)
                if interface['state'] == 'supported':
                    row.update(state='abstained', accepted_projected_connection=False)
                    row['reasons'].append('additional_supported_native_body_port')
        records.append(row)
    incident = defaultdict(list)
    for row in records:
        if row['accepted_projected_connection']:
            for port in row['ports']:
                incident[port['id']].append(row)
    for row in records:
        if row['accepted_projected_connection'] and any(len(incident[p['id']]) != 1 for p in row['ports']):
            row.update(state='abstained', accepted_projected_connection=False)
            row['reasons'].append('competing_supported_fitting_destinations')
    return {'schema_version': '0.1.0', 'layer': 'mep_fitting_hypotheses', 'document': deepcopy(graph['document']),
        'm3_payload_sha256': _sha256(graph), 'm35_payload_sha256': _sha256(composites), 'fitting_hypotheses': records,
        'source_query_manifest': [{'id': q['id'], 'sha256': _sha256(q), 'page_ref': q['page_ref'],
                                   'search_complete': q['search']['complete']} for q in source_queries],
        'accepted_projected_connection_count': sum(row['accepted_projected_connection'] for row in records),
        'physical_continuation_established': False, 'quantity_eligible': False}


def replay_fitting_hypotheses(payload, *, source_queries, composites, graph):
    """Rebuild every outcome and evidence field from immutable source inputs."""
    expected = build_fitting_hypotheses(source_queries=source_queries, composites=composites, graph=graph)
    if _sha256(payload) != _sha256(expected):
        raise ValueError('fitting hypothesis payload differs from full source replay')
    return expected
