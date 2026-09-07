"""Projected connections proved by two native boundary traces, never gaps.

The initial certificate is deliberately restricted to collinear equal-width
ports. A coupling may bridge an interval only when both sidewalls have complete
native line coverage. This is projected annotation applicability, not a fitting
identity, physical continuation, installed length or quantity.
"""

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import math

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible
from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id, build_mep_terminology_proposals
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import _semantic_key


TOLERANCE = .001  # native PDF coordinate replay tolerance, not a snap distance


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def curve_may_touch_boundary(row, paths):
    """Conservative control-hull box test over entire boundary edges.

    A curve meeting the middle of a long native sidewall must not disappear
    because neither sidewall vertex lies inside the curve's search box.
    """
    box = row.get('search_bbox_display', row['bbox_display'])
    for path in paths:
        for a, b in zip(path['points_display'], path['points_display'][1:]):
            low, high = 0., 1.
            for axis in (0, 1):
                start, end = box[axis] - TOLERANCE, box[axis + 2] + TOLERANCE
                delta = b[axis] - a[axis]
                if delta == 0:
                    if not start <= a[axis] <= end:
                        high = -1.
                        break
                else:
                    enter, leave = sorted(((start - a[axis]) / delta, (end - a[axis]) / delta))
                    low, high = max(low, enter), min(high, leave)
            if low <= high:
                return True
    return False


def _ports(composites, fragments):
    for composite in composites:
        points = composite['derived_geometry']['centreline_points_display']
        if len(points) != 2:
            continue
        members = [fragments[ref] for ref in composite['member_fragment_refs']]
        width = composite['geometry_metrics']['mean_separation_display_points']
        for end in (0, 1):
            center = points[end]
            direction = _sub(center, points[1 - end])
            length = math.hypot(*direction)
            if not length:
                continue
            # Recover the two native ends; their mean must replay the centre.
            sides = [min((r['geometry']['points_display'][0], r['geometry']['points_display'][-1]),
                         key=lambda p: math.dist(p, center)) for r in members]
            if math.dist([(a + b) / 2 for a, b in zip(*sides)], center) > TOLERANCE:
                continue
            yield {'id': _stable_id('mep_composite_port', composite['id'], end),
                   'composite_ref': composite['id'], 'end': end, 'center': center,
                   'sides': sides, 'outward': [v / length for v in direction],
                   'width': width, 'style': members[0]['style'], 'page_ref': composite['page_ref']}


def boundary_coverage(start, end, sources, style):
    """Replay union coverage on one boundary. Every source stays immutable."""
    delta = _sub(end, start)
    length = math.hypot(*delta)
    if length <= TOLERANCE:
        return []
    direction = [v / length for v in delta]
    intervals = []
    for row in sources:
        native = row['source_native_segment']
        if (native['kind'] != 'line' or native['style'].get('stroke') is None
                or not _style_compatible(style, _normalise_style(native['style']))):
            continue
        points = row['points_display']
        positions = [_dot(_sub(p, start), direction) for p in points]
        if any(math.dist(p, [start[i] + t * direction[i] for i in (0, 1)]) > TOLERANCE
               for p, t in zip(points, positions)):
            continue
        a, b = sorted(positions)
        if b >= -TOLERANCE and a <= length + TOLERANCE and b - a > TOLERANCE:
            intervals.append((a, b, row['source_primitive_ref']))
    covered = 0.
    used = []
    for a, b, ref in sorted(intervals):
        if a > covered + TOLERANCE:
            break
        if b >= 0:
            used.append(ref)
            covered = max(covered, b)
    return sorted(set(used)) if covered >= length - TOLERANCE else None


def discover_native_boundary_connections(*, index, composites, graph, page_ref):
    fragments = {r['id']: r for p in graph['pages'] for r in p['fragments']}
    ports = list(_ports([r for r in composites['accepted_composites'] if r['page_ref'] == page_ref], fragments))
    records = []
    for left, right in combinations(ports, 2):
        if left['composite_ref'] == right['composite_ref']:
            continue
        delta = _sub(right['center'], left['center'])
        gap = _dot(delta, left['outward'])
        if (gap < -TOLERANCE or gap > min(left['width'], right['width'])
                or _dot(left['outward'], right['outward']) > -1 + 1e-8
                or math.dist(delta, [gap * v for v in left['outward']]) > TOLERANCE
                or abs(left['width'] - right['width']) > TOLERANCE
                or not _style_compatible(left['style'], right['style'])):
            continue
        pairs = [(a, min(right['sides'], key=lambda b: math.dist(a, b))) for a in left['sides']]
        if len({tuple(b) for _, b in pairs}) != 2:
            continue
        points = [p for pair in pairs for p in pair]
        radius = max(left['width'], right['width'])
        box = [min(p[0] for p in points) - radius, min(p[1] for p in points) - radius,
               max(p[0] for p in points) + radius, max(p[1] for p in points) + radius]
        sources, complete, regions = index.query(box)
        # Even exactly coincident outline endpoints need independent native
        # bridge strokes extending across the port plane on BOTH sides.
        members = {ref for port in (left, right) for ref in
                   next(c for c in composites['accepted_composites'] if c['id'] == port['composite_ref'])['member_source_primitive_refs']}
        bridge_sources = [r for r in sources if r['source_primitive_ref'] not in members]
        if gap <= TOLERANCE:
            extension = min(left['width'], right['width']) * .05
            pairs = [([p[i] - extension * left['outward'][i] for i in (0, 1)],
                      [q[i] + extension * left['outward'][i] for i in (0, 1)]) for p, q in pairs]
        paths = [boundary_coverage(a, b, bridge_sources, left['style']) for a, b in pairs]
        reasons = []
        if not complete:
            reasons.append('incomplete_native_boundary_search')
        if any(path is None for path in paths):
            reasons.append('native_sidewall_coverage_missing')
        refs = sorted({ref for path in paths if path is not None for ref in path})
        records.append({'id': _stable_id('mep_native_boundary_connection', left['id'], right['id']),
            'record_type': 'mep_native_boundary_connection', 'record_version': '0.1.0',
            'page_ref': page_ref, 'composite_refs': [left['composite_ref'], right['composite_ref']],
            'port_refs': [left['id'], right['id']], 'ports': [left, right],
            'boundary_paths': [{'points_display': [a, b], 'source_primitive_refs': path or []}
                               for (a, b), path in zip(pairs, paths)],
            'source_primitive_refs': refs,
            'source_rows': index.with_initial_search_refs(sources),
            'search': {'region_refs': regions, 'bbox_display': box, 'complete': complete,
                       'all_source_refs_sha256': _sha256(sorted(r['source_primitive_ref'] for r in sources))},
            'relation_type': 'projected_collinear_boundary_join',
            'state': 'accepted' if not reasons else 'abstained', 'reasons': reasons,
            'physical_continuation_established': False, 'quantity_eligible': False})
    # Multiple geometrically admissible destinations are never resolved by a
    # score or by selecting whichever happened to retain a bridge first.
    incident = defaultdict(list)
    for row in records:
        for ref in row['port_refs']:
            incident[ref].append(row)
    for row in records:
        if any(len(incident[ref]) != 1 for ref in row['port_refs']):
            row['state'] = 'abstained'
            row['reasons'].append('non_unique_projected_port_destination')
    return records


def propagate_collinear_applicability(*, terminology, binding_evidence, connections, direct_bindings,
                                     proposal_types=('system', 'inline_size')):
    """Create separate, traceable M2 scope overlays along certified sidewalls.

    Direct anchors are untouched. Callers may request system-only extension.
    The default retains the historical system/size replay contract. Only selected attributes travel over certified
    paired boundary geometry. Elevation requires a separate physical applicability
    certificate and is intentionally not propagated by projected topology.
    """
    adjacency = defaultdict(list)
    proposal_types = frozenset(proposal_types)
    if not proposal_types or not proposal_types <= {'system', 'inline_size'}:
        raise ValueError('boundary applicability permits explicit system/size channels only')
    if direct_bindings['m2_contract_ref']['payload_sha256'] != _sha256(terminology):
        raise ValueError('direct M4 applicability does not match frozen M2')
    accepted_evidence = {ref for r in direct_bindings['relations'] if r['state'] == 'accepted'
                         for ref in r['binding_evidence_refs']}
    direct_values = defaultdict(set)
    for relation in direct_bindings['relations']:
        if relation['state'] == 'accepted' and relation['relation_type'] in {'route_system', 'route_size'}:
            kind = {'route_system': 'system', 'route_size': 'inline_size'}[relation['relation_type']]
            for ref in relation['target_refs']:
                direct_values[ref, kind].add(_semantic_key(relation['candidate']))
    for row in connections:
        # Multi-port geometry does not prove annotation coverage of every arm.
        if (row['state'] == 'accepted' and len(row['composite_refs']) == 2
                and row['relation_type'] in {'projected_collinear_boundary_join', 'projected_native_bend'}):
            a, b = row['composite_refs']
            adjacency[a].append((b, row))
            adjacency[b].append((a, row))
    proposals = {r['id']: r for r in terminology['proposals']}
    observations = {r['id']: r for r in terminology['source_observations']}
    derived, specs = [], []
    for evidence in binding_evidence:
        proposal = proposals[evidence['proposal_ref']]
        if (evidence['id'] not in accepted_evidence or proposal['state'] != 'proposed'
                or proposal['proposal_type'] not in proposal_types
                or evidence['target_kind'] != 'route_composite' or len(evidence['target_refs']) != 1):
            continue
        start = evidence['target_refs'][0]
        paths = {start: []}
        queue = [start]
        for ref in queue:
            for other, connection in adjacency[ref]:
                values = direct_values.get((other, proposal['proposal_type']), set())
                if values and values != {_semantic_key(proposal['candidate'])}:
                    continue  # An observed attribute change ends applicability.
                if other not in paths:
                    paths[other] = paths[ref] + [connection]
                    queue.append(other)
        for target, path in sorted(paths.items()):
            if target == start:
                continue
            source = observations[proposal['anchor_ref']]
            identifier = _stable_id('mep_boundary_applicability', source['id'], target, proposal['proposal_type'])
            overlay = deepcopy(source)
            overlay.update(id=identifier, record_type='mep_boundary_applicability_observation',
                source_observation_ref=source['id'], immutable_source_observation=False,
                epistemic_state='derived', region_role='drawing_inline', interpretation_scope_ref=target,
                applicability_evidence={'source_binding_evidence_ref': evidence['id'],
                    'source_composite_ref': start, 'target_composite_ref': target,
                    'boundary_connection_refs': [r['id'] for r in path],
                    'projected_only': True, 'physical_continuation_established': False})
            derived.append(overlay)
            specs.append((identifier, target, proposal, evidence, path))
    if not derived:
        return terminology, binding_evidence
    additions = build_mep_terminology_proposals(document=terminology['document'], observations=derived)
    by_key = {(r['anchor_ref'], r['proposal_type'], _sha256(r['candidate'])): r for r in additions['proposals']}
    new_proposals, new_evidence = [], []
    for identifier, target, original, direct, path in specs:
        proposal = by_key[identifier, original['proposal_type'], _sha256(original['candidate'])]
        new_proposals.append(proposal)
        row = deepcopy(direct)
        row.update(id=_stable_id('mep_boundary_binding_evidence', proposal['id'], target),
            proposal_ref=proposal['id'], target_refs=[target],
            method={'name': 'paired_native_sidewall_applicability', 'version': '0.1.0'},
            geometric_evidence_refs=[target, identifier],
            boundary_connection_refs=[r['id'] for r in path])
        new_evidence.append(row)
    result = deepcopy(terminology)
    result['source_observations'].extend(derived)
    result['proposals'].extend(new_proposals)
    result['summary']['source_observation_count'] = len(result['source_observations'])
    result['summary']['proposal_count'] = len(result['proposals'])
    for field, key in (('proposal_type_counts', 'proposal_type'), ('state_counts', 'state')):
        result['summary'][field] = {v: sum(r[key] == v for r in result['proposals'])
                                   for v in sorted({r[key] for r in result['proposals']})}
    return result, [*binding_evidence, *new_evidence]
