"""Native equipment review evidence, without adjacency-based item identity.

Closed rectangular outlines and paired boundary contacts are geometry
observations only. Neither is an equipment body or port certificate. The
separate tag, body and port outcomes deliberately cannot feed an M4 accept.
"""

from collections import defaultdict
from copy import deepcopy
import math

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_hvac_inventory import EQUIPMENT_CLASSES
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


VERSION = "0.1.0"
_EPS = .001  # Native numeric replay only; no endpoint clustering or snapping.


def _style(row):
    style = row['source_native_segment']['style']
    return (style.get('width'), style.get('dash'), style.get('fill') is not None)


def rectangular_outline_candidates(rows, minimum_side):
    """Enumerate exact four-native-edge rectangles; keep duplicate provenance.

    This bounded subset excludes fragmented sides and nonrectangular bodies;
    an empty result never means equipment is absent. All competing query rows
    remain available in the parent evidence artifact.
    """
    by_style = defaultdict(dict)
    for row in rows:
        native = row['source_native_segment']
        if native['kind'] != 'line' or native['style'].get('stroke') is None:
            continue
        a, b = map(tuple, (row['points_display'][0], row['points_display'][-1]))
        if a == b or not (a[0] == b[0] or a[1] == b[1]):
            continue
        by_style[_style(row)].setdefault(tuple(sorted((a, b))), []).append(row['source_primitive_ref'])
    candidates = []
    for style, edges in by_style.items():
        horizontal, vertical = defaultdict(list), defaultdict(list)
        for a, b in edges:
            if a[1] == b[1] and b[0] - a[0] >= minimum_side:
                horizontal[a].append(b)
            if a[0] == b[0] and b[1] - a[1] >= minimum_side:
                vertical[a].append(b)
        for a in sorted(set(horizontal).intersection(vertical)):
            for b in horizontal[a]:
                for d in vertical[a]:
                    c = (b[0], d[1])
                    sides = [(a, b), tuple(sorted((b, c))), tuple(sorted((d, c))), (a, d)]
                    if not all(side in edges for side in sides):
                        continue
                    refs = sorted({ref for side in sides for ref in edges[side]})
                    page_ref = rows[0]['page_ref']
                    candidates.append({'id': _stable_id('mep_native_body_outline_candidate', page_ref, refs),
                        'record_type': 'mep_native_body_outline_candidate',
                        'bbox_display': [*a, *c], 'points_display': [a, b, c, d, a],
                        'source_primitive_refs': refs,
                        'state': 'observed', 'epistemic_state': 'observed',
                        'geometry_kind': 'closed_four_edge_rectangular_outline',
                        'non_color_style': list(style),
                        'equipment_body_identity_established': False,
                        'quantity_eligible': False})
    return sorted(candidates, key=lambda row: row['id'])


def boundary_contact_candidates(outline, rows):
    """Preserve terminal incidents separately from lines continuing through.

    Paired native strokes may be pipe outlines, fittings, supports or other
    drawing content. Matching stroke geometry supplies no equipment-port
    identity and no invented centreline endpoint.
    """
    x0, y0, x1, y1 = outline['bbox_display']
    contacts = []
    own = set(outline['source_primitive_refs'])
    for row in rows:
        native = row['source_native_segment']
        if native['kind'] != 'line' or row['source_primitive_ref'] in own:
            continue
        a, b = row['points_display'][0], row['points_display'][-1]
        for side, axis, value, low, high in [('left', 0, x0, y0, y1), ('right', 0, x1, y0, y1),
                                            ('top', 1, y0, x0, x1), ('bottom', 1, y1, x0, x1)]:
            delta = b[axis] - a[axis]
            if abs(delta) <= _EPS or abs(b[1-axis] - a[1-axis]) > _EPS:
                continue
            t = (value - a[axis]) / delta
            coordinate = a[1-axis] + t * (b[1-axis] - a[1-axis])
            if not 0 <= t <= 1 or not low + _EPS < coordinate < high - _EPS:
                continue
            point = [0., 0.]; point[axis] = value; point[1-axis] = coordinate
            terminal = min(math.dist(point, a), math.dist(point, b)) <= _EPS
            contacts.append({'source_primitive_ref': row['source_primitive_ref'],
                'side': side, 'point_display': point,
                'contact_kind': 'native_endpoint_incidence' if terminal else 'native_through_stroke',
                'source_points_display': deepcopy(row['points_display']),
                'non_color_style': list(_style(row))})
    grouped = {}
    for contact in contacts:
        key = (contact['side'], tuple(contact['point_display']),
               tuple(sorted(map(tuple, contact['source_points_display']))),
               tuple(contact['non_color_style']), contact['contact_kind'])
        if key not in grouped:
            grouped[key] = {**contact, 'source_primitive_refs': []}
            grouped[key].pop('source_primitive_ref')
        grouped[key]['source_primitive_refs'].append(contact['source_primitive_ref'])
    contacts = list(grouped.values())
    pairs = []
    for i, left in enumerate(contacts):
        for right in contacts[i+1:]:
            if left['side'] != right['side'] or left['non_color_style'] != right['non_color_style']:
                continue
            separation = math.dist(left['point_display'], right['point_display'])
            if not .1 <= separation <= min(x1-x0, y1-y0) / 4:
                continue
            axis = 0 if left['side'] in {'left', 'right'} else 1
            intervals = [sorted(p[axis] for p in contact['source_points_display']) for contact in (left, right)]
            low, high = max(i[0] for i in intervals), min(i[1] for i in intervals)
            if left['side'] in {'left', 'top'}:
                high = min(high, left['point_display'][axis])
            else:
                low = max(low, left['point_display'][axis])
            if high - low < separation:
                continue
            refs = sorted(set(left['source_primitive_refs'] + right['source_primitive_refs']))
            through = any(c['contact_kind'] == 'native_through_stroke' for c in (left, right))
            pairs.append({'id': _stable_id('mep_native_port_geometry_candidate', outline['id'], refs, left['side']),
                'record_type': 'mep_native_port_geometry_candidate', 'body_outline_candidate_ref': outline['id'],
                'source_primitive_refs': refs, 'contacts': [left, right],
                'observed_separation_display_points': separation,
                'state': 'unresolved', 'epistemic_state': 'observed',
                'port_identity_established': False,
                'unresolved_reasons': ['through_strokes_are_not_terminal_ports' if through else
                                       'terminal_pair_requires_independent_port_opening_or_symbol_certificate'],
                'quantity_eligible': False})
    return sorted(pairs, key=lambda row: row['id'])


def build_equipment_review_outcome(*, proposal, observation, source_rows, search_bbox_display,
                                   search_complete, region_refs, leader_paths, leader_search_complete,
                                   pdf_grouping):
    if proposal['page_ref'] != observation['page_ref'] or any(
            r['page_ref'] != proposal['page_ref'] for r in source_rows):
        raise ValueError('equipment source evidence must remain on one page')
    candidate = proposal['candidate']
    if candidate.get('kind') != 'equipment_tag' or candidate.get('equipment_class_token') not in EQUIPMENT_CLASSES:
        raise ValueError('only existing supported equipment classes are eligible')
    box = observation['bbox_display']
    height = max(1., min(box[2]-box[0], box[3]-box[1]))
    outlines = rectangular_outline_candidates(source_rows, minimum_side=height / 4)
    port_candidates = [port for outline in outlines for port in boundary_contact_candidates(outline, source_rows)]
    contained = [r['id'] for r in outlines if r['bbox_display'][0] <= box[0] and r['bbox_display'][1] <= box[1]
                 and r['bbox_display'][2] >= box[2] and r['bbox_display'][3] >= box[3]]
    remote = observation['text'].upper().strip().startswith(('UP TO ', 'DOWN TO ', 'TO '))
    tag_reasons = ['remote_destination_mention_is_not_a_local_equipment_tag'] if remote else [
        'no_certified_tag_to_equipment_body_relation']
    if not leader_paths:
        tag_reasons.append('no_supported_native_dot_leader_chain_recovered')
    if not leader_search_complete:
        tag_reasons.append('leader_search_has_unsupported_or_competing_native_geometry')
    if not contained:
        tag_reasons.append('tag_is_not_contained_in_a_recovered_rectangular_outline')
    if not pdf_grouping.get('form_xobject_count') and not pdf_grouping.get('marked_content_count'):
        tag_reasons.append('source_pdf_has_no_form_or_marked_content_body_tag_ownership')
    row = {'id': _stable_id('mep_equipment_review_outcome', proposal['id'], _sha256(source_rows)),
        'record_type': 'mep_equipment_review_outcome', 'record_version': VERSION,
        'page_ref': proposal['page_ref'], 'page_number': observation['page_number'],
        'proposal_ref': proposal['id'], 'source_observation_refs': [observation['id']],
        'tag': candidate['tag'], 'equipment_class': EQUIPMENT_CLASSES[candidate['equipment_class_token']],
        'bbox_display': deepcopy(box), 'search_bbox_display': search_bbox_display,
        'state': 'abstained', 'epistemic_state': 'observed',
        'tag_body_outcome': {'state': 'unresolved', 'relation_established': False,
            'reasons': tag_reasons, 'native_leader_paths': deepcopy(leader_paths),
            'leader_search_complete': leader_search_complete, 'containing_outline_candidate_refs': contained},
        'body_outcome': {'state': 'unresolved', 'identity_established': False,
            'reasons': ['closed_rectangular_geometry_does_not_establish_equipment_body_identity'],
            'discovery_subset': 'four_exact_unsplit_axis_aligned_native_edges',
            'minimum_side_display_points': height / 4,
            'other_body_geometry_unresolved': True,
            'candidate_refs': [r['id'] for r in outlines]},
        'port_outcome': {'state': 'unresolved', 'identity_established': False,
            'reasons': ['no_independent_body_port_opening_or_symbol_certificate'],
            'discovery_subset': 'paired_axis_aligned_perpendicular_boundary_contacts',
            'candidate_refs': [r['id'] for r in port_candidates],
            'native_contact_geometry_observed': bool(port_candidates)},
        'body_outline_candidates': outlines, 'port_geometry_candidates': port_candidates,
        'native_query': {'complete': search_complete, 'region_refs': region_refs,
            'source_rows_sha256': _sha256(source_rows), 'source_primitive_count': len(source_rows),
            'source_primitive_refs': [r['source_primitive_ref'] for r in source_rows]},
        'pdf_grouping': deepcopy(pdf_grouping),
        'authority': {'tag_body_identity_established': False, 'equipment_port_binding_established': False,
            'physical_item_identity_established': False, 'quantity_eligible': False},
        'quantity_eligible': False}
    return row
