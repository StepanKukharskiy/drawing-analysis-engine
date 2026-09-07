"""Duct-specific inline evidence for existing M3.5/M4 and bounded M5A.

Geometry is frozen before text is bound. Native queries include all source
roles; explicit air-service text inside one complete corridor establishes
page-local applicability, never physical connectivity or a purchase quantity.
"""
from copy import deepcopy
import math
import re

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import FrozenNativeQueries
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import validate_mep_outlined_route_composites
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id, validate_mep_terminology_proposals


def observe_native_duct_shapes(query):
    """Replay bounded shape observations where semantic duct attributes are missing.

    Authored closed quadrilaterals nominate panels or tapers; they do not name
    ducts. Paired curved chains nominate portals for the existing exhaustive
    bend replay. Exact duplicate geometry shares traversal edges but retains
    every native reference. Colour never selects membership. No gap is filled.
    """
    from collections import defaultdict
    from itertools import combinations
    from src.drawing_engine.disciplines.mep.mep_native_bend_connections import trace_bend_boundaries
    from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import TOLERANCE
    from src.drawing_engine.disciplines.mep.mep_route_observations import _normalise_style
    from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _style_compatible, _samples
    from src.drawing_engine.disciplines.mep.mep_route_body_partition import partition_corridor

    index = FrozenNativeQueries(query)
    if not query['search']['complete']:
        raise ValueError('complete native shape query required')
    rows = query['source_rows']
    by_paint, edges = defaultdict(list), {}
    for row in rows:
        native = row['source_native_segment']
        by_paint[native['drawing_ref']].append(row)
        if native['kind'] != 'line' or native['style'].get('fill') is not None:
            continue
        p, q = map(tuple, (row['points_display'][0], row['points_display'][-1]))
        style = _normalise_style(native['style'])
        key = (tuple(sorted((p, q))), style['width_display_points'], style['dash_pattern'])
        edges.setdefault(key, []).append(row['source_primitive_ref'])

    def midpoint(a, b):
        return [(a[k]+b[k])/2 for k in (0, 1)]

    def portal(a, b, other, style):
        center = midpoint(a, b)
        u = [b[k]-a[k] for k in (0, 1)]
        w = math.hypot(*u)
        n = [-u[1]/w, u[0]/w]
        if sum(n[k]*(other[k]-center[k]) for k in (0, 1)) < 0:
            n = [-v for v in n]
        return {'center': center, 'sides': [list(a), list(b)], 'width': w,
                'outward': n, 'style': style}

    # Only explicit, sequential authored closure; no implicit missing edge.
    polygons = {}
    for paint_rows in by_paint.values():
        chain = []
        for row in sorted(paint_rows, key=lambda r: (r['source_native_segment']['item_index'],
                                                     r['source_native_segment']['part_index'])):
            if (row['source_native_segment']['kind'] != 'line'
                    or row['source_native_segment']['style'].get('fill') is not None):
                chain = []
                continue
            if chain and chain[-1]['points_display'][-1] != row['points_display'][0]:
                chain = []
            chain.append(row)
            points = [r['points_display'][0] for r in chain]
            if row['points_display'][-1] != points[0]:
                continue
            if len(points) == 4 and len(set(map(tuple, points))) == 4:
                style = _normalise_style(chain[0]['source_native_segment']['style'])
                if all(_style_compatible(style, _normalise_style(r['source_native_segment']['style'])) for r in chain):
                    key = tuple(sorted(tuple(sorted((tuple(p), tuple(q))))
                                       for p, q in zip(points, points[1:]+points[:1])))
                    polygons.setdefault((key, style['width_display_points'], style['dash_pattern']),
                        {'points': points, 'style': style, 'refs': []})['refs'].extend(r['source_primitive_ref'] for r in chain)
            chain = []

    parts = []
    for (key, width, dash), polygon in sorted(polygons.items()):
        pts, style = polygon['points'], polygon['style']
        vectors = [[q[k]-p[k] for k in (0, 1)] for p, q in zip(pts, pts[1:]+pts[:1])]
        lengths = [math.hypot(*v) for v in vectors]
        if min(lengths) <= max(TOLERANCE, 3*width):
            continue
        cross = [vectors[i][0]*vectors[(i+1)%4][1]-vectors[i][1]*vectors[(i+1)%4][0] for i in range(4)]
        if not (all(v>0 for v in cross) or all(v<0 for v in cross)):
            continue
        parallel = [i for i in (0, 1) if abs(vectors[i][0]*vectors[i+2][1]-vectors[i][1]*vectors[i+2][0])
                    <= TOLERANCE*max(lengths)]
        choices = []
        for i in parallel:
            a, b = pts[i], pts[(i+1)%4]
            c, d = pts[(i+2)%4], pts[(i+3)%4]
            centers = [midpoint(a, b), midpoint(c, d)]
            axial = math.dist(*centers)
            skew = abs(sum((centers[1][k]-centers[0][k])*vectors[i][k]/lengths[i] for k in (0, 1)))
            if skew > TOLERANCE or axial <= max(lengths[i], lengths[i+2])*.6:
                continue
            kind = 'straight_panel' if len(parallel)==2 else 'tapered_transition'
            if kind == 'straight_panel' and axial < 1.25*max(lengths[i], lengths[i+2]):
                continue
            choices.append((i, kind, centers, axial))
        if len(choices) != 1:
            continue
        i, kind, centers, axial = choices[0]
        refs = sorted({r for edge in key for r in edges.get((edge, width, dash), [])})
        ports = [portal(pts[i], pts[(i+1)%4], centers[1], style),
                 portal(pts[(i+2)%4], pts[(i+3)%4], centers[0], style)]
        part = {'id': _stable_id('mep_duct_shape_observation', query['page_ref'], key, refs),
            'record_type': 'mep_duct_shape_observation', 'page_ref': query['page_ref'],
            'kind': kind, 'state': 'observed', 'boundary_points_display': pts+[pts[0]],
            'centreline_points_display': centers, 'projected_length_display_points': axial,
            'ports': ports, 'source_primitive_refs': refs,
            'system_identity_established': False, 'physical_connection_established': False,
            'quantity_eligible': False}
        if kind == 'straight_panel':
            candidate = {'id': part['id'], 'polyline_display': centers,
                         'corridor_width_display_points': ports[0]['width']}
            part['interior_partition'] = partition_corridor(candidate, query, set(refs), style)
        parts.append(part)

    # Curved line chains only nominate. The existing bend checker then tests
    # all compatible source ink, including axis-aligned branches and crossings.
    groups = defaultdict(dict)
    for (edge, width, dash), refs in edges.items():
        p, q = edge
        if all(abs(p[k]-q[k]) > TOLERANCE for k in (0, 1)):
            groups[(width, dash)][edge] = refs
    bends = []
    for (width, dash), curved in sorted(groups.items()):
        adjacency = defaultdict(set)
        full_adjacency = defaultdict(set)
        for (edge, ew, ed), refs in edges.items():
            if (ew, ed) == (width, dash):
                p, q = edge
                full_adjacency[p].add(q); full_adjacency[q].add(p)
        for p, q in curved:
            adjacency[p].add(q); adjacency[q].add(p)
        chains, seen = [], set()
        for start in sorted(adjacency):
            if len(adjacency[start]) != 1 or start in seen:
                continue
            path, current = [start], start
            while True:
                options = adjacency[current]-set(path)
                if len(options) != 1:
                    break
                current = next(iter(options)); path.append(current)
                if len(adjacency[current]) != 2:
                    break
            seen.update(path)
            if len(path) >= 5 and len(adjacency[current]) == 1:
                # Quantized circular outlines finish in short axis-aligned
                # native strokes. Extend only unique exact source incidence;
                # stop at the first cap/branch, never extrapolate a tangent.
                for reverse in (False, True):
                    if reverse:
                        path.reverse()
                    walked = 0.
                    while len(full_adjacency[path[-1]]) == 2:
                        options = full_adjacency[path[-1]]-set(path)
                        if len(options) != 1:
                            break
                        other = next(iter(options))
                        step = math.dist(path[-1], other)
                        if walked+step > 2*width:
                            break
                        path.append(other); walked += step
                    if reverse:
                        path.reverse()
                chains.append(path)
        style = {'width_display_points': width, 'dash_pattern': dash, 'fill': None}
        for a, b in combinations(chains, 2):
            if math.dist(a[0], b[0])+math.dist(a[-1], b[-1]) > math.dist(a[0], b[-1])+math.dist(a[-1], b[0]):
                b = list(reversed(b))
            # Nominate common transverse stations using observed vertices only.
            # Flange stubs may end one boundary beyond the other. Keep that ink
            # in the full replay query, but never invent its missing partner.
            ends = []
            for aa, bb in ((a,b), (list(reversed(a)),list(reversed(b)))):
                pairs = [(math.dist(aa[0],p)+math.dist(bb[0],q), p, q)
                    for p in aa if math.dist(aa[0],p)<=2*width
                    for q in bb if math.dist(bb[0],q)<=2*width
                    if min(abs(p[k]-q[k]) for k in (0,1))<=TOLERANCE]
                if not pairs:
                    break
                _, p, q = min(pairs)
                ends.append((p,q))
            if len(ends)!=2:
                continue
            a = [ends[0][0], ends[1][0]]
            b = [ends[0][1], ends[1][1]]
            c, d = midpoint(a[0], b[0]), midpoint(a[-1], b[-1])
            w0, w1 = math.dist(a[0], b[0]), math.dist(a[-1], b[-1])
            if min(w0,w1) <= 3*width or abs(w0-w1)>TOLERANCE:
                continue
            u, v = [a[0][k]-b[0][k] for k in (0,1)], [a[-1][k]-b[-1][k] for k in (0,1)]
            if abs(sum(x*y for x,y in zip(u,v))) > TOLERANCE*w0*w1:
                continue
            ports = [portal(a[0], b[0], d, style), portal(a[-1], b[-1], c, style)]
            paths, reasons = trace_bend_boundaries(ports, rows)
            if reasons:
                continue
            paired = [_samples(p['points_display'], 33) for p in paths]
            center = [midpoint(p,q) for p,q in zip(*paired)]
            refs = sorted({r for p in paths for r in p['source_primitive_refs']})
            bends.append({'id': _stable_id('mep_duct_shape_bend', query['page_ref'], refs),
                'record_type': 'mep_duct_shape_observation', 'page_ref': query['page_ref'],
                'kind': 'projected_bend', 'state': 'derived', 'boundary_paths': paths,
                'centreline_points_display': center, 'ports': ports, 'source_primitive_refs': refs,
                'projected_length_display_points': sum(math.dist(p,q) for p,q in zip(center,center[1:])),
                'system_identity_established': False, 'physical_connection_established': False,
                'quantity_eligible': False})
    parts += bends
    contacts = []
    for a, b in combinations(parts, 2):
        for i, pa in enumerate(a['ports']):
            for j, pb in enumerate(b['ports']):
                if not _style_compatible(pa['style'], pb['style']):
                    continue
                gap = math.dist(pa['center'], pb['center'])
                if gap > max(pa['style']['width_display_points'], pb['style']['width_display_points']):
                    continue
                if (abs(pa['width']-pb['width']) > TOLERANCE or
                    sum(x*y for x,y in zip(pa['outward'],pb['outward'])) > -.999):
                    continue
                residuals = [math.dist(x,y) for x,y in zip(sorted(pa['sides']),sorted(pb['sides']))]
                if max(residuals)-min(residuals) > TOLERANCE:
                    continue
                contacts.append({'part_refs': [a['id'], b['id']], 'port_indices': [i,j],
                    'gap_display_points': gap, 'side_residuals_display_points': residuals,
                    'state': 'projected_interface_candidate', 'physical_connection_established': False,
                    'gap_geometry_added': False})
    incidence = defaultdict(list)
    for contact in contacts:
        for ref, i in zip(contact['part_refs'],contact['port_indices']):
            incidence[(ref,i)].append(contact)
    for contact in contacts:
        contact['mutually_unique'] = all(len(incidence[(ref,i)])==1 for ref,i in zip(contact['part_refs'],contact['port_indices']))
    selected = {b['id'] for b in bends}
    while True:
        more = {ref for c in contacts if c['mutually_unique'] and selected.intersection(c['part_refs']) for ref in c['part_refs']}
        if more <= selected:
            break
        selected |= more
    return {'method': 'native_panel_and_replayed_bend_observations_v1',
        'page_ref': query['page_ref'], 'source_rows_sha256': query['search']['source_rows_sha256'],
        'parts': sorted(parts,key=lambda r:r['id']), 'contacts': contacts,
        'selected_part_refs': sorted(selected), 'selection_basis': 'projected_interface_candidates_connected_to_replayed_bend',
        'native_primitive_count': len(index.rows), 'system_identity_established': False,
        'network_complete': False, 'quantity_eligible': False}


def _inline(box, composite):
    points = composite['derived_geometry']['centreline_points_display']
    if len(points) != 2:
        return False
    a, b = points
    length = math.dist(a, b)
    if length <= 0:
        return False
    tangent = [(b[k] - a[k]) / length for k in (0, 1)]
    normal = [-tangent[1], tangent[0]]
    corners = [(box[i], box[j]) for i in (0, 2) for j in (1, 3)]
    along = [sum((p[k]-a[k])*tangent[k] for k in (0, 1)) for p in corners]
    across = [sum((p[k]-a[k])*normal[k] for k in (0, 1)) for p in corners]
    width = composite['derived_geometry']['corridor_width_display_points']
    return (min(along) > 0 and max(along) < length
            and max(abs(v) for v in across) < width / 2
            and max(along)-min(along) > 2*(max(across)-min(across)))


def inline_duct_binding_evidence(*, terminology, targets, native_queries):
    """Return measured inline ownership to M4, keeping competing targets open."""
    errors = (validate_mep_terminology_proposals(terminology)
              + validate_mep_outlined_route_composites(targets['composites']))
    if errors or terminology['document'] != targets['route_graph']['document']:
        raise ValueError('invalid or mismatched duct binding inputs: ' + '; '.join(errors))
    sources = {o['id']: o for o in terminology['source_observations']}
    by_anchor = {}
    for p in terminology['proposals']:
        by_anchor.setdefault(p['anchor_ref'], []).append(p)
    indexes = {ref: FrozenNativeQueries(q) for ref, q in native_queries.items()}
    evidence, outcomes = [], []
    for anchor, proposals in by_anchor.items():
        size_proposals = [p for p in proposals
                          if p['candidate']['kind'] == 'rectangular_duct_size']
        has_air_system = any(
            p['proposal_type'] == 'system' and p['candidate']['kind'] in
            {'supply_air', 'return_air', 'outside_air', 'exhaust_air'}
            for p in proposals)
        explicit_rectangular_text = any(
            re.search(r'(?<!\d)\d+(?:\.\d+)?\s*[xX×]\s*'
                      r'\d+(?:\.\d+)?(?!\d)',
                      str(p.get('candidate', {}).get('raw_text') or ''))
            for p in size_proposals)
        if not size_proposals or not (has_air_system or explicit_rectangular_text):
            continue
        observation = sources[anchor]
        page_ref = observation['page_ref']
        matches = [c for c in targets['composites']['accepted_composites']
                   if c['page_ref'] == page_ref and _inline(observation['bbox_display'], c)]
        reasons = []
        if len(matches) != 1:
            reasons.append('inline_callout_not_inside_one_accepted_corridor')
        if observation.get('region_role') not in {'unknown', 'drawing_inline'}:
            reasons.append('excluded_or_unresolved_native_text_role')
        if observation.get('native_overlap_alternatives') or observation.get('ocr_overlap_alternatives'):
            reasons.append('overlapping_text_alternatives')
        measured = {}
        if len(matches) == 1:
            c = matches[0]
            points = c['derived_geometry']['corridor_boundary_points_display']
            width = c['derived_geometry']['corridor_width_display_points']
            box = [min(p[0] for p in points)-width, min(p[1] for p in points)-width,
                   max(p[0] for p in points)+width, max(p[1] for p in points)+width]
            index = indexes.get(page_ref)
            rows, complete, refs = index.query(box) if index else ([], False, [])
            present = {r['source_primitive_ref'] for r in rows}
            if not complete or not set(c['member_source_primitive_refs']).issubset(present):
                reasons.append('complete_native_corridor_query_missing')
            competitors = [r for r in targets['envelope_searches'] if r['page_ref'] == page_ref
                and r.get('derived_geometry') and _inline(observation['bbox_display'], r)
                and set(r['member_source_primitive_refs']) != set(c['member_source_primitive_refs'])
                and r['state'] not in {'rejected_no_native_cap_path'}]
            if competitors:
                reasons.append('competing_native_corridor_at_inline_text')
            measured = {'query_bbox_display': box, 'query_complete': complete,
                'query_refs': refs, 'query_source_rows_sha256': _sha256(rows),
                'source_primitive_refs': sorted(present),
                'competing_envelope_refs': [r['id'] for r in competitors],
                'native_text_bbox_display': observation['bbox_display'],
                'native_source_query_sha256': _sha256(native_queries.get(page_ref)),
                'geometry_frozen_before_binding': True, 'color_used_for_identity': False}
        outcome = {'record_type': 'mep_duct_inline_ownership',
            'id': _stable_id('mep_duct_inline_ownership', anchor), 'page_ref': page_ref,
            'source_observation_ref': anchor, 'text': observation['text'],
            'state': 'accepted' if not reasons else 'abstained', 'reasons': reasons,
            'target_refs': [c['id'] for c in matches], 'measured_evidence': measured,
            'quantity_eligible': False}
        outcomes.append(outcome)
        if reasons:
            continue
        for proposal in proposals:
            if proposal['proposal_type'] not in {'system', 'inline_size', 'elevation'}:
                continue
            evidence.append({'id': _stable_id('mep_duct_inline_binding', proposal['id'], c['id']),
                'page_ref': page_ref, 'proposal_ref': proposal['id'], 'state': 'observed',
                'method': {'name': 'unique_native_corridor_inline_air_callout', 'version': '1.0.0'},
                'target_kind': 'route_composite', 'target_refs': [c['id']],
                'geometric_evidence_refs': [c['id'], *c['member_source_primitive_refs']],
                'explicit_branch_coverage': False, 'duct_inline_certificate': deepcopy(outcome)})
    return evidence, outcomes


def _dimension_note_basis(observation):
    if observation.get('region_role') not in {'unknown', 'note'}:
        return None
    text = ' '.join(observation['text'].split())
    if re.search(r'DUCT SIZES.*FREE AREA', text, re.I):
        return 'free_area'
    if (observation.get('region_role') != 'note'
            or observation.get('region_role_evidence', {}).get('region_body_complete') is not True
            or re.search(r'\b(?:NOT|EXCEPT)\b', text, re.I)):
        return None
    outside = (
        r'(?:ALL )?DUCT(?:WORK)? (?:SIZES(?: SHOWN(?: ON (?:THE )?PLANS?)?)?'
        r'|DIMENSIONS(?: INDICATED)?) (?:ARE |REFER TO )'
        r'(?:EXTERNAL|OUTSIDE|OVERALL|SHEET METAL) DIMENSIONS?'
    )
    inside = (
        r'(?:ALL )?DUCT(?:WORK)? SIZES(?: SHOWN(?: ON (?:THE )?PLANS?)?)? ARE '
        r'(?:INSIDE|INTERNAL|NET INTERNAL)(?: AIR FLOW)?(?: DIMENSIONS?)?'
    )
    if re.match(outside, text, re.I):
        return 'exterior'
    if re.match(inside, text, re.I):
        return 'interior'
    return None


def _native_text_refs(observation):
    if 'native_pdf_text' not in observation.get('evidence_channels', []):
        return []
    return sorted({str(span.get('source_native_ref'))
                   for span in observation.get('native_spans', [])
                   if span.get('source_native_ref')}
                  | ({str(observation['source_native_ref'])}
                     if observation.get('source_native_ref') else set()))


def _dimension_note_scope(observation):
    """Return the textual applicability scope of an explicit duct convention."""
    text = ' '.join(observation['text'].split())
    if re.search(r'\bSHOWN ON (?:THE )?PLAN\b', text, re.I):
        return 'page'
    return 'document'


def _duct_dimension_unit(observation):
    """Return an explicit native rectangular-duct unit convention."""
    text = ' '.join(observation.get('text', '').split())
    if (observation.get('region_role') not in {'legend', 'note'}
            or not _native_text_refs(observation)):
        return None
    if re.search(r'\bRECT(?:ANGULAR|\.)?\s+DUCT\s+SIZE\s*\(INCHES\)', text, re.I):
        return 'in'
    if re.search(r'\bRECT(?:ANGULAR|\.)?\s+DUCT\s+SIZE\s*\(MM\)', text, re.I):
        return 'mm'
    return None


def build_duct_size_applicability_certificates(*, terminology, bindings):
    """Certify one native W x H callout on one native duct interval.

    M4 already owns leader/inline geometry and target uniqueness.  This adapter
    replays that authority, traces the proposal back to native text, and keeps
    the outside-dimension convention as a separate evidence gate.  A document
    default must say that all ductwork dimensions use that basis; a note that
    names ``the plan`` remains page-local.  Conflicting, inside, free-area, or
    missing conventions abstain.  The certificate establishes only page-local
    size applicability, never a 3D envelope, continuation, or quantity.
    """
    contract = bindings.get('m2_contract_ref', {})
    if (contract and contract.get('payload_sha256') != _sha256(terminology)):
        raise ValueError('duct size applicability requires matching frozen M2/M4')
    if (bindings.get('document') is not None
            and terminology.get('document') != bindings.get('document')):
        raise ValueError('duct size applicability document mismatch')

    proposals = {str(row['id']): row for row in terminology.get('proposals', [])}
    observations = {str(row['id']): row
                    for row in terminology.get('source_observations', [])}
    evidence = {str(row['id']): row for row in bindings.get('binding_evidence', [])}
    composites = {str(row['id']): row
                  for row in bindings.get('outlined_route_composites', [])}

    def original_observation(proposal):
        ref = str(proposal.get('anchor_ref'))
        seen = set()
        while ref in observations and observations[ref].get('source_observation_ref'):
            if ref in seen:
                raise ValueError('cyclic duct size observation lineage')
            seen.add(ref)
            ref = str(observations[ref]['source_observation_ref'])
        return observations.get(ref)

    notes = []
    unit_conventions = []
    for observation in observations.values():
        basis = _dimension_note_basis(observation)
        native_refs = _native_text_refs(observation)
        if basis and native_refs:
            notes.append((observation, basis, _dimension_note_scope(observation),
                          native_refs))
        unit = _duct_dimension_unit(observation)
        if unit:
            unit_conventions.append((observation, unit))

    relations = [row for row in bindings.get('relations', [])
                 if row.get('relation_type') == 'route_size'
                 and row.get('candidate', {}).get('kind') == 'rectangular_duct_size']
    accepted_values_by_target = {}
    for relation in relations:
        if relation.get('state') != 'accepted' or len(relation.get('target_refs', [])) != 1:
            continue
        candidate = relation['candidate']
        key = (candidate.get('width'), candidate.get('height'), candidate.get('unit'))
        accepted_values_by_target.setdefault(str(relation['target_refs'][0]), set()).add(key)

    rows = []
    for relation in relations:
        proposal = proposals.get(str(relation.get('proposal_ref')))
        observation = original_observation(proposal or {})
        page_ref = str(relation.get('page_ref') or '')
        target_refs = [str(ref) for ref in relation.get('target_refs', [])]
        target_ref = target_refs[0] if len(target_refs) == 1 else None
        composite = composites.get(target_ref) if target_ref else None
        reasons = []

        candidate = relation.get('candidate', {})
        dimensions = (candidate.get('width'), candidate.get('height'))
        native_size_refs = _native_text_refs(observation or {})
        native_size_text = bool(
            observation and proposal
            and proposal.get('proposal_type') == 'inline_size'
            and candidate.get('kind') == 'rectangular_duct_size'
            and all(type(value) in {int, float} and math.isfinite(float(value))
                    and value > 0 for value in dimensions)
            and re.search(r'(?<!\d)\d+(?:\.\d+)?\s*(?:[xX\u00d7/])\s*'
                          r'\d+(?:\.\d+)?(?!\d)', observation.get('text', ''))
            and bool(native_size_refs))
        if not native_size_text:
            reasons.append('native_width_height_text_not_established')
        units = {row[1] for row in unit_conventions}
        explicit_unit = candidate.get('unit') if candidate.get('unit') in {'in', 'mm'} else None
        resolved_unit = explicit_unit or next(iter(units)) if len(units) == 1 else explicit_unit
        unit_closed = resolved_unit in {'in', 'mm'} and not (
            explicit_unit and units and units != {explicit_unit})
        if not unit_closed:
            reasons.append('rectangular_duct_dimension_unit_unresolved')

        association_rows = [evidence[ref] for ref in relation.get('binding_evidence_refs', [])
                            if str(ref) in evidence]
        association_kinds = set()
        direct_association_refs = []
        for item in association_rows:
            method = item.get('method', {}).get('name')
            common = (item.get('state') == 'observed'
                      and item.get('target_kind') == 'route_composite'
                      and [str(ref) for ref in item.get('target_refs', [])] == target_refs
                      and target_ref in {str(ref) for ref in
                                         item.get('geometric_evidence_refs', [])})
            if method == 'complete_native_dot_leader_contact':
                search = item.get('automatic_search_certificate', {})
                direct = (common and observation is not None
                          and str(search.get('leader_observation_ref')) == str(observation['id'])
                          and bool(search.get('source_searches_sha256'))
                          and search.get('reviewed_selectors_used') is False)
                kind = 'leader'
            elif method == 'unique_native_corridor_inline_air_callout':
                inline = item.get('duct_inline_certificate', {})
                measured = inline.get('measured_evidence', {})
                direct = (common and observation is not None
                          and inline.get('state') == 'accepted'
                          and str(inline.get('source_observation_ref')) == str(observation['id'])
                          and measured.get('query_complete') is True
                          and measured.get('geometry_frozen_before_binding') is True
                          and measured.get('color_used_for_identity') is False)
                kind = 'inline'
            else:
                direct, kind = False, None
            if direct:
                direct_association_refs.append(str(item['id']))
                association_kinds.add(kind)
        direct_association = bool(direct_association_refs) and len(association_kinds) == 1
        if not direct_association:
            reasons.append('unique_native_leader_or_inline_association_not_established')

        certificates = relation.get('certificates', {})
        geometry = composite.get('derived_geometry', {}) if composite else {}
        native_interval = bool(
            relation.get('state') == 'accepted'
            and relation.get('target_kind') == 'route_composite'
            and target_ref and composite
            and composite.get('state') == 'accepted'
            and str(composite.get('page_ref')) == page_ref
            and len(geometry.get('centreline_points_display', [])) == 2
            and bool(composite.get('member_source_primitive_refs'))
            and certificates
            and all(value is True for value in certificates.values())
            and len(accepted_values_by_target.get(target_ref, set())) == 1)
        if not native_interval:
            reasons.append('unique_native_duct_interval_not_established')

        applicable_notes = [row for row in notes
                            if row[2] == 'document'
                            or str(row[0].get('page_ref')) == page_ref]
        bases = {row[1] for row in applicable_notes}
        outside = bases == {'exterior'}
        if not outside:
            if len(bases) > 1:
                reasons.append('conflicting_duct_dimension_conventions')
            elif bases == {'interior'}:
                reasons.append('inside_dimensions_do_not_establish_outside_dimensions')
            elif bases == {'free_area'}:
                reasons.append('free_area_dimensions_do_not_establish_outside_dimensions')
            else:
                reasons.append('outside_dimension_convention_not_established')

        state = 'accepted' if not reasons else 'abstained'
        rows.append({
            'record_type': 'mep_duct_size_applicability_certificate',
            'id': _stable_id('mep_duct_size_applicability_certificate',
                             relation.get('id'), observation.get('id') if observation else None,
                             target_refs, sorted(row[0]['id']
                                                 for row in applicable_notes)),
            'page_ref': page_ref,
            'source_pdf_sha256': (terminology.get('document') or {}).get(
                'source_pdf_sha256'),
            'source_observation_ref': observation.get('id') if observation else None,
            'proposal_ref': proposal.get('id') if proposal else None,
            'route_size_relation_ref': relation.get('id'),
            'route_composite_ref': target_ref,
            'candidate': deepcopy(candidate),
            'association_kind': next(iter(association_kinds))
                if direct_association else None,
            'dimension_basis': 'outside' if outside else
                next(iter(bases)) if len(bases) == 1 else
                'conflicting' if bases else 'unknown',
            'dimension_unit': resolved_unit if unit_closed else None,
            'state': state,
            'reasons': sorted(set(reasons)),
            'certificates': {
                'native_width_height_text': native_size_text,
                'direct_leader_or_inline_association': direct_association,
                'unique_native_duct_interval': native_interval,
                'outside_dimension_convention': outside,
                'rectangular_duct_dimension_unit': unit_closed,
            },
            'evidence': {
                'native_size_text_refs': native_size_refs,
                'binding_evidence_refs': sorted(direct_association_refs),
                'interval_source_primitive_refs': sorted(
                    str(ref) for ref in
                    (composite or {}).get('member_source_primitive_refs', [])),
                'dimension_convention_refs': sorted(
                    str(row[0]['id']) for row in applicable_notes),
                'dimension_convention_native_refs': sorted({
                    ref for row in applicable_notes for ref in row[3]}),
                'dimension_unit_refs': sorted(
                    str(row[0]['id']) for row in unit_conventions),
            },
            'authority': {
                'outside_size_applicability_established': state == 'accepted',
                'physical_3d_envelope_established': False,
                'physical_continuation_established': False,
                'installed_length_established': False,
            },
            'quantity_eligible': False,
        })
    return rows


def duct_section_evidence(*, registry, terminology, bindings):
    """Resolve dimensional alternatives against measured plan width, not tables.

Unknown unit/axis conventions are enumerated, not silently assumed. A free-area
note blocks an exterior envelope even when the plotted width agrees exactly.
"""
    pages = {p['page_ref']: p for p in registry['pages']}
    notes = [o for o in terminology['source_observations'] if _dimension_note_basis(o)]
    rows = []
    for c in bindings['outlined_route_composites']:
        relations = [r for r in bindings['relations'] if r['state'] == 'accepted'
                     and c['id'] in r['target_refs']]
        sizes = [r for r in relations if r['relation_type'] == 'route_size'
                 and r['candidate'].get('kind') == 'rectangular_duct_size']
        if not sizes:
            continue
        page = pages[c['page_ref']]
        scale = page['fields']['scale']
        factor = scale.get('drawing_inches_per_paper_inch')
        if scale['state'] not in {'observed', 'derived'} or not factor:
            factor = None
        factor = factor * .0254 / 72 if factor else None
        width = c['derived_geometry']['corridor_width_display_points']
        precision = c['geometry_metrics']['member_width_display_points']
        alternatives = []
        for relation in sizes:
            size = relation['candidate']
            for unit in ([size['unit']] if size.get('unit') else ['mm', 'in']):
                multiplier = {'mm': .001, 'in': .0254}.get(unit)
                if multiplier is None or factor is None:
                    continue
                for w, h in set(((size['width'], size['height']), (size['height'], size['width']))):
                    residual = abs(w*multiplier - width*factor)
                    if w > 0 and h > 0 and residual <= precision*factor:
                        alternatives.append({'unit': unit, 'plan_width_m': w*multiplier,
                            'vertical_height_m': h*multiplier, 'width_residual_m': residual,
                            'size_relation_ref': relation['id']})
        # Repeated annotations do not manufacture distinct physical alternatives.
        unique = {(a['unit'], a['plan_width_m'], a['vertical_height_m']): a for a in alternatives}
        relevant_notes = [o for o in notes if o['page_ref'] == c['page_ref']]
        bases = {_dimension_note_basis(o) for o in relevant_notes}
        free_area = 'free_area' in bases
        exterior = bases == {'exterior'}
        reasons = []
        if len(unique) != 1:
            reasons.append('duct_unit_or_cross_section_orientation_unresolved')
        if any(r['candidate'].get('unit') is None for r in sizes):
            reasons.append('duct_unit_convention_not_explicitly_established')
        if free_area:
            reasons.append('free_area_size_does_not_determine_exterior_height_or_bottom_offset')
        elif not exterior:
            # Absence of a note is not an exterior-size convention certificate.
            reasons.append('duct_dimension_basis_not_explicitly_established')
        rows.append({'record_type': 'mep_duct_cross_section_evidence',
            'id': _stable_id('mep_duct_cross_section', c['id'], relations, relevant_notes),
            'route_composite_ref': c['id'], 'page_ref': c['page_ref'],
            'state': 'accepted' if not reasons else 'abstained',
            'dimension_basis': 'free_area' if free_area else 'exterior' if exterior else 'unknown',
            'orientation_and_unit_alternatives': list(unique.values()),
            'plan_cross_section_resolved': len(unique) == 1,
            'measured_plan_width_m': width*factor if factor else None,
            'measurement_tolerance_m': precision*factor if factor else None,
            'scale_evidence_refs': scale.get('evidence_refs', []),
            'source_primitive_refs': c['member_source_primitive_refs'],
            'dimension_basis_evidence_refs': [o['id'] for o in relevant_notes],
            'm4_relation_refs': [r['id'] for r in relations], 'reasons': reasons,
            'physical_outer_envelope_established': not reasons, 'quantity_eligible': False})
    return rows


def build_partial_duct_3d_gate(*, shapes, schedule, unresolved_interfaces):
    """Freeze projected duct evidence that is insufficient for metric 3D.

    Plan-scaled portal spacing remains an observation of plotted geometry.  It
    cannot become an annotated rectangular size, supply a missing height, or
    borrow a nearby pipe elevation.  This gate is part of the existing duct
    evidence flow and deliberately emits no surface or quantity.
    """
    selected = {p['id']: p for p in shapes['parts']
                if p['id'] in shapes['selected_part_refs']}
    if {row['shape_ref'] for row in schedule} != set(selected):
        raise ValueError('partial duct gate requires the exact selected schedule')
    values = {}
    for row in schedule:
        for value in row.get('plan_widths_m') or []:
            key = round(value, 8)
            values.setdefault(key, {'shape_refs': set(), 'schedule_refs': set()})
            values[key]['shape_refs'].add(row['shape_ref'])
            values[key]['schedule_refs'].add(row['id'])
    observed = []
    for index, value in enumerate(sorted(values)):
        refs = values[value]
        observed.append({
            'record_type': 'mep_duct_plotted_spacing_observation',
            'id': _stable_id('mep_duct_plotted_spacing', value,
                             sorted(refs['shape_refs'])),
            'kind': ('plotted_envelope_spacing' if index == 0
                     else 'taper_wide_portal_spacing'),
            'state': 'observed',
            'value_m': value,
            'value_mm': round(value * 1000, 1),
            'shape_refs': sorted(refs['shape_refs']),
            'schedule_refs': sorted(refs['schedule_refs']),
            'establishes_annotated_duct_dimension': False,
            'establishes_vertical_height': False,
            'quantity_eligible': False,
        })
    selected_contacts = [c for c in shapes['contacts']
                         if set(c['part_refs']).issubset(selected)]
    topology = {
        'straight_panel_count': sum(p['kind'] == 'straight_panel'
                                    for p in selected.values()),
        'projected_bend_count': sum(p['kind'] == 'projected_bend'
                                    for p in selected.values()),
        'tapered_transition_count': sum(p['kind'] == 'tapered_transition'
                                        for p in selected.values()),
    }
    return {
        'record_type': 'mep_duct_partial_3d_gate',
        'id': _stable_id('mep_duct_partial_3d_gate',
                         shapes['source_rows_sha256'], sorted(selected)),
        'state': 'partial_geometry_only',
        'positive_metric_3d_eligible': False,
        'certified': {
            'projected_panel_topology': topology,
            'shape_refs': sorted(selected),
            'projected_portal_alignment_refs': sorted(
                _stable_id('mep_duct_projected_portal_alignment',
                           c['part_refs'], c['port_indices'],
                           c['gap_display_points'])
                for c in selected_contacts if c['mutually_unique']),
            'native_source_rows_sha256': shapes['source_rows_sha256'],
            'native_provenance_preserved': True,
            'analysis_boundary_count': len(unresolved_interfaces),
            'analysis_boundaries_are_physical_caps': False,
        },
        'observed_only': observed,
        'unknown': [
            'system_identity', 'annotated_width', 'annotated_height',
            'duct_local_elevation', 'connection_details', 'installed_length',
        ],
        'prohibited_inferences': [
            'borrow_nearby_pipe_system_or_size',
            'borrow_nearby_pipe_elevation',
            'promote_plotted_spacing_to_annotated_duct_size',
            'assume_standard_duct_height',
            'treat_analysis_boundaries_as_physical_caps',
            'emit_installed_length_or_quantity',
        ],
        'metric_reconstruction': {
            'state': 'abstained',
            'surface_generated': False,
            'watertightness_claimed': False,
            'reasons': [
                'explicit_duct_system_unbound',
                'annotated_width_and_height_unbound',
                'duct_local_elevation_unbound',
            ],
        },
        'schematic_preview_policy': {
            'permitted': True,
            'height_basis': 'arbitrary_display_only',
            'required_label': 'NOT DIMENSIONALLY RESOLVED',
            'may_enter_quantities_or_marketplace_takeoff': False,
        },
        'quantity_eligible': False,
    }


def build_duct_3d_authority_levels(*, qualification, assembly=None,
                                   absolute_placement=None):
    """Separate relative duct geometry from building-coordinate authority."""
    checks = [
        ('unique_plan_to_section_identity', 'plan_to_section_identity', 'mutually_unique'),
        ('drawing_derived_width_and_height', 'cross_section', 'drawing_derived'),
        ('complete_centreline_and_fitting_topology', 'route_topology',
         'complete_within_analysis_boundaries'),
        ('relative_vertical_offsets', 'relative_z',
         'drawing_derived_from_scaled_section'),
        ('plan_and_section_metric_scale', 'two_view_scale',
         'plan_and_section_metric'),
    ]
    gates = []
    for label, key, flag in checks:
        row = qualification.get(key, {})
        passed = row.get('state') == 'accepted' and row.get(flag) is True
        gates.append({'gate': label, 'state': 'accepted' if passed else 'unresolved',
            'evidence_refs': sorted(str(ref) for ref in row.get('evidence_refs', [])),
            'reasons': [] if passed else list(row.get('reasons', [key + '_unresolved']))})
    assembly = assembly or {}
    assembly_passed = (assembly.get('state') == 'accepted'
        and assembly.get('relative_3d_established') is True
        and assembly.get('verification_envelope_watertight') is True
        and assembly.get('internal_interfaces_gap_free') is True
        and {row.get('view') for row in assembly.get('reprojections', [])
             if row.get('state') == 'accepted'} >= {'plan', 'section'})
    gates.append({'gate': 'watertight_assembly_and_two_view_reprojection',
        'state': 'accepted' if assembly_passed else 'unresolved',
        'evidence_refs': [assembly['id']] if assembly.get('id') else [],
        'reasons': [] if assembly_passed else list(
            assembly.get('reasons', ['relative_duct_assembly_unresolved']))})
    relative_passed = all(row['state'] == 'accepted' for row in gates)
    placement = absolute_placement or {}
    portal_by_ref = {str(row.get('id')): row
                     for row in qualification.get('portals', [])}
    reference_portal_ref = str(placement.get('reference_portal_ref'))
    reference_portal = portal_by_ref.get(reference_portal_ref, {})
    absolute_placement_closed = (placement.get('state') == 'accepted'
        and placement.get('basis') in {'BOD', 'TOD', 'CL'}
        and isinstance(placement.get('floor_datum_ref'), str)
        and bool(placement.get('floor_datum_ref'))
        and reference_portal_ref in set(assembly.get('ordered_portal_refs', []))
        and isinstance(reference_portal.get('centre_xyz_m'), list)
        and len(reference_portal.get('centre_xyz_m')) == 3
        and type(reference_portal.get('height_m')) in {int, float}
        and math.isfinite(float(reference_portal['height_m']))
        and float(reference_portal['height_m']) > 0
        and isinstance(placement.get('value_m'), (int, float))
        and math.isfinite(float(placement['value_m']))
        and isinstance(placement.get('floor_datum_elevation_m'), (int, float))
        and math.isfinite(float(placement['floor_datum_elevation_m'])))
    placement_passed = relative_passed and absolute_placement_closed
    z_translation = None
    if absolute_placement_closed:
        target_centre_z = (float(placement['floor_datum_elevation_m'])
                           + float(placement['value_m']))
        height = float(reference_portal['height_m'])
        if placement['basis'] == 'BOD':
            target_centre_z += height / 2
        elif placement['basis'] == 'TOD':
            target_centre_z -= height / 2
        z_translation = target_centre_z - float(reference_portal['centre_xyz_m'][2])
    relative_reasons = sorted({reason for row in gates if row['state'] != 'accepted'
                               for reason in row['reasons']})
    return {'record_type': 'mep_duct_3d_authority_levels',
        'id': _stable_id('mep_duct_3d_authority_levels', qualification.get('id'),
                         assembly.get('id'), placement.get('id')),
        'relative_3d': {'state': 'accepted' if relative_passed else 'abstained',
            'gates': gates, 'reasons': relative_reasons,
            'absolute_building_datum_required': False,
            'resolved_relative_centerline_length_m': assembly.get(
                'resolved_relative_centerline_length_m')
                if relative_passed else None,
            'resolved_relative_centerline_length_closed': relative_passed
                and assembly.get('resolved_relative_centerline_length_m') is not None},
        'building_placed_3d': {'state': 'accepted' if placement_passed else 'abstained',
            'absolute_elevation_unresolved': not absolute_placement_closed,
            'placement_evidence_refs': sorted(str(ref) for ref in placement.get('evidence_refs', [])),
            'floor_datum_ref': placement.get('floor_datum_ref') if placement_passed else None,
            'reference_portal_ref': reference_portal_ref if placement_passed else None,
            'relative_to_building_z_translation_m': round(z_translation, 8)
                if placement_passed else None,
            'cross_system_placement_eligible': placement_passed,
            'bounded_clash_verification_eligible': placement_passed,
            'reasons': [] if placement_passed else
                (([] if relative_passed else ['relative_3d_unresolved'])
                 + ([] if absolute_placement_closed else ['absolute_elevation_unresolved']))},
        'confirmed_clash_established': False,
        'quantity_eligible': False}


def validate_duct_3d_authority_levels(payload):
    """Return invariant errors for the two duct 3D authority levels."""
    errors = []
    relative = payload.get('relative_3d', {})
    building = payload.get('building_placed_3d', {})
    all_relative_gates = (bool(relative.get('gates')) and all(
        row.get('state') == 'accepted' for row in relative.get('gates', [])))
    if (relative.get('state') == 'accepted') != all_relative_gates:
        errors.append('relative state does not match its evidence gates')
    if (relative.get('resolved_relative_centerline_length_m') is not None
            or relative.get('resolved_relative_centerline_length_closed') is True) and not all_relative_gates:
        errors.append('relative centreline length escaped an unresolved relative gate')
    building_accepted = building.get('state') == 'accepted'
    if building_accepted and not all_relative_gates:
        errors.append('building placement escaped an unresolved relative gate')
    if building_accepted and building.get('absolute_elevation_unresolved') is not False:
        errors.append('building placement lacks closed absolute elevation')
    if building_accepted and not isinstance(
            building.get('relative_to_building_z_translation_m'), (int, float)):
        errors.append('building placement lacks a numeric frame transform')
    if ((building.get('cross_system_placement_eligible') is True
         or building.get('bounded_clash_verification_eligible') is True)
            and not building_accepted):
        errors.append('coordination authority escaped building placement')
    if payload.get('confirmed_clash_established') is not False:
        errors.append('authority levels cannot confirm a clash')
    if payload.get('quantity_eligible') is not False:
        errors.append('duct 3D authority levels cannot grant quantity authority')
    return errors
