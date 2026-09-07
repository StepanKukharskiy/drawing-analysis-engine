"""Versioned 2D equipment interpretation, independent of ports and 3D.

The observed geometry is a cabinet motif, not a physical unit. A repeated
same-class drawing convention plus mutually unique native typographic ownership
can accept an *inferred projected body/tag identity*. No port, physical instance,
quantity or old M4 equipment-endpoint authority follows from that certificate.
"""

from collections import defaultdict
from copy import deepcopy

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_hvac_inventory import EQUIPMENT_CLASSES
from src.drawing_engine.disciplines.mep.mep_native_equipment_observations import rectangular_outline_candidates, boundary_contact_candidates
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


VERSION = '0.1.0'
POLICY = {'motif': 'three_panel_inset_cabinet', 'major_minor_aspect_range': [3., 8.],
    'panel_minor_coverage_range': [.7, .98], 'inset_minor_coverage_range': [.3, .7],
    'minimum_central_to_side_width_ratio': 2., 'maximum_frame_margin_fraction': .12,
    'maximum_panel_asymmetry_fraction': .04, 'minimum_label_span_overlap': .5,
    'maximum_label_gap_in_text_heights': 3., 'maximum_body_span_in_text_heights': 4.,
    'maximum_normalized_motif_residual': .025,
    'minimum_distinct_calibration_tags': 2, 'minimum_calibration_pages': 2,
    'bold_native_typography_required': True}


def _inside(a, b):
    return a[0] < b[0] and a[1] < b[1] and b[2] < a[2] and b[3] < a[3]


def _contains(a, b):
    return a != b and a[0] <= b[0] and a[1] <= b[1] and b[2] <= a[2] and b[3] <= a[3]


def _axes(box, major):
    return [box[major], box[1-major], box[major+2], box[3-major]]


def discover_cabinet_motifs(source_rows):
    """Find geometry before consulting tags, with every competing rectangle kept."""
    widths = [r['source_native_segment']['style'].get('width') for r in source_rows]
    positive = [float(w) for w in widths if isinstance(w, (int, float)) and w > 0]
    minimum_side = min(positive, default=.25) * 4
    rectangles = rectangular_outline_candidates(source_rows, minimum_side)
    motifs = []
    for body in rectangles:
        box = body['bbox_display']
        major = 0 if box[2]-box[0] >= box[3]-box[1] else 1
        b = _axes(box, major)
        width, height = b[2]-b[0], b[3]-b[1]
        if not POLICY['major_minor_aspect_range'][0] <= width / height <= POLICY['major_minor_aspect_range'][1]:
            continue
        children = [r for r in rectangles if r['non_color_style'] == body['non_color_style']
                    and _inside(box, r['bbox_display'])]
        panels = [r for r in children if not any(_contains(other['bbox_display'], r['bbox_display'])
                                                for other in children)]
        panels = [r for r in panels if POLICY['panel_minor_coverage_range'][0] <=
                  (_axes(r['bbox_display'], major)[3]-_axes(r['bbox_display'], major)[1]) / height
                  <= POLICY['panel_minor_coverage_range'][1]]
        if len(panels) != 3:
            continue
        panels.sort(key=lambda r: _axes(r['bbox_display'], major)[0])
        p = [_axes(r['bbox_display'], major) for r in panels]
        if any(a[2] >= c[0] for a, c in zip(p, p[1:])):
            continue
        tolerance = POLICY['maximum_panel_asymmetry_fraction'] * height
        if max(v[1] for v in p)-min(v[1] for v in p) > tolerance or max(v[3] for v in p)-min(v[3] for v in p) > tolerance:
            continue
        panel_widths = [v[2]-v[0] for v in p]
        if (abs(panel_widths[0]-panel_widths[2]) > POLICY['maximum_panel_asymmetry_fraction'] * width
                or panel_widths[1] < POLICY['minimum_central_to_side_width_ratio'] * max(panel_widths[0], panel_widths[2])
                or max(p[0][0]-b[0], b[2]-p[-1][2]) > POLICY['maximum_frame_margin_fraction'] * width):
            continue
        insets = []
        for panel, geometry in zip(panels, p):
            nested = [r for r in children if _inside(panel['bbox_display'], r['bbox_display'])
                      and POLICY['inset_minor_coverage_range'][0] <=
                      (_axes(r['bbox_display'], major)[3]-_axes(r['bbox_display'], major)[1]) / (geometry[3]-geometry[1])
                      <= POLICY['inset_minor_coverage_range'][1]]
            nested = [r for r in nested if not any(_contains(other['bbox_display'], r['bbox_display']) for other in nested)]
            if len(nested) != 1:
                break
            insets.append(nested[0])
        if len(insets) != 3:
            continue
        inset_boxes = [_axes(r['bbox_display'], major) for r in insets]
        if max(v[1] for v in inset_boxes)-min(v[1] for v in inset_boxes) > tolerance or max(v[3] for v in inset_boxes)-min(v[3] for v in inset_boxes) > tolerance:
            continue
        components = [body, *panels, *insets]
        refs = sorted({ref for r in components for ref in r['source_primitive_refs']})
        normalized = [(value-b[i % 2]) / (width if i % 2 == 0 else height)
                      for r in [*panels, *insets] for i, value in enumerate(_axes(r['bbox_display'], major))]
        motifs.append({'id': _stable_id('mep_equipment_cabinet_motif', source_rows[0]['page_ref'], refs),
            'record_type': 'mep_equipment_cabinet_motif', 'record_version': VERSION,
            'page_ref': source_rows[0]['page_ref'], 'bbox_display': box, 'major_axis': major,
            'motif_kind': POLICY['motif'], 'components': deepcopy(components),
            'normalized_signature': [width/height, *normalized], 'source_primitive_refs': refs,
            'state': 'observed', 'epistemic_state': 'derived', 'quantity_eligible': False})
    return motifs, rectangles


def _label_zone(motif, observation):
    # The tag may be placed above or below the body, but its printed span must
    # substantially overlap the cabinet span. This is only one evidence gate.
    b, t = _axes(motif['bbox_display'], motif['major_axis']), _axes(observation['bbox_display'], motif['major_axis'])
    height = max(1., min(observation['bbox_display'][2]-observation['bbox_display'][0],
                         observation['bbox_display'][3]-observation['bbox_display'][1]))
    if b[2]-b[0] > POLICY['maximum_body_span_in_text_heights'] * height:
        return None
    overlap = max(0., min(b[2], t[2])-max(b[0], t[0])) / min(b[2]-b[0], t[2]-t[0])
    if t[3] <= b[1]:
        side, gap = 'before_minor_axis', b[1]-t[3]
    elif b[3] <= t[1]:
        side, gap = 'after_minor_axis', t[1]-b[3]
    else:
        return None
    if overlap < POLICY['minimum_label_span_overlap'] or gap > POLICY['maximum_label_gap_in_text_heights'] * height:
        return None
    return {'side': side, 'printed_span_overlap_fraction': overlap,
            'gap_in_text_heights': gap/height, 'quantity_eligible': False}


def _typography_valid(record, observation):
    return bool(record and record.get('source_observation_ref') == observation['id']
        and record.get('page_ref') == observation['page_ref']
        and record.get('bbox_display') == observation['bbox_display']
        and record.get('native_text') == observation['text']
        and record.get('bold') is True and record.get('horizontal') is True
        and record.get('source_pdf_sha256') == observation['source_pdf_sha256'])


def _lexical_tag_eligible(proposal, observation):
    # The new symbol/typography certificate may resolve only the old unknown
    # region context. It never overrides a legend, competing interpretation,
    # unsupported class or semantic conflict, and never rewrites frozen M2.
    context_only = (proposal.get('state') == 'abstained'
                    and set(proposal.get('reasons', [])) == {'region_applicability_unresolved'})
    return ((proposal.get('state') == 'proposed' or context_only)
            and not proposal.get('conflicts') and not proposal.get('alternatives')
            and not proposal.get('legend_membership')
            and observation.get('region_role') not in {'legend', 'table', 'schedule', 'note', 'title', 'cut_sheet'}
            and proposal['candidate'].get('equipment_class_token') in EQUIPMENT_CLASSES
            and observation['text'].strip() == proposal['candidate'].get('tag'))


def build_equipment_2d_identity(*, native_inputs, typography, calibration=None):
    """Infer unique projected identities under an independently frozen convention.

    When calibration is absent, derive it from all eligible input scopes. A
    held-out invocation receives that exact frozen calibration and cannot add
    its own labels to satisfy the repeated-example gate.
    """
    sources, items, motifs_by_page = {}, [], defaultdict(dict)
    for page in native_inputs['pages']:
        rows = {r['id']: r for r in page['source_rows']}
        for item in page['items']:
            inp = item['inputs']; query = page['queries'][item['query_key']]
            local = [rows[ref] for ref in query['source_row_refs']]
            if _sha256(local) != query['source_rows_sha256'] or inp['observation']['source_pdf_sha256'] != native_inputs['source_pdf_sha256']:
                raise ValueError('equipment identity native evidence does not replay')
            if any(r['page_ref'] != inp['observation']['page_ref'] for r in local):
                raise ValueError('equipment source rows cross page scopes')
            motifs, rectangles = discover_cabinet_motifs(local)
            motifs_by_page[page['page_ref']].update((r['id'], r) for r in motifs)
            sources.update(((r['page_ref'], r['source_primitive_ref']), r) for r in local)
            items.append({'inputs': inp, 'query': query, 'motifs': motifs, 'rectangles': rectangles,
                          'source_rows': local, 'page_number': page['page_number']})
    observations = {r['inputs']['observation']['id']: r['inputs']['observation'] for r in items}
    typographies = {r['source_observation_ref']: r for r in typography}
    candidates, by_motif = {}, defaultdict(list)
    for item in items:
        inp = item['inputs']; obs, proposal = inp['observation'], inp['proposal']
        choices = [(motif, _label_zone(motif, obs)) for motif in motifs_by_page[obs['page_ref']].values()]
        choices = [(motif, zone) for motif, zone in choices if zone is not None]
        candidates[obs['id']] = choices
        for motif, _ in choices:
            by_motif[motif['id']].append(obs['id'])
    seeds = []
    for item in items:
        obs, proposal = item['inputs']['observation'], item['inputs']['proposal']
        choices = candidates[obs['id']]
        if (len(choices) == 1 and len(by_motif[choices[0][0]['id']]) == 1 and item['query']['complete']
                and _lexical_tag_eligible(proposal, obs)
                and _typography_valid(typographies.get(obs['id']), obs)):
            motif, zone = choices[0]
            seeds.append({'motif_ref': motif['id'], 'page_ref': obs['page_ref'], 'tag': proposal['candidate']['tag'],
                'class_token': proposal['candidate']['equipment_class_token'], 'source_observation_ref': obs['id'],
                'proposal_ref': proposal['id'], 'normalized_signature': motif['normalized_signature'],
                'font': typographies[obs['id']]['font'], 'zone': zone})
    if calibration is None:
        groups = []
        for token in sorted({s['class_token'] for s in seeds}):
            members = [s for s in seeds if s['class_token'] == token]
            if len({s['tag'] for s in members}) < POLICY['minimum_distinct_calibration_tags'] or len({s['page_ref'] for s in members}) < POLICY['minimum_calibration_pages']:
                continue
            if any(max(abs(x-y) for x, y in zip(a['normalized_signature'], b['normalized_signature'])) > POLICY['maximum_normalized_motif_residual']
                   or a['font'] != b['font'] for a in members for b in members):
                continue
            groups.append({'class_token': token, 'members': deepcopy(members), 'font': members[0]['font'],
                           'reference_signature': members[0]['normalized_signature']})
        calibration = {'record_type': 'mep_equipment_2d_drawing_convention', 'version': VERSION,
            'policy_sha256': _sha256(POLICY), 'source_pdf_sha256': native_inputs['source_pdf_sha256'],
            'groups': groups, 'epistemic_state': 'inferred', 'quantity_eligible': False}
        calibration['id'] = _stable_id('mep_equipment_2d_drawing_convention', calibration)
    elif calibration.get('policy_sha256') != _sha256(POLICY) or calibration.get('source_pdf_sha256') != native_inputs['source_pdf_sha256']:
        raise ValueError('equipment identity convention does not match policy/source')
    output = []
    for item in items:
        obs, proposal = item['inputs']['observation'], item['inputs']['proposal']
        choices = candidates[obs['id']]; reasons = []
        seed = next((s for s in seeds if s['source_observation_ref'] == obs['id']), None)
        if not item['query']['complete']:
            reasons.append('incomplete_native_body_and_competitor_search')
        if not choices:
            reasons.append('no_supported_cabinet_motif_in_typographic_label_zone')
        elif len(choices) != 1 or len(by_motif[choices[0][0]['id']]) != 1:
            reasons.append('body_tag_ownership_is_not_mutually_unique')
        if seed is None:
            reasons.append('native_tag_typography_or_unconflicted_tag_gate_not_closed')
        groups = [g for g in calibration['groups'] if seed and g['class_token'] == seed['class_token']
                  and g['font'] == seed['font'] and max(abs(x-y) for x,y in zip(g['reference_signature'],seed['normalized_signature'])) <= POLICY['maximum_normalized_motif_residual']]
        if len(groups) != 1:
            reasons.append('independent_repeated_symbol_and_tag_class_convention_not_closed')
        accepted = not reasons
        motif = choices[0][0] if len(choices) == 1 else None
        contacts = boundary_contact_candidates(motif['components'][0], item['source_rows']) if motif else []
        output.append({'id': _stable_id('mep_projected_equipment_identity', proposal['id'], motif['id'] if motif else None),
            'record_type': 'mep_projected_equipment_identity', 'record_version': VERSION,
            'page_ref': obs['page_ref'], 'page_number': item['page_number'], 'proposal_ref': proposal['id'],
            'source_observation_refs': [obs['id']], 'equipment_tag': proposal['candidate']['tag'],
            'source_m2_state': proposal['state'], 'source_m2_reasons': proposal.get('reasons', []),
            'resolved_context_abstentions': (['region_applicability_unresolved'] if accepted
                and proposal['state'] == 'abstained' else []),
            'equipment_class': EQUIPMENT_CLASSES.get(proposal['candidate']['equipment_class_token']),
            'body_motif_ref': motif['id'] if motif else None, 'body_bbox_display': motif['bbox_display'] if motif else None,
            'source_primitive_refs': motif['source_primitive_refs'] if motif else [],
            'native_query_sha256': item['query']['source_rows_sha256'], 'native_query_complete': item['query']['complete'],
            'drawing_convention_ref': calibration['id'], 'typography_ref': typographies.get(obs['id'], {}).get('id'),
            'label_zone_evidence': choices[0][1] if len(choices) == 1 else None,
            'candidate_body_motif_refs': [m['id'] for m, _ in choices], 'reasons': reasons,
            'state': 'accepted' if accepted else 'abstained', 'epistemic_state': 'inferred' if accepted else 'unknown',
            'identity_scope': 'page_local_projected_body_and_tag_only',
            'port_outcome': {'state': 'unresolved', 'port_geometry_candidates': contacts,
                'reason': 'body_tag_identity_does_not_certify_equipment_ports', 'equipment_port_binding_established': False},
            'authority': {'projected_body_tag_identity_established': accepted, 'equipment_port_binding_established': False,
                'physical_item_identity_established': False, 'physical_placement_established': False,
                'physical_count_established': False, 'quantity_eligible': False}, 'quantity_eligible': False})
    used = {(m['page_ref'], ref) for page in motifs_by_page.values() for m in page.values() for ref in m['source_primitive_refs']}
    return {'schema_version': VERSION, 'layer': 'mep_projected_equipment_identity',
        'document': native_inputs['document'], 'policy': deepcopy(POLICY), 'drawing_convention': calibration,
        'identities': output, 'body_motifs': [m for page in motifs_by_page.values() for m in page.values()],
        'native_source_evidence': [sources[key] for key in sorted(used)], 'typography_evidence': deepcopy(typography),
        'source_observations': list(observations.values()),
        'summary': {'assessed_tag_occurrences': len(output), 'accepted_projected_body_tag_identities': sum(r['state']=='accepted' for r in output),
            'accepted_equipment_port_bindings': 0, 'physical_item_count': None}, 'quantity_eligible': False}


def _check_frozen_terms(native_inputs, terminology):
    from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals
    errors = validate_mep_terminology_proposals(terminology)
    if errors or native_inputs['document'] != terminology['document']:
        raise ValueError('equipment identity terminology/source contract mismatch')
    proposals = {r['id']: r for r in terminology['proposals']}
    observations = {r['id']: r for r in terminology['source_observations']}
    pages = {p['page_ref'] for p in native_inputs['pages']}
    expected = {r['id'] for r in terminology['proposals']
                if r['page_ref'] in pages and r['candidate'].get('kind') == 'equipment_tag'}
    provided = [item['inputs']['proposal']['id'] for page in native_inputs['pages'] for item in page['items']]
    if len(provided) != len(set(provided)) or set(provided) != expected:
        raise ValueError('equipment identity tag competitor search is incomplete or duplicated')
    for page in native_inputs['pages']:
        for item in page['items']:
            inp = item['inputs']
            if inp['proposal'] != proposals.get(inp['proposal']['id']) or inp['observation'] != observations.get(inp['observation']['id']):
                raise ValueError('equipment identity tag evidence differs from frozen M2')
            if inp['proposal']['page_ref'] != page['page_ref'] or inp['observation']['page_ref'] != page['page_ref']:
                raise ValueError('equipment identity tag evidence crosses execution pages')


def replay_equipment_2d_identity(payload, *, native_inputs, typography, terminology,
                                calibration_payload=None, calibration_native_inputs=None,
                                calibration_typography=None, calibration_terminology=None):
    """Replay every accepted/abstained identity before an integration may use it.

    The caller loads native inputs and typography from independently hashed
    evidence artifacts, not by extracting evidence back out of the claim being
    verified. Held-out claims must also replay their original calibration.
    """
    _check_frozen_terms(native_inputs, terminology)
    if payload.get('input_payload_sha256') != {'terminology-proposals': _sha256(terminology)}:
        raise ValueError('equipment identity references a different frozen M2 payload')
    if payload.get('native_payload_sha256') != _sha256(native_inputs) or payload.get('typography_payload_sha256') != _sha256(typography):
        raise ValueError('equipment identity native/typography evidence hash mismatch')
    if calibration_payload is not None:
        if any(value is None for value in (calibration_native_inputs, calibration_typography, calibration_terminology)):
            raise ValueError('held-out equipment identity requires complete calibration replay evidence')
        replay_equipment_2d_identity(calibration_payload, native_inputs=calibration_native_inputs,
            typography=calibration_typography, terminology=calibration_terminology)
    if payload.get('calibration_payload_sha256') != (_sha256(calibration_payload) if calibration_payload else None):
        raise ValueError('equipment identity calibration payload hash mismatch')
    result = build_equipment_2d_identity(native_inputs=native_inputs, typography=typography,
        calibration=calibration_payload['drawing_convention'] if calibration_payload else None)
    if _sha256({key: payload.get(key) for key in result}) != _sha256(result):
        raise ValueError('equipment identity certificate does not replay exactly')
    expected_coverage = {'execution_page_numbers': [p['page_number'] for p in native_inputs['pages']],
        'calibration_page_refs': sorted({s['page_ref'] for g in result['drawing_convention']['groups'] for s in g['members']}),
        'calibration_consumes_current_scope': calibration_payload is None,
        'whole_package_equipment_coverage_established': False, 'engineer_approved': False,
        'reviewed_selectors_consumed': False}
    if payload.get('coverage') != expected_coverage:
        raise ValueError('equipment identity coverage claims differ from execution/calibration scopes')
    return result
