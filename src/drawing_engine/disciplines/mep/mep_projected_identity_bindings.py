"""M4 companion certificates for projected identity, independent of port/3D gates.

Discovery certificates are replayed before use. This layer never adds an old
M4 equipment_endpoint relation, an M7 occurrence, physical identity or quantity.
"""

from copy import deepcopy

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_equipment_identity import replay_equipment_2d_identity
from src.drawing_engine.disciplines.mep.mep_fitting_hypotheses import replay_fitting_hypotheses
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


def projected_fitting_bindings(replay):
    return [{**deepcopy(row), 'id': _stable_id('mep_projected_fitting_binding', row['id']),
             'record_type': 'mep_projected_fitting_binding',
             'relation_type': 'projected_fitting_body_branch', 'source_certificate_ref': row['id'],
             'automatic_attribute_propagation': False}
            for row in replay['fitting_hypotheses']]


def build_projected_identity_bindings(*, attribute_bindings, terminology,
                                      equipment_identity=None, equipment_replay=None,
                                      fitting_hypotheses=None, fitting_replay=None):
    errors = validate_mep_attribute_bindings(attribute_bindings)
    if errors or attribute_bindings['m2_contract_ref']['payload_sha256'] != _sha256(terminology):
        raise ValueError('projected identity requires matching frozen M2/M4 inputs')
    equipment, fittings = [], []
    inputs = {'attribute-bindings': _sha256(attribute_bindings), 'terminology-proposals': _sha256(terminology)}
    if equipment_identity is not None:
        if equipment_replay is None:
            raise ValueError('projected equipment identity requires independent native replay inputs')
        replay = replay_equipment_2d_identity(equipment_identity, terminology=terminology, **equipment_replay)
        if replay['document'] != attribute_bindings['document']:
            raise ValueError('projected equipment identity belongs to another document')
        inputs['equipment-2d-identities'] = _sha256(equipment_identity)
        motifs = {row['id']: row for row in replay['body_motifs']}
        for identity in replay['identities']:
            motif = motifs.get(identity['body_motif_ref'])
            equipment.append({'id': _stable_id('mep_projected_equipment_binding', identity['id']),
                'record_type': 'mep_projected_equipment_binding', 'record_version': '0.1.0',
                'relation_type': 'projected_equipment_body_tag', 'page_ref': identity['page_ref'],
                'proposal_ref': identity['proposal_ref'], 'source_observation_refs': identity['source_observation_refs'],
                'source_primitive_refs': identity['source_primitive_refs'],
                'source_certificate_ref': identity['id'], 'body_motif_ref': identity['body_motif_ref'],
                'body_bbox_display': identity['body_bbox_display'],
                'body_paths': [deepcopy(row) for row in motif['components']] if motif else [],
                'equipment_class': identity['equipment_class'], 'equipment_tag': identity['equipment_tag'],
                'state': identity['state'], 'epistemic_state': identity['epistemic_state'],
                'identity_scope': identity['identity_scope'], 'reasons': identity['reasons'],
                'source_m2_state': identity['source_m2_state'], 'source_m2_reasons': identity['source_m2_reasons'],
                'resolved_context_abstentions': identity['resolved_context_abstentions'],
                'drawing_convention_ref': identity['drawing_convention_ref'],
                'port_outcome': deepcopy(identity['port_outcome']),
                'authority': deepcopy(identity['authority']), 'quantity_eligible': False})
    if fitting_hypotheses is not None:
        if fitting_replay is None:
            raise ValueError('projected fitting identity requires independent native replay inputs')
        replay = replay_fitting_hypotheses(fitting_hypotheses, **fitting_replay)
        if (replay['document'] != attribute_bindings['document'] or
                replay['m3_payload_sha256'] != attribute_bindings['m3_contract_ref']['payload_sha256']):
            raise ValueError('projected fitting identity differs from frozen M3/M4')
        inputs['fitting-hypotheses'] = _sha256(fitting_hypotheses)
        fittings = projected_fitting_bindings(replay)
    return {'schema_version': '0.1.0', 'layer': 'mep_projected_identity_bindings',
        'document': deepcopy(attribute_bindings['document']), 'input_payload_sha256': inputs,
        'equipment_bindings': equipment, 'fitting_bindings': fittings,
        'summary': {'projected_equipment_identity_count': sum(row['state'] == 'accepted' for row in equipment),
                    'projected_fitting_connection_count': sum(row['accepted_projected_connection'] for row in fittings),
                    'equipment_port_binding_count': 0, 'physical_item_count': None},
        'exchange_contract': {'projected_identity_only': True, 'existing_m4_port_gate_unchanged': True,
            'automatic_elevation_propagation': False, 'physical_item_identity_established': False,
            'physical_continuation_established': False, 'quantity_eligible': False}, 'quantity_eligible': False}
