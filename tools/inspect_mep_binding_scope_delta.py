#!/usr/bin/env python3
"""Freeze binding losses and automatically enumerate their cap-review universe.

This is a read-only diagnostic. It grants no binding or rejection authority.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import _contact
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256


from src.drawing_engine.project.mep_json_field import read_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['reference_run', 'run', 'out']:
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text())
    before = load(args.reference_run / 'attribute-bindings.json')
    after = load(args.run / 'attribute-bindings.json')
    old_terms = load(args.reference_run / 'terminology-proposals.json')
    new_terms = load(args.run / 'terminology-proposals.json')
    if before['document'] != after['document']:
        raise ValueError('binding comparison source differs')
    prior = {r['id']: r for r in before['relations'] if r['state'] == 'accepted'}
    scope = {r['page_ref'] for r in before['relations']}
    current = {r['id']: r for r in after['relations'] if r['state'] == 'accepted' and r['page_ref'] in scope}
    old_obs = {r['id']: r for r in old_terms['source_observations']}
    new_obs = {r['id']: r for r in new_terms['source_observations']}
    old_props = {r['id']: r for r in old_terms['proposals']}
    losses = []
    for ref in sorted(set(prior) - set(current)):
        relation = prior[ref]
        chain = [old_props[relation['proposal_ref']]['anchor_ref']]
        while old_obs[chain[-1]].get('source_observation_ref') in old_obs:
            chain.append(old_obs[chain[-1]]['source_observation_ref'])
        losses.append({'prior_relation': relation, 'source_observation_chain': chain,
                       'native_source_observation_ref': chain[-1]})
    unresolved = read_field(args.run / 'automatic-discovery.json', 'unresolved_proposals')
    blocked = {ref for row in unresolved
        if row['reason'] == 'unresolved_competing_envelope_at_contact' and len(row['target_refs']) == 1
        for ref in row['evidence_refs'] if ref in new_obs}
    selected = {}
    cases = []
    loss_refs = {r['native_source_observation_ref'] for r in losses}
    for number in sorted({new_obs[ref]['page_number'] for ref in blocked}):
        path = args.run / f'page-{number:03d}.automatic-targets.json'
        leaders = {r['observation_ref']: r for r in read_field(path, 'leader_observations')}
        witnesses = read_field(path, 'envelope_searches')
        refs = sorted(ref for ref in blocked if new_obs[ref]['page_number'] == number)
        for ref in refs:
            leader = leaders[ref]
            if not leader['search_complete'] or len(leader['paths']) != 1 or not leader['paths'][0]['search_complete']:
                raise ValueError('reported otherwise-unique source does not have a complete unique leader')
            point = leader['paths'][0]['contact_point_display']
            blockers = [w for w in witnesses if w['page_ref'] == new_obs[ref]['page_ref']
                and w['state'] == 'unresolved' and _contact(point, w)]
            for witness in blockers:
                selected[witness['id']] = witness
            if ref not in loss_refs:
                continue
            old_path = args.reference_run / f'page-{number:03d}.automatic-targets.json'
            old_leader = next(r for r in read_field(old_path, 'leader_observations') if r['observation_ref'] == ref)
            old_witnesses = {r['id']: r for r in read_field(old_path, 'envelope_searches')}
            cases.append({'native_source_observation_ref': ref, 'page_number': number,
                'source_observation_unchanged': old_obs[ref] == new_obs[ref],
                'native_source_observation': new_obs[ref], 'prior_leader': old_leader, 'current_leader': leader,
                'current_automatic_reason': 'unresolved_competing_envelope_at_contact',
                'witness_comparisons': [{'id': w['id'], 'prior': old_witnesses.get(w['id']), 'current': w,
                    'changed_fields': [k for k in sorted(set(w) | set(old_witnesses.get(w['id'], {})))
                        if w.get(k) != old_witnesses.get(w['id'], {}).get(k)]} for w in blockers]})
    payload = {'schema_version': '0.1.0', 'layer': 'mep_binding_scope_delta', 'document': after['document'],
        'authority': {'evaluation_only': True, 'quantity_eligible': False},
        'input_payload_sha256': {'prior_m4': _sha256(before), 'current_m4': _sha256(after),
            'prior_m2': _sha256(old_terms), 'current_m2': _sha256(new_terms)},
        'summary': {'prior_scope_accepted': len(prior), 'current_same_scope_accepted': len(current),
            'prior_exact_accepted_records_retained': sum(prior[r] == current[r] for r in set(prior) & set(current)),
            'lost_relation_count': len(losses), 'lost_types': dict(Counter(r['prior_relation']['relation_type'] for r in losses)),
            'automatically_blocked_source_annotation_count': len(blocked),
            'automatically_selected_legacy_cap_witness_count': len(selected),
            'fresh_cap_page_counts': dict(Counter(r['page_ref'] for r in selected.values()))},
        'losses': losses, 'native_loss_cases': cases,
        'fresh_cap_selection': 'All legacy unresolved witnesses that block otherwise unique complete automatic leader contacts; no reviewed IDs',
        'fresh_cap_witnesses': [selected[k] for k in sorted(selected)],
        'negative_cap_replay_available_from_receipts_alone': False,
        'missing_replay_evidence': 'Actual native cap-query rows and full clustered endpoint graph; receipts retain only source-ID hashes/counts.',
        'quantity_eligible': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'summary': payload['summary'], 'out': str(args.out), 'file_sha256': _file_sha256(args.out)}))


if __name__ == '__main__':
    main()
