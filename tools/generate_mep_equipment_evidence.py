#!/usr/bin/env python3
"""Freeze native equipment discovery outcomes, including unresolved gates.

No reviewed coordinates, expected tag, body selector or M4 accept is an input.
The requested page numbers bound execution; every supported frozen M2 equipment
proposal on those pages is searched independently.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import _leader_paths
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_hvac_inventory import EQUIPMENT_CLASSES
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_equipment_observations import build_equipment_review_outcome
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _search_rect
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id


class RecordingQueries:
    def __init__(self, index):
        self.index, self.queries, self.sources = index, {}, {}

    def query(self, box):
        key = _sha256(box)
        if key in self.queries:
            query = self.queries[key]
            return [self.sources[ref] for ref in query['source_row_refs']], query['complete'], query['region_refs']
        rows, complete, regions = self.index.query(box)
        self.sources.update((r['id'], r) for r in rows)
        self.queries[key] = {'bbox_display': box, 'complete': complete, 'region_refs': regions,
            'source_row_refs': [r['id'] for r in rows], 'source_rows_sha256': _sha256(rows)}
        return rows, complete, regions


def grouping_evidence(document, page):
    streams = [document.xref_stream(ref) for ref in page.get_contents()]
    return {'page_xref': page.xref, 'content_xrefs': page.get_contents(),
        'content_stream_sha256': [hashlib.sha256(data).hexdigest() for data in streams],
        'form_xobject_count': len(page.get_xobjects()),
        'marked_content_count': sum(data.count(b'BDC') + data.count(b'BMC') for data in streams),
        'mcid_count': sum(data.count(b'/MCID') for data in streams),
        'document_ocg_count': len(document.get_ocgs()),
        'grouping_is_equipment_ownership': False}


def replay_outcomes(frozen):
    results = []
    for page in frozen['pages']:
        rows = {r['id']: r for r in page['source_rows']}
        for item in page['items']:
            # Unsupported tags remain in the raw native input as ownership
            # competitors; this older body/port diagnostic has no such class.
            if item['inputs']['proposal']['candidate'].get('equipment_class_token') not in EQUIPMENT_CLASSES:
                continue
            query = page['queries'][item['query_key']]
            sources = [rows[ref] for ref in query['source_row_refs']]
            if _sha256(sources) != query['source_rows_sha256']:
                raise ValueError('equipment native source rows changed')
            results.append(build_equipment_review_outcome(**item['inputs'], source_rows=sources))
    return results


def bounded_review_payload(payload, frozen):
    """Present compact per-tag outcomes; candidate geometry keeps no authority."""
    outcomes, evidence_records = [], []
    source_rows = {(r['page_ref'], r['source_primitive_ref']): r for page in frozen['pages'] for r in page['source_rows']}
    used_sources = set()
    for row in payload['outcomes']:
        query_id = _stable_id('mep_equipment_native_query', row['page_ref'], row['native_query']['source_rows_sha256'])
        native_query = {key: value for key, value in row['native_query'].items() if key != 'source_primitive_refs'}
        native_query.update({'id': query_id, 'record_type': 'mep_equipment_native_query',
            'page_ref': row['page_ref'], 'bbox_display': row['search_bbox_display'],
            'native_evidence_artifact': payload['native_evidence_artifact'], 'quantity_eligible': False})
        evidence_records.append(native_query)
        for candidate in row['body_outline_candidates'] + row['port_geometry_candidates']:
            source_keys = [(row['page_ref'], ref) for ref in candidate['source_primitive_refs']]
            used_sources.update(source_keys)
            evidence_records.append({**candidate, 'page_ref': row['page_ref'], 'query_ref': query_id,
                'source_evidence_refs': [source_rows[key]['id'] for key in source_keys]})
        source_paths = [{'page_ref': row['page_ref'], 'bbox_display': row['bbox_display'],
            'source_observation_refs': row['source_observation_refs'], 'role': 'equipment_tag_observation'},
            {'page_ref': row['page_ref'], 'bbox_display': row['search_bbox_display'],
             'source_primitive_refs': [], 'role': 'complete_native_search_scope'}]
        source_paths.extend({'page_ref': row['page_ref'], 'points_display': candidate['points_display'],
            'source_primitive_refs': candidate['source_primitive_refs'], 'role': 'unresolved_body_outline_candidate'}
            for candidate in row['body_outline_candidates'])
        reasons = sorted(set(row['tag_body_outcome']['reasons'] + row['body_outcome']['reasons']
                             + row['port_outcome']['reasons']))
        outcomes.append({'id': row['id'], 'record_type': 'mep_bounded_review_outcome',
            'page_ref': row['page_ref'], 'page_number': row['page_number'], 'subject_kind': 'equipment',
            'state': 'abstained', 'subject_refs': [row['proposal_ref'], *row['source_observation_refs']],
            'reason_codes': reasons, 'evidence_refs': [row['proposal_ref'], *row['source_observation_refs'], query_id],
            'source_paths': source_paths,
            'description': f"{row['tag']}: native outline/contact geometry retained; tag ownership and equipment ports remain unresolved.",
            'stage_outcomes': {**{key: row[key] for key in ['tag_body_outcome', 'body_outcome', 'port_outcome',
                                                          'pdf_grouping']}, 'native_query': native_query},
            'native_equipment_evidence_ref': payload['native_evidence_artifact'],
            'quantity_eligible': False, 'physical_continuation_established': False,
            'physical_item_identity_established': False, 'evaluation_only': True})
    evidence_records.extend(source_rows[key] for key in sorted(used_sources))
    return {'schema_version': '0.1.0', 'layer': 'mep_bounded_review_outcomes',
        'document': {**payload['document'], 'source_pdf_sha256': payload['source_pdf_sha256']},
        'input_payload_sha256': {'terminology-proposals': payload['terminology_payload_sha256']},
        'coverage': payload['coverage'], 'authority': {'evaluation_only': True}, 'outcomes': outcomes,
        'evidence_records': evidence_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--terminology', type=Path)
    parser.add_argument('--pages', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--include-unsupported', action='store_true',
                        help='Retain every M2 equipment tag as a possible ownership competitor; does not add supported classes')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replay-from', type=Path)
    parser.add_argument('--freeze-only', action='store_true', help='write only the compressed native replay fixture')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.replay_from:
        with gzip.open(args.replay_from, 'rt') as stream:
            frozen = json.load(stream)
    else:
        if not args.source or not args.terminology:
            parser.error('--source and --terminology are required for native discovery')
        terminology = json.loads(args.terminology.read_text())
        source_sha256 = _file_sha256(args.source)
        observations = {r['id']: r for r in terminology['source_observations']}
        proposals = [r for r in terminology['proposals']
                     if r['candidate'].get('kind') == 'equipment_tag'
                     and (args.include_unsupported or r['candidate'].get('equipment_class_token') in EQUIPMENT_CLASSES)]
        frozen = {'schema_version': '0.1.0', 'layer': 'mep_equipment_native_evidence_replay',
            'source_pdf_sha256': source_sha256, 'terminology_payload_sha256': _sha256(terminology),
            'document': terminology['document'], 'pages': [],
            'coverage': {'execution_page_numbers': sorted(set(args.pages)),
                'whole_package_equipment_coverage_established': False,
                'reviewed_selectors_consumed': False, 'engineer_approval_established': False}}
        with fitz.open(args.source) as document:
            for number in sorted(set(args.pages)):
                selected = [(proposal, observations[ref]) for proposal in proposals for ref in proposal['evidence_refs']
                            if ref in observations and observations[ref].get('page_number') == number]
                if not selected:
                    continue
                if any(o.get('source_pdf_sha256') != source_sha256 for _, o in selected):
                    raise ValueError('equipment observation source PDF differs from native query source')
                page = document[number - 1]
                index = RecordingQueries(NativeBoundaryQueries(page, selected[0][0]['page_ref']))
                grouping = grouping_evidence(document, page)
                items = []
                for proposal, observation in selected:
                    box = _search_rect(observation['bbox_display'], [page.rect.width, page.rect.height], 6.)
                    rows, complete, regions = index.query(box)
                    paths, leader_complete = _leader_paths(index, observation)
                    inputs = {'proposal': proposal, 'observation': observation, 'search_bbox_display': box,
                        'search_complete': complete, 'region_refs': regions,
                        'leader_paths': paths, 'leader_search_complete': leader_complete, 'pdf_grouping': grouping}
                    items.append({'query_key': _sha256(box), 'inputs': inputs})
                    print(f"page {number}: {proposal['candidate']['tag']}, {len(rows)} original primitives", flush=True)
                frozen['pages'].append({'page_number': number, 'page_ref': selected[0][0]['page_ref'],
                    'source_rows': list(index.sources.values()), 'queries': index.queries, 'items': items})
    outcomes = replay_outcomes(frozen)
    frozen['expected_outcomes_sha256'] = _sha256(outcomes)
    evidence_path = args.output_dir / 'native-equipment-evidence.json.gz'
    with evidence_path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            stream.write(json.dumps(frozen, separators=(',', ':')).encode())
    if args.freeze_only:
        print(f'frozen {len(outcomes)} equipment outcomes: {evidence_path}', flush=True)
        return
    payload = {'schema_version': '0.1.0', 'layer': 'mep_equipment_review_outcomes',
        'document': frozen['document'], 'source_pdf_sha256': frozen['source_pdf_sha256'],
        'terminology_payload_sha256': frozen['terminology_payload_sha256'],
        'native_evidence_artifact': {'filename': evidence_path.name, 'sha256': _file_sha256(evidence_path)},
        'coverage': frozen['coverage'], 'outcomes': outcomes,
        'summary': {'equipment_observation_count': len(outcomes), 'accepted_body_tag_port_chain_count': 0,
            'state_counts': dict(Counter(r['state'] for r in outcomes)),
            'body_outline_candidate_count': sum(len(r['body_outline_candidates']) for r in outcomes),
            'port_geometry_candidate_count': sum(len(r['port_geometry_candidates']) for r in outcomes)},
        'quantity_eligible': False}
    (args.output_dir / 'equipment-review-outcomes.json').write_text(json.dumps(payload, indent=2) + '\n')
    (args.output_dir / 'bounded-equipment-review-outcomes.json').write_text(
        json.dumps(bounded_review_payload(payload, frozen), indent=2) + '\n')
    print(json.dumps(payload['summary']), flush=True)


if __name__ == '__main__':
    main()
