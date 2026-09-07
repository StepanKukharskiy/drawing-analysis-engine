#!/usr/bin/env python3
"""Capture complete native endpoint queries for all automatically blocked caps.

Writes a separate negative-only replay layer. Existing package files remain
immutable; no previous accepted binding or reviewed item selects a witness.
"""

import argparse
from collections import Counter
import gc
import gzip
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fitz

from tools.inspect_mep_binding_scope_delta import read_field
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_cap_replay import (
    VERSION, implementation_manifest, select_native_cap_witnesses,
    cap_query_plan, build_native_cap_page,
)
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _native_display_geometry
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.core.vector_topology import iter_native_segments


def _write(path, payload):
    staging = path.with_name(path.name + '.partial')
    with staging.open('w') as stream:
        json.dump(payload, stream, ensure_ascii=True, separators=(',', ':'))
        stream.write('\n')
    staging.replace(path)


def _write_archive(path, payload):
    staging = path.with_name(path.name + '.partial')
    with staging.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=1) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=True, separators=(',', ':'))
    staging.replace(path)


def capture_native_cap_page(*, index, registry, witnesses, selection, selection_inputs, implementation, progress=None):
    """Capture original native member IDs and deduplicated full endpoint queries."""
    page_ref = index.page_ref
    scope = next(p for p in registry['pages'] if p['page_ref'] == page_ref)
    by_ref = {r['id']: r for r in witnesses}
    selected_refs = {ref for selected in selection for ref in by_ref[selected['witness_ref']]['member_source_primitive_refs']}
    indices = set()
    for ref in selected_refs:
        match = re.fullmatch(r'drawing\[(\d+)\]\.item\[\d+\]\.segment\[\d+\]', ref)
        if match is None or int(match[1]) >= len(index.drawings):
            raise ValueError('cap member is not an original canonical native primitive')
        indices.add(int(match[1]))
    identities = {id(index.drawings[i]) for i in indices}
    members = {}
    for native in iter_native_segments(index.drawings, drawing_filter=lambda d: id(d) in identities):
        if native['id'] not in selected_refs:
            continue
        points, bounds, search = _native_display_geometry(native, index.rotation)
        row = {'id': _stable_id('mep_native_target_primitive', page_ref, native['id']),
            'page_ref': page_ref, 'source_primitive_ref': native['id'], 'source_native_segment': native,
            'points_display': points, 'bbox_display': bounds, 'search_bbox_display': search,
            'pdf_to_display_matrix': list(index.rotation), 'state': 'observed', 'quantity_eligible': False}
        members[native['id']] = row
    if set(members) != selected_refs:
        raise ValueError('fresh PDF does not contain every selected native cap member')
    sources = {r['id']: r for r in members.values()}
    queries, cases = {}, []
    for ordinal, selected in enumerate(selection, 1):
        witness = by_ref[selected['witness_ref']]
        member_rows = [members[r] for r in witness['member_source_primitive_refs']]
        _, _, _, _, boxes = cap_query_plan(witness, member_rows)
        keys = []
        for box in boxes:
            key = _sha256(box)
            keys.append(key)
            if key in queries:
                continue
            rows, complete, regions = index.query(box)
            queries[key] = {'bbox_display': box, 'complete': complete, 'region_refs': regions,
                'source_row_refs': [r['id'] for r in rows], 'source_rows_sha256': _sha256(rows)}
            for row in rows:
                if row['id'] in sources and sources[row['id']] != row:
                    raise ValueError('native cap member/query source records disagree')
                sources[row['id']] = row
        cases.append({'witness_ref': witness['id'], 'input_witness': witness, 'selection': selected,
            'member_row_refs': [r['id'] for r in member_rows], 'query_keys': keys})
        if progress:
            progress({'page': scope['page_number'], 'captured_witnesses': ordinal, 'selected_witnesses': len(selection),
                      'deduplicated_queries': len(queries), 'deduplicated_native_rows': len(sources)})
    return {'schema_version': VERSION, 'layer': 'mep_native_cap_replay_sources',
        'document': registry['document'], 'm1_payload_sha256': _sha256(registry),
        'page_ref': page_ref, 'page_number': scope['page_number'],
        'implementation_sha256': implementation, 'selection_inputs': selection_inputs,
        'source_rows': list(sources.values()), 'queries': queries, 'cases': cases,
        'capture_scope': 'complete fresh full-native endpoint queries for automatically selected unresolved witnesses',
        'quantity_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'registry', 'run', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--selection-only', action='store_true')
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    registry = load(args.registry)
    terminology = load(args.run / 'terminology-proposals.json')
    context = load(args.run / 'run-context.json')
    if (_file_sha256(args.source) != registry['document']['source_pdf_sha256']
            or terminology['document'] != registry['document']
            or context['inputs_sha256']['registry'] != _sha256(registry)):
        raise ValueError('native cap inputs do not reference one frozen source and M1')
    unresolved = read_field(args.run / 'automatic-discovery.json', 'unresolved_proposals')
    blocked = {ref for r in unresolved if r.get('reason') == 'unresolved_competing_envelope_at_contact'
               and len(r.get('target_refs', [])) == 1 for ref in r.get('evidence_refs', [])}
    numbers = {r['page_number'] for r in terminology['source_observations'] if r['id'] in blocked}
    selected_pages = []
    for scope in registry['pages']:
        number, page_ref = scope['page_number'], scope['page_ref']
        if number not in numbers:
            continue
        target_path = args.run / f'page-{number:03d}.automatic-targets.json'
        checkpoint = load(args.run / f'page-{number:03d}.checkpoint.json')
        if (checkpoint['run_context_sha256'] != _sha256(context)
                or checkpoint['record_sha256'] != _sha256(checkpoint['record'])
                or checkpoint['targets_sha256'] != _file_sha256(target_path)):
            raise ValueError('native cap selection upstream checkpoint differs')
        leaders = read_field(target_path, 'leader_observations')
        witnesses = read_field(target_path, 'envelope_searches')
        selection = select_native_cap_witnesses(terminology, unresolved, leaders, witnesses)
        if selection:
            selected_pages.append({'page_ref': page_ref, 'page_number': number,
                'target_file_sha256': checkpoint['targets_sha256'], 'selection': selection})
    summary = {'automatically_selected_witnesses': sum(len(p['selection']) for p in selected_pages),
               'pages': {p['page_number']: len(p['selection']) for p in selected_pages}}
    print(json.dumps({'phase': 'native_cap_selection', **summary}), flush=True)
    if args.selection_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'native-cap-replay-manifest.json').exists():
        raise ValueError('native cap output is already frozen; choose a new output directory')
    implementation = implementation_manifest()
    manifest = {'schema_version': VERSION, 'layer': 'mep_native_cap_replay_manifest',
        'document': registry['document'], 'm1_payload_sha256': _sha256(registry),
        'terminology_payload_sha256': _sha256(terminology), 'upstream_context_payload_sha256': _sha256(context),
        'upstream_run': str(args.run.resolve()), 'source_pdf': str(args.source.resolve()),
        'implementation_sha256': implementation, 'selection_summary': summary, 'pages': [],
        'authority': {'negative_cap_witness_only': True, 'positive_acceptance_authority': False, 'quantity_eligible': False}}
    with fitz.open(args.source) as document:
        for selected_page in selected_pages:
            number, page_ref = selected_page['page_number'], selected_page['page_ref']
            target_path = args.run / f'page-{number:03d}.automatic-targets.json'
            leaders = read_field(target_path, 'leader_observations')
            witnesses = read_field(target_path, 'envelope_searches')
            page_observations = [r for r in terminology['source_observations'] if r['page_ref'] == page_ref]
            observation_refs = {r['id'] for r in page_observations}
            inputs = {'terminology': {'document': terminology['document'], 'source_observations': page_observations},
                'full_terminology_payload_sha256': _sha256(terminology),
                'unresolved_proposals': [r for r in unresolved if observation_refs.intersection(r.get('evidence_refs', []))],
                'leader_observations': leaders}
            print(json.dumps({'phase': 'native_cap_page_start', 'page': number}), flush=True)
            index = NativeBoundaryQueries(document[number - 1], page_ref)
            native = capture_native_cap_page(index=index, registry=registry, witnesses=witnesses,
                selection=selected_page['selection'], selection_inputs=inputs, implementation=implementation,
                progress=lambda row: print(json.dumps({'phase': 'native_cap_query', **row}), flush=True))
            del index
            gc.collect()
            payload = build_native_cap_page(native_page=native, registry=registry, upstream_witnesses=witnesses,
                                            selection=selected_page['selection'])
            archive = args.out / f'page-{number:03d}.native-cap-queries.json.gz'
            outcomes = args.out / f'page-{number:03d}.native-cap-outcomes.json'
            _write_archive(archive, native)
            _write(outcomes, payload)
            manifest['pages'].append({'page_ref': page_ref, 'page_number': number,
                'upstream_target_file_sha256': selected_page['target_file_sha256'],
                'native_archive': {'path': archive.name, 'file_sha256': _file_sha256(archive), 'payload_sha256': _sha256(native)},
                'outcomes': {'path': outcomes.name, 'file_sha256': _file_sha256(outcomes), 'payload_sha256': _sha256(payload)},
                'summary': payload['summary'], 'native_query_count': len(native['queries']),
                'native_source_row_count': len(native['source_rows'])})
            print(json.dumps({'phase': 'native_cap_page_complete', **manifest['pages'][-1]}), flush=True)
            _write(args.out / 'native-cap-replay-progress.json', manifest)
            del native, payload, witnesses, leaders, inputs
            gc.collect()
    if implementation != implementation_manifest():
        raise RuntimeError('native cap implementation changed during capture')
    manifest['summary'] = dict(sum((Counter(p['summary']) for p in manifest['pages']), Counter()))
    _write(args.out / 'native-cap-replay-manifest.json', manifest)
    print(json.dumps({'phase': 'native_cap_replay_complete', 'out': str(args.out), 'summary': manifest['summary']}), flush=True)


if __name__ == '__main__':
    main()
