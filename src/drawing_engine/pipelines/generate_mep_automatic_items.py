#!/usr/bin/env python3
"""Automatic bounded item discovery into existing M4/M5A/M7A certificates.

Page numbers select execution coverage, never geometry or expected outcomes.
Reviewed truth is deliberately not an input. The existing M1 registration is
frozen upstream evidence, not claimed as a newly automatic registration.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import fitz

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import (
    RegionIndex, build_automatic_page_components, build_automatic_targets, automatic_binding_evidence,
    apply_geometric_text_applicability,
)
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_bounded_local_3d import build_mep_bounded_local_3d_segments
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import build_mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_item_catalog import build_mep_item_catalog, flatten_mep_item_catalog
from src.drawing_engine.disciplines.mep.mep_item_ocr_observations import build_mep_item_ocr_proposals
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _seeds, iter_bounded_native_page_regions
from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import propagate_collinear_applicability
from src.drawing_engine.disciplines.mep.mep_route_observations import build_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals
from src.drawing_engine.disciplines.mep.mep_text_observations import build_mep_document_text_proposals
from src.drawing_engine.disciplines.mep.mep_v3_page_cache import MepV3PageCache, page_cache_key, page_component_record_stream
from src.drawing_engine.project.project_packed_store import PackedProjectStore


def _write(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=True, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def run(*, source, registry, native_text, output_dir, annotations, ocr=None,
        page_numbers=None, region_size=64, region_budget=4000, minimum_member_length=24, resume=False,
        refine_from=None, rebind_from=None, native_cap_replay_dir=None,
        ownership_from=None, stroke_ownership_queries=None, v3_shadow_database=None,
        v3_page_cache_database=None, page_cache_seed_dir=None, page_cache_parity_report=None,
        export_json=True, export_json_scope='all', envelope_engine='per_candidate',
        native_scratch_dir=None):
    if v3_page_cache_database is None and not export_json:
        raise ValueError('automatic runs require a v3 page cache; use --export-json only for diagnostics')
    if (page_cache_seed_dir is None) != (page_cache_parity_report is None):
        raise ValueError('page-cache seed input requires an exact parity report')
    if page_cache_seed_dir is not None and v3_page_cache_database is None:
        raise ValueError('page-cache seed input requires a detached v3 cache database')
    if ownership_from is not None and (refine_from is not None or rebind_from is not None or stroke_ownership_queries is None):
        raise ValueError('stroke ownership rebind requires its frozen query capture and a single geometry source')
    if stroke_ownership_queries is not None and ownership_from is None:
        raise ValueError('stroke ownership queries require an explicit frozen geometry source')
    if rebind_from is not None and (refine_from is not None or native_cap_replay_dir is None):
        raise ValueError('cap rebind requires its native replay archive and cannot also refine geometry')
    if native_cap_replay_dir is not None and rebind_from is None:
        raise ValueError('native cap replay requires an explicit frozen rebind source')
    implementation_paths = [Path(__file__), *(ROOT / name for name in (
        'src/drawing_engine/disciplines/mep/mep_automatic_target_binding.py', 'src/drawing_engine/disciplines/mep/mep_native_target_discovery.py', 'src/drawing_engine/core/vector_topology.py',
        'src/drawing_engine/disciplines/mep/mep_native_descriptor_pack.py', 'src/drawing_engine/disciplines/mep/mep_packed_spatial_index.py',
        'src/drawing_engine/disciplines/mep/mep_terminology_proposals.py', 'src/drawing_engine/disciplines/mep/mep_text_observations.py', 'src/drawing_engine/disciplines/mep/mep_item_ocr_observations.py',
        'src/drawing_engine/disciplines/mep/mep_route_observations.py', 'src/drawing_engine/disciplines/mep/mep_outlined_route_composites.py', 'src/drawing_engine/disciplines/mep/mep_attribute_binding.py',
        'src/drawing_engine/disciplines/mep/mep_bounded_local_3d.py', 'src/drawing_engine/disciplines/mep/mep_cross_sheet_runs.py', 'src/drawing_engine/disciplines/mep/mep_item_catalog.py',
        'src/drawing_engine/disciplines/mep/mep_native_boundary_connections.py', 'src/drawing_engine/disciplines/mep/mep_native_bend_connections.py', 'src/drawing_engine/disciplines/mep/mep_native_boundary_queries.py',
        'src/drawing_engine/disciplines/mep/mep_native_branch_connections.py'))]
    if rebind_from is not None:
        implementation_paths.extend([ROOT / 'src/drawing_engine/disciplines/mep/mep_native_cap_replay.py',
                                     ROOT / 'tools/rebind_mep_package_attributes.py'])
    if ownership_from is not None:
        implementation_paths.extend([ROOT / 'src/drawing_engine/disciplines/mep/mep_projected_trace_completion.py',
                                     ROOT / 'src/drawing_engine/disciplines/mep/mep_native_equipment_observations.py',
                                     ROOT / 'src/drawing_engine/core/dimension_attachment.py',
                                     ROOT / 'src/drawing_engine/pipelines/generate_mep_stroke_ownership.py'])
    implementation = {str(p.relative_to(ROOT)): _file_sha256(p) for p in implementation_paths}
    if _file_sha256(source) != registry['document']['source_pdf_sha256']:
        raise ValueError('source PDF hash does not match M1')
    if native_text['document'] != registry['document'] or native_text['m1_payload_sha256'] != _sha256(registry):
        raise ValueError('native text does not match frozen M1')
    if annotations['document']['source_pdf_sha256'] != registry['document']['source_pdf_sha256']:
        raise ValueError('annotation source mismatch')
    selected = set(range(1, registry['document']['page_count'] + 1)) if page_numbers is None else set(page_numbers)
    if not selected or any(type(n) is not int or n < 1 or n > registry['document']['page_count'] for n in selected):
        raise ValueError('invalid page execution scope')
    selected_refs = {row['page_ref'] for row in registry['pages'] if row['page_number'] in selected}
    terminology = build_mep_document_text_proposals(native_text)
    if ocr is not None:
        if (ocr['document'] != registry['document'] or ocr['m1_payload_sha256'] != _sha256(registry)
                or ocr['native_text_payload_sha256'] != _sha256(native_text)):
            raise ValueError('OCR provenance does not match native/M1 evidence')
        additional = build_mep_item_ocr_proposals(ocr)
        terminology['source_observations'].extend(additional['source_observations'])
        terminology['proposals'].extend(additional['proposals'])
        terminology['observation_diagnostics'].extend(additional['observation_diagnostics'])
    invalid = {row['observation_ref'] for row in terminology.get('observation_diagnostics', [])}
    seed_refs = {row['observation_ref'] for row in _seeds(terminology['source_observations'], invalid)}
    observations = [row for row in terminology['source_observations'] if row['id'] in seed_refs and row['page_ref'] in selected_refs]
    terminology['unprocessed_proposal_refs'] = [row['id'] for row in terminology['proposals'] if row['page_ref'] not in selected_refs]
    terminology['proposals'] = [row for row in terminology['proposals'] if row['page_ref'] in selected_refs]
    terminology['summary']['source_observation_count'] = len(terminology['source_observations'])
    terminology['summary']['proposal_count'] = len(terminology['proposals'])
    for field, key in (('proposal_type_counts', 'proposal_type'), ('state_counts', 'state')):
        terminology['summary'][field] = {value: sum(row[key] == value for row in terminology['proposals'])
                                        for value in sorted({row[key] for row in terminology['proposals']})}
    errors = validate_mep_terminology_proposals(terminology)
    if errors:
        raise ValueError('; '.join(errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_context = {'implementation_sha256': implementation,
        'inputs_sha256': {name: _sha256(value) for name, value in
                         {'registry': registry, 'native_text': native_text,
                          'annotations': annotations, 'ocr': ocr}.items()},
        'parameters': {'page_numbers': sorted(selected), 'region_size': region_size,
                       'region_budget': region_budget, 'minimum_member_length': minimum_member_length,
                       'envelope_engine': envelope_engine}}
    source_phase_dir = ownership_from or (rebind_from if rebind_from is not None else refine_from)
    if source_phase_dir is not None:
        upstream_context = json.loads((source_phase_dir / 'run-context.json').read_text())
        if upstream_context['inputs_sha256'] != run_context['inputs_sha256']:
            raise ValueError('refinement inputs differ from frozen discovery inputs')
        run_context['refinement_input'] = {'run_context_sha256': _sha256(upstream_context),
            'page_target_sha256': {str(n): _file_sha256(source_phase_dir / f'page-{n:03d}.automatic-targets.json')
                                   for n in selected}}
    if rebind_from is not None:
        from src.drawing_engine.disciplines.mep.mep_native_cap_replay import load_native_cap_manifest
        cap_terminology = json.loads((rebind_from / 'terminology-proposals.json').read_text())
        cap_unresolved = json.loads((rebind_from / 'automatic-discovery.json').read_text())['unresolved_proposals']
        cap_manifest = load_native_cap_manifest(native_cap_replay_dir, registry=registry,
            terminology=cap_terminology, upstream_context=upstream_context)
        if cap_manifest.get('authority') != {'negative_cap_witness_only': True,
                'positive_acceptance_authority': False, 'quantity_eligible': False}:
            raise ValueError('native cap replay may only reject unsupported envelope witnesses')
        run_context['attribute_rebind'] = {
            'native_cap_manifest_sha256': _sha256(cap_manifest),
            'source_phase_context': upstream_context,
            'source_geometry_and_boundary_reused': True,
            'full_page_candidate_search_reexecuted': False,
            'binding_result_rows_reused': False}
    ownership_queries = None
    if ownership_from is not None:
        from src.drawing_engine.pipelines.generate_mep_stroke_ownership import read as read_ownership
        ownership_capture = read_ownership(stroke_ownership_queries)
        if (ownership_capture['document'] != registry['document']
                or ownership_capture['pages_requested'] != sorted(selected)):
            raise ValueError('stroke ownership capture differs from source/page scope')
        captured_pages = [page['page_number'] for page in ownership_capture['pages']]
        captured_refs = [scope['observation']['id'] for page in ownership_capture['pages']
                         for scope in page['scope_candidates']]
        if (sorted(captured_pages) != sorted(selected)
                or len(captured_refs) != len(set(captured_refs))):
            raise ValueError('stroke ownership capture has missing or duplicate scopes')
        ownership_queries = {scope['observation']['id']: query for page in ownership_capture['pages']
                             for scope, query in zip(page['scope_candidates'], page['queries'], strict=True)}
        run_context['stroke_ownership_rebind'] = {'capture_file_sha256': _file_sha256(stroke_ownership_queries),
            'capture_payload_sha256': _sha256(ownership_capture), 'source_phase_context': upstream_context,
            'binding_result_rows_reused': False, 'full_page_candidate_search_reexecuted': False}
    context_path = output_dir / 'run-context.json'
    if context_path.exists():
        if not resume:
            raise ValueError('output run already exists; use resume or a new output directory')
        if json.loads(context_path.read_text()) != run_context:
            raise ValueError('checkpoint inputs or implementation differ; use a new output directory')
    else:
        if any(output_dir.glob('*.json')):
            raise ValueError('existing output has no verified run context; use a new output directory')
        _write(context_path, run_context)
    page_records, page_graphs, witnesses, leaders, all_composites, connections = [], {}, [], [], [], []
    cache_store = (PackedProjectStore(v3_page_cache_database,
                   create=not Path(v3_page_cache_database).exists())
                   if v3_page_cache_database is not None else None)
    cache = MepV3PageCache(cache_store) if cache_store is not None else None
    compact_downstream_connections = (
        cache is not None and (not export_json or export_json_scope == 'audit')
        and v3_shadow_database is None
        and source_phase_dir is None)

    def consume_cached_page(cache_key_value, record):
        found = set()
        for kind, payload in cache.iter_records(cache_key_value, kinds={
                'page_record', 'route_graph', 'composite_candidate', 'envelope_search',
                'leader_observation', 'boundary_connection'}):
            if kind == 'page_record':
                record.update(payload)
                found.add(kind)
            elif kind == 'route_graph':
                page_graphs[record['page_ref']] = next(
                    row for row in payload['pages'] if row['page_ref'] == record['page_ref'])
                found.add(kind)
            elif kind == 'composite_candidate':
                all_composites.append(payload)
            elif kind == 'envelope_search':
                witnesses.append(payload)
            elif kind == 'leader_observation':
                leaders.append(payload)
            elif kind == 'boundary_connection':
                connections.append({key: payload[key] for key in
                    ('id', 'state', 'composite_refs', 'relation_type')}
                    if compact_downstream_connections else payload)
        if not {'page_record', 'route_graph'}.issubset(found):
            raise ValueError('v3 page cache omits required downstream page records')

    pdf = None
    try:
        for scope in registry['pages']:
            number, ref = scope['page_number'], scope['page_ref']
            record = {'page_ref': ref, 'page_number': number, 'item_inventory_complete': False,
                      'discovery_state': 'not_processed', 'quantity_eligible': False}
            page_records.append(record)
            if number not in selected:
                continue
            target_path = output_dir / f'page-{number:03d}.automatic-targets.json'
            checkpoint_path = output_dir / f'page-{number:03d}.checkpoint.json'
            page_observations = [row for row in observations if row['page_ref'] == ref]
            extraction_hash = _sha256({
                'fitz_version': fitz.VersionBind,
                'implementation': {key: value for key, value in implementation.items()
                                   if key.endswith(('mep_native_target_discovery.py', 'vector_topology.py',
                                                    'mep_native_descriptor_pack.py'))},
            })
            ruleset_hash = _sha256({
                'implementation': {key: value for key, value in implementation.items()
                                   if key.startswith('src/') and not key.endswith(
                                       ('mep_native_target_discovery.py', 'vector_topology.py',
                                        'mep_native_descriptor_pack.py'))},
                'envelope_engine': envelope_engine,
            })
            configuration_hash = _sha256({
                'region_size': region_size, 'region_budget': region_budget,
                'minimum_region_size': min(8, region_size),
                'minimum_member_length': minimum_member_length,
                'page_scope_sha256': _sha256(scope),
                'page_observations_sha256': _sha256(page_observations),
                'source_phase_sha256': (_sha256(run_context['refinement_input'])
                                        if source_phase_dir is not None else None),
            })
            cache_key_value, cache_inputs = page_cache_key(
                source_pdf_sha256=registry['document']['source_pdf_sha256'],
                page_identity={'page_ref': ref, 'page_number': number,
                               'page_size_display': scope['page_size_display']},
                extraction_implementation_sha256=extraction_hash,
                configuration_sha256=configuration_hash, ruleset_sha256=ruleset_hash)
            if cache is not None and cache.manifest(cache_key_value) is not None:
                cache_read_started = perf_counter()
                consume_cached_page(cache_key_value, record)
                if export_json and export_json_scope == 'all':
                    local_export = cache.export_legacy_page(cache_key_value)
                    cached_record, cached_regions, _ = cache.page_evidence(cache_key_value)
                    diagnostic_record = {**cached_record, 'regions': cached_regions}
                    _write(target_path, local_export)
                    _write(checkpoint_path, {'run_context_sha256': _sha256(run_context),
                        'record': diagnostic_record, 'record_sha256': _sha256(diagnostic_record),
                        'targets_sha256': _file_sha256(target_path)})
                    del local_export, cached_regions
                print(json.dumps({'page': number, 'phase': 'verified_v3_page_cache_reused',
                                  'page_cache_key': cache_key_value,
                                  'seconds': round(perf_counter() - cache_read_started, 6)}), flush=True)
                continue
            if page_cache_seed_dir is not None:
                target_path_seed = page_cache_seed_dir / f'page-{number:03d}.automatic-targets.json'
                checkpoint_path_seed = page_cache_seed_dir / f'page-{number:03d}.checkpoint.json'
                seed_context = json.loads((page_cache_seed_dir / 'run-context.json').read_text())
                checkpoint_seed = json.loads(checkpoint_path_seed.read_text())
                parity = json.loads(page_cache_parity_report.read_text())
                parity_row = next((row for row in parity.get('pages', [])
                                   if row.get('page_number') == number), None)
                exact_fields = ('canonical_payload_exact', 'completeness_witnesses_exact',
                    'outcomes_exact', 'quantity_authority_exact', 'relation_certificates_exact',
                    'stable_ids_and_native_references_exact')
                if (not parity.get('passed') or parity_row is None
                        or not all(parity_row.get(field) is True for field in exact_fields)
                        or parity_row.get('candidate_sha256') != _file_sha256(target_path_seed)
                        or checkpoint_seed.get('targets_sha256') != _file_sha256(target_path_seed)
                        or checkpoint_seed.get('record_sha256') != _sha256(checkpoint_seed.get('record'))
                        or seed_context.get('inputs_sha256') != run_context['inputs_sha256']):
                    raise ValueError('frozen page-cache seed lacks exact input-bound semantic parity')
                local_seed = json.loads(target_path_seed.read_text())
                seed_record = checkpoint_seed['record']
                seed_regions = seed_record.pop('regions')
                seed_record.update(region_count=len(seed_regions), page_cache_key=cache_key_value)
                route_graph = local_seed.pop('route_graph')
                page_composites = local_seed.pop('composites')
                page_witnesses = local_seed.pop('envelope_searches')
                page_leaders = local_seed.pop('leader_observations')
                page_connections = local_seed.pop('boundary_connections')
                if local_seed:
                    raise ValueError('frozen page-cache seed has unknown target fields')
                cache_result = cache.write(cache_key=cache_key_value, cache_inputs=cache_inputs,
                    page_ref=ref, records=page_component_record_stream(
                        page_record=seed_record, regions=seed_regions, route_graph=route_graph,
                        composites=page_composites, envelope_searches=page_witnesses,
                        leader_observations=page_leaders, boundary_connections=page_connections))
                consume_cached_page(cache_key_value, record)
                print(json.dumps({'page': number, 'phase': 'exact_parity_page_cache_seeded',
                    'page_cache_key': cache_key_value,
                    'artifact_sha256': cache_result['artifact_sha256']}), flush=True)
                continue
            if resume and checkpoint_path.exists() and cache is None:
                checkpoint = json.loads(checkpoint_path.read_text())
                if (checkpoint['run_context_sha256'] != _sha256(run_context)
                        or checkpoint['record_sha256'] != _sha256(checkpoint['record'])
                        or checkpoint['record']['page_ref'] != ref
                        or checkpoint['record']['page_number'] != number
                        or not target_path.exists()
                        or checkpoint['targets_sha256'] != _file_sha256(target_path)):
                    raise ValueError('page checkpoint evidence or execution scope differs')
                local = json.loads(target_path.read_text())
                record.update(checkpoint['record'])
                page_graphs[ref] = next(row for row in local['route_graph']['pages'] if row['page_ref'] == ref)
                witnesses.extend(local['envelope_searches'])
                leaders.extend(local['leader_observations'])
                all_composites.extend(local['composites']['candidates'])
                connections.extend(local['boundary_connections'])
                print(json.dumps({'page': number, 'phase': 'verified_checkpoint_reused'}), flush=True)
                del local
                continue
            if pdf is None:
                pdf = fitz.open(source)
            started = perf_counter()
            regions = []
            performance_profiles = {ref: {}}

            if rebind_from is not None or ownership_from is not None:
                from src.drawing_engine.disciplines.mep.mep_native_cap_replay import replay_native_cap_page
                checkpoint = json.loads((source_phase_dir / f'page-{number:03d}.checkpoint.json').read_text())
                if (checkpoint['run_context_sha256'] != _sha256(upstream_context)
                        or checkpoint['targets_sha256'] != run_context['refinement_input']['page_target_sha256'][str(number)]
                        or checkpoint['record_sha256'] != _sha256(checkpoint['record'])
                        or checkpoint['record']['page_ref'] != ref
                        or checkpoint['record']['page_number'] != number):
                    raise ValueError('attribute rebind source checkpoint does not replay')
                local = json.loads((source_phase_dir / f'page-{number:03d}.automatic-targets.json').read_text())
                rejections = replay_native_cap_page(native_cap_replay_dir, cap_manifest,
                    page_ref=ref, registry=registry, terminology=cap_terminology,
                    unresolved_proposals=cap_unresolved, page_leaders=local['leader_observations'],
                    upstream_witnesses=local['envelope_searches']) if rebind_from is not None else {}
                if ownership_from is not None:
                    from src.drawing_engine.pipelines.generate_mep_stroke_ownership import contact_scopes
                    expected = contact_scopes(local, {r['id']: r for r in terminology['source_observations']
                                              if r['record_type'] == 'mep_native_text_line'})
                    captured = next((p for p in ownership_capture['pages'] if p['page_number'] == number), None)
                    if captured is None or captured['scope_candidates'] != expected:
                        raise ValueError('stroke ownership capture omits or changes automatic contact scopes')
                for witness in local['envelope_searches']:
                    rejection = rejections.get(witness['id'])
                    if rejection is not None:
                        if (rejection['input_witness_sha256'] != _sha256(witness)
                                or rejection['state'] != 'rejected_no_native_cap_path'
                                or witness['state'] != 'unresolved'):
                            raise ValueError('cap rejection does not match its unchanged unresolved witness')
                        witness['source_phase_witness_sha256'] = _sha256(witness)
                        witness['native_cap_rejection_ref'] = rejection['id']
                        witness['state'] = 'rejected_no_native_cap_path'
                regions = checkpoint['record']['regions']
                source_count = checkpoint['record']['native_source_count']
                index = None
                print(json.dumps({'page': number, 'phase': 'native_cap_evidence_replayed' if rebind_from else 'stroke_ownership_geometry_verified',
                                  'rejected_competing_envelopes': len(rejections)}), flush=True)
            elif refine_from is not None:
                from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
                from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import discover_native_boundary_connections
                from src.drawing_engine.disciplines.mep.mep_native_bend_connections import discover_native_bends
                from src.drawing_engine.disciplines.mep.mep_native_branch_connections import discover_native_branches
                checkpoint = json.loads((refine_from / f'page-{number:03d}.checkpoint.json').read_text())
                if (checkpoint['run_context_sha256'] != _sha256(upstream_context)
                        or checkpoint['targets_sha256'] != run_context['refinement_input']['page_target_sha256'][str(number)]
                        or checkpoint['record_sha256'] != _sha256(checkpoint['record'])
                        or checkpoint['record']['page_ref'] != ref):
                    raise ValueError('frozen page discovery checkpoint does not replay')
                local = json.loads((refine_from / f'page-{number:03d}.automatic-targets.json').read_text())
                regions = checkpoint['record']['regions']
                source_count = checkpoint['record']['native_source_count']
                index = NativeBoundaryQueries(pdf[number - 1], ref)
                arguments = dict(index=index, composites=local['composites'], graph=local['route_graph'], page_ref=ref)
                local['boundary_connections'] = [*discover_native_boundary_connections(**arguments),
                                                  *discover_native_bends(**arguments),
                                                  *discover_native_branches(**arguments)]
                incident = defaultdict(list)
                for connection in local['boundary_connections']:
                    for port in connection['port_refs']:
                        incident[port].append(connection)
                for connection in local['boundary_connections']:
                    if any(len(incident[port]) > 1 for port in connection['port_refs']):
                        connection['state'] = 'abstained'
                        connection['reasons'] = sorted(set(connection['reasons'] + ['non_unique_projected_port_destination']))
                print(json.dumps({'page': number, 'phase': 'frozen_envelopes_native_boundary_refined',
                    'accepted_connections': sum(r['state'] == 'accepted' for r in local['boundary_connections'])}), flush=True)
            else:
                scratch = Path(native_scratch_dir) if native_scratch_dir is not None else ROOT / 'tmp' / 'mep-native-index'
                scratch.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=scratch, prefix=f'page-{number:03d}-') as directory:
                    with RegionIndex.from_native_page(
                            page=pdf[number - 1], page_ref=ref,
                            page_size=scope['page_size_display'],
                            region_size_display_points=region_size,
                            max_candidates_per_region=region_budget,
                            minimum_region_size_display_points=min(8, region_size),
                            sqlite_path=Path(directory) / 'sources.sqlite',
                            performance_profile=performance_profiles[ref],
                            observations=[row for row in observations if row['page_ref'] == ref],
                            minimum_member_length=minimum_member_length) as index:
                        regions = index.regions
                        source_count = performance_profiles[ref]['native_descriptor_count']
                        native_profile = performance_profiles[ref]
                        print(json.dumps({'page': number, 'phase': 'compact_native_index_complete',
                            'regions': len(regions),
                            'native_sources': source_count,
                            'sqlite_survivors': native_profile['sqlite_survivor_count'],
                            'excluded_sources': native_profile['excluded_descriptor_count'],
                            'incomplete_regions': sum(not r['relevant_competitor_search_complete'] for r in regions),
                            'seconds': round(perf_counter() - started, 2),
                            'native_descriptor_scan_seconds': round(native_profile['native_descriptor_scan_seconds'], 3),
                            'sqlite_insert_seconds': round(native_profile['sqlite_insert_seconds'], 3),
                            'region_refinement_seconds': round(native_profile['region_refinement_seconds'], 3),
                            'refinement_candidate_decode_count': native_profile['refinement_candidate_decode_count'],
                            'competitor_coverage_sha256': native_profile['competitor_coverage_sha256']}), flush=True)
                        components = build_automatic_page_components(
                            sheet_registry=registry, page_indexes={ref: index}, observations=observations,
                            minimum_member_length=minimum_member_length,
                            performance_profiles=performance_profiles,
                            envelope_engine=envelope_engine)
                        route_graph, page_composites, page_witnesses, page_leaders, page_connections = components
            if source_phase_dir is not None:
                route_graph, page_composites = local['route_graph'], local['composites']
                page_witnesses, page_leaders = local['envelope_searches'], local['leader_observations']
                page_connections = local['boundary_connections']
            page_elapsed_seconds = round(perf_counter() - started, 3)
            record.update(discovery_state='bounded_native_target_search',
                          region_count=len(regions), page_cache_key=cache_key_value,
                          native_source_count=source_count,
                          eligible_text_seed_count=sum(row['page_ref'] == ref for row in observations),
                          geometric_proposal_count=len(page_composites['accepted_composites']))
            if rebind_from is not None or ownership_from is not None:
                record.update(discovery_state='frozen_geometry_native_cap_rebind' if rebind_from else 'frozen_geometry_stroke_ownership_rebind',
                    source_phase_record_sha256=checkpoint['record_sha256'],
                    source_phase_elapsed_seconds=checkpoint['record'].get('elapsed_seconds'),
                    full_page_candidate_search_reexecuted=False,
                    native_cap_rejection_refs=sorted(row['id'] for row in rejections.values()))
            if cache is not None:
                cache_started = perf_counter()
                cache_result = cache.write(cache_key=cache_key_value, cache_inputs=cache_inputs,
                    page_ref=ref, records=page_component_record_stream(
                        page_record=deepcopy(record), regions=regions, route_graph=route_graph,
                        composites=page_composites, envelope_searches=page_witnesses,
                        leader_observations=page_leaders, boundary_connections=page_connections,
                        performance_profile=performance_profiles.get(ref)))
                cache_seconds = perf_counter() - cache_started
                consume_cached_page(cache_key_value, record)
                if export_json and export_json_scope == 'all':
                    local_export = cache.export_legacy_page(cache_key_value)
                    cached_record, cached_regions, _ = cache.page_evidence(cache_key_value)
                    diagnostic_record = {**cached_record, 'regions': cached_regions}
                    _write(target_path, local_export)
                    _write(checkpoint_path, {'run_context_sha256': _sha256(run_context),
                        'record': diagnostic_record, 'record_sha256': _sha256(diagnostic_record),
                        'targets_sha256': _file_sha256(target_path)})
                    del local_export, cached_regions
            else:
                serialization_started = perf_counter()
                record['regions'] = regions
                if source_phase_dir is None:
                    local = {'route_graph': route_graph, 'composites': page_composites,
                        'envelope_searches': page_witnesses,
                        'leader_observations': page_leaders,
                        'boundary_connections': page_connections}
                _write(target_path, local)
                page_graphs[ref] = next(row for row in route_graph['pages'] if row['page_ref'] == ref)
                witnesses.extend(page_witnesses)
                leaders.extend(page_leaders)
                all_composites.extend(page_composites['candidates'])
                connections.extend(page_connections)
                _write(checkpoint_path, {'run_context_sha256': _sha256(run_context),
                    'record': {**record, 'regions': regions},
                    'record_sha256': _sha256({**record, 'regions': regions}),
                    'targets_sha256': _file_sha256(target_path)})
            if source_phase_dir is None:
                profile = performance_profiles[ref]
                profile.setdefault('phases', {})['v3_page_transaction' if cache is not None else 'output_serialization'] = {
                    'seconds': cache_seconds if cache is not None else perf_counter() - serialization_started,
                    'calls': 1,
                }
                profile['run_context_sha256'] = _sha256(run_context)
                if cache is not None:
                    profile['page_cache_key'] = cache_key_value
                    profile['page_cache_artifact_sha256'] = cache_result['artifact_sha256']
                    profile['page_cache_bytes'] = cache_result['byte_count']
                else:
                    profile['targets_sha256'] = _file_sha256(target_path)
                    profile['target_bytes'] = target_path.stat().st_size
                _write(output_dir / f'page-{number:03d}.performance.json', profile)
            print(json.dumps({**{key: value for key, value in record.items() if key != 'regions'},
                              'elapsed_seconds': page_elapsed_seconds}), flush=True)
            del index
            if cache is None:
                del local
            gc.collect()
    finally:
        if pdf is not None:
            pdf.close()
        if cache_store is not None:
            cache_store.connection.close()
    # Assemble the per-page M3 graphs without reinterpreting selected geometry.
    downstream_started = perf_counter()
    graph = build_mep_route_graph(sheet_registry=registry, page_inputs={})
    graph['pages'] = [page_graphs.get(row['page_ref'], row) for row in graph['pages']]
    for key in graph['summary']:
        if key != 'page_count':
            graph['summary'][key] = sum(row['summary'][key] for row in graph['pages'])
    from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import build_mep_outlined_route_composites
    composites = build_mep_outlined_route_composites(route_graph=build_mep_route_graph(sheet_registry=registry, page_inputs={}))
    composites['m3_contract_ref']['payload_sha256'] = _sha256(graph)
    composites['candidates'] = all_composites
    composites['accepted_composites'] = [deepcopy(row) for row in all_composites if row['state'] == 'accepted']
    composites['summary'] = {'candidate_count': len(all_composites),
        'accepted_composite_count': len(composites['accepted_composites']),
        'abstained_candidate_count': sum(row['state'] == 'abstained' for row in all_composites),
        'state_counts': {state: sum(row['state'] == state for row in all_composites) for state in ('accepted', 'abstained')}}
    targets = {'route_graph': graph, 'composites': composites, 'envelope_searches': witnesses, 'leader_observations': leaders}
    evidence, unresolved = automatic_binding_evidence(targets=targets, terminology=terminology,
                                                     stroke_ownership_queries=ownership_queries)
    proposal_anchors = {row['anchor_ref'] for row in terminology['proposals']}
    for row in observations:
        if row['id'] not in proposal_anchors:
            unresolved.append({'id': row['id'], 'page_ref': row['page_ref'], 'description': row['text'],
                'reason': 'Unclassified tag; equipment identity and explicit port connectivity are unresolved.',
                'target_refs': [], 'evidence_refs': [row['id']],
                'source_geometry': {'bbox_display': row['bbox_display']}, 'quantity_eligible': False})
    terminology, evidence = apply_geometric_text_applicability(terminology=terminology, binding_evidence=evidence)
    direct_bindings = build_mep_attribute_bindings(terminology_proposals=terminology, route_graph=graph,
        binding_evidence=evidence, outlined_route_composites=composites)
    terminology, evidence = propagate_collinear_applicability(terminology=terminology,
        binding_evidence=evidence, connections=connections, direct_bindings=direct_bindings)
    bindings = build_mep_attribute_bindings(terminology_proposals=terminology, route_graph=graph,
        binding_evidence=evidence, outlined_route_composites=composites)
    unresolved_ids = {row['id'] for row in unresolved}
    proposal_by_id = {row['id']: row for row in terminology['proposals']}
    source_by_id = {row['id']: row for row in terminology['source_observations']}
    for relation in bindings['relations']:
        if relation['state'] != 'abstained' or relation['proposal_ref'] in unresolved_ids:
            continue
        proposal = proposal_by_id[relation['proposal_ref']]
        source_row = source_by_id[proposal['anchor_ref']]
        unresolved.append({'id': relation['id'], 'page_ref': relation['page_ref'], 'description': source_row['text'],
            'reason': 'M4 abstained: ' + '; '.join(relation['reasons']),
            'target_refs': relation['target_refs'], 'evidence_refs': relation['proposal_evidence_refs'],
            'source_geometry': {'bbox_display': source_row['bbox_display']}, 'quantity_eligible': False})
    runs = build_mep_cross_sheet_runs(sheet_registry=registry, route_graph=graph, attribute_bindings=bindings)
    geometry = build_mep_bounded_local_3d_segments(sheet_registry=registry, outlined_route_composites=composites,
        attribute_bindings=bindings, cross_sheet_runs=runs)
    catalog = build_mep_item_catalog(sheet_registry=registry, attribute_bindings=bindings, bounded_local_3d=geometry)
    named = {ref for row in catalog['item_occurrences'] for ref in row['target_refs']}
    for row in composites['accepted_composites']:
        if row['id'] not in named:
            unresolved.append({'id': row['id'], 'page_ref': row['page_ref'],
                'description': 'Unlabelled closed envelope - geometric proposal only',
                'reason': 'No certified system, size or equipment identity; not an accepted item.',
                'target_refs': [], 'evidence_refs': row['member_source_primitive_refs'],
                'source_geometry': {'points_display': row['derived_geometry']['centreline_points_display']},
                'quantity_eligible': False})
    # Group semantic alternatives on the same native line into one audit row;
    # every proposal ID is retained as evidence, never counted as an item.
    grouped = {}
    for row in unresolved:
        key = (row['page_ref'], tuple(row['evidence_refs']), row['description'])
        if key not in grouped:
            grouped[key] = deepcopy(row)
            grouped[key]['evidence_refs'] = sorted({row['id'], *row['evidence_refs']})
        else:
            grouped[key]['evidence_refs'] = sorted({row['id'], *grouped[key]['evidence_refs']})
            grouped[key]['target_refs'] = sorted({*row['target_refs'], *grouped[key]['target_refs']})
            grouped[key]['reason'] = '; '.join(sorted(set(grouped[key]['reason'].split('; ') + row['reason'].split('; '))))
    items_by_target = defaultdict(list)
    for row in catalog['item_occurrences']:
        for ref in row['target_refs']:
            items_by_target[ref].append(row['id'])
    discovery = {'schema_version': '0.1.0', 'layer': 'mep_automatic_item_discovery',
        'document': deepcopy(registry['document']), 'execution_mode': 'automatic_frozen_replay',
        'geometry_selection_basis': 'automatic bounded native envelope and dot-leader search; frozen upstream M1 registry',
        'authority': {'reviewed_selectors_used': False, 'quantity_eligible': False, 'document_completeness_established': False},
        'input_payload_sha256': {key: _sha256(value) for key, value in
                                {'catalog': catalog, 'geometry': geometry, 'composites': composites, 'annotations': annotations}.items()},
        'upstream_input_sha256': {'registry': _sha256(registry), 'native_text': _sha256(native_text),
                                  'ocr': _sha256(ocr) if ocr else None},
        'parameters': {'page_numbers': sorted(selected), 'region_size': region_size, 'region_budget': region_budget,
                       'minimum_member_length': minimum_member_length, 'minimum_region_size': min(8, region_size),
                       'retained_source_storage': 'exact_sqlite_cell_index'},
        'implementation_sha256': implementation,
        'pages': page_records, 'accepted_targets': [
            {'id': row['id'], 'page_ref': row['page_ref'], 'item_occurrence_refs': items_by_target[row['id']]}
            for row in composites['accepted_composites'] if row['id'] in named],
        'unresolved_proposals': list(grouped.values()),
        'summary': {'certified_item_occurrences': len(catalog['item_occurrences']),
                    'identified_item_occurrences': sum(row['system'] is not None and bool(row['typed_dimensions'])
                                                      for row in catalog['item_occurrences']),
                    'accepted_m4_relations': bindings['summary']['accepted_relation_count'],
                    'bounded_local_3d_segments': len(geometry['bounded_local_3d_segments']),
                    'calculated_amounts': 0, 'reviewed_item_recall': None}}
    if implementation != {str(p.relative_to(ROOT)): _file_sha256(p) for p in implementation_paths}:
        raise RuntimeError('automatic implementation changed during run; rerun before publishing certificates')
    payloads = {'route-observations': graph, 'outlined-route-composites': composites,
        'terminology-proposals': terminology, 'attribute-bindings': bindings,
        'cross-sheet-runs': runs, 'bounded-local-3d': geometry, 'item-catalog': catalog,
        'automatic-discovery': discovery,
        'outlined-route-connections': {'schema_version': '0.1.0', 'layer': 'mep_native_boundary_connections',
            'document': deepcopy(registry['document']), 'm3_payload_sha256': _sha256(graph),
            'm35_payload_sha256': _sha256(composites), 'connections': connections, 'quantity_eligible': False}}
    if v3_shadow_database is not None:
        from src.drawing_engine.project.project_v3_direct_writer import shadow_write
        shadow = shadow_write(v3_shadow_database, 'outlined-route-connections',
                              payloads['outlined-route-connections'])
        _write(output_dir / 'outlined-route-connections.v3-shadow.json', {
            'schema_version': '0.1.0', 'layer': 'project_v3_direct_write_shadow', **shadow})
    if export_json:
      names = (payloads if export_json_scope == 'all' else {
          key: payloads[key] for key in ('outlined-route-composites',
              'terminology-proposals', 'attribute-bindings', 'bounded-local-3d',
              'item-catalog', 'automatic-discovery')})
      for name, payload in names.items():
        _write(output_dir / (name + '.json'), payload)
    if ownership_from is not None and export_json:
        # Preserve complete query bodies once; M4 roles link to their exact IDs
        # and hashes rather than embedding repeated native inventories.
        _write(output_dir / 'stroke-ownership-queries.json', ownership_capture)
    rows = flatten_mep_item_catalog(catalog) if export_json_scope == 'all' and export_json else []
    if rows:
        with (output_dir / 'item-catalog.review.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'phase': 'downstream_recomputation_complete',
                      'seconds': round(perf_counter() - downstream_started, 6),
                      'page_count': len(selected)}), flush=True)
    print(json.dumps(discovery['summary']), flush=True)
    return discovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'registry', 'native-text', 'annotations', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--ocr', type=Path)
    parser.add_argument('--pages', type=int, nargs='+')
    parser.add_argument('--region-size', type=float, default=64)
    parser.add_argument('--region-budget', type=int, default=4000)
    parser.add_argument('--resume', action='store_true', help='Reuse only hash-verified completed pages from this exact run')
    parser.add_argument('--refine-from', type=Path, help='Refine verified frozen envelopes with new full-source native boundary queries')
    parser.add_argument('--ownership-from', type=Path, help='Rebind scoped stroke ownership while preserving frozen geometry')
    parser.add_argument('--stroke-ownership-queries', type=Path, help='Complete native query capture for ownership rebind')
    parser.add_argument('--v3-shadow-database', type=Path,
                        help='Direct-write the dominant artifact to a detached v3 store and prove canonical parity')
    parser.add_argument('--v3-page-cache-database', type=Path,
                        help='Content-addressed bounded page cache in a detached schema-v3 store')
    parser.add_argument('--page-cache-seed-dir', type=Path,
                        help='Maintenance-only frozen page JSON source with exact parity proof')
    parser.add_argument('--page-cache-parity-report', type=Path,
                        help='Exact semantic parity proof binding a maintenance cache seed')
    parser.add_argument('--export-json', nargs='?', const='all', choices=('all', 'audit'),
                        help='Explicitly export all diagnostic JSON/CSV, or only bounded audit inputs')
    parser.add_argument('--envelope-engine', choices=('per_candidate', 'page_topology'), default='per_candidate')
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    run(source=args.source, registry=load(args.registry), native_text=load(args.native_text),
        annotations=load(args.annotations), ocr=load(args.ocr) if args.ocr else None, refine_from=args.refine_from,
        ownership_from=args.ownership_from, stroke_ownership_queries=args.stroke_ownership_queries,
        output_dir=args.output_dir, page_numbers=args.pages, region_size=args.region_size, region_budget=args.region_budget,
        resume=args.resume, v3_shadow_database=args.v3_shadow_database,
        v3_page_cache_database=args.v3_page_cache_database,
        page_cache_seed_dir=args.page_cache_seed_dir,
        page_cache_parity_report=args.page_cache_parity_report,
        export_json=args.export_json is not None,
        export_json_scope=args.export_json or 'all',
        envelope_engine=args.envelope_engine)


if __name__ == '__main__':
    main()
