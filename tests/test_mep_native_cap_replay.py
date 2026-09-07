from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import fitz

from tools.generate_mep_native_cap_replay import capture_native_cap_page
from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import RegionIndex, discover_envelopes
from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries
from src.drawing_engine.disciplines.mep.mep_native_cap_replay import (
    select_native_cap_witnesses, build_native_cap_page, replay_native_cap_rejections,
)
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import iter_bounded_native_page_regions
from src.drawing_engine.disciplines.mep.mep_sheet_registry import build_mep_sheet_registry, build_sheet_page_record
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations


def fixture(directory, *, cap=None, clutter=0):
    source = Path(directory) / 'cap.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=160)
        page.draw_line((40, 70), (240, 70), width=.7)
        page.draw_line((40, 74), (240, 74), width=.7)
        if cap == 'line':
            page.draw_line((40, 70), (40, 74), width=.7)
        elif cap == 'curve':
            page.draw_bezier((40, 70), (37, 70), (37, 74), (40, 74), width=.7)
        for _ in range(clutter):
            page.draw_line((40, 70), (39, 70), width=.7)
        pdf.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scope = build_sheet_page_record(page_ref='test.page', page_number=1, page_width=300, page_height=160,
        native_tokens=[], quality={'route': 'native', 'reason': 'synthetic'})
    registry = build_mep_sheet_registry(document={'document_key': 'pdf-sha256:' + digest,
        'source_pdf_sha256': digest, 'source_bytes': source.stat().st_size, 'page_count': 1}, pages=[scope])
    with fitz.open(source) as pdf:
        packets = list(iter_bounded_native_page_regions(pdf[0], 'test.page', region_size_display_points=64,
                                                       max_candidates_per_region=4000))
        with RegionIndex(primitives=[r for p in packets for r in p['primitive_candidates']],
                regions=[p['region'] for p in packets], page_size=[300, 160]) as index:
            _evidence, witnesses = discover_envelopes(index=index, page_scope=scope)
        if len(witnesses) != 1:
            raise AssertionError(witnesses)
        # Simulate the old observation policy, which retained no-path witnesses
        # as unresolved. The replay must ignore the old closure/state receipt.
        witnesses[0]['state'] = 'unresolved'
        terminology = {'document': registry['document'], 'source_observations': [
            {'id': 'source.annotation', 'page_ref': 'test.page', 'page_number': 1}]}
        unresolved = [{'reason': 'unresolved_competing_envelope_at_contact',
                       'target_refs': ['other.envelope'], 'evidence_refs': ['source.annotation']}]
        leaders = [{'observation_ref': 'source.annotation', 'search_complete': True, 'paths': [
            {'search_complete': True, 'contact_point_display': [100, 72]}]}]
        selection = select_native_cap_witnesses(terminology, unresolved, leaders, witnesses)
        inputs = {'terminology': terminology, 'unresolved_proposals': unresolved, 'leader_observations': leaders}
        native = capture_native_cap_page(index=NativeBoundaryQueries(pdf[0], 'test.page'), registry=registry,
            witnesses=witnesses, selection=selection, selection_inputs=inputs, implementation={})
    return registry, witnesses, selection, native


class NativeCapReplayTest(unittest.TestCase):
    def test_package_rebind_recomputes_bindings_and_preserves_frozen_geometry(self):
        from src.drawing_engine.pipelines.generate_mep_automatic_items import run
        from tools.mep_project import build
        from src.drawing_engine.project.project_knowledge_store import ProjectKnowledgeStore
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'binding.pdf'
            with fitz.open() as pdf:
                page = pdf.new_page(width=400, height=300)
                for y in [136, 140, 144]:
                    page.draw_line((70, y), (270, y), width=.7)
                page.draw_line((70, 140), (70, 144), width=.7)
                page.insert_text((160, 83), '2" HHWS', fontsize=10)
                page.draw_line((150, 83), (160, 83), width=.2)
                page.draw_line((100, 140), (150, 83), width=.2)
                page.draw_circle((100, 140), 2, color=(0, 0, 0), fill=(0, 0, 0), width=.2)
                pdf.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            scope = build_sheet_page_record(page_ref='test.page', page_number=1, page_width=400, page_height=300,
                native_tokens=[], quality={'route': 'native', 'reason': 'synthetic'})
            registry = build_mep_sheet_registry(document={'document_key': 'pdf-sha256:' + digest,
                'source_pdf_sha256': digest, 'source_bytes': source.stat().st_size, 'page_count': 1}, pages=[scope])
            native = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
            annotations = {'document': registry['document'], 'annotations': []}
            upstream, output, caps = root / 'legacy', root / 'rebound', root / 'caps'

            def legacy_policy(**kwargs):
                selected, witnesses = discover_envelopes(**kwargs)
                for witness in witnesses:
                    if witness['state'] == 'rejected_no_native_cap_path':
                        witness['state'] = 'unresolved'
                return selected, witnesses

            with patch('src.drawing_engine.disciplines.mep.mep_automatic_target_binding.discover_envelopes', side_effect=legacy_policy):
                run(source=source, registry=registry, native_text=native,
                    output_dir=upstream, annotations=annotations)
            load = lambda path: json.loads(path.read_text())
            prior = load(upstream / 'attribute-bindings.json')
            self.assertEqual(prior['summary']['accepted_relation_count'], 0)
            registry_path = root / 'registry.json'
            registry_path.write_text(json.dumps(registry))
            result = subprocess.run([sys.executable, '-B', 'tools/generate_mep_native_cap_replay.py',
                '--source', str(source), '--registry', str(registry_path), '--run', str(upstream), '--out', str(caps)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = {name: (upstream / name).read_bytes() for name in [
                'route-observations.json', 'outlined-route-composites.json', 'outlined-route-connections.json',
                'page-001.automatic-targets.json', 'attribute-bindings.json']}
            context = load(upstream / 'run-context.json')
            (upstream / 'package-policy-manifest.json').write_text(json.dumps({
                'schema_version': '0.1.0', 'layer': 'mep_package_native_policy',
                'document': registry['document'], 'execution_page_numbers': [1],
                'current_interpretation_context': context,
                'current_interpretation_context_sha256': _sha256(context),
                'pilot_or_prior_results_unioned': False}))
            for name, value in [('native-text', native), ('annotations', annotations), ('ocr', None)]:
                (root / (name + '.json')).write_text(json.dumps(value))
            result = subprocess.run([sys.executable, '-B', 'tools/rebind_mep_package_attributes.py',
                '--source', str(source), '--registry', str(registry_path),
                '--native-text', str(root / 'native-text.json'), '--annotations', str(root / 'annotations.json'),
                '--ocr', str(root / 'ocr.json'), '--upstream', str(upstream),
                '--native-cap-replay', str(caps), '--output-dir', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load(output / 'attribute-bindings.json')['summary']['accepted_relation_count'], 2)
            for name, frozen in original.items():
                self.assertEqual((upstream / name).read_bytes(), frozen)
            for name in ['route-observations.json', 'outlined-route-composites.json', 'outlined-route-connections.json']:
                self.assertEqual(load(output / name), load(upstream / name))
            rebound = load(output / 'page-001.automatic-targets.json')
            self.assertTrue(any(r.get('native_cap_rejection_ref') for r in rebound['envelope_searches']))
            self.assertTrue(all(not r['quantity_eligible'] for r in load(output / 'attribute-bindings.json')['relations']))
            database = root / 'project.sqlite'
            report = build(source=source, registry_path=registry_path, run_dir=output, output_dir=root / 'checkpoint',
                           database=database, project_id='synthetic', document_id='cap-rebind')
            with ProjectKnowledgeStore(database) as store:
                rows = store.query(project_id='synthetic', document_id='cap-rebind', snapshot_id=report['snapshot_id'],
                                   record_type='mep_native_cap_replay_outcome', state='rejected_no_native_cap_path')
                self.assertEqual(len(rows), 2)
                self.assertEqual({r['page_ref'] for r in rows}, {'test.page'})

    def test_fresh_no_path_is_negative_only_and_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, witnesses, selection, native = fixture(directory)
        before = deepcopy(witnesses)
        payload = build_native_cap_page(native_page=native, registry=registry,
            upstream_witnesses=witnesses, selection=selection)
        negatives = replay_native_cap_rejections(payload, native_page=native, registry=registry)
        self.assertEqual(set(negatives), {witnesses[0]['id']})
        result = next(iter(negatives.values()))
        self.assertEqual(result['cap_proof']['cap_result'], 'no_path')
        self.assertEqual(result['input_witness_sha256'], _sha256(witnesses[0]))
        self.assertFalse(result['positive_acceptance_authority'])
        self.assertFalse(result['physical_item_identity_established'])
        self.assertFalse(result['quantity_eligible'])
        self.assertEqual(witnesses, before)

    def test_found_straight_or_curved_cap_retains_ambiguity(self):
        for cap in ['line', 'curve']:
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                registry, witnesses, selection, native = fixture(directory, cap=cap)
                payload = build_native_cap_page(native_page=native, registry=registry,
                    upstream_witnesses=witnesses, selection=selection)
                self.assertEqual(payload['outcomes'][0]['cap_proof']['cap_result'], 'found_path')
                self.assertEqual(replay_native_cap_rejections(payload, native_page=native, registry=registry), {})
                self.assertTrue(payload['outcomes'][0]['cap_proof']['closure_source_primitive_refs'])

    def test_incomplete_query_and_reachable_budget_never_reject(self):
        for limited in ['query', 'budget']:
            with self.subTest(limited=limited), tempfile.TemporaryDirectory() as directory:
                registry, witnesses, selection, native = fixture(directory, clutter=1201 if limited == 'budget' else 0)
                if limited == 'query':
                    next(iter(native['queries'].values()))['complete'] = False
                payload = build_native_cap_page(native_page=native, registry=registry,
                    upstream_witnesses=witnesses, selection=selection)
                self.assertEqual(payload['outcomes'][0]['state'], 'unresolved')
                expected = 'incomplete_query' if limited == 'query' else 'reachable_source_budget_exceeded'
                self.assertEqual(payload['outcomes'][0]['cap_proof']['cap_result'], expected)
                self.assertEqual(replay_native_cap_rejections(payload, native_page=native, registry=registry), {})

    def test_replay_rejects_authority_witness_query_and_geometry_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, witnesses, selection, native = fixture(directory)
        payload = build_native_cap_page(native_page=native, registry=registry,
            upstream_witnesses=witnesses, selection=selection)
        changed = deepcopy(payload)
        changed['outcomes'][0]['quantity_eligible'] = True
        with self.assertRaises(ValueError):
            replay_native_cap_rejections(changed, native_page=native, registry=registry)
        changed = deepcopy(witnesses)
        changed[0]['cap_candidate_source_count'] += 1
        with self.assertRaises(ValueError):
            replay_native_cap_rejections(payload, native_page=native, registry=registry,
                                          upstream_witnesses=changed, selection=selection)
        for field in ['query', 'geometry']:
            changed = deepcopy(native)
            if field == 'query':
                changed['queries'].pop(next(iter(changed['queries'])))
            else:
                changed['source_rows'][0]['points_display'][0][0] += 1
            with self.assertRaises(ValueError):
                replay_native_cap_rejections(payload, native_page=changed, registry=registry)

    def test_selection_preserves_every_competitor_and_excludes_incomplete_leaders(self):
        with tempfile.TemporaryDirectory() as directory:
            _, witnesses, _, native = fixture(directory)
        inputs = native['selection_inputs']
        other = deepcopy(witnesses[0])
        other['id'] = 'another.competing.witness'
        selected = select_native_cap_witnesses(inputs['terminology'], inputs['unresolved_proposals'],
                                               inputs['leader_observations'], [*witnesses, other])
        self.assertEqual(len(selected), 2)
        inputs['leader_observations'][0]['paths'][0]['search_complete'] = False
        self.assertEqual(select_native_cap_witnesses(inputs['terminology'], inputs['unresolved_proposals'],
                                                    inputs['leader_observations'], witnesses), [])

    def test_real_package_native_cap_negatives_replay_with_all_competitors(self):
        root = Path(__file__).resolve().parents[1] / 'fixtures/mep/m_and_p_coordination'
        registry = json.loads((root / 'm_and_p_coordination.sheet-registry.json').read_text())
        native = json.loads(gzip.decompress((root / 'native-cap-replay/page-003.native-cap-queries.json.gz').read_bytes()))
        payload = json.loads((root / 'native-cap-replay/page-003.native-cap-outcomes.json').read_text())
        negatives = replay_native_cap_rejections(payload, native_page=native, registry=registry)
        self.assertEqual(len(payload['outcomes']), 60)
        self.assertEqual(len(negatives), 4)
        self.assertEqual(sum(r['state'] == 'unresolved' for r in payload['outcomes']), 56)
        prefix = 'mep_automatic_envelope_search.'
        for suffix, source_count, reachable_count in [('c342c0882eefec239766', 519, 24),
                ('3fa91a5e443eb4ff2b87', 2967, 477), ('a94a3d0c5181b4ade1cf', 2382, 416)]:
            proof = negatives[prefix + suffix]['cap_proof']
            self.assertEqual(proof['input_source_count'], source_count)
            self.assertEqual(proof['reachable_source_count'], reachable_count)
            self.assertEqual(proof['cap_result'], 'no_path')
            self.assertTrue(proof['full_source_endpoint_clustering_preserved'])
        # Evaluation IDs occur only after capture, never in candidate discovery.
        with self.assertRaises(ValueError):
            changed = deepcopy(payload)
            next(r for r in changed['outcomes'] if r['state'] == 'unresolved')['state'] = 'rejected_no_native_cap_path'
            replay_native_cap_rejections(changed, native_page=native, registry=registry)


if __name__ == '__main__':
    unittest.main()
