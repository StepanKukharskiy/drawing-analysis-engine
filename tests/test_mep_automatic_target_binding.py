from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import RegionIndex, build_automatic_targets, automatic_binding_evidence, apply_geometric_text_applicability, _reachable_cap_fragments
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import _shortest_cap_path
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import iter_bounded_native_page_regions
from src.drawing_engine.disciplines.mep.mep_sheet_registry import build_mep_sheet_registry, build_sheet_page_record
from src.drawing_engine.disciplines.mep.mep_text_observations import extract_mep_text_observations, build_mep_document_text_proposals
from src.drawing_engine.disciplines.mep.mep_attribute_binding import build_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_item_catalog import build_mep_item_catalog
from src.drawing_engine.pipelines.generate_mep_automatic_items import run
from src.drawing_engine.audit.render_mep_partial_audit import build_automatic_manifest
from src.drawing_engine.disciplines.mep.mep_v3_page_cache import MepV3PageCache
from src.drawing_engine.project.project_packed_store import PackedProjectStore


def automatic_fixture(directory, *, ambiguous=False, budget=4000, rotation=0, labelled=True, short_crossing=False, curved_crossing=False,
                      cap_clutter=None, label_text='2" HHWS', disconnected_leader=False,
                      unrelated_text=False):
    source = Path(directory) / 'neutral.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=300)
        page.draw_line((70, 140), (270, 140), width=.7)
        page.draw_line((70, 144), (270, 144), width=.7)
        page.draw_line((70, 140), (70, 144), width=.7)
        if cap_clutter:
            for _ in range(1300):
                page.draw_line((70, 140) if cap_clutter == 'connected' else (66, 138),
                               (69, 140) if cap_clutter == 'connected' else (66.1, 138), width=.7)
        if ambiguous:
            page.draw_line((70, 136), (270, 136), width=.7)
            page.draw_line((70, 136), (70, 140), width=.7)
        if short_crossing:
            page.draw_line((98, 130), (98, 150), width=.7)
            page.draw_line((102, 130), (102, 150), width=.7)
            page.draw_line((98, 150), (102, 150), width=.7)
        if curved_crossing:
            page.draw_bezier((90, 130), (100, 144), (100, 136), (110, 150), width=.7)
        if labelled:
            page.insert_text((160, 83), label_text, fontsize=10)
            page.draw_line((150, 83), (160, 83), width=.2)
            page.draw_line((100, 140), (150, 82.8 if disconnected_leader else 83), width=.2)
            page.draw_circle((100, 140), 2, color=(0, 0, 0), fill=(0, 0, 0), width=.2)
        if unrelated_text:
            page.insert_text((160, 110), '3" CHWS', fontsize=10)
        page.set_rotation(rotation)
        size = list(page.rect.br)
        pdf.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    scope = build_sheet_page_record(page_ref='page.1', page_number=1,
        page_width=size[0], page_height=size[1], native_tokens=[], quality={'route': 'native', 'reason': 'synthetic'})
    registry = build_mep_sheet_registry(document={'document_key': 'pdf-sha256:' + digest,
        'source_pdf_sha256': digest, 'source_bytes': source.stat().st_size, 'page_count': 1}, pages=[scope])
    native = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
    terminology = build_mep_document_text_proposals(native)
    with fitz.open(source) as pdf:
        packets = list(iter_bounded_native_page_regions(pdf[0], 'page.1', region_size_display_points=64,
                                                       max_candidates_per_region=budget))
    index = RegionIndex(primitives=[row for packet in packets for row in packet['primitive_candidates']],
        regions=[packet['region'] for packet in packets], page_size=size)
    targets = build_automatic_targets(sheet_registry=registry, page_indexes={'page.1': index},
                                      observations=terminology['source_observations'])
    evidence, unresolved = automatic_binding_evidence(targets=targets, terminology=terminology)
    terminology, evidence = apply_geometric_text_applicability(terminology=terminology, binding_evidence=evidence)
    bindings = build_mep_attribute_bindings(terminology_proposals=terminology,
        route_graph=targets['route_graph'], outlined_route_composites=targets['composites'], binding_evidence=evidence)
    catalog = build_mep_item_catalog(sheet_registry=registry, attribute_bindings=bindings)
    return registry, terminology, targets, evidence, unresolved, bindings, catalog, index


class MepAutomaticTargetBindingTest(unittest.TestCase):
    def test_disconnected_leader_and_nearby_unrelated_text_do_not_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, evidence, unresolved, bindings, _, _ = automatic_fixture(
                directory, disconnected_leader=True)
            self.assertFalse(evidence)
            self.assertEqual(0, bindings['summary']['accepted_relation_count'])
            self.assertTrue(unresolved)
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, unresolved, bindings, _, _ = automatic_fixture(directory, unrelated_text=True)
            systems = [r for r in bindings['relations']
                       if r['state'] == 'accepted' and r['relation_type'] == 'route_system']
            self.assertEqual(['heating_hot_water_supply'], [r['candidate']['kind'] for r in systems])
            self.assertTrue(any('CHWS' in r['description'] for r in unresolved))

    def test_route_ink_cannot_also_certify_the_annotation_leader(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, _, targets, *_ = automatic_fixture(directory)
            terminology = build_mep_document_text_proposals(extract_mep_text_observations(
                pdf_path=Path(directory) / 'neutral.pdf', sheet_registry=registry))
            path = targets['leader_observations'][0]['paths'][0]
            path['source_primitive_refs'].append(
                targets['composites']['accepted_composites'][0]['member_source_primitive_refs'][0])
            evidence, unresolved = automatic_binding_evidence(targets=targets, terminology=terminology)
            self.assertFalse(evidence)
            self.assertTrue(all('annotation_route_dual_use' in r['reason'] for r in unresolved))

    def test_v3_page_cache_has_exact_page_parity_no_retained_json_and_hash_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, *_ = automatic_fixture(directory)
            source = Path(directory) / 'neutral.pdf'
            native = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
            annotations = {'document': registry['document'], 'annotations': []}
            legacy_output = Path(directory) / 'legacy'
            cached_output = Path(directory) / 'cached'
            resumed_output = Path(directory) / 'resumed'
            database = Path(directory) / 'page-cache.sqlite'
            legacy = run(source=source, registry=registry, native_text=native,
                annotations=annotations, output_dir=legacy_output, export_json=True,
                envelope_engine='page_topology')
            cached = run(source=source, registry=registry, native_text=native,
                annotations=annotations, output_dir=cached_output, export_json=False,
                v3_page_cache_database=database, envelope_engine='page_topology')
            cache_key = cached['pages'][0]['page_cache_key']
            context = json.loads((cached_output / 'run-context.json').read_text())
            for name in ('mep_native_descriptor_pack.py', 'mep_packed_spatial_index.py'):
                relative = 'src/drawing_engine/disciplines/mep/' + name
                module = Path(__file__).resolve().parents[1] / relative
                self.assertEqual(context['implementation_sha256'][relative],
                                 hashlib.sha256(module.read_bytes()).hexdigest())
            with PackedProjectStore(database) as store:
                cache = MepV3PageCache(store)
                self.assertEqual(cache.export_legacy_page(cache_key),
                    json.loads((legacy_output / 'page-001.automatic-targets.json').read_text()))
                self.assertEqual(store.connection.execute('pragma quick_check').fetchone()[0], 'ok')
            self.assertEqual(cached['summary'], legacy['summary'])
            self.assertFalse(any(cached_output.glob('page-*.automatic-targets.json')))
            self.assertFalse((cached_output / 'automatic-discovery.json').exists())
            with patch('src.drawing_engine.pipelines.generate_mep_automatic_items.RegionIndex.from_native_page',
                       side_effect=AssertionError('completed page was rescanned')):
                resumed = run(source=source, registry=registry, native_text=native,
                    annotations=annotations, output_dir=resumed_output, export_json=False,
                    v3_page_cache_database=database, envelope_engine='page_topology')
            self.assertEqual(resumed, cached)
            audit_output = Path(directory) / 'audit-export'
            with patch('src.drawing_engine.pipelines.generate_mep_automatic_items.RegionIndex.from_native_page',
                       side_effect=AssertionError('audit export rescanned a cached page')):
                audit = run(source=source, registry=registry, native_text=native,
                    annotations=annotations, output_dir=audit_output, export_json=True,
                    export_json_scope='audit', v3_page_cache_database=database,
                    envelope_engine='page_topology')
            self.assertEqual(audit, cached)
            self.assertFalse(any(audit_output.glob('page-*.automatic-targets.json')))
            self.assertEqual({path.stem for path in audit_output.glob('*.json')
                              if path.name != 'run-context.json'}, {
                'outlined-route-composites', 'terminology-proposals',
                'attribute-bindings', 'bounded-local-3d', 'item-catalog',
                'automatic-discovery'})

    def test_shared_page_topology_is_exact_legacy_parity(self):
        for kwargs in ({}, {'ambiguous': True}, {'short_crossing': True}, {'curved_crossing': True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                registry, terminology, legacy, *_, index = automatic_fixture(directory, **kwargs)
                with RegionIndex(primitives=list(index.primitives.values()), regions=index.regions,
                        page_size=index.page_size, sqlite_path=Path(directory) / 'sources.sqlite') as disk:
                    current = build_automatic_targets(sheet_registry=registry, page_indexes={'page.1': disk},
                        observations=terminology['source_observations'], envelope_engine='page_topology')
                for key in ('route_graph', 'composites', 'envelope_searches', 'boundary_connections'):
                    self.assertEqual(current[key], legacy[key])

    def test_cap_reachability_preserves_all_bounded_paths_and_endpoint_assignments(self):
        def fragment(identifier, start, end, length):
            return {'id': identifier, 'source_primitive_ref': identifier,
                    'endpoint_vertex_refs': [start, end],
                    'geometry': {'path_display_points': length},
                    'style': {'width_display_points': .7, 'dash_pattern': None},
                    'source_kind': 'native_pdf_vector',
                    'provenance': {'method': 'native_pdf_vector_topology', 'coordinate_space': 'page'},
                    'ownership': {'page_ref': 'page.1', 'view_scope_ref': None, 'package_ref': None}}
        rows = [fragment('left', 'a0', 'a1', 100), fragment('right', 'b0', 'b1', 100),
                fragment('start_cap_1', 'a0', 'u', 1), fragment('start_cap_2', 'u', 'b0', 1),
                fragment('alternative_1', 'a0', 'v', 1.5), fragment('alternative_2', 'v', 'b0', 1.5),
                fragment('end_cap', 'a1', 'b1', 2), fragment('exact_bound_branch', 'a0', 'z', 5),
                fragment('out_of_bound', 'a0', 'far', 5.1), fragment('disconnected', 'c', 'd', 1)]
        before = deepcopy(rows)
        metrics = {'right_reversed': False, 'mean_separation_display_points': 2,
                   'member_width_display_points': .7}
        reduced = _reachable_cap_fragments(rows, rows[0], rows[1], 5)
        self.assertEqual(rows, before)
        self.assertEqual({r['id'] for r in reduced}, {r['id'] for r in rows[:-2]})
        self.assertTrue(all(row is original for row in reduced for original in rows if row['id'] == original['id']))
        self.assertEqual(_shortest_cap_path({'fragments': rows}, rows[0], rows[1], metrics),
                         _shortest_cap_path({'fragments': reduced}, rows[0], rows[1], metrics))
        incompatible = deepcopy(rows)
        incompatible[2]['style']['dash_pattern'] = 'different'
        self.assertNotIn('start_cap_1', {row['id'] for row in _reachable_cap_fragments(incompatible, incompatible[0], incompatible[1], 5)})

    def test_disconnected_cap_clutter_does_not_consume_reachable_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, evidence, _, _, catalog, _ = automatic_fixture(directory, cap_clutter='disconnected')
            witness = next(row for row in targets['envelope_searches'] if row['state'] == 'closed_geometric_proposal')
            self.assertGreater(witness['cap_candidate_source_count'], 1200)
            self.assertLess(witness['cap_source_reachability']['reachable_source_count'], 1200)
            self.assertTrue(witness['cap_search_complete'])
            self.assertTrue(witness['cap_source_reachability']['full_source_endpoint_clustering_preserved'])
            self.assertEqual(len(evidence), 2)
            self.assertEqual(len(catalog['item_occurrences']), 1)

    def test_reachable_cap_alternatives_still_exhaust_the_unchanged_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, evidence, _, _, catalog, _ = automatic_fixture(directory, cap_clutter='connected')
            witness = targets['envelope_searches'][0]
            self.assertGreater(witness['cap_source_reachability']['reachable_source_count'], 1200)
            self.assertFalse(witness['cap_search_complete'])
            self.assertEqual(witness['cap_source_reachability']['reachable_source_budget'], 1200)
            self.assertEqual(evidence, [])
            self.assertEqual(catalog['item_occurrences'], [])

    def test_native_leader_closes_item_without_reviewed_selectors(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, evidence, unresolved, bindings, catalog, _ = automatic_fixture(directory)
            self.assertEqual(len(targets['composites']['accepted_composites']), 1)
            self.assertEqual(len(evidence), 2)
            self.assertEqual(unresolved, [])
            self.assertEqual(bindings['summary']['accepted_relation_count'], 2)
            self.assertEqual(len(catalog['item_occurrences']), 1)
            item = catalog['item_occurrences'][0]
            self.assertEqual(item['system']['kind'], 'heating_hot_water_supply')
            self.assertFalse(item['quantity_eligible'])
            self.assertIsNone(catalog['takeoff_lines'][0]['value_channels']['calculated']['value'])

    def test_competing_pair_and_truncation_do_not_manufacture_uniqueness(self):
        for kwargs in ({'ambiguous': True}, {'budget': 1}, {'short_crossing': True}, {'curved_crossing': True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                _, _, _, evidence, unresolved, bindings, catalog, _ = automatic_fixture(directory, **kwargs)
                self.assertEqual(evidence, [])
                self.assertTrue(unresolved)
                self.assertEqual(bindings['summary']['accepted_relation_count'], 0)
                self.assertEqual(catalog['item_occurrences'], [])

    def test_unlabelled_envelope_stays_geometric_proposal_not_item(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, evidence, _, _, catalog, _ = automatic_fixture(directory, labelled=False)
            self.assertEqual(len(targets['composites']['accepted_composites']), 1)
            self.assertTrue(targets['envelope_searches'])
            self.assertEqual(evidence, [])
            self.assertEqual(catalog['item_occurrences'], [])

    def test_region_boundary_crossings_keep_full_native_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, targets, _, _, _, _, index = automatic_fixture(directory)
            row = targets['envelope_searches'][0]
            self.assertTrue(row['competitor_search_complete'])
            self.assertGreater(len(row['region_refs']), 1)
            self.assertTrue(any(r['cross_boundary_primitive_refs'] for r in index.regions))
            for page in targets['route_graph']['pages']:
                for fragment in page['fragments']:
                    native = index.primitives[fragment['source_primitive_ref']]
                    self.assertEqual(fragment['geometry']['points_display'],
                                     [[round(v, 6) for v in p] for p in native['points_display']])

    def test_ocr_overlap_remains_unresolved_even_with_closed_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, terminology, targets, _, _, _, _, _ = automatic_fixture(directory)
            terminology = deepcopy(terminology)
            for row in terminology['source_observations']:
                row['ocr_overlap_alternatives'] = [{'observation_ref': 'other'}]
            evidence, unresolved = automatic_binding_evidence(targets=targets, terminology=terminology)
            self.assertEqual(evidence, [])
            self.assertTrue(all('unresolved_ocr_overlap_alternatives' in r['reason'] for r in unresolved))

    def test_missing_tile_or_retained_source_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            *_, index = automatic_fixture(directory)
            with self.assertRaisesRegex(ValueError, 'partition'):
                RegionIndex(primitives=list(index.primitives.values()), regions=index.regions[1:], page_size=index.page_size)
            with self.assertRaisesRegex(ValueError, 'missing retained'):
                RegionIndex(primitives=list(index.primitives.values())[1:], regions=index.regions, page_size=index.page_size)
            legacy = deepcopy(list(index.primitives.values()))
            next(row for row in legacy if row['source_native_segment']['kind'] == 'cubic').pop('search_bbox_display')
            with self.assertRaisesRegex(ValueError, 'conservative search extent'):
                RegionIndex(primitives=legacy, regions=index.regions, page_size=index.page_size)

    def test_source_text_is_preserved_beneath_applicability_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            _, terminology, _, _, _, _, _, _ = automatic_fixture(directory)
            native, overlay = terminology['source_observations']
            self.assertEqual(native['region_role'], 'unknown')
            self.assertEqual(native['epistemic_state'], 'observed')
            self.assertEqual(overlay['region_role'], 'drawing_inline')
            self.assertEqual(overlay['source_observation_ref'], native['id'])
            self.assertNotEqual(overlay['id'], native['id'])
            self.assertTrue(overlay['applicability_evidence'])

    def test_pdf_to_catalog_to_adjacent_audit_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, *_ = automatic_fixture(directory)
            source = Path(directory) / 'neutral.pdf'
            native = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
            annotations = {'document': registry['document'], 'annotations': []}
            output = Path(directory) / 'automatic'
            shadow_database = Path(directory) / 'v3-shadow.sqlite'
            discovery = run(source=source, registry=registry, native_text=native,
                            annotations=annotations, output_dir=output,
                            v3_shadow_database=shadow_database)
            self.assertEqual(discovery['summary']['identified_item_occurrences'], 1)
            self.assertEqual(discovery['summary']['calculated_amounts'], 0)
            self.assertEqual(discovery['parameters']['retained_source_storage'], 'exact_sqlite_cell_index')
            self.assertEqual([row['page_number'] for row in discovery['pages']
                              if row['discovery_state'] != 'not_processed'], [1])
            load = lambda name: json.loads((output / (name + '.json')).read_text())
            shadow = load('outlined-route-connections.v3-shadow')
            self.assertTrue(shadow['canonical_parity'])
            self.assertEqual(shadow['sha256'], shadow['canonical_sha256'])
            manifest = build_automatic_manifest({'catalog': load('item-catalog'),
                'composites': load('outlined-route-composites'), 'geometry': load('bounded-local-3d'),
                'annotations': annotations, 'discovery': discovery})
            self.assertEqual(manifest['execution_mode'], 'automatic_frozen_replay')
            self.assertEqual(len(manifest['item_rows']), 1)
            self.assertEqual(manifest['item_rows'][0]['id'], load('item-catalog')['item_occurrences'][0]['id'])
            resumed = run(source=source, registry=registry, native_text=native,
                annotations=annotations, output_dir=output, resume=True)
            self.assertEqual(resumed, discovery)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(source=source, registry=registry, native_text=native,
                    annotations=annotations, output_dir=output)
            with self.assertRaisesRegex(ValueError, 'inputs or implementation differ'):
                run(source=source, registry=registry, native_text=native,
                    annotations=annotations, output_dir=output, resume=True, region_size=32)
            target_path = output / 'page-001.automatic-targets.json'
            target_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checkpoint evidence'):
                run(source=source, registry=registry, native_text=native,
                    annotations=annotations, output_dir=output, resume=True)

    def test_default_execution_visits_pages_without_item_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, *_ = automatic_fixture(directory)
            source = Path(directory) / 'all-pages.pdf'
            with fitz.open(Path(directory) / 'neutral.pdf') as pdf:
                pdf.new_page(width=400, height=300).insert_text((40, 50), 'Cover')
                pdf.save(source)
            registry = build_mep_sheet_registry(document={
                'document_key': 'pdf-sha256:' + hashlib.sha256(source.read_bytes()).hexdigest(),
                'source_pdf_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'source_bytes': source.stat().st_size, 'page_count': 2},
                pages=[registry['pages'][0], build_sheet_page_record(page_ref='page.2', page_number=2,
                    page_width=400, page_height=300, native_tokens=[],
                    quality={'route': 'native', 'reason': 'synthetic'})])
            native = extract_mep_text_observations(pdf_path=source, sheet_registry=registry)
            discovery = run(source=source, registry=registry, native_text=native,
                annotations={'document': registry['document'], 'annotations': []},
                output_dir=Path(directory) / 'automatic')
            self.assertEqual(discovery['parameters']['page_numbers'], [1, 2])
            self.assertTrue(all(row['discovery_state'] == 'bounded_native_target_search'
                                for row in discovery['pages']))
            self.assertEqual(discovery['pages'][1]['eligible_text_seed_count'], 0)
            self.assertEqual(discovery['pages'][1]['native_source_count'], 0)
            self.assertFalse(discovery['pages'][1]['item_inventory_complete'])
            self.assertEqual(discovery['summary']['identified_item_occurrences'], 1)
