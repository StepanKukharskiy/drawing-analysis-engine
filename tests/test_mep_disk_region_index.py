from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import fitz

from src.drawing_engine.disciplines.mep.mep_automatic_target_binding import (
    RegionIndex, _DiskPrimitives, _box_lookup, _cold_selection_plan,
    automatic_binding_evidence, build_automatic_targets, discover_envelopes,
)
import src.drawing_engine.disciplines.mep.mep_outlined_route_composites as composites_module
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import iter_bounded_native_page_regions
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import RECORD, NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_packed_spatial_index import PackedSpatialIndex
from test_mep_automatic_target_binding import automatic_fixture


class _UnfilteredQueries:
    """Replay the previous envelope flow: query all rows, then filter in caller."""

    def __init__(self, index):
        self.index = index

    def __getattr__(self, name):
        return getattr(self.index, name)

    def query(self, box, *, minimum_member_length=None):
        return self.index.query(box)


class MepDiskRegionIndexTest(unittest.TestCase):
    def test_bulk_secondary_indexes_preserve_rows_queries_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            *_, source = automatic_fixture(directory)
            disks = [_DiskPrimitives(Path(directory) / f'{deferred}.sqlite',
                                     defer_secondary_indexes=deferred) for deferred in (False, True)]
            try:
                for disk in disks:
                    for row in source.primitives.values():
                        disk.add(row, [])
                    row = next(iter(source.primitives.values()))
                    disk.add(row, [])
                    changed = deepcopy(row)
                    changed['points_display'][0][0] += 1
                    with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
                        disk.add(changed, [])
                    disk.build_secondary_indexes()
                    disk.connection.commit()
                for table in ('sources', 'bounds', 'cells'):
                    self.assertEqual(*[list(d.connection.execute(f'SELECT * FROM {table} ORDER BY ordinal'))
                                       for d in disks])
                for query in ('SELECT ordinal FROM bounds WHERE member_length >= 24 ORDER BY ordinal',
                              'SELECT ordinal FROM bounds WHERE start_x BETWEEN 0 AND 300 ORDER BY ordinal',
                              'SELECT ordinal FROM bounds WHERE end_y BETWEEN 0 AND 300 ORDER BY ordinal'):
                    self.assertEqual(*[list(d.connection.execute(query)) for d in disks])
            finally:
                for disk in disks:
                    disk.close()

    def test_cold_plans_share_source_samples_with_exact_uncached_parity(self):
        with tempfile.TemporaryDirectory() as directory, fitz.open() as pdf:
            page = pdf.new_page(width=300, height=160)
            for ordinal in range(8):
                ends = [(40, 40 + ordinal), (240, 40 + ordinal)]
                if ordinal % 2:
                    ends.reverse()
                page.draw_line(*ends, width=.7)
            packets = list(iter_bounded_native_page_regions(
                page, 'page.shared', region_size_display_points=400,
                max_candidates_per_region=100))
            writer = NativeDescriptorPackWriter(
                Path(directory) / 'native.pack', page_ref='page.shared',
                pdf_to_display_matrix=list(page.rotation_matrix), minimum_member_length=23.52)
            for packet in packets:
                for row in packet['primitive_candidates']:
                    writer.append(row)
            pack = NativeDescriptorPack(writer.path, writer.finish())
            before = deepcopy(writer.long_members)
            with patch.object(pack, 'open_mmap', side_effect=AssertionError('planning must stream')):
                for _ in range(2):  # No preparation escapes its page invocation.
                    with patch.object(composites_module, '_samples', wraps=composites_module._samples) as calls:
                        actual = _cold_selection_plan(pack, writer.long_members, [], cell_size=8)
                        self.assertEqual(calls.call_count, len(writer.long_members))
            def uncached(left, right, **ignored):
                return composites_module._parallel_metrics(left, right)
            with patch('src.drawing_engine.disciplines.mep.mep_automatic_target_binding._parallel_metrics', side_effect=uncached):
                expected = _cold_selection_plan(pack, writer.long_members, [], cell_size=8)
            self.assertEqual(actual, expected)
            self.assertEqual(writer.long_members, before)
            plans = actual[2]
            self.assertGreater(len(plans), len(writer.long_members))
            points = [point for plan in plans for side in ('left_samples', 'right_samples')
                      for point in plan['metrics'][side]]
            self.assertLessEqual(len({id(point) for point in points}), 17 * len(writer.long_members))

    def test_measured_selection_and_exact_duplicate_removal_preserve_queries(self):
        neighborhoods = [([0, 0, 12, 12], None), ([0, 0, 12, 12], None),
                         ([20, 20, 25, 25], 2), ([20, 20, 25, 25], 3),
                         ([-10, -10, -5, -5], None)]
        expected = _box_lookup(neighborhoods, cell_size=8)
        measured_profile, compact_profile = {}, {}
        measured = _box_lookup(neighborhoods, cell_size=8,
                               performance_profile=measured_profile)
        compact = _box_lookup(list(dict.fromkeys((tuple(box), limit)
                                                for box, limit in neighborhoods)),
                              cell_size=8, performance_profile=compact_profile)
        for box in ([12, 12, 13, 13], [13, 13, 14, 14], [22, 22, 23, 23],
                    [-5, -5, -5, -5], [-100, -100, 100, 100]):
            for length in (0, 2, 2.5, 3, 4):
                self.assertEqual(measured(box, length), expected(box, length))
                self.assertEqual(compact(box, length), expected(box, length))
        self.assertEqual(measured_profile['query_count'], 25)
        self.assertEqual(measured_profile['neighborhood_count'], 5)
        self.assertEqual(compact_profile['neighborhood_count'], 4)
        self.assertLess(compact_profile['cell_memberships'], measured_profile['cell_memberships'])

    def test_packed_spatial_index_exact_batch_overflow_and_atomic_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'page.spatial'
            bounds = {
                0: ([1, 1, 2, 2], 1.0, 1, 0),
                1: ([9, 9, 11, 11], 2.0, 1, 0),
                # Forced into the separate overflow list.
                2: ([-100, -100, 100, 100], 300.0, 1, 0),
                # Shares a coarse cell with query 0 but does not intersect it.
                3: ([7, 7, 8, 8], 1.0, 1, 0),
            }
            with PackedSpatialIndex.build(
                    path, descriptor_count=4, cell_size=10,
                    maximum_anchor_span_cells=1,
                    descriptors=((ordinal, row[0], row[1])
                                 for ordinal, row in bounds.items()),
                    descriptor_lookup=lambda ordinal: bounds[ordinal][:2]) as index:
                self.assertEqual(index.overflow_count, 1)
                self.assertEqual(index.survivor_count, 4)
                self.assertEqual(index.cell_size, 10)
                self.assertEqual(index.query([0, 0, 3, 3], bounds.__getitem__), {0, 2})
                self.assertEqual(index.query_many(
                    ([0, 0, 3, 3], [10, 10, 12, 12]), bounds.__getitem__), {0, 1, 2})
                self.assertEqual(list(index.survivors()), [0, 1, 2, 3])

            compact_path = root / 'page-without-inline-descriptors.spatial'
            with PackedSpatialIndex.build(
                    compact_path, descriptor_count=4, cell_size=10,
                    maximum_anchor_span_cells=1,
                    inline_query_descriptors=False,
                    descriptors=((ordinal, row[0], row[1])
                                 for ordinal, row in bounds.items())) as index:
                self.assertFalse(index.has_inline_query_descriptors)
                self.assertEqual(index.query(
                    [0, 0, 3, 3], bounds.__getitem__), {0, 2})
                with self.assertRaises(ValueError):
                    index.query_ranges(([0, 0, 3, 3],))

            interrupted = root / 'interrupted.spatial'
            def fail_midstream():
                yield 0, bounds[0][0], bounds[0][1]
                raise RuntimeError('synthetic interruption')
            with self.assertRaises(RuntimeError):
                PackedSpatialIndex.build(
                    interrupted, descriptor_count=4, cell_size=10,
                    descriptors=fail_midstream(),
                    descriptor_lookup=lambda ordinal: bounds[ordinal][:2])
            self.assertFalse(interrupted.exists())
            self.assertFalse(interrupted.with_name(interrupted.name + '.partial').exists())

            # Buffer removal recovers numeric descriptors during file writing;
            # an interruption there must also remove the partially written file.
            def fail_lookup(ordinal):
                raise RuntimeError('interrupted descriptor recovery')
            with self.assertRaisesRegex(RuntimeError, 'interrupted descriptor recovery'):
                PackedSpatialIndex.build(
                    interrupted, descriptor_count=4, cell_size=10,
                    descriptors=((ordinal, row[0], row[1])
                                 for ordinal, row in bounds.items()),
                    descriptor_lookup=fail_lookup)
            self.assertFalse(interrupted.exists())
            self.assertFalse(interrupted.with_name(interrupted.name + '.partial').exists())

    def test_fixed_width_descriptor_pack_round_trips_curves_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory, fitz.open() as pdf:
            page = pdf.new_page(width=200, height=120)
            page.draw_line((10, 20), (180, 20), width=.7)
            page.draw_bezier((20, 80), (70, 10), (130, 110), (180, 80), width=.5)
            packets = list(iter_bounded_native_page_regions(
                page, 'page.pack', region_size_display_points=200,
                max_candidates_per_region=100))
            rows = [row for packet in packets for row in packet['primitive_candidates']]
            path = Path(directory) / 'native.pack'
            writer = NativeDescriptorPackWriter(
                path, page_ref='page.pack', pdf_to_display_matrix=list(page.rotation_matrix),
                minimum_member_length=23.52)
            for row in rows:
                writer.append(row)
            manifest = writer.finish()
            pack = NativeDescriptorPack(path, manifest)
            stream, mapping = pack.open_mmap()
            try:
                self.assertEqual(list(pack.raw_records()),
                                 [pack.raw_at(i, mapping) for i in range(len(pack))])
            finally:
                mapping.close()
                stream.close()
            self.assertEqual(path.stat().st_size, len(rows) * RECORD.size)
            self.assertTrue(pack.verify_hashes())
            expected = deepcopy(rows)
            for row in expected:
                row['search_refs'] = []  # Region ownership is a separate manifest.
            self.assertEqual(list(pack.rows()), expected)
            self.assertEqual([row['source_primitive_ref'] for row in writer.long_members],
                             [row['source_primitive_ref'] for row in rows
                             if row['source_native_segment']['kind'] == 'line'])

            # Exercise the chunk boundary, short final block, early close and
            # fail-closed length checks without building a large PDF fixture.
            body = path.read_bytes()[:RECORD.size]
            chunk_path = Path(directory) / 'chunks.pack'
            chunk_path.write_bytes(body * 4097)
            chunks = NativeDescriptorPack(chunk_path, {**manifest, 'record_count': 4097})
            self.assertEqual(list(chunks.raw_records()), [RECORD.unpack(body)] * 4097)
            records = chunks.raw_records()
            next(records)
            records.close()
            with chunk_path.open('r+b') as output:
                output.truncate(4097 * RECORD.size - 1)
            with self.assertRaisesRegex(ValueError, 'truncated'):
                list(chunks.raw_records())
            chunk_path.write_bytes(body * 4096)
            with self.assertRaisesRegex(ValueError, 'record count mismatch'):
                list(chunks.raw_records())
            chunk_path.write_bytes(body * 4098)
            with self.assertRaisesRegex(ValueError, 'record count mismatch'):
                list(chunks.raw_records())

    def test_descriptor_refinement_matches_overloaded_legacy_partition(self):
        with tempfile.TemporaryDirectory() as directory, fitz.open() as pdf:
            page = pdf.new_page(width=128, height=128)
            for ordinal in range(40):
                page.draw_line((1, ordinal * 3 + 1), (127, ordinal * 3 + 1), width=.7)
            packets = list(iter_bounded_native_page_regions(
                page, 'page.dense', region_size_display_points=64,
                max_candidates_per_region=2,
                minimum_region_size_display_points=8))
            expected = [packet['region'] for packet in packets]
            profile = {}
            with RegionIndex.from_native_page(
                    page=page, page_ref='page.dense', page_size=[128, 128],
                    region_size_display_points=64, max_candidates_per_region=2,
                    minimum_region_size_display_points=8,
                    sqlite_path=Path(directory) / 'dense.sqlite',
                    performance_profile=profile) as direct:
                self.assertEqual(direct.regions, expected)
                self.assertEqual(profile['native_descriptor_count'], 40)
                self.assertEqual(len(direct._partition_coverage), len(expected))
                self.assertTrue(any(row['excluded_candidate_count']
                                    for row in direct._partition_coverage.values()))

    def test_survivor_plan_preserves_envelopes_leaders_and_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, terminology, expected, *_ = automatic_fixture(
                directory, curved_crossing=True)
            source = Path(directory) / 'neutral.pdf'
            scope = registry['pages'][0]
            profile = {}
            with fitz.open(source) as pdf, RegionIndex.from_native_page(
                    page=pdf[0], page_ref='page.1', page_size=scope['page_size_display'],
                    region_size_display_points=64, max_candidates_per_region=4000,
                    minimum_region_size_display_points=8,
                    sqlite_path=Path(directory) / 'survivors.sqlite',
                    performance_profile=profile,
                    observations=terminology['source_observations']) as direct:
                actual = build_automatic_targets(
                    sheet_registry=registry, page_indexes={'page.1': direct},
                    observations=terminology['source_observations'],
                    envelope_engine='page_topology')
            self.assertEqual(actual, expected)
            self.assertLessEqual(profile['sqlite_survivor_count'],
                                 profile['native_descriptor_count'])

    def test_cold_backends_preserve_identical_candidates_and_evidence(self):
        for options in ({}, {'ambiguous': True}, {'budget': 1},
                        {'short_crossing': True}, {'curved_crossing': True},
                        {'rotation': 90}, {'labelled': False}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                registry, terminology, *_ = automatic_fixture(directory, **options)
                scope = registry['pages'][0]
                results = []
                for threshold, backend in ((0, 'packed_mmap_inline_cell_descriptors'),
                                           (10**9, 'sqlite_bulk_rtree')):
                    with self.subTest(backend=backend), patch(
                            'src.drawing_engine.disciplines.mep.mep_automatic_target_binding.PACKED_SPATIAL_MIN_SOURCE_COUNT',
                            threshold), fitz.open(Path(directory) / 'neutral.pdf') as pdf:
                        profile = {}
                        with RegionIndex.from_native_page(
                                page=pdf[0], page_ref=scope['page_ref'],
                                page_size=scope['page_size_display'],
                                max_candidates_per_region=options.get('budget', 4000),
                                sqlite_path=Path(directory) / (backend + '.sqlite'),
                                performance_profile=profile,
                                observations=terminology['source_observations']) as index:
                            actual = build_automatic_targets(
                                sheet_registry=registry, page_indexes={scope['page_ref']: index},
                                observations=terminology['source_observations'],
                                envelope_engine='page_topology')
                        self.assertEqual(profile['temporary_spatial_backend'], backend)
                        results.append((actual, automatic_binding_evidence(
                            targets=actual, terminology=terminology)))
                self.assertEqual(results[0], results[1])

    def test_direct_cold_scan_skips_json_spool_and_preserves_page_results(self):
        with tempfile.TemporaryDirectory() as directory, fitz.open() as pdf:
            page = pdf.new_page(width=300, height=160)
            page.draw_line((40, 70), (240, 70), width=.7)
            page.draw_line((40, 74), (240, 74), width=.7)
            page.draw_line((40, 70), (40, 74), width=.7)
            packets = list(iter_bounded_native_page_regions(
                page, 'page.synthetic', region_size_display_points=64,
                max_candidates_per_region=4000,
                minimum_region_size_display_points=8))
            regions = [packet['region'] for packet in packets]
            rows = [row for packet in packets for row in packet['primitive_candidates']]
            profile = {}
            with RegionIndex(primitives=rows, regions=regions, page_size=[300, 160]) as memory:
                with RegionIndex.from_native_page(
                        page=page, page_ref='page.synthetic', page_size=[300, 160],
                        region_size_display_points=64, max_candidates_per_region=4000,
                        minimum_region_size_display_points=8,
                        sqlite_path=Path(directory) / 'direct.sqlite',
                        performance_profile=profile) as direct:
                    self.assertEqual(direct.regions, memory.regions)
                    query = [0, 0, 300, 160]
                    direct_rows = direct.query(query)[0]
                    memory_rows = memory.query(query)[0]
                    self.assertEqual(direct.with_initial_search_refs(direct_rows),
                                     memory_rows)
                    scope = {'id': 'page.synthetic', 'page_ref': 'page.synthetic',
                             'page_number': 1}
                    selected_memory, witnesses_memory = discover_envelopes(
                        index=memory, page_scope=scope,
                        engine='page_topology')
                    selected_direct, witnesses_direct = discover_envelopes(
                        index=direct, page_scope=scope,
                        engine='page_topology')
                    self.assertEqual((selected_direct, witnesses_direct),
                                     (selected_memory, witnesses_memory))
            self.assertEqual(profile['native_descriptor_count'], 3)
            self.assertEqual(len(profile['competitor_coverage_sha256']), 64)
            self.assertIn('sqlite_insert_seconds', profile)

    def test_scene_targets_and_binding_evidence_equal_memory_backend(self):
        for options in ({}, {'ambiguous': True}, {'budget': 1}, {'short_crossing': True},
                        {'curved_crossing': True}, {'rotation': 90}, {'labelled': False}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                registry, terminology, targets, _, _, _, _, memory = automatic_fixture(directory, **options)
                targets = build_automatic_targets(sheet_registry=registry,
                    page_indexes={registry['pages'][0]['page_ref']: memory},
                    observations=terminology['source_observations'])
                previous = build_automatic_targets(sheet_registry=registry,
                    page_indexes={registry['pages'][0]['page_ref']: _UnfilteredQueries(memory)},
                    observations=terminology['source_observations'])
                self.assertEqual(targets, previous)
                rows = list(memory.primitives.values())
                # Same IDs in multiple input packets must deduplicate without
                # changing insertion order, source rows or competitor searches.
                source = iter([*rows, deepcopy(rows[0])])
                with RegionIndex(primitives=source, regions=memory.regions, page_size=memory.page_size,
                                 sqlite_path=Path(directory) / 'index.sqlite') as disk:
                    self.assertIsInstance(disk.primitives, Mapping)
                    self.assertEqual(list(disk.primitives), list(memory.primitives))
                    self.assertEqual(list(disk.primitives.values()), rows)
                    self.assertEqual(list(disk.primitives.items()), list(memory.primitives.items()))
                    self.assertEqual(len(disk.primitives), len(rows))
                    self.assertFalse(disk.cells)
                    self.assertEqual(disk.native(reversed(list(memory.primitives))), memory.native(memory.primitives))
                    self.assertEqual(list(disk.iter_member_lines(23.52)), list(memory.iter_member_lines(23.52)))
                    replay = build_automatic_targets(sheet_registry=registry,
                        page_indexes={registry['pages'][0]['page_ref']: disk},
                        observations=terminology['source_observations'])
                    self.assertEqual(replay, targets)
                    self.assertEqual(automatic_binding_evidence(targets=replay, terminology=terminology),
                                     automatic_binding_evidence(targets=targets, terminology=terminology))

    def test_exact_queries_boundaries_curves_and_streamed_region_population(self):
        with tempfile.TemporaryDirectory() as directory, fitz.open() as pdf:
            page = pdf.new_page(width=256, height=256)
            page.draw_line((64, 0), (64, 256), width=.7)
            page.draw_line((0, 128), (256, 128), width=.7)
            page.draw_bezier((20, 80), (200, 220), (25, 250), (80, 20), width=.7)
            packets = list(iter_bounded_native_page_regions(page, 'page.synthetic',
                region_size_display_points=64, max_candidates_per_region=2,
                minimum_region_size_display_points=8))
            regions = []
            def streamed():
                for packet in packets:
                    regions.append(packet['region'])
                    yield from packet['primitive_candidates']
            with RegionIndex(primitives=streamed(), regions=regions, page_size=[256, 256],
                             sqlite_path=Path(directory) / 'index.sqlite') as disk:
                memory = RegionIndex(primitives=[r for p in packets for r in p['primitive_candidates']],
                                     regions=regions, page_size=[256, 256])
                boxes = [[0, 0, 256, 256], [-5, -5, 0, 0], [256, 256, 257, 257],
                         [64, 128, 64, 128], [150, 200, 151, 201], [257, 257, 260, 260]]
                for x in (0, 64, 128, 256):
                    for delta in (-math.inf, math.inf):
                        boundary = math.nextafter(x, delta)
                        boxes.append([boundary, 0, boundary, 256])
                for box in boxes:
                    self.assertEqual(disk.query(box), memory.query(box), box)
                    for threshold in (0, 23.52, 256, math.nextafter(256, math.inf)):
                        self.assertEqual(disk.query(box, minimum_member_length=threshold),
                                         memory.query(box, minimum_member_length=threshold), (box, threshold))
                self.assertEqual(disk.primitives.get('missing'), None)
                with self.assertRaises(KeyError):
                    disk.primitives['missing']
                row = next(disk.primitives.values())
                union_boxes = boxes[:2]
                required_refs = [row['source_primitive_ref']]
                expected_refs = set(required_refs)
                for box in union_boxes:
                    expected_refs.update(source['source_primitive_ref'] for source in
                                         memory.query(box)[0]
                                         if source['source_native_segment']['length_points'] <= 256)
                self.assertEqual(disk.source_ref_union_count(
                    union_boxes, 256, required_refs), len(expected_refs))
                before = deepcopy(row)
                row['points_display'][0][0] += 10
                self.assertEqual(disk.primitives[before['source_primitive_ref']], before)

    def test_incomplete_partition_and_missing_or_conflicting_evidence_reject_equally(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, _, _, memory = automatic_fixture(directory)
            rows = list(memory.primitives.values())
            conflict = deepcopy(rows[0])
            conflict['points_display'][0][0] += .1
            reused_id = deepcopy(rows[1])
            reused_id['id'] = rows[0]['id']
            reused_id['source_primitive_ref'] = 'source.candidate-id-collision'
            missing_hull = deepcopy(rows)
            next(r for r in missing_hull if r['source_native_segment']['kind'] == 'cubic').pop('search_bbox_display')
            cases = [(rows, memory.regions[1:]), (rows[1:], memory.regions),
                     ([*rows, conflict], memory.regions), ([*rows, reused_id], memory.regions),
                     (missing_hull, memory.regions)]
            for number, (sources, regions) in enumerate(cases):
                for disk in (False, True):
                    with self.subTest(number=number, disk=disk), self.assertRaises(ValueError):
                        RegionIndex(primitives=iter(sources), regions=regions, page_size=memory.page_size,
                                    sqlite_path=Path(directory) / f'bad-{number}.sqlite' if disk else None)
            incomplete = deepcopy(memory.regions)
            for region in incomplete:
                region['relevant_competitor_search_complete'] = False
            with RegionIndex(primitives=iter(rows), regions=incomplete, page_size=memory.page_size,
                             sqlite_path=Path(directory) / 'incomplete.sqlite') as disk:
                query = [0, 0, *memory.page_size]
                self.assertFalse(disk.query(query)[1])
                filtered, complete, region_refs = disk.query(query, minimum_member_length=math.inf)
                self.assertEqual(filtered, [])
                self.assertFalse(complete)
                self.assertEqual(region_refs, disk.query(query)[2])
                self.assertEqual(disk.query(query), RegionIndex(primitives=rows, regions=incomplete,
                                                               page_size=memory.page_size).query(query))

    def test_joined_queries_decode_each_returned_source_once_and_do_not_lookup_per_row(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, _, _, memory = automatic_fixture(directory, curved_crossing=True)
            with RegionIndex(primitives=iter(memory.primitives.values()), regions=memory.regions,
                             page_size=memory.page_size, sqlite_path=Path(directory) / 'index.sqlite') as disk:
                query = [0, 0, *memory.page_size]
                for threshold in (None, 23.52):
                    expected = memory.query(query, minimum_member_length=threshold)
                    with patch.object(type(disk.primitives), '__getitem__', side_effect=AssertionError('per-row lookup')):
                        with patch('src.drawing_engine.disciplines.mep.mep_automatic_target_binding.json.loads', wraps=json.loads) as decode:
                            actual = disk.query(query, minimum_member_length=threshold)
                            self.assertEqual(decode.call_count, len(actual[0]))
                    self.assertEqual(actual, expected)
                all_rows = disk.query(query)[0]
                self.assertTrue(any(r['source_native_segment']['kind'] == 'cubic' for r in all_rows))
                self.assertTrue(any(r['source_native_segment']['length_points'] < 23.52 for r in all_rows))

    def test_lifecycle_and_bounded_database_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, _, _, memory = automatic_fixture(directory)
            path = Path(directory) / 'index.sqlite'
            with RegionIndex(primitives=iter(memory.primitives.values()), regions=memory.regions,
                             page_size=memory.page_size, sqlite_path=path) as disk:
                connection = disk._disk.connection
                self.assertEqual(connection.execute('PRAGMA cache_size').fetchone()[0], -16384)
                self.assertEqual(connection.execute('PRAGMA temp_store').fetchone()[0], 1)
                self.assertEqual(connection.execute('PRAGMA mmap_size').fetchone()[0], 0)
                with self.assertRaises(ValueError):
                    RegionIndex(primitives=(), regions=(), page_size=memory.page_size, sqlite_path=path)
            disk.close()  # Cleanup is idempotent.
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute('SELECT 1')
            self.assertTrue(path.exists())  # The caller owns deletion of its temporary database.


if __name__ == '__main__':
    unittest.main()
