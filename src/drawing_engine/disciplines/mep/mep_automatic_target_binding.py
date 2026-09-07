"""Bounded automatic straight-envelope / native dot-leader evidence.

This deliberately narrow search supplies existing M3.5/M4 certificates. It does
not use reviewed selectors, tags as identity, colour, schedules, or a nearest
neighbour winner. A missing tile, truncated competitor search, an untraced
leader, or more than one contacted envelope closes the acceptance gate.
"""

from collections import defaultdict
from collections.abc import Mapping
from array import array
from copy import deepcopy
import heapq
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter
import zlib

import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import (
    _bounded_page_records, _cells, _intersects,
    _refined_bounded_native_page_regions, _refined_regions_from_descriptor_pack,
)
from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import RECORD, NativeDescriptorPack, NativeDescriptorPackWriter
from src.drawing_engine.disciplines.mep.mep_packed_spatial_index import PackedSpatialIndex
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import (
    _parallel_metrics, _prepare_polyline, _shortest_cap_path, _style_compatible,
    _provenance_compatible, _ownership_compatible,
    build_mep_outlined_route_composites,
)
from src.drawing_engine.disciplines.mep.mep_route_observations import (
    _normalise_style, _native_candidates, _cluster_endpoints, _fragment_records,
    build_mep_route_graph,
)
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id, build_mep_terminology_proposals, interpret_mep_text
from src.drawing_engine.core.vector_topology import point_distance_to_segment


# SQLite's bounded bulk R-tree is already faster for ordinary dense pages. The
# packed mmap path is reserved for ultra-dense denominators where page-local
# SQLite count queries dominate wall time. This is a source-cost policy, never
# a drawing name, sheet identity or reviewed-coordinate dispatch.
PACKED_SPATIAL_MIN_SOURCE_COUNT = 3_000_000


def _expand(box, radius):
    return [box[0] - radius, box[1] - radius, box[2] + radius, box[3] + radius]


def _point_box(point, radius):
    return _expand([*point, *point], radius)


def _member_line(row, minimum_length):
    native = row['source_native_segment']
    return (native['kind'] == 'line' and native['style'].get('stroke') is not None
            and native['length_points'] >= minimum_length)


def _box_lookup(neighborhoods, *, cell_size=64, performance_profile=None):
    cells = defaultdict(list)
    for box, maximum_source_length in neighborhoods:
        for cell in _cells(box, cell_size):
            cells[cell].append((box, maximum_source_length))

    if performance_profile is not None:
        performance_profile.update(
            neighborhood_count=len(neighborhoods),
            unique_neighborhood_count=len({(tuple(box), limit) for box, limit in neighborhoods}),
            cell_count=len(cells), cell_memberships=sum(map(len, cells.values())),
            query_count=0, candidate_visits=0, exact_intersection_tests=0,
            duplicate_candidate_visits=0)

        def measured_matches(box, source_length):
            seen = set()
            visits = tests = duplicates = 0
            found = False
            for cell in _cells(box, cell_size):
                for candidate, maximum_source_length in cells.get(cell, ()):
                    visits += 1
                    marker = id(candidate)
                    if marker in seen:
                        duplicates += 1
                    else:
                        tests += 1
                        if (_intersects(box, candidate)
                                and (maximum_source_length is None
                                     or source_length <= maximum_source_length)):
                            found = True
                            break
                    seen.add(marker)
                if found:
                    break
            performance_profile['query_count'] += 1
            performance_profile['candidate_visits'] += visits
            performance_profile['exact_intersection_tests'] += tests
            performance_profile['duplicate_candidate_visits'] += duplicates
            return found
        return measured_matches

    def matches(box, source_length):
        seen = set()
        for cell in _cells(box, cell_size):
            for candidate, maximum_source_length in cells.get(cell, ()):
                marker = id(candidate)
                if (marker not in seen and _intersects(box, candidate)
                        and (maximum_source_length is None
                             or source_length <= maximum_source_length)):
                    return True
                seen.add(marker)
        return False
    return matches


def _cold_selection_plan(pack, long_members, observations, *, cell_size=64,
                         allocation_observer=None):
    """Conservative envelope, cap and leader query coverage.

    Every eligible long member is retained independently.  A passing envelope
    cannot require a partner beyond .216L+.35 or a cap source beyond .4L+.1:
    the latter follows directly from the unchanged 10*width and cap=4*width
    gates.  Leader expansion over-approximates all eight permitted hops and
    keeps the full contact-alternative radius at each endpoint.
    """
    neighborhoods = []
    if not len(pack):
        return neighborhoods, set(), [], {
            'eligible_long_member_count': 0, 'partner_candidate_pair_count': 0,
            'cap_neighborhood_pair_count': 0, 'selection_neighborhood_count': 0,
        }
    member_cells = defaultdict(list)
    # Plans share read-only source samples. Preparing them per pair retained
    # many identical point lists while the descriptor mmap was resident.
    # The existing helper preserves the exact unrounded certificate values.
    prepared_members = [_prepare_polyline({
        'geometry': {'points_display': member['points_display']},
        'style': _normalise_style(pack.manifest['styles'][member['style_id']]),
    }) for member in long_members]
    for ordinal, member in enumerate(long_members):
        for cell in _cells(member['search_bbox_display'], 64):
            member_cells[cell].append(ordinal)
    candidate_pairs, envelope_plans, cap_pair_count = set(), [], 0
    for left_ordinal, left in enumerate(long_members):
        radius = .216 * left['length_points'] + .35
        partner_box = _point_box(left['points_display'][0], radius)
        around = set()
        for cell in _cells(partner_box, 64):
            around.update(member_cells.get(cell, ()))
        for right_ordinal in around:
            if right_ordinal == left_ordinal:
                continue
            right = long_members[right_ordinal]
            if not _intersects(right['search_bbox_display'], partner_box):
                continue
            key = tuple(sorted((left_ordinal, right_ordinal)))
            if key in candidate_pairs:
                continue
            candidate_pairs.add(key)
            left_style = pack.manifest['styles'][left['style_id']]
            right_style = pack.manifest['styles'][right['style_id']]
            a = {'id': left['source_primitive_ref'],
                 'geometry': {'points_display': left['points_display']},
                 'style': _normalise_style(left_style)}
            b = {'id': right['source_primitive_ref'],
                 'geometry': {'points_display': right['points_display']},
                 'style': _normalise_style(right_style)}
            if not _style_compatible(a['style'], b['style']):
                continue
            metrics = _parallel_metrics(
                a, b, left_prepared=prepared_members[left_ordinal],
                right_prepared=prepared_members[right_ordinal])
            mean = metrics['mean_separation_display_points']
            width = metrics['member_width_display_points']
            length = min(metrics['left_length_display_points'],
                         metrics['right_length_display_points'])
            if not (length >= max(10 * width, 5 * mean)
                    and metrics['length_ratio'] >= .98
                    and max(.1, .15 * width) < mean <= max(12 * width, .08 * length)
                    and metrics['maximum_separation_deviation_display_points'] <= max(.35, .08 * mean)
                    and metrics['maximum_tangent_difference_degrees'] <= 2
                    and all(abs(value - mean) <= max(.35, .08 * mean)
                            for value in metrics['endpoint_separations_display_points'])):
                continue
            limit = max(2.5 * mean, 4 * width, 1) + .1
            # Preserve the same orientation that the source-ID-sorted runtime
            # traversal would encounter first.  These exact metrics can then
            # be reused after the second scan instead of repeating all pair
            # generation and certificate arithmetic over SQLite.
            by_ref = {left['source_primitive_ref']: (a, prepared_members[left_ordinal]),
                      right['source_primitive_ref']: (b, prepared_members[right_ordinal])}
            source_key = tuple(sorted(by_ref))
            oriented_metrics = _parallel_metrics(
                by_ref[source_key[0]][0], by_ref[source_key[1]][0],
                left_prepared=by_ref[source_key[0]][1],
                right_prepared=by_ref[source_key[1]][1])
            envelope_plans.append({
                'member_source_primitive_refs': source_key,
                'metrics': oriented_metrics,
                'cap_limit_display_points': limit - .1,
            })
            # Preserve both orientations. The unchanged discovery order may
            # encounter either member first when only one start box overlaps.
            # Selection keeps every exact-bound source in the cap locality.
            # The source-length limit belongs to reachability, not storage:
            # longer strokes can still be required by later boundary replay.
            neighborhoods.extend((_point_box(point, limit), None)
                                 for member in (left, right)
                                 for point in member['points_display'])
            cap_pair_count += 1

    leader_selected = set()
    if not len(pack):
        return neighborhoods, leader_selected, envelope_plans, {
            'eligible_long_member_count': len(long_members),
            'partner_candidate_pair_count': len(candidate_pairs),
            'cap_neighborhood_pair_count': cap_pair_count,
            'selection_neighborhood_count': len(neighborhoods),
            'leader_descriptor_scan_passes': 0,
        }

    # Leader paths are endpoint-topology searches, not arbitrary area walks.
    # Repeatedly probing the all-primitive cell directory made dense hatch ink
    # dominate cold planning.  Eight sequential fixed-width scans are bounded
    # and deterministic: pass zero finds exact text-edge starts; each later
    # pass joins line endpoints only to the preceding frontier.  Contact boxes
    # still retain every curve, dot and rival in the separate survivor scan.
    initial_cells = defaultdict(list)
    for observation_ordinal, observation in enumerate(observations or ()):
        text_box = observation['bbox_display']
        height = max(1, min(text_box[2] - text_box[0], text_box[3] - text_box[1]))
        initial = _expand(text_box, .4 * height)
        neighborhoods.append((initial, None))
        entry = (observation_ordinal, text_box, height, initial)
        for cell in _cells(initial, cell_size):
            initial_cells[cell].append(entry)

    normal_styles = {
        ordinal: _normalise_style(style)
        for ordinal, style in enumerate(pack.manifest['styles'])
    }
    style_keys = {
        ordinal: json.dumps(style, sort_keys=True, separators=(',', ':'))
        for ordinal, style in normal_styles.items()
    }
    frontier = []
    scan_passes = 1
    if allocation_observer:
        allocation_observer('before_descriptor_scan', {
            'neighborhoods': len(neighborhoods), 'candidate_pairs': len(candidate_pairs),
            'envelope_plans': len(envelope_plans), 'member_cells': len(member_cells),
            'descriptor_pack_bytes': len(pack) * RECORD.size,
            'descriptor_scan_chunk_bytes': 4096 * RECORD.size,
        })
    # These scans only move forward. Do not keep the whole descriptor mapping
    # resident alongside pair plans; byte-identical fixed-width chunks suffice.
    for ordinal, raw in enumerate(pack.raw_records()):
        if allocation_observer and ordinal % 524288 == 0:
            allocation_observer('initial_scan_progress', {
                'descriptor_ordinal': ordinal, 'frontier': len(frontier)})
        style = pack.manifest['styles'][raw[3]]
        if raw[4] != 1 or style.get('fill') is not None:
            continue
        possible = {}
        for cell in _cells(raw[27:31], cell_size):
            for entry in initial_cells.get(cell, ()):
                possible[entry[0]] = entry
        for _, text_box, height, initial in possible.values():
            if not _intersects(raw[27:31], initial):
                continue
            for end, point in enumerate(
                    ((raw[15], raw[16]), (raw[17], raw[18]))):
                if (min(abs(point[0] - text_box[0]),
                        abs(point[0] - text_box[2])) <= .4 * height
                        and text_box[1] - .25 * height
                        <= point[1] <= text_box[3] + .25 * height):
                    frontier.append((ordinal, raw, end, raw[3], height))

    if allocation_observer:
        allocation_observer('initial_scan_complete', {'frontier': len(frontier)})
    visited = set()
    for _ in range(8):
        targets = defaultdict(list)
        target_records = []
        for current_ordinal, current, entered, start_style_id, height in frontier:
            state = (current_ordinal, entered, style_keys[start_style_id], height)
            if state in visited:
                continue
            visited.add(state)
            point = ((current[17], current[18]) if entered == 0
                     else (current[15], current[16]))
            neighborhoods.append((_point_box(point, height), None))
            target_ordinal = len(target_records)
            target_records.append(
                (point, current_ordinal, start_style_id, height, []))
            targets[(math.floor(point[0] / .05),
                     math.floor(point[1] / .05))].append(
                target_ordinal)
        if not targets:
            break
        scan_passes += 1
        for ordinal, raw in enumerate(pack.raw_records()):
            style = pack.manifest['styles'][raw[3]]
            if raw[4] != 1 or style.get('fill') is not None:
                continue
            for end, endpoint in enumerate(
                    ((raw[15], raw[16]), (raw[17], raw[18]))):
                cell = (math.floor(endpoint[0] / .05),
                        math.floor(endpoint[1] / .05))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for target_ordinal in targets.get(
                                (cell[0] + dx, cell[1] + dy), ()):
                            (point, current_ordinal, start_style_id,
                             height, matches) = target_records[target_ordinal]
                            if (ordinal != current_ordinal
                                    and math.dist(point, endpoint) <= .05
                                    and _style_compatible(
                                        normal_styles[start_style_id],
                                        normal_styles[raw[3]])):
                                matches.append(
                                    (ordinal, raw, end, start_style_id, height))
        # This is the same fail-closed non-branching condition used by
        # _leader_paths.  Continuing every branch is unnecessary for
        # evidence parity because the real certificate stops there.
        frontier = [matches[0] for *_, matches in target_records
                    if len(matches) == 1]
        if allocation_observer:
            allocation_observer('leader_scan_complete', {
                'scan_passes': scan_passes, 'frontier': len(frontier),
                'target_records': len(target_records), 'visited': len(visited)})
    envelope_plans.sort(key=lambda row: row['member_source_primitive_refs'])
    if allocation_observer:
        allocation_observer('planning_complete', {'neighborhoods': len(neighborhoods)})
    return neighborhoods, leader_selected, envelope_plans, {
        'eligible_long_member_count': len(long_members),
        'partner_candidate_pair_count': len(candidate_pairs),
        'cap_neighborhood_pair_count': cap_pair_count,
        'selection_neighborhood_count': len(neighborhoods),
        'leader_descriptor_scan_passes': scan_passes,
    }


class _DiskPrimitives(Mapping):
    """Temporary exact row/cell storage, never a geometry approximation."""

    def __init__(self, path, *, defer_secondary_indexes=False):
        path = Path(path)
        if path.exists() and path.stat().st_size:
            raise ValueError('native source database must be new or empty')
        self.connection = sqlite3.connect(path)
        try:
            self.connection.executescript('''
                PRAGMA cache_size=-16384;
                PRAGMA temp_store=FILE;
                PRAGMA mmap_size=0;
                CREATE TABLE sources (
                    ordinal INTEGER PRIMARY KEY,
                    source_ref TEXT COLLATE BINARY NOT NULL UNIQUE,
                    candidate_id TEXT NOT NULL UNIQUE,
                    body BLOB,
                    page_ref TEXT, close_path INTEGER, matrix_body TEXT,
                    drawing_ref TEXT, primitive_ref TEXT,
                    item_index INTEGER, part_index INTEGER,
                    kind TEXT, axis TEXT,
                    native_start_x REAL, native_start_y REAL,
                    native_end_x REAL, native_end_y REAL,
                    control_body TEXT, sample_body TEXT,
                    native_x0 REAL, native_y0 REAL, native_x1 REAL, native_y1 REAL,
                    native_metadata_body TEXT, points_body TEXT, search_refs_body TEXT,
                    descriptor_ordinal INTEGER);
                CREATE TABLE bounds (
                    ordinal INTEGER PRIMARY KEY,
                    member_length REAL,
                    source_length REAL NOT NULL,
                    start_x REAL NOT NULL, start_y REAL NOT NULL,
                    end_x REAL NOT NULL, end_y REAL NOT NULL,
                    style_body TEXT,
                    gx0 REAL NOT NULL, gy0 REAL NOT NULL, gx1 REAL NOT NULL, gy1 REAL NOT NULL,
                    x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL);
                CREATE VIRTUAL TABLE cells USING rtree(
                    ordinal, min_x, max_x, min_y, max_y);
            ''')
            self._compact_sources = []
            self._compact_bounds = []
            self._compact_cells = []
            self._next_ordinal = 1
            if not defer_secondary_indexes:
                self.build_secondary_indexes()
        except BaseException:
            self.close()
            raise

    def build_secondary_indexes(self):
        # Bulk sorting avoids maintaining three B-trees for every inserted
        # descriptor. Source/candidate uniqueness remains enforced throughout.
        self._flush_compact()
        for name, columns in (('member_lengths', 'member_length'),
                              ('bounds_start_xy', 'start_x,start_y'),
                              ('bounds_end_xy', 'end_x,end_y')):
            self.connection.execute(
                f'CREATE INDEX IF NOT EXISTS {name} ON bounds({columns})')

    def add(self, row, cells):
        ref = row['source_primitive_ref']
        native = row['source_native_segment']
        length = native['length_points'] if _member_line(row, -math.inf) else None
        points = row['points_display']
        try:
            cursor = self.connection.execute(
                'INSERT INTO sources(source_ref,candidate_id,body) VALUES (?,?,?)',
                (ref, row['id'], zlib.compress(
                    json.dumps(row, separators=(',', ':')).encode(), level=1)))
        except sqlite3.IntegrityError:
            if ref in self and self[ref] == row:
                return
            raise ValueError('conflicting duplicate native source or candidate ID') from None
        self.connection.execute('INSERT INTO bounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                (cursor.lastrowid, length, native['length_points'],
                                 *points[0], *points[-1], json.dumps(native.get('style'), separators=(',', ':')),
                                 *row['bbox_display'],
                                 *row.get('search_bbox_display', row['bbox_display'])))
        box = row.get('search_bbox_display', row['bbox_display'])
        self.connection.execute('INSERT INTO cells VALUES (?,?,?,?,?)',
                                (cursor.lastrowid, box[0], box[2], box[1], box[3]))

    def add_compact(self, row, cells):
        """Store numeric line descriptors without a nested per-source JSON body."""
        ref = row['source_primitive_ref']
        native = row['source_native_segment']
        length = native['length_points'] if _member_line(row, -math.inf) else None
        points = row['points_display']
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._compact_sources.append(
            (ordinal, ref, row['id'], row['page_ref'], int(row['source_drawing_close_path']),
             json.dumps(row['pdf_to_display_matrix'], separators=(',', ':')),
             native['drawing_ref'], native['primitive_ref'], native['item_index'], native['part_index'],
             native['kind'], native['axis'], *native['start_display'], *native['end_display'],
             json.dumps(native['control_points_display'], separators=(',', ':'))
                if native['control_points_display'] else None,
             json.dumps(native['sample_points_display'], separators=(',', ':'))
                if native['sample_points_display'] else None,
             *native['bbox_display'],
             json.dumps(native.get('native_metadata'), separators=(',', ':'))
                if native.get('native_metadata') else None,
             json.dumps(points, separators=(',', ':')) if native['kind'] != 'line' else None,
             json.dumps(row.get('search_refs', []), separators=(',', ':')),
             row.get('_native_descriptor_ordinal')))
        self._compact_bounds.append(
            (ordinal, length, native['length_points'], *points[0], *points[-1],
             json.dumps(native.get('style'), separators=(',', ':'), sort_keys=True),
             *row['bbox_display'], *row.get('search_bbox_display', row['bbox_display'])))
        box = row.get('search_bbox_display', row['bbox_display'])
        self._compact_cells.append((ordinal, box[0], box[2], box[1], box[3]))
        if len(self._compact_sources) >= 4096:
            self._flush_compact()

    def add_native_descriptor(self, raw, descriptor_ordinal, *, page_ref,
                              matrix_body, styles, style_bodies,
                              payload_stream):
        """Insert one fixed-width pack row without rebuilding a nested dict."""
        drawing, item, part, style_id = raw[:4]
        kind = 'line' if raw[4] == 1 else 'cubic'
        axis = (None, 'horizontal', 'vertical', 'oblique')[raw[5]]
        source_ref = f'drawing[{drawing}].item[{item}].segment[{part}]'
        drawing_ref = f'drawing[{drawing}]'
        primitive_ref = f'{drawing_ref}.item[{item}]'
        payload = {}
        if raw[10]:
            payload_stream.seek(raw[9])
            payload = json.loads(payload_stream.read(raw[10]))
        control = payload.get('control_points_display', [])
        sample = payload.get('sample_points_display', [])
        metadata = payload.get('native_metadata')
        points = payload.get('points_display')
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._compact_sources.append((
            ordinal, source_ref,
            'mep_native_target_primitive.' + raw[8].hex(), page_ref,
            int(raw[6]), matrix_body, drawing_ref, primitive_ref, item, part,
            kind, axis, raw[11], raw[12], raw[13], raw[14],
            json.dumps(control, separators=(',', ':')) if control else None,
            json.dumps(sample, separators=(',', ':')) if sample else None,
            raw[19], raw[20], raw[21], raw[22],
            json.dumps(metadata, separators=(',', ':')) if metadata else None,
            json.dumps(points, separators=(',', ':')) if points else None,
            '[]', descriptor_ordinal))
        member_length = (raw[31] if kind == 'line'
                         and styles[style_id].get('stroke') is not None else None)
        self._compact_bounds.append((
            ordinal, member_length, raw[31], raw[15], raw[16], raw[17], raw[18],
            style_bodies[style_id], raw[23], raw[24], raw[25], raw[26],
            raw[27], raw[28], raw[29], raw[30]))
        self._compact_cells.append((
            ordinal, raw[27], raw[29], raw[28], raw[30]))
        if len(self._compact_sources) >= 4096:
            self._flush_compact()

    def _flush_compact(self):
        if not self._compact_sources:
            return
        self.connection.executemany(
            'INSERT INTO sources(ordinal,source_ref,candidate_id,page_ref,close_path,matrix_body,'
            'drawing_ref,primitive_ref,item_index,part_index,kind,axis,'
            'native_start_x,native_start_y,native_end_x,native_end_y,control_body,sample_body,'
            'native_x0,native_y0,native_x1,native_y1,native_metadata_body,points_body,search_refs_body,'
            'descriptor_ordinal) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            self._compact_sources)
        self.connection.executemany(
            'INSERT INTO bounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', self._compact_bounds)
        self.connection.executemany('INSERT INTO cells VALUES (?,?,?,?,?)', self._compact_cells)
        self._compact_sources.clear()
        self._compact_bounds.clear()
        self._compact_cells.clear()

    def _compact_record(self, ref):
        row = self.connection.execute(
            'SELECT s.candidate_id,s.page_ref,s.close_path,s.matrix_body,s.drawing_ref,'
            's.primitive_ref,s.item_index,s.part_index,s.kind,s.axis,'
            's.native_start_x,s.native_start_y,s.native_end_x,s.native_end_y,'
            's.control_body,s.sample_body,s.native_x0,s.native_y0,s.native_x1,s.native_y1,'
            's.native_metadata_body,s.points_body,s.search_refs_body,'
            'b.source_length,b.start_x,b.start_y,b.end_x,b.end_y,'
            'b.style_body,b.gx0,b.gy0,b.gx1,b.gy1,b.x0,b.y0,b.x1,b.y1 '
            'FROM sources s JOIN bounds b USING (ordinal) WHERE s.source_ref=?', (ref,)).fetchone()
        if row is None:
            raise KeyError(ref)
        return self._compact_record_values(ref, row)

    @staticmethod
    def _compact_record_values(ref, row):
        (candidate_id, page_ref, close_path, matrix_body, drawing_ref, primitive_ref,
         item_index, part_index, kind, axis, nsx, nsy, nex, ney, control_body,
         sample_body, nx0, ny0, nx1, ny1, metadata_body, points_body, search_refs_body, source_length,
         sx, sy, ex, ey, style_body, gx0, gy0, gx1, gy1, x0, y0, x1, y1) = row
        native = {
            'id': ref, 'drawing_ref': drawing_ref, 'primitive_ref': primitive_ref,
            'item_index': item_index, 'part_index': part_index, 'kind': kind,
            'start_display': [nsx, nsy], 'end_display': [nex, ney],
            'control_points_display': json.loads(control_body) if control_body else [],
            'sample_points_display': json.loads(sample_body) if sample_body else [],
            'axis': axis, 'length_points': source_length,
            'bbox_display': [nx0, ny0, nx1, ny1], 'style': json.loads(style_body),
        }
        if metadata_body:
            native['native_metadata'] = json.loads(metadata_body)
        points = json.loads(points_body) if points_body else [[sx, sy], [ex, ey]]
        return {
            'id': candidate_id, 'page_ref': page_ref, 'source_primitive_ref': ref,
            'source_native_segment': native, 'source_drawing_close_path': bool(close_path),
            'points_display': points, 'bbox_display': [gx0, gy0, gx1, gy1],
            'search_bbox_display': [x0, y0, x1, y1], 'search_bbox_is_geometry': False,
            'pdf_to_display_matrix': json.loads(matrix_body),
            'search_refs': json.loads(search_refs_body),
            'state': 'observed', 'role': 'unclassified_native_geometry_candidate',
            'geometry_is_not_a_route_or_equipment': True, 'quantity_eligible': False,
        }

    def compact_records(self, refs):
        """Decode compact rows in bounded joins, avoiding one SQL query per row."""
        output = {}
        refs = list(refs)
        fields = (
            's.source_ref,s.candidate_id,s.page_ref,s.close_path,s.matrix_body,s.drawing_ref,'
            's.primitive_ref,s.item_index,s.part_index,s.kind,s.axis,'
            's.native_start_x,s.native_start_y,s.native_end_x,s.native_end_y,'
            's.control_body,s.sample_body,s.native_x0,s.native_y0,s.native_x1,s.native_y1,'
            's.native_metadata_body,s.points_body,s.search_refs_body,'
            'b.source_length,b.start_x,b.start_y,b.end_x,b.end_y,'
            'b.style_body,b.gx0,b.gy0,b.gx1,b.gy1,b.x0,b.y0,b.x1,b.y1 ')
        for start in range(0, len(refs), 500):
            batch = refs[start:start + 500]
            marks = ','.join('?' for _ in batch)
            for row in self.connection.execute(
                    'SELECT ' + fields + 'FROM sources s JOIN bounds b USING (ordinal) '
                    'WHERE s.source_ref IN (' + marks + ')', batch):
                output[row[0]] = self._compact_record_values(row[0], row[1:])
        if len(output) != len(set(refs)):
            raise KeyError('compact source batch is incomplete')
        return output

    def descriptor_ordinals(self, refs):
        """Return cold-pack ordinals for a bounded final evidence row set."""
        output = {}
        refs = list(refs)
        for start in range(0, len(refs), 500):
            batch = refs[start:start + 500]
            marks = ','.join('?' for _ in batch)
            for ref, ordinal in self.connection.execute(
                    'SELECT source_ref,descriptor_ordinal FROM sources '
                    'WHERE source_ref IN (' + marks + ')', batch):
                output[ref] = ordinal
        if len(output) != len(set(refs)):
            raise KeyError('compact source ordinal batch is incomplete')
        return output

    def __getitem__(self, ref):
        result = self.connection.execute('SELECT body FROM sources WHERE source_ref=?', (ref,)).fetchone()
        if result is None:
            raise KeyError(ref)
        return self._decode(result[0]) if result[0] is not None else self._compact_record(ref)

    @staticmethod
    def _decode(body):
        return json.loads(zlib.decompress(body))

    def candidate(self, candidate_id):
        result = self.connection.execute(
            'SELECT source_ref,body FROM sources WHERE candidate_id=?', (candidate_id,)).fetchone()
        if result is None:
            raise KeyError(candidate_id)
        return self._decode(result[1]) if result[1] is not None else self._compact_record(result[0])

    def candidate_bounds(self, candidate_id):
        self._flush_compact()
        result = self.connection.execute(
            'SELECT b.x0,b.y0,b.x1,b.y1 FROM sources s JOIN bounds b USING (ordinal) '
            'WHERE s.candidate_id=?', (candidate_id,)).fetchone()
        if result is None:
            raise KeyError(candidate_id)
        return {'search_bbox_display': list(result), 'bbox_display': list(result)}

    def candidate_bounds_many(self, candidate_ids):
        self._flush_compact()
        output = {}
        candidate_ids = list(candidate_ids)
        for start in range(0, len(candidate_ids), 500):
            batch = candidate_ids[start:start + 500]
            marks = ','.join('?' for _ in batch)
            for row in self.connection.execute(
                    'SELECT s.candidate_id,b.x0,b.y0,b.x1,b.y1 FROM sources s '
                    'JOIN bounds b USING (ordinal) WHERE s.candidate_id IN (' + marks + ')', batch):
                output[row[0]] = {'search_bbox_display': list(row[1:]),
                                  'bbox_display': list(row[1:])}
        if len(output) != len(set(candidate_ids)):
            raise KeyError('candidate bound batch is incomplete')
        return ((ref, output[ref]) for ref in candidate_ids)

    def __contains__(self, ref):
        return self.connection.execute('SELECT 1 FROM sources WHERE source_ref=?', (ref,)).fetchone() is not None

    def __iter__(self):
        return (row[0] for row in self.connection.execute('SELECT source_ref FROM sources ORDER BY ordinal'))

    def __len__(self):
        return self.connection.execute('SELECT count(*) FROM sources').fetchone()[0]

    def values(self):
        return (self._decode(body) if body is not None else self._compact_record(ref)
                for ref, body in self.connection.execute(
                    'SELECT source_ref,body FROM sources ORDER BY ordinal'))

    def items(self):
        return ((ref, self._decode(body) if body is not None else self._compact_record(ref))
                for ref, body in self.connection.execute(
            'SELECT source_ref,body FROM sources ORDER BY ordinal'))

    def has_candidate(self, ref):
        return self.connection.execute('SELECT 1 FROM sources WHERE candidate_id=?', (ref,)).fetchone() is not None

    def cell_rows(self, box, minimum_member_length=None):
        # R-tree float32 bounds are only a conservative accelerator. Binary64
        # values in ``bounds`` replay the authoritative inclusive predicate.
        sql = ('SELECT s.source_ref,s.body FROM cells c JOIN bounds b USING (ordinal) '
               'JOIN sources s USING (ordinal) WHERE '
               'c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
               'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=?')
        parameters = (box[2], box[0], box[3], box[1],
                      box[2], box[0], box[3], box[1])
        if minimum_member_length is not None:
            sql += ' AND b.member_length>=?'
            parameters += (minimum_member_length,)
        return self.connection.execute(sql, parameters)

    def member_lines(self, minimum_length):
        return (self._decode(body) if body is not None else self._compact_record(ref)
                for ref, body in self.connection.execute(
            'SELECT source_ref,body FROM bounds JOIN sources USING (ordinal) WHERE member_length>=? '
            'ORDER BY source_ref COLLATE BINARY',
            (minimum_length,)))

    @staticmethod
    def _descriptor(row):
        source_ref, length, start_x, start_y, end_x, end_y, style_body = row
        return {
            'source_primitive_ref': source_ref,
            'points_display': [[start_x, start_y], [end_x, end_y]],
            'source_native_segment': {
                'kind': 'line',
                'length_points': length,
                'style': json.loads(style_body),
            },
        }

    def member_descriptors(self, minimum_length):
        rows = self.connection.execute(
            'SELECT source_ref,member_length,start_x,start_y,end_x,end_y,style_body '
            'FROM bounds JOIN sources USING (ordinal) WHERE member_length>=? '
            'ORDER BY source_ref COLLATE BINARY', (minimum_length,))
        return (self._descriptor(row) for row in rows)

    def cell_member_descriptors(self, box, minimum_member_length):
        rows = self.connection.execute(
            'SELECT s.source_ref,b.member_length,b.start_x,b.start_y,b.end_x,b.end_y,b.style_body '
            'FROM cells c JOIN bounds b USING (ordinal) JOIN sources s USING (ordinal) '
            'WHERE c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
            'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=? '
            'AND b.member_length>=?', (box[2], box[0], box[3], box[1],
                box[2], box[0], box[3], box[1], minimum_member_length))
        return rows

    def cell_source_refs(self, box, maximum_source_length=None):
        sql = ('SELECT s.source_ref FROM cells c JOIN bounds b USING (ordinal) '
               'JOIN sources s USING (ordinal) WHERE '
               'c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
               'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=?')
        parameters = (box[2], box[0], box[3], box[1],
                      box[2], box[0], box[3], box[1])
        if maximum_source_length is not None:
            sql += ' AND b.source_length<=?'
            parameters += (maximum_source_length,)
        return self.connection.execute(sql, parameters)

    def source_ref_union_count(self, boxes, maximum_source_length, required_refs):
        """Count an exact multi-box source union without returning every row."""
        if len(boxes) == 2:
            def count_box(box):
                return self.connection.execute(
                    'SELECT count(*) FROM cells c JOIN bounds b USING (ordinal) WHERE '
                    'c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
                    'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=? '
                    'AND b.source_length<=?',
                    (box[2], box[0], box[3], box[1], box[2], box[0],
                     box[3], box[1], maximum_source_length)).fetchone()[0]

            left, right = boxes
            intersection = self.connection.execute(
                'SELECT count(*) FROM cells c JOIN bounds b USING (ordinal) WHERE '
                'c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
                'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=? '
                'AND c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
                'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=? '
                'AND b.source_length<=?',
                (left[2], left[0], left[3], left[1], left[2], left[0],
                 left[3], left[1], right[2], right[0], right[3], right[1],
                 right[2], right[0], right[3], right[1],
                 maximum_source_length)).fetchone()[0]
            required_refs = sorted(set(required_refs))
            outside = 0
            if required_refs:
                marks = ','.join('?' for _ in required_refs)
                outside = self.connection.execute(
                    'SELECT count(*) FROM sources s JOIN bounds b USING (ordinal) '
                    'WHERE s.source_ref IN (' + marks + ') AND NOT (b.source_length<=? AND ('
                    '(b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=?) OR '
                    '(b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=?)))',
                    (*required_refs, maximum_source_length,
                     left[2], left[0], left[3], left[1],
                     right[2], right[0], right[3], right[1])).fetchone()[0]
            return count_box(left) + count_box(right) - intersection + outside

        selects, parameters = [], []
        for box in boxes:
            selects.append(
                'SELECT c.ordinal FROM cells c JOIN bounds b USING (ordinal) WHERE '
                'c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
                'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=? '
                'AND b.source_length<=?')
            parameters.extend((box[2], box[0], box[3], box[1],
                               box[2], box[0], box[3], box[1],
                               maximum_source_length))
        required_refs = sorted(set(required_refs))
        required = ('SELECT ordinal FROM sources WHERE source_ref IN ('
                    + ','.join('?' for _ in required_refs) + ')')
        parameters.extend(required_refs)
        query = ('WITH queried(ordinal) AS (' + ' UNION '.join(selects)
                 + '), required(ordinal) AS (' + required
                 + ') SELECT count(*) FROM ('
                 + 'SELECT ordinal FROM queried UNION SELECT ordinal FROM required)')
        return self.connection.execute(query, parameters).fetchone()[0]

    def cell_geometry_descriptors(self, box):
        """Return only fields consumed by leader traversal and competitors."""
        return self.connection.execute(
            'SELECT s.source_ref,s.body,s.drawing_ref,s.kind,s.points_body,'
            'b.start_x,b.start_y,b.end_x,b.end_y,b.style_body,'
            'b.gx0,b.gy0,b.gx1,b.gy1,b.x0,b.y0,b.x1,b.y1 '
            'FROM cells c JOIN bounds b USING (ordinal) JOIN sources s USING (ordinal) '
            'WHERE c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
            'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=?',
            (box[2], box[0], box[3], box[1],
             box[2], box[0], box[3], box[1]))

    @staticmethod
    def _geometry_fields():
        return ('s.source_ref,s.body,s.drawing_ref,s.kind,s.points_body,'
                'b.start_x,b.start_y,b.end_x,b.end_y,b.style_body,'
                'b.gx0,b.gy0,b.gx1,b.gy1,b.x0,b.y0,b.x1,b.y1 ')

    def endpoint_geometry_descriptors(self, box):
        """Return sources with at least one exact display endpoint in ``box``."""
        fields = self._geometry_fields()
        parameters = (box[0], box[2], box[1], box[3])
        return self.connection.execute(
            'SELECT ' + fields + 'FROM bounds b JOIN sources s USING (ordinal) '
            'WHERE b.start_x>=? AND b.start_x<=? AND b.start_y>=? AND b.start_y<=? '
            'UNION SELECT ' + fields + 'FROM bounds b JOIN sources s USING (ordinal) '
            'WHERE b.end_x>=? AND b.end_x<=? AND b.end_y>=? AND b.end_y<=?',
            (*parameters, *parameters))

    def curve_geometry_descriptors(self, box):
        return self.connection.execute(
            'SELECT ' + self._geometry_fields()
            + 'FROM cells c JOIN bounds b USING (ordinal) JOIN sources s USING (ordinal) '
            "WHERE (s.kind='cubic' OR s.body IS NOT NULL) "
            'AND c.min_x<=? AND c.max_x>=? AND c.min_y<=? AND c.max_y>=? '
            'AND b.x0<=? AND b.x1>=? AND b.y0<=? AND b.y1>=?',
            (box[2], box[0], box[3], box[1],
             box[2], box[0], box[3], box[1]))

    @staticmethod
    def _geometry_descriptor(row):
        (source_ref, body, drawing_ref, kind, points_body, sx, sy, ex, ey,
         style_body, gx0, gy0, gx1, gy1, x0, y0, x1, y1) = row
        if body is not None:
            full = _DiskPrimitives._decode(body)
            native = full['source_native_segment']
            return {
                'source_primitive_ref': source_ref,
                'points_display': full['points_display'],
                'bbox_display': full['bbox_display'],
                'search_bbox_display': full.get('search_bbox_display', full['bbox_display']),
                'source_native_segment': {
                    'kind': native['kind'], 'drawing_ref': native['drawing_ref'],
                    'style': native.get('style') or {},
                },
            }
        return {
            'source_primitive_ref': source_ref,
            'points_display': json.loads(points_body) if points_body else [[sx, sy], [ex, ey]],
            'bbox_display': [gx0, gy0, gx1, gy1],
            'search_bbox_display': [x0, y0, x1, y1],
            'source_native_segment': {
                'kind': kind, 'drawing_ref': drawing_ref,
                'style': json.loads(style_body),
            },
        }

    def compact_topology(self, page_scope, refs, tolerance=.05):
        """Build connectivity once without materializing full M3 records."""
        self.connection.executescript('''
            DROP TABLE IF EXISTS topology_refs;
            DROP TABLE IF EXISTS topology_ordinals;
            CREATE TEMP TABLE topology_refs (source_ref TEXT COLLATE BINARY PRIMARY KEY) WITHOUT ROWID;
            CREATE TEMP TABLE topology_ordinals (ordinal INTEGER PRIMARY KEY) WITHOUT ROWID;
        ''')
        self.connection.executemany('INSERT INTO topology_refs VALUES (?)', ((ref,) for ref in refs))
        # Resolve string identities through the compact covering unique index,
        # then read the wide source/bounds rows in physical ordinal order.  A
        # direct source_ref-ordered join caused tens of GiB of random reads on
        # page 11 before returning the same deterministic source order.
        self.connection.execute(
            'INSERT INTO topology_ordinals '
            'SELECT s.ordinal FROM topology_refs t JOIN sources s USING (source_ref)')
        if self.connection.execute('SELECT count(*) FROM topology_ordinals').fetchone()[0] != len(refs):
            raise KeyError('compact topology source set is incomplete')
        self.connection.commit()
        parent = array('I')
        cells = defaultdict(list)
        topology, styles = {}, {}
        reconstruction_seconds = 0.0
        clustering_seconds = 0.0

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        rows = self.connection.execute(
            'SELECT s.source_ref,s.body,s.kind,s.points_body,'
            'round(b.start_x,6),round(b.start_y,6),round(b.end_x,6),round(b.end_y,6),'
            'b.style_body,round(sqrt('
            '(round(b.end_x,6)-round(b.start_x,6))*(round(b.end_x,6)-round(b.start_x,6))+'
            '(round(b.end_y,6)-round(b.start_y,6))*(round(b.end_y,6)-round(b.start_y,6))),6) '
            'FROM topology_ordinals t JOIN sources s USING (ordinal) '
            'JOIN bounds b USING (ordinal) ORDER BY s.source_ref COLLATE BINARY')
        for source_ref, body, kind, points_body, sx, sy, ex, ey, style_body, line_length in rows:
            phase_started = perf_counter()
            if body is not None:
                row = self._decode(body)
                source_points = row['points_display']
                style = row['source_native_segment'].get('style')
                points = [[round(float(value), 6) for value in point]
                          for point in source_points]
                path_length = round(sum(math.dist(left, right)
                    for left, right in zip(points, points[1:])), 6)
            elif kind != 'line':
                source_points = json.loads(points_body)
                style = json.loads(style_body)
                points = [[round(float(value), 6) for value in point]
                          for point in source_points]
                path_length = round(sum(math.dist(left, right)
                    for left, right in zip(points, points[1:])), 6)
            else:
                style = json.loads(style_body)
                points = [[sx, sy], [ex, ey]]
                path_length = line_length
            fragment_id = _stable_id('mep_route_primitive', page_scope['page_ref'],
                                     'native_pdf_vector', source_ref, points)
            style_key = json.dumps(_normalise_style(style),
                                   ensure_ascii=True, sort_keys=True, separators=(',', ':'))
            styles.setdefault(style_key, json.loads(style_key))
            reconstruction_seconds += perf_counter() - phase_started
            endpoint_ordinals = []
            phase_started = perf_counter()
            for point in (points[0], points[-1]):
                ordinal = len(parent)
                parent.append(ordinal)
                x, y = point
                cell = (math.floor(x / tolerance), math.floor(y / tolerance))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for other, other_x, other_y in cells.get((cell[0] + dx, cell[1] + dy), ()):
                            if math.hypot(x - other_x, y - other_y) <= tolerance:
                                union(ordinal, other)
                cells[cell].append((ordinal, x, y))
                endpoint_ordinals.append(ordinal)
            clustering_seconds += perf_counter() - phase_started
            topology[source_ref] = [fragment_id, *endpoint_ordinals, path_length, style_key]
        cells.clear()
        for row in topology.values():
            row[1], row[2] = find(row[1]), find(row[2])
        self.connection.executescript('DROP TABLE topology_refs; DROP TABLE topology_ordinals;')
        return topology, styles, reconstruction_seconds, clustering_seconds

    def close(self):
        self.connection.close()


class _PackedPrimitives(Mapping):
    """Exact native rows and transient spatial queries backed by mmap packs."""

    def __init__(self, pack, spatial, *, cap_path, member_ordinals=()):
        self.pack = pack
        self.spatial = spatial
        self.cap_path = Path(cap_path)
        self._descriptor_stream, self._descriptor_map = pack.open_mmap()
        self._payload_stream = pack.payload_path.open('rb')
        # Match the canonical JSON value types previously supplied by the
        # temporary SQLite row round-trip (notably RGB arrays, not tuples).
        self._styles = json.loads(json.dumps(pack.manifest['styles']))
        numbers_offset = RECORD.size - 21 * 8
        import numpy as np
        self._numpy = np
        self._search_x0 = np.ndarray(
            (len(pack),), dtype='<f8', buffer=self._descriptor_map,
            offset=numbers_offset + 16 * 8, strides=(RECORD.size,))
        self._search_y0 = np.ndarray(
            (len(pack),), dtype='<f8', buffer=self._descriptor_map,
            offset=numbers_offset + 17 * 8, strides=(RECORD.size,))
        self._search_x1 = np.ndarray(
            (len(pack),), dtype='<f8', buffer=self._descriptor_map,
            offset=numbers_offset + 18 * 8, strides=(RECORD.size,))
        self._search_y1 = np.ndarray(
            (len(pack),), dtype='<f8', buffer=self._descriptor_map,
            offset=numbers_offset + 19 * 8, strides=(RECORD.size,))
        self._lengths = np.ndarray(
            (len(pack),), dtype='<f8', buffer=self._descriptor_map,
            offset=numbers_offset + 20 * 8, strides=(RECORD.size,))
        self._member_ordinals = tuple(member_ordinals)
        self._cap_stream = None
        self._cap_offsets = None

    @staticmethod
    def _source_key(ref):
        try:
            drawing, rest = ref.removeprefix('drawing[').split('].item[', 1)
            item, part = rest.split('].segment[', 1)
            return int(drawing), int(item), int(part.removesuffix(']'))
        except (AttributeError, ValueError) as error:
            raise KeyError(ref) from error

    def _raw(self, ordinal):
        return self.pack.raw_at(ordinal, self._descriptor_map)

    def _ordinal_for_ref(self, ref, *, require_survivor=True):
        target = self._source_key(ref)
        low, high = 0, len(self.pack)
        while low < high:
            middle = (low + high) // 2
            key = self._raw(middle)[:3]
            if key < target:
                low = middle + 1
            else:
                high = middle
        if low >= len(self.pack) or self._raw(low)[:3] != target:
            raise KeyError(ref)
        if require_survivor and not self.spatial.contains(low):
            raise KeyError(ref)
        return low

    def _descriptor(self, ordinal):
        descriptor = self.pack._decode_descriptor(self._raw(ordinal))
        descriptor['descriptor_ordinal'] = ordinal
        return descriptor

    def _row(self, ordinal):
        descriptor = self._descriptor(ordinal)
        row = self.pack.row(descriptor, self._payload_stream)
        row['source_native_segment']['style'] = self._style(descriptor['style_id'])
        return row

    def _bounds(self, ordinal):
        raw = self._raw(ordinal)
        return raw[27:31], raw[31], raw[4], raw[3]

    def _style(self, style_id):
        return self._styles[style_id]

    def _is_member(self, raw, minimum_length):
        return (raw[4] == 1 and self._style(raw[3]).get('stroke') is not None
                and raw[31] >= minimum_length)

    def query_ordinals(self, box, predicate=None):
        return self.spatial.query(box, self._bounds, predicate)

    def query_many_ordinals(self, boxes, predicate=None):
        return self.spatial.query_many(boxes, self._bounds, predicate)

    def query_rows(self, box, minimum_member_length=None):
        predicate = None
        if minimum_member_length is not None:
            predicate = lambda ordinal, *_: self._is_member(
                self._raw(ordinal), minimum_member_length)
        rows = [self._row(ordinal) for ordinal in self.query_ordinals(box, predicate)]
        return sorted(rows, key=lambda row: row['source_primitive_ref'])

    def descriptor_ordinals(self, refs):
        return {ref: self._ordinal_for_ref(ref) for ref in refs}

    def member_lines(self, minimum_length):
        rows = [self._row(ordinal) for ordinal in self._member_ordinals
                if self._is_member(self._raw(ordinal), minimum_length)]
        return iter(sorted(rows, key=lambda row: row['source_primitive_ref']))

    def _member_descriptor(self, ordinal):
        descriptor = self._descriptor(ordinal)
        return {
            'source_primitive_ref': descriptor['source_primitive_ref'],
            'points_display': [descriptor['display_start'], descriptor['display_end']],
            'source_native_segment': {
                'kind': 'line', 'length_points': descriptor['length_points'],
                'style': self._style(descriptor['style_id']),
            },
        }

    def member_descriptors(self, minimum_length):
        rows = [self._member_descriptor(ordinal) for ordinal in self._member_ordinals
                if self._is_member(self._raw(ordinal), minimum_length)]
        return iter(sorted(rows, key=lambda row: row['source_primitive_ref']))

    def query_member_descriptors(self, box, minimum_length):
        ordinals = self.query_ordinals(
            box, lambda ordinal, *_: self._is_member(self._raw(ordinal), minimum_length))
        return sorted((self._member_descriptor(ordinal) for ordinal in ordinals),
                      key=lambda row: row['source_primitive_ref'])

    def query_source_refs(self, box, maximum_source_length=None):
        predicate = None if maximum_source_length is None else (
            lambda ordinal, exact_box, length, kind, style: length <= maximum_source_length)
        return sorted(self._descriptor(ordinal)['source_primitive_ref']
                      for ordinal in self.query_ordinals(box, predicate))

    def source_ref_union_count(self, boxes, maximum_source_length, required_refs):
        np = self._numpy
        required_ordinals = {self._ordinal_for_ref(ref) for ref in set(required_refs)}
        found_required = set()
        count = 0
        # Anchor-cell ranges and the overflow range are disjoint: each
        # survivor occurs exactly once. Ultra-dense pages store bounds beside
        # each range; smaller pages gather exact values from the native pack
        # and avoid retaining a duplicate numeric descriptor array.
        if self.spatial.has_inline_query_descriptors:
            ranges = self.spatial.query_ranges(boxes)
        else:
            ranges = ((offset, None, range_count)
                      for offset, range_count in self.spatial.ordinal_ranges(boxes))
        for ordinal_offset, descriptor_offset, range_count in ranges:
            ordinals = np.frombuffer(
                self.spatial._map, dtype='<u4', count=range_count,
                offset=ordinal_offset)
            if descriptor_offset is None:
                x0, y0 = self._search_x0[ordinals], self._search_y0[ordinals]
                x1, y1 = self._search_x1[ordinals], self._search_y1[ordinals]
                matched = self._lengths[ordinals] <= maximum_source_length
            else:
                descriptors = np.ndarray(
                    (range_count, 5), dtype='<f8', buffer=self.spatial._map,
                    offset=descriptor_offset)
                x0, y0 = descriptors[:, 0], descriptors[:, 1]
                x1, y1 = descriptors[:, 2], descriptors[:, 3]
                matched = descriptors[:, 4] <= maximum_source_length
            intersects = np.zeros(range_count, dtype=bool)
            for box in boxes:
                intersects |= ((x0 <= box[2]) & (x1 >= box[0])
                               & (y0 <= box[3]) & (y1 >= box[1]))
            selected = ordinals[matched & intersects]
            count += len(selected)
            for ordinal in required_ordinals - found_required:
                if np.any(selected == ordinal):
                    found_required.add(ordinal)
        return count + len(required_ordinals - found_required)

    def _geometry_descriptor(self, ordinal):
        descriptor = self._descriptor(ordinal)
        if descriptor['kind'] == 'line':
            points = [descriptor['display_start'], descriptor['display_end']]
        else:
            points = self._row(ordinal)['points_display']
        return {
            'source_primitive_ref': descriptor['source_primitive_ref'],
            'points_display': points,
            'bbox_display': descriptor['bbox_display'],
            'search_bbox_display': descriptor['search_bbox_display'],
            'source_native_segment': {
                'kind': descriptor['kind'],
                'drawing_ref': f"drawing[{descriptor['drawing_ordinal']}]",
                'style': self._style(descriptor['style_id']),
            },
        }

    def query_geometry_descriptors(self, box):
        return sorted((self._geometry_descriptor(ordinal)
                       for ordinal in self.query_ordinals(box)),
                      key=lambda row: row['source_primitive_ref'])

    def query_endpoint_descriptors(self, box):
        def endpoint_inside(ordinal, *_):
            raw = self._raw(ordinal)
            return any(box[0] <= x <= box[2] and box[1] <= y <= box[3]
                       for x, y in ((raw[15], raw[16]), (raw[17], raw[18])))
        return sorted((self._geometry_descriptor(ordinal)
                       for ordinal in self.query_ordinals(box, endpoint_inside)),
                      key=lambda row: row['source_primitive_ref'])

    def query_leader_adjacency(self, point, search_box, endpoint_box):
        endpoint_ordinals = self.query_ordinals(
            endpoint_box, lambda ordinal, *_: any(
                math.dist(point, endpoint) <= .05 for endpoint in (
                    self._raw(ordinal)[15:17], self._raw(ordinal)[17:19])))
        curve_ordinals = self.query_ordinals(
            search_box, lambda ordinal, exact_box, length, kind, style: kind == 2)
        return sorted((self._geometry_descriptor(ordinal)
                       for ordinal in endpoint_ordinals | curve_ordinals),
                      key=lambda row: row['source_primitive_ref'])

    def __getitem__(self, ref):
        return self._row(self._ordinal_for_ref(ref))

    def __contains__(self, ref):
        try:
            self._ordinal_for_ref(ref)
            return True
        except KeyError:
            return False

    def __iter__(self):
        return (self._descriptor(ordinal)['source_primitive_ref']
                for ordinal in self.spatial.survivors())

    def __len__(self):
        return self.spatial.survivor_count

    def values(self):
        return (self._row(ordinal) for ordinal in self.spatial.survivors())

    def items(self):
        return ((row['source_primitive_ref'], row) for row in self.values())

    def has_candidate(self, candidate_id):
        digest = bytes.fromhex(candidate_id.rsplit('.', 1)[-1])
        return any(self._raw(ordinal)[8] == digest
                   for ordinal in self.spatial.survivors())

    def compact_topology(self, page_scope, refs, tolerance=.05):
        parent = array('I')
        cells = defaultdict(list)
        topology, styles = {}, {}
        reconstruction_seconds = clustering_seconds = 0.0

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for source_ref in sorted(refs):
            phase_started = perf_counter()
            ordinal = self._ordinal_for_ref(source_ref)
            descriptor = self._descriptor(ordinal)
            if descriptor['kind'] == 'line':
                points = [[round(value, 6) for value in descriptor['display_start']],
                          [round(value, 6) for value in descriptor['display_end']]]
                path_length = round(math.dist(*points), 6)
            else:
                source_points = self._row(ordinal)['points_display']
                points = [[round(float(value), 6) for value in point]
                          for point in source_points]
                path_length = round(sum(math.dist(left, right)
                    for left, right in zip(points, points[1:])), 6)
            style = self._style(descriptor['style_id'])
            fragment_id = _stable_id('mep_route_primitive', page_scope['page_ref'],
                                     'native_pdf_vector', source_ref, points)
            style_key = json.dumps(_normalise_style(style), ensure_ascii=True,
                                   sort_keys=True, separators=(',', ':'))
            styles.setdefault(style_key, json.loads(style_key))
            reconstruction_seconds += perf_counter() - phase_started
            endpoint_ordinals = []
            phase_started = perf_counter()
            for point in (points[0], points[-1]):
                endpoint_ordinal = len(parent)
                parent.append(endpoint_ordinal)
                x, y = point
                cell = (math.floor(x / tolerance), math.floor(y / tolerance))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for other, other_x, other_y in cells.get(
                                (cell[0] + dx, cell[1] + dy), ()):
                            if math.hypot(x - other_x, y - other_y) <= tolerance:
                                union(endpoint_ordinal, other)
                cells[cell].append((endpoint_ordinal, x, y))
                endpoint_ordinals.append(endpoint_ordinal)
            clustering_seconds += perf_counter() - phase_started
            topology[source_ref] = [fragment_id, *endpoint_ordinals,
                                    path_length, style_key]
        cells.clear()
        for row in topology.values():
            row[1], row[2] = find(row[1]), find(row[2])
        return topology, styles, reconstruction_seconds, clustering_seconds

    def begin_cap_source_batches(self):
        if self.cap_path.exists():
            raise ValueError('packed cap batch path must be new')
        self._cap_stream = self.cap_path.open('xb+')
        self._cap_offsets = {}

    def store_cap_source_batch(self, search_ordinal, refs):
        ordinals = array('I', (self._ordinal_for_ref(ref) for ref in sorted(refs)))
        offset = self._cap_stream.tell()
        self._cap_stream.write(ordinals.tobytes())
        self._cap_offsets[search_ordinal] = (offset, len(ordinals))

    def _cap_ordinals(self, search_ordinal):
        offset, count = self._cap_offsets[search_ordinal]
        self._cap_stream.seek(offset)
        body = self._cap_stream.read(count * 4)
        if len(body) != count * 4:
            raise ValueError('truncated packed cap batch')
        values = array('I')
        values.frombytes(body)
        return values

    def cap_source_batch(self, search_ordinal):
        return tuple(self._descriptor(ordinal)['source_primitive_ref']
                     for ordinal in self._cap_ordinals(search_ordinal))

    def cap_source_union(self):
        present = bytearray((len(self.pack) + 7) // 8)
        for search_ordinal in sorted(self._cap_offsets):
            for ordinal in self._cap_ordinals(search_ordinal):
                byte, bit = divmod(ordinal, 8)
                present[byte] |= 1 << bit
        refs = [self._descriptor(ordinal)['source_primitive_ref']
                for ordinal in range(len(self.pack))
                if present[ordinal // 8] & (1 << (ordinal % 8))]
        return sorted(refs)

    def end_cap_source_batches(self):
        if self._cap_stream is not None:
            self._cap_stream.close()
            self._cap_stream = None
        self._cap_offsets = None
        try:
            self.cap_path.unlink()
        except FileNotFoundError:
            pass

    def close(self):
        self.end_cap_source_batches()
        self._payload_stream.close()
        self._descriptor_map.close()
        self._descriptor_stream.close()
        self.spatial.close()


class RegionIndex:
    """Spatial query over immutable retained sources, with coverage witnesses.

    Optional sqlite_path keeps source bodies/cells on disk. Input may stream
    while populating regions, which are validated after consumption. The caller
    owns the new/empty database path and its deletion; close or context exit
    releases it. Region evidence and complete local query results remain in RAM.
    """

    def __init__(self, *, primitives, regions, page_size, cell_size=64, sqlite_path=None):
        self._disk = _DiskPrimitives(sqlite_path) if sqlite_path is not None else None
        self._packed = None
        self.primitives = self._disk if self._disk is not None else {}
        self.page_size = page_size
        self.cell_size = cell_size
        self.cells = defaultdict(set)
        self.region_cells = defaultdict(list)
        self._cap_sources = None
        self._partition_coverage = None
        self._selection_boxes = None
        self._cold_envelope_plans = None
        self._cold_envelope_candidate_pair_count = None
        self._all_member_minimum = None
        self._initial_region_membership = None
        candidate_ids = {}
        try:
            for row in primitives:
                ref = row['source_primitive_ref']
                if row['source_native_segment']['kind'] == 'cubic' and 'search_bbox_display' not in row:
                    raise ValueError('curved source requires conservative search extent')
                cells = _cells(self._clip(row.get('search_bbox_display', row['bbox_display'])), cell_size)
                if self._disk is not None:
                    self._disk.add(row, cells)
                else:
                    if ref in self.primitives and self.primitives[ref] != row:
                        raise ValueError('conflicting duplicate native source')
                    if row['id'] in candidate_ids and candidate_ids[row['id']] != ref:
                        raise ValueError('conflicting duplicate native candidate ID')
                    self.primitives[ref] = row
                    candidate_ids[row['id']] = ref
                    for cell in cells:
                        self.cells[cell].add(ref)
            if self._disk is not None:
                self._disk.connection.commit()
            self._validate_regions(regions, candidate_ids)
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_native_page(cls, *, page, page_ref, page_size, sqlite_path,
                         region_size_display_points=64,
                         max_candidates_per_region=4000,
                         minimum_region_size_display_points=8,
                         cell_size=64, performance_profile=None,
                         observations=(), minimum_member_length=24):
        """Build a survivor-only cold index from a sequential descriptor pack."""
        self = cls.__new__(cls)
        self._disk = None
        self._packed = None
        self.primitives = {}
        self.page_size = page_size
        self.cell_size = cell_size
        self.cells = defaultdict(set)
        self.region_cells = defaultdict(list)
        self._cap_sources = None
        self._partition_coverage = None
        self._selection_boxes = None
        self._cold_envelope_plans = None
        self._cold_envelope_candidate_pair_count = None
        self._all_member_minimum = .98 * minimum_member_length
        self._initial_region_membership = None
        profile = performance_profile if performance_profile is not None else {}
        descriptor_path = Path(sqlite_path).with_suffix('.native-descriptors.pack')
        spatial_path = Path(sqlite_path).with_suffix('.packed-spatial')
        writer = NativeDescriptorPackWriter(
            descriptor_path, page_ref=page_ref,
            pdf_to_display_matrix=list(page.rotation_matrix),
            minimum_member_length=self._all_member_minimum,
            cell_size=minimum_region_size_display_points)
        try:
            scan_started = perf_counter()
            scanned = _bounded_page_records(
                page, page_ref, region_size_display_points,
                max_candidates_per_region, candidate_sink=writer.append,
                include_unretained=False, stream_native_drawings=True,
                sink_all_candidates=True)
            pack_manifest = writer.finish()
            pack = NativeDescriptorPack(descriptor_path, pack_manifest)
            profile['native_descriptor_scan_seconds'] = perf_counter() - scan_started
            profile['native_drawing_record_count'] = scanned['native_drawing_record_count']
            if not pack.verify_hashes():
                raise ValueError('native descriptor pack hash verification failed')

            refinement_started = perf_counter()
            regions, partition_coverage, refinement_count, retention_max_ordinals = (
                _refined_regions_from_descriptor_pack(
                    scanned['regions'], pack.descriptors, page_ref=page_ref,
                    page_size=page_size,
                    max_candidates_per_region=max_candidates_per_region,
                    minimum_region_size_display_points=minimum_region_size_display_points,
                    initial_retention_max_ordinal=
                        writer.region_retention_max_ordinal))
            profile['region_refinement_seconds'] = perf_counter() - refinement_started
            profile['refinement_candidate_decode_count'] = refinement_count
            profile['region_count'] = len(regions)

            # Serialized primitive evidence historically carries membership in
            # the coarse regions from the first scan, even when those regions
            # are subsequently replaced by refined leaves.  Preserve that
            # exact compatibility field independently from the refined query
            # partition; otherwise a boundary source silently changes IDs in
            # an otherwise semantically identical cold run.
            initial_region_membership = defaultdict(list)
            for region in scanned['regions']:
                threshold = writer.region_retention_max_ordinal.get(region['id'])
                if threshold is None:
                    continue
                entry = (region['id'], region['bbox_display'], threshold)
                for region_cell in _cells(region['bbox_display'], region_size_display_points):
                    initial_region_membership[region_cell].append(entry)
            self._initial_region_membership = (
                region_size_display_points, dict(initial_region_membership))

            planning_started = perf_counter()
            selection_boxes, leader_refs, envelope_plans, selection_counts = _cold_selection_plan(
                pack, writer.long_members, observations,
                cell_size=minimum_region_size_display_points)
            # Pair plans can request the same immutable cap window thousands
            # of times. Remove exact duplicates only; preserve first-occurrence
            # order, binary64 bounds and the source-length predicate.
            selection_boxes = list(dict.fromkeys(
                (tuple(box), limit) for box, limit in selection_boxes))
            selection_profile = profile.setdefault('selection_lookup', {})
            selection_lookup = _box_lookup(
                selection_boxes, cell_size=minimum_region_size_display_points,
                performance_profile=selection_profile)
            long_ordinals = {row['descriptor_ordinal'] for row in writer.long_members}
            profile['selection_planning_seconds'] = perf_counter() - planning_started
            profile.update(selection_counts)

            selection_started = perf_counter()
            survivor_count = 0
            profile['descriptor_selection_scan_seconds'] = 0.0
            profile['selection_descriptor_decode_count'] = 0
            profile['index_descriptor_recovery_count'] = 0

            def selected_raws():
                nonlocal survivor_count
                if not len(pack):
                    return
                descriptor_stream, descriptor_map = pack.open_mmap()
                try:
                    active_started = perf_counter()
                    previous_source_key = None
                    for descriptor_ordinal in range(len(pack)):
                        raw = pack.raw_at(descriptor_ordinal, descriptor_map)
                        profile['selection_descriptor_decode_count'] += 1
                        source_key = raw[:3]
                        if (previous_source_key is not None
                                and source_key <= previous_source_key):
                            raise ValueError(
                                'native descriptor source ordinals are not strictly ordered')
                        previous_source_key = source_key
                        if (descriptor_ordinal not in long_ordinals
                                and not selection_lookup(raw[27:31], raw[31])):
                            continue
                        survivor_count += 1
                        profile['descriptor_selection_scan_seconds'] += perf_counter() - active_started
                        yield descriptor_ordinal, raw
                        active_started = perf_counter()
                    profile['descriptor_selection_scan_seconds'] += perf_counter() - active_started
                finally:
                    descriptor_map.close()
                    descriptor_stream.close()

            if len(pack) >= PACKED_SPATIAL_MIN_SOURCE_COUNT:
                lookup_stream, lookup_map = pack.open_mmap()
                try:
                    def descriptor_lookup(ordinal):
                        profile['index_descriptor_recovery_count'] += 1
                        raw = pack.raw_at(ordinal, lookup_map)
                        return raw[27:31], raw[31]

                    spatial = PackedSpatialIndex.build(
                        spatial_path, descriptor_count=len(pack),
                        # The evidence-region grid may stay coarse, but
                        # dense-page primitive lookup needs the bounded
                        # refinement cell.
                        cell_size=minimum_region_size_display_points,
                        descriptors=((ordinal, raw[27:31], raw[31])
                                     for ordinal, raw in selected_raws()),
                        descriptor_lookup=descriptor_lookup)
                finally:
                    lookup_map.close()
                    lookup_stream.close()
                self._packed = _PackedPrimitives(
                    pack, spatial,
                    cap_path=Path(sqlite_path).with_suffix('.packed-cap-ordinals'),
                    member_ordinals=(row['descriptor_ordinal']
                                     for row in writer.long_members))
                self._disk = self._packed
                self.primitives = self._packed
                profile['sqlite_commit_seconds'] = 0.0
                profile['sqlite_insert_seconds'] = 0.0
                profile['temporary_spatial_backend'] = (
                    'packed_mmap_inline_cell_descriptors')
                profile['packed_spatial_bytes'] = spatial_path.stat().st_size
                profile['packed_spatial_cell_size'] = spatial.cell_size
                profile['packed_spatial_cell_count'] = spatial.cell_count
                profile['packed_spatial_entry_count'] = spatial.entry_count
                profile['packed_spatial_overflow_count'] = spatial.overflow_count
            else:
                self._disk = _DiskPrimitives(sqlite_path)
                self.primitives = self._disk
                insertion_started = perf_counter()
                styles = pack.manifest['styles']
                style_bodies = [json.dumps(
                    style, separators=(',', ':'), sort_keys=True)
                    for style in styles]
                matrix_body = json.dumps(
                    pack.manifest['pdf_to_display_matrix'], separators=(',', ':'))
                with pack.payload_path.open('rb') as payload_stream:
                    for descriptor_ordinal, raw in selected_raws():
                        self._disk.add_native_descriptor(
                            raw, descriptor_ordinal, page_ref=page_ref,
                            matrix_body=matrix_body, styles=styles,
                            style_bodies=style_bodies,
                            payload_stream=payload_stream)
                self._disk._flush_compact()
                profile['sqlite_insert_seconds'] = (
                    perf_counter() - insertion_started)
                commit_started = perf_counter()
                self._disk.connection.commit()
                profile['sqlite_commit_seconds'] = perf_counter() - commit_started
                profile['temporary_spatial_backend'] = 'sqlite_bulk_rtree'
            profile['descriptor_selection_seconds'] = perf_counter() - selection_started
            profile['temporary_index_construction_seconds'] = (
                profile['descriptor_selection_seconds'] - profile['descriptor_selection_scan_seconds'])
            profile['native_descriptor_count'] = len(pack)
            profile['sqlite_survivor_count'] = survivor_count
            profile['excluded_descriptor_count'] = len(pack) - survivor_count
            profile['descriptor_pack_bytes'] = descriptor_path.stat().st_size
            profile['descriptor_payload_bytes'] = pack.payload_path.stat().st_size
            profile['descriptor_pack_sha256'] = pack_manifest['descriptor_sha256']
            profile['descriptor_payload_sha256'] = pack_manifest['payload_sha256']
            profile['partition_coverage_sha256'] = _sha256(partition_coverage)
            profile['competitor_coverage_sha256'] = _sha256([{
                'id': row['id'], 'bbox_display': row['bbox_display'],
                'candidate_count': row['candidate_count'],
                'primitive_candidate_refs': row['primitive_candidate_refs'],
                'relevant_competitor_search_complete': row['relevant_competitor_search_complete'],
            } for row in regions])
            self._partition_coverage = {row['region_ref']: row for row in partition_coverage}
            self._selection_boxes = None
            self._cold_envelope_plans = envelope_plans
            self._cold_envelope_candidate_pair_count = selection_counts[
                'partner_candidate_pair_count']
            del selection_lookup, selection_boxes, leader_refs, long_ordinals
            self._validate_regions(regions, {})
            return self
        except BaseException:
            writer.abort()
            self.close()
            raise

    def _validate_regions(self, regions, candidate_ids):
        page_size = self.page_size
        boxes = {tuple(row['bbox_display']) for row in regions}
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        if (not boxes or any(not (0 <= b[0] < b[2] <= page_size[0] and 0 <= b[1] < b[3] <= page_size[1]) for b in boxes)
                or not math.isclose(area, page_size[0] * page_size[1], rel_tol=1e-12)
                or len(boxes) != len(regions) or len({r['id'] for r in regions}) != len(regions)):
            raise ValueError('incomplete or overlapping native region partition')
        if self._partition_coverage is None:
            has_candidate = self._disk.has_candidate if self._disk is not None else candidate_ids.__contains__
            if any(not all(has_candidate(ref) for ref in row['primitive_candidate_refs']) for row in regions):
                raise ValueError('region references missing retained native evidence')
        else:
            for row in regions:
                receipt = self._partition_coverage.get(row['id'])
                if (receipt is None or receipt['bbox_display'] != row['bbox_display']
                        or receipt['candidate_count'] != row['candidate_count']
                        or receipt['retained_candidate_count'] != len(row['primitive_candidate_refs'])
                        or receipt['retained_candidate_refs_sha256']
                        != _sha256(row['primitive_candidate_refs'])):
                    raise ValueError('native region partition coverage receipt mismatch')
        self.regions = regions
        for region in regions:
            for cell in _cells(region['bbox_display'], self.cell_size):
                b = region['bbox_display']
                if any(min(b[2], r['bbox_display'][2]) > max(b[0], r['bbox_display'][0])
                       and min(b[3], r['bbox_display'][3]) > max(b[1], r['bbox_display'][1])
                       for r in self.region_cells[cell]):
                    raise ValueError('overlapping native region partition')
                self.region_cells[cell].append(region)

    def _clip(self, box):
        return [max(0, box[0]), max(0, box[1]), min(self.page_size[0], box[2]), min(self.page_size[1], box[3])]

    def query(self, box, *, minimum_member_length=None):
        refs, regions = set(), {}
        decoded = {}
        compact_refs = set()
        if self._packed is not None:
            decoded = {row['source_primitive_ref']: row
                       for row in self._packed.query_rows(box, minimum_member_length)}
        elif self._disk is not None:
            for ref, body in self._disk.cell_rows(box, minimum_member_length):
                if body is not None:
                    decoded[ref] = self._disk._decode(body)
                else:
                    compact_refs.add(ref)
        for cell in _cells(self._clip(box), self.cell_size):
            if self._disk is None:
                refs.update(self.cells.get(cell, ()))
            for region in self.region_cells.get(cell, ()):
                if _intersects(box, region['bbox_display']):
                    regions[region['id']] = region
        if compact_refs:
            decoded.update(self._disk.compact_records(compact_refs))
        rows = []
        for ref in sorted(decoded if self._disk is not None else refs):
            row = decoded[ref] if self._disk is not None else self.primitives[ref]
            if (_intersects(box, row.get('search_bbox_display', row['bbox_display']))
                    and (minimum_member_length is None or _member_line(row, minimum_member_length))):
                rows.append(row)
        # Regions are the complete disjoint page partition supplied by the
        # scanner, including regions with no eligible returned members. Source
        # bounds are clipped only for indexing, never geometry.
        complete = bool(regions) and all(row['relevant_competitor_search_complete'] for row in regions.values())
        return rows, complete, sorted(regions)

    def with_initial_search_refs(self, rows):
        """Hydrate exact first-scan region memberships for serialized evidence.

        Internal geometry and topology queries do not consume ``search_refs``.
        Deferring this compatibility field avoids rebuilding it for almost one
        million cold-path survivors and for every repeated spatial query. The
        refined leaves remain authoritative for completeness queries; this
        field replays only the immutable legacy source-row representation.
        """
        output = deepcopy(list(rows))
        if self._disk is None or self._initial_region_membership is None:
            return output
        region_size, membership = self._initial_region_membership
        ordinals = self._disk.descriptor_ordinals(
            row['source_primitive_ref'] for row in output)
        for row in output:
            box = row.get('search_bbox_display', row['bbox_display'])
            possible = {}
            for cell in _cells(self._clip(box), region_size):
                for region_ref, region_box, threshold in membership.get(cell, ()):
                    possible[region_ref] = (region_box, threshold)
            ordinal = ordinals[row['source_primitive_ref']]
            row['search_refs'] = sorted(
                region_ref for region_ref, (region_box, threshold) in possible.items()
                if ordinal <= threshold and _intersects(box, region_box))
        return output

    def iter_member_lines(self, minimum_length):
        if self._disk is not None:
            return self._disk.member_lines(minimum_length)
        return iter(sorted((row for row in self.primitives.values() if _member_line(row, minimum_length)),
                           key=lambda row: row['source_primitive_ref']))

    def preplanned_envelopes(self, minimum_member_length):
        """Return pass-one certificates only for the matching cold threshold."""
        if (self._cold_envelope_plans is None
                or not math.isclose(.98 * minimum_member_length,
                                    self._all_member_minimum,
                                    rel_tol=0, abs_tol=1e-12)):
            return None
        return (self._cold_envelope_plans,
                self._cold_envelope_candidate_pair_count)

    def release_preplanned_envelopes(self):
        """Release pass-one metrics before the much larger cap topology."""
        self._cold_envelope_plans = None
        self._cold_envelope_candidate_pair_count = None

    def iter_member_descriptors(self, minimum_length):
        if self._disk is not None:
            return self._disk.member_descriptors(minimum_length)
        return iter(sorted(({
            'source_primitive_ref': row['source_primitive_ref'],
            'points_display': [row['points_display'][0], row['points_display'][-1]],
            'source_native_segment': {
                'kind': 'line',
                'length_points': row['source_native_segment']['length_points'],
                'style': row['source_native_segment']['style'],
            },
        } for row in self.primitives.values() if _member_line(row, minimum_length)),
            key=lambda row: row['source_primitive_ref']))

    def _regions_for_box(self, box):
        regions = {}
        for cell in _cells(self._clip(box), self.cell_size):
            for region in self.region_cells.get(cell, ()):
                if _intersects(box, region['bbox_display']):
                    regions[region['id']] = region
        complete = bool(regions) and all(row['relevant_competitor_search_complete'] for row in regions.values())
        return complete, sorted(regions)

    def query_member_descriptors(self, box, minimum_member_length):
        if self._disk is None:
            rows, complete, regions = self.query(box, minimum_member_length=minimum_member_length)
            return [{
                'source_primitive_ref': row['source_primitive_ref'],
                'points_display': [row['points_display'][0], row['points_display'][-1]],
                'source_native_segment': {
                    'kind': 'line',
                    'length_points': row['source_native_segment']['length_points'],
                    'style': row['source_native_segment']['style'],
                },
            } for row in rows], complete, regions
        if self._packed is not None:
            complete, regions = self._regions_for_box(box)
            return self._packed.query_member_descriptors(
                box, minimum_member_length), complete, regions
        compact = {row[0]: row for row in self._disk.cell_member_descriptors(
            box, minimum_member_length)}
        complete, regions = self._regions_for_box(box)
        return [self._disk._descriptor(compact[ref]) for ref in sorted(compact)], complete, regions

    def query_source_refs(self, box, maximum_source_length=None):
        if self._disk is None:
            rows, complete, regions = self.query(box)
            refs = [row['source_primitive_ref'] for row in rows
                    if maximum_source_length is None
                    or row['source_native_segment']['length_points'] <= maximum_source_length]
            return refs, complete, regions
        if self._packed is not None:
            refs = self._packed.query_source_refs(box, maximum_source_length)
        else:
            refs = {row[0] for row in self._disk.cell_source_refs(
                box, maximum_source_length)}
        complete, regions = self._regions_for_box(box)
        return sorted(refs), complete, regions

    def source_ref_union_count(self, boxes, maximum_source_length, required_refs=()):
        """Count the exact union used by an incomplete cap witness."""
        if self._disk is not None:
            return self._disk.source_ref_union_count(
                boxes, maximum_source_length, required_refs)
        refs = set(required_refs)
        for box in boxes:
            rows, _, _ = self.query(box)
            refs.update(row['source_primitive_ref'] for row in rows
                        if row['source_native_segment']['length_points']
                        <= maximum_source_length)
        return len(refs)

    def query_geometry_descriptors(self, box):
        """Query compact geometry for leader search without full source rows."""
        if self._disk is None:
            rows, complete, regions = self.query(box)
            return [{
                'source_primitive_ref': row['source_primitive_ref'],
                'points_display': row['points_display'],
                'bbox_display': row['bbox_display'],
                'search_bbox_display': row.get('search_bbox_display', row['bbox_display']),
                'source_native_segment': {
                    'kind': row['source_native_segment']['kind'],
                    'drawing_ref': row['source_native_segment']['drawing_ref'],
                    'style': row['source_native_segment'].get('style') or {},
                },
            } for row in rows], complete, regions
        if self._packed is not None:
            complete, regions = self._regions_for_box(box)
            return self._packed.query_geometry_descriptors(box), complete, regions
        compact = {row[0]: row for row in self._disk.cell_geometry_descriptors(box)}
        complete, regions = self._regions_for_box(box)
        return [self._disk._geometry_descriptor(compact[ref])
                for ref in sorted(compact)], complete, regions

    def query_endpoint_descriptors(self, box):
        """Query only sources whose exact start or end lies in ``box``."""
        if self._disk is None:
            rows, _, _ = self.query(box)
            compact = [row for row in rows if any(
                box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
                for point in (row['points_display'][0], row['points_display'][-1]))]
            complete, regions = self._regions_for_box(box)
            return compact, complete, regions
        if self._packed is not None:
            complete, regions = self._regions_for_box(box)
            return self._packed.query_endpoint_descriptors(box), complete, regions
        raw = self._disk.endpoint_geometry_descriptors(box)
        compact = {row[0]: row for row in raw}
        complete, regions = self._regions_for_box(box)
        return [self._disk._geometry_descriptor(compact[ref])
                for ref in sorted(compact)], complete, regions

    def query_leader_adjacency_descriptors(self, point, search_radius):
        """Exact rows that can affect one leader endpoint decision."""
        search_box = _point_box(point, search_radius)
        endpoint_box = _point_box(point, .05)
        if self._disk is None:
            rows, complete, regions = self.query_geometry_descriptors(search_box)
            return [row for row in rows if (
                row['source_native_segment']['kind'] == 'cubic'
                or any(math.dist(point, endpoint) <= .05
                       for endpoint in (row['points_display'][0],
                                        row['points_display'][-1])))], complete, regions
        if self._packed is not None:
            complete, regions = self._regions_for_box(search_box)
            return self._packed.query_leader_adjacency(
                point, search_box, endpoint_box), complete, regions
        compact = {row[0]: row for row in self._disk.endpoint_geometry_descriptors(
            endpoint_box)}
        for row in self._disk.curve_geometry_descriptors(search_box):
            descriptor = self._disk._geometry_descriptor(row)
            if descriptor['source_native_segment']['kind'] == 'cubic':
                compact.setdefault(row[0], row)
        complete, regions = self._regions_for_box(search_box)
        return [self._disk._geometry_descriptor(compact[ref])
                for ref in sorted(compact)], complete, regions

    def begin_cap_source_batches(self):
        if self._packed is not None:
            self._packed.begin_cap_source_batches()
            return
        if self._disk is None:
            self._cap_sources = {}
            return
        self._disk.connection.executescript('''
            DROP TABLE IF EXISTS cap_sources;
            CREATE TABLE cap_sources (
                search_ordinal INTEGER PRIMARY KEY,
                source_refs_zlib BLOB NOT NULL);
        ''')

    def store_cap_source_batch(self, search_ordinal, refs):
        if self._packed is not None:
            self._packed.store_cap_source_batch(search_ordinal, refs)
            return
        if self._disk is None:
            self._cap_sources[search_ordinal] = tuple(sorted(refs))
            return
        body = json.dumps(sorted(refs), ensure_ascii=True, separators=(',', ':')).encode()
        self._disk.connection.execute('INSERT INTO cap_sources VALUES (?,?)',
                                      (search_ordinal, zlib.compress(body, 1)))

    def cap_source_batch(self, search_ordinal):
        if self._packed is not None:
            return self._packed.cap_source_batch(search_ordinal)
        if self._disk is None:
            return self._cap_sources[search_ordinal]
        body = self._disk.connection.execute(
            'SELECT source_refs_zlib FROM cap_sources WHERE search_ordinal=?',
            (search_ordinal,)).fetchone()[0]
        return tuple(json.loads(zlib.decompress(body)))

    def cap_source_union(self):
        if self._packed is not None:
            return self._packed.cap_source_union()
        if self._disk is None:
            return sorted({ref for refs in self._cap_sources.values() for ref in refs})
        self._disk.connection.commit()
        refs = set()
        for row in self._disk.connection.execute(
                'SELECT source_refs_zlib FROM cap_sources ORDER BY search_ordinal'):
            refs.update(json.loads(zlib.decompress(row[0])))
        return sorted(refs)

    def end_cap_source_batches(self):
        if self._packed is not None:
            self._packed.end_cap_source_batches()
            return
        if self._disk is None:
            self._cap_sources = None
            return
        self._disk.connection.executescript('DROP TABLE cap_sources;')

    def compact_cap_topology(self, page_scope, refs):
        if self._disk is None:
            return None
        return self._disk.compact_topology(page_scope, refs)

    def close(self):
        if self._disk is not None:
            self._disk.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def native(self, refs):
        output = []
        for ref in sorted(set(refs)):
            row = self.primitives[ref]
            native = deepcopy(row['source_native_segment'])
            points = row['points_display']
            native['start_display'], native['end_display'] = points[0], points[-1]
            native['bbox_display'] = row['bbox_display']
            if native['sample_points_display']:
                native['sample_points_display'] = points
            native['control_points_display'] = [list(fitz.Point(p) * fitz.Matrix(row['pdf_to_display_matrix']))
                                                 for p in native['control_points_display']]
            output.append(native)
        return output


def _fragment(row):
    native = row['source_native_segment']
    return {'id': row['source_primitive_ref'],
            'geometry': {'points_display': row['points_display']},
            'style': _normalise_style(native['style'])}


def _reachable_cap_fragments(fragments, left, right, limit):
    """Keep every compatible edge reachable within the complete cap bound.

    Endpoint vertices must already be clustered from ALL local originals. This
    does not re-cluster, choose a path or assert closure. Any admissible cap path
    starts at one left-member endpoint and has cumulative length <= limit, so
    each of its edges belongs to this conservative shortest-distance union.
    Both starts and every branching alternative survive; unrelated drawing
    clutter cannot consume the downstream reachable-source processing budget.
    """
    members = {left['id'], right['id']}
    edges = defaultdict(list)
    for row in fragments:
        vertices = row.get('endpoint_vertex_refs', [])
        if (row['id'] in members or len(vertices) != 2 or vertices[0] == vertices[1]
                or not _style_compatible(left.get('style', {}), row.get('style', {}))
                or not _provenance_compatible(left, row) or not _ownership_compatible(left, row)):
            continue
        length = float(row.get('geometry', {}).get('path_display_points') or 0)
        edges[vertices[0]].append((vertices[1], length, row['id']))
        edges[vertices[1]].append((vertices[0], length, row['id']))
    distances = {vertex: 0.0 for vertex in left.get('endpoint_vertex_refs', [])}
    queue = [(distance, vertex) for vertex, distance in distances.items()]
    heapq.heapify(queue)
    reachable = set(members)
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for target, length, ref in edges[vertex]:
            candidate = distance + length
            if candidate > limit:
                continue
            reachable.add(ref)
            if candidate < distances.get(target, math.inf):
                distances[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return [row for row in fragments if row['id'] in reachable]


def _profile_add(profile, phase, seconds, **counts):
    if profile is None:
        return
    phases = profile.setdefault('phases', {})
    row = phases.setdefault(phase, {'seconds': 0.0, 'calls': 0})
    row['seconds'] += seconds
    row['calls'] += 1
    totals = profile.setdefault('counts', {})
    for key, value in counts.items():
        totals[key] = totals.get(key, 0) + value


def _discover_envelopes_page_topology(*, index, page_scope, minimum_member_length,
                                      max_local_sources, performance_profile):
    """Replay the existing certificate over one shared bounded page topology."""
    if performance_profile is not None:
        performance_profile.update({
            'schema_version': '0.1.0',
            'layer': 'mep_envelope_discovery_performance',
            'page_ref': page_scope['page_ref'],
            'engine': 'page_endpoint_topology',
            'phases': {},
            'counts': {},
        })
    phase_started = perf_counter()
    members = list(index.iter_member_descriptors(.98 * minimum_member_length))
    _profile_add(performance_profile, 'source_iteration', perf_counter() - phase_started,
                 eligible_member_lines=len(members))
    preplanned = index.preplanned_envelopes(minimum_member_length)
    candidate_pairs, complete_by_member = set(), {}
    candidate_pair_count = 0
    witnesses, plans = [], []
    index.begin_cap_source_batches()

    def retain_plan(key, left, metrics, region_refs):
        mean = metrics['mean_separation_display_points']
        width = metrics['member_width_display_points']
        limit = max(2.5 * mean, 4 * width, 1)
        local_refs, cap_regions, cap_complete = set(key), set(region_refs), True
        boxes = [_point_box(point, limit + .1) for point in left['points_display']]
        for box in boxes:
            closed, regions = index._regions_for_box(box)
            cap_complete &= closed
            cap_regions.update(regions)
        if cap_complete:
            for box in boxes:
                phase_started = perf_counter()
                local, _, _ = index.query_source_refs(
                    box, maximum_source_length=limit + .1)
                _profile_add(performance_profile, 'spatial_queries', perf_counter() - phase_started,
                             spatial_query_count=1, cap_spatial_query_count=1,
                             spatial_query_result_rows=len(local))
                local_refs.update(local)
            source_count = len(local_refs)
        else:
            phase_started = perf_counter()
            source_count = index.source_ref_union_count(
                boxes, limit + .1, required_refs=local_refs)
            _profile_add(performance_profile, 'spatial_query_counts',
                         perf_counter() - phase_started,
                         spatial_count_query_count=1,
                         spatial_count_query_box_count=len(boxes),
                         spatial_count_query_result_union_count=source_count)
        record = {'id': _stable_id('mep_automatic_envelope_search', page_scope['page_ref'], key),
                  'page_ref': page_scope['page_ref'], 'member_source_primitive_refs': list(key),
                  'derived_geometry': {'centreline_points_display': [
                      [(p + q) / 2 for p, q in zip(left_point, right_point)]
                      for left_point, right_point in zip(metrics['left_samples'], metrics['right_samples'])]},
                  'geometry_metrics': {'mean_separation_display_points': mean},
                  'region_refs': sorted(cap_regions), 'cap_search_complete': cap_complete,
                  'cap_candidate_source_count': source_count, 'cap_source_reachability': None,
                  'competitor_search_complete': False, 'closure_source_primitive_refs': [],
                  'state': 'unresolved', 'quantity_eligible': False}
        witnesses.append(record)
        if cap_complete:
            ordinal = len(plans)
            phase_started = perf_counter()
            index.store_cap_source_batch(ordinal, local_refs)
            _profile_add(performance_profile, 'cap_batch_storage', perf_counter() - phase_started,
                         cap_batch_store_count=1,
                         cap_batch_stored_source_instance_count=len(local_refs))
            # Closure needs only orientation and the two scalar cap-limit
            # inputs. Do not keep sampled pair geometry alive while the shared
            # half-million-source topology is materialized.
            closure_metrics = {
                'right_reversed': metrics['right_reversed'],
                'mean_separation_display_points': mean,
                'member_width_display_points': width,
            }
            plans.append((key, closure_metrics, limit, ordinal, record))

    if preplanned is not None:
        prepared, candidate_pair_count = preplanned
        member_by_ref = {row['source_primitive_ref']: row for row in members}
        region_refs_by_member = {}
        phase_started = perf_counter()
        for left in members:
            radius = .216 * left['source_native_segment']['length_points'] + .35
            complete, region_refs = index._regions_for_box(
                _point_box(left['points_display'][0], radius))
            complete_by_member[left['source_primitive_ref']] = complete
            region_refs_by_member[left['source_primitive_ref']] = region_refs
        _profile_add(performance_profile, 'coverage_queries', perf_counter() - phase_started,
                     coverage_query_count=len(members))
        _profile_add(performance_profile, 'parallel_pair_tests', 0,
                     candidate_pair_count=candidate_pair_count,
                     parallel_pair_test_count=candidate_pair_count,
                     cap_search_count=len(prepared), preplanned_pair_count=len(prepared))
        for plan in prepared:
            key = tuple(plan['member_source_primitive_refs'])
            retain_plan(key, member_by_ref[key[0]], plan['metrics'],
                        region_refs_by_member[key[0]])
        index.release_preplanned_envelopes()
        del prepared, preplanned, member_by_ref, region_refs_by_member
    else:
        for left in members:
            lref = left['source_primitive_ref']
            radius = .216 * left['source_native_segment']['length_points'] + .35
            phase_started = perf_counter()
            around, complete, region_refs = index.query_member_descriptors(
                _point_box(left['points_display'][0], radius), .98 * minimum_member_length)
            _profile_add(performance_profile, 'spatial_queries', perf_counter() - phase_started,
                         spatial_query_count=1, spatial_query_result_rows=len(around))
            complete_by_member[lref] = complete
            for right in around:
                rref = right['source_primitive_ref']
                if rref == lref:
                    continue
                key = tuple(sorted((lref, rref)))
                if key in candidate_pairs:
                    continue
                candidate_pairs.add(key)
                phase_started = perf_counter()
                a, b = _fragment(left), _fragment(right)
                if not _style_compatible(a['style'], b['style']):
                    _profile_add(performance_profile, 'parallel_pair_tests', perf_counter() - phase_started,
                                 candidate_pair_count=1, style_rejected_pair_count=1)
                    continue
                metrics = _parallel_metrics(a, b)
                mean, width = metrics['mean_separation_display_points'], metrics['member_width_display_points']
                length = min(metrics['left_length_display_points'], metrics['right_length_display_points'])
                if not (length >= max(10 * width, 5 * mean)
                        and metrics['length_ratio'] >= .98
                        and max(.1, .15 * width) < mean <= max(12 * width, .08 * length)
                        and metrics['maximum_separation_deviation_display_points'] <= max(.35, .08 * mean)
                        and metrics['maximum_tangent_difference_degrees'] <= 2
                        and all(abs(value - mean) <= max(.35, .08 * mean)
                                for value in metrics['endpoint_separations_display_points'])):
                    _profile_add(performance_profile, 'parallel_pair_tests', perf_counter() - phase_started,
                                 candidate_pair_count=1, parallel_pair_test_count=1,
                                 geometric_rejected_pair_count=1)
                    continue
                _profile_add(performance_profile, 'parallel_pair_tests', perf_counter() - phase_started,
                             candidate_pair_count=1, parallel_pair_test_count=1, cap_search_count=1)
                retain_plan(key, left, metrics, region_refs)
        candidate_pair_count = len(candidate_pairs)

    by_source = {}
    compact_topology = None
    compact_styles = None
    phase_started = perf_counter()
    topology_refs = index.cap_source_union()
    _profile_add(performance_profile, 'topology_union', perf_counter() - phase_started,
                 topology_union_count=1, topology_union_source_count=len(topology_refs))
    topology_source_count = len(topology_refs)
    if topology_refs:
        phase_started = perf_counter()
        compact = index.compact_cap_topology(page_scope, topology_refs)
        if compact is not None:
            compact_topology, compact_styles, reconstruction_seconds, clustering_seconds = compact
            topology_elapsed = perf_counter() - phase_started
            _profile_add(performance_profile, 'native_reconstruction', reconstruction_seconds,
                         native_reconstruction_count=1,
                         native_reconstructed_source_count=len(compact_topology),
                         logical_native_source_instance_count=sum(
                             plan[4]['cap_candidate_source_count'] for plan in plans))
            _profile_add(performance_profile, 'endpoint_clustering', clustering_seconds,
                         endpoint_clustering_count=1,
                         endpoint_clustering_source_count=len(compact_topology))
            _profile_add(performance_profile, 'topology_materialization',
                         max(0.0, topology_elapsed - reconstruction_seconds - clustering_seconds),
                         topology_materialization_count=1)
        else:
            phase_started = perf_counter()
            native_candidates = _native_candidates(page_scope, {'segments': index.native(topology_refs)})
            _profile_add(performance_profile, 'native_reconstruction', perf_counter() - phase_started,
                         native_reconstruction_count=1,
                         native_reconstructed_source_count=len(native_candidates),
                         logical_native_source_instance_count=sum(
                             plan[4]['cap_candidate_source_count'] for plan in plans))
            phase_started = perf_counter()
            _, assignments = _cluster_endpoints(page_scope['page_ref'], native_candidates, .05)
            _profile_add(performance_profile, 'endpoint_clustering', perf_counter() - phase_started,
                         endpoint_clustering_count=1,
                         endpoint_clustering_source_count=len(native_candidates))
            fragments, _ = _fragment_records(page_scope, None, native_candidates, assignments)
            by_source = {row['source_primitive_ref']: row for row in fragments}
    del topology_refs

    evidence = {}
    compact_provenance = {'method': 'native_pdf_vector_topology',
                          'coordinate_space': 'rotation-normalized PyMuPDF page display coordinates'}
    compact_ownership = {'page_ref': page_scope['page_ref'],
                         'view_scope_ref': page_scope.get('id') or page_scope['page_ref'],
                         'package_ref': None}
    for key, metrics, limit, ordinal, record in plans:
        phase_started = perf_counter()
        local_refs = index.cap_source_batch(ordinal)
        if compact_topology is not None:
            fragments = []
            for ref in local_refs:
                fragment_id, start_vertex, end_vertex, path_length, style_key = compact_topology[ref]
                fragments.append({
                    'id': fragment_id,
                    'source_primitive_ref': ref,
                    'endpoint_vertex_refs': [start_vertex, end_vertex],
                    'geometry': {'path_display_points': path_length},
                    'style': compact_styles[style_key],
                    'source_kind': 'native_pdf_vector',
                    'provenance': compact_provenance,
                    'ownership': compact_ownership,
                })
            by_source = {row['source_primitive_ref']: row for row in fragments}
        else:
            fragments = [by_source[ref] for ref in local_refs]
        # Match the canonical fragment-record ordering used by the legacy
        # per-search reconstruction. Stable order is part of shortest-path
        # tie-breaking even when the reachable source set is unchanged.
        fragments.sort(key=lambda row: row['id'])
        _profile_add(performance_profile, 'cap_batch_loading', perf_counter() - phase_started,
                     cap_batch_load_count=1,
                     cap_batch_loaded_source_instance_count=len(local_refs))
        phase_started = perf_counter()
        reachable = _reachable_cap_fragments(fragments, by_source[key[0]], by_source[key[1]], limit)
        _profile_add(performance_profile, 'cap_reachability', perf_counter() - phase_started,
                     cap_reachability_count=1,
                     cap_reachability_input_source_count=len(fragments),
                     cap_reachable_source_count=len(reachable))
        record['cap_source_reachability'] = {
            'method': 'complete_length_bounded_endpoint_reachability', 'version': '1.0.0',
            'maximum_path_display_points': limit, 'reachable_source_count': len(reachable),
            'input_source_refs_sha256': _sha256(sorted(local_refs)),
            'reachable_source_refs_sha256': _sha256(sorted(row['source_primitive_ref'] for row in reachable)),
            'full_source_endpoint_clustering_preserved': True,
            'all_admissible_cap_paths_preserved': True,
            'reachable_source_budget': max_local_sources,
        }
        if len(reachable) > max_local_sources:
            record['cap_search_complete'] = False
            continue
        phase_started = perf_counter()
        closure, refs = _shortest_cap_path({'fragments': reachable}, by_source[key[0]],
                                           by_source[key[1]], metrics)
        _profile_add(performance_profile, 'shortest_path_closure', perf_counter() - phase_started,
                     shortest_path_closure_count=1,
                     shortest_path_found_count=int(bool(closure)))
        if closure:
            by_id = {row['id']: row for row in reachable}
            record['closure_source_primitive_refs'] = sorted(
                by_id[ref]['source_primitive_ref'] for ref in refs)
            evidence[key] = record
        else:
            record['state'] = 'rejected_no_native_cap_path'

    supported = defaultdict(list)
    for key in evidence:
        for ref in key:
            supported[ref].append(key)
    # A member's competitor certificate is incomplete when any incident
    # witness has an incomplete cap search. Compute that reduction once. The
    # prior expression rescanned every witness for every surviving pair.
    cap_complete_by_member = {}
    for witness in witnesses:
        for ref in witness['member_source_primitive_refs']:
            cap_complete_by_member[ref] = (
                cap_complete_by_member.get(ref, True)
                and witness['cap_search_complete'])
    selected = set()
    for key, record in evidence.items():
        complete = all(complete_by_member.get(ref, False) for ref in key)
        complete &= all(cap_complete_by_member.get(ref, True) for ref in key)
        record['competitor_search_complete'] = complete
        if (complete and all(len(supported[ref]) == 1 for ref in key)
                and min(index.primitives[ref]['source_native_segment']['length_points']
                        for ref in key) >= minimum_member_length):
            record['state'] = 'closed_geometric_proposal'
            selected.update(key)
            selected.update(record['closure_source_primitive_refs'])
    if performance_profile is not None:
        performance_profile['counts'].update({
            'unique_candidate_pair_count': candidate_pair_count,
            'envelope_witness_count': len(witnesses),
            'selected_source_count': len(selected),
            'shared_topology_source_count': topology_source_count,
            'closed_geometric_proposal_count': sum(
                row['state'] == 'closed_geometric_proposal' for row in witnesses),
            'rejected_no_native_cap_path_count': sum(
                row['state'] == 'rejected_no_native_cap_path' for row in witnesses),
            'unresolved_envelope_count': sum(row['state'] == 'unresolved' for row in witnesses),
        })
    index.end_cap_source_batches()
    return selected, witnesses


def discover_envelopes(*, index, page_scope, minimum_member_length=24, max_local_sources=1200,
                       performance_profile=None, engine='per_candidate'):
    """Search every eligible line's full metric competitor neighbourhood.

    The 0.2*length bound follows the existing certificate's length >= 5*mean
    separation condition. Endpoint deviation adds 0.35 points. No competing
    pair can pass that certificate outside the searched endpoint rectangles.
    Cap searches cover the complete path-length ball, including neighbouring
    tiles and curves, with all original segments retained uncut.
    """
    if engine == 'page_topology':
        return _discover_envelopes_page_topology(
            index=index, page_scope=page_scope, minimum_member_length=minimum_member_length,
            max_local_sources=max_local_sources, performance_profile=performance_profile)
    if engine != 'per_candidate':
        raise ValueError('unknown envelope discovery engine')
    candidate_pairs, evidence, witnesses = set(), {}, []
    complete_by_member = {}
    if performance_profile is not None:
        performance_profile.update({
            'schema_version': '0.1.0',
            'layer': 'mep_envelope_discovery_performance',
            'page_ref': page_scope['page_ref'],
            'engine': 'per_candidate_topology',
            'phases': {},
            'counts': {},
        })
    members = index.iter_member_lines(.98 * minimum_member_length)
    while True:
        phase_started = perf_counter()
        try:
            left = next(members)
        except StopIteration:
            _profile_add(performance_profile, 'source_iteration', perf_counter() - phase_started)
            break
        _profile_add(performance_profile, 'source_iteration', perf_counter() - phase_started,
                     eligible_member_lines=1)
        lref = left['source_primitive_ref']
        radius = .216 * left['source_native_segment']['length_points'] + .35
        phase_started = perf_counter()
        around, complete, region_refs = index.query(_point_box(left['points_display'][0], radius),
                                                   minimum_member_length=.98 * minimum_member_length)
        _profile_add(performance_profile, 'spatial_queries', perf_counter() - phase_started,
                     spatial_query_count=1, spatial_query_result_rows=len(around))
        complete_by_member[lref] = complete
        for right in around:
            rref = right['source_primitive_ref']
            if rref == lref or not _member_line(right, .98 * minimum_member_length):
                continue
            key = tuple(sorted((lref, rref)))
            if key in candidate_pairs:
                continue
            candidate_pairs.add(key)
            phase_started = perf_counter()
            a, b = _fragment(left), _fragment(right)
            if not _style_compatible(a['style'], b['style']):
                _profile_add(performance_profile, 'parallel_pair_tests', perf_counter() - phase_started,
                             candidate_pair_count=1, style_rejected_pair_count=1)
                continue
            metrics = _parallel_metrics(a, b)
            mean, width = metrics['mean_separation_display_points'], metrics['member_width_display_points']
            length = min(metrics['left_length_display_points'], metrics['right_length_display_points'])
            if not (length >= max(10 * width, 5 * mean)
                    and metrics['length_ratio'] >= .98
                    and max(.1, .15 * width) < mean <= max(12 * width, .08 * length)
                    and metrics['maximum_separation_deviation_display_points'] <= max(.35, .08 * mean)
                    and metrics['maximum_tangent_difference_degrees'] <= 2
                    and all(abs(value - mean) <= max(.35, .08 * mean)
                            for value in metrics['endpoint_separations_display_points'])):
                _profile_add(performance_profile, 'parallel_pair_tests', perf_counter() - phase_started,
                             candidate_pair_count=1, parallel_pair_test_count=1,
                             geometric_rejected_pair_count=1)
                continue
            _profile_add(performance_profile, 'parallel_pair_tests', perf_counter() - phase_started,
                         candidate_pair_count=1, parallel_pair_test_count=1,
                         cap_search_count=1)
            limit = max(2.5 * mean, 4 * width, 1)
            local_refs, cap_regions, cap_complete = set(key), set(region_refs), True
            for point in left['points_display']:
                phase_started = perf_counter()
                local, closed, regions = index.query(_point_box(point, limit + .1))
                _profile_add(performance_profile, 'spatial_queries', perf_counter() - phase_started,
                             spatial_query_count=1, cap_spatial_query_count=1,
                             spatial_query_result_rows=len(local))
                cap_complete &= closed
                cap_regions.update(regions)
                local_refs.update(row['source_primitive_ref'] for row in local
                                  if row['source_native_segment']['length_points'] <= limit + .1)
            record = {'id': _stable_id('mep_automatic_envelope_search', page_scope['page_ref'], key),
                      'page_ref': page_scope['page_ref'], 'member_source_primitive_refs': list(key),
                      'derived_geometry': {'centreline_points_display': [
                          [(a + b) / 2 for a, b in zip(p, q)]
                          for p, q in zip(metrics['left_samples'], metrics['right_samples'])]},
                      'geometry_metrics': {'mean_separation_display_points': mean},
                      'region_refs': sorted(cap_regions), 'cap_search_complete': cap_complete,
                      'cap_candidate_source_count': len(local_refs), 'cap_source_reachability': None,
                      'competitor_search_complete': False, 'closure_source_primitive_refs': [],
                      'state': 'unresolved', 'quantity_eligible': False}
            witnesses.append(record)
            if not cap_complete:
                continue
            # Cap closure consumes fragments/endpoint vertices only. Computing
            # all local crossings, gaps and repetitions here changes no cap
            # decision and is quadratic on dense hatching. The final M3 graph
            # still computes and preserves those observations for its sources.
            phase_started = perf_counter()
            native_candidates = _native_candidates(page_scope, {'segments': index.native(local_refs)})
            _profile_add(performance_profile, 'native_reconstruction', perf_counter() - phase_started,
                         native_reconstruction_count=1,
                         native_reconstructed_source_count=len(native_candidates))
            phase_started = perf_counter()
            _, assignments = _cluster_endpoints(page_scope['page_ref'], native_candidates, .05)
            _profile_add(performance_profile, 'endpoint_clustering', perf_counter() - phase_started,
                         endpoint_clustering_count=1,
                         endpoint_clustering_source_count=len(native_candidates))
            fragments, _ = _fragment_records(page_scope, None, native_candidates, assignments)
            by_source = {row['source_primitive_ref']: row for row in fragments}
            phase_started = perf_counter()
            reachable = _reachable_cap_fragments(fragments, by_source[lref], by_source[rref], limit)
            _profile_add(performance_profile, 'cap_reachability', perf_counter() - phase_started,
                         cap_reachability_count=1,
                         cap_reachability_input_source_count=len(fragments),
                         cap_reachable_source_count=len(reachable))
            record['cap_source_reachability'] = {
                'method': 'complete_length_bounded_endpoint_reachability', 'version': '1.0.0',
                'maximum_path_display_points': limit, 'reachable_source_count': len(reachable),
                'input_source_refs_sha256': _sha256(sorted(local_refs)),
                'reachable_source_refs_sha256': _sha256(sorted(row['source_primitive_ref'] for row in reachable)),
                'full_source_endpoint_clustering_preserved': True,
                'all_admissible_cap_paths_preserved': True,
                'reachable_source_budget': max_local_sources,
            }
            if len(reachable) > max_local_sources:
                record['cap_search_complete'] = False
                continue
            page = {'fragments': reachable}
            phase_started = perf_counter()
            closure, refs = _shortest_cap_path(page, by_source[lref], by_source[rref], metrics)
            _profile_add(performance_profile, 'shortest_path_closure', perf_counter() - phase_started,
                         shortest_path_closure_count=1,
                         shortest_path_found_count=int(bool(closure)))
            if closure:
                by_id = {row['id']: row for row in page['fragments']}
                record['closure_source_primitive_refs'] = sorted(by_id[ref]['source_primitive_ref'] for ref in refs)
                evidence[key] = record
            else:
                record['state'] = 'rejected_no_native_cap_path'
    supported = defaultdict(list)
    for key, record in evidence.items():
        for ref in key:
            supported[ref].append(key)
    selected = set()
    for key, record in evidence.items():
        complete = all(complete_by_member.get(ref, False) for ref in key)
        # An incomplete cap search for any geometrically admissible competitor
        # blocks uniqueness even if that competitor has not supplied a cap yet.
        complete &= all(w['cap_search_complete'] for w in witnesses
                        if set(key).intersection(w['member_source_primitive_refs']))
        record['competitor_search_complete'] = complete
        if (complete and all(len(supported[ref]) == 1 for ref in key)
                and min(index.primitives[ref]['source_native_segment']['length_points'] for ref in key) >= minimum_member_length):
            record['state'] = 'closed_geometric_proposal'
            selected.update(key)
            selected.update(record['closure_source_primitive_refs'])
    if performance_profile is not None:
        performance_profile['counts'].update({
            'unique_candidate_pair_count': len(candidate_pairs),
            'envelope_witness_count': len(witnesses),
            'selected_source_count': len(selected),
            'closed_geometric_proposal_count': sum(
                row['state'] == 'closed_geometric_proposal' for row in witnesses),
            'rejected_no_native_cap_path_count': sum(
                row['state'] == 'rejected_no_native_cap_path' for row in witnesses),
            'unresolved_envelope_count': sum(row['state'] == 'unresolved' for row in witnesses),
        })
    return selected, witnesses


def _leader_paths(index, observation, *, max_path_segments=8):
    """Enumerate nonbranching native shoulders ending at a filled closed dot."""
    def geometry_query(box):
        method = getattr(index, 'query_geometry_descriptors', None)
        return method(box) if method is not None else index.query(box)

    def endpoint_query(box):
        method = getattr(index, 'query_endpoint_descriptors', None)
        if method is not None:
            return method(box)
        rows, complete, regions = geometry_query(box)
        return [row for row in rows if any(
            box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
            for point in (row['points_display'][0], row['points_display'][-1]))], complete, regions

    def adjacency_query(point, radius):
        method = getattr(index, 'query_leader_adjacency_descriptors', None)
        if method is not None:
            return method(point, radius)
        rows, complete, regions = geometry_query(_point_box(point, radius))
        return [row for row in rows if (
            row['source_native_segment']['kind'] == 'cubic'
            or any(math.dist(point, endpoint) <= .05
                   for endpoint in (row['points_display'][0],
                                    row['points_display'][-1])))], complete, regions

    box = observation['bbox_display']
    height = max(1, min(box[2] - box[0], box[3] - box[1]))
    rows, complete, regions = endpoint_query(_expand(box, .4 * height))
    starts = []
    for row in rows:
        native = row['source_native_segment']
        for end, point in enumerate((row['points_display'][0], row['points_display'][-1])):
            if (min(abs(point[0] - box[0]), abs(point[0] - box[2])) <= .4 * height
                    and box[1] - .25 * height <= point[1] <= box[3] + .25 * height):
                if native['kind'] != 'line':
                    complete = False  # An unsupported curved leader remains an alternative.
                elif native['style'].get('fill') is None:
                    starts.append((row, end))
    paths = []
    for start, end in starts:
        path, current, entered = [], start, end
        local_complete = complete
        region_refs = set(regions)
        for _ in range(max_path_segments):
            ref = current['source_primitive_ref']
            if ref in path:
                break
            path.append(ref)
            point = current['points_display'][1 - entered]
            around, closed, nearby_regions = adjacency_query(point, .4 * height)
            local_complete &= closed
            complete &= closed
            region_refs.update(nearby_regions)
            dot_groups = defaultdict(list)
            for row in around:
                native = row['source_native_segment']
                if native['kind'] == 'cubic' and native['style'].get('fill') is not None:
                    dot_groups[native['drawing_ref']].append(row)
            dots = []
            for group in dot_groups.values():
                if len(group) != 4:
                    continue
                extent = [min(r['bbox_display'][0] for r in group), min(r['bbox_display'][1] for r in group),
                          max(r['bbox_display'][2] for r in group), max(r['bbox_display'][3] for r in group)]
                w, h = extent[2] - extent[0], extent[3] - extent[1]
                center = [(extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2]
                if (.1 * height <= min(w, h) <= max(w, h) <= .8 * height
                        and min(w, h) / max(w, h) >= .8 and math.dist(center, point) <= .35
                        and all(.35 * min(w, h) <= math.dist(center, p) <= .6 * max(w, h)
                                for row in group for p in row['points_display'])
                        and all(any(other is not row and math.dist(row['points_display'][-1], other['points_display'][0]) <= .05
                                    for other in group) for row in group)):
                    dots.append(group)
            if len(dots) == 1:
                contact_rows, contact_complete, contact_regions = (
                    geometry_query(_point_box(point, height)))
                complete &= contact_complete
                paths.append({'source_primitive_refs': [*path, *(r['source_primitive_ref'] for r in dots[0])],
                              'contact_point_display': point, 'region_refs': sorted(region_refs),
                              'search_complete': local_complete and contact_complete,
                              'contact_search_radius': height,
                              'contact_region_refs': contact_regions,
                              'contact_stroke_alternatives': [
                                  {'source_primitive_ref': r['source_primitive_ref'],
                                   'distance_display_points': _contact_stroke_distance(point, r),
                                   'distance_basis': ('segment_distance' if r['source_native_segment']['kind'] == 'line'
                                                      else 'control_hull_box_lower_bound')}
                                  for r in contact_rows
                                  if any(r['source_native_segment']['style'].get(key) is not None for key in ('stroke', 'fill'))]})
                break
            if len(dots) > 1:
                complete = False
            adjacent = []
            for row in around:
                native = row['source_native_segment']
                if row['source_primitive_ref'] in path:
                    continue
                if native['kind'] != 'line':
                    if any(math.dist(point, p) <= .05 for p in (row['points_display'][0], row['points_display'][-1])):
                        complete = False
                    continue
                if native['style'].get('fill') is not None:
                    continue
                if not _style_compatible(_fragment(start)['style'], _fragment(row)['style']):
                    continue
                for end_index, endpoint in enumerate(row['points_display']):
                    if math.dist(point, endpoint) <= .05:
                        adjacent.append((row, end_index))
            if len(adjacent) != 1:
                if len(adjacent) > 1:
                    complete = False
                break
            current, entered = adjacent[0]
        else:
            complete = False
    # Duplicate traversal cannot create a competing target or silently erase it.
    return list({tuple(row['source_primitive_refs']): row for row in paths}.values()), complete


def _contact_stroke_distance(point, row):
    if row['source_native_segment']['kind'] == 'line':
        return point_distance_to_segment(tuple(point), tuple(row['points_display'][0]), tuple(row['points_display'][-1]))
    # A Bezier lies inside its control hull. This lower bound may abstain on
    # extra curves, but cannot clear a curved competitor by using its chord.
    box = row.get('search_bbox_display', row['bbox_display'])
    return math.hypot(max(box[0] - point[0], 0, point[0] - box[2]),
                      max(box[1] - point[1], 0, point[1] - box[3]))


def _contact(point, composite):
    geometry = composite['derived_geometry']
    points = geometry['centreline_points_display']
    width = composite['geometry_metrics']['mean_separation_display_points']
    return any(point_distance_to_segment(tuple(point), tuple(a), tuple(b)) <= width / 2 + .35
               for a, b in zip(points, points[1:]))


def _shared_route_ink(refs, sources, protected_refs):
    """A duplicate source ID cannot hide geometry shared with a route wall."""
    by_ref = {r['source_primitive_ref']: r for r in sources}
    dual = set(refs).intersection(protected_refs)
    for ref in set(refs) - dual:
        row = by_ref[ref]
        if row['source_native_segment']['kind'] != 'line':
            continue
        a, b = row['points_display']
        length = math.dist(a, b)
        if length <= .001:
            continue
        along = [(b[i] - a[i]) / length for i in (0, 1)]
        for protected in protected_refs:
            other = by_ref[protected]
            if other['source_native_segment']['kind'] != 'line':
                continue
            local = [[sum((p[i] - a[i]) * along[i] for i in (0, 1)),
                      (p[0] - a[0]) * along[1] - (p[1] - a[1]) * along[0]] for p in other['points_display']]
            if (all(abs(p[1]) <= .001 for p in local)
                    and min(length, max(p[0] for p in local)) - max(0, min(p[0] for p in local)) > .001):
                dual.add(ref)
                break
    return sorted(dual)


def _two_anchor_body_roles(sources, box, protected_refs, annotation_refs):
    """Bounded transverse plate/support interpretation from complete native ink.

    Two closed end motifs and a closed plate are required. An open arm, an
    axial continuation, or shared route sidewall preserves competing ownership.
    This proves a local drawing role, never a physical support or fitting count.
    """
    from src.drawing_engine.disciplines.mep.mep_native_equipment_observations import rectangular_outline_candidates
    from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import native_outer_cycle, _inside, _on_segment
    from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import TOLERANCE

    rectangles = rectangular_outline_candidates(sources, TOLERANCE)
    bodies = []
    for rectangle in rectangles:
        b = rectangle['bbox_display']
        spans = [b[i + 2] - b[i] for i in (0, 1)]
        axis = int(spans[1] > spans[0])
        minor, major = spans[1 - axis], spans[axis]
        if major < 3 * minor or not all(box[i] < b[i] and b[i + 2] < box[i + 2] for i in (0, 1)):
            continue
        # Nested native rails do not create separate bodies; the unique outer
        # closed perimeter retains all source IDs from those rails below.
        if any(other['id'] != rectangle['id'] and all(other['bbox_display'][i] <= b[i]
                and b[i + 2] <= other['bbox_display'][i + 2] for i in (0, 1))
                for other in rectangles):
            continue
        inside_rows = [r for r in sources if r['source_native_segment']['kind'] == 'line'
            and all(b[i] < p[i] < b[i + 2] for p in r['points_display'] for i in (0, 1))
            and math.dist(r['points_display'][0], r['points_display'][-1]) < minor / 2]
        groups = []
        for row in inside_rows:
            matches = [g for g in groups if any(math.dist(p, q) <= TOLERANCE
                for r in g for p in r['points_display'] for q in row['points_display'])]
            group = [row]
            for match in matches:
                group.extend(match)
                groups.remove(match)
            groups.append(group)
        anchors = []
        for group in groups:
            nonzero = [r for r in group if math.dist(*r['points_display']) > TOLERANCE]
            hull = native_outer_cycle(nonzero)
            if len(hull) < 8:
                continue
            bounds = [min(p[i] for p in hull) for i in (0, 1)] + [max(p[i] for p in hull) for i in (0, 1)]
            sizes = [bounds[i + 2] - bounds[i] for i in (0, 1)]
            center = [(bounds[i] + bounds[i + 2]) / 2 for i in (0, 1)]
            radius = sum(sizes) / 4
            if (min(sizes) < .7 * max(sizes) or max(sizes) > .5 * minor
                    or max(abs(math.dist(p, center) - radius) for p in hull) > .35 * radius):
                continue
            anchors.append({'center_display': center, 'boundary_points_display': hull,
                'source_primitive_refs': sorted(r['source_primitive_ref'] for r in group)})
        if len(anchors) != 2:
            continue
        anchors.sort(key=lambda a: a['center_display'][axis])
        a, z = [r['center_display'] for r in anchors]
        if (a[axis] - b[axis] > minor or b[axis + 2] - z[axis] > minor
                or abs(a[1 - axis] - z[1 - axis]) > .1 * minor
                or abs((a[1 - axis] + z[1 - axis]) / 2 - (b[1-axis] + b[3-axis]) / 2) > .15 * minor):
            continue
        perimeter = rectangle['points_display'][:-1]
        motif_refs = {ref for anchor in anchors for ref in anchor['source_primitive_refs']}
        owned, competitors, crossings = set(), set(), set()
        for row in sources:
            ref = row['source_primitive_ref']
            points = row['points_display']
            if ref in annotation_refs:
                continue
            hits = [any(_on_segment(p, c, d) for c, d in zip(perimeter, perimeter[1:] + perimeter[:1]))
                    or _inside(p, perimeter) for p in (points[0], points[-1])]
            if row['source_native_segment']['kind'] != 'line':
                if _intersects(b, row['search_bbox_display']):
                    competitors.add(ref)
                continue
            if all(hits):
                on_perimeter = any(all(_on_segment(p, c, d) for p in points)
                    for c, d in zip(perimeter, perimeter[1:] + perimeter[:1]))
                nested_rail = (abs(points[0][1-axis] - points[-1][1-axis]) <= TOLERANCE
                    and abs(min(p[axis] for p in points) - b[axis]) <= TOLERANCE
                    and abs(max(p[axis] for p in points) - b[axis + 2]) <= TOLERANCE)
                if ref in motif_refs or on_perimeter or nested_rail:
                    owned.add(ref)
                else:
                    competitors.add(ref)  # Containment alone does not explain interior equipment/route ink.
                continue
            if any(hits):
                competitors.add(ref)
                continue
            # Only a completely through-going transverse stroke is independent
            # of this body. Axial and diagonal continuations remain competitors.
            p, q = points
            if _intersects(b, row['search_bbox_display']):
                cross = 1 - axis
                if (abs(p[axis] - q[axis]) <= TOLERANCE
                        and min(p[cross], q[cross]) < b[cross] - TOLERANCE
                        and max(p[cross], q[cross]) > b[cross + 2] + TOLERANCE):
                    crossings.add(ref)
                else:
                    competitors.add(ref)
        dual = _shared_route_ink(owned, sources, protected_refs)
        reasons = []
        if competitors:
            reasons.append('unresolved_body_incidence_or_axial_continuation')
        if dual:
            reasons.append('body_strokes_also_support_a_certified_route')
        # At least one protected route wall must visibly traverse the body.
        if not crossings.intersection(protected_refs):
            reasons.append('no_independently_certified_transverse_route')
        bodies.append({'id': _stable_id('mep_two_anchor_body_role', rectangle['id'], anchors),
            'role': 'transverse_two_anchor_body_detail', 'epistemic_state': 'inferred',
            'state': 'accepted' if not reasons else 'ambiguous', 'reasons': reasons,
            'source_primitive_refs': sorted(owned), 'dual_role_source_refs': dual,
            'competing_source_primitive_refs': sorted(competitors),
            'independent_transverse_source_refs': sorted(crossings),
            'native_perimeter': rectangle, 'native_end_motifs': anchors,
            'physical_support_identity_established': False, 'quantity_eligible': False})
    return bodies


def _named_grid_roles(scope, query):
    """Own dashed axis ink only through named end bubbles and a full cadence.

    Reuse M1's opposing-label/group nomination. Exact native contacts then
    anchor a gap-separated long/short sequence at both closed bubbles. Every
    interval and gap must replay; a missing dash, extra arm or dual route
    boundary retains ambiguity. Width and colour cannot supply this role.
    """
    from statistics import median
    from src.drawing_engine.disciplines.mep.mep_sheet_registry import propose_grid_axes
    context = query['grid_context']
    texts = context['source_observations']
    if any(t.get('page_ref') != scope['page_ref'] or t.get('method') != 'native_pdf_text'
           or t.get('source_pdf_sha256') != context['source_pdf_sha256'] for t in texts):
        raise ValueError('named grid labels differ from native page evidence')
    axes = propose_grid_axes(texts, page_ref=scope['page_ref'],
        page_width=context['page_size_display'][0], page_height=context['page_size_display'][1])
    sources, box = query['source_rows'], query['search']['bbox_display']
    groups = defaultdict(list)
    for source in sources:
        native = source['source_native_segment']
        if native['kind'] == 'cubic' and native['style'].get('fill') is None:
            groups[native['drawing_ref']].append(source)
    bubbles = {}
    for group in groups.values():
        if len(group) != 4:
            continue
        points = [p for r in group for p in r['points_display']]
        extent = [min(p[i] for p in points) for i in (0, 1)] + [max(p[i] for p in points) for i in (0, 1)]
        center = [(extent[i]+extent[i+2])/2 for i in (0, 1)]
        radius = (extent[2]-extent[0]+extent[3]-extent[1])/4
        if radius <= 0 or any(not box[i] < extent[i] < extent[i+2] < box[i+2] for i in (0, 1)):
            continue
        if max(abs(math.dist(p, center)-radius) for p in points) > .035*radius:
            continue
        if any(sum(math.dist(r['points_display'][-1], other['points_display'][0]) <= .001
                   for other in group) != 1 for r in group):
            continue
        key = tuple(round(v, 3) for v in extent)
        bubble = bubbles.setdefault(key, {'bbox_display': extent, 'points': [], 'source_primitive_refs': []})
        bubble['points'].extend(r['points_display'][0] for r in group)
        bubble['source_primitive_refs'].extend(r['source_primitive_ref'] for r in group)
    output = []
    for axis in axes:
        if axis['state'] != 'observed' or axis['method'] != 'opposing_equal_label_alignment_v1':
            continue
        cross = 0 if axis['orientation_display'] == 'vertical' else 1
        along = 1-cross
        labels = [t for t in texts if t['text'].strip() == axis['label']]
        ends = [b for b in bubbles.values() if any(all(b['bbox_display'][i] < t['bbox_display'][i]
                    and t['bbox_display'][i+2] < b['bbox_display'][i+2] for i in (0, 1)) for t in labels)]
        if len(ends) != 2:
            continue
        ends.sort(key=lambda b: b['bbox_display'][along])
        start = max(ends[0]['points'], key=lambda p: p[along])
        end = min(ends[1]['points'], key=lambda p: p[along])
        if abs(start[cross]-end[cross]) > .001:
            continue
        by_style = defaultdict(dict)
        for row in sources:
            native = row['source_native_segment']
            a, b = row['points_display'][0], row['points_display'][-1]
            if native['kind'] != 'line' or native['style'].get('stroke') is None:
                continue
            # Dash centre coordinates may differ within their actual ink
            # width. This is role ownership, never endpoint snapping/joining.
            if any(abs(p[cross]-start[cross]) > (native['style'].get('width') or 0)/2+.001 for p in (a, b)):
                continue
            lo, hi = sorted((a[along], b[along]))
            if hi-lo <= .001 or lo < start[along]-.001 or hi > end[along]+.001:
                continue
            style = native['style']
            by_style[(style.get('width'), style.get('dash'), style.get('fill') is not None)].setdefault((lo, hi), []).append(row['source_primitive_ref'])
        candidates = []
        for intervals in by_style.values():
            spans = sorted(intervals)
            if len(spans) < 14 or abs(spans[0][0]-start[along]) > .001 or abs(spans[-1][1]-end[along]) > .001:
                continue
            # Learn the cadence immediately after the independently anchored
            # first dash. Follow only a unique full sequence to the other
            # bubble; intervening non-sequence ink remains a competitor.
            sizes = [spans[i][1]-spans[i][0] for i in (1, 2)]
            gap = median(spans[i+1][0]-spans[i][1] for i in range(4))
            if gap <= .001 or min(sizes) <= .001 or max(sizes) < 2*min(sizes):
                continue
            chain = [spans[0]]
            while abs(chain[-1][1]-end[along]) > .001 and len(chain) <= len(spans):
                size = sizes[(len(chain)-1) % 2]
                options = [s for s in spans if abs(s[0]-chain[-1][1]-gap) <= .03*gap
                    and (abs((s[1]-s[0])-size) <= .03*size
                         or abs(s[1]-end[along]) <= .001 and 0 < s[1]-s[0] <= size*1.03)]
                if len(options) != 1:
                    break
                chain.append(options[0])
            if len(chain) < 14 or abs(chain[-1][1]-end[along]) > .001:
                continue
            candidates.append({'intervals': chain, 'source_primitive_refs': sorted(ref for span in chain for ref in intervals[span]),
                'native_dash_lengths': sizes, 'native_gap_length': gap,
                'sequence_competitor_refs': sorted(ref for span in spans if span not in chain for ref in intervals[span]),
                'axis_ink_half_width_used_for_role_only': True})
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        dual = _shared_route_ink(candidate['source_primitive_refs'], sources, scope['route_boundary_source_refs'])
        # A non-axis stroke ending on an owned dash is a competing role even
        # if its width/colour differ. Ordinary through-crossings are retained.
        arms, ambiguous = [], set(dual)
        by_ref = {r['source_primitive_ref']: r for r in sources}
        for row in sources:
            if row['source_primitive_ref'] in candidate['source_primitive_refs']:
                continue
            points = (row['points_display'][0], row['points_display'][-1])
            if any(abs(p[cross]-start[cross]) <= .001 and any(lo+.001 < p[along] < hi-.001
                   for lo,hi in candidate['intervals']) for p in points):
                arms.append(row['source_primitive_ref'])
                for ref in candidate['source_primitive_refs']:
                    a,b = by_ref[ref]['points_display'][0],by_ref[ref]['points_display'][-1]
                    if any(min(a[along],b[along])-.001 <= p[along] <= max(a[along],b[along])+.001 for p in points):
                        ambiguous.add(ref)
        for is_ambiguous in (False, True):
            refs = sorted(ref for ref in candidate['source_primitive_refs'] if (ref in ambiguous) == is_ambiguous)
            if not refs:
                continue
            reasons = (['grid_ink_also_supports_certified_route'] if set(refs).intersection(dual) else []) + (
                ['additional_endpoint_on_grid_dash'] if is_ambiguous and arms else [])
            output.append({'id': _stable_id('mep_native_named_grid_role', axis['id'], refs),
                'role': 'named_grid_axis_dash_sequence', 'state': 'ambiguous' if is_ambiguous else 'accepted',
                'epistemic_state': 'derived', **candidate, 'named_axis': axis,
                'sequence_source_primitive_refs': candidate['source_primitive_refs'], 'source_primitive_refs': refs,
                'end_bubble_source_primitive_refs': sorted(ref for b in ends for ref in b['source_primitive_refs']),
                'endpoint_contacts_display': [start, end], 'additional_incident_source_refs': sorted(arms) if is_ambiguous else [],
                'dual_role_source_refs': sorted(set(refs).intersection(dual)), 'reasons': reasons,
                'attribute_applicability_established': False, 'quantity_eligible': False})
    return output


def scoped_native_stroke_roles(scope, query):
    """Rebuild local role evidence; retain the original rows and all alternatives.

    Consumers supply protected, already certified native route boundaries. The
    role overlay may explain an annotation/body detail but cannot remove a
    protected boundary or propagate a label beyond its independently bound scope.
    """
    from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import FrozenNativeQueries
    index = FrozenNativeQueries(query)
    if scope['page_ref'] != query['page_ref']:
        raise ValueError('stroke ownership scope differs from native query page')
    sources = {r['source_primitive_ref']: r for r in query['source_rows']}
    protected = set(scope['route_boundary_source_refs'])
    if not protected.issubset(sources):
        raise ValueError('stroke ownership omits protected native boundary evidence')
    roles, annotation_refs = [], set()
    for observation in query.get('source_observations', []):
        if observation['page_ref'] != scope['page_ref']:
            raise ValueError('annotation ownership crosses page scope')
        paths, complete = _leader_paths(index, observation)
        if not complete or len(paths) != 1 or not paths[0]['search_complete']:
            continue
        path = paths[0]
        dual = _shared_route_ink(path['source_primitive_refs'], query['source_rows'], protected)
        role = {'id': _stable_id('mep_native_leader_role', observation['id'], path['source_primitive_refs']),
            'role': 'native_text_dot_leader', 'state': 'ambiguous' if dual else 'accepted',
            'epistemic_state': 'derived', 'source_observation_ref': observation['id'],
            'source_observation_sha256': _sha256(observation),
            'source_primitive_refs': path['source_primitive_refs'], 'native_path': path,
            'dual_role_source_refs': dual, 'reasons': ['leader_also_supports_certified_route'] if dual else [],
            'attribute_applicability_established': False, 'quantity_eligible': False}
        roles.append(role)
        if not dual:
            annotation_refs.update(path['source_primitive_refs'])
    if query['search']['complete'] and not query['search'].get('budget_exhausted') and query.get('grid_context'):
        roles.extend(_named_grid_roles(scope, query))
    elif query['search']['complete'] and not query['search'].get('budget_exhausted'):
        from src.drawing_engine.disciplines.mep.mep_projected_trace_completion import native_dimension_role_candidates
        for candidate in native_dimension_role_candidates(scope, query):
            dual = _shared_route_ink(candidate['source_primitive_refs'], query['source_rows'], protected)
            reasons = list(candidate['reasons'])
            if dual:
                reasons.append('dimension_also_supports_certified_route')
            accepted = candidate['role_geometry_established'] and not reasons
            roles.append({**candidate, 'state': 'accepted' if accepted else 'ambiguous',
                'exclusive_role': accepted, 'dual_role_source_refs': dual, 'reasons': reasons,
                'attribute_applicability_established': False})
            if accepted:
                annotation_refs.update(candidate['source_primitive_refs'])
        roles.extend(_two_anchor_body_roles(query['source_rows'], query['search']['bbox_display'], protected, annotation_refs))
    return {'scope_ref': scope['id'], 'scope': deepcopy(scope), 'page_ref': scope['page_ref'], 'scope_sha256': _sha256(scope),
        'source_query_ref': query.get('id'), 'source_query_sha256': _sha256(query), 'roles': roles,
        'all_source_rows_retained': True, 'physical_continuation_established': False,
        'attribute_applicability_established': False, 'quantity_eligible': False}


def binding_ownership_context(targets, observation, leader):
    page_ref = observation['page_ref']
    return {'source_pdf_sha256': targets['route_graph']['document']['source_pdf_sha256'],
        'observation_sha256': _sha256(observation), 'leader_sha256': _sha256(leader),
        'page_composites_sha256': _sha256([r for r in targets['composites']['accepted_composites'] if r['page_ref'] == page_ref]),
        'page_envelopes_sha256': _sha256([r for r in targets['envelope_searches'] if r['page_ref'] == page_ref])}


def binding_stroke_roles(targets, observation, leader, query, observations):
    """Bind a role query to exact native annotations and current competitors."""
    context = binding_ownership_context(targets, observation, leader)
    if query.get('binding_context') != context:
        raise ValueError('stroke ownership evidence belongs to different binding inputs')
    if any(observations.get(row['id']) != row for row in query.get('source_observations', [])):
        raise ValueError('stroke ownership native text differs from frozen terminology')
    if observation not in query.get('source_observations', []):
        raise ValueError('stroke ownership omits the binding annotation')
    native_refs = {r['source_primitive_ref'] for r in query['source_rows']}
    protected = sorted({ref for c in targets['composites']['accepted_composites']
        if c['page_ref'] == observation['page_ref'] for ref in c['member_source_primitive_refs'] if ref in native_refs})
    scope = {'id': _stable_id('mep_binding_ownership_scope', observation['id'], context, query['search']['bbox_display']),
        'page_ref': observation['page_ref'], 'source_observation_ref': observation['id'],
        'route_boundary_source_refs': protected}
    return scoped_native_stroke_roles(scope, query)


def build_automatic_page_components(*, sheet_registry, page_indexes, observations,
                                    minimum_member_length=24, performance_profiles=None,
                                    envelope_engine='per_candidate'):
    """Freeze automatic native target/leader observations, then run M3.5.

    All geometric envelope proposals (including unlabelled ones) survive in the
    register. Only a unique, completely searched dot-leader contact supplies M4
    evidence. Equipment without certified ports remains an unresolved proposal.
    """
    page_inputs, witnesses, leader_records = {}, [], []
    for scope in sheet_registry['pages']:
        page_ref = scope['page_ref']
        index = page_indexes.get(page_ref)
        if index is None:
            continue
        profile = None if performance_profiles is None else performance_profiles.setdefault(page_ref, {})
        phase_started = perf_counter()
        selected, searches = discover_envelopes(index=index, page_scope=scope,
                                                minimum_member_length=minimum_member_length,
                                                performance_profile=profile, engine=envelope_engine)
        envelope_seconds = perf_counter() - phase_started
        witnesses.extend(searches)
        phase_started = perf_counter()
        for observation in observations:
            if observation['page_ref'] != page_ref:
                continue
            paths, search_complete = _leader_paths(index, observation)
            for path in paths:
                selected.update(path['source_primitive_refs'])
            leader_records.append({'observation_ref': observation['id'], 'page_ref': page_ref,
                                   'paths': paths, 'search_complete': search_complete})
        leader_seconds = perf_counter() - phase_started
        if profile is not None:
            profile.setdefault('phases', {}).update({
                'envelope_discovery_total': {'seconds': envelope_seconds, 'calls': 1},
                'leader_discovery_total': {'seconds': leader_seconds, 'calls': 1},
            })
        page_inputs[page_ref] = {'native_topology': {'segments': index.native(selected)},
                                'vertex_tolerance_display_points': .05,
                                'gap_tolerance_display_points': .05}
    phase_started = perf_counter()
    graph = build_mep_route_graph(sheet_registry=sheet_registry, page_inputs=page_inputs)
    targets = certify_automatic_targets(graph=graph, witnesses=witnesses, leader_records=leader_records)
    graph_seconds = perf_counter() - phase_started
    from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import discover_native_boundary_connections
    phase_started = perf_counter()
    connections = [row for page_ref, index in page_indexes.items()
        for row in discover_native_boundary_connections(index=index, composites=targets['composites'],
                                                         graph=graph, page_ref=page_ref)]
    boundary_seconds = perf_counter() - phase_started
    if performance_profiles is not None:
        for page_ref in page_indexes:
            profile = performance_profiles.setdefault(page_ref, {})
            profile.setdefault('phases', {}).update({
                'route_graph_and_target_certification': {'seconds': graph_seconds, 'calls': 1},
                'boundary_connection_discovery': {'seconds': boundary_seconds, 'calls': 1},
            })
    return (targets['route_graph'], targets['composites'], targets['envelope_searches'],
            targets['leader_observations'], connections)


def build_automatic_targets(*, sheet_registry, page_indexes, observations, minimum_member_length=24,
                            performance_profiles=None, envelope_engine='per_candidate'):
    """Compatibility wrapper for callers that explicitly need legacy page JSON."""
    graph, composites, witnesses, leaders, connections = build_automatic_page_components(
        sheet_registry=sheet_registry, page_indexes=page_indexes, observations=observations,
        minimum_member_length=minimum_member_length, performance_profiles=performance_profiles,
        envelope_engine=envelope_engine)
    return {'route_graph': graph, 'composites': composites, 'envelope_searches': witnesses,
            'leader_observations': leaders, 'boundary_connections': connections}


def certify_automatic_targets(*, graph, witnesses, leader_records):
    """Run unchanged composite gates, intersected with full-source witnesses."""
    composites = build_mep_outlined_route_composites(route_graph=graph)
    closed = {(row['page_ref'], tuple(row['member_source_primitive_refs'])): row for row in witnesses
              if row['state'] == 'closed_geometric_proposal'}
    # The final graph may assemble unrelated pairs from the evidence union.
    # Such pairs have not passed the full-source competitor search: never
    # expose them as accepted to M4, irrespective of a small-graph uniqueness.
    for row in composites['candidates']:
        search = closed.get((row['page_ref'], tuple(row['member_source_primitive_refs'])))
        if row['state'] == 'accepted' and (not search or search['page_ref'] != row['page_ref']):
            row['state'], row['epistemic_state'] = 'abstained', 'unknown'
            row['reasons'] = ['full_source_competitor_search_not_closed']
            row['certificates']['full_source_competitor_search'] = False
    composites['accepted_composites'] = [deepcopy(row) for row in composites['candidates'] if row['state'] == 'accepted']
    composites['summary']['accepted_composite_count'] = len(composites['accepted_composites'])
    composites['summary']['abstained_candidate_count'] = len(composites['candidates']) - len(composites['accepted_composites'])
    composites['summary']['state_counts'] = {key: sum(row['state'] == key for row in composites['candidates']) for key in ('accepted', 'abstained')}
    return {'route_graph': graph, 'composites': composites, 'envelope_searches': witnesses,
            'leader_observations': leader_records}


def automatic_binding_evidence(*, targets, terminology, stroke_ownership_queries=None):
    evidence, unresolved = [], []
    source_searches_sha256 = None
    observations = {row['id']: row for row in terminology['source_observations']}
    leaders = {row['observation_ref']: row for row in targets['leader_observations']}
    composites = targets['composites']['accepted_composites']
    role_cache = {}
    for proposal in terminology['proposals']:
        observation = next((observations[ref] for ref in proposal['evidence_refs'] if ref in observations), None)
        if observation is None:
            continue
        row = leaders.get(observation['id'])
        ownership, explained = None, set()
        query = (stroke_ownership_queries or {}).get(observation['id'])
        if query is not None:
            if observation['id'] not in role_cache:
                role_cache[observation['id']] = binding_stroke_roles(targets, observation, row, query, observations)
            ownership = role_cache[observation['id']]
            explained = {ref for role in ownership['roles'] if role['state'] == 'accepted'
                         for ref in role['source_primitive_refs']}
            explained.difference_update(ref for role in ownership['roles'] if role['state'] != 'accepted'
                                        for ref in role['source_primitive_refs'])
        reasons, contacts = [], []
        paths = row['paths'] if row else []
        if not paths:
            reasons.append('no_complete_native_dot_leader')
        if row is None or not row['search_complete'] or any(not path['search_complete'] for path in paths):
            reasons.append('incomplete_relevant_competitor_search')
        if observation.get('ocr_overlap_alternatives') or observation.get('native_overlap_alternatives'):
            reasons.append('unresolved_ocr_overlap_alternatives')
        if proposal['proposal_type'] not in {'system', 'inline_size', 'elevation'}:
            reasons.append('semantic_kind_or_equipment_port_not_certified')
        for path in paths:
            for composite in composites:
                if composite['page_ref'] == proposal['page_ref'] and _contact(path['contact_point_display'], composite):
                    contacts.append((composite, path))
            for candidate in targets['envelope_searches']:
                if (candidate['page_ref'] == proposal['page_ref'] and _contact(path['contact_point_display'], candidate)
                        and candidate['state'] not in {'closed_geometric_proposal', 'rejected_no_native_cap_path'}):
                    if not set(candidate['member_source_primitive_refs']).issubset(explained):
                        reasons.append('unresolved_competing_envelope_at_contact')
        target_refs = sorted({target['id'] for target, _ in contacts})
        if len(target_refs) != 1 or len(paths) != 1:
            reasons.append('no_unique_leader_contact_target')
        if not reasons:
            composite, path = contacts[0]
            protected = set(composite['member_source_primitive_refs'])
            dual = protected.intersection(path['source_primitive_refs'])
            if query is not None:
                present = {r['source_primitive_ref'] for r in query['source_rows']}
                dual.update(_shared_route_ink(
                    set(path['source_primitive_refs']) & present,
                    query['source_rows'], protected & present))
            if dual:
                reasons.append('annotation_route_dual_use')
            radius = composite['geometry_metrics']['mean_separation_display_points'] / 2 + .35
            covered = {*composite['member_source_primitive_refs'], *path['source_primitive_refs']}
            covered.update(explained)
            by_fragment = {r['id']: r['source_primitive_ref'] for p in targets['route_graph']['pages'] for r in p['fragments']}
            covered.update(by_fragment[ref] for ref in composite['supporting_closure_fragment_refs'])
            if radius > path['contact_search_radius']:
                reasons.append('contact_competitor_search_radius_insufficient')
            if any(r['source_primitive_ref'] not in covered and r['distance_display_points'] <= radius
                   for r in path['contact_stroke_alternatives']):
                reasons.append('unresolved_native_stroke_at_leader_contact')
        if not reasons:
            composite, path = contacts[0]
            if source_searches_sha256 is None:
                source_searches_sha256 = _sha256(targets['envelope_searches'])
            evidence.append({'id': _stable_id('mep_automatic_binding_evidence', proposal['id'], composite['id']),
                'proposal_ref': proposal['id'], 'page_ref': proposal['page_ref'], 'state': 'observed',
                'method': {'name': 'complete_native_dot_leader_contact', 'version': '1.0.0'},
                'target_kind': 'route_composite', 'target_refs': [composite['id']],
                'geometric_evidence_refs': [composite['id'], *composite['member_source_primitive_refs'], *path['source_primitive_refs']],
                'explicit_branch_coverage': False,
                'automatic_search_certificate': {'leader_observation_ref': observation['id'],
                    'region_refs': path['region_refs'], 'contact_point_display': path['contact_point_display'],
                    'source_searches_sha256': source_searches_sha256,
                    'reviewed_selectors_used': False}})
            if ownership:
                evidence[-1]['scoped_stroke_ownership'] = ownership
        else:
            unresolved.append({'id': proposal['id'], 'page_ref': proposal['page_ref'],
                'description': observation['text'], 'reason': '; '.join(sorted(set(reasons))),
                'target_refs': target_refs, 'evidence_refs': list(proposal['evidence_refs']),
                'source_geometry': {'bbox_display': observation['bbox_display']}, 'quantity_eligible': False})
            if ownership:
                unresolved[-1]['scoped_stroke_ownership'] = ownership
    return evidence, unresolved


def apply_geometric_text_applicability(*, terminology, binding_evidence):
    """Add a derived applicability observation; never rewrite native/OCR text.

    M2 intentionally abstains on unknown regions. A closed leader-to-envelope
    certificate resolves only that reason. The immutable source line remains in
    source_observations, and its new, separately identified overlay cites both
    the original line and the exact geometric certificate. Semantic conflicts
    and all other M2 reasons continue to abstain.
    """
    proposals = {row['id']: row for row in terminology['proposals']}
    sources = {row['id']: row for row in terminology['source_observations']}
    replacements, origins = {}, {}
    for evidence in binding_evidence:
        proposal = proposals[evidence['proposal_ref']]
        if set(proposal['reasons']) != {'region_applicability_unresolved'}:
            continue
        source_ref = proposal['anchor_ref']
        source = sources[source_ref]
        if source.get('region_role') != 'unknown':
            continue
        replacements.setdefault(source_ref, []).append(evidence)
    derived = []
    for source_ref, rows in sorted(replacements.items()):
        source = sources[source_ref]
        targets = {tuple(row['target_refs']) for row in rows}
        if len(targets) != 1:
            continue
        identifier = _stable_id('mep_geometric_text_applicability', source_ref, next(iter(targets)))
        overlay = deepcopy(source)
        overlay.update(id=identifier, record_type='mep_geometric_text_applicability_observation',
                       region_role='drawing_inline', epistemic_state='derived',
                       immutable_source_observation=False, source_observation_ref=source_ref,
                       interpretation_scope_ref=identifier,
                       applicability_evidence=deepcopy(rows))
        overlay['text_interpretation'] = interpret_mep_text(overlay)
        derived.append(overlay)
        origins[source_ref] = identifier
    if not derived:
        return terminology, binding_evidence
    # Re-propose only the overlays; existing proposal IDs and diagnostics remain
    # unchanged for every unrelated or quarantined source line.
    additions = build_mep_terminology_proposals(document=terminology['document'], observations=derived)
    result = deepcopy(terminology)
    result['source_observations'].extend(derived)
    result['proposals'] = [row for row in result['proposals'] if row['anchor_ref'] not in origins] + additions['proposals']
    mapping = {(row['anchor_ref'], row['proposal_type'], _sha256(row['candidate'])): row['id'] for row in additions['proposals']}
    updated = deepcopy(binding_evidence)
    for row in updated:
        proposal = proposals[row['proposal_ref']]
        if proposal['anchor_ref'] in origins:
            row['proposal_ref'] = mapping[(origins[proposal['anchor_ref']], proposal['proposal_type'], _sha256(proposal['candidate']))]
            row['id'] = _stable_id('mep_automatic_binding_evidence', row['proposal_ref'], row['target_refs'])
    result['summary']['source_observation_count'] = len(result['source_observations'])
    result['summary']['proposal_count'] = len(result['proposals'])
    for field, key in (('proposal_type_counts', 'proposal_type'), ('state_counts', 'state')):
        result['summary'][field] = {value: sum(row[key] == value for row in result['proposals'])
                                   for value in sorted({row[key] for row in result['proposals']})}
    result['automatic_applicability_input_sha256'] = _sha256(terminology)
    return result, updated
