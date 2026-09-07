"""Exact local native queries for refinement of frozen full-page envelopes.

Every original drawing participates in the spatial index. Queries retain whole
primitives and original indices, including curves and unsupported alternatives;
no reviewed box or annotation selects which connections are attempted.
"""

from collections import defaultdict
from itertools import chain
import math

import fitz

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.disciplines.mep.mep_native_target_discovery import _cells, _intersects, _drawing_search_box, _native_display_geometry
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import _stable_id
from src.drawing_engine.core.vector_topology import iter_native_segments


class FrozenNativeQueries:
    """Replay subqueries only inside one completely captured native extent.

    Whole source strokes survive even when they extend beyond the extent. Their
    presence never proves that a search outside the captured extent is complete.
    """

    def __init__(self, query):
        self.page_ref = query['page_ref']
        self.search = query['search']
        self.rows = query['source_rows']
        self.box = self.search['bbox_display']
        if (len(self.box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in self.box)
                or any(self.box[i] >= self.box[i + 2] for i in (0, 1))):
            raise ValueError('frozen native query has invalid search extent')
        refs = [row['source_primitive_ref'] for row in self.rows]
        if (len(refs) != len(set(refs))
                or self.search.get('source_rows_sha256') != _sha256(self.rows)
                or self.search.get('all_source_refs_sha256') != _sha256(sorted(refs))
                or type(self.search.get('complete')) is not bool):
            raise ValueError('frozen native query inventory or completeness changed')
        for row in self.rows:
            native = row['source_native_segment']
            if (row['page_ref'] != self.page_ref or native['id'] != row['source_primitive_ref']
                    or row['id'] != _stable_id('mep_native_target_primitive', self.page_ref, native['id'])):
                raise ValueError('frozen native query page or primitive identity changed')
            points, bounds, search = _native_display_geometry(native, fitz.Matrix(row['pdf_to_display_matrix']))
            if (points != row['points_display'] or bounds != row['bbox_display']
                    or search != row['search_bbox_display'] or not _intersects(self.box, search)):
                raise ValueError('frozen native query geometry does not replay')

    def query(self, box):
        rows = [r for r in self.rows if _intersects(box, r['search_bbox_display'])]
        inside = all(self.box[i] <= box[i] and box[i + 2] <= self.box[i + 2] for i in (0, 1))
        complete = inside and self.search['complete'] and not self.search.get('budget_exhausted', False)
        ref = _stable_id('mep_full_native_boundary_query', self.page_ref, box,
                        sorted(r['source_primitive_ref'] for r in rows))
        return rows, complete, [ref]


class NativeBoundaryQueries:
    def __init__(self, page, page_ref):
        self.page_ref = page_ref
        self.rotation = page.rotation_matrix
        self.drawings = page.get_drawings()
        self.cells = defaultdict(set)
        self.boxes = []
        for ordinal, drawing in enumerate(self.drawings):
            box = _drawing_search_box(drawing, self.rotation)
            self.boxes.append(box)
            for cell in _cells(box, 64):
                self.cells[cell].add(ordinal)

    def query(self, box):
        candidates = {i for cell in _cells(box, 64) for i in self.cells.get(cell, ())}
        relevant = {i for i in candidates if _intersects(box, self.boxes[i])}
        complete = all(item[0] in {'l', 'c', 're', 'qu'}
                       for i in relevant for item in self.drawings[i]['items'])
        # The canonical iterator's explicit offset preserves page-global IDs.
        # Do not rescan millions of unrelated paths for each tiny query.
        natives = chain.from_iterable(iter_native_segments(
            [self.drawings[i]], drawing_index_offset=i) for i in sorted(relevant))
        rows = []
        for native in natives:
            points, bounds, search = _native_display_geometry(native, self.rotation)
            if not _intersects(box, search):
                continue
            rows.append({'id': _stable_id('mep_native_target_primitive', self.page_ref, native['id']),
                'page_ref': self.page_ref, 'source_primitive_ref': native['id'],
                'source_native_segment': native, 'points_display': points,
                'bbox_display': bounds, 'search_bbox_display': search,
                'pdf_to_display_matrix': list(self.rotation), 'state': 'observed',
                'quantity_eligible': False})
        witness = _stable_id('mep_full_native_boundary_query', self.page_ref, box,
                             sorted(r['source_primitive_ref'] for r in rows))
        return rows, complete, [witness]
