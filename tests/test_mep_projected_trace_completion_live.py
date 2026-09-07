import gzip
import json
from pathlib import Path
import unittest

from src.drawing_engine.disciplines.mep.mep_declared_data import _file_sha256, _sha256
from src.drawing_engine.disciplines.mep.mep_native_boundary_queries import NativeBoundaryQueries


class MepProjectedTraceCompletionLiveTest(unittest.TestCase):
    def test_original_pdf_replays_the_complete_positive_corridor_query(self):
        import fitz
        root = Path(__file__).resolve().parents[1]
        folder = root / 'fixtures/mep/m_and_p_coordination'
        fixture = json.loads(gzip.decompress((folder / 'projected-trace-completion-replay.json.gz').read_bytes()))
        positive = next(r for r in fixture['expected_completion']['scoped_traces'] if r['projected_scope_complete'])
        query = next(q for q in fixture['source_queries'] if q['id'] == positive['source_query_ref'])
        registry = json.loads((folder / 'm_and_p_coordination.sheet-registry.json').read_text())
        page = next(p for p in registry['pages'] if p['page_ref'] == query['page_ref'])
        source = root / 'M&P mark-up against shop systems piping.pdf'
        self.assertEqual(_file_sha256(source), fixture['document']['source_pdf_sha256'])
        with fitz.open(source) as pdf:
            index = NativeBoundaryQueries(pdf[page['page_number'] - 1], query['page_ref'])
            rows, complete, regions = index.query(query['search']['bbox_display'])
        self.assertTrue(complete)
        self.assertEqual(_sha256(rows), query['search']['source_rows_sha256'])
        self.assertEqual(_sha256(rows), _sha256(query['source_rows']))
        self.assertEqual(regions, query['search']['region_refs'])


if __name__ == '__main__':
    unittest.main()
