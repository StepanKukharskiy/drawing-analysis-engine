import tempfile
from pathlib import Path
import unittest

from tools.run_mep_parallel_pages import (
    estimate_page_costs, merge_page_caches, scheduled_page_order,
)
from src.drawing_engine.disciplines.mep.mep_v3_page_cache import MepV3PageCache
from src.drawing_engine.project.project_packed_store import PackedProjectStore


class ParallelPageWorkerTests(unittest.TestCase):
    def test_preflight_serializes_only_costly_pages(self):
        registry = {'pages': [
            {'page_number': 1, 'page_size_display': [640, 640],
             'quality_route': {'base_structural_quality_route': {'metrics': {
                 'native_path_paint_operation_count': 10}}}},
            {'page_number': 2, 'page_size_display': [3456, 2592],
             'quality_route': {'base_structural_quality_route': {'metrics': {
                 'native_path_paint_operation_count': 2_000_000}}}},
            {'page_number': 3, 'page_size_display': [640, 640],
             'quality_route': {'base_structural_quality_route': {'metrics': {
                 'native_path_paint_operation_count': 20}}}},
        ]}
        estimates = estimate_page_costs(registry, [3, 2, 1])
        self.assertEqual([row['scheduling_class'] for row in estimates],
                         ['lightweight', 'disk_heavy', 'lightweight'])
        self.assertEqual(scheduled_page_order(estimates), [2, 1, 3])
        heavy = estimates[1]
        self.assertEqual(heavy['spatial_cell_count'], 54 * 41)
        self.assertGreater(heavy['expected_index_bytes'], 256 * 1024 * 1024)

    def test_ordered_merge_preserves_content_addressed_page_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for page_number in (2, 1):
                path = root / f"source-{page_number}.sqlite"
                store = PackedProjectStore(path, create=True)
                cache = MepV3PageCache(store)
                key = f"{page_number:064x}"
                inputs = {"fixture": page_number}
                manifest = cache.write(
                    cache_key=key, cache_inputs=inputs, page_ref=f"page.{page_number}",
                    records=iter((("page_record", {"page_number": page_number}),)))
                store.connection.close()
                results.append({
                    "page_number": page_number, "database": path,
                    "page_cache_key": key,
                    "artifact_sha256": manifest["artifact_sha256"],
                })
            target = root / "merged.sqlite"
            merged = merge_page_caches(results, target)
            self.assertEqual([row["page_number"] for row in merged], [1, 2])
            store = PackedProjectStore(target)
            cache = MepV3PageCache(store)
            self.assertEqual(
                [next(cache.iter_records(row["page_cache_key"]))[1]["page_number"]
                 for row in merged], [1, 2])
            store.connection.close()


if __name__ == "__main__":
    unittest.main()
