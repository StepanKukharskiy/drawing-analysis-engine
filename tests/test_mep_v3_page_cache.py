import copy
from pathlib import Path
import tempfile
import unittest

from src.drawing_engine.disciplines.mep.mep_v3_page_cache import (
    MepV3PageCache,
    legacy_page_record_stream,
    page_component_record_stream,
    page_cache_key,
)
from src.drawing_engine.project.project_packed_store import PackedProjectStore


def _digest(character):
    return character * 64


def _page_target():
    candidate = {
        "id": "composite.1",
        "record_type": "mep_outlined_route_composite",
        "page_ref": "page.5",
        "state": "accepted",
        "member_source_primitive_refs": ["native.1", "native.2"],
    }
    return {
        "route_graph": {
            "schema_version": "0.1.0",
            "layer": "mep_route_observation_graph",
            "pages": [{"page_ref": "page.5", "fragments": []}],
        },
        "composites": {
            "schema_version": "0.1.0",
            "layer": "mep_outlined_route_composites",
            "summary": {"candidate_count": 1},
            "candidates": [candidate],
            "accepted_composites": [copy.deepcopy(candidate)],
        },
        "envelope_searches": [{
            "id": "search.1", "record_type": "mep_envelope_search",
            "page_ref": "page.5", "state": "closed_geometric_proposal",
        }],
        "leader_observations": [{
            "id": "leader.1", "record_type": "mep_native_leader_observation",
            "page_ref": "page.5", "state": "observed",
        }],
        "boundary_connections": [{
            "id": "connection.1", "record_type": "mep_native_boundary_connection",
            "page_ref": "page.5", "state": "accepted",
        }],
    }


class MepV3PageCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "page-cache.sqlite"
        self.inputs = {
            "source_pdf_sha256": _digest("a"),
            "page_identity": {"page_ref": "page.5", "page_number": 5},
            "extraction_implementation_sha256": _digest("b"),
            "configuration_sha256": _digest("c"),
            "ruleset_sha256": _digest("d"),
        }

    def test_key_binds_every_required_input(self):
        key, inputs = page_cache_key(**self.inputs)
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(key, page_cache_key(**copy.deepcopy(self.inputs))[0])
        replacements = {
            "source_pdf_sha256": _digest("e"),
            "page_identity": {"page_ref": "page.6", "page_number": 6},
            "extraction_implementation_sha256": _digest("e"),
            "configuration_sha256": _digest("e"),
            "ruleset_sha256": _digest("e"),
        }
        for field, value in replacements.items():
            changed = copy.deepcopy(self.inputs)
            changed[field] = value
            self.assertNotEqual(key, page_cache_key(**changed)[0], field)

    def test_write_iterate_export_and_resume_without_consuming_input(self):
        key, inputs = page_cache_key(**self.inputs)
        expected = _page_target()
        drained = copy.deepcopy(expected)
        with PackedProjectStore(self.path, create=True) as store:
            cache = MepV3PageCache(store, transaction_bytes=4 * 1024 * 1024)
            result = cache.write(cache_key=key, cache_inputs=inputs, page_ref="page.5",
                                 records=legacy_page_record_stream(drained))
            self.assertFalse(result["reused"])
            self.assertEqual(drained, {})
            self.assertEqual(cache.export_legacy_page(key), expected)
            self.assertEqual([kind for kind, _ in cache.iter_records(key)], [
                "route_graph", "composites_header", "composite_candidate",
                "envelope_search", "leader_observation", "boundary_connection",
            ])

            def must_not_run():
                raise AssertionError("completed cache hash did not resume")
                yield

            resumed = cache.write(cache_key=key, cache_inputs=inputs, page_ref="page.5",
                                  records=must_not_run())
            self.assertTrue(resumed["reused"])
            self.assertEqual(resumed["artifact_key"], result["artifact_key"])
            self.assertEqual(store.connection.execute("pragma quick_check").fetchone()[0], "ok")

    def test_interrupted_stream_publishes_nothing(self):
        key, inputs = page_cache_key(**self.inputs)

        def interrupted():
            yield "route_graph", _page_target()["route_graph"]
            raise RuntimeError("simulated page interruption")

        with PackedProjectStore(self.path, create=True) as store:
            cache = MepV3PageCache(store, transaction_bytes=4 * 1024 * 1024)
            before = store.connection.execute("select count(*) from artifacts").fetchone()[0]
            with self.assertRaisesRegex(RuntimeError, "simulated page interruption"):
                cache.write(cache_key=key, cache_inputs=inputs, page_ref="page.5",
                            records=interrupted())
            self.assertIsNone(cache.manifest(key))
            self.assertEqual(store.connection.execute("select count(*) from artifacts").fetchone()[0], before)
            self.assertEqual(store.connection.execute("select count(*) from artifact_chunks").fetchone()[0], 0)

    def test_runtime_profile_does_not_change_page_artifact_identity(self):
        key, inputs = page_cache_key(**self.inputs)
        hashes = []
        for ordinal, seconds in enumerate((1.0, 999.0)):
            target = _page_target()
            path = Path(self.directory.name) / f"deterministic-{ordinal}.sqlite"
            with PackedProjectStore(path, create=True) as store:
                cache = MepV3PageCache(store)
                result = cache.write(
                    cache_key=key, cache_inputs=inputs, page_ref="page.5",
                    records=page_component_record_stream(
                        page_record={"page_ref": "page.5", "page_number": 5},
                        regions=[], route_graph=target["route_graph"],
                        composites=target["composites"],
                        envelope_searches=target["envelope_searches"],
                        leader_observations=target["leader_observations"],
                        boundary_connections=target["boundary_connections"],
                        performance_profile={"seconds": seconds}))
                hashes.append(result["artifact_sha256"])
        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
