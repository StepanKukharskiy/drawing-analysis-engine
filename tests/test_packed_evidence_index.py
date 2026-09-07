from pathlib import Path
import tempfile
import unittest
import zlib

from src.drawing_engine.project.packed_evidence_index import PackedEvidenceIndexPilot


class PackedEvidenceIndexPilotTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "packed.sqlite"
        self.rows = [{
            "id": "native-row-" + str(index), "record_type": "native_target",
            "page_ref": "page.a", "state": "observed",
            "source_primitive_ref": "drawing[" + str(index) + "]",
            "source_native_segment": {
                "id": "drawing[" + str(index) + "]", "points": [[index, 0], [index, 1]]},
        } for index in range(100)]
        self.artifact = {
            "schema_version": "0.1.0", "layer": "pilot",
            "items": [{
                "id": "connection", "record_type": "connection", "page_ref": "page.a",
                "state": "accepted", "source_rows": self.rows,
            }, {
                "id": "label", "record_type": "text", "page_ref": "page.a",
                "state": "observed", "text": "AHU-1",
            }],
        }

    def test_dense_rows_are_packed_but_every_record_and_payload_stays_queryable(self):
        with PackedEvidenceIndexPilot(self.path, pack_records=32) as store:
            store.import_artifact("example", self.artifact)
            stats = store.stats()
            self.assertEqual(stats["logical_records"], 202)
            self.assertEqual(stats["regular_records"], 2)
            self.assertEqual(stats["pack_rows"], 7)
            self.assertEqual(stats["relational_rows"], 9)
            self.assertEqual(stats["edge_rows"], 0)
            row = store.query(source_id="native-row-73")[0]
            self.assertEqual(store.node_payload(row["id"]), self.rows[73])
            segment = store.query(source_id="drawing[73]", page_ref="page.a")[0]
            self.assertEqual(store.node_payload(segment["id"]), self.rows[73]["source_native_segment"])
            self.assertEqual(store.artifact("example"), self.artifact)

    def test_packed_neighbors_preserve_resolution_and_raw_page_scope(self):
        duplicate = dict(self.rows[7]["source_native_segment"])
        self.artifact["items"].append({
            "id": duplicate["id"], "record_type": "primitive", "page_ref": "page.b",
            "points": duplicate["points"],
        })
        with PackedEvidenceIndexPilot(self.path, pack_records=16) as store:
            store.import_artifact("example", self.artifact)
            row = store.query(source_id="native-row-7")[0]
            relation = next(item for item in store.neighbors(row["id"])
                            if item["target_ref"] == "drawing[7]")
            self.assertEqual(relation["resolution"], "unique_record")
            self.assertEqual({target["page_ref"] for target in relation["targets"]}, {"page.a"})
            self.assertFalse(relation["engineering_authority_inferred"])

    def test_filters_paging_idempotence_and_corruption_fail_closed(self):
        with PackedEvidenceIndexPilot(self.path, pack_records=20) as store:
            digest = store.import_artifact("example", self.artifact)
            self.assertEqual(store.import_artifact("example", self.artifact), digest)
            self.assertEqual(store.import_artifact("same-content", self.artifact), digest)
            filtered = store.query(record_type="native_target", state="observed",
                                   page_ref="page.a", limit=10, offset=10)
            self.assertEqual(len(filtered), 10)
            self.assertTrue(all(row["record_type"] == "native_target" for row in filtered))
            store.connection.execute(
                "UPDATE pilot_dense_packs SET payload_zlib=? "
                "WHERE id=(SELECT id FROM pilot_dense_packs LIMIT 1)",
                (zlib.compress(b"[]"),))
            with self.assertRaisesRegex(ValueError, "pack content hash changed"):
                store.query(record_type="native_target")

    def test_corrupt_skip_metadata_cannot_hide_a_pack(self):
        with PackedEvidenceIndexPilot(self.path, pack_records=20) as store:
            store.import_artifact("example", self.artifact)
            store.connection.execute(
                "UPDATE pilot_dense_packs SET source_bloom=zeroblob(length(source_bloom)) "
                "WHERE id=(SELECT id FROM pilot_dense_packs LIMIT 1)")
            with self.assertRaisesRegex(ValueError, "metadata hash changed"):
                store.query(source_id="native-row-1")


if __name__ == "__main__":
    unittest.main()
