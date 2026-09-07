"""Schema-v3 packed projection store for project packages.

The immutable canonical artifact remains authoritative. Dense ``source_rows``
records are descriptors inside verified compressed packs; they are never
expanded into ordinary SQLite node or edge rows. Stable public IDs and JSON
pointers are reconstructed at the API boundary.
"""

from collections import defaultdict, OrderedDict
from contextlib import AbstractContextManager
import bisect
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import struct
import zlib

from src.drawing_engine.project.packed_evidence_index import _bloom, _maybe_contains
from src.drawing_engine.project.project_knowledge_store import (_ARTIFACT_ONLY_REFERENCE_FIELDS, CHUNK_BYTES,
                                         _fts_text, _json)


VERSION = "0.3.0"
SCHEMA_VERSION = 3
PACK_RECORDS = 4096
_NODE_ENTRY = struct.Struct(">32sIH")
_SOURCE_ENTRY = struct.Struct(">16sIH")
_REVERSE_ENTRY = struct.Struct(">16sIHH")


def _binary_hash(value):
    return hashlib.sha256(_json(value).encode()).digest()


def _public_id(artifact_sha256, pointer):
    return _binary_hash([artifact_sha256, pointer])


def _hash16(value):
    return hashlib.sha256(value.encode()).digest()[:16]


def _decode_verified(payload_zlib, payload_sha256, label):
    raw = zlib.decompress(payload_zlib)
    if hashlib.sha256(raw).digest() != payload_sha256:
        raise ValueError(label + " content hash changed")
    return raw


class _PackCache:
    def __init__(self, maximum=8):
        self.maximum = maximum
        self.values = OrderedDict()

    def get(self, key, loader):
        value = self.values.pop(key, None)
        if value is None:
            value = loader()
        self.values[key] = value
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return value


class PackedProjectStore(AbstractContextManager):
    """Read/write interface for a schema-v3 packed project package."""

    def __init__(self, path, *, create=False, readonly=False):
        self.path = Path(path)
        if create and readonly:
            raise ValueError("cannot create a read-only store")
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if create and self.path.exists():
            raise ValueError("fresh schema-v3 path already exists")
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60) if readonly else sqlite3.connect(self.path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if create:
            if version:
                raise ValueError("fresh schema-v3 database is not empty")
            self._create_schema()
        elif version != SCHEMA_VERSION:
            self.connection.close()
            raise ValueError("schema-v3 packed project database is required")
        self._descriptor_cache = _PackCache()
        self._locator_cache = _PackCache(maximum=4)
        self._artifact_chunk_cache = _PackCache(maximum=8)
        self._artifact_chunk_rows_cache = _PackCache(maximum=8)
        self._dictionary_cache = {}

    def _create_schema(self):
        self.connection.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE dictionaries (
                kind TEXT NOT NULL, key INTEGER NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY(kind,key), UNIQUE(kind,value)) WITHOUT ROWID;
            CREATE TABLE artifacts (
                artifact_key INTEGER PRIMARY KEY, sha256 BLOB NOT NULL UNIQUE,
                byte_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL);
            CREATE TABLE artifact_chunks (
                artifact_key INTEGER NOT NULL REFERENCES artifacts(artifact_key),
                ordinal INTEGER NOT NULL, uncompressed_offset INTEGER NOT NULL,
                uncompressed_size INTEGER NOT NULL, payload_sha256 BLOB NOT NULL,
                payload_zlib BLOB NOT NULL, PRIMARY KEY(artifact_key,ordinal)) WITHOUT ROWID;
            CREATE TABLE snapshots (
                snapshot_key INTEGER PRIMARY KEY, id BLOB NOT NULL UNIQUE,
                project_id TEXT NOT NULL, document_id TEXT NOT NULL,
                source_sha256 BLOB NOT NULL, context_json TEXT NOT NULL,
                manifest_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE active_documents (
                project_id TEXT NOT NULL, document_id TEXT NOT NULL,
                snapshot_key INTEGER NOT NULL REFERENCES snapshots(snapshot_key),
                PRIMARY KEY(project_id,document_id)) WITHOUT ROWID;
            CREATE TABLE snapshot_artifacts (
                snapshot_key INTEGER NOT NULL REFERENCES snapshots(snapshot_key),
                name TEXT NOT NULL, artifact_key INTEGER NOT NULL REFERENCES artifacts(artifact_key),
                PRIMARY KEY(snapshot_key,name)) WITHOUT ROWID;
            CREATE TABLE collections (
                collection_key INTEGER PRIMARY KEY, artifact_key INTEGER NOT NULL REFERENCES artifacts(artifact_key),
                pointer TEXT NOT NULL, UNIQUE(artifact_key,pointer));
            CREATE TABLE descriptor_packs (
                pack_key INTEGER PRIMARY KEY, collection_key INTEGER NOT NULL REFERENCES collections(collection_key),
                record_count INTEGER NOT NULL, payload_sha256 BLOB NOT NULL, payload_zlib BLOB NOT NULL,
                node_bloom BLOB NOT NULL, source_bloom BLOB NOT NULL,
                page_keys_json TEXT NOT NULL, type_keys_json TEXT NOT NULL,
                state_keys_json TEXT NOT NULL, metadata_sha256 BLOB NOT NULL);
            CREATE TABLE node_locators (
                bucket INTEGER PRIMARY KEY, entry_count INTEGER NOT NULL,
                payload_sha256 BLOB NOT NULL, payload_zlib BLOB NOT NULL) WITHOUT ROWID;
            CREATE TABLE source_locators (
                bucket INTEGER PRIMARY KEY, entry_count INTEGER NOT NULL,
                payload_sha256 BLOB NOT NULL, payload_zlib BLOB NOT NULL) WITHOUT ROWID;
            CREATE TABLE packed_reverse_edges (
                bucket INTEGER PRIMARY KEY, entry_count INTEGER NOT NULL,
                payload_sha256 BLOB NOT NULL, payload_zlib BLOB NOT NULL) WITHOUT ROWID;
            CREATE TABLE hot_nodes (
                node_key INTEGER PRIMARY KEY, public_id BLOB NOT NULL UNIQUE,
                artifact_key INTEGER NOT NULL REFERENCES artifacts(artifact_key), pointer TEXT NOT NULL,
                source_id TEXT, type_key INTEGER NOT NULL, page_key INTEGER, state_key INTEGER,
                payload_start INTEGER NOT NULL, payload_length INTEGER NOT NULL);
            CREATE INDEX hot_nodes_source ON hot_nodes(source_id,page_key);
            CREATE INDEX hot_nodes_kind ON hot_nodes(type_key,state_key,page_key);
            CREATE TABLE hot_edges (
                node_key INTEGER NOT NULL REFERENCES hot_nodes(node_key), field_key INTEGER NOT NULL,
                target_ref TEXT NOT NULL, PRIMARY KEY(node_key,field_key,target_ref)) WITHOUT ROWID;
            CREATE INDEX hot_edges_target ON hot_edges(target_ref);
            CREATE TABLE fts_node_map (
                rowid INTEGER PRIMARY KEY, public_id BLOB NOT NULL UNIQUE);
            CREATE VIRTUAL TABLE fts_authored_text USING fts5(
                source_id,record_type,page_ref,state,body,content='',tokenize='unicode61');
            CREATE TABLE reviews (
                id BLOB PRIMARY KEY, snapshot_key INTEGER NOT NULL REFERENCES snapshots(snapshot_key),
                public_node_id BLOB NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE pilot_manifest (
                key TEXT PRIMARY KEY, value_json TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE migration_artifacts (
                sha256 BLOB PRIMARY KEY REFERENCES artifacts(sha256),
                source_schema INTEGER NOT NULL, status TEXT NOT NULL,
                node_count INTEGER NOT NULL, edge_count INTEGER NOT NULL,
                timing_json TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE migration_phase_timings (
                phase TEXT PRIMARY KEY, elapsed_seconds REAL NOT NULL,
                detail_json TEXT NOT NULL) WITHOUT ROWID;
            PRAGMA user_version=3;
        """)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def dictionary_key(self, kind, value):
        if value is None:
            return None
        cache = self._dictionary_cache.setdefault(kind, {})
        if value in cache:
            return cache[value]
        row = self.connection.execute(
            "SELECT key FROM dictionaries WHERE kind=? AND value=?", (kind, value)).fetchone()
        if row:
            key = row[0]
        else:
            key = self.connection.execute(
                "SELECT coalesce(max(key),0)+1 FROM dictionaries WHERE kind=?", (kind,)).fetchone()[0]
            self.connection.execute("INSERT INTO dictionaries VALUES(?,?,?)", (kind, key, value))
        cache[value] = key
        return key

    def dictionary_value(self, kind, key):
        if key is None:
            return None
        cache = self._dictionary_cache.setdefault(kind + ":reverse", {})
        if key not in cache:
            row = self.connection.execute(
                "SELECT value FROM dictionaries WHERE kind=? AND key=?", (kind, key)).fetchone()
            if row is None:
                raise ValueError("packed dictionary key is missing")
            cache[key] = row[0]
        return cache[key]

    def put_manifest(self, key, value):
        self.connection.execute(
            "INSERT INTO pilot_manifest VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, _json(value)))

    def manifest(self, key):
        row = self.connection.execute("SELECT value_json FROM pilot_manifest WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def snapshot(self, *, project_id, document_id, snapshot_id=None):
        clauses = ["project_id=?", "document_id=?"]
        parameters = [project_id, document_id]
        if snapshot_id is None:
            sql = ("SELECT s.* FROM active_documents a JOIN snapshots s ON s.snapshot_key=a.snapshot_key "
                   "WHERE a.project_id=? AND a.document_id=?")
        else:
            clauses.append("id=?")
            parameters.append(bytes.fromhex(snapshot_id))
            sql = "SELECT * FROM snapshots WHERE " + " AND ".join(clauses)
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            raise KeyError("project snapshot not found")
        result = dict(row)
        result["id"] = result["id"].hex()
        result["source_sha256"] = result["source_sha256"].hex()
        result["context"] = json.loads(result.pop("context_json"))
        result["manifest"] = json.loads(result.pop("manifest_json"))
        return result

    def _artifact_keys(self, snapshot_key):
        return {row[0] for row in self.connection.execute(
            "SELECT artifact_key FROM snapshot_artifacts WHERE snapshot_key=?", (snapshot_key,))}

    def snapshots(self, *, project_id, document_id):
        """Return retained versions without exposing schema-v3 storage keys."""
        return [{**dict(row), "id": row["id"].hex(),
                 "source_sha256": row["source_sha256"].hex()}
                for row in self.connection.execute(
                    "SELECT s.id,s.created_at,s.source_sha256,"
                    "(a.snapshot_key=s.snapshot_key) AS active FROM snapshots s "
                    "LEFT JOIN active_documents a ON a.project_id=s.project_id "
                    "AND a.document_id=s.document_id WHERE s.project_id=? "
                    "AND s.document_id=? ORDER BY s.created_at DESC,s.id",
                    (project_id, document_id))]

    def artifact_names(self, *, project_id, document_id, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id,
                                 snapshot_id=snapshot_id)
        return [row[0] for row in self.connection.execute(
            "SELECT name FROM snapshot_artifacts WHERE snapshot_key=? ORDER BY name",
            (snapshot["snapshot_key"],))]

    def collection(self, *, project_id, document_id, name, collection,
                   snapshot_id=None, source_ids=None, incoming_refs=None,
                   include_payload=True):
        """Read one direct canonical record array in original array order.

        The app-facing records live in ``hot_nodes``. Dense nested evidence
        remains in descriptor packs and must be reached through source/query
        APIs, so this method can never accidentally expand a ``source_rows``
        collection.
        """
        if (not isinstance(collection, str) or not collection
                or "/" in collection or collection == "source_rows"):
            raise ValueError("collection must name one direct record array")
        snapshot = self.snapshot(project_id=project_id, document_id=document_id,
                                 snapshot_id=snapshot_id)
        artifact = self.connection.execute(
            "SELECT sa.artifact_key FROM snapshot_artifacts sa "
            "WHERE sa.snapshot_key=? AND sa.name=?",
            (snapshot["snapshot_key"], name)).fetchone()
        if artifact is None:
            raise KeyError(name)
        prefix = "/" + collection + "/"
        clauses = ["h.artifact_key=?", "h.pointer LIKE ?",
                   "instr(substr(h.pointer,?),'/')=0"]
        parameters = [artifact[0], prefix + "%", len(prefix) + 1]
        if source_ids is not None:
            values = sorted(set(source_ids))
            if not values:
                return []
            clauses.append("h.source_id IN (" + ",".join("?" for _ in values) + ")")
            parameters.extend(values)
        if incoming_refs is not None:
            values = sorted(set(incoming_refs))
            if not values:
                return []
            clauses.append("EXISTS (SELECT 1 FROM hot_edges e WHERE e.node_key=h.node_key "
                           "AND e.target_ref IN (" + ",".join("?" for _ in values) + "))")
            parameters.extend(values)
        sql = ("SELECT h.* FROM hot_nodes h WHERE " + " AND ".join(clauses)
               + " ORDER BY CAST(substr(h.pointer,?) AS INTEGER)")
        parameters.append(len(prefix) + 1)
        rows = []
        for stored in self.connection.execute(sql, parameters):
            row = self._hot_descriptor(stored)
            payload = None
            if include_payload:
                payload = json.loads(self._chunk_bytes(
                    row["artifact_key"], row["payload_start"], row["payload_length"]))
                if not isinstance(payload, dict) or payload.get("id") != row["source_id"]:
                    raise ValueError("artifact pointer differs from indexed record")
            rows.append({**self._public_row(row), "payload": payload,
                         "references": [{"field": field, "target_ref": target}
                                        for field, target in sorted(row["references"])]})
        return rows

    def _decode_descriptor_pack(self, pack_key):
        def load():
            row = self.connection.execute(
                "SELECT p.*,c.pointer,c.artifact_key,a.sha256 FROM descriptor_packs p "
                "JOIN collections c ON c.collection_key=p.collection_key "
                "JOIN artifacts a ON a.artifact_key=c.artifact_key WHERE p.pack_key=?", (pack_key,)).fetchone()
            if row is None:
                raise ValueError("descriptor pack is missing")
            metadata = _binary_hash([row["collection_key"], row["record_count"],
                row["payload_sha256"].hex(), row["node_bloom"].hex(), row["source_bloom"].hex(),
                row["page_keys_json"], row["type_keys_json"], row["state_keys_json"]])
            if metadata != row["metadata_sha256"]:
                raise ValueError("descriptor pack metadata hash changed")
            raw = _decode_verified(row["payload_zlib"], row["payload_sha256"], "descriptor pack")
            body = json.loads(raw)
            if len(body["r"]) != row["record_count"]:
                raise ValueError("descriptor pack record count changed")
            return row, body
        return self._descriptor_cache.get(pack_key, load)

    def _descriptor(self, pack_key, ordinal):
        pack, body = self._decode_descriptor_pack(pack_key)
        try:
            record = body["r"][ordinal]
        except IndexError as exc:
            raise ValueError("packed node ordinal is invalid") from exc
        local_ordinal, suffix_index, source_id, type_key, page_key, state_key, start, length, refs = record
        suffix = body["s"][suffix_index]
        pointer = pack["pointer"] + "/" + str(local_ordinal) + suffix
        record_type = str(local_ordinal) if type_key == 0 else self.dictionary_value("type", type_key)
        targets = body["t"]
        references = [(self.dictionary_value("field", field_key), targets[target_index])
                      for field_key, target_index in refs]
        public_id = _public_id(pack["sha256"].hex(), pointer)
        return {"id": public_id.hex(), "_public_id": public_id, "artifact_sha256": pack["sha256"].hex(),
                "artifact_key": pack["artifact_key"], "pointer": pointer, "source_id": source_id,
                "record_type": record_type, "page_ref": self.dictionary_value("page", page_key),
                "state": self.dictionary_value("state", state_key), "payload_json": None,
                "payload_start": start, "payload_length": length, "references": references,
                "_pack_key": pack_key, "_pack_ordinal": ordinal}

    def _hot_descriptor(self, row):
        edges = [(self.dictionary_value("field", edge[0]), edge[1]) for edge in self.connection.execute(
            "SELECT field_key,target_ref FROM hot_edges WHERE node_key=? ORDER BY field_key,target_ref", (row["node_key"],))]
        digest = self.connection.execute("SELECT sha256 FROM artifacts WHERE artifact_key=?",
                                         (row["artifact_key"],)).fetchone()[0]
        return {"id": row["public_id"].hex(), "_public_id": row["public_id"],
                "artifact_sha256": digest.hex(), "artifact_key": row["artifact_key"],
                "pointer": row["pointer"], "source_id": row["source_id"],
                "record_type": self.dictionary_value("type", row["type_key"]),
                "page_ref": self.dictionary_value("page", row["page_key"]),
                "state": self.dictionary_value("state", row["state_key"]), "payload_json": None,
                "payload_start": row["payload_start"], "payload_length": row["payload_length"],
                "references": edges, "_node_key": row["node_key"]}

    def _locator_payload(self, table, bucket, entry):
        key = (table, bucket)
        def load():
            row = self.connection.execute(
                f"SELECT * FROM {table} WHERE bucket=?", (bucket,)).fetchone()
            if row is None:
                return b""
            raw = _decode_verified(row["payload_zlib"], row["payload_sha256"], table)
            if len(raw) != row["entry_count"] * entry.size:
                raise ValueError(table + " entry count changed")
            return raw
        return self._locator_cache.get(key, load)

    @staticmethod
    def _fixed_matches(raw, entry, needle):
        count = len(raw) // entry.size
        keys = [raw[index * entry.size:index * entry.size + len(needle)] for index in range(count)]
        left = bisect.bisect_left(keys, needle)
        right = bisect.bisect_right(keys, needle)
        return [entry.unpack_from(raw, index * entry.size) for index in range(left, right)]

    def _node(self, snapshot_key, node_id):
        try:
            public_id = bytes.fromhex(node_id)
        except ValueError as exc:
            raise KeyError("node does not belong to frozen snapshot") from exc
        if len(public_id) != 32:
            raise KeyError("node does not belong to frozen snapshot")
        artifacts = self._artifact_keys(snapshot_key)
        hot = self.connection.execute("SELECT * FROM hot_nodes WHERE public_id=?", (public_id,)).fetchone()
        if hot and hot["artifact_key"] in artifacts:
            return self._hot_descriptor(hot)
        raw = self._locator_payload("node_locators", public_id[0], _NODE_ENTRY)
        matches = self._fixed_matches(raw, _NODE_ENTRY, public_id)
        rows = [self._descriptor(pack_key, ordinal) for _, pack_key, ordinal in matches]
        rows = [row for row in rows if row["artifact_key"] in artifacts and row["_public_id"] == public_id]
        if len(rows) != 1:
            raise KeyError("node does not belong to frozen snapshot")
        return rows[0]

    def iter_nodes(self, *, project_id, document_id, snapshot_id=None):
        """Stream every projected node in exact public-ID order with bounded RSS."""
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        artifacts = self._artifact_keys(snapshot["snapshot_key"])
        hot_by_bucket = defaultdict(list)
        for row in self.connection.execute("SELECT * FROM hot_nodes ORDER BY public_id"):
            if row["artifact_key"] in artifacts:
                decoded = self._hot_descriptor(row)
                hot_by_bucket[row["public_id"][0]].append(decoded)
        for bucket in range(256):
            rows = list(hot_by_bucket.get(bucket, ()))
            raw = self._locator_payload("node_locators", bucket, _NODE_ENTRY)
            by_pack = defaultdict(list)
            for offset in range(0, len(raw), _NODE_ENTRY.size):
                public_id, pack_key, ordinal = _NODE_ENTRY.unpack_from(raw, offset)
                by_pack[pack_key].append((public_id, ordinal))
            # Hash ordering distributes every collection pack across buckets.
            # Grouping here guarantees one decode per pack per bucket instead
            # of thrashing the bounded point-query cache once per node.
            for pack_key in sorted(by_pack):
                self._decode_descriptor_pack(pack_key)
                for public_id, ordinal in by_pack[pack_key]:
                    decoded = self._descriptor(pack_key, ordinal)
                    if decoded["artifact_key"] in artifacts and decoded["_public_id"] == public_id:
                        rows.append(decoded)
            rows.sort(key=lambda row: row["_public_id"])
            yield from rows

    def iter_storage_nodes(self, *, project_id, document_id, snapshot_id=None):
        """Stream every node once in physical storage order.

        This is intended for whole-store verification and export planning where
        public-ID ordering is irrelevant.  It deliberately avoids decoding each
        descriptor pack again for every public-ID hash bucket.
        """
        snapshot = self.snapshot(project_id=project_id, document_id=document_id,
                                 snapshot_id=snapshot_id)
        artifacts = self._artifact_keys(snapshot["snapshot_key"])
        for row in self.connection.execute("SELECT * FROM hot_nodes ORDER BY node_key"):
            if row["artifact_key"] in artifacts:
                yield self._hot_descriptor(row)
        for pack_key, record_count in self.connection.execute(
                "SELECT p.pack_key,p.record_count FROM descriptor_packs p "
                "JOIN collections c ON c.collection_key=p.collection_key "
                "WHERE c.artifact_key IN (SELECT artifact_key FROM snapshot_artifacts "
                "WHERE snapshot_key=?) ORDER BY p.pack_key", (snapshot["snapshot_key"],)):
            self._decode_descriptor_pack(pack_key)
            for ordinal in range(record_count):
                yield self._descriptor(pack_key, ordinal)

    def _source_matches(self, snapshot_key, source_id):
        artifacts = self._artifact_keys(snapshot_key)
        output = [self._hot_descriptor(row) for row in self.connection.execute(
            "SELECT * FROM hot_nodes WHERE source_id=?", (source_id,)) if row["artifact_key"] in artifacts]
        digest = _hash16(source_id)
        raw = self._locator_payload("source_locators", digest[0], _SOURCE_ENTRY)
        for packed_hash, pack_key, ordinal in self._fixed_matches(raw, _SOURCE_ENTRY, digest):
            row = self._descriptor(pack_key, ordinal)
            if row["artifact_key"] in artifacts and row["source_id"] == source_id:
                output.append(row)
        return output

    @staticmethod
    def _matches(row, filters):
        return all(value is None or row[field] == value for field, value in filters.items())

    @staticmethod
    def _public_row(row):
        return {key:value for key, value in row.items() if not key.startswith("_") and key not in {"references", "artifact_key"}}

    def query(self, *, project_id, document_id, snapshot_id=None, record_type=None,
              state=None, page_ref=None, source_id=None, limit=100, offset=0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("query limit must be 1..1000 and offset nonnegative")
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        snapshot_key = snapshot["snapshot_key"]
        filters = {"record_type": record_type, "state": state, "page_ref": page_ref, "source_id": source_id}
        if source_id is not None:
            rows = [row for row in self._source_matches(snapshot_key, source_id) if self._matches(row, filters)]
            selected = sorted(rows, key=lambda row: row["id"])[offset:offset + limit]
        else:
            matching = (row for row in self.iter_nodes(project_id=project_id, document_id=document_id,
                        snapshot_id=snapshot["id"]) if self._matches(row, filters))
            selected = list(itertools.islice(matching, offset, offset + limit))
        return [{**self._public_row(row), "payload": None, "payload_in_artifact_only": True,
                 "artifact_only_reference_fields": sorted(_ARTIFACT_ONLY_REFERENCE_FIELDS)} for row in selected]

    def neighbors(self, *, project_id, document_id, node_id, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        node = self._node(snapshot["snapshot_key"], node_id)
        output = []
        for field, target_ref in sorted(node["references"]):
            raw = target_ref.startswith(("drawing[", "page["))
            matches = [row for row in self._source_matches(snapshot["snapshot_key"], target_ref)
                       if not raw or row["page_ref"] == node["page_ref"]]
            matches.sort(key=lambda row: row["id"])
            output.append({"field": field, "target_ref": target_ref,
                "resolution": "unresolved" if not matches else "unique_record" if len(matches) == 1 else "multiple_records",
                "targets": [{key:row[key] for key in ("id","source_id","page_ref","record_type","state")} for row in matches],
                "engineering_authority_inferred": False})
        return output

    def reverse_neighbors(self, *, project_id, document_id, node_id, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        target = self._node(snapshot["snapshot_key"], node_id)
        if not target["source_id"]:
            return []
        artifacts = self._artifact_keys(snapshot["snapshot_key"])
        output = []
        for row in self.connection.execute(
                "SELECT h.*,e.field_key,e.target_ref FROM hot_edges e JOIN hot_nodes h ON h.node_key=e.node_key WHERE e.target_ref=?",
                (target["source_id"],)):
            if row["artifact_key"] in artifacts:
                output.append((self._hot_descriptor(row), self.dictionary_value("field", row["field_key"])))
        digest = _hash16(target["source_id"])
        raw = self._locator_payload("packed_reverse_edges", digest[0], _REVERSE_ENTRY)
        for packed_hash, pack_key, ordinal, field_key in self._fixed_matches(raw, _REVERSE_ENTRY, digest):
            source = self._descriptor(pack_key, ordinal)
            if source["artifact_key"] not in artifacts:
                continue
            field = self.dictionary_value("field", field_key)
            if (field, target["source_id"]) in source["references"]:
                output.append((source, field))
        output.sort(key=lambda item: (item[0]["id"], item[1]))
        return [{"field":field, "target_ref":target["source_id"], "source":{
            key:source[key] for key in ("id","source_id","page_ref","record_type","state")},
            "engineering_authority_inferred": False} for source,field in output]

    def _chunk_bytes(self, artifact_key, start, length):
        end, output = start + length, bytearray()
        chunks = self._artifact_chunk_rows_cache.get(artifact_key,lambda:[dict(row) for row in
            self.connection.execute("SELECT * FROM artifact_chunks WHERE artifact_key=? ORDER BY ordinal",
                                    (artifact_key,))])
        for row in chunks:
            if row['uncompressed_offset'] >= end:
                break
            if row['uncompressed_offset'] + row['uncompressed_size'] <= start:
                continue
            raw = self._artifact_chunk_cache.get((artifact_key,row['ordinal']),
                lambda row=row:_decode_verified(row["payload_zlib"],row["payload_sha256"],
                                                 "artifact chunk"))
            if len(raw) != row["uncompressed_size"]:
                raise ValueError("artifact chunk size changed")
            left = max(start,row["uncompressed_offset"]) - row["uncompressed_offset"]
            right = min(end,row["uncompressed_offset"] + len(raw)) - row["uncompressed_offset"]
            output.extend(raw[left:right])
        if len(output) != length:
            raise ValueError("artifact chunk coverage is incomplete")
        return bytes(output)

    def node_payload(self, *, project_id, document_id, node_id, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        node = self._node(snapshot["snapshot_key"], node_id)
        return json.loads(self._chunk_bytes(node["artifact_key"], node["payload_start"], node["payload_length"]))

    def iter_artifact_bytes(self, *, project_id, document_id, name, snapshot_id=None):
        snapshot = self.snapshot(project_id=project_id, document_id=document_id, snapshot_id=snapshot_id)
        row = self.connection.execute(
            "SELECT a.* FROM snapshot_artifacts sa JOIN artifacts a ON a.artifact_key=sa.artifact_key "
            "WHERE sa.snapshot_key=? AND sa.name=?", (snapshot["snapshot_key"],name)).fetchone()
        if row is None:
            raise KeyError(name)
        digest, offset = hashlib.sha256(), 0
        for chunk in self.connection.execute("SELECT * FROM artifact_chunks WHERE artifact_key=? ORDER BY ordinal",
                                             (row["artifact_key"],)):
            raw = _decode_verified(chunk["payload_zlib"], chunk["payload_sha256"], "artifact chunk")
            if len(raw) != chunk["uncompressed_size"] or chunk["uncompressed_offset"] != offset:
                raise ValueError("artifact chunk coverage changed")
            digest.update(raw); offset += len(raw); yield raw
        if offset != row["byte_count"] or digest.digest() != row["sha256"]:
            raise ValueError("artifact content hash changed")

    def search(self, *, project_id, document_id, text, snapshot_id=None, limit=100, offset=0):
        if not isinstance(text,str) or not text.strip() or len(text)>500:
            raise ValueError("search text must contain 1..500 characters")
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("search limit must be 1..1000 and offset nonnegative")
        snapshot = self.snapshot(project_id=project_id,document_id=document_id,snapshot_id=snapshot_id)
        rows=[]
        for hit in self.connection.execute(
            "SELECT m.public_id,bm25(fts_authored_text) rank FROM fts_authored_text "
            "JOIN fts_node_map m ON m.rowid=fts_authored_text.rowid WHERE fts_authored_text MATCH ? "
            "ORDER BY rank,m.public_id LIMIT ? OFFSET ?",(text,limit,offset)):
            try: row=self._node(snapshot["snapshot_key"],hit["public_id"].hex())
            except KeyError: continue
            rows.append({**self._public_row(row),"rank":hit["rank"],"retrieval_authority":"proposal_only",
                         "engineering_authority_inferred":False})
        return rows

    def reviews(self, *, project_id, document_id, snapshot_id=None):
        snapshot=self.snapshot(project_id=project_id,document_id=document_id,snapshot_id=snapshot_id)
        return [{"id":row["id"].hex(),"snapshot_id":snapshot["id"],"node_id":row["public_node_id"].hex(),
                 "payload_json":row["payload_json"],"created_at":row["created_at"],
                 "review":json.loads(row["payload_json"])} for row in self.connection.execute(
                     "SELECT * FROM reviews WHERE snapshot_key=? ORDER BY id",(snapshot["snapshot_key"],))]

    def stats(self):
        return {name:self.connection.execute("SELECT count(*) FROM " + name).fetchone()[0] for name in
                ("artifacts","artifact_chunks","collections","descriptor_packs","node_locators",
                 "source_locators","packed_reverse_edges","hot_nodes","hot_edges","fts_node_map","reviews")}

    def quick_check(self):
        """Return SQLite's bounded structural check and fail on any extra row."""
        rows = [row[0] for row in self.connection.execute("PRAGMA quick_check")]
        return {"rows": rows, "passed": rows == ["ok"]}
