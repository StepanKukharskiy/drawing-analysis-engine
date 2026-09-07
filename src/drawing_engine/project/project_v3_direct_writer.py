"""Bounded direct writer for canonical artifacts in a schema-v3 package.

This is a persistence adapter only.  It serializes the producer's frozen
artifact directly into verified SQLite chunks and builds the same hot/packed
projection used by the migrated store.  It never changes engineering states,
identities, references, or authority.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

from src.drawing_engine.project.packed_evidence_index import _bloom
from src.drawing_engine.operations.artifact_disk_usage import artifact_disk_usage, check_capacity
from src.drawing_engine.project.project_knowledge_store import (_ARTIFACT_ONLY_REFERENCE_FIELDS,
    CHUNK_BYTES, _fts_text, _json, _references)
from src.drawing_engine.project.project_packed_store import (_NODE_ENTRY, _REVERSE_ENTRY, _SOURCE_ENTRY,
    PACK_RECORDS, PackedProjectStore, _binary_hash, _decode_verified, _hash16,
    _public_id)


DIRECT_ARTIFACTS = frozenset({
    "trace-source-queries", "outlined-route-connections", "mep-automatic-page-cache",
})


class RecordStream:
    """Single-use canonical array input consumed without list materialization."""

    def __init__(self, records):
        if not isinstance(records, Iterable):
            raise TypeError("record stream must be iterable")
        self.records = iter(records)
        self.consumed = False

    def __iter__(self):
        if self.consumed:
            raise ValueError("record stream is single-use")
        self.consumed = True
        return self.records


def _escape(key):
    return str(key).replace("~", "~0").replace("/", "~1")


class _LocatorSpool:
    def __init__(self, directory, name, entry):
        self.directory = Path(directory)
        self.name = name
        self.entry = entry
        self.buffers = defaultdict(bytearray)
        self.counts = defaultdict(int)

    def append(self, bucket, *values):
        self.buffers[bucket].extend(self.entry.pack(*values))
        self.counts[bucket] += 1
        if len(self.buffers[bucket]) >= 1024 * 1024:
            self._flush(bucket)

    def _path(self, bucket):
        return self.directory / f"{self.name}-{bucket:03d}.bin"

    def _flush(self, bucket):
        if not self.buffers[bucket]:
            return
        with self._path(bucket).open("ab") as stream:
            stream.write(self.buffers[bucket])
        self.buffers[bucket].clear()

    def merge(self, connection, table, *, compression_level):
        for bucket in sorted(self.counts):
            self._flush(bucket)
            incoming = self._path(bucket).read_bytes()
            if len(incoming) != self.counts[bucket] * self.entry.size:
                raise ValueError("direct locator spool length changed")
            prior = connection.execute(
                f"SELECT payload_sha256,payload_zlib FROM {table} WHERE bucket=?",
                (bucket,)).fetchone()
            existing = b"" if prior is None else _decode_verified(
                prior["payload_zlib"], prior["payload_sha256"], table + " bucket")
            rows = [self.entry.unpack_from(body, offset)
                    for body in (existing, incoming)
                    for offset in range(0, len(body), self.entry.size)]
            rows.sort()
            packed = b"".join(self.entry.pack(*row) for row in rows)
            connection.execute(
                f"INSERT INTO {table}(bucket,entry_count,payload_sha256,payload_zlib) "
                "VALUES(?,?,?,?) ON CONFLICT(bucket) DO UPDATE SET "
                "entry_count=excluded.entry_count,payload_sha256=excluded.payload_sha256,"
                "payload_zlib=excluded.payload_zlib",
                (bucket, len(rows), hashlib.sha256(packed).digest(),
                 zlib.compress(packed, level=compression_level)))


class _DirectDescriptorPacker:
    def __init__(self, store, artifact_key, *, compression_level):
        self.store = store
        self.connection = store.connection
        self.artifact_key = artifact_key
        self.compression_level = compression_level
        self.collection_pointer = None
        self.records = []
        self._suffixes = {}
        self._targets = {}
        self.pack_keys = []

    @staticmethod
    def split_pointer(pointer):
        prefix, tail = pointer.split("/source_rows/", 1)
        ordinal, *suffix = tail.split("/", 1)
        return prefix + "/source_rows", int(ordinal), "" if not suffix else "/" + suffix[0]

    def add(self, row, references):
        collection, ordinal, suffix = self.split_pointer(row["pointer"])
        if self.collection_pointer is not None and (
                collection != self.collection_pointer or len(self.records) >= PACK_RECORDS):
            self.flush()
        self.collection_pointer = collection
        suffix_index = self._suffixes.setdefault(suffix, len(self._suffixes))
        encoded_refs = []
        for field, target in references:
            field_key = self.store.dictionary_key("field", field)
            encoded_refs.append([field_key, self._targets.setdefault(target, len(self._targets))])
        record_type = row["record_type"]
        type_key = (0 if suffix == "/source_native_segment" and record_type == str(ordinal)
                    else self.store.dictionary_key("type", record_type))
        self.records.append([ordinal, suffix_index, row["source_id"], type_key,
            self.store.dictionary_key("page", row["page_ref"]),
            self.store.dictionary_key("state", row["state"]),
            row["payload_start"], row["payload_length"], encoded_refs])

    def flush(self):
        if not self.records:
            return
        row = self.connection.execute(
            "SELECT collection_key FROM collections WHERE artifact_key=? AND pointer=?",
            (self.artifact_key, self.collection_pointer)).fetchone()
        collection_key = row[0] if row else self.connection.execute(
            "INSERT INTO collections(artifact_key,pointer) VALUES(?,?)",
            (self.artifact_key, self.collection_pointer)).lastrowid
        suffixes = [None] * len(self._suffixes)
        for value, index in self._suffixes.items():
            suffixes[index] = value
        targets = [None] * len(self._targets)
        for value, index in self._targets.items():
            targets[index] = value
        body = _json({"s": suffixes, "t": targets, "r": self.records}).encode()
        payload_sha = hashlib.sha256(body).digest()
        # Public IDs need the final artifact digest, so locator blooms are
        # finalized after the streaming hash closes.
        pages = _json(sorted({record[4] for record in self.records if record[4] is not None}))
        types = _json(sorted({record[3] for record in self.records}))
        states = _json(sorted({record[5] for record in self.records if record[5] is not None}))
        node_bloom = _bloom(())
        source_bloom = _bloom(record[2] for record in self.records if record[2])
        metadata_sha = _binary_hash([collection_key, len(self.records), payload_sha.hex(),
            node_bloom.hex(), source_bloom.hex(), pages, types, states])
        pack_key = self.connection.execute(
            "INSERT INTO descriptor_packs(collection_key,record_count,payload_sha256,"
            "payload_zlib,node_bloom,source_bloom,page_keys_json,type_keys_json,"
            "state_keys_json,metadata_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (collection_key, len(self.records), payload_sha,
             zlib.compress(body, level=self.compression_level), node_bloom, source_bloom,
             pages, types, states, metadata_sha)).lastrowid
        self.pack_keys.append(pack_key)
        self.records = []
        self.collection_pointer = None
        self._suffixes = {}
        self._targets = {}


class _CanonicalEmitter:
    def __init__(self, store, artifact_key, *, compression_level, transaction_bytes):
        self.store = store
        self.connection = store.connection
        self.artifact_key = artifact_key
        self.compression_level = compression_level
        self.transaction_bytes = transaction_bytes
        self.pending = bytearray()
        self.offset = 0
        self.ordinal = 0
        self.committed_offset = 0
        self.digest = hashlib.sha256()
        self.hot = []
        self.packer = _DirectDescriptorPacker(
            store, artifact_key, compression_level=compression_level)

    def emit(self, body):
        self.pending.extend(body)
        self.digest.update(body)
        while len(self.pending) >= CHUNK_BYTES:
            self.flush_chunk(CHUNK_BYTES)

    def flush_chunk(self, size):
        if self.offset - self.committed_offset + size >= self.transaction_bytes:
            check_capacity([self.store.path], reserve_bytes=self.transaction_bytes)
        raw = bytes(self.pending[:size])
        del self.pending[:size]
        self.connection.execute("INSERT INTO artifact_chunks VALUES(?,?,?,?,?,?)", (
            self.artifact_key, self.ordinal, self.offset, len(raw),
            hashlib.sha256(raw).digest(), zlib.compress(raw, level=self.compression_level)))
        self.offset += len(raw)
        self.ordinal += 1
        if self.offset - self.committed_offset >= self.transaction_bytes:
            self.connection.commit()
            self.committed_offset = self.offset

    def walk(self, value, pointer="", page_ref=None):
        start = self.offset + len(self.pending)
        if isinstance(value, dict):
            source = value.get("source")
            inherited_page = (value.get("page_ref")
                or (source.get("page_ref") if isinstance(source, dict) else None)
                or page_ref)
            self.emit(b"{")
            for index, key in enumerate(sorted(value)):
                if index:
                    self.emit(b",")
                self.emit(_json(str(key)).encode())
                self.emit(b":")
                self.walk(value[key], pointer + "/" + _escape(key), inherited_page)
            self.emit(b"}")
            if pointer and (isinstance(value.get("id"), str) or value.get("record_type")
                            or (value.get("page_ref") and "pages/" in pointer)):
                record_type = (value.get("record_type") or value.get("entity_type")
                               or pointer.split("/")[-2])
                row = {"pointer": pointer, "source_id": value.get("id"),
                    "record_type": record_type, "page_ref": inherited_page,
                    "state": value.get("state") or value.get("epistemic_state"),
                    "payload_start": start,
                    "payload_length": self.offset + len(self.pending) - start}
                references = _references(value)
                if "/source_rows/" in pointer:
                    self.packer.add(row, references)
                else:
                    body = _json(value)
                    self.hot.append((row, references,
                        body if len(body) <= 65536 else None))
        elif isinstance(value, (list, RecordStream)):
            self.emit(b"[")
            for index, child in enumerate(value):
                if index:
                    self.emit(b",")
                self.walk(child, pointer + "/" + str(index), page_ref)
            self.emit(b"]")
        else:
            self.emit(_json(value).encode())

    def finish(self):
        self.packer.flush()
        if self.pending or self.ordinal == 0:
            self.flush_chunk(len(self.pending))
        self.connection.commit()
        return self.digest.digest(), self.offset, self.ordinal


class V3DirectArtifactWriter:
    """Write one immutable producer artifact without a retained JSON file."""

    def __init__(self, store, *, compression_level=1, transaction_bytes=16 * 1024 * 1024):
        if not isinstance(store, PackedProjectStore):
            raise TypeError("schema-v3 packed store is required")
        if compression_level not in range(1, 10):
            raise ValueError("compression level must be 1..9")
        if transaction_bytes < CHUNK_BYTES:
            raise ValueError("transaction bound must cover at least one canonical chunk")
        self.store = store
        self.connection = store.connection
        self.compression_level = compression_level
        self.transaction_bytes = transaction_bytes

    def _placeholder(self):
        while True:
            value = hashlib.sha256(os.urandom(32)).digest()
            if self.connection.execute("SELECT 1 FROM artifacts WHERE sha256=?", (value,)).fetchone() is None:
                return value

    def _discard(self, artifact_key):
        collections = [row[0] for row in self.connection.execute(
            "SELECT collection_key FROM collections WHERE artifact_key=?", (artifact_key,))]
        if collections:
            marks = ",".join("?" for _ in collections)
            self.connection.execute("DELETE FROM descriptor_packs WHERE collection_key IN (" + marks + ")", collections)
            self.connection.execute("DELETE FROM collections WHERE collection_key IN (" + marks + ")", collections)
        nodes = [row[0] for row in self.connection.execute(
            "SELECT node_key FROM hot_nodes WHERE artifact_key=?", (artifact_key,))]
        if nodes:
            marks = ",".join("?" for _ in nodes)
            self.connection.execute("DELETE FROM hot_edges WHERE node_key IN (" + marks + ")", nodes)
            public_ids = [row[0] for row in self.connection.execute(
                "SELECT public_id FROM hot_nodes WHERE artifact_key=?", (artifact_key,))]
            for public_id in public_ids:
                mapped = self.connection.execute(
                    "SELECT rowid FROM fts_node_map WHERE public_id=?", (public_id,)).fetchone()
                if mapped:
                    self.connection.execute("DELETE FROM fts_authored_text WHERE rowid=?", (mapped[0],))
                    self.connection.execute("DELETE FROM fts_node_map WHERE rowid=?", (mapped[0],))
            self.connection.execute("DELETE FROM hot_nodes WHERE artifact_key=?", (artifact_key,))
        self.connection.execute("DELETE FROM artifact_chunks WHERE artifact_key=?", (artifact_key,))
        self.connection.execute("DELETE FROM artifacts WHERE artifact_key=?", (artifact_key,))

    def _insert_hot(self, artifact_key, digest, rows):
        for index, (row, references, payload_json) in enumerate(rows, 1):
            public_id = _public_id(digest.hex(), row["pointer"])
            node_key = self.connection.execute(
                "INSERT INTO hot_nodes(public_id,artifact_key,pointer,source_id,type_key,"
                "page_key,state_key,payload_start,payload_length) VALUES(?,?,?,?,?,?,?,?,?)",
                (public_id, artifact_key, row["pointer"], row["source_id"],
                 self.store.dictionary_key("type", row["record_type"]),
                 self.store.dictionary_key("page", row["page_ref"]),
                 self.store.dictionary_key("state", row["state"]),
                 row["payload_start"], row["payload_length"])).lastrowid
            self.connection.executemany("INSERT INTO hot_edges VALUES(?,?,?)", [
                (node_key, self.store.dictionary_key("field", field), target)
                for field, target in references])
            if row["source_id"] and payload_json:
                body = _fts_text(payload_json)
                if body:
                    mapped = self.connection.execute(
                        "INSERT INTO fts_node_map(public_id) VALUES(?)", (public_id,)).lastrowid
                    self.connection.execute(
                        "INSERT INTO fts_authored_text(rowid,source_id,record_type,page_ref,state,body) "
                        "VALUES(?,?,?,?,?,?)", (mapped, row["source_id"], row["record_type"],
                                                row["page_ref"], row["state"], body))
            if index % 1000 == 0:
                self.connection.commit()
        self.connection.commit()

    def _merge_locators(self, artifact_key, digest, pack_keys):
        with tempfile.TemporaryDirectory(prefix="project-v3-direct-locators-") as directory:
            spools = {
                "node": _LocatorSpool(directory, "node", _NODE_ENTRY),
                "source": _LocatorSpool(directory, "source", _SOURCE_ENTRY),
                "reverse": _LocatorSpool(directory, "reverse", _REVERSE_ENTRY),
            }
            for pack_key in pack_keys:
                pack, body = self.store._decode_descriptor_pack(pack_key)
                for ordinal, record in enumerate(body["r"]):
                    local_ordinal, suffix_index, source_id = record[:3]
                    pointer = pack["pointer"] + "/" + str(local_ordinal) + body["s"][suffix_index]
                    public_id = _public_id(digest.hex(), pointer)
                    spools["node"].append(public_id[0], public_id, pack_key, ordinal)
                    if source_id:
                        source_hash = _hash16(source_id)
                        spools["source"].append(source_hash[0], source_hash, pack_key, ordinal)
                    for field_key, target_index in record[8]:
                        if field_key > 65535:
                            raise ValueError("reference field dictionary exceeds packed key width")
                        target_hash = _hash16(body["t"][target_index])
                        spools["reverse"].append(target_hash[0], target_hash,
                                                 pack_key, ordinal, field_key)
            spools["node"].merge(self.connection, "node_locators",
                                  compression_level=self.compression_level)
            spools["source"].merge(self.connection, "source_locators",
                                    compression_level=self.compression_level)
            spools["reverse"].merge(self.connection, "packed_reverse_edges",
                                     compression_level=self.compression_level)
        self.connection.commit()
        self.store._locator_cache.values.clear()

    def write(self, name, payload):
        database = Path(self.store.path)
        paths = [database, *(Path(str(database) + suffix) for suffix in ('-wal', '-shm', '-journal'))]
        with artifact_disk_usage('v3:' + name, paths, reserve_bytes=self.transaction_bytes):
            return self._write(name, payload)

    def _write(self, name, payload):
        if name not in DIRECT_ARTIFACTS:
            raise ValueError("artifact is outside the bounded direct-write rollout")
        if not isinstance(payload, dict) or not payload.get("schema_version") or not payload.get("layer"):
            raise ValueError("direct artifact must declare schema_version and layer")
        placeholder = self._placeholder()
        artifact_key = self.connection.execute(
            "INSERT INTO artifacts(sha256,byte_count,chunk_count) VALUES(?,0,0)",
            (placeholder,)).lastrowid
        self.connection.commit()
        emitter = _CanonicalEmitter(self.store, artifact_key,
            compression_level=self.compression_level,
            transaction_bytes=self.transaction_bytes)
        try:
            emitter.walk(payload)
            digest, byte_count, chunk_count = emitter.finish()
            existing = self.connection.execute(
                "SELECT artifact_key,byte_count,chunk_count FROM artifacts WHERE sha256=?",
                (digest,)).fetchone()
            if existing is not None:
                with self.connection:
                    self._discard(artifact_key)
                return {"name": name, "sha256": digest.hex(), "artifact_key": existing[0],
                        "byte_count": existing[1], "chunk_count": existing[2], "reused": True}
            self.connection.execute(
                "UPDATE artifacts SET sha256=?,byte_count=?,chunk_count=? WHERE artifact_key=?",
                (digest, byte_count, chunk_count, artifact_key))
            self.connection.commit()
            artifact_local = name == "mep-automatic-page-cache"
            if artifact_local:
                # Page-cache consumers iterate physical descriptor packs by
                # artifact. Global source/reverse locators and FTS would add
                # work and authority the cache does not need.
                if emitter.hot:
                    self._insert_hot(artifact_key, digest, emitter.hot)
            else:
                self._insert_hot(artifact_key, digest, emitter.hot)
                self._merge_locators(artifact_key, digest, emitter.packer.pack_keys)
            self.store.put_manifest("direct-write:" + digest.hex(), {
                "status": "complete", "name": name, "byte_count": byte_count,
                "chunk_count": chunk_count, "transaction_bytes": self.transaction_bytes,
                "index_scope": ("artifact_local_descriptor_packs" if artifact_local
                                else "new_content_addressed_artifact_only"),
                "fts_scope": ("none" if artifact_local else "new_hot_records_only")})
            self.connection.commit()
            return {"name": name, "sha256": digest.hex(), "artifact_key": artifact_key,
                    "byte_count": byte_count, "chunk_count": chunk_count, "reused": False,
                    "hot_node_count": len(emitter.hot),
                    "descriptor_pack_count": len(emitter.packer.pack_keys)}
        except Exception:
            self.connection.rollback()
            with self.connection:
                self._discard(artifact_key)
            self.store._dictionary_cache.clear()
            raise

    def iter_bytes(self, artifact_key):
        row = self.connection.execute(
            "SELECT sha256,byte_count FROM artifacts WHERE artifact_key=?", (artifact_key,)).fetchone()
        if row is None:
            raise KeyError(artifact_key)
        digest = hashlib.sha256()
        offset = 0
        for chunk in self.connection.execute(
                "SELECT * FROM artifact_chunks WHERE artifact_key=? ORDER BY ordinal", (artifact_key,)):
            raw = _decode_verified(chunk["payload_zlib"], chunk["payload_sha256"],
                                   "direct artifact chunk")
            if chunk["uncompressed_offset"] != offset or len(raw) != chunk["uncompressed_size"]:
                raise ValueError("direct artifact chunk coverage changed")
            digest.update(raw)
            offset += len(raw)
            yield raw
        if offset != row["byte_count"] or digest.digest() != row["sha256"]:
            raise ValueError("direct artifact digest changed")


def shadow_write(database, name, payload):
    """Direct-write one artifact and prove exact parity with canonical JSON."""
    database = Path(database)
    with PackedProjectStore(database, create=not database.exists()) as store:
        if store.connection.execute("SELECT count(*) FROM active_documents").fetchone()[0]:
            raise ValueError("shadow writes require a detached schema-v3 database")
        writer = V3DirectArtifactWriter(store)
        result = writer.write(name, payload)
        exported = b"".join(writer.iter_bytes(result["artifact_key"]))
    expected = _json(payload).encode()
    if exported != expected:
        raise ValueError("direct v3 export differs from canonical producer artifact")
    return {**result, "canonical_parity": True,
            "canonical_sha256": hashlib.sha256(expected).hexdigest()}
