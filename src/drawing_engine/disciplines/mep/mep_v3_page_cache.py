"""Content-addressed, fail-closed MEP page cache in schema-v3 SQLite."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
from src.drawing_engine.project.project_packed_store import PackedProjectStore
from src.drawing_engine.project.project_v3_direct_writer import RecordStream, V3DirectArtifactWriter


CACHE_SCHEMA_VERSION = "0.1.0"
CACHE_LAYER = "mep_automatic_page_cache"
_MANIFEST_PREFIX = "mep-page-cache:"
_RECORD_TYPE_PREFIX = "mep_automatic_page_cache_"


def page_cache_key(*, source_pdf_sha256, page_identity, extraction_implementation_sha256,
                   configuration_sha256, ruleset_sha256):
    inputs = {
        "source_pdf_sha256": source_pdf_sha256,
        "page_identity": page_identity,
        "extraction_implementation_sha256": extraction_implementation_sha256,
        "configuration_sha256": configuration_sha256,
        "ruleset_sha256": ruleset_sha256,
    }
    if (not isinstance(page_identity, dict) or not page_identity.get("page_ref")
            or type(page_identity.get("page_number")) is not int):
        raise ValueError("page cache identity requires page_ref and page_number")
    for field in ("source_pdf_sha256", "extraction_implementation_sha256",
                  "configuration_sha256", "ruleset_sha256"):
        value = inputs[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(field + " must be a sha256 hex digest")
        bytes.fromhex(value)
    return _sha256(inputs), inputs


def _drain(rows):
    rows.reverse()
    while rows:
        yield rows.pop()


def legacy_page_record_stream(local):
    """Drain one legacy page result into independently iterable cache rows."""
    route_graph = local.pop("route_graph")
    composites = local.pop("composites")
    accepted = composites.pop("accepted_composites")
    candidates = composites.pop("candidates")
    if accepted != [row for row in candidates if row["state"] == "accepted"]:
        raise ValueError("accepted composite projection differs from candidates")
    yield "route_graph", route_graph
    yield "composites_header", composites
    for row in _drain(candidates):
        yield "composite_candidate", row
    for row in _drain(local.pop("envelope_searches")):
        yield "envelope_search", row
    for row in _drain(local.pop("leader_observations")):
        yield "leader_observation", row
    for row in _drain(local.pop("boundary_connections")):
        yield "boundary_connection", row
    if local:
        raise ValueError("unrecognized legacy page target fields")


def page_component_record_stream(*, page_record, regions, route_graph, composites,
                                 envelope_searches, leader_observations,
                                 boundary_connections, performance_profile=None):
    """Drain a computed page into compact rows without constructing page JSON."""
    if "regions" in page_record:
        raise ValueError("page cache metadata must not embed native regions")
    # Runtime timings and memory observations describe one execution, not the
    # page evidence.  Keep them in benchmark reports so identical page inputs
    # always produce an identical content-addressed cache artifact.
    accepted = composites.pop("accepted_composites")
    candidates = composites.pop("candidates")
    if accepted != [row for row in candidates if row["state"] == "accepted"]:
        raise ValueError("accepted composite projection differs from candidates")
    yield "page_record", page_record
    for row in _drain(regions):
        yield "native_region", row
    yield "route_graph", route_graph
    yield "composites_header", composites
    for row in _drain(candidates):
        yield "composite_candidate", row
    for row in _drain(envelope_searches):
        yield "envelope_search", row
    for row in _drain(leader_observations):
        yield "leader_observation", row
    for row in _drain(boundary_connections):
        yield "boundary_connection", row


class MepV3PageCache:
    def __init__(self, store, *, compression_level=1, transaction_bytes=16 * 1024 * 1024):
        if not isinstance(store, PackedProjectStore):
            raise TypeError("schema-v3 packed store is required")
        self.store = store
        self.writer = V3DirectArtifactWriter(
            store, compression_level=compression_level, transaction_bytes=transaction_bytes)

    @staticmethod
    def _manifest_key(cache_key):
        return _MANIFEST_PREFIX + cache_key

    def manifest(self, cache_key):
        row = self.store.manifest(self._manifest_key(cache_key))
        if row is None:
            return None
        if (row.get("status") != "complete" or row.get("cache_key") != cache_key
                or row.get("cache_schema_version") != CACHE_SCHEMA_VERSION):
            raise ValueError("page cache manifest is incomplete or mismatched")
        artifact = self.store.connection.execute(
            "SELECT sha256,byte_count,chunk_count FROM artifacts WHERE artifact_key=?",
            (row["artifact_key"],)).fetchone()
        if (artifact is None or artifact["sha256"].hex() != row["artifact_sha256"]
                or artifact["byte_count"] != row["byte_count"]
                or artifact["chunk_count"] != row["chunk_count"]):
            raise ValueError("published page cache artifact changed")
        return row

    def write(self, *, cache_key, cache_inputs, page_ref, records):
        prior = self.manifest(cache_key)
        if prior is not None:
            return {**prior, "reused": True}

        def wrapped():
            for ordinal, (kind, payload) in enumerate(records):
                if not isinstance(kind, str) or not kind or not isinstance(payload, dict):
                    raise ValueError("page cache rows require a kind and object payload")
                yield {
                    "record_type": _RECORD_TYPE_PREFIX + kind,
                    "id": "mep_page_cache_record." + hashlib.sha256(
                        f"{cache_key}:{ordinal}:{kind}".encode()).hexdigest()[:20],
                    "page_ref": page_ref,
                    "state": "direct",
                    "cache_collection": kind,
                    "collection_ordinal": ordinal,
                    "payload": payload,
                    "quantity_eligible": False,
                }

        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "layer": CACHE_LAYER,
            "cache_key": cache_key,
            "cache_inputs": cache_inputs,
            "page_ref": page_ref,
            "source_rows": RecordStream(wrapped()),
        }
        result = self.writer.write("mep-automatic-page-cache", payload)
        manifest = {
            "status": "complete",
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "cache_inputs": cache_inputs,
            "page_ref": page_ref,
            "artifact_key": result["artifact_key"],
            "artifact_sha256": result["sha256"],
            "byte_count": result["byte_count"],
            "chunk_count": result["chunk_count"],
            "record_storage": "artifact_local_descriptor_packs",
            "published_only_after_complete_artifact": True,
        }
        self.store.put_manifest(self._manifest_key(cache_key), manifest)
        self.store.connection.commit()
        return {**manifest, "reused": False}

    def iter_records(self, cache_key, *, kinds=None):
        manifest = self.manifest(cache_key)
        if manifest is None:
            raise KeyError(cache_key)
        accepted = None if kinds is None else set(kinds)
        artifact_key = manifest["artifact_key"]
        collections = self.store.connection.execute(
            "SELECT collection_key FROM collections WHERE artifact_key=? AND pointer='/source_rows'",
            (artifact_key,)).fetchall()
        if len(collections) != 1:
            raise ValueError("page cache source-row collection is missing or ambiguous")
        for pack_key, in self.store.connection.execute(
                "SELECT pack_key FROM descriptor_packs WHERE collection_key=? ORDER BY pack_key",
                (collections[0][0],)):
            pack, body = self.store._decode_descriptor_pack(pack_key)
            rows = []
            for record in body["r"]:
                if body["s"][record[1]] != "":
                    continue
                record_type = self.store.dictionary_value("type", record[3])
                if not record_type.startswith(_RECORD_TYPE_PREFIX):
                    raise ValueError("page cache descriptor type changed")
                kind = record_type[len(_RECORD_TYPE_PREFIX):]
                if accepted is not None and kind not in accepted:
                    continue
                raw = self.store._chunk_bytes(artifact_key, record[6], record[7])
                wrapper = json.loads(raw)
                if (wrapper.get("record_type") != record_type
                        or wrapper.get("page_ref") != manifest["page_ref"]):
                    raise ValueError("page cache record provenance changed")
                rows.append(wrapper)
            for wrapper in sorted(rows, key=lambda row: row["collection_ordinal"]):
                yield wrapper["cache_collection"], wrapper["payload"]

    def export_legacy_page(self, cache_key):
        route_graph = None
        composites_header = None
        candidates, searches, leaders, connections = [], [], [], []
        for kind, payload in self.iter_records(cache_key, kinds={
                "route_graph", "composites_header", "composite_candidate",
                "envelope_search", "leader_observation", "boundary_connection"}):
            if kind == "route_graph":
                if route_graph is not None:
                    raise ValueError("duplicate cached route graph")
                route_graph = payload
            elif kind == "composites_header":
                if composites_header is not None:
                    raise ValueError("duplicate cached composite header")
                composites_header = payload
            elif kind == "composite_candidate":
                candidates.append(payload)
            elif kind == "envelope_search":
                searches.append(payload)
            elif kind == "leader_observation":
                leaders.append(payload)
            elif kind == "boundary_connection":
                connections.append(payload)
            elif kind in {"page_record", "performance_profile", "native_region"}:
                continue
            else:
                raise ValueError("unknown page cache record kind")
        if route_graph is None or composites_header is None:
            raise ValueError("page cache omits required page graph components")
        composites = deepcopy(composites_header)
        composites["candidates"] = candidates
        composites["accepted_composites"] = [deepcopy(row) for row in candidates
                                               if row["state"] == "accepted"]
        return {
            "route_graph": route_graph,
            "composites": composites,
            "envelope_searches": searches,
            "leader_observations": leaders,
            "boundary_connections": connections,
        }

    def page_evidence(self, cache_key):
        """Return compact metadata and exact regions for explicit diagnostics."""
        record = profile = None
        regions = []
        for kind, payload in self.iter_records(
                cache_key, kinds={"page_record", "performance_profile", "native_region"}):
            if kind == "page_record":
                if record is not None:
                    raise ValueError("duplicate cached page metadata")
                record = payload
            elif kind == "performance_profile":
                if profile is not None:
                    raise ValueError("duplicate cached performance profile")
                profile = payload
            else:
                regions.append(payload)
        if record is None:
            raise ValueError("page cache omits page metadata")
        return record, regions, profile
