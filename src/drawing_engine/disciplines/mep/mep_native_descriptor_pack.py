"""Fixed-width cold-path native primitive descriptors.

The pack is a sequential intermediate, not an engineering graph.  Every native
primitive has one fixed-width record; only curves and optional PDF metadata use
the side payload stream.  Source and candidate identifiers are deterministically
reconstructed from the canonical drawing / item / part ordinals.
"""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
import struct


PACK_VERSION = "1.0.0"
_KIND_TO_CODE = {"line": 1, "cubic": 2}
_CODE_TO_KIND = {value: key for key, value in _KIND_TO_CODE.items()}
_AXIS_TO_CODE = {"horizontal": 1, "vertical": 2, "oblique": 3}
_CODE_TO_AXIS = {value: key for key, value in _AXIS_TO_CODE.items()}

# drawing, item, part, style; kind, axis, close, payload-present; payload
# offset/length; native endpoints, display endpoints, native bbox, display
# geometry bbox, conservative display search bbox, path length.
RECORD = struct.Struct("<IIIIBBBB10sQI21d")


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def _stable_id(kind, *parts):
    encoded = _canonical(parts).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _ordinals(source_ref):
    try:
        drawing, rest = source_ref.removeprefix("drawing[").split("].item[", 1)
        item, part = rest.split("].segment[", 1)
        return int(drawing), int(item), int(part.removesuffix("]"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"non-canonical native source ref: {source_ref}") from error


class NativeDescriptorPackWriter:
    """Append exact fixed-width descriptors and a sparse variable payload."""

    def __init__(self, path, *, page_ref, pdf_to_display_matrix,
                 minimum_member_length, cell_size=64):
        self.path = Path(path)
        self.payload_path = self.path.with_suffix(self.path.suffix + ".payload")
        if any(candidate.exists() for candidate in (self.path, self.payload_path)):
            raise ValueError("native descriptor pack paths must be new")
        self.page_ref = page_ref
        self.pdf_to_display_matrix = list(pdf_to_display_matrix)
        self.minimum_member_length = float(minimum_member_length)
        self.cell_size = float(cell_size)
        self._pack = self.path.open("xb")
        self._payload = self.payload_path.open("xb")
        self._styles = []
        self._style_ids = {}
        self._pack_hash = hashlib.sha256()
        self._payload_hash = hashlib.sha256()
        self.count = 0
        self.long_members = []
        # Geometry-first regions retain the first configured number of
        # intersecting primitives in canonical scan order.  One ordinal
        # threshold per initial region is enough to reconstruct those legacy
        # memberships later; retaining a source-id list per primitive would
        # recreate the giant in-memory structure this pack replaces.
        self.region_retention_max_ordinal = {}
        self.closed = False

    def append(self, row):
        native = row["source_native_segment"]
        drawing = int(native["drawing_ref"][8:-1])
        item, part = native["item_index"], native["part_index"]
        style = native.get("style") or {}
        style_key = (style.get("width"), tuple(style.get("stroke") or ()),
                     tuple(style.get("fill") or ()), style.get("dash"))
        style_id = self._style_ids.get(style_key)
        if style_id is None:
            style_id = len(self._styles)
            self._style_ids[style_key] = style_id
            self._styles.append(style)
        payload = {}
        for key in ("control_points_display", "sample_points_display", "native_metadata"):
            if native.get(key):
                payload[key] = native[key]
        if native["kind"] != "line":
            payload["points_display"] = row["points_display"]
        payload_body = (_canonical(payload).encode("utf-8") if payload else b"")
        payload_offset = self._payload.tell()
        if payload_body:
            self._payload.write(payload_body)
            self._payload_hash.update(payload_body)
        values = (
            drawing, item, part, style_id,
            _KIND_TO_CODE[native["kind"]], _AXIS_TO_CODE[native["axis"]],
            int(row["source_drawing_close_path"]), int(bool(payload_body)),
            bytes.fromhex((row.get("id") or _stable_id(
                "mep_native_target_primitive", self.page_ref,
                row["source_primitive_ref"])).rsplit('.', 1)[1]),
            payload_offset, len(payload_body),
            *native["start_display"], *native["end_display"],
            *row["points_display"][0], *row["points_display"][-1],
            *native["bbox_display"], *row["bbox_display"],
            *row.get("search_bbox_display", row["bbox_display"]),
            native["length_points"],
        )
        body = RECORD.pack(*values)
        self._pack.write(body)
        self._pack_hash.update(body)
        for region_ref in row.get("search_refs", ()):
            self.region_retention_max_ordinal[region_ref] = self.count
        self.count += 1
        if (native["kind"] == "line" and native.get("style", {}).get("stroke") is not None
                and native["length_points"] >= self.minimum_member_length):
            self.long_members.append({
                "descriptor_ordinal": self.count - 1,
                "source_primitive_ref": row["source_primitive_ref"],
                "points_display": [row["points_display"][0], row["points_display"][-1]],
                "search_bbox_display": list(row.get("search_bbox_display", row["bbox_display"])),
                "length_points": native["length_points"], "style_id": style_id,
            })

    def finish(self):
        if self.closed:
            raise ValueError("descriptor pack writer is closed")
        self._pack.flush()
        self._payload.flush()
        self._pack.close()
        self._payload.close()
        self.closed = True
        return {
            "schema_version": "0.1.0", "layer": "mep_native_descriptor_pack",
            "format_version": PACK_VERSION, "record_size": RECORD.size,
            "record_count": self.count, "page_ref": self.page_ref,
            "pdf_to_display_matrix": self.pdf_to_display_matrix,
            "minimum_member_length": self.minimum_member_length,
            "styles": self._styles,
            "descriptor_sha256": self._pack_hash.hexdigest(),
            "payload_sha256": self._payload_hash.hexdigest(),
            "payload_bytes": self.payload_path.stat().st_size,
        }

    def abort(self):
        if not self.closed:
            self._pack.close()
            self._payload.close()
            self.closed = True


class NativeDescriptorPack:
    """Sequential reader with exact compact-row reconstruction."""

    def __init__(self, path, manifest):
        self.path = Path(path)
        self.payload_path = self.path.with_suffix(self.path.suffix + ".payload")
        self.manifest = manifest
        if (manifest.get("format_version") != PACK_VERSION
                or manifest.get("record_size") != RECORD.size
                or self.path.stat().st_size != manifest["record_count"] * RECORD.size):
            raise ValueError("native descriptor pack manifest mismatch")

    def __len__(self):
        return self.manifest["record_count"]

    def descriptors(self):
        with self.path.open("rb") as stream:
            ordinal = 0
            while True:
                body = stream.read(RECORD.size)
                if not body:
                    return
                if len(body) != RECORD.size:
                    raise ValueError("truncated native descriptor pack")
                descriptor = self._decode_descriptor(RECORD.unpack(body))
                descriptor["descriptor_ordinal"] = ordinal
                ordinal += 1
                yield descriptor

    def descriptor_at(self, ordinal, stream=None):
        if not 0 <= ordinal < len(self):
            raise IndexError(ordinal)
        owned = stream is None
        stream = stream or self.path.open('rb')
        try:
            offset = ordinal * RECORD.size
            if isinstance(stream, mmap.mmap):
                descriptor = self._decode_descriptor(RECORD.unpack_from(stream, offset))
                descriptor["descriptor_ordinal"] = ordinal
                return descriptor
            stream.seek(offset)
            body = stream.read(RECORD.size)
            if len(body) != RECORD.size:
                raise ValueError("truncated native descriptor pack")
            descriptor = self._decode_descriptor(RECORD.unpack(body))
            descriptor["descriptor_ordinal"] = ordinal
            return descriptor
        finally:
            if owned:
                stream.close()

    def raw_at(self, ordinal, mapping):
        if not 0 <= ordinal < len(self):
            raise IndexError(ordinal)
        return RECORD.unpack_from(mapping, ordinal * RECORD.size)

    def raw_records(self):
        """Scan exact records with bounded buffers, without a resident full mmap."""
        count = 0
        with self.path.open('rb') as stream:
            while body := stream.read(4096 * RECORD.size):
                if len(body) % RECORD.size:
                    raise ValueError('truncated native descriptor pack')
                for raw in RECORD.iter_unpack(body):
                    if count >= len(self):
                        raise ValueError('native descriptor pack record count mismatch')
                    count += 1
                    yield raw
        if count != len(self):
            raise ValueError('native descriptor pack record count mismatch')

    def open_mmap(self):
        stream = self.path.open('rb')
        try:
            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            stream.close()
            raise
        return stream, mapping

    def _decode_descriptor(self, values):
        (drawing, item, part, style_id, kind_code, axis_code, close_path,
         payload_present, candidate_digest, payload_offset, payload_length, *numbers) = values
        native_start = numbers[0:2]
        native_end = numbers[2:4]
        display_start = numbers[4:6]
        display_end = numbers[6:8]
        native_box = numbers[8:12]
        geometry_box = numbers[12:16]
        search_box = numbers[16:20]
        length = numbers[20]
        source_ref = f"drawing[{drawing}].item[{item}].segment[{part}]"
        return {
            "drawing_ordinal": drawing, "item_ordinal": item, "part_ordinal": part,
            "source_primitive_ref": source_ref,
            "candidate_id": "mep_native_target_primitive." + candidate_digest.hex(),
            "style_id": style_id, "kind": _CODE_TO_KIND[kind_code],
            "axis": _CODE_TO_AXIS[axis_code], "close_path": bool(close_path),
            "payload_offset": payload_offset if payload_present else None,
            "payload_length": payload_length,
            "native_start": list(native_start), "native_end": list(native_end),
            "display_start": list(display_start), "display_end": list(display_end),
            "native_bbox": list(native_box), "bbox_display": list(geometry_box),
            "search_bbox_display": list(search_box), "length_points": length,
        }

    def row(self, descriptor, payload_stream=None):
        owned = payload_stream is None
        payload_stream = payload_stream or self.payload_path.open("rb")
        try:
            payload = {}
            if descriptor["payload_length"]:
                payload_stream.seek(descriptor["payload_offset"])
                payload = json.loads(payload_stream.read(descriptor["payload_length"]))
            source_ref = descriptor["source_primitive_ref"]
            drawing_ref = f"drawing[{descriptor['drawing_ordinal']}]"
            primitive_ref = f"{drawing_ref}.item[{descriptor['item_ordinal']}]"
            native = {
                "id": source_ref, "drawing_ref": drawing_ref,
                "primitive_ref": primitive_ref,
                "item_index": descriptor["item_ordinal"],
                "part_index": descriptor["part_ordinal"],
                "kind": descriptor["kind"],
                "start_display": descriptor["native_start"],
                "end_display": descriptor["native_end"],
                "control_points_display": payload.get("control_points_display", []),
                "sample_points_display": payload.get("sample_points_display", []),
                "axis": descriptor["axis"], "length_points": descriptor["length_points"],
                "bbox_display": descriptor["native_bbox"],
                "style": self.manifest["styles"][descriptor["style_id"]],
            }
            if "native_metadata" in payload:
                native["native_metadata"] = payload["native_metadata"]
            points = payload.get("points_display",
                                 [descriptor["display_start"], descriptor["display_end"]])
            return {
                "id": descriptor["candidate_id"], "page_ref": self.manifest["page_ref"],
                "source_primitive_ref": source_ref, "source_native_segment": native,
                "source_drawing_close_path": descriptor["close_path"],
                "points_display": points, "bbox_display": descriptor["bbox_display"],
                "search_bbox_display": descriptor["search_bbox_display"],
                "search_bbox_is_geometry": False,
                "pdf_to_display_matrix": self.manifest["pdf_to_display_matrix"],
                "search_refs": [], "state": "observed",
                "role": "unclassified_native_geometry_candidate",
                "geometry_is_not_a_route_or_equipment": True,
                "quantity_eligible": False,
            }
        finally:
            if owned:
                payload_stream.close()

    def rows(self, predicate=None):
        with self.payload_path.open("rb") as payload:
            for descriptor in self.descriptors():
                if predicate is None or predicate(descriptor):
                    yield self.row(descriptor, payload)

    def verify_hashes(self):
        def digest(path):
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            return hasher.hexdigest()
        return (digest(self.path) == self.manifest["descriptor_sha256"]
                and digest(self.payload_path) == self.manifest["payload_sha256"])
