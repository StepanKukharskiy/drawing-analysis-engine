"""Compact authored-PDF-path records over the native MEP denominator.

The native descriptor pack is segment-complete.  This companion groups
consecutive descriptors by their PyMuPDF drawing ordinal (one authored PDF
paint path) before any semantic classifier runs.  Segment provenance is kept
losslessly through the first descriptor ordinal and segment count.

Path records establish source context only.  They never certify a route,
system, connection, length quantity, or physical identity.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import struct
from typing import Any, Iterator, Mapping

from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack
from src.drawing_engine.disciplines.mep.mep_source_primitive_denominator import (
    CODE_TO_DISPOSITION, CODE_TO_REGION, DISPOSITION_TO_CODE,
    DISPOSITION_TO_BIT, REGION_TO_CODE, SourceDispositionPack,
)


SCHEMA_VERSION = "0.1.0"
FORMAT_VERSION = "1.0.0"
MIXED_STYLE_ID = 0xFFFFFFFF

# drawing, first segment ordinal, segment/item counts, homogeneous style,
# primary/candidate masks, homogeneous region, axis/kind masks, flags,
# bbox, path endpoints, and summed source-segment length.
RECORD = struct.Struct("<IQIIIHHBBBB9d")

AXIS_BITS = {"horizontal": 1, "vertical": 2, "oblique": 4}
KIND_BITS = {"line": 1, "cubic": 2}
FLAG_ANY_CLOSE_PATH = 1
FLAG_ENDPOINT_CLOSED = 2
FLAG_MIXED_STYLE = 4
FLAG_MIXED_REGION = 8


def _mask_names(mask: int) -> list[str]:
    return [name for name in DISPOSITION_TO_CODE
            if mask & DISPOSITION_TO_BIT[name]]


class NativePathPack:
    """Sequential reader for authored path summaries."""

    def __init__(self, path: Path | str, manifest: Mapping[str, Any]):
        self.path = Path(path)
        self.manifest = manifest
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported native path pack format")
        if manifest.get("record_size") != RECORD.size:
            raise ValueError("native path record size mismatch")
        if self.path.stat().st_size != manifest.get("record_count", -1) * RECORD.size:
            raise ValueError("native path pack length mismatch")

    def __len__(self) -> int:
        return int(self.manifest["record_count"])

    def records(self) -> Iterator[dict[str, Any]]:
        with self.path.open("rb") as stream:
            ordinal = 0
            while body := stream.read(RECORD.size):
                if len(body) != RECORD.size:
                    raise ValueError("truncated native path pack")
                values = RECORD.unpack(body)
                (drawing, first, count, item_count, style_id,
                 primary_mask, candidate_mask, region_code, axis_mask,
                 kind_mask, flags, *numbers) = values
                yield {
                    "path_ordinal": ordinal,
                    "drawing_ordinal": drawing,
                    "first_descriptor_ordinal": first,
                    "source_segment_count": count,
                    "native_item_path_count": item_count,
                    "style_id": None if style_id == MIXED_STYLE_ID else style_id,
                    "mixed_style": bool(flags & FLAG_MIXED_STYLE),
                    "primary_dispositions": _mask_names(primary_mask),
                    "candidate_roles": _mask_names(candidate_mask),
                    "region_role": (None if region_code == 0
                                    else CODE_TO_REGION[region_code]),
                    "mixed_region": bool(flags & FLAG_MIXED_REGION),
                    "axis_mask": axis_mask,
                    "kind_mask": kind_mask,
                    "any_close_path": bool(flags & FLAG_ANY_CLOSE_PATH),
                    "endpoint_closed": bool(flags & FLAG_ENDPOINT_CLOSED),
                    "bbox_display": list(numbers[0:4]),
                    "start_display": list(numbers[4:6]),
                    "end_display": list(numbers[6:8]),
                    "source_segment_length_points": numbers[8],
                }
                ordinal += 1

    def verify_hash(self) -> bool:
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == self.manifest.get("sha256")


def write_native_path_pack(
    *, descriptor_pack: NativeDescriptorPack,
    disposition_pack: SourceDispositionPack,
    output_path: Path | str,
) -> dict[str, Any]:
    """Group every descriptor into exactly one authored drawing path."""
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("native path output path must be new")

    path_count = segment_count = 0
    segment_count_distribution = Counter()
    primary_path_counts = Counter()
    region_path_counts = Counter()
    digest = hashlib.sha256()
    current: dict[str, Any] | None = None

    def finish(stream) -> None:
        nonlocal current, path_count
        if current is None:
            return
        flags = 0
        if current["any_close"]:
            flags |= FLAG_ANY_CLOSE_PATH
        if (abs(current["start"][0] - current["end"][0]) <= 1e-6
                and abs(current["start"][1] - current["end"][1]) <= 1e-6):
            flags |= FLAG_ENDPOINT_CLOSED
        style_id = current["style_id"]
        if current["mixed_style"]:
            flags |= FLAG_MIXED_STYLE
            style_id = MIXED_STYLE_ID
        region_code = current["region_code"]
        if current["mixed_region"]:
            flags |= FLAG_MIXED_REGION
            region_code = 0
        body = RECORD.pack(
            current["drawing"], current["first"], current["count"],
            len(current["items"]), style_id, current["primary_mask"],
            current["candidate_mask"], region_code, current["axis_mask"],
            current["kind_mask"], flags, *current["bbox"], *current["start"],
            *current["end"], current["length"])
        stream.write(body)
        digest.update(body)
        path_count += 1
        segment_count_distribution[current["count"]] += 1
        for name in _mask_names(current["primary_mask"]):
            primary_path_counts[name] += 1
        region_path_counts[CODE_TO_REGION[region_code] if region_code else "mixed"] += 1
        current = None

    descriptors = descriptor_pack.descriptors()
    dispositions = disposition_pack.records()
    with output_path.open("xb") as stream:
        for descriptor, disposition in zip(descriptors, dispositions):
            ordinal = int(descriptor["descriptor_ordinal"])
            drawing = int(descriptor["drawing_ordinal"])
            primary_bit = DISPOSITION_TO_BIT[disposition["primary_disposition"]]
            region_code = REGION_TO_CODE[disposition["region_role"]]
            if current is None or drawing != current["drawing"]:
                finish(stream)
                current = {
                    "drawing": drawing, "first": ordinal, "count": 0,
                    "items": set(), "style_id": descriptor["style_id"],
                    "mixed_style": False, "primary_mask": 0,
                    "candidate_mask": 0, "region_code": region_code,
                    "mixed_region": False, "axis_mask": 0, "kind_mask": 0,
                    "any_close": False,
                    "bbox": list(descriptor["bbox_display"]),
                    "start": list(descriptor["display_start"]),
                    "end": list(descriptor["display_end"]), "length": 0.0,
                }
            elif ordinal != current["first"] + current["count"]:
                raise ValueError("one authored path is not descriptor-contiguous")
            current["count"] += 1
            current["items"].add(descriptor["item_ordinal"])
            current["mixed_style"] |= descriptor["style_id"] != current["style_id"]
            current["primary_mask"] |= primary_bit
            current["candidate_mask"] |= int(disposition["candidate_role_mask"])
            current["mixed_region"] |= region_code != current["region_code"]
            current["axis_mask"] |= AXIS_BITS[descriptor["axis"]]
            current["kind_mask"] |= KIND_BITS[descriptor["kind"]]
            current["any_close"] |= bool(descriptor["close_path"])
            box = descriptor["bbox_display"]
            current["bbox"] = [min(current["bbox"][0], box[0]),
                               min(current["bbox"][1], box[1]),
                               max(current["bbox"][2], box[2]),
                               max(current["bbox"][3], box[3])]
            current["end"] = list(descriptor["display_end"])
            current["length"] += float(descriptor["length_points"])
            segment_count += 1
        finish(stream)

    if segment_count != len(descriptor_pack):
        raise ValueError("path pack did not consume every native descriptor")
    if segment_count != disposition_pack.manifest.get("record_count"):
        raise ValueError("path pack did not consume every source disposition")
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "mep_native_authored_path_pack",
        "format_version": FORMAT_VERSION,
        "record_size": RECORD.size,
        "record_count": path_count,
        "source_segment_count": segment_count,
        "sha256": digest.hexdigest(),
        "single_segment_path_count": segment_count_distribution.get(1, 0),
        "multi_segment_path_count": path_count - segment_count_distribution.get(1, 0),
        "maximum_segments_per_path": max(segment_count_distribution, default=0),
        "primary_disposition_path_counts": dict(sorted(primary_path_counts.items())),
        "region_role_path_counts": dict(sorted(region_path_counts.items())),
        "authority": {
            "authored_pdf_path_context_established": True,
            "segment_provenance_preserved": True,
            "route_identity_established": False,
            "system_identity_established": False,
            "quantity_eligible": False,
        },
    }


def validate_native_path_pack(
    *, descriptor_pack: NativeDescriptorPack,
    disposition_pack: SourceDispositionPack,
    path_pack: NativePathPack,
) -> list[str]:
    errors = []
    manifest = path_pack.manifest
    if manifest.get("source_segment_count") != len(descriptor_pack):
        errors.append("path source-segment count differs from denominator")
    if manifest.get("source_segment_count") != disposition_pack.manifest.get("record_count"):
        errors.append("path source-segment count differs from disposition count")
    rows = list(path_pack.records())
    if len(rows) != manifest.get("record_count"):
        errors.append("path record count mismatch")
    if sum(row["source_segment_count"] for row in rows) != len(descriptor_pack):
        errors.append("authored paths do not partition all source segments")
    if rows and any(
        right["first_descriptor_ordinal"]
        != left["first_descriptor_ordinal"] + left["source_segment_count"]
        for left, right in zip(rows, rows[1:])
    ):
        errors.append("authored path descriptor intervals are not contiguous")
    if rows and rows[0]["first_descriptor_ordinal"] != 0:
        errors.append("authored path pack does not begin at descriptor zero")
    if not path_pack.verify_hash():
        errors.append("authored path pack hash verification failed")
    return errors
