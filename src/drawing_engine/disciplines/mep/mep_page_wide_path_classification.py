"""Page-wide authored-path classification for the MEP source denominator.

Every authored PDF path receives exactly one provisional role.  The classifier
uses region ownership, prior independently accepted role evidence, measured
path geometry, text association, and anchored route-style certificates.  It
does not use legacy M3 membership to grant route status.

The output is a compact fixed-width sidecar; ambiguous paths remain unresolved.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_native_path_pack import NativePathPack


SCHEMA_VERSION = "0.1.0"
FORMAT_VERSION = "1.0.0"

ROLES = (
    "excluded_non_view_content",
    "equipment_fitting_evidence",
    "annotation_dimension",
    "text_associated_stroke_candidate",
    "measured_hatch_candidate",
    "architectural_boundary_candidate",
    "drawing_furniture",
    "outlined_corridor_source_candidate",
    "anchored_route_style_candidate",
    "unresolved_drawing_view_geometry",
)
ROLE_TO_CODE = {name: index + 1 for index, name in enumerate(ROLES)}
CODE_TO_ROLE = {value: key for key, value in ROLE_TO_CODE.items()}

FLAG_LEGACY_M3_CANDIDATE = 1
FLAG_ANCHORED_ROUTE_STYLE = 2
FLAG_TEXT_ASSOCIATED = 4
FLAG_HATCH_CLUSTER = 8
FLAG_ARCHITECTURAL_GEOMETRY = 16
FLAG_OUTLINED_CORRIDOR_MEMBER = 32
FLAG_INDEPENDENT_PRIOR_ROLE = 64

RECORD = struct.Struct("<BH")


def _intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _point_near_text(point: Sequence[float], boxes: Sequence[Sequence[float]],
                     factor: float = .35) -> bool:
    for box in boxes:
        height = max(1.0, float(box[3]) - float(box[1]))
        margin = factor * height
        if (float(box[0]) - margin <= point[0] <= float(box[2]) + margin
                and float(box[1]) - margin <= point[1] <= float(box[3]) + margin):
            return True
    return False


def _text_cells(boxes: Sequence[Sequence[float]], cell_size: float = 64.0):
    cells: dict[tuple[int, int], list[Sequence[float]]] = {}
    for box in boxes:
        x0, y0 = math.floor(box[0] / cell_size), math.floor(box[1] / cell_size)
        x1, y1 = math.floor(box[2] / cell_size), math.floor(box[3] / cell_size)
        for x in range(x0 - 1, x1 + 2):
            for y in range(y0 - 1, y1 + 2):
                cells.setdefault((x, y), []).append(box)
    return cells


def _nearby_text(path: Mapping[str, Any], cells, cell_size: float = 64.0):
    boxes = []
    seen = set()
    for point in (path["start_display"], path["end_display"]):
        cell = (math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for box in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    marker = tuple(box)
                    if marker not in seen:
                        seen.add(marker)
                        boxes.append(box)
    return boxes


def _hatch_key(path: Mapping[str, Any], *, cell_size: float = 96.0):
    if (path["kind_mask"] != 1 or path["axis_mask"] != 4
            or path["endpoint_closed"] or not 1.0 <= path["source_segment_length_points"] <= 40.0):
        return None
    start, end = path["start_display"], path["end_display"]
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    length_bucket = round(math.log2(max(1.0, path["source_segment_length_points"])), 1)
    return (path["style_id"], round(angle / 5.0), length_bucket,
            math.floor(midpoint[0] / cell_size), math.floor(midpoint[1] / cell_size))


class PageWidePathRolePack:
    def __init__(self, path: Path | str, manifest: Mapping[str, Any]):
        self.path = Path(path)
        self.manifest = manifest
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported page-wide path-role pack format")
        if manifest.get("record_size") != RECORD.size:
            raise ValueError("page-wide path-role record size mismatch")
        if self.path.stat().st_size != manifest.get("record_count", -1) * RECORD.size:
            raise ValueError("page-wide path-role pack length mismatch")

    def records(self) -> Iterator[dict[str, Any]]:
        with self.path.open("rb") as stream:
            ordinal = 0
            while body := stream.read(RECORD.size):
                if len(body) != RECORD.size:
                    raise ValueError("truncated page-wide path-role pack")
                role, flags = RECORD.unpack(body)
                if role not in CODE_TO_ROLE:
                    raise ValueError("unknown page-wide path role")
                yield {"path_ordinal": ordinal, "role": CODE_TO_ROLE[role],
                       "flags": flags}
                ordinal += 1

    def verify_hash(self) -> bool:
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == self.manifest.get("sha256")


def write_page_wide_path_roles(
    *, path_pack: NativePathPack, styles: Sequence[Mapping[str, Any]],
    output_path: Path | str, text_boxes_display: Sequence[Sequence[float]],
    anchored_style_ids: Iterable[int], outlined_member_drawing_ordinals: Iterable[int],
    page_rect_display: Sequence[float],
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("page-wide path-role output must be new")
    anchored = set(map(int, anchored_style_ids))
    outlined = set(map(int, outlined_member_drawing_ordinals))
    text_cells = _text_cells(text_boxes_display)

    # First pass measures dense, local, same-style oblique repetition.  It is
    # intentionally conservative: axis-aligned repetition remains unresolved.
    hatch_counts = Counter()
    for path in path_pack.records():
        if path["region_role"] != "main_plan_view":
            continue
        key = _hatch_key(path)
        if key is not None and path["style_id"] not in anchored:
            hatch_counts[key] += 1
    accepted_hatch_keys = {key for key, count in hatch_counts.items() if count >= 12}

    counts = Counter()
    flag_counts = Counter()
    digest = hashlib.sha256()
    segment_counts = Counter()
    page_width = float(page_rect_display[2]) - float(page_rect_display[0])
    page_height = float(page_rect_display[3]) - float(page_rect_display[1])
    with output_path.open("xb") as stream:
        for path in path_pack.records():
            flags = 0
            candidates = set(path["candidate_roles"])
            primaries = set(path["primary_dispositions"])
            if "legacy_m3_route_candidate" in candidates:
                flags |= FLAG_LEGACY_M3_CANDIDATE
            if path["style_id"] in anchored:
                flags |= FLAG_ANCHORED_ROUTE_STYLE
            if path["drawing_ordinal"] in outlined:
                flags |= FLAG_OUTLINED_CORRIDOR_MEMBER

            nearby = _nearby_text(path, text_cells)
            associated = bool(nearby and (
                _point_near_text(path["start_display"], nearby)
                or _point_near_text(path["end_display"], nearby)))
            if associated:
                flags |= FLAG_TEXT_ASSOCIATED

            hatch = _hatch_key(path) in accepted_hatch_keys
            if hatch:
                flags |= FLAG_HATCH_CLUSTER

            length = float(path["source_segment_length_points"])
            box = path["bbox_display"]
            closed_boundary = (path["endpoint_closed"]
                               and box[2] - box[0] >= .01 * page_width
                               and box[3] - box[1] >= .01 * page_height)
            long_axis_rule = (path["kind_mask"] == 1 and path["axis_mask"] in {1, 2}
                              and length >= .22 * min(page_width, page_height)
                              and path["style_id"] not in anchored)
            architectural = closed_boundary or long_axis_rule
            if architectural:
                flags |= FLAG_ARCHITECTURAL_GEOMETRY

            if path["region_role"] != "main_plan_view":
                role = "excluded_non_view_content"
            elif "equipment_fitting_evidence" in primaries:
                role = "equipment_fitting_evidence"
                flags |= FLAG_INDEPENDENT_PRIOR_ROLE
            elif "annotation_dimension" in primaries:
                role = "annotation_dimension"
                flags |= FLAG_INDEPENDENT_PRIOR_ROLE
            elif "drawing_furniture" in primaries:
                role = "drawing_furniture"
                flags |= FLAG_INDEPENDENT_PRIOR_ROLE
            elif associated and path["kind_mask"] == 1 and not path["endpoint_closed"]:
                role = "text_associated_stroke_candidate"
            elif hatch:
                role = "measured_hatch_candidate"
            elif architectural:
                role = "architectural_boundary_candidate"
            elif path["drawing_ordinal"] in outlined:
                role = "outlined_corridor_source_candidate"
            elif path["style_id"] in anchored:
                role = "anchored_route_style_candidate"
            else:
                role = "unresolved_drawing_view_geometry"
            body = RECORD.pack(ROLE_TO_CODE[role], flags)
            stream.write(body)
            digest.update(body)
            counts[role] += 1
            segment_counts[role] += int(path["source_segment_count"])
            for name, bit in (
                ("legacy_m3_candidate", FLAG_LEGACY_M3_CANDIDATE),
                ("anchored_route_style", FLAG_ANCHORED_ROUTE_STYLE),
                ("text_associated", FLAG_TEXT_ASSOCIATED),
                ("measured_hatch", FLAG_HATCH_CLUSTER),
                ("architectural_geometry", FLAG_ARCHITECTURAL_GEOMETRY),
                ("outlined_corridor_member", FLAG_OUTLINED_CORRIDOR_MEMBER),
            ):
                if flags & bit:
                    flag_counts[name] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "mep_page_wide_authored_path_roles",
        "format_version": FORMAT_VERSION,
        "record_size": RECORD.size,
        "record_count": len(path_pack),
        "source_segment_count": path_pack.manifest["source_segment_count"],
        "sha256": digest.hexdigest(),
        "role_path_counts": dict(sorted(counts.items())),
        "role_source_segment_counts": dict(sorted(segment_counts.items())),
        "measured_flag_path_counts": dict(sorted(flag_counts.items())),
        "hatch_cluster_key_count": len(accepted_hatch_keys),
        "text_bbox_count": len(text_boxes_display),
        "anchored_style_ids": sorted(anchored),
        "outlined_member_drawing_ordinal_count": len(outlined),
        "authority": {
            "every_authored_path_classified_once": True,
            "legacy_M3_direct_route_authority": False,
            "unclassified_paths_may_remain_unresolved": True,
            "route_identity_established": False,
            "system_identity_established": False,
            "quantity_eligible": False,
        },
    }


def validate_page_wide_path_roles(
    *, path_pack: NativePathPack, role_pack: PageWidePathRolePack,
) -> list[str]:
    errors = []
    if role_pack.manifest.get("record_count") != len(path_pack):
        errors.append("path-role count differs from authored path count")
    if role_pack.manifest.get("source_segment_count") != path_pack.manifest.get(
            "source_segment_count"):
        errors.append("path-role segment coverage differs from denominator")
    if sum(role_pack.manifest.get("role_path_counts", {}).values()) != len(path_pack):
        errors.append("path-role counts are not exhaustive")
    if sum(role_pack.manifest.get("role_source_segment_counts", {}).values()) != path_pack.manifest.get(
            "source_segment_count"):
        errors.append("path-role source-segment counts are not exhaustive")
    if not role_pack.verify_hash():
        errors.append("path-role pack hash verification failed")
    if any(row["role"] == "anchored_route_style_candidate"
           and not row["flags"] & FLAG_ANCHORED_ROUTE_STYLE
           for row in role_pack.records()):
        errors.append("route-style role lacks independent anchored-style flag")
    return errors
