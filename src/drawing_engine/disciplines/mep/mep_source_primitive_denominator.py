"""Complete compact source denominator for MEP drawing-view geometry.

The native descriptor pack establishes which PDF segments exist.  This module
adds one fixed-width disposition record per descriptor without expanding the
inventory into JSON.  Candidate-role bits preserve competing interpretations;
the primary disposition is nevertheless singular and exhaustive.

This layer establishes source coverage only.  In particular, ``route_evidence``
does not certify a semantic route, topology, physical identity, or quantity.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_native_descriptor_pack import NativeDescriptorPack


SCHEMA_VERSION = "0.1.0"
FORMAT_VERSION = "1.0.0"

DISPOSITIONS = (
    "route_evidence",
    "equipment_fitting_evidence",
    "annotation_dimension",
    "drawing_furniture",
    "unresolved_drawing_view_geometry",
    "excluded_non_view_content",
    "legacy_m3_route_candidate",
)
DISPOSITION_TO_CODE = {name: index + 1 for index, name in enumerate(DISPOSITIONS)}
CODE_TO_DISPOSITION = {value: key for key, value in DISPOSITION_TO_CODE.items()}
DISPOSITION_TO_BIT = {name: 1 << index for index, name in enumerate(DISPOSITIONS)}

REGION_ROLES = (
    "main_plan_view",
    "section_detail_riser",
    "title_block_revision_stamp",
    "border",
    "legend",
    "schedule_table",
    "notes_specifications",
    "unknown_region",
)
REGION_TO_CODE = {name: index + 1 for index, name in enumerate(REGION_ROLES)}
CODE_TO_REGION = {value: key for key, value in REGION_TO_CODE.items()}

VIEW_ROLES = {"main_plan_view", "section_detail_riser"}
NON_VIEW_ROLES = {
    "title_block_revision_stamp", "border", "legend", "schedule_table",
    "notes_specifications",
}

# One primary disposition, all candidate roles, and the owning sheet region.
RECORD = struct.Struct("<BBB")
EVIDENCE_ROLE_BITS = (
    "route_evidence", "equipment_fitting_evidence", "annotation_dimension",
    "drawing_furniture", "legacy_m3_route_candidate",
)


def _contains(outer: Sequence[float], inner: Sequence[float], tolerance: float = 1.0) -> bool:
    return (outer[0] - tolerance <= inner[0]
            and outer[1] - tolerance <= inner[1]
            and outer[2] + tolerance >= inner[2]
            and outer[3] + tolerance >= inner[3])


def _edge_anchored_long_rule(box: Sequence[float], page_rect: Sequence[float]) -> bool:
    width = float(page_rect[2]) - float(page_rect[0])
    height = float(page_rect[3]) - float(page_rect[1])
    if width <= 0 or height <= 0:
        raise ValueError("page rectangle must have positive dimensions")
    span_x = (float(box[2]) - float(box[0])) / width
    span_y = (float(box[3]) - float(box[1])) / height
    margin_x, margin_y = .02 * width, .02 * height
    anchored = (float(box[0]) <= float(page_rect[0]) + margin_x
                or float(box[2]) >= float(page_rect[2]) - margin_x
                or float(box[1]) <= float(page_rect[1]) + margin_y
                or float(box[3]) >= float(page_rect[3]) - margin_y)
    return anchored and max(span_x, span_y) >= .70


def _region_role(descriptor: Mapping[str, Any], *, page: Mapping[str, Any],
                 regions: Sequence[Mapping[str, Any]]) -> str:
    box = descriptor["bbox_display"]
    matches = [row for row in regions
               if row.get("state") == "accepted"
               and row.get("region_role") in NON_VIEW_ROLES
               and row.get("bbox_display")
               and _contains(row["bbox_display"], box, tolerance=2.0)]
    if matches:
        # More specific accepted regions win deterministically.
        matches.sort(key=lambda row: (
            (row["bbox_display"][2] - row["bbox_display"][0])
            * (row["bbox_display"][3] - row["bbox_display"][1]),
            row.get("id", "")))
        return matches[0]["region_role"]
    if _edge_anchored_long_rule(box, page["page_rect_display"]):
        return "border"
    role = page.get("accepted_view_role") or "unknown_region"
    return role if role in REGION_TO_CODE else "unknown_region"


def _candidate_mask(source_ref: str, role_refs: Mapping[str, set[str]],
                    *, region_role: str) -> int:
    mask = 0
    for role in EVIDENCE_ROLE_BITS:
        if source_ref in role_refs.get(role, set()):
            mask |= DISPOSITION_TO_BIT[role]
    if region_role in NON_VIEW_ROLES:
        mask |= DISPOSITION_TO_BIT["excluded_non_view_content"]
    elif region_role in VIEW_ROLES:
        mask |= DISPOSITION_TO_BIT["unresolved_drawing_view_geometry"]
    else:
        mask |= DISPOSITION_TO_BIT["unresolved_drawing_view_geometry"]
    return mask


def _primary_disposition(mask: int, *, region_role: str) -> str:
    if region_role in NON_VIEW_ROLES:
        return "excluded_non_view_content"
    # Legacy M3 membership is intentionally absent: reusing it as primary
    # route evidence would make this supposedly independent denominator
    # circular.  Only independently supplied route evidence may receive the
    # route_evidence primary disposition.
    for role in ("equipment_fitting_evidence", "annotation_dimension",
                 "route_evidence", "drawing_furniture"):
        if mask & DISPOSITION_TO_BIT[role]:
            return role
    return "unresolved_drawing_view_geometry"


class SourceDispositionPack:
    """Reader for the compact exactly-one source-disposition sidecar."""

    def __init__(self, path: Path | str, manifest: Mapping[str, Any]):
        self.path = Path(path)
        self.manifest = manifest
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported source disposition pack format")
        if manifest.get("record_size") != RECORD.size:
            raise ValueError("source disposition record size mismatch")
        if self.path.stat().st_size != manifest.get("record_count", -1) * RECORD.size:
            raise ValueError("source disposition pack length mismatch")

    def records(self):
        with self.path.open("rb") as stream:
            ordinal = 0
            while True:
                body = stream.read(RECORD.size)
                if not body:
                    return
                if len(body) != RECORD.size:
                    raise ValueError("truncated source disposition pack")
                primary, candidates, region = RECORD.unpack(body)
                if primary not in CODE_TO_DISPOSITION or region not in CODE_TO_REGION:
                    raise ValueError("unknown source disposition code")
                yield {
                    "descriptor_ordinal": ordinal,
                    "primary_disposition": CODE_TO_DISPOSITION[primary],
                    "candidate_role_mask": candidates,
                    "candidate_roles": [name for name in DISPOSITIONS
                                        if candidates & DISPOSITION_TO_BIT[name]],
                    "region_role": CODE_TO_REGION[region],
                }
                ordinal += 1

    def verify_hash(self) -> bool:
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == self.manifest.get("sha256")


def write_source_disposition_pack(
    *, descriptor_pack: NativeDescriptorPack, output_path: Path | str,
    page: Mapping[str, Any], regions: Sequence[Mapping[str, Any]],
    role_refs: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """Write one compact disposition for every native descriptor."""
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("source disposition output path must be new")
    normalized_refs = {name: set(role_refs.get(name, ())) for name in EVIDENCE_ROLE_BITS}
    seen_refs = {name: set() for name in EVIDENCE_ROLE_BITS}
    counts, region_counts, candidate_counts = Counter(), Counter(), Counter()
    conflicts = Counter()
    drawing_ordinals, item_paths = set(), set()
    digest = hashlib.sha256()
    record_count = 0
    with output_path.open("xb") as stream:
        for descriptor in descriptor_pack.descriptors():
            source_ref = descriptor["source_primitive_ref"]
            region_role = _region_role(descriptor, page=page, regions=regions)
            mask = _candidate_mask(source_ref, normalized_refs, region_role=region_role)
            primary = _primary_disposition(mask, region_role=region_role)
            body = RECORD.pack(DISPOSITION_TO_CODE[primary], mask, REGION_TO_CODE[region_role])
            stream.write(body)
            digest.update(body)
            counts[primary] += 1
            region_counts[region_role] += 1
            for role in DISPOSITIONS:
                if mask & DISPOSITION_TO_BIT[role]:
                    candidate_counts[role] += 1
            non_fallback = [role for role in EVIDENCE_ROLE_BITS
                            if mask & DISPOSITION_TO_BIT[role]]
            if len(non_fallback) > 1:
                conflicts["|".join(non_fallback)] += 1
            for role in EVIDENCE_ROLE_BITS:
                if source_ref in normalized_refs[role]:
                    seen_refs[role].add(source_ref)
            drawing = descriptor["drawing_ordinal"]
            item = descriptor["item_ordinal"]
            drawing_ordinals.add(drawing)
            item_paths.add((drawing, item))
            record_count += 1
    if record_count != len(descriptor_pack):
        raise ValueError("disposition count differs from native descriptor count")
    unaccounted = record_count - sum(counts.values())
    unmatched = {role: sorted(refs - seen_refs[role])
                 for role, refs in normalized_refs.items() if refs - seen_refs[role]}
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "mep_source_primitive_denominator_dispositions",
        "format_version": FORMAT_VERSION,
        "record_size": RECORD.size,
        "record_count": record_count,
        "sha256": digest.hexdigest(),
        "native_drawing_path_count": len(drawing_ordinals),
        "native_item_path_count": len(item_paths),
        "disposition_counts": dict(sorted(counts.items())),
        "candidate_role_counts": dict(sorted(candidate_counts.items())),
        "region_role_counts": dict(sorted(region_counts.items())),
        "competing_evidence_counts": dict(sorted(conflicts.items())),
        "unmatched_evidence_reference_count": sum(map(len, unmatched.values())),
        "unmatched_evidence_reference_counts": {
            role: len(refs) for role, refs in sorted(unmatched.items())},
        "unmatched_evidence_refs_sha256": hashlib.sha256(
            repr(sorted((role, tuple(refs)) for role, refs in unmatched.items())).encode()
        ).hexdigest(),
        "unaccounted_segment_count": unaccounted,
        "exactly_one_primary_disposition": unaccounted == 0,
        "authority": {
            "source_existence_and_coverage_established": True,
            "route_identity_established": False,
            "topology_established": False,
            "physical_identity_established": False,
            "quantity_eligible": False,
        },
    }


def validate_source_denominator(
    *, descriptor_pack: NativeDescriptorPack,
    disposition_pack: SourceDispositionPack,
) -> list[str]:
    errors = []
    manifest = disposition_pack.manifest
    if len(descriptor_pack) != manifest.get("record_count"):
        errors.append("native descriptor and disposition counts differ")
    if sum(manifest.get("disposition_counts", {}).values()) != manifest.get("record_count"):
        errors.append("primary disposition counts are not exhaustive")
    if manifest.get("unaccounted_segment_count") != 0:
        errors.append("accepted source segments remain unaccounted")
    if manifest.get("exactly_one_primary_disposition") is not True:
        errors.append("exactly-one disposition gate failed")
    if not descriptor_pack.verify_hashes():
        errors.append("native descriptor pack hash verification failed")
    if not disposition_pack.verify_hash():
        errors.append("source disposition pack hash verification failed")
    observed_count = 0
    for row in disposition_pack.records():
        observed_count += 1
        if row["primary_disposition"] not in row["candidate_roles"]:
            errors.append("primary disposition missing from candidate mask")
            break
        if (row["region_role"] in NON_VIEW_ROLES
                and row["primary_disposition"] != "excluded_non_view_content"):
            errors.append("non-view source promoted outside exclusion disposition")
            break
    if observed_count != len(descriptor_pack):
        errors.append("disposition reader did not replay every descriptor")
    return errors
