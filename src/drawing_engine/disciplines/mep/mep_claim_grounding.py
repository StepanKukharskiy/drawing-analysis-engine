"""M6A markup-claim grounding against partial projected MEP routes.

M0 annotation pairs remain authored markup claims.  This layer may bind one
claim to one uniquely overlapping accepted outlined-route composite when M5
also preserves that target as a partial 2.5D centreline.  Multiple overlaps
remain explicit alternatives in a 2D overlap candidate, and missing targets
abstain.  No record in this layer has 3D clash or M7 quantity authority.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_annotation_observations import validate_pdf_annotation_observations
from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_cross_sheet_runs import validate_mep_cross_sheet_runs
from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import validate_mep_outlined_route_composites


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_markup_claim_grounding"
METHOD_VERSION = "1.0.0"


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{kind}.{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _point(value: Sequence[object]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _rect_polygon(rect: Sequence[object]) -> list[list[float]]:
    x0, y0, x1, y1 = (float(value) for value in rect)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _orientation(a: Sequence[object], b: Sequence[object], c: Sequence[object]) -> float:
    ax, ay = _point(a)
    bx, by = _point(b)
    cx, cy = _point(c)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(
    point: Sequence[object], start: Sequence[object], end: Sequence[object], tolerance: float = 1e-7
) -> bool:
    px, py = _point(point)
    ax, ay = _point(start)
    bx, by = _point(end)
    return (
        abs(_orientation(start, end, point)) <= tolerance
        and min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance
        and min(ay, by) - tolerance <= py <= max(ay, by) + tolerance
    )


def _segments_intersect(
    a: Sequence[object], b: Sequence[object], c: Sequence[object], d: Sequence[object]
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if ((ab_c > 0 > ab_d) or (ab_c < 0 < ab_d)) and (
        (cd_a > 0 > cd_b) or (cd_a < 0 < cd_b)
    ):
        return True
    return any(
        abs(value) <= 1e-7 and _on_segment(point, start, end)
        for value, point, start, end in (
            (ab_c, c, a, b),
            (ab_d, d, a, b),
            (cd_a, a, c, d),
            (cd_b, b, c, d),
        )
    )


def _point_in_polygon(point: Sequence[object], polygon: Sequence[Sequence[object]]) -> bool:
    if len(polygon) < 3:
        return False
    if any(
        _on_segment(point, left, right)
        for left, right in zip(polygon, [*polygon[1:], polygon[0]])
    ):
        return True
    x, y = _point(point)
    inside = False
    for left, right in zip(polygon, [*polygon[1:], polygon[0]]):
        ax, ay = _point(left)
        bx, by = _point(right)
        if (ay > y) == (by > y):
            continue
        crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
        if crossing_x > x:
            inside = not inside
    return inside


def _polygons_overlap(
    left: Sequence[Sequence[object]], right: Sequence[Sequence[object]]
) -> bool:
    if len(left) < 3 or len(right) < 3:
        return False
    if any(_point_in_polygon(point, right) for point in left):
        return True
    if any(_point_in_polygon(point, left) for point in right):
        return True
    left_edges = list(zip(left, [*left[1:], left[0]]))
    right_edges = list(zip(right, [*right[1:], right[0]]))
    return any(
        _segments_intersect(a, b, c, d)
        for a, b in left_edges
        for c, d in right_edges
    )


def _claim_polygon(cloud: Mapping[str, Any]) -> tuple[list[list[float]], str]:
    geometry = cloud.get("geometry", {})
    vertices = geometry.get("vertices_display", []) or []
    if len(vertices) >= 3:
        return [[float(point[0]), float(point[1])] for point in vertices], "cloud_vertices"
    rect = geometry.get("rect_display", []) or []
    if len(rect) == 4:
        return _rect_polygon(rect), "cloud_rect"
    return [], "missing_cloud_geometry"


def _eligible_targets(
    outlined_route_composites: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    cross_sheet_runs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    partial_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for row in cross_sheet_runs.get("partial_2_5d_centreline_segments", []):
        partial_by_target.setdefault(str(row.get("route_target_ref")), []).append(row)
    resolved_targets = {
        str(row.get("route_target_ref"))
        for row in cross_sheet_runs.get("resolved_3d_centreline_segments", [])
    }
    embedded_composites = {
        str(row["id"]): row
        for row in attribute_bindings.get("outlined_route_composites", [])
    }
    accepted_relations: dict[str, list[Mapping[str, Any]]] = {}
    for relation in attribute_bindings.get("relations", []):
        if (
            relation.get("state") != "accepted"
            or relation.get("target_kind") != "route_composite"
            or relation.get("authority", {}).get("page_local_binding_established")
            is not True
        ):
            continue
        for ref in relation.get("target_refs", []):
            accepted_relations.setdefault(str(ref), []).append(relation)
    targets = []
    for composite in outlined_route_composites.get("accepted_composites", []):
        composite_ref = str(composite["id"])
        partial_rows = partial_by_target.get(composite_ref, [])
        relations = accepted_relations.get(composite_ref, [])
        if (
            len(partial_rows) != 1
            or composite_ref in resolved_targets
            or not relations
            or embedded_composites.get(composite_ref) != composite
        ):
            continue
        partial = partial_rows[0]
        if str(partial.get("page_ref")) != str(composite.get("page_ref")):
            continue
        corridor = composite.get("derived_geometry", {}).get(
            "corridor_boundary_points_display", []
        )
        if len(corridor) < 3:
            continue
        target_id = _stable_id(
            "mep_claim_grounding_target",
            composite_ref,
            partial.get("id"),
        )
        targets.append(
            {
                "record_type": "mep_claim_grounding_target",
                "record_version": SCHEMA_VERSION,
                "id": target_id,
                "page_ref": str(composite["page_ref"]),
                "m3_5_composite_ref": composite_ref,
                "m5_partial_2_5d_centreline_ref": str(partial["id"]),
                "m4_binding_relation_refs": sorted(
                    str(relation["id"]) for relation in relations
                ),
                "source_fragment_refs": sorted(
                    str(ref) for ref in composite.get("member_fragment_refs", [])
                ),
                "corridor_boundary_points_display": deepcopy(corridor),
                "centreline_points_display": deepcopy(
                    composite.get("derived_geometry", {}).get(
                        "centreline_points_display", []
                    )
                ),
                "m3_5_composite_sha256": _canonical_sha256(composite),
                "m5_partial_segment_sha256": _canonical_sha256(partial),
                "geometry_state": "partial_2_5d",
                "three_dimensional_authority": False,
                "quantity_eligible": False,
            }
        )
    return sorted(targets, key=lambda row: (row["page_ref"], row["id"]))


def _claim_provenance(
    pair: Mapping[str, Any],
    cloud: Mapping[str, Any],
    callout: Mapping[str, Any],
    claim_polygon: list[list[float]],
    geometry_basis: str,
) -> dict[str, Any]:
    return {
        "m0_annotation_pair_ref": str(pair["id"]),
        "m0_annotation_pair_sha256": _canonical_sha256(pair),
        "cloud_annotation_ref": str(cloud["id"]),
        "cloud_annotation_sha256": _canonical_sha256(cloud),
        "callout_annotation_ref": str(callout["id"]),
        "callout_annotation_sha256": _canonical_sha256(callout),
        "pair_relationship": pair.get("relationship"),
        "pair_epistemic_state": pair.get("epistemic_state"),
        "pair_evidence": deepcopy(pair.get("evidence", {})),
        "callout_text": str(callout.get("text") or ""),
        "callout_author": str(callout.get("author") or ""),
        "claim_polygon_display": claim_polygon,
        "claim_geometry_basis": geometry_basis,
        "coordinate_space": "page_display_points_top_left",
        "markup_claim_only_at_source": True,
    }


def _alternatives_for_claim(
    page_ref: str,
    claim_polygon: Sequence[Sequence[object]],
    targets: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for target in targets:
        if str(target["page_ref"]) != page_ref:
            continue
        corridor = target.get("corridor_boundary_points_display", [])
        if not _polygons_overlap(claim_polygon, corridor):
            continue
        output.append(
            {
                "target_ref": str(target["id"]),
                "m3_5_composite_ref": str(target["m3_5_composite_ref"]),
                "m5_partial_2_5d_centreline_ref": str(
                    target["m5_partial_2_5d_centreline_ref"]
                ),
                "relation": "claim_polygon_overlaps_route_corridor_in_2d",
                "coordinate_space": "page_display_points_top_left",
                "three_dimensional_authority": False,
            }
        )
    return sorted(output, key=lambda row: row["target_ref"])


def build_mep_claim_grounding(
    *,
    markup_observations: Mapping[str, Any],
    outlined_route_composites: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    cross_sheet_runs: Mapping[str, Any],
) -> dict[str, Any]:
    """Ground M0 claims to unique M3.5+M4+M5 partial targets or fail closed."""

    upstream_errors = {
        "M0": validate_pdf_annotation_observations(markup_observations),
        "M3.5": validate_mep_outlined_route_composites(outlined_route_composites),
        "M4": validate_mep_attribute_bindings(attribute_bindings),
        "M5": validate_mep_cross_sheet_runs(cross_sheet_runs),
    }
    failed = [f"{name}: {error}" for name, rows in upstream_errors.items() for error in rows]
    if failed:
        raise ValueError("invalid upstream claim-grounding payload:\n" + "\n".join(failed))
    documents = [
        markup_observations.get("document", {}),
        outlined_route_composites.get("document", {}),
        attribute_bindings.get("document", {}),
        cross_sheet_runs.get("document", {}),
    ]
    document_keys = {str(document.get("document_key")) for document in documents}
    if len(document_keys) != 1:
        raise ValueError("M0, M3.5, M4, and M5 document keys do not match")
    composite_hash = _canonical_sha256(outlined_route_composites)
    binding_hash = _canonical_sha256(attribute_bindings)
    if (
        attribute_bindings.get("m3_5_contract_ref", {}).get("payload_sha256")
        != composite_hash
    ):
        raise ValueError("M4 does not reference the supplied M3.5 composite contract")
    if (
        attribute_bindings.get("m3_contract_ref", {}).get("payload_sha256")
        != outlined_route_composites.get("m3_contract_ref", {}).get("payload_sha256")
    ):
        raise ValueError("M3.5 and M4 do not reference the same M3 route contract")
    if (
        cross_sheet_runs.get("m3_contract_ref", {}).get("payload_sha256")
        != outlined_route_composites.get("m3_contract_ref", {}).get("payload_sha256")
    ):
        raise ValueError("M3.5 and M5 do not reference the same M3 route contract")
    if cross_sheet_runs.get("m4_contract_ref", {}).get("payload_sha256") != binding_hash:
        raise ValueError("M5 does not reference the supplied M4 binding contract")

    annotations = {
        str(row["id"]): row for row in markup_observations.get("annotations", [])
    }
    targets = _eligible_targets(
        outlined_route_composites, attribute_bindings, cross_sheet_runs
    )
    grounding_records = []
    for pair in sorted(
        markup_observations.get("annotation_pairs", []), key=lambda row: str(row["id"])
    ):
        cloud = annotations[str(pair["cloud_annotation_ref"])]
        callout = annotations[str(pair["callout_annotation_ref"])]
        polygon, geometry_basis = _claim_polygon(cloud)
        provenance = _claim_provenance(
            pair, cloud, callout, polygon, geometry_basis
        )
        alternatives = _alternatives_for_claim(
            str(pair["page_ref"]), polygon, targets
        )
        page_targets = [
            target
            for target in targets
            if str(target["page_ref"]) == str(pair["page_ref"])
        ]
        common = {
            "record_version": SCHEMA_VERSION,
            "page_ref": str(pair["page_ref"]),
            "claim_pair_ref": str(pair["id"]),
            "claim_provenance": provenance,
            "alternatives": alternatives,
            "ambiguity": {
                "eligible_2d_overlap_count": len(alternatives),
                "resolved": len(alternatives) == 1,
            },
            "method": {
                "name": "m0_cloud_to_m3_5_corridor_overlap_with_m4_binding_and_m5_partial_target_gate",
                "version": METHOD_VERSION,
            },
            "three_dimensional_authority": False,
            "m7_eligible": False,
            "quantity_eligible": False,
        }
        if len(alternatives) == 1:
            alternative = alternatives[0]
            record = {
                **common,
                "record_type": "mep_claim_target_grounding",
                "id": _stable_id(
                    "mep_claim_target_grounding",
                    pair["id"],
                    alternative["target_ref"],
                ),
                "target_ref": alternative["target_ref"],
                "m3_5_composite_ref": alternative["m3_5_composite_ref"],
                "m5_partial_2_5d_centreline_ref": alternative[
                    "m5_partial_2_5d_centreline_ref"
                ],
                "state": "accepted",
                "epistemic_state": "derived",
                "reasons": [],
            }
        elif alternatives:
            record = {
                **common,
                "record_type": "2d_overlap_candidate",
                "id": _stable_id(
                    "mep_2d_overlap_candidate",
                    pair["id"],
                    [row["target_ref"] for row in alternatives],
                ),
                "target_ref": None,
                "m3_5_composite_ref": None,
                "m5_partial_2_5d_centreline_ref": None,
                "state": "candidate",
                "epistemic_state": "unknown",
                "reasons": ["multiple_partial_2_5d_targets_overlap_claim_in_2d"],
            }
        else:
            if geometry_basis == "missing_cloud_geometry":
                abstention_reason = "missing_claim_cloud_geometry"
            elif not page_targets:
                abstention_reason = "no_eligible_partial_2_5d_target_on_claim_page"
            else:
                abstention_reason = "claim_polygon_does_not_overlap_eligible_page_target"
            record = {
                **common,
                "record_type": "mep_claim_grounding_abstention",
                "id": _stable_id("mep_claim_grounding_abstention", pair["id"]),
                "target_ref": None,
                "m3_5_composite_ref": None,
                "m5_partial_2_5d_centreline_ref": None,
                "state": "abstained",
                "epistemic_state": "unknown",
                "reasons": [abstention_reason],
            }
        grounding_records.append(record)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(markup_observations.get("document", {}))),
        "m0_contract_ref": {
            "layer": markup_observations.get("layer"),
            "schema_version": markup_observations.get("schema_version"),
            "payload_sha256": _canonical_sha256(markup_observations),
        },
        "m3_5_contract_ref": {
            "layer": outlined_route_composites.get("layer"),
            "schema_version": outlined_route_composites.get("schema_version"),
            "payload_sha256": composite_hash,
            "m3_payload_sha256": outlined_route_composites.get(
                "m3_contract_ref", {}
            ).get("payload_sha256"),
        },
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": binding_hash,
            "m3_payload_sha256": attribute_bindings.get(
                "m3_contract_ref", {}
            ).get("payload_sha256"),
            "m3_5_payload_sha256": attribute_bindings.get(
                "m3_5_contract_ref", {}
            ).get("payload_sha256"),
        },
        "m5_contract_ref": {
            "layer": cross_sheet_runs.get("layer"),
            "schema_version": cross_sheet_runs.get("schema_version"),
            "payload_sha256": _canonical_sha256(cross_sheet_runs),
            "m3_payload_sha256": cross_sheet_runs.get("m3_contract_ref", {}).get(
                "payload_sha256"
            ),
            "m4_payload_sha256": cross_sheet_runs.get("m4_contract_ref", {}).get(
                "payload_sha256"
            ),
        },
        "eligible_targets": targets,
        "grounding_records": grounding_records,
        "summary": {
            "claim_count": len(grounding_records),
            "eligible_partial_2_5d_target_count": len(targets),
            "accepted_unique_grounding_count": sum(
                row["state"] == "accepted" for row in grounding_records
            ),
            "ambiguous_2d_overlap_candidate_count": sum(
                row["record_type"] == "2d_overlap_candidate"
                for row in grounding_records
            ),
            "abstention_count": sum(
                row["state"] == "abstained" for row in grounding_records
            ),
        },
        "exchange_contract": {
            "m0_markup_claim_provenance_preserved": True,
            "m3_5_m4_and_m5_contract_hashes_preserved": True,
            "accepted_target_requires_unique_2d_overlap": True,
            "alternatives_and_ambiguity_preserved": True,
            "partial_2_5d_target_required": True,
            "three_dimensional_geometry_resolved": False,
            "clash_authority_enabled": False,
            "calculated_severity_enabled": False,
            "installed_length_emitted": False,
            "m7_outputs_enabled": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_claim_grounding(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def _walk_items(
    value: object, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk_items(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, (*path, f"[{index}]"))


def validate_mep_claim_grounding(payload: Mapping[str, Any]) -> list[str]:
    """Validate M6A references, ambiguity, and the unresolved-3D boundary."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for name in (
        "m0_contract_ref",
        "m3_5_contract_ref",
        "m4_contract_ref",
        "m5_contract_ref",
    ):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    m3_5_ref = payload.get("m3_5_contract_ref", {})
    m4_ref = payload.get("m4_contract_ref", {})
    m5_ref = payload.get("m5_contract_ref", {})
    if m3_5_ref.get("m3_payload_sha256") != m4_ref.get("m3_payload_sha256"):
        errors.append("M3.5 and M4 M3 contract hashes do not match")
    if m3_5_ref.get("payload_sha256") != m4_ref.get("m3_5_payload_sha256"):
        errors.append("M4 M3.5 contract hash does not match")
    if m4_ref.get("m3_payload_sha256") != payload.get(
        "m5_contract_ref", {}
    ).get("m3_payload_sha256"):
        errors.append("M4 and M5 M3 contract hashes do not match")
    if m4_ref.get("payload_sha256") != m5_ref.get("m4_payload_sha256"):
        errors.append("M5 M4 contract hash does not match")

    false_authority_keys = {
        "three_dimensional_authority",
        "three_dimensional_geometry_resolved",
        "clash_authority_enabled",
        "calculated_severity_enabled",
        "installed_length_emitted",
        "m7_outputs_enabled",
        "m7_eligible",
    }
    forbidden_keys = {
        "confirmed_clash",
        "calculated_severity",
        "severity",
        "installed_length",
        "quantity",
        "takeoff",
        "clash_status",
        "m7_output",
    }
    forbidden_prefixes = (
        "confirmed_clash_",
        "calculated_severity_",
        "installed_length_",
        "quantity_",
        "m7_output_",
    )
    allowed_prefixed = {
        "calculated_severity_enabled",
        "installed_length_emitted",
        "quantity_eligible",
        "m7_outputs_enabled",
    }
    for path, key, value in _walk_items(payload):
        location = ".".join((*path, key))
        if key in forbidden_keys or (
            key.startswith(forbidden_prefixes) and key not in allowed_prefixed
        ):
            errors.append(f"{location}: forbidden M6A authority output")
        if key in false_authority_keys and value is not False:
            errors.append(f"{location}: authority flag must remain false")
        if key == "quantity_eligible" and value is not False:
            errors.append(f"{location}: quantity_eligible must remain false")

    targets = {str(row.get("id")): row for row in payload.get("eligible_targets", [])}
    if len(targets) != len(payload.get("eligible_targets", [])):
        errors.append("eligible target IDs must be unique")
    for target in targets.values():
        if target.get("geometry_state") != "partial_2_5d":
            errors.append(f"{target.get('id')}: target is not partial 2.5D")
        if not target.get("m3_5_composite_ref") or not target.get(
            "m5_partial_2_5d_centreline_ref"
        ):
            errors.append(f"{target.get('id')}: upstream target references are incomplete")
        if not target.get("m4_binding_relation_refs"):
            errors.append(f"{target.get('id')}: target lacks accepted M4 binding relation")

    records = list(payload.get("grounding_records", []))
    record_ids = [str(row.get("id")) for row in records]
    if len(record_ids) != len(set(record_ids)):
        errors.append("grounding record IDs must be unique")
    for row in records:
        alternatives = list(row.get("alternatives", []))
        alternative_refs = [str(item.get("target_ref")) for item in alternatives]
        provenance = row.get("claim_provenance", {})
        claim_polygon = provenance.get("claim_polygon_display", [])
        if str(row.get("claim_pair_ref")) != str(
            provenance.get("m0_annotation_pair_ref")
        ):
            errors.append(f"{row.get('id')}: claim provenance pair reference mismatch")
        if any(ref not in targets for ref in alternative_refs):
            errors.append(f"{row.get('id')}: alternative references unknown target")
        for alternative in alternatives:
            target = targets.get(str(alternative.get("target_ref")))
            if target is not None and (
                alternative.get("m3_5_composite_ref")
                != target.get("m3_5_composite_ref")
                or alternative.get("m5_partial_2_5d_centreline_ref")
                != target.get("m5_partial_2_5d_centreline_ref")
            ):
                errors.append(f"{row.get('id')}: alternative upstream references mismatch")
            if target is not None and str(row.get("page_ref")) != str(
                target.get("page_ref")
            ):
                errors.append(f"{row.get('id')}: alternative is not page-local")
            if target is not None and not _polygons_overlap(
                claim_polygon, target.get("corridor_boundary_points_display", [])
            ):
                errors.append(f"{row.get('id')}: alternative lacks geometric overlap")
        if len(alternative_refs) != len(set(alternative_refs)):
            errors.append(f"{row.get('id')}: duplicate alternatives")
        ambiguity = row.get("ambiguity", {})
        if ambiguity.get("eligible_2d_overlap_count") != len(alternatives):
            errors.append(f"{row.get('id')}: ambiguity count mismatch")
        record_type = row.get("record_type")
        if record_type == "mep_claim_target_grounding":
            alternative = alternatives[0] if len(alternatives) == 1 else {}
            if (
                row.get("state") != "accepted"
                or len(alternatives) != 1
                or str(row.get("target_ref")) != alternative_refs[0]
                or row.get("m3_5_composite_ref")
                != alternative.get("m3_5_composite_ref")
                or row.get("m5_partial_2_5d_centreline_ref")
                != alternative.get("m5_partial_2_5d_centreline_ref")
                or ambiguity.get("resolved") is not True
                or row.get("reasons")
            ):
                errors.append(f"{row.get('id')}: accepted grounding is not uniquely closed")
        elif record_type == "2d_overlap_candidate":
            if (
                row.get("state") != "candidate"
                or len(alternatives) < 2
                or row.get("target_ref") is not None
                or ambiguity.get("resolved") is not False
                or not row.get("reasons")
            ):
                errors.append(f"{row.get('id')}: ambiguous overlap was promoted")
        elif record_type == "mep_claim_grounding_abstention":
            if (
                row.get("state") != "abstained"
                or alternatives
                or row.get("target_ref") is not None
                or ambiguity.get("resolved") is not False
                or not row.get("reasons")
            ):
                errors.append(f"{row.get('id')}: invalid grounding abstention")
        else:
            errors.append(f"{row.get('id')}: unsupported grounding record type")
        for key in (
            "m0_annotation_pair_ref",
            "m0_annotation_pair_sha256",
            "cloud_annotation_ref",
            "cloud_annotation_sha256",
            "callout_annotation_ref",
            "callout_annotation_sha256",
        ):
            if not provenance.get(key):
                errors.append(f"{row.get('id')}: incomplete M0 claim provenance")

    summary = payload.get("summary", {})
    expected_summary = {
        "claim_count": len(records),
        "eligible_partial_2_5d_target_count": len(targets),
        "accepted_unique_grounding_count": sum(
            row.get("state") == "accepted" for row in records
        ),
        "ambiguous_2d_overlap_candidate_count": sum(
            row.get("record_type") == "2d_overlap_candidate" for row in records
        ),
        "abstention_count": sum(row.get("state") == "abstained" for row in records),
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            errors.append(f"summary.{key} must be {expected}")

    contract = payload.get("exchange_contract", {})
    for key in (
        "m0_markup_claim_provenance_preserved",
        "m3_5_m4_and_m5_contract_hashes_preserved",
        "accepted_target_requires_unique_2d_overlap",
        "alternatives_and_ambiguity_preserved",
        "partial_2_5d_target_required",
    ):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in (
        "three_dimensional_geometry_resolved",
        "clash_authority_enabled",
        "calculated_severity_enabled",
        "installed_length_emitted",
        "m7_outputs_enabled",
        "quantity_eligible",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    return errors
