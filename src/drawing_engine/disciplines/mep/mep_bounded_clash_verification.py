"""M6B bounded clash verification from frozen M6A claims and M5A envelopes.

The layer can verify only the local intersection of two independently accepted
M5A envelopes in one registered metric frame.  A 2D markup overlap is merely
the claim scope: both alternatives must resolve uniquely to different bounded
segments, and a geometric intersection must remain outside both unresolved
analysis-cap zones.  The result says nothing about whole runs, installed
length, calculated severity, takeoff, or quantity.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_bounded_local_3d import validate_mep_bounded_local_3d_segments
from src.drawing_engine.disciplines.mep.mep_claim_grounding import validate_mep_claim_grounding


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_bounded_clash_verification"
METHOD_VERSION = "1.0.0"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _point3(value: Sequence[object]) -> tuple[float, float, float]:
    return float(value[0]), float(value[1]), float(value[2])


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(item * factor for item in value)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _closest_segment_points(
    p1: Sequence[float], q1: Sequence[float], p2: Sequence[float], q2: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float, float]:
    """Return closest points, parameters, and distance for two finite 3D segments."""

    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    offset = _sub(p1, p2)
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    epsilon = 1e-15
    if a <= epsilon and e <= epsilon:
        left = _point3(p1)
        right = _point3(p2)
        return left, right, 0.0, 0.0, math.dist(left, right)
    if a <= epsilon:
        left_t = 0.0
        right_t = max(0.0, min(1.0, _dot(d2, offset) / e))
    else:
        c = _dot(d1, offset)
        if e <= epsilon:
            right_t = 0.0
            left_t = max(0.0, min(1.0, -c / a))
        else:
            b = _dot(d1, d2)
            denominator = a * e - b * b
            left_t = 0.0 if abs(denominator) <= epsilon else max(
                0.0, min(1.0, (b * _dot(d2, offset) - c * e) / denominator)
            )
            right_t = (b * left_t + _dot(d2, offset)) / e
            if right_t < 0.0:
                right_t = 0.0
                left_t = max(0.0, min(1.0, -c / a))
            elif right_t > 1.0:
                right_t = 1.0
                left_t = max(0.0, min(1.0, (b - c) / a))
    left = _add(p1, _scale(d1, left_t))
    right = _add(p2, _scale(d2, right_t))
    return left, right, left_t, right_t, math.dist(left, right)


def _polyline_lengths(points: list[tuple[float, float, float]]) -> tuple[list[float], float]:
    cumulative = [0.0]
    for left, right in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(left, right))
    return cumulative, cumulative[-1]


def _closest_polylines(
    left: list[tuple[float, float, float]],
    right: list[tuple[float, float, float]],
) -> dict[str, Any] | None:
    if len(left) < 2 or len(right) < 2:
        return None
    left_lengths, left_total = _polyline_lengths(left)
    right_lengths, right_total = _polyline_lengths(right)
    best: dict[str, Any] | None = None
    for left_index, (left_a, left_b) in enumerate(zip(left, left[1:])):
        left_span = math.dist(left_a, left_b)
        for right_index, (right_a, right_b) in enumerate(zip(right, right[1:])):
            right_span = math.dist(right_a, right_b)
            left_point, right_point, left_t, right_t, distance = _closest_segment_points(
                left_a, left_b, right_a, right_b
            )
            candidate = {
                "centreline_distance_m": distance,
                "left_point_xyz_m": left_point,
                "right_point_xyz_m": right_point,
                "left_path_offset_m": left_lengths[left_index] + left_t * left_span,
                "right_path_offset_m": right_lengths[right_index] + right_t * right_span,
                "left_total_path_m": left_total,
                "right_total_path_m": right_total,
            }
            if best is None or distance < best["centreline_distance_m"]:
                best = candidate
    return best


def _frame_signature(frame: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "id": frame.get("id"),
            "state": frame.get("state"),
            "reference_page_ref": frame.get("reference_page_ref"),
            "member_page_refs": sorted(str(ref) for ref in frame.get("member_page_refs", [])),
            "display_to_reference_display_matrices": frame.get(
                "display_to_reference_display_matrices"
            ),
            "reference_scale_m_per_display_point": frame.get(
                "reference_scale_m_per_display_point"
            ),
            "axes": frame.get("axes"),
            "absolute_origin_established": frame.get("absolute_origin_established"),
        }
    )


def _segment_geometry(segment: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    envelope = segment.get("local_envelope", {})
    points = segment.get("centreline_points_xyz_m", [])
    envelope_points = envelope.get("centreline_points_xyz_m", [])
    try:
        centreline = [_point3(point) for point in points]
        width = float(segment.get("physical_envelope_dimension", {}).get("representative_outer_width_m"))
        radius = float(envelope.get("radius_m"))
    except (TypeError, ValueError, IndexError):
        return None, "invalid_bounded_envelope_geometry"
    if (
        len(centreline) < 2
        or any(not all(math.isfinite(value) for value in point) for point in centreline)
        or envelope.get("kind") != "circular_swept_corridor"
        or envelope_points != points
        or not math.isfinite(width)
        or width <= 0
        or not math.isfinite(radius)
        or radius <= 0
        or not math.isclose(radius * 2.0, width, rel_tol=1e-8, abs_tol=1e-9)
    ):
        return None, "invalid_bounded_envelope_geometry"
    observations = segment.get("physical_envelope_dimension", {}).get(
        "occurrence_observations", []
    )
    resolution = max(
        [float(row.get("measurement_resolution_m") or 0.0) for row in observations]
        or [0.0]
    )
    width_residual = float(
        segment.get("physical_envelope_dimension", {}).get("maximum_width_residual_m")
        or 0.0
    )
    frame_scale = float(
        segment.get("m1_metric_frame", {}).get("reference_scale_m_per_display_point")
        or 0.0
    )
    path_residual = float(
        segment.get("duplicate_occurrence_agreement", {}).get(
            "maximum_path_residual_display_points"
        )
        or 0.0
    ) * frame_scale
    return {
        "points": centreline,
        "radius_m": radius,
        "geometry_uncertainty_m": max(1e-9, resolution / 2.0 + width_residual / 2.0 + path_residual),
    }, None


def _authority() -> dict[str, bool]:
    return {
        "bounded_clash_verification_established": False,
        "physical_run_identity_established": False,
        "physical_continuation_established": False,
        "whole_run_conclusion_established": False,
        "installed_length_emitted": False,
        "calculated_severity_emitted": False,
        "m7_takeoff_emitted": False,
        "quantity_eligible": False,
    }


def _abstention(
    grounding: Mapping[str, Any],
    *,
    segment_refs: Iterable[str],
    reasons: Iterable[str],
) -> dict[str, Any]:
    return {
        "record_type": "mep_bounded_clash_abstention",
        "record_version": SCHEMA_VERSION,
        "id": _stable_id("mep_bounded_clash_abstention", grounding.get("id"), sorted(reasons)),
        "claim_grounding_ref": str(grounding.get("id")),
        "claim_pair_ref": str(grounding.get("claim_pair_ref")),
        "claim_provenance": deepcopy(grounding.get("claim_provenance", {})),
        "claim_grounding_sha256": _canonical_sha256(grounding),
        "input_grounding_type": grounding.get("record_type"),
        "finding_kind": "clash",
        "bounded_segment_refs": sorted(set(str(ref) for ref in segment_refs)),
        "state": "abstained",
        "epistemic_state": "unknown",
        "reasons": sorted(set(str(reason) for reason in reasons)),
        "authority": _authority(),
        "quantity_eligible": False,
    }


def build_mep_bounded_clash_verification(
    *,
    attribute_bindings: Mapping[str, Any],
    bounded_local_3d: Mapping[str, Any],
    claim_grounding: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify claim-scoped local envelope intersections or abstain explicitly."""

    upstream = (
        ("M4", validate_mep_attribute_bindings(attribute_bindings)),
        ("M5A", validate_mep_bounded_local_3d_segments(bounded_local_3d)),
        ("M6A", validate_mep_claim_grounding(claim_grounding)),
    )
    failures = [f"{name}: {error}" for name, rows in upstream for error in rows]
    if failures:
        raise ValueError("invalid upstream bounded-clash payload:\n" + "\n".join(failures))
    documents = [
        attribute_bindings.get("document", {}),
        bounded_local_3d.get("document", {}),
        claim_grounding.get("document", {}),
    ]
    if len({str(row.get("document_key")) for row in documents}) != 1:
        raise ValueError("M4, M5A, and M6A document keys do not match")
    binding_hash = _canonical_sha256(attribute_bindings)
    if bounded_local_3d.get("m4_contract_ref", {}).get("payload_sha256") != binding_hash:
        raise ValueError("M5A does not reference the supplied M4 binding contract")
    if claim_grounding.get("m4_contract_ref", {}).get("payload_sha256") != binding_hash:
        raise ValueError("M6A does not reference the supplied M4 binding contract")

    segments = {
        str(row["id"]): row
        for row in bounded_local_3d.get("bounded_local_3d_segments", [])
    }
    segments_by_composite: dict[str, list[str]] = {}
    for segment_ref, segment in segments.items():
        for composite_ref in segment.get("source_route_composite_refs", []):
            segments_by_composite.setdefault(str(composite_ref), []).append(segment_ref)
    targets = {
        str(row["id"]): row for row in claim_grounding.get("eligible_targets", [])
    }

    records: list[dict[str, Any]] = []
    for grounding in sorted(
        claim_grounding.get("grounding_records", []), key=lambda row: str(row.get("id"))
    ):
        if grounding.get("record_type") != "2d_overlap_candidate":
            reason = (
                "second_independent_bounded_target_missing"
                if grounding.get("record_type") == "mep_claim_target_grounding"
                else "m6a_claim_target_not_grounded"
            )
            records.append(_abstention(grounding, segment_refs=[], reasons=[reason]))
            continue
        alternatives = list(grounding.get("alternatives", []))
        if len(alternatives) != 2:
            records.append(
                _abstention(
                    grounding,
                    segment_refs=[],
                    reasons=["bounded_target_pair_not_mutually_unique"],
                )
            )
            continue
        mapped: list[str] = []
        mapping_failed = False
        for alternative in alternatives:
            target = targets.get(str(alternative.get("target_ref")))
            candidates = (
                []
                if target is None
                else segments_by_composite.get(str(target.get("m3_5_composite_ref")), [])
            )
            if len(candidates) != 1:
                mapping_failed = True
            else:
                mapped.append(candidates[0])
        if mapping_failed:
            records.append(
                _abstention(
                    grounding,
                    segment_refs=mapped,
                    reasons=["m6a_target_does_not_map_uniquely_to_m5a_segment"],
                )
            )
            continue
        if len(set(mapped)) != 2:
            records.append(
                _abstention(
                    grounding,
                    segment_refs=mapped,
                    reasons=["alternatives_resolve_to_same_bounded_segment"],
                )
            )
            continue
        left, right = (segments[ref] for ref in mapped)
        independent = (
            left.get("canonical_projected_segment_ref")
            != right.get("canonical_projected_segment_ref")
            and set(left.get("source_page_occurrence_refs", [])).isdisjoint(
                right.get("source_page_occurrence_refs", [])
            )
        )
        if not independent:
            records.append(
                _abstention(
                    grounding,
                    segment_refs=mapped,
                    reasons=["bounded_envelopes_are_not_independent"],
                )
            )
            continue
        left_frame = left.get("m1_metric_frame", {})
        right_frame = right.get("m1_metric_frame", {})
        if (
            left_frame.get("state") != "accepted"
            or right_frame.get("state") != "accepted"
            or _frame_signature(left_frame) != _frame_signature(right_frame)
        ):
            records.append(
                _abstention(
                    grounding,
                    segment_refs=mapped,
                    reasons=["bounded_envelope_frames_are_incompatible"],
                )
            )
            continue
        left_geometry, left_reason = _segment_geometry(left)
        right_geometry, right_reason = _segment_geometry(right)
        if left_geometry is None or right_geometry is None:
            records.append(
                _abstention(
                    grounding,
                    segment_refs=mapped,
                    reasons=[left_reason or right_reason or "invalid_bounded_envelope_geometry"],
                )
            )
            continue
        closest = _closest_polylines(left_geometry["points"], right_geometry["points"])
        if closest is None:
            records.append(
                _abstention(
                    grounding,
                    segment_refs=mapped,
                    reasons=["invalid_bounded_envelope_geometry"],
                )
            )
            continue
        radius_sum = left_geometry["radius_m"] + right_geometry["radius_m"]
        uncertainty = left_geometry["geometry_uncertainty_m"] + right_geometry[
            "geometry_uncertainty_m"
        ]
        left_cap_clearance = min(
            closest["left_path_offset_m"],
            closest["left_total_path_m"] - closest["left_path_offset_m"],
        )
        right_cap_clearance = min(
            closest["right_path_offset_m"],
            closest["right_total_path_m"] - closest["right_path_offset_m"],
        )
        cap_exclusion = radius_sum + uncertainty
        away_from_caps = min(left_cap_clearance, right_cap_clearance) > cap_exclusion
        signed_clearance = closest["centreline_distance_m"] - radius_sum
        minimum_penetration = -signed_clearance - uncertainty
        minimum_clearance = signed_clearance - uncertainty
        observation = {
            "frame_ref": str(left_frame.get("id")),
            "left_closest_point_xyz_m": [round(value, 9) for value in closest["left_point_xyz_m"]],
            "right_closest_point_xyz_m": [round(value, 9) for value in closest["right_point_xyz_m"]],
            "centreline_distance_m": round(closest["centreline_distance_m"], 9),
            "combined_envelope_radius_m": round(radius_sum, 9),
            "signed_envelope_clearance_m": round(signed_clearance, 9),
            "geometry_uncertainty_m": round(uncertainty, 9),
            "minimum_certified_penetration_m": round(max(0.0, minimum_penetration), 9),
            "minimum_certified_clearance_m": round(max(0.0, minimum_clearance), 9),
            "left_terminal_path_clearance_m": round(left_cap_clearance, 9),
            "right_terminal_path_clearance_m": round(right_cap_clearance, 9),
            "analysis_cap_exclusion_distance_m": round(cap_exclusion, 9),
            "intersection_away_from_analysis_caps": away_from_caps,
        }
        if minimum_penetration > 0.0 and not away_from_caps:
            records.append(
                {
                    **_abstention(
                        grounding,
                        segment_refs=mapped,
                        reasons=["intersection_overlaps_unresolved_analysis_cap"],
                    ),
                    "geometry_observation": observation,
                }
            )
            continue
        if minimum_penetration <= 0.0 and minimum_clearance <= 0.0:
            records.append(
                {
                    **_abstention(
                        grounding,
                        segment_refs=mapped,
                        reasons=["envelope_contact_within_measurement_uncertainty"],
                    ),
                    "geometry_observation": observation,
                }
            )
            continue
        confirmed = minimum_penetration > 0.0 and away_from_caps
        authority = _authority()
        authority["bounded_clash_verification_established"] = True
        record = {
            "record_type": "mep_bounded_clash_certificate",
            "record_version": SCHEMA_VERSION,
            "id": _stable_id("mep_bounded_clash_certificate", grounding.get("id"), sorted(mapped)),
            "claim_grounding_ref": str(grounding.get("id")),
            "claim_pair_ref": str(grounding.get("claim_pair_ref")),
            "claim_provenance": deepcopy(grounding.get("claim_provenance", {})),
            "claim_grounding_sha256": _canonical_sha256(grounding),
            "input_grounding_type": grounding.get("record_type"),
            "finding_kind": "clash",
            "bounded_segment_refs": sorted(mapped),
            "geometry_observation": observation,
            "clash_status": "confirmed_bounded_clash" if confirmed else "clear_in_bounded_scope",
            "confirmed_clash": confirmed,
            "state": "accepted",
            "epistemic_state": "derived",
            "reasons": [],
            "authority": {
                **authority,
                "confirmed_bounded_clash_established": confirmed,
            },
            "quantity_eligible": False,
        }
        records.append(record)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(claim_grounding.get("document", {}))),
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": binding_hash,
        },
        "m5a_contract_ref": {
            "layer": bounded_local_3d.get("layer"),
            "schema_version": bounded_local_3d.get("schema_version"),
            "payload_sha256": _canonical_sha256(bounded_local_3d),
            "m4_payload_sha256": bounded_local_3d.get("m4_contract_ref", {}).get(
                "payload_sha256"
            ),
        },
        "m6a_contract_ref": {
            "layer": claim_grounding.get("layer"),
            "schema_version": claim_grounding.get("schema_version"),
            "payload_sha256": _canonical_sha256(claim_grounding),
            "m4_payload_sha256": claim_grounding.get("m4_contract_ref", {}).get(
                "payload_sha256"
            ),
        },
        "verification_records": sorted(records, key=lambda row: row["id"]),
        "summary": {
            "claim_count": len(records),
            "confirmed_bounded_clash_count": sum(
                row.get("confirmed_clash") is True for row in records
            ),
            "clear_in_bounded_scope_count": sum(
                row.get("clash_status") == "clear_in_bounded_scope" for row in records
            ),
            "abstention_count": sum(row.get("state") == "abstained" for row in records),
            "whole_run_conclusion_count": 0,
        },
        "exchange_contract": {
            "m6a_claim_provenance_preserved": True,
            "two_independent_m5a_envelopes_required": True,
            "compatible_registered_metric_frame_required": True,
            "intersection_away_from_unresolved_analysis_caps_required": True,
            "measurement_uncertainty_preserved": True,
            "clearance_and_access_require_explicit_zone_geometry": True,
            "m5b_whole_run_reasoning_separate": True,
            "physical_run_identity_established": False,
            "physical_continuation_established": False,
            "whole_run_conclusion_established": False,
            "installed_length_emitted": False,
            "calculated_severity_emitted": False,
            "m7_takeoff_emitted": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_bounded_clash_verification(payload)
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


def validate_mep_bounded_clash_verification(payload: Mapping[str, Any]) -> list[str]:
    """Validate bounded authority and reject M5B/M7 promotion recursively."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for name in ("m4_contract_ref", "m5a_contract_ref", "m6a_contract_ref"):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    if payload.get("m4_contract_ref", {}).get("payload_sha256") != payload.get(
        "m5a_contract_ref", {}
    ).get("m4_payload_sha256"):
        errors.append("M5A M4 contract hash mismatch")
    if payload.get("m4_contract_ref", {}).get("payload_sha256") != payload.get(
        "m6a_contract_ref", {}
    ).get("m4_payload_sha256"):
        errors.append("M6A M4 contract hash mismatch")

    false_authority = {
        "physical_run_identity_established",
        "physical_continuation_established",
        "whole_run_conclusion_established",
        "installed_length_emitted",
        "calculated_severity_emitted",
        "m7_takeoff_emitted",
        "quantity_eligible",
    }
    forbidden = {
        "installed_length",
        "fitting_count",
        "calculated_severity",
        "severity",
        "takeoff",
        "quantity",
        "physical_run_hypotheses",
        "whole_run_conclusion",
    }
    for path, key, value in _walk_items(payload):
        location = ".".join((*path, key))
        if key in forbidden:
            errors.append(f"{location}: forbidden M5B/M7 authority output")
        if key in false_authority and value is not False:
            errors.append(f"{location}: authority flag must remain false")
    records = list(payload.get("verification_records", []))
    record_ids = [str(row.get("id")) for row in records]
    claim_refs = [str(row.get("claim_grounding_ref")) for row in records]
    if len(record_ids) != len(set(record_ids)):
        errors.append("verification record IDs must be unique")
    if len(claim_refs) != len(set(claim_refs)):
        errors.append("each M6A grounding may have only one M6B record")
    for row in records:
        identifier = str(row.get("id"))
        if row.get("quantity_eligible") is not False:
            errors.append(f"{identifier}: quantity_eligible must remain false")
        if _canonical_sha256(row.get("claim_provenance", {})) == _canonical_sha256({}):
            errors.append(f"{identifier}: claim provenance is missing")
        if not isinstance(row.get("claim_grounding_sha256"), str) or len(
            row.get("claim_grounding_sha256", "")
        ) != 64:
            errors.append(f"{identifier}: claim grounding hash missing")
        authority = row.get("authority", {})
        for key in false_authority:
            if authority.get(key) is not False:
                errors.append(f"{identifier}: authority.{key} must be false")
        finding_kind = row.get("finding_kind")
        if finding_kind in {"clearance", "access"}:
            zone = row.get("explicit_zone_geometry", {})
            if (
                not isinstance(zone, Mapping)
                or zone.get("state") != "accepted"
                or not zone.get("frame_ref")
                or not zone.get("geometry")
                or not zone.get("evidence_refs")
            ):
                errors.append(f"{identifier}: clearance/access requires explicit accepted zone geometry")
        elif finding_kind != "clash":
            errors.append(f"{identifier}: unsupported finding kind")
        if row.get("record_type") == "mep_bounded_clash_certificate":
            observation = row.get("geometry_observation", {})
            if (
                row.get("state") != "accepted"
                or len(row.get("bounded_segment_refs", [])) != 2
                or len(set(row.get("bounded_segment_refs", []))) != 2
                or authority.get("bounded_clash_verification_established") is not True
                or row.get("reasons")
                or not observation.get("frame_ref")
            ):
                errors.append(f"{identifier}: accepted bounded certificate has an open gate")
            status = row.get("clash_status")
            confirmed = row.get("confirmed_clash")
            if status == "confirmed_bounded_clash":
                if (
                    confirmed is not True
                    or authority.get("confirmed_bounded_clash_established") is not True
                    or observation.get("minimum_certified_penetration_m", 0) <= 0
                    or observation.get("intersection_away_from_analysis_caps") is not True
                ):
                    errors.append(f"{identifier}: confirmed clash lacks bounded geometric closure")
            elif status == "clear_in_bounded_scope":
                if (
                    confirmed is not False
                    or authority.get("confirmed_bounded_clash_established") is not False
                    or observation.get("minimum_certified_clearance_m", 0) <= 0
                ):
                    errors.append(f"{identifier}: bounded clear result lacks geometric closure")
            else:
                errors.append(f"{identifier}: invalid clash status")
        elif row.get("record_type") == "mep_bounded_clash_abstention":
            if (
                row.get("state") != "abstained"
                or authority.get("bounded_clash_verification_established") is not False
                or not row.get("reasons")
                or "confirmed_clash" in row
                or "clash_status" in row
            ):
                errors.append(f"{identifier}: invalid bounded clash abstention")
        else:
            errors.append(f"{identifier}: unsupported verification record type")
    summary = payload.get("summary", {})
    expected = {
        "claim_count": len(records),
        "confirmed_bounded_clash_count": sum(
            row.get("confirmed_clash") is True for row in records
        ),
        "clear_in_bounded_scope_count": sum(
            row.get("clash_status") == "clear_in_bounded_scope" for row in records
        ),
        "abstention_count": sum(row.get("state") == "abstained" for row in records),
        "whole_run_conclusion_count": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary.{key} must be {value}")
    contract = payload.get("exchange_contract", {})
    for key in (
        "m6a_claim_provenance_preserved",
        "two_independent_m5a_envelopes_required",
        "compatible_registered_metric_frame_required",
        "intersection_away_from_unresolved_analysis_caps_required",
        "measurement_uncertainty_preserved",
        "clearance_and_access_require_explicit_zone_geometry",
        "m5b_whole_run_reasoning_separate",
    ):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in (
        "physical_run_identity_established",
        "physical_continuation_established",
        "whole_run_conclusion_established",
        "installed_length_emitted",
        "calculated_severity_emitted",
        "m7_takeoff_emitted",
        "schedule_values_used",
        "quantity_eligible",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    return errors
