"""Certify a narrow sign-invariant, plan-bounded orthogonal sweep.

This certificate does not resolve a cutting-plane normal.  It permits a local
canonical preview only when two mirrored section transforms act on one
dimension-bounded rectangular prism and therefore preserve its solid,
interfaces, orthographic extents, and analytic volume.
"""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), *(str(part) for part in parts))).encode("utf-8")
    return f"unsigned_bounded_sweep_certificate.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def certify_unsigned_bounded_sweep(
    *,
    page_number: int,
    hypothesis_ref: str,
    physical_scope_ref: str,
    orientation_certificate: Mapping[str, Any],
    signed_transform: Mapping[str, Any],
    extents_by_axis_mm: Mapping[str, float],
    plan_bounded_axes: Iterable[str],
    dimension_refs_by_axis: Mapping[str, Iterable[str]],
    external_interface_count: int,
    evidence_refs: Iterable[str],
    interface_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Return a certificate only for the explicitly proven invariant subset."""

    if orientation_certificate.get("status") in {"accepted", "contradiction"}:
        return None
    axes = set(map(str, extents_by_axis_mm))
    bounded_axes = set(map(str, plan_bounded_axes))
    dimension_refs = {
        str(axis): sorted({str(ref) for ref in refs if str(ref)})
        for axis, refs in dimension_refs_by_axis.items()
    }
    interfaces = [dict(item) for item in interface_records]
    candidates = signed_transform.get("candidates", []) or []
    signs = {
        item.get("sign")
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("sign") in {-1, 1}
        and isinstance(item.get("offset_mm"), (int, float))
        and not isinstance(item.get("offset_mm"), bool)
        and math.isfinite(float(item["offset_mm"]))
    }
    if (
        len(axes) != 3
        or len(bounded_axes) not in {1, 2}
        or not bounded_axes < axes
        or any(not _positive_finite(value) for value in extents_by_axis_mm.values())
        or set(dimension_refs) != axes
        or any(not dimension_refs[axis] for axis in axes)
        or signed_transform.get("state") != "unresolved"
        or len(candidates) != 2
        or signs != {-1, 1}
        or external_interface_count != len(interfaces)
    ):
        return None
    if interfaces:
        positions = {str(item.get("position")) for item in interfaces}
        face_extents = {
            tuple(sorted(float(value) for value in item.get("face_extents_mm", []) or []))
            for item in interfaces
        }
        if (
            len(interfaces) != 2
            or positions != {"minimum", "maximum"}
            or len(face_extents) != 1
            or any(item.get("termination") != "clear_span_inner_face" for item in interfaces)
        ):
            return None

    ordered_extents = {axis: float(extents_by_axis_mm[axis]) for axis in sorted(axes)}
    volume_mm3 = math.prod(ordered_extents.values())
    refs = sorted(
        {
            str(orientation_certificate.get("id") or ""),
            hypothesis_ref,
            physical_scope_ref,
            *(str(ref) for ref in evidence_refs if str(ref)),
            *(ref for refs in dimension_refs.values() for ref in refs),
        }
        - {""}
    )
    certificate_id = _stable_id(
        page_number,
        hypothesis_ref,
        physical_scope_ref,
        *ordered_extents.items(),
        *refs,
    )
    return {
        "id": certificate_id,
        "record_type": "unsigned_bounded_sweep_certificate",
        "record_version": SCHEMA_VERSION,
        "status": "accepted",
        "state": "derived",
        "hypothesis_ref": hypothesis_ref,
        "physical_scope_ref": physical_scope_ref,
        "orientation_sensitivity": "sign_invariant",
        "signed_orientation_state": "unresolved",
        "signed_orientation_certificate_ref": orientation_certificate.get("id"),
        "extents_by_axis_mm": ordered_extents,
        "plan_bounded_axes": sorted(bounded_axes),
        "dimension_refs_by_axis": {
            axis: dimension_refs[axis] for axis in sorted(dimension_refs)
        },
        "signed_transform_alternatives": [
            {"sign": int(item["sign"]), "offset_mm": float(item["offset_mm"])}
            for item in sorted(candidates, key=lambda row: int(row["sign"]), reverse=True)
        ],
        "invariance_checks": {
            "congruent_solid_extents": "pass",
            "identical_interface_topology": "pass",
            "identical_orthographic_extents": "pass",
            "identical_analytic_volume": "pass",
        },
        "external_interface_count": external_interface_count,
        "interface_records": interfaces,
        "invariant_volume_mm3": volume_mm3,
        "canonical_preview_authorized": True,
        "signed_physical_placement_resolved": False,
        "quantity_eligible": False,
        "evidence_refs": refs,
    }


def validate_unsigned_bounded_sweep_certificate(record: Mapping[str, Any]) -> list[str]:
    """Validate the publication boundary before Slice 2 consumes it."""

    errors = []
    if record.get("record_type") != "unsigned_bounded_sweep_certificate":
        errors.append("record_type must be unsigned_bounded_sweep_certificate")
    if record.get("record_version") != SCHEMA_VERSION:
        errors.append(f"record_version must be {SCHEMA_VERSION}")
    if record.get("status") != "accepted" or record.get("orientation_sensitivity") != "sign_invariant":
        errors.append("certificate must be accepted and sign_invariant")
    if record.get("signed_orientation_state") != "unresolved":
        errors.append("signed orientation must remain unresolved")
    alternatives = record.get("signed_transform_alternatives", []) or []
    if len(alternatives) != 2 or {item.get("sign") for item in alternatives} != {-1, 1}:
        errors.append("exactly two mirrored signed alternatives are required")
    extents = record.get("extents_by_axis_mm", {}) or {}
    bounded = set(map(str, record.get("plan_bounded_axes", []) or []))
    refs = record.get("dimension_refs_by_axis", {}) or {}
    if len(extents) != 3 or len(bounded) not in {1, 2} or not bounded < set(extents):
        errors.append("one or two of three positive extents must be plan bounded")
    if any(not _positive_finite(value) for value in extents.values()):
        errors.append("all extents must be positive finite values")
    if set(refs) != set(extents) or any(not refs[axis] for axis in refs):
        errors.append("every extent requires dimension evidence")
    checks = record.get("invariance_checks", {}) or {}
    if set(checks.values()) != {"pass"} or len(checks) != 4:
        errors.append("all four invariance checks must pass")
    interfaces = record.get("interface_records", []) or []
    if record.get("external_interface_count") != len(interfaces):
        errors.append("external interface count must match interface records")
    if interfaces and (
        len(interfaces) != 2
        or {item.get("position") for item in interfaces} != {"minimum", "maximum"}
        or len(
            {
                tuple(sorted(float(value) for value in item.get("face_extents_mm", []) or []))
                for item in interfaces
            }
        )
        != 1
        or any(item.get("termination") != "clear_span_inner_face" for item in interfaces)
    ):
        errors.append("external interfaces must be a reflection-invariant clear-span endpoint pair")
    if record.get("canonical_preview_authorized") is not True:
        errors.append("canonical preview must be explicitly authorized")
    if record.get("signed_physical_placement_resolved") is not False:
        errors.append("signed physical placement must remain unresolved")
    if record.get("quantity_eligible") is not False:
        errors.append("certificate cannot directly authorize a quantity")
    return errors
