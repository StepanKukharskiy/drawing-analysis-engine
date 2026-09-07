"""Bind resolved Step 3 profiles to certified Step 2 physical scopes.

The binding is deliberately narrower than a solid hypothesis.  It certifies
relative physical-scope membership and exposes an input to Step 5, while
leaving component geometry, placement, axis signs, and quantities unresolved.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping


SCHEMA_VERSION = "0.1.0"
_CLOSURE_FIELDS = (
    "closed",
    "branch_free",
    "unique_completion",
    "scale_bounded",
    "dimensionally_redundant",
)


def _stable_id(kind: str, page_number: int, *parts: str) -> str:
    encoded = "\0".join((str(page_number), *parts)).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _scope_membership_role(scope: Mapping[str, Any], view_ref: str) -> str | None:
    roles = (
        ("section_projection", scope.get("section_view_ids", [])),
        ("parent_projection", scope.get("parent_view_ids", [])),
        ("supporting_projection", scope.get("supporting_projection_view_ids", [])),
        ("member_projection", scope.get("view_ids", [])),
    )
    for role, values in roles:
        if view_ref in set(map(str, values or [])):
            return role
    return None


def _step2_certificate(
    scope: Mapping[str, Any],
    relations_by_id: Mapping[str, Mapping[str, Any]],
    coordinates_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    scope_ref = str(scope.get("id") or "")
    if (
        scope.get("state") not in {"resolved", "resolved_relative"}
        or scope.get("physical_object_identity_state") not in {"resolved", "resolved_relative"}
    ):
        return None, "physical scope is not resolved by Step 2"

    relation_refs = sorted(set(map(str, scope.get("relation_refs", []) or [])))
    relations = [relations_by_id.get(ref) for ref in relation_refs]
    if not relation_refs or any(item is None for item in relations):
        return None, "physical scope does not cite complete Step 2 relations"
    if any(
        item.get("state") != "accepted"
        or item.get("integration_certificate", {}).get("status") != "passed"
        for item in relations
        if item is not None
    ):
        return None, "physical scope relation certificate is not accepted"

    coordinate_ref = str(scope.get("shared_coordinate_scope_id") or "")
    coordinate = coordinates_by_id.get(coordinate_ref)
    if coordinate is None or coordinate.get("state") != "resolved_relative":
        return None, "shared-coordinate scope is not resolved-relative"
    passed_reprojections = [
        item
        for item in coordinate.get("reprojection_validations", []) or []
        if item.get("status") == "pass"
    ]
    declared_reprojection_refs = sorted(
        set(map(str, scope.get("reprojection_validation_refs", []) or []))
    )
    available_reprojection_refs = {
        str(item.get("id") or f"{coordinate_ref}.reprojection.{index}")
        for index, item in enumerate(
            coordinate.get("reprojection_validations", []) or [], start=1
        )
        if item.get("status") == "pass"
    }
    if (
        not passed_reprojections
        or not declared_reprojection_refs
        or not set(declared_reprojection_refs) <= available_reprojection_refs
    ):
        return None, "physical scope does not cite a passed shared-axis reprojection"

    return (
        {
            "status": "passed",
            "physical_scope_ref": scope_ref,
            "physical_scope_state": scope.get("state"),
            "physical_object_identity_state": scope.get(
                "physical_object_identity_state"
            ),
            "relation_refs": relation_refs,
            "shared_coordinate_scope_ref": coordinate_ref,
            "reprojection_validation_refs": declared_reprojection_refs,
        },
        None,
    )


def bind_profiles_to_physical_scopes(
    scoped_profile_assembly: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Certify unique Step 3 profile membership in a Step 2 physical scope."""

    profile_scopes = {
        str(item.get("scope_ref")): item
        for item in scoped_profile_assembly.get("scopes", []) or []
        if item.get("scope_ref") is not None
    }
    scope_count_by_view = Counter(
        str(item.get("view_ref"))
        for item in profile_scopes.values()
        if item.get("view_ref") is not None
    )
    published_scope_refs_by_profile: dict[str, set[str]] = {}
    for scope_ref, scope in profile_scopes.items():
        for item in scope.get("profiles", []) or []:
            if item.get("id") is not None:
                published_scope_refs_by_profile.setdefault(str(item["id"]), set()).add(
                    scope_ref
                )
    relations_by_id = {
        str(item.get("id")): item
        for item in view_frame_graph.get("relations", []) or []
        if item.get("id") is not None
    }
    coordinates_by_id = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get(
            "scopes", []
        )
        or []
        if item.get("id") is not None
    }
    certified_scopes = []
    rejected_scope_reasons: dict[str, str] = {}
    for scope in sorted(
        view_frame_graph.get("object_scopes", []) or [],
        key=lambda item: str(item.get("id")),
    ):
        certificate, reason = _step2_certificate(
            scope, relations_by_id, coordinates_by_id
        )
        if certificate is None:
            rejected_scope_reasons[str(scope.get("id"))] = str(reason)
        else:
            certified_scopes.append((scope, certificate))

    bindings = []
    abstentions = []
    for profile in sorted(
        scoped_profile_assembly.get("profiles", []) or [],
        key=lambda item: str(item.get("id")),
    ):
        profile_ref = str(profile.get("id") or "")
        profile_scope_ref = str(profile.get("scope_ref") or "")
        scoped_result = profile_scopes.get(profile_scope_ref)
        view_ref = str((scoped_result or {}).get("view_ref") or "")
        closure = profile.get("closure", {}) or {}
        reason_code = None
        reason = None
        candidates = []
        if (
            not profile_ref
            or profile.get("state") != "resolved"
            or profile.get("quantity_eligible") is not False
            or not all(closure.get(field) is True for field in _CLOSURE_FIELDS)
        ):
            reason_code = "profile_certificate_unresolved"
            reason = "profile does not preserve every resolved, quantity-ineligible Step 3 gate"
        elif scoped_result is None:
            reason_code = "profile_scope_missing"
            reason = "profile does not resolve to one published Step 3 scope"
        elif published_scope_refs_by_profile.get(profile_ref) != {profile_scope_ref}:
            reason_code = "profile_scope_membership_mismatch"
            reason = "profile is not published exactly once by its claimed Step 3 scope"
        else:
            for physical_scope, certificate in certified_scopes:
                exact_role = _scope_membership_role(physical_scope, profile_scope_ref)
                alias_role = _scope_membership_role(physical_scope, view_ref)
                if exact_role is not None:
                    candidates.append(
                        (physical_scope, certificate, exact_role, "exact_title_scope")
                    )
                elif alias_role is not None and scope_count_by_view[view_ref] == 1:
                    candidates.append(
                        (
                            physical_scope,
                            certificate,
                            alias_role,
                            "unique_unsplit_source_view",
                        )
                    )
        if reason_code is None and len(candidates) != 1:
            reason_code = (
                "ambiguous_physical_scope_membership"
                if len(candidates) > 1
                else "physical_scope_membership_unresolved"
            )
            reason = (
                "profile matches more than one certified Step 2 physical scope"
                if candidates
                else "profile title scope is not a unique member of a certified Step 2 physical scope"
            )

        if reason_code is not None:
            evidence_refs = sorted(
                {
                    profile_ref,
                    profile_scope_ref,
                    *map(str, profile.get("evidence_refs", []) or []),
                }
                - {""}
            )
            abstention = {
                "id": _stable_id(
                    "profile_physical_scope_abstention",
                    page_number,
                    profile_ref,
                    profile_scope_ref,
                    reason_code,
                    *evidence_refs,
                ),
                "record_type": "profile_physical_scope_abstention",
                "record_version": SCHEMA_VERSION,
                "page": page_number,
                "state": "abstained",
                "profile_ref": profile_ref,
                "profile_scope_ref": profile_scope_ref,
                "reason_code": reason_code,
                "reason": reason,
                "candidate_physical_scope_refs": sorted(
                    str(item[0].get("id")) for item in candidates
                ),
                "evidence_refs": evidence_refs,
                "quantity_eligible": False,
            }
            abstentions.append(abstention)
            continue

        physical_scope, certificate, role, match_basis = candidates[0]
        physical_scope_ref = str(physical_scope["id"])
        evidence_refs = sorted(
            {
                profile_ref,
                profile_scope_ref,
                view_ref,
                physical_scope_ref,
                *map(str, profile.get("evidence_refs", []) or []),
                *certificate["relation_refs"],
                certificate["shared_coordinate_scope_ref"],
                *certificate["reprojection_validation_refs"],
            }
            - {""}
        )
        binding = {
            "record_type": "profile_physical_scope_binding",
            "record_version": SCHEMA_VERSION,
            "id": _stable_id(
                "profile_physical_scope_binding",
                page_number,
                profile_ref,
                physical_scope_ref,
                role,
                match_basis,
                *evidence_refs,
            ),
            "page": page_number,
            "state": "accepted",
            "epistemic_state": "derived",
            "profile_ref": profile_ref,
            "profile_scope_ref": profile_scope_ref,
            "source_view_ref": view_ref,
            "physical_scope_ref": physical_scope_ref,
            "physical_scope_membership_state": "resolved_relative",
            "membership_role": role,
            "membership_match_basis": match_basis,
            "step2_certificate": certificate,
            "evidence_refs": evidence_refs,
            "step5_reconstruction_input_eligible": True,
            "step4_kernel_invocation_eligible": False,
            "additive_component_identity_established": False,
            "physical_component_ref": None,
            "component_geometry_established": False,
            "physical_transform_established": False,
            "quantity_eligible": False,
        }
        bindings.append(binding)

    scope_results = []
    for scope_ref, scope in sorted(profile_scopes.items()):
        profile_refs = sorted(
            str(item.get("id"))
            for item in scope.get("profiles", []) or []
            if item.get("id") is not None
        )
        binding_refs = sorted(
            item["id"] for item in bindings if item["profile_ref"] in profile_refs
        )
        abstention_refs = sorted(
            item["id"] for item in abstentions if item["profile_ref"] in profile_refs
        )
        scope_results.append(
            {
                "profile_scope_ref": scope_ref,
                "source_view_ref": scope.get("view_ref"),
                "state": (
                    "bound"
                    if profile_refs and len(binding_refs) == len(profile_refs)
                    else "partial"
                    if binding_refs
                    else "upstream_abstained"
                    if not profile_refs
                    else "abstained"
                ),
                "profile_refs": profile_refs,
                "binding_refs": binding_refs,
                "abstention_refs": abstention_refs,
                "upstream_abstention_refs": sorted(
                    str(item.get("id"))
                    for item in scope.get("abstentions", []) or []
                    if item.get("id") is not None
                ),
            }
        )

    status = (
        "resolved"
        if bindings and not abstentions
        else "resolved_subset"
        if bindings
        else "abstained"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "profile_physical_scope_binding",
        "page": page_number,
        "status": status,
        "bindings": bindings,
        "abstentions": abstentions,
        "scope_results": scope_results,
        "step5_reconstruction_inputs": [
            {
                "binding_ref": item["id"],
                "profile_ref": item["profile_ref"],
                "physical_scope_ref": item["physical_scope_ref"],
                "shared_coordinate_scope_ref": item["step2_certificate"][
                    "shared_coordinate_scope_ref"
                ],
                "membership_role": item["membership_role"],
                "state": "eligible_for_component_reconstruction",
                "additive_component_eligible": False,
                "component_geometry_required": True,
                "quantity_eligible": False,
            }
            for item in bindings
        ],
        "summary": {
            "profile_count": len(scoped_profile_assembly.get("profiles", []) or []),
            "certified_physical_scope_count": len(certified_scopes),
            "binding_count": len(bindings),
            "abstention_count": len(abstentions),
            "step5_reconstruction_input_count": len(bindings),
        },
        "diagnostics": {
            "rejected_physical_scopes": rejected_scope_reasons,
        },
        "contract": {
            "resolved_step3_profile_required": True,
            "certified_step2_physical_scope_required": True,
            "exact_or_unique_unsplit_view_membership_required": True,
            "ambiguous_membership_abstains": True,
            "relative_physical_scope_membership_established": bool(bindings),
            "physical_origin_established": False,
            "absolute_axis_signs_established": False,
            "component_geometry_established": False,
            "additive_component_identity_established": False,
            "step4_kernel_invocation_authorized": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def validate_profile_physical_scope_binding(payload: Mapping[str, Any]) -> list[str]:
    """Validate the binding boundary without promoting it to a solid."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("layer") != "profile_physical_scope_binding":
        errors.append("layer must be profile_physical_scope_binding")
    contract = payload.get("contract", {}) or {}
    for field in (
        "physical_origin_established",
        "absolute_axis_signs_established",
        "component_geometry_established",
        "step4_kernel_invocation_authorized",
        "quantity_eligible",
        "schedule_values_used",
    ):
        if contract.get(field) is not False:
            errors.append(f"contract.{field} must be false")
    ids = []
    pairs = []
    for index, item in enumerate(payload.get("bindings", []) or []):
        prefix = f"bindings[{index}]"
        ids.append(str(item.get("id")))
        pairs.append((str(item.get("profile_ref")), str(item.get("physical_scope_ref"))))
        if item.get("record_type") != "profile_physical_scope_binding":
            errors.append(f"{prefix}.record_type must be profile_physical_scope_binding")
        if item.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version must be {SCHEMA_VERSION}")
        if item.get("state") != "accepted":
            errors.append(f"{prefix}.state must be accepted")
        if item.get("physical_scope_membership_state") != "resolved_relative":
            errors.append(f"{prefix} must preserve resolved-relative membership")
        if item.get("quantity_eligible") is not False:
            errors.append(f"{prefix} must remain quantity-ineligible")
        if item.get("step5_reconstruction_input_eligible") is not True:
            errors.append(f"{prefix} must be eligible only as a Step 5 reconstruction input")
        if item.get("step4_kernel_invocation_eligible") is not False:
            errors.append(f"{prefix} must not directly authorize Step 4 kernel invocation")
        if item.get("additive_component_identity_established") is not False:
            errors.append(f"{prefix} must not establish an additive component")
        if item.get("physical_component_ref") is not None:
            errors.append(f"{prefix}.physical_component_ref must remain unresolved")
        certificate = item.get("step2_certificate", {}) or {}
        if (
            certificate.get("status") != "passed"
            or not certificate.get("relation_refs")
            or not certificate.get("shared_coordinate_scope_ref")
            or not certificate.get("reprojection_validation_refs")
        ):
            errors.append(f"{prefix}.step2_certificate is incomplete")
        if not item.get("evidence_refs"):
            errors.append(f"{prefix}.evidence_refs are required")
    if len(ids) != len(set(ids)):
        errors.append("binding ids must be globally unique")
    if len(pairs) != len(set(pairs)):
        errors.append("profile-to-physical-scope pairs must be unique")
    input_binding_refs = [
        str(item.get("binding_ref"))
        for item in payload.get("step5_reconstruction_inputs", []) or []
    ]
    if sorted(input_binding_refs) != sorted(ids):
        errors.append("step5_reconstruction_inputs must exactly match accepted bindings")
    return errors
