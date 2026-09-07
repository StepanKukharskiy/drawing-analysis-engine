"""Stable Step 0 exchange records for geometry candidates and transforms."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"


def _refs(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value is not None and str(value)})


def build_exchange_records(
    dimension_proposals: Mapping[str, Any],
    view_segmentation: Mapping[str, Any],
    contours: Iterable[Mapping[str, Any]],
    shared_coordinate_system: Mapping[str, Any],
    *,
    profile_assembly: Mapping[str, Any] | None = None,
    profile_scope_binding: Mapping[str, Any] | None = None,
    physical_component_hypothesis: Mapping[str, Any] | None = None,
    physical_component_reclosure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project active graph layers into versioned, future-lane-safe records."""

    dimension_chains = []
    for item in dimension_proposals.get("attachments", []) or []:
        dimension_chains.append(
            {
                "record_type": "dimension_chain_candidate",
                "record_version": SCHEMA_VERSION,
                "id": str(item["id"]),
                "state": (
                    "detector_qualified"
                    if item.get("proposal_status") == "accepted"
                    else "candidate"
                ),
                "value_mm": item.get("value_mm"),
                "orientation": item.get("orientation"),
                "terminal_style": item.get("terminal_style", "terminal_less"),
                "bbox_display": list(item.get("text_bbox_display", [])),
                "dimension_points_display": list(item.get("dimension_points_display", [])),
                "measured_points_display": list(item.get("measured_points_display", [])),
                "terminal_refs_by_endpoint": [
                    _refs(refs) for refs in item.get("terminal_refs_by_endpoint", [[], []])
                ],
                "primitive_refs": _refs(item.get("primitive_refs", [])),
                "evidence_refs": _refs([item.get("id"), *item.get("primitive_refs", [])]),
            }
        )

    view_scopes = []
    for item in view_segmentation.get("segments", []) or []:
        view_scopes.append(
            {
                "record_type": "view_scope_candidate",
                "record_version": SCHEMA_VERSION,
                "id": str(item["id"]),
                "state": str(item.get("state", "unresolved")),
                "view_ref": str(item.get("view_id")),
                "source_view_ref": str(item.get("source_view_id") or item.get("view_id")),
                "title_anchor_ref": item.get("title_anchor_ref"),
                "title": item.get("title"),
                "normalised_title": item.get("normalised_title"),
                "scale_text_ref": item.get("scale_text_ref"),
                "scale_ratio": item.get("scale_ratio"),
                "bbox_display": list(item.get("bbox_display", [])),
                "primitive_refs": _refs(item.get("primitive_refs", [])),
                "excluded_primitive_refs": _refs(item.get("excluded_primitive_refs", [])),
                "evidence_refs": _refs(item.get("evidence_refs", [])),
            }
        )

    raw_contours = []
    for item in contours:
        topology = item.get("topology", {}) or {}
        closed = bool(topology.get("closed")) or topology.get("class") in {
            "closed_polygon",
            "closed_polyline",
        }
        raw_contours.append(
            {
                "record_type": "raw_contour_candidate",
                "record_version": SCHEMA_VERSION,
                "id": f"raw_contour_candidate.{len(raw_contours) + 1:04d}",
                "state": "observed_closed" if closed else "observed_open",
                "source_contour_ref": str(item["id"]),
                "bbox_display": list(item.get("bbox_display", [])),
                "primitive_refs": _refs(item.get("primitive_refs", [])),
                "observed_closed": closed,
                "evidence_refs": _refs([item.get("id"), *item.get("primitive_refs", [])]),
            }
        )
    assembled_profiles = [
        dict(item)
        for item in (profile_assembly or {}).get("profiles", []) or []
    ]
    profile_scope_bindings = [
        dict(item)
        for item in (profile_scope_binding or {}).get("bindings", []) or []
    ]
    physical_component_hypotheses = [
        dict(item)
        for item in (physical_component_hypothesis or {}).get("hypotheses", []) or []
    ]
    physical_component_identity_placement_certificates = [
        dict(item)
        for item in (physical_component_reclosure or {}).get("certificates", []) or []
    ]

    component_transforms = []
    for scope in shared_coordinate_system.get("scopes", []) or []:
        mappings = scope.get("view_mappings", {}) or {}
        for view_ref, mapping in sorted(mappings.items()):
            component_transforms.append(
                {
                    "record_type": "physical_component_transform",
                    "record_version": SCHEMA_VERSION,
                    "id": f"physical_component_transform.{len(component_transforms) + 1:04d}",
                    "state": str(scope.get("state", "unresolved")),
                    "component_ref": None,
                    "view_ref": str(view_ref),
                    "scope_ref": str(scope.get("id")),
                    "axis_mapping": {
                        "u": mapping.get("u"),
                        "v": mapping.get("v"),
                        "normal": mapping.get("normal"),
                    },
                    "origin_xyz_mm": None,
                    "axis_signs": "unresolved",
                    "evidence_refs": _refs(scope.get("evidence_refs", [])),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "step0_exchange_records",
        "dimension_chain_candidates": dimension_chains,
        "view_scope_candidates": view_scopes,
        "raw_contour_candidates": raw_contours,
        "assembled_profile_candidates": assembled_profiles,
        "profile_physical_scope_bindings": profile_scope_bindings,
        "physical_component_hypotheses": physical_component_hypotheses,
        "physical_component_identity_placement_certificates": physical_component_identity_placement_certificates,
        "physical_component_transforms": component_transforms,
        "contract": {
            "stable_record_type_and_version_required": True,
            "source_primitive_ids_preserved": True,
            "unknown_transform_origins_and_signs_remain_explicit": True,
            "profile_scope_bindings_are_reconstruction_inputs_only": True,
            "component_hypotheses_are_quantity_free_alternatives_only": True,
            "component_identity_placement_certificates_do_not_authorize_step4": True,
            "schedule_values_used": False,
        },
    }


def validate_exchange_records(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    expected = {
        "dimension_chain_candidates": "dimension_chain_candidate",
        "view_scope_candidates": "view_scope_candidate",
        "raw_contour_candidates": "raw_contour_candidate",
        "assembled_profile_candidates": "assembled_profile_candidate",
        "profile_physical_scope_bindings": "profile_physical_scope_binding",
        "physical_component_hypotheses": "physical_component_hypothesis",
        "physical_component_identity_placement_certificates": "physical_component_identity_placement_certificate",
        "physical_component_transforms": "physical_component_transform",
    }
    for collection, record_type in expected.items():
        if not isinstance(payload.get(collection), list):
            errors.append(f"{collection} must be a list")
            continue
        for index, item in enumerate(payload[collection]):
            prefix = f"{collection}[{index}]"
            if item.get("record_type") != record_type:
                errors.append(f"{prefix}.record_type must be {record_type}")
            if item.get("record_version") != SCHEMA_VERSION:
                errors.append(f"{prefix}.record_version must be {SCHEMA_VERSION}")
            if not item.get("id"):
                errors.append(f"{prefix}.id is required")
            if not isinstance(item.get("evidence_refs"), list):
                errors.append(f"{prefix}.evidence_refs must be a list")
            if record_type == "view_scope_candidate":
                primitive_refs = item.get("primitive_refs")
                excluded_refs = item.get("excluded_primitive_refs")
                if not isinstance(primitive_refs, list):
                    errors.append(f"{prefix}.primitive_refs must be a list")
                if not isinstance(excluded_refs, list):
                    errors.append(f"{prefix}.excluded_primitive_refs must be a list")
                if item.get("state") == "resolved" and not item.get("title_anchor_ref"):
                    errors.append(f"{prefix}.resolved scope requires title_anchor_ref")
                if item.get("state") == "resolved" and not primitive_refs:
                    errors.append(f"{prefix}.resolved scope requires native primitive membership")
                if isinstance(primitive_refs, list) and isinstance(excluded_refs, list):
                    overlap = set(map(str, primitive_refs)) & set(map(str, excluded_refs))
                    if overlap:
                        errors.append(f"{prefix}.included and excluded primitive refs overlap")
            if record_type == "assembled_profile_candidate" and item.get("state") == "resolved":
                closure = item.get("closure") or {}
                if item.get("quantity_eligible") is not False:
                    errors.append(f"{prefix}.resolved profile must remain quantity-ineligible")
                if not item.get("scope_ref"):
                    errors.append(f"{prefix}.resolved profile requires scope_ref")
                if not item.get("source_edge_refs"):
                    errors.append(f"{prefix}.resolved profile requires source_edge_refs")
                for field in (
                    "closed",
                    "branch_free",
                    "unique_completion",
                    "scale_bounded",
                    "dimensionally_redundant",
                ):
                    if closure.get(field) is not True:
                        errors.append(f"{prefix}.closure.{field} must be true")
            if record_type == "profile_physical_scope_binding":
                if item.get("state") != "accepted":
                    errors.append(f"{prefix}.state must be accepted")
                if item.get("quantity_eligible") is not False:
                    errors.append(f"{prefix} must remain quantity-ineligible")
                if not item.get("profile_ref") or not item.get("physical_scope_ref"):
                    errors.append(f"{prefix} requires profile and physical scope refs")
                if item.get("step5_reconstruction_input_eligible") is not True:
                    errors.append(f"{prefix} must be a Step 5 reconstruction input")
                if item.get("step4_kernel_invocation_eligible") is not False:
                    errors.append(f"{prefix} cannot directly authorize Step 4")
                if item.get("additive_component_identity_established") is not False:
                    errors.append(f"{prefix} cannot establish an additive component")
            if record_type == "physical_component_hypothesis":
                if item.get("state") != "candidate":
                    errors.append(f"{prefix}.state must remain candidate")
                if not item.get("profile_refs") or not item.get("binding_refs"):
                    errors.append(f"{prefix} requires profile and binding refs")
                if item.get("additive_component_identity_established") is not False:
                    errors.append(f"{prefix} cannot establish an additive component")
                if item.get("physical_component_ref") is not None:
                    errors.append(f"{prefix}.physical_component_ref must remain unresolved")
                if item.get("physical_transform_refs") != []:
                    errors.append(f"{prefix}.physical_transform_refs must remain empty")
                if item.get("step4_kernel_invocation_eligible") is not False:
                    errors.append(f"{prefix} cannot authorize Step 4")
                if item.get("quantity_eligible") is not False:
                    errors.append(f"{prefix} must remain quantity-ineligible")
            if record_type == "physical_component_identity_placement_certificate":
                if item.get("status") not in {
                    "reclosed_pass",
                    "reclosed_fail",
                    "insufficient_constraints",
                }:
                    errors.append(f"{prefix}.status is unsupported")
                if item.get("slice3_input_eligible") is not (
                    item.get("status") == "reclosed_pass"
                ):
                    errors.append(f"{prefix}.slice3_input_eligible must match status")
                if item.get("physical_component_ref") is not None:
                    errors.append(f"{prefix}.physical_component_ref must remain unresolved")
                if item.get("physical_transform_ref") is not None:
                    errors.append(f"{prefix}.physical_transform_ref must remain unresolved")
                if item.get("step4_kernel_invocation_eligible") is not False:
                    errors.append(f"{prefix} cannot authorize Step 4")
                if item.get("quantity_eligible") is not False:
                    errors.append(f"{prefix} must remain quantity-ineligible")

    memberships: defaultdict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(payload.get("view_scope_candidates", []) or []):
        if item.get("state") != "resolved":
            continue
        source = str(item.get("source_view_ref") or item.get("view_ref"))
        refs = set(map(str, item.get("primitive_refs", []) or []))
        if refs & memberships[source]:
            errors.append(
                f"view_scope_candidates[{index}].primitive_refs overlap another resolved scope from {source}"
            )
        memberships[source].update(refs)
    profile_ids = [str(item.get("id")) for item in payload.get("assembled_profile_candidates", []) or []]
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("assembled_profile_candidates ids must be globally unique")
    binding_ids = [str(item.get("id")) for item in payload.get("profile_physical_scope_bindings", []) or []]
    if len(binding_ids) != len(set(binding_ids)):
        errors.append("profile_physical_scope_bindings ids must be globally unique")
    hypothesis_ids = [str(item.get("id")) for item in payload.get("physical_component_hypotheses", []) or []]
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        errors.append("physical_component_hypotheses ids must be globally unique")
    certificate_ids = [
        str(item.get("id"))
        for item in payload.get(
            "physical_component_identity_placement_certificates", []
        )
        or []
    ]
    if len(certificate_ids) != len(set(certificate_ids)):
        errors.append(
            "physical_component_identity_placement_certificates ids must be globally unique"
        )
    return errors
