"""Integrate multi-path profile assembly with resolved title-anchored views."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from src.drawing_engine.disciplines.concrete.multi_path_profile_assembly import (
    SCHEMA_VERSION,
    assemble_multi_path_profiles,
    validate_multi_path_profile_assembly,
)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _drawing_ref(value: Any) -> str:
    return str(value).split(".item[", 1)[0]


def _abstention_id(page_number: int, scope_ref: str, item: Mapping[str, Any]) -> str:
    evidence = "\0".join(
        (
            str(page_number),
            scope_ref,
            str(item.get("reason_code")),
            *sorted(map(str, item.get("evidence_refs", []) or [])),
        )
    ).encode("utf-8")
    return f"profile_assembly_abstention.page_{page_number:04d}.evidence_{hashlib.sha256(evidence).hexdigest()[:16]}"


def _owned_dimensions_for_scope(
    scope: Mapping[str, Any],
    resolved_scope_count_by_view: Mapping[str, int],
    dimensions_by_ref: Mapping[str, Any],
    ownership: Mapping[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    scope_ref = str(scope["id"])
    view_ref = str(scope.get("view_id"))
    allowed = set(map(str, scope.get("primitive_refs", []) or []))
    excluded = set(map(str, scope.get("excluded_primitive_refs", []) or []))
    selected = []
    decisions = []
    for item in sorted(
        ownership.get("attachments", []) or [],
        key=lambda row: str(row.get("dimension_ref")),
    ):
        dimension_ref = str(item.get("dimension_ref"))
        dimension = dimensions_by_ref.get(dimension_ref)
        accepted = (
            dimension is not None
            and str(_field(dimension, "status", "")) == "accepted"
            and item.get("status") == "accepted"
        )
        owner_views = set(map(str, item.get("view_refs", []) or []))
        target_drawings = {
            _drawing_ref(ref)
            for ref in item.get("primitive_refs", []) or []
            if str(ref)
        }
        reason = None
        if not accepted:
            reason = "dimension or ownership is not accepted"
        elif owner_views != {view_ref}:
            reason = "accepted dimension is owned by another preliminary view"
        elif target_drawings and (not target_drawings <= allowed or target_drawings & excluded):
            reason = "owned endpoint geometry is outside this resolved scope"
        elif not target_drawings and resolved_scope_count_by_view.get(view_ref, 0) != 1:
            reason = "ownership lacks exact endpoint primitives inside a split preliminary view"
        if reason is None:
            selected.append(dimension)
        decisions.append(
            {
                "dimension_ref": dimension_ref,
                "ownership_ref": item.get("id"),
                "scope_ref": scope_ref,
                "state": "accepted" if reason is None else "excluded",
                "target_drawing_refs": sorted(target_drawings),
                "reason": reason,
            }
        )
    return selected, decisions


def assemble_profiles_by_view_scope(
    topology: Mapping[str, Any],
    dimensions: Iterable[Any],
    dimension_ownership: Mapping[str, Any],
    view_segmentation: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Run Step 3 once per resolved Step 1B scope with local evidence only."""

    dimensions_by_ref = {
        str(_field(item, "attachment_id", _field(item, "id", ""))): item
        for item in dimensions
        if _field(item, "attachment_id", _field(item, "id"))
    }
    resolved_scopes = sorted(
        (
            item
            for item in view_segmentation.get("segments", []) or []
            if item.get("state") == "resolved"
        ),
        key=lambda item: str(item.get("id")),
    )
    scope_count_by_view: dict[str, int] = {}
    for scope in resolved_scopes:
        view_ref = str(scope.get("view_id"))
        scope_count_by_view[view_ref] = scope_count_by_view.get(view_ref, 0) + 1

    scope_results = []
    for scope in resolved_scopes:
        scope_ref = str(scope["id"])
        included = set(map(str, scope.get("primitive_refs", []) or []))
        excluded = set(map(str, scope.get("excluded_primitive_refs", []) or []))
        allowed = sorted(included - excluded)
        owned_dimensions, decisions = _owned_dimensions_for_scope(
            scope,
            scope_count_by_view,
            dimensions_by_ref,
            dimension_ownership,
        )
        assembly = assemble_multi_path_profiles(
            topology,
            owned_dimensions,
            page_number=page_number,
            scope_ref=scope_ref,
            allowed_primitive_refs=allowed,
        )
        validation_errors = validate_multi_path_profile_assembly(assembly)
        abstentions = []
        for item in assembly["abstentions"]:
            row = {**item, "scope_ref": scope_ref, "page": page_number}
            row["id"] = _abstention_id(page_number, scope_ref, row)
            abstentions.append(row)
        if validation_errors:
            row = {
                "scope_ref": scope_ref,
                "page": page_number,
                "reason_code": "invalid_profile_assembly_record",
                "reason": "; ".join(validation_errors),
                "evidence_refs": sorted(allowed),
            }
            row["id"] = _abstention_id(page_number, scope_ref, row)
            abstentions.append(row)
        scope_results.append(
            {
                "scope_ref": scope_ref,
                "view_ref": str(scope.get("view_id")),
                "title_anchor_ref": scope.get("title_anchor_ref"),
                "title": scope.get("title"),
                "state": assembly["status"] if not validation_errors else "invalid",
                "primitive_refs": allowed,
                "excluded_primitive_refs": sorted(excluded),
                "accepted_dimension_refs": sorted(
                    str(_field(item, "attachment_id", _field(item, "id")))
                    for item in owned_dimensions
                ),
                "dimension_scope_decisions": decisions,
                "profiles": assembly["profiles"] if not validation_errors else [],
                "abstentions": abstentions,
                "validation": {
                    "status": "pass" if not validation_errors else "fail",
                    "errors": validation_errors,
                },
            }
        )

    profiles = sorted(
        (
            profile
            for scope in scope_results
            for profile in scope["profiles"]
        ),
        key=lambda item: item["id"],
    )
    abstentions = sorted(
        (
            item
            for scope in scope_results
            for item in scope["abstentions"]
        ),
        key=lambda item: item["id"],
    )
    states = {scope["state"] for scope in scope_results}
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "scoped_multi_path_profile_assembly",
        "page": page_number,
        "status": (
            "resolved"
            if scope_results and states == {"resolved"}
            else "partial"
            if profiles
            else "abstained"
        ),
        "scopes": scope_results,
        "profiles": profiles,
        "abstentions": abstentions,
        "summary": {
            "resolved_scope_count": len(resolved_scopes),
            "scope_with_profile_count": sum(bool(scope["profiles"]) for scope in scope_results),
            "profile_count": len(profiles),
            "abstention_count": len(abstentions),
        },
        "contract": {
            "resolved_title_scope_required": True,
            "scope_primitive_membership_is_authoritative": True,
            "excluded_primitives_never_enter_bridge_search": True,
            "accepted_same_scope_dimension_ownership_required": True,
            "profile_ids_are_page_scope_and_evidence_namespaced": True,
            "physical_object_identity_established": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
