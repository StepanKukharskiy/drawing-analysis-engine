"""Rerun generic profile assembly from title-scope-local metric evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Iterable, Mapping

from src.drawing_engine.disciplines.concrete.multi_path_profile_assembly import (
    assemble_multi_path_profiles,
    validate_multi_path_profile_assembly,
)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _scope_local_accepted_copy(item: Any) -> Any:
    if isinstance(item, Mapping):
        return {**item, "status": "accepted"}
    return replace(item, status="accepted")


def reclose_profiles_in_title_scopes(
    topology: Mapping[str, Any],
    dimensions: Iterable[Any],
    title_segmentation: Mapping[str, Any],
    local_dimension_reclosure: Mapping[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    """Replay the existing assembler only in locally reclosed title scopes.

    The output deliberately does not classify or select flight profiles. A
    two-profile candidate set is available only when the unchanged generic
    assembler itself returns exactly two profiles in the target scope. This
    stage never emits the pair certificate because the required semantic
    boundary and interface checks are intentionally downstream.
    """

    dimensions_by_ref = {
        str(_field(item, "attachment_id", _field(item, "id", ""))): item
        for item in dimensions
        if _field(item, "attachment_id", _field(item, "id"))
    }
    scopes = {
        str(item.get("id")): item
        for item in title_segmentation.get("segments", []) or []
        if item.get("state") == "resolved"
    }
    results = []
    for local in local_dimension_reclosure.get("scope_results", []) or []:
        scope_ref = str(local.get("scope_ref"))
        scope = scopes.get(scope_ref)
        accepted_refs = list(map(str, local.get("accepted_dimension_refs", []) or []))
        selected_dimensions = [
            _scope_local_accepted_copy(dimensions_by_ref[ref])
            for ref in accepted_refs
            if ref in dimensions_by_ref
        ]
        if scope is None or local.get("status") != "resolved_subset":
            results.append(
                {
                    "scope_ref": scope_ref,
                    "status": "insufficient_constraints",
                    "reason_code": "title_scope_metric_reclosure_unresolved",
                    "accepted_dimension_refs": accepted_refs,
                    "assembly": None,
                    "pair_certificate": None,
                    "quantity_eligible": False,
                }
            )
            continue
        allowed = sorted(
            set(map(str, scope.get("primitive_refs", []) or []))
            - set(map(str, scope.get("excluded_primitive_refs", []) or []))
        )
        assembly = assemble_multi_path_profiles(
            topology,
            selected_dimensions,
            page_number=page_number,
            scope_ref=scope_ref,
            allowed_primitive_refs=allowed,
        )
        validation_errors = validate_multi_path_profile_assembly(assembly)
        profiles = list(assembly.get("profiles", []) or []) if not validation_errors else []
        profile_count = len(profiles)
        two_profile_candidate_set = profile_count == 2
        results.append(
            {
                "scope_ref": scope_ref,
                "status": (
                    "two_profile_candidate_set_available"
                    if two_profile_candidate_set
                    else "pair_unresolved"
                ),
                "reason_code": (
                    None
                    if two_profile_candidate_set
                    else "generic_assembler_did_not_return_unique_two_profile_pair"
                ),
                "accepted_dimension_refs": accepted_refs,
                "independent_metric_check_count": int(
                    local.get("independent_metric_check_count", 0)
                ),
                "assembly": assembly,
                "validation": {
                    "status": "pass" if not validation_errors else "fail",
                    "errors": validation_errors,
                },
                "profile_candidate_count": profile_count,
                "profile_vertex_count_histogram": {
                    str(count): frequency
                    for count, frequency in sorted(
                        Counter(
                            len(item.get("ordered_boundary_display", []) or [])
                            for item in profiles
                        ).items()
                    )
                },
                "pair_selection_assessment": {
                    "two_profile_candidate_refs": (
                        [str(item.get("id")) for item in profiles]
                        if two_profile_candidate_set
                        else []
                    ),
                    "required_validation": [
                        "lower_flight_stepped_surface_and_waist_underside",
                        "upper_flight_stepped_surface_and_waist_underside",
                        "rise_run_agreement_with_local_158_and_275_evidence",
                        "approximately_175_mm_waist_thickness",
                        "vertical_extent_agreement_with_1575_1425_and_3150",
                        "unique_landing_end_interfaces",
                        "reinforcement_paths_excluded_from_concrete_boundaries",
                    ],
                    "validation_status": "not_run",
                },
                "pair_certificate": None,
                "profile_selector_used": False,
                "physical_component_identity_established": False,
                "quantity_eligible": False,
            }
        )

    pair_count = sum(
        item.get("status") == "two_profile_candidate_set_available" for item in results
    )
    return {
        "schema_version": "0.1.0",
        "layer": "scope_local_profile_reclosure",
        "page": page_number,
        "status": "two_profile_candidate_set_available" if pair_count else "pair_unresolved",
        "scope_results": results,
        "summary": {
            "evaluated_scope_count": len(results),
            "candidate_pair_scope_count": pair_count,
            "profile_candidate_count": sum(
                int(item.get("profile_candidate_count", 0)) for item in results
            ),
        },
        "contract": {
            "existing_generic_assembler_reused": True,
            "target_scopes_come_from_physical_section_relations": True,
            "profile_selector_used": False,
            "concrete_boundary_classification_established": False,
            "reinforcement_path_exclusion_required_before_pair_acceptance": True,
            "landing_folded_into_flight": False,
            "physical_component_identity_established": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
