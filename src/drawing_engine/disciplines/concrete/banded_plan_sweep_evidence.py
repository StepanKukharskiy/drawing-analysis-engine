"""Resolve repeated, disjoint plan sweep bands without choosing a view sign."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping


SCHEMA_VERSION = "0.1.0"


def _stable_id(page_number: int, *parts: Any) -> str:
    encoded = "\0".join((str(page_number), *(str(part) for part in parts))).encode("utf-8")
    return f"banded_plan_sweep_evidence.page_{page_number:04d}.evidence_{sha256(encoded).hexdigest()[:16]}"


def _abstain(page_number: int, reason_code: str, evidence_refs: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "banded_plan_sweep_evidence",
        "page": page_number,
        "status": "insufficient_constraints",
        "reason_code": reason_code,
        "plan_band_certificate": None,
        "search_physical_object_scope": None,
        "profile_split_assessment": None,
        "evidence_refs": sorted(set(evidence_refs or [])),
        "contract": {
            "accepted_arithmetic_chain_required": True,
            "two_disjoint_equal_sweep_bands_required": True,
            "signed_orientation_resolved": False,
            "physical_component_identity_established": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def derive_banded_plan_sweep_evidence(
    dimension_ownership: Mapping[str, Any],
    dimension_adjudication: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    scoped_profile_assembly: Mapping[str, Any],
    *,
    page_number: int,
    scope_local_profile_reclosure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close a five-term plan chain into two equal bands and one clear gap.

    This stage deliberately does not infer what the swept components are.  It
    proves only their two disjoint plan intervals and reports whether a paired
    section-profile split is already available to a later reconstruction.
    """

    scopes = view_frame_graph.get("object_scopes", []) or []
    relations = {
        str(item.get("id")): item for item in view_frame_graph.get("relations", []) or []
    }
    adjudication_by_ref = {
        str(item.get("dimension_ref")): item
        for item in dimension_adjudication.get("records", []) or []
    }
    ownership_by_ref = {
        str(item.get("dimension_ref")): item
        for item in dimension_ownership.get("attachments", []) or []
    }
    search_scope = None
    relation: Mapping[str, Any] = {}
    if len(scopes) == 1:
        scope = scopes[0]
        cuts = [
            relations[ref]
            for ref in map(str, scope.get("relation_refs", []) or [])
            if ref in relations
            and relations[ref].get("type") == "cut_at"
            and relations[ref].get("state") == "accepted"
        ]
        if len(cuts) != 1:
            return _abstain(page_number, "unique_parent_section_relation_unresolved")
        relation = cuts[0]
        parent_view = str(relation.get("parent_view_id") or "")
        section_view = str(relation.get("section_view_id") or "")
    else:
        local_sections = [
            item
            for item in (scope_local_profile_reclosure or {}).get("scope_results", []) or []
            if int(item.get("independent_metric_check_count") or 0) >= 2
        ]
        plan_contexts = set()
        for certificate in dimension_adjudication.get(
            "terminal_less_redundancy_certificates", []
        ) or []:
            terms = list(map(float, certificate.get("term_values_mm", []) or []))
            refs = [
                str(certificate.get("overall_dimension_ref") or ""),
                *map(str, certificate.get("term_dimension_refs", []) or []),
            ]
            records = [adjudication_by_ref.get(ref) for ref in refs]
            views = {
                str(view)
                for record in records
                if record is not None
                for view in record.get("view_refs", []) or []
            }
            if (
                certificate.get("kind") == "arithmetic_chain"
                and len(terms) == 5
                and len(records) == 6
                and all(record is not None and record.get("status") == "accepted" for record in records)
                and len(views) == 1
            ):
                plan_contexts.add(next(iter(views)))
        if len(local_sections) != 1 or len(plan_contexts) != 1:
            return _abstain(page_number, "unique_physical_scope_unresolved")
        parent_view = next(iter(plan_contexts))
        section_view = str(local_sections[0].get("scope_ref") or "")
        scope_id = (
            f"physical_object_scope.search.page_{page_number:04d}."
            f"evidence_{sha256((parent_view + '\0' + section_view).encode('utf-8')).hexdigest()[:16]}"
        )
        scope = {
            "id": scope_id,
            "record_type": "physical_object_scope",
            "state": "resolved_search_hypothesis",
            "parent_view_id": parent_view,
            "section_view_id": section_view,
            "accepted_cross_view_identity": False,
            "quantity_eligible": False,
            "evidence_refs": [parent_view, section_view],
        }
        search_scope = scope
    candidates = []
    seen_overall_refs: set[str] = set()
    for certificate in dimension_adjudication.get(
        "terminal_less_redundancy_certificates", []
    ) or []:
        terms = list(map(float, certificate.get("term_values_mm", []) or []))
        term_refs = list(map(str, certificate.get("term_dimension_refs", []) or []))
        overall_ref = str(certificate.get("overall_dimension_ref") or "")
        if overall_ref in seen_overall_refs:
            continue
        if overall_ref:
            seen_overall_refs.add(overall_ref)
        refs = [overall_ref, *term_refs]
        records = [adjudication_by_ref.get(ref) for ref in refs]
        owners = [ownership_by_ref.get(ref) for ref in term_refs]
        if (
            certificate.get("kind") != "arithmetic_chain"
            or len(terms) != 5
            or len(term_refs) != 5
            or not overall_ref
            or any(not math.isfinite(value) or value <= 0 for value in terms)
            or abs(terms[1] - terms[3]) > max(2.0, 0.01 * max(terms[1], terms[3]))
            or abs(sum(terms) - float(certificate.get("total_value_mm") or 0.0)) > 1e-6
            or abs(float(certificate.get("arithmetic_residual_mm") or 0.0)) > 1e-6
            or any(record is None or record.get("status") != "accepted" for record in records)
            or any(set(map(str, record.get("view_refs", []) or [])) != {parent_view} for record in records)
            or any(owner is None or owner.get("status") != "accepted" for owner in owners)
            or any(len(owner.get("geometry_anchor_refs", []) or []) != 2 for owner in owners)
        ):
            continue
        candidates.append((certificate, terms, term_refs, owners))
    if len(candidates) != 1:
        return _abstain(page_number, "unique_five_term_equal_band_chain_unresolved")

    certificate, terms, term_refs, owners = candidates[0]
    offsets = [0.0]
    for value in terms:
        offsets.append(offsets[-1] + value)
    band_indices = (1, 3)
    bands = []
    for ordinal, term_index in enumerate(band_indices, start=1):
        owner = owners[term_index]
        bands.append(
            {
                "id": _stable_id(page_number, certificate["overall_dimension_ref"], term_refs[term_index]),
                "ordinal": ordinal,
                "state": "resolved_relative_unsigned",
                "interval_mm": [offsets[term_index], offsets[term_index + 1]],
                "sweep_width_mm": terms[term_index],
                "dimension_ref": term_refs[term_index],
                "endpoint_refs": list(map(str, owner.get("geometry_anchor_refs", []) or [])),
                "physical_side_assignment": "unresolved_reflection",
                "evidence_refs": [
                    term_refs[term_index],
                    *map(str, owner.get("geometry_anchor_refs", []) or []),
                ],
            }
        )

    coordinate_ref = str(scope.get("shared_coordinate_scope_id") or "")
    coordinates = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
    }
    orientation = (coordinates.get(coordinate_ref, {}).get("signed_orientation_certificate") or {})
    section_scope = next(
        (
            item
            for item in scoped_profile_assembly.get("scopes", []) or []
            if str(item.get("scope_ref")) == section_view
        ),
        None,
    )
    section_profiles = list((section_scope or {}).get("profiles", []) or [])
    section_abstentions = list((section_scope or {}).get("abstentions", []) or [])
    local_profile_scope = next(
        (
            item
            for item in (scope_local_profile_reclosure or {}).get("scope_results", []) or []
            if str(item.get("scope_ref")) == section_view
        ),
        None,
    )
    if local_profile_scope is not None:
        local_assembly = local_profile_scope.get("assembly") or {}
        section_profiles = list(local_assembly.get("profiles", []) or [])
        section_abstentions = list(local_assembly.get("abstentions", []) or [])
    profile_split_resolved = len(section_profiles) == 2
    profile_assessment = {
        "state": "candidate_pair_available" if profile_split_resolved else "unresolved",
        "section_scope_ref": section_view,
        "profile_evidence_basis": (
            "title_scope_local_metric_reclosure"
            if local_profile_scope is not None
            else "global_dimension_ownership"
        ),
        "scope_local_profile_reclosure_status": (
            local_profile_scope.get("status") if local_profile_scope is not None else None
        ),
        "resolved_profile_refs": [str(item.get("id")) for item in section_profiles],
        "resolved_profile_count": len(section_profiles),
        "abstention_reason_codes": sorted(
            {str(item.get("reason_code")) for item in section_abstentions if item.get("reason_code")}
        ),
        "required_next_certificate": (
            "unique_profile_to_band_pairing_and_landing_footprint"
            if profile_split_resolved
            else "unique_two_profile_section_split_and_landing_footprint"
        ),
        "plan_sweep_width_is_resolved_independently": True,
        "evidence_refs": [
            section_view,
            *[str(item.get("id")) for item in section_profiles],
            *[str(item.get("id")) for item in section_abstentions if item.get("id")],
        ],
    }

    evidence_refs = sorted(
        {
            str(scope.get("id") or ""),
            str(relation.get("id") or ""),
            coordinate_ref,
            str(orientation.get("id") or ""),
            str(certificate.get("overall_dimension_ref") or ""),
            *term_refs,
            *[ref for band in bands for ref in band["endpoint_refs"]],
            section_view,
        }
        - {""}
    )
    record_id = _stable_id(page_number, *evidence_refs)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "banded_plan_sweep_evidence",
        "page": page_number,
        "status": "plan_bands_resolved_profile_split_pending",
        "reason_code": "section_profile_split_unresolved" if not profile_split_resolved else None,
        "search_physical_object_scope": search_scope,
        "plan_band_certificate": {
            "id": record_id,
            "state": "resolved_relative_unsigned",
            "overall_dimension_ref": certificate["overall_dimension_ref"],
            "overall_extent_mm": float(certificate["total_value_mm"]),
            "term_dimension_refs": term_refs,
            "term_values_mm": terms,
            "arithmetic_residual_mm": float(certificate["arithmetic_residual_mm"]),
            "bands": bands,
            "clear_gap_mm": terms[2],
            "clear_gap_interval_mm": [offsets[2], offsets[3]],
            "equal_sweep_width_mm": terms[1],
            "bands_are_disjoint": offsets[2] <= offsets[3],
            "orientation_certificate_ref": orientation.get("id"),
            "signed_orientation_state": (
                "resolved" if orientation.get("status") == "accepted" else "unresolved"
            ),
            "evidence_refs": evidence_refs,
        },
        "profile_split_assessment": profile_assessment,
        "evidence_refs": evidence_refs,
        "contract": {
            "accepted_arithmetic_chain_required": True,
            "two_disjoint_equal_sweep_bands_required": True,
            "plan_sweep_width_resolved": True,
            "signed_orientation_resolved": orientation.get("status") == "accepted",
            "unresolved_sign_does_not_invalidate_plan_bands": True,
            "physical_component_identity_established": False,
            "landing_footprint_resolved": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }
