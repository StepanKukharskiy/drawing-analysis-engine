"""Reviewed 18-page occurrence coverage audit for M7A/M7B.

The audit distinguishes accepted M4 occurrences from reviewed hints that lack
an accepted M4 target.  A markup claim may identify a category worth auditing,
but it never becomes an item occurrence, physical identity, or count.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.drawing_engine.disciplines.mep.mep_annotation_observations import validate_pdf_annotation_observations
from src.drawing_engine.disciplines.mep.mep_attribute_binding import validate_mep_attribute_bindings
from src.drawing_engine.disciplines.mep.mep_sheet_registry import validate_mep_sheet_registry


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_occurrence_coverage_audit"
DISCRETE_CATEGORIES = ("equipment", "valve", "fitting", "damper", "fixture")
DISCRETE_RELATION_TYPES = {"equipment_endpoint", "valve", "fitting", "damper"}
ROUTE_RELATION_TYPES = {"route_system", "route_size", "route_elevation"}


def validate_mep_discrete_target_review(
    payload: Mapping[str, Any],
    *,
    sheet_registry: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
) -> list[str]:
    """Validate the reviewed M4-negative page freeze against its exact inputs."""

    errors = []
    if payload.get("schema_version") != "0.1.0":
        errors.append("M4 target review schema_version mismatch")
    if payload.get("layer") != "mep_m4_discrete_target_review_freeze":
        errors.append("M4 target review layer mismatch")
    if payload.get("document", {}).get("document_key") != sheet_registry.get(
        "document", {}
    ).get("document_key"):
        errors.append("M4 target review document key mismatch")
    if payload.get("m4_contract_ref", {}).get("payload_sha256") != _sha256(
        attribute_bindings
    ):
        errors.append("M4 target review does not bind supplied M4")

    registry_pages = {
        str(row.get("page_ref")): row for row in sheet_registry.get("pages", [])
    }
    m4_relations = {
        str(row.get("id")): row for row in attribute_bindings.get("relations", [])
    }
    accepted_route_pages = {
        str(row.get("page_ref"))
        for row in m4_relations.values()
        if row.get("state") == "accepted"
        and str(row.get("relation_type")) in ROUTE_RELATION_TYPES
    }
    expected_page_refs = {
        page_ref
        for page_ref, page in registry_pages.items()
        if page.get("role") != "divider" and page_ref not in accepted_route_pages
    }
    pages = list(payload.get("pages", []))
    page_refs = [str(row.get("page_ref")) for row in pages]
    if len(page_refs) != len(set(page_refs)) or set(page_refs) != expected_page_refs:
        errors.append("M4 target review must cover each route-uncovered item-bearing page once")
    reviewed_relation_refs = []
    state_counts = Counter()
    accepted_target_count = 0
    for page in pages:
        page_ref = str(page.get("page_ref"))
        source_page = registry_pages.get(page_ref)
        state = page.get("review_state")
        relation_refs = [str(ref) for ref in page.get("relation_refs", [])]
        proposal_refs = [str(ref) for ref in page.get("proposal_refs", [])]
        if source_page is None or page.get("pdf_page_number") != source_page.get(
            "page_number"
        ):
            errors.append(f"{page_ref}: M4 target review page mismatch")
        if state not in {
            "reviewed_no_acceptable_discrete_target",
            "candidate_inventory_not_established",
        }:
            errors.append(f"{page_ref}: invalid M4 target review state")
        else:
            state_counts[str(state)] += 1
        if state == "reviewed_no_acceptable_discrete_target" and not relation_refs:
            errors.append(f"{page_ref}: reviewed negative page lacks relations")
        if state == "candidate_inventory_not_established" and (
            relation_refs or proposal_refs
        ):
            errors.append(f"{page_ref}: inventory-pending page has promoted references")
        if page.get("document_occurrence_completeness_established") is not False:
            errors.append(f"{page_ref}: document occurrence completeness must be false")
        for relation_ref in relation_refs:
            relation = m4_relations.get(relation_ref)
            if relation is None:
                errors.append(f"{page_ref}: unknown M4 target-review relation")
                continue
            if relation.get("page_ref") != page_ref:
                errors.append(f"{page_ref}: cross-page M4 target-review relation")
            if relation.get("relation_type") not in DISCRETE_RELATION_TYPES:
                errors.append(f"{page_ref}: non-discrete M4 target-review relation")
            if str(relation.get("proposal_ref")) not in proposal_refs:
                errors.append(f"{page_ref}: proposal membership mismatch")
            accepted_target_count += relation.get("state") == "accepted"
        if page.get("accepted_discrete_target_count") != sum(
            m4_relations.get(ref, {}).get("state") == "accepted"
            for ref in relation_refs
        ):
            errors.append(f"{page_ref}: accepted target count mismatch")
        reviewed_relation_refs.extend(relation_refs)
    expected_summary = {
        "uncovered_item_bearing_page_count": len(pages),
        "review_state_counts": dict(sorted(state_counts.items())),
        "reviewed_discrete_proposal_count": len(reviewed_relation_refs),
        "accepted_discrete_target_count": accepted_target_count,
    }
    if payload.get("summary") != expected_summary:
        errors.append("M4 target review summary mismatch")
    if payload.get("coverage_conclusion") != {
        "reviewed_markup_regions_with_no_acceptable_target": state_counts.get(
            "reviewed_no_acceptable_discrete_target", 0
        ),
        "pages_with_candidate_inventory_not_established": state_counts.get(
            "candidate_inventory_not_established", 0
        ),
        "document_occurrence_completeness": "not_established",
    }:
        errors.append("M4 target review conclusion mismatch")
    if payload.get("authority") != {
        "item_occurrence_established": False,
        "physical_identity_established": False,
        "calculated_count_established": False,
        "quantity_eligible": False,
    }:
        errors.append("M4 target review authority mismatch")
    return errors


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accepted_m4_occurrences(bindings: Mapping[str, Any]) -> list[dict[str, Any]]:
    route_groups: dict[tuple[str, str], list[str]] = {}
    discrete = []
    for relation in bindings.get("relations", []):
        if relation.get("state") != "accepted":
            continue
        relation_type = str(relation.get("relation_type"))
        targets = [str(ref) for ref in relation.get("target_refs", [])]
        if relation_type in ROUTE_RELATION_TYPES and len(targets) == 1:
            route_groups.setdefault(
                (str(relation.get("page_ref")), targets[0]), []
            ).append(str(relation.get("id")))
        elif relation_type in DISCRETE_RELATION_TYPES:
            discrete.append({
                "occurrence_kind": "discrete_item",
                "category": (
                    "equipment" if relation_type == "equipment_endpoint" else relation_type
                ),
                "page_ref": str(relation.get("page_ref")),
                "target_refs": targets,
                "m4_relation_refs": [str(relation.get("id"))],
                "state": "accepted_m4_occurrence",
                "physical_identity_established": False,
                "calculated_count": None,
            })
    routes = [
        {
            "occurrence_kind": "routed_material",
            "category": "routed_material",
            "page_ref": page_ref,
            "target_refs": [target_ref],
            "m4_relation_refs": sorted(refs),
            "state": "accepted_m4_occurrence",
            "physical_identity_established": False,
            "calculated_count": None,
        }
        for (page_ref, target_ref), refs in sorted(route_groups.items())
    ]
    return sorted(
        [*routes, *discrete],
        key=lambda row: (row["page_ref"], row["occurrence_kind"], row["target_refs"]),
    )


def build_mep_occurrence_coverage_audit(
    *,
    sheet_registry: Mapping[str, Any],
    attribute_bindings: Mapping[str, Any],
    annotation_observations: Mapping[str, Any],
    reviewed_coverage: Mapping[str, Any],
    discrete_target_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reviewed coverage audit without promoting reviewed hints."""

    upstream_errors = [
        *[f"M0: {error}" for error in validate_pdf_annotation_observations(annotation_observations)],
        *[f"M1: {error}" for error in validate_mep_sheet_registry(sheet_registry)],
        *[f"M4: {error}" for error in validate_mep_attribute_bindings(attribute_bindings)],
    ]
    if upstream_errors:
        raise ValueError("invalid coverage-audit input:\n" + "\n".join(upstream_errors))
    document_keys = {
        str(payload.get("document", {}).get("document_key"))
        for payload in (sheet_registry, attribute_bindings, annotation_observations)
    }
    if len(document_keys) != 1:
        raise ValueError("M0, M1, and M4 document keys must match")
    if reviewed_coverage.get("document_key") not in document_keys:
        raise ValueError("reviewed coverage document key mismatch")
    if reviewed_coverage.get("review", {}).get("status") != "reviewed_occurrence_coverage":
        raise ValueError("reviewed coverage status is required")
    target_review_by_page: dict[str, Mapping[str, Any]] = {}
    if discrete_target_review is not None:
        review_errors = validate_mep_discrete_target_review(
            discrete_target_review,
            sheet_registry=sheet_registry,
            attribute_bindings=attribute_bindings,
        )
        if review_errors:
            raise ValueError("invalid M4 target review:\n" + "\n".join(review_errors))
        target_review_by_page = {
            str(row.get("page_ref")): row
            for row in discrete_target_review.get("pages", [])
        }

    pairs = {
        str(row.get("id")): row
        for row in annotation_observations.get("annotation_pairs", [])
    }
    annotations = {
        str(row.get("id")): row
        for row in annotation_observations.get("annotations", [])
    }
    findings = []
    for source in reviewed_coverage.get("reviewed_findings", []):
        pair_ref = str(source.get("annotation_pair_ref"))
        pair = pairs.get(pair_ref)
        if pair is None:
            raise ValueError(f"unknown reviewed annotation pair: {pair_ref}")
        categories = sorted(set(str(value) for value in source.get("categories", [])))
        if not categories or any(value not in DISCRETE_CATEGORIES for value in categories):
            raise ValueError(f"invalid reviewed discrete categories: {categories}")
        callout_ref = str(pair.get("callout_annotation_ref"))
        callout = annotations.get(callout_ref, {})
        findings.append({
            "record_type": "mep_occurrence_coverage_finding",
            "record_version": SCHEMA_VERSION,
            "id": str(source.get("id")),
            "page_ref": str(pair.get("page_ref")),
            "pdf_page_number": int(pair.get("page_number")),
            "categories": categories,
            "description": callout.get("text"),
            "annotation_pair_ref": pair_ref,
            "annotation_refs": sorted({
                str(pair.get("cloud_annotation_ref")),
                callout_ref,
            }),
            "review_state": "reviewed_unresolved_hint",
            "disposition": "omitted_from_accepted_occurrences",
            "reason": str(source.get("reason")),
            "m4_target_ref": None,
            "observed_occurrence_count": None,
            "deduplicated_projected_count": None,
            "physical_instance_count": None,
            "calculated_count": None,
            "authority": {
                "item_occurrence_established": False,
                "physical_identity_established": False,
                "calculated_count_established": False,
            },
        })
    findings.sort(key=lambda row: (row["pdf_page_number"], row["id"]))
    reviewed_category_rows = reviewed_coverage.get("category_review", [])
    if [row.get("category") for row in reviewed_category_rows] != list(DISCRETE_CATEGORIES):
        raise ValueError("reviewed category coverage must include every discrete category")
    for row in reviewed_category_rows:
        category = str(row.get("category"))
        expected_count = sum(category in finding["categories"] for finding in findings)
        if row.get("reviewed_unresolved_hint_count") != expected_count:
            raise ValueError(f"reviewed category count mismatch: {category}")
        if row.get("source_absence_established") is not False:
            raise ValueError(f"review cannot establish source absence: {category}")

    occurrences = _accepted_m4_occurrences(attribute_bindings)
    occurrence_refs_by_page: dict[str, list[str]] = {}
    for index, occurrence in enumerate(occurrences):
        occurrence["id"] = f"mep_coverage_accepted_occurrence.{_sha256(occurrence)[:20]}"
        occurrence_refs_by_page.setdefault(occurrence["page_ref"], []).append(
            occurrence["id"]
        )
    finding_refs_by_page: dict[str, list[str]] = {}
    for finding in findings:
        finding_refs_by_page.setdefault(finding["page_ref"], []).append(finding["id"])

    pages = []
    for page in sorted(
        sheet_registry.get("pages", []), key=lambda row: int(row.get("page_number", 0))
    ):
        page_ref = str(page.get("page_ref"))
        role = str(page.get("role") or "unknown")
        accepted_refs = sorted(occurrence_refs_by_page.get(page_ref, []))
        finding_refs = sorted(finding_refs_by_page.get(page_ref, []))
        if role == "divider":
            coverage_state = "not_applicable_divider"
        elif accepted_refs:
            coverage_state = "current_m4_covered_subset"
        else:
            coverage_state = "not_m4_occurrence_covered"
        pages.append({
            "record_type": "mep_occurrence_coverage_page",
            "record_version": SCHEMA_VERSION,
            "page_ref": page_ref,
            "pdf_page_number": int(page.get("page_number")),
            "drawing_sheet_number": page.get("fields", {}).get("sheet_number", {}).get("value"),
            "sheet_role": role,
            "coverage_state": coverage_state,
            "accepted_m4_occurrence_refs": accepted_refs,
            "reviewed_unresolved_finding_refs": finding_refs,
            "m4_discrete_target_review_state": (
                target_review_by_page.get(page_ref, {}).get("review_state")
                if discrete_target_review is not None
                else None
            ),
            "m4_discrete_target_relation_refs": sorted(
                str(ref)
                for ref in target_review_by_page.get(page_ref, {}).get(
                    "relation_refs", []
                )
            ),
            "document_occurrence_completeness_established": False,
        })

    category_coverage = []
    for category in DISCRETE_CATEGORIES:
        accepted_count = sum(row["category"] == category for row in occurrences)
        hint_count = sum(category in row["categories"] for row in findings)
        category_coverage.append({
            "category": category,
            "accepted_m4_occurrence_count": accepted_count,
            "reviewed_unresolved_hint_count": hint_count,
            "source_absence_established": False,
            "coverage_state": (
                "accepted_and_unresolved" if accepted_count and hint_count
                else "accepted_subset" if accepted_count
                else "reviewed_unresolved_hints_only" if hint_count
                else "not_covered_no_reviewed_hint"
            ),
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(sheet_registry.get("document", {}))),
        "review": deepcopy(dict(reviewed_coverage.get("review", {}))),
        "m0_contract_ref": {
            "layer": annotation_observations.get("layer"),
            "schema_version": annotation_observations.get("schema_version"),
            "payload_sha256": _sha256(annotation_observations),
        },
        "m1_contract_ref": {
            "layer": sheet_registry.get("layer"),
            "schema_version": sheet_registry.get("schema_version"),
            "payload_sha256": _sha256(sheet_registry),
        },
        "m4_contract_ref": {
            "layer": attribute_bindings.get("layer"),
            "schema_version": attribute_bindings.get("schema_version"),
            "payload_sha256": _sha256(attribute_bindings),
        },
        "m4_discrete_target_review_contract_ref": (
            {
                "layer": discrete_target_review.get("layer"),
                "schema_version": discrete_target_review.get("schema_version"),
                "payload_sha256": _sha256(discrete_target_review),
                "document_occurrence_completeness": discrete_target_review.get(
                    "coverage_conclusion", {}
                ).get("document_occurrence_completeness"),
            }
            if discrete_target_review is not None
            else None
        ),
        "pages": pages,
        "accepted_m4_occurrences": occurrences,
        "reviewed_unresolved_findings": findings,
        "category_coverage": category_coverage,
        "summary": {
            "page_count": len(pages),
            "item_bearing_page_count": sum(row["sheet_role"] != "divider" for row in pages),
            "m4_covered_page_count": sum(row["coverage_state"] == "current_m4_covered_subset" for row in pages),
            "uncovered_item_bearing_page_count": sum(row["coverage_state"] == "not_m4_occurrence_covered" for row in pages),
            "accepted_m4_occurrence_count": len(occurrences),
            "accepted_routed_material_occurrence_count": sum(row["occurrence_kind"] == "routed_material" for row in occurrences),
            "accepted_discrete_occurrence_count": sum(row["occurrence_kind"] == "discrete_item" for row in occurrences),
            "reviewed_unresolved_finding_count": len(findings),
            "reviewed_negative_target_page_count": sum(
                row.get("m4_discrete_target_review_state")
                == "reviewed_no_acceptable_discrete_target"
                for row in pages
            ),
            "candidate_inventory_not_established_page_count": sum(
                row.get("m4_discrete_target_review_state")
                == "candidate_inventory_not_established"
                for row in pages
            ),
        },
        "coverage_conclusion": {
            "accepted_set_scope": "complete_for_frozen_m4_payload_only",
            "document_occurrence_completeness": "not_established",
            "four_routed_material_occurrences_are_current_m4_covered_subset": True,
            "reviewed_hints_are_not_item_occurrences": True,
            "zero_reviewed_hints_does_not_establish_category_absence": True,
        },
        "authority": {
            "physical_identity_established": False,
            "calculated_count_established": False,
            "installed_length_emitted": False,
            "quantity_eligible": False,
        },
    }
    errors = validate_mep_occurrence_coverage_audit(payload)
    if errors:
        raise ValueError("\n".join(errors))
    return payload


def validate_mep_occurrence_coverage_audit(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for name in ("m0_contract_ref", "m1_contract_ref", "m4_contract_ref"):
        digest = payload.get(name, {}).get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{name}.payload_sha256 must be a SHA-256 digest")
    target_review_ref = payload.get("m4_discrete_target_review_contract_ref")
    if target_review_ref is not None:
        digest = target_review_ref.get("payload_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("m4_discrete_target_review_contract_ref.payload_sha256 must be a SHA-256 digest")
        if target_review_ref.get("document_occurrence_completeness") != "not_established":
            errors.append("M4 target review cannot establish document completeness")
    pages = list(payload.get("pages", []))
    if [row.get("pdf_page_number") for row in pages] != list(range(1, len(pages) + 1)):
        errors.append("coverage pages must be complete and ordered")
    page_refs = {str(row.get("page_ref")) for row in pages}
    accepted = list(payload.get("accepted_m4_occurrences", []))
    accepted_ids = {str(row.get("id")) for row in accepted}
    if len(accepted_ids) != len(accepted):
        errors.append("accepted M4 occurrence IDs must be unique")
    for row in accepted:
        identifier = str(row.get("id"))
        if str(row.get("page_ref")) not in page_refs:
            errors.append(f"{identifier}: page is not registered")
        if row.get("state") != "accepted_m4_occurrence":
            errors.append(f"{identifier}: accepted occurrence state mismatch")
        if row.get("physical_identity_established") is not False:
            errors.append(f"{identifier}: physical identity must remain unresolved")
        if row.get("calculated_count") is not None:
            errors.append(f"{identifier}: calculated count must remain null")
    findings = list(payload.get("reviewed_unresolved_findings", []))
    finding_ids = {str(row.get("id")) for row in findings}
    if len(finding_ids) != len(findings):
        errors.append("reviewed finding IDs must be unique")
    for row in findings:
        identifier = str(row.get("id"))
        if str(row.get("page_ref")) not in page_refs:
            errors.append(f"{identifier}: page is not registered")
        if row.get("disposition") != "omitted_from_accepted_occurrences":
            errors.append(f"{identifier}: reviewed hint must remain omitted")
        if row.get("m4_target_ref") is not None:
            errors.append(f"{identifier}: reviewed hint cannot gain an M4 target")
        for key in (
            "observed_occurrence_count",
            "deduplicated_projected_count",
            "physical_instance_count",
            "calculated_count",
        ):
            if row.get(key) is not None:
                errors.append(f"{identifier}: {key} must remain null")
        if any(row.get("authority", {}).values()):
            errors.append(f"{identifier}: reviewed hint authority must remain false")
    for page in pages:
        page_ref = str(page.get("page_ref"))
        expected_accepted_refs = sorted(
            str(row.get("id")) for row in accepted if str(row.get("page_ref")) == page_ref
        )
        expected_finding_refs = sorted(
            str(row.get("id")) for row in findings if str(row.get("page_ref")) == page_ref
        )
        if page.get("accepted_m4_occurrence_refs") != expected_accepted_refs:
            errors.append(f"{page_ref}: accepted occurrence membership mismatch")
        if page.get("reviewed_unresolved_finding_refs") != expected_finding_refs:
            errors.append(f"{page_ref}: reviewed finding membership mismatch")
        if page.get("document_occurrence_completeness_established") is not False:
            errors.append(f"{page_ref}: document completeness cannot be established")
        target_state = page.get("m4_discrete_target_review_state")
        target_refs = page.get("m4_discrete_target_relation_refs", [])
        if target_review_ref is not None:
            if target_state not in {
                None,
                "reviewed_no_acceptable_discrete_target",
                "candidate_inventory_not_established",
            }:
                errors.append(f"{page_ref}: invalid M4 discrete-target review state")
            if target_state == "reviewed_no_acceptable_discrete_target" and not target_refs:
                errors.append(f"{page_ref}: reviewed negative target needs relation evidence")
            if target_state == "candidate_inventory_not_established" and target_refs:
                errors.append(f"{page_ref}: inventory-pending page cannot cite target relations")
        elif target_state is not None or target_refs:
            errors.append(f"{page_ref}: target review fields require a contract")
    categories = list(payload.get("category_coverage", []))
    if [row.get("category") for row in categories] != list(DISCRETE_CATEGORIES):
        errors.append("all discrete coverage categories are required in stable order")
    if any(row.get("source_absence_established") is not False for row in categories):
        errors.append("coverage audit cannot establish category absence")
    for row in categories:
        category = str(row.get("category"))
        if row.get("accepted_m4_occurrence_count") != sum(
            item.get("category") == category for item in accepted
        ):
            errors.append(f"{category}: accepted occurrence category count mismatch")
        if row.get("reviewed_unresolved_hint_count") != sum(
            category in item.get("categories", []) for item in findings
        ):
            errors.append(f"{category}: reviewed hint category count mismatch")
    conclusion = payload.get("coverage_conclusion", {})
    if conclusion.get("accepted_set_scope") != "complete_for_frozen_m4_payload_only":
        errors.append("accepted set scope mismatch")
    if conclusion.get("document_occurrence_completeness") != "not_established":
        errors.append("document occurrence completeness must remain unresolved")
    for key in (
        "four_routed_material_occurrences_are_current_m4_covered_subset",
        "reviewed_hints_are_not_item_occurrences",
        "zero_reviewed_hints_does_not_establish_category_absence",
    ):
        if conclusion.get(key) is not True:
            errors.append(f"coverage_conclusion.{key} must remain true")
    if payload.get("authority") != {
        "physical_identity_established": False,
        "calculated_count_established": False,
        "installed_length_emitted": False,
        "quantity_eligible": False,
    }:
        errors.append("coverage audit authority boundary mismatch")
    summary = payload.get("summary", {})
    expected = {
        "page_count": len(pages),
        "item_bearing_page_count": sum(row.get("sheet_role") != "divider" for row in pages),
        "m4_covered_page_count": sum(row.get("coverage_state") == "current_m4_covered_subset" for row in pages),
        "uncovered_item_bearing_page_count": sum(row.get("coverage_state") == "not_m4_occurrence_covered" for row in pages),
        "accepted_m4_occurrence_count": len(accepted),
        "accepted_routed_material_occurrence_count": sum(row.get("occurrence_kind") == "routed_material" for row in accepted),
        "accepted_discrete_occurrence_count": sum(row.get("occurrence_kind") == "discrete_item" for row in accepted),
        "reviewed_unresolved_finding_count": len(findings),
        "reviewed_negative_target_page_count": sum(
            row.get("m4_discrete_target_review_state")
            == "reviewed_no_acceptable_discrete_target"
            for row in pages
        ),
        "candidate_inventory_not_established_page_count": sum(
            row.get("m4_discrete_target_review_state")
            == "candidate_inventory_not_established"
            for row in pages
        ),
    }
    if summary != expected:
        errors.append("summary mismatch")
    return errors
