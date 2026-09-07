"""Generate quantity-free Step 5 physical-component alternatives.

This is Slice 1 of Step 5.  It partitions accepted Step 3 profile bindings
into bounded candidate groups using only profile topology, metric extents,
shared boundary evidence, and projection role.  It never accepts a physical
component, resolves a transform, invokes Step 4, or emits a quantity.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "0.1.0"


def _stable_id(kind: str, page_number: int, *parts: str) -> str:
    encoded = "\0".join((str(page_number), *parts)).encode("utf-8")
    return f"{kind}.page_{page_number:04d}.evidence_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _profile_metric(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    box = profile.get("bbox_display")
    certificate = profile.get("dimension_certificate", {}) or {}
    scale = certificate.get("scale_points_per_mm")
    boundary = profile.get("ordered_boundary_display")
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not isinstance(boundary, (list, tuple))
        or len(boundary) < 3
        or scale is None
        or float(scale) <= 0
    ):
        return None
    width = max(0.0, float(box[2]) - float(box[0])) / float(scale)
    height = max(0.0, float(box[3]) - float(box[1])) / float(scale)
    if width <= 0 or height <= 0:
        return None
    edge_lengths = []
    points = [tuple(map(float, point)) for point in boundary]
    for index, point in enumerate(points):
        edge_lengths.append(math.dist(point, points[(index + 1) % len(points)]) / float(scale))
    return {
        "scale_points_per_mm": round(float(scale), 9),
        "width_mm": round(width, 6),
        "height_mm": round(height, 6),
        "vertex_count": len(points),
        "source_edge_count": len(profile.get("source_edge_refs", []) or []),
        "edge_lengths_mm": [round(value, 6) for value in sorted(edge_lengths)],
    }


def _boundary_edges(profile: Mapping[str, Any]) -> dict[tuple[tuple[float, float], tuple[float, float]], set[str]]:
    points = [tuple(map(float, point)) for point in profile.get("ordered_boundary_display", []) or []]
    source_refs = list(map(str, profile.get("source_edge_refs", []) or []))
    output: dict[tuple[tuple[float, float], tuple[float, float]], set[str]] = {}
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        key = tuple(sorted(((round(point[0], 6), round(point[1], 6)), (round(other[0], 6), round(other[1], 6)))))
        refs = {source_refs[index]} if index < len(source_refs) else set(source_refs)
        output.setdefault(key, set()).update(refs)
    return output


def _within(left: float, right: float, tolerance_ratio: float) -> bool:
    tolerance = max(2.0, tolerance_ratio * max(abs(left), abs(right)))
    return abs(left - right) <= tolerance


def _metric_topology_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    tolerance_ratio: float,
) -> bool:
    return (
        left["vertex_count"] == right["vertex_count"]
        and left["source_edge_count"] == right["source_edge_count"]
        and _within(float(left["width_mm"]), float(right["width_mm"]), tolerance_ratio)
        and _within(float(left["height_mm"]), float(right["height_mm"]), tolerance_ratio)
    )


def _components(nodes: Iterable[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    remaining = set(nodes)
    groups = []
    while remaining:
        seed = min(remaining)
        group = set()
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(sorted(adjacency.get(current, set()) - group, reverse=True))
        remaining -= group
        groups.append(sorted(group))
    return sorted(groups, key=tuple)


def _signed_transform_blocker(
    physical_scope_ref: str,
    object_scopes: Mapping[str, Mapping[str, Any]],
    coordinates: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, list[str]]:
    physical_scope = object_scopes.get(physical_scope_ref)
    coordinate_ref = str((physical_scope or {}).get("shared_coordinate_scope_id") or "")
    coordinate = coordinates.get(coordinate_ref, {})
    orientation = coordinate.get("signed_orientation_certificate") or {}
    if orientation and orientation.get("status") != "accepted":
        constraints = orientation.get("constraint_certificates", []) or []
        reasons = sorted(
            {
                str(gate.get("reason"))
                for item in constraints
                for gate in (
                    item.get("oriented_cutting_plane", {}) or {},
                    item.get("signed_shared_axis", {}) or {},
                )
                if gate.get("status") not in {"pass", "accepted"} and gate.get("reason")
            }
        )
        evidence_refs = sorted(
            {
                coordinate_ref,
                str(orientation.get("id") or ""),
                *[
                    str(ref)
                    for item in constraints
                    for gate in (
                        item.get("oriented_cutting_plane", {}) or {},
                        item.get("signed_shared_axis", {}) or {},
                    )
                    for ref in gate.get("evidence_refs", []) or []
                ],
            }
            - {""}
        )
        return (
            (
                "signed_orientation_certificate_contradiction"
                if orientation.get("status") == "contradiction"
                else "signed_orientation_certificate_unresolved"
            ),
            "; ".join(reasons) or "scope-level cut normal and shared-axis signs remain unresolved",
            evidence_refs,
        )
    for correspondence in coordinate.get("contour_correspondences", []) or []:
        if correspondence.get("state") != "accepted":
            continue
        selected = correspondence.get("selected", {}) or {}
        transform = selected.get("signed_transform", {}) or {}
        candidates = transform.get("candidates", []) or []
        if transform.get("state") != "resolved" and len(candidates) > 1:
            evidence_refs = sorted(
                {
                    coordinate_ref,
                    str(correspondence.get("id") or ""),
                    *map(str, selected.get("primitive_refs", []) or []),
                }
                - {""}
            )
            return (
                "mirrored_contour_transform_unresolved",
                str(transform.get("reason") or "forward and mirrored contour transforms remain equivalent"),
                evidence_refs,
            )
    return (
        "cross_view_identity_and_placement_not_reclosed",
        "Slice 1 alternatives require the deterministic cross-view identity and placement certificate before acceptance",
        sorted({physical_scope_ref, coordinate_ref} - {""}),
    )


def generate_physical_component_hypotheses(
    scoped_profile_assembly: Mapping[str, Any],
    profile_scope_binding: Mapping[str, Any],
    view_frame_graph: Mapping[str, Any],
    *,
    page_number: int,
    max_hypotheses: int = 256,
    metric_tolerance_ratio: float = 0.02,
) -> dict[str, Any]:
    """Publish bounded alternatives without establishing additive identity."""

    if not 1 <= max_hypotheses <= 4096:
        raise ValueError("max_hypotheses must be between 1 and 4096")
    if not 0 < metric_tolerance_ratio <= 0.1:
        raise ValueError("metric_tolerance_ratio must be positive and at most 0.1")

    profiles = {
        str(item.get("id")): item
        for item in scoped_profile_assembly.get("profiles", []) or []
        if item.get("id") is not None
    }
    object_scopes = {
        str(item.get("id")): item
        for item in view_frame_graph.get("object_scopes", []) or []
        if item.get("id") is not None
    }
    coordinates = {
        str(item.get("id")): item
        for item in view_frame_graph.get("shared_coordinate_system", {}).get("scopes", []) or []
        if item.get("id") is not None
    }
    rows_by_partition: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    abstentions = []
    for binding in sorted(
        profile_scope_binding.get("bindings", []) or [], key=lambda item: str(item.get("id"))
    ):
        profile_ref = str(binding.get("profile_ref") or "")
        profile = profiles.get(profile_ref)
        metric = _profile_metric(profile or {})
        if (
            binding.get("state") != "accepted"
            or binding.get("step5_reconstruction_input_eligible") is not True
            or profile is None
            or profile.get("state") != "resolved"
            or metric is None
        ):
            evidence_refs = sorted(
                {str(binding.get("id") or ""), profile_ref, *map(str, binding.get("evidence_refs", []) or [])}
                - {""}
            )
            abstentions.append(
                {
                    "id": _stable_id("physical_component_hypothesis_abstention", page_number, profile_ref, "invalid_step5_input", *evidence_refs),
                    "record_type": "physical_component_hypothesis_abstention",
                    "record_version": SCHEMA_VERSION,
                    "page": page_number,
                    "state": "abstained",
                    "reason_code": "invalid_step5_input",
                    "reason": "profile binding does not preserve a resolved metric Step 5 reconstruction input",
                    "profile_refs": [profile_ref] if profile_ref else [],
                    "surviving_hypothesis_refs": [],
                    "evidence_refs": evidence_refs,
                    "quantity_eligible": False,
                }
            )
            continue
        partition = (
            str(binding.get("physical_scope_ref") or ""),
            str(binding.get("source_view_ref") or ""),
            str(binding.get("membership_role") or ""),
        )
        rows_by_partition[partition].append(
            {
                "binding": binding,
                "profile": profile,
                "metric": metric,
                "boundary_edges": _boundary_edges(profile),
            }
        )

    hypotheses = []
    alternative_sets = []
    hypotheses_by_scope: dict[str, list[str]] = defaultdict(list)
    for partition, rows in sorted(rows_by_partition.items()):
        physical_scope_ref, source_view_ref, projection_role = partition
        by_profile = {str(row["profile"]["id"]): row for row in rows}
        adjacency: dict[str, set[str]] = defaultdict(set)
        shared_by_pair: dict[tuple[str, str], list[str]] = {}
        refs = sorted(by_profile)
        for left_index, left_ref in enumerate(refs):
            left = by_profile[left_ref]
            for right_ref in refs[left_index + 1 :]:
                right = by_profile[right_ref]
                shared_native = set(map(str, left["profile"].get("source_edge_refs", []) or [])) & set(
                    map(str, right["profile"].get("source_edge_refs", []) or [])
                )
                shared_geometry = set(left["boundary_edges"]) & set(right["boundary_edges"])
                if not shared_native and not shared_geometry:
                    continue
                if not _metric_topology_compatible(left["metric"], right["metric"], metric_tolerance_ratio):
                    continue
                adjacency[left_ref].add(right_ref)
                adjacency[right_ref].add(left_ref)
                shared_refs = set(shared_native)
                for edge in shared_geometry:
                    shared_refs.update(left["boundary_edges"][edge])
                    shared_refs.update(right["boundary_edges"][edge])
                shared_by_pair[(left_ref, right_ref)] = sorted(shared_refs)

        groups = _components(refs, adjacency)
        if len(hypotheses) + len(groups) > max_hypotheses:
            evidence_refs = sorted(
                {
                    *(str(row["binding"].get("id")) for row in rows),
                    *(str(row["profile"].get("id")) for row in rows),
                }
            )
            abstentions.append(
                {
                    "id": _stable_id("physical_component_hypothesis_abstention", page_number, physical_scope_ref, "hypothesis_search_not_bounded", *evidence_refs),
                    "record_type": "physical_component_hypothesis_abstention",
                    "record_version": SCHEMA_VERSION,
                    "page": page_number,
                    "state": "abstained",
                    "physical_scope_ref": physical_scope_ref,
                    "reason_code": "hypothesis_search_not_bounded",
                    "reason": f"profile grouping would exceed the bound of {max_hypotheses} hypotheses",
                    "profile_refs": refs,
                    "surviving_hypothesis_refs": [],
                    "evidence_refs": evidence_refs,
                    "quantity_eligible": False,
                }
            )
            continue

        partition_hypotheses = []
        for group in groups:
            binding_refs = sorted(str(by_profile[ref]["binding"]["id"]) for ref in group)
            shared_edge_refs = sorted(
                {
                    edge_ref
                    for pair, pair_refs in shared_by_pair.items()
                    if set(pair) <= set(group)
                    for edge_ref in pair_refs
                }
            )
            evidence_refs = sorted(
                {
                    *group,
                    *binding_refs,
                    *shared_edge_refs,
                    *(ref for profile_ref in group for ref in map(str, by_profile[profile_ref]["profile"].get("evidence_refs", []) or [])),
                }
            )
            hypothesis_id = _stable_id(
                "physical_component_hypothesis",
                page_number,
                physical_scope_ref,
                source_view_ref,
                projection_role,
                *group,
                *shared_edge_refs,
            )
            hypothesis = {
                "record_type": "physical_component_hypothesis",
                "record_version": SCHEMA_VERSION,
                "id": hypothesis_id,
                "page": page_number,
                "state": "candidate",
                "epistemic_state": "inferred",
                "physical_scope_ref": physical_scope_ref,
                "source_view_ref": source_view_ref,
                "projection_role": projection_role,
                "profile_refs": group,
                "binding_refs": binding_refs,
                "alternative_set_ref": None,
                "compatibility_certificate": {
                    "status": "passed",
                    "topology": {
                        "status": "passed",
                        "profile_signatures": {
                            ref: {
                                "vertex_count": by_profile[ref]["metric"]["vertex_count"],
                                "source_edge_count": by_profile[ref]["metric"]["source_edge_count"],
                            }
                            for ref in group
                        },
                    },
                    "metric_extents": {
                        "status": "passed",
                        "tolerance_ratio": metric_tolerance_ratio,
                        "profiles": {
                            ref: {
                                key: by_profile[ref]["metric"][key]
                                for key in ("scale_points_per_mm", "width_mm", "height_mm")
                            }
                            for ref in group
                        },
                    },
                    "shared_edges": {
                        "status": "passed" if len(group) > 1 else "not_required_single_profile",
                        "edge_refs": shared_edge_refs,
                    },
                    "projection_role": {
                        "status": "passed",
                        "physical_scope_ref": physical_scope_ref,
                        "source_view_ref": source_view_ref,
                        "role": projection_role,
                    },
                    "grouping_basis": "shared_boundary_edges" if len(group) > 1 else "single_supporting_profile",
                },
                "additive_count_interpretation": "unresolved",
                "additive_component_identity_established": False,
                "physical_component_ref": None,
                "physical_transform_refs": [],
                "cross_view_identity_and_placement_state": "unresolved",
                "step4_kernel_invocation_eligible": False,
                "quantity_eligible": False,
                "evidence_refs": evidence_refs,
            }
            hypotheses.append(hypothesis)
            partition_hypotheses.append(hypothesis)
            hypotheses_by_scope[physical_scope_ref].append(hypothesis_id)

        # Metric/topology-equivalent, disjoint groups are alternatives, never
        # an additive count.  Complete-link clustering avoids tolerance chains.
        clusters: list[list[dict[str, Any]]] = []
        for hypothesis in sorted(partition_hypotheses, key=lambda item: item["id"]):
            representative = by_profile[hypothesis["profile_refs"][0]]["metric"]
            matching = None
            for cluster in clusters:
                if all(
                    len(item["profile_refs"]) == len(hypothesis["profile_refs"])
                    and _metric_topology_compatible(
                        representative,
                        by_profile[item["profile_refs"][0]]["metric"],
                        metric_tolerance_ratio,
                    )
                    for item in cluster
                ):
                    matching = cluster
                    break
            if matching is None:
                clusters.append([hypothesis])
            else:
                matching.append(hypothesis)

        for cluster in clusters:
            hypothesis_refs = sorted(item["id"] for item in cluster)
            profile_refs = sorted({ref for item in cluster for ref in item["profile_refs"]})
            alternative_set_ref = _stable_id(
                "physical_component_alternative_set",
                page_number,
                physical_scope_ref,
                source_view_ref,
                projection_role,
                *profile_refs,
            )
            for item in cluster:
                item["alternative_set_ref"] = alternative_set_ref
            alternative_sets.append(
                {
                    "id": alternative_set_ref,
                    "record_type": "physical_component_alternative_set",
                    "record_version": SCHEMA_VERSION,
                    "page": page_number,
                    "state": "candidate_set",
                    "physical_scope_ref": physical_scope_ref,
                    "source_view_ref": source_view_ref,
                    "projection_role": projection_role,
                    "hypothesis_refs": hypothesis_refs,
                    "profile_refs": profile_refs,
                    "basis": "metric_topology_equivalence_without_additive_count",
                    "additive_count_interpretation": "unresolved",
                    "evidence_refs": sorted({*hypothesis_refs, *profile_refs}),
                    "quantity_eligible": False,
                }
            )

    for physical_scope_ref, hypothesis_refs in sorted(hypotheses_by_scope.items()):
        reason_code, reason, blocker_refs = _signed_transform_blocker(
            physical_scope_ref, object_scopes, coordinates
        )
        abstentions.append(
            {
                "id": _stable_id("physical_component_hypothesis_abstention", page_number, physical_scope_ref, reason_code, *sorted(hypothesis_refs), *blocker_refs),
                "record_type": "physical_component_hypothesis_abstention",
                "record_version": SCHEMA_VERSION,
                "page": page_number,
                "state": "abstained",
                "physical_scope_ref": physical_scope_ref,
                "reason_code": reason_code,
                "reason": reason,
                "profile_refs": sorted(
                    {ref for item in hypotheses if item["physical_scope_ref"] == physical_scope_ref for ref in item["profile_refs"]}
                ),
                "surviving_hypothesis_refs": sorted(hypothesis_refs),
                "required_next_certificate": (
                    "signed_orientation"
                    if reason_code == "signed_orientation_certificate_contradiction"
                    else "signed_orientation_or_unsigned_bounded_sweep"
                    if reason_code in {
                        "mirrored_contour_transform_unresolved",
                        "signed_orientation_certificate_unresolved",
                    }
                    else "cross_view_identity_and_placement"
                ),
                "evidence_refs": sorted({physical_scope_ref, *blocker_refs, *hypothesis_refs}),
                "quantity_eligible": False,
            }
        )

    hypotheses = sorted(hypotheses, key=lambda item: item["id"])
    alternative_sets = sorted(alternative_sets, key=lambda item: item["id"])
    abstentions = sorted(abstentions, key=lambda item: item["id"])
    if hypotheses:
        status = "insufficient_constraints"
        generation_status = "bounded_alternatives"
    elif abstentions:
        status = "abstained"
        generation_status = "no_bounded_hypotheses"
    else:
        status = "abstained"
        generation_status = "no_eligible_inputs"
        abstentions.append(
            {
                "id": _stable_id("physical_component_hypothesis_abstention", page_number, "no_eligible_step5_inputs"),
                "record_type": "physical_component_hypothesis_abstention",
                "record_version": SCHEMA_VERSION,
                "page": page_number,
                "state": "abstained",
                "reason_code": "no_eligible_step5_inputs",
                "reason": "no accepted profile-to-physical-scope bindings are eligible for Step 5",
                "profile_refs": [],
                "surviving_hypothesis_refs": [],
                "evidence_refs": [],
                "quantity_eligible": False,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "physical_component_hypothesis_generation",
        "page": page_number,
        "status": status,
        "generation_status": generation_status,
        "hypotheses": hypotheses,
        "alternative_sets": alternative_sets,
        "abstentions": abstentions,
        "summary": {
            "eligible_profile_count": sum(len(item["profile_refs"]) for item in hypotheses),
            "hypothesis_count": len(hypotheses),
            "alternative_set_count": len(alternative_sets),
            "multi_profile_hypothesis_count": sum(len(item["profile_refs"]) > 1 for item in hypotheses),
            "abstention_count": len(abstentions),
            "accepted_component_count": 0,
        },
        "contract": {
            "topology_metric_extent_shared_edge_and_projection_role_required": True,
            "bounded_hypothesis_count": max_hypotheses,
            "stair_specific_grouping_used": False,
            "separate_profile_implies_additive_count": False,
            "additive_component_identity_established": False,
            "physical_component_refs_assigned": False,
            "physical_transforms_resolved": False,
            "step4_kernel_invocation_authorized": False,
            "quantity_eligible": False,
            "schedule_values_used": False,
        },
    }


def validate_physical_component_hypotheses(payload: Mapping[str, Any]) -> list[str]:
    """Validate the Slice 1 trust boundary."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("layer") != "physical_component_hypothesis_generation":
        errors.append("layer must be physical_component_hypothesis_generation")
    contract = payload.get("contract", {}) or {}
    for field in (
        "stair_specific_grouping_used",
        "separate_profile_implies_additive_count",
        "additive_component_identity_established",
        "physical_component_refs_assigned",
        "physical_transforms_resolved",
        "step4_kernel_invocation_authorized",
        "quantity_eligible",
        "schedule_values_used",
    ):
        if contract.get(field) is not False:
            errors.append(f"contract.{field} must be false")
    hypothesis_ids = []
    assigned_profiles = []
    for index, item in enumerate(payload.get("hypotheses", []) or []):
        prefix = f"hypotheses[{index}]"
        hypothesis_ids.append(str(item.get("id")))
        assigned_profiles.extend(map(str, item.get("profile_refs", []) or []))
        if item.get("record_type") != "physical_component_hypothesis":
            errors.append(f"{prefix}.record_type must be physical_component_hypothesis")
        if item.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version must be {SCHEMA_VERSION}")
        if item.get("state") != "candidate":
            errors.append(f"{prefix}.state must remain candidate")
        if not item.get("profile_refs") or not item.get("binding_refs"):
            errors.append(f"{prefix} requires profile and binding refs")
        certificate = item.get("compatibility_certificate", {}) or {}
        for gate in ("topology", "metric_extents", "shared_edges", "projection_role"):
            if not isinstance(certificate.get(gate), Mapping) or not certificate[gate].get("status"):
                errors.append(f"{prefix}.compatibility_certificate.{gate} is required")
        if item.get("additive_component_identity_established") is not False:
            errors.append(f"{prefix} cannot establish additive component identity")
        if item.get("physical_component_ref") is not None:
            errors.append(f"{prefix}.physical_component_ref must remain unresolved")
        if item.get("physical_transform_refs") != []:
            errors.append(f"{prefix}.physical_transform_refs must remain empty")
        if item.get("step4_kernel_invocation_eligible") is not False:
            errors.append(f"{prefix} cannot authorize Step 4")
        if item.get("quantity_eligible") is not False:
            errors.append(f"{prefix} must remain quantity-ineligible")
        if not item.get("alternative_set_ref"):
            errors.append(f"{prefix}.alternative_set_ref is required")
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        errors.append("hypothesis ids must be globally unique")
    if len(assigned_profiles) != len(set(assigned_profiles)):
        errors.append("each eligible profile must occur in exactly one candidate hypothesis")
    sets_by_id = {
        str(item.get("id")): item for item in payload.get("alternative_sets", []) or []
    }
    if len(sets_by_id) != len(payload.get("alternative_sets", []) or []):
        errors.append("alternative set ids must be globally unique")
    referenced_hypotheses = []
    for index, item in enumerate(payload.get("alternative_sets", []) or []):
        prefix = f"alternative_sets[{index}]"
        referenced_hypotheses.extend(map(str, item.get("hypothesis_refs", []) or []))
        if item.get("record_type") != "physical_component_alternative_set":
            errors.append(f"{prefix}.record_type must be physical_component_alternative_set")
        if item.get("record_version") != SCHEMA_VERSION:
            errors.append(f"{prefix}.record_version must be {SCHEMA_VERSION}")
        if item.get("state") != "candidate_set":
            errors.append(f"{prefix}.state must remain candidate_set")
        if not item.get("hypothesis_refs") or not item.get("profile_refs"):
            errors.append(f"{prefix} requires hypothesis and profile refs")
        if item.get("additive_count_interpretation") != "unresolved":
            errors.append(f"{prefix} cannot establish additive count")
        if item.get("quantity_eligible") is not False:
            errors.append(f"{prefix} must remain quantity-ineligible")
    if sorted(referenced_hypotheses) != sorted(hypothesis_ids):
        errors.append("alternative sets must partition candidate hypotheses exactly once")
    for item in payload.get("hypotheses", []) or []:
        alternative = sets_by_id.get(str(item.get("alternative_set_ref")))
        if alternative is None or str(item.get("id")) not in set(map(str, alternative.get("hypothesis_refs", []) or [])):
            errors.append(f"{item.get('id')}.alternative_set_ref does not point to a matching set")
    if payload.get("hypotheses") and payload.get("status") != "insufficient_constraints":
        errors.append("bounded Slice 1 hypotheses must remain insufficient_constraints")
    if any(item.get("state") == "accepted" for item in payload.get("hypotheses", []) or []):
        errors.append("Slice 1 cannot accept a physical component")
    return errors
