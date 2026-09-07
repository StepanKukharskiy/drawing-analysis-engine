"""M4 page-local MEP identity and attribute binding.

This layer consumes the frozen M2 proposal and M3 projected-route contracts.
It builds page-local route scopes, then binds proposals only through explicit,
unique geometric target evidence.  Bindings stop at branches and published
size/elevation change points.  Continuation symbols bind to one terminal
endpoint only; cross-sheet joining remains an M5 concern.

The output is deliberately quantity-free.  It cannot publish installed
length, fitting counts, physical continuation, clash status, or takeoff lines.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from src.drawing_engine.disciplines.mep.mep_outlined_route_composites import validate_mep_outlined_route_composites
from src.drawing_engine.disciplines.mep.mep_route_observations import validate_mep_route_graph
from src.drawing_engine.disciplines.mep.mep_terminology_proposals import validate_mep_terminology_proposals


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_page_local_attribute_bindings"
METHOD_VERSION = "1.0.0"

_ATTRIBUTE_RELATIONS = {"route_system", "route_size", "route_elevation"}
_CHANGE_RELATIONS = {"route_size": "size", "route_elevation": "elevation"}
_PHYSICAL_ENDPOINT_RELATIONS = {
    "physical_terminal",
    "package_boundary",
    "detail_section_interface",
}
_NEGATIVE_TARGET_GATES = {
    "no_unique_geometric_target",
    "nearby_equipment_without_port_connectivity",
    "ambiguous_inline_symbol_target",
    "broad_region_without_unique_target",
    "unconnected_crossing",
    "annotation_claim_only_no_target",
}
_ALLOWED_TARGETS = {
    "route_system": {"route_fragment", "route_fragment_set", "route_composite"},
    "route_size": {"route_fragment", "route_fragment_set", "route_composite"},
    "route_elevation": {"route_fragment", "route_fragment_set", "route_composite"},
    "equipment_endpoint": {"route_endpoint"},
    "valve": {"route_fragment", "route_vertex"},
    "fitting": {"route_fragment", "route_vertex"},
    "damper": {"route_fragment", "route_vertex"},
    "riser_drop": {"route_endpoint", "route_vertex"},
    "service_zone": {
        "route_fragment",
        "route_fragment_set",
        "route_endpoint",
        "route_vertex",
    },
    "continuation_endpoint": {"route_endpoint"},
    "physical_terminal": {"route_endpoint"},
    "package_boundary": {"route_endpoint"},
    "detail_section_interface": {"route_endpoint"},
}


def _stable_id(kind: str, *parts: object) -> str:
    encoded = json.dumps(
        parts, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{kind}.{hashlib.sha256(encoded).hexdigest()[:20]}"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relation_type(proposal: Mapping[str, Any]) -> str | None:
    proposal_type = proposal.get("proposal_type")
    candidate = proposal.get("candidate", {})
    kind = str(candidate.get("kind") or "")
    category = str(candidate.get("category") or "")
    if proposal_type == "system":
        return "route_system"
    if proposal_type == "inline_size":
        return "route_size"
    if proposal_type == "elevation":
        return "route_elevation"
    if proposal_type == "equipment":
        return "equipment_endpoint"
    if proposal_type == "vertical_transition" and kind in {"riser", "drop"}:
        return "riser_drop"
    if proposal_type == "service_zone":
        return "service_zone"
    if proposal_type == "continuation":
        return "continuation_endpoint"
    if proposal_type == "terminal":
        if kind == "capped_physical_terminal":
            return "physical_terminal"
        if kind == "drawing_scope_boundary":
            return "package_boundary"
        if kind == "detail_section_interface":
            return "detail_section_interface"
    if proposal_type == "component":
        if category == "valve" or "valve" in kind:
            return "valve"
        if category == "fitting" or kind in {
            "fitting", "elbow", "tee", "reducer", "coupling"
        }:
            return "fitting"
        if category == "damper" or "damper" in kind:
            return "damper"
    return None


def _components(
    fragment_ids: Iterable[str],
    vertices: Iterable[Mapping[str, Any]],
    *,
    blocked_vertex_refs: set[str] | None = None,
) -> list[list[str]]:
    fragments = sorted(set(fragment_ids))
    fragment_set = set(fragments)
    parent = {fragment_ref: fragment_ref for fragment_ref in fragments}
    blocked = blocked_vertex_refs or set()

    def find(fragment_ref: str) -> str:
        while parent[fragment_ref] != fragment_ref:
            parent[fragment_ref] = parent[parent[fragment_ref]]
            fragment_ref = parent[fragment_ref]
        return fragment_ref

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    for vertex in vertices:
        if str(vertex.get("id")) in blocked:
            continue
        incident = sorted(
            set(str(ref) for ref in vertex.get("fragment_refs", [])) & fragment_set
        )
        for fragment_ref in incident[1:]:
            union(incident[0], fragment_ref)
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for fragment_ref in fragments:
        groups[find(fragment_ref)].append(fragment_ref)
    return sorted((sorted(rows) for rows in groups.values()), key=lambda rows: rows)


def _scope_records(
    page: Mapping[str, Any],
    groups: Iterable[Iterable[str]],
    *,
    record_type: str,
    id_kind: str,
    attribute_kind: str | None = None,
    change_relation_refs: Mapping[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    page_ref = str(page["page_ref"])
    vertices = list(page.get("vertices", []))
    endpoints = list(page.get("endpoints", []))
    output = []
    for group in groups:
        fragment_refs = sorted(set(str(ref) for ref in group))
        fragment_set = set(fragment_refs)
        vertex_refs = sorted(
            str(vertex["id"])
            for vertex in vertices
            if fragment_set.intersection(str(ref) for ref in vertex.get("fragment_refs", []))
        )
        boundary_vertex_refs = sorted(
            str(vertex["id"])
            for vertex in vertices
            if str(vertex["id"]) in vertex_refs
            and (
                len(
                    fragment_set.intersection(
                        str(ref) for ref in vertex.get("fragment_refs", [])
                    )
                )
                == 1
                or any(
                    str(ref) not in fragment_set
                    for ref in vertex.get("fragment_refs", [])
                )
            )
        )
        scope_id = _stable_id(id_kind, page_ref, attribute_kind, fragment_refs)
        output.append(
            {
                "record_type": record_type,
                "record_version": SCHEMA_VERSION,
                "id": scope_id,
                "page_ref": page_ref,
                "attribute_kind": attribute_kind,
                "fragment_refs": fragment_refs,
                "vertex_refs": vertex_refs,
                "boundary_vertex_refs": boundary_vertex_refs,
                "endpoint_refs": sorted(
                    str(endpoint["id"])
                    for endpoint in endpoints
                    if str(endpoint.get("fragment_ref")) in fragment_set
                    and str(endpoint.get("vertex_ref")) in boundary_vertex_refs
                ),
                "change_relation_refs": sorted(
                    {
                        relation_ref
                        for fragment_ref in fragment_refs
                        for relation_ref in (change_relation_refs or {}).get(
                            fragment_ref, []
                        )
                    }
                ),
                "state": "derived",
                "page_local_only": True,
                "quantity_eligible": False,
            }
        )
    return sorted(output, key=lambda item: item["id"])


def build_page_local_route_scopes(
    page: Mapping[str, Any],
    outlined_route_composites: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return maximal M3 chains that do not propagate through M3 branches."""

    composites = sorted(
        (
            composite
            for composite in outlined_route_composites
            if composite.get("state") == "accepted"
            and str(composite.get("page_ref")) == str(page["page_ref"])
        ),
        key=lambda row: str(row["id"]),
    )
    consumed = {
        str(ref)
        for composite in composites
        for ref in [
            *composite.get("member_fragment_refs", []),
            *composite.get("supporting_closure_fragment_refs", []),
        ]
    }
    fragment_ids = [
        str(fragment["id"])
        for fragment in page.get("fragments", [])
        if str(fragment["id"]) not in consumed
    ]
    branch_vertices = {
        str(branch["vertex_ref"]) for branch in page.get("branches", [])
    }
    groups = _components(
        fragment_ids,
        page.get("vertices", []),
        blocked_vertex_refs=branch_vertices,
    )
    ordinary = _scope_records(
        page,
        groups,
        record_type="mep_page_local_route_scope",
        id_kind="mep_page_local_route_scope",
    )
    composite_scopes = []
    for composite in composites:
        fragment_refs = sorted(str(ref) for ref in composite["member_fragment_refs"])
        composite_scopes.append(
            {
                "record_type": "mep_page_local_route_scope",
                "record_version": SCHEMA_VERSION,
                "id": _stable_id(
                    "mep_page_local_route_scope",
                    page["page_ref"],
                    "outlined_route_composite",
                    composite["id"],
                ),
                "page_ref": str(page["page_ref"]),
                "attribute_kind": None,
                "fragment_refs": fragment_refs,
                "vertex_refs": [],
                "boundary_vertex_refs": [],
                "endpoint_refs": [],
                "change_relation_refs": [],
                "route_composite_ref": str(composite["id"]),
                "centreline_points_display": deepcopy(
                    composite.get("derived_geometry", {}).get(
                        "centreline_points_display", []
                    )
                ),
                "corridor_width_display_points": composite.get(
                    "derived_geometry", {}
                ).get("corridor_width_display_points"),
                "state": "derived",
                "page_local_only": True,
                "quantity_eligible": False,
            }
        )
    return sorted([*ordinary, *composite_scopes], key=lambda row: row["id"])


def _page_indexes(
    page: Mapping[str, Any],
    composites: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    fragments = {str(item["id"]): item for item in page.get("fragments", [])}
    vertices = {str(item["id"]): item for item in page.get("vertices", [])}
    endpoints = {str(item["id"]): item for item in page.get("endpoints", [])}
    return {
        "fragments": fragments,
        "vertices": vertices,
        "endpoints": endpoints,
        "composites": {
            str(item["id"]): item
            for item in composites
            if item.get("state") == "accepted"
            and str(item.get("page_ref")) == str(page["page_ref"])
        },
    }


def _target_fragments(
    evidence: Mapping[str, Any], indexes: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    target_kind = str(evidence.get("target_kind") or "")
    target_refs = sorted(set(str(ref) for ref in evidence.get("target_refs", [])))
    errors = []
    fragments = indexes["fragments"]
    vertices = indexes["vertices"]
    endpoints = indexes["endpoints"]
    composites = indexes.get("composites", {})
    if target_kind == "route_fragment":
        if len(target_refs) != 1 or target_refs[0] not in fragments:
            errors.append("route_fragment_target_is_not_unique_and_known")
        return ([target_refs[0]] if not errors else []), errors
    if target_kind == "route_fragment_set":
        if not target_refs or any(ref not in fragments for ref in target_refs):
            errors.append("route_fragment_set_contains_unknown_target")
        return (target_refs if not errors else []), errors
    if target_kind == "route_endpoint":
        if len(target_refs) != 1 or target_refs[0] not in endpoints:
            errors.append("route_endpoint_target_is_not_unique_and_known")
            return [], errors
        return [str(endpoints[target_refs[0]]["fragment_ref"])], errors
    if target_kind == "route_vertex":
        if len(target_refs) != 1 or target_refs[0] not in vertices:
            errors.append("route_vertex_target_is_not_unique_and_known")
            return [], errors
        return sorted(str(ref) for ref in vertices[target_refs[0]]["fragment_refs"]), errors
    if target_kind == "route_composite":
        if len(target_refs) != 1 or target_refs[0] not in composites:
            errors.append("route_composite_target_is_not_unique_and_accepted")
            return [], errors
        return sorted(
            str(ref) for ref in composites[target_refs[0]]["member_fragment_refs"]
        ), errors
    errors.append("unsupported_geometric_target_kind")
    return [], errors


def _semantic_candidate(candidate: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            key: value
            for key, value in candidate.items()
            if key not in {"raw_text", "terminology_entry_ref", "observed_symbol_kind"}
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _relation_authority() -> dict[str, bool]:
    return {
        "page_local_binding_established": False,
        "physical_continuation_established": False,
        "cross_sheet_join_established": False,
        "installed_length_emitted": False,
        "fitting_count_emitted": False,
        "confirmed_clash_established": False,
        "quantity_eligible": False,
    }


def _preliminary_relation(
    proposal: Mapping[str, Any],
    evidence_rows: list[Mapping[str, Any]],
    pages: Mapping[str, Mapping[str, Any]],
    route_scopes_by_page: Mapping[str, list[Mapping[str, Any]]],
    composites_by_page: Mapping[str, list[Mapping[str, Any]]],
    known_evidence_refs: set[str],
) -> dict[str, Any]:
    proposal_ref = str(proposal["id"])
    page_ref = str(proposal.get("page_ref") or "")
    relation_type = _relation_type(proposal)
    reasons = []
    target_kind: str | None = None
    target_refs: list[str] = []
    target_fragment_refs: list[str] = []
    route_scope_refs: list[str] = []
    selected_evidence_refs: list[str] = []
    abstention_evidence_refs: list[str] = []
    candidate = deepcopy(dict(proposal.get("candidate", {})))
    certificates = {
        "upstream_proposal_is_unconflicted": False,
        "unique_page_scope": False,
        "unique_geometric_target": False,
        "target_is_topologically_connected": False,
        "branch_coverage_is_explicit_or_not_required": False,
        "equipment_port_connectivity_is_explicit_or_not_applicable": False,
        "continuation_target_is_unique_terminal_endpoint_or_not_applicable": False,
    }

    if relation_type is None:
        reasons.append("proposal_kind_is_outside_m4_binding_contract")
    if proposal.get("state") != "proposed" or proposal.get("conflicts"):
        reasons.append("upstream_proposal_abstained_or_conflicted")
    else:
        certificates["upstream_proposal_is_unconflicted"] = True
    page = pages.get(page_ref)
    if not page_ref or page is None:
        reasons.append("proposal_does_not_resolve_to_one_m3_page")
        indexes = {"fragments": {}, "vertices": {}, "endpoints": {}, "composites": {}}
    else:
        certificates["unique_page_scope"] = True
        indexes = _page_indexes(page, composites_by_page.get(page_ref, []))

    valid_rows = []
    invalid_evidence_reasons = []
    for row in evidence_rows:
        row_reasons = []
        if str(row.get("page_ref") or "") != page_ref:
            row_reasons.append("binding_evidence_page_mismatch")
        if row.get("state") != "observed":
            row_reasons.append("binding_evidence_is_not_observed")
        method = row.get("method", {})
        if not method.get("name") or not method.get("version"):
            row_reasons.append("binding_evidence_lacks_method_version")
        geometric_refs = [str(ref) for ref in row.get("geometric_evidence_refs", [])]
        if not geometric_refs or any(ref not in known_evidence_refs for ref in geometric_refs):
            row_reasons.append("geometric_evidence_refs_are_not_closed")
        if row.get("disposition") == "abstain":
            negative_reasons = sorted(
                set(str(reason) for reason in row.get("negative_gate_reasons", []))
            )
            if row.get("target_kind") is not None or row.get("target_refs"):
                row_reasons.append("negative_binding_evidence_cannot_name_a_target")
            if not negative_reasons or any(
                reason not in _NEGATIVE_TARGET_GATES for reason in negative_reasons
            ):
                row_reasons.append("negative_binding_evidence_gate_is_invalid")
            if row_reasons:
                invalid_evidence_reasons.extend(row_reasons)
            else:
                reasons.extend(negative_reasons)
                abstention_evidence_refs.append(str(row.get("id")))
            continue
        if not set(str(ref) for ref in row.get("target_refs", [])).issubset(
            geometric_refs
        ):
            row_reasons.append("target_refs_are_not_preserved_as_geometric_evidence")
        fragments, target_errors = _target_fragments(row, indexes)
        row_reasons.extend(target_errors)
        if relation_type is not None and row.get("target_kind") not in _ALLOWED_TARGETS.get(
            relation_type, set()
        ):
            row_reasons.append("target_kind_is_invalid_for_relation")
        if not row_reasons:
            valid_rows.append((row, fragments))
        else:
            invalid_evidence_reasons.extend(row_reasons)

    if invalid_evidence_reasons:
        reasons.extend(invalid_evidence_reasons)

    signatures = {
        (
            str(row.get("target_kind")),
            tuple(sorted(str(ref) for ref in row.get("target_refs", []))),
        )
        for row, _fragments in valid_rows
    }
    if len(signatures) != 1:
        reasons.append(
            "no_valid_geometric_target"
            if not signatures
            else "multiple_geometric_targets"
        )
    else:
        certificates["unique_geometric_target"] = True
        selected = [
            (row, fragments)
            for row, fragments in valid_rows
            if (
                str(row.get("target_kind")),
                tuple(sorted(str(ref) for ref in row.get("target_refs", []))),
            )
            == next(iter(signatures))
        ]
        row, target_fragment_refs = selected[0]
        target_kind = str(row["target_kind"])
        target_refs = sorted(str(ref) for ref in row.get("target_refs", []))
        selected_evidence_refs = sorted(str(item[0]["id"]) for item in selected)

        if target_kind == "route_composite":
            connected = bool(target_refs) and target_refs[0] in indexes["composites"]
        else:
            full_components = _components(
                indexes["fragments"], page.get("vertices", []) if page else []
            )
            component_by_fragment = {
                fragment_ref: index
                for index, component in enumerate(full_components)
                for fragment_ref in component
            }
            connected = bool(target_fragment_refs) and len(
                {component_by_fragment.get(ref) for ref in target_fragment_refs}
            ) == 1
        if not connected:
            reasons.append("target_fragments_are_not_topologically_connected")
        else:
            certificates["target_is_topologically_connected"] = True

        scopes = route_scopes_by_page.get(page_ref, [])
        scope_by_fragment = {
            str(fragment_ref): str(scope["id"])
            for scope in scopes
            for fragment_ref in scope.get("fragment_refs", [])
        }
        route_scope_refs = sorted(
            {scope_by_fragment[ref] for ref in target_fragment_refs if ref in scope_by_fragment}
        )
        crosses_branch = len(route_scope_refs) > 1
        explicit_branch_coverage = all(
            bool(item[0].get("explicit_branch_coverage")) for item in selected
        )
        if relation_type in _ATTRIBUTE_RELATIONS and crosses_branch and not explicit_branch_coverage:
            reasons.append("branch_coverage_is_not_explicit")
        else:
            certificates["branch_coverage_is_explicit_or_not_required"] = True

        if relation_type == "equipment_endpoint":
            port_evidence = {
                str(ref)
                for item in selected
                for ref in item[0].get("port_geometry_evidence_refs", [])
            }
            if (
                not all(bool(item[0].get("explicit_port_connectivity")) for item in selected)
                or not port_evidence
                or any(ref not in known_evidence_refs for ref in port_evidence)
                or not set(target_refs).issubset(port_evidence)
            ):
                reasons.append("equipment_port_connectivity_is_not_explicit")
            else:
                certificates[
                    "equipment_port_connectivity_is_explicit_or_not_applicable"
                ] = True
        else:
            certificates[
                "equipment_port_connectivity_is_explicit_or_not_applicable"
            ] = True

        if relation_type == "continuation_endpoint":
            endpoint = indexes["endpoints"].get(target_refs[0]) if len(target_refs) == 1 else None
            vertex = (
                indexes["vertices"].get(str(endpoint.get("vertex_ref")))
                if endpoint is not None
                else None
            )
            if endpoint is None or vertex is None or int(vertex.get("degree", 0)) != 1:
                reasons.append("continuation_target_is_not_one_terminal_endpoint")
            else:
                certificates[
                    "continuation_target_is_unique_terminal_endpoint_or_not_applicable"
                ] = True
        else:
            certificates[
                "continuation_target_is_unique_terminal_endpoint_or_not_applicable"
            ] = True

        if relation_type in _PHYSICAL_ENDPOINT_RELATIONS:
            certificates[
                "physical_endpoint_index_is_explicit_or_not_applicable"
            ] = False
            certificates[
                "physical_endpoint_is_unique_degree_one_or_not_applicable"
            ] = False
            endpoint = indexes["endpoints"].get(
                target_refs[0]
            ) if len(target_refs) == 1 else None
            vertex = (
                indexes["vertices"].get(str(endpoint.get("vertex_ref")))
                if endpoint is not None else None
            )
            if endpoint is None or vertex is None or int(vertex.get("degree", 0)) != 1:
                reasons.append("physical_terminal_target_is_not_one_degree_one_endpoint")
            else:
                certificates[
                    "physical_endpoint_is_unique_degree_one_or_not_applicable"
                ] = True
            endpoint_indexes = {
                item[0].get("projected_endpoint_index") for item in selected
            }
            if endpoint_indexes not in ({0}, {1}):
                reasons.append("physical_endpoint_index_is_not_explicit_and_unique")
            else:
                endpoint_index = next(iter(endpoint_indexes))
                candidate["endpoint_index"] = endpoint_index
                candidate["terminal_kind"] = {
                    "physical_terminal": "physical_terminal",
                    "package_boundary": "package_boundary",
                    "detail_section_interface": "detail_section_interface",
                }[relation_type]
                certificates[
                    "physical_endpoint_index_is_explicit_or_not_applicable"
                ] = True

    state = "accepted" if not reasons else "abstained"
    authority = _relation_authority()
    authority["page_local_binding_established"] = state == "accepted"
    relation = {
        "record_type": "mep_page_local_binding_relation",
        "record_version": SCHEMA_VERSION,
        "id": _stable_id(
            "mep_page_local_binding_relation",
            proposal_ref,
            relation_type,
            target_kind,
            target_refs,
        ),
        "page_ref": page_ref,
        "proposal_ref": proposal_ref,
        "relation_type": relation_type,
        "candidate": candidate,
        "target_kind": target_kind,
        "target_refs": target_refs,
        "target_fragment_refs": sorted(target_fragment_refs),
        "route_scope_refs": route_scope_refs,
        "applicability_scope_refs": [],
        "binding_evidence_refs": selected_evidence_refs,
        "proposal_evidence_refs": sorted(
            str(ref) for ref in proposal.get("evidence_refs", [])
        ),
        "method": {
            "name": "unique_page_scope_and_geometric_target_binding",
            "version": METHOD_VERSION,
        },
        "state": state,
        "epistemic_state": "derived" if state == "accepted" else "unknown",
        "reasons": sorted(set(reasons)),
        "conflicts": [],
        "certificates": certificates,
        "authority": authority,
        "quantity_eligible": False,
    }
    if abstention_evidence_refs:
        relation["abstention_evidence_refs"] = sorted(abstention_evidence_refs)
    return relation


def _apply_relation_conflicts(relations: list[dict[str, Any]]) -> None:
    for index, left in enumerate(relations):
        if left["state"] != "accepted" or left["relation_type"] is None:
            continue
        for right in relations[index + 1 :]:
            if right["state"] != "accepted" or right["relation_type"] != left["relation_type"]:
                continue
            if left["page_ref"] != right["page_ref"]:
                continue
            if left["relation_type"] == "route_system":
                same_target = bool(
                    set(left["route_scope_refs"]) & set(right["route_scope_refs"])
                )
            else:
                same_target = bool(
                    set(left["target_refs"]) & set(right["target_refs"])
                    or set(left["target_fragment_refs"])
                    & set(right["target_fragment_refs"])
                )
            if not same_target or _semantic_candidate(left["candidate"]) == _semantic_candidate(
                right["candidate"]
            ):
                continue
            left["conflicts"].append(right["id"])
            right["conflicts"].append(left["id"])
    for relation in relations:
        if relation["conflicts"]:
            relation["conflicts"] = sorted(set(relation["conflicts"]))
            relation["state"] = "abstained"
            relation["epistemic_state"] = "unknown"
            relation["reasons"] = sorted(
                set([*relation["reasons"], "conflicting_bound_annotations"])
            )
            relation["authority"]["page_local_binding_established"] = False


def _attribute_scopes(
    page: Mapping[str, Any],
    relations: list[dict[str, Any]],
    composites: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    composite_by_id = {
        str(row["id"]): row
        for row in composites
        if row.get("state") == "accepted"
        and str(row.get("page_ref")) == str(page["page_ref"])
    }
    consumed = {
        str(ref)
        for composite in composite_by_id.values()
        for ref in [
            *composite.get("member_fragment_refs", []),
            *composite.get("supporting_closure_fragment_refs", []),
        ]
    }
    fragment_ids = [
        str(fragment["id"])
        for fragment in page.get("fragments", [])
        if str(fragment["id"]) not in consumed
    ]
    fragment_by_id = {str(fragment["id"]): fragment for fragment in page.get("fragments", [])}
    branch_vertices = {
        str(branch["vertex_ref"]) for branch in page.get("branches", [])
    }
    output = []
    for relation_type, attribute_kind in _CHANGE_RELATIONS.items():
        accepted = [
            relation
            for relation in relations
            if relation["page_ref"] == str(page["page_ref"])
            and relation["relation_type"] == relation_type
            and relation["state"] == "accepted"
        ]
        composite_relations = [
            relation
            for relation in accepted
            if relation.get("target_kind") == "route_composite"
            and len(relation.get("target_refs", [])) == 1
            and str(relation["target_refs"][0]) in composite_by_id
        ]
        accepted = [
            relation
            for relation in accepted
            if relation not in composite_relations
        ]
        change_vertices = set(branch_vertices)
        changes_by_fragment: defaultdict[str, list[str]] = defaultdict(list)
        for relation in accepted:
            for fragment_ref in relation["target_fragment_refs"]:
                fragment = fragment_by_id[fragment_ref]
                change_vertices.update(str(ref) for ref in fragment["endpoint_vertex_refs"])
                changes_by_fragment[fragment_ref].append(relation["id"])
        groups = _components(
            fragment_ids,
            page.get("vertices", []),
            blocked_vertex_refs=change_vertices,
        )
        scopes = _scope_records(
            page,
            groups,
            record_type="mep_attribute_applicability_scope",
            id_kind="mep_attribute_applicability_scope",
            attribute_kind=attribute_kind,
            change_relation_refs=changes_by_fragment,
        )
        output.extend(scopes)
        scope_by_fragment = {
            str(fragment_ref): str(scope["id"])
            for scope in scopes
            for fragment_ref in scope["fragment_refs"]
        }
        for relation in accepted:
            relation["applicability_scope_refs"] = sorted(
                {
                    scope_by_fragment[fragment_ref]
                    for fragment_ref in relation["target_fragment_refs"]
                }
            )
        # Conflicting candidates have already abstained. Repeated agreeing
        # annotations keep separate relations but share one applicability scope.
        for composite_ref in sorted({str(row["target_refs"][0]) for row in composite_relations}):
            same_scope = [row for row in composite_relations if str(row["target_refs"][0]) == composite_ref]
            composite = composite_by_id[composite_ref]
            scope_id = _stable_id(
                "mep_attribute_applicability_scope",
                page["page_ref"],
                attribute_kind,
                composite_ref,
            )
            output.append(
                {
                    "record_type": "mep_attribute_applicability_scope",
                    "record_version": SCHEMA_VERSION,
                    "id": scope_id,
                    "page_ref": str(page["page_ref"]),
                    "attribute_kind": attribute_kind,
                    "fragment_refs": sorted(
                        str(ref) for ref in composite["member_fragment_refs"]
                    ),
                    "vertex_refs": [],
                    "boundary_vertex_refs": [],
                    "endpoint_refs": [],
                    "change_relation_refs": sorted(str(row["id"]) for row in same_scope),
                    "route_composite_ref": composite_ref,
                    "state": "derived",
                    "page_local_only": True,
                    "quantity_eligible": False,
                }
            )
            for relation in same_scope:
                relation["applicability_scope_refs"] = [scope_id]
    return sorted(output, key=lambda item: item["id"])


def build_mep_attribute_bindings(
    *,
    terminology_proposals: Mapping[str, Any],
    route_graph: Mapping[str, Any],
    binding_evidence: Iterable[Mapping[str, Any]],
    outlined_route_composites: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the isolated M4 page-local relation layer."""

    terminology_errors = validate_mep_terminology_proposals(terminology_proposals)
    if terminology_errors:
        raise ValueError("invalid M2 proposal payload:\n" + "\n".join(terminology_errors))
    route_errors = validate_mep_route_graph(route_graph)
    if route_errors:
        raise ValueError("invalid M3 route payload:\n" + "\n".join(route_errors))
    composite_rows: list[Mapping[str, Any]] = []
    if outlined_route_composites is not None:
        composite_errors = validate_mep_outlined_route_composites(
            outlined_route_composites
        )
        if composite_errors:
            raise ValueError(
                "invalid M3.5 composite payload:\n" + "\n".join(composite_errors)
            )
        if (
            outlined_route_composites.get("m3_contract_ref", {}).get("payload_sha256")
            != _canonical_sha256(route_graph)
        ):
            raise ValueError("M3.5 composite payload does not reference this M3 graph")
        composite_rows = list(
            outlined_route_composites.get("accepted_composites", [])
        )
    m2_document = terminology_proposals.get("document", {})
    m3_document = route_graph.get("document", {})
    if (
        m2_document.get("document_key")
        and m3_document.get("document_key")
        and m2_document.get("document_key") != m3_document.get("document_key")
    ):
        raise ValueError("M2 and M3 document keys do not match")

    evidence_rows = sorted(
        (deepcopy(dict(row)) for row in binding_evidence),
        key=lambda row: str(row.get("id") or ""),
    )
    evidence_ids = [str(row.get("id") or "") for row in evidence_rows]
    if "" in evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("binding evidence IDs must be present and unique")
    proposals = list(terminology_proposals.get("proposals", []))
    proposal_refs = {str(proposal["id"]) for proposal in proposals}
    unknown_proposals = sorted(
        str(row.get("proposal_ref"))
        for row in evidence_rows
        if str(row.get("proposal_ref")) not in proposal_refs
    )
    if unknown_proposals:
        raise ValueError(f"binding evidence references unknown proposals: {unknown_proposals}")

    pages = {str(page["page_ref"]): page for page in route_graph.get("pages", [])}
    composites_by_page: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for composite in composite_rows:
        composites_by_page[str(composite["page_ref"])].append(composite)
    route_scopes_by_page = {
        page_ref: build_page_local_route_scopes(
            page, composites_by_page.get(page_ref, [])
        )
        for page_ref, page in pages.items()
    }
    evidence_by_proposal: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_proposal[str(row["proposal_ref"])].append(row)
    known_evidence_refs = {
        *evidence_ids,
        *(str(row["id"]) for row in terminology_proposals.get("source_observations", [])),
    }
    for page in pages.values():
        for collection in ("fragments", "vertices", "endpoints", "branches", "crossings"):
            for item in page.get(collection, []):
                known_evidence_refs.add(str(item["id"]))
                if item.get("source_primitive_ref"):
                    known_evidence_refs.add(str(item["source_primitive_ref"]))
    for composite in composite_rows:
        known_evidence_refs.add(str(composite["id"]))
        known_evidence_refs.update(
            str(ref)
            for ref in [
                *composite.get("member_fragment_refs", []),
                *composite.get("member_source_primitive_refs", []),
                *composite.get("supporting_closure_fragment_refs", []),
            ]
        )

    relations = [
        _preliminary_relation(
            proposal,
            evidence_by_proposal.get(str(proposal["id"]), []),
            pages,
            route_scopes_by_page,
            composites_by_page,
            known_evidence_refs,
        )
        for proposal in proposals
    ]
    _apply_relation_conflicts(relations)
    applicability_scopes = [
        scope
        for page in pages.values()
        for scope in _attribute_scopes(
            page, relations, composites_by_page.get(str(page["page_ref"]), [])
        )
    ]
    route_scopes = [
        scope
        for page_ref in sorted(route_scopes_by_page)
        for scope in route_scopes_by_page[page_ref]
    ]
    relations.sort(key=lambda item: (item["page_ref"], item["relation_type"] or "", item["id"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "document": deepcopy(dict(m3_document or m2_document)),
        "m2_contract_ref": {
            "schema_version": terminology_proposals.get("schema_version"),
            "layer": terminology_proposals.get("layer"),
            "terminology_pack_id": terminology_proposals.get("terminology_pack", {}).get("id"),
            "terminology_pack_version": terminology_proposals.get("terminology_pack", {}).get("version"),
            "payload_sha256": _canonical_sha256(terminology_proposals),
        },
        "m3_contract_ref": {
            "schema_version": route_graph.get("schema_version"),
            "layer": route_graph.get("layer"),
            "document_key": m3_document.get("document_key"),
            "payload_sha256": _canonical_sha256(route_graph),
        },
        "m3_5_contract_ref": (
            None
            if outlined_route_composites is None
            else {
                "schema_version": outlined_route_composites.get("schema_version"),
                "layer": outlined_route_composites.get("layer"),
                "payload_sha256": _canonical_sha256(outlined_route_composites),
            }
        ),
        "outlined_route_composites": deepcopy(composite_rows),
        "binding_evidence_sha256": _canonical_sha256(evidence_rows),
        "binding_evidence": evidence_rows,
        "route_scopes": route_scopes,
        "attribute_applicability_scopes": applicability_scopes,
        "relations": relations,
        "summary": {
            "page_count": len(pages),
            "route_scope_count": len(route_scopes),
            "attribute_applicability_scope_count": len(applicability_scopes),
            "relation_count": len(relations),
            "accepted_relation_count": sum(row["state"] == "accepted" for row in relations),
            "abstained_relation_count": sum(row["state"] == "abstained" for row in relations),
            "state_counts": dict(sorted(Counter(row["state"] for row in relations).items())),
        },
        "exchange_contract": {
            "page_local_binding_only": True,
            "branch_and_change_point_applicability_split": True,
            "continuations_are_endpoint_relations_only": True,
            "cross_sheet_join_established": False,
            "physical_continuation_established": False,
            "installed_length_emitted": False,
            "fitting_count_emitted": False,
            "confirmed_clash_established": False,
            "schedule_values_used": False,
            "quantity_eligible": False,
        },
        "quantity_eligible": False,
    }
    errors = validate_mep_attribute_bindings(payload)
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


def validate_mep_attribute_bindings(payload: Mapping[str, Any]) -> list[str]:
    """Validate reference closure and the M4 authority boundary."""

    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if payload.get("layer") != LAYER:
        errors.append("layer mismatch")
    for path in (
        ("m2_contract_ref", "payload_sha256"),
        ("m3_contract_ref", "payload_sha256"),
    ):
        value = payload.get(path[0], {}).get(path[1])
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{'.'.join(path)} must be a SHA-256 digest")
    composite_contract = payload.get("m3_5_contract_ref")
    if composite_contract is not None:
        value = composite_contract.get("payload_sha256")
        if not isinstance(value, str) or len(value) != 64:
            errors.append("m3_5_contract_ref.payload_sha256 must be a SHA-256 digest")
    evidence_digest = payload.get("binding_evidence_sha256")
    if evidence_digest != _canonical_sha256(payload.get("binding_evidence", [])):
        errors.append("binding_evidence_sha256 does not replay")
    allowed_false = {
        "physical_continuation_established",
        "cross_sheet_join_established",
        "installed_length_emitted",
        "fitting_count_emitted",
        "confirmed_clash_established",
    }
    forbidden_exact = {
        "installed_length",
        "fitting_count",
        "physical_continuation",
        "confirmed_clash",
        "clash_status",
        "quantity",
        "takeoff",
    }
    forbidden_prefixes = (
        "installed_length_",
        "fitting_count_",
        "physical_continuation_",
        "confirmed_clash_",
        "quantity_",
        "takeoff_",
    )
    for path, key, value in _walk_items(payload):
        location = ".".join((*path, key))
        if key in forbidden_exact or (
            key.startswith(forbidden_prefixes)
            and key not in allowed_false
            and key != "quantity_eligible"
        ):
            errors.append(f"{location}: forbidden M4 engineering output")
        if key in allowed_false and value is not False:
            errors.append(f"{location}: authority flag must remain false")
        if key == "quantity_eligible" and value is not False:
            errors.append(f"{location}: quantity_eligible must remain false")

    route_scopes = list(payload.get("route_scopes", []))
    applicability_scopes = list(payload.get("attribute_applicability_scopes", []))
    relations = list(payload.get("relations", []))
    binding_evidence = list(payload.get("binding_evidence", []))
    composites = list(payload.get("outlined_route_composites", []))
    composite_refs = {str(row.get("id")) for row in composites}
    route_scope_refs = {str(row.get("id")) for row in route_scopes}
    applicability_refs = {str(row.get("id")) for row in applicability_scopes}
    evidence_refs = {str(row.get("id")) for row in binding_evidence}
    relation_refs = {str(row.get("id")) for row in relations}
    if len(route_scope_refs) != len(route_scopes):
        errors.append("route scope IDs must be unique")
    if len(applicability_refs) != len(applicability_scopes):
        errors.append("attribute applicability scope IDs must be unique")
    if len(relation_refs) != len(relations):
        errors.append("relation IDs must be unique")
    if len(composite_refs) != len(composites):
        errors.append("outlined route composite IDs must be unique")
    if composite_contract is None and composites:
        errors.append("outlined route composites lack an M3.5 contract ref")
    for composite in composites:
        if composite.get("state") != "accepted":
            errors.append(f"{composite.get('id')}: embedded composite is not accepted")
    for scope in [*route_scopes, *applicability_scopes]:
        if not scope.get("fragment_refs"):
            errors.append(f"{scope.get('id')}: empty scope")
        if scope.get("page_local_only") is not True:
            errors.append(f"{scope.get('id')}: scope is not page-local")
        if scope.get("route_composite_ref") is not None and str(
            scope.get("route_composite_ref")
        ) not in composite_refs:
            errors.append(f"{scope.get('id')}: unknown outlined route composite")
    for relation in relations:
        relation_ref = str(relation.get("id"))
        if relation.get("record_type") != "mep_page_local_binding_relation":
            errors.append(f"{relation_ref}: invalid record_type")
        if relation.get("relation_type") not in set(_ALLOWED_TARGETS) | {None}:
            errors.append(f"{relation_ref}: invalid relation_type")
        if any(str(ref) not in route_scope_refs for ref in relation.get("route_scope_refs", [])):
            errors.append(f"{relation_ref}: unknown route scope")
        if any(str(ref) not in applicability_refs for ref in relation.get("applicability_scope_refs", [])):
            errors.append(f"{relation_ref}: unknown applicability scope")
        if any(str(ref) not in evidence_refs for ref in relation.get("binding_evidence_refs", [])):
            errors.append(f"{relation_ref}: unknown binding evidence")
        if any(str(ref) not in evidence_refs for ref in relation.get("abstention_evidence_refs", [])):
            errors.append(f"{relation_ref}: unknown abstention evidence")
        if relation.get("state") == "accepted" and relation.get("abstention_evidence_refs"):
            errors.append(f"{relation_ref}: accepted relation retained negative evidence")
        if relation.get("target_kind") == "route_composite" and (
            len(relation.get("target_refs", [])) != 1
            or str(relation.get("target_refs", [None])[0]) not in composite_refs
        ):
            errors.append(f"{relation_ref}: unknown outlined route composite target")
        authority = relation.get("authority", {})
        if relation.get("state") == "accepted":
            if relation.get("reasons") or not relation.get("target_refs"):
                errors.append(f"{relation_ref}: accepted relation lacks a clean target certificate")
            if authority.get("page_local_binding_established") is not True:
                errors.append(f"{relation_ref}: accepted relation lacks page-local authority")
            if relation.get("relation_type") in _CHANGE_RELATIONS and not relation.get(
                "applicability_scope_refs"
            ):
                errors.append(f"{relation_ref}: accepted change relation lacks applicability scope")
            certificates = relation.get("certificates", {})
            if not all(bool(value) for value in certificates.values()):
                errors.append(f"{relation_ref}: accepted relation has an open certificate")
        elif relation.get("state") == "abstained":
            if authority.get("page_local_binding_established") is not False:
                errors.append(f"{relation_ref}: abstention retained binding authority")
            if not relation.get("reasons"):
                errors.append(f"{relation_ref}: abstention lacks reason")
        else:
            errors.append(f"{relation_ref}: invalid state")
    contract = payload.get("exchange_contract", {})
    for key in (
        "page_local_binding_only",
        "branch_and_change_point_applicability_split",
        "continuations_are_endpoint_relations_only",
    ):
        if contract.get(key) is not True:
            errors.append(f"exchange_contract.{key} must be true")
    for key in (
        "cross_sheet_join_established",
        "physical_continuation_established",
        "installed_length_emitted",
        "fitting_count_emitted",
        "confirmed_clash_established",
        "schedule_values_used",
        "quantity_eligible",
    ):
        if contract.get(key) is not False:
            errors.append(f"exchange_contract.{key} must be false")
    return errors


def elevation_applicability_outcomes(*, terminology, bindings, graph, boundary_connections):
    """Separate direct annotation ownership from proposed projected extents.

    This is a read-only M4 scope assessment, not elevation propagation. A
    native boundary join can reconstruct one straight annotation extent only
    when both native sidewalls, unique two-port topology and collinearity close.
    Bends and branches never inherit an elevation. The resulting certificate
    remains projected annotation applicability: it cannot establish physical
    continuation, a vertical datum, or section/detail applicability. Existing
    direct M4 scopes are preserved verbatim. No caller-supplied boolean can
    authorize an extension; unsupported extent evidence stays explicitly open.
    """
    from src.drawing_engine.disciplines.mep.mep_declared_data import _sha256
    from src.drawing_engine.disciplines.mep.mep_boundary_connection_replay import replay_boundary_connections
    from src.drawing_engine.disciplines.mep.mep_native_boundary_connections import _ports

    if (bindings['m2_contract_ref']['payload_sha256'] != _sha256(terminology)
            or bindings['m3_contract_ref']['payload_sha256'] != _sha256(graph)
            or terminology['document'] != graph['document']):
        raise ValueError('elevation applicability requires matching frozen M2/M3/M4')
    connections = replay_boundary_connections(boundary_connections, graph, bindings)
    composites = {r['id']: r for r in bindings['outlined_route_composites']}
    fragments = {r['id']: r for p in graph['pages'] for r in p['fragments']}
    observations = {r['id']: r for r in terminology['source_observations']}
    evidence = {r['id']: r for r in bindings['binding_evidence']}
    scopes = {r['id']: r for r in bindings['attribute_applicability_scopes']}
    relations = defaultdict(list)
    for relation in bindings['relations']:
        relations[relation['proposal_ref']].append(relation)
    incident, ports = defaultdict(list), defaultdict(list)
    for connection in connections:
        for port in connection['ports']:
            incident[port['id']].append(connection)
    for port in _ports(list(composites.values()), fragments):
        ports[port['composite_ref']].append(port)

    def value_key(candidate):
        return tuple(candidate.get(k) for k in ('kind', 'basis', 'value', 'unit'))

    def straight_inline_certificate(connection, left_ref, right_ref):
        """Certify one native-path split inside a straight annotation scope.

        This deliberately does not reuse the broader system/size propagation
        rule: curves, bends and multi-port junctions remain ineligible. The
        connection replay already proves style and native boundary coverage;
        the checks here retain the exact positive evidence needed by the
        elevation-specific scope decision.
        """
        if (connection.get('state') != 'accepted'
                or connection.get('relation_type') != 'projected_collinear_boundary_join'
                or sorted(connection.get('composite_refs', [])) != sorted((left_ref, right_ref))
                or len(connection.get('port_refs', [])) != 2
                or len(connection.get('boundary_paths', [])) != 2
                or connection.get('reasons')
                or connection.get('search', {}).get('complete') is not True
                or any(not row.get('source_primitive_refs')
                       for row in connection.get('boundary_paths', []))):
            return None
        left, right = composites[left_ref], composites[right_ref]
        left_points = left.get('derived_geometry', {}).get('centreline_points_display', [])
        right_points = right.get('derived_geometry', {}).get('centreline_points_display', [])
        if len(left_points) != 2 or len(right_points) != 2:
            return None
        left_vector = [left_points[1][i] - left_points[0][i] for i in (0, 1)]
        right_vector = [right_points[1][i] - right_points[0][i] for i in (0, 1)]
        left_length, right_length = math.hypot(*left_vector), math.hypot(*right_vector)
        if not left_length or not right_length:
            return None
        cross = abs(left_vector[0] * right_vector[1] - left_vector[1] * right_vector[0])
        if cross > 1e-8 * left_length * right_length:
            return None
        left_width = left.get('geometry_metrics', {}).get('mean_separation_display_points')
        right_width = right.get('geometry_metrics', {}).get('mean_separation_display_points')
        if (not isinstance(left_width, (int, float)) or not isinstance(right_width, (int, float))
                or abs(left_width - right_width) > .001):
            return None
        return {'method': 'complete_native_straight_inline_elevation_extent',
            'version': '0.1.0', 'boundary_connection_ref': connection['id'],
            'boundary_path_source_primitive_refs': [
                deepcopy(row['source_primitive_refs']) for row in connection['boundary_paths']],
            'native_query_complete': True, 'unique_two_port_join': True,
            'same_width_collinear_members': True, 'bend_crossed': False,
            'branch_crossed': False, 'physical_continuation_established': False}

    # Include independently anchored changes and unresolved transition
    # proposals. Absence in this retained inventory never establishes absence
    # on the drawing; the search states below remain partial.
    elevations, transitions = defaultdict(list), defaultdict(list)
    for relation in bindings['relations']:
        if relation['state'] == 'accepted' and relation['relation_type'] == 'route_elevation':
            for target in relation['target_refs']:
                elevations[target].append(relation)
        if relation['relation_type'] == 'riser_drop':
            for target in relation.get('target_fragment_refs', []):
                transitions[target].append(relation)
    grouped = {}
    for proposal in terminology['proposals']:
        if proposal['proposal_type'] != 'elevation':
            continue
        source, seen = observations[proposal['anchor_ref']], set()
        while source.get('source_observation_ref') in observations:
            if source['id'] in seen:
                raise ValueError('cyclic elevation observation lineage')
            seen.add(source['id'])
            source = observations[source['source_observation_ref']]
        key = source['id'], value_key(proposal['candidate'])
        group = grouped.setdefault(key, {'source': source, 'candidate': proposal['candidate'], 'relations': []})
        group['relations'].extend(relations[proposal['id']])
    outcomes = []
    for (source_ref, _), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        source, candidate = group['source'], group['candidate']
        direct = [r for r in group['relations'] if r['state'] == 'accepted'
                  and r['target_kind'] == 'route_composite'
                  and all(evidence[ref]['method']['name'] == 'complete_native_dot_leader_contact'
                          for ref in r['binding_evidence_refs']) and r['binding_evidence_refs']]
        starts = sorted({ref for r in direct for ref in r['target_refs']})
        paths = {ref: [] for ref in starts}
        path_certificates = {ref: [] for ref in starts}
        queue, boundaries = list(starts), []
        for target in queue:
            for port in ports[target]:
                rows = incident[port['id']]
                accepted = [r for r in rows if r['state'] == 'accepted']
                reasons, next_refs, extent_certificate = [], [], None
                if len(accepted) != 1 or any(r['state'] != 'accepted' for r in rows):
                    reasons.append('unresolved_projected_destination' if rows else 'retained_geometry_scope_exit')
                elif len(accepted[0]['composite_refs']) != 2:
                    reasons.append('explicit_branch_boundary')
                elif accepted[0].get('relation_type') != 'projected_collinear_boundary_join':
                    reasons.append('bend_or_non_collinear_join_requires_independent_elevation_evidence')
                else:
                    next_refs = [ref for ref in accepted[0]['composite_refs'] if ref != target]
                    if len(next_refs) == 1:
                        extent_certificate = straight_inline_certificate(
                            accepted[0], target, next_refs[0])
                    if extent_certificate is None:
                        reasons.append('straight_inline_elevation_extent_certificate_open')
                transition_rows = {r['id']: r for ref in composites[target]['member_fragment_refs']
                                   for r in transitions[ref]}
                if transition_rows:
                    reasons.append('riser_drop_applicability_requires_resolution')
                change_refs = [r['id'] for ref in next_refs for r in elevations[ref]
                               if value_key(r['candidate']) != value_key(candidate)]
                if change_refs:
                    reasons.append('different_direct_elevation_or_basis')
                if not reasons:
                    for ref in next_refs:
                        if ref not in paths:
                            paths[ref] = paths[target] + [accepted[0]['id']]
                            path_certificates[ref] = path_certificates[target] + [extent_certificate]
                            queue.append(ref)
                boundaries.append({'port_ref': port['id'], 'composite_ref': target,
                    'point_display': deepcopy(port['center']), 'candidate_next_target_refs': next_refs,
                    'incident_connection_refs': sorted(r['id'] for r in rows),
                    'change_relation_refs': sorted(change_refs), 'riser_drop_relation_refs': sorted(transition_rows),
                    'candidate_extent_stops_here': bool(reasons),
                    'reason_codes': reasons,
                    'straight_inline_extent_certificate': deepcopy(extent_certificate),
                    'annotation_applicability_established': not reasons and extent_certificate is not None,
                    'elevation_transfer_established': False})
        targets = []
        for target, path in sorted(paths.items()):
            composite = composites[target]
            local = [r for r in direct if target in r['target_refs']]
            extended = bool(path) and not local
            targets.append({'target_ref': target, 'points_display': deepcopy(composite['derived_geometry']['centreline_points_display']),
                'source_primitive_refs': deepcopy(composite['member_source_primitive_refs']),
                'projected_path_connection_refs': path,
                'straight_inline_extent_certificates': deepcopy(path_certificates[target]),
                'state': ('existing_direct_binding' if local
                          else 'accepted_straight_annotation_extent' if extended else 'undetermined'),
                'accepted_relation_refs': [r['id'] for r in local],
                'accepted_scope_refs': sorted({ref for r in local for ref in r['applicability_scope_refs']}),
                'positive_extent_evidence_refs': ([r['id'] for r in local]
                    if local else [r['boundary_connection_ref'] for r in path_certificates[target]]),
                'annotation_applicability_established': bool(local or extended),
                'elevation_transfer_established': False,
                'reason_codes': [] if local or extended else ['positive_elevation_extent_evidence_missing']})
        extension_count = sum(r['state'] == 'accepted_straight_annotation_extent' for r in targets)
        outcomes.append({'id': _stable_id('mep_elevation_applicability_outcome', source_ref, candidate),
            'record_type': 'mep_elevation_applicability_outcome', 'page_ref': source['page_ref'],
            'source_observation_ref': source_ref, 'literal_text': source['text'],
            'annotation_bbox_display': deepcopy(source['bbox_display']), 'candidate': deepcopy(candidate),
            'direct_target_refs': starts, 'direct_relation_refs': [r['id'] for r in direct],
            'direct_contact_evidence': [{k: deepcopy(evidence[ref].get(k)) for k in
                ('id', 'geometric_evidence_refs', 'automatic_search_certificate', 'scoped_stroke_ownership')}
                for ref in sorted({ref for r in direct for ref in r['binding_evidence_refs']})],
            'accepted_existing_scopes': [deepcopy(scopes[ref]) for ref in
                sorted({ref for r in direct for ref in r['applicability_scope_refs']})],
            'candidate_extents': targets, 'boundaries': boundaries,
            'state': ('straight_annotation_extent_resolved' if extension_count
                      else 'direct_target_only' if starts else 'direct_target_unresolved'),
            'search_coverage': {'leaders': 'existing_native_binding_replayed',
                'change_annotations': 'retained_m4_relations_only', 'risers_drops': 'retained_m4_relations_only',
                'section_detail_applicability': 'not_established', 'whole_source_search_complete': False,
                'bends': 'excluded_without_independent_elevation_evidence',
                'branches': 'hard_scope_boundary'},
            'extension_accepted_count': extension_count,
            'constant_elevation_scope_established': bool(starts and extension_count),
            'physical_continuation_established': False, 'quantity_eligible': False})
    extended_outcomes = [r for r in outcomes if r['extension_accepted_count']]
    return {'schema_version': '0.1.0', 'layer': 'mep_elevation_applicability',
        'document': deepcopy(graph['document']),
        'input_payload_sha256': {'terminology-proposals': _sha256(terminology),
            'attribute-bindings': _sha256(bindings), 'route-observations': _sha256(graph),
            'outlined-route-connections': _sha256(boundary_connections)},
        'outcomes': outcomes, 'summary': {'annotation_value_count': len(outcomes),
            'directly_bound_annotation_count': sum(bool(r['direct_target_refs']) for r in outcomes),
            'extended_elevation_scope_count': len(extended_outcomes),
            'accepted_straight_extent_target_count': sum(
                r['extension_accepted_count'] for r in extended_outcomes)},
        'reviewed_selectors_used': False, 'quantity_eligible': False}
