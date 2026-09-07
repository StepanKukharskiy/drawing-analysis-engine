"""Materialize machine-certified graph relations as fail-closed solver inputs.

The canonical graph remains immutable.  This module does not reinterpret a
review decision as geometry: it exposes only accepted typed relations to the
solver stages that can validate them again against their native constraints.
Until a stage reports reclosure, its previously emitted quantities are kept
unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping

from src.drawing_engine.core.contour_correspondence import reclose_contour_correspondence
from src.drawing_engine.disciplines.concrete.multi_component_solid_replay import replay_multi_component_solid
from src.drawing_engine.project.review_feedback import canonical_graph_sha256
from src.drawing_engine.core.shared_coordinate_system import reclose_shared_coordinate_system


SCHEMA_VERSION = "0.1.0"
SOLVER_RELATION_TYPES = {
    "shared_coordinate_system": {"cut_at", "projects_to", "section_of"},
    "contour_correspondence": {"cut_at", "dimension_of", "section_of"},
    "solid_hypothesis": {"cut_at", "dimension_of", "projects_to", "section_of"},
    "physical_bar_family": {
        "callout_targets",
        "detail_defines",
        "projects_to",
        "same_bar_family",
        "spacing_of",
    },
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _quantity_snapshot(pages: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        {
            "page": page.get("page"),
            "quantities": deepcopy(page.get("quantities", [])),
            "reinforcement_quantities": deepcopy(page.get("reinforcement_quantities")),
            "estimated_reinforcement_quantities": deepcopy(page.get("estimated_reinforcement_quantities")),
        }
        for page in pages
    ]
    encoded = _json(values).encode("utf-8")
    return {
        "sha256": sha256(encoded).hexdigest(),
        "byte_count": len(encoded),
        "serialization": "canonical_json_utf8",
        "pages": values,
    }


def _page_keys(entity: Mapping[str, Any]) -> set[str]:
    value = entity.get("provenance", {}).get("page_key")
    return {str(value)} if value else set()


def _native_entity(entity: Mapping[str, Any]) -> dict[str, Any]:
    """Expose exact native aliases and evidence behind one canonical endpoint."""

    provenance = entity.get("provenance", {}) or {}
    attributes = entity.get("attributes", {}) or {}
    source_ids = sorted(map(str, provenance.get("source_ids", []) or []))
    row = {
        "canonical_id": str(entity.get("id")),
        "entity_type": entity.get("entity_type"),
        "page_key": provenance.get("page_key"),
        "native_source_ids": source_ids,
        "source_path": provenance.get("source_path"),
        "evidence_refs": sorted(map(str, provenance.get("evidence_refs", []) or [])),
    }
    if entity.get("entity_type") == "view":
        frame = attributes.get("frame") or {}
        row.update(
            {
                "native_entity_id": frame.get("view_id"),
                "native_frame_id": frame.get("id"),
                "native_frame": deepcopy(frame),
            }
        )
    elif entity.get("entity_type") == "object":
        row.update(
            {
                "native_entity_id": source_ids[0] if len(source_ids) == 1 else None,
                "canonical_view_ids": sorted(map(str, attributes.get("view_ids", []) or [])),
            }
        )
    elif entity.get("entity_type") == "contour":
        row["native_entity_id"] = source_ids[0] if len(source_ids) == 1 else None
    elif entity.get("entity_type") in {"dimension", "dimension_attachment"}:
        row["native_entity_id"] = source_ids[0] if len(source_ids) == 1 else None
    return row


def _layout_mode(primary: Mapping[str, Any], section: Mapping[str, Any]) -> str | None:
    left = primary.get("native_frame", {}).get("bbox_display") or []
    right = section.get("native_frame", {}).get("bbox_display") or []
    if len(left) != 4 or len(right) != 4:
        return None
    if float(right[0]) >= float(left[2]):
        return "right"
    if float(right[2]) <= float(left[0]):
        return "left"
    if float(right[1]) >= float(left[3]):
        return "below"
    if float(right[3]) <= float(left[1]):
        return "above"
    return None


def _translated_cut(
    constraint: Mapping[str, Any],
    native_entities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    relation_id = str(constraint["relation_id"])
    source = native_entities.get(str(constraint["from"]), {})
    target = native_entities.get(str(constraint["to"]), {})
    source_native = source.get("native_entity_id")
    target_native = target.get("native_entity_id")
    if source.get("entity_type") != "view" or target.get("entity_type") != "view":
        return None, f"{relation_id} cut_at endpoints are not native views"
    if not source_native or not target_native:
        return None, f"{relation_id} has a view endpoint without one exact native entity ID"
    trace = (constraint.get("relation_attributes", {}) or {}).get("trace") or {}
    if trace.get("orientation") not in {"horizontal", "vertical"} or len(trace.get("bbox_display", [])) != 4:
        return None, f"{relation_id} has no complete native cutting trace"
    return (
        {
            "id": (constraint.get("native_relation_ids") or [relation_id])[0],
            "canonical_relation_id": relation_id,
            "type": "cut_at",
            "state": "accepted",
            "parent_view_id": str(target_native),
            "section_view_id": str(source_native),
            "trace": deepcopy(trace),
            "evidence_refs": list(constraint.get("native_evidence_refs", [])),
        },
        None,
    )


def _shared_coordinate_native_inputs(
    constraints: list[Mapping[str, Any]],
    native_entities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Translate the certified canonical slice into native solver records."""

    errors = []
    uniqueness_errors = []
    cuts = []
    projects_by_object: dict[str, set[str]] = defaultdict(set)
    primary_by_object: dict[str, set[str]] = defaultdict(set)
    sections_by_object: dict[str, set[str]] = defaultdict(set)
    constraint_ids = []
    for constraint in constraints:
        relation_id = str(constraint["relation_id"])
        constraint_ids.append(relation_id)
        source = native_entities.get(str(constraint["from"]), {})
        target = native_entities.get(str(constraint["to"]), {})
        source_native = source.get("native_entity_id")
        target_native = target.get("native_entity_id")
        if not source_native or not target_native:
            errors.append(f"{relation_id} has an endpoint without one exact native entity ID")
            continue
        relation_type = constraint["relation_type"]
        attributes = constraint.get("relation_attributes", {}) or {}
        if relation_type == "cut_at":
            cut, error = _translated_cut(constraint, native_entities)
            if error:
                errors.append(error)
            elif cut is not None:
                cuts.append(cut)
        elif relation_type == "projects_to":
            if source.get("entity_type") != "object" or target.get("entity_type") != "view":
                errors.append(f"{relation_id} projects_to endpoints are not object-to-view")
                continue
            object_id = str(source_native)
            view_id = str(target_native)
            projects_by_object[object_id].add(view_id)
            if attributes.get("projection_role") == "primary_metric_projection":
                primary_by_object[object_id].add(view_id)
        elif relation_type == "section_of":
            if source.get("entity_type") != "view" or target.get("entity_type") != "object":
                errors.append(f"{relation_id} section_of endpoints are not view-to-object")
                continue
            sections_by_object[str(target_native)].add(str(source_native))

    instances = []
    object_ids = sorted(set(projects_by_object) | set(sections_by_object))
    for object_id in object_ids:
        sections = sections_by_object.get(object_id, set())
        projections = projects_by_object.get(object_id, set())
        primaries = primary_by_object.get(object_id, set()) or (projections - sections)
        if len(sections) > 1 or len(primaries) > 1:
            uniqueness_errors.append(
                f"{object_id} has non-unique certified section or primary projections"
            )
            continue
        if len(sections) != 1 or len(primaries) != 1 or not sections <= projections:
            errors.append(
                f"{object_id} requires one certified section and one certified primary projection"
            )
            continue
        section_id = next(iter(sections))
        primary_id = next(iter(primaries))
        primary = next((item for item in native_entities.values() if item.get("native_entity_id") == primary_id), {})
        section = next((item for item in native_entities.values() if item.get("native_entity_id") == section_id), {})
        mode = _layout_mode(primary, section)
        if mode is None:
            errors.append(f"{object_id} native projection layout is not uniquely disjoint")
            continue
        instances.append(
            {
                "id": object_id,
                "primary_view_id": primary_id,
                "section_view_id": section_id,
                "view_ids": sorted({primary_id, section_id}),
                "layout_mode": mode,
                "basis": constraint_ids,
            }
        )

    frames = [
        deepcopy(item["native_frame"])
        for item in native_entities.values()
        if item.get("entity_type") == "view" and isinstance(item.get("native_frame"), Mapping)
    ]
    return {
        "frames": frames,
        "relations": cuts,
        "object_instance_graph": {"instances": instances},
        "translation_errors": sorted(errors),
        "uniqueness_errors": sorted(uniqueness_errors),
        "certified_constraint_ids": sorted(constraint_ids),
    }


def _rows_by_id(rows: Any) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    output: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for item in rows or []:
        if not isinstance(item, Mapping) or item.get("id") is None:
            continue
        item_id = str(item["id"])
        if item_id in output and output[item_id] != item:
            duplicates.add(item_id)
            continue
        output[item_id] = item
    return output, duplicates


def _contour_native_inputs(
    constraints: list[Mapping[str, Any]],
    native_entities: Mapping[str, Mapping[str, Any]],
    engineering_pages: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve certified contour constraints against exact frozen page records."""

    pages_by_key = {f"page:{item.get('page')}": item for item in engineering_pages}
    constraints_by_page: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    errors = []
    for constraint in constraints:
        page_keys = list(map(str, constraint.get("page_keys", []) or []))
        if len(page_keys) != 1:
            errors.append(f"{constraint['relation_id']} does not resolve to exactly one native page")
            continue
        constraints_by_page[page_keys[0]].append(constraint)

    page_inputs = []
    for page_key, page_constraints in sorted(constraints_by_page.items()):
        page = pages_by_key.get(page_key)
        if page is None:
            errors.append(f"{page_key} is absent from the frozen engineering pages")
            continue
        views_by_id, duplicate_views = _rows_by_id(page.get("view_hypotheses", []))
        frames_by_view: dict[str, Mapping[str, Any]] = {}
        duplicate_frames = set()
        for frame in page.get("view_frame_graph", {}).get("frames", []) or []:
            if not isinstance(frame, Mapping) or frame.get("view_id") is None:
                continue
            view_id = str(frame["view_id"])
            if view_id in frames_by_view and frames_by_view[view_id] != frame:
                duplicate_frames.add(view_id)
                continue
            frames_by_view[view_id] = frame
        contours_by_id, duplicate_contours = _rows_by_id(
            [
                *(page.get("contour_hypotheses", []) or []),
                *(page.get("calculation_contours", []) or []),
            ]
        )
        attachments = page.get("dimension_ownership", {}).get("attachments", []) or []
        native_dimension_relations, duplicate_dimension_relations = _rows_by_id(
            page.get("dimension_ownership", {}).get("relations", [])
        )
        native_cut_relations, duplicate_cut_relations = _rows_by_id(
            page.get("view_frame_graph", {}).get("relations", [])
        )
        native_objects, duplicate_objects = _rows_by_id(
            page.get("object_instance_graph", {}).get("instances", [])
        )
        cuts = []
        ownership = []
        section_targets: dict[str, set[str]] = defaultdict(set)
        relation_ids = []
        page_errors = []
        page_uniqueness_errors = []
        relevant_view_ids = set()

        for constraint in page_constraints:
            relation_id = str(constraint["relation_id"])
            relation_ids.append(relation_id)
            relation_type = constraint["relation_type"]
            source = native_entities.get(str(constraint["from"]), {})
            target = native_entities.get(str(constraint["to"]), {})
            if relation_type == "cut_at":
                cut, error = _translated_cut(constraint, native_entities)
                if error:
                    page_errors.append(error)
                    continue
                native_relation_ids = set(map(str, constraint.get("native_relation_ids", []) or []))
                native_cut_ids = sorted(native_relation_ids & native_cut_relations.keys())
                if len(native_cut_ids) != 1 or native_cut_ids[0] in duplicate_cut_relations:
                    page_errors.append(f"{relation_id} does not resolve to one exact native cut relation")
                    continue
                native_cut = native_cut_relations[native_cut_ids[0]]
                if (
                    str(native_cut.get("parent_view_id")) != cut["parent_view_id"]
                    or str(native_cut.get("section_view_id")) != cut["section_view_id"]
                    or native_cut.get("state") != "accepted"
                ):
                    page_uniqueness_errors.append(f"{relation_id} contradicts its native cut relation")
                    continue
                cut = deepcopy(dict(native_cut))
                cut["canonical_relation_id"] = relation_id
                cut["evidence_refs"] = sorted(
                    {
                        *map(str, constraint.get("native_evidence_refs", []) or []),
                        *map(str, cut.get("trace", {}).get("primitive_refs", []) or []),
                    }
                )
                cuts.append(cut)
                relevant_view_ids.update((cut["parent_view_id"], cut["section_view_id"]))
            elif relation_type == "section_of":
                section_id = source.get("native_entity_id")
                object_id = target.get("native_entity_id")
                if source.get("entity_type") != "view" or target.get("entity_type") != "object":
                    page_errors.append(f"{relation_id} section_of endpoints are not view-to-object")
                elif not section_id or not object_id:
                    page_errors.append(f"{relation_id} has no exact native section/object IDs")
                elif str(section_id) not in views_by_id or str(object_id) not in native_objects:
                    page_errors.append(f"{relation_id} native section/object records are missing")
                elif str(object_id) in duplicate_objects:
                    page_uniqueness_errors.append(f"native object {object_id} is not unique")
                elif str(section_id) not in set(map(str, native_objects[str(object_id)].get("view_ids", []) or [])):
                    page_uniqueness_errors.append(
                        f"{relation_id} contradicts native object projection membership"
                    )
                else:
                    section_targets[str(section_id)].add(str(object_id))
            elif relation_type == "dimension_of":
                target_aliases = {
                    *map(str, target.get("native_source_ids", []) or []),
                    *([str(target["native_entity_id"])] if target.get("native_entity_id") else []),
                }
                contour_ids = sorted(target_aliases & contours_by_id.keys())
                source_aliases = {
                    *map(str, source.get("native_source_ids", []) or []),
                    *([str(source["native_entity_id"])] if source.get("native_entity_id") else []),
                }
                matching_attachments = [
                    item
                    for item in attachments
                    if str(item.get("id")) in source_aliases
                    or str(item.get("dimension_ref")) in source_aliases
                ]
                if len(contour_ids) == 1 and len(matching_attachments) > 1:
                    owned_matches = [
                        item
                        for item in matching_attachments
                        if contour_ids[0] in set(map(str, item.get("owner_entity_refs", []) or []))
                    ]
                    if owned_matches:
                        matching_attachments = owned_matches
                native_relation_ids = set(map(str, constraint.get("native_relation_ids", []) or []))
                dimension_relation_ids = sorted(native_relation_ids & native_dimension_relations.keys())
                if len(contour_ids) != 1 or len(matching_attachments) != 1:
                    page_errors.append(
                        f"{relation_id} does not resolve to one native contour and dimension attachment"
                    )
                    continue
                if len(dimension_relation_ids) != 1 or dimension_relation_ids[0] in duplicate_dimension_relations:
                    page_errors.append(f"{relation_id} does not resolve to one exact native dimension relation")
                    continue
                native_dimension_relation = native_dimension_relations[dimension_relation_ids[0]]
                if (
                    str(native_dimension_relation.get("from")) not in source_aliases
                    or str(native_dimension_relation.get("to")) != contour_ids[0]
                ):
                    page_uniqueness_errors.append(
                        f"{relation_id} contradicts its native dimension ownership relation"
                    )
                    continue
                attachment = deepcopy(dict(matching_attachments[0]))
                attachment["status"] = "accepted"
                attachment["owner_entity_refs"] = [contour_ids[0]]
                attachment["certified_relation_id"] = relation_id
                ownership.append(attachment)

        for section_id, object_ids in sorted(section_targets.items()):
            if len(object_ids) > 1:
                page_uniqueness_errors.append(
                    f"native section {section_id} has multiple certified physical objects"
                )
        for view_id in sorted(relevant_view_ids):
            if view_id in duplicate_views or view_id in duplicate_frames:
                page_uniqueness_errors.append(f"native view {view_id} is not unique")
            if view_id not in views_by_id or view_id not in frames_by_view:
                page_errors.append(f"native view/frame {view_id} is missing")

        selected_views = [deepcopy(dict(views_by_id[item])) for item in sorted(relevant_view_ids) if item in views_by_id]
        contour_ids = {
            str(contour_id)
            for view in selected_views
            for contour_id in view.get("contour_refs", []) or []
        }
        for contour_id in sorted(contour_ids):
            if contour_id in duplicate_contours:
                page_uniqueness_errors.append(f"native contour {contour_id} is not unique")
            if contour_id not in contours_by_id:
                page_errors.append(f"native contour {contour_id} is missing")
        selected_contours = [
            deepcopy(dict(contours_by_id[item]))
            for item in sorted(contour_ids)
            if item in contours_by_id
        ]
        selected_frames = [
            deepcopy(dict(frames_by_view[item]))
            for item in sorted(relevant_view_ids)
            if item in frames_by_view
        ]
        page_inputs.append(
            {
                "page_key": page_key,
                "certified_constraint_ids": sorted(relation_ids),
                "cut_constraint_ids": sorted(
                    str(item["relation_id"])
                    for item in page_constraints
                    if item["relation_type"] == "cut_at"
                ),
                "frames": selected_frames,
                "relations": cuts,
                "views": selected_views,
                "contours": selected_contours,
                "dimension_ownership": {"attachments": ownership},
                "native_segments": deepcopy(page.get("native_segments", []) or []),
                "translation_errors": sorted(set(page_errors)),
                "uniqueness_errors": sorted(set(page_uniqueness_errors)),
            }
        )
    return {"pages": page_inputs, "errors": sorted(set(errors))}


def build_solver_replay(
    canonical_graph: Mapping[str, Any],
    effective_overlay: Mapping[str, Any] | None,
    engineering_pages: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic per-solver inputs without changing quantities."""

    base_hash = canonical_graph_sha256(canonical_graph)
    if effective_overlay is None:
        effective_overlay = {
            "layer": "reviewed_canonical_overlay",
            "base_canonical_graph_sha256": base_hash,
            "effective_entities": deepcopy(canonical_graph.get("entities", []) or []),
            "effective_relations": deepcopy(canonical_graph.get("relations", []) or []),
            "validation": {"status": "pass"},
        }
    if effective_overlay.get("layer") != "reviewed_canonical_overlay":
        raise ValueError("effective overlay has an unsupported layer")
    if effective_overlay.get("base_canonical_graph_sha256") != base_hash:
        raise ValueError("effective overlay does not match the canonical graph")
    if effective_overlay.get("validation", {}).get("status") != "pass":
        raise ValueError("effective overlay validation did not pass")

    entities = {
        str(item["id"]): item
        for item in effective_overlay.get("effective_entities", []) or []
    }
    certified_relations = [
        item
        for item in effective_overlay.get("effective_relations", []) or []
        if item.get("review_status") in {"accepted", "corrected"}
    ]
    certified_relations.sort(key=lambda item: str(item["id"]))

    relation_pages: dict[str, list[str]] = {}
    for relation in certified_relations:
        pages = set()
        page_key = relation.get("provenance", {}).get("page_key")
        if page_key:
            pages.add(str(page_key))
        pages.update(_page_keys(entities.get(str(relation.get("from")), {})))
        pages.update(_page_keys(entities.get(str(relation.get("to")), {})))
        relation_pages[str(relation["id"])] = sorted(pages)

    solver_inputs = {}
    for solver, allowed_types in SOLVER_RELATION_TYPES.items():
        constraints = []
        native_entity_ids: set[str] = set()
        for relation in certified_relations:
            relation_type = str(relation.get("type"))
            if relation_type not in allowed_types:
                continue
            source = entities.get(str(relation.get("from")), {})
            target = entities.get(str(relation.get("to")), {})
            if solver == "shared_coordinate_system" and {source.get("entity_type"), target.get("entity_type")} - {"object", "view"}:
                continue
            if solver == "contour_correspondence":
                endpoint_types = (source.get("entity_type"), target.get("entity_type"))
                if (
                    relation_type == "cut_at" and endpoint_types != ("view", "view")
                    or relation_type == "section_of" and endpoint_types != ("view", "object")
                    or relation_type == "dimension_of"
                    and endpoint_types not in {
                        ("dimension", "contour"),
                        ("dimension_attachment", "contour"),
                    }
                ):
                    continue
            native_entity_ids.update((str(relation["from"]), str(relation["to"])))
            native_evidence = sorted(
                {
                    *map(str, relation.get("provenance", {}).get("evidence_refs", []) or []),
                    *map(str, source.get("provenance", {}).get("evidence_refs", []) or []),
                    *map(str, target.get("provenance", {}).get("evidence_refs", []) or []),
                }
            )
            constraints.append(
                {
                    "relation_id": relation["id"],
                    "relation_type": relation_type,
                    "from": relation["from"],
                    "to": relation["to"],
                    "page_keys": relation_pages[str(relation["id"])],
                    "decision_id": relation.get("review_decision_id"),
                    "evidence_refs": sorted(relation.get("provenance", {}).get("evidence_refs", []) or []),
                    "native_relation_ids": sorted(
                        map(str, relation.get("provenance", {}).get("source_ids", []) or [])
                    ),
                    "native_evidence_refs": native_evidence,
                    "relation_attributes": deepcopy(relation.get("attributes", {}) or {}),
                }
            )
        solver_inputs[solver] = {
            "status": "certified_constraints_available" if constraints else "no_new_certified_constraints",
            "constraints": constraints,
            "native_entities": [
                _native_entity(entities[entity_id])
                for entity_id in sorted(native_entity_ids)
                if entity_id in entities
            ],
            "requires_native_constraint_reclosure": True,
            "replay_contract": {
                "input_relation_types": sorted(allowed_types),
                "certified_relations_only": True,
                "canonical_endpoints_translate_to_exact_native_ids": True,
                "native_evidence_is_preserved": True,
                "required_outcomes": ["reclosed_pass", "reclosed_fail", "insufficient_constraints"],
                "quantity_writes_allowed": False,
            },
        }

    family_by_mark: dict[str, set[str]] = defaultdict(set)
    paths_by_mark: dict[str, set[str]] = defaultdict(set)
    projected_paths_by_family: dict[str, set[str]] = defaultdict(set)
    for relation in certified_relations:
        relation_type = relation.get("type")
        if relation_type == "same_bar_family":
            family_by_mark[str(relation["from"])].add(str(relation["to"]))
        elif relation_type == "callout_targets":
            paths_by_mark[str(relation["from"])].add(str(relation["to"]))
        elif relation_type == "projects_to":
            projected_paths_by_family[str(relation["from"])].add(str(relation["to"]))
    family_triangles = sorted(
        {
            (mark_id, family_id, path_id)
            for mark_id, family_ids in family_by_mark.items()
            for family_id in family_ids
            for path_id in paths_by_mark.get(mark_id, set())
            if path_id in projected_paths_by_family.get(family_id, set())
        }
    )
    before = _quantity_snapshot(engineering_pages)
    coordinate_input = solver_inputs["shared_coordinate_system"]
    native_entities = {
        str(item["canonical_id"]): item
        for item in coordinate_input["native_entities"]
    }
    translated = _shared_coordinate_native_inputs(
        coordinate_input["constraints"],
        native_entities,
    )
    coordinate_reclosure = reclose_shared_coordinate_system(
        translated["frames"],
        translated["relations"],
        translated["object_instance_graph"],
        certified_constraint_ids=translated["certified_constraint_ids"],
        translation_errors=translated["translation_errors"],
        input_uniqueness_errors=translated["uniqueness_errors"],
    )
    coordinate_input["reclosure"] = coordinate_reclosure
    contour_input = solver_inputs["contour_correspondence"]
    contour_entities = {
        str(item["canonical_id"]): item
        for item in contour_input["native_entities"]
    }
    contour_translation = _contour_native_inputs(
        contour_input["constraints"],
        contour_entities,
        engineering_pages,
    )
    contour_pages = []
    for page_input in contour_translation["pages"]:
        page_coordinate_reclosure = reclose_shared_coordinate_system(
            page_input["frames"],
            page_input["relations"],
            certified_constraint_ids=page_input["cut_constraint_ids"],
            translation_errors=page_input["translation_errors"],
            input_uniqueness_errors=page_input["uniqueness_errors"],
        )
        contour_reclosure = reclose_contour_correspondence(
            page_input["frames"],
            page_coordinate_reclosure.get("coordinate_system"),
            page_input["contours"],
            page_input["views"],
            page_input["dimension_ownership"],
            page_input["native_segments"],
            certified_constraint_ids=page_input["certified_constraint_ids"],
            translation_errors=page_input["translation_errors"],
            input_uniqueness_errors=page_input["uniqueness_errors"],
            coordinate_reclosure_status=page_coordinate_reclosure["status"],
        )
        contour_pages.append(
            {
                "page_key": page_input["page_key"],
                "coordinate_reclosure": page_coordinate_reclosure,
                **contour_reclosure,
            }
        )
    contour_statuses = [item["status"] for item in contour_pages]
    contour_status = (
        "reclosed_fail"
        if "reclosed_fail" in contour_statuses
        else "reclosed_pass"
        if contour_statuses and all(item == "reclosed_pass" for item in contour_statuses)
        else "insufficient_constraints"
    )
    contour_reclosure = {
        "schema_version": SCHEMA_VERSION,
        "stage": "contour_correspondence",
        "status": contour_status,
        "pages": contour_pages,
        "translation_errors": contour_translation["errors"],
        "summary": {
            "page_count": len(contour_pages),
            "reclosed_pass_count": sum(item == "reclosed_pass" for item in contour_statuses),
            "reclosed_fail_count": sum(item == "reclosed_fail" for item in contour_statuses),
            "insufficient_constraints_count": sum(item == "insufficient_constraints" for item in contour_statuses),
        },
        "contract": contour_input["replay_contract"],
    }
    if contour_translation["errors"] and contour_status != "reclosed_fail":
        contour_reclosure["status"] = "insufficient_constraints"
    contour_input["reclosure"] = contour_reclosure
    solid_input = solver_inputs["solid_hypothesis"]
    solid_reclosure = replay_multi_component_solid(
        canonical_graph_sha256=base_hash,
        constraints=solid_input["constraints"],
        native_entities=solid_input["native_entities"],
        engineering_pages=engineering_pages,
        shared_coordinate_reclosure=coordinate_reclosure,
    )
    solid_input["reclosure"] = solid_reclosure
    after = _quantity_snapshot(engineering_pages)
    byte_identical = (
        before["sha256"] == after["sha256"]
        and before["byte_count"] == after["byte_count"]
        and before["pages"] == after["pages"]
    )
    if not byte_identical:
        raise RuntimeError("solver replay changed engineering quantities")
    if canonical_graph_sha256(canonical_graph) != base_hash:
        raise RuntimeError("canonical graph changed during solver replay")
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "certified_solver_replay",
        "document_key": canonical_graph.get("document_key"),
        "base_canonical_graph_sha256": base_hash,
        "status": "solver_inputs_materialized" if certified_relations else "no_certified_relations",
        "solver_inputs": solver_inputs,
        "solver_reclosures": {
            "shared_coordinate_system": coordinate_reclosure,
            "contour_correspondence": contour_reclosure,
            "solid_hypothesis": solid_reclosure,
        },
        "solid_preview": solid_reclosure.get("solid_preview"),
        "closed_relation_certificates": {
            "bar_family_path_triangles": [
                {"mark_id": mark, "family_id": family, "path_id": path}
                for mark, family, path in family_triangles
            ],
        },
        "quantity_replay": {
            "status": "reclosures_completed_quantities_unchanged",
            "before": before,
            "after": after,
            "changed": False,
            "byte_identical": byte_identical,
            "reason": "coordinate, contour, and solid reclosure validate geometry relations without changing quantities",
        },
        "summary": {
            "certified_relation_count": len(certified_relations),
            "bar_family_path_triangle_count": len(family_triangles),
            "solver_input_count": sum(len(item["constraints"]) for item in solver_inputs.values()),
            "shared_coordinate_reclosure_status": coordinate_reclosure["status"],
            "contour_correspondence_reclosure_status": contour_reclosure["status"],
            "solid_hypothesis_reclosure_status": solid_reclosure["status"],
        },
        "contract": {
            "canonical_graph_mutated": False,
            "automatic_acceptance_is_not_direct_geometry": True,
            "quantity_change_requires_solver_reclosure": True,
            "shared_coordinate_reclosure_cannot_change_quantities": True,
            "contour_correspondence_reclosure_cannot_change_quantities": True,
            "solid_hypothesis_reclosure_cannot_change_quantities": True,
            "reusable_solver_replay_contract": {
                "certified_relation_slice": True,
                "canonical_to_native_translation": True,
                "native_gate_reexecution": True,
                "explicit_reclosure_outcome": True,
                "quantity_snapshot_before_and_after": True,
                "next_consumers": [
                    "physical_bar_family",
                ],
                "completed_consumers": ["solid_hypothesis"],
            },
            "schedule_values_used": False,
        },
    }
