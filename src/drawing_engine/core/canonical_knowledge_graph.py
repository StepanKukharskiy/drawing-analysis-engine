"""Canonical, provenance-preserving entities and relations for drawing evidence.

This module is intentionally an adapter, not another inference solver.  It
assigns document-wide stable identifiers to facts already present in an
engineering graph and emits typed relations only when an existing record
supports them.  In particular, equal mark text and geometric proximity never
merge physical entities here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence


RELATION_TYPES = (
    "section_of",
    "cut_at",
    "dimension_of",
    "callout_targets",
    "detail_defines",
    "projects_to",
    "same_bar_family",
    "spacing_of",
)

_EPISTEMIC_STATES = {
    "direct",
    "observed",
    "derived",
    "inferred",
    "convention_dependent",
    "unknown",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(*parts: Any, length: int = 16) -> str:
    return sha256(_json(parts).encode("utf-8")).hexdigest()[:length]


def canonical_entity_id(
    page_key: str,
    entity_type: str,
    source_id: str,
    *,
    document_key: str = "document:unspecified",
) -> str:
    """Return a stable document- and page-qualified canonical ID."""

    return f"canonical.{entity_type}.{_digest(document_key, page_key, entity_type, source_id)}"


def _epistemic_state(record: Mapping[str, Any], default: str = "unknown") -> str:
    for key in ("epistemic_state", "state"):
        state = record.get(key)
        if state in _EPISTEMIC_STATES:
            return str(state)
    status = record.get("status") or record.get("state")
    if isinstance(status, str):
        if status.startswith("derived_") or status.endswith("_resolved"):
            return "derived"
        if status.startswith("observed_"):
            return "observed"
        if status.endswith("_candidate"):
            return "inferred"
    if status in {"accepted", "resolved", "pass", "geometry_resolved", "fabrication_resolved"}:
        return "derived"
    if status in {"candidate", "review_candidate", "ambiguous", "partial", "unresolved", "rejected"}:
        return "unknown"
    return default


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                yield item


def _evidence_refs(value: Any) -> list[str]:
    """Collect immutable PDF/evidence references without copying whole records."""

    refs: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key.endswith("_ref") or key.endswith("_refs") or key in {
                    "evidence_ref",
                    "evidence_refs",
                    "primitive_ref",
                    "primitive_refs",
                    "drawing_ref",
                    "drawing_refs",
                    "source_path_ref",
                }:
                    refs.update(_strings(child))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sorted(refs)


def _provenance(
    document_key: str,
    page_key: str,
    source_path: str,
    source_ids: Iterable[str],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "document_key": document_key,
        "page_key": page_key,
        "source_path": source_path,
        "source_ids": sorted({str(item) for item in source_ids if item is not None}),
        "evidence_refs": _evidence_refs(record),
    }
    if record.get("basis") is not None:
        result["basis"] = deepcopy(record["basis"])
    return result


def _view_frame(view: Mapping[str, Any]) -> dict[str, Any]:
    explicit = view.get("canonical_frame") or view.get("view_frame") or view.get("frame")
    if isinstance(explicit, Mapping):
        frame = deepcopy(dict(explicit))
        frame.setdefault("state", _epistemic_state(explicit))
        return frame
    fields = {
        key: deepcopy(view[key])
        for key in (
            "scale",
            "points_per_mm",
            "mm_per_point",
            "origin_display",
            "origin_metric",
            "axes",
            "projection_direction",
        )
        if key in view
    }
    return {"state": _epistemic_state(view) if fields else "unknown", **fields}


def _page_identity_rows(page: Mapping[str, Any]) -> list[str]:
    rows: list[str] = []
    for collection in ("view_hypotheses", "claims"):
        for item in page.get(collection, []) or []:
            if isinstance(item, Mapping) and item.get("id") is not None:
                rows.append(f"{collection}:{item['id']}")
    for item in page.get("object_instance_graph", {}).get("instances", []) or []:
        if item.get("id") is not None:
            rows.append(f"objects:{item['id']}")
    return sorted(rows)


def _coerce_pages(engineering: Any) -> list[Mapping[str, Any]]:
    if isinstance(engineering, Mapping) and isinstance(engineering.get("engineering_graph"), Mapping):
        engineering = engineering["engineering_graph"]
    if isinstance(engineering, Mapping) and isinstance(engineering.get("pages"), list):
        pages = engineering["pages"]
    elif isinstance(engineering, Mapping):
        pages = [engineering]
    elif isinstance(engineering, Sequence) and not isinstance(engineering, (str, bytes, bytearray)):
        pages = list(engineering)
    else:
        raise TypeError("engineering must be a page mapping, page sequence, or document mapping with pages")
    if not all(isinstance(page, Mapping) for page in pages):
        raise TypeError("every engineering page must be a mapping")
    return pages


def _document_key(engineering: Any, pages: Sequence[Mapping[str, Any]], explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    current = engineering
    if isinstance(current, Mapping):
        for key in ("document_key", "source_pdf_sha256", "source_sha256"):
            if current.get(key):
                return str(current[key])
        nested = current.get("engineering_graph")
        if isinstance(nested, Mapping):
            for key in ("document_key", "source_pdf_sha256", "source_sha256"):
                if nested.get(key):
                    return str(nested[key])
    page_digests = sorted(_digest(page, length=64) for page in pages)
    return f"engineering-sha256:{_digest(page_digests, length=64)}"


def _page_keys(pages: Sequence[Mapping[str, Any]]) -> list[str]:
    bases = [f"page:{page.get('page', index)}" for index, page in enumerate(pages)]
    counts = Counter(bases)
    seen: Counter[str] = Counter()
    keys: list[str] = []
    for base, page in zip(bases, pages):
        if counts[base] == 1:
            keys.append(base)
            continue
        signature = _digest(_page_identity_rows(page), length=10)
        candidate = f"{base}:{signature}"
        seen[candidate] += 1
        keys.append(candidate if seen[candidate] == 1 else f"{candidate}:{seen[candidate]}")
    return keys


class _Builder:
    def __init__(self, document_key: str) -> None:
        self.document_key = document_key
        self.entities: dict[str, dict[str, Any]] = {}
        self.relations: dict[str, dict[str, Any]] = {}
        self.source_index: dict[tuple[str, str, str], str] = {}
        self.source_any: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.duplicate_sources: list[dict[str, str]] = []
        self.unresolved: list[dict[str, Any]] = []

    def entity(
        self,
        page_key: str,
        entity_type: str,
        source_id: str,
        record: Mapping[str, Any],
        source_path: str,
        attributes: Mapping[str, Any] | None = None,
        aliases: Iterable[str] = (),
    ) -> str:
        source_id = str(source_id)
        entity_id = canonical_entity_id(
            page_key,
            entity_type,
            source_id,
            document_key=self.document_key,
        )
        node = {
            "id": entity_id,
            "entity_type": entity_type,
            "state": _epistemic_state(record),
            "status": record.get("status") or record.get("state"),
            "attributes": deepcopy(dict(attributes or {})),
            "provenance": _provenance(
                self.document_key,
                page_key,
                source_path,
                [source_id, *aliases],
                record,
            ),
        }
        previous = self.entities.get(entity_id)
        if previous is None:
            self.entities[entity_id] = node
        elif previous != node:
            self.duplicate_sources.append(
                {"page_key": page_key, "entity_type": entity_type, "source_id": source_id}
            )
        for alias in (source_id, *aliases):
            alias = str(alias)
            key = (page_key, entity_type, alias)
            previous_id = self.source_index.get(key)
            if previous_id is not None and previous_id != entity_id:
                self.duplicate_sources.append(
                    {"page_key": page_key, "entity_type": entity_type, "source_id": alias}
                )
                continue
            self.source_index[key] = entity_id
            self.source_any[(page_key, alias)].add(entity_id)
        return entity_id

    def resolve(self, page_key: str, source_id: Any, kinds: Iterable[str] = ()) -> str | None:
        if source_id is None:
            return None
        source_id = str(source_id)
        for kind in kinds:
            result = self.source_index.get((page_key, kind, source_id))
            if result is not None:
                return result
        candidates = self.source_any.get((page_key, source_id), set())
        return next(iter(candidates)) if len(candidates) == 1 else None

    def reference(self, page_key: str, source_id: str, source_path: str) -> str:
        existing = self.resolve(page_key, source_id)
        if existing is not None:
            return existing
        record = {"id": source_id, "state": "observed"}
        return self.entity(
            page_key,
            "source_reference",
            source_id,
            record,
            source_path,
            {"semantic_type": "unclassified"},
        )

    def relation(
        self,
        page_key: str,
        relation_type: str,
        source: str | None,
        target: str | None,
        record: Mapping[str, Any],
        source_path: str,
        source_key: str,
        attributes: Mapping[str, Any] | None = None,
        state: str | None = None,
    ) -> str | None:
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"unsupported canonical relation type: {relation_type}")
        if source is None or target is None:
            self.unresolved.append(
                {
                    "page_key": page_key,
                    "relation_type": relation_type,
                    "source_key": source_key,
                    "reason": "relation endpoint is not a canonical entity",
                }
            )
            return None
        if relation_type == "same_bar_family" and source == target:
            return None
        relation_id = f"canonical.relation.{_digest(self.document_key, page_key, relation_type, source, target, source_key)}"
        node = {
            "id": relation_id,
            "type": relation_type,
            "from": source,
            "to": target,
            "state": state or _epistemic_state(record, "derived"),
            "attributes": deepcopy(dict(attributes or {})),
            "provenance": _provenance(
                self.document_key,
                page_key,
                source_path,
                [source_key],
                record,
            ),
        }
        self.relations.setdefault(relation_id, node)
        return relation_id


def _component_for_fragment(fragment_to_paths: Mapping[str, set[str]], fragment_id: Any) -> str | None:
    paths = fragment_to_paths.get(str(fragment_id), set())
    return next(iter(paths)) if len(paths) == 1 else None


def _contour_geometry(contour: Mapping[str, Any], primitive_refs: Iterable[str]) -> dict[str, Any]:
    geometry = {
        key: deepcopy(contour[key])
        for key in (
            "segments_display",
            "polygon_points_display",
            "points_display",
            "geometry",
        )
        if contour.get(key) is not None
    }
    refs = sorted(set(primitive_refs))
    if geometry:
        return {"state": "observed_or_derived", "representation": "explicit_geometry", **geometry}
    if refs:
        return {
            "state": "observed",
            "representation": "native_vector_reference",
            "primitive_refs": refs,
        }
    return {
        "state": "partial",
        "representation": "compact_bbox_only",
        "reason": "the compact engineering record does not carry native path segments",
    }


def _build_page(builder: _Builder, page: Mapping[str, Any], page_key: str) -> None:
    views = page.get("view_hypotheses", []) or []
    objects = page.get("object_instance_graph", {}).get("instances", []) or []
    contours = page.get("contour_hypotheses", []) or []
    calculation_contours = page.get("calculation_contours", []) or []
    program = page.get("rebar_program", {}) or {}
    path_graph = program.get("physical_path_graph", {}) or {}
    details_graph = program.get("native_vector_detail_linking", {}) or {}
    components = path_graph.get("components", []) or []
    fragments = path_graph.get("fragments", []) or []
    composite_projected_bars = path_graph.get("composite_projected_bars", {}).get("proposals", []) or []
    details = details_graph.get("details", []) or []
    groups = program.get("groups", []) or []
    physical_families = details_graph.get("physical_families", []) or []
    placements = details_graph.get("placement_associations", []) or []
    dimension_ownership = page.get("dimension_ownership", {}) or {}
    view_frame_graph = page.get("view_frame_graph", {}) or {}
    frames_by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for frame in view_frame_graph.get("frames", []) or []:
        if isinstance(frame, Mapping) and frame.get("view_id") is not None:
            frames_by_view[str(frame["view_id"])].append(frame)
    accepted_frame_cuts = {
        str(item.get("section_view_id"))
        for item in view_frame_graph.get("relations", []) or []
        if isinstance(item, Mapping)
        and item.get("type") == "cut_at"
        and item.get("state") == "accepted"
        and item.get("integration_certificate", {}).get("status") == "passed"
        and item.get("integration_certificate", {}).get("title_segmentation_closed") is True
        and item.get("integration_certificate", {}).get("dimension_adjudication_closed") is True
    }

    view_ids: dict[str, str] = {}
    for view in views:
        source_id = str(view["id"])
        frame_candidates = frames_by_view.get(source_id, [])
        frame = frame_candidates[0] if len(frame_candidates) == 1 else None
        source_record = deepcopy(dict(view))
        aliases: list[str] = []
        source_path = f"view_hypotheses[id={source_id}]"
        if frame is not None:
            source_record["view_frame_graph"] = deepcopy(frame)
            source_path += f";view_frame_graph.frames[id={frame.get('id')}]"
            if frame.get("id") is not None:
                aliases.append(str(frame["id"]))
        elif len(frame_candidates) > 1:
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "view_frame_assignment",
                    "source_id": source_id,
                    "status": "ambiguous",
                    "reason": "multiple view-frame records refer to the same view",
                    "frame_ids": [item.get("id") for item in frame_candidates],
                }
            )
        view_ids[source_id] = builder.entity(
            page_key,
            "view",
            source_id,
            source_record,
            source_path,
            {
                "role": view.get("role_hypothesis"),
                "bbox_display": deepcopy(view.get("bbox_display")),
                "confidence": view.get("confidence"),
                "frame": deepcopy(frame) if frame is not None else _view_frame(view),
            },
            aliases,
        )
    for frame_view_id, frame_candidates in frames_by_view.items():
        if frame_view_id in view_ids:
            continue
        title_segments = {
            str(item.get("id")): item
            for item in page.get("title_anchored_view_segmentation", {}).get("segments", []) or []
            if item.get("id") is not None
        }
        if len(frame_candidates) == 1 and frame_view_id in title_segments:
            frame = frame_candidates[0]
            segment = title_segments[frame_view_id]
            source_record = {
                **deepcopy(dict(segment)),
                "view_frame_graph": deepcopy(frame),
            }
            view_ids[frame_view_id] = builder.entity(
                page_key,
                "view",
                frame_view_id,
                source_record,
                (
                    f"title_anchored_view_segmentation.segments[id={frame_view_id}]"
                    f";view_frame_graph.frames[id={frame.get('id')}]"
                ),
                {
                    "role": frame.get("view_role_hypothesis"),
                    "bbox_display": deepcopy(frame.get("bbox_display")),
                    "confidence": frame.get("confidence"),
                    "frame": deepcopy(frame),
                    "source_view_id": segment.get("source_view_id"),
                },
                [str(frame.get("id"))] if frame.get("id") is not None else [],
            )
            continue
        for frame in frame_candidates:
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "view_frame_assignment",
                    "source_id": frame.get("id"),
                    "status": frame.get("state"),
                    "reason": "view-frame record references a missing view entity",
                    "view_id": frame_view_id,
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"view_frame_graph.frames[id={frame.get('id')}]",
                        [str(frame.get("id")), frame_view_id],
                        frame,
                    ),
                }
            )

    ownership_by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for attachment in dimension_ownership.get("attachments", []) or []:
        if not isinstance(attachment, Mapping) or attachment.get("status") != "accepted":
            continue
        for owner_ref in attachment.get("owner_entity_refs", []) or []:
            ownership_by_owner[str(owner_ref)].append(attachment)

    contour_ids: dict[str, str] = {}
    for collection_name, rows in (
        ("contour_hypotheses", contours),
        ("calculation_contours", calculation_contours),
    ):
        for contour in rows:
            if not isinstance(contour, Mapping) or contour.get("id") is None:
                continue
            source_id = str(contour["id"])
            ownership_evidence = ownership_by_owner.get(source_id, [])
            source_record = deepcopy(dict(contour))
            if ownership_evidence:
                source_record["dimension_ownership_evidence"] = deepcopy(ownership_evidence)
            primitive_refs = _evidence_refs(source_record)
            contour_ids[source_id] = builder.entity(
                page_key,
                "contour",
                source_id,
                source_record,
                f"{collection_name}[id={source_id}]",
                {
                    "kind": contour.get("kind") or contour.get("role"),
                    "bbox_display": deepcopy(contour.get("bbox_display")),
                    "closed": contour.get("closed"),
                    "geometry": _contour_geometry(contour, primitive_refs),
                    "dimensions": deepcopy(contour.get("dimensions")),
                    "closure_validation": deepcopy(contour.get("closure_validation")),
                },
            )

    object_ids: dict[str, str] = {}
    for item in objects:
        source_id = str(item["id"])
        object_ids[source_id] = builder.entity(
            page_key,
            "object",
            source_id,
            item,
            f"object_instance_graph.instances[id={source_id}]",
            {
                "role": item.get("role"),
                "designation": item.get("designation"),
                "confidence": item.get("confidence"),
                "geometry_status": item.get("geometry_status"),
                "view_ids": [view_ids[source] for source in item.get("view_ids", []) if source in view_ids],
            },
        )

    fragment_to_paths: dict[str, set[str]] = defaultdict(set)
    primitive_to_fragments: dict[str, set[str]] = defaultdict(set)
    fragment_by_id: dict[str, Mapping[str, Any]] = {}
    for fragment in fragments:
        fragment_id = str(fragment["id"])
        fragment_by_id[fragment_id] = fragment
        for primitive_ref in (fragment.get("primitive_ref"), fragment.get("source_path_ref")):
            if primitive_ref is not None:
                primitive_to_fragments[str(primitive_ref)].add(fragment_id)

    path_ids: dict[str, str] = {}
    for component in components:
        source_id = str(component["id"])
        source_fragments = [
            fragment_by_id[str(fragment_id)]
            for fragment_id in component.get("fragment_ids", [])
            if str(fragment_id) in fragment_by_id
        ]
        component_record = deepcopy(dict(component))
        component_record["source_fragments"] = deepcopy(source_fragments)
        component_record.setdefault("epistemic_state", "observed")
        component_record.setdefault("status", component.get("physical_path_state"))
        path_id = builder.entity(
            page_key,
            "projected_path",
            source_id,
            component_record,
            f"rebar_program.physical_path_graph.components[id={source_id}]",
            {
                "topology": component.get("topology"),
                "bbox_display": deepcopy(component.get("bbox_display")),
                "projection_dimensionality": deepcopy(component.get("projection_dimensionality")),
                "projected_length": deepcopy(component.get("projected_length")),
                "physical_path_state": component.get("physical_path_state"),
                "view_ids": [view_ids[source] for source in component.get("view_ids", []) if source in view_ids],
                "object_ids": [object_ids[source] for source in component.get("object_instance_ids", []) if source in object_ids],
                "coordinate_scope_ids": deepcopy(component.get("coordinate_scope_ids", [])),
                "marks": deepcopy(component.get("mark_hypotheses", [])),
                "source_fragments": [
                    {
                        "id": fragment.get("id"),
                        "primitive_ref": fragment.get("primitive_ref"),
                        "source_path_ref": fragment.get("source_path_ref"),
                        "geometry": deepcopy(fragment.get("geometry")),
                    }
                    for fragment in source_fragments
                ],
            },
        )
        path_ids[source_id] = path_id
        for fragment_id in component.get("fragment_ids", []):
            fragment_to_paths[str(fragment_id)].add(path_id)

    for composite in composite_projected_bars:
        source_id = str(composite["id"])
        member_path_ids = [
            path_ids[str(component_id)]
            for component_id in composite.get("component_ids", []) or []
            if str(component_id) in path_ids
        ]
        builder.entity(
            page_key,
            "projected_bar_composite",
            source_id,
            composite,
            f"rebar_program.physical_path_graph.composite_projected_bars.proposals[id={source_id}]",
            {
                "mark": composite.get("mark"),
                "view_id": view_ids.get(str(composite.get("view_id"))),
                "member_projected_path_ids": member_path_ids,
                "source_fragment_ids": deepcopy(composite.get("fragment_ids", [])),
                "geometry_metrics": deepcopy(composite.get("geometry_metrics", {})),
                "certificate": composite.get("certificate"),
                "representation_state": composite.get("state"),
                "physical_placement_identity_inferred": False,
                "quantities_changed": False,
            },
        )
    detail_ids: dict[str, str] = {}
    for detail in details:
        source_id = str(detail["id"])
        detail_ids[source_id] = builder.entity(
            page_key,
            "detail_shape",
            source_id,
            detail,
            f"rebar_program.native_vector_detail_linking.details[id={source_id}]",
            {
                "marks": deepcopy(detail.get("marks", [])),
                "mark_display": detail.get("mark_display"),
                "bbox_display": deepcopy(detail.get("bbox_display")),
                "geometry_paths": deepcopy(detail.get("geometry_paths", [])),
                "fabrication_dimensions": deepcopy(detail.get("fabrication_dimensions", [])),
                "raster_fabrication_dimension_observations": deepcopy(
                    detail.get("raster_fabrication_dimension_observations", [])
                ),
            },
        )

    group_family_ids: dict[str, str] = {}
    for group in groups:
        source_id = str(group["id"])
        mark = group.get("identity", {}).get("mark", {}).get("value")
        group_record = deepcopy(dict(group))
        if group_record.get("epistemic_state") is None and group_record.get("state") is None:
            nested_state = (
                group.get("identity", {}).get("mark", {}).get("state")
                or group.get("role", {}).get("state")
            )
            group_record["epistemic_state"] = nested_state if nested_state in _EPISTEMIC_STATES else "derived"
        group_family_ids[source_id] = builder.entity(
            page_key,
            "bar_family",
            source_id,
            group_record,
            f"rebar_program.groups[id={source_id}]",
            {
                "source_family_type": "rebar_group",
                "mark": mark,
                "role": deepcopy(group.get("role")),
                "bar_spec": deepcopy(group.get("bar_spec")),
                "topology": deepcopy(group.get("topology")),
                "quantity": deepcopy(group.get("quantity")),
            },
        )

    physical_family_ids: dict[str, str] = {}
    for family in physical_families:
        source_id = str(family["id"])
        physical_family_ids[source_id] = builder.entity(
            page_key,
            "bar_family",
            source_id,
            family,
            f"rebar_program.native_vector_detail_linking.physical_families[id={source_id}]",
            {
                "source_family_type": "native_detail_family",
                "mark": family.get("mark"),
                "constraint_status": family.get("constraint_status"),
                "constraint_reason": family.get("constraint_reason"),
                "count": family.get("count"),
                "count_state": family.get("count_state"),
                "detail_ids": [detail_ids[source] for source in family.get("detail_ids", []) if source in detail_ids],
                "view_ids": [view_ids[source] for source in family.get("view_ids", []) if source in view_ids],
            },
        )

    dimension_ids: dict[str, str] = {}
    dimension_records: list[tuple[Mapping[str, Any], str]] = []
    for claim in page.get("claims", []) or []:
        if claim.get("kind") == "metric_dimension":
            dimension_records.append((claim, f"claims[id={claim.get('id')}]") )
    for claim in page.get("dimension_claims", []) or []:
        if isinstance(claim, Mapping):
            dimension_records.append((claim, f"dimension_claims[id={claim.get('id')}]") )
    for dimension, source_path in dimension_records:
        source_id = str(dimension.get("subject") or dimension.get("id") or f"dimension.{_digest(dimension)}")
        aliases = [str(dimension["id"])] if dimension.get("id") is not None else []
        dimension_id = builder.entity(
            page_key,
            "dimension",
            source_id,
            dimension,
            source_path,
            {
                "value": dimension.get("value") if dimension.get("value") is not None else dimension.get("value_mm"),
                "unit": dimension.get("unit", "mm" if dimension.get("value_mm") is not None else None),
                "kind": dimension.get("kind", "metric_dimension"),
                "confidence": dimension.get("confidence"),
            },
            aliases,
        )
        dimension_ids[source_id] = dimension_id
        for owner_key in ("owner_id", "target_id", "feature_id"):
            owner_source = dimension.get(owner_key)
            if owner_source is None:
                continue
            owner = builder.resolve(page_key, owner_source)
            builder.relation(
                page_key,
                "dimension_of",
                dimension_id,
                owner,
                dimension,
                source_path,
                f"{source_id}:{owner_key}:{owner_source}",
                {"owner_field": owner_key},
            )

    for detail in details:
        detail_source = str(detail["id"])
        detail_id = detail_ids[detail_source]
        for dimension in [
            *(detail.get("fabrication_dimensions", []) or []),
            *(detail.get("raster_fabrication_dimension_observations", []) or []),
        ]:
            if not isinstance(dimension, Mapping):
                continue
            source_id = str(dimension.get("id") or f"{detail_source}.dimension.{_digest(dimension)}")
            dimension_record = deepcopy(dict(dimension))
            dimension_record["source_detail_id"] = detail_source
            dimension_record["evidence_refs"] = sorted(
                {
                    *_evidence_refs(dimension),
                    *(detail.get("primitive_refs") or []),
                    detail_source,
                }
            )
            dimension_id = builder.entity(
                page_key,
                "dimension",
                source_id,
                dimension_record,
                f"rebar_program.native_vector_detail_linking.details[id={detail_source}].dimensions[{_digest(dimension)}]",
                {
                    "value": dimension.get("value") if dimension.get("value") is not None else dimension.get("value_mm"),
                    "unit": dimension.get("unit", "mm" if dimension.get("value_mm") is not None else None),
                    "kind": dimension.get("kind"),
                    "bbox_display": deepcopy(dimension.get("bbox_display")),
                    "confidence": dimension.get("confidence"),
                    "text": dimension.get("text"),
                    "method": dimension.get("method"),
                    "source_detail_id": detail_id,
                },
            )
            builder.relation(
                page_key,
                "dimension_of",
                dimension_id,
                detail_id,
                dimension_record,
                f"rebar_program.native_vector_detail_linking.details[id={detail_source}].dimensions",
                f"{source_id}:{detail_source}",
                {"ownership_scope": "detail_shape"},
                "derived",
            )

    for attachment in dimension_ownership.get("attachments", []) or []:
        if not isinstance(attachment, Mapping):
            continue
        source_id = str(attachment.get("id") or f"dimension_attachment.{_digest(attachment)}")
        attachment_id = builder.entity(
            page_key,
            "dimension_attachment",
            source_id,
            attachment,
            f"dimension_ownership.attachments[id={source_id}]",
            {
                "dimension_ref": attachment.get("dimension_ref"),
                "value_mm": attachment.get("value_mm"),
                "orientation": attachment.get("orientation"),
                "role": attachment.get("role"),
                "semantic_role": attachment.get("semantic_role"),
                "owner_entity_refs": deepcopy(attachment.get("owner_entity_refs", [])),
                "view_refs": deepcopy(attachment.get("view_refs", [])),
                "measured_endpoints": deepcopy(attachment.get("measured_endpoints")),
                "topology_segment_refs": deepcopy(attachment.get("topology_segment_refs", [])),
                "geometry_anchor_refs": deepcopy(attachment.get("geometry_anchor_refs", [])),
                "reason": attachment.get("reason"),
            },
        )
        if attachment.get("status") != "accepted":
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "dimension_ownership_attachment",
                    "attachment_id": attachment_id,
                    "source_id": source_id,
                    "status": attachment.get("status"),
                    "reason": attachment.get("reason") or "dimension owner is not uniquely supported",
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"dimension_ownership.attachments[id={source_id}]",
                        [source_id, str(attachment.get("dimension_ref"))],
                        attachment,
                    ),
                }
            )

    spacing_ids: dict[str, str] = {}
    for spacing in program.get("spacing_constraints", []) or []:
        source_id = str(spacing["id"])
        spacing_ids[source_id] = builder.entity(
            page_key,
            "spacing_constraint",
            source_id,
            spacing,
            f"rebar_program.spacing_constraints[id={source_id}]",
            {
                "spacing_mm": spacing.get("spacing_mm"),
                "interval_count": spacing.get("interval_count"),
                "extent_mm": spacing.get("extent_mm"),
                "mark_token": spacing.get("mark_token"),
                "constraint_type": spacing.get("type"),
                "bbox_display": deepcopy(spacing.get("bbox_display")),
                "arithmetic": deepcopy(spacing.get("arithmetic")),
            },
        )

    occurrence_ids: dict[tuple[str, str], str] = {}
    occurrence_targets: dict[str, set[str]] = defaultdict(set)
    for hypothesis in path_graph.get("mark_hypotheses", []) or []:
        source_id = str(hypothesis["id"])
        token = str(hypothesis.get("token", ""))
        occurrence_id = builder.entity(
            page_key,
            "mark_occurrence",
            source_id,
            hypothesis,
            f"rebar_program.physical_path_graph.mark_hypotheses[id={source_id}]",
            {
                "token": token,
                "source_occurrence_type": hypothesis.get("source_occurrence_type", "path_leader"),
                "text_role_id": hypothesis.get("text_role_id"),
                "bbox_display": deepcopy(hypothesis.get("bbox_display")),
                "source_role_basis": hypothesis.get("source_role_basis"),
                "leader_trace": deepcopy(hypothesis.get("leader_trace")),
            },
        )
        occurrence_ids[(source_id, token)] = occurrence_id
        target = _component_for_fragment(fragment_to_paths, hypothesis.get("fragment_id"))
        if hypothesis.get("state") == "accepted" and target is not None:
            builder.relation(
                page_key,
                "callout_targets",
                occurrence_id,
                target,
                hypothesis,
                f"rebar_program.physical_path_graph.mark_hypotheses[id={source_id}]",
                source_id,
                {"terminal_method": hypothesis.get("leader_trace", {}).get("terminal_method")},
                "derived",
            )
            occurrence_targets[occurrence_id].add(target)

    placement_by_id = {str(item["id"]): item for item in placements}
    placement_occurrences: dict[tuple[str, str], str] = {}
    for placement in placements:
        placement_source = str(placement["id"])
        for mark in placement.get("marks", []) or []:
            token = str(mark)
            source_id = f"{placement_source}.mark.{token}"
            occurrence_id = builder.entity(
                page_key,
                "mark_occurrence",
                source_id,
                placement,
                f"rebar_program.native_vector_detail_linking.placement_associations[id={placement_source}]",
                {
                    "token": token,
                    "source_occurrence_type": "detail_placement_callout",
                    "bbox_display": deepcopy(placement.get("target_mark_bbox_display")),
                    "view_ids": [view_ids[placement["target_view_id"]]] if placement.get("target_view_id") in view_ids else [],
                    "leader_trace": deepcopy(placement.get("leader_trace")),
                },
            )
            placement_occurrences[(placement_source, token)] = occurrence_id
            if placement.get("state") != "accepted":
                continue
            for attachment in placement.get("path_attachments", []) or []:
                if attachment.get("state") != "accepted":
                    continue
                target = _component_for_fragment(fragment_to_paths, attachment.get("fragment_id"))
                if target is None:
                    continue
                builder.relation(
                    page_key,
                    "callout_targets",
                    occurrence_id,
                    target,
                    placement,
                    f"rebar_program.native_vector_detail_linking.placement_associations[id={placement_source}]",
                    f"{source_id}:{attachment.get('fragment_id')}",
                    {"terminal_display": deepcopy(attachment.get("terminal_display"))},
                    "derived",
                )
                occurrence_targets[occurrence_id].add(target)

    section_occurrences: dict[tuple[str, str], str] = {}
    for section in page.get("section_rebar_observations", {}).get("sections", []) or []:
        section_source = str(section.get("section_id"))
        for candidate in section.get("candidates", []) or []:
            candidate_source = str(candidate.get("candidate_id"))
            for mark in candidate.get("leader_marks", []) or []:
                token = str(mark)
                source_id = f"{candidate_source}.mark.{token}"
                occurrence_id = builder.entity(
                    page_key,
                    "mark_occurrence",
                    source_id,
                    candidate,
                    f"section_rebar_observations.sections[id={section_source}].candidates[id={candidate_source}]",
                    {
                        "token": token,
                        "source_occurrence_type": "section_leader",
                        "section_id": section_source,
                        "bbox_display": deepcopy(candidate.get("bbox_display")),
                    },
                )
                section_occurrences[(candidate_source, token)] = occurrence_id
                fragment_sources: set[str] = set()
                for primitive in (candidate.get("primitive_ref"),):
                    if primitive is not None:
                        fragment_sources.update(primitive_to_fragments.get(str(primitive), set()))
                targets = {
                    target
                    for fragment_id in fragment_sources
                    if (target := _component_for_fragment(fragment_to_paths, fragment_id)) is not None
                }
                if candidate.get("state") == "resolved" and len(targets) == 1:
                    target = next(iter(targets))
                    builder.relation(
                        page_key,
                        "callout_targets",
                        occurrence_id,
                        target,
                        candidate,
                        f"section_rebar_observations.sections[id={section_source}].candidates[id={candidate_source}]",
                        source_id,
                        {"section_id": section_source},
                        "derived",
                    )
                    occurrence_targets[occurrence_id].add(target)

    for item in objects:
        object_source = str(item["id"])
        object_id = object_ids[object_source]
        section_source = item.get("section_view_id")
        if section_source in view_ids:
            builder.relation(
                page_key,
                "section_of",
                view_ids[section_source],
                object_id,
                item,
                f"object_instance_graph.instances[id={object_source}]",
                f"{object_source}:section:{section_source}",
                {"projection_role": item.get("projection_roles", {}).get(section_source)},
                _epistemic_state(item, "inferred"),
            )
        for source_view in item.get("view_ids", []) or []:
            if source_view not in view_ids:
                continue
            builder.relation(
                page_key,
                "projects_to",
                object_id,
                view_ids[source_view],
                item,
                f"object_instance_graph.instances[id={object_source}]",
                f"{object_source}:view:{source_view}",
                {"projection_role": item.get("projection_roles", {}).get(source_view)},
                _epistemic_state(item, "inferred"),
            )

    for view in views:
        parent_source = view.get("parent_view_id")
        if parent_source is None or str(view["id"]) in accepted_frame_cuts:
            continue
        source_id = str(view["id"])
        integration = view.get("integration_certificate", {}) or {}
        if not (
            integration.get("status") == "passed"
            and integration.get("title_segmentation_closed") is True
            and integration.get("dimension_adjudication_closed") is True
        ):
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "cut_at_relation_candidate",
                    "source_id": f"{source_id}:parent:{parent_source}",
                    "status": "candidate",
                    "reason": "legacy parent-view link lacks the passed title-segmentation and dimension-adjudication certificate",
                    "candidate": deepcopy(dict(view)),
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"view_hypotheses[id={source_id}]",
                        [source_id, str(parent_source)],
                        view,
                    ),
                }
            )
            continue
        builder.relation(
            page_key,
            "cut_at",
            view_ids[source_id],
            view_ids.get(str(parent_source)),
            view,
            f"view_hypotheses[id={source_id}]",
            f"{source_id}:parent:{parent_source}",
            {
                "cutting_plane": deepcopy(view.get("cutting_plane")),
                "integration_certificate": deepcopy(integration),
            },
        )

    for relation in view_frame_graph.get("relations", []) or []:
        if not isinstance(relation, Mapping) or relation.get("type") != "cut_at":
            continue
        source_id = str(relation.get("id") or f"cut_at.{_digest(relation)}")
        integration = relation.get("integration_certificate", {}) or {}
        certificate_passed = (
            integration.get("status") == "passed"
            and integration.get("title_segmentation_closed") is True
            and integration.get("dimension_adjudication_closed") is True
        )
        if relation.get("state") != "accepted" or not certificate_passed:
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "cut_at_relation_candidate",
                    "source_id": source_id,
                    "status": relation.get("state"),
                    "reason": relation.get("reason") or (
                        "cutting-plane relation lacks the passed title-segmentation and dimension-adjudication certificate"
                        if relation.get("state") == "accepted"
                        else "cutting-plane relation is not accepted"
                    ),
                    "candidate": deepcopy(dict(relation)),
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"view_frame_graph.relations[id={source_id}]",
                        [source_id],
                        relation,
                    ),
                }
            )
            continue
        section_source = str(relation.get("section_view_id"))
        parent_source = str(relation.get("parent_view_id"))
        section_id = view_ids.get(section_source)
        parent_id = view_ids.get(parent_source)
        if section_id is None or parent_id is None:
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "cut_at_relation_candidate",
                    "source_id": source_id,
                    "status": "invalid_endpoints",
                    "reason": "accepted cutting-plane relation references a missing canonical view",
                    "candidate": deepcopy(dict(relation)),
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"view_frame_graph.relations[id={source_id}]",
                        [source_id],
                        relation,
                    ),
                }
            )
            continue
        builder.relation(
            page_key,
            "cut_at",
            section_id,
            parent_id,
            relation,
            f"view_frame_graph.relations[id={source_id}]",
            source_id,
            {
                "section_label": relation.get("section_label"),
                "section_id": relation.get("section_id"),
                "parent_object": deepcopy(relation.get("parent_object")),
                "trace": deepcopy(relation.get("trace")),
                "integration_certificate": deepcopy(integration),
                "confidence": relation.get("confidence"),
                "unresolved_fields": deepcopy(relation.get("unresolved_fields", [])),
            },
            "derived",
        )

    for candidate in view_frame_graph.get("relation_candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        source_id = str(candidate.get("id") or f"cut_at_candidate.{_digest(candidate)}")
        builder.unresolved.append(
            {
                "page_key": page_key,
                "kind": "cut_at_relation_candidate",
                "source_id": source_id,
                "status": candidate.get("state", "candidate"),
                "reason": candidate.get("reason") or "cutting-plane relation remains ambiguous",
                "candidate": deepcopy(dict(candidate)),
                "provenance": _provenance(
                    builder.document_key,
                    page_key,
                    f"view_frame_graph.relation_candidates[id={source_id}]",
                    [source_id],
                    candidate,
                ),
            }
        )

    for group in groups:
        group_source = str(group["id"])
        family_id = group_family_ids[group_source]
        for component_source in group.get("placement", {}).get("path_component_ids", []) or []:
            if component_source in path_ids:
                builder.relation(
                    page_key,
                    "projects_to",
                    family_id,
                    path_ids[component_source],
                    group,
                    f"rebar_program.groups[id={group_source}]",
                    f"{group_source}:component:{component_source}",
                    {"family_source": "rebar_group"},
                )

    for identity in path_graph.get("cross_view_projection_identities", []) or []:
        if identity.get("state") != "accepted" or len(set(identity.get("view_ids", []))) < 2:
            continue
        canonical_paths = sorted({path_ids[source] for source in identity.get("component_ids", []) if source in path_ids})
        if identity.get("group_id") in group_family_ids:
            family_id = group_family_ids[identity["group_id"]]
            for path_id in canonical_paths:
                builder.relation(
                    page_key,
                    "projects_to",
                    family_id,
                    path_id,
                    identity,
                    f"rebar_program.physical_path_graph.cross_view_projection_identities[id={identity.get('id')}]",
                    f"{identity.get('id')}:family_projection:{path_id}",
                    {"identity_id": identity.get("id")},
                )
        elif len(canonical_paths) > 1:
            anchor = canonical_paths[0]
            for path_id in canonical_paths[1:]:
                builder.relation(
                    page_key,
                    "same_bar_family",
                    path_id,
                    anchor,
                    identity,
                    f"rebar_program.physical_path_graph.cross_view_projection_identities[id={identity.get('id')}]",
                    f"{identity.get('id')}:{path_id}:{anchor}",
                    {"identity_id": identity.get("id")},
                )

    for family in physical_families:
        family_source = str(family["id"])
        family_id = physical_family_ids[family_source]
        for detail_source in family.get("detail_ids", []) or []:
            if detail_source in detail_ids:
                builder.relation(
                    page_key,
                    "detail_defines",
                    detail_ids[detail_source],
                    family_id,
                    family,
                    f"rebar_program.native_vector_detail_linking.physical_families[id={family_source}]",
                    f"{family_source}:detail:{detail_source}",
                )
        for component_source in family.get("component_ids", []) or []:
            if component_source in path_ids:
                builder.relation(
                    page_key,
                    "projects_to",
                    family_id,
                    path_ids[component_source],
                    family,
                    f"rebar_program.native_vector_detail_linking.physical_families[id={family_source}]",
                    f"{family_source}:component:{component_source}",
                    {"family_source": "native_detail_family"},
                )
        token = str(family.get("mark", ""))
        for placement_source in family.get("placement_association_ids", []) or []:
            occurrence_id = placement_occurrences.get((str(placement_source), token))
            if occurrence_id is None:
                continue
            builder.relation(
                page_key,
                "same_bar_family",
                occurrence_id,
                family_id,
                family,
                f"rebar_program.native_vector_detail_linking.physical_families[id={family_source}]",
                f"{family_source}:placement:{placement_source}:{token}",
            )
            for target in occurrence_targets.get(occurrence_id, set()):
                builder.relation(
                    page_key,
                    "projects_to",
                    family_id,
                    target,
                    placement_by_id.get(str(placement_source), family),
                    f"rebar_program.native_vector_detail_linking.placement_associations[id={placement_source}]",
                    f"{family_source}:placement_target:{placement_source}:{target}",
                )
        for candidate_source in family.get("section_candidate_ids", []) or []:
            occurrence_id = section_occurrences.get((str(candidate_source), token))
            if occurrence_id is None:
                continue
            builder.relation(
                page_key,
                "same_bar_family",
                occurrence_id,
                family_id,
                family,
                f"rebar_program.native_vector_detail_linking.physical_families[id={family_source}]",
                f"{family_source}:section_candidate:{candidate_source}:{token}",
            )

    accepted_spacing = {
        (str(item.get("constraint_id")), str(item.get("group_id"))): item
        for item in program.get("spacing_associations", []) or []
        if item.get("status") == "accepted"
    }
    for (constraint_source, group_source), association in accepted_spacing.items():
        builder.relation(
            page_key,
            "spacing_of",
            spacing_ids.get(constraint_source),
            group_family_ids.get(group_source),
            association,
            f"rebar_program.spacing_associations[id={association.get('id')}]",
            str(association.get("id") or f"{constraint_source}:{group_source}"),
            {"view_id": view_ids.get(str(association.get("view_id")))},
            "derived",
        )

    for relation in dimension_ownership.get("relations", []) or []:
        if not isinstance(relation, Mapping) or relation.get("type") != "dimension_of":
            continue
        status = relation.get("status")
        state = relation.get("state")
        accepted = status == "accepted" if status is not None else state in (_EPISTEMIC_STATES - {"unknown"})
        if not accepted:
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "dimension_ownership_relation",
                    "source_id": relation.get("id"),
                    "status": status or state,
                    "reason": relation.get("reason") or "dimension ownership relation is not accepted",
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"dimension_ownership.relations[id={relation.get('id')}]",
                        [str(relation.get("id"))],
                        relation,
                    ),
                }
            )
            continue
        if relation.get("from") is None or relation.get("to") is None:
            builder.unresolved.append(
                {
                    "page_key": page_key,
                    "kind": "dimension_ownership_relation",
                    "source_id": relation.get("id"),
                    "status": status or state,
                    "reason": "accepted dimension ownership relation has a missing endpoint",
                    "provenance": _provenance(
                        builder.document_key,
                        page_key,
                        f"dimension_ownership.relations[id={relation.get('id')}]",
                        [str(relation.get("id"))],
                        relation,
                    ),
                }
            )
            continue
        source_ref = str(relation["from"])
        target_ref = str(relation["to"])
        source = builder.resolve(page_key, source_ref) or builder.reference(
            page_key, source_ref, "dimension_ownership.relations"
        )
        target = builder.resolve(page_key, target_ref) or builder.reference(
            page_key, target_ref, "dimension_ownership.relations"
        )
        builder.relation(
            page_key,
            "dimension_of",
            source,
            target,
            relation,
            f"dimension_ownership.relations[id={relation.get('id')}]",
            str(relation.get("id") or _digest(relation)),
            {"source_payload": "dimension_ownership"},
            state if state in _EPISTEMIC_STATES else None,
        )

    for relation in page.get("relations", []) or []:
        relation_type = relation.get("type")
        source_path = f"relations[type={relation_type}]"
        if relation_type in RELATION_TYPES:
            if relation.get("from") is None or relation.get("to") is None:
                builder.unresolved.append(
                    {
                        "page_key": page_key,
                        "relation_type": relation_type,
                        "source_key": str(relation.get("id") or _digest(relation)),
                        "reason": "explicit typed relation has a missing endpoint",
                    }
                )
                continue
            source_ref = str(relation["from"])
            target_ref = str(relation["to"])
            source = builder.resolve(page_key, source_ref) or builder.reference(page_key, source_ref, source_path)
            target = builder.resolve(page_key, target_ref) or builder.reference(page_key, target_ref, source_path)
            builder.relation(
                page_key,
                relation_type,
                source,
                target,
                relation,
                source_path,
                str(relation.get("id") or _digest(relation)),
            )
        elif relation_type == "projection_of_object_instance":
            view = builder.resolve(page_key, relation.get("from"), ("view",))
            obj = builder.resolve(page_key, relation.get("to"), ("object",))
            builder.relation(
                page_key,
                "projects_to",
                obj,
                view,
                relation,
                source_path,
                str(relation.get("id") or _digest(relation)),
                {"source_relation_type": relation_type},
            )


def build_canonical_knowledge_graph(
    engineering: Any,
    *,
    document_key: str | None = None,
) -> dict[str, Any]:
    """Normalize one page or a multi-page engineering document.

    Accepted inputs are a page mapping, a sequence of page mappings, a document
    mapping with ``pages``, or a record containing ``engineering_graph``.  The
    input is never modified.  Unresolved evidence remains an entity without a
    semantic relation rather than being joined by mark text or proximity.
    """

    pages = _coerce_pages(engineering)
    resolved_document_key = _document_key(engineering, pages, document_key)
    keys = _page_keys(pages)
    builder = _Builder(resolved_document_key)
    for page, page_key in zip(pages, keys):
        _build_page(builder, page, page_key)

    entities = sorted(builder.entities.values(), key=lambda item: item["id"])
    relations = sorted(builder.relations.values(), key=lambda item: item["id"])
    entity_ids = {item["id"] for item in entities}
    relation_counts = Counter(item["type"] for item in relations)
    entity_counts = Counter(item["entity_type"] for item in entities)
    cut_relations = [item for item in relations if item["type"] == "cut_at"]
    cut_pairs = [(item["from"], item["to"]) for item in cut_relations]
    unresolved_counts = Counter(item.get("kind") or item.get("relation_type") or "other" for item in builder.unresolved)
    source_index = [
        {
            "document_key": resolved_document_key,
            "page_key": page_key,
            "entity_type": entity_type,
            "source_id": source_id,
            "canonical_id": canonical_id,
        }
        for (page_key, entity_type, source_id), canonical_id in sorted(builder.source_index.items())
    ]
    return {
        "schema_version": "0.1.0",
        "layer": "canonical_drawing_knowledge_graph",
        "status": "populated" if entities else "empty",
        "document_key": resolved_document_key,
        "page_keys": keys,
        "entities": entities,
        "relations": relations,
        "source_index": source_index,
        "unresolved": sorted(builder.unresolved, key=_json),
        "validation": {
            "unique_entity_ids": len(entity_ids) == len(entities),
            "unique_relation_ids": len({item["id"] for item in relations}) == len(relations),
            "relation_endpoints_exist": all(
                item["from"] in entity_ids and item["to"] in entity_ids for item in relations
            ),
            "supported_relation_types_only": all(item["type"] in RELATION_TYPES for item in relations),
            "duplicate_source_records": builder.duplicate_sources,
            "filename_dispatch_used": False,
            "schedule_values_used": False,
            "implicit_equal_mark_merge_used": False,
            "accepted_cut_relation_section_parent_pairs_are_unique": len(cut_pairs) == len(set(cut_pairs)),
        },
        "contract": {
            "canonical_ids_are_document_qualified": True,
            "canonical_ids_are_page_qualified": True,
            "source_evidence_remains_authoritative": True,
            "equal_mark_text_is_not_physical_identity": True,
            "ambiguous_relations_fail_closed": True,
            "missing_view_frame_fields_remain_unknown": True,
        },
        "summary": {
            "page_count": len(pages),
            "entity_count": len(entities),
            "relation_count": len(relations),
            "unresolved_count": len(builder.unresolved),
            "cut_candidate_count": unresolved_counts.get("cut_at_relation_candidate", 0),
            "entities_by_type": {key: entity_counts.get(key, 0) for key in sorted(entity_counts)},
            "relations_by_type": {key: relation_counts.get(key, 0) for key in RELATION_TYPES},
            "unresolved_by_type": {key: unresolved_counts[key] for key in sorted(unresolved_counts)},
        },
    }
