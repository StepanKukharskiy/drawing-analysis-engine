"""Replay certified solid-hypothesis records into the Step 4 kernel.

This adapter translates exact canonical aliases back to frozen native records.
It consumes shared-coordinate and profile reclosure certificates produced by
their owning stages; it never derives meshes, placement, axes, or projections.
Kernel acceptance is necessary but not sufficient: replay acceptance also
requires a current graph binding and certified, successfully reclosed upstream
relations.  Quantities remain read-only even after the assembly is previewable.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from src.drawing_engine.disciplines.concrete.multi_component_solid_kernel import solve_multi_component_solid


SCHEMA_VERSION = "0.1.0"
RECORD_COLLECTION = "solid_hypothesis_replay_records"


class _InsufficientConstraints(ValueError):
    pass


class _ReplayContradiction(ValueError):
    pass


def _refs(value: Any, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise _InsufficientConstraints(f"{field} has no evidence_refs")
    refs = sorted({str(item) for item in value if item is not None and str(item)})
    if not refs:
        raise _InsufficientConstraints(f"{field} has no evidence_refs")
    return refs


def _rows_by_id(
    rows: Any,
    field: str,
    *,
    key: str = "id",
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        raise _InsufficientConstraints(f"{field} records are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get(key) is None:
            raise _InsufficientConstraints(f"{field} contains a record without an id")
        row_id = str(row[key])
        if row_id in result:
            raise _ReplayContradiction(f"{field} contains duplicate id {row_id}")
        result[row_id] = row
    return result


def _native_alias(entity: Mapping[str, Any], field: str) -> str:
    explicit = entity.get("native_entity_id")
    if explicit is not None:
        return str(explicit)
    aliases = sorted({str(item) for item in entity.get("native_source_ids", []) or [] if str(item)})
    if len(aliases) != 1:
        raise _InsufficientConstraints(f"{field} does not translate to one exact native id")
    return aliases[0]


def _result(
    status: str,
    reason: str,
    *,
    record_id: str | None,
    gates: Mapping[str, Any] | None = None,
    translation: Mapping[str, Any] | None = None,
    kernel_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = status == "reclosed_pass" and kernel_result is not None
    validation_records = {
        "components": deepcopy(kernel_result.get("components", [])) if accepted else [],
        "interfaces": deepcopy(kernel_result.get("shared_interfaces", [])) if accepted else [],
        "overlaps": deepcopy(kernel_result.get("pair_validations", [])) if accepted else [],
        "volume": deepcopy(kernel_result.get("volume_validation", {})) if accepted else {},
        "reprojections": deepcopy(kernel_result.get("supplied_view_reprojections", [])) if accepted else [],
    }
    preview = None
    if accepted:
        preview = {
            "mesh": deepcopy(kernel_result["assembly_mesh"]),
            "source": "certified_multi_component_solid_replay",
            "kernel_input_sha256": kernel_result["input_sha256"],
            "validation": deepcopy(validation_records),
            "quantity_eligible_for_step5": True,
            "quantity_writes_allowed": False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "solid_hypothesis",
        "status": status,
        "reason": reason,
        "record_id": record_id,
        "gates": deepcopy(dict(gates or {})),
        "translation": deepcopy(dict(translation or {})),
        "kernel_result": deepcopy(dict(kernel_result)) if accepted else None,
        "solid_preview": preview,
        "audit": {
            "status": "accepted" if accepted else "abstained",
            "reason": reason,
            "validation_records": validation_records,
            "abstention": None if accepted else {"status": status, "reason": reason},
        },
        "contract": {
            "certified_relations_only": True,
            "exact_native_id_translation_required": True,
            "shared_coordinate_reclosure_required": True,
            "profile_reclosure_required": True,
            "kernel_acceptance_required": True,
            "kernel_quantity_eligible": bool(
                accepted and kernel_result.get("contract", {}).get("quantity_eligible")
            ),
            "upstream_reclosure_proven": accepted,
            "quantity_eligible_for_step5": accepted,
            "quantity_writes_allowed": False,
            "schedule_values_used": False,
        },
    }


def _candidate_records(engineering_pages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        record
        for page in engineering_pages
        for record in page.get(RECORD_COLLECTION, []) or []
        if isinstance(record, Mapping)
    ]


def replay_multi_component_solid(
    *,
    canonical_graph_sha256: str,
    constraints: Sequence[Mapping[str, Any]],
    native_entities: Sequence[Mapping[str, Any]],
    engineering_pages: Sequence[Mapping[str, Any]],
    shared_coordinate_reclosure: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate one frozen native replay record and rerun every Step 4 gate."""

    candidates = _candidate_records(engineering_pages)
    if not candidates:
        return _result(
            "insufficient_constraints",
            "no frozen solid-hypothesis replay record supplies profiles, transforms, interfaces, and projections",
            record_id=None,
        )
    if len(candidates) != 1:
        return _result(
            "reclosed_fail",
            "multiple frozen solid-hypothesis replay records are not uniquely selectable",
            record_id=None,
        )
    record = candidates[0]
    record_id = str(record.get("id") or "") or None
    gates: dict[str, Any] = {}
    translation: dict[str, Any] = {}
    try:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise _ReplayContradiction("solid replay record has an unsupported schema_version")
        if record_id is None:
            raise _InsufficientConstraints("solid replay record has no stable id")
        if record.get("base_canonical_graph_sha256") != canonical_graph_sha256:
            raise _ReplayContradiction("solid replay record is bound to a stale canonical graph hash")
        gates["graph_binding"] = {"status": "pass", "canonical_graph_sha256": canonical_graph_sha256}
        _refs(record.get("evidence_refs"), str(record_id or "solid replay record"))

        constraints_by_id = _rows_by_id(
            constraints,
            "certified solid constraints",
            key="relation_id",
        )
        for constraint_id, constraint in constraints_by_id.items():
            if constraint.get("relation_type") not in {"cut_at", "dimension_of", "projects_to", "section_of"}:
                raise _ReplayContradiction(f"{constraint_id} has an unsupported solid relation type")
            _refs(constraint.get("native_evidence_refs"), constraint_id)
            _refs(constraint.get("native_relation_ids"), constraint_id)
            if not constraint.get("decision_id"):
                raise _InsufficientConstraints(f"{constraint_id} has no certification decision id")
        gates["certified_relations"] = {
            "status": "pass",
            "constraint_ids": sorted(constraints_by_id),
        }

        if shared_coordinate_reclosure.get("status") == "reclosed_fail":
            raise _ReplayContradiction("shared-coordinate replay failed")
        if shared_coordinate_reclosure.get("status") != "reclosed_pass":
            raise _InsufficientConstraints("shared-coordinate replay has not reclosed")
        coordinate_gates = shared_coordinate_reclosure.get("gates", {}) or {}
        if any(
            coordinate_gates.get(name, {}).get("status") != "pass"
            for name in ("metric", "uniqueness", "axis", "reprojection")
        ):
            raise _InsufficientConstraints("shared-coordinate replay gates are incomplete")
        coordinate_ids = set(
            map(str, shared_coordinate_reclosure.get("certified_constraint_ids", []) or [])
        )
        if not coordinate_ids:
            raise _InsufficientConstraints("shared-coordinate replay has no certified relation ids")
        if not coordinate_ids <= constraints_by_id.keys():
            raise _InsufficientConstraints("shared-coordinate replay cites relations outside the certified solid slice")
        gates["shared_coordinate_reclosure"] = {
            "status": "pass",
            "certified_constraint_ids": sorted(coordinate_ids),
        }
        reclosed_transform_ids = set(
            map(str, shared_coordinate_reclosure.get("physical_component_transform_ids", []) or [])
        )
        reclosed_projection_view_ids = set(
            map(str, shared_coordinate_reclosure.get("projection_view_ids", []) or [])
        )
        if not reclosed_transform_ids:
            raise _InsufficientConstraints(
                "shared-coordinate replay does not certify physical component transforms"
            )
        if len(reclosed_projection_view_ids) < 2:
            raise _InsufficientConstraints(
                "shared-coordinate replay does not certify two projection-view axes"
            )

        profile_reclosure = record.get("profile_reclosure")
        if not isinstance(profile_reclosure, Mapping):
            raise _InsufficientConstraints("profile reclosure record is missing")
        if profile_reclosure.get("base_canonical_graph_sha256") != canonical_graph_sha256:
            raise _ReplayContradiction("profile reclosure is bound to a stale canonical graph hash")
        if profile_reclosure.get("status") == "reclosed_fail":
            raise _ReplayContradiction("profile replay failed")
        if profile_reclosure.get("status") != "reclosed_pass":
            raise _InsufficientConstraints("profile replay has not reclosed")
        profile_gates = profile_reclosure.get("gates", {}) or {}
        if any(
            profile_gates.get(name, {}).get("status") != "pass"
            for name in ("closure", "uniqueness", "evidence")
        ):
            raise _InsufficientConstraints("profile replay gates are incomplete")
        profile_constraint_ids = set(
            map(str, profile_reclosure.get("certified_constraint_ids", []) or [])
        )
        if not profile_constraint_ids or not profile_constraint_ids <= constraints_by_id.keys():
            raise _InsufficientConstraints("profile replay does not cite certified solid relations")
        gates["profile_reclosure"] = {
            "status": "pass",
            "certified_constraint_ids": sorted(profile_constraint_ids),
        }
        reclosed_profile_ids = set(map(str, profile_reclosure.get("profile_ids", []) or []))
        if not reclosed_profile_ids:
            raise _InsufficientConstraints("profile replay has no reclosed profile ids")
        reclosed_interface_ids = set(map(str, profile_reclosure.get("interface_ids", []) or []))

        entities_by_id = _rows_by_id(
            native_entities,
            "native entity translations",
            key="canonical_id",
        )
        profiles_by_id = _rows_by_id(record.get("profiles"), "assembled profiles")
        transforms_by_id = _rows_by_id(record.get("physical_component_transforms"), "physical transforms")
        component_rows = record.get("components")
        if not isinstance(component_rows, (list, tuple)) or not component_rows:
            raise _InsufficientConstraints("at least one solid component record is required")
        if not profiles_by_id:
            raise _InsufficientConstraints("assembled profile records are missing")
        if not transforms_by_id:
            raise _InsufficientConstraints("physical component transforms are missing")

        kernel_components = []
        component_translation: dict[str, dict[str, str]] = {}
        canonical_component_by_native: dict[str, str] = {}
        for component in sorted(component_rows, key=lambda item: str(item.get("id"))):
            component_id = str(component.get("id"))
            if not component_id or component_id == "None":
                raise _InsufficientConstraints("solid component record has no native id")
            _refs(component.get("evidence_refs"), component_id)
            canonical_ref = str(component.get("canonical_component_ref"))
            canonical_profile_ref = str(component.get("canonical_profile_ref"))
            entity = entities_by_id.get(canonical_ref)
            profile_entity = entities_by_id.get(canonical_profile_ref)
            if entity is None or profile_entity is None:
                raise _InsufficientConstraints(f"{component_id} canonical component or profile translation is missing")
            if _native_alias(entity, canonical_ref) != component_id:
                raise _ReplayContradiction(f"{component_id} contradicts its canonical native alias")
            profile_ref = str(component.get("profile_ref"))
            if _native_alias(profile_entity, canonical_profile_ref) != profile_ref:
                raise _ReplayContradiction(f"{component_id} profile_ref contradicts its canonical native alias")
            profile = profiles_by_id.get(profile_ref)
            if profile is None:
                raise _InsufficientConstraints(f"{component_id} assembled profile is missing")
            if (
                profile.get("record_type") != "assembled_profile_candidate"
                or profile.get("record_version") != SCHEMA_VERSION
            ):
                raise _ReplayContradiction(f"{profile_ref} has an unsupported profile record contract")
            closure = profile.get("closure", {}) or {}
            if profile.get("state") != "resolved" or not all(
                closure.get(field) is True
                for field in (
                    "closed",
                    "branch_free",
                    "unique_completion",
                    "scale_bounded",
                    "dimensionally_redundant",
                )
            ):
                raise _InsufficientConstraints(f"{profile_ref} profile closure is unresolved")
            if profile_ref not in reclosed_profile_ids:
                raise _InsufficientConstraints(f"{profile_ref} is absent from the profile reclosure certificate")
            if profile.get("quantity_eligible") is not False:
                raise _ReplayContradiction(f"{profile_ref} profile bypasses the Step 3 quantity boundary")
            _refs(profile.get("evidence_refs"), profile_ref)
            profile_relation_ref = str(component.get("profile_relation_ref"))
            profile_relation = constraints_by_id.get(profile_relation_ref)
            if (
                profile_relation is None
                or profile_relation.get("relation_type") != "dimension_of"
                or str(profile_relation.get("to")) != canonical_profile_ref
                or profile_relation_ref not in profile_constraint_ids
            ):
                raise _InsufficientConstraints(f"{component_id} profile is not backed by one reclosed dimension relation")

            transform_ref = str(component.get("transform_ref"))
            transform = transforms_by_id.get(transform_ref)
            if transform is None:
                raise _InsufficientConstraints(f"{component_id} physical transform is missing")
            if transform_ref not in reclosed_transform_ids:
                raise _InsufficientConstraints(
                    f"{transform_ref} is absent from the shared-coordinate reclosure certificate"
                )
            if str(transform.get("component_ref")) != component_id:
                raise _ReplayContradiction(f"{transform_ref} references another component")
            if transform.get("axis_signs") != "resolved":
                raise _InsufficientConstraints(f"{transform_ref} axis signs are unresolved")
            if transform.get("state") != "resolved":
                raise _InsufficientConstraints(f"{transform_ref} physical transform is unresolved")
            if transform.get("placement_role") != "physical":
                raise _ReplayContradiction(f"{transform_ref} is a presentation-only transform")
            _refs(transform.get("evidence_refs"), transform_ref)
            transform_certificate_ids = set(
                map(str, transform.get("certified_relation_ids", []) or [])
            )
            if not transform_certificate_ids or not transform_certificate_ids <= coordinate_ids:
                raise _InsufficientConstraints(f"{transform_ref} is not backed by reclosed coordinate relations")

            canonical_component_by_native[component_id] = canonical_ref
            component_translation[component_id] = {
                "canonical_component_ref": canonical_ref,
                "native_component_ref": component_id,
                "canonical_profile_ref": canonical_profile_ref,
                "native_profile_ref": profile_ref,
                "physical_transform_ref": transform_ref,
            }
            kernel_components.append(
                {
                    "id": component_id,
                    "mesh": deepcopy(component.get("mesh")),
                    "transform": deepcopy(dict(transform)),
                    "analytic_volume": deepcopy(component.get("analytic_volume")),
                    "evidence_refs": list(component.get("evidence_refs", [])),
                }
            )

        interfaces = record.get("shared_interfaces")
        if not isinstance(interfaces, (list, tuple)):
            raise _InsufficientConstraints("shared interface records are missing")
        if len(kernel_components) == 1:
            if interfaces or reclosed_interface_ids:
                raise _ReplayContradiction(
                    "a single component cannot declare an inter-component shared interface"
                )
            if profile_reclosure.get("no_external_interfaces_required") is not True:
                raise _InsufficientConstraints(
                    "single-component profile replay must explicitly require no external interface"
                )
        elif not interfaces or not reclosed_interface_ids:
            raise _InsufficientConstraints("profile replay has no reclosed shared interface ids")
        for interface in interfaces:
            interface_id = str(interface.get("id"))
            _refs(interface.get("evidence_refs"), interface_id)
            if interface.get("state") != "resolved":
                raise _InsufficientConstraints(f"{interface_id} shared interface is unresolved")
            if interface_id not in reclosed_interface_ids:
                raise _InsufficientConstraints(
                    f"{interface_id} is absent from the profile reclosure certificate"
                )
            certificate_ids = set(map(str, interface.get("certified_relation_ids", []) or []))
            if not certificate_ids or not certificate_ids <= constraints_by_id.keys():
                raise _InsufficientConstraints(f"{interface_id} has no certified relation support")
            interface_component_refs = set(map(str, interface.get("component_refs", []) or []))
            certified_component_refs = {
                component_ref
                for component_ref, canonical_ref in canonical_component_by_native.items()
                for relation_id in certificate_ids
                if constraints_by_id[relation_id].get("relation_type") == "projects_to"
                and str(constraints_by_id[relation_id].get("from")) == canonical_ref
            }
            if certified_component_refs != interface_component_refs:
                raise _InsufficientConstraints(
                    f"{interface_id} is not supported by certified projections of both components"
                )

        supplied_views = record.get("supplied_views")
        if not isinstance(supplied_views, (list, tuple)) or len(supplied_views) < 2:
            raise _InsufficientConstraints("two supplied projection records are required")
        translated_views = []
        view_translation = {}
        for view in supplied_views:
            view_id = str(view.get("id"))
            canonical_view_ref = str(view.get("canonical_view_ref"))
            entity = entities_by_id.get(canonical_view_ref)
            if entity is None:
                raise _InsufficientConstraints(f"{view_id} canonical view translation is missing")
            if _native_alias(entity, canonical_view_ref) != view_id:
                raise _ReplayContradiction(f"{view_id} contradicts its canonical native alias")
            if view_id not in reclosed_projection_view_ids:
                raise _InsufficientConstraints(
                    f"{view_id} axes are absent from the shared-coordinate reclosure certificate"
                )
            _refs(view.get("evidence_refs"), view_id)
            projections = view.get("component_projections")
            if not isinstance(projections, (list, tuple)) or len(projections) != len(kernel_components):
                raise _InsufficientConstraints(f"{view_id} does not project every component")
            for projection in projections:
                component_ref = str(projection.get("component_ref"))
                canonical_component_ref = canonical_component_by_native.get(component_ref)
                relation_ref = str(projection.get("certified_relation_id"))
                relation = constraints_by_id.get(relation_ref)
                if (
                    canonical_component_ref is None
                    or relation is None
                    or relation.get("relation_type") != "projects_to"
                    or str(relation.get("from")) != canonical_component_ref
                    or str(relation.get("to")) != canonical_view_ref
                ):
                    raise _InsufficientConstraints(
                        f"{view_id}.{component_ref} projection is not backed by one certified projects_to relation"
                    )
                _refs(projection.get("evidence_refs"), f"{view_id}.{component_ref}")
            translated = deepcopy(dict(view))
            translated.pop("canonical_view_ref", None)
            for projection in translated.get("component_projections", []) or []:
                projection.pop("certified_relation_id", None)
            translated_views.append(translated)
            view_translation[canonical_view_ref] = view_id

        translation = {
            "status": "pass",
            "components": component_translation,
            "views": view_translation,
        }
        gates["input_completeness"] = {
            "status": "pass",
            "component_count": len(kernel_components),
            "profile_count": len(profiles_by_id),
            "transform_count": len(transforms_by_id),
            "interface_count": len(interfaces),
            "projection_view_count": len(translated_views),
        }
        kernel_result = solve_multi_component_solid(
            kernel_components,
            [
                {
                    key: deepcopy(value)
                    for key, value in interface.items()
                    if key not in {"state", "certified_relation_ids"}
                }
                for interface in interfaces
            ],
            translated_views,
        )
        if kernel_result.get("status") != "accepted":
            raise _ReplayContradiction("multi-component solid kernel did not accept the translated record")
        gates["kernel"] = {
            "status": "pass",
            "input_sha256": kernel_result["input_sha256"],
        }
        return _result(
            "reclosed_pass",
            "certified native solid inputs and every multi-component kernel gate reclosed",
            record_id=record_id,
            gates=gates,
            translation=translation,
            kernel_result=kernel_result,
        )
    except _InsufficientConstraints as exc:
        return _result(
            "insufficient_constraints",
            str(exc),
            record_id=record_id,
            gates=gates,
            translation=translation,
        )
    except (_ReplayContradiction, ValueError, TypeError, KeyError) as exc:
        return _result(
            "reclosed_fail",
            str(exc),
            record_id=record_id,
            gates=gates,
            translation=translation,
        )
