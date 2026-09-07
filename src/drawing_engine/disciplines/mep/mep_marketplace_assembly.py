"""Serial MEP physical-run and marketplace-assembly promotion.

M5C projected networks are immutable inputs.  This layer accepts a physical
run only when one evidence record accounts for every projected member, closes
every port, and supplies a replayable three-dimensional centreline.  Assembly
rules and catalog mappings are separate, versioned inputs; absent commercial
facts remain explicit requirements and never become guessed SKUs.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Iterable, Mapping, Sequence

from src.drawing_engine.project.takeoff_intelligence import canonical_sha256, stable_id


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_marketplace_assembly_takeoff"
TERMINAL_KINDS = {
    "physical_terminal", "equipment_port", "fitting_port", "riser_drop",
    "cross_sheet_continuation", "detail_section_interface",
    "package_boundary", "unresolved",
}
FIXED_PORT_COUNTS = {
    "elbow": 2, "tee": 3, "wye": 3, "cross": 4,
    "reducer": 2, "transition": 2, "valve": 2, "damper": 2,
}
CRITICAL_CATALOG_FIELDS = {
    "straight_material": ("material", "specification", "nominal_size", "connection_standard"),
    "component": ("component_type", "nominal_size", "connection_standard"),
}


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _polyline_length(points: Sequence[Sequence[object]]) -> float | None:
    if len(points) < 2:
        return None
    total = 0.0
    for left, right in zip(points, points[1:]):
        if len(left) != 3 or len(right) != 3 or not all(_finite(value) for value in (*left, *right)):
            return None
        step = math.dist([float(value) for value in left], [float(value) for value in right])
        if step <= 0:
            return None
        total += step
    return total


def _value_channel(
    value: float | None, unit: str = "m", *, scope: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "calculated": {
            "value": value, "unit": unit if value is not None else None,
            "measurement_scope": deepcopy(dict(scope)) if scope else None,
        },
        "declared": {"value": None, "unit": None},
        "reviewed": {"value": None, "unit": None},
        "approved": {"value": None, "unit": None},
    }


def _projected_measurements(
    run_ref: str, rows: Iterable[Mapping[str, Any]]
) -> tuple[float | None, list[str], list[str], dict[str, Any] | None]:
    candidates = [row for row in rows if str(row.get("network_run_ref")) == run_ref]
    reasons = []
    accepted = [row for row in candidates if row.get("state") == "observed"]
    values = {(float(row["value_m"]), row.get("measurement_scope_ref"))
              for row in accepted if _finite_nonnegative(row.get("value_m"))}
    if len(accepted) != 1 or len(values) != 1:
        if candidates:
            reasons.append("projected_length_measurement_not_unique")
        evidence_refs = sorted({
            str(ref) for row in accepted
            for ref in (row.get("measurement_scope_ref"), *row.get("evidence_refs", []))
            if ref
        })
        scope = None if not accepted else {
            "scope_kind": "multiple_bounded_projected_scopes",
            "complete_run_coverage": False,
            "candidates": [{
                "scope_ref": row.get("measurement_scope_ref"),
                "value_m": row.get("value_m"),
                "scope_kind": row.get("measurement_scope_kind", "explicit_projected_scope"),
            } for row in sorted(accepted, key=lambda item: str(item.get("id")))],
        }
        return None, evidence_refs, reasons, scope
    value, scope_ref = next(iter(values))
    if not scope_ref or not accepted[0].get("evidence_refs"):
        return None, [], ["projected_length_measurement_lacks_scope_or_evidence"], None
    scope = {
        "scope_ref": scope_ref,
        "scope_kind": accepted[0].get("measurement_scope_kind", "explicit_projected_scope"),
        "complete_run_coverage": accepted[0].get("complete_run_coverage") is True,
    }
    return value, sorted({scope_ref, *map(str, accepted[0]["evidence_refs"])}), [], scope


def projected_length_observations_from_m5c(
    network_hierarchy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Adapt replayed complete subtraces into explicitly scoped 2D lengths."""
    if network_hierarchy.get("layer") != "mep_projected_network_hierarchy":
        raise ValueError("M5C projected network hierarchy is required")
    target_to_segment = {
        str(target): str(segment.get("id"))
        for segment in network_hierarchy.get("segments", [])
        for target in segment.get("route_target_refs", [])
    }
    segment_to_runs: dict[str, list[str]] = {}
    run_segments = {}
    for run in network_hierarchy.get("runs", []):
        run_ref = str(run.get("id"))
        run_segments[run_ref] = set(map(str, run.get("segment_refs", [])))
        for segment_ref in run_segments[run_ref]:
            segment_to_runs.setdefault(segment_ref, []).append(run_ref)
    output = []
    for trace in network_hierarchy.get("connected_trace_completion", {}).get("complete_subtraces", []):
        if trace.get("state") != "derived" or trace.get("projected_scope_complete") is not True:
            continue
        segment_refs = {target_to_segment.get(str(ref)) for ref in trace.get("composite_refs", [])}
        if None in segment_refs or not segment_refs:
            continue
        candidate_runs = {
            run_ref for segment_ref in segment_refs
            for run_ref in segment_to_runs.get(str(segment_ref), [])
        }
        if len(candidate_runs) != 1 or not _finite_nonnegative(trace.get("projected_length_m")):
            continue
        run_ref = next(iter(candidate_runs))
        if not segment_refs.issubset(run_segments.get(run_ref, set())):
            continue
        trace_ref = str(trace.get("id"))
        evidence_refs = sorted({
            trace_ref,
            *map(str, trace.get("composite_refs", [])),
            *map(str, trace.get("covered_junction_refs", [])),
        })
        output.append({
            "id": stable_id("mep_projected_length_observation", trace_ref, run_ref),
            "network_run_ref": run_ref, "state": "observed",
            "value_m": float(trace["projected_length_m"]),
            "measurement_scope_ref": trace_ref,
            "measurement_scope_kind": "bounded_complete_projected_subtrace",
            "complete_run_coverage": (
                segment_refs == run_segments.get(run_ref, set())
                and trace.get("external_continuation_resolved") is True
            ),
            "evidence_refs": evidence_refs,
        })
    return sorted(output, key=lambda row: (row["network_run_ref"], row["id"]))


def _port_graph(evidence: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    reasons = []
    port_owner: dict[str, str] = {}
    port_size: dict[str, object] = {}
    components = []
    for segment in evidence.get("centreline_segments", []):
        ports = list(map(str, segment.get("port_refs", [])))
        if len(ports) != 2 or len(set(ports)) != 2:
            reasons.append("centreline_segment_requires_two_unique_ports")
        for ref in ports:
            if ref in port_owner:
                reasons.append("port_identity_reused_by_multiple_bodies")
            port_owner[ref] = str(segment.get("id"))
            port_size[ref] = evidence.get("cross_section", {}).get("nominal_size")
    for component in evidence.get("components", []):
        component_type = str(component.get("component_type"))
        ports = list(component.get("ports", []))
        expected = FIXED_PORT_COUNTS.get(component_type)
        if expected is not None and len(ports) != expected:
            reasons.append(f"{component_type}_port_count_mismatch")
        if component_type == "equipment" and not ports:
            reasons.append("equipment_requires_explicit_ports")
        if component_type not in {*FIXED_PORT_COUNTS, "equipment", "accessory"}:
            reasons.append("unsupported_component_type")
        for port in ports:
            ref = str(port.get("id"))
            if not ref or ref in port_owner:
                reasons.append("component_port_identity_not_unique")
            port_owner[ref] = str(component.get("id"))
            port_size[ref] = port.get("nominal_size")
        components.append(deepcopy(dict(component)))
    usage = Counter()
    standards = set()
    for connection in evidence.get("connections", []):
        refs = list(map(str, connection.get("port_refs", [])))
        if len(refs) != 2 or len(set(refs)) != 2:
            reasons.append("connection_requires_two_distinct_ports")
            continue
        if any(ref not in port_owner for ref in refs):
            reasons.append("connection_references_unknown_port")
        if len({port_owner.get(ref) for ref in refs}) != 2:
            reasons.append("connection_cannot_join_one_body_to_itself")
        if all(port_size.get(ref) is not None for ref in refs) and len(
            {canonical_sha256(port_size[ref]) for ref in refs}
        ) > 1:
            owners = {port_owner.get(ref) for ref in refs}
            reducers = {str(row.get("id")) for row in components
                        if row.get("component_type") in {"reducer", "transition"}}
            if not owners.intersection(reducers):
                reasons.append("size_change_requires_reducer_or_transition")
        usage.update(refs)
        standard = connection.get("connection_standard")
        if not isinstance(standard, str) or not standard:
            reasons.append("connection_standard_unresolved")
        else:
            standards.add(standard)
        if not connection.get("evidence_refs"):
            reasons.append("connection_evidence_missing")
    terminal_rows = list(evidence.get("terminals", []))
    for terminal in terminal_rows:
        ref = str(terminal.get("port_ref"))
        kind = terminal.get("terminal_kind")
        if ref not in port_owner:
            reasons.append("terminal_references_unknown_port")
        if kind not in TERMINAL_KINDS or kind == "unresolved":
            reasons.append("endpoint_classification_unresolved")
        if not terminal.get("evidence_refs"):
            reasons.append("terminal_evidence_missing")
        standard = terminal.get("connection_standard")
        if not isinstance(standard, str) or not standard:
            reasons.append("connection_standard_unresolved")
        else:
            standards.add(standard)
        usage[ref] += 1
    if set(usage) != set(port_owner) or any(count != 1 for count in usage.values()):
        reasons.append("every_port_must_be_resolved_exactly_once")
    return components, sorted(set(reasons)), sorted(standards)


def _takeout_total(
    components: Sequence[Mapping[str, Any]],
    segment_lengths: Mapping[str, float],
) -> tuple[float | None, list[str]]:
    """Replay non-overlapping fitting/equipment takeouts on 3D segments."""
    reasons = []
    intervals: dict[str, list[tuple[float, float, str]]] = {}
    declared_total = 0.0
    allocated_total = 0.0
    for component in components:
        component_ref = str(component.get("id"))
        declared = component.get("centreline_takeout_m", 0.0)
        if not _finite_nonnegative(declared):
            reasons.append("component_takeout_invalid")
            continue
        declared_total += float(declared)
        allocations = list(component.get("takeout_allocations", []))
        if float(declared) > 0 and not allocations:
            reasons.append("positive_component_takeout_requires_allocations")
        component_allocated = 0.0
        for allocation in allocations:
            segment_ref = str(allocation.get("centreline_segment_ref"))
            start = allocation.get("start_m")
            end = allocation.get("end_m")
            if (segment_ref not in segment_lengths or not _finite_nonnegative(start)
                    or not _finite_nonnegative(end) or float(end) <= float(start)
                    or float(end) > segment_lengths.get(segment_ref, -1) + 1e-9):
                reasons.append("component_takeout_allocation_invalid")
                continue
            interval = (float(start), float(end), component_ref)
            intervals.setdefault(segment_ref, []).append(interval)
            component_allocated += float(end) - float(start)
            allocated_total += float(end) - float(start)
        if abs(component_allocated - float(declared)) > 1e-8:
            reasons.append("component_takeout_allocation_does_not_replay")
    for rows in intervals.values():
        ordered = sorted(rows)
        if any(right[0] < left[1] - 1e-9 for left, right in zip(ordered, ordered[1:])):
            reasons.append("component_takeout_allocations_overlap")
    if abs(allocated_total - declared_total) > 1e-8:
        reasons.append("aggregate_component_takeout_does_not_replay")
    if reasons:
        return None, sorted(set(reasons))
    return declared_total, []


def _rule_items(
    evidence: Mapping[str, Any], rule_pack: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    output = []
    reasons = []
    if not rule_pack.get("id") or not rule_pack.get("version"):
        return output, ["assembly_rule_pack_identity_missing"]
    rules = list(rule_pack.get("assembly_rules", []))
    observed_standards = {
        str(row.get("connection_standard"))
        for row in (*evidence.get("connections", []), *evidence.get("terminals", []))
        if row.get("connection_standard")
    }
    shared_standard = next(iter(observed_standards)) if len(observed_standards) == 1 else None
    for component in evidence.get("components", []):
        context = {
            "component_type": component.get("component_type"),
            "connection_standard": component.get("connection_standard") or shared_standard,
            "nominal_size": evidence.get("cross_section", {}).get("nominal_size"),
            "material": evidence.get("material", {}).get("material"),
        }
        matches = [rule for rule in rules if all(
            context.get(key) == value for key, value in rule.get("match", {}).items()
        )]
        if len(matches) > 1:
            reasons.append("assembly_rule_match_not_unique")
            continue
        if not matches:
            reasons.append(f"assembly_rule_missing_for_{context['component_type']}")
            continue
        rule = matches[0]
        for derived in rule.get("derived_items", []):
            quantity = derived.get("quantity_per_component")
            if not _finite_nonnegative(quantity) or float(quantity) <= 0:
                reasons.append("assembly_rule_quantity_invalid")
                continue
            output.append({
                "engineering_item_kind": derived.get("engineering_item_kind"),
                "quantity": float(quantity),
                "unit": derived.get("unit"),
                "derivation": {
                    "state": "derived", "rule_pack_ref": rule_pack.get("id"),
                    "rule_pack_version": rule_pack.get("version"),
                    "rule_ref": rule.get("id"), "physical_component_ref": component.get("id"),
                },
            })
    return output, sorted(set(reasons))


def _catalog_mapping(
    item: Mapping[str, Any], catalog_pack: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    kind = str(item.get("engineering_item_kind"))
    requirements = dict(item.get("requirements", {}))
    if not catalog_pack.get("id") or not catalog_pack.get("version"):
        return {
            "stage": "generic_engineering_item", "product_family": None,
            "manufacturer": None, "model": None, "sku": None,
            "approved_substitute_refs": [],
        }, ["catalog_pack_identity_missing"]
    matches = [product for product in catalog_pack.get("products", [])
               if product.get("engineering_item_kind") == kind and all(
                   requirements.get(key) == value
                   for key, value in product.get("match", {}).items()
               )]
    if len(matches) != 1:
        return {
            "stage": "generic_engineering_item", "product_family": None,
            "manufacturer": None, "model": None, "sku": None,
            "approved_substitute_refs": [],
        }, ["catalog_mapping_missing" if not matches else "catalog_mapping_not_unique"]
    product = matches[0]
    critical = CRITICAL_CATALOG_FIELDS.get(
        "straight_material" if kind == "straight_material" else "component", ()
    )
    exact = bool(product.get("sku")) and all(
        requirements.get(key) is not None and product.get("match", {}).get(key) == requirements.get(key)
        for key in critical
    )
    stage = "exact_sku" if exact else "manufacturer_model" if product.get("model") else "product_family"
    reasons = [] if exact else ["exact_sku_requirements_unresolved"]
    return {
        "stage": stage, "product_family": product.get("product_family"),
        "manufacturer": product.get("manufacturer"), "model": product.get("model"),
        "sku": product.get("sku") if exact else None,
        "approved_substitute_refs": sorted(map(str, product.get("approved_substitute_refs", []))),
        "catalog_product_ref": product.get("id"),
    }, reasons


def _review_state(
    evidence_ref: str | None, network_hash: str, calculation_hash: str,
    decisions: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    rows = [row for row in decisions if str(row.get("promotion_evidence_ref")) == str(evidence_ref)]
    if not rows:
        return {"state": "unreviewed"}, {"state": "not_approved"}, []
    if len(rows) != 1:
        return {"state": "conflicted"}, {"state": "not_approved"}, ["review_overlay_not_unique"]
    row = rows[0]
    if (row.get("network_payload_sha256") != network_hash
            or row.get("calculation_payload_sha256") != calculation_hash
            or not row.get("evidence_refs")):
        return {"state": "invalid"}, {"state": "not_approved"}, ["review_overlay_not_bound_to_frozen_calculation"]
    review = {"state": row.get("decision"), "engineer": row.get("engineer"),
              "reviewed_at": row.get("reviewed_at"), "evidence_refs": deepcopy(row.get("evidence_refs"))}
    review["calculation_payload_sha256"] = calculation_hash
    if row.get("decision") != "approved":
        return review, {"state": "not_approved"}, []
    if not all(row.get(key) for key in ("engineer", "reviewed_at", "reason")):
        return review, {"state": "invalid"}, ["approval_metadata_incomplete"]
    return review, {"state": "approved", "engineer": row["engineer"],
                    "approved_at": row["reviewed_at"], "reason": row["reason"],
                    "evidence_refs": deepcopy(row["evidence_refs"]),
                    "calculation_payload_sha256": calculation_hash}, []


def build_mep_marketplace_assemblies(
    *, network_hierarchy: Mapping[str, Any],
    physical_run_evidence: Sequence[Mapping[str, Any]] = (),
    projected_length_observations: Sequence[Mapping[str, Any]] = (),
    assembly_rule_pack: Mapping[str, Any] | None = None,
    catalog_pack: Mapping[str, Any] | None = None,
    review_decisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build fail-closed marketplace assemblies from frozen M5C runs."""
    if network_hierarchy.get("layer") != "mep_projected_network_hierarchy":
        raise ValueError("M5C projected network hierarchy is required")
    network_hash = canonical_sha256(network_hierarchy)
    document = deepcopy(network_hierarchy.get("document", {}))
    rule_pack = deepcopy(dict(assembly_rule_pack or {}))
    catalog = deepcopy(dict(catalog_pack or {}))
    evidence_by_run: dict[str, list[Mapping[str, Any]]] = {}
    for row in physical_run_evidence:
        evidence_by_run.setdefault(str(row.get("network_run_ref")), []).append(row)
    assemblies = []
    segment_by_id = {
        str(row.get("id")): row for row in network_hierarchy.get("segments", [])
    }
    for run in sorted(network_hierarchy.get("runs", []), key=lambda row: str(row.get("id"))):
        run_ref = str(run.get("id"))
        identifier = stable_id("mep_marketplace_assembly", network_hash, run_ref)
        projected, projected_refs, projected_reasons, projected_scope = _projected_measurements(
            run_ref, projected_length_observations
        )
        rows = evidence_by_run.get(run_ref, [])
        evidence = rows[0] if len(rows) == 1 else None
        # A missing or disputed 2D scalar does not invalidate independently
        # closed 3D geometry; the channels remain separate all the way out.
        reasons: list[str] = []
        if len(rows) != 1:
            reasons.append("physical_run_evidence_missing" if not rows else "physical_run_evidence_not_unique")
            if not rows:
                reasons.extend([
                    "complete_topology_coverage_required",
                    "system_service_required",
                    "material_specification_required",
                    "nominal_physical_cross_section_required",
                    "endpoint_identity_required",
                    "resolved_three_dimensional_centreline_required",
                    "connection_standard_required",
                    "versioned_assembly_rule_pack_required",
                    "marketplace_catalog_mapping_required",
                ])
        installed = net = purchase = None
        components: list[dict[str, Any]] = []
        material_items: list[dict[str, Any]] = []
        standards: list[str] = []
        evidence_ref = str(evidence.get("id")) if evidence else None
        if evidence:
            if evidence.get("state") != "observed":
                reasons.append("physical_run_evidence_not_observed")
            if not evidence.get("evidence_refs"):
                reasons.append("physical_run_evidence_refs_missing")
            if evidence.get("network_payload_sha256") != network_hash:
                reasons.append("physical_run_evidence_not_bound_to_frozen_network")
            expected = list(map(str, run.get("segment_refs", [])))
            actual = list(map(str, evidence.get("accounted_network_segment_refs", [])))
            if Counter(actual) != Counter(expected) or len(actual) != len(set(actual)):
                reasons.append("network_segments_not_accounted_exactly_once")
            expected_occurrences = [
                str(ref) for segment_ref in expected
                for ref in segment_by_id.get(segment_ref, {}).get("source_occurrence_refs", [])
            ]
            actual_occurrences = list(map(str, evidence.get("accounted_projected_occurrence_refs", [])))
            if (Counter(actual_occurrences) != Counter(expected_occurrences)
                    or len(actual_occurrences) != len(set(actual_occurrences))
                    or len(expected_occurrences) != len(set(expected_occurrences))):
                reasons.append("projected_occurrences_not_accounted_exactly_once")
            centreline = list(evidence.get("centreline_segments", []))
            centreline_refs = list(map(str, (row.get("network_segment_ref") for row in centreline)))
            if Counter(centreline_refs) != Counter(expected) or len(centreline_refs) != len(set(centreline_refs)):
                reasons.append("three_dimensional_segment_coverage_not_exact")
            lengths = [_polyline_length(row.get("points_xyz_m", [])) for row in centreline]
            if any(value is None for value in lengths):
                reasons.append("invalid_three_dimensional_centreline")
            coverage = evidence.get("coverage", {})
            for key in (
                "all_projected_occurrences_accounted_once", "reprojection_passed",
                "duplicate_occurrences_eliminated", "vertical_spans_resolved",
            ):
                if coverage.get(key) is not True:
                    reasons.append(f"coverage_{key}_not_closed")
            if not evidence.get("system", {}).get("kind") or not evidence.get("system", {}).get("service"):
                reasons.append("system_or_service_unresolved")
            cross_section = evidence.get("cross_section", {})
            if not cross_section.get("nominal_size") or not cross_section.get("physical_dimensions"):
                reasons.append("nominal_or_physical_cross_section_unresolved")
            # Centreline length is a geometry channel.  Connector standards,
            # material and catalog rules may block net/purchase selection, but
            # they cannot erase a separately certified XYZ centreline.
            if not reasons:
                installed = sum(float(value) for value in lengths if value is not None)
            material = evidence.get("material", {})
            components, port_reasons, standards = _port_graph(evidence)
            reasons.extend(port_reasons)
            if installed is not None and not port_reasons:
                segment_lengths = {
                    str(row.get("id")): float(length)
                    for row, length in zip(centreline, lengths) if length is not None
                }
                takeout, takeout_reasons = _takeout_total(components, segment_lengths)
                reasons.extend(takeout_reasons)
                if takeout is not None and takeout > installed:
                    reasons.append("component_takeouts_exceed_installed_length")
                elif takeout is not None:
                    net = installed - takeout
            if not material.get("material") or not material.get("specification"):
                reasons.append("material_or_specification_unresolved")
            derived, rule_reasons = _rule_items(evidence, rule_pack)
            reasons.extend(rule_reasons)
            if net is not None and material.get("material") and material.get("specification"):
                stock_rules = [row for row in rule_pack.get("stock_material_rules", []) if all(
                    {
                        "material": material.get("material"),
                        "specification": material.get("specification"),
                        "nominal_size": cross_section.get("nominal_size"),
                    }.get(key) == value for key, value in row.get("match", {}).items()
                )]
                if len(stock_rules) != 1:
                    reasons.append("stock_material_rule_missing" if not stock_rules else "stock_material_rule_not_unique")
                else:
                    stock = stock_rules[0].get("stock_length_m")
                    waste = stock_rules[0].get("waste_fraction")
                    if not _finite_nonnegative(stock) or float(stock) <= 0 or not _finite_nonnegative(waste):
                        reasons.append("stock_material_rule_invalid")
                    else:
                        purchase = math.ceil(net * (1 + float(waste)) / float(stock)) * float(stock)
            straight = {
                "engineering_item_kind": "straight_material", "quantity": purchase,
                "unit": "m" if purchase is not None else None,
                "requirements": {
                    "material": material.get("material"), "specification": material.get("specification"),
                    "nominal_size": cross_section.get("nominal_size"),
                    "physical_cross_section": deepcopy(cross_section.get("physical_dimensions")),
                    "connection_standard": standards[0] if len(standards) == 1 else None,
                },
                "derivation": {"state": "calculated" if purchase is not None else "unknown",
                               "rule_pack_ref": rule_pack.get("id"),
                               "rule_pack_version": rule_pack.get("version")},
            }
            material_items = [straight]
            for component in components:
                material_items.append({
                    "engineering_item_kind": component.get("component_type"),
                    "quantity": 1.0, "unit": "ea",
                    "requirements": {
                        "component_type": component.get("component_type"),
                        "nominal_size": cross_section.get("nominal_size"),
                        "connection_standard": (
                            component.get("connection_standard")
                            or (standards[0] if len(standards) == 1 else None)
                        ),
                        "material": material.get("material"),
                        "specification": material.get("specification"),
                    },
                    "derivation": {
                        "state": "calculated", "physical_component_ref": component.get("id"),
                        "evidence_refs": deepcopy(component.get("evidence_refs", [])),
                    },
                })
            for item in derived:
                item["requirements"] = {
                    "component_type": item["engineering_item_kind"],
                    "nominal_size": cross_section.get("nominal_size"),
                    "connection_standard": standards[0] if len(standards) == 1 else None,
                    "material": material.get("material"),
                    "specification": material.get("specification"),
                }
                material_items.append(item)
            for item in material_items:
                mapping, mapping_reasons = _catalog_mapping(item, catalog)
                item["catalog_mapping"] = mapping
                reasons.extend(mapping_reasons)
        reasons.extend(projected_reasons)
        calculation_hash = canonical_sha256({
            "network_payload_sha256": network_hash, "run": run,
            "physical_run_evidence": evidence,
            "projected_length": {"value": projected, "evidence_refs": projected_refs},
            "assembly_rule_pack_sha256": canonical_sha256(rule_pack),
            "catalog_pack_sha256": canonical_sha256(catalog),
        })
        review, approval, review_reasons = _review_state(
            evidence_ref, network_hash, calculation_hash, review_decisions
        )
        reasons.extend(review_reasons)
        reasons = sorted(set(reasons))
        assembly = {
            "record_type": "mep_marketplace_assembly", "record_version": SCHEMA_VERSION,
            "id": identifier, "network_run_ref": run_ref,
            "promotion_evidence_ref": evidence_ref,
            "calculation_payload_sha256": calculation_hash,
            "system_and_service": deepcopy(evidence.get("system")) if evidence else None,
            "source_pages_details_sections": deepcopy(evidence.get("source")) if evidence else {
                "page_refs": sorted({
                    str(page_ref) for segment_ref in run.get("segment_refs", [])
                    for page_ref in segment_by_id.get(str(segment_ref), {}).get("page_refs", [])
                }),
                "detail_refs": [], "section_refs": [],
            },
            "material_and_specification": deepcopy(evidence.get("material")) if evidence else None,
            "nominal_size_and_physical_cross_section": deepcopy(evidence.get("cross_section")) if evidence else None,
            "start_and_end_terminals": deepcopy(evidence.get("terminals", [])) if evidence else [],
            "centreline_segments_3d": deepcopy(evidence.get("centreline_segments", [])) if evidence else [],
            "length_channels": {
                "projected_length": _value_channel(projected, scope=projected_scope),
                "resolved_centreline_length": _value_channel(installed),
                "net_material_length": _value_channel(net),
                "purchase_length": _value_channel(purchase),
            },
            "fittings_and_accessories": [row for row in components
                                         if row.get("component_type") not in {"equipment"}],
            "equipment_and_ports": [row for row in components
                                    if row.get("component_type") == "equipment"],
            "connection_standards": standards,
            "material_items": material_items,
            "evidence_refs": sorted({*projected_refs, *map(str, evidence.get("evidence_refs", []))}
                                    if evidence else set(projected_refs)),
            "uncertainty": {"state": "closed" if not reasons else "unresolved",
                            "requirements": reasons},
            "review": review, "approval": approval,
            "physical_run_established": installed is not None,
            "quantity_eligible": net is not None,
            "marketplace_ready": bool(material_items) and not reasons and all(
                row.get("catalog_mapping", {}).get("stage") == "exact_sku" for row in material_items
            ),
            "approved_for_quote": approval.get("state") == "approved" and not reasons,
        }
        assemblies.append(assembly)
    orphan_evidence = sorted(set(evidence_by_run) - {str(row.get("id")) for row in network_hierarchy.get("runs", [])})
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": LAYER, "document": document,
        "input_payload_sha256": {
            "m5c": network_hash, "physical_run_evidence": canonical_sha256(physical_run_evidence),
            "projected_length_observations": canonical_sha256(projected_length_observations),
            "assembly_rule_pack": canonical_sha256(rule_pack), "catalog_pack": canonical_sha256(catalog),
            "review_decisions": canonical_sha256(review_decisions),
        },
        "assembly_rule_pack": {"id": rule_pack.get("id"), "version": rule_pack.get("version")},
        "catalog_pack": {"id": catalog.get("id"), "version": catalog.get("version")},
        "assemblies": assemblies,
        "orphan_physical_run_evidence_refs": sorted(
            str(row.get("id")) for ref in orphan_evidence for row in evidence_by_run[ref]
        ),
        "contract": {
            "projected_resolved_net_purchase_lengths_separate": True,
            "drawing_and_rule_pack_derivations_separate": True,
            "generic_requirements_precede_sku_selection": True,
            "calculated_declared_reviewed_approved_separate": True,
            "m5c_not_mutated": True,
        },
        "summary": {
            "assembly_candidate_count": len(assemblies),
            "physical_run_count": sum(row["physical_run_established"] for row in assemblies),
            "quantity_eligible_count": sum(row["quantity_eligible"] for row in assemblies),
            "marketplace_ready_count": sum(row["marketplace_ready"] for row in assemblies),
            "approved_for_quote_count": sum(row["approved_for_quote"] for row in assemblies),
            "unresolved_requirement_count": sum(len(row["uncertainty"]["requirements"]) for row in assemblies),
        },
    }
    errors = validate_mep_marketplace_assemblies(payload)
    if errors:
        raise ValueError("invalid marketplace assembly output:\n" + "\n".join(errors))
    return payload


def validate_mep_marketplace_assemblies(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("layer") != LAYER:
        errors.append("marketplace assembly schema or layer mismatch")
    contract = payload.get("contract", {})
    for key in (
        "projected_resolved_net_purchase_lengths_separate",
        "drawing_and_rule_pack_derivations_separate",
        "generic_requirements_precede_sku_selection",
        "calculated_declared_reviewed_approved_separate", "m5c_not_mutated",
    ):
        if contract.get(key) is not True:
            errors.append(f"contract.{key} must be true")
    ids = []
    for row in payload.get("assemblies", []):
        identifier = str(row.get("id"))
        ids.append(identifier)
        calculation_hash = row.get("calculation_payload_sha256")
        if not isinstance(calculation_hash, str) or len(calculation_hash) != 64:
            errors.append(f"{identifier}: calculation payload digest is invalid")
        channels = row.get("length_channels", {})
        if set(channels) != {"projected_length", "resolved_centreline_length",
                            "net_material_length", "purchase_length"}:
            errors.append(f"{identifier}: four distinct length channels are required")
        for name, channel in channels.items():
            values = channel if isinstance(channel, Mapping) else {}
            calculated = values.get("calculated", {})
            value = calculated.get("value")
            if value is not None and not _finite_nonnegative(value):
                errors.append(f"{identifier}: {name} is invalid")
            for state in ("declared", "reviewed", "approved"):
                if values.get(state, {}).get("value") is not None:
                    errors.append(f"{identifier}: {state} length cannot be synthesized")
        installed = channels.get("resolved_centreline_length", {}).get("calculated", {}).get("value")
        net = channels.get("net_material_length", {}).get("calculated", {}).get("value")
        purchase = channels.get("purchase_length", {}).get("calculated", {}).get("value")
        if row.get("physical_run_established") is not (installed is not None):
            errors.append(f"{identifier}: physical-run authority mismatch")
        if row.get("quantity_eligible") is not (net is not None):
            errors.append(f"{identifier}: quantity authority mismatch")
        if installed is not None and net is not None and float(net) > float(installed) + 1e-9:
            errors.append(f"{identifier}: net length exceeds installed length")
        if purchase is not None and net is not None and float(purchase) + 1e-9 < float(net):
            errors.append(f"{identifier}: purchase length is below net length")
        if row.get("marketplace_ready") and (
            row.get("uncertainty", {}).get("requirements") or not row.get("material_items")
            or any(item.get("catalog_mapping", {}).get("stage") != "exact_sku"
                   for item in row.get("material_items", []))
        ):
            errors.append(f"{identifier}: marketplace readiness is unsupported")
        if row.get("approved_for_quote") and (
            row.get("approval", {}).get("state") != "approved" or not row.get("marketplace_ready")
            or row.get("approval", {}).get("calculation_payload_sha256") != calculation_hash
            or row.get("review", {}).get("calculation_payload_sha256") != calculation_hash
        ):
            errors.append(f"{identifier}: quote approval is unsupported")
    if len(ids) != len(set(ids)):
        errors.append("assembly IDs must be unique")
    summary = payload.get("summary", {})
    expected = {
        "assembly_candidate_count": len(payload.get("assemblies", [])),
        "physical_run_count": sum(bool(row.get("physical_run_established")) for row in payload.get("assemblies", [])),
        "quantity_eligible_count": sum(bool(row.get("quantity_eligible")) for row in payload.get("assemblies", [])),
        "marketplace_ready_count": sum(bool(row.get("marketplace_ready")) for row in payload.get("assemblies", [])),
        "approved_for_quote_count": sum(bool(row.get("approved_for_quote")) for row in payload.get("assemblies", [])),
        "unresolved_requirement_count": sum(len(row.get("uncertainty", {}).get("requirements", []))
                                            for row in payload.get("assemblies", [])),
    }
    if summary != expected:
        errors.append("summary does not replay")
    return errors
