"""Evidence-first classification for M3 rows deferred by outline recovery.

The adapter groups native segments by their native drawing path and classifies
every deferred denominator row.  It never consumes unclassified recovered
components.  A closed circular symbol may become a supported interface
candidate only when its centre has unique strict incidence to an already
identified or independently supported route component.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
LAYER = "mep_deferred_drawing_geometry_classification"
MEP_STATES = {"identified_mep_route", "supported_unidentified_mep_candidate"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(kind: str, *parts: Any) -> str:
    return f"{kind}.{hashlib.sha256(_canonical(parts).encode()).hexdigest()[:20]}"


def _drawing_ordinal(source_ref: str) -> int:
    return int(source_ref.removeprefix("drawing[").split("]", 1)[0])


def _point_segment_distance(point: Sequence[float], start: Sequence[float],
                            end: Sequence[float]) -> float:
    dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return math.dist(point[:2], start[:2])
    ratio = ((float(point[0]) - float(start[0])) * dx
             + (float(point[1]) - float(start[1])) * dy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    projected = (float(start[0]) + ratio * dx, float(start[1]) + ratio * dy)
    return math.dist(point[:2], projected)


def _path_distance(point: Sequence[float], path: Sequence[Sequence[float]]) -> float:
    return min((_point_segment_distance(point, a, b) for a, b in zip(path, path[1:])),
               default=math.inf)


def _circle_metrics(fragments: Sequence[Mapping[str, Any]], tolerance: float) -> dict[str, Any] | None:
    if len(fragments) != 4 or any(row["geometry"].get("kind") != "cubic" for row in fragments):
        return None
    points = [point for row in fragments for point in row["geometry"]["points_display"]]
    if not points:
        return None
    bbox = [min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points)]
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if min(width, height) <= tolerance * 4.0 or abs(width - height) > max(tolerance * 4.0, width * .08):
        return None
    centre = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    radius = (width + height) / 4.0
    residual = max(abs(math.dist(point[:2], centre) - radius) for point in points)
    if residual > max(tolerance * 4.0, radius * .08):
        return None
    endpoint_degrees = Counter()
    for row in fragments:
        endpoints = [row["geometry"]["points_display"][0],
                     row["geometry"]["points_display"][-1]]
        for point in endpoints:
            key = (round(point[0] / tolerance), round(point[1] / tolerance))
            endpoint_degrees[key] += 1
    if len(endpoint_degrees) != 4 or any(value != 2 for value in endpoint_degrees.values()):
        return None
    return {
        "shape": "closed_circular_native_symbol",
        "centre_display": [round(value, 6) for value in centre],
        "radius_display_points": round(radius, 6),
        "maximum_radial_residual_display_points": round(residual, 6),
        "bbox_display": [round(value, 6) for value in bbox],
    }


def classify_deferred_geometry(
    *, recovery: Mapping[str, Any], route_page: Mapping[str, Any],
    consolidated_classification: Mapping[str, Any],
    component_paths: Mapping[str, Sequence[Sequence[Sequence[float]]]],
    negative_certificates: Mapping[str, Any],
) -> dict[str, Any]:
    page_ref = recovery["page_ref"]
    tolerance = max(float(route_page["tolerances"]["vertex_display_points"]), 1e-6)
    fragments = {row["id"]: row for row in route_page["fragments"]}
    deferred_rows = [row for row in recovery["unrecovered_route_evidence"]
                     if row["reason_category"] == "no_outlined_route_candidate_membership"]
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in deferred_rows:
        grouped[_drawing_ordinal(row["source_primitive_ref"])].append(row)

    allowed_components = {
        row["component_ref"]: row for row in consolidated_classification["classifications"]
        if row["state"] in MEP_STATES
    }
    negative_by_entity = {row["entity_ref"]: row
                          for row in negative_certificates["evaluations"]}
    entities = []
    segment_records = []
    for drawing_ordinal, rows in sorted(grouped.items()):
        entity_ref = f"mep_deferred_native_drawing_group.{drawing_ordinal}"
        member_fragments = [fragments[row["fragment_ref"]] for row in rows]
        source_refs = sorted(row["source_primitive_ref"] for row in rows)
        circle = _circle_metrics(member_fragments, tolerance)
        negative = negative_by_entity.get(entity_ref, {})
        hatch_state = negative.get("hatch_certificate", {}).get("state")
        architectural_state = negative.get("architectural_boundary_certificate", {}).get("state")
        independent_support = []
        alternatives = []
        state = "unclassified_view_geometry"
        geometry_role = "unclassified_native_drawing_geometry"
        reason = "no_independent_MEP_evidence"

        if hatch_state == "accepted" or architectural_state == "accepted":
            state = "non_route_drawing_content"
            geometry_role = "measured_negative_drawing_content"
            reason = "accepted_measured_hatch_or_architectural_boundary_certificate"
        elif circle:
            centre = circle["centre_display"]
            incidence_limit = max(tolerance * 4.0,
                                  float(circle["radius_display_points"]) * .45)
            distances = []
            for component_ref, paths in component_paths.items():
                if component_ref not in allowed_components:
                    continue
                distance = min((_path_distance(centre, path) for path in paths),
                               default=math.inf)
                if distance <= incidence_limit:
                    distances.append((distance, component_ref))
            incident_components = sorted({component_ref for _, component_ref in distances})
            alternatives = [{"component_ref": component_ref,
                             "centreline_distance_display_points": round(distance, 6)}
                            for distance, component_ref in sorted(distances)]
            if len(incident_components) == 1:
                component_ref = incident_components[0]
                state = "supported_unidentified_mep_candidate"
                geometry_role = "fitting_or_equipment_port_symbol_candidate"
                reason = "closed_circle_with_unique_strict_incidence_to_supported_MEP_component"
                independent_support.append({
                    "kind": "unique_route_incidence",
                    "component_ref": component_ref,
                    "component_state": allowed_components[component_ref]["state"],
                    "incidence_limit_display_points": round(incidence_limit, 6),
                    "connection_identity_established": False,
                })
            elif len(incident_components) > 1:
                reason = "closed_circle_has_ambiguous_supported_route_incidence"
                geometry_role = "unresolved_circular_symbol"
            else:
                reason = "closed_circle_has_no_supported_route_incidence"
                geometry_role = "unresolved_circular_symbol"

        entity_record = {
            "id": _stable_id("mep_deferred_geometry_entity", entity_ref, state,
                             geometry_role, independent_support),
            "record_type": "mep_deferred_geometry_entity_classification",
            "state": state,
            "page_ref": page_ref,
            "entity_ref": entity_ref,
            "native_drawing_ordinal": drawing_ordinal,
            "fragment_refs": sorted(row["fragment_ref"] for row in rows),
            "source_primitive_refs": source_refs,
            "geometry_role": geometry_role,
            "geometry_shape": circle,
            "classification_reason": reason,
            "independent_mep_support": independent_support,
            "competing_route_incidences": alternatives,
            "unclassified_recovered_components_consumed": False,
            "identified_system": None,
            "route_length_eligible": False,
            "physical_continuity_established": False,
            "quantity_eligible": False,
        }
        entities.append(entity_record)
        for row in rows:
            segment_records.append({
                "id": _stable_id("mep_deferred_segment_classification",
                                 row["source_primitive_ref"], entity_record["id"]),
                "record_type": "mep_deferred_source_segment_classification",
                "state": state,
                "page_ref": page_ref,
                "fragment_ref": row["fragment_ref"],
                "source_primitive_ref": row["source_primitive_ref"],
                "entity_classification_ref": entity_record["id"],
                "geometry_role": geometry_role,
                "identified_system": None,
                "route_length_eligible": False,
                "quantity_eligible": False,
            })

    entity_counts = Counter(row["state"] for row in entities)
    segment_counts = Counter(row["state"] for row in segment_records)
    all_refs = {row["source_primitive_ref"] for row in deferred_rows}
    classified_refs = {row["source_primitive_ref"] for row in segment_records}
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": LAYER,
        "development_status": "diagnostic_deferred_geometry_reclassification",
        "page_ref": page_ref,
        "entity_classifications": entities,
        "source_segment_classifications": segment_records,
        "summary": {
            "deferred_native_drawing_group_count": len(entities),
            "deferred_source_segment_count": len(segment_records),
            "entity_state_counts": dict(sorted(entity_counts.items())),
            "source_segment_state_counts": dict(sorted(segment_counts.items())),
            "closed_circular_symbol_count": sum(row["geometry_shape"] is not None for row in entities),
            "uniquely_incident_interface_candidate_count": sum(
                row["geometry_role"] == "fitting_or_equipment_port_symbol_candidate"
                for row in entities),
            "unresolved_circular_symbol_count": sum(
                row["geometry_role"] == "unresolved_circular_symbol" for row in entities),
        },
        "acceptance_gate": {
            "every_deferred_source_segment_classified_once": (
                len(segment_records) == len(all_refs) == len(classified_refs)),
            "unaccounted_source_segment_refs": sorted(all_refs - classified_refs),
            "duplicate_source_segment_count": len(segment_records) - len(classified_refs),
            "unclassified_recovered_components_consumed": False,
            "circle_alone_establishes_connection_identity": False,
            "status": "accepted_diagnostic_deferred_reclassification",
        },
        "authority": {
            "route_length_added": False,
            "system_identity_created": False,
            "physical_connection_created": False,
            "installed_length_established": False,
            "purchase_length_established": False,
        },
    }


def validate_deferred_geometry_classification(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    rows = payload.get("source_segment_classifications", [])
    if payload.get("layer") != LAYER:
        errors.append("unexpected deferred classification layer")
    if len(rows) != len({row.get("source_primitive_ref") for row in rows}):
        errors.append("deferred source segments are missing or duplicated")
    if not payload.get("acceptance_gate", {}).get("every_deferred_source_segment_classified_once"):
        errors.append("not every deferred source segment is classified")
    if any(row.get("route_length_eligible") for row in rows):
        errors.append("deferred classifications cannot add route length")
    return errors
