"""Precision-derived, fail-closed near-join certificates for MEP M3 pages."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from statistics import median
from typing import Any, Mapping


SCHEMA_VERSION = "0.1.0"
EXACT_TOLERANCE = .05
ABSOLUTE_MAXIMUM = .25


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(kind: str, *parts: Any) -> str:
    return f"{kind}.{hashlib.sha256(_canonical(parts).encode()).hexdigest()[:20]}"


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _style_key(fragment: Mapping[str, Any]) -> str:
    style = fragment["style"]
    return _canonical({
        "stroke": style.get("stroke"),
        "width_display_points": style.get("width_display_points"),
        "dash_pattern": style.get("dash_pattern"),
    })


def _unit(vector):
    length = math.hypot(*vector)
    return None if length <= 1e-9 else [vector[0] / length, vector[1] / length]


def _endpoint_rows(page: Mapping[str, Any]):
    output = []
    for fragment in page.get("fragments", []):
        points = fragment["geometry"]["points_display"]
        if len(points) < 2:
            continue
        for role, point, neighbor in (
            ("start", points[0], points[1]),
            ("end", points[-1], points[-2]),
        ):
            inward = _unit([neighbor[0] - point[0], neighbor[1] - point[1]])
            if inward is None:
                continue
            output.append({
                "id": f"{fragment['id']}:{role}",
                "fragment_ref": fragment["id"], "role": role,
                "point_display": point, "inward_unit": inward,
                "vertex_ref": fragment["endpoint_vertex_refs"][0 if role == "start" else 1],
                "style_key": _style_key(fragment),
                "style": fragment["style"],
            })
    return output


def _precision(endpoints, width):
    coordinates = [set(), set()]
    for endpoint in endpoints:
        for axis, value in enumerate(endpoint["point_display"]):
            coordinates[axis].add(round(float(value), 3))
    maximum = min(ABSOLUTE_MAXIMUM, max(EXACT_TOLERANCE, width / 2))
    differences = [
        right - left
        for values in coordinates
        for left, right in zip(sorted(values), sorted(values)[1:])
        if EXACT_TOLERANCE < right - left <= maximum
    ]
    bins = defaultdict(list)
    for value in differences:
        bins[round(value / .01)].append(value)
    if not bins:
        return {"state": "unknown", "native_coordinate_quantum_display_points": None,
                "sample_count": 0, "quantized_difference_count": 0}
    values = max(bins.values(), key=lambda rows: (len(rows), -median(rows)))
    quantum = median(values)
    agreeing = sum(abs(value - round(value / quantum) * quantum) <= .02
                    for value in differences)
    state = "observed_grid" if len(values) >= 4 and agreeing / len(differences) >= .65 else "unknown"
    return {
        "state": state,
        "native_coordinate_quantum_display_points": round(quantum, 6) if state == "observed_grid" else None,
        "sample_count": len(differences),
        "modal_sample_count": len(values),
        "quantized_difference_count": agreeing,
    }


def _candidate_pairs(endpoints, maximum):
    cells = defaultdict(list)
    for index, endpoint in enumerate(endpoints):
        x, y = endpoint["point_display"]
        cell = (math.floor(x / maximum), math.floor(y / maximum))
        cells[cell].append(index)
    output = []
    for left_index, left in enumerate(endpoints):
        x, y = left["point_display"]
        cell = (math.floor(x / maximum), math.floor(y / maximum))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for right_index in cells.get((cell[0] + dx, cell[1] + dy), []):
                    if right_index <= left_index:
                        continue
                    right = endpoints[right_index]
                    if left["fragment_ref"] == right["fragment_ref"]:
                        continue
                    distance = math.dist(left["point_display"], right["point_display"])
                    if not EXACT_TOLERANCE < distance <= maximum:
                        continue
                    dot = sum(a * b for a, b in zip(left["inward_unit"], right["inward_unit"]))
                    if dot > -math.cos(math.radians(3)):
                        continue
                    output.append({
                        "left": left, "right": right, "residual": distance,
                        "opposed_tangent_dot": dot,
                    })
    return output


def build_precision_near_join_certificates(*, route_page: Mapping[str, Any]) -> dict[str, Any]:
    """Publish exact, unique-near, ambiguous-near, and crossing classes."""
    page_ref = route_page["page_ref"]
    endpoints = _endpoint_rows(route_page)
    by_style = defaultdict(list)
    for endpoint in endpoints:
        by_style[endpoint["style_key"]].append(endpoint)
    style_certificates = []
    accepted, ambiguous = [], []
    for style_key, rows in sorted(by_style.items()):
        width = rows[0]["style"].get("width_display_points")
        width = float(width) if width is not None and float(width) > 0 else EXACT_TOLERANCE * 2
        hard_maximum = min(ABSOLUTE_MAXIMUM, max(EXACT_TOLERANCE, width / 2))
        precision = _precision(rows, width)
        raw_pairs = _candidate_pairs(rows, hard_maximum)
        residuals = sorted(pair["residual"] for pair in raw_pairs)
        tolerance = EXACT_TOLERANCE
        if precision["state"] == "observed_grid" and residuals:
            percentile_index = min(len(residuals) - 1, math.floor(.9 * (len(residuals) - 1)))
            tolerance = min(
                hard_maximum,
                max(EXACT_TOLERANCE,
                    precision["native_coordinate_quantum_display_points"],
                    residuals[percentile_index]))
        eligible = [pair for pair in raw_pairs if pair["residual"] <= tolerance]
        competitors = defaultdict(list)
        for pair in eligible:
            competitors[pair["left"]["id"]].append(pair)
            competitors[pair["right"]["id"]].append(pair)
        seen = set()
        for pair in eligible:
            pair_key = tuple(sorted((pair["left"]["id"], pair["right"]["id"])))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            refs = sorted((pair["left"]["fragment_ref"], pair["right"]["fragment_ref"]))
            record = {
                "id": _stable_id("mep_precision_near_join", page_ref, pair_key),
                "record_type": "mep_precision_near_join_certificate",
                "page_ref": page_ref, "fragment_refs": refs,
                "endpoint_refs": list(pair_key),
                "points_display": [pair["left"]["point_display"], pair["right"]["point_display"]],
                "residual_display_points": round(pair["residual"], 6),
                "opposed_tangent_dot": round(pair["opposed_tangent_dot"], 8),
                "style_key": style_key,
                "tolerance_display_points": round(tolerance, 6),
                "absolute_maximum_display_points": ABSOLUTE_MAXIMUM,
                "crossing_join": False,
                "physical_continuation_established": False,
                "quantity_eligible": False,
            }
            unique = (len(competitors[pair["left"]["id"]]) == 1
                      and len(competitors[pair["right"]["id"]]) == 1)
            record["state"] = "accepted" if unique else "ambiguous"
            record["mutual_unique"] = unique
            record["competitor_endpoint_refs"] = sorted({
                candidate[side]["id"]
                for endpoint in (pair["left"], pair["right"])
                for candidate in competitors[endpoint["id"]]
                for side in ("left", "right")
                if candidate[side]["id"] not in pair_key})
            (accepted if unique else ambiguous).append(record)
        style_certificates.append({
            "id": _stable_id("mep_view_style_precision", page_ref, style_key),
            "page_ref": page_ref, "style_key": style_key,
            "style": rows[0]["style"], "endpoint_count": len(rows),
            "line_width_display_points": width,
            "native_coordinate_precision": precision,
            "observed_seam_residual_count": len(residuals),
            "observed_seam_residual_min_display_points": (
                round(residuals[0], 6) if residuals else None),
            "observed_seam_residual_max_display_points": (
                round(residuals[-1], 6) if residuals else None),
            "derived_near_join_tolerance_display_points": round(tolerance, 6),
            "strict_style_maximum_display_points": round(hard_maximum, 6),
            "global_tolerance_increase_used": False,
            "state": "derived" if precision["state"] == "observed_grid" else "unknown",
        })
    exact = []
    for vertex in route_page.get("vertices", []):
        refs = sorted(set(vertex.get("fragment_refs", [])))
        if len(refs) < 2:
            continue
        exact.append({
            "id": _stable_id("mep_exact_join", page_ref, vertex["id"], refs),
            "record_type": "mep_exact_join_certificate", "state": "accepted",
            "page_ref": page_ref, "vertex_ref": vertex["id"],
            "point_display": vertex["point_display"], "fragment_refs": refs,
            "basis": "shared_M3_endpoint_vertex_within_exact_tolerance",
            "quantity_eligible": False,
        })
    crossings = route_page.get("crossings", [])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layer": "mep_precision_near_join_certificates",
        "page_ref": page_ref,
        "exact_joins": exact,
        "uniquely_certified_near_joins": sorted(accepted, key=lambda row: row["id"]),
        "ambiguous_near_join_candidates": sorted(ambiguous, key=lambda row: row["id"]),
        "disconnected_crossings": {
            "count": len(crossings),
            "source_crossing_refs_sha256": _sha256(sorted(row["id"] for row in crossings)),
            "join_authority": False,
        },
        "view_style_precision_certificates": style_certificates,
        "summary": {
            "exact_join_count": len(exact),
            "uniquely_certified_near_join_count": len(accepted),
            "ambiguous_near_join_candidate_count": len(ambiguous),
            "disconnected_crossing_count": len(crossings),
            "maximum_derived_tolerance_display_points": max(
                (row["derived_near_join_tolerance_display_points"]
                 for row in style_certificates), default=EXACT_TOLERANCE),
        },
        "authority": {
            "global_snap_tolerance_increased": False,
            "crossings_establish_joins": False,
            "near_join_changes_native_geometry": False,
            "physical_continuation_established": False,
            "quantity_eligible": False,
        },
    }
    return payload


def validate_precision_near_joins(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("layer") != "mep_precision_near_join_certificates":
        errors.append("unexpected layer")
    if payload.get("authority", {}).get("global_snap_tolerance_increased") is not False:
        errors.append("global snap tolerance cannot increase")
    for row in payload.get("view_style_precision_certificates", []):
        tolerance = row.get("derived_near_join_tolerance_display_points")
        maximum = row.get("strict_style_maximum_display_points")
        if tolerance is None or maximum is None or tolerance > maximum or maximum > ABSOLUTE_MAXIMUM:
            errors.append("style tolerance exceeds strict bound")
    for row in payload.get("uniquely_certified_near_joins", []):
        if row.get("state") != "accepted" or row.get("mutual_unique") is not True:
            errors.append("accepted near join is not mutually unique")
        if row.get("residual_display_points", math.inf) > row.get("tolerance_display_points", -math.inf):
            errors.append("accepted near join exceeds derived tolerance")
        if row.get("crossing_join") is not False:
            errors.append("crossing promoted as near join")
    return errors
