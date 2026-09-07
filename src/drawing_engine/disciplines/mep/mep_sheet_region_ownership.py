"""Generic sheet-region ownership gate for MEP route observations.

Geometric route certificates are necessary but not sufficient: drawing
furniture can contain the same parallel-line motifs as outlined services.  This
layer classifies page scopes before takeoff promotion and preserves excluded
geometry as explicit non-route content.  It never deletes source observations
or grants physical, installed-length, or quantity authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "0.1.0"
NON_ROUTE_ROLES = {
    "title_block_revision_stamp", "border", "legend", "schedule_table",
    "notes_specifications",
}
VIEW_ROLES = {"main_plan_view", "section_detail_riser"}

_TEXT_PATTERNS = {
    "view_plan": re.compile(r"\b(?:plan|floor plan|piping plan|ductwork plan)\b", re.I),
    "view_detail": re.compile(r"\b(?:section|detail|riser|schematic|diagram)\b", re.I),
    "title_block": re.compile(
        r"\b(?:revision|drawing|sheet|project|checked|approved|drawn|scale|date)\b", re.I),
    "legend": re.compile(r"\b(?:legend|abbreviations?)\b", re.I),
    "schedule_table": re.compile(r"\b(?:schedule|table|quantity|qty\.?|item)\b", re.I),
    "notes_specifications": re.compile(r"\b(?:general notes?|notes?|specifications?)\b", re.I),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(kind: str, *parts: Any) -> str:
    return kind + "." + _sha256([kind, *parts])[:20]


def _bbox(points: Sequence[Sequence[float]]) -> list[float] | None:
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _contains(outer: Sequence[float], inner: Sequence[float], tolerance: float = 1.0) -> bool:
    return (outer[0] - tolerance <= inner[0] and outer[1] - tolerance <= inner[1]
            and outer[2] + tolerance >= inner[2] and outer[3] + tolerance >= inner[3])


def _intersects(left: Sequence[float], right: Sequence[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0]
                or left[3] < right[1] or right[3] < left[1])


def _union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _geometry_signature(points: Sequence[Sequence[float]], width: float, height: float) -> str:
    normalised = tuple((round(float(x) / width, 4), round(float(y) / height, 4))
                       for x, y in points)
    reverse = tuple(reversed(normalised))
    return _sha256(min(normalised, reverse))


def _edge_anchors(box: Sequence[float], width: float, height: float) -> list[str]:
    margin = .02
    output = []
    if box[0] / width <= margin:
        output.append("left")
    if (width - box[2]) / width <= margin:
        output.append("right")
    if box[1] / height <= margin:
        output.append("top")
    if (height - box[3]) / height <= margin:
        output.append("bottom")
    return output


def _axis_counts(candidates: Sequence[Mapping[str, Any]], region: Sequence[float]) -> dict[str, int]:
    horizontal = vertical = 0
    for candidate in candidates:
        points = candidate.get("points_display", [])
        for left, right in zip(points, points[1:]):
            segment = [min(left[0], right[0]), min(left[1], right[1]),
                       max(left[0], right[0]), max(left[1], right[1])]
            if not _intersects(segment, region):
                continue
            dx, dy = abs(right[0] - left[0]), abs(right[1] - left[1])
            if dx > 1 and dy <= .5:
                horizontal += 1
            elif dy > 1 and dx <= .5:
                vertical += 1
    return {"horizontal_rule_count": horizontal, "vertical_rule_count": vertical}


def _text_roles(blocks: Sequence[Mapping[str, Any]], region: Sequence[float] | None = None) -> Counter:
    roles = Counter()
    for block in blocks:
        box = block.get("bbox_display")
        if region is not None and box and not _intersects(box, region):
            continue
        text = str(block.get("text") or "")
        for role, pattern in _TEXT_PATTERNS.items():
            if pattern.search(text):
                roles[role] += 1
    return roles


def _page_view_role(page: Mapping[str, Any]) -> tuple[str, list[str]]:
    registry_role = str(page.get("registry_role") or "").lower()
    roles = _text_roles(page.get("text_blocks", []))
    evidence = []
    if "plan" in registry_role or roles["view_plan"]:
        evidence.extend(["m1_page_role" if "plan" in registry_role else "native_view_title"])
        return "main_plan_view", evidence
    if any(token in registry_role for token in ("detail", "section", "riser", "schematic")) \
            or roles["view_detail"]:
        evidence.extend(["m1_page_role" if registry_role else "native_view_title"])
        return "section_detail_riser", evidence
    return "unknown_region", ["no_unique_view_title_or_m1_view_role"]


def build_mep_sheet_region_ownership(
    *, document: Mapping[str, Any], pages: Sequence[Mapping[str, Any]],
    route_candidates: Sequence[Mapping[str, Any]], source_pdf_sha256: str,
) -> dict[str, Any]:
    """Classify route candidates into accepted view or drawing-furniture scopes."""
    page_by_ref = {page["page_ref"]: page for page in pages}
    if len(page_by_ref) != len(pages):
        raise ValueError("page refs must be unique")
    by_page: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    signature_pages: dict[str, set[str]] = defaultdict(set)
    prepared = []
    for candidate in route_candidates:
        page = page_by_ref.get(candidate.get("page_ref"))
        if page is None:
            raise ValueError("route candidate references an unregistered page")
        width, height = page["page_rect_display"][2], page["page_rect_display"][3]
        box = _bbox(candidate.get("points_display", []))
        signature = _geometry_signature(candidate.get("points_display", []), width, height)
        row = {**candidate, "bbox_display": box, "normalised_geometry_signature": signature}
        prepared.append(row)
        by_page[row["page_ref"]].append(row)
        signature_pages[signature].add(row["page_ref"])

    regions = []
    title_regions_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    text_grid_regions_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page_ref, candidates in by_page.items():
        page = page_by_ref[page_ref]
        width, height = page["page_rect_display"][2], page["page_rect_display"][3]
        seeds_by_edge: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            box = candidate["bbox_display"]
            if box is None or len(signature_pages[candidate["normalised_geometry_signature"]]) < 3:
                continue
            span = max((box[2] - box[0]) / width, (box[3] - box[1]) / height)
            if span >= .55:
                continue
            for edge in _edge_anchors(box, width, height):
                seeds_by_edge[edge].append(candidate)
        for edge, seeds in sorted(seeds_by_edge.items()):
            if len(seeds) < 4:
                continue
            region_box = _union([row["bbox_display"] for row in seeds])
            axis = _axis_counts(candidates, region_box)
            roles = _text_roles(page.get("text_blocks", []), region_box)
            cross_sheet_count = max(len(signature_pages[row["normalised_geometry_signature"]]) for row in seeds)
            ruled = max(axis.values()) >= 4
            title_text = roles["title_block"] > 0
            if not (ruled and (title_text or len(seeds) >= 6)):
                continue
            record = {
                "id": _stable_id("mep_sheet_region", page_ref, edge, region_box),
                "record_type": "mep_sheet_region_certificate", "state": "accepted",
                "page_ref": page_ref, "region_role": "title_block_revision_stamp",
                "bbox_display": [round(value, 6) for value in region_box],
                "membership_geometry": "closed_bbox_from_repeated_edge_anchored_rules",
                "evidence": {
                    "page_edge_anchor": edge,
                    "repeated_cross_sheet_geometry_page_count": cross_sheet_count,
                    "repeated_rule_candidate_refs": sorted(row["occurrence_ref"] for row in seeds),
                    "ruled_grid_topology": axis,
                    "text_role_counts": dict(sorted(roles.items())),
                    "view_title_exclusion": True,
                },
                "quantity_eligible": False,
            }
            regions.append(record)
            title_regions_by_page[page_ref].append(record)

        for block in page.get("text_blocks", []):
            block_box = block.get("bbox_display")
            if not block_box:
                continue
            block_roles = _text_roles([block])
            role = next((name for name in ("legend", "schedule_table", "notes_specifications")
                         if block_roles[name]), None)
            if role is None:
                continue
            seeds = [row for row in candidates if row["bbox_display"]
                     and _intersects(row["bbox_display"], block_box)]
            if not seeds:
                continue
            connected = list(seeds)
            seed_boxes = [row["bbox_display"] for row in seeds]
            for row in candidates:
                if row in connected or row["bbox_display"] is None:
                    continue
                if any(_intersects(row["bbox_display"], seed_box) for seed_box in seed_boxes):
                    connected.append(row)
            region_box = _union([block_box, *[row["bbox_display"] for row in connected]])
            axis = _axis_counts(candidates, region_box)
            if min(axis.values()) < 1 or max(axis.values()) < 4:
                continue
            record = {
                "id": _stable_id("mep_sheet_region", page_ref, role, region_box),
                "record_type": "mep_sheet_region_certificate", "state": "accepted",
                "page_ref": page_ref, "region_role": role,
                "bbox_display": [round(value, 6) for value in region_box],
                "membership_geometry": "ruled_component_seeded_by_native_text_role",
                "evidence": {"native_text_block_ref": block.get("id"),
                             "ruled_grid_topology": axis,
                             "text_role_counts": {role: 1},
                             "page_edge_anchor": None,
                             "repeated_cross_sheet_geometry_page_count": 0,
                             "view_title_exclusion": True},
                "quantity_eligible": False,
            }
            regions.append(record)
            text_grid_regions_by_page[page_ref].append(record)

    ownership = []
    non_route = []
    role_counts = Counter()
    excluded_lengths = Counter()
    for candidate in prepared:
        page = page_by_ref[candidate["page_ref"]]
        width, height = page["page_rect_display"][2], page["page_rect_display"][3]
        box = candidate["bbox_display"]
        view_role, view_evidence = _page_view_role(page)
        role, state, reasons, region_ref = view_role, "accepted" if view_role in VIEW_ROLES else "unknown", [], None
        if box is None:
            role, state, reasons = "unknown_region", "unknown", ["candidate_has_no_display_geometry"]
        else:
            span_x, span_y = (box[2] - box[0]) / width, (box[3] - box[1]) / height
            anchors = _edge_anchors(box, width, height)
            if anchors and max(span_x, span_y) >= .70:
                role, state, reasons = "border", "excluded", ["page_edge_anchored_long_rule"]
            else:
                for region in title_regions_by_page[candidate["page_ref"]]:
                    if _contains(region["bbox_display"], box, tolerance=2.0):
                        role, state = "title_block_revision_stamp", "excluded"
                        reasons = ["inside_accepted_title_block_region"]
                        region_ref = region["id"]
                        break
                if state != "excluded":
                    for region in text_grid_regions_by_page[candidate["page_ref"]]:
                        if _contains(region["bbox_display"], box, tolerance=2.0):
                            role, state = region["region_role"], "excluded"
                            reasons = ["inside_accepted_%s_region" % role]
                            region_ref = region["id"]
                            break
            if state not in {"excluded"}:
                local_roles = _text_roles(page.get("text_blocks", []), box)
                axis = _axis_counts(by_page[candidate["page_ref"]], box)
                ruled = max(axis.values()) >= 4
                for text_role, output_role in (("legend", "legend"),
                                                ("schedule_table", "schedule_table"),
                                                ("notes_specifications", "notes_specifications")):
                    if local_roles[text_role] and ruled:
                        role, state = output_role, "excluded"
                        reasons = ["ruled_region_with_native_text_role"]
                        break
        membership_id = _stable_id("mep_route_region_membership", candidate["occurrence_ref"], role)
        ownership.append({
            "id": membership_id, "record_type": "mep_route_region_membership",
            "state": state, "page_ref": candidate["page_ref"],
            "route_ledger_ref": candidate["route_ledger_ref"],
            "occurrence_ref": candidate["occurrence_ref"], "region_role": role,
            "region_ref": region_ref, "bbox_display": box,
            "view_scope_evidence": view_evidence, "reasons": reasons,
            "route_certification_eligible": state == "accepted" and role in VIEW_ROLES,
            "quantity_eligible": False,
        })
        role_counts[role] += 1
        if state == "excluded":
            length = candidate.get("projected_2d_length_m")
            if length is not None:
                excluded_lengths[role] += float(length)
            non_route.append({
                "id": _stable_id("non_route_drawing_content", candidate["occurrence_ref"]),
                "record_type": "non_route_drawing_content", "state": "observed",
                "page_ref": candidate["page_ref"], "region_role": role,
                "region_membership_ref": membership_id, "source_route_ledger_ref": candidate["route_ledger_ref"],
                "source_occurrence_ref": candidate["occurrence_ref"],
                "source_primitive_refs": sorted(candidate.get("source_primitive_refs", [])),
                "points_display": candidate.get("points_display", []),
                "projected_length_m_excluded_from_route": length,
                "preservation_reason": "drawing_furniture_retained_without_route_or_quantity_authority",
                "route_length_counted": False, "quantity_eligible": False,
            })

    title_rows = [row for row in non_route if row["region_role"] == "title_block_revision_stamp"]
    payload = {
        "schema_version": SCHEMA_VERSION, "layer": "mep_sheet_region_ownership",
        "document": dict(document), "source_pdf_sha256": source_pdf_sha256,
        "authority": {
            "sheet_region_ownership_required_before_route_certification": True,
            "excluded_content_preserved": True,
            "route_identity_established": False, "physical_run_identity_established": False,
            "installed_length_established": False, "quantity_eligible": False,
        },
        "pages": [{
            "page_ref": page["page_ref"], "page_number": page["page_number"],
            "registry_role": page.get("registry_role"),
            "accepted_view_role": _page_view_role(page)[0],
            "view_scope_evidence": _page_view_role(page)[1],
            "page_rect_display": page["page_rect_display"],
            "text_evidence_sha256": _sha256(page.get("text_blocks", [])),
        } for page in pages],
        "regions": sorted(regions, key=lambda row: row["id"]),
        "candidate_ownership": sorted(ownership, key=lambda row: row["id"]),
        "non_route_drawing_content": sorted(non_route, key=lambda row: row["id"]),
        "summary": {
            "registered_page_count": len(pages), "candidate_count": len(route_candidates),
            "accepted_view_candidate_count": sum(row["route_certification_eligible"] for row in ownership),
            "excluded_non_route_candidate_count": len(non_route),
            "unknown_region_candidate_count": sum(row["state"] == "unknown" for row in ownership),
            "region_role_counts": dict(sorted(role_counts.items())),
            "excluded_projected_length_m_by_role": {
                role: round(value, 8) for role, value in sorted(excluded_lengths.items())},
        },
    }
    payload["acceptance_gate"] = {
        "every_candidate_assigned_once": len(ownership) == len(route_candidates)
            and len({row["occurrence_ref"] for row in ownership}) == len(route_candidates),
        "excluded_content_preserved_once": len(non_route) == sum(row["state"] == "excluded" for row in ownership),
        "title_block_geometry_counted_as_route_length": False,
        "title_block_measurable_excluded_length_m": round(sum(
            row.get("projected_length_m_excluded_from_route") or 0 for row in title_rows), 8),
        "status": "accepted_sheet_region_ownership_certificate",
    }
    payload["reproducibility"] = {
        "page_evidence_sha256": _sha256(pages),
        "candidate_ownership_sha256": _sha256(payload["candidate_ownership"]),
        "non_route_content_sha256": _sha256(payload["non_route_drawing_content"]),
    }
    return payload


def validate_mep_sheet_region_ownership(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("layer") != "mep_sheet_region_ownership":
        errors.append("unexpected layer")
    gate = payload.get("acceptance_gate", {})
    if gate.get("every_candidate_assigned_once") is not True:
        errors.append("every candidate must be assigned once")
    if gate.get("excluded_content_preserved_once") is not True:
        errors.append("excluded content must be preserved once")
    if gate.get("title_block_geometry_counted_as_route_length") is not False:
        errors.append("title-block geometry cannot count as route length")
    memberships = {row.get("id"): row for row in payload.get("candidate_ownership", [])}
    for row in payload.get("non_route_drawing_content", []):
        membership = memberships.get(row.get("region_membership_ref"))
        if (membership is None or membership.get("state") != "excluded"
                or row.get("route_length_counted") is not False):
            errors.append("non-route content lacks excluded membership: " + str(row.get("id")))
    for row in memberships.values():
        eligible = row.get("route_certification_eligible") is True
        if eligible != (row.get("state") == "accepted" and row.get("region_role") in VIEW_ROLES):
            errors.append("route eligibility differs from region membership: " + str(row.get("id")))
    return errors
