"""Title-, scale-, whitespace-, and geometry-backed drawing view scopes."""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import median
from typing import Any, Iterable, Mapping

import fitz


_TITLE_ROLES = {
    "section_label",
    "reinforcement_view_candidate",
    "formwork_view_candidate",
    "section_view_candidate",
    "plan_view_candidate",
    "detail_view_candidate",
}
_SECTION_LABEL = re.compile(
    r"(?<!\w)([1-9]\d*|[A-Z\u0410-\u042f])\s*[-\u2013\u2014]\s*\1(?!\w)",
    re.I,
)
_SCALE = re.compile(r"\b(?:SCALE|\u041c\u0410\u0421\u0428\u0422\u0410\u0411)\s*[:.]?\s*(\d+)\s*:\s*(\d+)\b", re.I)
_PLAN_WORDS = {"PLAN", "\u041f\u041b\u0410\u041d"}
_SECTION_WORDS = {"SECTION", "\u0420\u0410\u0417\u0420\u0415\u0417", "\u0421\u0415\u0427\u0415\u041d\u0418\u0415"}
_DETAIL_WORDS = {"DETAIL", "DETAILS", "\u0414\u0415\u0422\u0410\u041b\u042c", "\u0414\u0415\u0422\u0410\u041b\u0418"}
_REINFORCEMENT_WORDS = {"REINFORCEMENT", "\u0410\u0420\u041c\u0418\u0420\u041e\u0412\u0410\u041d\u0418\u0415"}
_FORMWORK_WORDS = {"FORMWORK", "\u041e\u041f\u0410\u041b\u0423\u0411\u041a\u0410"}


def _normalise_title(text: Any) -> str:
    value = " ".join(str(text or "").strip().upper().split())
    match = _SECTION_LABEL.search(value)
    return re.sub(r"\s*[-\u2013\u2014]\s*", "-", match.group(0)) if match else value


def _title_role(text: Any, explicit_role: Any = None) -> str | None:
    role = str(explicit_role or "")
    if role in _TITLE_ROLES:
        return role
    value = " ".join(str(text or "").strip().upper().split())
    words = {item.strip(".,:;()[]") for item in value.split()}
    if _SECTION_LABEL.search(value) or words & _SECTION_WORDS:
        return "section_label"
    if words & _PLAN_WORDS:
        return "plan_view_candidate"
    if words & _REINFORCEMENT_WORDS:
        return "reinforcement_view_candidate"
    if words & _FORMWORK_WORDS:
        return "formwork_view_candidate"
    if words & _DETAIL_WORDS:
        return "detail_view_candidate"
    return None


def _is_title_anchor(item: Mapping[str, Any]) -> bool:
    role = str(item.get("resolved_role", ""))
    if role in _TITLE_ROLES:
        return True
    text = str(item.get("text") or "").strip()
    letters = [character for character in text if character.isalpha()]
    return bool(
        _title_role(text)
        and len(text.split()) >= 2
        and len(letters) >= 5
        and sum(character.isupper() for character in letters) / len(letters) >= 0.8
    )


def _distance(left: fitz.Rect, right: fitz.Rect) -> float:
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return math.hypot(dx, dy)


def _rect_union(items: Iterable[fitz.Rect]) -> fitz.Rect:
    boxes = list(items)
    if not boxes:
        return fitz.Rect()
    output = fitz.Rect(boxes[0])
    for box in boxes[1:]:
        output |= box
    return output


def _line_groups(text_roles: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reconstitute native words into spatial lines without replacing them."""

    tokens = []
    seen: list[tuple[str, fitz.Rect]] = []
    for item in text_roles:
        if len(item.get("bbox_display", [])) != 4:
            continue
        box = fitz.Rect(item["bbox_display"])
        text = str(item.get("text") or "").strip()
        if box.is_empty or not text:
            continue
        if any(
            text == previous
            and (box & previous_box).get_area()
            >= 0.65 * min(box.get_area(), previous_box.get_area())
            for previous, previous_box in seen
        ):
            continue
        seen.append((text, box))
        tokens.append((box, item))
    tokens.sort(
        key=lambda row: (
            (row[0].y0 + row[0].y1) / 2,
            row[0].x0,
            str(row[1].get("id")),
        )
    )

    rows: list[list[tuple[fitz.Rect, Mapping[str, Any]]]] = []
    for box, item in tokens:
        center_y = (box.y0 + box.y1) / 2
        matching = []
        for index, row in enumerate(rows):
            row_box = _rect_union(entry[0] for entry in row)
            row_center_y = (row_box.y0 + row_box.y1) / 2
            tolerance = max(1.5, 0.45 * max(box.height, row_box.height))
            if abs(center_y - row_center_y) <= tolerance:
                matching.append((abs(center_y - row_center_y), index))
        if matching:
            rows[min(matching)[1]].append((box, item))
        else:
            rows.append([(box, item)])

    lines = []
    for row in rows:
        row.sort(key=lambda entry: (entry[0].x0, str(entry[1].get("id"))))
        typical_height = median(entry[0].height for entry in row)
        runs: list[list[tuple[fitz.Rect, Mapping[str, Any]]]] = []
        for entry in row:
            if not runs:
                runs.append([entry])
                continue
            gap = entry[0].x0 - runs[-1][-1][0].x1
            if gap > max(14.0, 3.0 * typical_height):
                runs.append([entry])
            else:
                runs[-1].append(entry)
        for run in runs:
            box = _rect_union(entry[0] for entry in run)
            texts = [str(entry[1].get("text") or "").strip() for entry in run]
            refs = sorted(
                {
                    str(ref)
                    for _, item in run
                    for ref in [item.get("id"), *item.get("primitive_refs", [])]
                    if ref
                }
            )
            lines.append(
                {
                    "text": " ".join(texts),
                    "bbox_display": list(box),
                    "primitive_refs": refs,
                    "explicit_roles": [
                        str(item.get("resolved_role", "")) for _, item in run
                    ],
                }
            )
    return sorted(
        lines,
        key=lambda item: (
            item["bbox_display"][1],
            item["bbox_display"][0],
            item["text"],
        ),
    )


def _build_anchors(text_roles: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lines = _line_groups(text_roles)
    scales = []
    titles = []
    for index, line in enumerate(lines, start=1):
        scale_match = _SCALE.search(line["text"])
        if scale_match:
            scales.append(
                {
                    "id": f"scale_anchor.line.{index:04d}",
                    **line,
                    "scale_ratio": (
                        f"{int(scale_match.group(1))}:{int(scale_match.group(2))}"
                    ),
                }
            )
            continue
        explicit_role = next(
            (role for role in line["explicit_roles"] if role in _TITLE_ROLES),
            None,
        )
        role = _title_role(line["text"], explicit_role)
        letters = [character for character in line["text"] if character.isalpha()]
        uppercase_ratio = (
            sum(character.isupper() for character in letters) / len(letters)
            if letters
            else 1.0
        )
        if role and (
            explicit_role
            or uppercase_ratio >= 0.8
            or _SECTION_LABEL.search(line["text"])
        ):
            titles.append(
                {
                    "id": f"title_anchor.line.{index:04d}",
                    "text": line["text"],
                    "bbox_display": line["bbox_display"],
                    "resolved_role": role,
                    "confidence": 0.96 if explicit_role else 0.9,
                    "primitive_refs": line["primitive_refs"],
                    "explicit_semantic_role": bool(explicit_role),
                    "basis": "native_text_line_semantic_title",
                }
            )

    scale_owner: dict[str, str] = {}
    for scale in scales:
        scale_box = fitz.Rect(scale["bbox_display"])
        rows = []
        for title in titles:
            title_box = fitz.Rect(title["bbox_display"])
            vertical_gap = max(
                scale_box.y0 - title_box.y1,
                title_box.y0 - scale_box.y1,
                0.0,
            )
            center_dx = abs(
                (scale_box.x0 + scale_box.x1 - title_box.x0 - title_box.x1) / 2
            )
            if vertical_gap > max(
                24.0, 3.5 * max(scale_box.height, title_box.height)
            ):
                continue
            if center_dx > 0.8 * max(scale_box.width, title_box.width, 24.0):
                continue
            rows.append((_distance(scale_box, title_box), center_dx, title["id"]))
        rows.sort()
        if rows and (len(rows) == 1 or rows[0][:2] != rows[1][:2]):
            scale_owner[scale["id"]] = rows[0][2]

    for title in titles:
        owned = [
            scale
            for scale in scales
            if scale_owner.get(scale["id"]) == title["id"]
        ]
        owned.sort(
            key=lambda scale: (
                _distance(
                    fitz.Rect(title["bbox_display"]),
                    fitz.Rect(scale["bbox_display"]),
                ),
                scale["id"],
            )
        )
        selected = owned[0] if owned else None
        title["scale_text_ref"] = selected["id"] if selected else None
        title["scale_text"] = selected["text"] if selected else None
        title["scale_ratio"] = selected["scale_ratio"] if selected else None
        title["scale_primitive_refs"] = (
            selected["primitive_refs"] if selected else []
        )

    titles = [
        item
        for item in titles
        if item["explicit_semantic_role"] or item["scale_text_ref"]
    ]

    for item in text_roles:
        if not _is_title_anchor(item) or len(item.get("bbox_display", [])) != 4:
            continue
        normalised = _normalise_title(item.get("text"))
        box = fitz.Rect(item["bbox_display"])
        if any(
            (
                _normalise_title(title["text"]) == normalised
                and _distance(box, fitz.Rect(title["bbox_display"])) <= 2.0
            )
            or str(item.get("id")) in title["primitive_refs"]
            for title in titles
        ):
            continue
        titles.append(
            {
                "id": str(item.get("id")),
                "text": str(item.get("text")),
                "bbox_display": list(box),
                "resolved_role": _title_role(
                    item.get("text"), item.get("resolved_role")
                ),
                "confidence": float(item.get("confidence", 0.0)),
                "primitive_refs": list(item.get("primitive_refs", [])),
                "explicit_semantic_role": True,
                "basis": str(
                    item.get("basis") or "explicit_semantic_text_role"
                ),
                "scale_text_ref": None,
                "scale_text": None,
                "scale_ratio": None,
                "scale_primitive_refs": [],
            }
        )

    output = []
    for anchor in sorted(
        titles,
        key=lambda item: (-float(item.get("confidence", 0.0)), item["id"]),
    ):
        box = fitz.Rect(anchor["bbox_display"])
        normalised = _normalise_title(anchor["text"])
        duplicate = next(
            (
                existing
                for existing in output
                if _normalise_title(existing["text"]) == normalised
                and _distance(box, fitz.Rect(existing["bbox_display"])) <= 2.0
                and math.dist(
                    box.tl, fitz.Rect(existing["bbox_display"]).tl
                )
                <= 8.0
            ),
            None,
        )
        if duplicate is None:
            output.append(anchor)
    return output


def _whitespace_boundary(
    left: float,
    right: float,
    primitive_boxes: Iterable[fitz.Rect],
    *,
    axis: str,
) -> tuple[float, int]:
    gap = right - left
    low = left + 0.25 * gap
    high = right - 0.25 * gap
    midpoint = (left + right) / 2
    boxes = list(primitive_boxes)
    candidates = [low + (high - low) * index / 80 for index in range(81)]

    def occupancy(value: float) -> int:
        if axis == "x":
            return sum(box.x0 < value < box.x1 for box in boxes)
        return sum(box.y0 < value < box.y1 for box in boxes)

    best = min(
        candidates,
        key=lambda value: (occupancy(value), abs(value - midpoint), value),
    )
    return round(best, 3), occupancy(best)


def _partition_primitives(
    view: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    primitive_bounds: Mapping[str, fitz.Rect],
) -> dict[str, Any] | None:
    if len(candidates) < 2 or not all(
        item.get("scale_text_ref") for item in candidates
    ):
        return None
    parent_refs = sorted(map(str, view.get("primitive_refs", []) or []))
    located = [
        (ref, primitive_bounds[ref])
        for ref in parent_refs
        if ref in primitive_bounds
    ]
    if len(located) < 8:
        return None
    centers = [
        (
            (
                fitz.Rect(item["bbox_display"]).x0
                + fitz.Rect(item["bbox_display"]).x1
            )
            / 2,
            (
                fitz.Rect(item["bbox_display"]).y0
                + fitz.Rect(item["bbox_display"]).y1
            )
            / 2,
            item,
        )
        for item in candidates
    ]
    x_spread = max(item[0] for item in centers) - min(item[0] for item in centers)
    y_spread = max(item[1] for item in centers) - min(item[1] for item in centers)
    axis = "x" if x_spread >= y_spread else "y"
    ordered = sorted(
        centers,
        key=lambda item: (
            item[0] if axis == "x" else item[1],
            item[2]["title_anchor_ref"],
        ),
    )
    if (x_spread if axis == "x" else y_spread) < 24.0:
        return None
    boundaries = []
    boxes = [box for _, box in located]
    for left, right in zip(ordered, ordered[1:]):
        left_value = left[0] if axis == "x" else left[1]
        right_value = right[0] if axis == "x" else right[1]
        value, occupancy = _whitespace_boundary(
            left_value, right_value, boxes, axis=axis
        )
        boundaries.append(
            {"value": value, "crossing_primitive_count": occupancy}
        )

    memberships = {item[2]["title_anchor_ref"]: [] for item in ordered}
    excluded = [ref for ref in parent_refs if ref not in primitive_bounds]
    for ref, box in located:
        low = box.x0 if axis == "x" else box.y0
        high = box.x1 if axis == "x" else box.y1
        crossing = any(
            low < boundary["value"] - 0.5
            and high > boundary["value"] + 0.5
            for boundary in boundaries
        )
        if crossing:
            excluded.append(ref)
            continue
        center = (low + high) / 2
        index = sum(center > boundary["value"] for boundary in boundaries)
        memberships[ordered[index][2]["title_anchor_ref"]].append(ref)
    if any(len(refs) < 4 for refs in memberships.values()):
        return None
    return {
        "axis": "vertical_whitespace" if axis == "x" else "horizontal_whitespace",
        "boundaries_display": boundaries,
        "memberships": memberships,
        "excluded_primitive_refs": sorted(set(excluded)),
        "located_parent_primitive_count": len(located),
    }


def segment_views_by_titles(
    page_rect: fitz.Rect,
    views: Iterable[Mapping[str, Any]],
    text_roles: Iterable[Mapping[str, Any]],
    *,
    geometry_primitives: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Certify view scopes without turning a title into physical identity.

    A connected native component may be split only when distinct semantic
    titles each have a uniquely associated scale and the primitive geometry
    supplies a deterministic whitespace boundary. Boundary-spanning paths are
    retained as excluded evidence; native primitives are never split or
    duplicated between resolved scopes.
    """

    views = list(views)
    anchors = _build_anchors(list(text_roles))
    primitive_bounds = {
        str(item.get("id")): fitz.Rect(item["bbox_display"])
        for item in geometry_primitives
        if item.get("id") is not None
        and len(item.get("bbox_display", [])) == 4
    }
    page_diagonal = max(1.0, math.hypot(page_rect.width, page_rect.height))
    maximum_distance = max(18.0, 0.035 * page_diagonal)

    def candidate_for(
        view: Mapping[str, Any], anchor: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        box = fitz.Rect(view.get("bbox_display", ()))
        anchor_box = fitz.Rect(anchor["bbox_display"])
        distance = _distance(box, anchor_box)
        if (
            box.is_empty
            or distance > maximum_distance
            or view.get("role_hypothesis") == "table_or_grid_candidate"
        ):
            return None
        role = str(view.get("role_hypothesis", ""))
        anchor_role = str(anchor.get("resolved_role"))
        compatible = (
            anchor_role == "section_label"
            and "section" in role
            or anchor_role == role
            or anchor_role == "detail_view_candidate"
            and role == "drawing_view_candidate"
        )
        contained = box.contains(anchor_box.tl) or box.contains(anchor_box.br)
        horizontal_overlap = max(
            0.0,
            min(box.x1, anchor_box.x1) - max(box.x0, anchor_box.x0),
        )
        aligned = horizontal_overlap >= 0.25 * min(box.width, anchor_box.width)
        score = (
            0.46
            + (0.18 if compatible else 0.0)
            + (0.10 if contained else 0.0)
            + (0.10 if aligned else 0.0)
            + (0.08 if anchor.get("scale_text_ref") else 0.0)
            + 0.08 * max(0.0, 1.0 - distance / maximum_distance)
        )
        primitive_refs = sorted(
            {
                *map(str, anchor.get("primitive_refs", [])),
                *map(str, anchor.get("scale_primitive_refs", [])),
            }
        )
        return {
            "title_anchor_ref": str(anchor.get("id")),
            "title": str(anchor.get("text")),
            "normalised_title": _normalise_title(anchor.get("text")),
            "role": anchor_role,
            "bbox_display": list(anchor_box),
            "score": round(min(0.99, score), 3),
            "distance": round(distance, 3),
            "primitive_refs": primitive_refs,
            "scale_text_ref": anchor.get("scale_text_ref"),
            "scale_text": anchor.get("scale_text"),
            "scale_ratio": anchor.get("scale_ratio"),
        }

    anchor_owner: dict[str, str] = {}
    for anchor in anchors:
        rows = []
        for view in views:
            candidate = candidate_for(view, anchor)
            if candidate is None:
                continue
            rows.append(
                (
                    -candidate["score"],
                    candidate["distance"],
                    fitz.Rect(view["bbox_display"]).get_area(),
                    str(view.get("id")),
                )
            )
        rows.sort()
        if rows and (len(rows) == 1 or rows[0][:3] != rows[1][:3]):
            anchor_owner[str(anchor.get("id"))] = rows[0][3]

    segments = []
    split_parent_views = set()
    for view in views:
        box = fitz.Rect(view.get("bbox_display", ()))
        if box.is_empty:
            continue
        candidates = []
        for anchor in anchors:
            if anchor_owner.get(str(anchor.get("id"))) != str(view.get("id")):
                continue
            candidate = candidate_for(view, anchor)
            if candidate is not None:
                candidates.append(candidate)
        strongest_by_title: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = candidate["normalised_title"]
            current = strongest_by_title.get(key)
            if current is None or (
                candidate["score"],
                -candidate["distance"],
                candidate["title_anchor_ref"],
            ) > (
                current["score"],
                -current["distance"],
                current["title_anchor_ref"],
            ):
                strongest_by_title[key] = candidate
        candidates = sorted(
            strongest_by_title.values(),
            key=lambda item: (-item["score"], item["title_anchor_ref"]),
        )
        partition = _partition_primitives(view, candidates, primitive_bounds)
        selected_candidates = candidates if partition else candidates[:1]
        unique = bool(candidates) and (
            len(candidates) == 1
            or partition is not None
            or candidates[0]["score"] - candidates[1]["score"] >= 0.12
        )
        if not unique:
            selected_candidates = []

        if partition:
            split_parent_views.add(str(view.get("id")))
        if selected_candidates:
            ordered_selected = sorted(
                selected_candidates,
                key=lambda item: (
                    fitz.Rect(item["bbox_display"]).x0,
                    fitz.Rect(item["bbox_display"]).y0,
                    item["title_anchor_ref"],
                ),
            )
            for scope_index, selected in enumerate(ordered_selected, start=1):
                primitive_refs = (
                    sorted(
                        partition["memberships"][selected["title_anchor_ref"]]
                    )
                    if partition
                    else sorted(map(str, view.get("primitive_refs", []) or []))
                )
                scope_boxes = [
                    primitive_bounds[ref]
                    for ref in primitive_refs
                    if ref in primitive_bounds
                ]
                scope_box = _rect_union(scope_boxes) if scope_boxes else box
                excluded = partition["excluded_primitive_refs"] if partition else []
                segment_id = f"title_view_segment.{len(segments) + 1:03d}"
                evidence_refs = sorted(
                    {
                        str(view.get("id")),
                        selected["title_anchor_ref"],
                        *selected["primitive_refs"],
                        *primitive_refs,
                        *excluded,
                    }
                )
                segments.append(
                    {
                        "id": segment_id,
                        "view_id": str(view.get("id")),
                        "source_view_id": str(view.get("id")),
                        "scope_index": scope_index,
                        "bbox_display": list(scope_box),
                        "primitive_refs": primitive_refs,
                        "excluded_primitive_refs": excluded,
                        "state": "resolved",
                        "title_anchor_ref": selected["title_anchor_ref"],
                        "title": selected["title"],
                        "normalised_title": selected["normalised_title"],
                        "role": selected["role"],
                        "confidence": selected["score"],
                        "scale_text_ref": selected.get("scale_text_ref"),
                        "scale_text": selected.get("scale_text"),
                        "scale_ratio": selected.get("scale_ratio"),
                        "candidate_anchors": candidates,
                        "spatial_partition": (
                            {
                                "axis": partition["axis"],
                                "boundaries_display": partition[
                                    "boundaries_display"
                                ],
                                "located_parent_primitive_count": partition[
                                    "located_parent_primitive_count"
                                ],
                            }
                            if partition
                            else None
                        ),
                        "evidence_refs": evidence_refs,
                        "reason": (
                            "semantic title and scale isolate exact native membership at a whitespace boundary"
                            if partition
                            else "unique semantic title anchors this native geometry scope"
                        ),
                    }
                )
            continue

        segment_id = f"title_view_segment.{len(segments) + 1:03d}"
        segments.append(
            {
                "id": segment_id,
                "view_id": str(view.get("id")),
                "source_view_id": str(view.get("id")),
                "scope_index": None,
                "bbox_display": list(box),
                "primitive_refs": sorted(
                    map(str, view.get("primitive_refs", []) or [])
                ),
                "excluded_primitive_refs": [],
                "state": "candidate" if candidates else "unresolved",
                "title_anchor_ref": None,
                "title": None,
                "normalised_title": None,
                "role": None,
                "confidence": candidates[0]["score"] if candidates else 0.0,
                "scale_text_ref": None,
                "scale_text": None,
                "scale_ratio": None,
                "candidate_anchors": candidates,
                "spatial_partition": None,
                "evidence_refs": [
                    str(view.get("id")),
                    *(item["title_anchor_ref"] for item in candidates),
                ],
                "reason": (
                    "multiple title anchors lack a unique scale-and-whitespace partition"
                    if candidates
                    else "table/grid geometry is excluded from title-certified views"
                    if view.get("role_hypothesis") == "table_or_grid_candidate"
                    else "no semantic title anchor is close enough to the geometry scope"
                ),
            }
        )

    counts = Counter(item["state"] for item in segments)
    return {
        "schema_version": "0.2.0",
        "layer": "title_anchored_view_segmentation",
        "status": "resolved_subset" if counts["resolved"] else "unresolved",
        "segments": segments,
        "summary": {
            "source_view_count": len(views),
            "view_count": len(segments),
            "resolved_segment_count": counts["resolved"],
            "ambiguous_segment_count": counts["candidate"],
            "unanchored_segment_count": counts["unresolved"],
            "split_parent_view_count": len(split_parent_views),
            "resolved_titles": sorted(
                {
                    str(item["title"])
                    for item in segments
                    if item["state"] == "resolved"
                }
            ),
        },
        "contract": {
            "geometry_without_unique_title_remains_uncertified": True,
            "titles_seed_roles_not_physical_identity": True,
            "titles_do_not_create_geometry": True,
            "resolved_scopes_preserve_exact_primitive_membership": True,
            "native_primitives_are_never_split_or_duplicated": True,
            "boundary_spanning_primitives_remain_excluded_evidence": True,
            "tables_and_title_blocks_are_not_view_scopes": True,
            "schedule_values_used": False,
        },
    }
